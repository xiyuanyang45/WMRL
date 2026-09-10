#!/usr/bin/env python3
"""
Segment 2: one multi-turn trajectory rollout.
A trajectory = up to K turns of (model generates action via vLLM) -> (env grades) -> (obs appended to history).
Trajectory reward = max over turns of the execute_code position score (outcome reward, design 2.3-A).

This module is the rollout engine. It needs:
  - a vLLM OpenAI server already running (HTTP), model name to query
  - the unified env's agentic_grader.py for grading (subprocess, PATH = unified env)
It returns the full trajectory record (every turn's prompt, gen, action, obs, reward, timing) for
both training (GRPO over trajectories) and logging.
"""
import os, sys, json, signal, subprocess, tempfile, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_CLK = os.sysconf("SC_CLK_TCK")                       # jiffies/sec, for utime+stime -> seconds
_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024         # statm pages -> kB

# Shared, rate-limited nvidia-smi compute-apps reader: many grades sample concurrently, but the per-PID GPU-memory
# table is global, so we refresh it at most once per _GPU_TTL and every sampler reads the cache -> bounded smi calls.
_GPU_TTL = 1.0
_gpu_lock = threading.Lock()
_gpu_cache = {"t": 0.0, "by_pid": {}}


def _gpu_mem_by_pid():
    """{pid(int): used_MB} for all GPU compute processes on the node, cached for _GPU_TTL. Empty if no GPU / smi fails."""
    import time as _t
    now = _t.monotonic()
    with _gpu_lock:
        if now - _gpu_cache["t"] < _GPU_TTL:
            return _gpu_cache["by_pid"]
        by_pid = {}
        try:
            out = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits",
                                 shell=True, capture_output=True, text=True, timeout=10).stdout
            for ln in out.strip().splitlines():
                p = [x.strip() for x in ln.split(",")]
                if len(p) >= 2 and p[0].isdigit():
                    by_pid[int(p[0])] = by_pid.get(int(p[0]), 0) + int(float(p[1]))
        except Exception:
            pass                                           # no GPU / smi unavailable -> gpu_mb stays 0
        _gpu_cache.update(t=now, by_pid=by_pid)
        return by_pid


def _pgroup_stats(root_pid):
    """Return (RSS_kB, cpu_jiffies, gpu_MB) summed over the WHOLE grade subtree, via /proc + cached nvidia-smi (no
    psutil). The dojo Sandbox runs the model's code in a NEW SESSION (its own process group), so matching only the
    grader's pgid MISSES the GPU-using grandchild (this silently logged gpu=0 / under-counted RSS). We instead walk
    the PPID tree from `root_pid` (the grader) AND union with root's process group, so the model code + its
    joblib/loky workers are all captured regardless of session."""
    info = {}                                              # pid -> (ppid, pgrp, cpu_jiffies, rss_kb)
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                data = f.read()
            rest = data[data.rindex(")") + 2:].split()     # fields after "(comm)": rest[0]=state, rest[1]=ppid, rest[2]=pgrp
            with open(f"/proc/{d}/statm") as f:
                rss = int(f.read().split()[1]) * _PAGE_KB  # resident pages -> kB
            info[int(d)] = (int(rest[1]), int(rest[2]), int(rest[11]) + int(rest[12]), rss)
        except (OSError, ValueError, IndexError):
            continue                                       # proc exited mid-read -> skip
    kids = {}
    for pid, (ppid, _, _, _) in info.items():
        kids.setdefault(ppid, []).append(pid)
    sel, stack = set(), [root_pid]                         # descendants of the grader (PPID tree)
    while stack:
        p = stack.pop()
        if p in sel or p not in info:
            continue
        sel.add(p); stack += kids.get(p, [])
    root_grp = info.get(root_pid, (0, root_pid, 0, 0))[1]  # union: anything still in the grader's process group
    sel |= {pid for pid, (_, pgrp, _, _) in info.items() if pgrp == root_grp}
    tot_rss = sum(info[p][3] for p in sel)
    tot_cpu = sum(info[p][2] for p in sel)
    gpu = _gpu_mem_by_pid()
    tot_gpu = sum(gpu.get(p, 0) for p in sel)
    return tot_rss, tot_cpu, tot_gpu


class _ResSampler(threading.Thread):
    """Poll a grade's process-group RSS + CPU + GPU memory (high-water) while it runs. One per grade subprocess."""
    def __init__(self, pgid, every=1.0):
        super().__init__(daemon=True)
        self.pgid, self.every = pgid, every
        self.peak_rss_kb, self.cpu_jiffies, self.peak_gpu_mb = 0, 0, 0
        self._ev = threading.Event()                       # NOTE: not self._stop — that shadows Thread._stop

    def run(self):
        while not self._ev.is_set():
            rss, cpu, gpu = _pgroup_stats(self.pgid)
            self.peak_rss_kb = max(self.peak_rss_kb, rss)
            self.cpu_jiffies = max(self.cpu_jiffies, cpu)   # cumulative -> last (largest) sample is the total
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu)
            self._ev.wait(self.every)

    def stop(self):
        self._ev.set()
        self.join(timeout=3)
        return {"peak_rss_mb": round(self.peak_rss_kb / 1024, 1), "cpu_s": round(self.cpu_jiffies / _CLK, 1),
                "peak_gpu_mb": self.peak_gpu_mb}

# The env that runs the grader harness (agentic_grader.py): the UNIFIED env with MLE-Dojo deps. Locally that was
# vllm-env; on the cluster it's verl-env (a clone of vllm-env + verl). env_server runs under that very interpreter,
# so default to sys.executable (self-correcting across envs); allow an explicit UNIFIED_PY override.
UNIFIED_PY = os.environ.get("UNIFIED_PY") or sys.executable
GRADER = ROOT / "agentic_grader.py"


def grade_actions(items, workers=8, gpu=None, slot=None):
    """Run the grader subprocess. NO fallback: if grading crashes (harness bug) we RAISE loudly — a silent
    grade_fail/reward=0 would train on fake data and hide the bug. `gpu` = the borrowed card for this grade;
    the model's code runs in the ERA-APPROPRIATE kaggle-grade env (SANDBOX_PY_BIN).
    `slot` = the pool's GLOBAL slot index, keys the disjoint taskset CPU block. With one sandbox per card,
    slot==gpu and the old gpu-keyed behavior is identical; with SLOTS_PER_GPU>1 two grades share a card, so
    the CPU block MUST be keyed by slot (gpu-keyed blocks would collide). gpu-only callers are unchanged."""
    inp = Path(tempfile.mktemp(suffix="_agin.json")); outp = Path(tempfile.mktemp(suffix="_agout.json"))
    inp.write_text(json.dumps({"items": items, "workers": workers}))
    env = dict(os.environ)
    # SANDBOX_NO_GPU=1: CPU-only sandboxes (measured 06-10: ~97% of model solutions are sklearn; sandbox GPUs
    # sat idle — 9B run peak 527MB). `gpu` then acts as a pure SLOT id: it still keys the disjoint taskset
    # core block below, but the sandbox sees no CUDA devices.
    if os.environ.get("SANDBOX_NO_GPU") == "1":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else env.get("GRADE_CUDA", "")
    # the dojo Sandbox runs the model's code via `["python3", code]` (sandbox.py:416) inheriting this PATH; put the
    # kaggle-grade BIN DIR FIRST so the code runs against ERA-APPROPRIATE libs (sklearn 1.4/pandas 2.2), NOT
    # vllm-env (1.8/3.0). SANDBOX_PY_BIN is the bin dir (has python3); if it's a binary path, use its parent.
    sbx = os.environ.get("SANDBOX_PY_BIN", "").rstrip("/")
    if sbx:
        sbx_dir = (str(Path(sbx).parent) if Path(sbx).name.startswith("python") else sbx) + ":"
    else:
        sbx_dir = ""
    env["PATH"] = sbx_dir + str(Path(UNIFIED_PY).parent) + ":" + env.get("PATH", "")
    # Put OUR env's lib (modern libstdc++, GLIBCXX-versioned = backward compatible) FIRST: on cluster the
    # container's `:latest` base image can carry libs (e.g. /opt/conda pyarrow 24) needing GLIBCXX newer than
    # the SYSTEM libstdc++ — killed a job at the FIRST valid tabular submission (metric.py import →
    # "GLIBCXX_3.4.31 not found"). Resolving libstdc++ from vllm-env/lib fixes the symbol for any such leak.
    env["LD_LIBRARY_PATH"] = str(Path(UNIFIED_PY).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # GPU libs for the MODEL's code: it runs under the SANDBOX env (SANDBOX_PY_BIN = kaggle-grade). torch finds its
    # cu13 libs via RPATH, but TensorFlow needs its bundled cu12 libs (nvidia-*-cu12) on LD_LIBRARY_PATH or it sees 0
    # GPUs. Prepend the sandbox env's nvidia/*/lib dirs (cu12 .so.12 + cu13 .so.13 coexist by SONAME, so torch is
    # unaffected). Glob is cheap (≈12 dirs) and tolerant if the env lacks them.
    if sbx:
        _sbx_root = Path(sbx).parent if Path(sbx).name == "bin" else Path(sbx).parent.parent
        _nvlibs = sorted(str(p) for p in (_sbx_root / "lib/python3.11/site-packages/nvidia").glob("*/lib"))
        if _nvlibs:
            env["LD_LIBRARY_PATH"] = ":".join(_nvlibs) + ":" + env["LD_LIBRARY_PATH"]
    # HARD CPU isolation: pin this grade (and its children — the model's code + joblib/loky workers, which inherit
    # affinity) to a disjoint block of SANDBOX_CORES cores keyed by the card id, so concurrent grades physically
    # cannot oversubscribe the 96-core box (a model's sklearn n_jobs=-1 would otherwise grab all cores).
    cmd = [UNIFIED_PY, str(GRADER), str(inp), str(outp)]
    _cores = int(os.environ.get("SANDBOX_CORES", "8"))
    _key = slot if slot is not None else gpu             # slot-keyed when cards are shared (SLOTS_PER_GPU>1)
    if _key is not None and os.path.exists("/usr/bin/taskset"):
        _ncpu = os.cpu_count() or 96
        _base = (int(_key) * _cores) % max(_cores, _ncpu - _cores)
        cmd = ["/usr/bin/taskset", "-c", f"{_base}-{_base + _cores - 1}"] + cmd
    # Wall budget for the WHOLE grader subprocess = model-code execution_timeout + buffer (data copy + scoring).
    # start_new_session => the grader is a process-group leader; on timeout we SIGKILL the whole group so the
    # model's loky/joblib workers (which can survive the dojo's in-process timeout as orphans and hang the pipe)
    # die too. A model-code timeout/hang is a REAL outcome (slow code), NOT a harness bug -> return a penalized
    # 'timeout' result, do NOT crash the run. Real harness errors (rc!=0) still raise loudly.
    wall = max((it.get("limits", {}).get("execution_timeout", 600) for it in items), default=600) + 180
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    sampler = _ResSampler(proc.pid)                  # pgid==pid (session leader); samples the whole grade tree
    sampler.start()
    try:
        try:
            _out, err = proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError: pass
            try: proc.wait(timeout=30)          # BOUNDED: never block the SPMD step on an (unkillable) reap
            except subprocess.TimeoutExpired: pass
            usage = sampler.stop()
            return [{"action": it.get("action", "execute_code"), "status": "timeout", "reward": 0.0,
                     "pos": None, "feedback": f"grade KILLED: exceeded {wall}s wall budget (hung/slow code)",
                     "exec_time": float(wall), "compiles": True, **usage} for it in items]
        usage = sampler.stop()                  # peak_rss_mb + cpu_s for this grade tree (CPU/mem usage per grade)
        if proc.returncode != 0:
            raise RuntimeError(f"grader subprocess failed (rc={proc.returncode}):\n{err[-2000:]}")
        res = json.loads(outp.read_text())
        if len(res) != len(items):
            raise RuntimeError(f"grader returned {len(res)} results for {len(items)} items")
        for r in res:                           # attach measured resource to every item of this grade call
            for _k, _v in usage.items():         # peak_rss_mb + cpu_s + peak_gpu_mb (the last was being dropped)
                r.setdefault(_k, _v)
        return res
    finally:
        if sampler.is_alive(): sampler.stop()
        for f in (inp, outp):
            if f.exists(): f.unlink()
