#!/usr/bin/env python3
"""
Multi-turn grader for agentic RLVR on MLE-Dojo.  Runs in the UNIFIED env (vllm-env now has the
full Kaggle stack). One call grades ONE trajectory's ONE action:
  - request_info: return the competition data structure (column names, file list, sample sub)
  - execute_code: run code in a dojo Sandbox with STANDARDIZED resource limits, return reward+feedback

Standardized sandbox (every exec, dojo-native setrlimit + CUDA_VISIBLE_DEVICES):
  gpu_device (the assigned card), gpu_memory_limit, cpu_time_limit, memory_limit, execution_timeout.
Reward ladder (dense): syntax/truncated 0.0 / compiles-but-runtime-err 0.1 / runs-no-valid-sub 0.2 /
  valid 0.5+0.5*HumanRank.

Usage: python agentic_grader.py <in.json> <out.json>
in.json: {"items":[{"task","data_dir","action","code"(if execute), "gpu_device", "limits":{...}}], "workers"}
out.json: [{"action","status","reward","pos","feedback","exec_time","compiles"}]
The dojo Sandbox runs solutions via PATH-resolved python3 → caller must put the unified env on PATH.
"""
import os, sys, json, time, shutil, tempfile, traceback
from pathlib import Path

DOJO = Path("/home/sagemaker-user/xiyuan_work_dir/auto_research/MLE-Dojo")
sys.path.insert(0, str(DOJO))

# default standardized resource quota (one A100 + 64GB + cpu budget); overridable per item
DEFAULT_LIMITS = {"gpu_memory_limit": 40, "memory_limit": 64, "cpu_time_limit": 600, "execution_timeout": 600}

_TOK = None
def _cap_feedback(fb, max_tokens=int(os.environ.get("FEEDBACK_MAX_TOKENS", "1000"))):
    """TOKEN-level head+tail cap on execution feedback (unified with the trainer's OBS_MAX=1000 budget).
    A Python traceback puts the exception name+message on the LAST line, and deep pandas/sklearn stacks
    are long — the old head-only fb[:1200] (CHARS) cut the punchline off 30% of runtime_err feedbacks
    (measured on k5 steps 150-189): the model debugged BLIND. Keep head 500 + tail (500 − marker) tokens
    so the TOTAL is ≤ max_tokens and the trainer-side _cap_obs never re-truncates (single cut point)."""
    if len(fb) <= max_tokens:           # every token covers ≥1 char → cheap early-out, no tokenizer load
        return fb
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer   # grader runs in the unified env; MODEL is set by the trainer
        _TOK = AutoTokenizer.from_pretrained(
            os.environ.get("MODEL", "/home/sagemaker-user/mleb-baseline/models/qwen3.5-4b"))
    ids = _TOK(fb, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return fb
    marker = "\n…(output truncated)…\n"
    n_marker = len(_TOK(marker, add_special_tokens=False).input_ids)
    head = max_tokens // 2
    tail = max_tokens - head - n_marker
    return (_TOK.decode(ids[:head], skip_special_tokens=False) + marker
            + _TOK.decode(ids[-tail:], skip_special_tokens=False))


def _make_env(task, data_dir, outdir, gpu_device, limits):
    from mledojo.gym.env import KaggleEnvironment
    from mledojo.gym.competition import CompetitionRegistry, CompInfo
    from mledojo.competitions import get_metric
    # `mleb__<slug>` tasks are MLE-bench-Lite competitions: not in MLE-Dojo's registry, so get_metric
    # returns None. Delegate to MLE-bench's OFFICIAL grade_fn behind the MLE-Dojo metric interface —
    # see mlebench_metric.py for why we delegate instead of reimplementing the metric.
    import mlebench_metric as _mlb
    if _mlb.is_mlebench_task(task):
        mc = _mlb.make_metric_class(task)
    else:
        mc = get_metric(task)
    if mc is None:
        raise RuntimeError("get_metric None")
    reg = CompetitionRegistry()
    reg.register(name=task, data_dir=str(data_dir),
                 comp_info=CompInfo(category="General", level="beginner",
                                    output_type="submission.csv", higher_is_better=True),
                 metric_class=mc)
    return KaggleEnvironment.make(
        competition_name=task, output_dir=str(outdir), competition_registry=reg,
        score_mode="position", gpu_device=gpu_device,
        gpu_memory_limit=limits.get("gpu_memory_limit"),
        cpu_time_limit=limits.get("cpu_time_limit"),
        memory_limit=limits.get("memory_limit"),
        execution_timeout=limits.get("execution_timeout"))


def _data_structure(data_dir):
    """request_info → human-readable data structure (file list + columns of csv/json + sample sub head)."""
    pub = Path(data_dir) / "public"
    lines = ["DATA STRUCTURE (read-only info):", f"Files in DATA_DIR:"]
    import pandas as pd, json as J
    for p in sorted(pub.glob("*")):
        if p.is_file():
            sz = p.stat().st_size
            lines.append(f"  {p.name} ({sz//1024}KB)")
            try:
                if p.suffix == ".csv":
                    d = pd.read_csv(p, nrows=3)
                    lines.append(f"      columns: {list(d.columns)}")
                    lines.append(f"      first rows:\n{d.head(2).to_string(max_cols=12)}")
                elif p.suffix == ".json":
                    d = J.load(open(p))
                    if isinstance(d, list) and d:
                        lines.append(f"      json list of {len(d)}; keys of item0: {list(d[0].keys()) if isinstance(d[0],dict) else type(d[0])}")
                # other suffixes (.zip/.txt/.md): just the filename+size already listed
            except (UnicodeDecodeError, J.JSONDecodeError, __import__("pandas").errors.ParserError):
                # genuinely-unparseable file (binary/corrupt) is data about the dataset, note it briefly
                lines.append(f"      (not parseable as csv/json)")
        else:
            lines.append(f"  {p.name}/ (dir)")
    return "\n".join(lines)[:4000]


def grade_one(item):
    """NO silent fallbacks (per design principle). Harness errors (can't build env, missing config,
    bad data dir) RAISE loudly. The ONE legitimate "error as data" is the model's generated code
    failing at runtime — dojo's env.step captures that into obs WITHOUT raising, so we read it as reward.
    A compile-error in the model code is also data (syntax_err reward 0). Everything else = bug = raise."""
    action = item["action"]                       # KeyError if caller forgot it -> loud, intended
    task = item["task"]; data_dir = item["data_dir"]
    # GPU mapping: the SCHEDULER already pinned this grader process to one physical GPU via
    # CUDA_VISIBLE_DEVICES (so inside here there's exactly one visible card = index 0). dojo's Sandbox
    # ALSO sets CUDA_VISIBLE_DEVICES=gpu_device for the model's subprocess, so we must pass 0 (the only
    # visible index), NOT the physical id — otherwise it points at a non-existent card. If no GPU was
    # assigned (CUDA_VISIBLE_DEVICES unset/empty) the sandbox runs CPU-only (gpu_device=None).
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_device = 0 if (_cvd != "" and item.get("gpu_device") is not None) else None
    limits = {**DEFAULT_LIMITS, **item.get("limits", {})}
    t0 = time.time()

    # request_info: describe the data. If this fails it's a HARNESS bug -> raise (don't hide as "peek failed").
    if action == "request_info":
        return {"action": action, "status": "info", "reward": 0.0, "pos": None,
                "feedback": _data_structure(data_dir), "exec_time": time.time()-t0, "compiles": True}

    if action != "execute_code":
        raise ValueError(f"unknown action {action!r}")

    code = item["code"]                            # KeyError if execute without code -> loud
    outdir = Path(tempfile.mkdtemp(prefix=f"ag_{task[:10]}_"))       # mledojo output_dir: env_history/ + submission.csv
    workdir = Path(tempfile.mkdtemp(prefix=f"agwork_{task[:10]}_"))  # SEPARATE agent cwd — ISOLATES env_history from the
    env = _make_env(task, data_dir, outdir, gpu_device, limits)     # model's untrusted cwd cleanup. ROOT CAUSE of the
    # 9b_sandbox crash: the model ran with cwd=outdir and env_history was outdir/env_history; model code that cleans its
    # working dir (shutil.rmtree of subdirs) deleted env_history -> mledojo's feedback read -> FileNotFoundError -> fatal.
    # Now the model cwd's into workdir; env_history stays in outdir (untouched by relative cwd cleanup).
    env.reset()
    # compile-check (model-code syntax error IS data, not a harness bug)
    import py_compile, tempfile as _tf
    _cf = _tf.NamedTemporaryFile("w", suffix=".py", delete=False); _cf.write(code); _cf.close()
    try:
        py_compile.compile(_cf.name, doraise=True); compiles = True
    except py_compile.PyCompileError:
        compiles = False                           # model wrote bad syntax = data
    finally:
        os.unlink(_cf.name)
    # PATH-contract header. CRITICAL: give the model an ISOLATED per-grade COPY of the data. Earlier we
    # symlinked the SHARED public/ dir under data/public/input — and a model that wrote through it
    # (e.g. zipfile.extractall('data/')) DELETED the source files, corrupting the dataset for every other
    # trajectory/grade. Copying per-grade means the model can read/extract/mutate freely with zero risk to
    # the shared original (the dojo still grades against the untouched data_dir). Small tasks -> copy <1s.
    pub = (Path(data_dir)/"public").as_posix()
    datac = (workdir/"input_data").as_posix()   # per-grade data copy lives in the agent's cwd (workdir), not the env dir
    # optional BLAS/OpenMP thread cap for the model's code: with ~40 cores and 5 concurrent sandboxes,
    # unbounded sklearn/numpy threads peg the CPU (load >100) AND balloon virtual memory. A small cap
    # cuts contention and flaky resource-kills with negligible effect on these small datasets. Gated by
    # SANDBOX_THREADS (unset -> no cap, unchanged behavior).
    _nthr = os.environ.get("SANDBOX_THREADS", "")
    _loky = os.environ.get("SANDBOX_LOKY", "")    # cap joblib/loky PROCESSES (sklearn n_jobs=-1) — else each grade
    thr_hdr = ""                                  # spawns ~ncpu(96) workers -> 4 grades oversubscribe the box
    if _nthr:
        thr_hdr = (f"import os\nfor _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',"
                   f"'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[_v]='{int(_nthr)}'\n")
    if _loky:
        thr_hdr += (f"import os\nfor _v in ['LOKY_MAX_CPU_COUNT','JOBLIB_CPU_COUNT']: os.environ[_v]='{int(_loky)}'\n")
    header = (
        thr_hdr +
        "import os, shutil, warnings; warnings.filterwarnings('ignore')\n"
        f"os.chdir(r'{workdir.as_posix()}')\n"   # agent works in workdir; env_history (outdir/) stays out of its reach
        f"shutil.copytree(r'{pub}', r'{datac}', dirs_exist_ok=True)\n"
        f"DATA_DIR = r'{datac}'\n"
        f"SUBMISSION_PATH = r'{(outdir/'submission.csv').as_posix()}'\n"
        "os.environ['DATA_DIR'] = DATA_DIR; os.environ['SUBMISSION_PATH'] = SUBMISSION_PATH\n"
        # legacy relative access -> point common names at the ISOLATED copy (safe to write through)
        "for _ln in ['data','public','input']:\n"
        "    if not os.path.exists(_ln): os.symlink(DATA_DIR, _ln)\n"
        "for _f in os.listdir(DATA_DIR):\n"
        "    if not os.path.exists(_f): os.symlink(os.path.join(DATA_DIR, _f), _f)\n")
    obs, r = env.step("execute_code", code=header + code)   # dojo captures model-code errors into obs
    dt = time.time() - t0
    fb = obs.get("feedback")
    if isinstance(fb, dict): fb = fb.get("base", {}).get("feedback", "")
    fb = str(fb)
    # reward ladder (env-configurable)
    R_COMPILE = float(os.environ.get("R_COMPILE", "0.1"))
    R_RUN = float(os.environ.get("R_RUN", "0.2"))
    R_VBASE = float(os.environ.get("R_VALID_BASE", "0.5"))
    R_VSCALE = float(os.environ.get("R_VALID_SCALE", "0.5"))
    pos = None
    if r and r > 0:
        status = "valid"; pos = float(r); reward = R_VBASE + R_VSCALE*float(r)
    elif "Execution successful" in fb:
        status = "ran_no_sub"; reward = R_RUN
    elif compiles:
        status = "runtime_err"; reward = R_COMPILE
    else:
        status = "syntax_err"; reward = 0.0
    env.close()
    shutil.rmtree(outdir, ignore_errors=True); shutil.rmtree(workdir, ignore_errors=True)
    return {"action": action, "status": status, "reward": reward, "pos": pos,
            "feedback": _cap_feedback(fb), "feedback_raw": fb,   # feedback = capped obs for the model; feedback_raw = FULL untruncated, for the audit log
            "exec_time": dt, "compiles": compiles}


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    spec = json.load(open(inp))
    items = spec["items"]; workers = spec.get("workers", 8)
    results = [None]*len(items)
    if workers <= 1 or len(items) == 1:
        for i, it in enumerate(items): results[i] = grade_one(it)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=min(workers, len(items))) as ex:
            futs = {ex.submit(grade_one, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()   # raises loudly if a worker hit a harness bug
    json.dump(results, open(outp, "w"))


if __name__ == "__main__":
    main()
