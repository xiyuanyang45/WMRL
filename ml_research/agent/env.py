#!/usr/bin/env python3
"""ENVIRONMENT SEAM — the single interface between the rollout and "whatever produces feedback + reward".

An env implements:  step(items, workers=1) -> list of grade-result dicts (same order as items), each with
  {status, reward, pos, feedback, exec_time, compiles}   (the agentic_grader result contract)
  + {cpu_s, peak_rss_mb, peak_gpu_mb}  — per-grade CPU seconds, process-tree peak RSS, peak GPU mem (grade_actions).
where each item = {task, data_dir, action, limits, [code]}.

TWO real-execution backends (both run the model's code in the standardized kaggle-grade sandbox; they differ
only in WHERE the grading capacity lives):
  SandboxEnv (ENV_IMPL=sandbox) — grade on THIS process's local SandboxPool (cards on this node). The original
    single-node path: trainer + grading share one box. Borrow a (gpu, slot) -> grade_actions -> return slot.
  RemoteEnv  (ENV_IMPL=remote)  — grade on env_server.py instances reachable over HTTP (this node and/or other
    nodes). The trainer holds NO pool; it POSTs each grade to a server with a free slot and blocks otherwise.
    This is the multi-node layout: a whole node of cards becomes a pure grading service. SBX_SERVERS REQUIRED.

A future learned/simulated env (world model: predict feedback + score WITHOUT executing) plugs in the same way:
implement step() + add a make_env branch. Unknown impl names FAIL LOUD — never silently fall back to (or from)
real execution (a typo silently burning hours of sandbox time, or silently stubbing a real run, are both bad).
"""
import os, json, time, threading, urllib.request, urllib.error
import agentic_rollout as AR


class SandboxEnv:
    """Real execution on a borrowed LOCAL pool slot (one grade owns one card-slot)."""

    def __init__(self, pool):
        self.pool = pool
        self.capacity = pool.capacity

    def step(self, items, workers=1):
        gpu, slot, lock = self.pool.borrow()     # shared cross-process pool: any free slot; blocks when all busy
        try:
            return AR.grade_actions(items, workers=workers, gpu=gpu, slot=slot)
        finally:
            self.pool.return_slot((gpu, slot, lock))


class RemoteEnv:
    """Real execution behind env_server.py HTTP services (multi-node grading).

    Each server advertises a slot capacity (cards × SLOTS_PER_GPU); we model that with one bounded semaphore per
    server, so at most `capacity` grades are in flight on a server at once and a free slot is picked round-robin.
    A grade is one POST /grade with the item batch (server-side it borrows a local pool slot, runs grade_actions,
    returns the result list). FAIL LOUD: any HTTP/transport error raises (a swallowed error would train on fake
    reward=0). The server is the thing that can die; if it does, the POST raises and the run aborts loudly —
    exactly what we want (a half-capacity silent degrade would corrupt the batch)."""

    def __init__(self, servers, timeout_s=None):
        if not servers:
            raise ValueError("RemoteEnv needs SBX_SERVERS=host:port[,host:port] — no implicit grading backend")
        self.servers = []                                    # [(url, semaphore, capacity)]
        for s in servers:
            cap = self._probe_capacity(s)                    # fail loud if a server is unreachable at startup
            self.servers.append([f"http://{s}", threading.BoundedSemaphore(cap), cap])
        self.capacity = sum(c for _, _, c in self.servers)
        self._rr = 0
        self._rr_lock = threading.Lock()
        # wall budget for one POST: the grader's own (execution_timeout+180) plus slack for queueing behind a
        # busy server + data copy. Default generous; the server's own grade timeout is the real bound.
        self.timeout_s = timeout_s or int(os.environ.get("REMOTE_GRADE_TIMEOUT", "1800"))

    @staticmethod
    def _probe_capacity(server):
        url = f"http://{server}/health"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                cap = int(json.loads(r.read())["capacity"])
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            raise RuntimeError(f"env_server {server} unreachable/bad /health at startup: {e}")
        if cap < 1:
            raise RuntimeError(f"env_server {server} reports capacity {cap}")
        return cap

    def describe(self):
        return [(u, c) for u, _, c in self.servers]

    def has_free_capacity(self):
        """Non-blocking probe: True if ANY server has a free slot right now. Used by demand-driven
        routing (grade on the real sandbox iff it has capacity, else WM) so the sandbox runs flat-out without
        overloading (no queue backup -> bounded staleness). Probe = try-acquire + instant-release (doesn't consume)."""
        for _url, sem, _cap in self.servers:
            if sem.acquire(blocking=False):
                sem.release()
                return True
        return False

    def _pick(self):
        """Round-robin over servers, blocking on each server's semaphore. Try every server once per sweep so a
        busy server doesn't starve a free one; sleep briefly only when ALL are saturated."""
        while True:
            with self._rr_lock:
                start = self._rr
                self._rr = (self._rr + 1) % len(self.servers)
            for k in range(len(self.servers)):
                url, sem, _cap = self.servers[(start + k) % len(self.servers)]
                if sem.acquire(blocking=False):
                    return url, sem
            time.sleep(0.05)

    def step(self, items, workers=1):
        url, sem = self._pick()
        try:
            body = json.dumps({"items": items, "workers": workers}).encode()
            # TOLERATE transient env_server failures: a single bad /grade (HTTP 500, malformed/short response) must NOT
            # crash the rollouter and kill the whole run — E#2 (reduce_both_v3f_r2) died at step 51 to one uncaught
            # HTTP 500. Retry the fast/transient faults; on a conn/timeout (possible hang) don't retry (the trainer's
            # queue-starvation guard handles genuine stalls); after exhausting, return a LOUD grade_error fallback
            # (reward 0, same 4-key shape as the proven "malformed" result) per item so the run survives.
            last_err = None
            for _attempt in range(3):
                retryable = True
                try:
                    req = urllib.request.Request(f"{url}/grade", data=body,
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                        res = json.loads(r.read())
                    if isinstance(res, dict) and "error" in res:
                        last_err = f"grade error: {res['error']}"
                    elif len(res) != len(items):
                        last_err = f"returned {len(res)} results for {len(items)} items"
                    else:
                        return res
                except urllib.error.HTTPError as e:                           # 500 etc — transient server fault, retry
                    last_err = f"HTTP {e.code}: {e.read().decode(errors='ignore')[:400] if hasattr(e, 'read') else ''}"
                except (urllib.error.URLError, OSError) as e:                 # conn/timeout — maybe a hang; don't retry
                    last_err = f"conn: {e}"; retryable = False
                if not retryable or _attempt == 2:
                    break
                time.sleep(0.5 * (_attempt + 1))
                print(f"[agentic_env] {url} /grade failed (try {_attempt + 1}/3): {last_err} -> retry", flush=True)
            print(f"[agentic_env] {url} /grade UNRECOVERABLE: {last_err} -> grade_error fallback (reward=0) "
                  f"for {len(items)} item(s)", flush=True)
            return [{"status": "grade_error", "reward": 0.0, "pos": None,
                     "feedback": f"[grade unavailable: {last_err}]"} for _ in items]
        finally:
            sem.release()


def make_env(kind, pool):
    if kind == "sandbox":
        return SandboxEnv(pool)
    if kind == "remote":
        servers = [s.strip() for s in os.environ.get("SBX_SERVERS", "").split(",") if s.strip()]
        return RemoteEnv(servers)
    raise ValueError(f"unknown ENV_IMPL={kind!r} — implement it in agentic_env.make_env (no silent fallback)")
