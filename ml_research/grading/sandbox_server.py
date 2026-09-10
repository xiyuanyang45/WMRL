#!/usr/bin/env python3
"""Sandbox grading SERVICE — exposes a node's GPUs as an HTTP grading backend for RemoteEnv (agentic_env).

WHY THIS EXISTS (the multi-node motivation): on one node, trainer + vLLM + sandboxes fight over 8 cards, so the
agent's experiments get few (or zero) GPU sandboxes. With multi-node we dedicate a WHOLE node to grading: run
this server there over all 8 cards, optionally PACKING 2 sandboxes per card (SLOTS_PER_GPU=2 => 16 concurrent
GPU sandboxes), and point the trainer node's RemoteEnv at it. The trainer node can ALSO run a small local server
on its leftover cards. Same standardized grading code path (agentic_rollout.grade_actions -> agentic_grader) as
the single-node run — only the transport changed.

ONE GPU SANDBOX = ONE POOL SLOT. The server owns a cross-process SandboxPool over ENV_GPUS × SLOTS_PER_GPU. Each
POST /grade borrows a (gpu, slot): the grade runs with CUDA_VISIBLE_DEVICES=gpu (one visible card) and tasksets
to a disjoint CPU block keyed by the GLOBAL slot index (so two sandboxes sharing a card don't share cores). When
packing (SLOTS_PER_GPU>1) the per-sandbox gpu_memory_limit is HALVED here explicitly (SBX_GPU_MEM_GB), so the
packed grades fit the card — the only place card-sharing's memory split is decided.

FAIL LOUD: a harness error in grade_actions raises and is returned as {"error": ...} with HTTP 500; RemoteEnv
turns that into a loud RuntimeError that aborts the run (a swallowed error would train on fake reward=0). Model
CODE failures (syntax/runtime/timeout) are normal graded outcomes, NOT errors — grade_actions returns them.

Config (env): ENV_GPUS (REQUIRED, e.g. "0,1,2,3,4,5,6,7"), SLOTS_PER_GPU (1), PORT (18000),
  SBX_GPU_MEM_GB (per-sandbox GPU GB — set to ~card/SLOTS_PER_GPU), SANDBOX_CORES, SANDBOX_THREADS, SANDBOX_LOKY,
  SANDBOX_PY_BIN (kaggle-grade bin), SANDBOX_NO_GPU (1 => CPU-only sandboxes), EXP (lockdir namespace).
Run: ENV_GPUS=0,1,..,7 SLOTS_PER_GPU=2 PORT=18000 vllm-env/bin/python3.11 env_server.py
"""
import os, sys, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_rollout as AR
import agentic_sched as ASch

ENV_GPUS = [int(x) for x in os.environ.get("ENV_GPUS", "").split(",") if x.strip() != ""]
if not ENV_GPUS:
    raise SystemExit("env_server: set ENV_GPUS to the grading cards (e.g. ENV_GPUS=0,1,2,3,4,5,6,7)")
SLOTS_PER_GPU = int(os.environ.get("SLOTS_PER_GPU", "1"))
PORT = int(os.environ.get("PORT", "18000"))
NO_GPU = os.environ.get("SANDBOX_NO_GPU", "0") == "1"
# per-sandbox GPU memory cap injected into every item. When packing 2/card on a 40GB A100, ~19GB each; the
# trainer's items carry NO gpu_memory_limit, so this server is where the card-sharing split is set (fail loud
# if someone packs without shrinking: a 40GB default × 2 on one card would OOM the second sandbox).
SBX_GPU_MEM_GB = int(os.environ.get("SBX_GPU_MEM_GB", "0")) or None
if SLOTS_PER_GPU > 1 and not NO_GPU and not SBX_GPU_MEM_GB:
    raise SystemExit(f"env_server: SLOTS_PER_GPU={SLOTS_PER_GPU} packs cards but SBX_GPU_MEM_GB is unset — set it "
                     f"to ~card_GB/{SLOTS_PER_GPU} so packed GPU sandboxes fit (or SANDBOX_NO_GPU=1 for CPU-only).")

POOL = ASch.SandboxPool(ENV_GPUS, max_tasks_per_gpu=SLOTS_PER_GPU,
                        lockdir=f"/tmp/envsrv_{os.environ.get('EXP', 'default')}_{PORT}")
_served = {"grades": 0, "errors": 0}
_log_lock = threading.Lock()


def _grade(items, workers):
    """Borrow a local slot, grade, return slot. Injects gpu enablement + the packed gpu_memory_limit per item."""
    gpu, slot, lock = POOL.borrow()
    try:
        prepared = []
        for it in items:
            it = dict(it)
            lim = dict(it.get("limits", {}))
            if SBX_GPU_MEM_GB:
                lim["gpu_memory_limit"] = SBX_GPU_MEM_GB
            it["limits"] = lim
            if not NO_GPU:
                it["gpu_device"] = 0            # non-None => grader uses the one visible card (index 0)
            prepared.append(it)
        return AR.grade_actions(prepared, workers=workers, gpu=gpu, slot=slot)
    finally:
        POOL.return_slot((gpu, slot, lock))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass                                    # silence per-request stderr spam; we log our own one-liners

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"capacity": POOL.capacity, "gpus": ENV_GPUS, "slots_per_gpu": SLOTS_PER_GPU,
                             "grades": _served["grades"], "errors": _served["errors"]})
        else:
            self._send(404, {"error": f"no GET {self.path}"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=httpd.shutdown, daemon=True).start()
            return
        if self.path != "/grade":
            self._send(404, {"error": f"no POST {self.path}"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        spec = json.loads(self.rfile.read(n))
        items, workers = spec["items"], spec.get("workers", 1)
        t0 = time.time()
        try:
            res = _grade(items, workers)
        except Exception as e:                  # harness bug -> 500 -> RemoteEnv aborts the run loudly
            with _log_lock:
                _served["errors"] += 1
            import traceback
            print(f"[env_server:{PORT}] GRADE ERROR ({time.time()-t0:.0f}s): {e}\n{traceback.format_exc()}",
                  flush=True)
            self._send(500, {"error": str(e)})
            return
        with _log_lock:
            _served["grades"] += 1
            n_done = _served["grades"]
        r0 = res[0] if res else {}
        # abs start->end timestamps (for the sandbox-lane Gantt: each grade is a [start,end] bar) + the full I/O metrics
        _t1 = time.time()
        print(f"[env_server:{PORT}] ts={time.strftime('%H:%M:%S', time.localtime(t0))}->{time.strftime('%H:%M:%S', time.localtime(_t1))} "
              f"grade#{n_done} task={items[0].get('task')} status={r0.get('status','?')} "
              f"wall={_t1-t0:.1f}s cpu={r0.get('cpu_s','?')}s exec={r0.get('exec_time','?')}s peak_rss={r0.get('peak_rss_mb','?')}MB "
              f"peak_gpu={r0.get('peak_gpu_mb','?')}MB reward={r0.get('reward','?')} pos={r0.get('pos','?')}", flush=True)
        self._send(200, res)


httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
httpd.daemon_threads = True

if __name__ == "__main__":
    print(f"[env_server:{PORT}] serving cards {ENV_GPUS} × {SLOTS_PER_GPU} slots = capacity {POOL.capacity} "
          f"| gpu={'OFF' if NO_GPU else f'ON ({SBX_GPU_MEM_GB or 38}GB/sandbox)'} | "
          f"cores={os.environ.get('SANDBOX_CORES','8')}", flush=True)
    httpd.serve_forever()
    print(f"[env_server:{PORT}] stopped", flush=True)
