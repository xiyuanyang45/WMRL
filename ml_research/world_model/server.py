#!/usr/bin/env python3
"""WORLD-MODEL grading SERVICE — a DROP-IN for env_server.py that PREDICTS the grading outcome with an LLM
instead of running the code in a real sandbox.

Same HTTP contract as env_server.py (GET /health -> {capacity,...}; POST /grade {items, workers} -> result list),
so the trainer's RemoteEnv (agentic_env.py) is UNCHANGED — point SBX_SERVERS at these instead of env_servers and
the whole async rollout works as-is. The WM is the SAME base model as the RL agent (no-distillation): a Qwen3.5-4B
agent is graded by a Qwen3.5-4B WM, 9B by 9B.

WHY ASYNC (the #1 design goal = MINIMIZE WM rollout time): one process hosts ONE vLLM AsyncLLMEngine on the single
visible card (launcher pins CUDA_VISIBLE_DEVICES). aiohttp serves many concurrent POST /grade on the SAME event
loop, each `await engine.generate(...)`; vLLM CONTINUOUS-BATCHES all in-flight grades. So while the agent generates
the next turn on node0, node1 batches every pending grade at once — the WM never serializes. Launch 8 of these (one
per card) for 8x throughput; RemoteEnv round-robins across them. Parallel is ~free; serial latency is what costs.

Per item (one turn's action):
  - request_info  -> the real data structure (deterministic, no LLM) — the agent's free data peek.
  - malformed     -> deterministic 0.0 (the loop already short-circuits this; handled for safety).
  - execute_code  -> deterministic gate (compile) for malformed/syntax (no LLM), else ONE holistic WM call
                     (prompts.HOLISTIC, guided JSON) -> wm_aggregate.aggregate_holistic(rwd_top, fdbk_top) ->
                     {status, reward, pos, feedback}, mirroring the real ladder. The WM env_feedback mimics the
                     sandbox traceback so the agent's next-turn observation looks identical to the real env.

Reward/feedback ABLATION FLAGS (env): WM_RWD_TOP, WM_FDBK_TOP in {1,2} -> passed straight to aggregate_holistic.
The WM is NON-THINK: guided JSON forces the structured output from the first token, so it emits ZERO reasoning
tokens (output is ~235 tok of pure JSON). No think mode.

Config (env): WM_MODEL (REQUIRED, = agent base model path), PORT (18000), WM_TP (1), WM_GPU_UTIL (0.90),
  WM_MAX_MODEL_LEN (12288), WM_CAPACITY (advertised concurrency for RemoteEnv's semaphore; 256), WM_MAX_TOKENS,
  WM_SEED, WM_ENFORCE_EAGER (0/1 — 0 = CUDA graphs, the FAST path; set 1 only if graph capture fails on this stack).
  vLLM accel is selected by the launcher via VLLM_ATTENTION_BACKEND / VLLM_USE_FLASHINFER_SAMPLER.
Run: CUDA_VISIBLE_DEVICES=0 WM_MODEL=<path> PORT=18000 vllm-env/bin/python3.11 wm_server.py
"""
import os, sys, json, time, asyncio, re, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import prompts as P
import wm_aggregate as A

# ---- config ----
WM_MODEL = os.environ["WM_MODEL"]                       # fail loud if unset — no implicit model
PORT = int(os.environ.get("PORT", "18000"))
RWD_TOP = int(os.environ.get("WM_RWD_TOP", "1"))
FDBK_TOP = int(os.environ.get("WM_FDBK_TOP", "2"))
TP = int(os.environ.get("WM_TP", "1"))
GPU_UTIL = float(os.environ.get("WM_GPU_UTIL", "0.90"))   # DEDICATED card on the env node -> reserve it for max KV/batch
MAX_MODEL_LEN = int(os.environ.get("WM_MAX_MODEL_LEN", "12288"))
# advertised in-flight cap per server (RemoteEnv's semaphore). The 4B/9B engine batches hundreds of these light grades
# easily; set generous so RemoteEnv never throttles a server (the real load is <=64 grades/step over 8 servers).
CAPACITY = int(os.environ.get("WM_CAPACITY", "256"))
# HARD CAP 1024, NO retry. Output is TINY (measured p50=235, p99=316, max=385 tok — the model stops at the JSON close
# via EOS), so 1024 never bites a normal grade. LENGTH IS EXPENSIVE (one rambler at 2k would extend the WHOLE
# continuous-batch decode tail, slowing every concurrent grade); WIDTH IS FREE. The rare ~2% stochastic rambler that
# runs past 1024 -> truncated JSON -> we just take the low default (logged + counted in /health). No retry (a 2nd
# sequential generation would lengthen the batch tail for ~1% of recovery — not worth it).
MAX_TOKENS = int(os.environ.get("WM_MAX_TOKENS", "1024"))
SEED = int(os.environ.get("WM_SEED", "0"))
ENFORCE_EAGER = os.environ.get("WM_ENFORCE_EAGER", "0") == "1"
OVERVIEW_MAX_CHARS = int(os.environ.get("WM_OVERVIEW_CHARS", "4000"))   # match the fidelity-test regime exactly
CODE_MAX_CHARS = int(os.environ.get("WM_CODE_CHARS", "8000"))

# ---- guided-JSON schema (identical to fidelity.py: maxLength caps keep each field short so the JSON ALWAYS closes
#      well under MAX_TOKENS -> fast decode + zero truncation/parse-fail). ----
_PRED = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["malformed", "syntax_err", "runtime_err", "ran_no_sub", "valid"]},
    "error_type": {"type": "string", "maxLength": 40}, "error_line": {"type": "integer"},
    "quality_bucket": {"type": "string", "enum": ["low", "fair", "mid", "high", ""]},
    "env_feedback": {"type": "string", "maxLength": 420}, "confidence": {"type": "number"}},
    "required": ["status", "error_type", "error_line", "quality_bucket", "env_feedback", "confidence"]}
HOLI_SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string", "maxLength": 300},
    "predictions": {"type": "array", "items": _PRED, "minItems": 1, "maxItems": 2}},
    "required": ["reason", "predictions"]}

_HDR_RE = re.compile(r"^\s*===[^\n]*===\s*\n")           # strip a leading "=== ... ===" so the loop's own prefix is the only one


def extract_json(text):
    """Robust: strip <think>..</think>, take the last balanced {...} that json-parses. None if unparseable."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    depth = 0; start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0: start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None: cands.append(text[start:i + 1])
    for c in reversed(cands):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _data_structure(data_dir, overview):
    """request_info -> the real data structure (file list + columns + sample head), like the sandbox grader. Falls
    back to the agent's overview (which already carries the inventory) if the listing can't be built."""
    try:
        import agentic_grader as G
        return G._data_structure(data_dir)
    except Exception as e:
        return (overview or "") + f"\n(data listing unavailable: {e})"


# ---- per-server live stats (for the 'minimize rollout time' instrumentation) ----
_stats = {"grades": 0, "llm_calls": 0, "parse_fail": 0, "gate": 0, "inflight": 0, "inflight_max": 0,
          "lat": collections.deque(maxlen=512), "gen_tok": collections.deque(maxlen=512),
          "status": collections.Counter(), "t_last_log": time.time(), "n_last_log": 0}


def _pctl(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs); return s[min(len(s) - 1, int(q * len(s)))]


def _maybe_log():
    """Throughput / latency / in-flight one-liner every ~30s or 200 grades — the WM-side time+util telemetry."""
    n = _stats["grades"]; now = time.time()
    if n - _stats["n_last_log"] < 200 and now - _stats["t_last_log"] < 30:
        return
    dt = max(1e-6, now - _stats["t_last_log"]); dn = n - _stats["n_last_log"]
    lat = list(_stats["lat"])
    print(f"[wm:{PORT}] grades={n} (+{dn} in {dt:.0f}s = {dn/dt:.1f}/s) inflight={_stats['inflight']}"
          f"(max {_stats['inflight_max']}) lat_s p50={_pctl(lat,.5):.2f} p95={_pctl(lat,.95):.2f} "
          f"max={max(lat) if lat else 0:.2f} gen_tok_p50={_pctl(list(_stats['gen_tok']),.5):.0f} "
          f"llm={_stats['llm_calls']} gate={_stats['gate']} parse_fail={_stats['parse_fail']} "
          f"status={dict(_stats['status'])}", flush=True)
    _stats["t_last_log"], _stats["n_last_log"] = now, n


# ---- the engine call ----
async def _wm_generate(app, messages, max_tokens=None, seed=None):
    """One holistic WM call -> (raw_text, n_gen_tok). Continuous-batched by vLLM across all concurrent grades.
    max_tokens/seed overridable so the parse-fail RETRY can both add token budget AND vary the sample (a same-seed
    re-run is deterministic => useless for a truncation)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    max_tokens = max_tokens or MAX_TOKENS
    seed = SEED if seed is None else seed
    tok = app["tok"]
    # MATCH fidelity.py EXACTLY: it never passed enable_thinking, so Qwen3.5's template default is used. The model
    # does NOT actually think — guided JSON forces the schema from the first token (zero reasoning tokens). Passing
    # enable_thinking=False instead changes the primer suffix and inflates FP 22->31, so we keep the default.
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.0,
                        max_tokens=max_tokens, seed=seed,
                        structured_outputs=StructuredOutputsParams(json=HOLI_SCHEMA))
    rid = f"{os.getpid()}-{_stats['llm_calls']}"
    _stats["llm_calls"] += 1
    final = None
    async for out in app["engine"].generate(prompt, sp, request_id=rid):
        final = out
    o = final.outputs[0]
    return o.text, len(o.token_ids)


async def grade_one(app, item):
    """Predict one action's grade. Result contract mirrors agentic_grader.grade_one:
    {action,status,reward,pos,feedback,feedback_raw,exec_time,compiles} (+ wm_audit for the rollout log)."""
    t0 = time.time()
    action = item.get("action")
    if action == "request_info":
        fb = _data_structure(item["data_dir"], item.get("overview"))
        return {"action": action, "status": "info", "reward": 0.0, "pos": None,
                "feedback": fb, "feedback_raw": fb, "exec_time": round(time.time() - t0, 3), "compiles": True}
    if action == "malformed":
        fb = "No valid action found. Emit a ```python``` code block."
        return {"action": action, "status": "malformed", "reward": 0.0, "pos": None,
                "feedback": fb, "feedback_raw": fb, "exec_time": round(time.time() - t0, 3), "compiles": False}
    if action != "execute_code":
        raise ValueError(f"WM: unknown action {action!r}")

    code = (item["code"] or "")[:CODE_MAX_CHARS]
    # ---- deterministic gate first: skip the LLM entirely for empty/syntactically-broken code (fast + exact) ----
    det = A.deterministic_gate(code)
    if det is not None:
        _stats["gate"] += 1
        fb = ("No valid action found. Emit a ```python``` code block." if det == "malformed"
              else "Execution failed: Traceback (most recent call last):\n  File \"solution.py\"\nSyntaxError: invalid syntax")
        return {"action": action, "status": det, "reward": A.reward_of(det), "pos": None,
                "feedback": fb, "feedback_raw": fb, "exec_time": round(time.time() - t0, 3),
                "compiles": False, "wm_audit": {"gate": det}}

    overview = item.get("overview")
    assert overview, "WM execute_code item missing 'overview' — the AgentLoop must send it (fail loud, no rebuild)"
    msgs_sys, msgs_user = P.holistic_messages(item["task"], overview[:OVERVIEW_MAX_CHARS], code)
    messages = [{"role": "system", "content": msgs_sys}, {"role": "user", "content": msgs_user}]

    # SINGLE generation at the 1024 cap — NO retry. A parse-fail here is a rare (~2%) STOCHASTIC rambler whose
    # `reason` ran past 1024 -> truncated JSON. A retry would be a 2nd sequential generation that lengthens the batch
    # TAIL for ~half-a-percent of recovery — not worth it (length is expensive, width is free). We just take the low
    # default for those, logged + counted (watch parse_fail in /health; if it ever climbs, revisit the prompt).
    text, ngen = await _wm_generate(app, messages, max_tokens=MAX_TOKENS, seed=SEED)
    truncated = ngen >= MAX_TOKENS                        # hit the cap with unclosed JSON => truncation
    j = extract_json(text)
    if not (j and (j.get("predictions") or j.get("status"))):
        j = None
    text_with_fence = f"```python\n{code}\n```"
    if j is None:
        _stats["parse_fail"] += 1
        print(f"[wm:{PORT}] PARSE_FAIL task={item.get('task')} n_gen_tok={ngen} truncated={truncated} "
              f"-> safe default runtime_err. raw[-160:]={text[-160:]!r}", flush=True)
        fb = "Execution failed: Traceback (most recent call last):\n  (world-model could not parse output)"
        agg = {"status": "runtime_err", "reward": 0.1, "env_feedback": fb, "error_type": "", "error_line": 0,
               "cand_status": ["runtime_err"], "stage": "parse_fail"}
    else:
        try:
            agg = A.aggregate_holistic(text_with_fence, j, rwd_top=RWD_TOP, fdbk_top=FDBK_TOP)
        except Exception as e:                            # defensive: a malformed-but-parseable JSON -> logged default
            _stats["parse_fail"] += 1
            print(f"[wm:{PORT}] AGG_ERROR {e} task={item.get('task')} -> safe default runtime_err", flush=True)
            agg = {"status": "runtime_err", "reward": 0.1, "env_feedback":
                   "Execution failed: Traceback (most recent call last):\n  (world-model aggregation error)",
                   "error_type": "", "error_line": 0, "cand_status": ["runtime_err"], "stage": "agg_error"}

    status = agg["status"]
    pos = A.BUCKET_POS.get(agg.get("quality_bucket")) if status == "valid" else None
    fb_full = agg.get("env_feedback", "") or ""
    fb_model = _HDR_RE.sub("", fb_full)                  # drop a leading "=== ... ===" so the loop's prefix is the only one
    _stats["status"][status] += 1
    return {"action": action, "status": status, "reward": agg["reward"], "pos": pos,
            "feedback": fb_model, "feedback_raw": fb_full, "exec_time": round(time.time() - t0, 3),
            "compiles": status not in ("malformed", "syntax_err"),
            "error_type": agg.get("error_type", ""), "error_line": agg.get("error_line", 0),
            "cand_status": agg.get("cand_status"),
            "wm_audit": {"reason": (j or {}).get("reason", ""), "predictions": agg.get("cand_errors"),
                         "n_gen_tok": ngen, "stage": agg.get("stage")}}


# ---- HTTP (aiohttp; one shared event loop with the engine -> concurrent grades continuous-batch) ----
async def _handle_grade(request):
    spec = await request.json()
    items = spec["items"]
    app = request.app
    _stats["inflight"] += len(items)
    _stats["inflight_max"] = max(_stats["inflight_max"], _stats["inflight"])
    try:
        results = await asyncio.gather(*[grade_one(app, it) for it in items])
    finally:
        _stats["inflight"] -= len(items)
    for r in results:
        _stats["grades"] += 1
        _stats["lat"].append(r.get("exec_time", 0.0))
        if "wm_audit" in r and r["wm_audit"].get("n_gen_tok"):
            _stats["gen_tok"].append(r["wm_audit"]["n_gen_tok"])
    _maybe_log()
    from aiohttp import web
    return web.json_response(results)


async def _handle_health(request):
    from aiohttp import web
    return web.json_response({"capacity": CAPACITY, "model": WM_MODEL, "port": PORT, "grades": _stats["grades"],
                              "parse_fail": _stats["parse_fail"], "rwd_top": RWD_TOP, "fdbk_top": FDBK_TOP,
                              "max_tokens": MAX_TOKENS, "inflight": _stats["inflight"]})


async def main():
    from aiohttp import web
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from transformers import AutoTokenizer
    print(f"[wm:{PORT}] loading {WM_MODEL} | tp={TP} gpu_util={GPU_UTIL} max_len={MAX_MODEL_LEN} "
          f"max_tokens={MAX_TOKENS} cap={CAPACITY} eager={ENFORCE_EAGER} rwd_top={RWD_TOP} fdbk_top={FDBK_TOP} "
          f"attn={os.environ.get('VLLM_ATTENTION_BACKEND','default')} "
          f"fi_sampler={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER','default')}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(WM_MODEL, trust_remote_code=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=WM_MODEL, tensor_parallel_size=TP, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN, trust_remote_code=True, enforce_eager=ENFORCE_EAGER, disable_log_stats=True))
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app["engine"] = engine; app["tok"] = tok
    app.router.add_get("/health", _handle_health)
    app.router.add_post("/grade", _handle_grade)
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    print(f"[wm:{PORT}] READY in {time.time()-t0:.0f}s | capacity {CAPACITY}", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
