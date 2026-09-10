#!/usr/bin/env python3
"""Held-out OOD evaluation of a saved checkpoint (or base model) with REAL-sandbox grading.

WHY THIS FILE EXISTS (and is not rl_mledojo/eval_ckpt.py): the 3 runs we evaluate were TRAINED with
multi_node_rl/agentic_core.py (system prompt carries the GPU+library ENV_BRIEF; build_overview adds the
pred-vs-copy submission note and token-caps the inventory). rl_mledojo/agentic_core.py is an OLDER, DIFFERENT
prompt builder. If the eval built prompts with the wrong agentic_core, the held-out prompts would be
off-distribution from training and the experiment invalid. So this harness routes EVERY `agentic_*` import to
multi_node_rl's versions (sys.path below) — proven byte-identical to training by verify_byte_identity.py.

The rollout orchestrator (offline vLLM, forced-K, obs/turn wrapping) is rl_mledojo's agentic_grpo_rollout.py,
copied next to this file. It imports `agentic_core` by bare name -> resolves to multi_node_rl's; its turn
wrapping (_derive_turn_wrap / _cap_obs / IM_END+USER_OPEN+obs+IM_END+ASST_OPEN_PRIMER, P0 = build_turn_prompt(
tok, task, ovr, [], K)) is byte-identical to verl_agentic/agentic_mledojo_loop.py:run, so the eval rollout
matches the training rollout token-for-token.

Defaults mirror the training config, so evaluation matches training:
  K=4, OVERVIEW_MAX=1500, OBS_MAX=1024, MAX_NEW_PER_TURN=4096, THINK=0 (ENABLE_THINKING=False), EXEC_CAP=600.

GRADING BACKEND = the EXACT path training used: a REMOTE GPU env_server (RemoteEnv). Training graded on GPU
(cfg SANDBOX_NO_GPU=0, SLOTS_PER_GPU=2, SBX_GPU_MEM_GB=19; env_server runs each grade with
CUDA_VISIBLE_DEVICES=<card> so the model's CNN code trains on a real GPU). Earlier this harness graded on
CPU (SANDBOX_NO_GPU=1) which made image comps (dogs-vs-cats, dog-breed) UNFAITHFUL — CPU is slow, the
submission code falls back / times out at the 600s cap. We now POST every grade to a GPU env_server via
make_env("remote", None) driven by SBX_SERVERS, matching training byte-for-byte on the grading side too.
The eval process owns NO sandbox pool; it only runs vLLM on its own card and POSTs grades. The env_server's
GPU/cores/threads/mem knobs live on the SERVER process (run_ood_evals_gpu.sh sets them to the training cfg).

Run (one process per ckpt — vLLM owns its eval GPU; the env_server owns the grading cards separately):
  ENV_IMPL=remote SBX_SERVERS=127.0.0.1:18200 \
  CUDA_VISIBLE_DEVICES=4 MODEL=<ckpt_dir_or_base> TAG=se4_step20 \
    TASKS=random-acts-of-pizza G=4 \
    /home/sagemaker-user/xiyuan_work_dir/envs/vllm-env/bin/python3.11 eval_ckpt_ood.py
Env: MODEL (req), TAG, TASKS (csv), G (8), K (4), MAX_NEW_PER_TURN (4096), MAXCTX (26000),
     OBS_MAX (1024), OVERVIEW_MAX (1500), EXEC_CAP (600), OUT (jsonl),
     ENV_IMPL=remote (req), SBX_SERVERS=host:port[,host:port] (req — the GPU env_server(s)),
     SANDBOX_CORES (6 — only used to size the item's cpu_time_limit; the SERVER's SANDBOX_CORES does the pinning),
     GRADE_CONC (max concurrent in-flight grades; default = server capacity).
"""
import os, sys, json, time
from pathlib import Path

# ---- env defaults set BEFORE importing agentic_* ----
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")          # A100 triton path (validated)
# SANDBOX_CORES here ONLY sizes the per-item cpu_time_limit (= SANDBOX_CORES * EXEC_CAP), matching the training
# loop (agentic_mledojo_loop.py:136 uses SANDBOX_CORES * EXEC_CAP); the actual CPU pinning / GPU enablement happens
# on the env_server. Default 6 = training cfg's SANDBOX_CORES.
os.environ.setdefault("SANDBOX_CORES", "6")
# grader makes an ISOLATED data copy per grade (mkdtemp) on the SERVER; the eval process never grades locally, but
# keep TMPDIR pointed at the big volume in case any incidental tmp use occurs.
os.environ.setdefault("TMPDIR", "/home/sagemaker-user/xiyuan_work_dir/tmp_grader")
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

# ---- ROUTE ALL agentic_* IMPORTS TO MULTI_NODE_RL (the version the runs trained with) ----
MNR = Path("/home/sagemaker-user/xiyuan_work_dir/auto_research/multi_node_rl")
sys.path.insert(0, str(Path(__file__).resolve().parent))           # eval_ood/ (has agentic_grpo_rollout.py)
sys.path.insert(0, str(MNR))                                       # multi_node_rl/ (agentic_core/env/sched/...)

MODEL = os.environ["MODEL"]                                         # ckpt dir or base — REQUIRED, no default
TAG = os.environ.get("TAG") or Path(MODEL).name
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(MNR / "data")))    # default: multi_node_rl/data (training root)
# default held-out comps that are already loop-ready under rl_mledojo/data; overridable via TASKS
DEFAULT_TASKS = "random-acts-of-pizza,tabular-playground-series-dec-2021,new-york-city-taxi-fare-prediction"
TASKS = [t for t in os.environ.get("TASKS", DEFAULT_TASKS).split(",") if t]
G = int(os.environ.get("G", "8"))                                   # samples/task = avg@G (noise ~ 1/sqrt(G))
K = int(os.environ.get("K", "4"))                                   # se4_long trained with K=4
EVAL_TEMP = float(os.environ.get("EVAL_TEMP", "0.8"))              # eval-time sampling temp (NOT the train temp 1.0)
MAX_NEW = int(os.environ.get("MAX_NEW_PER_TURN", "4096"))
MAXCTX = int(os.environ.get("MAXCTX", "26000"))
OBS_MAX = int(os.environ.get("OBS_MAX", "1024"))                    # se4_long trained with OBS_MAX=1024
OVERVIEW_MAX = int(os.environ.get("OVERVIEW_MAX", "1500"))
EXEC_CAP = int(os.environ.get("EXEC_CAP", "600"))
OUT = Path(os.environ.get("OUT", str(MNR / "eval_ood" / "logs" / f"{TAG}.jsonl")))
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    import agentic_core as AC
    import agentic_grpo_rollout as AGR
    import agentic_env
    assert str(MNR) in AC.__file__, f"WRONG agentic_core resolved: {AC.__file__} (must be multi_node_rl's)"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    AC.ENABLE_THINKING = (os.environ.get("THINK", "0") == "1")     # se4_long: THINK unset -> False
    print(f"agentic_core   : {AC.__file__}")
    print(f"rollout host   : {AGR.__file__}")
    print(f"ENABLE_THINKING: {AC.ENABLE_THINKING}  K={K}  OVERVIEW_MAX={OVERVIEW_MAX}  OBS_MAX={OBS_MAX}")
    print(f"EVAL sampling  : temp={EVAL_TEMP} top_p=1.0  avg@G={G}  EXEC_CAP={EXEC_CAP}s")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # CHAT_EXTRA_KW ("k=v[,k=v]") injects extra chat-template kwargs into EVERY render (P0 build_turn_prompt,
    # _derive_turn_wrap, history re-renders). Needed for gpt-oss: its harmony template cannot disable reasoning
    # (no enable_thinking); reasoning_effort=low is the closest analog to the THINK=0 convention every other
    # baseline runs under. Injected as a DEFAULT (setdefault) so explicit call-site kwargs still win.
    # GEN_PRIMER_SUFFIX is appended to every add_generation_prompt=True render. For gpt-oss this is
    # '<|channel|>final<|message|>' — harmony has NO thinking off-switch, so we FORCE the final channel at every
    # generation point (the standard skip-reasoning technique), aligning it with the THINK=0 non-thinking setting
    # all other baselines run under. Because _derive_turn_wrap derives TURN_TAIL from these same renders, the
    # forcing flows into P0 AND every turn wrap consistently (incremental build stays == full template render).
    _ckw = os.environ.get("CHAT_EXTRA_KW", "")
    _gps = os.environ.get("GEN_PRIMER_SUFFIX", "")
    if _ckw or _gps:
        _extra = dict(p.split("=", 1) for p in _ckw.split(",") if "=" in p)
        _orig_actt = tok.apply_chat_template
        def _actt(*a, **kw):
            for _k, _v in _extra.items():
                kw.setdefault(_k, _v)
            out = _orig_actt(*a, **kw)
            if _gps and kw.get("add_generation_prompt") and isinstance(out, str):
                out += _gps
            return out
        tok.apply_chat_template = _actt
        print(f"chat-template extra kwargs: {_extra}  gen_primer_suffix={_gps!r}")
    data_map, ovr_map = {}, {}
    from mledojo.competitions import get_metric
    for t in TASKS:
        d = DATA_ROOT / t / "data"
        if not (d / "public" / "sample_submission.csv").exists():
            raise SystemExit(f"task {t}: no prepared data at {d}")
        # MLE-bench-Lite tasks (`mleb__<slug>`) are deliberately NOT in MLE-Dojo's registry — they are
        # graded by MLE-bench's OFFICIAL grade_fn via mlebench_metric, which agentic_grader selects on
        # the same prefix. Validate THAT path here instead, so a broken MLE-bench task still fails in
        # seconds rather than after the model load.
        import mlebench_metric as _mlb
        if _mlb.is_mlebench_task(t):
            try:
                _mlb.make_metric_class(t)
            except Exception as _e:
                raise SystemExit(f"task {t}: mlebench grader unavailable ({type(_e).__name__}: {_e})")
        elif get_metric(t) is None:                                # fail in seconds, not after model load
            raise SystemExit(f"task {t}: not in dojo registry (get_metric None)")
        data_map[t] = d
        ovr_map[t] = AC.build_overview(tok, t, str(d), max_tokens=OVERVIEW_MAX)

    # GRADING BACKEND: REMOTE GPU env_server (RemoteEnv) — the exact path training used. No local SandboxPool: the
    # eval process holds no grading capacity, it POSTs every grade to the env_server(s) named in SBX_SERVERS (set by
    # run_ood_evals_gpu.sh). make_env("remote", None) probes each server's /health at startup and FAILS LOUD if a
    # server is unreachable/zero-capacity (a swallowed grading error would record fake reward=0). The server runs the
    # SAME agentic_grader on GPU (SANDBOX_NO_GPU=0), so image-comp CNN code trains on a real card — faithful to train.
    env_impl = os.environ.get("ENV_IMPL", "remote")
    if env_impl != "remote":
        raise SystemExit(f"ENV_IMPL={env_impl!r}: this harness grades via the GPU env_server (ENV_IMPL=remote). "
                         f"Set ENV_IMPL=remote SBX_SERVERS=host:port (see run_ood_evals_gpu.sh).")
    env = agentic_env.make_env("remote", None)   # reads SBX_SERVERS; fail-loud on unreachable server
    print(f"grading backend: REMOTE env_server(s) {env.describe()}  total_capacity={env.capacity}")

    # TP/GPU_MEM_UTIL/QUANT env-configurable for large baseline models (default = the trained-4B single-card setup).
    _tp = int(os.environ.get("TP", "1"))
    _pp = int(os.environ.get("PP", "1"))   # pipeline-parallel (prequant-bnb models need PP, not TP, in vLLM 0.21)
    _gmu = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
    _quant = (os.environ.get("QUANT", "") or "").strip() or None
    _cpu_off = float(os.environ.get("CPU_OFFLOAD_GB", "0") or "0")   # >0 => offload N GB of weights to CPU so a big
    _llm_kw = dict(model=MODEL, tensor_parallel_size=_tp, pipeline_parallel_size=_pp,     # 4bit-MoE model fits at TP=1
                   gpu_memory_utilization=_gmu, max_model_len=MAXCTX, trust_remote_code=True)  # on one 40GB card (TP=1 keeps
    if _quant:                                                                            # MoE dims unsharded->Marlin ok)
        _llm_kw["quantization"] = _quant
    if _cpu_off > 0:
        _llm_kw["cpu_offload_gb"] = _cpu_off
    if _quant == "bitsandbytes":   # bnb isn't in the staged verl-env; install on-demand (ONLY for the bnb baseline)
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            import subprocess
            print("[eval] bitsandbytes missing -> pip install (bnb baseline only)", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.48.1"], check=True)
    print(f"[eval] vLLM LLM(tp={_tp}, gpu_mem_util={_gmu}, quant={_quant}, cpu_offload_gb={_cpu_off}, max_len={MAXCTX}) model={MODEL}", flush=True)
    llm = LLM(**_llm_kw)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end, int) or im_end < 0:   # non-Qwen models lack <|im_end|> (e.g. Llama4 turn-end = <|eot|> = its
        im_end = tok.eos_token_id                    # eos) -> stop at the tokenizer's EOS instead of a None stop_token_id
        # Non-Qwen models may signal end-of-turn with an eos that is NOT tokenizer.eos_token: GLM-4.5 emits
        # <|user|> (151336) / <|observation|> (151338) while eos_token is <|endoftext|> (151329). Stopping only at
        # eos_token would let it ramble past its turn into hallucinated user turns. Use the FULL eos set from the
        # model's generation_config.json (the model author's declared stop set). Qwen path above is untouched.
        stop_ids = {im_end}
        _gc = Path(MODEL) / "generation_config.json"
        if _gc.exists():
            _ge = json.loads(_gc.read_text()).get("eos_token_id")
            if _ge is not None:
                stop_ids |= {_ge} if isinstance(_ge, int) else set(_ge)
        stop_ids = sorted(stop_ids)
    else:
        stop_ids = [im_end]                          # Qwen: byte-identical to the historical behavior

    _n = [0]
    def generate_fn(seq_list):
        # EVAL sampling temperature (EVAL_TEMP, default 0.8) — a DELIBERATE eval-time choice, NOT the training temp.
        # Training rolled out at temp 1.0 (full exploration distribution); for held-out *evaluation* we sample at a
        # slightly sharpened temp 0.8 to estimate the policy's quality with lower variance (standard eval convention).
        # Applied UNIFORMLY to every cell, so the hybrid-vs-sandbox / 4B-vs-9B comparison stays fair; the one caveat
        # is that this is not the exact training-time sampling distribution. top_p=1.0 (pure temperature scaling).
        # Per-call seeds make reruns reproducible. Stop at im_end (rollout appends the IM_END text itself).
        sps = [SamplingParams(temperature=EVAL_TEMP, top_p=1.0, max_tokens=MAX_NEW,
                              stop_token_ids=stop_ids, seed=1000 * _n[0] + i)
               for i in range(len(seq_list))]
        _n[0] += 1
        outs = llm.generate([{"prompt_token_ids": s} for s in seq_list], sps)
        return [(list(o.outputs[0].token_ids), [0.0] * len(o.outputs[0].token_ids)) for o in outs]

    # grade_conc = max grades in flight from THIS eval. Default = the server's advertised capacity (RemoteEnv's
    # semaphore is the real enforcer; it blocks extra POSTs until a slot frees). Override via GRADE_CONC if running
    # several eval procs against ONE shared env_server, to leave slots for the others.
    grade_conc = int(os.environ.get("GRADE_CONC", str(env.capacity)))
    tasks_rep = [t for t in TASKS for _ in range(G)]
    t0 = time.time()
    results = AGR.rollout_batch(tok, generate_fn, tasks_rep, data_map, ovr_map, K=K,
                                max_new_per_turn=MAX_NEW,
                                limits={"execution_timeout": EXEC_CAP,
                                        "cpu_time_limit": int(os.environ["SANDBOX_CORES"]) * EXEC_CAP},
                                max_ctx=MAXCTX, max_traj=None, obs_max=OBS_MAX,
                                grade_conc=grade_conc, env=env)
    wall = time.time() - t0

    # FULL_OUT=1 (the training job entry sets it): write the COMPLETE per-traj record — prompt text, full decoded
    # output, per-turn timing/status/reward — so the OOD curve is auditable from S3 (not just the slim summary).
    # peak_gpu is per-GRADE and lives in the env_server log (shipped to S3 by the entry); the GPU-grading proof is
    # there. This changes ONLY what is serialized — prompts/sampling/grading are untouched (fidelity invariant).
    full_out = os.environ.get("FULL_OUT", "0") == "1"
    with open(OUT, "w") as fh:
        for r in results:
            rec = {k: r[k] for k in ("task", "traj_reward", "best_pos", "n_turns", "statuses", "turn_times")}
            rec["model"], rec["tag"] = MODEL, TAG
            if full_out:
                p0 = tok.decode(r["prompt_ids"], skip_special_tokens=False)
                comp = tok.decode(r["completion_ids"], skip_special_tokens=False)
                rec["prompt"] = p0
                rec["output"] = comp                      # full multi-turn completion (gen + masked obs/template)
                rec["seq_text"] = r.get("seq_text", "")   # prompt+completion as one decoded string
                rec["n_prompt_tok"] = len(r["prompt_ids"])
                rec["n_comp_tok"] = len(r["completion_ids"])
            fh.write(json.dumps(rec) + "\n")

    print(f"\n== {TAG}  ({len(results)} trajs, wall {wall/60:.1f} min)")
    overall_v, overall_p = [], []
    for t in TASKS:
        rs = [r for r in results if r["task"] == t]
        valid = [r for r in rs if any(s == "valid" for s in r["statuses"])]
        pos = [r["best_pos"] for r in valid if r["best_pos"] is not None]
        overall_v += [len(valid) / len(rs)]; overall_p += pos
        if pos:
            print(f"   {t[:42]:42s} valid {len(valid)}/{len(rs)}  best_pos={sum(pos)/len(pos):.3f}")
        else:
            print(f"   {t[:42]:42s} valid {len(valid)}/{len(rs)}  best_pos=n/a")
    if overall_p:
        print(f"   OVERALL valid-rate={sum(overall_v)/len(overall_v):.3f}  "
              f"mean best_pos(valid)={sum(overall_p)/len(overall_p):.3f}")
    else:
        print(f"   OVERALL valid-rate={sum(overall_v)/len(overall_v):.3f}  no valid trajs")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
