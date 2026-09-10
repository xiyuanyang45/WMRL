"""verl AgentLoop port of the rl_mledojo agentic rollout (Stage 2 of the verl build).

Each trajectory = K forced turns of (LLM generates code -> grade in a kaggle sandbox -> append observation), built
at the TOKEN level (response_mask=1 on generated tokens, 0 on observation tokens — exactly the old tool_mask).
We REUSE the proven seams unchanged:
  - agentic_core      : prompt building + action parsing (byte-identical prompts -> on-policy)
  - agentic_env.RemoteEnv -> env_server.py : sandbox grading over HTTP (the dedicated grading node)
The ONLY new thing vs the TRL stack is the trainer/rollout host: verl's async AgentLoop instead of the TRL
colocate SPMD rollout. verl's server_manager.generate is async + token-level, so trajectories run independently
(no per-turn lockstep). Reward = max position-ladder score over the trajectory's execute turns (baseline traj
credit -> AgentLoopOutput.reward_score -> GRPO group advantage).

Registered as agent_name "agentic_mledojo" (see agent_loop_config.yaml); select via
rollout.agent.default_agent_loop=agentic_mledojo. Needs rollout.mode=async and SBX_SERVERS pointing at env_server.
"""
import os
import sys
import json
import time
import asyncio
import threading
from uuid import uuid4
from typing import Any

# the shared seams live in the multi_node_rl dir (one level up); reuse them verbatim
_MNR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MNR not in sys.path:
    sys.path.insert(0, _MNR)
import agentic_core as AC
import agentic_env as ENVMOD
from agentic_core import _derive_turn_wrap, _cap_obs

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

# ---- run knobs (env) — mirror the rl_mledojo recipe ----
K = int(os.environ.get("K", "5"))
MAX_NEW_PER_TURN = int(os.environ.get("MAX_NEW_PER_TURN", "4096"))
OBS_MAX = int(os.environ.get("OBS_MAX", "1200"))
OVERVIEW_MAX = int(os.environ.get("OVERVIEW_MAX", "1500"))
EXEC_CAP = int(os.environ.get("EXEC_CAP", "600"))
SANDBOX_CORES = int(os.environ.get("SANDBOX_CORES", "5"))
# host RAM hard cap per grade (GB), enforced by the dojo as setrlimit(RLIMIT_AS). NOTE: RLIMIT_AS caps VIRTUAL
# address space, and CUDA/torch reserve large virtual ranges to init — so keep this >= ~64GB whenever the sandbox
# may run GPU code, else CUDA init itself OOMs (it is NOT a GPU-memory cap; that's SBX_GPU_MEM_GB).
SANDBOX_MEM_GB = int(os.environ.get("SANDBOX_MEM_GB", "64"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_MNR, "data"))
AC.ENABLE_THINKING = (os.environ.get("THINK", "0") == "1")

# ONE shared grading backend per worker process (RemoteEnv -> env_server over HTTP). Built lazily on first grade
# so importing the registry doesn't require env_server to be up yet. SBX_SERVERS REQUIRED at first use.
_ENV = None
# hybrid (Option A): on grounding steps the batch grades against the REAL-sandbox pool (SBX_SERVERS_SANDBOX) instead
# of the WM pool (SBX_SERVERS), so the agent sees REAL feedback. The trainer tags each batch with non_tensor
# `grade_mode` ('wm'|'sandbox'); run() routes to the matching pool. Both pools speak the identical /grade contract.
_GRADER = os.environ.get("GRADER", "sandbox")
_ADAW_MODES = ("wmrl",)
_ENV_SBX = None

# ADAW r_wm side-grade: computed SYNCHRONOUSLY inside run() for sandbox-graded trajs and carried in the AgentLoopOutput's
# extra_fields -> verl's _postprocess copies every extra_fields key into non_tensor_batch (aligned by traj), so the
# trainer reads r_wm[i] perfectly aligned with grade_mode[i]/traj_reward[i] — NO out-of-band store, NO response-key
# matching, NO produce/consume staleness race (the earlier grade_store+create_task/thread design stored pairs the trainer
# could never pop, eta2_count stuck at 0). Cost: ~K WM grades (gathered, ~3s) added to sandbox trajs only — those already
# take minutes for the real-sandbox grade, so it's marginal. _ADAW_N counts side-grades for a verifiable rollout-log print.
_ADAW_N = [0]
_OVR_CACHE: dict[str, str] = {}     # task -> capped overview (built once per worker, byte-identical to training)

# ---- per-trajectory rollout log (one jsonl line per trajectory) ----
# ROLLOUT_LOG_DIR is set by model_verl.launch_trainer; each rollout-worker process appends to its own pid-keyed file
# (no cross-process lock needed), guarded in-process by a lock since many trajectories run concurrently as coroutines.
_RLOG_DIR = os.environ.get("ROLLOUT_LOG_DIR")
_RLOG_PATH = os.path.join(_RLOG_DIR, f"rollouts_pid{os.getpid()}.jsonl") if _RLOG_DIR else None
_RLOG_LOCK = threading.Lock()


def _rlog(rec: dict):
    """Append one trajectory record. Best-effort: a logging failure must never crash a rollout (fail loud lives in
    the gradient path, not the telemetry path)."""
    if not _RLOG_PATH:
        return
    try:
        line = json.dumps(rec, ensure_ascii=False)
        with _RLOG_LOCK:
            with open(_RLOG_PATH, "a") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[agentic_mledojo_loop] rollout-log write failed: {e}", flush=True)


def _env():
    global _ENV
    if _ENV is None:
        _ENV = ENVMOD.make_env("remote", None)            # SBX_SERVERS (WM pool for hybrid; single pool otherwise)
    return _ENV


def _env_sbx():
    """hybrid grounding pool: the REAL sandbox env_server(s) at SBX_SERVERS_SANDBOX. Built lazily on first use."""
    global _ENV_SBX
    if _ENV_SBX is None:
        servers = [s.strip() for s in os.environ.get("SBX_SERVERS_SANDBOX", "").split(",") if s.strip()]
        if not servers:
            raise RuntimeError("hybrid: SBX_SERVERS_SANDBOX empty — no real-sandbox grounding pool")
        _ENV_SBX = ENVMOD.RemoteEnv(servers)
    return _ENV_SBX


def _env_for(grade_mode):
    """Pick the grading pool for this trajectory. hybrid/ADAW modes: 'sandbox'->real sandbox, else WM pool."""
    if _GRADER == "wmrl" and grade_mode == "sandbox":
        return _env_sbx()
    return _env()


async def _adaw_wm_grade(task, codes, overview):
    """ADAW: WM-grade each execute-turn code via the WM pool (concurrently), return the MAX (mirrors r_sbx = max over
    execute turns). Awaited inside run() before the output is built, so r_wm is carried aligned in extra_fields. Returns
    None if every grade fails/None (consumer skips Nones). Best-effort — never raises into the rollout."""
    try:
        wm_env = _env()                          # WM pool only — never _env_sbx()

        async def _g(code):
            item = {"task": task, "data_dir": os.path.join(DATA_ROOT, task, "data"),
                    "action": "execute_code", "code": code, "overview": overview,
                    "limits": {"execution_timeout": EXEC_CAP, "cpu_time_limit": SANDBOX_CORES * EXEC_CAP,
                               "memory_limit": SANDBOX_MEM_GB}}
            res = (await asyncio.to_thread(wm_env.step, [item], 1))[0]
            r = res.get("reward")
            return float(r) if r is not None else None

        rs = await asyncio.gather(*[_g(c) for c in codes], return_exceptions=True)
        vals = [r for r in rs if isinstance(r, float)]
        if vals:
            _ADAW_N[0] += 1
            if _ADAW_N[0] <= 8 or _ADAW_N[0] % 64 == 0:
                print(f"[adaw] r_wm computed #{_ADAW_N[0]} = {max(vals):.3f} (n_codes={len(codes)})", flush=True)
            return max(vals)
    except Exception as e:
        print(f"[adaw] r_wm grade best-effort failure: {e}", flush=True)
    return None


def _overview(tok, task):
    if task not in _OVR_CACHE:
        _OVR_CACHE[task] = AC.build_overview(tok, task, os.path.join(DATA_ROOT, task, "data"), max_tokens=OVERVIEW_MAX)
    return _OVR_CACHE[task]


@register("agentic_mledojo")
class AgenticMLEDojoLoop(AgentLoopBase):
    """Forced-K multi-turn MLE-Dojo agentic rollout on verl's async LLM server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        # chat-template pieces between turns, derived from the tokenizer (reproduces the gen-prompt primer exactly)
        self.ASST_OPEN, self.IM_END, self.USER_OPEN = _derive_turn_wrap(self.tokenizer, AC.ENABLE_THINKING)

    async def _grade(self, env, task, action, payload, overview):
        """Grade one turn's action via `env` (a RemoteEnv; blocking HTTP -> run off the event loop). Returns the full
        result dict (status/reward/pos/feedback + per-grade cpu_s/peak_rss_mb/exec_time from the env node).
        `overview` (the task+data inventory the agent saw) is sent in the item: the SANDBOX grader ignores it, the
        WM grader (wm_server.py) NEEDS it to predict the outcome — same item shape, one backend reads it. `env` is
        chosen per-batch by run(): the WM pool, or (hybrid grounding steps) the REAL-sandbox pool."""
        if action == "malformed":
            return {"status": "malformed", "reward": 0.0, "pos": None,
                    "feedback": "No valid action found. Emit a ```python``` code block."}
        item = {"task": task, "data_dir": os.path.join(DATA_ROOT, task, "data"), "action": action,
                "overview": overview,
                "limits": {"execution_timeout": EXEC_CAP, "cpu_time_limit": SANDBOX_CORES * EXEC_CAP,
                           "memory_limit": SANDBOX_MEM_GB}}
        if action == "execute_code":
            item["code"] = payload
        res = (await asyncio.to_thread(env.step, [item], 1))[0]
        res["feedback"] = "=== Execution result ===\n" + res["feedback"]
        return res

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        task = kwargs["raw_prompt"][-1]["content"] if isinstance(kwargs["raw_prompt"], list) else kwargs["raw_prompt"]
        # Pick this trajectory's grading pool ONCE (consistent feedback source across all K turns). grade_mode is
        # stamped per-trajectory by the backend, and read here from kwargs (non_tensor field -> run() kwarg):
        # - hybrid (one_step_off): the trainer stamps grade_mode per-batch (whole batch sandbox on grounding steps).
        # - fully async: the rollouter (patch-7) stamps it per-GROUP via the shared sbx_gate — the whole
        #   GRPO group goes sandbox iff the gate reserved NUM_GEN sandbox slots for it, else WM. GROUP-pure (group-norm
        #   needs one reward scale) + demand-driven (sandbox flat-out, no oversubscribe; overflow -> fast WM).
        grade_mode = kwargs.get("grade_mode", "wm")
        env = _env_for(grade_mode)
        tok = self.tokenizer
        ovr = _overview(tok, task)
        prompt_ids = tok(AC.build_turn_prompt(tok, task, ovr, [], K), add_special_tokens=False).input_ids

        seq_ids = list(prompt_ids)
        resp_ids: list[int] = []
        resp_mask: list[int] = []
        resp_lps: list[float] = []
        best_reward, best_pos = 0.0, None
        statuses = []
        turns: list[dict] = []                            # per-turn telemetry for the rollout log
        sp = dict(sampling_params)
        sp["max_tokens"] = MAX_NEW_PER_TURN
        sp["logprobs"] = 1
        metrics: dict[str, Any] = {}

        for t in range(K):                                # forced-K (no silent early-break): fail loud below if over budget
            _tg = time.time()
            with simple_timer("generate_sequences", metrics):
                out: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=seq_ids, sampling_params=sp)
            gen_s = round(time.time() - _tg, 2)
            gen_ids = out.token_ids
            # FAIL LOUD: in one_step_off BYPASS mode (bypass_mode=True, the entrypoint default we run — verified in
            # real20/real50 resolved config) these behavior-policy (rollout) logprobs BECOME old_log_probs, so the actor
            # loss ratio is exp(actor_lp - rollout_lp). They are required for the GRADIENT, not just metrics
            # (apply_bypass_mode itself raises if they're absent). Zero-filling silently corrupts the ratio (a real
            # logprob is <=0; a fake 0.0 means behavior "prob 1"). vLLM with calculate_log_probs=True+logprobs=1 must
            # always return one logprob per generated token.
            if out.log_probs is None or len(out.log_probs) != len(gen_ids):
                raise RuntimeError(
                    f"vLLM returned no/misaligned logprobs "
                    f"({None if out.log_probs is None else len(out.log_probs)} for {len(gen_ids)} tokens) — "
                    f"required as old_log_probs for one_step_off bypass-mode PPO ratio")
            gen_lps = out.log_probs
            resp_ids += gen_ids
            resp_mask += [1] * len(gen_ids)               # model-generated -> trained
            resp_lps += gen_lps
            seq_ids += gen_ids

            gen_text = tok.decode(gen_ids, skip_special_tokens=False)   # FULL completion (analysis + code) — logged for audit
            action, payload = AC.parse_action(gen_text)
            _tgr = time.time()
            with simple_timer("tool_calls", metrics):
                gres = await self._grade(env, task, action, payload, ovr)
            grade_s = round(time.time() - _tgr, 2)
            status, reward, pos, obs = gres["status"], gres["reward"], gres.get("pos"), gres["feedback"]
            statuses.append(status)
            # FULL audit record per turn — nothing truncated: raw model output, parsed action+code, and the
            # COMPLETE untruncated env feedback (feedback_raw from the grader). obs_to_model (capped) is added below.
            turn_rec = {"turn": t, "action": action, "gen_s": gen_s, "grade_s": grade_s,
                        # abs epoch timestamps for the per-turn Gantt: gen [t_gen_start, +gen_s], grade [t_grade_start, +grade_s]
                        "t_gen_start": round(_tg, 3), "t_grade_start": round(_tgr, 3), "t_done": round(time.time(), 3),
                        "grade_mode": grade_mode,
                        "n_gen_tok": len(gen_ids), "status": status, "reward": reward, "pos": pos,
                        "cpu_s": gres.get("cpu_s"), "peak_rss_mb": gres.get("peak_rss_mb"),
                        "peak_gpu_mb": gres.get("peak_gpu_mb"), "exec_time": gres.get("exec_time"),
                        "text": gen_text, "code": payload if action == "execute_code" else None,
                        "feedback_raw": gres.get("feedback_raw", obs), "obs_to_model": None,
                        "wm_audit": gres.get("wm_audit")}   # WM-only: reason+ranked predictions (None for sandbox grades)
            turns.append(turn_rec)
            if action == "execute_code" and reward > best_reward:
                best_reward, best_pos = reward, pos

            if t < K - 1:                                 # append (env obs + next gen-prompt) tokens, MASKED
                obs = _cap_obs(tok, obs, OBS_MAX)
                turn_rec["obs_to_model"] = obs            # exactly what the model saw as the next turn's observation
                delta = tok(self.IM_END + self.USER_OPEN + obs + self.IM_END + self.ASST_OPEN,
                            add_special_tokens=False).input_ids
                resp_ids += delta
                resp_mask += [0] * len(delta)             # observation -> masked from loss
                resp_lps += [0.0] * len(delta)
                seq_ids += delta

        # FAIL LOUD: never silently truncate the trajectory to fit the budget — that drops generated tokens while
        # best_reward still credits them (off-policy mismatch). The original rollout treated truncation as a
        # correctness bug. At the shipped cfg (K×MAX_NEW + obs < max_response_length) this never trips; it raises the
        # moment K/MAX_NEW_PER_TURN are scaled toward the real recipe, so the budget gets fixed instead of silently
        # corrupting the gradient.
        n = self.response_length
        if len(resp_ids) > n:
            raise RuntimeError(
                f"trajectory length {len(resp_ids)} > response_length {n} (K={K}, "
                f"MAX_NEW_PER_TURN={MAX_NEW_PER_TURN}, OBS_MAX={OBS_MAX}) — refusing to silently truncate off-policy. "
                f"Raise data.max_response_length or lower K / MAX_NEW_PER_TURN / OBS_MAX so a full K-turn trajectory fits.")
        # FULL trajectory record: the complete initial prompt (system + data overview + task, untruncated) + every
        # turn's raw model output / parsed code / raw env feedback. ts lets post-hoc bucketing by training step
        # (one step's trajectories are logged in a contiguous burst). Reconstruct the whole conversation from
        # prompt + per-turn {text, obs_to_model}.
        _rlog({"ts": time.time(), "task": task, "grade_mode": grade_mode,
               "prompt": tok.decode(prompt_ids, skip_special_tokens=False),
               "overview": ovr, "traj_reward": best_reward, "best_pos": best_pos, "n_turns": len(statuses),
               "statuses": statuses, "n_prompt": len(prompt_ids), "n_comp": len(resp_ids),
               "n_unmasked": sum(resp_mask), "turns": turns})
        # ADAW: for sandbox-graded trajs, WM-grade the same execute-turn codes -> r_wm, carried ALIGNED in extra_fields
        # (verl _postprocess copies extra_fields -> non_tensor_batch[i], so the trainer reads r_wm[i] next to
        # grade_mode[i]/traj_reward[i] — no out-of-band store/key/race). Synchronous (awaited) so it's ready at return.
        r_wm = None
        if _GRADER == "wmrl" and grade_mode == "sandbox":
            _codes = [tr["code"] for tr in turns if tr["action"] == "execute_code" and tr.get("code")]
            if _codes:
                r_wm = await _adaw_wm_grade(task, _codes, ovr)
        if _GRADER == "wmrl":
            # bridge our grade to the fully_async streaming reward loop (different actor, can't see reward_score):
            # key by the decoded response == the reward loop's solution_str. See grade_store.py / mledojo_reward.py.
            try:
                import grade_store
                # WMRL (wmrl): Path B. SANDBOX (anchor) groups push (r_wm, r_sbx) pairs to fit f and
                # keep ground-truth reward. WM groups get the recalibrated reward f(r_wm) (bias removed). f=identity
                # until RECAL_MIN_PAIRS anchor pairs seen ⇒ opening behaviour ≡ B (an abandoned variant). Monotone f ⇒ f(max)=
                # max(f), so applying to best_reward (= max-over-turns WM grade) is valid.
                if _GRADER == "wmrl":
                    if grade_mode == "sandbox":
                        if r_wm is not None:
                            grade_store.push_recal_pair(r_wm, best_reward)
                    else:
                        best_reward = grade_store.recal(best_reward)
                grade_store.put_grade(tok.decode(resp_ids, skip_special_tokens=True), best_reward)
                # release this trajectory's sandbox-gate slot (reserved per-group by the rollouter). Only sandbox
                # trajectories hold one; WM ones don't. A leaked slot only biases routing toward WM (safe), never stalls.
                if grade_mode == "sandbox":
                    grade_store.sbx_release(1)
            except Exception as _e:
                print(f"[agentic_mledojo_loop] grade_store put/release failed: {_e}", flush=True)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=resp_ids,
            response_mask=resp_mask,
            response_logprobs=resp_lps,
            reward_score=best_reward,                     # baseline trajectory credit -> GRPO group advantage
            num_turns=len(statuses),
            metrics=AgentLoopMetrics(**metrics),
            # traj_reward carries the agentic grade into tool_extra_fields -> the reward loop's extra_info, so the
            # fully_async  reward loop returns OUR grade via custom_reward_function (mledojo_reward.py)
            # instead of crashing on default_compute_score(data_source='mledojo'). one_step_off ignores it (uses
            # reward_score directly). best_reward == reward_score, so it's a no-op there.
            # r_wm (ADAW): WM-grade of the sandbox traj's codes; None for WM-routed/non-ADAW trajs. _postprocess copies
            # it into non_tensor_batch["r_wm"][i], aligned with grade_mode/traj_reward, for the patch-7d η² consumer.
            extra_fields={"best_pos": best_pos, "statuses": statuses, "traj_reward": best_reward, "r_wm": r_wm},
        )


# run() must be traced like the reference loops; apply verl's decorator explicitly (avoids import-order issues)
from verl.utils.rollout_trace import rollout_trace_op  # noqa: E402
AgenticMLEDojoLoop.run = rollout_trace_op(AgenticMLEDojoLoop.run)
