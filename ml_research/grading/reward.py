"""Custom reward function for the fully_async (hybrid_async) backend.

fully_async's streaming RewardLoopWorker populates rm_scores by calling this per trajectory; it must run (disabling it
-> KeyError('rm_scores')). Our real reward is the AgentLoop WM/sandbox grade, which the reward loop CANNOT see (verified
via DIAG: its data is only task + response_text). So the AgentLoop stashes its grade in a shared Ray actor keyed by the
decoded response (grade_store.py); here we look it up by solution_str == that same decoded response. Signature matches
verl reward_manager/naive.py's call. one_step_off (sandbox/wm/hybrid) does NOT use this (uses reward_score directly)."""
import grade_store

_HIT = [0]
_MISS = [0]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    r = grade_store.get_grade(solution_str)
    if r is not None:
        _HIT[0] += 1
        if (_HIT[0] + _MISS[0]) % 200 == 0:
            print(f"[mledojo_reward] grade-store hits={_HIT[0]} misses={_MISS[0]}", flush=True)
        return float(r)
    # MISS: the AgentLoop's grade wasn't found for this response (key mismatch or a race). Log (rate-limited) and
    # return 0 -- a penalty, not a crash. If misses are common the key needs fixing (watch the hit/miss counts).
    _MISS[0] += 1
    if _MISS[0] <= 6 or _MISS[0] % 50 == 0:
        k = grade_store.grade_key(solution_str)
        print(f"[mledojo_reward] STORE MISS #{_MISS[0]} (hits={_HIT[0]}) key={k[:12]} sol_len={len(solution_str)} "
              f"sol_head={solution_str[:90]!r} sol_tail={solution_str[-60:]!r}", flush=True)
    return 0.0
