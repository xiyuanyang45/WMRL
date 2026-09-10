#!/usr/bin/env python3
"""Score a policy on LIBERO-Long.

Reports the three columns of Table 3. The split between them is the point:

**In-domain** is initial states the policy trained on. **Out-of-distribution** is
initial states held out entirely. **Overall** averages over every initial state,
seen and unseen.

Each initial state gets eight rollouts, which is what makes the two secondary
numbers meaningful. ``Best@8`` counts an initial state solved if any of the
eight succeeded; ``All@8`` counts it solved only if all eight did. They bracket
the average from either side, and the gap between them is how reliable the
policy is rather than how capable: a policy that solves a task one time in eight
and one that solves it every time can share an average.

    python -m embodied.evaluate --checkpoint checkpoints/embodied/wmrl --data data/libero
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROLLOUTS_PER_STATE = 8

__all__ = ["summarize", "main"]


def summarize(successes) -> dict:
    """Turn per-initial-state rollout outcomes into the reported numbers.

    Args:
        successes: mapping of initial-state id to a list of booleans, one per
            rollout of that state.

    Returns:
        ``avg``, ``best_at_8`` and ``all_at_8`` as percentages.
    """
    if not successes:
        return {"avg": 0.0, "best_at_8": 0.0, "all_at_8": 0.0, "n_states": 0}

    flat = [ok for runs in successes.values() for ok in runs]
    best = [any(runs) for runs in successes.values()]
    every = [all(runs) for runs in successes.values()]

    pct = lambda xs: round(100.0 * sum(xs) / len(xs), 1)
    return {
        "avg": pct(flat),
        "best_at_8": pct(best),
        "all_at_8": pct(every),
        "n_states": len(successes),
    }


def _report(results: dict) -> str:
    rows = [
        ("In-Domain", results.get("in_domain", {})),
        ("Out-of-Distribution", results.get("ood", {})),
        ("Overall", results.get("overall", {})),
    ]
    out = [f"{'':<22}{'Avg':>8}{'Best@8':>9}{'All@8':>8}{'states':>9}"]
    for name, r in rows:
        if not r:
            continue
        out.append(
            f"{name:<22}{r['avg']:>8.1f}{r['best_at_8']:>9.1f}"
            f"{r['all_at_8']:>8.1f}{r['n_states']:>9}"
        )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS_PER_STATE)
    ap.add_argument("--out", help="write the full result as JSON")
    ap.add_argument("--tasks", nargs="*", help="evaluate only these task names")
    args = ap.parse_args(argv)

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    try:
        from embodied.rollout_worker import evaluate_policy
    except ImportError as e:
        raise SystemExit(
            f"cannot import the rollout worker ({e}).\n"
            "Evaluation needs the simulator and the policy backbone:\n"
            "  ./embodied/run.sh setup"
        )

    print(f"evaluating {ckpt} with {args.rollouts} rollout(s) per initial state", flush=True)
    results = evaluate_policy(
        checkpoint=str(ckpt),
        data_root=args.data,
        rollouts_per_state=args.rollouts,
        tasks=args.tasks,
    )

    print("\n" + _report(results))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
