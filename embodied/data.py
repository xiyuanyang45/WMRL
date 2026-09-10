#!/usr/bin/env python3
"""Fetch the LIBERO-Long demonstrations.

LIBERO-Long is ten long-horizon manipulation tasks, each with fifty human
demonstrations. The demonstrations are used twice: once for behaviour cloning,
which produces the checkpoint every reinforcement-learning run starts from, and
once to define the initial states an evaluation samples from.

The split between seen and unseen initial states is what makes the
out-of-distribution column meaningful. It is fixed by seed here rather than
chosen per run, so two runs are comparable.

    python -m embodied.data --out data/libero
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# The benchmark suite this track trains and evaluates on.
SUITE = "libero_10"
N_TASKS = 10
DEMOS_PER_TASK = 50

# Initial states held out of training entirely. Fixed so the out-of-distribution
# number means the same thing across runs.
HELDOUT_INITIAL_STATES = 10
SPLIT_SEED = 0

__all__ = ["fetch", "describe_split", "main"]


def _looks_complete(root: pathlib.Path) -> bool:
    hdf5 = list(root.rglob("*.hdf5"))
    return len(hdf5) >= N_TASKS


def fetch(out: pathlib.Path) -> pathlib.Path:
    """Download the suite's demonstrations into ``out``."""
    try:
        from libero.libero import benchmark, get_libero_path
    except ImportError:
        raise SystemExit(
            "LIBERO is not installed. It carries the simulator, the task suite and\n"
            "the demonstrations, and is not pinned in this repository because its\n"
            "rendering stack depends on your machine:\n"
            "  pip install -r requirements-train.txt\n"
            "  pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO\n"
            "See embodied/README.md."
        )

    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    dataset_root = pathlib.Path(get_libero_path("datasets"))

    print(f"{SUITE}: {suite.n_tasks} task(s), demonstrations under {dataset_root}", flush=True)
    for i in range(suite.n_tasks):
        name = suite.get_task(i).name
        src = dataset_root / SUITE / f"{name}_demo.hdf5"
        dst = out / f"{name}_demo.hdf5"
        if dst.exists():
            print(f"  [{i + 1}/{suite.n_tasks}] {name}: present", flush=True)
            continue
        if not src.exists():
            raise SystemExit(
                f"missing demonstration file {src}.\n"
                "LIBERO downloads these separately; see its README for the download step."
            )
        dst.symlink_to(src)
        print(f"  [{i + 1}/{suite.n_tasks}] {name}: linked", flush=True)

    return out


def describe_split() -> str:
    return (
        f"Each task has {DEMOS_PER_TASK} demonstrations. The last "
        f"{HELDOUT_INITIAL_STATES} initial states of each are held out of training "
        f"(seed {SPLIT_SEED}) and form the out-of-distribution column; the rest are "
        "in-domain. Evaluation runs eight rollouts per initial state."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="report what is present and stop")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)

    if args.check:
        print(f"{out}: {'complete' if _looks_complete(out) else 'incomplete or absent'}")
        print(describe_split())
        return 0 if _looks_complete(out) else 1

    fetch(out)
    print(f"\n{describe_split()}")
    print(f"\nNext: ./embodied/run.sh sft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
