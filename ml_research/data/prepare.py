#!/usr/bin/env python3
"""Download and prepare the benchmark tasks.

Two benchmarks, both built on real Kaggle competitions:

**MLE-Dojo** is the training set and the in-distribution test set. Each task is
prepared into a public half the agent sees (description, training data, a
sample submission) and a private half it never sees (the held-out answers the
grader scores against). Preparation is per-competition and can fail for reasons
outside this repository, mostly rules that have to be accepted on the
competition page before its data can be downloaded, so failures are collected
and reported at the end rather than aborting the run.

**DSBench** is the out-of-distribution test set, disjoint from everything
trained on. It is never prepared here for training, only for evaluation.

Preparation is resumable. A task whose public and private halves are both
present is skipped, so an interrupted run continues where it stopped.

    python -m ml_research.data.prepare --tasks ml_research/configs/tasks_train.txt --out data
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

__all__ = ["read_task_list", "is_prepared", "prepare_task", "main"]


def read_task_list(path: str) -> list[str]:
    """One competition slug per line; blank lines and ``#`` comments ignored."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"task list not found: {path}")
    out = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.split()[-1])
    return out


def is_prepared(root: pathlib.Path, slug: str) -> bool:
    """Both halves present. The private half is what makes a task gradable."""
    public = root / slug / "data" / "public" / "description.txt"
    private = root / slug / "data" / "private" / "test_answer.csv"
    return public.exists() and private.exists()


def prepare_task(root: pathlib.Path, slug: str, timeout: float = 1800) -> tuple[bool, str]:
    """Prepare one competition. Returns ``(ok, detail)``."""
    task_dir = root / slug
    if task_dir.exists():
        shutil.rmtree(task_dir)  # a partial tree would read as prepared later

    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("mledojo_task.py")),
        "--competitions", slug,
        "--data-dir", str(root),
        "--logs-dir", str(root / "_prepare_logs"),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"

    if is_prepared(root, slug):
        # The raw download is large and nothing reads it after preparation.
        shutil.rmtree(task_dir / "raw", ignore_errors=True)
        return True, "prepared"

    tail = (r.stderr or r.stdout or "").strip().splitlines()
    hint = next((l for l in reversed(tail) if any(
        w in l.lower() for w in ("error", "forbidden", "not found", "accept", "rules"))), "")
    return False, hint[:160] or f"exit {r.returncode}, no usable output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, help="file of competition slugs")
    ap.add_argument("--out", required=True, help="where prepared tasks go")
    ap.add_argument("--timeout", type=float, default=1800, help="per-task seconds")
    ap.add_argument("--only", nargs="*", help="prepare just these slugs")
    ap.add_argument("--force", action="store_true", help="re-prepare tasks already present")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    slugs = read_task_list(args.tasks)
    if args.only:
        slugs = [s for s in slugs if s in set(args.only)]

    prepared, skipped, failed = [], [], []
    started = time.time()

    for i, slug in enumerate(slugs, 1):
        if not args.force and is_prepared(root, slug):
            skipped.append(slug)
            print(f"[{i}/{len(slugs)}] {slug}: already prepared", flush=True)
            continue

        print(f"[{i}/{len(slugs)}] {slug}: preparing", flush=True)
        ok, detail = prepare_task(root, slug, args.timeout)
        (prepared if ok else failed).append((slug, detail))
        print(f"[{i}/{len(slugs)}] {slug}: {'ok' if ok else 'FAILED ' + detail}", flush=True)

    mins = (time.time() - started) / 60
    print(f"\n{len(prepared)} prepared, {len(skipped)} already present, "
          f"{len(failed)} failed, in {mins:.1f} min")

    if failed:
        print("\nfailed:")
        for slug, detail in failed:
            print(f"  {slug:<52} {detail}")
        print(
            "\nMost failures are competition rules that have to be accepted on the "
            "competition page with the same Kaggle account before its data can be "
            "downloaded. Accept them, then re-run: prepared tasks are skipped."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
