#!/usr/bin/env python
"""MLE-Dojo-compatible metric class that delegates to MLE-bench's OFFICIAL grader.

Our sandbox grades through `mledojo.gym.env.KaggleEnvironment`, which needs a `CompetitionMetrics`
subclass. MLE-bench competitions are not in MLE-Dojo's registry (`get_metric` returns None), so this
adapter presents MLE-bench's per-competition `grade.py` behind the MLE-Dojo interface.

Delegating rather than reimplementing is deliberate: MLE-bench's grade_fn defines the score that its
leaderboard — and therefore every published percentile/medal — is expressed in. A hand-written metric
that differed even slightly would silently make our numbers non-comparable.

Task naming: our eval tasks are `mleb__<competition-slug>`; strip the prefix to find the grader.
"""
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# The mlebench package must be importable in BOTH places:
#   local dev  -> /home/sagemaker-user/mleb-baseline/mle-bench/mlebench
#   in-container -> <repo root>/mlebench (eval_job_entry.stage() syncs S3 code/ to the repo root)
# Resolve by import and derive every other path from the package itself, so no absolute host path
# is baked in — a hardcoded local path would import fine here and fail only inside the job.
_DEV = Path("/home/sagemaker-user/mleb-baseline/mle-bench")
if _DEV.is_dir() and str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))


def _stub_prep_only_deps():
    """mlebench/utils.py imports py7zr at module top-level, but uses it ONLY inside
    sevenz_compress/extract — i.e. DATA-PREPARATION helpers we never call; we grade an
    already-prepared submission. py7zr is not installable here (no PyPI reachability from this
    box or the container), so register a stub that satisfies the import and would raise loudly if
    anything ever actually tried to use it.
    """
    if "py7zr" in sys.modules:
        return
    import types
    m = types.ModuleType("py7zr")

    def _unavailable(*a, **k):
        raise RuntimeError("py7zr is stubbed: archive (de)compression is prep-only and must not "
                           "run during grading")
    m.SevenZipFile = _unavailable
    sys.modules["py7zr"] = m


def _pkg_dir() -> Path:
    _stub_prep_only_deps()
    import mlebench
    return Path(mlebench.__file__).parent


PREFIX = "mleb__"


def is_mlebench_task(task: str) -> bool:
    return task.startswith(PREFIX)


def comp_of(task: str) -> str:
    return task[len(PREFIX):] if task.startswith(PREFIX) else task


def _load_grade_fn(comp: str):
    """Import the competition's official grade() from the mlebench package.

    config.yaml spells grade_fn as `module:function` (colon), and the module path keeps the slug's
    HYPHENS (mlebench.competitions.spooky-author-identification.grade). Rather than re-deriving that
    convention, reuse mlebench's own `import_fn` — it is the loader the official Grader uses.
    """
    import yaml
    _stub_prep_only_deps()
    from mlebench.utils import import_fn
    cfg = _pkg_dir() / "competitions" / comp / "config.yaml"
    fn_path = None
    if cfg.is_file():
        c = yaml.safe_load(cfg.read_text()) or {}
        g = c.get("grader") or {}
        fn_path = g.get("grade_fn") if isinstance(g, dict) else None
    if not fn_path:
        fn_path = f"mlebench.competitions.{comp}.grade:grade"
    return import_fn(fn_path)


def make_metric_class(task: str):
    """Build a CompetitionMetrics subclass for one MLE-bench competition."""
    from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

    comp = comp_of(task)
    grade_fn = _load_grade_fn(comp)

    # Direction comes from the leaderboard we prepped (rank,score sorted best-first), which was
    # itself ordered by mlebench's own is_lower_better rule — so the two agree by construction.
    lb = _pkg_dir() / "competitions" / comp / "leaderboard.csv"
    higher = True
    if lb.is_file():
        s = pd.read_csv(lb)["score"].dropna()
        if len(s) > 1:
            higher = not bool(s.iloc[0] < s.iloc[-1])

    class MLEBenchMetric(CompetitionMetrics):
        def __init__(self, value: str = "", higher_is_better: bool = higher):
            super().__init__(higher_is_better)
            self.value = value
            self.competition = comp

        def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
            # mlebench grade_fn signature is (submission, answers)
            score = grade_fn(y_pred, y_true)
            if score is None:
                raise InvalidSubmissionError(f"{comp}: official grader returned None")
            return float(score)

        def validate_submission(self, submission: Any, ground_truth: Any) -> str:
            if not isinstance(submission, pd.DataFrame):
                raise InvalidSubmissionError("Submission must be a pandas DataFrame.")
            if not isinstance(ground_truth, pd.DataFrame):
                raise InvalidSubmissionError("Ground truth must be a pandas DataFrame.")
            if len(submission) != len(ground_truth):
                raise InvalidSubmissionError(
                    f"Number of rows in submission ({len(submission)}) does not match "
                    f"ground truth ({len(ground_truth)}).")
            # Deliberately do NOT duplicate mlebench's column/shape checks here: each competition's
            # grade_fn raises InvalidSubmissionError with its own precise message, and re-checking
            # would risk rejecting submissions the official grader would have accepted.
            return "Submission is valid."

    MLEBenchMetric.__name__ = f"MLEBench_{comp.replace('-', '_')}_Metric"
    return MLEBenchMetric
