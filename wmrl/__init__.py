"""WMRL: reinforcement learning on world model rewards, corrected by a thin
stream of real execution.

Reference implementation of the two mechanisms in *Scaling Automatic Research
Agents via World Models* (arXiv:2608.12564). The world model replaces the
expensive environment, and its two failure modes are corrected one apiece:

* :mod:`wmrl.debias` removes the systematic error with a monotone recalibration
  fit online against anchor pairs.
* :mod:`wmrl.denoise` suppresses the random error by fusing the anchor and world
  model streams weighted by inverse variance.

Nothing here depends on a particular RL framework, model, or environment. The
two tracks in this repository, ``ml_research/`` and ``embodied/``, both call into
these modules.

A minimal end-to-end loop against a synthetic grader lives in
``examples/minimal_loop.py`` and runs on a laptop in seconds.
"""

from wmrl.advantage import Group, group_advantages
from wmrl.anchor import AnchorScheduler
from wmrl.debias import (
    OnlineDebiaser,
    apply_calibration,
    bias_stats,
    fit_calibration,
)
from wmrl.denoise import InverseVarianceWeighter, group_disagreement

__version__ = "0.1.0"

__all__ = [
    "OnlineDebiaser",
    "fit_calibration",
    "apply_calibration",
    "bias_stats",
    "InverseVarianceWeighter",
    "group_disagreement",
    "Group",
    "group_advantages",
    "AnchorScheduler",
]
