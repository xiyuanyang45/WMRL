"""Inverse-Variance Denoising: fusing the anchor and world model streams.

Calibration removes the systematic part of the world model's error. What is left
is zero-mean noise, and noise cannot be subtracted off pointwise: there is no
per-trajectory estimate of it. It can only be averaged down.

Two reward streams reach the gradient. The anchor stream is graded by real
execution, so it is clean but scarce. The world model stream is abundant but
noisy. Both are estimates of the same policy gradient, so the minimal-variance
combination weights each by the inverse of its variance, and the result sits
strictly below either stream on its own.

In practice that weight is applied as an up-weight on anchor groups. Writing
``V_WM = (1 + c s^2) V_E`` for the inflated variance of a world-model-graded
group, the optimal anchor weight is ``1 + c s^2``, and the only quantity that has
to be measured online is the disagreement ``s^2`` between the calibrated world
model score and the truth. The scale constant ``c`` is fixed once, after a short
warmup, so the weight tracks the *changes* in disagreement rather than depending
on an absolute calibration of it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["group_disagreement", "InverseVarianceWeighter"]


def group_disagreement(r_wm_calibrated, r_env):
    """Normalized within-group disagreement between calibrated scores and truth.

    Centered for the same reason the bias diagnostic is centered: GRPO only sees
    within-group differences. Normalized by the variance of the true scores so
    the quantity is dimensionless and comparable across groups of very different
    difficulty.

    Returns:
        The disagreement, or ``None`` for a degenerate group in which every true
        score is the same. Such a group carries no ranking information and must
        not enter the running estimate.
    """
    d = np.asarray(r_wm_calibrated, dtype=float)
    s = np.asarray(r_env, dtype=float)
    var_s = s.var()
    if var_s < 1e-9:
        return None
    return float((((d - d.mean()) - (s - s.mean())) ** 2).mean() / var_s)


class InverseVarianceWeighter:
    """Adaptive anchor weight driven by measured disagreement.

    Holds an exponentially weighted estimate of the disagreement and turns it
    into a multiplicative weight on anchor-group advantages,
    ``w = clip(1 + c * s^2, 1, w_max)``.

    ``c`` is calibrated once, after ``warmup`` observations, so that the weight
    equals ``target_weight`` at that moment, then frozen. Freezing matters: it
    makes the weight respond to how the disagreement moves over training rather
    than to its absolute scale, which depends on reward units nobody should have
    to tune.

    Before calibration the weight is exactly 1, so a run that never gathers
    enough anchor groups degrades to unweighted fusion instead of to an
    arbitrary constant.
    """

    def __init__(
        self,
        half_life: float = 64,
        warmup: int = 32,
        w_max: float = 4.0,
        target_weight: float = 2.0,
    ):
        self.half_life = float(half_life)
        self.warmup = int(warmup)
        self.w_max = float(w_max)
        self.target_weight = float(target_weight)

        self.disagreement = None
        self.n = 0
        self.c = None

    def push(self, s2: float) -> float:
        """Fold one group's disagreement into the running estimate."""
        alpha = 1.0 - 0.5 ** (1.0 / max(1.0, self.half_life))
        if self.disagreement is None:
            self.disagreement = float(s2)
        else:
            self.disagreement = (1 - alpha) * self.disagreement + alpha * float(s2)
        self.n += 1
        return self.disagreement

    def maybe_calibrate(self, rho: float = 1.0):
        """Fix the scale constant once, after warmup. Returns ``c``, or ``None``."""
        if (
            self.c is None
            and self.n >= self.warmup
            and self.disagreement is not None
            and self.disagreement * rho > 1e-12
        ):
            self.c = (self.target_weight - 1.0) / (self.disagreement * max(1e-12, rho))
        return self.c

    def weight(self) -> float:
        """The current anchor weight. Exactly 1.0 until calibrated."""
        if self.c is None or self.disagreement is None:
            return 1.0
        return float(min(self.w_max, max(1.0, 1.0 + self.c * self.disagreement)))

    @property
    def calibrated(self) -> bool:
        return self.c is not None

    def observe_group(self, r_wm_calibrated, r_env, rho: float = 1.0):
        """Convenience: measure one anchor group, fold it in, calibrate if due.

        Degenerate groups are ignored rather than pushed as zero, which would
        otherwise drag the estimate toward zero and silently disable the weight.
        """
        s2 = group_disagreement(r_wm_calibrated, r_env)
        if s2 is None:
            return self.weight()
        self.push(s2)
        self.maybe_calibrate(rho)
        return self.weight()
