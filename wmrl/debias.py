"""Online Debiasing: monotone recalibration of world model rewards.

A world model grades a trajectory without running it, so its score is
systematically off. The offset is not a constant: it drifts as the policy moves
into parts of the solution space the world model reads differently.

The fix is to keep a thin stream of ground truth. A small fraction of groups is
graded by *both* the world model and real execution, and each such group yields
score pairs ``(r_wm, r_env)``. Fitting a monotone map ``f`` on those pairs and
pushing every world model score through ``f`` before advantages are formed
removes the systematic part.

Two properties of the map matter, and both are deliberate:

* **Monotone.** ``f`` never reorders trajectories within a group, so it cannot
  destroy the ranking the world model got right.
* **Non-affine.** GRPO standardizes rewards within a group, which cancels any
  affine map. A piecewise-linear isotonic fit survives that standardization,
  which is the whole point: an affine recalibration would be a no-op here.

The fit is bucketed isotonic regression by pool-adjacent-violators, in plain
numpy, with no scikit-learn dependency and no randomness.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fit_calibration",
    "apply_calibration",
    "bias_stats",
    "OnlineDebiaser",
]


def fit_calibration(pairs, bins: int = 10):
    """Fit a monotone map from world model score to expected execution score.

    Bins ``x`` over ``[0, 1]``, takes the mean of ``y`` in each populated bin,
    then pool-adjacent-violators to force the bin means to be non-decreasing.

    Args:
        pairs: iterable of ``(r_wm, r_env)`` score pairs from anchor groups.
        bins: number of equal-width bins over ``[0, 1]``.

    Returns:
        ``(fx, fy)`` breakpoint arrays for :func:`apply_calibration`, or ``None``
        when fewer than two bins are populated. ``None`` means the caller has not
        seen enough ground truth to calibrate and should stay at identity.
    """
    P = np.asarray(list(pairs), dtype=float)
    if P.ndim != 2 or P.shape[0] < 2:
        return None
    x, y = P[:, 0], P[:, 1]

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, int(bins) - 1)

    cx, cy, cw = [], [], []
    for b in range(int(bins)):
        m = idx == b
        if m.sum() == 0:
            continue
        cx.append(float(x[m].mean()))
        cy.append(float(y[m].mean()))
        cw.append(float(m.sum()))
    if len(cx) < 2:
        return None

    # Pool adjacent violators: merge any bin whose mean dips below its left
    # neighbour, weighting by bin population, until the sequence is monotone.
    val, wt, cnt = [], [], []
    for v, w in zip(cy, cw):
        val.append(v)
        wt.append(w)
        cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2, c2 = val.pop(), wt.pop(), cnt.pop()
            v1, w1, c1 = val.pop(), wt.pop(), cnt.pop()
            nw = w1 + w2
            val.append((v1 * w1 + v2 * w2) / nw)
            wt.append(nw)
            cnt.append(c1 + c2)

    fy = []
    for v, c in zip(val, cnt):
        fy += [v] * c
    return np.asarray(cx), np.asarray(fy)


def apply_calibration(f, r):
    """Push a score, or an array of scores, through a fitted map.

    ``f=None`` is the identity, which is what an uncalibrated run does and what
    every run does before it has seen enough anchor pairs.
    """
    if f is None:
        return float(r) if np.isscalar(r) else np.asarray(r, dtype=float)
    fx, fy = f
    if np.isscalar(r):
        return float(np.interp(float(r), fx, fy))
    return np.interp(np.asarray(r, dtype=float), fx, fy)


def bias_stats(x, y, f):
    """Within-group centered squared residual, before and after calibration.

    Centering is what makes this the right diagnostic: GRPO only ever sees
    within-group differences, so a bias measure has to be blind to any constant
    offset for the same reason the advantage is.

    Returns:
        ``(before, after, reduction)`` where ``reduction = 1 - after / before``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yhat = apply_calibration(f, x)
    before = float((((x - x.mean()) - (y - y.mean())) ** 2).mean())
    after = float((((yhat - yhat.mean()) - (y - y.mean())) ** 2).mean())
    return before, after, 1.0 - after / max(1e-9, before)


class OnlineDebiaser:
    """Anchor buffer plus opportunistic refit, for use inside a training loop.

    Starts at identity and stays there until ``min_pairs`` anchor pairs have
    arrived, so a run that never sees ground truth degrades exactly to training
    on raw world model rewards rather than to something undefined. After that it
    refits every ``refit_every`` new pairs, which is what lets the map track the
    drift instead of freezing an early estimate.

    The buffer is FIFO and capped, so old pairs age out and the fit reflects the
    world model's behaviour on the *current* policy.
    """

    def __init__(
        self,
        min_pairs: int = 200,
        refit_every: int = 64,
        bins: int = 10,
        cap: int = 5000,
    ):
        self.min_pairs = int(min_pairs)
        self.refit_every = int(refit_every)
        self.bins = int(bins)
        self.cap = int(cap)

        self._pairs: list[tuple[float, float]] = []
        self._since = 0

        self.f = None
        self.n_fits = 0
        self.bias_before = None
        self.bias_after = None
        self.reduction = None

    def push(self, r_wm: float, r_env: float) -> None:
        """Record one anchor pair, refitting when enough new ones have arrived."""
        self._pairs.append((float(r_wm), float(r_env)))
        if len(self._pairs) > self.cap:
            self._pairs = self._pairs[-self.cap:]
        self._since += 1
        if len(self._pairs) >= self.min_pairs and self._since >= self.refit_every:
            self.refit()

    def push_many(self, r_wm, r_env) -> None:
        for a, b in zip(r_wm, r_env):
            self.push(a, b)

    def refit(self) -> bool:
        """Refit now. Returns whether the map changed."""
        f = fit_calibration(self._pairs, self.bins)
        if f is None:
            return False
        self.f = f
        self._since = 0
        self.n_fits += 1
        P = np.asarray(self._pairs, dtype=float)
        self.bias_before, self.bias_after, self.reduction = bias_stats(P[:, 0], P[:, 1], f)
        return True

    def apply(self, r):
        """Calibrate a score or an array of scores."""
        return apply_calibration(self.f, r)

    @property
    def calibrated(self) -> bool:
        return self.f is not None

    def __len__(self) -> int:
        return len(self._pairs)
