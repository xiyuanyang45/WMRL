"""Deciding which groups pay for real execution.

The anchor stream is the only ground truth in the loop, and it is the only thing
in the loop that costs real machine time. Scheduling it is therefore a budget
problem with two constraints pulling against each other:

* Too few anchor groups and the calibration map goes stale, because the world
  model's error drifts as the policy moves.
* Too many and the speedup disappears, since sandbox execution is exactly the
  bottleneck the world model was introduced to remove.

The scheduler below spends a fixed fraction of groups on real execution, with a
floor per step so a step never goes entirely without ground truth, and a hard
concurrency cap so a burst of anchor groups cannot exhaust the sandbox pool and
stall the rollout workers behind it.

Selection is deterministic given a seed. Reproducibility of *which* groups were
anchored matters when a run has to be resumed or explained.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AnchorScheduler"]


class AnchorScheduler:
    """Choose the anchor groups for each training step.

    Args:
        fraction: target share of groups graded by real execution.
        min_per_step: floor on anchor groups per step, applied even when
            ``fraction`` would round down to zero. A step with no anchor group
            contributes nothing to the calibration map.
        max_concurrent: ceiling on anchor groups in flight, reflecting how many
            sandbox slots exist. ``None`` disables the cap.
        seed: seed for the selection.
    """

    def __init__(
        self,
        fraction: float = 0.10,
        min_per_step: int = 1,
        max_concurrent: int | None = None,
        seed: int = 0,
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.fraction = float(fraction)
        self.min_per_step = int(min_per_step)
        self.max_concurrent = max_concurrent
        self._rng = np.random.default_rng(seed)

        self.n_steps = 0
        self.n_groups = 0
        self.n_anchor = 0

    def select(self, n_groups: int, in_flight: int = 0):
        """Return the sorted indices of the groups to grade by real execution.

        Args:
            n_groups: how many groups this step has.
            in_flight: anchor groups already occupying sandbox slots, counted
                against ``max_concurrent``.
        """
        if n_groups <= 0:
            return []

        want = max(self.min_per_step, int(round(self.fraction * n_groups)))
        want = min(want, n_groups)

        if self.max_concurrent is not None:
            room = max(0, int(self.max_concurrent) - int(in_flight))
            want = min(want, room)

        idx = sorted(self._rng.choice(n_groups, size=want, replace=False).tolist()) if want else []

        self.n_steps += 1
        self.n_groups += n_groups
        self.n_anchor += len(idx)
        return idx

    @property
    def realized_fraction(self) -> float:
        """Share of groups actually anchored so far.

        Worth logging: it drifts below ``fraction`` whenever the concurrency cap
        binds, and that is the signal that the sandbox pool, not the policy, is
        setting the pace.
        """
        return self.n_anchor / self.n_groups if self.n_groups else 0.0
