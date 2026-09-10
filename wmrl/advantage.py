"""GRPO advantages over groups graded by a mixture of sources.

Standard GRPO centers and scales rewards within each group of responses to the
same prompt. Two things change once part of the batch is graded by real
execution and the rest by a world model:

* Anchor groups carry a cleaner signal, so their advantages are up-weighted by
  the factor :mod:`wmrl.denoise` computes.
* A rollout can come back ungraded, because the sandbox timed out or the world
  model server dropped it. An ungraded rollout is *excluded*, not scored zero.
  Scoring it zero would be a confident claim that the attempt was bad, which is
  the one thing the missing grade does not tell us.
"""

from __future__ import annotations

import numpy as np

__all__ = ["group_advantages", "Group"]


class Group:
    """One prompt's group of rollouts, with whatever grades came back.

    Args:
        rewards: one entry per rollout; ``None`` marks an ungraded rollout.
        source: ``"env"`` for a group graded by real execution (an anchor
            group), ``"wm"`` for one graded by the world model.
    """

    __slots__ = ("rewards", "source", "advantages")

    def __init__(self, rewards, source: str = "wm"):
        if source not in ("env", "wm"):
            raise ValueError(f"source must be 'env' or 'wm', got {source!r}")
        self.rewards = list(rewards)
        self.source = source
        self.advantages = [None] * len(self.rewards)

    @property
    def graded(self):
        return [i for i, r in enumerate(self.rewards) if r is not None]

    def __len__(self):
        return len(self.rewards)


def group_advantages(
    groups,
    anchor_weight: float = 1.0,
    eps: float = 1e-4,
    min_graded: int = 2,
    standardize: bool = True,
):
    """Fill in ``group.advantages`` for every group, in place.

    A group with fewer than ``min_graded`` graded rollouts is skipped whole: with
    one grade there is nothing to compare against, so every advantage in it would
    be zero or arbitrary.

    Args:
        groups: iterable of :class:`Group`.
        anchor_weight: multiplier applied to groups whose ``source`` is
            ``"env"``. Comes from :class:`wmrl.denoise.InverseVarianceWeighter`.
        eps: added to the standard deviation to keep near-deterministic groups
            from exploding.
        min_graded: minimum graded rollouts for a group to contribute.
        standardize: divide by the group standard deviation, as classic GRPO
            does. Setting this to ``False`` centers only, which avoids inflating
            updates from near-deterministic groups at the expense of the
            informative middle ones.

    Returns:
        Counts of what was used: ``n_groups``, ``n_rollouts``, ``n_anchor_groups``,
        ``n_degenerate`` (groups used whose rewards were all equal).
    """
    n_groups = n_rollouts = n_anchor = n_degenerate = 0

    for g in groups:
        idx = g.graded
        if len(idx) < int(min_graded):
            continue

        arr = np.asarray([g.rewards[i] for i in idx], dtype=float)
        mu = arr.mean()
        sd = arr.std()
        if sd <= 0:
            n_degenerate += 1

        adv = (arr - mu) / (sd + float(eps)) if standardize else (arr - mu)
        if g.source == "env":
            adv = adv * float(anchor_weight)
            n_anchor += 1

        for i, a in zip(idx, adv):
            g.advantages[i] = float(a)

        n_groups += 1
        n_rollouts += len(idx)

    return {
        "n_groups": n_groups,
        "n_rollouts": n_rollouts,
        "n_anchor_groups": n_anchor,
        "n_degenerate": n_degenerate,
    }
