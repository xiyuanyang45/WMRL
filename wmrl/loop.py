"""The method, as one function.

Everything else in :mod:`wmrl` is a piece of this. :class:`CorrectionLoop` puts
them together so a training loop can call one method per step and get corrected
scores back, without having to remember the order the pieces go in.

That order is not arbitrary and is easy to get subtly wrong:

1. Choose which groups pay for the trusted scorer.
2. Score those with *both*, so each yields calibration pairs.
3. Push the pairs into the calibration **before** measuring disagreement, so
   the disagreement is measured on the calibrated residual. This is what
   couples the two mechanisms: as the map removes bias the residual shrinks and
   the weight relaxes on its own.
4. Score everything else with the cheap scorer and push it through the map.
5. Form advantages, with anchor groups carrying the weight.

Using it::

    loop = CorrectionLoop(ScorerPair(cheap=world_model, trusted=sandbox))

    for step in range(steps):
        result = loop.step(groups)          # groups: list of lists of trajectories
        train_on(result.groups)             # each Group has .advantages
        log(loop.stats())

For a run that is not shaped like this, use the pieces directly; nothing else
in the package imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wmrl.advantage import Group, group_advantages
from wmrl.anchor import AnchorScheduler
from wmrl.debias import OnlineDebiaser
from wmrl.denoise import InverseVarianceWeighter
from wmrl.scorer import ScorerPair

__all__ = ["CorrectionLoop", "StepResult"]


@dataclass
class StepResult:
    """What one step produced."""

    groups: list = field(default_factory=list)
    anchor_ids: list = field(default_factory=list)
    anchor_weight: float = 1.0
    advantage_stats: dict = field(default_factory=dict)

    @property
    def n_anchor(self) -> int:
        return len(self.anchor_ids)


class CorrectionLoop:
    """Owns the anchor schedule, the calibration and the weight for a whole run.

    One instance per run. Keeping all three in one object is what keeps them
    consistent: they are driven by a single stream of anchor pairs, and a run
    that fitted the calibration on one set of pairs while measuring
    disagreement on another would have two mechanisms disagreeing about what
    the world model is currently doing.
    """

    def __init__(
        self,
        scorers: ScorerPair,
        *,
        anchor_fraction: float = 0.10,
        min_anchor_groups: int = 1,
        anchor_concurrency: int | None = None,
        seed: int = 0,
        debiaser: OnlineDebiaser | None = None,
        weighter: InverseVarianceWeighter | None = None,
        standardize: bool = True,
    ):
        self.scorers = scorers
        self.standardize = standardize

        self.anchors = AnchorScheduler(
            fraction=anchor_fraction,
            min_per_step=min_anchor_groups,
            max_concurrent=anchor_concurrency,
            seed=seed,
        )
        # `or` would be wrong here: OnlineDebiaser defines __len__, so a freshly
        # constructed one is falsy and a caller's configured debiaser would be
        # silently swapped for a default.
        self.debias = debiaser if debiaser is not None else OnlineDebiaser()
        self.weighter = weighter if weighter is not None else InverseVarianceWeighter()
        self.n_steps = 0

    def step(self, groups, in_flight: int = 0) -> StepResult:
        """Score and correct one step's groups.

        Args:
            groups: a list of groups, each a list of trajectories.
            in_flight: trusted-scorer work already outstanding, counted against
                the concurrency cap.
        """
        anchor_ids = self.anchors.select(len(groups), in_flight=in_flight)
        anchor_set = set(anchor_ids)
        out = []

        for gid, trajectories in enumerate(groups):
            if gid in anchor_set:
                trusted = self.scorers.trusted.score(trajectories)
                cheap = self.scorers.cheap.score(trajectories)

                pairs = [(c, t) for c, t in zip(cheap, trusted)
                         if c is not None and t is not None]
                if pairs:
                    self.debias.push_many([c for c, _ in pairs], [t for _, t in pairs])
                    # after the push, so the residual measured is the calibrated one
                    self.weighter.observe_group(
                        [self.debias.apply(c) for c, _ in pairs],
                        [t for _, t in pairs],
                    )
                out.append(Group(trusted, source="env"))
            else:
                cheap = self.scorers.cheap.score(trajectories)
                out.append(Group(
                    [None if c is None else self.debias.apply(c) for c in cheap],
                    source="wm",
                ))

        weight = self.weighter.weight()
        stats = group_advantages(out, anchor_weight=weight, standardize=self.standardize)
        self.n_steps += 1

        return StepResult(
            groups=out,
            anchor_ids=list(anchor_ids),
            anchor_weight=weight,
            advantage_stats=stats,
        )

    def stats(self) -> dict:
        """Four numbers worth logging every step, and what each one tells you."""
        return {
            "step": self.n_steps,
            # drifts below target when the trusted scorer's capacity binds
            "anchor_fraction": round(self.anchors.realized_fraction, 4),
            # false means the run is training on raw cheap scores
            "calibrated": self.debias.calibrated,
            # how much within-group bias the current map removes
            "bias_reduction": (
                round(self.debias.reduction, 4) if self.debias.reduction is not None else None
            ),
            # 1.0 until calibrated, then rises with measured disagreement
            "anchor_weight": round(self.weighter.weight(), 3),
        }
