"""The two things a WMRL run needs from you.

Strip away the agent, the benchmark and the cluster, and the method needs
exactly two ways to score a trajectory:

**A cheap scorer.** Fast, batchable, and wrong. In the paper this is a language
model reading the submitted solution and predicting how it would have scored. It
could equally be a learned reward model, a heuristic, a smaller model, or a
cached lookup. The only requirement is that it is cheap enough to run on
everything.

**A trusted scorer.** Slow, expensive, and right. In the paper this is executing
the solution in a sandbox and reading the leaderboard percentile. It could be a
unit-test suite, a human label queue, a simulator, or a physical experiment. The
only requirement is that you believe it.

Everything else in this repository is machinery for running those two at scale.
The method itself only cares that one is cheap, one is trusted, and both return
a number per trajectory.

Implementing your own is one method::

    class MyTests(Scorer):
        name = "pytest"
        kind = "trusted"

        def score(self, trajectories):
            return [run_suite(t) or None for t in trajectories]

and then handing it to the loop in place of a built-in. Nothing else changes:
the calibration will fit against whatever your trusted scorer says, and the
weight will follow whatever disagreement it finds.

Two contract points are worth reading before you write one.

**Return ``None``, never ``0.0``, for a trajectory you could not score.** A
timeout, a crashed harness, a rate limit: all of those mean *unknown*. Scoring
them zero asserts the attempt was bad, which is a different and usually false
claim, and it poisons both the group's advantage and the calibration.

**Keep the scale stable, not necessarily correct.** The calibration handles a
cheap scorer that is systematically wrong. It cannot handle one whose meaning
changes underneath it faster than the anchor stream can re-fit.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

__all__ = ["Scorer", "ScorerPair", "CallableScorer"]


@runtime_checkable
class Scorer(Protocol):
    """Scores a batch of trajectories.

    Attributes:
        name: what to call this in logs.
        kind: ``"cheap"`` or ``"trusted"``. This decides which stream the scorer
            feeds, and therefore whether its output is corrected or is the thing
            corrections are fitted against.
    """

    name: str
    kind: str

    def score(self, trajectories: Sequence) -> list:
        """Return one score per trajectory, in order, or ``None`` for unknown."""
        ...


class CallableScorer:
    """Wrap a plain function as a scorer, for the common simple case.

    ::

        cheap = CallableScorer("my-reward-model", "cheap", predict_batch)
    """

    def __init__(self, name: str, kind: str, fn):
        if kind not in ("cheap", "trusted"):
            raise ValueError(f"kind must be 'cheap' or 'trusted', got {kind!r}")
        self.name = name
        self.kind = kind
        self._fn = fn

    def score(self, trajectories):
        out = list(self._fn(trajectories))
        if len(out) != len(trajectories):
            raise ValueError(
                f"{self.name} returned {len(out)} score(s) for {len(trajectories)} "
                "trajectory/ies. Scores are matched by position."
            )
        return [None if s is None else float(s) for s in out]

    def __repr__(self):
        return f"CallableScorer({self.name!r}, {self.kind!r})"


class ScorerPair:
    """The cheap and trusted scorers a run uses, checked once at construction.

    Catching a swapped pair here matters: a run with the two the wrong way round
    would train happily, calibrating a trusted signal against a cheap one, and
    the only symptom would be results quietly worse than the uncorrected
    baseline.
    """

    def __init__(self, cheap: Scorer, trusted: Scorer):
        for role, s in (("cheap", cheap), ("trusted", trusted)):
            if not hasattr(s, "score") or not callable(s.score):
                raise TypeError(f"{role} scorer {s!r} has no score() method")
            kind = getattr(s, "kind", None)
            if kind != role:
                raise ValueError(
                    f"the {role} scorer declares kind={kind!r}. "
                    "A run with these the wrong way round trains without error and "
                    "ends up worse than doing nothing, so the pair is checked here."
                )
        self.cheap = cheap
        self.trusted = trusted

    def describe(self) -> str:
        return (
            f"cheap:   {getattr(self.cheap, 'name', type(self.cheap).__name__)}\n"
            f"trusted: {getattr(self.trusted, 'name', type(self.trusted).__name__)}"
        )

    def __repr__(self):
        return f"ScorerPair(cheap={self.cheap!r}, trusted={self.trusted!r})"
