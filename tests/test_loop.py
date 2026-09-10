"""The whole method as one loop, and the scorer contract it runs on."""

import numpy as np
import pytest

from wmrl import CallableScorer, CorrectionLoop, ScorerPair
from wmrl.debias import OnlineDebiaser
from wmrl.denoise import InverseVarianceWeighter


class Truth:
    """A trusted scorer. Slow and right, standing in for real execution."""
    name, kind = "truth", "trusted"

    def score(self, trajectories):
        return [float(t) for t in trajectories]


class Squashed:
    """A cheap scorer: monotone in the truth, but with the top compressed."""
    name, kind = "squashed", "cheap"

    def __init__(self, power=0.35, offset=0.1):
        self.power, self.offset = power, offset

    def score(self, trajectories):
        return [min(1.0, self.offset + float(t) ** self.power) for t in trajectories]


class Flaky:
    """Fails to score every third trajectory."""
    name, kind = "flaky", "cheap"

    def score(self, trajectories):
        return [None if i % 3 == 0 else float(t) for i, t in enumerate(trajectories)]


def rand_groups(rng, n_groups=8, n=8):
    return [list(rng.uniform(0, 1, n)) for _ in range(n_groups)]


# ------------------------------------------------------------------ contract


def test_a_swapped_pair_is_caught_at_construction():
    """A run with these the wrong way round trains without error and ends up
    worse than doing nothing, so it has to fail here."""
    with pytest.raises(ValueError, match="kind="):
        ScorerPair(cheap=Truth(), trusted=Squashed())


def test_a_thing_with_no_score_method_is_rejected():
    with pytest.raises(TypeError, match="no score"):
        ScorerPair(cheap=object(), trusted=Truth())


def test_callable_scorer_wraps_a_function():
    s = CallableScorer("mine", "cheap", lambda ts: [0.5] * len(ts))
    assert s.score(["a", "b"]) == [0.5, 0.5]


def test_callable_scorer_rejects_a_wrong_length_reply():
    s = CallableScorer("mine", "cheap", lambda ts: [0.5])
    with pytest.raises(ValueError, match="matched by position"):
        s.score(["a", "b", "c"])


def test_callable_scorer_rejects_a_bad_kind():
    with pytest.raises(ValueError, match="cheap"):
        CallableScorer("mine", "expensive", lambda ts: [])


# ---------------------------------------------------------------------- loop


def test_anchor_groups_are_scored_by_the_trusted_scorer():
    rng = np.random.default_rng(0)
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()),
                          anchor_fraction=0.25, min_anchor_groups=1)
    groups = rand_groups(rng)
    result = loop.step(groups)

    assert result.n_anchor >= 1
    for gid in result.anchor_ids:
        np.testing.assert_allclose(result.groups[gid].rewards, groups[gid])
        assert result.groups[gid].source == "env"


def test_every_group_gets_advantages():
    rng = np.random.default_rng(1)
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()))
    result = loop.step(rand_groups(rng))
    assert result.advantage_stats["n_groups"] == 8
    for g in result.groups:
        assert all(a is not None for a in g.advantages)


def test_the_calibration_fits_and_then_changes_the_cheap_scores():
    rng = np.random.default_rng(2)
    loop = CorrectionLoop(
        ScorerPair(Squashed(), Truth()),
        anchor_fraction=0.5, min_anchor_groups=2,
        debiaser=OnlineDebiaser(min_pairs=40, refit_every=20, bins=10),
    )
    for _ in range(30):
        loop.step(rand_groups(rng))

    assert loop.debias.calibrated
    assert loop.stats()["bias_reduction"] > 0.3

    raw = Squashed().score([0.9])[0]
    assert loop.debias.apply(raw) != pytest.approx(raw, abs=1e-6)


def test_disagreement_is_measured_after_calibration():
    """The coupling: a shrinking residual should let the weight relax.

    Measured by running the same stream twice, once with the calibration
    disabled, and comparing what each weighter ended up believing.
    """
    def run(min_pairs):
        rng = np.random.default_rng(3)
        loop = CorrectionLoop(
            ScorerPair(Squashed(), Truth()),
            anchor_fraction=0.5, min_anchor_groups=2, seed=3,
            debiaser=OnlineDebiaser(min_pairs=min_pairs, refit_every=20, bins=10),
            weighter=InverseVarianceWeighter(half_life=16, warmup=4, target_weight=2.0),
        )
        for _ in range(40):
            loop.step(rand_groups(rng))
        return loop

    calibrated = run(min_pairs=40)          # fits early
    never = run(min_pairs=10_000)           # never fits

    assert calibrated.debias.calibrated and not never.debias.calibrated
    assert calibrated.weighter.disagreement < never.weighter.disagreement


def test_ungraded_trajectories_survive_the_loop():
    loop = CorrectionLoop(ScorerPair(Flaky(), Truth()),
                          anchor_fraction=0.0, min_anchor_groups=0)
    result = loop.step([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    g = result.groups[0]
    assert g.rewards[0] is None and g.advantages[0] is None
    assert any(a is not None for a in g.advantages)


def test_a_run_with_no_anchors_degrades_to_raw_cheap_scores():
    """Not an error, but it must be exactly the uncorrected behaviour."""
    rng = np.random.default_rng(4)
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()),
                          anchor_fraction=0.0, min_anchor_groups=0)
    for _ in range(20):
        result = loop.step(rand_groups(rng))

    assert not loop.debias.calibrated
    assert loop.stats()["anchor_weight"] == 1.0
    assert result.n_anchor == 0

    raw = Squashed().score([0.7])[0]
    assert loop.debias.apply(raw) == pytest.approx(raw)


def test_the_concurrency_cap_reaches_the_scheduler():
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()),
                          anchor_fraction=1.0, anchor_concurrency=2)
    assert loop.step(rand_groups(np.random.default_rng(5)), in_flight=2).n_anchor == 0
    assert loop.step(rand_groups(np.random.default_rng(5)), in_flight=0).n_anchor == 2


def test_stats_report_what_a_run_should_watch():
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()))
    loop.step(rand_groups(np.random.default_rng(6)))
    s = loop.stats()
    assert set(s) == {"step", "anchor_fraction", "calibrated", "bias_reduction", "anchor_weight"}
    assert s["step"] == 1


def test_a_custom_scorer_pair_just_works():
    """The documented extension point, exercised end to end."""
    cheap = CallableScorer("heuristic", "cheap", lambda ts: [len(str(t)) / 20 for t in ts])
    trusted = CallableScorer("unit-tests", "trusted", lambda ts: [float(t) for t in ts])

    loop = CorrectionLoop(ScorerPair(cheap, trusted), anchor_fraction=0.5, min_anchor_groups=1)
    result = loop.step([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    assert result.advantage_stats["n_groups"] == 2


def test_a_configured_debiaser_is_not_swapped_for_a_default():
    """OnlineDebiaser defines __len__, so an empty one is falsy. A truthiness
    check here would silently discard the caller's configuration."""
    mine = OnlineDebiaser(min_pairs=9999, refit_every=7, bins=4)
    loop = CorrectionLoop(ScorerPair(Squashed(), Truth()), debiaser=mine)
    assert loop.debias is mine
    assert loop.debias.min_pairs == 9999

    w = InverseVarianceWeighter(warmup=123)
    assert CorrectionLoop(ScorerPair(Squashed(), Truth()), weighter=w).weighter is w
