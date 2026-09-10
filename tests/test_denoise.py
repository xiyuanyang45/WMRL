import numpy as np
import pytest

from wmrl.denoise import InverseVarianceWeighter, group_disagreement


def test_weight_is_one_before_calibration():
    w = InverseVarianceWeighter(warmup=32)
    assert w.weight() == 1.0
    w.push(0.4)
    assert not w.calibrated
    assert w.weight() == 1.0, "an uncalibrated weighter must not perturb the gradient"


def test_calibration_hits_the_target_then_freezes():
    w = InverseVarianceWeighter(half_life=8, warmup=4, target_weight=2.0, w_max=4.0)
    for _ in range(4):
        w.push(0.5)
    w.maybe_calibrate()
    assert w.calibrated
    assert w.weight() == pytest.approx(2.0, abs=1e-9)

    c = w.c
    for _ in range(20):
        w.push(0.9)
        w.maybe_calibrate()
    assert w.c == c, "the scale constant must be fixed once, not re-fit every step"
    assert w.weight() > 2.0, "a larger disagreement must raise the anchor weight"


def test_weight_is_clipped_to_its_band():
    w = InverseVarianceWeighter(half_life=2, warmup=2, target_weight=2.0, w_max=3.0)
    for _ in range(2):
        w.push(0.1)
    w.maybe_calibrate()

    for _ in range(50):
        w.push(100.0)
    assert w.weight() == pytest.approx(3.0), "must saturate at w_max"

    for _ in range(200):
        w.push(0.0)
    assert w.weight() >= 1.0, "must never fall below 1: anchors are never worth less than world model groups"


def test_degenerate_group_is_ignored_not_pushed_as_zero():
    w = InverseVarianceWeighter(half_life=8, warmup=2, target_weight=2.0)
    assert group_disagreement([0.3, 0.5, 0.9], [0.4, 0.4, 0.4]) is None

    w.observe_group([0.1, 0.5, 0.9], [0.2, 0.5, 0.8])
    w.observe_group([0.1, 0.5, 0.9], [0.2, 0.5, 0.8])
    n_before, d_before = w.n, w.disagreement

    w.observe_group([0.3, 0.5, 0.9], [0.4, 0.4, 0.4])  # degenerate
    assert w.n == n_before, "a group with no ranking information must not enter the estimate"
    assert w.disagreement == d_before


def test_perfect_agreement_gives_zero_disagreement():
    r = [0.1, 0.4, 0.9]
    assert group_disagreement(r, r) == pytest.approx(0.0, abs=1e-12)


def test_disagreement_grows_with_error():
    truth = np.array([0.1, 0.3, 0.6, 0.9])
    close = truth + np.array([0.01, -0.01, 0.01, -0.01])
    far = truth[::-1]
    assert group_disagreement(close, truth) < group_disagreement(far, truth)


def test_fused_estimator_beats_either_stream_alone():
    """The claim the mechanism rests on, checked numerically.

    Two unbiased estimators of the same quantity, combined by inverse variance,
    have a variance below both. Fusion here up-weights the low-variance anchor
    stream by exactly the ratio this test uses.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    var_env, var_wm = 1.0, 4.0

    g_env = rng.normal(0.0, np.sqrt(var_env), n)
    g_wm = rng.normal(0.0, np.sqrt(var_wm), n)

    w_env = 1.0 / var_env
    w_wm = 1.0 / var_wm
    fused = (w_env * g_env + w_wm * g_wm) / (w_env + w_wm)

    assert fused.var() < g_env.var()
    assert fused.var() < g_wm.var()
    expected = 1.0 / (w_env + w_wm)
    assert fused.var() == pytest.approx(expected, rel=0.02)
