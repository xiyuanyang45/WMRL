import numpy as np
import pytest

from wmrl.debias import (
    OnlineDebiaser,
    apply_calibration,
    bias_stats,
    fit_calibration,
)


def biased_grader(r_true, rng, bias=0.25, noise=0.05):
    """A world model that compresses the scale and shifts it up, plus noise.

    Deliberately non-affine in its effect on ranking: the compression is
    stronger at the top, which is what a world model reading code without
    running it actually does. It cannot see that an almost-correct solution
    crashes, so it bunches the good ones together.
    """
    r = np.asarray(r_true, dtype=float)
    return np.clip(bias + 0.55 * r + 0.20 * r**2 + rng.normal(0, noise, r.shape), 0, 1)


def test_identity_before_any_fit():
    d = OnlineDebiaser(min_pairs=10)
    assert not d.calibrated
    assert d.apply(0.42) == pytest.approx(0.42)
    np.testing.assert_allclose(d.apply(np.array([0.1, 0.9])), [0.1, 0.9])


def test_returns_none_when_too_few_bins_populated():
    assert fit_calibration([], bins=10) is None
    assert fit_calibration([(0.5, 0.5)], bins=10) is None
    # every point in one bin: one populated bin is not enough to interpolate
    assert fit_calibration([(0.51, 0.3), (0.52, 0.4), (0.53, 0.5)], bins=10) is None


def test_fitted_map_is_monotone():
    rng = np.random.default_rng(0)
    r_true = rng.uniform(0, 1, 4000)
    pairs = list(zip(biased_grader(r_true, rng), r_true))
    fx, fy = fit_calibration(pairs, bins=10)
    assert np.all(np.diff(fy) >= -1e-12), "pool-adjacent-violators must leave a non-decreasing map"
    assert np.all(np.diff(fx) > 0), "breakpoints must be strictly increasing for interp"


def test_calibration_removes_almost_all_of_a_noise_free_bias():
    """With the noise turned down, what is left is the systematic part alone,
    and the calibration should take out nearly all of it."""
    rng = np.random.default_rng(1)
    r_true = rng.uniform(0, 1, 4000)
    r_wm = biased_grader(r_true, rng, noise=0.005)
    f = fit_calibration(list(zip(r_wm, r_true)), bins=10)

    before, after, reduction = bias_stats(r_wm, r_true, f)
    assert after < before
    assert reduction > 0.9, f"expected nearly all of the bias removed, got {reduction:.3f}"


def test_calibration_still_helps_once_noise_is_present():
    """Noise is not removable by a pointwise map, so it caps the reduction.
    What matters is that the systematic part still goes away."""
    rng = np.random.default_rng(1)
    r_true = rng.uniform(0, 1, 4000)
    r_wm = biased_grader(r_true, rng, noise=0.05)
    f = fit_calibration(list(zip(r_wm, r_true)), bins=10)

    before, after, reduction = bias_stats(r_wm, r_true, f)
    assert after < before
    assert reduction > 0.4, f"expected a substantial reduction, got {reduction:.3f}"


def test_calibration_survives_within_group_standardization():
    """The property that makes a monotone fit the right choice.

    GRPO standardizes rewards inside each group, which cancels any affine map.
    A calibration that only rescaled would therefore be a no-op on the gradient.
    This checks that the fitted map still changes the standardized scores.
    """
    rng = np.random.default_rng(2)
    r_true = rng.uniform(0, 1, 4000)
    r_wm = biased_grader(r_true, rng)
    f = fit_calibration(list(zip(r_wm, r_true)), bins=10)

    def standardize(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-4)

    group = rng.uniform(0, 1, 8)
    raw = biased_grader(group, rng)
    z_raw = standardize(raw)
    z_cal = standardize(apply_calibration(f, raw))

    assert not np.allclose(z_raw, z_cal, atol=1e-3), (
        "a calibration cancelled by standardization would not change the gradient"
    )


def test_ranking_is_never_reordered():
    rng = np.random.default_rng(3)
    r_true = rng.uniform(0, 1, 2000)
    f = fit_calibration(list(zip(biased_grader(r_true, rng), r_true)), bins=10)

    probe = np.linspace(0, 1, 200)
    out = apply_calibration(f, probe)
    assert np.all(np.diff(out) >= -1e-12), "a monotone map must preserve order"


def test_online_debiaser_refits_and_tracks_drift():
    rng = np.random.default_rng(4)
    d = OnlineDebiaser(min_pairs=200, refit_every=100, bins=10, cap=1000)

    r_true = rng.uniform(0, 1, 600)
    d.push_many(biased_grader(r_true, rng, bias=0.25), r_true)
    assert d.calibrated
    first_fits = d.n_fits
    assert first_fits >= 1

    # the world model's error moves: refeed with a different offset and check
    # the map follows rather than staying frozen on the early estimate
    before = d.apply(0.5)
    r_true2 = rng.uniform(0, 1, 1500)
    d.push_many(biased_grader(r_true2, rng, bias=0.60), r_true2)
    assert d.n_fits > first_fits
    assert d.apply(0.5) != pytest.approx(before, abs=1e-6)


def test_buffer_is_capped():
    rng = np.random.default_rng(5)
    d = OnlineDebiaser(min_pairs=50, refit_every=25, cap=200)
    r_true = rng.uniform(0, 1, 1000)
    d.push_many(biased_grader(r_true, rng), r_true)
    assert len(d) == 200


def test_uncalibrated_run_degrades_to_raw_scores():
    """A run that never sees ground truth must behave exactly like no correction."""
    d = OnlineDebiaser(min_pairs=10_000)
    rng = np.random.default_rng(6)
    r_true = rng.uniform(0, 1, 100)
    r_wm = biased_grader(r_true, rng)
    d.push_many(r_wm, r_true)
    np.testing.assert_allclose(d.apply(r_wm), r_wm)
