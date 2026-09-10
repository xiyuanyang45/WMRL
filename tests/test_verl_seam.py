"""The seam where the anchor weight reaches the loss.

This is the one place the inverse-variance mechanism actually changes a
gradient, so it gets tested directly rather than only through its parts.
"""

import numpy as np
import pytest

from ml_research.cluster.verl_patches import PatchError, apply_overlays, weight_anchor_advantages
from wmrl.debias import OnlineDebiaser
from wmrl.denoise import InverseVarianceWeighter


def calibrated_weighter(target=2.0, w_max=4.0):
    w = InverseVarianceWeighter(half_life=8, warmup=2, target_weight=target, w_max=w_max)
    for _ in range(2):
        w.push(0.5)
    w.maybe_calibrate()
    return w


def test_only_anchor_rows_are_scaled():
    adv = [1.0, -1.0, 2.0, -2.0]
    mode = ["env", "env", "wm", "wm"]
    gid = [0, 0, 1, 1]

    out, info = weight_anchor_advantages(adv, mode, gid, weighter=calibrated_weighter())

    np.testing.assert_allclose(out[:2], [2.0, -2.0])
    np.testing.assert_allclose(out[2:], [2.0, -2.0], err_msg="world model rows must be untouched")
    assert info["anchor_rows"] == 2


def test_without_a_weighter_nothing_changes():
    """The uncorrected path must be exactly the trainer's own behaviour."""
    adv = [0.3, -0.3, 1.0]
    out, info = weight_anchor_advantages(adv, ["env", "env", "wm"], [0, 0, 1])
    np.testing.assert_allclose(out, adv)
    assert info["anchor_weight"] == 1.0


def test_an_uncalibrated_weighter_is_inert():
    w = InverseVarianceWeighter(warmup=1000)
    adv = [1.0, -1.0]
    out, _ = weight_anchor_advantages(adv, ["env", "env"], [0, 0], weighter=w)
    np.testing.assert_allclose(out, adv)


def test_the_input_is_not_modified():
    adv = np.array([1.0, -1.0])
    out, _ = weight_anchor_advantages(adv, ["env", "env"], [0, 0], weighter=calibrated_weighter())
    np.testing.assert_allclose(adv, [1.0, -1.0])
    assert out is not adv


def test_disagreement_is_measured_within_group_and_after_calibration():
    """The coupling between the two mechanisms lives here.

    As the calibration removes bias the residual shrinks, so the weight relaxes.
    A run that measured the raw residual instead would keep the weight pinned
    high forever and never notice the debiaser working.
    """
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 1, 400)
    predicted = np.clip(0.25 + 0.5 * truth, 0, 1)  # biased, noise free

    debias = OnlineDebiaser(min_pairs=50, refit_every=50, bins=10)
    debias.push_many(predicted, truth)
    assert debias.calibrated

    raw = InverseVarianceWeighter(half_life=4, warmup=1, target_weight=2.0)
    cal = InverseVarianceWeighter(half_life=4, warmup=1, target_weight=2.0)

    mode = ["env"] * 8
    gid = [0] * 8
    wm, env = predicted[:8], truth[:8]

    weight_anchor_advantages(np.zeros(8), mode, gid, wm, env, debias=None, weighter=raw)
    weight_anchor_advantages(np.zeros(8), mode, gid, wm, env, debias=debias, weighter=cal)

    assert cal.disagreement < raw.disagreement, (
        "calibrated residual should be smaller than the raw one"
    )


def test_a_degenerate_anchor_group_is_skipped():
    w = calibrated_weighter()
    before = w.n
    _, info = weight_anchor_advantages(
        np.zeros(3), ["env"] * 3, [0] * 3,
        r_world_model=[0.2, 0.5, 0.9], r_env=[0.4, 0.4, 0.4], weighter=w,
    )
    assert info["groups_measured"] == 0
    assert w.n == before, "a group where every measured score is equal says nothing"


def test_a_single_row_group_is_skipped():
    w = calibrated_weighter()
    _, info = weight_anchor_advantages(
        np.zeros(1), ["env"], [0], r_world_model=[0.5], r_env=[0.6], weighter=w
    )
    assert info["groups_measured"] == 0


def test_ungraded_rows_do_not_enter_the_measurement():
    w = calibrated_weighter()
    _, info = weight_anchor_advantages(
        np.zeros(4), ["env"] * 4, [0] * 4,
        r_world_model=[0.2, None, 0.7, 0.9], r_env=[0.3, 0.5, None, 0.8], weighter=w,
    )
    assert info["groups_measured"] == 1  # only the two complete pairs


def test_misaligned_batch_is_rejected():
    """Rows are matched by position, so a length mismatch would weight the wrong ones."""
    with pytest.raises(ValueError, match="misaligned"):
        weight_anchor_advantages([1.0, 2.0], ["env"], [0, 0])


def test_weight_is_capped_at_w_max():
    w = calibrated_weighter(target=2.0, w_max=2.5)
    for _ in range(50):
        w.push(100.0)
    out, info = weight_anchor_advantages([1.0], ["env"], [0], weighter=w)
    assert info["anchor_weight"] == pytest.approx(2.5)
    np.testing.assert_allclose(out, [2.5])


# ------------------------------------------------------------------ overlays


def test_overlay_fails_loudly_when_the_trainer_has_drifted(tmp_path):
    """Silently skipping a patch is worse than refusing to start."""
    target = tmp_path / "verl" / "experimental" / "fully_async_policy"
    target.mkdir(parents=True)
    (target / "detach_utils.py").write_text("def handler():\n    pass\n")

    with pytest.raises(PatchError, match="anchor text not found"):
        apply_overlays(str(tmp_path))


def test_overlay_reports_a_missing_installation(tmp_path):
    with pytest.raises(PatchError, match="does not exist"):
        apply_overlays(str(tmp_path))


def test_overlay_is_idempotent(tmp_path):
    target = tmp_path / "verl" / "experimental" / "fully_async_policy"
    target.mkdir(parents=True)
    f = target / "detach_utils.py"
    f.write_text(
        "    except Exception as e:\n"
        '        print(f"Task {task.get_name()} failed with exception: {e}")\n'
        "        raise e\n"
    )

    assert apply_overlays(str(tmp_path))["cancellation"] == "applied"
    once = f.read_text()
    assert apply_overlays(str(tmp_path))["cancellation"] == "already applied"
    assert f.read_text() == once, "a second run must not stack the patch"
