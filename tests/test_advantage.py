import numpy as np
import pytest

from wmrl.advantage import Group, group_advantages


def test_ungraded_rollouts_are_excluded_not_zeroed():
    """A missing grade means we do not know, which is not the same as a bad attempt."""
    g = Group([0.2, None, 0.8, 0.5], source="wm")
    stats = group_advantages([g])

    assert g.advantages[1] is None, "an ungraded rollout must carry no advantage at all"
    assert stats["n_rollouts"] == 3

    graded = [a for a in g.advantages if a is not None]
    assert sum(graded) == pytest.approx(0.0, abs=1e-9), "advantages must be centered on the graded members only"


def test_group_below_min_graded_is_skipped_whole():
    g = Group([0.5, None, None], source="wm")
    stats = group_advantages([g], min_graded=2)
    assert all(a is None for a in g.advantages)
    assert stats["n_groups"] == 0


def test_anchor_weight_applies_only_to_env_groups():
    rewards = [0.1, 0.4, 0.9, 0.6]
    wm = Group(list(rewards), source="wm")
    env = Group(list(rewards), source="env")
    group_advantages([wm, env], anchor_weight=3.0)

    a_wm = np.array(wm.advantages, dtype=float)
    a_env = np.array(env.advantages, dtype=float)
    np.testing.assert_allclose(a_env, 3.0 * a_wm)


def test_counts_are_reported():
    groups = [
        Group([0.1, 0.9], source="env"),
        Group([0.2, 0.3, 0.4], source="wm"),
        Group([0.5, None], source="wm"),  # only one graded, skipped
    ]
    stats = group_advantages(groups, min_graded=2)
    assert stats == {
        "n_groups": 2,
        "n_rollouts": 5,
        "n_anchor_groups": 1,
        "n_degenerate": 0,
    }


def test_degenerate_group_is_counted_and_yields_zero_advantages():
    g = Group([0.5, 0.5, 0.5], source="wm")
    stats = group_advantages([g])
    assert stats["n_degenerate"] == 1
    np.testing.assert_allclose(np.array(g.advantages, dtype=float), 0.0)


def test_standardize_toggle_changes_only_the_scale():
    rewards = [0.1, 0.4, 0.9, 0.6]
    a = Group(list(rewards))
    b = Group(list(rewards))
    group_advantages([a], standardize=True)
    group_advantages([b], standardize=False)

    va = np.array(a.advantages, dtype=float)
    vb = np.array(b.advantages, dtype=float)
    assert np.corrcoef(va, vb)[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(va, vb)


def test_source_is_validated():
    with pytest.raises(ValueError):
        Group([0.1, 0.2], source="sandbox")
