import pytest

from wmrl.anchor import AnchorScheduler


def test_fraction_is_respected_over_many_steps():
    s = AnchorScheduler(fraction=0.10, min_per_step=0, seed=0)
    for _ in range(500):
        s.select(n_groups=32)
    assert s.realized_fraction == pytest.approx(0.10, abs=0.01)


def test_min_per_step_floor_beats_a_fraction_that_rounds_to_zero():
    s = AnchorScheduler(fraction=0.01, min_per_step=1, seed=0)
    for _ in range(20):
        assert len(s.select(n_groups=8)) == 1, "a step with no anchor group cannot feed the calibration"


def test_concurrency_cap_binds():
    s = AnchorScheduler(fraction=0.5, min_per_step=1, max_concurrent=4, seed=0)
    assert len(s.select(n_groups=32, in_flight=0)) == 4
    assert len(s.select(n_groups=32, in_flight=3)) == 1
    assert s.select(n_groups=32, in_flight=4) == []


def test_realized_fraction_falls_below_target_when_the_cap_binds():
    """The diagnostic that tells you the sandbox pool is setting the pace."""
    s = AnchorScheduler(fraction=0.5, min_per_step=1, max_concurrent=2, seed=0)
    for _ in range(100):
        s.select(n_groups=32, in_flight=0)
    assert s.realized_fraction < 0.5
    assert s.realized_fraction == pytest.approx(2 / 32, abs=1e-9)


def test_selection_is_deterministic_given_a_seed():
    a = AnchorScheduler(fraction=0.25, seed=7)
    b = AnchorScheduler(fraction=0.25, seed=7)
    for _ in range(10):
        assert a.select(16) == b.select(16)


def test_indices_are_sorted_unique_and_in_range():
    s = AnchorScheduler(fraction=0.3, seed=1)
    for _ in range(50):
        idx = s.select(n_groups=20)
        assert idx == sorted(idx)
        assert len(set(idx)) == len(idx)
        assert all(0 <= i < 20 for i in idx)


def test_degenerate_inputs():
    s = AnchorScheduler(fraction=0.1, min_per_step=1)
    assert s.select(n_groups=0) == []

    s_all = AnchorScheduler(fraction=1.0, seed=0)
    assert len(s_all.select(n_groups=5)) == 5

    with pytest.raises(ValueError):
        AnchorScheduler(fraction=1.5)
