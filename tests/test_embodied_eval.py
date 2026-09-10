"""The reported metrics for the embodied track."""

import pytest

from embodied.evaluate import summarize


def test_a_perfectly_reliable_policy():
    s = summarize({"a": [True] * 8, "b": [True] * 8})
    assert s == {"avg": 100.0, "best_at_8": 100.0, "all_at_8": 100.0, "n_states": 2}


def test_a_policy_that_never_succeeds():
    s = summarize({"a": [False] * 8})
    assert (s["avg"], s["best_at_8"], s["all_at_8"]) == (0.0, 0.0, 0.0)


def test_best_and_all_bracket_the_average():
    """They are the reliability bracket, so this ordering must always hold."""
    successes = {
        "a": [True] + [False] * 7,   # solved once
        "b": [True] * 8,             # solved every time
        "c": [True] * 4 + [False] * 4,
    }
    s = summarize(successes)
    assert s["all_at_8"] <= s["avg"] <= s["best_at_8"]


def test_the_metrics_separate_capable_from_reliable():
    """Two policies with the same average, told apart by the other two columns."""
    flaky = {f"s{i}": [True] + [False] * 7 for i in range(8)}
    steady = {f"s{i}": [True] * 8 if i == 0 else [False] * 8 for i in range(8)}

    a, b = summarize(flaky), summarize(steady)
    assert a["avg"] == pytest.approx(b["avg"]), "same average by construction"
    assert a["best_at_8"] > b["best_at_8"], "the flaky policy touches more tasks"
    assert a["all_at_8"] < b["all_at_8"], "the steady policy actually finishes one"


def test_no_results_is_not_a_crash():
    assert summarize({})["n_states"] == 0


def test_percentages_are_rounded_to_one_decimal():
    s = summarize({"a": [True, False, False]})
    assert s["avg"] == 33.3
