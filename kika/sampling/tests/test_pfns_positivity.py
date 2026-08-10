"""The P⁰-weighted projection, and the regression that says why it is weighted.

The projection has one job: put a drawn ratio vector back on
``{r ≥ 0, Σ P⁰ r = S}`` while moving it as little as possible. The whole
argument for weighting it is in
:func:`test_projection_does_not_annihilate_small_groups`, which the textbook
unweighted simplex projection fails outright — and fails by destroying a third
of a fission spectrum's grid to repair a 0.003 % normalisation error.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.sampling.pfns_positivity import check_ratios, project_ratios_to_simplex

#: Cf-252's group probabilities span roughly this range, and that span is the
#: reason the weighting matters.
WIDE_P0 = np.array([1e-17, 1e-12, 1e-6, 0.5, 0.5])


def unweighted_simplex(probabilities: np.ndarray, target: float) -> np.ndarray:
    """The textbook Euclidean projection onto ``{P ≥ 0, ΣP = S}``.

    Present only so the test below can show what it does to this data. It is
    the right projection of the wrong thing: it treats a probability of 1e-17
    and one of 1e-1 as equally movable in absolute terms.
    """
    ordered = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(ordered)
    k = np.arange(1, probabilities.size + 1)
    condition = ordered - (cumulative - target) / k > 0
    rho = int(np.nonzero(condition)[0][-1]) + 1
    theta = (cumulative[rho - 1] - target) / rho
    return np.maximum(probabilities - theta, 0.0)


# ---------------------------------------------------------------------------
# The identities
# ---------------------------------------------------------------------------

def test_projection_is_identity_when_already_feasible():
    """``t == 0.0`` exactly, and the vector comes back untouched.

    This is the common case — the measured drift is ~3e-5 — so if the
    projection did arithmetic here it would be a second, invisible perturbation
    applied to every sample.
    """
    ratios = np.ones(5)
    result, shift, clipped = project_ratios_to_simplex(
        ratios, WIDE_P0, WIDE_P0.sum()
    )
    assert shift == 0.0
    assert clipped == 0
    np.testing.assert_array_equal(result, ratios)


def test_projection_restores_the_sum_to_machine_precision():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        p0 = np.abs(rng.lognormal(0.0, 6.0, n))
        p0 /= p0.sum()
        ratios = 1.0 + rng.normal(0.0, 0.4, n)

        result, _, _ = project_ratios_to_simplex(ratios, p0, 1.0)
        assert np.dot(p0, result) == pytest.approx(1.0, abs=1e-12)
        assert np.all(result >= 0.0)


def test_projection_does_not_annihilate_small_groups():
    """**The regression the weighting exists for.**

    Five groups spanning 1e-17 to 0.5, and a normalisation drift of 3e-5 — the
    measured budget. The weighted projection shifts every *ratio* by the same
    ~3e-5 and leaves every group strictly positive. The unweighted one absorbs
    the drift in absolute probability, which needs a threshold of order 1e-5
    and therefore zeroes the three smallest groups outright.
    """
    target = WIDE_P0.sum()
    ratios = np.ones(WIDE_P0.size)
    ratios[3] += 3e-5 * target / WIDE_P0[3]          # inject the drift

    result, shift, clipped = project_ratios_to_simplex(ratios, WIDE_P0, target)

    assert clipped == 0
    assert np.all(result > 0.0), "the weighted projection zeroed a group"
    assert np.max(np.abs(result - 1.0)) < 1e-4, "ratios moved by more than the drift"
    assert shift == pytest.approx(-3e-5, rel=1e-3)

    # And the comparison that makes the point rather than asserting it.
    drifted = WIDE_P0 * ratios
    naive = unweighted_simplex(drifted, target)
    assert np.count_nonzero(naive == 0.0) >= 3, (
        "the unweighted projection was expected to destroy the small groups; "
        "if it no longer does, this test has stopped being evidence"
    )


def test_projection_clips_only_genuinely_negative_ratios():
    p0 = np.array([0.2, 0.3, 0.25, 0.25])
    ratios = np.array([1.0, -0.5, 1.0, 1.0])

    result, _, clipped = project_ratios_to_simplex(ratios, p0, p0.sum())

    assert clipped == 1
    assert result[1] == 0.0
    assert np.all(result[[0, 2, 3]] > 0.0)
    assert np.dot(p0, result) == pytest.approx(p0.sum(), abs=1e-14)


def test_a_zero_weight_group_is_left_alone():
    """``P⁰ = 0`` carries no weight, so the constraint says nothing about it.

    Its ratio is undefined rather than free, and the caller has already frozen
    its delta. Moving it here would be inventing a value.
    """
    p0 = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
    ratios = np.array([7.0, 1.0, 1.0, 1.0, 1.0 + 1e-4])

    result, _, _ = project_ratios_to_simplex(ratios, p0, p0.sum())

    assert result[0] == 7.0
    assert np.dot(p0, result) == pytest.approx(p0.sum(), abs=1e-14)


def test_an_all_zero_weight_vector_is_a_no_op():
    ratios = np.array([1.0, 2.0, 3.0])
    result, shift, clipped = project_ratios_to_simplex(ratios, np.zeros(3), 0.0)
    np.testing.assert_array_equal(result, ratios)
    assert (shift, clipped) == (0.0, 0)


def test_everything_negative_still_produces_a_feasible_answer():
    """The degenerate branch: no prefix survives its own shift."""
    p0 = np.array([0.5, 0.5])
    ratios = np.array([-3.0, -4.0])
    result, _, _ = project_ratios_to_simplex(ratios, p0, 1.0)
    assert np.all(result >= 0.0)
    assert np.dot(p0, result) == pytest.approx(1.0, abs=1e-12)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="3 ratios for 4 probabilities"):
        project_ratios_to_simplex(np.ones(3), np.ones(4), 4.0)


def test_the_variance_metric_is_refused_with_its_reason():
    """Deferred, not forgotten: the parameter is here so adding it is not a break."""
    with pytest.raises(NotImplementedError, match="variance-weighted"):
        project_ratios_to_simplex(np.ones(3), np.ones(3), 3.0, metric="variance")


# ---------------------------------------------------------------------------
# check_* before project_*
# ---------------------------------------------------------------------------

def test_check_reports_infeasibility_without_changing_anything():
    """Measure first, so the projection's effect is a difference, not a claim."""
    p0 = np.array([0.25, 0.25, 0.5])
    ratios = np.array([1.1, -0.2, 1.0])

    report = check_ratios(ratios, p0, 1.0)

    assert report["is_feasible"] is False
    assert report["n_negative"] == 1
    assert report["sum_error"] == pytest.approx(np.dot(p0, ratios) - 1.0)
    assert report["min_ratio"] == -0.2
    assert report["negative_mass_fraction"] == pytest.approx(0.05)


def test_check_calls_a_feasible_vector_feasible():
    p0 = np.array([0.5, 0.5])
    report = check_ratios(np.ones(2), p0, 1.0)
    assert report["is_feasible"] is True
    assert report["n_negative"] == 0
    assert report["sum_error"] == 0.0
