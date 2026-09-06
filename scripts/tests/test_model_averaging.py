"""Tests for ``scripts.model_averaging`` — Phase 1 of the MF4 research roadmap.

The load-bearing properties are the two the roadmap names as exit criteria:

* the scalar mixture reproduces ``E[a]=pm`` and ``Var(a)=ps²+p(1-p)m²``, so the
  analytic mixture really is what the discrete degree-sampling MC estimates;
* **angular equivalence** — averaging the coefficients equals averaging the
  angular distributions they predict. This is what licenses averaging in
  coefficient space at all, and it is where the zero-padding convention is
  either right or wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.model_averaging import (  # noqa: E402
    conditional_mean,
    effective_n_models,
    ic_weights,
    inclusion_probabilities,
    mixture_moments,
    nested_masks,
    pad_to,
    stack_padded,
)

L_MAX = 6


# --------------------------------------------------------------------------
# IC weights
# --------------------------------------------------------------------------

def test_ic_weights_match_the_akaike_definition() -> None:
    scores = np.array([10.0, 12.0, 16.0])
    w = ic_weights(scores)
    raw = np.exp(-0.5 * (scores - scores.min()))
    assert w == pytest.approx(raw / raw.sum())
    assert w.sum() == pytest.approx(1.0)


def test_ic_weights_survive_scores_that_would_overflow_exp() -> None:
    """chi2-based IC scores run to hundreds; exp(+score/2) overflows.

    The shift-to-minimum is not cosmetic — without it these come back NaN.
    """
    scores = np.array([2000.0, 2001.0, 2400.0])
    w = ic_weights(scores)
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    assert w[2] == pytest.approx(0.0, abs=1e-80)


def test_ic_weights_are_invariant_to_a_constant_shift() -> None:
    """Only differences matter — the property the whole scheme relies on."""
    scores = np.array([10.0, 12.0, 16.0])
    base = ic_weights(scores)
    for shift in (-500.0, 0.0, 1e4):
        assert ic_weights(scores + shift) == pytest.approx(base, rel=1e-12)


def test_non_finite_scores_are_dropped_not_propagated() -> None:
    """A rank-deficient degree yields NaN; it must not poison the vector."""
    w = ic_weights([10.0, np.nan, 12.0, np.inf])
    assert np.all(np.isfinite(w))
    assert w[1] == 0.0 and w[3] == 0.0
    assert w.sum() == pytest.approx(1.0)
    assert w[[0, 2]] == pytest.approx(ic_weights([10.0, 12.0]))


def test_all_scores_non_finite_gives_zero_weights_not_nan() -> None:
    w = ic_weights([np.nan, np.inf])
    assert np.all(w == 0.0)


def test_weight_floor_drops_and_renormalises() -> None:
    w = ic_weights([10.0, 12.0, 30.0], floor=0.01)
    assert w[2] == 0.0
    assert w.sum() == pytest.approx(1.0)


def test_effective_n_models_counts_supported_models() -> None:
    assert effective_n_models([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_n_models([0.25] * 4) == pytest.approx(4.0)
    assert effective_n_models([]) == 0.0


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------

def test_pad_to_zero_fills_absent_orders() -> None:
    assert pad_to([0.3, 0.2], L_MAX) == pytest.approx([0.3, 0.2, 0, 0, 0, 0])


def test_pad_to_refuses_to_truncate() -> None:
    """Truncating would silently score one model and average a different one."""
    with pytest.raises(ValueError, match="different model"):
        pad_to([0.1] * 7, L_MAX)


def test_stack_padded_builds_the_candidate_matrix() -> None:
    A = stack_padded([[0.3], [0.3, 0.2], [0.3, 0.2, 0.1]], L_MAX)
    assert A.shape == (3, L_MAX)
    assert A[0] == pytest.approx([0.3, 0, 0, 0, 0, 0])
    assert A[2] == pytest.approx([0.3, 0.2, 0.1, 0, 0, 0])


def test_nested_masks_and_inclusion_probabilities_are_tail_sums() -> None:
    degrees = [2, 3, 5]
    w = np.array([0.5, 0.3, 0.2])
    q = inclusion_probabilities(w, nested_masks(degrees, L_MAX))
    # order 1,2 in all; 3 in the last two; 4,5 in the last; 6 in none
    assert q == pytest.approx([1.0, 1.0, 0.5, 0.2, 0.2, 0.0])


# --------------------------------------------------------------------------
# The scalar reframing: E[a]=pm, Var=ps^2+p(1-p)m^2
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p,m,s", [(0.2, 0.02, 0.01), (0.5, -0.3, 0.05), (0.9, 1.4, 0.2)])
def test_scalar_two_model_mixture_reproduces_the_closed_form(p, m, s) -> None:
    """One model includes the order (mean m, var s^2), the other does not."""
    weights = np.array([1.0 - p, p])
    means = np.array([[0.0], [m]])
    covs = np.array([[[0.0]], [[s ** 2]]])

    out = mixture_moments(weights, means, covs)
    assert out["mean"][0] == pytest.approx(p * m)
    assert out["total"][0, 0] == pytest.approx(p * s ** 2 + p * (1 - p) * m ** 2)
    assert out["within"][0, 0] == pytest.approx(p * s ** 2)
    assert out["between"][0, 0] == pytest.approx(p * (1 - p) * m ** 2)


def test_large_relative_sigma_at_a_near_zero_mean_is_expected() -> None:
    """The roadmap's §4.2 worked example — 230 % relative sigma, nothing wrong.

    Pinned because a number like this looks like a covariance failure and has
    previously been read as one.
    """
    p, m, s = 0.2, 0.02, 0.01
    out = mixture_moments([1 - p, p], np.array([[0.0], [m]]),
                          np.array([[[0.0]], [[s ** 2]]]))
    mean = out["mean"][0]
    sigma = np.sqrt(out["total"][0, 0])
    assert mean == pytest.approx(0.004)
    assert sigma == pytest.approx(0.0092, abs=5e-5)
    assert sigma / mean == pytest.approx(2.3, abs=0.05)


# --------------------------------------------------------------------------
# Law of total covariance
# --------------------------------------------------------------------------

def test_total_is_exactly_within_plus_between() -> None:
    rng = np.random.default_rng(0)
    w = ic_weights(rng.normal(20, 3, size=5))
    means = rng.normal(size=(5, L_MAX)) * 0.1
    covs = np.stack([(lambda A: A @ A.T)(rng.normal(size=(L_MAX, L_MAX)) * 0.05)
                     for _ in range(5)])
    out = mixture_moments(w, means, covs)
    assert out["total"] == pytest.approx(out["within"] + out["between"])


def test_total_covariance_is_psd_for_psd_inputs() -> None:
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(2, 7))
        w = ic_weights(rng.normal(50, 10, size=n))
        means = rng.normal(size=(n, L_MAX)) * 0.3
        covs = np.stack([(lambda A: A @ A.T)(rng.normal(size=(L_MAX, L_MAX)))
                         for _ in range(n)])
        total = mixture_moments(w, means, covs)["total"]
        assert np.allclose(total, total.T)
        assert np.linalg.eigvalsh(total).min() > -1e-9


def test_between_model_term_vanishes_when_candidates_agree() -> None:
    means = np.tile(np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0]), (3, 1))
    out = mixture_moments([0.5, 0.3, 0.2], means)
    assert out["between"] == pytest.approx(np.zeros((L_MAX, L_MAX)))


def test_a_single_candidate_reduces_to_that_candidate() -> None:
    mu = np.array([[0.3, 0.2, 0.1, 0.0, 0.0, 0.0]])
    cov = np.eye(L_MAX)[None, :, :] * 0.01
    out = mixture_moments([1.0], mu, cov)
    assert out["mean"] == pytest.approx(mu[0])
    assert out["total"] == pytest.approx(cov[0])
    assert out["between"] == pytest.approx(np.zeros((L_MAX, L_MAX)))


def test_weights_are_renormalised_and_zero_sum_raises() -> None:
    mu = np.array([[0.1] * L_MAX, [0.2] * L_MAX])
    a = mixture_moments([1.0, 1.0], mu)["mean"]
    b = mixture_moments([0.5, 0.5], mu)["mean"]
    assert a == pytest.approx(b)
    with pytest.raises(ValueError, match="positive finite"):
        mixture_moments([0.0, 0.0], mu)


# --------------------------------------------------------------------------
# Angular equivalence — the property that licenses averaging in a-space
# --------------------------------------------------------------------------

def _pdf(a: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
    """ENDF MF4: p(mu) = 1/2 [1 + sum_{l>=1} (2l+1) a_l P_l(mu)]."""
    c = np.concatenate([[1.0], (2 * np.arange(1, a.size + 1) + 1) * a])
    return 0.5 * legval(mu_grid, c)


def test_averaging_coefficients_equals_averaging_angular_distributions() -> None:
    """The exit criterion. If this fails, a-space averaging is not justified.

    Candidates of genuinely different degree, so the zero padding is exercised:
    a degree-2 model must contribute exactly zero to a_3..a_6, and the identity
    only holds if that zero is the right model statement.
    """
    mu_grid = np.linspace(-1.0, 1.0, 2001)
    cands = [[0.30, 0.18],
             [0.28, 0.20, 0.09],
             [0.31, 0.17, 0.11, -0.04],
             [0.29, 0.19, 0.10, -0.03, 0.02, -0.01]]
    A = stack_padded(cands, L_MAX)
    w = ic_weights([10.0, 10.8, 12.5, 14.0])

    mean = mixture_moments(w, A)["mean"]
    pdf_of_mean = _pdf(mean, mu_grid)
    mean_of_pdfs = np.tensordot(w, np.stack([_pdf(a, mu_grid) for a in A]), axes=1)

    assert pdf_of_mean == pytest.approx(mean_of_pdfs, rel=1e-12, abs=1e-12)


def test_the_averaged_distribution_still_integrates_to_one() -> None:
    """Normalisation is preserved because a_0 is excluded from the averaging.

    Gauss-Legendre rather than a dense trapezoid: p(mu) is a polynomial of
    degree <= L_MAX, so an 8-node rule is exact to machine precision and the
    assertion tests the identity instead of a quadrature error floor.
    """
    nodes, wts = np.polynomial.legendre.leggauss(8)
    A = stack_padded([[0.3, 0.18], [0.28, 0.20, 0.09], [0.31, 0.17, 0.11, -0.04]],
                     L_MAX)
    mean = mixture_moments(ic_weights([10.0, 11.0, 13.0]), A)["mean"]
    assert float(wts @ _pdf(mean, nodes)) == pytest.approx(1.0, rel=1e-14)


# --------------------------------------------------------------------------
# Conditional mean — diagnostic only
# --------------------------------------------------------------------------

def test_zero_padded_mean_is_the_conditional_mean_times_inclusion_probability() -> None:
    """Makes the E[a] = p*m decomposition explicit and checkable per order."""
    degrees = [2, 4, 6]
    w = ic_weights([10.0, 11.0, 13.0])
    A = stack_padded([[0.30, 0.18],
                      [0.28, 0.20, 0.09, -0.04],
                      [0.31, 0.17, 0.11, -0.03, 0.02, -0.01]], L_MAX)
    masks = nested_masks(degrees, L_MAX)

    mean = mixture_moments(w, A)["mean"]
    q = inclusion_probabilities(w, masks)
    m = conditional_mean(w, A, masks)

    assert mean == pytest.approx(q * m)
    # Shrinkage is real wherever an order is not universally present.
    assert np.all(np.abs(mean[q < 1.0]) < np.abs(m[q < 1.0]) + 1e-15)


def test_conditional_mean_is_zero_where_no_candidate_offers_an_opinion() -> None:
    w = ic_weights([10.0, 11.0])
    A = stack_padded([[0.3, 0.2], [0.3, 0.2, 0.1]], L_MAX)
    m = conditional_mean(w, A, nested_masks([2, 3], L_MAX))
    assert m[3:] == pytest.approx(np.zeros(3))
    assert np.all(np.isfinite(m))
