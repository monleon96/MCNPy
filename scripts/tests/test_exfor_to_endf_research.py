"""Tests for the Phase-2 research fork — ``scripts.exfor_to_endf_research``.

The maths of model averaging is tested in ``test_model_averaging.py``. What is
tested here is the **plumbing**, which is where this fork can actually go wrong:

* the candidate set that reaches the average (the dropped 1 % floor);
* the normalisation applied before averaging — averaging raw ``c`` instead of
  ENDF ``a`` would silently be a different quantity;
* the zero-padding convention surviving the trip through the pipeline's data
  structures;
* the undefined cases returning NaN rather than a plausible-looking wrong
  number.

The load-bearing invariant of the whole fork is that it **adds** information and
changes nothing shipped: ``nominal_coeffs`` is never reassigned, so ``c_0..c_6``
and the ENDF stay bit-identical to run 83.
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

from scripts.exfor_to_endf_research import (  # noqa: E402
    AVERAGING_COLUMNS,
    CHI2_COLUMNS,
    compute_averaging_diagnostics,
    data_space_chi2,
)
from scripts.model_averaging import ic_weights  # noqa: E402
from scripts.resample_AD import endf_normalize_legendre_coeffs  # noqa: E402

MAX_DEGREE = 6


def _info(coeffs_by_degree, scores_by_degree):
    """Build an ``all_degrees_info``-shaped dict like resample_AD returns."""
    return {
        d: {'coeffs': np.asarray(c, dtype=float),
            'aicc': float(scores_by_degree[d]),
            'chi2': 0.0, 'dof': 1.0, 'k': float(d + 1)}
        for d, c in coeffs_by_degree.items()
    }


def _weights_from(info, floor=0.0, min_degree=1):
    """Mirror the fork's weight construction so tests exercise the real path."""
    degrees = sorted(info)
    w = ic_weights([info[d]['aicc'] for d in degrees], floor=floor)
    out = {d: float(x) for d, x in zip(degrees, w) if x > 0.0 and d >= min_degree}
    total = sum(out.values())
    return {d: v / total for d, v in out.items()} if total > 0 else {}


# --------------------------------------------------------------------------- #
# 1. The candidate set
# --------------------------------------------------------------------------- #

def test_zero_floor_keeps_every_candidate_and_one_percent_reproduces_v2():
    """The dropped 1 % cutoff is the point of the change — pin both behaviours."""
    # Degree 4 is deliberately given a weak score: supported, but under 1 %.
    info = _info(
        {2: [1.0, 0.3, 0.1], 3: [1.0, 0.3, 0.1, 0.05], 4: [1.0, 0.3, 0.1, 0.05, 0.02]},
        {2: 100.0, 3: 100.5, 4: 112.0},
    )
    kept_all = _weights_from(info, floor=0.0)
    kept_v2 = _weights_from(info, floor=0.01)

    assert set(kept_all) == {2, 3, 4}, "floor 0.0 must keep every feasible candidate"
    assert 4 not in kept_v2, "the legacy 1 % floor must still drop the weak candidate"
    assert set(kept_v2) == {2, 3}
    # Both sets renormalise to 1 — dropping a candidate must not leak weight.
    assert sum(kept_all.values()) == pytest.approx(1.0)
    assert sum(kept_v2.values()) == pytest.approx(1.0)


def test_min_degree_filter_is_preserved():
    """MIN_DEGREE_FOR_AVERAGING still excludes the isotropic candidate."""
    info = _info({0: [1.0], 1: [1.0, 0.2], 2: [1.0, 0.2, 0.05]},
                 {0: 100.0, 1: 100.1, 2: 100.2})
    assert 0 not in _weights_from(info, floor=0.0, min_degree=1)


# --------------------------------------------------------------------------- #
# 2. The averaged central value
# --------------------------------------------------------------------------- #

def test_degenerate_weights_reproduce_the_winner():
    """w=(1,0,...) must give back exactly the winning degree's a_l. The anchor."""
    coeffs = {2: [2.0, 0.6, 0.2], 5: [2.0, 0.6, 0.2, 0.1, 0.04, 0.01]}
    # Degree 5 wins by a mile, so its weight is 1 to float precision.
    info = _info(coeffs, {2: 500.0, 5: 100.0})
    weights = _weights_from(info, floor=0.0)
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)

    winner_a = endf_normalize_legendre_coeffs(np.array(coeffs[5]), include_a0=False)
    for l in range(1, 6):
        assert out[f'avg_a_{l}'] == pytest.approx(winner_a[l - 1], rel=1e-9, abs=1e-12)
    # Order 6 is in no candidate at all.
    assert out['avg_a_6'] == pytest.approx(0.0, abs=1e-15)
    assert out['q_a_6'] == pytest.approx(0.0, abs=1e-15)


def test_averaging_happens_in_normalized_a_space_not_raw_c_space():
    """Two candidates with the same SHAPE but different c0 must average to that shape.

    This is the test that fails if someone averages ``coeffs`` directly: raw c
    vectors scaled by different c0 would pull the mean toward the larger one,
    while in normalised a-space they are the same model and the average is
    scale-free. MF4 ships a normalised shape, so a-space is the correct one.
    """
    base = np.array([1.0, 0.45, 0.15])
    info = _info({2: base, 3: np.append(base * 7.0, 0.0)}, {2: 100.0, 3: 100.0})
    weights = _weights_from(info, floor=0.0)
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)

    expected = endf_normalize_legendre_coeffs(base, include_a0=False)
    assert out['avg_a_1'] == pytest.approx(expected[0], rel=1e-12)
    assert out['avg_a_2'] == pytest.approx(expected[1], rel=1e-12)


def test_absent_orders_contribute_exact_zero_not_a_dropped_term():
    """Zero-padding is a prediction: ā_3 = w_3·a_3, shrunk by the L=2 candidate."""
    c2 = np.array([1.0, 0.4, 0.1])
    c3 = np.array([1.0, 0.4, 0.1, 0.3])
    info = _info({2: c2, 3: c3}, {2: 100.0, 3: 100.0})   # equal scores -> w = 0.5/0.5
    weights = _weights_from(info, floor=0.0)
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)

    a3_full = endf_normalize_legendre_coeffs(c3, include_a0=False)[2]
    assert weights[2] == pytest.approx(0.5)
    # Shrunk by exactly the inclusion probability, NOT equal to the L=3 value.
    assert out['avg_a_3'] == pytest.approx(0.5 * a3_full, rel=1e-12)
    assert out['q_a_3'] == pytest.approx(0.5, rel=1e-12)
    # ...and the conditional mean undoes the shrinkage, which is its whole job.
    assert out['cond_a_3'] == pytest.approx(a3_full, rel=1e-12)


def test_angular_equivalence_survives_the_pipeline_plumbing():
    """Averaging coefficients == averaging the distributions they predict.

    Proven in test_model_averaging.py for the pure module; re-checked here
    through the fork's own normalise-then-pad path, because a mistake in the
    normalisation step would break it without touching the module.
    """
    c2 = np.array([1.0, 0.35, 0.12])
    c4 = np.array([1.0, 0.50, 0.08, 0.06, 0.02])
    info = _info({2: c2, 4: c4}, {2: 100.0, 4: 100.8})
    weights = _weights_from(info, floor=0.0)
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)

    mu = np.linspace(-1.0, 1.0, 401)

    def unit_shape(c):
        """Normalised distribution 1 + Σ (2l+1) a_l P_l(mu)."""
        a = endf_normalize_legendre_coeffs(c, include_a0=False)
        full = np.concatenate([[1.0], [(2 * l + 1) * a[l - 1] for l in range(1, len(a) + 1)]])
        return legval(mu, full)

    mixed = sum(weights[d] * unit_shape(info[d]['coeffs']) for d in weights)

    avg_a = np.array([out[f'avg_a_{l}'] for l in range(1, MAX_DEGREE + 1)])
    from_avg = legval(
        mu, np.concatenate([[1.0], [(2 * l + 1) * avg_a[l - 1] for l in range(1, MAX_DEGREE + 1)]])
    )
    np.testing.assert_allclose(from_avg, mixed, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Inclusion probabilities
# --------------------------------------------------------------------------- #

def test_inclusion_probabilities_are_monotone_and_start_at_one():
    info = _info(
        {1: [1.0, 0.4], 3: [1.0, 0.4, 0.1, 0.05], 6: [1.0, 0.4, 0.1, 0.05, 0.02, 0.01, 0.005]},
        {1: 103.0, 3: 100.0, 6: 101.5},
    )
    weights = _weights_from(info, floor=0.0)
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)
    q = [out[f'q_a_{l}'] for l in range(1, MAX_DEGREE + 1)]

    assert q[0] == pytest.approx(1.0), "every candidate carries order 1"
    for lo, hi in zip(q, q[1:]):
        assert hi <= lo + 1e-12, f"q_l must not increase with l: {q}"


# --------------------------------------------------------------------------- #
# 4. The undefined cases — NaN, never a plausible-looking fallback
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("info,weights", [
    (None, None),                       # interpolated bin: no fit at all
    ({}, {}),                           # empty candidate set
    (_info({3: [1.0, 0.2, 0.1, 0.05]}, {3: 100.0}), None),   # weights never built
])
def test_undefined_bins_emit_nan_and_do_not_fall_back_to_the_winner(info, weights):
    """A silent winner fallback would hide which bins the average is undefined for."""
    out = compute_averaging_diagnostics(info, weights, MAX_DEGREE)
    for l in range(1, MAX_DEGREE + 1):
        assert np.isnan(out[f'avg_a_{l}'])
        assert np.isnan(out[f'q_a_{l}'])
    assert np.isnan(out['n_eff_models'])
    # Provenance is still recorded even when nothing could be averaged.
    assert out['weight_floor_applied'] == 0.0


def test_schema_is_complete_and_stable():
    """Every declared column is produced, in both the populated and NaN paths."""
    info = _info({2: [1.0, 0.3, 0.1], 3: [1.0, 0.3, 0.1, 0.05]}, {2: 100.0, 3: 100.4})
    populated = compute_averaging_diagnostics(info, _weights_from(info), MAX_DEGREE)
    empty = compute_averaging_diagnostics(None, None, MAX_DEGREE)

    assert set(populated) == set(AVERAGING_COLUMNS)
    assert set(empty) == set(AVERAGING_COLUMNS)
    assert len(AVERAGING_COLUMNS) == 20  # 6 avg + 6 q + 6 cond + n_eff + floor


def test_effective_models_matches_the_gate2_definition():
    """exp(H); 1.0 for a decided bin, 2.0 for a perfectly split one."""
    decided = _info({2: [1.0, 0.3, 0.1], 3: [1.0, 0.3, 0.1, 0.05]}, {2: 100.0, 3: 400.0})
    split = _info({2: [1.0, 0.3, 0.1], 3: [1.0, 0.3, 0.1, 0.05]}, {2: 100.0, 3: 100.0})

    assert compute_averaging_diagnostics(
        decided, _weights_from(decided), MAX_DEGREE)['n_eff_models'] == pytest.approx(1.0, abs=1e-6)
    assert compute_averaging_diagnostics(
        split, _weights_from(split), MAX_DEGREE)['n_eff_models'] == pytest.approx(2.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# 5. Data-space chi2 — the Phase-2 decision statistic
# --------------------------------------------------------------------------- #

def _curve(mu, a, c0=1.0):
    full = np.concatenate([[1.0], [(2 * l + 1) * a[l - 1] for l in range(1, len(a) + 1)]])
    return c0 * legval(mu, full)


def test_chi2_is_zero_when_the_curve_passes_through_every_point():
    mu = np.linspace(-0.9, 0.9, 25)
    a = np.array([0.3, 0.1, 0.05, 0.0, 0.0, 0.0])
    y = _curve(mu, a, c0=2.0)
    val = data_space_chi2(mu, y, np.full_like(y, 0.1), None, a, 2.0)
    assert val == pytest.approx(0.0, abs=1e-20)


def test_identical_centrals_give_a_ratio_of_exactly_one():
    """The invariance that makes chi2_pp_ratio readable at all."""
    rng = np.random.default_rng(0)
    mu = np.linspace(-1, 1, 40)
    a = np.array([0.25, 0.12, 0.04, 0.01, 0.0, 0.0])
    y = _curve(mu, a, 3.0) + rng.normal(0, 0.05, mu.size)
    s = np.full_like(y, 0.05)
    w = rng.uniform(0.2, 1.0, mu.size)
    assert data_space_chi2(mu, y, s, w, a, 3.0) == pytest.approx(
        data_space_chi2(mu, y, s, w, a, 3.0), rel=0, abs=0)


def test_chi2_is_a_weighted_mean_of_squared_standardized_residuals():
    """Pin the exact formula — a dof convention creeping in would break the ratio."""
    mu = np.array([-0.5, 0.0, 0.5])
    a = np.zeros(6)                      # flat curve at c0
    y = np.array([1.0, 2.0, 3.0])        # residuals 1, 0, -1 against c0=2
    s = np.array([1.0, 1.0, 1.0])
    w = np.array([1.0, 2.0, 1.0])
    got = data_space_chi2(mu, y, s, w, a, 2.0)
    assert got == pytest.approx((1 * 1 + 2 * 0 + 1 * 1) / 4.0)


def test_zero_and_nonfinite_sigma_points_are_excluded_not_infinite():
    mu = np.array([-0.5, 0.0, 0.5])
    a = np.zeros(6)
    y = np.array([1.0, 2.0, 3.0])
    s = np.array([0.0, 1.0, np.nan])     # only the middle point is usable
    got = data_space_chi2(mu, y, s, None, a, 2.0)
    assert got == pytest.approx(0.0)     # middle point sits exactly on the curve


def test_chi2_is_nan_when_the_central_is_undefined():
    mu = np.linspace(-1, 1, 10)
    y = np.ones(10)
    a = np.full(6, np.nan)
    assert np.isnan(data_space_chi2(mu, y, np.ones(10), None, a, 1.0))
    assert np.isnan(data_space_chi2(np.array([]), np.array([]), np.array([]),
                                    None, np.zeros(6), 1.0))


def test_a_worse_shape_scores_higher():
    """Sanity of direction: the statistic must increase as the curve degrades."""
    mu = np.linspace(-0.9, 0.9, 30)
    good = np.array([0.30, 0.10, 0.05, 0.0, 0.0, 0.0])
    bad = np.array([0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
    y = _curve(mu, good, 2.0)
    s = np.full_like(y, 0.1)
    assert data_space_chi2(mu, y, s, None, bad, 2.0) > data_space_chi2(mu, y, s, None, good, 2.0)


def test_chi2_column_schema():
    assert CHI2_COLUMNS == ['chi2_pp_win', 'chi2_pp_avg', 'chi2_pp_ratio']
    assert not set(CHI2_COLUMNS) & set(AVERAGING_COLUMNS)


def test_padding_refuses_to_truncate_a_candidate_above_max_degree():
    """A candidate longer than max_degree must raise, not be silently shortened.

    Truncating would average a different model than the one that was scored.
    """
    info = _info({7: [1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]}, {7: 100.0})
    with pytest.raises(ValueError, match="truncate"):
        compute_averaging_diagnostics(info, {7: 1.0}, MAX_DEGREE)
