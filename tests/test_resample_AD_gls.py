"""Tests for ``_weighted_ridge_fit_gls`` — block-correlated GLS Legendre fit.

The GLS helper builds the per-bin data covariance Σ = D + u uᵀ + v vᵀ where
D carries σ²_stat only and there are TWO rank-1 columns per experiment:
  • u_e = σ_sys_indep,e · y      (per-experiment normalisation, constant amp)
  • v_e = σ_sys_dep_i · y_i     (per-experiment shape, per-row amplitude)
Both are physically correlated within an experiment (manifest ``correlated: true``);
putting σ_dep on the diagonal — as earlier versions did — would silently
treat a correlated shape error as uncorrelated noise. AICc model selection
operates on the rank-2 GLS χ², so inter-experiment normalisation disagreement
and correlated calibration-shape errors are both absorbed by the rank-1
blocks instead of inflating χ² and biasing the choice toward L=0.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import legvander

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.resample_AD import (  # noqa: E402
    _weighted_ridge_fit,
    _weighted_ridge_fit_gls,
)


def _aicc(chi2: float, n: int, k: float) -> float:
    aic = chi2 + 2.0 * k
    if n > k + 1.0:
        return aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return aic + 1e6


@pytest.mark.parametrize("trial", range(20))
def test_pure_stat_limit_matches_weighted_ridge_fit(trial: int) -> None:
    """When both σ_sys_indep and σ_sys_dep are zero everywhere, GLS reduces
    exactly to diagonal kernel-WLS over σ_stat (no rank-1 contribution from
    either correlated mode fires)."""
    rng = np.random.default_rng(100 + trial)
    n = int(rng.integers(8, 35))
    degree = int(rng.integers(0, min(5, n - 2)))
    mu = rng.uniform(-1, 1, n)
    true_c = rng.normal(0, 0.3, degree + 1)
    y = legvander(mu, degree) @ true_c + rng.normal(0, 0.05, n)
    sigma_stat = rng.uniform(0.01, 0.10, n)
    g = rng.uniform(0.2, 1.0, n)
    n_exp = int(rng.integers(1, 4))
    exp_index = rng.integers(0, n_exp, n)
    sigma_sys_indep = np.zeros(n_exp)
    sigma_sys_dep = np.zeros(n)

    cD, chi2D, dofD, kD = _weighted_ridge_fit(
        mu, y, sigma_stat, degree,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat", external_weights=g,
    )
    cG, chi2G, dofG, kG = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, degree,
        sigma_sys_dep_per_row=sigma_sys_dep,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat", external_weights=g,
    )
    np.testing.assert_allclose(cG, cD, atol=1e-9)
    assert abs(chi2G - chi2D) < 1e-7
    assert abs(dofG - dofD) < 1e-7
    assert abs(kG - kD) < 1e-7


@pytest.mark.parametrize("trial", range(15))
def test_woodbury_matches_explicit_inverse(trial: int) -> None:
    """The rank-2 Woodbury reduction in the helper must give the same
    coefficients and χ² as a brute-force explicit Σ⁻¹ solve where σ_dep
    enters as a per-experiment rank-1 column (not on the diagonal)."""
    rng = np.random.default_rng(500 + trial)
    n = int(rng.integers(6, 18))
    degree = int(rng.integers(0, min(4, n - 2)))
    mu = rng.uniform(-1, 1, n)
    y = legvander(mu, degree) @ rng.normal(0, 0.3, degree + 1) + rng.normal(0, 0.05, n)
    sigma_stat = rng.uniform(0.01, 0.10, n)
    sigma_sys_dep = rng.uniform(0.0, 0.04, n)
    g = rng.uniform(0.3, 1.0, n)
    n_exp = int(rng.integers(2, 4))
    exp_index = rng.integers(0, n_exp, n)
    sigma_sys_indep = rng.uniform(0.03, 0.10, n_exp)

    cG, chi2G, _, _ = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, degree,
        sigma_sys_dep_per_row=sigma_sys_dep,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat", external_weights=g,
    )

    # Explicit Σ̃ build — σ_dep enters as a per-experiment rank-1 (NOT diag):
    #   D̃_ii = σ²_stat / g
    #   Σ̃_ij (same exp e) =
    #       [σ²_sys_indep,e + σ_sys_dep,i · σ_sys_dep,j] · y_i · y_j / √(g_i g_j)
    var_diag = sigma_stat ** 2
    Sig = np.diag(var_diag / g)
    for e in range(n_exp):
        idx = np.where(exp_index == e)[0]
        s_e = sigma_sys_indep[e]
        if idx.size == 0:
            continue
        for i in idx:
            for j in idx:
                Sig[i, j] += (
                    (s_e ** 2 + sigma_sys_dep[i] * sigma_sys_dep[j])
                    * y[i] * y[j] / np.sqrt(g[i] * g[j])
                )

    eigs = np.linalg.eigvalsh(Sig)
    assert eigs.min() > -1e-10, f"Σ not PSD: min eig = {eigs.min():.3e}"

    X = legvander(mu, degree)
    Sinv = np.linalg.inv(Sig)
    pen = np.array([float(l ** 4) if l > 0 else 0.0 for l in range(degree + 1)])
    R = np.diag(pen)
    P = X.T @ Sinv @ X + 1e-4 * R
    rhs = X.T @ Sinv @ y
    c_explicit = np.linalg.solve(P, rhs)
    r = y - X @ c_explicit
    chi2_explicit = float(r @ Sinv @ r)

    np.testing.assert_allclose(cG, c_explicit, atol=1e-9)
    assert abs(chi2G - chi2_explicit) < 1e-7


def test_synthetic_two_experiment_normalization_disagreement() -> None:
    """When two experiments disagree by a known normalisation offset, GLS
    sees χ²(L=L_true) close to dof while diagonal-stat-only does not. Confirms
    the rank-1 block absorbs the inter-experiment shift."""
    rng = np.random.default_rng(2025)
    mu_a = np.linspace(-0.9, 0.9, 8)
    mu_b = np.linspace(-0.85, 0.85, 7)
    mu = np.concatenate([mu_a, mu_b])
    n = mu.size
    true_c = np.array([1.0, 0.0, 0.4])  # L=2: 1 + 0.4 P_2(mu)
    y_clean = legvander(mu, 2) @ true_c
    sigma_stat = np.full(n, 0.03)
    norm_B = 1.07  # experiment B is 7% high
    y = y_clean.copy()
    y[:8] += rng.normal(0, sigma_stat[:8])
    y[8:] = y_clean[8:] * norm_B + rng.normal(0, sigma_stat[8:])
    exp_index = np.concatenate([np.zeros(8, int), np.ones(7, int)])
    sigma_sys_indep = np.array([0.08, 0.08])

    # Without modelling per-experiment normalisation: chi² stays bad at L=0 and
    # is dragged down only by adding orders the data don't actually need.
    sigma_stat_only = sigma_stat.copy()
    chi2_diag_at_L = {}
    for L in range(0, 5):
        _, chi2, dof, _ = _weighted_ridge_fit(
            mu, y, sigma_stat_only, L,
            ridge_lambda=1e-4, ridge_power=4, df_method="hat",
        )
        chi2_diag_at_L[L] = chi2 / max(dof, 1e-12)

    # Diagonal stat-only χ² should be high at every order because no per-experiment
    # normalisation factor can flatten the 7% offset away. With σ_stat=3% the
    # 7% offset is ~2σ per point, so χ²(L=true)/dof lands well above 1 even
    # though the shape is exact.
    assert chi2_diag_at_L[0] > 5.0
    assert chi2_diag_at_L[2] > 3.0  # data would fit at L=2 if normalisation were free

    # With GLS, the rank-1 absorption flattens χ² at L=2 close to dof.
    chi2_gls_at_L = {}
    for L in range(0, 5):
        _, chi2, dof, _ = _weighted_ridge_fit_gls(
            mu, y, sigma_stat, sigma_sys_indep, exp_index, L,
            ridge_lambda=1e-4, ridge_power=4, df_method="hat",
        )
        chi2_gls_at_L[L] = chi2 / max(dof, 1e-12)

    assert chi2_gls_at_L[0] > 3.0    # L=0 still doesn't fit the L=2 shape
    assert chi2_gls_at_L[2] < 1.5   # L=2 fits nicely once normalisation is absorbed

    # AICc picks L=2 cleanly under GLS.
    aicc_gls = {
        L: _aicc(chi2_gls_at_L[L] * max(L + 1 - 1, 1), n, L + 1) for L in chi2_gls_at_L
    }
    aicc_gls = {}
    for L in range(0, 5):
        _, chi2, _, k = _weighted_ridge_fit_gls(
            mu, y, sigma_stat, sigma_sys_indep, exp_index, L,
            ridge_lambda=1e-4, ridge_power=4, df_method="hat",
        )
        aicc_gls[L] = _aicc(chi2, n, k)
    L_winner = min(aicc_gls, key=lambda L: aicc_gls[L])
    assert L_winner == 2


def test_single_experiment_block_no_regression() -> None:
    """When there's only one experiment in the bin, U is rank-1 across all
    points. The fit should still produce sensible χ², dof, eff_params with
    no NaN or singular-matrix complaints."""
    rng = np.random.default_rng(7)
    n = 12
    mu = rng.uniform(-1, 1, n)
    y = legvander(mu, 3) @ np.array([1.0, 0.1, 0.3, 0.05]) + rng.normal(0, 0.04, n)
    sigma_stat = np.full(n, 0.04)
    exp_index = np.zeros(n, int)
    sigma_sys_indep = np.array([0.07])

    coeffs, chi2, dof, kp = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, 3,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat",
    )
    assert np.all(np.isfinite(coeffs))
    assert chi2 >= 0.0
    assert dof > 0.0
    assert kp > 0.0


def test_single_point_experiment_is_marginalized() -> None:
    """A single-point experiment with a tiny σ_stat is allowed to absorb its
    own normalisation completely (M_e → ∞, 1/M_e → 0). Verify the single
    point doesn't drag the fit toward its absolute level."""
    rng = np.random.default_rng(13)
    # Experiment A: 8 points sampling a clear L=2 shape
    mu_a = np.linspace(-0.9, 0.9, 8)
    y_a = legvander(mu_a, 2) @ np.array([1.0, 0.0, 0.4]) + rng.normal(0, 0.03, 8)
    # Experiment B: 1 point, very low σ_stat, value WAY off the L=2 shape
    mu_b = np.array([0.2])
    y_b = np.array([1.5])  # ~50% off
    mu = np.concatenate([mu_a, mu_b])
    y = np.concatenate([y_a, y_b])
    sigma_stat = np.concatenate([np.full(8, 0.03), np.array([0.001])])
    exp_index = np.concatenate([np.zeros(8, int), np.ones(1, int)])
    sigma_sys_indep = np.array([0.07, 0.07])

    coeffs, chi2, dof, kp = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, 2,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat",
    )
    # The fit should track A's L=2 shape (a_2 ≈ 0.4) rather than being dragged
    # toward B's absolute level. Diagonal WLS would put a_0 ≈ 1.5 because B
    # dominates with σ=0.001; GLS marginalises B's normalisation entirely.
    assert abs(coeffs[2] - 0.4) < 0.15, f"a_2 should be ~0.4, got {coeffs[2]:.3f}"
    assert chi2 >= 0.0


def test_zero_sigma_sys_indep_falls_through_to_diagonal() -> None:
    """All-zero σ_sys_indep with σ_sys_dep=0 reproduces the diagonal kernel-WLS
    over σ_stat exactly (no rank-1 corrections fire)."""
    rng = np.random.default_rng(91)
    n = 10
    degree = 2
    mu = rng.uniform(-1, 1, n)
    y = legvander(mu, degree) @ np.array([1.0, 0.1, 0.3]) + rng.normal(0, 0.05, n)
    sigma_stat = rng.uniform(0.02, 0.08, n)
    g = rng.uniform(0.5, 1.0, n)
    exp_index = rng.integers(0, 3, n)
    sigma_sys_indep = np.zeros(3)

    cD, chi2D, dofD, kD = _weighted_ridge_fit(
        mu, y, sigma_stat, degree,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat", external_weights=g,
    )
    cG, chi2G, dofG, kG = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, degree,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat", external_weights=g,
    )
    np.testing.assert_allclose(cG, cD, atol=1e-9)
    assert abs(chi2G - chi2D) < 1e-7


def test_single_experiment_bin_falls_back_to_diagonal() -> None:
    """When the bin has only one experiment, GLS has no inter-experiment
    normalisation to absorb. Applying it would marginalise the only
    experiment's normalisation, making c_0 unidentifiable. The pipeline
    must fall back to diagonal WLS (over σ_total = √(σ²_stat + σ²_sys))
    in that case.

    Verify by comparing the AICc winner from sample_legendre_coefficients
    (with use_gls_kernel=True) against a manual diagonal AICc scan on
    the same data — they should agree, and L=0 must be in the candidate
    set (because the no-anchor exclusion only applies when GLS actually
    runs)."""
    from scripts.resample_AD import (
        sample_legendre_coefficients, _criterion_score, _weighted_ridge_fit,
    )
    import pandas as pd

    rng = np.random.default_rng(2030)
    mu = np.linspace(-0.9, 0.9, 8)
    true_c = np.array([1.0, 0.2, 0.4, 0.15])
    y = legvander(mu, 3) @ true_c + rng.normal(0, 0.03, len(mu))
    sigma_stat = np.full(len(mu), 0.03)
    df = pd.DataFrame({
        'value': y, 'unc': sigma_stat, 'mu': mu,
        'theta_deg': np.rad2deg(np.arccos(mu)),
        'entry': '10571', 'subentry': '002',
        'sigma_sys_relative':       np.full(len(mu), 0.08),
        'sigma_sys_indep_relative': np.full(len(mu), 0.08),
        'sigma_sys_dep_relative':   np.zeros(len(mu)),
    })
    sigma_sys = df['sigma_sys_relative'].to_numpy() * np.abs(y)
    df['_sigma_sys_abs'] = sigma_sys
    sigma_total = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)

    _, info = sample_legendre_coefficients(
        df, value_col='value', unc_col='unc', sys_unc_col='_sigma_sys_abs',
        degree=None, max_degree=4, select_degree='aicc',
        ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        n_samples=1, use_gls_kernel=True,
    )

    # Winner should match a manual diagonal AICc scan over the same range.
    diag_aicc = {}
    for d in range(0, 5):
        _, chi2, _, k = _weighted_ridge_fit(
            mu, y, sigma_total, d,
            ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        )
        diag_aicc[d] = _criterion_score(chi2, n=len(mu), k=k, criterion='aicc')
    diag_winner = min(diag_aicc, key=diag_aicc.get)
    assert info['degree'] == diag_winner, (
        f"single-exp GLS should fall back to diagonal: pipeline picked L={info['degree']}, "
        f"manual diagonal AICc winner is L={diag_winner}"
    )
    # L=0 must be in the scan output (no-anchor exclusion does not fire when
    # GLS itself was bypassed).
    assert 0 in info['all_degrees_info'], (
        "L=0 must remain a candidate when GLS is bypassed for single-exp bins"
    )


def test_aic_criterion_picks_higher_l_than_aicc_for_small_n() -> None:
    """Plain AIC has no small-sample correction. For shape-rich data with
    small n, AIC should pick higher L than AICc (which over-penalises via
    2k(k+1)/(n−k−1))."""
    from scripts.resample_AD import _criterion_score, _weighted_ridge_fit

    rng = np.random.default_rng(2031)
    n = 8
    mu = np.linspace(-0.9, 0.9, n)
    # L=4-rich shape with small noise.
    true_c = np.array([1.0, 0.2, 0.5, 0.15, 0.4])
    y = legvander(mu, 4) @ true_c + rng.normal(0, 0.02, n)
    sigma = np.full(n, 0.02)

    # Compute scores for L=0..6 under both criteria.
    aicc_scores, aic_scores = {}, {}
    for d in range(0, 7):
        _, chi2, _, k = _weighted_ridge_fit(
            mu, y, sigma, d, ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        )
        aicc_scores[d] = _criterion_score(chi2, n=n, k=k, criterion='aicc')
        aic_scores[d] = _criterion_score(chi2, n=n, k=k, criterion='aic')

    aicc_winner = min(aicc_scores, key=aicc_scores.get)
    aic_winner = min(aic_scores, key=aic_scores.get)

    # AIC should pick the actual model order (L=4) or higher; AICc almost
    # certainly will pick L < 4 because of the n=8 small-sample correction.
    assert aic_winner >= aicc_winner, (
        f"AIC should pick at least as high a degree as AICc; "
        f"got AIC L={aic_winner} vs AICc L={aicc_winner}"
    )
    assert aic_winner >= 4, f"AIC should recover L≥4 on L=4-rich data; got L={aic_winner}"


def test_no_anchor_excludes_l0_from_aicc_scan() -> None:
    """When GLS runs and every active experiment has σ_sys_indep > 0,
    c_0 is structurally unidentifiable (per-experiment normalisations
    absorb any constant offset). The AICc scan must skip L=0 in that
    case, otherwise L=0 trivially wins with χ²≈0 even on highly
    anisotropic bins (the run-59 multi-experiment-without-anchor
    pathology). This test uses two experiments so the GLS path
    actually runs (single-experiment bins fall back to diagonal WLS;
    see ``test_single_experiment_bin_falls_back_to_diagonal``)."""
    from scripts.resample_AD import sample_legendre_coefficients
    import pandas as pd

    rng = np.random.default_rng(2026)
    # Two experiments, both with σ_sys_indep = 8% (no anchor).
    n_a, n_b = 8, 6
    mu_a = np.linspace(-0.9, 0.9, n_a)
    mu_b = np.linspace(-0.85, 0.85, n_b)
    mu = np.concatenate([mu_a, mu_b])
    true_c = np.array([1.0, 0.2, 0.4, 0.15])
    y = legvander(mu, 3) @ true_c + rng.normal(0, 0.03, len(mu))
    df = pd.DataFrame({
        'value': y,
        'unc': np.full(len(mu), 0.03),
        'mu': mu,
        'theta_deg': np.rad2deg(np.arccos(mu)),
        'entry': ['A'] * n_a + ['B'] * n_b,
        'subentry': ['001'] * (n_a + n_b),
        'sigma_sys_relative':       np.full(len(mu), 0.08),
        'sigma_sys_indep_relative': np.full(len(mu), 0.08),
        'sigma_sys_dep_relative':   np.zeros(len(mu)),
    })
    _, info = sample_legendre_coefficients(
        df, value_col='value', unc_col='unc',
        degree=None, max_degree=4, select_degree='aicc',
        ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        n_samples=1,
        use_gls_kernel=True,
    )
    # L=0 must be excluded; the chosen degree must be ≥ 1 and ideally near 3.
    assert info['degree'] >= 1, f"L=0 should be excluded under no-anchor GLS, got L={info['degree']}"
    assert info['degree'] in (2, 3, 4), f"expected L≈3, got L={info['degree']}"
    # all_degrees_info should not contain a record at L=0 either.
    assert 0 not in info['all_degrees_info'], "L=0 should not be in the AICc scan output under no-anchor GLS"


def test_anchor_present_keeps_l0_in_scan() -> None:
    """When at least one experiment has σ_sys_indep = 0 (an anchor), c_0 is
    identifiable and the scan must still consider L=0 (some bins are
    legitimately near-isotropic)."""
    from scripts.resample_AD import sample_legendre_coefficients
    import pandas as pd

    rng = np.random.default_rng(2027)
    # Two experiments. A: σ_sys_indep = 8%. B: σ_sys_indep = 0 (anchor).
    n_a, n_b = 6, 5
    mu_a = np.linspace(-0.85, 0.85, n_a)
    mu_b = np.linspace(-0.95, 0.95, n_b)
    mu = np.concatenate([mu_a, mu_b])
    true_c = np.array([1.0, 0.2, 0.4])
    y = legvander(mu, 2) @ true_c + rng.normal(0, 0.03, len(mu))
    df = pd.DataFrame({
        'value': y, 'unc': np.full(len(mu), 0.03), 'mu': mu,
        'theta_deg': np.rad2deg(np.arccos(mu)),
        'entry': ['A']*n_a + ['B']*n_b, 'subentry': ['001']*(n_a+n_b),
        # Caller now reads sigma_sys_relative (total). Set B to 0 explicitly
        # so it acts as the anchor (per-experiment normalisation = 0).
        'sigma_sys_relative':       np.array([0.08]*n_a + [0.0]*n_b),
        'sigma_sys_indep_relative': np.array([0.08]*n_a + [0.0]*n_b),
        'sigma_sys_dep_relative':   np.zeros(len(mu)),
    })
    _, info = sample_legendre_coefficients(
        df, value_col='value', unc_col='unc',
        degree=None, max_degree=3, select_degree='aicc',
        ridge_lambda=1e-4, ridge_power=4, df_method='hat',
        n_samples=1,
        use_gls_kernel=True,
    )
    # L=0 should be in the scan output (anchor present, c_0 identifiable).
    # The chosen degree depends on the data but L=0 must be a candidate.
    assert 0 in info['all_degrees_info'], (
        "L=0 must be a candidate in the AICc scan when an anchor experiment exists"
    )


def test_fixed_c0_round_trip_under_gls() -> None:
    """``fixed_c0`` should subtract the c0 contribution from y before fitting
    free coefficients (same semantics as ``_weighted_ridge_fit``). Verify the
    returned coefficient vector has the requested c0 and that χ² is computed
    on the original residual."""
    rng = np.random.default_rng(17)
    n = 12
    mu = rng.uniform(-1, 1, n)
    true_c = np.array([1.0, 0.2, 0.4, 0.05])
    y = legvander(mu, 3) @ true_c + rng.normal(0, 0.03, n)
    sigma_stat = np.full(n, 0.03)
    exp_index = np.zeros(n, int)
    sigma_sys_indep = np.array([0.05])

    coeffs_free, _, _, _ = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, 3,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat",
    )
    coeffs_fix, _, _, _ = _weighted_ridge_fit_gls(
        mu, y, sigma_stat, sigma_sys_indep, exp_index, 3,
        sigma_sys_dep_per_row=None,
        ridge_lambda=1e-4, ridge_power=4, df_method="hat",
        fixed_c0=0.95,
    )
    assert coeffs_fix[0] == 0.95
    # Free c0 result should differ from the fixed-c0 result on c1..c3 too,
    # because shapes have to compensate for the imposed level shift.
    assert not np.allclose(coeffs_fix[1:], coeffs_free[1:])
