"""Gate 1b of docs/chi2-mf4/mf4_research_roadmap.md — is the IC score a valid likelihood?

The order-treatment work in §4 rests entirely on the Akaike weights
``w_L = exp(-Δ_L/2) / Σ exp(-Δ_K/2)``. Those weights are only meaningful if the
score the pipeline minimises really is ``-2 log L + 2k`` up to a constant that
is **the same for every candidate degree**. If the constant moves with the
degree, every Δ_L is contaminated and §4 is dead.

The production score is ``chi2 + 2k`` (`_criterion_score`, SELECT_DEGREE="aic").
For a Gaussian model, ``-2 log L = chi2 + log det(2πΣ)``. So the omitted term is
``log det(2πΣ)`` and the question is whether Σ is fixed across the scan.

Two things make that non-obvious in this pipeline:

1. The fit is **GLS with a rank-2 per-experiment systematic block**, so Σ is not
   diagonal and is assembled by Woodbury rather than built explicitly.
2. The fit is **kernel-weighted** — production passes ``external_weights=g``,
   the energy-kernel weights. A weighted misfit is not automatically a
   log-likelihood for any covariance at all.

These tests establish that it is, identify what Σ actually is, and pin the
things that would break the argument if someone changed them.

Production settings this is written against (`exfor_to_endf_sampling_v2.py`):
    SELECT_DEGREE = "aic"   RIDGE_LAMBDA = 1e-4
    RIDGE_POWER = 4         DF_METHOD = "hat"     USE_GLS_KERNEL = True
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import legvander

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.resample_AD import (  # noqa: E402
    _criterion_score,
    _weighted_ridge_fit,
    _weighted_ridge_fit_gls,
)

RIDGE_LAMBDA = 1e-4
RIDGE_POWER = 4


def _case(n: int = 9, seed: int = 0):
    """A small bin with three experiments, unequal kernel weights, both sys modes."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(-0.9, 0.9, n)
    exp_index = np.repeat(np.arange(3), n // 3)[:n]
    y = 1.0 + 0.4 * mu + 0.2 * mu ** 2 + 0.02 * rng.standard_normal(n)
    sigma = 0.05 * np.ones(n)
    s_indep = np.array([0.10, 0.05, 0.08])
    d_dep = 0.03 * np.ones(n)
    g = np.linspace(1.0, 0.2, n)          # kernel weights, deliberately != 1
    return mu, y, sigma, s_indep, d_dep, exp_index, g


def _dense_sigma(y, sigma, s_indep, d_dep, exp_index, g):
    """Σ̃ = G^(-1/2) (D + Σ_e [u_e u_eᵀ + v_e v_eᵀ]) G^(-1/2)."""
    n = y.size
    Sig = np.diag(sigma ** 2)
    for e in range(int(exp_index.max()) + 1):
        idx = np.where(exp_index == e)[0]
        if idx.size == 0:
            continue
        u = np.zeros(n)
        u[idx] = s_indep[e] * y[idx]
        v = np.zeros(n)
        v[idx] = d_dep[idx] * y[idx]
        Sig += np.outer(u, u) + np.outer(v, v)
    Gm = np.diag(1.0 / np.sqrt(g))
    return Gm @ Sig @ Gm


# --------------------------------------------------------------------------
# 1. The score's chi2 term IS a quadratic form in a genuine covariance
# --------------------------------------------------------------------------

def test_gls_chi2_is_a_quadratic_form_in_an_explicit_covariance() -> None:
    """chi2 == rᵀ Σ̃⁻¹ r, with Σ̃ built densely and independently.

    This is the load-bearing check. The Woodbury correction in
    ``_weighted_ridge_fit_gls`` applies √g to the residual projection but no g
    to the uᵀD⁻¹u term, which looks inconsistent until you work out that it
    corresponds to u_e = s_e·y/√g. The net effect is that the kernel weight
    scales the WHOLE covariance, statistical and systematic alike, not just the
    diagonal. If it scaled only the diagonal, chi2 would not be a log-likelihood
    for any Σ and the Akaike weights would be meaningless.
    """
    mu, y, sigma, s_indep, d_dep, exp_index, g = _case()
    deg = 2
    c = np.array([1.0, 0.35, 0.25])
    fixed = {l: float(c[l]) for l in range(deg + 1)}

    _, chi2_code, _, _ = _weighted_ridge_fit_gls(
        mu, y, sigma, s_indep, exp_index, deg,
        sigma_sys_dep_per_row=d_dep, external_weights=g, fixed_coeffs=fixed,
    )

    r = y - legvander(mu, deg) @ c
    Sig = _dense_sigma(y, sigma, s_indep, d_dep, exp_index, g)
    chi2_dense = float(r @ np.linalg.solve(Sig, r))

    assert chi2_code == pytest.approx(chi2_dense, rel=1e-10), (
        "GLS chi2 is not rᵀΣ⁻¹r for the covariance the code claims to model. "
        "If this fails the IC score is not a likelihood and §4 cannot proceed."
    )


def test_kernel_weight_scales_the_whole_covariance_not_just_the_diagonal() -> None:
    """Pins the interpretation above, so a 'fix' to the Woodbury cannot pass silently."""
    mu, y, sigma, s_indep, d_dep, exp_index, g = _case()
    deg = 2
    c = np.array([1.0, 0.35, 0.25])
    fixed = {l: float(c[l]) for l in range(deg + 1)}
    _, chi2_code, _, _ = _weighted_ridge_fit_gls(
        mu, y, sigma, s_indep, exp_index, deg,
        sigma_sys_dep_per_row=d_dep, external_weights=g, fixed_coeffs=fixed,
    )
    r = y - legvander(mu, deg) @ c

    # The wrong model: kernel weight on the diagonal only.
    n = y.size
    Sig_wrong = np.diag(sigma ** 2 / g)
    for e in range(3):
        idx = np.where(exp_index == e)[0]
        u = np.zeros(n)
        u[idx] = s_indep[e] * y[idx]
        v = np.zeros(n)
        v[idx] = d_dep[idx] * y[idx]
        Sig_wrong += np.outer(u, u) + np.outer(v, v)
    chi2_wrong = float(r @ np.linalg.solve(Sig_wrong, r))

    assert not np.isclose(chi2_code, chi2_wrong), (
        "diagonal-only kernel scaling now matches — the covariance model changed"
    )


# --------------------------------------------------------------------------
# 2. The omitted normalisation is constant across candidate degrees
# --------------------------------------------------------------------------

def test_the_omitted_log_det_does_not_depend_on_the_candidate_degree() -> None:
    """Σ̃ is assembled from (sigma, s_indep, d_dep, g) only — never from `degree`.

    Verified behaviourally: evaluate the SAME residual vector through fits
    declared at different degrees (padding the coefficient vector with zeros).
    Since the covariance is what turns r into chi2, an equal chi2 across
    declared degrees means the covariance did not move with the degree.
    """
    mu, y, sigma, s_indep, d_dep, exp_index, g = _case()
    c_true = np.array([1.0, 0.35, 0.25])

    scores = []
    for deg in range(2, 7):
        c = np.zeros(deg + 1)
        c[: c_true.size] = c_true          # same predicted curve, wider declaration
        fixed = {l: float(c[l]) for l in range(deg + 1)}
        _, chi2_d, _, _ = _weighted_ridge_fit_gls(
            mu, y, sigma, s_indep, exp_index, deg,
            sigma_sys_dep_per_row=d_dep, external_weights=g, fixed_coeffs=fixed,
        )
        scores.append(chi2_d)

    assert np.allclose(scores, scores[0], rtol=1e-12), (
        f"chi2 for an identical residual moved with the declared degree: {scores}. "
        "The omitted log-det would then differ between candidates and every Δ_L "
        "would be contaminated."
    )


def test_akaike_weights_are_invariant_to_the_omitted_constant() -> None:
    """Adding any constant to every score leaves the weights unchanged.

    This is why omitting log det(2πΣ) is harmless *given* the previous test:
    the weights depend on differences only.
    """
    scores = np.array([12.0, 13.5, 14.2, 18.0])

    def weights(s):
        d = s - s.min()
        raw = np.exp(-0.5 * d)
        return raw / raw.sum()

    for shift in (-1e3, 0.0, 7.25, 1e4):
        assert np.allclose(weights(scores), weights(scores + shift), rtol=1e-12)


# --------------------------------------------------------------------------
# 3. Ridge: the penalty must not leak into the likelihood term
# --------------------------------------------------------------------------

def test_chi2_excludes_the_ridge_penalty() -> None:
    """chi2 is the pure data misfit, not misfit + λ cᵀRc.

    If the penalty leaked in, the score would double-count complexity: once in
    the penalised misfit and again in 2k.
    """
    mu, y, sigma, *_ = _case()
    w = 1.0 / sigma ** 2

    for deg in range(1, 7):
        coeffs, chi2, _, _ = _weighted_ridge_fit(
            mu, y, sigma, deg,
            ridge_lambda=RIDGE_LAMBDA, ridge_power=RIDGE_POWER, df_method="hat",
        )
        misfit = float(np.sum(w * (y - legvander(mu, deg) @ coeffs) ** 2))
        pen = RIDGE_LAMBDA * float(
            np.sum([l ** RIDGE_POWER * coeffs[l] ** 2 for l in range(1, deg + 1)])
        )
        assert chi2 == pytest.approx(misfit, rel=1e-12), f"degree {deg}"
        if pen > 0:
            assert chi2 != pytest.approx(misfit + pen, rel=1e-12)


def test_effective_params_are_the_hat_trace_and_shrink_with_ridge() -> None:
    """k = tr(H) ≤ degree+1, decreasing in λ, and consistent across degrees."""
    mu, y, sigma, *_ = _case()

    for deg in range(1, 7):
        ks = []
        for lam in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            _, _, _, k = _weighted_ridge_fit(
                mu, y, sigma, deg,
                ridge_lambda=lam, ridge_power=RIDGE_POWER, df_method="hat",
            )
            ks.append(k)
            assert k <= deg + 1 + 1e-9, f"degree {deg}, lambda {lam}: k={k}"
        assert np.all(np.diff(ks) <= 1e-9), (
            f"effective parameters not monotone decreasing in lambda at "
            f"degree {deg}: {ks}"
        )


def test_ridge_penalty_is_scaled_per_order_so_higher_degrees_are_shrunk_more() -> None:
    """λ·l^4 means the k for a high degree is well below degree+1.

    Not a defect — it is the intended behaviour — but it means the AIC penalty
    2k is NOT 2(degree+1), and anyone reasoning about the scan as if it were
    will get the wrong answer.
    """
    mu, y, sigma, *_ = _case()
    gaps = []
    for deg in range(1, 7):
        _, _, _, k = _weighted_ridge_fit(
            mu, y, sigma, deg,
            ridge_lambda=RIDGE_LAMBDA, ridge_power=RIDGE_POWER, df_method="hat",
        )
        gaps.append((deg + 1) - k)
    assert gaps[-1] > gaps[0], (
        f"shrinkage does not grow with degree under l^{RIDGE_POWER} ridge: {gaps}"
    )


# --------------------------------------------------------------------------
# 4. The criterion choice is doing real work at production sample sizes
# --------------------------------------------------------------------------

# Representative per-bin point counts observed in run 82's nominal_fits.parquet.
RUN82_NPTS = (13, 19, 25, 29)


def test_aic_and_aicc_disagree_at_production_sample_sizes() -> None:
    """Documents that SELECT_DEGREE="aic" is a live choice, not a formality.

    At n≈13-29 with k up to 7, AICc's correction 2k(k+1)/(n-k-1) is large — tens
    of units — so it penalises high orders far harder than AIC. The pipeline
    chose AIC deliberately (see `_criterion_score`'s docstring: AICc combined
    with a Kish ESS was structurally banning L≥2). The consequence is that the
    weights §4 consumes are conditional on that choice, and a sensitivity study
    over the criterion belongs in the write-up.
    """
    chi2_by_k = {1: 40.0, 2: 22.0, 3: 14.0, 4: 11.0, 5: 9.5, 6: 9.0, 7: 8.8}
    disagreements = 0
    for n in RUN82_NPTS:
        aic = {k: _criterion_score(c, n=n, k=k, criterion="aic")
               for k, c in chi2_by_k.items()}
        aicc = {k: _criterion_score(c, n=n, k=k, criterion="aicc")
                for k, c in chi2_by_k.items()}
        if min(aic, key=aic.get) != min(aicc, key=aicc.get):
            disagreements += 1
    assert disagreements > 0, (
        "AIC and AICc pick the same order at every production sample size — "
        "if this ever becomes true the criterion choice stops mattering"
    )


def test_aicc_correction_is_never_silently_negative() -> None:
    """The n ≤ k+1 branch must penalise, not reward.

    A negative correction would make an infeasible model win outright.
    """
    for n in range(2, 12):
        for k in range(1, 9):
            aic = _criterion_score(10.0, n=n, k=k, criterion="aic")
            aicc = _criterion_score(10.0, n=n, k=k, criterion="aicc")
            assert aicc >= aic - 1e-12, f"n={n} k={k}: AICc below AIC"


def test_aic_penalty_is_exactly_two_k() -> None:
    """Guards the one line every weight in §4 depends on."""
    for k in (1.0, 2.5, 7.0):
        assert _criterion_score(0.0, n=20, k=k, criterion="aic") == pytest.approx(2.0 * k)


# --------------------------------------------------------------------------
# 5. The tau-IRLS / GLS solver mismatch
# --------------------------------------------------------------------------

def test_tau_differs_between_wls_and_gls_residuals_on_shape_disagreement() -> None:
    """Documents the mechanism behind roadmap §5 Gate 1b finding (2).

    Both degree scans branch on ``use_gls_kernel`` (`resample_AD.py:1947` and
    `:2131`), but the tau-IRLS refit between them calls ``_weighted_ridge_fit``
    unconditionally (`:2069`) — plain WLS with sigma_sys on the DIAGONAL. So in
    production (USE_GLS_KERNEL=True) tau is estimated from the residuals of a
    different model than the one every candidate is then scored under.

    It matters because the two models absorb experimental disagreement
    differently. GLS gives each experiment a rank-1 normalisation direction, so
    a discrepant experiment can shift; diagonal WLS cannot shift it and books
    the discrepancy as band misfit, inflating tau. The post-tau GLS rescan then
    receives an already-inflated sigma_eff AND still has the rank-1 blocks, so
    the same disagreement is accommodated twice.

    This test only shows the mechanism is real on a constructed bin. It does
    NOT establish how much of run 82's observed tau is artifact — tau is active
    in 98.5 % of bins at median ~2.0, but EXFOR genuinely disagrees beyond
    declared uncertainties, which is why tau exists. Separating the two needs a
    tau-IRLS re-run under GLS on real data; that is fork work, see the roadmap.
    """
    from scripts.resample_AD import compute_angular_band_discrepancy
    from numpy.polynomial.legendre import legval

    rng = np.random.default_rng(7)
    mu = np.concatenate([np.linspace(-0.95, 0.95, 11)] * 3)
    exp_index = np.repeat([0, 1, 2], 11)
    y = legval(mu, np.array([1.0, 0.45, 0.30, 0.12]))
    # Experiment 2 carries a genuine backward-angle SHAPE distortion.
    y = y.copy()
    y[(exp_index == 2) & (mu < -0.3)] *= 1.25
    y *= np.where(exp_index == 0, 1.02, np.where(exp_index == 1, 0.98, 1.0))
    y += 0.004 * rng.standard_normal(y.size)

    sigma = 0.008 * np.ones(y.size)
    s_indep = np.full(3, 0.05)          # declared normalisation uncertainty
    sigma_sys = 0.05 * y
    deg = 3

    c_wls, *_ = _weighted_ridge_fit(
        mu, y, np.sqrt(sigma ** 2 + sigma_sys ** 2), deg,
        ridge_lambda=RIDGE_LAMBDA, ridge_power=RIDGE_POWER,
    )
    _, tau_wls = compute_angular_band_discrepancy(
        mu=mu, y=y, sigma=sigma, y_fit=legval(mu, c_wls), sigma_sys=sigma_sys,
    )
    c_gls, *_ = _weighted_ridge_fit_gls(
        mu, y, sigma, s_indep, exp_index, deg,
        ridge_lambda=RIDGE_LAMBDA, ridge_power=RIDGE_POWER,
    )
    _, tau_gls = compute_angular_band_discrepancy(
        mu=mu, y=y, sigma=sigma, y_fit=legval(mu, c_gls), sigma_sys=sigma_sys,
    )

    assert float(tau_wls["tau_B"]) > float(tau_gls["tau_B"]) + 0.1, (
        f"tau_B from WLS residuals {tau_wls['tau_B']:.3f} vs GLS "
        f"{tau_gls['tau_B']:.3f} — the solver mismatch no longer changes tau, "
        "so the Gate 1b finding may have been addressed; check the roadmap."
    )
