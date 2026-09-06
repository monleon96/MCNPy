from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any, Sequence, List, Union
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legvander, legval
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

from kika.utils.energy_folding import tof_energy_resolution

# Import from kika.exfor (new module)
try:
    from kika.exfor.transforms import (
        cos_cm_from_cos_lab,
        jacobian_cm_to_lab,
        transform_lab_to_cm,
    )
    _EXFOR_AVAILABLE = True
except ImportError:
    # Fallback to legacy import for backward compatibility
    _exfor_utils_path = Path(__file__).parent.parent / "EXFOR"
    if str(_exfor_utils_path) not in sys.path:
        sys.path.insert(0, str(_exfor_utils_path))
    try:
        from angular_distribution_utils import (
            cos_cm_from_cos_lab,
            jacobian_cm_to_lab,
            transform_lab_to_cm,
        )
        _EXFOR_AVAILABLE = True
    except ImportError:
        _EXFOR_AVAILABLE = False


@dataclass
class FitResult:
    coeffs: np.ndarray          # c_0..c_L for y(mu)=sum c_l P_l(mu)
    chi2: float
    dof: float                  # can be non-integer if ridge df="hat"
    chi2_red: float
    degree: int
    scale_factor: float         # Birge factor applied to uncertainties
    eff_params: float           # effective number of parameters (trace(H) if df="hat" else degree+1)


# =============================================================================
# ANGULAR-BAND DISCREPANCY MODEL
# =============================================================================

def robust_residual_scale(residuals: np.ndarray) -> float:
    """
    Compute MAD-based robust scale estimate.

    Uses the formula: 1.4826 * median(|r - median(r)|)

    The factor 1.4826 makes this consistent with standard deviation
    for normally distributed data.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals (normalized or raw)

    Returns
    -------
    float
        Robust scale estimate (equivalent to std for Gaussian)
    """
    if len(residuals) < 2:
        return 1.0
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad


def compute_angular_band_discrepancy(
    mu: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    y_fit: np.ndarray,
    min_points_per_band: int = 3,
    max_band_scale: float = 3.0,
    experiment_ids: Optional[np.ndarray] = None,
    method: str = "mad",
    sigma_sys: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Estimate per-band multiplicative scale factor s_b and return effective
    uncertainties.

    This replaces global Birge scaling with angular-band specific
    uncertainty inflation. The bands are:
      - Forward:  μ > 0.5  (θ < 60°)
      - Mid:      |μ| ≤ 0.5 (60° ≤ θ ≤ 120°)
      - Backward: μ < -0.5 (θ > 120°)

    For each band b:
      1. Compute normalized residuals: r_i = (y_i - y_fit_i) / σ_i
      2. Compute robust scale: s_b according to ``method``:
         - 'mad' (default): MAD-based estimate, robust to outliers
         - 'rms': sqrt(mean(r²)), responds to all dispersion (research)
         - 'hybrid': max(MAD, RMS), MAD floor + RMS sensitivity (research)
      3. If s_b > 1: uncertainties are under-estimated; apply s_b as a
         multiplicative scale factor.
      4. Apply ceiling: s_b = min(s_b, max_band_scale).

    The effective uncertainty is: σ_eff,i = s_b · σ_i

    This multiplicative model preserves the relative weights between data
    points within a band: a well-measured point (small σ) retains more
    influence on the fit than a poorly measured one.

    Parameters
    ----------
    mu : np.ndarray
        Cosine of scattering angle
    y : np.ndarray
        Measured cross section values
    sigma : np.ndarray
        Experimental uncertainties
    y_fit : np.ndarray
        Fitted values from Legendre polynomial
    min_points_per_band : int
        Minimum points required to estimate s for a band.
        If fewer, use the mid-band s value.
    max_band_scale : float
        Maximum allowed scale factor per band (safety cap).
    experiment_ids : np.ndarray, optional
        Per-point experiment identifiers (e.g. EXFOR entry numbers).
        When provided and a band scale is capped, per-experiment
        diagnostics are computed and returned in ``tau_info['exp_diag']``.
    method : str
        Scale estimator: 'mad' (default, pipeline behavior), 'rms', or
        'hybrid' (= max(MAD, RMS)). 'rms' and 'hybrid' are research options.
    sigma_sys : np.ndarray, optional
        Per-point systematic uncertainty (absolute, same units as ``sigma``).
        When provided, normalised residuals are r_i = (y_i - y_fit_i) / σ_total
        with σ_total² = σ_stat² + σ_sys², so τ measures discrepancy beyond
        what stat *and* sys together predict. The returned ``sigma_eff`` is
        still τ·σ_stat — sys is *not* folded into σ_eff so callers can
        compose σ_total_eff = sqrt(σ_eff² + σ_sys²) themselves when needed.

    Returns
    -------
    sigma_eff : np.ndarray
        Effective uncertainties scaled by band factors
    tau_info : Dict
        Dictionary with per-band scale factors (s_F, s_M, s_B ≥ 1.0).
        Keys are 'tau_F', 'tau_M', 'tau_B' for backward compatibility.
        Always includes 'mad_F/M/B' and 'rms_F/M/B' for diagnostics.
        When ``experiment_ids`` is given and a band is capped,
        ``tau_info['exp_diag']`` contains per-experiment band diagnostics.
    """
    if method not in ("mad", "rms", "hybrid"):
        raise ValueError(f"Unknown method={method!r}; expected 'mad', 'rms', or 'hybrid'")

    n = len(mu)
    sigma_eff = sigma.copy()

    # Define band masks
    forward_mask = mu > 0.5      # θ < 60°
    backward_mask = mu < -0.5   # θ > 120°
    mid_mask = ~forward_mask & ~backward_mask  # 60° ≤ θ ≤ 120°

    bands = {
        'F': forward_mask,
        'M': mid_mask,
        'B': backward_mask,
    }

    # Normalize residuals by σ_total = sqrt(σ_stat² + σ_sys²) when σ_sys is
    # provided. This stops τ from absorbing variance that's already accounted
    # for by the per-experiment systematic — without it, two experiments that
    # disagree at a few × σ_total but ~10 × σ_stat would force τ to the cap.
    sys_aware = sigma_sys is not None
    if sys_aware:
        sigma_for_resid = np.sqrt(sigma ** 2 + np.asarray(sigma_sys, dtype=float) ** 2)
    else:
        sigma_for_resid = sigma
    r_all = (y - y_fit) / sigma_for_resid

    # Values are multiplicative scale factors (1.0 = no inflation)
    tau_values = {'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0}
    # Raw (uncapped) scale estimates for diagnostics
    tau_values['raw_F'] = 1.0
    tau_values['raw_M'] = 1.0
    tau_values['raw_B'] = 1.0
    # Per-band MAD and RMS, always populated when the band has enough points
    tau_values['mad_F'] = 1.0
    tau_values['mad_M'] = 1.0
    tau_values['mad_B'] = 1.0
    tau_values['rms_F'] = 1.0
    tau_values['rms_M'] = 1.0
    tau_values['rms_B'] = 1.0

    # Per-experiment diagnostics for capped bands
    exp_diag = {}

    # Track which bands had a band-derived τ (vs default / inherited). Used by
    # the inheritance pass below and by apply_tau_prior_floor to filter the
    # donor pool to genuinely band-derived values.
    band_derived = {'F': False, 'M': False, 'B': False}
    band_npts = {bn: int(np.sum(m)) for bn, m in bands.items()}

    # First pass: estimate scale factor for bands with enough points
    for band_name, mask in bands.items():
        n_band = band_npts[band_name]

        if n_band < min_points_per_band:
            continue

        # Normalized residuals in this band
        r_band = r_all[mask]

        # Robust (MAD) and non-robust (RMS) estimators, both reported.
        mad_band = float(robust_residual_scale(r_band))
        rms_band = float(np.sqrt(np.mean(r_band ** 2)))
        tau_values[f'mad_{band_name}'] = max(1.0, mad_band)
        tau_values[f'rms_{band_name}'] = max(1.0, rms_band)

        if method == "mad":
            s_band = mad_band
        elif method == "rms":
            s_band = rms_band
        else:  # hybrid
            s_band = max(mad_band, rms_band)

        # Store raw value before capping (floored at 1.0 but not capped)
        raw_val = max(1.0, s_band)
        tau_values[f'raw_{band_name}'] = raw_val

        # Apply: s_b = max(1, s_method), capped at max_band_scale
        s_b = min(raw_val, max_band_scale)

        tau_values[f'tau_{band_name}'] = s_b
        band_derived[band_name] = True

        # Per-experiment diagnostics when band is capped
        if experiment_ids is not None and raw_val > max_band_scale:
            ids_band = experiment_ids[mask]
            r_band_vals = r_band
            band_entries = {}
            for exp_id in np.unique(ids_band):
                exp_mask = ids_band == exp_id
                r_exp = r_band_vals[exp_mask]
                n_exp = len(r_exp)
                band_entries[exp_id] = {
                    'n': n_exp,
                    'mad_scale': robust_residual_scale(r_exp) if n_exp >= 2 else float('nan'),
                    'mean_abs_r': float(np.mean(np.abs(r_exp))),
                    'max_abs_r': float(np.max(np.abs(r_exp))),
                }
            exp_diag[band_name] = band_entries

    if exp_diag:
        tau_values['exp_diag'] = exp_diag

    tau_values['sys_aware'] = bool(sys_aware)

    # Surface the per-band derivation flags so downstream consumers (notably
    # apply_tau_prior_floor) can distinguish band-derived τ from inherited.
    for bn in ('F', 'M', 'B'):
        tau_values[f'band_derived_{bn}'] = bool(band_derived[bn])
        tau_values[f'n_pts_{bn}'] = int(band_npts[bn])

    # Second pass: for bands with too few points, inherit τ from the
    # *most-supported* band-derived band (donor with the largest n_pts among
    # those that cleared min_points_per_band). Falling back to mid is wrong
    # for this dataset because mid is frequently the under-supported band
    # itself; picking the largest band-derived donor avoids that bias and
    # falls back to 1.0 (no inflation) only when no band is well-supported.
    derived_bands = [bn for bn, ok in band_derived.items() if ok]
    if derived_bands:
        donor = max(derived_bands, key=lambda bn: band_npts[bn])
        s_donor = tau_values[f'tau_{donor}']
        tau_values['donor_band'] = donor
        for band_name in bands:
            if not band_derived[band_name] and band_npts[band_name] > 0:
                tau_values[f'tau_{band_name}'] = s_donor
    else:
        tau_values['donor_band'] = None

    # Apply multiplicative scaling
    for band_name, mask in bands.items():
        s_b = tau_values[f'tau_{band_name}']
        if s_b > 1.0:
            sigma_eff[mask] = sigma[mask] * s_b

    return sigma_eff, tau_values


def sigma_eff_from_tau(
    mu: np.ndarray,
    sigma: np.ndarray,
    tau_info: Dict[str, float],
) -> np.ndarray:
    """Reconstruct sigma_eff from known band scale factors (no re-estimation).

    Uses the same band definitions as compute_angular_band_discrepancy:
      Forward:  mu > 0.5   -> tau_F (scale factor)
      Mid:      |mu| <= 0.5 -> tau_M (scale factor)
      Backward: mu < -0.5  -> tau_B (scale factor)

    sigma_eff_i = s_b * sigma_i  (multiplicative scaling)
    """
    sigma_eff = sigma.copy()
    bands = {
        'tau_F': mu > 0.5,
        'tau_M': (mu >= -0.5) & (mu <= 0.5),
        'tau_B': mu < -0.5,
    }
    for key, mask in bands.items():
        s_b = tau_info.get(key, 1.0)
        if s_b > 1.0 and np.any(mask):
            sigma_eff[mask] = sigma[mask] * s_b
    return sigma_eff


def smooth_tau_in_energy(
    tau_by_energy: Dict[float, Dict[str, float]],
    window: int = 3,
) -> Dict[float, Dict[str, float]]:
    """
    Apply moving median smoothing to τ_b(E) across energy grid.

    This reduces statistical fluctuations in τ estimates across
    neighboring energy points.

    Parameters
    ----------
    tau_by_energy : Dict[float, Dict[str, float]]
        {energy: {'tau_F': ..., 'tau_M': ..., 'tau_B': ...}}
    window : int
        Window size for moving median (must be odd)

    Returns
    -------
    Dict[float, Dict[str, float]]
        Smoothed τ values
    """
    if window < 1:
        return tau_by_energy

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    energies = sorted(tau_by_energy.keys())
    n_energies = len(energies)

    if n_energies < window:
        return tau_by_energy

    half_window = window // 2

    # Extract τ arrays
    tau_F = np.array([tau_by_energy[E]['tau_F'] for E in energies])
    tau_M = np.array([tau_by_energy[E]['tau_M'] for E in energies])
    tau_B = np.array([tau_by_energy[E]['tau_B'] for E in energies])

    # Apply moving median
    tau_F_smooth = np.zeros_like(tau_F)
    tau_M_smooth = np.zeros_like(tau_M)
    tau_B_smooth = np.zeros_like(tau_B)

    for i in range(n_energies):
        i_start = max(0, i - half_window)
        i_end = min(n_energies, i + half_window + 1)

        tau_F_smooth[i] = np.median(tau_F[i_start:i_end])
        tau_M_smooth[i] = np.median(tau_M[i_start:i_end])
        tau_B_smooth[i] = np.median(tau_B[i_start:i_end])

    # Rebuild dictionary
    smoothed = {}
    for i, E in enumerate(energies):
        smoothed[E] = {
            'tau_F': tau_F_smooth[i],
            'tau_M': tau_M_smooth[i],
            'tau_B': tau_B_smooth[i],
        }

    return smoothed


def apply_tau_prior_floor(
    nominal_results: List,
    n_eff_threshold: float = 5.0,
    percentile: float = 50.0,
) -> Dict[str, float]:
    """
    Compute a per-band tau baseline from well-supported bins and enforce it
    as a floor on low-support bins (partial pooling).

    Bins with N_eff >= n_eff_threshold are considered well-estimated (used to
    compute the baseline). Bins with N_eff < n_eff_threshold get the floor applied.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results (modified in-place).
    n_eff_threshold : float
        Minimum N_eff for a bin to be considered well-estimated.
    percentile : float
        Percentile of well-estimated tau values to use as baseline (e.g. 50 = median).

    Returns
    -------
    Dict[str, float]
        Baseline tau values per band {'tau_F': ..., 'tau_M': ..., 'tau_B': ...}.
    """
    bands = ['tau_F', 'tau_M', 'tau_B']
    baselines: Dict[str, float] = {b: 0.0 for b in bands}

    # Step 1: Collect tau values from well-supported bins (high N_eff), but
    # only when the band itself was *band-derived* (i.e. had enough points
    # in that band to estimate τ from its own residuals). Inherited values
    # — propagated from another band by compute_angular_band_discrepancy —
    # would otherwise contaminate the donor pool toward the donor band.
    well_estimated: Dict[str, List[float]] = {b: [] for b in bands}
    for r in nominal_results:
        if not r.has_data or r.interpolated:
            continue
        r_neff = getattr(r.kernel_diagnostics, 'n_eff', 0.0) if hasattr(r, 'kernel_diagnostics') else 0.0
        if r_neff >= n_eff_threshold:
            for b in bands:
                bn = b.split('_', 1)[1]  # 'tau_F' -> 'F'
                if not r.tau_info.get(f'band_derived_{bn}', True):
                    # Skip bands inherited from another band of this bin.
                    continue
                val = r.tau_info.get(b, 0.0)
                if val > 1.0:  # exclude neutral (no-inflation) values
                    well_estimated[b].append(val)

    # Step 2: Compute baseline per band (need >= 3 well-estimated bins)
    for b in bands:
        vals = well_estimated[b]
        if len(vals) >= 3:
            baselines[b] = float(np.percentile(vals, percentile))

    # Step 3: Apply floor. A band is floored when EITHER (a) the bin is
    # under-supported overall (N_eff < threshold) OR (b) the band itself
    # was not band-derived in this bin. (b) catches well-supported bins
    # whose forward/backward bands inherited τ from another band — the
    # floor pulls them up to the per-band baseline rather than leaving
    # them at the inherited value.
    for r in nominal_results:
        if not r.has_data or r.interpolated:
            continue
        r_neff = getattr(r.kernel_diagnostics, 'n_eff', 0.0) if hasattr(r, 'kernel_diagnostics') else 0.0
        bin_under = r_neff < n_eff_threshold
        updated = dict(r.tau_info)
        any_change = False
        for b in bands:
            bn = b.split('_', 1)[1]
            band_derived_b = updated.get(f'band_derived_{bn}', True)
            if bin_under or not band_derived_b:
                new_val = max(updated.get(b, 1.0), baselines[b])
                if new_val != updated.get(b, 1.0):
                    updated[b] = new_val
                    any_change = True
        if any_change:
            r.tau_info = updated

    return baselines


# =============================================================================
# KERNEL DIAGNOSTICS
# =============================================================================

def compute_n_eff(
    kernel_weights: np.ndarray,
    sigma_eff: np.ndarray,
) -> float:
    """
    Compute effective sample size N_eff.

    Formula: N_eff = (sum(w_i))^2 / sum(w_i^2)
    where w_i = g_ij / sigma_eff_i^2

    This measures how many "effective" independent data points contribute
    to the fit. N_eff = n for equal weights, N_eff = 1 for single-point
    dominance.

    Parameters
    ----------
    kernel_weights : np.ndarray
        Gaussian kernel weights g_ij
    sigma_eff : np.ndarray
        Effective uncertainties (stat + band discrepancy)

    Returns
    -------
    float
        Effective sample size. Higher is better.
    """
    if len(kernel_weights) == 0:
        return 0.0

    # Mask out points with non-positive or non-finite sigma_eff
    valid = np.isfinite(sigma_eff) & (sigma_eff > 0)
    if not np.any(valid):
        return 0.0
    # Combined weights: kernel weight / variance (valid points only)
    w = kernel_weights[valid] / (sigma_eff[valid] ** 2)

    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)

    if sum_w2 < 1e-30:
        return 0.0

    return (sum_w ** 2) / sum_w2


def compute_angular_support_diagnostics(
    mu: np.ndarray,
    kernel_weights: np.ndarray,
    max_degree: int,
) -> Dict[str, float]:
    """Compute diagnostics for angular coverage quality.

    Returns dict with:
    - n_unique_mu: distinct mu values (rounded to 4 decimals)
    - mu_coverage: (max_mu - min_mu) / 2
    - max_mu_gap: largest gap between consecutive sorted unique mu
    - legendre_cond: condition number of weighted Legendre design at max_degree
    - recommended_mc_order: suggested max order for MC sampling
    """
    from numpy.polynomial.legendre import legvander

    unique_mu = np.unique(np.round(mu, decimals=4))
    n_unique = len(unique_mu)
    sorted_mu = np.sort(unique_mu)

    mu_coverage = (sorted_mu[-1] - sorted_mu[0]) / 2.0 if n_unique > 1 else 0.0
    max_gap = float(np.max(np.diff(sorted_mu))) if n_unique > 1 else 2.0

    # Weighted Legendre design matrix condition number
    W_sqrt = np.sqrt(np.maximum(kernel_weights, 0.0))
    w_max = W_sqrt.max()
    if w_max > 0:
        W_sqrt = W_sqrt / w_max

    # Compute condition number at max_degree
    V = legvander(mu, max_degree)
    VW = V * W_sqrt[:, None]
    try:
        cond = float(np.linalg.cond(VW))
    except Exception:
        cond = 1e12

    # Heuristic for recommended MC order:
    # 1. Need >= 2*(L+1) unique mu values
    # 2. Condition number < 1e6 at that order
    max_by_points = max(0, n_unique // 2 - 1)

    max_by_cond = max_degree
    for L_test in range(max_degree, 0, -1):
        V_test = legvander(mu, L_test)
        VW_test = V_test * W_sqrt[:, None]
        try:
            c = float(np.linalg.cond(VW_test))
            if c < 1e6:
                max_by_cond = L_test
                break
        except Exception:
            continue
    else:
        max_by_cond = 1

    recommended = min(max_by_points, max_by_cond, max_degree)

    return {
        'n_unique_mu': n_unique,
        'mu_coverage': mu_coverage,
        'max_mu_gap': max_gap,
        'legendre_cond': cond,
        'recommended_mc_order': max(1, recommended),
    }


def compute_between_experiment_coeffs(
    exfor_df: pd.DataFrame,
    degree: int,
    fixed_c0: float,
    min_points: int = 3,
    ridge_lambda: float = 1e-6,
    min_mu_coverage: float = 1.0,
    max_cond: float = 1e4,
    freeze_c0: bool = True,
    min_experiments: int = 2,
) -> Optional[Dict]:
    """Compute per-experiment Legendre coefficients and their weighted scatter.

    For each qualifying experiment, fit Legendre polynomials at the same pooled
    nominal order with c0 frozen to the pooled value. Then compute the weighted
    scatter of the resulting ENDF a_l coefficients across experiments.

    This provides a "between-experiment" uncertainty floor analogous to the PDG
    external error: if independent measurements disagree beyond their internal
    errors, the uncertainty must reflect that disagreement.

    Experiments must have enough angular points to support the full pooled order
    (n_pts >= degree + 2, since c0 is frozen). An angular quality gate further
    ensures that only experiments with sufficient angular coverage and numerical
    stability contribute to the scatter estimate.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with columns 'mu', 'value', 'unc', 'entry', 'kernel_weight'.
    degree : int
        Maximum Legendre order from the pooled nominal fit.
    fixed_c0 : float
        c0 coefficient from the pooled fit (frozen for per-experiment fits).
    min_points : int
        Minimum angular points per experiment to qualify (default 3).
    ridge_lambda : float
        Small ridge parameter to stabilize near-singular per-experiment fits.
    min_mu_coverage : float
        Minimum angular span ``max(mu) - min(mu)`` required (default 1.0,
        i.e. 50 % of the full [-1, 1] range).
    max_cond : float
        Maximum condition number of the per-experiment Legendre design matrix
        (default 1e4).  Experiments whose design matrix exceeds this are
        excluded because the fitted coefficients would be numerically
        unreliable.
    freeze_c0 : bool
        ``True`` (default, historical) freezes every per-experiment fit to the
        pooled ``fixed_c0``.  ``False`` lets each experiment carry its own c0.

        ⚠ WHICH ONE YOU WANT DEPENDS ON WHAT THE SCATTER IS FOR, and the
        historical default is wrong for MF34.  ``a_l = c_l / c_0``, so freezing
        c0 to the pooled value pushes each experiment's NORMALISATION offset
        straight into its ``a_l``: a measurement that is uniformly 5 % high
        comes out with every a_l 5 % high and is counted as a SHAPE
        disagreement.  MF34 declares shape; normalisation belongs to MF33.
        With ``freeze_c0=False`` a pure normalisation offset cancels exactly and
        the scatter is shape only.  Kept defaulting to True so existing callers
        are byte-identical.
    min_experiments : int
        Minimum qualifying experiments before a result is returned (default 2,
        historical).  Pass 1 to get the census — ``per_experiment`` and
        ``n_experiments`` — for bins where a single experiment qualifies.  Those
        are exactly the bins where the joint fit can see NO disagreement at all,
        so a floor has to reach them, and it cannot if the estimator refuses to
        report them.  ``scatter`` is ``None`` when fewer than 2 qualify.

    Returns
    -------
    dict or None
        If >= 2 qualifying experiments:
        - 'scatter': np.ndarray of weighted scatter for l=1..degree (ENDF a_l units)
        - 'L_common': int, equal to degree (kept for API compatibility)
        - 'n_experiments': int, number of qualifying experiments
        - 'per_experiment': dict mapping entry -> (a_l array, n_points, weight_sum)
        - 'skipped_experiments': list of (entry, reason) for experiments that
          failed the quality gate
        Returns None if fewer than 2 experiments qualify.
    """
    from numpy.polynomial.legendre import legvander

    if 'entry' not in exfor_df.columns:
        return None

    # Group by experiment entry
    grouped = exfor_df.groupby('entry')
    per_experiment = {}
    skipped = []

    for entry_val, grp in grouped:
        n_pts = len(grp)
        if n_pts < min_points:
            skipped.append((entry_val, f"too few points ({n_pts} < {min_points})"))
            continue

        mu = grp['mu'].to_numpy()
        y = grp['value'].to_numpy()
        sigma = grp['unc'].to_numpy()

        # Require enough points to support the full pooled order
        # (c0 frozen -> degree free params, so need n_pts >= degree + 2)
        if n_pts < degree + 2:
            skipped.append((entry_val, f"too few points for pooled order (n_pts={n_pts} < {degree + 2})"))
            continue

        # --- Angular quality gate ---
        unique_mu = np.unique(np.round(mu, decimals=4))
        n_unique = len(unique_mu)

        # (a) Angular coverage: must span enough of the cosine range
        #     If points cluster in a narrow angular window the Legendre
        #     polynomials are nearly collinear there and the fitted
        #     coefficients reflect local noise, not the true shape.
        mu_span = float(unique_mu.max() - unique_mu.min()) if n_unique > 1 else 0.0
        if mu_span < min_mu_coverage:
            skipped.append((entry_val, f"poor angular coverage (span={mu_span:.2f} < {min_mu_coverage})"))
            continue

        # (b) Condition number of per-experiment Legendre design matrix
        #     Catches ill-conditioning from any source (clustered angles,
        #     gaps, too few points for the order, etc.).
        V = legvander(mu, degree)
        try:
            cond = float(np.linalg.cond(V))
        except Exception:
            cond = np.inf
        if cond > max_cond:
            skipped.append((entry_val, f"ill-conditioned design (cond={cond:.0f} > {max_cond:.0f})"))
            continue

        # Use kernel_weight sum as scatter weight for this experiment
        if 'kernel_weight' in grp.columns:
            weight_sum = float(grp['kernel_weight'].sum())
        else:
            weight_sum = float(n_pts)

        try:
            coeffs, _chi2, _dof, _k = _weighted_ridge_fit(
                mu, y, sigma, degree,
                fixed_c0=(fixed_c0 if freeze_c0 else None),
                ridge_lambda=ridge_lambda,
            )
            a_l = endf_normalize_legendre_coeffs(coeffs)  # returns a_1..a_L
            per_experiment[entry_val] = (a_l, n_pts, weight_sum)
        except (ValueError, np.linalg.LinAlgError):
            skipped.append((entry_val, "fit failed"))
            continue

    if len(per_experiment) < max(1, min_experiments):
        return None

    # All experiments were fitted at the same pooled order
    L_common = degree
    if L_common < 1:
        return None

    if len(per_experiment) < 2:
        # Census only: one qualifying experiment means there is nothing to take
        # a scatter of.  The caller still needs to know THAT, and who it was.
        return {
            'scatter': None,
            'L_common': L_common,
            'n_experiments': len(per_experiment),
            'per_experiment': per_experiment,
            'skipped_experiments': skipped,
        }

    # Compute weighted scatter for each order l=1..L_common
    scatter = np.zeros(L_common)
    entries = list(per_experiment.keys())
    N = len(entries)
    weights = np.array([per_experiment[e][2] for e in entries])
    W_total = weights.sum()

    for l_idx in range(L_common):
        a_vals = np.array([per_experiment[e][0][l_idx] for e in entries])

        if N == 2:
            # For N=2: scatter = sqrt(w1*w2/(w1+w2)) * |a1 - a2|
            w1, w2 = weights[0], weights[1]
            scatter[l_idx] = np.sqrt(w1 * w2 / (w1 + w2)) * abs(a_vals[0] - a_vals[1])
        else:
            # Weighted scatter: sqrt(sum w_j*(a_j - a_bar)^2 / ((N-1)/N * sum_w))
            a_bar = np.average(a_vals, weights=weights)
            var = np.sum(weights * (a_vals - a_bar) ** 2) / ((N - 1) / N * W_total)
            scatter[l_idx] = np.sqrt(var)

    return {
        'scatter': scatter,
        'L_common': L_common,
        'n_experiments': N,
        'per_experiment': per_experiment,
        'skipped_experiments': skipped,
    }


def compute_weight_span_95(
    kernel_weights: np.ndarray,
    exfor_energies: np.ndarray,
    target_energy: float,
) -> float:
    """
    Compute 95% weight span - smallest energy interval containing 95% of total weight.

    This diagnostic shows the effective energy range that contributes to the fit.
    A wide span (relative to σE) indicates the kernel may be averaging over
    energy-dependent structure (e.g., resonances).

    Parameters
    ----------
    kernel_weights : np.ndarray
        Gaussian kernel weights g_ij
    exfor_energies : np.ndarray
        EXFOR experiment energies (MeV) for each point
    target_energy : float
        Target ENDF grid energy (MeV)

    Returns
    -------
    float
        95% weight span in MeV (full width, not half-width)
    """
    if len(kernel_weights) < 2:
        return 0.0

    # Sort points by distance from target
    delta_E = np.abs(exfor_energies - target_energy)
    sort_idx = np.argsort(delta_E)

    sorted_weights = kernel_weights[sort_idx]
    sorted_delta_E = delta_E[sort_idx]

    # Normalize weights
    total_weight = np.sum(sorted_weights)
    if total_weight < 1e-30:
        return 0.0

    # Accumulate weights until 95%
    cumsum = np.cumsum(sorted_weights) / total_weight
    idx_95 = np.searchsorted(cumsum, 0.95)
    idx_95 = min(idx_95, len(sorted_delta_E) - 1)

    # Span is 2x the distance to farthest included point (symmetric around target)
    return 2.0 * sorted_delta_E[idx_95]


# =============================================================================
# TOF ENERGY RESOLUTION AND GAUSSIAN KERNEL
# =============================================================================

def compute_energy_resolution_tof(
    E_mev: float,
    delta_t_ns: float = 5.0,
    flight_path_m: float = 27.037,
    delta_t_is_fwhm: bool = True,
) -> float:
    """
    Compute energy resolution σE from TOF parameters.

    Thin adapter over :func:`kika.utils.energy_folding.tof_energy_resolution`,
    which is the single definition of the formula.

    Parameters
    ----------
    E_mev : float
        Neutron energy in MeV
    delta_t_ns : float
        Timing spread in nanoseconds; see ``delta_t_is_fwhm``.
    flight_path_m : float
        Flight path length in meters (default: 27.037 m)
    delta_t_is_fwhm : bool, default True
        Whether ``delta_t_ns`` is a FWHM (usual experimental convention) or
        already a sigma.  The two differ by a factor 2.355 in the result.

    Returns
    -------
    float
        Energy resolution σE in MeV
    """
    if E_mev <= 0:
        return 0.0

    return tof_energy_resolution(
        E_mev,
        flight_path_m=flight_path_m,
        delta_t_ns=delta_t_ns,
        delta_t_is_fwhm=delta_t_is_fwhm,
    )


def compute_energy_kernel_weights(
    E_target: float,
    E_exfor: np.ndarray,
    sigma_E: float,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Compute Gaussian kernel weights for EXFOR data points.

    The kernel weight is:
      g_ij = exp(-0.5 * ((E_i - E_j) / σE)²)

    Only points within ±n_sigma * σE are included (others get weight 0).

    Parameters
    ----------
    E_target : float
        Target ENDF grid energy (MeV)
    E_exfor : np.ndarray
        Array of EXFOR experiment energies (MeV)
    sigma_E : float
        Energy resolution σE (MeV)
    n_sigma : float
        Cutoff in units of σE (default: 3.0)

    Returns
    -------
    np.ndarray
        Kernel weights (same shape as E_exfor)
    """
    if sigma_E <= 0:
        # Fallback: equal weights for all points
        return np.ones_like(E_exfor)

    # Compute distances in units of σE
    z = (E_exfor - E_target) / sigma_E

    # Apply cutoff
    mask = np.abs(z) <= n_sigma

    # Compute Gaussian weights
    weights = np.zeros_like(E_exfor)
    weights[mask] = np.exp(-0.5 * z[mask]**2)

    return weights


# =============================================================================
# LEGENDRE FITTING FUNCTIONS
# =============================================================================

def _infer_mu(
    df: pd.DataFrame,
    mu_col: Optional[str] = None,
    theta_deg_col: Optional[str] = "theta_deg",
    cos_col_candidates: Sequence[str] = ("mu", "cos_theta", "cos", "cth", "costheta"),
) -> np.ndarray:
    """
    Return mu=cos(theta) from a dataframe. Priority:
    1) mu_col if provided
    2) any column in cos_col_candidates
    3) theta_deg_col (degrees) -> mu=cos(theta)
    """
    if mu_col is not None:
        if mu_col not in df.columns:
            raise ValueError(f"mu_col='{mu_col}' not found in dataframe columns.")
        mu = df[mu_col].to_numpy(dtype=float)
        return mu

    for c in cos_col_candidates:
        if c in df.columns:
            mu = df[c].to_numpy(dtype=float)
            return mu

    if theta_deg_col is not None and theta_deg_col in df.columns:
        theta_deg = df[theta_deg_col].to_numpy(dtype=float)
        return np.cos(np.deg2rad(theta_deg))

    raise ValueError(
        "Could not infer mu. Provide mu_col or include one of "
        f"{list(cos_col_candidates)} or '{theta_deg_col}' in the dataframe."
    )


def _weighted_ridge_fit(
    mu: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    degree: int,
    ridge_lambda: float = 0.0,
    ridge_power: int = 4,
    df_method: Literal["naive", "hat"] = "hat",
    external_weights: Optional[np.ndarray] = None,
    fixed_c0: Optional[float] = None,
    fixed_coeffs: Optional[Dict[int, float]] = None,
    compute_dof: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Fit y(mu) = sum_{l=0..L} c_l P_l(mu) using weighted least squares,
    optionally with ridge regularization on l>=1 terms.

    Parameters
    ----------
    mu : np.ndarray
        Cosine of scattering angle
    y : np.ndarray
        Cross section values
    sigma : np.ndarray
        Uncertainties
    degree : int
        Legendre polynomial degree
    ridge_lambda : float
        Ridge regularization parameter
    ridge_power : int
        Power for ridge penalty (default: 4, i.e., l^4)
    df_method : str
        Method for degrees of freedom ("naive" or "hat")
    external_weights : Optional[np.ndarray]
        Additional weights (e.g., Gaussian kernel weights g_ij).
        If provided, combined weight is: w_ij = g_ij / σ²_i
    fixed_c0 : Optional[float]
        If provided, fix c0 to this value. Shorthand for fixed_coeffs={0: c0}.
    fixed_coeffs : Optional[Dict[int, float]]
        Dict mapping Legendre order l to its fixed value.
        Fixed coefficients are subtracted from y and not fitted.
        Can be combined with fixed_c0 (fixed_c0 takes precedence for l=0).

    Returns
    -------
    Tuple[np.ndarray, float, float, float]
        (coeffs, chi2, dof, eff_params)
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(sigma)):
        raise ValueError("mu, y, sigma must be finite.")
    if np.any(sigma <= 0):
        raise ValueError("All sigma must be > 0.")

    n = mu.size

    # Weighting: combine external weights with inverse variance
    if external_weights is not None:
        w = external_weights / (sigma ** 2)  # w_ij = g_ij / σ²_i
    else:
        w = 1.0 / (sigma ** 2)
    sw = np.sqrt(w)

    # Merge fixed_c0 and fixed_coeffs into a single dict
    fixed_values: Dict[int, float] = {}
    if fixed_coeffs is not None:
        fixed_values.update(fixed_coeffs)
    if fixed_c0 is not None:
        fixed_values[0] = fixed_c0

    # Full design matrix: N x (L+1)
    A_full = legvander(mu, degree)
    all_indices = list(range(degree + 1))

    if fixed_values:
        free_indices = [l for l in all_indices if l not in fixed_values]
        fixed_indices = sorted(fixed_values.keys())

        if not free_indices:
            # All coefficients are fixed — nothing to fit
            coeffs = np.array([fixed_values.get(l, 0.0) for l in all_indices])
            yhat = A_full @ coeffs
            chi2 = float(np.sum(w * (y - yhat) ** 2))
            return coeffs, chi2, float(max(1, n)), 0.0

        # Subtract fixed contributions from y
        fixed_vals = np.array([fixed_values[l] for l in fixed_indices])
        y_adj = y - A_full[:, fixed_indices] @ fixed_vals

        # Reduced design matrix with only free columns
        A = A_full[:, free_indices]
        Aw = A * sw[:, None]
        yw = y_adj * sw

        n_free = len(free_indices)

        # Ridge penalty on free indices, using original l for penalty scaling
        if ridge_lambda > 0.0:
            pen = np.array([float(l ** ridge_power) if l > 0 else 0.0
                            for l in free_indices])
            R = np.diag(pen)
        else:
            R = np.zeros((n_free, n_free), dtype=float)

        M = Aw.T @ Aw + ridge_lambda * R
        rhs = Aw.T @ yw

        # Solve for free coefficients
        try:
            coeffs_partial = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            coeffs_partial, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
            import warnings
            warnings.warn(
                f"Matrix M is rank-deficient (rank={rank}/{M.shape[0]}) for degree={degree} "
                f"with {len(fixed_values)} fixed coefficients. Using least-squares solution.",
                RuntimeWarning,
                stacklevel=2
            )

        # Reconstruct full coefficient vector
        coeffs = np.zeros(degree + 1, dtype=float)
        for l in fixed_indices:
            coeffs[l] = fixed_values[l]
        for i, l in enumerate(free_indices):
            coeffs[l] = coeffs_partial[i]

        # Compute chi2 on original scale
        yhat = A_full @ coeffs
        chi2 = float(np.sum(w * (y - yhat) ** 2))

        # Degrees of freedom (only free parameters count)
        if df_method == "naive" or ridge_lambda <= 0.0 or not compute_dof:
            eff_params = float(n_free)
            dof = float(max(1, n - n_free))
        else:
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                Minv = np.linalg.pinv(M)
            H = Aw @ Minv @ Aw.T
            eff_params = float(np.trace(H))
            dof = float(max(1e-12, n - eff_params))

        return coeffs, chi2, dof, eff_params

    # Standard mode: fit all c0..cL (no fixed coefficients)
    A = A_full
    Aw = A * sw[:, None]
    yw = y * sw

    # Ridge penalty matrix R (L+1 x L+1), no penalty on l=0
    if ridge_lambda > 0.0:
        pen = np.zeros(degree + 1, dtype=float)
        for l in range(1, degree + 1):
            pen[l] = float(l ** ridge_power)
        R = np.diag(pen)
    else:
        R = np.zeros((degree + 1, degree + 1), dtype=float)

    M = Aw.T @ Aw + ridge_lambda * R
    rhs = Aw.T @ yw

    # Solve for coefficients
    # Try direct solve first; fall back to lstsq for singular/ill-conditioned matrices
    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Matrix is singular - use least-squares solution instead
        coeffs, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
        import warnings
        warnings.warn(
            f"Matrix M is rank-deficient (rank={rank}/{M.shape[0]}) for degree={degree}. "
            f"Using least-squares solution.",
            RuntimeWarning,
            stacklevel=2
        )

    # Compute chi2 on original scale, weighted by the same w used in the fit
    # (so AICc/degree selection sees the WLS objective, not an unweighted form)
    yhat = A @ coeffs
    chi2 = float(np.sum(w * (y - yhat) ** 2))

    # Degrees of freedom
    if df_method == "naive" or ridge_lambda <= 0.0 or not compute_dof:
        eff_params = float(degree + 1)
        dof = float(max(1, n - (degree + 1)))
    else:
        # Effective parameters via trace(H), where H = A (A^T W A + λR)^-1 A^T W
        # We'll compute in weighted form: H = (Aw) M^-1 (Aw)^T
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            Minv = np.linalg.pinv(M)
        H = Aw @ Minv @ Aw.T
        eff_params = float(np.trace(H))
        dof = float(max(1e-12, n - eff_params))

    return coeffs, chi2, dof, eff_params


def _weighted_ridge_fit_gls(
    mu: np.ndarray,
    y: np.ndarray,
    sigma_stat: np.ndarray,
    sigma_sys_indep_per_exp: np.ndarray,
    exp_index: np.ndarray,
    degree: int,
    *,
    sigma_sys_dep_per_row: Optional[np.ndarray] = None,
    ridge_lambda: float = 0.0,
    ridge_power: int = 4,
    df_method: Literal["naive", "hat"] = "hat",
    external_weights: Optional[np.ndarray] = None,
    fixed_c0: Optional[float] = None,
    fixed_coeffs: Optional[Dict[int, float]] = None,
    compute_dof: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Block-correlated GLS Legendre ridge fit with per-experiment systematic
    correlation modelled as **two** rank-1 modes per experiment. Same return
    contract as ``_weighted_ridge_fit``.

    Data covariance model
    ---------------------
        Σ̃_ii  = σ²_stat,i / g_i                                            (diagonal)
        Σ̃_ij  = [σ²_sys_indep,e + σ_sys_dep,i·σ_sys_dep,j] · y_i y_j
                     / √(g_i g_j)                                          (i, j in same experiment e)
        Σ̃_ij  = 0                                                         (different experiments)

    Two correlated rank-1 modes per experiment:
      • **u_e** (normalisation): per-experiment scalar σ_sys_indep,e times y_i.
        Same amplitude across the experiment, modelling a flux/efficiency
        normalisation that shifts the whole curve up or down together.
      • **v_e** (shape): per-row σ_sys_dep_i times y_i. Amplitude varies with
        energy/angle, modelling a piecewise-energy calibration error like
        Cierjacks's ERR-T whose magnitude is energy-dependent but whose draw
        is shared across all points of the experiment.

    Both modes are marked ``correlated: true`` in the manifest (see
    ``uncertainty_manifest.py``); putting σ_sys_dep on the diagonal — as
    earlier versions of this function did — would silently treat a correlated
    shape error as uncorrelated noise and inflate the effective degrees of
    freedom.

    Algorithm
    ---------
    Let U = [u_e, v_e]_e be N×2n_exp. Σ̃ = D + UUᵀ. Woodbury gives
        Σ̃⁻¹ = D⁻¹ − D⁻¹ U M⁻¹ Uᵀ D⁻¹,   M = I_{2n_exp} + Uᵀ D⁻¹ U
    which is block-diagonal with 2×2 blocks per experiment (because each
    experiment's two columns share the same disjoint support). Each 2×2
    inverse is closed form, so the cost stays O(n·k + n_exp·k).

    Reduces to ``_weighted_ridge_fit(σ = σ_stat)`` exactly when BOTH
    ``sigma_sys_indep_per_exp`` and ``sigma_sys_dep_per_row`` are zero.

    Parameters
    ----------
    mu, y : np.ndarray
        Cosines and cross-section values.
    sigma_stat : np.ndarray
        Per-row statistical uncertainty (absolute units, b/sr). σ_stat ONLY —
        do not pre-add σ_sys in quadrature.
    sigma_sys_indep_per_exp : np.ndarray, shape (n_exp,)
        Per-experiment normalisation σ (relative fraction, e.g. 0.08 = 8%).
    exp_index : np.ndarray, shape (n,) of int
        Block id 0..n_exp−1 telling which experiment each row belongs to.
    sigma_sys_dep_per_row : np.ndarray, optional
        Per-row correlated shape σ (relative fraction). Becomes the second
        rank-1 column v_e per experiment. If None or all zero, GLS collapses
        to the single-mode rank-1 form.
    external_weights : np.ndarray, optional
        Kernel weights g_i (e.g. Gaussian TOF kernel).
    fixed_c0, fixed_coeffs, ridge_lambda, ridge_power, df_method, compute_dof :
        Same semantics as ``_weighted_ridge_fit``.

    Returns
    -------
    (coeffs, chi2, dof, eff_params) : same contract as ``_weighted_ridge_fit``.

    Notes
    -----
    Edge cases:
    - σ_sys_indep_per_exp[e] = 0 AND σ_sys_dep on exp e all-zero: experiment e
      contributes only to D (no rank-1 absorption).
    - Single-point experiment: each rank-1 mode marginalises its own
      normalisation — the lone point's absolute level becomes unconstrained.
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(sigma_stat)):
        raise ValueError("mu, y, sigma_stat must be finite.")
    if np.any(sigma_stat <= 0):
        raise ValueError("All sigma_stat must be > 0.")

    n = mu.size
    n_exp = int(np.asarray(sigma_sys_indep_per_exp).size)

    # Kernel weights (default 1)
    if external_weights is not None:
        g = np.asarray(external_weights, dtype=float)
    else:
        g = np.ones(n, dtype=float)

    # Per-row correlated shape sys (default 0). Now a *rank-1* column per
    # experiment, NOT a diagonal contribution — see docstring.
    if sigma_sys_dep_per_row is not None:
        sigma_sys_dep = np.asarray(sigma_sys_dep_per_row, dtype=float)
    else:
        sigma_sys_dep = np.zeros(n, dtype=float)

    # Diagonal of Σ̃: D̃_ii = σ²_stat,i / g_i (σ_dep is in V, not D).
    var_diag = sigma_stat ** 2
    var_diag = np.maximum(var_diag, 1e-30)
    inv_D = g / var_diag  # 1D length n

    # Fixed coefficients — mirror _weighted_ridge_fit's logic exactly
    fixed_values: Dict[int, float] = {}
    if fixed_coeffs is not None:
        fixed_values.update(fixed_coeffs)
    if fixed_c0 is not None:
        fixed_values[0] = fixed_c0

    A_full = legvander(mu, degree)
    all_indices = list(range(degree + 1))

    if fixed_values:
        free_indices = [l for l in all_indices if l not in fixed_values]
        fixed_indices = sorted(fixed_values.keys())

        if not free_indices:
            coeffs = np.array([fixed_values.get(l, 0.0) for l in all_indices])
            yhat = A_full @ coeffs
            r = y - yhat
            chi2_diag_only = float(np.sum((r ** 2) * inv_D))
            chi2_corr = 0.0
            for e in range(n_exp):
                s_e = float(sigma_sys_indep_per_exp[e])
                idx = np.where(exp_index == e)[0]
                if idx.size == 0:
                    continue
                vd = var_diag[idx]
                d_e = sigma_sys_dep[idx]
                has_indep = s_e > 0.0
                has_dep = bool(np.any(d_e > 0.0))
                if not (has_indep or has_dep):
                    continue
                sqg = np.sqrt(np.maximum(g[idx], 1e-30))
                # 2×2 M_e and its determinant; Woodbury reduces correctly to
                # rank-1 when only one mode is active (off-diagonal & one
                # diagonal entry are zero, det = (1 + s_uu) or (1 + s_vv)).
                s_uu = (s_e ** 2) * float(np.sum((y[idx] ** 2) / vd)) if has_indep else 0.0
                s_vv = float(np.sum(((d_e * y[idx]) ** 2) / vd)) if has_dep else 0.0
                s_uv = (
                    s_e * float(np.sum(d_e * (y[idx] ** 2) / vd))
                    if (has_indep and has_dep) else 0.0
                )
                M_uu = 1.0 + s_uu
                M_vv = 1.0 + s_vv
                det = M_uu * M_vv - s_uv ** 2
                if det <= 0.0:
                    continue
                # rᵀ D⁻¹ u_e and rᵀ D⁻¹ v_e
                u_r = (
                    s_e * float(np.sum(y[idx] * r[idx] * sqg / vd))
                    if has_indep else 0.0
                )
                v_r = (
                    float(np.sum(d_e * y[idx] * r[idx] * sqg / vd))
                    if has_dep else 0.0
                )
                chi2_corr += (
                    M_vv * u_r ** 2 - 2.0 * s_uv * u_r * v_r + M_uu * v_r ** 2
                ) / det
            chi2 = chi2_diag_only - chi2_corr
            return coeffs, chi2, float(max(1, n)), 0.0

        fixed_vals = np.array([fixed_values[l] for l in fixed_indices])
        y_adj = y - A_full[:, fixed_indices] @ fixed_vals
        A = A_full[:, free_indices]
        n_free = len(free_indices)

        if ridge_lambda > 0.0:
            pen = np.array([float(l ** ridge_power) if l > 0 else 0.0
                            for l in free_indices])
            R = np.diag(pen)
        else:
            R = np.zeros((n_free, n_free), dtype=float)
    else:
        free_indices = all_indices
        fixed_indices = []
        y_adj = y
        A = A_full
        n_free = degree + 1

        if ridge_lambda > 0.0:
            pen = np.zeros(degree + 1, dtype=float)
            for l in range(1, degree + 1):
                pen[l] = float(l ** ridge_power)
            R = np.diag(pen)
        else:
            R = np.zeros((degree + 1, degree + 1), dtype=float)

    # Diagonal Gram and rhs (no Woodbury correction yet):
    #   X^T D̃⁻¹ X = Σ_i inv_D_i · X[i,:] X[i,:]^T
    #   X^T D̃⁻¹ y = Σ_i inv_D_i · X[i,:] · y_adj_i
    A_iD = A * inv_D[:, None]
    G_diag = A.T @ A_iD            # (n_free, n_free)
    rhs_diag = A_iD.T @ y_adj      # (n_free,)

    # Woodbury per-experiment correction. Each experiment contributes two
    # rank-1 columns to U (u_e: normalisation; v_e: shape, per-row amplitude),
    # so the per-experiment block of M = I + UᵀD⁻¹U is 2×2 (and disjoint across
    # experiments because the column supports don't overlap). Cost still
    # O(n_exp · n_e · n_free) up to a small constant.
    G_corr = np.zeros_like(G_diag)
    rhs_corr = np.zeros(n_free, dtype=float)
    # Cache per-experiment 2×2 (M_uu, M_vv, M_uv, det) for the χ² recomputation
    # below; tuple of NaN signals "no rank-1 contribution from this experiment".
    M_e_cache: list = [None] * n_exp
    for e in range(n_exp):
        s_e = float(sigma_sys_indep_per_exp[e])
        idx = np.where(exp_index == e)[0]
        if idx.size == 0:
            continue
        vd = var_diag[idx]
        d_e = sigma_sys_dep[idx]
        has_indep = s_e > 0.0
        has_dep = bool(np.any(d_e > 0.0))
        if not (has_indep or has_dep):
            continue
        sqg = np.sqrt(np.maximum(g[idx], 1e-30))
        # 2×2 M_e components
        s_uu = (s_e ** 2) * float(np.sum((y[idx] ** 2) / vd)) if has_indep else 0.0
        s_vv = float(np.sum(((d_e * y[idx]) ** 2) / vd)) if has_dep else 0.0
        s_uv = (
            s_e * float(np.sum(d_e * (y[idx] ** 2) / vd))
            if (has_indep and has_dep) else 0.0
        )
        M_uu = 1.0 + s_uu
        M_vv = 1.0 + s_vv
        det = M_uu * M_vv - s_uv ** 2
        if det <= 0.0:
            continue
        M_e_cache[e] = (M_uu, M_vv, s_uv, det)
        # z_u[j] = Xᵀ D̃⁻¹ u_e_{:,j} = s_e · Σ A[i,j] · y_i · √g_i / var_i
        # z_v[j] = Xᵀ D̃⁻¹ v_e_{:,j} = Σ A[i,j] · d_i · y_i · √g_i / var_i
        if has_indep:
            z_u = s_e * (A[idx, :].T @ ((sqg * y[idx]) / vd))
        else:
            z_u = np.zeros(n_free, dtype=float)
        if has_dep:
            z_v = A[idx, :].T @ ((sqg * d_e * y[idx]) / vd)
        else:
            z_v = np.zeros(n_free, dtype=float)
        # u_e^T D̃⁻¹ y_adj , v_e^T D̃⁻¹ y_adj
        Uy = (
            s_e * float(np.sum(y[idx] * y_adj[idx] * sqg / vd))
            if has_indep else 0.0
        )
        Vy = (
            float(np.sum(d_e * y[idx] * y_adj[idx] * sqg / vd))
            if has_dep else 0.0
        )
        # Rank-2 Woodbury: G_corr += [z_u, z_v] M_e⁻¹ [z_u; z_v]ᵀ
        # rhs_corr  += [z_u, z_v] M_e⁻¹ [Uy; Vy]
        inv_det = 1.0 / det
        G_corr += inv_det * (
            M_vv * np.outer(z_u, z_u)
            - s_uv * (np.outer(z_u, z_v) + np.outer(z_v, z_u))
            + M_uu * np.outer(z_v, z_v)
        )
        rhs_corr += inv_det * (
            (M_vv * Uy - s_uv * Vy) * z_u
            + (M_uu * Vy - s_uv * Uy) * z_v
        )

    # Normal equations: (X^T Σ̃⁻¹ X + λ R) β̂ = X^T Σ̃⁻¹ y
    G_full = G_diag - G_corr
    rhs_full = rhs_diag - rhs_corr
    P = G_full + ridge_lambda * R

    try:
        coeffs_partial = np.linalg.solve(P, rhs_full)
    except np.linalg.LinAlgError:
        coeffs_partial, _, rank, _ = np.linalg.lstsq(P, rhs_full, rcond=None)
        import warnings
        warnings.warn(
            f"GLS normal-equations matrix is rank-deficient (rank={rank}/{P.shape[0]}) "
            f"for degree={degree} with {len(fixed_values)} fixed coefficients. "
            f"Using least-squares solution.",
            RuntimeWarning,
            stacklevel=2,
        )

    if fixed_values:
        coeffs = np.zeros(degree + 1, dtype=float)
        for l in fixed_indices:
            coeffs[l] = fixed_values[l]
        for i, l in enumerate(free_indices):
            coeffs[l] = coeffs_partial[i]
    else:
        coeffs = coeffs_partial

    # χ²_GLS = rᵀ Σ̃⁻¹ r (rank-2 closed-form Woodbury reduction):
    #   χ²_diag = Σ_i r²_i · inv_D_i        (kernel-weighted diagonal piece)
    #   χ²_corr = Σ_e [Ur, Vr] M_e⁻¹ [Ur; Vr]   (two-mode rank-1 absorption)
    #   χ²     = χ²_diag − χ²_corr
    yhat_full = A_full @ coeffs
    r = y - yhat_full
    chi2_diag = float(np.sum((r ** 2) * inv_D))
    chi2_corr = 0.0
    for e in range(n_exp):
        cache = M_e_cache[e]
        if cache is None:
            continue
        M_uu, M_vv, s_uv, det = cache
        idx = np.where(exp_index == e)[0]
        if idx.size == 0:
            continue
        vd = var_diag[idx]
        d_e = sigma_sys_dep[idx]
        s_e = float(sigma_sys_indep_per_exp[e])
        sqg = np.sqrt(np.maximum(g[idx], 1e-30))
        Ur = (
            s_e * float(np.sum(y[idx] * r[idx] * sqg / vd))
            if s_e > 0.0 else 0.0
        )
        Vr = (
            float(np.sum(d_e * y[idx] * r[idx] * sqg / vd))
            if np.any(d_e > 0.0) else 0.0
        )
        chi2_corr += (
            M_vv * Ur ** 2 - 2.0 * s_uv * Ur * Vr + M_uu * Vr ** 2
        ) / det
    chi2 = chi2_diag - chi2_corr
    # Numerical floor: χ² should be non-negative (the Woodbury subtraction can
    # push slightly below zero from rounding when the rank-1 absorption is
    # near-perfect). Clamp at zero to keep AICc finite.
    if chi2 < 0.0:
        chi2 = 0.0

    # Effective parameters via GLS hat-trace.
    # H_GLS = X (P)⁻¹ Xᵀ Σ̃⁻¹  →  trace(H_GLS) = trace(P⁻¹ G_full)
    # (G_full = X^T Σ̃⁻¹ X). Reduces to the diagonal hat-trace when corr=0.
    if df_method == "naive" or ridge_lambda <= 0.0 or not compute_dof:
        eff_params = float(n_free)
        dof = float(max(1, n - n_free))
    else:
        try:
            P_inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(P)
        eff_params = float(np.trace(P_inv @ G_full))
        dof = float(max(1e-12, n - eff_params))

    return coeffs, chi2, dof, eff_params


def _batch_mc_ridge_solve(
    mu: np.ndarray,
    Y_perturbed: np.ndarray,
    sigma: np.ndarray,
    degree: int,
    ridge_lambda: float = 0.0,
    ridge_power: int = 4,
    external_weights: Optional[np.ndarray] = None,
    fixed_c0: Optional[float] = None,
    fixed_coeffs: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Batch-solve MC ridge regressions: factorize M once, solve all RHS at once.

    Parameters
    ----------
    mu : np.ndarray
        Cosine of scattering angle, shape (n,).
    Y_perturbed : np.ndarray
        Perturbed y vectors, shape (n_draws, n).
    sigma : np.ndarray
        Uncertainties, shape (n,).
    degree : int
        Legendre polynomial degree.
    ridge_lambda, ridge_power, external_weights, fixed_c0, fixed_coeffs :
        Same as ``_weighted_ridge_fit``.

    Returns
    -------
    np.ndarray
        Fitted coefficients, shape (n_draws, degree + 1).
    """
    # Weights (identical for all samples)
    if external_weights is not None:
        w = external_weights / (sigma ** 2)
    else:
        w = 1.0 / (sigma ** 2)
    sw = np.sqrt(w)

    # Merge fixed coefficient specs
    fixed_values: Dict[int, float] = {}
    if fixed_coeffs is not None:
        fixed_values.update(fixed_coeffs)
    if fixed_c0 is not None:
        fixed_values[0] = fixed_c0

    A_full = legvander(mu, degree)
    all_indices = list(range(degree + 1))
    n_draws = Y_perturbed.shape[0]

    if fixed_values:
        free_indices = [l for l in all_indices if l not in fixed_values]
        fixed_indices = sorted(fixed_values.keys())

        if not free_indices:
            # All coefficients fixed — nothing to solve
            coeffs_row = np.array([fixed_values.get(l, 0.0) for l in all_indices])
            return np.tile(coeffs_row, (n_draws, 1))

        fixed_vals = np.array([fixed_values[l] for l in fixed_indices])
        # Subtract fixed contributions from all rows at once
        Y_adj = Y_perturbed - (A_full[:, fixed_indices] @ fixed_vals)[np.newaxis, :]

        A = A_full[:, free_indices]
        Aw = A * sw[:, None]
        n_free = len(free_indices)

        if ridge_lambda > 0.0:
            pen = np.array([float(l ** ridge_power) if l > 0 else 0.0
                            for l in free_indices])
            R = np.diag(pen)
        else:
            R = np.zeros((n_free, n_free), dtype=float)
    else:
        Y_adj = Y_perturbed
        A = A_full
        Aw = A * sw[:, None]
        n_free = degree + 1

        if ridge_lambda > 0.0:
            pen = np.arange(degree + 1, dtype=float) ** ridge_power
            pen[0] = 0.0
            R = np.diag(pen)
        else:
            R = np.zeros((n_free, n_free), dtype=float)
        free_indices = all_indices
        fixed_indices = []

    M = Aw.T @ Aw + ridge_lambda * R

    # Factorize M once
    try:
        factor = cho_factor(M)
        use_cho = True
    except np.linalg.LinAlgError:
        M_pinv = np.linalg.pinv(M)
        use_cho = False

    # Weighted RHS for all samples: shape (n_free, n_draws)
    Yw = Y_adj * sw[np.newaxis, :]  # (n_draws, n)
    RHS = Aw.T @ Yw.T              # (n_free, n_draws)

    # Batch solve
    if use_cho:
        X = cho_solve(factor, RHS)  # (n_free, n_draws)
    else:
        X = M_pinv @ RHS

    # Reconstruct full coefficient matrix
    if fixed_values:
        coeffs_all = np.zeros((n_draws, degree + 1), dtype=float)
        for l in fixed_indices:
            coeffs_all[:, l] = fixed_values[l]
        for i, l in enumerate(free_indices):
            coeffs_all[:, l] = X[i, :]
    else:
        coeffs_all = X.T  # (n_draws, n_free) = (n_draws, degree+1)

    return coeffs_all


def _criterion_score(
    chi2: float,
    n: int,
    k: float,
    criterion: Literal["aicc", "bic", "aic"]
) -> float:
    """
    Model selection score (lower is better). Three options:

    - ``"aic"``  → ``χ² + 2k``                        (Akaike Information Criterion)
    - ``"aicc"`` → ``χ² + 2k + 2k(k+1)/(n−k−1)``     (AIC w/ small-sample correction)
    - ``"bic"``  → ``χ² + k·ln(max(1, n))``           (Bayesian Information Criterion)

    AICc reduces to AIC as n → ∞; for small n its correction term blows up
    fast and over-penalises higher orders (penalty ≈ 30 at n=8, k=5). AIC
    is the right choice when small-sample bins persist as L=1 picks despite
    χ² clearly preferring higher L. BIC is more conservative than AIC for
    n ≥ 8 and more permissive than AICc for n ≤ ~12; useful as a third
    reference point but neither minimax-optimal nor consistent for our
    setting.
    """
    if criterion == "bic":
        return chi2 + k * np.log(max(1, n))
    if criterion == "aic":
        return chi2 + 2.0 * k
    # AICc (default)
    aic = chi2 + 2.0 * k
    # AICc correction (requires n > k + 1)
    if n > (k + 1.0):
        return aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    return aic + 1e6  # penalize impossible region


def sample_legendre_coefficients(
    df: pd.DataFrame,
    value_col: str = "value",
    unc_col: str = "unc",
    mu_col: Optional[str] = None,
    theta_deg_col: Optional[str] = "theta_deg",
    *,
    # order control
    degree: Optional[int] = None,
    max_degree: int = 20,
    select_degree: Optional[Literal["aicc", "bic", "aic"]] = None,
    # ridge
    ridge_lambda: float = 0.0,
    ridge_power: int = 4,
    df_method: Literal["naive", "hat"] = "hat",
    # external weights (for Gaussian kernel)
    external_weights: Optional[np.ndarray] = None,
    # systematic-uncertainty column for sys-aware fitting
    sys_unc_col: Optional[str] = None,
    # sampling
    n_samples: int = 1,
    stochastic: bool = False,
    rescale_unc_by_chi2: bool = True,
    allow_shrink_unc: bool = False,
    random_state: Optional[int] = None,
    # angular-band discrepancy model
    use_band_discrepancy: bool = False,
    min_points_per_band: int = 3,
    max_band_scale: float = 3.0,
    tau_irls_max_iters: int = 1,
    tau_irls_tol: float = 1e-3,
    tau_irls_damping: float = 0.0,
    band_scale_method: Literal["mad", "rms", "hybrid"] = "mad",
    # fixed-c0 mode (Improvement 1.2)
    freeze_c0: bool = False,
    fixed_c0_value: Optional[float] = None,
    # correlated normalization uncertainty (Improvement 1.3)
    sigma_norm: float = 0.0,
    sigma_norm_common_mode: float = 0.0,
    norm_group_cols: Tuple[str, ...] = ("entry",),
    norm_dist: Literal["lognormal", "normal"] = "lognormal",
    # freeze higher-order coefficients during MC sampling
    max_sample_order: Optional[int] = None,
    # Post-τ AICc re-scan: rebuild model-degree weights using the τ-refined
    # σ_eff so per-sample degree draws sample a model-degree distribution
    # consistent with the band-discrepancy uncertainty model. When the
    # post-τ winner differs from the pre-τ winner, the fit is also refit
    # at the new degree under τ-IRLS to keep coeffs and τ mutually
    # consistent. No-op unless ``select_degree`` is set and
    # ``use_band_discrepancy`` is True.
    rerun_aicc_post_tau: bool = False,
    # Block-correlated GLS kernel for the IC model-selection scan, the
    # initial nominal fit, and the post-τ rescan. Reduces to diagonal WLS
    # when σ_sys_indep = 0. freeze_c0 refit and MC sampler always stay
    # diagonal; the τ-IRLS refit follows ``tau_refit_use_gls``.
    use_gls_kernel: bool = False,
    # Solver for the coefficient refit *inside* the τ-IRLS loop. The default
    # (False) keeps the historical diagonal WLS refit, which folds σ_sys into
    # the fit weights as if it were uncorrelated. That is inconsistent with
    # the scan and the post-τ rescan, which both use the block-correlated GLS
    # kernel when ``use_gls_kernel`` is on — so with the default the shipped
    # coeffs0 and the AICc scores that selected its degree come from different
    # noise models. Set True to run the τ refit under the same GLS kernel.
    #
    # ⚠ The GLS branch takes σ_eff (= τ·σ_stat) and carries σ_sys as rank-1
    #   structure. It must NOT be handed ``sigma_refit`` (σ_sys already folded
    #   into the diagonal) or σ_sys is counted twice.
    #
    # No-op unless ``use_gls_kernel`` is True and GLS inputs were supplied.
    tau_refit_use_gls: bool = False,
    # Phase-2 elastic magnitude channel. When True, record the closed-form
    # fixed-shape c0 scale of every (perturbed) sample against the nominal
    # bin curve into ``info["c0_samples"]`` — a read-only side channel that
    # leaves ``coef_df`` and the shape fit untouched (default off → the info
    # dict is unchanged). ``c0_scale_ref_coeffs`` is the shape to project onto
    # (the pipeline nominal bin coeffs); when None the fit's own ``coeffs0``
    # is used.
    record_c0_scale: bool = False,
    c0_scale_ref_coeffs: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit Legendre coefficients c_l for y(mu) = sum c_l P_l(mu) and return samples.

    - If n_samples == 1: returns a single row with the nominal fitted coefficients.
    - If n_samples > 1: computes reduced chi2 from nominal fit, rescales uncertainties,
      then generates n_samples fits from jittered data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with angular distribution data
    value_col, unc_col : str
        Column names for values and uncertainties
    mu_col, theta_deg_col : str
        Column names for cos(theta) or angle in degrees
    degree : int, optional
        Fixed Legendre order. If None, auto-select.
    max_degree : int
        Maximum Legendre order to consider
    select_degree : str, optional
        Criterion for order selection: "aicc" (default), "aic", or "bic".
        See ``_criterion_score`` for the formulas.
    ridge_lambda : float
        Ridge regularization parameter
    external_weights : np.ndarray, optional
        Gaussian kernel weights g_ij. Combined with uncertainties as w_ij = g_ij / σ²
    n_samples : int
        Number of MC samples to generate
    stochastic : bool
        If True, force stochastic sampling (normalization + pointwise noise) even when
        n_samples=1. This is useful for energy-jitter MC where each iteration needs
        a single stochastic sample rather than the nominal fit.
    rescale_unc_by_chi2 : bool
        Apply global Birge scaling (only if use_band_discrepancy=False)
    use_band_discrepancy : bool
        Use angular-band discrepancy model instead of global Birge scaling
    min_points_per_band : int
        Minimum points per band for τ estimation
    max_band_scale : float
        Maximum multiplicative scale factor per angular band
    freeze_c0 : bool
        If True, fix c0 to either fixed_c0_value (if provided) or the nominal fit c0.
        This enables shape-only refits where MF3 is fixed and MF34 explains discrepancies.
    fixed_c0_value : float, optional
        Explicit value to fix c0 at. Only used if freeze_c0=True.
    sigma_norm : float
        Per-experiment normalization uncertainty (e.g., 0.05 for 5%).
        Applies correlated multiplicative noise per experiment group in MC sampling.
    norm_group_cols : Tuple[str, ...]
        Column names used to group data by experiment (default: ("entry",))
    norm_dist : str
        Distribution for normalization factor: "lognormal" (default, always positive)
        or "normal" (can go negative for large sigma_norm).
    max_sample_order : int, optional
        Maximum Legendre order to vary during MC sampling. Coefficients above this
        order are frozen at their nominal (best-fit) values. None = vary all orders.
        E.g., max_sample_order=3 means only c0-c3 are re-fitted from perturbed data.

    Returns
    -------
    coef_df : pd.DataFrame
        DataFrame with columns c0..cL and n_samples rows
    info : dict
        Fit metadata (degree, chi2_red, scale_factor, tau_info, etc.)
    """
    if value_col not in df.columns or unc_col not in df.columns:
        raise ValueError(f"Dataframe must contain '{value_col}' and '{unc_col}' columns.")

    work = df[[value_col, unc_col] + [c for c in df.columns if c not in (value_col, unc_col)]].copy()
    work = work.reset_index(drop=True)  # Reset index for consistent filtering

    # Build a mask for valid rows (finite values and positive uncertainties)
    valid_mask = (
        ~work[value_col].replace([np.inf, -np.inf], np.nan).isna() &
        ~work[unc_col].replace([np.inf, -np.inf], np.nan).isna() &
        (work[unc_col] > 0)
    )
    work = work[valid_mask].copy()

    # Filter external_weights if provided to match the filtered data
    if external_weights is not None:
        external_weights = external_weights[valid_mask.to_numpy()]

    mu = _infer_mu(work, mu_col=mu_col, theta_deg_col=theta_deg_col)
    y = work[value_col].to_numpy(dtype=float)
    sigma = work[unc_col].to_numpy(dtype=float)

    # Per-point systematic uncertainty (absolute units, same as ``sigma``).
    # When provided, fit weights use σ_total² = σ_stat² + σ_sys² so the WLS
    # doesn't pretend per-experiment normalization scatter is zero, and τ is
    # computed against the same σ_total — keeping the residual interpretation
    # consistent across initial fit, IRLS refit, and τ estimation.
    sigma_sys: Optional[np.ndarray] = None
    if sys_unc_col is not None and sys_unc_col in work.columns:
        sigma_sys = work[sys_unc_col].to_numpy(dtype=float)
        sigma_sys = np.where(np.isfinite(sigma_sys) & (sigma_sys > 0), sigma_sys, 0.0)
    sigma_for_fit = (
        np.sqrt(sigma ** 2 + sigma_sys ** 2) if sigma_sys is not None else sigma
    )

    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 points to fit anything meaningful.")

    # Block-correlated GLS inputs (used only when use_gls_kernel is True).
    # Built once and reused across the AICc scan, the initial nominal fit,
    # and the post-τ AICc rescan. Group key matches ``norm_group_cols`` so
    # the GLS partition is consistent with how the MC sampler draws per-
    # experiment normalisation factors downstream.
    gls_indep_per_exp: Optional[np.ndarray] = None
    gls_exp_index: Optional[np.ndarray] = None
    gls_dep_per_row: Optional[np.ndarray] = None
    if use_gls_kernel:
        group_cols_avail = [c for c in norm_group_cols if c in work.columns]
        if group_cols_avail:
            # Build a single string key per row by joining the group columns.
            # np.unique over object arrays doesn't support axis=0, so collapse
            # to a 1D string index.
            cols_str = [work[c].astype(str).to_numpy() for c in group_cols_avail]
            keys_1d = np.array(['\x00'.join(parts) for parts in zip(*cols_str)])
            _u_keys, gls_exp_index = np.unique(keys_1d, return_inverse=True)
            n_exp_local = len(_u_keys)
            # Honest manifest split — both modes are correlated within an
            # experiment (manifest ``correlated: true``), so we pass each to
            # ``_weighted_ridge_fit_gls`` as a separate rank-1 column:
            #   * gls_indep_per_exp[e] = scalar σ_sys_indep_relative for
            #     experiment e (energy-independent normalisation, e.g. flux).
            #   * gls_dep_per_row[i]  = per-row σ_sys_dep_relative, which
            #     enters as v_e = σ_dep_i · y_i — a rank-1 SHAPE mode whose
            #     amplitude varies with energy/angle but whose draw is shared
            #     across rows of one experiment (e.g. Cierjacks's piecewise
            #     ERR-T).
            # Putting σ_dep on the diagonal would treat a correlated shape
            # error as independent noise — that was the bug Fix 1a addresses.
            gls_indep_per_exp = np.zeros(n_exp_local, dtype=float)
            if 'sigma_sys_indep_rel' in work.columns:
                indep_arr = work['sigma_sys_indep_rel'].to_numpy(dtype=float)
            elif 'sigma_sys_indep_relative' in work.columns:
                indep_arr = work['sigma_sys_indep_relative'].to_numpy(dtype=float)
            else:
                indep_arr = None
            if indep_arr is not None:
                for e in range(n_exp_local):
                    mask = (gls_exp_index == e)
                    vals = indep_arr[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        # σ_indep is scalar per experiment, so all rows of e
                        # share one value; take the first (mean handles
                        # rounding noise without changing the value).
                        gls_indep_per_exp[e] = float(np.mean(vals))

            if 'sigma_sys_dep_rel' in work.columns:
                gls_dep_per_row = work['sigma_sys_dep_rel'].to_numpy(dtype=float)
            elif 'sigma_sys_dep_relative' in work.columns:
                gls_dep_per_row = work['sigma_sys_dep_relative'].to_numpy(dtype=float)
            else:
                gls_dep_per_row = None
            if gls_dep_per_row is not None:
                gls_dep_per_row = np.where(
                    np.isfinite(gls_dep_per_row) & (gls_dep_per_row > 0),
                    gls_dep_per_row, 0.0,
                )

            # Single-experiment bin fallback: GLS exists to absorb *inter-*
            # experiment normalisation disagreement. With only one experiment
            # there is no such disagreement — applying GLS just marginalises
            # that one experiment's normalisation, which makes c_0 structurally
            # unidentifiable, shrinks the GLS-fit c_0 toward zero, and biases
            # AICc to pick L=1 (or L=0 if not excluded) with a visually broken
            # nominal level. Detected and observed empirically on 91/152
            # single-Kinney bins in the 1.5–2.5 MeV range. Fall back to
            # diagonal WLS over σ_total = sqrt(σ²_stat + σ²_sys) for these
            # bins — that keeps c_0 anchored and lets the AICc scan compare
            # honest shape χ² across degrees.
            if gls_indep_per_exp is not None and gls_indep_per_exp.size <= 1:
                gls_indep_per_exp = None
                gls_exp_index = None
                gls_dep_per_row = None

    # When GLS marginalises every active experiment's normalisation (i.e. no
    # "anchor experiment" with σ_sys_indep = 0), the c_0 direction is
    # structurally unidentifiable: per-experiment scale factors absorb any
    # constant offset, so χ²(L=0) collapses to ~0 regardless of the data.
    # AICc would then trivially pick L=0 even on highly anisotropic bins
    # (observed: 152/154 single-Kinney bins picked L=0 in the first GLS run).
    # Exclude L=0 from the AICc scan in that case so the scan compares only
    # shape models that the data can actually constrain. The chosen L still
    # has c_0 in its parameter list — c_0 just gets fitted as a nuisance
    # direction with no χ² penalty, exactly how the marginalisation intends.
    gls_no_anchor = (
        use_gls_kernel
        and gls_indep_per_exp is not None
        and gls_indep_per_exp.size > 0
        and bool(np.all(gls_indep_per_exp > 0))
    )

    # AICc sample size: use the RAW point count, not the Kish ESS. The Kish
    # ESS — (Σw)^2/Σw^2 with w = kw/σ_for_fit² — collapses to ~k+1 whenever
    # one point has a much smaller σ than the others (e.g. a low-cross-section
    # point near the angular minimum, which gets w ≈ 10^4× the typical row).
    # Combined with `_criterion_score`'s `+1e6` hard penalty for n ≤ k+1,
    # this STRUCTURALLY BANS every degree above ~1 even when the χ² at L=0 is
    # catastrophic — observed: 0.847 MeV bin, χ²(L=0)=144 vs χ²(L=2)=8, but
    # n_eff_aicc=3.0 banned L≥2 and L=0 won by default. AICc's complexity
    # penalty is about "how many parameters can I afford given my MEASUREMENT
    # COUNT" (≈ raw n), not "how concentrated is my fit's information"
    # (Kish ESS). The chi² already absorbs the weighting; the parsimony term
    # should not double-count it.
    n_aicc = float(n)

    # Choose degree based on number of UNIQUE mu values, not total points.
    # The Legendre Vandermonde matrix has rank = n_unique_mu, so we need at least
    # degree+1 unique angles to avoid a rank-deficient design matrix.
    n_unique_mu = len(np.unique(np.round(mu, decimals=6)))  # Round to handle numerical noise
    max_feasible = min(max_degree, n_unique_mu - 1)  # Need at least degree+1 unique angles
    if max_feasible < 0:
        max_feasible = 0

    # Store all degrees info for model averaging
    all_degrees_info = {}

    if degree is None:
        if select_degree is None:
            degree_use = max_feasible
        else:
            # Scan degrees [d_min..max_feasible] and pick best score. Skip
            # L=0 when GLS has no anchor experiment (c_0 unidentifiable —
            # see ``gls_no_anchor`` definition above). When max_feasible=0
            # we still allow L=0 as the only option — that's a degenerate
            # bin and the choice is forced anyway.
            d_min = 1 if (gls_no_anchor and max_feasible >= 1) else 0
            best = None
            best_res = None
            for d in range(d_min, max_feasible + 1):
                if use_gls_kernel and gls_indep_per_exp is not None:
                    coeffs_d, chi2_d, dof_d, k_d = _weighted_ridge_fit_gls(
                        mu, y, sigma, gls_indep_per_exp, gls_exp_index, d,
                        sigma_sys_dep_per_row=gls_dep_per_row,
                        ridge_lambda=ridge_lambda,
                        ridge_power=ridge_power,
                        df_method=df_method,
                        external_weights=external_weights,
                    )
                else:
                    coeffs_d, chi2_d, dof_d, k_d = _weighted_ridge_fit(
                        mu, y, sigma_for_fit, d,
                        ridge_lambda=ridge_lambda,
                        ridge_power=ridge_power,
                        df_method=df_method,
                        external_weights=external_weights,
                    )
                score = _criterion_score(chi2_d, n=n_aicc, k=k_d, criterion=select_degree)

                # Store info for all viable degrees (for model averaging)
                all_degrees_info[d] = {
                    'coeffs': coeffs_d.copy(),
                    'chi2': chi2_d,
                    'dof': dof_d,
                    'eff_params': k_d,
                    'aicc': score,
                }

                if best is None or score < best:
                    best = score
                    best_res = (d, coeffs_d, chi2_d, dof_d, k_d)
            assert best_res is not None
            degree_use, coeffs0, chi2_0, dof_0, k_0 = best_res
    else:
        degree_use = int(degree)
        if degree_use > max_degree:
            raise ValueError(f"degree={degree_use} exceeds max_degree={max_degree}.")
        # If user asks degree larger than n-1, we can either fail or allow if ridge is on.
        if degree_use > (n - 1) and ridge_lambda <= 0.0:
            raise ValueError(
                f"degree={degree_use} is too high for N={n} points without ridge. "
                f"Use degree <= {n-1} or set ridge_lambda > 0."
            )

    # Nominal fit (if not already computed by selection step)
    if not (degree is None and select_degree is not None):
        if use_gls_kernel and gls_indep_per_exp is not None:
            coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit_gls(
                mu, y, sigma, gls_indep_per_exp, gls_exp_index, degree_use,
                sigma_sys_dep_per_row=gls_dep_per_row,
                ridge_lambda=ridge_lambda,
                ridge_power=ridge_power,
                df_method=df_method,
                external_weights=external_weights,
            )
        else:
            coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit(
                mu, y, sigma_for_fit, degree_use,
                ridge_lambda=ridge_lambda,
                ridge_power=ridge_power,
                df_method=df_method,
                external_weights=external_weights,
            )

    chi2_red = float(chi2_0 / max(1e-12, dof_0))

    # Compute effective uncertainties
    tau_info = {'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0}

    # Phase B audit follow-up: defaults so they exist on every code path.
    all_degrees_info_pre_tau: Optional[Dict[int, Dict[str, Any]]] = None
    post_tau_winner_changed: bool = False

    if use_band_discrepancy:
        # IRLS: alternate (estimate τ from residuals) ↔ (refit with σ_eff = τ·σ)
        # until τ stabilizes, so the returned coeffs and τ are mutually
        # consistent. With max_iters=1 this reproduces the legacy single-step
        # refit (one refit, then a final τ recompute against those coeffs).
        #
        # Geometric-mean damping (tau_irls_damping=α∈[0,1)) blends the new
        # τ_target with the previous iteration's used τ:
        #   τ_used = τ_target^(1-α) · τ_used_prev^α
        # This breaks limit cycles caused by MAD's non-monotonicity on
        # bimodal-residual bands. α=0 reproduces the un-damped pipeline.
        damping = float(max(0.0, min(0.999, tau_irls_damping)))
        forward_mask = mu > 0.5
        backward_mask = mu < -0.5
        mid_mask = ~forward_mask & ~backward_mask
        band_masks_irls = {'F': forward_mask, 'M': mid_mask, 'B': backward_mask}

        tau_used_prev: Optional[Dict[str, float]] = None
        tau_target_prev: Optional[Dict[str, float]] = None
        for _ in range(max(1, int(tau_irls_max_iters))):
            y_fit = legval(mu, coeffs0)
            sigma_eff, tau_info = compute_angular_band_discrepancy(
                mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                min_points_per_band=min_points_per_band,
                max_band_scale=max_band_scale,
                method=band_scale_method,
                sigma_sys=sigma_sys,
            )
            # Apply damping (if any) to the τ values used for the refit.
            if damping > 0 and tau_used_prev is not None:
                for b in ('tau_F', 'tau_M', 'tau_B'):
                    t_target = float(tau_info[b])
                    t_old = tau_used_prev[b]
                    t_damped = (t_target ** (1.0 - damping)) * (t_old ** damping)
                    t_damped = max(1.0, min(t_damped, max_band_scale))
                    tau_info[b] = t_damped
                # Rebuild sigma_eff from the damped τ values.
                sigma_eff = sigma.copy()
                for bn, bm in band_masks_irls.items():
                    s_b = float(tau_info[f'tau_{bn}'])
                    if s_b > 1.0:
                        sigma_eff[bm] = sigma[bm] * s_b
            # Compose total uncertainty for refit weights: σ_total_eff² =
            # (τ·σ_stat)² + σ_sys². σ_eff itself stays τ·σ_stat (preserved
            # for downstream MC and sigma_eff_from_tau).
            sigma_refit = (
                np.sqrt(sigma_eff ** 2 + sigma_sys ** 2)
                if sigma_sys is not None else sigma_eff
            )
            if (tau_refit_use_gls and use_gls_kernel
                    and gls_indep_per_exp is not None):
                # σ_eff (τ-inflated stat) on D; σ_sys stays rank-1 in U/V.
                # Deliberately NOT sigma_refit — see tau_refit_use_gls above.
                coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit_gls(
                    mu, y, sigma_eff, gls_indep_per_exp, gls_exp_index,
                    degree_use,
                    sigma_sys_dep_per_row=gls_dep_per_row,
                    ridge_lambda=ridge_lambda,
                    ridge_power=ridge_power,
                    df_method=df_method,
                    external_weights=external_weights,
                )
            else:
                coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit(
                    mu, y, sigma_refit, degree_use,
                    ridge_lambda=ridge_lambda,
                    ridge_power=ridge_power,
                    df_method=df_method,
                    external_weights=external_weights,
                )
            # Convergence: compare *raw* targets between consecutive iterations
            # (damping smooths the trajectory but we still want to see whether
            # the underlying estimator has settled).
            tau_target_curr = {
                b: float(tau_info.get(f'raw_{b[-1]}', tau_info[b]))
                for b in ('tau_F', 'tau_M', 'tau_B')
            }
            if tau_target_prev is not None:
                dtau = max(
                    abs(tau_target_curr[b] - tau_target_prev[b])
                    for b in ('tau_F', 'tau_M', 'tau_B')
                )
                if dtau < tau_irls_tol:
                    break
            tau_used_prev = {b: float(tau_info[b]) for b in ('tau_F', 'tau_M', 'tau_B')}
            tau_target_prev = tau_target_curr
        # Final τ on the final coefficients — guarantees mutual consistency.
        y_fit = legval(mu, coeffs0)
        sigma_eff, tau_info = compute_angular_band_discrepancy(
            mu=mu, y=y, sigma=sigma, y_fit=y_fit,
            min_points_per_band=min_points_per_band,
            max_band_scale=max_band_scale,
            method=band_scale_method,
            sigma_sys=sigma_sys,
        )
        chi2_red = float(chi2_0 / max(1e-12, dof_0))
        scale = 1.0  # No global scaling when using band model

        # Post-τ AICc re-scan. The original AICc loop (lines ~1415-1444) used
        # the pre-τ ``sigma_for_fit``, so the resulting model-degree weights
        # reflect a stat-only uncertainty model. When per-sample degree
        # draws use those weights downstream (v3's USE_DEGREE_SAMPLING_IN_MC),
        # the model-degree distribution is inconsistent with the τ-refined
        # nominal. Re-running the AICc body with σ_total_eff = √(σ_eff² +
        # σ_sys²) rebuilds the weights against the same noise model the
        # τ-IRLS converged under. If the post-τ winner differs from the
        # pre-τ winner, refit at the new degree under τ-IRLS once so coeffs
        # and τ remain mutually consistent. Stops after one re-scan to avoid
        # the AICc↔τ feedback loop (the new τ would in turn shift AICc).
        if (rerun_aicc_post_tau
                and degree is None
                and select_degree is not None
                and all_degrees_info):
            all_degrees_info_pre_tau = {
                d: dict(rec) for d, rec in all_degrees_info.items()
            }
            sigma_for_fit_post_tau = (
                np.sqrt(sigma_eff ** 2 + sigma_sys ** 2)
                if sigma_sys is not None else sigma_eff
            )
            new_all: Dict[int, Dict[str, Any]] = {}
            new_best: Optional[float] = None
            new_best_res: Optional[Tuple[int, np.ndarray, float, float, float]] = None
            d_min_post = 1 if (gls_no_anchor and max_feasible >= 1) else 0
            for d in range(d_min_post, max_feasible + 1):
                if use_gls_kernel and gls_indep_per_exp is not None:
                    # τ-inflated stat goes on D; per-exp normalisation stays in U.
                    coeffs_d, chi2_d, dof_d, k_d = _weighted_ridge_fit_gls(
                        mu, y, sigma_eff, gls_indep_per_exp, gls_exp_index, d,
                        sigma_sys_dep_per_row=gls_dep_per_row,
                        ridge_lambda=ridge_lambda,
                        ridge_power=ridge_power,
                        df_method=df_method,
                        external_weights=external_weights,
                    )
                else:
                    coeffs_d, chi2_d, dof_d, k_d = _weighted_ridge_fit(
                        mu, y, sigma_for_fit_post_tau, d,
                        ridge_lambda=ridge_lambda,
                        ridge_power=ridge_power,
                        df_method=df_method,
                        external_weights=external_weights,
                    )
                score = _criterion_score(
                    chi2_d, n=n_aicc, k=k_d, criterion=select_degree,
                )
                new_all[d] = {
                    'coeffs': coeffs_d.copy(),
                    'chi2': chi2_d,
                    'dof': dof_d,
                    'eff_params': k_d,
                    'aicc': score,
                }
                if new_best is None or score < new_best:
                    new_best = score
                    new_best_res = (d, coeffs_d, chi2_d, dof_d, k_d)
            assert new_best_res is not None
            all_degrees_info = new_all  # post-τ becomes the canonical record
            new_winner = new_best_res[0]
            if new_winner != degree_use:
                # Refit at the new winner under τ-IRLS so coeffs and τ stay
                # mutually consistent. Reuse the same iteration limits.
                post_tau_winner_changed = True
                degree_use, coeffs0, chi2_0, dof_0, k_0 = new_best_res
                tau_used_prev = None
                tau_target_prev = None
                for _ in range(max(1, int(tau_irls_max_iters))):
                    y_fit = legval(mu, coeffs0)
                    sigma_eff, tau_info = compute_angular_band_discrepancy(
                        mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                        min_points_per_band=min_points_per_band,
                        max_band_scale=max_band_scale,
                        method=band_scale_method,
                        sigma_sys=sigma_sys,
                    )
                    if damping > 0 and tau_used_prev is not None:
                        for b in ('tau_F', 'tau_M', 'tau_B'):
                            t_target = float(tau_info[b])
                            t_old = tau_used_prev[b]
                            t_damped = (
                                (t_target ** (1.0 - damping)) * (t_old ** damping)
                            )
                            t_damped = max(1.0, min(t_damped, max_band_scale))
                            tau_info[b] = t_damped
                        sigma_eff = sigma.copy()
                        for bn, bm in band_masks_irls.items():
                            s_b = float(tau_info[f'tau_{bn}'])
                            if s_b > 1.0:
                                sigma_eff[bm] = sigma[bm] * s_b
                    sigma_refit = (
                        np.sqrt(sigma_eff ** 2 + sigma_sys ** 2)
                        if sigma_sys is not None else sigma_eff
                    )
                    if (tau_refit_use_gls and use_gls_kernel
                            and gls_indep_per_exp is not None):
                        # Same contract as the first τ-IRLS loop above.
                        coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit_gls(
                            mu, y, sigma_eff, gls_indep_per_exp,
                            gls_exp_index, degree_use,
                            sigma_sys_dep_per_row=gls_dep_per_row,
                            ridge_lambda=ridge_lambda,
                            ridge_power=ridge_power,
                            df_method=df_method,
                            external_weights=external_weights,
                        )
                    else:
                        coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit(
                            mu, y, sigma_refit, degree_use,
                            ridge_lambda=ridge_lambda,
                            ridge_power=ridge_power,
                            df_method=df_method,
                            external_weights=external_weights,
                        )
                    tau_target_curr = {
                        b: float(tau_info.get(f'raw_{b[-1]}', tau_info[b]))
                        for b in ('tau_F', 'tau_M', 'tau_B')
                    }
                    if tau_target_prev is not None:
                        dtau = max(
                            abs(tau_target_curr[b] - tau_target_prev[b])
                            for b in ('tau_F', 'tau_M', 'tau_B')
                        )
                        if dtau < tau_irls_tol:
                            break
                    tau_used_prev = {
                        b: float(tau_info[b]) for b in ('tau_F', 'tau_M', 'tau_B')
                    }
                    tau_target_prev = tau_target_curr
                # Final τ on the new-winner coeffs.
                y_fit = legval(mu, coeffs0)
                sigma_eff, tau_info = compute_angular_band_discrepancy(
                    mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                    min_points_per_band=min_points_per_band,
                    max_band_scale=max_band_scale,
                    method=band_scale_method,
                    sigma_sys=sigma_sys,
                )
                chi2_red = float(chi2_0 / max(1e-12, dof_0))
    elif rescale_unc_by_chi2:
        # Global Birge scaling
        scale = float(np.sqrt(chi2_red))
        if not allow_shrink_unc:
            scale = max(1.0, scale)
        sigma_eff = sigma * scale
    else:
        scale = 1.0
        sigma_eff = sigma.copy()

    rng = np.random.default_rng(random_state)

    # Handle freeze_c0 mode (Improvement 1.2)
    c0_fix = None
    if freeze_c0:
        c0_fix = fixed_c0_value if fixed_c0_value is not None else float(coeffs0[0])
        # Refit nominal with fixed_c0 to get consistent c1..cL
        coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit(
            mu, y, sigma_for_fit, degree_use,
            ridge_lambda=ridge_lambda,
            ridge_power=ridge_power,
            df_method=df_method,
            external_weights=external_weights,
            fixed_c0=c0_fix,
        )
        chi2_red = float(chi2_0 / max(1e-12, dof_0))

    # Build fixed_coeffs dict for freezing higher-order coefficients during MC
    fixed_high: Optional[Dict[int, float]] = None
    if max_sample_order is not None and max_sample_order < degree_use:
        fixed_high = {
            l: float(coeffs0[l])
            for l in range(max_sample_order + 1, degree_use + 1)
        }

    # Build group mapping for correlated normalization (Improvement 1.3).
    # Groups are needed whenever the global sigma_norm is active OR the
    # manifest supplies per-experiment sigma_sys_relative values; gating on
    # sigma_norm alone would silently drop all manifest-driven normalization
    # perturbations when sigma_norm=0.
    _has_manifest_sys = (
        'sigma_sys_relative' in work.columns
        and bool(np.any(work['sigma_sys_relative'].to_numpy(dtype=float) > 0.0))
    )
    group_indices: Dict[Tuple, List[int]] = {}
    group_keys: List[Tuple] = []
    if sigma_norm > 0.0 or _has_manifest_sys:
        # Check if required columns exist
        available_cols = [col for col in norm_group_cols if col in work.columns]
        if available_cols:
            from collections import defaultdict
            group_indices = defaultdict(list)
            for i in range(n):
                key = tuple(work[col].iloc[i] for col in available_cols)
                group_indices[key].append(i)
            group_keys = list(group_indices.keys())

    # Sampling fits
    # Perturbed data + σ captured for the optional fixed-shape c0 recording
    # (Phase-2 magnitude channel); stays None on the default path.
    c0_record_Y = None
    c0_record_sigma = None
    if n_samples <= 1 and not stochastic:
        coef_mat = coeffs0[np.newaxis, :]
        if record_c0_scale:
            c0_record_Y = y[np.newaxis, :]
            c0_record_sigma = sigma_for_fit
    else:
        n_draws = max(1, int(n_samples))

        # Generate all perturbed y vectors at once
        Y_perturbed = np.tile(y, (n_draws, 1))  # (n_draws, n)

        # Global common-mode normalization factor: one draw per sample, applied to all points
        # (models the shared monitor / reference XS uncertainty).
        if sigma_norm_common_mode > 0.0:
            if norm_dist == "lognormal":
                N_common_mode = rng.lognormal(
                    mean=-0.5 * sigma_norm_common_mode ** 2,
                    sigma=sigma_norm_common_mode,
                    size=n_draws,
                )
            else:
                N_common_mode = 1.0 + rng.normal(0.0, sigma_norm_common_mode, size=n_draws)
            Y_perturbed *= N_common_mode[:, np.newaxis]

        # Apply correlated normalization uncertainty per experiment (Improvement 1.3).
        # Each group's sigma is taken from the manifest-derived
        # 'sigma_sys_relative' column on the data, falling back to the global
        # sigma_norm when the column is absent or zero.
        if group_keys:
            has_sys_col = 'sigma_sys_relative' in work.columns
            for key in group_keys:
                indices = group_indices[key]
                if has_sys_col and indices:
                    ex_sigma_sys = float(work['sigma_sys_relative'].iloc[indices[0]])
                else:
                    ex_sigma_sys = 0.0
                # Per-experiment scalar normalization sigma — kept as a local
                # variable name to avoid shadowing the per-point ``sigma_eff``
                # array that is used for additive noise + fit weights below.
                ex_norm_sigma = ex_sigma_sys if ex_sigma_sys > 0 else sigma_norm
                if ex_norm_sigma <= 0.0:
                    continue
                if norm_dist == "lognormal":
                    N_g = rng.lognormal(
                        mean=-0.5 * ex_norm_sigma ** 2,
                        sigma=ex_norm_sigma,
                        size=n_draws,
                    )
                else:  # "normal"
                    N_g = 1.0 + rng.normal(0.0, ex_norm_sigma, size=n_draws)
                Y_perturbed[:, indices] *= N_g[:, np.newaxis]

        # Add pointwise noise. Stat-only by design: σ_sys is already in the
        # per-experiment N_g multiplicative factor above; using σ_total_eff
        # here would double-count sys.
        Y_perturbed += rng.normal(loc=0.0, scale=sigma_eff, size=(n_draws, n))

        # MC fit weights match Level-2 nominal: 1/σ_total_eff² with
        # σ_total_eff² = (τ·σ_stat)² + σ_sys². The marginal point variance
        # of Y_perturbed is σ_total_eff² (additive contributes σ_eff²;
        # multiplicative N_g contributes ≈ y²·σ_indep² ≈ σ_sys²), so this
        # weighting matches the noise model the fit is observing.
        sigma_for_mc_fit = (
            np.sqrt(sigma_eff ** 2 + sigma_sys ** 2)
            if sigma_sys is not None else sigma_eff
        )

        # Batch solve: factorize M once, solve all RHS simultaneously
        coef_mat = _batch_mc_ridge_solve(
            mu, Y_perturbed, sigma_for_mc_fit, degree_use,
            ridge_lambda=ridge_lambda,
            ridge_power=ridge_power,
            external_weights=external_weights,
            fixed_c0=c0_fix,
            fixed_coeffs=fixed_high,
        )
        if record_c0_scale:
            c0_record_Y = Y_perturbed
            c0_record_sigma = sigma_for_mc_fit
    coef_df = pd.DataFrame(coef_mat, columns=[f"c{l}" for l in range(degree_use + 1)])

    # Optional fixed-shape c0 (elastic magnitude) recording. Projects each
    # perturbed sample onto the frozen nominal shape using the same weights the
    # fit used; read-only, so ``coef_df`` above is untouched.
    c0_samples = None
    if record_c0_scale and c0_record_Y is not None:
        ref_coeffs = (
            coeffs0 if c0_scale_ref_coeffs is None
            else np.asarray(c0_scale_ref_coeffs, dtype=float)
        )
        _, c0_samples = fixed_shape_c0_scale(
            c0_record_Y, mu, ref_coeffs,
            external_weights=external_weights,
            sigma_for_fit=c0_record_sigma,
        )
        c0_samples = np.atleast_1d(c0_samples)

    info = dict(
        n_points=n,
        degree=degree_use,
        max_degree=max_degree,
        ridge_lambda=ridge_lambda,
        ridge_power=ridge_power,
        df_method=df_method,
        chi2=chi2_0,
        dof=dof_0,
        chi2_red=chi2_red,
        scale_factor=scale,
        eff_params=k_0,
        sampled=n_samples > 1 or stochastic,
        tau_info=tau_info,
        use_band_discrepancy=use_band_discrepancy,
        # Model averaging info
        all_degrees_info=all_degrees_info if all_degrees_info else None,
        # New info fields
        freeze_c0=freeze_c0,
        fixed_c0_value=c0_fix,
        sigma_norm=sigma_norm,
        n_experiments_for_norm=len(group_keys),
        max_sample_order=max_sample_order,
        n_frozen_high=len(fixed_high) if fixed_high else 0,
        # Phase B audit follow-up: keep the pre-τ AICc snapshot (when the
        # post-τ rescan ran) so callers can log how much the model-degree
        # weights moved after τ. ``post_tau_winner_changed`` is True iff
        # the rescan picked a different degree and triggered a refit.
        all_degrees_info_pre_tau=all_degrees_info_pre_tau,
        post_tau_winner_changed=post_tau_winner_changed,
    )
    # Magnitude side channel: only present when explicitly requested so the
    # default info dict is byte-for-byte what every existing caller sees.
    if record_c0_scale:
        info["c0_samples"] = c0_samples
    return coef_df, info


def endf_normalize_legendre_coeffs(
    c: np.ndarray,
    *,
    include_a0: bool = False,
    require_positive_c0: bool = False,
) -> np.ndarray:
    """
    Convert coefficients c_l of y(mu)=sum c_l P_l(mu) into ENDF MF=4 style a_l:
      a0 = 1
      a_l = (c_l / c0) / (2l+1) for l>=1

    Returns a array [a0, a1, ..., aL] if include_a0 else [a1..aL].
    """
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size < 1:
        raise ValueError("c must be a 1D array with at least one element (c0).")

    c0 = float(c[0])
    if require_positive_c0 and not (c0 > 0.0):
        raise ValueError(f"c0 must be > 0 to normalize (got c0={c0}).")

    L = c.size - 1
    a = np.zeros_like(c)
    a[0] = 1.0
    for l in range(1, L + 1):
        a[l] = (c[l] / c0) / (2.0 * l + 1.0)

    return a if include_a0 else a[1:]


def evaluate_legendre_series(mu: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Evaluate y(mu)=sum c_l P_l(mu) using numpy's Legendre evaluator.
    """
    mu = np.asarray(mu, dtype=float)
    c = np.asarray(c, dtype=float)
    return legval(mu, c)


def fixed_shape_c0_scale(
    Y: np.ndarray,
    mu: np.ndarray,
    nominal_coeffs: np.ndarray,
    external_weights: Optional[np.ndarray] = None,
    sigma_for_fit: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form "fit c0 with the shape frozen" magnitude scale.

    The elastic magnitude channel (MF3 MT2 / MF33 MT2) of the EXFOR pipeline.
    With the angular *shape* held at the nominal bin curve
    ``y_nom(mu) = sum_l c_l^nom P_l(mu)``, the only free parameter of a
    perturbed dataset ``Y`` is a single multiplicative scale ``s``.  The
    weighted-least-squares estimate of that scale is the closed form::

        s = sum(w * Y * y_nom) / sum(w * y_nom**2)          c0 = s * c0_nom

    with the *same* fit weights the shape refit uses, ``w = external_weights /
    sigma_for_fit**2`` (``external_weights`` = the kernel/ESS weights ``g``;
    ``sigma_for_fit`` = the per-point σ the WLS observes, i.e. τ-inflated σ_stat
    optionally combined in quadrature with σ_sys).  ``c0_nom = nominal_coeffs[0]``
    and ``sigma_el = 4*pi*c0``.

    Because this only *reads* the perturbed data ``Y`` and never re-solves the
    shape fit, recording it alongside the frozen-c0 MF34 refits leaves MF4 and
    MF34 bit-identical.

    Parameters
    ----------
    Y : np.ndarray
        Perturbed data values (barns/sr), either 1-D ``(n_points,)`` for a
        single sample or 2-D ``(n_draws, n_points)`` for a batch of samples.
    mu : np.ndarray
        ``cos(theta)`` of each point, shape ``(n_points,)``.
    nominal_coeffs : np.ndarray
        Nominal Legendre coefficients ``[c0, c1, ...]`` defining the frozen
        shape; ``nominal_coeffs[0]`` is ``c0_nom``.
    external_weights : np.ndarray, optional
        Per-point kernel/ESS weights ``g`` (shape ``(n_points,)``).  Defaults to
        ones (unweighted) when omitted.
    sigma_for_fit : np.ndarray, optional
        Per-point σ the WLS uses (shape ``(n_points,)``).  Defaults to ones
        (plain kernel-weighted) when omitted; points with non-finite or
        non-positive σ are dropped.

    Returns
    -------
    (s, c0) : Tuple[np.ndarray, np.ndarray]
        The scale ``s`` and magnitude ``c0 = s * c0_nom``.  Both are 0-D arrays
        for 1-D ``Y`` and shape ``(n_draws,)`` for 2-D ``Y``.
    """
    Y = np.asarray(Y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nominal_coeffs = np.asarray(nominal_coeffs, dtype=float)

    y_nom = legval(mu, nominal_coeffs)

    g = (
        np.ones_like(mu)
        if external_weights is None
        else np.asarray(external_weights, dtype=float)
    )
    if sigma_for_fit is None:
        w = g
    else:
        sigma = np.asarray(sigma_for_fit, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_var = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma ** 2, 0.0)
        w = g * inv_var

    denom = float(np.sum(w * y_nom ** 2))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(
            "fixed_shape_c0_scale: sum(w * y_nom**2) is non-positive; the "
            "nominal curve is zero over all weighted points."
        )

    # numer = sum_j w_j Y_j y_nom_j  — matmul broadcasts over the batch axis
    # of a 2-D Y and reduces a 1-D Y to a scalar.
    numer = Y @ (w * y_nom)
    s = np.asarray(numer / denom)
    c0 = s * float(nominal_coeffs[0])
    return s, c0


def combine_c0_covariance(
    c0_pass1: np.ndarray,
    c0_pass2: np.ndarray,
    c0_nominal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Congruence-combine the two-pass fixed-shape c0 samples into an MF33 block.

    Mirrors the Legendre-vector combine used for MF34
    (``cov = corr_pass1 * outer(std_pass2, std_pass2)``): cross-bin correlations
    come from the Pass-1 shared-draw samples, marginal variances from the
    Pass-2 independent-per-bin samples.  The result is the elastic **magnitude**
    covariance — one square block over energy bins.

    Parameters
    ----------
    c0_pass1 : np.ndarray
        Pass-1 c0 samples, shape ``(n_samples_1, n_bins)`` (shared draws →
        correlations).  Missing entries may be NaN; correlations use the
        pairwise-complete observations.
    c0_pass2 : np.ndarray
        Pass-2 c0 samples, shape ``(n_samples_2, n_bins)`` (independent per-bin
        draws → variances).  NaNs are ignored per column.
    c0_nominal : np.ndarray
        Nominal magnitude ``c0_nom`` per bin, shape ``(n_bins,)``.

    Returns
    -------
    (rel_cov, cov_abs) : Tuple[np.ndarray, np.ndarray]
        ``rel_cov`` is the **relative** covariance
        ``Cov(c0_i, c0_j) / (c0_nom_i * c0_nom_j)`` (the MF33 writer input);
        ``cov_abs`` is the absolute covariance in c0 units.  Both are
        ``(n_bins, n_bins)`` symmetric.  Bins with a non-positive nominal get a
        zero relative row/column.
    """
    c0_pass1 = np.asarray(c0_pass1, dtype=float)
    c0_pass2 = np.asarray(c0_pass2, dtype=float)
    c0_nominal = np.asarray(c0_nominal, dtype=float)
    n_bins = c0_nominal.size

    # Pass-1 correlations (pairwise-complete over shared-draw samples).
    p1 = np.ma.masked_invalid(c0_pass1)
    corr = np.ma.corrcoef(p1, rowvar=False)
    corr = np.ma.filled(np.ma.asarray(corr), 0.0)
    corr = np.atleast_2d(corr)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Pass-2 marginal std per bin.
    std = np.nanstd(c0_pass2, axis=0, ddof=1)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)

    cov_abs = corr * np.outer(std, std)
    np.fill_diagonal(cov_abs, std ** 2)
    cov_abs = 0.5 * (cov_abs + cov_abs.T)

    denom = np.outer(c0_nominal, c0_nominal)
    rel_cov = np.divide(
        cov_abs, denom, out=np.zeros((n_bins, n_bins)), where=denom > 0.0,
    )
    rel_cov = 0.5 * (rel_cov + rel_cov.T)
    return rel_cov, cov_abs


def plot_sampled_angular_distributions(
    coef_df: pd.DataFrame,
    exfor_df: Optional[pd.DataFrame] = None,
    n_plot: int = 10,
    analysis_energy: float = None,
    random_state: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 8),
    show_nominal: bool = True,
    yscale: Literal['linear', 'log'] = 'linear',
    xaxis: Literal['theta', 'mu'] = 'mu',
    library_data: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Plot sampled angular distributions with EXFOR experimental data.

    Parameters
    ----------
    coef_df : pd.DataFrame
        DataFrame with Legendre coefficients (columns c0, c1, c2, ...)
        from sample_legendre_coefficients()
    exfor_df : pd.DataFrame, optional
        EXFOR data with columns theta_deg, value, unc, mu
    n_plot : int, optional
        Number of sample curves to plot (default: 10)
    analysis_energy : float, optional
        Energy in MeV for plot title
    random_state : int, optional
        Random seed for selecting samples to plot
    figsize : tuple, optional
        Figure size (default: (12, 8))
    show_nominal : bool, optional
        Whether to highlight the nominal (first) fit (default: True)
    yscale : str, optional
        Y-axis scale: 'linear' or 'log' (default: 'linear')
    xaxis : str, optional
        X-axis variable: 'theta' (degrees) or 'mu' (cos(theta)) (default: 'mu')
    library_data : dict, optional
        Dictionary of library data to overlay. Keys are library names,
        values are either:
        - Full results dict from perform_systematic_analysis (with uncertainties)
        - Dict with 'mu', 'baseline', 'uncertainties'=False (baseline only)

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Evaluation grid (1° resolution)
    mu_grid = np.linspace(-1, 1, 181)
    theta_grid = np.rad2deg(np.arccos(mu_grid))

    # Choose x-axis data
    if xaxis == 'mu':
        x_grid = mu_grid
    else:
        x_grid = theta_grid

    # Extract coefficient matrix
    coef_cols = [c for c in coef_df.columns if c.startswith('c')]
    coef_mat = coef_df[coef_cols].to_numpy()
    n_samples = len(coef_mat)

    # Select samples to plot
    rng = np.random.default_rng(random_state)
    if n_samples <= n_plot:
        sample_indices = list(range(n_samples))
    else:
        sample_indices = sorted(rng.choice(n_samples, size=n_plot, replace=False))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot sampled curves
    for idx in sample_indices:
        if idx == 0 and show_nominal:
            continue  # Plot nominal last for visibility
        y_sample = evaluate_legendre_series(mu_grid, coef_mat[idx])
        ax.plot(x_grid, y_sample, 'C0-', alpha=0.3, linewidth=1.0, zorder=2)

    # Plot nominal fit
    if show_nominal and n_samples > 0:
        y_nominal = evaluate_legendre_series(mu_grid, coef_mat[0])
        ax.plot(x_grid, y_nominal, 'C0-', linewidth=2.5,
                label='Nominal fit', zorder=3)

    # Plot EXFOR data
    if exfor_df is not None and len(exfor_df) > 0:
        # Choose x-coordinate for experimental data
        if xaxis == 'mu':
            x_exp = exfor_df['mu']
        else:
            x_exp = exfor_df['theta_deg']

        # Group by experiment
        for (entry, subentry), group in exfor_df.groupby(['entry', 'subentry']):
            author = group['author'].iloc[0]
            year = group['year'].iloc[0]
            reaction = group['reaction'].iloc[0] if 'reaction' in group.columns else ''

            # Check if this is a natural iron experiment
            is_natural = '26-FE-0' in reaction or 'FE-0' in reaction

            # Build label with (natural) suffix if applicable
            label = f"{author} ({year})"
            if is_natural:
                label += " (natural)"

            if xaxis == 'mu':
                x_exp_group = group['mu']
            else:
                x_exp_group = group['theta_deg']

            ax.errorbar(
                x_exp_group, group['value'], yerr=group['unc'],
                fmt='o', markersize=4, capsize=3, capthick=1.5,
                label=label, zorder=4
            )

    # Plot library data
    if library_data is not None and len(library_data) > 0:
        library_colors = {'JEFF-4.0': 'C1', 'JENDL-5': 'C2', 'TENDL-2023': 'C3'}

        for lib_name, lib_result in library_data.items():
            color = library_colors.get(lib_name, 'gray')

            if not isinstance(lib_result, dict):
                continue

            # Check if this is baseline-only (has 'uncertainties': False flag)
            is_baseline_only = lib_result.get('uncertainties') is False

            if is_baseline_only:
                # Baseline only (no uncertainties)
                mu_lib = lib_result.get('mu')
                dsig_lib = lib_result.get('baseline')

                if mu_lib is None or dsig_lib is None:
                    continue

                if xaxis == 'mu':
                    x_lib = mu_lib
                else:
                    x_lib = np.rad2deg(np.arccos(mu_lib))

                ax.plot(x_lib, dsig_lib, color=color, linewidth=2.0,
                        linestyle='--', label=f'{lib_name}', zorder=5)
            else:
                # Full results from perform_systematic_analysis
                # Has 'combined_lower' and 'combined_upper' for uncertainty bands
                mu_lib = lib_result.get('mu')
                dsig_baseline = lib_result.get('baseline')
                dsig_combined_lower = lib_result.get('combined_lower')
                dsig_combined_upper = lib_result.get('combined_upper')

                if mu_lib is None or dsig_baseline is None:
                    continue

                if xaxis == 'mu':
                    x_lib = mu_lib
                else:
                    x_lib = np.rad2deg(np.arccos(mu_lib))

                # Determine label based on whether uncertainties are available
                has_uncertainties = dsig_combined_lower is not None and dsig_combined_upper is not None
                label = f'{lib_name} ±1σ' if has_uncertainties else f'{lib_name}'

                # Plot baseline
                ax.plot(x_lib, dsig_baseline, color=color, linewidth=2.0,
                        linestyle='--', label=label, zorder=5)

                # Plot uncertainty band if available (no label to avoid duplicate legend entry)
                if has_uncertainties:
                    ax.fill_between(x_lib, dsig_combined_lower, dsig_combined_upper,
                                    color=color, alpha=0.2, zorder=1)

    # Formatting
    if xaxis == 'mu':
        ax.set_xlabel('cos(θ)', fontsize=12)
        ax.set_xlim(-1, 1)
    else:
        ax.set_xlabel('Scattering Angle θ (degrees)', fontsize=12)
        ax.set_xlim(0, 180)

    ax.set_ylabel('dσ/dΩ (b/sr)', fontsize=12)

    title = 'Sampled Angular Distributions'
    if analysis_energy is not None:
        title += f' at E = {analysis_energy:.3f} MeV'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add info text
    degree = len(coef_cols) - 1
    info_text = f'Legendre order: L={degree}\n'
    if n_samples > 1:
        info_text += f'Samples shown: {len(sample_indices)}/{n_samples}'
    else:
        info_text += 'Single fit (no sampling)'

    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    return fig


# =============================================================================
# POSITIVITY-CONSTRAINED PROJECTION FOR ANGULAR DISTRIBUTIONS
# =============================================================================

def check_angular_distribution_positivity(
    coeffs: np.ndarray,
    n_points: int = 50,
) -> bool:
    """
    Check whether a Legendre expansion produces non-negative σ(θ) everywhere.

    Evaluates σ(μ) = Σ_l c_l P_l(μ) at n_points evenly spaced in [-1, 1].

    Parameters
    ----------
    coeffs : np.ndarray
        Legendre coefficients c_0, c_1, ..., c_L (raw, not ENDF-normalized).
    n_points : int
        Number of check points in [-1, 1].

    Returns
    -------
    bool
        True if σ(μ) >= 0 for all check points, False otherwise.
    """
    mu_grid = np.linspace(-1, 1, n_points)
    sigma = legval(mu_grid, coeffs)
    return bool(np.all(sigma >= 0))


def project_to_positive_distribution(
    coeffs: np.ndarray,
    n_points: int = 50,
    frozen_indices: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Project Legendre coefficients to the nearest set producing non-negative σ(θ).

    Solves: minimize ||c* - c||² subject to Σ c*_l P_l(μ_j) >= 0 for all j.

    Uses SLSQP constrained optimization. Returns original coefficients unchanged
    if the distribution is already non-negative everywhere.

    Parameters
    ----------
    coeffs : np.ndarray
        Legendre coefficients c_0, c_1, ..., c_L.
    n_points : int
        Number of constraint points in [-1, 1].
    frozen_indices : dict, optional
        Mapping of coefficient index → fixed value. These coefficients are
        pinned during optimization (not allowed to change).

    Returns
    -------
    np.ndarray
        Projected coefficients (same shape as input).
    """
    from scipy.optimize import minimize

    mu_grid = np.linspace(-1, 1, n_points)
    sigma = legval(mu_grid, coeffs)

    # Already non-negative — return as-is
    if np.all(sigma >= 0):
        return coeffs

    n_coeffs = len(coeffs)

    # Build Legendre Vandermonde-like matrix: P[j, l] = P_l(μ_j)
    # legvander returns shape (n_points, n_coeffs) with columns P_0, P_1, ...
    P_matrix = legvander(mu_grid, n_coeffs - 1)

    def objective(c):
        diff = c - coeffs
        return 0.5 * np.dot(diff, diff)

    def grad(c):
        return c - coeffs

    # Constraints: P_matrix @ c >= 0 for all rows
    constraints = {
        'type': 'ineq',
        'fun': lambda c: P_matrix @ c,  # >= 0
        'jac': lambda c: P_matrix,
    }

    # Pin frozen coefficients via bounds: (val, val) for frozen, (None, None) for free
    bounds = None
    if frozen_indices:
        bounds = [(None, None)] * n_coeffs
        for idx, val in frozen_indices.items():
            bounds[idx] = (val, val)

    result = minimize(
        objective,
        x0=coeffs.copy(),
        jac=grad,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-12, 'maxiter': 500},
    )

    return result.x
