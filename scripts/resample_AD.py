from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any, Sequence, List, Union
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legvander, legval
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    max_tau_fraction: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Estimate per-band discrepancy τ_b and return effective uncertainties.

    This replaces global Birge scaling with angular-band specific
    uncertainty inflation. The bands are:
      - Forward:  μ > 0.5  (θ < 60°)
      - Mid:      |μ| ≤ 0.5 (60° ≤ θ ≤ 120°)
      - Backward: μ < -0.5 (θ > 120°)

    For each band b:
      1. Compute normalized residuals: r_i = (y_i - y_fit_i) / σ_i
      2. Compute robust scale: s_b = MAD-based estimate
      3. If s_b > 1: τ_b = median(σ_b) * sqrt(s_b² - 1)
      4. Apply ceiling: τ_b = min(τ_b, max_tau_fraction * median(y_b))

    The effective uncertainty is: σ²_i,eff = σ²_i + τ²_b

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
        Minimum points required to estimate τ for a band.
        If fewer, use the mid-band τ value.
    max_tau_fraction : float
        Cap τ_b at this fraction of median cross section in band.

    Returns
    -------
    sigma_eff : np.ndarray
        Effective uncertainties with band discrepancy added in quadrature
    tau_info : Dict[str, float]
        Dictionary with τ_F, τ_M, τ_B values for each band
    """
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

    tau_values = {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0}

    # First pass: estimate τ for bands with enough points
    for band_name, mask in bands.items():
        n_band = np.sum(mask)

        if n_band < min_points_per_band:
            continue

        # Normalized residuals in this band
        r_band = (y[mask] - y_fit[mask]) / sigma[mask]

        # Robust scale estimate
        s_band = robust_residual_scale(r_band)

        if s_band <= 1.0:
            tau_b = 0.0
        else:
            # τ_b = median(σ) * sqrt(s² - 1)
            median_sigma = np.median(sigma[mask])
            tau_b = median_sigma * np.sqrt(s_band**2 - 1)

        # Apply ceiling
        median_y = np.median(np.abs(y[mask]))
        tau_ceiling = max_tau_fraction * median_y
        tau_b = min(tau_b, tau_ceiling)

        tau_values[f'tau_{band_name}'] = tau_b

    # Second pass: for bands with too few points, use mid-band τ
    tau_mid = tau_values['tau_M']
    for band_name, mask in bands.items():
        n_band = np.sum(mask)
        if n_band < min_points_per_band and n_band > 0:
            tau_values[f'tau_{band_name}'] = tau_mid

    # Apply τ to get effective uncertainties
    for band_name, mask in bands.items():
        tau_b = tau_values[f'tau_{band_name}']
        if tau_b > 0:
            sigma_eff[mask] = np.sqrt(sigma[mask]**2 + tau_b**2)

    return sigma_eff, tau_values


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
    min_experiments: int = 2,
    percentile: float = 50.0,
) -> Dict[str, float]:
    """
    Compute a per-band tau baseline from multi-experiment bins and enforce it
    as a floor on single-experiment bins (partial pooling).

    For bins with fewer than `min_experiments` experiments, the tau values are
    raised to at least the baseline computed from well-estimated bins.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results (modified in-place).
    min_experiments : int
        Minimum number of experiments for a bin to be considered well-estimated.
    percentile : float
        Percentile of well-estimated tau values to use as baseline (e.g. 50 = median).

    Returns
    -------
    Dict[str, float]
        Baseline tau values per band {'tau_F': ..., 'tau_M': ..., 'tau_B': ...}.
    """
    bands = ['tau_F', 'tau_M', 'tau_B']
    baselines: Dict[str, float] = {b: 0.0 for b in bands}

    # Step 1: Collect tau values from well-estimated bins
    well_estimated: Dict[str, List[float]] = {b: [] for b in bands}
    for r in nominal_results:
        if not r.has_data or r.interpolated:
            continue
        n_exp = len(r.experiments_info)
        if n_exp >= min_experiments:
            for b in bands:
                val = r.tau_info.get(b, 0.0)
                well_estimated[b].append(val)

    # Step 2: Compute baseline per band (need >= 3 well-estimated bins)
    for b in bands:
        vals = well_estimated[b]
        if len(vals) >= 3:
            baselines[b] = float(np.percentile(vals, percentile))

    # Step 3: Apply floor to under-estimated bins
    for r in nominal_results:
        if not r.has_data or r.interpolated:
            continue
        n_exp = len(r.experiments_info)
        if n_exp < min_experiments:
            updated = dict(r.tau_info)
            for b in bands:
                updated[b] = max(updated.get(b, 0.0), baselines[b])
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

    # Combined weights: kernel weight / variance
    w = kernel_weights / (sigma_eff ** 2)

    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)

    if sum_w2 < 1e-30:
        return 0.0

    return (sum_w ** 2) / sum_w2


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
) -> float:
    """
    Compute energy resolution σE from TOF parameters.

    For neutrons measured by time-of-flight:
      E = m_n * L² / (2 * t²)
      δE/E = 2 * δt/t
      t = L / v = L * sqrt(m_n / (2E))

    Therefore:
      σE = E * 2 * δt / t
         = E * 2 * δt * v / L
         = E * 2 * δt * sqrt(2E/m_n) / L

    Parameters
    ----------
    E_mev : float
        Neutron energy in MeV
    delta_t_ns : float
        Time resolution in nanoseconds (default: 10 ns)
    flight_path_m : float
        Flight path length in meters (default: 27.037 m)

    Returns
    -------
    float
        Energy resolution σE in MeV
    """
    if E_mev <= 0:
        return 0.0

    # Physical constants
    m_n_kg = 1.674927e-27       # Neutron mass in kg
    MeV_to_J = 1.602176634e-13  # MeV to Joules

    E_J = E_mev * MeV_to_J      # Energy in Joules
    delta_t_s = delta_t_ns * 1e-9  # Time resolution in seconds

    # Velocity: v = sqrt(2E/m)
    v = np.sqrt(2 * E_J / m_n_kg)  # m/s

    # Time of flight: t = L/v
    t = flight_path_m / v  # seconds

    # Energy resolution: σE/E = 2 * δt/t
    sigma_E_rel = 2 * delta_t_s / t
    sigma_E_mev = E_mev * sigma_E_rel

    return sigma_E_mev


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
            chi2 = float(np.sum(((y - yhat) / sigma) ** 2))
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
        chi2 = float(np.sum(((y - yhat) / sigma) ** 2))

        # Degrees of freedom (only free parameters count)
        if df_method == "naive" or ridge_lambda <= 0.0:
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

    # Compute chi2 on original scale
    yhat = A @ coeffs
    chi2 = float(np.sum(((y - yhat) / sigma) ** 2))

    # Degrees of freedom
    if df_method == "naive" or ridge_lambda <= 0.0:
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


def _criterion_score(
    chi2: float,
    n: int,
    k: float,
    criterion: Literal["aicc", "bic"]
) -> float:
    """
    Model selection based on chi2. This is an approximate AICc/BIC style score.
    Lower is better.
    """
    if criterion == "bic":
        return chi2 + k * np.log(max(1, n))
    # AICc
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
    select_degree: Optional[Literal["aicc", "bic"]] = None,
    # ridge
    ridge_lambda: float = 0.0,
    ridge_power: int = 4,
    df_method: Literal["naive", "hat"] = "hat",
    # external weights (for Gaussian kernel)
    external_weights: Optional[np.ndarray] = None,
    # sampling
    n_samples: int = 1,
    stochastic: bool = False,
    rescale_unc_by_chi2: bool = True,
    allow_shrink_unc: bool = False,
    random_state: Optional[int] = None,
    # angular-band discrepancy model
    use_band_discrepancy: bool = False,
    min_points_per_band: int = 3,
    max_tau_fraction: float = 0.25,
    # fixed-c0 mode (Improvement 1.2)
    freeze_c0: bool = False,
    fixed_c0_value: Optional[float] = None,
    # correlated normalization uncertainty (Improvement 1.3)
    sigma_norm: float = 0.0,
    norm_group_cols: Tuple[str, ...] = ("entry", "subentry"),
    norm_dist: Literal["lognormal", "normal"] = "lognormal",
    # freeze higher-order coefficients during MC sampling
    max_sample_order: Optional[int] = None,
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
        Criterion for order selection ("aicc" or "bic")
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
    max_tau_fraction : float
        Cap τ_b at this fraction of cross section
    freeze_c0 : bool
        If True, fix c0 to either fixed_c0_value (if provided) or the nominal fit c0.
        This enables shape-only refits where MF3 is fixed and MF34 explains discrepancies.
    fixed_c0_value : float, optional
        Explicit value to fix c0 at. Only used if freeze_c0=True.
    sigma_norm : float
        Per-experiment normalization uncertainty (e.g., 0.05 for 5%).
        Applies correlated multiplicative noise per experiment group in MC sampling.
    norm_group_cols : Tuple[str, ...]
        Column names used to group data by experiment (default: ("entry", "subentry"))
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

    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 points to fit anything meaningful.")

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
            # Scan degrees 0..max_feasible and pick best score
            best = None
            best_res = None
            for d in range(0, max_feasible + 1):
                coeffs_d, chi2_d, dof_d, k_d = _weighted_ridge_fit(
                    mu, y, sigma, d,
                    ridge_lambda=ridge_lambda,
                    ridge_power=ridge_power,
                    df_method=df_method,
                    external_weights=external_weights,
                )
                score = _criterion_score(chi2_d, n=n, k=k_d, criterion=select_degree)

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
        coeffs0, chi2_0, dof_0, k_0 = _weighted_ridge_fit(
            mu, y, sigma, degree_use,
            ridge_lambda=ridge_lambda,
            ridge_power=ridge_power,
            df_method=df_method,
            external_weights=external_weights,
        )

    chi2_red = float(chi2_0 / max(1e-12, dof_0))

    # Compute effective uncertainties
    tau_info = {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0}

    if use_band_discrepancy:
        # Angular-band discrepancy model: compute τ_F, τ_M, τ_B
        y_fit = legval(mu, coeffs0)
        sigma_eff, tau_info = compute_angular_band_discrepancy(
            mu=mu, y=y, sigma=sigma, y_fit=y_fit,
            min_points_per_band=min_points_per_band,
            max_tau_fraction=max_tau_fraction,
        )
        scale = 1.0  # No global scaling when using band model
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
            mu, y, sigma, degree_use,
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

    # Build group mapping for correlated normalization (Improvement 1.3)
    group_indices: Dict[Tuple, List[int]] = {}
    group_keys: List[Tuple] = []
    if sigma_norm > 0.0:
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
    samples = []
    if n_samples <= 1 and not stochastic:
        samples.append(coeffs0)
    else:
        n_draws = max(1, int(n_samples))
        for _ in range(n_draws):
            y_s = y.copy()

            # Apply correlated normalization uncertainty per experiment (Improvement 1.3)
            if sigma_norm > 0.0 and group_keys:
                for key in group_keys:
                    indices = group_indices[key]
                    if norm_dist == "lognormal":
                        # Lognormal: always positive, multiplicative
                        N_g = rng.lognormal(mean=0.0, sigma=sigma_norm)
                    else:  # "normal"
                        N_g = 1.0 + rng.normal(0.0, sigma_norm)
                    y_s[indices] *= N_g

            # Add pointwise noise
            y_s = y_s + rng.normal(loc=0.0, scale=sigma_eff, size=n)

            coeffs_s, _, _, _ = _weighted_ridge_fit(
                mu, y_s, sigma_eff, degree_use,
                ridge_lambda=ridge_lambda,
                ridge_power=ridge_power,
                df_method=df_method,
                external_weights=external_weights,
                fixed_c0=c0_fix,  # Pass fixed c0 if freeze_c0=True
                fixed_coeffs=fixed_high,  # Freeze higher orders if max_sample_order set
            )
            samples.append(coeffs_s)

    coef_mat = np.vstack(samples)
    coef_df = pd.DataFrame(coef_mat, columns=[f"c{l}" for l in range(degree_use + 1)])

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
        n_experiments_for_norm=len(group_keys) if sigma_norm > 0.0 else 0,
        max_sample_order=max_sample_order,
        n_frozen_high=len(fixed_high) if fixed_high else 0,
    )
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


def load_exfor_for_fitting(
    exfor_directory: str,
    energy_mev: float,
    tolerance: float = 0.015,
    m_proj_u: float = 1.008665,
    m_targ_u: float = 55.93494,
) -> pd.DataFrame:
    """
    Load EXFOR experimental data for a specific energy and format for Legendre fitting.

    This function loads EXFOR data at the specified energy (within tolerance),
    transforms LAB frame data to CM frame if needed, and returns a DataFrame
    compatible with sample_legendre_coefficients().

    Parameters
    ----------
    exfor_directory : str
        Path to directory containing EXFOR JSON files
    energy_mev : float
        Target energy in MeV
    tolerance : float, optional
        Energy matching tolerance in MeV (default: 0.015)
    m_proj_u : float, optional
        Projectile mass in atomic mass units (default: 1.008665 for neutron)
    m_targ_u : float, optional
        Target mass in atomic mass units (default: 55.93494 for Fe-56)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        - 'theta_deg': scattering angle in degrees
        - 'value': differential cross section dσ/dΩ in barns/sr
        - 'unc': uncertainty (error_stat) in barns/sr
        - 'mu': cos(theta) in CM frame
        - 'frame': reference frame ('CM' or 'LAB')
        - 'entry': EXFOR entry number
        - 'subentry': EXFOR subentry number
        - 'author': first author name
        - 'year': publication year
        - 'reaction': reaction string (e.g., '26-FE-56(N,EL)26-FE-56' or '26-FE-0(N,EL)26-FE-0' for natural)

    Notes
    -----
    - All LAB frame data is automatically converted to CM frame
    - Only statistical uncertainties (error_stat) are included
    - Multiple EXFOR experiments at matching energies are concatenated
    - Returns empty DataFrame if no matching data found
    """
    if not _EXFOR_AVAILABLE:
        raise ImportError(
            "EXFOR utilities not available. Ensure angular_distribution_utils.py "
            "and uncertainty_analysis_utils.py are in ../EXFOR/"
        )

    exfor_data = load_exfor_data_within_tolerance(
        exfor_directory, energy_mev, tolerance
    )

    if exfor_data is None or len(exfor_data) == 0:
        print(f"No EXFOR data found for E={energy_mev:.3f} MeV ± {tolerance:.3f} MeV")
        return pd.DataFrame(columns=[
            'theta_deg', 'value', 'unc', 'mu', 'frame', 'entry', 'subentry', 'author', 'year', 'reaction'
        ])

    all_frames = []
    for df, meta in exfor_data:
        # Extract metadata
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        matched_energy = meta.get('matched_energy', energy_mev)
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        # Extract author and year
        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Extract columns
        angles_deg = df['angle'].to_numpy(dtype=float)
        dsig = df['dsig'].to_numpy(dtype=float)
        error_stat = df['error_stat'].to_numpy(dtype=float)

        # Transform to CM frame if needed
        if frame == 'LAB':
            mu_lab = np.cos(np.deg2rad(angles_deg))
            mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                mu_lab, dsig, error_stat, m_proj_u, m_targ_u
            )

            angles_cm_deg = np.rad2deg(np.arccos(mu_cm))

            # Create transformed dataframe
            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'frame': 'CM',
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
            })
        else:
            # Already in CM frame
            mu_cm = np.cos(np.deg2rad(angles_deg))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_deg,
                'value': dsig,
                'unc': error_stat,
                'mu': mu_cm,
                'frame': frame,
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
            })

        all_frames.append(transformed_df)

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)

    print(f"Loaded {len(exfor_data)} EXFOR experiment(s) with {len(result)} data points")
    print(f"Energy match: {energy_mev:.3f} MeV (tolerance: ±{tolerance:.3f} MeV)")

    return result


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
        EXFOR data from load_exfor_for_fitting() with columns
        theta_deg, value, unc, mu
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
