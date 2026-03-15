"""Collapse LegendreCovariance to MultigroupLegendreCovariance.

Uses generic functions from kika.processing.multigroup for the math,
adds Legendre-specific logic (MF4 nominal coefficients, union grids, frame handling).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from scipy import integrate, interpolate, special
import warnings

from ..legendre_covariance import LegendreCovariance
from .mg_legendre_covariance import MultigroupLegendreCovariance
from ...energy_grids import grids
from ...exfor.angular_distribution import ExforAngularDistribution
from ...processing.multigroup import (compute_rebin_operator, collapse_covariance, relative_to_absolute, absolute_to_relative, WeightingFunction)


def validate_frame_consistency(mf4_frame: str, mf34_frame: str) -> None:
    """
    Validate that MF4 and MF34 data are in consistent reference frames.

    Parameters
    ----------
    mf4_frame : str
        Reference frame from MF4 data ("LAB", "CM", or "same-as-MF4")
    mf34_frame : str
        Reference frame from MF34 data ("LAB", "CM", or "same-as-MF4")

    Raises
    ------
    ValueError
        If frames are inconsistent
    """
    # Handle "same-as-MF4" case
    if mf34_frame == "same-as-MF4":
        return  # Always consistent

    if mf4_frame != mf34_frame:
        raise ValueError(
            f"Frame mismatch between MF4 ({mf4_frame}) and MF34 ({mf34_frame}). "
            f"Angular distribution covariance processing requires consistent reference frames. "
            f"Please transform one dataset to match the other before proceeding."
        )


def compute_overlap_weights(base_energy_grid: np.ndarray,
                          mg_energy_grid: np.ndarray,
                          phi_func: Callable = WeightingFunction.constant,
                          phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> np.ndarray:
    """
    Compute base→group overlap weights w_{i,g}.

    w_{i,g} = [Phi(min(E_{i+1}, G_{g+1})) - Phi(max(E_i, G_g))]_+

    Parameters
    ----------
    base_energy_grid : np.ndarray
        Base energy grid edges [E_0, E_1, ..., E_N]
    mg_energy_grid : np.ndarray
        Multigroup energy grid edges [G_0, G_1, ..., G_n]
    phi_func : Callable, optional
        Weighting function phi(E)
    phi_antiderivative : Callable, optional
        Antiderivative of weighting function Phi(E)

    Returns
    -------
    np.ndarray
        Overlap weights matrix of shape (N_base_cells, N_mg_groups)
    """
    # Vectorized overlap computation using broadcasting
    E_lo = base_energy_grid[:-1, None]   # (N_base, 1)
    E_hi = base_energy_grid[1:, None]    # (N_base, 1)
    G_lo = mg_energy_grid[None, :-1]     # (1, N_mg)
    G_hi = mg_energy_grid[None, 1:]      # (1, N_mg)

    lower = np.maximum(E_lo, G_lo)       # (N_base, N_mg)
    upper = np.minimum(E_hi, G_hi)       # (N_base, N_mg)
    has_overlap = upper > lower

    # Evaluate antiderivative on the overlap bounds (only where needed)
    weights = np.zeros_like(lower)
    if np.any(has_overlap):
        weights[has_overlap] = np.maximum(
            phi_antiderivative(upper[has_overlap]) - phi_antiderivative(lower[has_overlap]),
            0.0
        )

    return weights


def _build_overlap_from_flux(union_grid: np.ndarray, mg_grid: np.ndarray,
                             flux_integrals: np.ndarray) -> np.ndarray:
    """Build overlap-weight matrix from per-cell flux integrals.

    Each union cell lies entirely within one MG group (by construction of the
    union grid), so each row of the returned matrix has exactly one nonzero
    entry equal to the corresponding flux integral.

    Parameters
    ----------
    union_grid : np.ndarray
        Union energy grid edges, shape (N_union+1,)
    mg_grid : np.ndarray
        Multigroup energy grid edges, shape (N_mg+1,)
    flux_integrals : np.ndarray
        Per-cell flux integrals, shape (N_union,)

    Returns
    -------
    np.ndarray
        Overlap-weight matrix, shape (N_union, N_mg)
    """
    N_union = len(union_grid) - 1
    N_mg = len(mg_grid) - 1
    midpoints = 0.5 * (union_grid[:-1] + union_grid[1:])
    group_idx = np.clip(np.searchsorted(mg_grid, midpoints, side='right') - 1, 0, N_mg - 1)
    weights = np.zeros((N_union, N_mg))
    weights[np.arange(N_union), group_idx] = flux_integrals
    return weights


def cm_to_lab_legendre(coeffs_cm: Dict[int, np.ndarray], alpha: float,
                       n_quad: int = 64) -> Dict[int, np.ndarray]:
    """Transform Legendre coefficients from CM to lab frame.

    For each energy point, reconstruct the CM-frame angular distribution from
    its Legendre expansion, then project onto lab-frame Legendre polynomials
    using Gauss-Legendre quadrature (matching NJOY's 64-point approach).

    Parameters
    ----------
    coeffs_cm : Dict[int, np.ndarray]
        {l_order: array_of_values_at_native_energies} in CM frame.
        Must include l=0 if present; missing orders are treated as zero.
    alpha : float
        Mass ratio m_projectile / m_target (= 1/AWR for neutrons).
    n_quad : int
        Number of Gauss-Legendre quadrature points (default 64).

    Returns
    -------
    Dict[int, np.ndarray]
        Same structure but coefficients in the lab frame.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.special import eval_legendre

    mu_q, w_q = leggauss(n_quad)  # quadrature on [-1, 1]
    mu_lab_q = ExforAngularDistribution.cos_lab_from_cos_cm(mu_q, alpha)

    max_l = max(coeffs_cm.keys())
    N_energies = len(next(iter(coeffs_cm.values())))

    # Precompute Legendre polynomials at quadrature points (CM and lab cosines)
    P_cm = np.array([eval_legendre(k, mu_q) for k in range(max_l + 1)])      # (L+1, n_quad)
    P_lab = np.array([eval_legendre(k, mu_lab_q) for k in range(max_l + 1)])  # (L+1, n_quad)

    # Build coefficient matrix: shape (max_l+1, N_energies)
    coeff_matrix = np.zeros((max_l + 1, N_energies))
    for k, a_k in coeffs_cm.items():
        coeff_matrix[k] = a_k

    # Weight factors (2k+1)/2 for the expansion
    weight_factors = np.array([(2 * k + 1) / 2.0 for k in range(max_l + 1)])  # (L+1,)

    # f_cm[q, e] = sum_k weight_factors[k] * coeff_matrix[k, e] * P_cm[k, q]
    weighted_coeffs = weight_factors[:, None] * coeff_matrix  # (L+1, N_energies)
    f_cm = P_cm.T @ weighted_coeffs  # (n_quad, N_energies)

    # Project onto lab-frame Legendre polynomials:
    # a_l^lab[e] = sum_q w_q[q] * f_cm[q, e] * P_lab[l, q]
    coeffs_lab = {}
    for l in coeffs_cm:
        # (n_quad,) * (n_quad, N_energies) summed over quadrature axis
        coeffs_lab[l] = (w_q[:, None] * f_cm * P_lab[l, :, None]).sum(axis=0)

    return coeffs_lab


def compute_base_cell_means(base_energy_grid: np.ndarray,
                          mf4_data,
                          legendre_orders: List[int],
                          phi_func: Callable = WeightingFunction.constant,
                          phi_antiderivative: Callable = WeightingFunction.constant_antiderivative,
                          cm_to_lab_alpha: Optional[float] = None,
                          pendf_energies: Optional[np.ndarray] = None,
                          return_flux_integrals: bool = False,
                          _coeffs_cache: Optional[Dict] = None) -> Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], np.ndarray]]:
    """
    Compute base-cell means A_{l,i} for each Legendre order.

    A_{l,i} = integral(phi(E) * a_l(E) dE) / integral(phi(E) dE) over [E_i, E_{i+1}]

    Uses native MF4 energy breakpoints for accurate integration. For lethargy weighting,
    applies analytic integration over linear segments. Ensures l>0 coefficients are zero
    outside the native MF4 energy range.

    Parameters
    ----------
    base_energy_grid : np.ndarray
        Base energy grid edges
    mf4_data : MF4MT object
        MF4 angular distribution data
    legendre_orders : List[int]
        List of Legendre orders to compute
    phi_func : Callable, optional
        Weighting function phi(E)
    phi_antiderivative : Callable, optional
        Antiderivative of weighting function
    cm_to_lab_alpha : float, optional
        If not None, apply CM→lab frame transformation with this mass ratio
        (alpha = m_projectile / m_target = 1/AWR for neutrons) before integration.
    pendf_energies : np.ndarray, optional
        PENDF energy grid to merge into segment meshes for sigma-weighted
        integration accuracy. When provided, ensures the numerator integral
        samples sigma(E)*a_l(E)*phi(E) at the same resolution as the
        denominator antiderivative.
    return_flux_integrals : bool, optional
        If True, also return the per-cell flux integrals (denominators) computed
        during the first Legendre order iteration. Default is False.

    Returns
    -------
    Dict[int, np.ndarray] or (Dict[int, np.ndarray], np.ndarray)
        Dictionary mapping Legendre orders to base-cell means arrays.
        If return_flux_integrals=True, returns a tuple of (base_means, flux_integrals).
    """
    N_base = len(base_energy_grid) - 1
    base_means: Dict[int, np.ndarray] = {}
    flux_integrals: Optional[np.ndarray] = np.zeros(N_base) if return_flux_integrals else None
    _first_order = True  # Track whether we need to store flux integrals

    if not legendre_orders:
        if return_flux_integrals:
            return base_means, np.zeros(N_base)
        return base_means

    max_order = max(legendre_orders)

    # Check coefficients cache (keyed by mf4_data identity and CM→lab alpha)
    _cache_key = (id(mf4_data), cm_to_lab_alpha)
    if _coeffs_cache is not None and _cache_key in _coeffs_cache:
        native_E, _all_coeffs = _coeffs_cache[_cache_key]
        # Filter to only needed orders
        coeffs_native = {l: v for l, v in _all_coeffs.items() if l <= max_order}
    else:
        # Extract full native MF4 energy grid for Legendre coefficients if available
        if hasattr(mf4_data, 'legendre_energies'):
            native_E = np.array(mf4_data.legendre_energies, dtype=float)
            # For LTT=3 (mixed), include tabulated energies so the full range is covered
            if hasattr(mf4_data, 'tabulated_energies') and mf4_data.tabulated_energies:
                tab_E = np.array(mf4_data.tabulated_energies, dtype=float)
                native_E = np.unique(np.concatenate([native_E, tab_E]))
        else:
            # Fallback: derive from first requested order using evaluation at base grid edges
            native_E = np.unique(base_energy_grid)

        # When CM-to-lab is requested, extract ALL available Legendre orders
        # because the nonlinear transformation couples all orders together
        if cm_to_lab_alpha is not None:
            extract_order = 64  # large enough; capped by file max inside extract
        else:
            extract_order = max_order

        # Evaluate coefficients at native energies directly using MF4-provided method
        # For mixed (LTT=3) data, disable auto-trim to avoid unnecessary overhead
        extract_kwargs = dict(out_of_range="zero")
        if hasattr(mf4_data, 'tabulated_energies'):
            extract_kwargs['trim'] = False
        coeffs_native = mf4_data.extract_legendre_coefficients(native_E, extract_order, **extract_kwargs)

        # Apply CM→lab frame transformation if requested
        if cm_to_lab_alpha is not None:
            # Ensure l=0 is included for the distribution reconstruction
            if 0 not in coeffs_native:
                coeffs_native[0] = np.ones(len(native_E))
            coeffs_native = cm_to_lab_legendre(coeffs_native, cm_to_lab_alpha)

        # Store in cache (all extracted orders, before filtering)
        if _coeffs_cache is not None:
            _coeffs_cache[_cache_key] = (native_E, coeffs_native)

        # Keep only orders up to max_order
        coeffs_native = {l: v for l, v in coeffs_native.items() if l <= max_order}

    # Helper to evaluate a_l(E) piecewise linearly between native points
    def eval_coeff(l: int, energies: np.ndarray) -> np.ndarray:
        values = coeffs_native.get(l)
        if values is None:
            return np.zeros_like(energies)
        # Outside range: l==0 hold boundary value, l>0 zero.
        # Use relative tolerance to avoid floating-point boundary mismatches
        # (e.g. SCALE grid 1e-5 eV vs ENDF native 1e-5 eV differing by 1 ULP)
        # that create phantom zero-width segments with catastrophic integration errors.
        E0 = native_E[0]
        E1 = native_E[-1]
        rtol = 1e-12
        E0_relaxed = E0 * (1 - rtol) if E0 > 0 else E0 - 1e-30
        E1_relaxed = E1 * (1 + rtol)
        vals = np.zeros_like(energies)
        inside = (energies >= E0_relaxed) & (energies <= E1_relaxed)
        # Clamp to [E0, E1] before interp to avoid numpy.interp extrapolation
        clamped = np.clip(energies[inside], E0, E1)
        vals[inside] = np.interp(clamped, native_E, values)
        # Low side extrapolation
        low_mask = energies < E0_relaxed
        if np.any(low_mask):
            vals[low_mask] = values[0] if l == 0 else 0.0
        # High side extrapolation
        high_mask = energies > E1_relaxed
        if np.any(high_mask):
            vals[high_mask] = values[-1] if l == 0 else 0.0
        return vals

    is_constant_weight = phi_func == WeightingFunction.constant
    is_lethargy_weight = phi_func == WeightingFunction.lethargy

    # Precompute logs of native energy for lethargy analytic segment integration
    if is_lethargy_weight:
        log_native_E = np.log(native_E)

    # --- Precompute segment meshes for all cells (reused across Legendre orders) ---
    cell_segments: List[np.ndarray] = []
    cell_valid = np.ones(N_base, dtype=bool)  # Track cells with valid denominator
    cell_denoms = np.zeros(N_base)
    cell_flux_vals = np.zeros(N_base)

    for i in range(N_base):
        E_lo = base_energy_grid[i]
        E_hi = base_energy_grid[i + 1]
        if E_hi <= E_lo:
            cell_segments.append(np.empty(0))
            cell_valid[i] = False
            continue

        denom = phi_antiderivative(E_hi) - phi_antiderivative(E_lo)
        if abs(denom) < 1e-30:
            cell_segments.append(np.empty(0))
            cell_valid[i] = False
            continue

        cell_denoms[i] = denom

        # Use searchsorted (O(log N)) instead of np.where (O(N))
        idx_lo = np.searchsorted(native_E, E_lo, side='right')
        idx_hi = np.searchsorted(native_E, E_hi, side='left')
        segment_E = np.concatenate(([E_lo], native_E[idx_lo:idx_hi], [E_hi]))

        # Include PENDF grid points for sigma-weighted accuracy
        if pendf_energies is not None:
            p_lo = np.searchsorted(pendf_energies, E_lo, side='right')
            p_hi = np.searchsorted(pendf_energies, E_hi, side='left')
            pendf_inside = pendf_energies[p_lo:p_hi]
            if len(pendf_inside) > 0:
                segment_E = np.concatenate([segment_E, pendf_inside])

        # Ensure strictly increasing
        segment_E = np.unique(segment_E)
        cell_segments.append(segment_E)

    # Precompute concatenated segment array and split indices for batch interp
    valid_indices = np.where(cell_valid)[0]
    if len(valid_indices) > 0:
        all_segment_E = np.concatenate([cell_segments[i] for i in valid_indices])
        cell_lengths = np.array([len(cell_segments[i]) for i in valid_indices])
        split_indices = np.cumsum(cell_lengths[:-1])
    else:
        all_segment_E = np.empty(0)
        cell_lengths = np.empty(0, dtype=int)
        split_indices = np.empty(0, dtype=int)

    # For general weighting, batch-compute phi values and flux integrals once
    _is_general_weight = not is_constant_weight and not is_lethargy_weight
    if _is_general_weight and len(all_segment_E) > 0:
        all_phi_vals = phi_func(all_segment_E)
        phi_per_cell = np.split(all_phi_vals, split_indices)
        # Precompute per-cell flux integrals (independent of Legendre order)
        _general_cell_flux = np.zeros(N_base)
        for vi, i in enumerate(valid_indices):
            _general_cell_flux[i] = integrate.trapezoid(phi_per_cell[vi], cell_segments[i])
    else:
        phi_per_cell = []
        _general_cell_flux = np.zeros(N_base)

    for l in legendre_orders:
        means = np.zeros(N_base)
        a_l_native = coeffs_native.get(l)
        if a_l_native is None:
            base_means[l] = means
            continue

        # Batch eval_coeff: single np.interp call for all cells
        if len(all_segment_E) > 0:
            all_a_vals = eval_coeff(l, all_segment_E)
            a_vals_per_cell = np.split(all_a_vals, split_indices)
        else:
            a_vals_per_cell = []

        for vi, i in enumerate(valid_indices):
            segment_E = cell_segments[i]
            a_vals = a_vals_per_cell[vi]

            if is_constant_weight:
                # phi=1 -> integral a(E) dE exact via trapezoid on linear segments
                E_lo = base_energy_grid[i]
                E_hi = base_energy_grid[i + 1]
                _cell_flux = E_hi - E_lo
                numer = integrate.trapezoid(a_vals, segment_E)
                means[i] = numer / _cell_flux
            elif is_lethargy_weight:
                # Analytic integral over each linear segment for a(E)/E
                # Vectorized over all segments in this cell
                E_lo = base_energy_grid[i]
                E_hi = base_energy_grid[i + 1]
                _cell_flux = np.log(E_hi) - np.log(E_lo)
                e0 = segment_E[:-1]
                e1 = segment_E[1:]
                a0 = a_vals[:-1]
                a1 = a_vals[1:]
                valid = e1 > e0
                if np.any(valid):
                    ln_ratio = np.log(e1[valid]) - np.log(e0[valid])
                    delta_e = e1[valid] - e0[valid]
                    m = (a1[valid] - a0[valid]) / delta_e
                    # General formula: a0*ln(e1/e0) + m*[(e1-e0) - e0*ln(e1/e0)]
                    # When a0==a1, m=0 so second term vanishes automatically
                    seg_numer = np.sum(a0[valid] * ln_ratio + m * (delta_e - e0[valid] * ln_ratio))
                else:
                    seg_numer = 0.0
                means[i] = seg_numer / _cell_flux
            else:
                # General weighting: use precomputed phi values
                phi_seg = phi_per_cell[vi]
                _cell_flux = _general_cell_flux[i]
                if abs(_cell_flux) < 1e-30:
                    continue
                numer = integrate.trapezoid(phi_seg * a_vals, segment_E)
                means[i] = numer / _cell_flux

            # Store flux integrals on first Legendre order iteration
            if _first_order and flux_integrals is not None:
                flux_integrals[i] = _cell_flux

        base_means[l] = means
        _first_order = False

    if return_flux_integrals:
        return base_means, flux_integrals
    return base_means


def compute_mg_means(base_means: Dict[int, np.ndarray],
                    overlap_weights: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute multigroup means bar{a}_{l,g} from base-cell means and overlap weights.

    bar{a}_{l,g} = sum_i(w_{i,g} * A_{l,i}) / W_g
    where W_g = sum_i(w_{i,g})

    Parameters
    ----------
    base_means : Dict[int, np.ndarray]
        Base-cell means for each Legendre order
    overlap_weights : np.ndarray
        Overlap weights matrix (N_base, N_mg)

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping Legendre orders to MG means arrays
    """
    N_mg = overlap_weights.shape[1]
    mg_means = {}

    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)

    for l, A_l_i in base_means.items():
        weighted = overlap_weights.T @ A_l_i       # (N_mg,)
        mg_means[l] = np.where(W_g > 1e-15, weighted / W_g, 0.0)

    return mg_means


def collapse_relative_covariance(relative_base_matrix: np.ndarray,
                                base_means_row: np.ndarray,
                                base_means_col: np.ndarray,
                                overlap_weights: np.ndarray,
                                mg_means_row: np.ndarray,
                                mg_means_col: np.ndarray) -> np.ndarray:
    """
    Collapse MF34 relative covariance to multigroup relative covariance.

    R^MG_{gg'} = sum_{ij}(w_{ig} * A_{l,i} * R_{ij} * A_{l',j} * w_{jg'}) /
                 (W_g * W_g' * bar{a}_{l,g} * bar{a}_{l',g'})

    Parameters
    ----------
    relative_base_matrix : np.ndarray
        Base-grid relative covariance matrix R_{ij}
    base_means_row : np.ndarray
        Base-cell means for row Legendre order A_{l,i}
    base_means_col : np.ndarray
        Base-cell means for column Legendre order A_{l',j}
    overlap_weights : np.ndarray
        Overlap weights matrix w_{i,g}
    mg_means_row : np.ndarray
        MG means for row Legendre order bar{a}_{l,g}
    mg_means_col : np.ndarray
        MG means for column Legendre order bar{a}_{l',g'}

    Returns
    -------
    np.ndarray
        Multigroup relative covariance matrix
    """
    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)

    # Vectorized: pre-multiply base matrix by means, then matrix product
    # weighted_base[i,j] = A_row[i] * R[i,j] * A_col[j]
    weighted_base = (base_means_row[:, None] * relative_base_matrix * base_means_col[None, :])
    numerator = overlap_weights.T @ weighted_base @ overlap_weights
    denom = np.outer(W_g * mg_means_row, W_g * mg_means_col)

    with np.errstate(divide='ignore', invalid='ignore'):
        mg_relative = np.where(np.abs(denom) > 1e-15, numerator / denom, np.nan)

    return mg_relative


def collapse_absolute_covariance(absolute_base_matrix: np.ndarray,
                                overlap_weights: np.ndarray) -> np.ndarray:
    """
    Collapse MF34 absolute covariance to multigroup absolute covariance.

    C^MG_{gg'} = sum_{ij}(w_{ig} * C^abs_{ij} * w_{jg'}) / (W_g * W_g')

    Parameters
    ----------
    absolute_base_matrix : np.ndarray
        Base-grid absolute covariance matrix C^abs_{ij}
    overlap_weights : np.ndarray
        Overlap weights matrix w_{i,g}

    Returns
    -------
    np.ndarray
        Multigroup absolute covariance matrix
    """
    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)

    # Vectorized: numerator[g,g'] = sum_{ij} w[i,g] * C[i,j] * w[j,g']
    numerator = overlap_weights.T @ absolute_base_matrix @ overlap_weights
    denom = np.outer(W_g, W_g)

    with np.errstate(divide='ignore', invalid='ignore'):
        mg_absolute = np.where(np.abs(denom) > 1e-15, numerator / denom, 0.0)

    return mg_absolute


def enforce_matrix_quality(matrix: np.ndarray,
                          matrix_type: str = "relative",
                          clip_small_negatives: bool = False,
                          symmetrize: bool = True,
                          project_to_psd: bool = False,
                          psd_floor: float = 0.0) -> np.ndarray:
    """
    Apply quality checks and corrections to covariance matrices.

    Parameters
    ----------
    matrix : np.ndarray
        Input covariance matrix
    matrix_type : str, optional
        Type of matrix ("relative" or "absolute")
    clip_small_negatives : bool, optional
        Whether to clip small negative diagonal elements to zero
    symmetrize : bool, optional
        Whether to enforce symmetry
    project_to_psd : bool, optional
        Whether to project matrix to nearest positive semi-definite matrix
        using Higham-style eigenvalue clipping
    psd_floor : float, optional
        Minimum eigenvalue for PSD projection (default 0.0)

    Returns
    -------
    np.ndarray
        Quality-corrected matrix
    """
    result = matrix.copy()

    # Enforce symmetry
    if symmetrize:
        result = (result + result.T) / 2.0

    # Clip small negative variances on diagonal
    if clip_small_negatives:
        diagonal = np.diag(result)
        small_negatives = (diagonal < 0) & (abs(diagonal) < 1e-10)
        if np.any(small_negatives):
            warnings.warn(f"Clipped {np.sum(small_negatives)} small negative diagonal elements to zero")
            np.fill_diagonal(result, np.where(small_negatives, 0.0, diagonal))

    # Optional PSD projection by clipping negative eigenvalues (Higham-style)
    if project_to_psd:
        w, v = np.linalg.eigh(result)
        w_clipped = np.maximum(w, psd_floor)
        result = (v @ np.diag(w_clipped) @ v.T)
        result = (result + result.T) / 2.0  # Re-symmetrize after reconstruction

    return result


def build_union_grid(mf34_grid: np.ndarray, mg_grid: np.ndarray) -> np.ndarray:
    """
    Build a union energy grid from MF34 and multigroup boundaries.

    The union grid ensures that each resulting cell falls entirely within
    exactly one MF34 cell AND one MG group, enabling accurate sub-cell
    averaging during covariance collapse.

    Parameters
    ----------
    mf34_grid : np.ndarray
        MF34 covariance energy grid edges
    mg_grid : np.ndarray
        Multigroup energy grid edges

    Returns
    -------
    np.ndarray
        Sorted unique union of both grids
    """
    return np.unique(np.concatenate([mf34_grid, mg_grid]))


def expand_matrix_to_union_grid(mf34_matrix: np.ndarray,
                                mf34_grid: np.ndarray,
                                union_grid: np.ndarray) -> np.ndarray:
    """
    Expand an MF34 covariance matrix from the MF34 grid to the union grid.

    Each union cell is mapped to its parent MF34 cell. If union cell k belongs
    to MF34 cell i and union cell l belongs to MF34 cell j, then
    R_union[k,l] = R_mf34[i,j].

    Parameters
    ----------
    mf34_matrix : np.ndarray
        Covariance matrix on MF34 grid, shape (N_mf34, N_mf34)
    mf34_grid : np.ndarray
        MF34 energy grid edges
    union_grid : np.ndarray
        Union energy grid edges (superset of mf34_grid)

    Returns
    -------
    np.ndarray
        Expanded covariance matrix on union grid, shape (N_union, N_union)
    """
    N_mf34 = len(mf34_grid) - 1
    N_union = len(union_grid) - 1

    # Map each union cell to its parent MF34 cell index
    # Use midpoints of union cells to determine which MF34 cell they belong to
    union_midpoints = 0.5 * (union_grid[:-1] + union_grid[1:])
    parent_idx = np.searchsorted(mf34_grid, union_midpoints, side='right') - 1
    # Clip to valid range
    parent_idx = np.clip(parent_idx, 0, N_mf34 - 1)

    # Build expanded matrix using vectorized fancy indexing
    expanded = mf34_matrix[np.ix_(parent_idx, parent_idx)]

    return expanded


def MF34_to_MG(endf_object,
               energy_grid: Union[str, List[float], np.ndarray],
               weighting_function: Union[str, Callable] = "constant",
               relative_normalization: str = "mf34_cell",
               isotope: Optional[int] = None,
               mt: Optional[int] = None,
               output_frame: str = "same",
               project_to_psd: bool = True) -> MultigroupLegendreCovariance:
    """
    Convert MF34 angular distribution covariance data to multigroup format.

    This function implements the complete algorithm for converting continuous-energy
    MF34 covariance matrices to multigroup format, following the mathematical
    procedure outlined in the specifications.

    Cross-section weighting (sigma(E)*phi(E)) is always applied, matching the
    NJOY ERRORR methodology. If ``endf_object.pendf`` is not populated, it is
    automatically reconstructed via ``endf_object.reconstruct_xs()``.

    Parameters
    ----------
    endf_object : ENDF object
        ENDF object containing both MF4 and MF34 data
    energy_grid : str, list, or np.ndarray
        Multigroup energy grid specification:
        - str: Name of predefined grid (e.g., "VITAMINJ174")
        - list/array: Custom energy group boundaries
    weighting_function : str or Callable, optional
        Base weighting spectrum phi(E) for multigroup collapse. The effective
        weight is always sigma(E)*phi(E):
        - "constant": phi(E) = 1 (default)
        - "lethargy": phi(E) = 1/E
        - "maxwellian": Maxwellian spectrum
        - "fission": Fission spectrum
        - Callable: Custom function
    relative_normalization : str, optional
        Normalization scheme for relative covariances:
        - "mf34_cell": Use MF34-derived MG means (ENDF-preserving, default)
          Reproduces evaluator's percent uncertainties within MF34 bins
        - "mg_cell": Use MG-grid means from MF4 (physics-preserving)
          Exposes within-cell energy variation of coefficients
    isotope : int, optional
        Specific isotope to process (if None, process all)
    mt : int, optional
        Specific MT reaction to process (if None, process all)
    output_frame : str, optional
        Reference frame for output Legendre coefficients:
        - "same" (default): preserve the MF4 native frame (backward compatible)
        - "lab": apply CM→lab kinematic transformation if MF4 is in CM frame
          (no-op if already in lab frame). Matches NJOY GROUPR convention.
    project_to_psd : bool, optional
        Whether to project output matrices to positive semi-definite (default True).
        Set to False to skip eigendecomposition for faster execution when PSD
        guarantee is not needed.

    Returns
    -------
    MultigroupLegendreCovariance
        Multigroup covariance matrix object

    Raises
    ------
    ValueError
        If input parameters are invalid or data is inconsistent
    """
    # Validate output_frame
    output_frame = output_frame.lower()
    if output_frame not in ("same", "lab"):
        raise ValueError(f"output_frame must be 'same' or 'lab', got '{output_frame}'")

    # Validate inputs and extract data
    if not hasattr(endf_object, 'mf') or 4 not in endf_object.mf or 34 not in endf_object.mf:
        raise ValueError("ENDF object must contain both MF4 and MF34 data")

    # Get MF34 covariance data (pass MF4 to populate Legendre coefficients)
    mf4_data = endf_object.mf[4]
    mf34_covmat = endf_object.mf[34].to_ang_covmat(mf4_data=mf4_data)

    # Process energy grid
    if isinstance(energy_grid, str):
        if hasattr(grids, energy_grid):
            mg_energy_edges = np.array(getattr(grids, energy_grid))
        else:
            raise ValueError(f"Unknown predefined energy grid: {energy_grid}")
    else:
        mg_energy_edges = np.array(energy_grid)

    # Validate energy grid
    if len(mg_energy_edges) < 2:
        raise ValueError("Energy grid must have at least 2 points (1 group)")
    if not np.all(np.diff(mg_energy_edges) > 0):
        raise ValueError("Energy grid must be strictly increasing")

    # Process weighting function
    if isinstance(weighting_function, str):
        if weighting_function == "constant":
            phi_func = WeightingFunction.constant
            phi_antiderivative = WeightingFunction.constant_antiderivative
            weight_desc = "constant"
        elif weighting_function == "lethargy":
            phi_func = WeightingFunction.lethargy
            phi_antiderivative = WeightingFunction.lethargy_antiderivative
            weight_desc = "lethargy"
        elif weighting_function == "maxwellian":
            phi_func = WeightingFunction.maxwellian
            phi_antiderivative = WeightingFunction.maxwellian_antiderivative
            weight_desc = "maxwellian"
        elif weighting_function == "fission":
            phi_func = WeightingFunction.fission_spectrum
            phi_antiderivative = WeightingFunction.fission_spectrum_antiderivative
            weight_desc = "fission spectrum"
        else:
            raise ValueError(f"Unknown weighting function: {weighting_function}")
    else:
        phi_func = weighting_function
        phi_antiderivative = None  # User must provide if needed
        weight_desc = "custom"

    # Check if antiderivative is available for custom functions
    if phi_antiderivative is None and weight_desc == "custom":
        raise NotImplementedError(f"Custom weighting functions require providing the antiderivative")

    # Auto-reconstruct PENDF if not present (sigma-weighting is always applied)
    if not hasattr(endf_object, 'pendf') or endf_object.pendf is None:
        endf_object.reconstruct_xs()
    _base_phi_func = phi_func
    _base_phi_antiderivative = phi_antiderivative
    weight_desc += " (sigma-weighted)"

    # Create result object
    result = MultigroupLegendreCovariance()
    result.energy_grid = mg_energy_edges
    result.weighting_function = weight_desc
    result.relative_normalization = relative_normalization
    result.energy_unit = mf34_covmat.energy_unit  # Propagate energy unit from source data

    # Cache for MG-grid base-cell means: keyed by (id(mf4_mt_data), l_order, output_frame)
    # These are constant across iterations since the MG grid never changes.
    _mg_means_cache: Dict[Tuple, Dict[int, np.ndarray]] = {}
    # Cache for union-grid base-cell means + flux integrals (keyed by mf4 id, l, frame, reaction, mf34 grid)
    _union_means_cache: Dict[Tuple, Tuple[Dict[int, np.ndarray], np.ndarray]] = {}
    # Cache for extracted+transformed Legendre coefficients (avoids re-extracting per l_order)
    _coeffs_cache: Dict[Tuple, Tuple[np.ndarray, Dict]] = {}

    # Process each matrix in the MF34 covariance data
    for i in range(mf34_covmat.num_matrices):
        # Get matrix information
        isotope_row = mf34_covmat.isotope_rows[i]
        reaction_row = mf34_covmat.reaction_rows[i]
        l_row = mf34_covmat.l_rows[i]
        isotope_col = mf34_covmat.isotope_cols[i]
        reaction_col = mf34_covmat.reaction_cols[i]
        l_col = mf34_covmat.l_cols[i]

        # Filter by isotope/MT if specified
        if isotope is not None and isotope_row != isotope:
            continue
        if mt is not None and reaction_row != mt:
            continue

        # Get matrix and metadata
        base_matrix = mf34_covmat.matrices[i]
        is_relative = mf34_covmat.is_relative[i]
        frame = mf34_covmat.frame[i]

        # Get corresponding MF4 data for row and column channels
        if reaction_row not in mf4_data.mt:
            warnings.warn(f"No MF4 data found for MT{reaction_row} (row), skipping")
            continue
        if reaction_col not in mf4_data.mt:
            warnings.warn(f"No MF4 data found for MT{reaction_col} (col), skipping")
            continue

        mf4_mt_data_row = mf4_data.mt[reaction_row]
        mf4_mt_data_col = mf4_data.mt[reaction_col]

        # Extract physics energy grids for accurate means computation
        if hasattr(mf4_mt_data_row, 'legendre_energies'):
            physics_energy_grid_row = np.array(mf4_mt_data_row.legendre_energies, dtype=float)
        else:
            physics_energy_grid_row = np.array(mf34_covmat.energy_grids[i], dtype=float)
            warnings.warn(f"Using MF34 energy grid as fallback for MT{reaction_row} (row)")

        if hasattr(mf4_mt_data_col, 'legendre_energies'):
            physics_energy_grid_col = np.array(mf4_mt_data_col.legendre_energies, dtype=float)
        else:
            physics_energy_grid_col = np.array(mf34_covmat.energy_grids[i], dtype=float)
            if reaction_col != reaction_row:
                warnings.warn(f"Using MF34 energy grid as fallback for MT{reaction_col} (col)")
        mf34_energy_grid = np.array(mf34_covmat.energy_grids[i])

        # Build effective weighting functions: sigma(E)*phi(E) for this matrix
        _mt_for_sigma = reaction_row
        if _mt_for_sigma in endf_object.pendf:
            _pendf_mt = endf_object.pendf[_mt_for_sigma]

            def _make_sigma_phi(pendf_mt, base_phi):
                def sigma_phi(energy):
                    E = np.atleast_1d(np.asarray(energy, dtype=float))
                    sigma = np.maximum(
                        np.asarray(pendf_mt.get_cross_section(E, out_of_range="hold")),
                        0.0,
                    )
                    return sigma * base_phi(E)
                return sigma_phi

            def _make_sigma_phi_antideriv(pendf_mt, base_phi):
                """Numerical antiderivative of sigma(E)*phi(E) via cumulative trapezoid."""
                E_grid = np.array(pendf_mt._energies, dtype=float)
                sigma_grid = np.maximum(np.array(pendf_mt._cross_sections, dtype=float), 0.0)
                phi_grid = base_phi(E_grid)
                integrand = sigma_grid * phi_grid

                cumul = np.zeros(len(E_grid))
                cumul[1:] = np.cumsum(
                    0.5 * (integrand[:-1] + integrand[1:]) * np.diff(E_grid)
                )

                def sigma_phi_antideriv(energy):
                    E = np.atleast_1d(np.asarray(energy, dtype=float))
                    return np.interp(E, E_grid, cumul)
                return sigma_phi_antideriv

            phi_func = _make_sigma_phi(_pendf_mt, _base_phi_func)
            phi_antiderivative = _make_sigma_phi_antideriv(_pendf_mt, _base_phi_func)
            _pendf_energies = np.array(_pendf_mt._energies, dtype=float)
        else:
            warnings.warn(
                f"No PENDF data for MT{_mt_for_sigma}, "
                f"falling back to flux-only weighting for this matrix"
            )
            phi_func = _base_phi_func
            phi_antiderivative = _base_phi_antiderivative
            _pendf_energies = None

        # Validate frame consistency for both row and column
        mf4_frame_row = mf4_mt_data_row.frame if hasattr(mf4_mt_data_row, 'frame') else "unknown"
        mf4_frame_col = mf4_mt_data_col.frame if hasattr(mf4_mt_data_col, 'frame') else "unknown"
        validate_frame_consistency(mf4_frame_row, frame)
        validate_frame_consistency(mf4_frame_col, frame)

        # Determine CM→lab alpha for this matrix pair (if output_frame='lab')
        _alpha_row = None
        _alpha_col = None
        _effective_frame = frame
        if output_frame == "lab":
            if mf4_frame_row == "CM" and hasattr(mf4_mt_data_row, '_awr') and mf4_mt_data_row._awr:
                _alpha_row = 1.0 / mf4_mt_data_row._awr
                _effective_frame = "LAB"
            if mf4_frame_col == "CM" and hasattr(mf4_mt_data_col, '_awr') and mf4_mt_data_col._awr:
                _alpha_col = 1.0 / mf4_mt_data_col._awr
                _effective_frame = "LAB"

        # Compute MG means directly on the MG grid (cached across iterations)
        legendre_orders_row = [l_row]
        legendre_orders_col = [l_col] if (l_col != l_row or reaction_col != reaction_row) else []

        cache_key_row = (id(mf4_mt_data_row), l_row, output_frame)
        if cache_key_row not in _mg_means_cache:
            _mg_means_cache[cache_key_row] = compute_base_cell_means(
                mg_energy_edges, mf4_mt_data_row, legendre_orders_row,
                phi_func, phi_antiderivative, cm_to_lab_alpha=_alpha_row,
                pendf_energies=_pendf_energies, _coeffs_cache=_coeffs_cache
            )
        mg_base_means_row = _mg_means_cache[cache_key_row]

        if legendre_orders_col:
            cache_key_col = (id(mf4_mt_data_col), l_col, output_frame)
            if cache_key_col not in _mg_means_cache:
                _mg_means_cache[cache_key_col] = compute_base_cell_means(
                    mg_energy_edges, mf4_mt_data_col, legendre_orders_col,
                    phi_func, phi_antiderivative, cm_to_lab_alpha=_alpha_col,
                    pendf_energies=_pendf_energies, _coeffs_cache=_coeffs_cache
                )
            mg_base_means_col = _mg_means_cache[cache_key_col]
        else:
            mg_base_means_col = mg_base_means_row

        mg_means_row_vals = mg_base_means_row[l_row]
        mg_means_col_vals = mg_base_means_col[l_col] if mg_base_means_col is not mg_base_means_row else mg_means_row_vals

        # Process covariance matrix
        base_matrix = mf34_covmat.matrices[i]

        if is_relative:
            # Direct collapse of relative covariance, matching NJOY ERRORR.
            #
            # The MF34 relative covariance R(i,j) is piecewise constant on its
            # coarse energy grid.  When a_l(E) varies strongly (including sign
            # changes) within an MF34 cell, converting R→absolute using the
            # cell-mean a_l dilutes the uncertainty through sign cancellation.
            # NJOY avoids this by collapsing R directly with flux-fraction
            # weights:
            #
            #   R_MG(g,g') = Σ_{i,j} f_i^g  f_j^{g'}  R(i,j)
            #
            # where f_i^g = W_i / W_g is the fraction of group g's flux that
            # falls in union cell i.  The absolute covariance is then derived
            # from the collapsed R_MG and the MG-mean Legendre coefficients.

            # 1) Build union grid (MF34 + MG boundaries)
            union_grid = build_union_grid(mf34_energy_grid, mg_energy_edges)

            # 2) Compute cell-averaged Legendre coefficients and flux integrals (cached)
            _ucache_key_row = (id(mf4_mt_data_row), l_row, output_frame, reaction_row,
                               mf34_energy_grid.tobytes())
            if _ucache_key_row in _union_means_cache:
                union_base_means_row, union_flux = _union_means_cache[_ucache_key_row]
            else:
                union_base_means_row, union_flux = compute_base_cell_means(
                    union_grid, mf4_mt_data_row, legendre_orders_row,
                    phi_func, phi_antiderivative, cm_to_lab_alpha=_alpha_row,
                    pendf_energies=_pendf_energies, return_flux_integrals=True,
                    _coeffs_cache=_coeffs_cache
                )
                _union_means_cache[_ucache_key_row] = (union_base_means_row, union_flux)
            if legendre_orders_col:
                _ucache_key_col = (id(mf4_mt_data_col), l_col, output_frame, reaction_col,
                                   mf34_energy_grid.tobytes())
                if _ucache_key_col in _union_means_cache:
                    union_base_means_col, _ = _union_means_cache[_ucache_key_col]
                else:
                    union_base_means_col, _col_flux = compute_base_cell_means(
                        union_grid, mf4_mt_data_col, legendre_orders_col,
                        phi_func, phi_antiderivative, cm_to_lab_alpha=_alpha_col,
                        pendf_energies=_pendf_energies, return_flux_integrals=True,
                        _coeffs_cache=_coeffs_cache
                    )
                    _union_means_cache[_ucache_key_col] = (union_base_means_col, _col_flux)
            else:
                union_base_means_col = union_base_means_row

            # 3) Expand relative MF34 matrix to union grid
            R_union = expand_matrix_to_union_grid(base_matrix, mf34_energy_grid, union_grid)

            # 4) Build flux-fraction matrix and collapse R directly
            union_overlap_weights = _build_overlap_from_flux(
                union_grid, mg_energy_edges, union_flux
            )
            W_g = np.sum(union_overlap_weights, axis=0)  # (N_mg,)
            # f[i, g] = W_i / W_g  (flux fraction of union cell i in group g)
            with np.errstate(divide='ignore', invalid='ignore'):
                flux_fractions = np.where(
                    W_g[None, :] > 1e-30,
                    union_overlap_weights / W_g[None, :],
                    0.0,
                )
            relative_fine = flux_fractions.T @ R_union @ flux_fractions

            # 5) Compute MG means for normalization and absolute covariance
            mf34_mg_means_row = compute_mg_means(union_base_means_row, union_overlap_weights)[l_row]
            if union_base_means_col is union_base_means_row:
                mf34_mg_means_col = mf34_mg_means_row
            else:
                mf34_mg_means_col = compute_mg_means(union_base_means_col, union_overlap_weights)[l_col]

            col_vals_symmetric = mg_means_col_vals if mg_base_means_col is not mg_base_means_row else mg_means_row_vals
            if relative_normalization.lower() == "mg_cell":
                absolute_fine = relative_to_absolute(
                    relative_fine, mg_means_row_vals, col_vals_symmetric
                )
            else:  # "mf34_cell" (default, ENDF-preserving)
                absolute_fine = relative_to_absolute(
                    relative_fine, mf34_mg_means_row, mf34_mg_means_col
                )
            mg_means_row_vals_to_store = mg_means_row_vals
            mg_means_col_vals_to_store = col_vals_symmetric
        else:
            # For absolute input: map to MG grid then derive relative using physics-grid means
            rebin_operator = compute_rebin_operator(
                mf34_energy_grid, mg_energy_edges, phi_func, phi_antiderivative
            )
            absolute_fine = collapse_covariance(base_matrix, rebin_operator)
            relative_fine = absolute_to_relative(
                absolute_fine, mg_means_row_vals, mg_means_col_vals
            )
            mg_means_row_vals_to_store = mg_means_row_vals
            mg_means_col_vals_to_store = mg_means_col_vals

        # Apply quality checks and add to result
        mg_relative = enforce_matrix_quality(relative_fine, "relative", project_to_psd=project_to_psd)
        mg_absolute = enforce_matrix_quality(absolute_fine, "absolute", project_to_psd=project_to_psd)
        result.add_matrix(
            isotope_row, reaction_row, l_row,
            isotope_col, reaction_col, l_col,
            mg_relative, mg_absolute,
            mg_means_row_vals_to_store, mg_means_col_vals_to_store,
            _effective_frame
        )

    return result


collapse_to_multigroup = MF34_to_MG
