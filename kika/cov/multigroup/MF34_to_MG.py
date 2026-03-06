"""
Convert MF34 angular distribution covariance data to multigroup format.

This module implements the conversion algorithm from continuous-energy MF34 
covariance matrices to multigroup angular distribution covariance matrices,
following the detailed mathematical procedure outlined in the user requirements.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from scipy import integrate, interpolate, special
import warnings

from ..mf34_covmat import MF34CovMat
from .mg_mf34_covmat import MGMF34CovMat
from ...energy_grids import grids


class WeightingFunction:
    """
    Class to represent various weighting functions phi(E) for multigroup collapse.
    """
    
    @staticmethod
    def constant(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Constant weighting function phi(E) = 1."""
        if isinstance(energy, np.ndarray):
            return np.ones_like(energy)
        return 1.0
    
    @staticmethod
    def constant_antiderivative(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Antiderivative of constant weighting: Phi(E) = E."""
        return energy
    
    @staticmethod
    def maxwellian(energy: Union[float, np.ndarray], temperature: float = 2.53e-2) -> Union[float, np.ndarray]:
        """Maxwellian spectrum phi(E) = sqrt(E) * exp(-E/kT)."""
        kT = temperature  # in eV, default = 0.0253 eV (room temperature)
        return np.sqrt(energy) * np.exp(-energy / kT)
    
    @staticmethod
    def maxwellian_antiderivative(energy: Union[float, np.ndarray], temperature: float = 2.53e-2) -> Union[float, np.ndarray]:
        """Antiderivative of Maxwellian: ∫ sqrt(E) e^{-E/kT} dE = (kT)^{3/2} Γ(3/2) * gammainc(3/2, E/kT)."""
        kT = temperature
        x = np.asarray(energy) / kT
        return (kT ** 1.5) * special.gamma(1.5) * special.gammainc(1.5, x)
    
    @staticmethod
    def fission_spectrum(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Simplified fission spectrum phi(E) = sqrt(E) * exp(-E/1.29e6)."""
        return np.sqrt(energy) * np.exp(-energy / 1.29e6)  # 1.29 MeV average
    
    @staticmethod
    def fission_spectrum_antiderivative(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Antiderivative of fission spectrum with kT≈1.29 MeV."""
        kT = 1.29e6  # 1.29 MeV in eV
        x = np.asarray(energy) / kT
        return (kT ** 1.5) * special.gamma(1.5) * special.gammainc(1.5, x)

    # --- New lethargy (phi=1/E) weighting for log-energy (flat in lethargy) averaging ---
    @staticmethod
    def lethargy(energy: Union[float, np.ndarray], epsilon: float = 1e-30) -> Union[float, np.ndarray]:
        """Lethargy weighting function phi(E)=1/E (flat per unit lethargy).

        This produces group averages consistent with a uniform distribution in lethargy
        u = ln(E). A tiny epsilon prevents division by zero for any accidental
        non-positive energies.
        """
        if isinstance(energy, np.ndarray):
            return 1.0 / np.maximum(energy, epsilon)
        return 1.0 / max(energy, epsilon)

    @staticmethod
    def lethargy_antiderivative(energy: Union[float, np.ndarray], epsilon: float = 1e-30) -> Union[float, np.ndarray]:
        """Antiderivative of 1/E: Phi(E)=ln(E). Safe for vector/scalar inputs."""
        if isinstance(energy, np.ndarray):
            return np.log(np.maximum(energy, epsilon))
        return float(np.log(max(energy, epsilon)))


def compute_energy_rebin_operator(coarse_energy_grid: np.ndarray, 
                                  fine_energy_grid: np.ndarray,
                                  phi_func: Callable = WeightingFunction.constant,
                                  phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> np.ndarray:
    """
    Build the H→T energy-rebin operator M following NJOY methodology.
    
    This creates a row-stochastic matrix M where M[g,h] represents the contribution
    of coarse energy bin h to fine energy bin g, weighted by the spectrum.
    
    M_{g,h} = ∫(E∈(g∩h)) w(E) dE / ∫(E∈g) w(E) dE
    
    Parameters
    ----------
    coarse_energy_grid : np.ndarray
        Coarse energy grid (H) from MF34 covariance data
    fine_energy_grid : np.ndarray  
        Fine target energy grid (T) for multigroup data
    phi_func : Callable
        Weighting function w(E)
    phi_antiderivative : Callable
        Antiderivative of weighting function
        
    Returns
    -------
    np.ndarray
        Row-stochastic rebin matrix M of shape (N_fine, N_coarse)
    """
    N_coarse = len(coarse_energy_grid) - 1
    N_fine = len(fine_energy_grid) - 1
    
    M = np.zeros((N_fine, N_coarse))
    
    for g in range(N_fine):  # Fine energy bins (target)
        E_g_low = fine_energy_grid[g]
        E_g_high = fine_energy_grid[g + 1]
        
        # Compute denominator: ∫(E∈g) w(E) dE 
        if phi_func == WeightingFunction.constant:
            denom = E_g_high - E_g_low
        else:
            denom = phi_antiderivative(E_g_high) - phi_antiderivative(E_g_low)
        
        if abs(denom) < 1e-15:
            continue
            
        for h in range(N_coarse):  # Coarse energy bins (source)
            E_h_low = coarse_energy_grid[h]
            E_h_high = coarse_energy_grid[h + 1]
            
            # Find intersection: (g ∩ h)
            E_intersect_low = max(E_g_low, E_h_low)
            E_intersect_high = min(E_g_high, E_h_high)
            
            if E_intersect_high > E_intersect_low:
                # Compute numerator: ∫(E∈(g∩h)) w(E) dE
                if phi_func == WeightingFunction.constant:
                    numer = E_intersect_high - E_intersect_low
                else:
                    numer = phi_antiderivative(E_intersect_high) - phi_antiderivative(E_intersect_low)
                
                M[g, h] = numer / denom
    
    return M


def map_covariance_matrix(coarse_matrix: np.ndarray,
                         rebin_operator: np.ndarray) -> np.ndarray:
    """
    Map covariance matrix from coarse grid to fine grid using rebin operator.
    
    For a single ℓ block: C^(T) = M @ C^(H) @ M.T
    
    Parameters
    ----------
    coarse_matrix : np.ndarray
        Covariance matrix on coarse energy grid
    rebin_operator : np.ndarray
        Energy rebin operator M (N_fine × N_coarse)
        
    Returns
    -------
    np.ndarray
        Covariance matrix on fine energy grid
    """
    return rebin_operator @ coarse_matrix @ rebin_operator.T


def convert_relative_to_absolute_covariance(relative_matrix: np.ndarray,
                                          means_row: np.ndarray,
                                          means_col: np.ndarray) -> np.ndarray:
    """
    Convert relative covariance to absolute covariance.
    
    C^abs = diag(μ_ℓ) @ R @ diag(μ_ℓ')
    
    Parameters
    ----------
    relative_matrix : np.ndarray
        Relative covariance matrix
    means_row : np.ndarray
        Row means (μ_ℓ)
    means_col : np.ndarray
        Column means (μ_ℓ')
        
    Returns
    -------
    np.ndarray
        Absolute covariance matrix
    """
    return np.diag(means_row) @ relative_matrix @ np.diag(means_col)


def convert_absolute_to_relative_covariance(absolute_matrix: np.ndarray,
                                          means_row: np.ndarray,
                                          means_col: np.ndarray,
                                          epsilon: float = 1e-15) -> np.ndarray:
    """
    Convert absolute covariance to relative covariance.
    
    R = diag(μ_ℓ)^(-1) @ C^abs @ diag(μ_ℓ')^(-1)
    
    Parameters
    ----------
    absolute_matrix : np.ndarray
        Absolute covariance matrix
    means_row : np.ndarray
        Row means (μ_ℓ)
    means_col : np.ndarray
        Column means (μ_ℓ')
    epsilon : float
        Small value to prevent division by zero
        
    Returns
    -------
    np.ndarray
        Relative covariance matrix
    """
    # Create inverse diagonal matrices with epsilon protection
    means_row_safe = np.where(np.abs(means_row) > epsilon, means_row, epsilon)
    means_col_safe = np.where(np.abs(means_col) > epsilon, means_col, epsilon)
    
    inv_diag_row = np.diag(1.0 / means_row_safe)
    inv_diag_col = np.diag(1.0 / means_col_safe)
    
    relative = inv_diag_row @ absolute_matrix @ inv_diag_col
    
    # Set elements to NaN where original means were too small
    mask_row = np.abs(means_row) <= epsilon
    mask_col = np.abs(means_col) <= epsilon
    
    bad_mask = mask_row[:, None] | mask_col[None, :]
    relative[bad_mask] = np.nan

    return relative


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


def compute_base_cell_means(base_energy_grid: np.ndarray,
                          mf4_data,
                          legendre_orders: List[int],
                          phi_func: Callable = WeightingFunction.constant,
                          phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> Dict[int, np.ndarray]:
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
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping Legendre orders to base-cell means arrays
    """
    N_base = len(base_energy_grid) - 1
    base_means: Dict[int, np.ndarray] = {}

    if not legendre_orders:
        return base_means

    max_order = max(legendre_orders)

    # Extract full native MF4 energy grid for Legendre coefficients if available
    if hasattr(mf4_data, 'legendre_energies'):
        native_E = np.array(mf4_data.legendre_energies, dtype=float)
    else:
        # Fallback: derive from first requested order using evaluation at base grid edges
        native_E = np.unique(base_energy_grid)

    # Evaluate coefficients at native energies directly using MF4-provided method
    coeffs_native = mf4_data.extract_legendre_coefficients(native_E, max_order, out_of_range="zero")

    # Helper to evaluate a_l(E) piecewise linearly between native points
    def eval_coeff(l: int, energies: np.ndarray) -> np.ndarray:
        values = coeffs_native.get(l)
        if values is None:
            return np.zeros_like(energies)
        # Outside range: l==0 hold left value on low side, zero for l>0; zero on high side for l>0, hold for l==0
        E0 = native_E[0]
        E1 = native_E[-1]
        vals = np.zeros_like(energies)
        inside = (energies >= E0) & (energies <= E1)
        # Interpolate inside using numpy.interp (linear)
        vals[inside] = np.interp(energies[inside], native_E, values)
        # Low side extrapolation
        low_mask = energies < E0
        if np.any(low_mask):
            vals[low_mask] = values[0] if l == 0 else 0.0
        # High side extrapolation
        high_mask = energies > E1
        if np.any(high_mask):
            vals[high_mask] = values[-1] if l == 0 else 0.0
        return vals

    is_constant_weight = phi_func == WeightingFunction.constant
    is_lethargy_weight = phi_func == WeightingFunction.lethargy

    # Precompute logs of native energy for lethargy analytic segment integration
    if is_lethargy_weight:
        log_native_E = np.log(native_E)

    for l in legendre_orders:
        means = np.zeros(N_base)
        a_l_native = coeffs_native.get(l)
        if a_l_native is None:
            base_means[l] = means
            continue

        for i in range(N_base):
            E_lo = base_energy_grid[i]
            E_hi = base_energy_grid[i + 1]
            if E_hi <= E_lo:
                continue

            # Denominator
            denom = phi_antiderivative(E_hi) - phi_antiderivative(E_lo)
            if abs(denom) < 1e-30:
                continue

            # Build segment mesh: native points within (E_lo,E_hi) plus boundaries
            # Identify indices of native_E inside interval
            inside_idx = np.where((native_E > E_lo) & (native_E < E_hi))[0]
            segment_E = np.concatenate(([E_lo], native_E[inside_idx], [E_hi]))
            # Ensure strictly increasing
            segment_E = np.unique(segment_E)

            # Evaluate coefficient values at segment_E
            a_vals = eval_coeff(l, segment_E)

            if is_constant_weight:
                # phi=1 -> integral a(E) dE exact via trapezoid on linear segments
                numer = integrate.trapezoid(a_vals, segment_E)
                means[i] = numer / (E_hi - E_lo)
            elif is_lethargy_weight:
                # Analytic integral over each linear segment for a(E)/E
                seg_numer = 0.0
                for k in range(len(segment_E) - 1):
                    e0 = segment_E[k]
                    e1 = segment_E[k + 1]
                    if e1 <= e0:
                        continue
                    a0 = a_vals[k]
                    a1 = a_vals[k + 1]
                    if a0 == a1:
                        # a(E)=a0 constant -> a0 * ln(e1/e0)
                        seg_numer += a0 * (np.log(e1) - np.log(e0))
                    else:
                        m = (a1 - a0) / (e1 - e0)
                        # Integral (a0 + m (E-e0))/E dE from e0 to e1
                        # = a0 ln(e1/e0) + m[(e1 - e0) - e0 ln(e1/e0)]
                        ln_ratio = np.log(e1) - np.log(e0)
                        seg_numer += a0 * ln_ratio + m * ((e1 - e0) - e0 * ln_ratio)
                means[i] = seg_numer / (np.log(E_hi) - np.log(E_lo))
            else:
                # General weighting: numeric integration on segment mesh refined with midpoints
                # Build a refined mesh with midpoints for better accuracy
                refined_E = []
                for k in range(len(segment_E) - 1):
                    refined_E.append(segment_E[k])
                    refined_E.append(0.5 * (segment_E[k] + segment_E[k + 1]))
                refined_E.append(segment_E[-1])
                refined_E = np.array(refined_E)
                a_ref = eval_coeff(l, refined_E)
                integrand = phi_func(refined_E) * a_ref
                numer = integrate.trapezoid(integrand, refined_E)
                means[i] = numer / denom

        base_means[l] = means

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
               mt: Optional[int] = None) -> MGMF34CovMat:
    """
    Convert MF34 angular distribution covariance data to multigroup format.
    
    This function implements the complete algorithm for converting continuous-energy
    MF34 covariance matrices to multigroup format, following the mathematical
    procedure outlined in the specifications.
    
    Parameters
    ----------
    endf_object : ENDF object
        ENDF object containing both MF4 and MF34 data
    energy_grid : str, list, or np.ndarray
        Multigroup energy grid specification:
        - str: Name of predefined grid (e.g., "VITAMINJ174")
        - list/array: Custom energy group boundaries
    weighting_function : str or Callable, optional
        Weighting function for multigroup collapse:
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
        
    Returns
    -------
    MGMF34CovMat
        Multigroup covariance matrix object
        
    Raises
    ------
    ValueError
        If input parameters are invalid or data is inconsistent
    """
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
    
    # Create result object
    result = MGMF34CovMat()
    result.energy_grid = mg_energy_edges
    result.weighting_function = weight_desc
    result.relative_normalization = relative_normalization
    result.energy_unit = mf34_covmat.energy_unit  # Propagate energy unit from source data
    
    # Cache for MG-grid base-cell means: keyed by (id(mf4_mt_data), l_order)
    # These are constant across iterations since the MG grid never changes.
    _mg_means_cache: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}

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
        
        # Validate frame consistency for both row and column
        mf4_frame_row = mf4_mt_data_row.frame if hasattr(mf4_mt_data_row, 'frame') else "unknown"
        mf4_frame_col = mf4_mt_data_col.frame if hasattr(mf4_mt_data_col, 'frame') else "unknown"
        validate_frame_consistency(mf4_frame_row, frame)
        validate_frame_consistency(mf4_frame_col, frame)
        
        # Compute MG means directly on the MG grid (cached across iterations)
        legendre_orders_row = [l_row]
        legendre_orders_col = [l_col] if (l_col != l_row or reaction_col != reaction_row) else []

        cache_key_row = (id(mf4_mt_data_row), l_row)
        if cache_key_row not in _mg_means_cache:
            _mg_means_cache[cache_key_row] = compute_base_cell_means(
                mg_energy_edges, mf4_mt_data_row, legendre_orders_row, phi_func, phi_antiderivative
            )
        mg_base_means_row = _mg_means_cache[cache_key_row]

        if legendre_orders_col:
            cache_key_col = (id(mf4_mt_data_col), l_col)
            if cache_key_col not in _mg_means_cache:
                _mg_means_cache[cache_key_col] = compute_base_cell_means(
                    mg_energy_edges, mf4_mt_data_col, legendre_orders_col, phi_func, phi_antiderivative
                )
            mg_base_means_col = _mg_means_cache[cache_key_col]
        else:
            mg_base_means_col = mg_base_means_row
        
        mg_means_row_vals = mg_base_means_row[l_row]
        mg_means_col_vals = mg_base_means_col[l_col] if mg_base_means_col is not mg_base_means_row else mg_means_row_vals

        # Process covariance matrix
        base_matrix = mf34_covmat.matrices[i]

        if is_relative:
            # Union-grid path: build a union of MF34 + MG boundaries so each
            # sub-cell falls entirely within one MF34 cell AND one MG group.
            # This gives accurate sub-cell averaging of a_l(E) when an MF34
            # cell straddles an MG boundary, matching NJOY's errorr approach.

            # 1) Build union grid
            union_grid = build_union_grid(mf34_energy_grid, mg_energy_edges)

            # 2) Compute cell-averaged Legendre coefficients on the union grid
            union_base_means_row = compute_base_cell_means(
                union_grid, mf4_mt_data_row, legendre_orders_row, phi_func, phi_antiderivative
            )
            if legendre_orders_col:
                union_base_means_col = compute_base_cell_means(
                    union_grid, mf4_mt_data_col, legendre_orders_col, phi_func, phi_antiderivative
                )
            else:
                union_base_means_col = union_base_means_row

            union_means_row_vals = union_base_means_row[l_row]
            union_means_col_vals = union_base_means_col[l_col] if union_base_means_col is not union_base_means_row else union_means_row_vals

            # 3) Expand relative MF34 matrix to union grid
            R_union = expand_matrix_to_union_grid(base_matrix, mf34_energy_grid, union_grid)

            # 4) Convert relative → absolute on union grid
            absolute_union = convert_relative_to_absolute_covariance(
                R_union, union_means_row_vals, union_means_col_vals
            )

            # 5) Compute overlap weights from union grid → MG grid
            #    (trivial: each union cell falls entirely within one MG group)
            union_overlap_weights = compute_overlap_weights(
                union_grid, mg_energy_edges, phi_func, phi_antiderivative
            )

            # 6) Collapse absolute covariance from union grid → MG
            absolute_fine = collapse_absolute_covariance(
                absolute_union, union_overlap_weights
            )

            # 7) Compute MG means for normalization via union grid path
            #    (ENDF-preserving: collapse union-grid means to MG)
            mf34_mg_means_row = compute_mg_means(union_base_means_row, union_overlap_weights)[l_row]
            if union_base_means_col is union_base_means_row:
                mf34_mg_means_col = mf34_mg_means_row
            else:
                mf34_mg_means_col = compute_mg_means(union_base_means_col, union_overlap_weights)[l_col]

            # Compute only the requested normalization (avoid doing both)
            col_vals_symmetric = mg_means_col_vals if mg_base_means_col is not mg_base_means_row else mg_means_row_vals
            if relative_normalization.lower() == "mg_cell":
                relative_fine = convert_absolute_to_relative_covariance(
                    absolute_fine, mg_means_row_vals, col_vals_symmetric
                )
            else:  # "mf34_cell" (default, ENDF-preserving)
                relative_fine = convert_absolute_to_relative_covariance(
                    absolute_fine, mf34_mg_means_row, mf34_mg_means_col
                )
            mg_means_row_vals_to_store = mg_means_row_vals
            mg_means_col_vals_to_store = col_vals_symmetric
        else:
            # For absolute input: map to MG grid then derive relative using physics-grid means
            rebin_operator = compute_energy_rebin_operator(
                mf34_energy_grid, mg_energy_edges, phi_func, phi_antiderivative
            )
            absolute_fine = map_covariance_matrix(base_matrix, rebin_operator)
            relative_fine = convert_absolute_to_relative_covariance(
                absolute_fine, mg_means_row_vals, mg_means_col_vals
            )
            mg_means_row_vals_to_store = mg_means_row_vals
            mg_means_col_vals_to_store = mg_means_col_vals
        
        # Apply quality checks and add to result
        mg_relative = enforce_matrix_quality(relative_fine, "relative", project_to_psd=True)
        mg_absolute = enforce_matrix_quality(absolute_fine, "absolute", project_to_psd=True)
        result.add_matrix(
            isotope_row, reaction_row, l_row,
            isotope_col, reaction_col, l_col,
            mg_relative, mg_absolute,
            mg_means_row_vals_to_store, mg_means_col_vals_to_store,
            frame
        )
    
    return result
