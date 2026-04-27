import math
import numpy as np
from scipy import stats
from scipy.stats import qmc
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from typing import List, Sequence, Optional, Tuple, Dict, Any

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.decomposition import (
    cap_variance_congruence,
    flag_threshold_bins,
    rescale_threshold_bins_congruence,
    cholesky_decomposition as _cholesky_decomposition,
    eigen_decomposition as _eigen_decomposition,
    svd_decomposition as _svd_decomposition,
    verify_cholesky_decomposition,
    verify_eigen_decomposition,
    verify_svd_decomposition,
    verify_pca_decomposition
)
from .diagnostics import (
    _diagnostics_samples_linear,
    _diagnostics_samples_log,
    _diagnostics_covariance,
    _diagnostics_endf_covariance,
    _diagnostics_endf_samples_linear,
    _diagnostics_endf_samples_log,
)


 
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _verify_sample_covariance(
    samples: np.ndarray,
    original_cov: np.ndarray,
    space: str,
    verbose: bool = True,
    logger = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Verify that the empirical covariance from samples matches the original covariance matrix.
    
    Parameters
    ----------
    samples : np.ndarray
        Generated samples of shape (n_samples, n_parameters)
    original_cov : np.ndarray
        Original covariance matrix of shape (n_parameters, n_parameters)
    space : str
        Sampling space: "linear" or "log"
    verbose : bool
        Whether to log verification results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing verification metrics
    """
    n_samples, n_params = samples.shape
    
    # Calculate empirical covariance from samples
    if space == "linear":
        # For linear space: samples = factors = 1 + X, so we need Cov(X) = Cov(factors - 1)
        centered_samples = samples - 1.0
        empirical_cov = np.cov(centered_samples.T, ddof=1)
    else:  # log space
        # For log space: samples = factors = exp(Y), we need to check Cov(log(factors))
        log_samples = np.log(np.maximum(samples, np.finfo(samples.dtype).tiny))
        # Remove the mean shift to get the underlying Y ~ N(m, Σ_log)
        # Since samples = exp(Y + m), we have log(samples) = Y + m
        # The original covariance is Cov(Y), so we need Cov(log(samples) - mean(log(samples)))
        log_samples_centered = log_samples - np.mean(log_samples, axis=0)
        empirical_cov = np.cov(log_samples_centered.T, ddof=1)
    
    # Mask non-finite entries for NaN-safe computation
    finite_mask = np.isfinite(original_cov) & np.isfinite(empirical_cov)
    n_nonfinite = int(np.count_nonzero(~finite_mask))

    # Compute Frobenius on finite entries only
    orig_clean = np.where(finite_mask, original_cov, 0.0)
    emp_clean = np.where(finite_mask, empirical_cov, 0.0)
    frobenius_original = np.linalg.norm(orig_clean, 'fro')
    frobenius_diff = np.linalg.norm(orig_clean - emp_clean, 'fro')
    relative_frobenius_error = frobenius_diff / frobenius_original if frobenius_original > 0 else 0.0

    # Calculate element-wise errors (finite only)
    abs_errors = np.abs(orig_clean - emp_clean)
    max_diagonal_error = np.max(abs_errors[np.eye(n_params, dtype=bool)])
    max_offdiag_error = np.max(abs_errors[~np.eye(n_params, dtype=bool)]) if n_params > 1 else 0.0

    # Calculate relative diagonal errors
    diagonal_original = np.diag(original_cov)
    diagonal_empirical = np.diag(empirical_cov)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diagonal_errors = np.abs(diagonal_original - diagonal_empirical) / np.abs(diagonal_original)
        relative_diagonal_errors = relative_diagonal_errors[np.isfinite(relative_diagonal_errors)]
    max_relative_diagonal_error = np.max(relative_diagonal_errors) if len(relative_diagonal_errors) > 0 else 0.0
    
    metrics = {
        'relative_frobenius_error': relative_frobenius_error,
        'max_diagonal_error': max_diagonal_error,
        'max_offdiag_error': max_offdiag_error,
        'max_relative_diagonal_error': max_relative_diagonal_error,
        'n_samples': n_samples,
        'n_parameters': n_params
    }
    
    # Quality assessment: combine Frobenius and diagonal error
    frob_pct = relative_frobenius_error * 100.0
    diag_pct = max_relative_diagonal_error * 100.0
    if frob_pct < 1.0 and diag_pct < 20.0:
        quality = "EXCELLENT"
    elif frob_pct < 5.0 and diag_pct < 50.0:
        quality = "GOOD"
    elif frob_pct < 15.0 and diag_pct < 100.0:
        quality = "FAIR"
    else:
        quality = "POOR"

    # Log results
    if verbose:
        ctx = f" [{label}]" if label else ""
        separator = "-" * 60
        def _out(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        _out(f"\n[SAMPLING] [QUALITY] Sample covariance verification{ctx}\n{separator}")
        _out(f"  Samples used: {n_samples}")
        _out(f"  Parameters: {n_params}")
        if n_nonfinite > 0:
            _out(f"  Non-finite entries masked: {n_nonfinite}")
        _out(f"  Relative Frobenius error: {frob_pct:.4f}%")
        _out(f"  Max relative diagonal error: {diag_pct:.4f}%")
        _out(f"  Max absolute off-diagonal error: {max_offdiag_error:.6e}")
        _out(f"  Sample quality assessment: {quality}")
        if n_samples < 1000:
            _out(f"  Note: Quality improves with more samples (current: {n_samples})")
        if n_samples <= n_params:
            _out(
                f"  Note: n_samples ({n_samples}) <= n_parameters ({n_params}); "
                f"empirical covariance is rank-deficient (rank <= {n_samples - 1}). "
                f"Frobenius error reflects this estimator limit, not a sampling defect — "
                f"uncertainty propagation from the perturbed files is unaffected."
            )
        _out(separator)
    
    return metrics


def _uncorrelated(
    dim: int,
    n: int,
    method: str,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw uncorrelated N(0, I) samples of shape (n, dim) via three methods:
      - 'random': plain RNG.normal
      - 'lhs': Latin Hypercube → inverse CDF
      - 'sobol': Sobol scramble → inverse CDF (with optional fast_forward)
    """
    m = method.lower()
    if m == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, dim))

    if m in ("lhs", "sobol"):
        if m == "lhs":
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
        else:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        U = sampler.random(n)
        return stats.norm.ppf(U)

    raise ValueError("method must be 'random', 'lhs' or 'sobol'")




# ----------------------------------------------------------------------
# PCA decomposition (shared by linear & log spaces)
# ----------------------------------------------------------------------
def _pca_decomposition_sampling(
    cov_mat: np.ndarray,
    n_samples: int,
    sampling_method: str,
    seed: Optional[int],
    trunc_threshold: float,
    verbose: bool,
    space: str = "log",
    logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Draw N(0, Σ) samples via PCA truncation while capturing
    `trunc_threshold` of the variance.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        (samples, eigenvalues, eigenvectors, k) where k is number of components used
    """
    # 1) Force symmetry
    T = (cov_mat + cov_mat.T) / 2.0

    # 2) Eigen-decompose
    eigvals, eigvecs = np.linalg.eigh(T)

    # 3) Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 4) Clamp negatives (numerical noise)
    eigvals = np.clip(eigvals, 0.0, None)

    # 5) Choose rank k
    total_var = eigvals.sum()
    cumvar    = np.cumsum(eigvals)
    k = int(np.searchsorted(cumvar / total_var, trunc_threshold) + 1)

    # 6) Print PCA info and verify decomposition quality
    if verbose:
        if logger:
            logger.info(f"[DECOMPOSITION] [PCA] Using k={k} components ({cumvar[k-1] / total_var:.4f} variance)")
        else:
            print(f"PCA: using k={k} components ({cumvar[k-1] / total_var:.4f} variance)")
        
        verify_pca_decomposition(
            original_matrix=T,
            eigvals=eigvals,
            eigvecs=eigvecs,
            k=k,
            space=space,
            verbose=verbose,
            logger=logger
        )

    # 7) Build transform L and draw uncorrelated Z
    Vred    = eigvecs[:, :k]
    sqrt_D  = np.sqrt(eigvals[:k])
    L       = Vred @ np.diag(sqrt_D)
    Z       = _uncorrelated(k, n_samples, sampling_method, seed)

    samples = Z @ L.T                      # shape (n_samples, p)
    return samples, eigvals, eigvecs, k



# ----------------------------------------------------------------------
#  Custom Exceptions
# ----------------------------------------------------------------------
class CovarianceFixError(Exception):
    """Exception raised when covariance matrix cannot be fixed to meet eigenvalue threshold."""
    pass

class SoftAutofixWarning(Exception):
    """Warning raised when soft autofix doesn't meet threshold but decomposition should still be attempted."""
    pass


# ----------------------------------------------------------------------
#  Main function
# ----------------------------------------------------------------------

def generate_samples(
    cov,
    n_samples: int,
    *,
    space: str = "log",          # "log" (default) or "linear"
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    autofix: Optional[str] = None,    # can be None/"soft"/"medium"/"hard"
    high_val_thresh: float = 5.0,
    accept_tol: float = -1.0e-4,
    psd_method: str = "higham",
    max_relative_std: Optional[float] = 3.0,
    mt_thresholds: Optional[Dict[int, float]] = None,
    verbose: bool = True,
    label: str = "",
) -> Tuple[np.ndarray, Optional[List[int]], Optional[Dict[str, Any]]]:
    """
    Draw multiplicative perturbation factors.

    Parameters
    ----------
    space : {"linear", "log"}
        * "linear": factors = 1 + X,   X ~ N(0, Σ_linear)
        * "log"   : factors = exp(Y),  Y ~ N(m, Σ_log) matched so that
                     Cov(factors) = Σ_linear and E[factors] = 1.
    autofix_level : {"soft", "medium", "hard"} or None/False
        If None or False, do not fix covariance. Otherwise, fix with the specified level.
    accept_tol : float
        Minimum eigenvalue threshold for accepting the covariance matrix
        
    Returns
    -------
    factors : np.ndarray
        Generated perturbation factors
    mt_numbers : Optional[List[int]]
        Final list of MT numbers (may be modified by autofix)
    fix_info : Optional[Dict[str, Any]]
        Information about covariance fixing, including removed correlations
    """
    # Try to get logger from ace_perturbation module
    try:
        from kika.sampling.ace_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None
    
    space  = space.lower()
    method = decomposition_method.lower()

    HIGH_VAR_LIN = 2.0
    HIGH_VAR_LOG = 2.0
    Z_LIMIT      = 3.0
    TRUNC_THRESHOLD = 0.999
    fix_info = None  # Initialize fix_info
    soft_autofix_failed = False  # Track if soft autofix failed to meet threshold

    # ------------------------------------------------------------------
    # 1. Fix the *linear* covariance if requested
    if autofix is not None:
        cov_fixed, fix_log = cov.fix_covariance(
            level=autofix, 
            high_val_thresh=high_val_thresh, 
            accept_tol=accept_tol,  
            verbose=verbose, 
            logger=logger
        )
        
        fix_info = fix_log  # Store the fix information
        
        # Check if covariance fixing was successful
        if not fix_log.get("converged", False):
            # For soft level, check if threshold was met
            if autofix.lower() == "soft" and not fix_log.get("soft_threshold_met", True):
                # Soft autofix didn't meet threshold, but we'll try decomposition anyway
                soft_autofix_failed = True
                min_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
                if logger:
                    logger.info(f"[COVARIANCE] [SOFT AUTOFIX] Threshold not met (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}), attempting decomposition anyway")
            else:
                # For medium/hard levels, this is a real failure
                min_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
                error_msg = (
                    f"Covariance matrix could not be fixed to meet eigenvalue threshold.\n"
                    f"  Final minimum eigenvalue: {min_eigenvalue:.4e}\n"
                    f"  Required threshold: {accept_tol:.4e}\n"
                    f"  Autofix level used: {autofix}\n"
                    f"  Suggestion: Try processing separately with a harder autofix level ('medium' or 'hard')"
                )
                
                # Log to file if logger is available
                if logger:
                    logger.error(f"[COVARIANCE] [ERROR] {error_msg}")
                else:
                    print(f"[ERROR] {error_msg}")
                    
                raise CovarianceFixError(f"min_eigenvalue={min_eigenvalue:.4e} below threshold={accept_tol:.4e}")
        
        # If we reach here and it's not a soft autofix failure, log success message
        if not soft_autofix_failed:
            final_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
            if verbose and logger:
                logger.info(f"[COVARIANCE] [SUCCESS] Matrix successfully fixed (final λ_min={final_eigenvalue:.4e})")
        
        cov_lin   = cov_fixed.covariance_matrix          # (p,p)             
        p         = cov_lin.shape[0]
        param_pairs = cov_fixed._get_param_pairs()
        num_groups  = cov_fixed.num_groups        

        if mt_numbers is not None and fix_log.get("removed_pairs"):
            # Extract removed MTs from the fix_log
            removed_pairs = fix_log.get("removed_pairs", [])
            removed_mts_from_autofix = set()
            
            # For "medium" level: look for removed block pairs and extract diagonal removals
            if autofix.lower() == "medium":
                for ra, rb in removed_pairs:
                    if ra == rb:  # Diagonal block removal means entire reaction removed
                        removed_mts_from_autofix.add(ra)
            
            # For "hard" level: look at removal_log for removed MTs
            elif autofix.lower() == "hard":
                removal_log = fix_log.get("removal_log", {})
                removed_mts_hard = removal_log.get("removed_mts", [])
                removed_mts_from_autofix.update(removed_mts_hard)
                
            if removed_mts_from_autofix:
                info_msg = f"  [INFO] MTs removed by fix_covariance: {sorted(removed_mts_from_autofix)}"
                if verbose:
                    if logger:
                        logger.info(info_msg)
                    else:
                        print(info_msg)
                        
                mt_numbers = [mt for mt in mt_numbers if mt not in removed_mts_from_autofix]
    else:
        cov_lin = cov.covariance_matrix
        p        = cov_lin.shape[0]
        param_pairs = cov._get_param_pairs()   
        num_groups  = cov.num_groups
        cov_fixed = cov.copy()   

    bins = np.asarray(energy_grid)

    # separate diagnostic for the input covariance
    cov_diagnostic_results = _diagnostics_covariance(
        cov_lin, param_pairs, num_groups, bins,
        HIGH_VAR_LIN if space == "linear" else HIGH_VAR_LOG,
        check_spd=False,
        verbose=verbose
    )

    # ------------------------------------------------------------------
    # 2. Decide which covariance to impose on the Gaussian draw
    if space == "linear":
        cov_mat = cov_lin
    elif space == "log":
        cov_mat = cov_fixed.log_covariance_matrix
    else:
        raise ValueError("space must be 'linear' or 'log'")

    # ------------------------------------------------------------------
    # 2a. Targeted rescaling of NJOY threshold-spanning bins.
    # Detect bins where a reaction threshold (from ACE σ(E)) falls inside a
    # multigroup boundary and rescale the variance to the per-MT median of
    # other groups. Runs before the global cap so the cap acts only on
    # residual outliers and Higham gets a cleaner input.
    threshold_flag_info: Optional[Dict[str, Any]] = None
    threshold_rescale_info: Optional[Dict[str, Any]] = None
    if mt_thresholds and num_groups > 0 and len(param_pairs) > 0:
        flagged_indices, targets, detection_log = flag_threshold_bins(
            cov_mat, mt_thresholds, param_pairs, num_groups, bins,
            space=space, verbose=verbose, logger=logger, label=label,
        )
        threshold_flag_info = {
            "n_flagged": len(flagged_indices),
            "detection_log": detection_log,
        }
        if flagged_indices:
            cov_mat, threshold_rescale_info = rescale_threshold_bins_congruence(
                cov_mat, flagged_indices, targets,
                param_pairs=param_pairs, num_groups=num_groups, bins=bins,
                verbose=verbose, logger=logger, label=label,
            )
        # Greppable summary metrics
        n_resc = threshold_rescale_info["n_actually_rescaled"] if threshold_rescale_info else 0
        if logger is not None:
            ctx = f" [{label}]" if label else ""
            logger.info(f">> threshold_bins_flagged{ctx} = {len(flagged_indices)}")
            logger.info(f">> threshold_bins_rescaled{ctx} = {n_resc}")

    # ------------------------------------------------------------------
    # 2b. Cap extreme variances (threshold-reaction spikes, etc.)
    if max_relative_std is not None and max_relative_std > 0:
        if space == "log":
            max_log_var = np.log(1.0 + max_relative_std ** 2)
        else:
            max_log_var = max_relative_std ** 2

        cov_mat, cap_info = cap_variance_congruence(
            cov_mat, max_log_var,
            param_pairs=param_pairs,
            num_groups=num_groups,
            bins=bins,
            verbose=verbose,
            logger=logger,
            label=label,
        )

    # ------------------------------------------------------------------
    # 3. Draw uncorrelated N(0,1)
    Z = _uncorrelated(dim=p, n=n_samples,
                      method=sampling_method, seed=seed)

    # ------------------------------------------------------------------
    # 4. Impose correlation (any of the four decompositions)
    try:
        if method == "pca":
            Y, _, _, _ = _pca_decomposition_sampling(
                cov_mat, n_samples, sampling_method, seed,
                TRUNC_THRESHOLD, verbose, space, logger
            )
        else:
            if method == "cholesky":
                L = _cholesky_decomposition(
                    space=space, psd_method=psd_method,
                    verbose=verbose, logger=logger, matrix=cov_mat,
                )
            elif method == "eigen":
                eigvals, eigvecs = _eigen_decomposition(
                    space=space, psd_method=psd_method,
                    verbose=verbose, logger=logger, matrix=cov_mat,
                )
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            elif method == "svd":
                U, S, Vt = _svd_decomposition(
                    space=space, psd_method=psd_method,
                    verbose=verbose, logger=logger, matrix=cov_mat,
                )
                L = U @ np.diag(np.sqrt(S))
            else:
                raise ValueError(
                    "decomposition_method must be 'pca', 'cholesky', 'eigen' or 'svd'"
                )
            Y = Z @ L.T
    except Exception as e:
        # If decomposition fails and we had a soft autofix failure, raise special exception
        if soft_autofix_failed:
            min_eigenvalue = fix_info.get("min_eigenvalue", float('nan'))
            error_msg = f"Soft autofix failed to meet threshold (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) and decomposition failed: {str(e)}"
            raise SoftAutofixWarning(error_msg)
        else:
            # Re-raise original exception for other cases
            raise e

    # ------------------------------------------------------------------
    # 5. Convert to multiplicative factors
    if space == "linear":
        factors = Y + 1.0

    else:  # log (moment-matched)
        m = -0.5 * np.diag(cov_mat)        # shift so mean → 1
        factors = np.exp(Y + m)            # strictly positive
    
    # ------------------------------------------------------------------
    # 6. Diagnostics of the *samples* (run before float32 cast to avoid
    #    underflow-to-zero that poisons log-space metrics with NaN)

    if space == "linear":
        sampling_diagnostic_results = _diagnostics_samples_linear(
            factors, cov_mat,
            param_pairs, num_groups,
            bins,
            Z_LIMIT, verbose
        )
    else:
        sampling_diagnostic_results = _diagnostics_samples_log(
            factors, cov_mat,
            param_pairs, num_groups,
            bins,
            Z_LIMIT, verbose
        )

    # ------------------------------------------------------------------
    # 7. Verify sample-based covariance matches original covariance
    if verbose:
        sample_verification_metrics = _verify_sample_covariance(
            samples=factors,
            original_cov=cov_mat,
            space=space,
            verbose=verbose,
            logger=logger,
            label=label,
        )

    # Convert perturbation factors to float32 for memory efficiency
    factors = factors.astype(np.float32)
    
    # Update fix_info to include soft autofix status
    if soft_autofix_failed and fix_info:
        fix_info["soft_autofix_failed"] = True
        fix_info["decomposition_succeeded"] = True
        
    return factors, mt_numbers, fix_info  # Return fix_info as third value


# ----------------------------------------------------------------------
#  ENDF-specific sampling function for MF34 angular covariance data
# ----------------------------------------------------------------------

def generate_endf_samples(
    mf34_cov,
    n_samples: int,
    *,
    space: str = "linear",          # "log" (default) or "linear"
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    psd_method: str = "higham",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]]]: 
    """
    Draw multiplicative perturbation factors for ENDF angular distribution data (MF34).
    
    This function is specifically designed for LegendreCovariance objects which contain
    covariance data for angular distributions with (isotope, reaction, legendre) triplets.
    
    Note: Unlike the standard generate_samples function, this does not include
    autofix capabilities since angular distribution covariance matrices have 
    different physical constraints than cross-section covariance matrices.

    Parameters
    ----------
    mf34_cov : LegendreCovariance
        MF34 angular distribution covariance matrix object
    n_samples : int
        Number of perturbation factor samples to generate
    space : {"linear", "log"}
        * "linear": factors = 1 + X,   X ~ N(0, Σ_linear)
        * "log"   : factors = exp(Y),  Y ~ N(m, Σ_log) matched so that
                     Cov(factors) = Σ_linear and E[factors] = 1.
    decomposition_method : str
        Method for matrix decomposition: "svd", "cholesky", "eigen", or "pca"
    sampling_method : str
        Method for generating uncorrelated samples: "sobol", "random", "lhs"
    seed : Optional[int]
        Random seed for reproducibility
    mt_numbers : Optional[Sequence[int]]
        List of MT numbers to include (for reference only, not used for filtering)
    energy_grid : Optional[Sequence[float]]
        Energy grid for diagnostics
    verbose : bool
        Whether to print diagnostic information
        
    Returns
    -------
    factors : np.ndarray
        Generated perturbation factors of shape (n_samples, n_parameters)
        where n_parameters corresponds to the flattened covariance matrix
    mt_numbers : Optional[List[int]]
        Unchanged list of MT numbers (returned for consistency with generate_samples)
    """
    # Try to get logger from endf_perturbation module
    try:
        from kika.sampling.endf_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None
    
    space = space.lower()
    method = decomposition_method.lower()

    HIGH_VAR_LIN = 2.0
    HIGH_VAR_LOG = 2.0
    Z_LIMIT = 3.0
    TRUNC_THRESHOLD = 0.999

    # ------------------------------------------------------------------
    # 1. Get covariance matrix and parameter information
    cov_lin = mf34_cov.covariance_matrix  # (p,p)
    p = cov_lin.shape[0]
    
    # For MF34 data, create parameter triplets and simplified param_pairs for diagnostics
    param_triplets = mf34_cov._get_param_triplets()  # List of (isotope, mt, legendre) triplets
    
    # Create param_pairs for diagnostics by converting triplets to (mt, l) pairs
    param_pairs = [(mt, l) for (iso, mt, l) in param_triplets]
    param_pairs = sorted(list(set(param_pairs)))
    
    # Calculate num_groups as maximum matrix size
    num_groups = max(matrix.shape[0] for matrix in mf34_cov.matrices) if mf34_cov.matrices else 0

    bins = np.asarray(energy_grid) if energy_grid is not None else None

    # Separate diagnostic for the input covariance
    if verbose:
        endf_cov_diagnostic_results = _diagnostics_endf_covariance(
            cov_lin, param_triplets, num_groups, bins,
            HIGH_VAR_LIN if space == "linear" else HIGH_VAR_LOG,
            verbose=verbose, logger=logger
        )
    else:
        endf_cov_diagnostic_results = None

    # ------------------------------------------------------------------
    # 2. Decide which covariance to impose on the Gaussian draw
    if space == "linear":
        cov_mat = cov_lin
    elif space == "log":
        cov_mat = mf34_cov.log_covariance_matrix
    else:
        raise ValueError("space must be 'linear' or 'log'")

    # ------------------------------------------------------------------
    # 3. Draw uncorrelated N(0,1)
    Z = _uncorrelated(dim=p, n=n_samples,
                      method=sampling_method, seed=seed)

    # ------------------------------------------------------------------
    # 4. Impose correlation using matrix decomposition
    if method == "pca":
        Y, _, _, _ = _pca_decomposition_sampling(
            cov_mat, n_samples, sampling_method, seed,
            TRUNC_THRESHOLD, verbose, space, logger
        )
    else:
        if method == "cholesky":
            try:
                L = mf34_cov.cholesky_decomposition(space=space, psd_method=psd_method, verbose=verbose, logger=logger)
                # Verify Cholesky decomposition quality
                if verbose:
                    verify_cholesky_decomposition(
                        original_matrix=cov_mat,
                        L=L,
                        space=space,
                        verbose=verbose,
                        logger=logger
                    )
            except Exception as chol_err:
                if verbose:
                    if logger:
                        logger.warning(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                    else:
                        print(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                raise  # Re-raise the original exception
        elif method == "eigen":
            eigvals, eigvecs = mf34_cov.eigen_decomposition(space=space, psd_method=psd_method, verbose=verbose, logger=logger)
            # Verify eigendecomposition quality
            if verbose:
                verify_eigen_decomposition(
                    original_matrix=cov_mat,
                    eigvals=eigvals,
                    eigvecs=eigvecs,
                    space=space,
                    verbose=verbose,
                    logger=logger
                )
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        elif method == "svd":
            U, S, Vt = mf34_cov.svd_decomposition(space=space, psd_method=psd_method, verbose=verbose, logger=logger)
            # Verify SVD quality
            if verbose:
                verify_svd_decomposition(
                    original_matrix=cov_mat,
                    U=U,
                    S=S,
                    Vt=Vt,
                    space=space,
                    verbose=verbose,
                    logger=logger
                )
            L = U @ np.diag(np.sqrt(S))
        else:
            raise ValueError(
                "decomposition_method must be 'pca', 'cholesky', 'eigen' or 'svd'"
            )
        Y = Z @ L.T

    # ------------------------------------------------------------------
    # 5. Convert to multiplicative factors
    if space == "linear":
        factors = Y + 1.0
    else:  # log (moment-matched)
        m = -0.5 * np.diag(cov_mat)        # shift so mean → 1
        factors = np.exp(Y + m)            # strictly positive
    
    # Convert perturbation factors to float32 for memory efficiency
    factors = factors.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Diagnostics of the samples
    if verbose:
        if space == "linear":
            endf_sampling_diagnostic_results = _diagnostics_endf_samples_linear(
                factors, cov_mat, param_triplets, num_groups,
                bins, Z_LIMIT, verbose=verbose, logger=logger
            )
        else:
            endf_sampling_diagnostic_results = _diagnostics_endf_samples_log(
                factors, cov_mat, param_triplets, num_groups,
                bins, Z_LIMIT, verbose=verbose, logger=logger
            )
            
        # ------------------------------------------------------------------
        # 7. Verify sample-based covariance matches original covariance
        sample_verification_metrics = _verify_sample_covariance(
            samples=factors,
            original_cov=cov_mat,
            space=space,
            verbose=verbose,
            logger=logger
        )
    else:
        endf_sampling_diagnostic_results = None
        
    return factors, mt_numbers, endf_sampling_diagnostic_results
