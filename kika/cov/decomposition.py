"""
Shared matrix decomposition methods for covariance matrix classes.

This module provides decomposition functionality that can be used by both
CrossSectionCovariance and LegendreCovariance classes without code duplication.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Protocol, runtime_checkable


@runtime_checkable
class CovarianceMatrixProtocol(Protocol):
    """Protocol defining the interface required for decomposition methods."""
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the linear-space covariance matrix."""
        ...
    
    @property 
    def log_covariance_matrix(self) -> np.ndarray:
        """Return the log-space covariance matrix."""
        ...


def _log_message(msg: str, logger=None, verbose: bool = True) -> None:
    """
    Helper function to log messages.
    
    Parameters
    ----------
    msg : str
        Message to log
    logger : optional
        Logger instance for file output. If provided, message is always logged to file.
    verbose : bool
        Whether to also print message to console
    """
    # Always log to file if logger is provided
    if logger is not None:
        logger.info(msg)
    
    # Only print to console if verbose is True
    if verbose:
        print(msg)


def _make_psd(
    M: np.ndarray,
    *,
    jitter_scale: float = 1e-10,
    max_jitter_ratio: float = 1e-3,
    verbose: bool = True,
    logger = None,
) -> Tuple[np.ndarray, float]:
    """
    Make matrix positive semi-definite by adding jitter to diagonal.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix to make PSD
    jitter_scale : float
        Base jitter scale factor
    max_jitter_ratio : float
        Maximum jitter relative to matrix norm
    verbose : bool
        Whether to log progress
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[np.ndarray, float]
        PSD matrix and jitter amount applied
    """
    # Force symmetry
    M_sym = (M + M.T) / 2.0
    
    # Check if already PSD
    try:
        np.linalg.cholesky(M_sym)
        if verbose:
            _log_message("[COV] [CHOLESKY] No adjustment necessary - matrix is already positive definite", logger, verbose)
        return M_sym, 0.0
    except np.linalg.LinAlgError:
        pass
    
    # Apply jitter
    eigvals = np.linalg.eigvals(M_sym)
    min_eigval = np.min(eigvals)
    
    if verbose:
        _log_message(f"[COV] Minimum eigenvalue: {min_eigval:.6e}", logger, verbose)
    
    # Calculate jitter amount
    matrix_norm = np.linalg.norm(M_sym, 'fro')
    base_jitter = jitter_scale * matrix_norm
    min_jitter = -min_eigval + base_jitter if min_eigval < 0 else base_jitter
    max_jitter = max_jitter_ratio * matrix_norm
    
    jitter = min(min_jitter, max_jitter)
    
    if verbose:
        _log_message(f"[COV] Adding jitter: {jitter:.6e}", logger, verbose)
    
    M_psd = M_sym + jitter * np.eye(M_sym.shape[0])

    return M_psd, jitter


def _robust_eigh(
    A: np.ndarray,
    *,
    label: str = "",
    verbose: bool = False,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition with SVD fallback for ill-conditioned matrices.

    Returns (eigenvalues, eigenvectors) just like ``np.linalg.eigh``.
    If ``eigh`` raises ``LinAlgError`` we fall back to SVD: for a real
    symmetric matrix ``A = V diag(s) V^T`` so ``s`` equals the absolute
    eigenvalues.  We recover the signs via ``diag(V^T A V)``.

    Sets ``_robust_eigh.used_svd = True`` on the last call that fell back.
    """
    try:
        result = np.linalg.eigh(A)
        _robust_eigh.used_svd = False
        return result
    except np.linalg.LinAlgError:
        _robust_eigh.used_svd = True
        ctx = f" ({label})" if label else ""
        if verbose:
            _log_message(
                f"[DECOMPOSITION] eigh failed{ctx} — falling back to SVD eigen-solver",
                logger, verbose,
            )
        U, s, Vt = np.linalg.svd(A)
        # Recover eigenvalue signs:  sign_i = sign(v_i^T A v_i)
        signs = np.sign(np.einsum("ij,ij->j", Vt.T, A @ Vt.T))
        signs[signs == 0] = 1.0
        return signs * s, Vt.T

_robust_eigh.used_svd = False


def _robust_eigvalsh(
    A: np.ndarray,
    *,
    label: str = "",
    verbose: bool = False,
    logger=None,
) -> np.ndarray:
    """Eigenvalues only, with SVD fallback."""
    try:
        return np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        ctx = f" ({label})" if label else ""
        if verbose:
            _log_message(
                f"[DECOMPOSITION] eigvalsh failed{ctx} — falling back to SVD eigen-solver",
                logger, verbose,
            )
        w, _ = _robust_eigh(A, label=label, verbose=False, logger=logger)
        return np.sort(w)


def cap_variance_congruence(
    cov_mat: np.ndarray,
    max_variance: float,
    *,
    param_pairs=None,
    num_groups: int = 0,
    bins=None,
    verbose: bool = True,
    logger=None,
    label: str = "",
) -> Tuple[np.ndarray, Dict]:
    """
    Cap diagonal variances via congruence transform, preserving correlations and PSD.

    For each diagonal entry ``σ²_i > max_variance``, compute a scale factor
    ``s_i = sqrt(max_variance / σ²_i)`` and apply the one-shot congruence
    transform ``Σ_capped = diag(s) @ Σ @ diag(s)``.

    Parameters
    ----------
    cov_mat : np.ndarray
        Covariance matrix (modified in-place is NOT done; a copy is returned).
    max_variance : float
        Maximum allowed diagonal variance.
    param_pairs : list of (zaid, mt) tuples, optional
        For logging which parameters were capped.
    num_groups : int
        Number of energy groups per reaction (for index→group mapping).
    bins : array-like, optional
        Energy bin edges (for logging energy ranges).
    verbose, logger, label : logging controls.

    Returns
    -------
    cov_capped : np.ndarray
        Capped covariance matrix.
    info : dict
        ``n_capped``, ``capped_entries`` list, ``max_original_variance``.
    """
    diag = np.diag(cov_mat).copy()
    n = len(diag)

    # Identify entries exceeding the cap (skip non-finite / zero)
    mask = np.isfinite(diag) & (diag > max_variance)
    n_capped = int(np.count_nonzero(mask))
    info = {
        "n_capped": n_capped,
        "max_original_variance": float(np.nanmax(diag)) if np.any(np.isfinite(diag)) else 0.0,
        "max_variance_cap": max_variance,
        "capped_entries": [],
    }

    if n_capped == 0:
        return cov_mat.copy(), info

    # Build scale vector
    s = np.ones(n)
    capped_indices = np.where(mask)[0]
    for idx in capped_indices:
        s[idx] = np.sqrt(max_variance / diag[idx])

    # Congruence transform: Σ_capped = diag(s) @ Σ @ diag(s)
    cov_capped = cov_mat * np.outer(s, s)

    # Build detailed log of capped entries
    capped_details = []
    for idx in capped_indices:
        entry = {"index": int(idx), "original_variance": float(diag[idx])}
        if param_pairs is not None and num_groups > 0:
            pair_idx, grp_idx = divmod(idx, num_groups)
            if pair_idx < len(param_pairs):
                zaid, mt = param_pairs[pair_idx]
                entry["zaid"] = int(zaid)
                entry["mt"] = int(mt)
                entry["group"] = int(grp_idx)
                if bins is not None and grp_idx < len(bins) - 1:
                    entry["energy_lo"] = float(bins[grp_idx])
                    entry["energy_hi"] = float(bins[grp_idx + 1])
        capped_details.append(entry)
    info["capped_entries"] = capped_details

    # Log
    if verbose:
        ctx = f" [{label}]" if label else ""
        separator = "-" * 60
        _log_message(
            f"\n[COVARIANCE] [VARIANCE CAP]{ctx}\n{separator}",
            logger, verbose,
        )
        _log_message(
            f"  Capped {n_capped}/{n} diagonal entries exceeding σ²={max_variance:.2f}",
            logger, verbose,
        )
        _log_message(
            f"  Max original variance: {info['max_original_variance']:.4e}",
            logger, verbose,
        )
        # Show up to 10 worst entries
        sorted_details = sorted(capped_details, key=lambda d: d["original_variance"], reverse=True)
        for d in sorted_details[:10]:
            if "mt" in d:
                e_range = ""
                if "energy_lo" in d:
                    e_range = f" [{d['energy_lo']:.2e},{d['energy_hi']:.2e}]"
                _log_message(
                    f"    MT={d['mt']}, G={d['group']}{e_range}: "
                    f"σ²={d['original_variance']:.4e} → {max_variance:.4e}",
                    logger, verbose,
                )
            else:
                _log_message(
                    f"    index={d['index']}: σ²={d['original_variance']:.4e} → {max_variance:.4e}",
                    logger, verbose,
                )
        if len(sorted_details) > 10:
            _log_message(f"    ... and {len(sorted_details) - 10} more", logger, verbose)
        _log_message(separator, logger, verbose)

    return cov_capped, info


def nearest_psd_higham(
    A: np.ndarray,
    *,
    preserve_diagonal: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-10,
    eigval_floor: float = 0.0,
    verbose: bool = True,
    logger=None,
) -> Tuple[np.ndarray, Dict]:
    """
    Find the nearest positive semi-definite matrix using Higham's alternating
    projection algorithm with Dykstra's correction.

    When ``preserve_diagonal=True`` the algorithm iterates between projecting
    onto the PSD cone and restoring the original diagonal (variances),
    converging to the nearest PSD matrix that keeps all variances unchanged.

    Reference: N.J. Higham, "Computing a nearest symmetric positive
    semidefinite matrix", Linear Algebra and its Applications, 1988.

    Parameters
    ----------
    A : np.ndarray
        Input symmetric matrix (need not be PSD).
    preserve_diagonal : bool
        If True, run the full Dykstra iteration that preserves the diagonal.
        If False, perform a single eigenvalue-clipping projection.
    max_iter : int
        Maximum number of alternating-projection iterations.
    tol : float
        Convergence tolerance on relative Frobenius change between iterations.
    eigval_floor : float
        Floor for eigenvalues in the PSD projection step (0.0 for exact PSD,
        small positive for strict PD).
    verbose : bool
        Whether to print diagnostic information.
    logger : optional
        Logger instance for file output.

    Returns
    -------
    Tuple[np.ndarray, dict]
        (X_psd, info) where info contains diagnostic metadata:
        - iterations, converged, frobenius_distance, relative_frobenius_error,
          max_diagonal_change, eigenvalue_range_before, eigenvalue_range_after,
          n_negative_eigenvalues_before
    """
    n = A.shape[0]
    A_sym = (A + A.T) / 2.0

    # Sanitise: replace NaN/Inf with zero so eigen-solvers don't choke
    bad_mask = ~np.isfinite(A_sym)
    if np.any(bad_mask):
        n_bad = int(np.count_nonzero(bad_mask))
        if verbose:
            _log_message(
                f"[PSD] [HIGHAM] Replacing {n_bad} non-finite entries with 0",
                logger, verbose,
            )
        A_sym[bad_mask] = 0.0
        A_sym = (A_sym + A_sym.T) / 2.0  # re-symmetrise after patching

    original_diag = np.diag(A_sym).copy()

    eigvals_orig = _robust_eigvalsh(A_sym, label="Higham input", verbose=verbose, logger=logger)
    min_eig_before = float(eigvals_orig.min())
    max_eig_before = float(eigvals_orig.max())
    n_negative = int(np.sum(eigvals_orig < -tol))

    # Already PSD — return early
    if min_eig_before >= -tol:
        info = {
            "iterations": 0,
            "converged": True,
            "frobenius_distance": 0.0,
            "relative_frobenius_error": 0.0,
            "max_diagonal_change": 0.0,
            "eigenvalue_range_before": (min_eig_before, max_eig_before),
            "eigenvalue_range_after": (min_eig_before, max_eig_before),
            "n_negative_eigenvalues_before": 0,
        }
        if verbose:
            _log_message(
                "[PSD] Matrix is already PSD — no projection needed",
                logger, verbose,
            )
        return A_sym, info

    if verbose:
        _log_message(
            f"[PSD] [HIGHAM] Input: {n}x{n}, {n_negative} negative eigenvalues "
            f"(λ_min={min_eig_before:.3e}, λ_max={max_eig_before:.3e})",
            logger, verbose,
        )

    # ------------------------------------------------------------------
    # Simple single-step clipping (no diagonal preservation)
    # ------------------------------------------------------------------
    if not preserve_diagonal:
        w, V = _robust_eigh(A_sym, label="Higham clip", verbose=verbose, logger=logger)
        w_clipped = np.maximum(w, eigval_floor)
        X = V @ np.diag(w_clipped) @ V.T
        X = (X + X.T) / 2.0
        eigvals_after = _robust_eigvalsh(X, label="Higham clip result", verbose=verbose, logger=logger)
        frob_dist = float(np.linalg.norm(X - A_sym, "fro"))
        frob_orig = float(np.linalg.norm(A_sym, "fro"))
        info = {
            "iterations": 1,
            "converged": True,
            "frobenius_distance": frob_dist,
            "relative_frobenius_error": frob_dist / frob_orig if frob_orig > 0 else 0.0,
            "max_diagonal_change": float(np.max(np.abs(np.diag(X) - original_diag))),
            "eigenvalue_range_before": (min_eig_before, max_eig_before),
            "eigenvalue_range_after": (float(eigvals_after[0]), float(eigvals_after[-1])),
            "n_negative_eigenvalues_before": n_negative,
        }
        if verbose:
            _log_message(
                f"[PSD] [CLIP] Single-step projection: Frobenius distance={frob_dist:.3e}, "
                f"max diag change={info['max_diagonal_change']:.3e}",
                logger, verbose,
            )
        return X, info

    # ------------------------------------------------------------------
    # Full Dykstra-Higham alternating projection (diagonal-preserving)
    # ------------------------------------------------------------------
    Y = A_sym.copy()
    D_S = np.zeros_like(A_sym)
    converged = False
    svd_fallback_logged = False  # log SVD fallback only once
    n_svd_iterations = 0  # count how many iterations needed SVD

    for k in range(max_iter):
        R = Y - D_S

        # Project onto PSD cone — only log SVD fallback on first occurrence
        w, V = _robust_eigh(
            R, label="Higham iteration",
            verbose=verbose and not svd_fallback_logged,
            logger=logger,
        )
        if _robust_eigh.used_svd:
            n_svd_iterations += 1
            svd_fallback_logged = True
        w_clipped = np.maximum(w, eigval_floor)
        X_psd = V @ np.diag(w_clipped) @ V.T

        # Dykstra correction
        D_S = X_psd - R

        # Restore original diagonal
        np.fill_diagonal(X_psd, original_diag)

        # Re-symmetrize (fill_diagonal doesn't break symmetry, but be safe)
        X_psd = (X_psd + X_psd.T) / 2.0

        # Convergence check
        change = float(np.linalg.norm(X_psd - Y, "fro"))
        norm_Y = float(np.linalg.norm(Y, "fro"))
        rel_change = change / norm_Y if norm_Y > 0 else 0.0

        Y = X_psd

        if rel_change < tol:
            converged = True
            break

    # Final PSD clean-up: the last iteration restored the diagonal which
    # may leave tiny negative eigenvalues.  Do a final eigenvalue clip
    # WITHOUT restoring the diagonal — the diagonal change is negligible
    # since the algorithm has converged.
    eigvals_final = _robust_eigvalsh(Y, label="Higham final check", verbose=verbose, logger=logger)
    if eigvals_final[0] < 0:
        w_f, V_f = _robust_eigh(Y, label="Higham final clip", verbose=verbose, logger=logger)
        w_f = np.maximum(w_f, eigval_floor)
        Y = V_f @ np.diag(w_f) @ V_f.T
        Y = (Y + Y.T) / 2.0

    # Final diagnostics
    eigvals_after = _robust_eigvalsh(Y, label="Higham result", verbose=verbose, logger=logger)
    frob_dist = float(np.linalg.norm(Y - A_sym, "fro"))
    frob_orig = float(np.linalg.norm(A_sym, "fro"))
    max_diag_change = float(np.max(np.abs(np.diag(Y) - original_diag)))
    iterations = k + 1

    info = {
        "iterations": iterations,
        "converged": converged,
        "frobenius_distance": frob_dist,
        "relative_frobenius_error": frob_dist / frob_orig if frob_orig > 0 else 0.0,
        "max_diagonal_change": max_diag_change,
        "eigenvalue_range_before": (min_eig_before, max_eig_before),
        "eigenvalue_range_after": (float(eigvals_after[0]), float(eigvals_after[-1])),
        "n_negative_eigenvalues_before": n_negative,
        "n_svd_fallback_iterations": n_svd_iterations,
    }

    if verbose:
        status = "converged" if converged else f"NOT converged (max_iter={max_iter})"
        svd_note = (
            f"\n  SVD fallback: used in {n_svd_iterations}/{iterations} iterations"
            if n_svd_iterations > 0 else ""
        )
        _log_message(
            f"\n[PSD] [HIGHAM] Diagonal-preserving projection ({status})\n"
            f"{'-' * 60}\n"
            f"  Iterations: {iterations}\n"
            f"  Relative Frobenius error: {info['relative_frobenius_error']*100:.6f}%\n"
            f"  Max diagonal change: {max_diag_change:.3e}\n"
            f"  Eigenvalue range: [{eigvals_after[0]:.3e}, {eigvals_after[-1]:.3e}]"
            f"{svd_note}\n"
            f"{'-' * 60}",
            logger, verbose,
        )

    return Y, info


def cholesky_decomposition(
    cov_obj: CovarianceMatrixProtocol = None,
    *,
    space: str = "log",
    psd_method: str = "higham",
    jitter_scale: float = 1e-10,
    max_jitter_ratio: float = 1e-3,
    verbose: bool = True,
    logger = None,
    matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Robust Cholesky factor L such that M ≈ L L^T.

    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol, optional
        Object containing covariance matrix data.
    space : str
        "linear" or "log" space for decomposition
    psd_method : str
        PSD enforcement method ("higham" or "jitter")
    jitter_scale : float
        Base jitter scale for PSD correction
    max_jitter_ratio : float
        Maximum jitter relative to matrix norm
    verbose : bool
        Whether to log progress
    logger : optional
        Logger instance for output
    matrix : np.ndarray, optional
        If provided, decompose this matrix directly instead of extracting
        from *cov_obj*. Useful when the matrix has been pre-processed
        (e.g. variance-capped).

    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor L
    """
    if matrix is not None:
        M = matrix
    else:
        M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)
    n = M.shape[0]

    if verbose:
        diag_range = f"diag ∈ [{np.min(np.diag(M)):.3e}, {np.max(np.diag(M)):.3e}]"
        _log_message(
            f"[DECOMPOSITION] Cholesky in {space} space  ({n}×{n}, {diag_range})",
            logger, verbose,
        )

    try:
        L = np.linalg.cholesky(M)
        if verbose:
            _log_message("[DECOMPOSITION] Cholesky successful (matrix was already PD)", logger, verbose)
        return L
    except np.linalg.LinAlgError:
        if psd_method == "higham":
            if verbose:
                _log_message("[DECOMPOSITION] Matrix not PD → applying Higham projection", logger, verbose)
            M_psd, psd_info = nearest_psd_higham(
                M, preserve_diagonal=True, eigval_floor=1e-14,
                verbose=verbose, logger=logger,
            )
            try:
                L = np.linalg.cholesky(M_psd)
            except np.linalg.LinAlgError:
                # Higham projection didn't produce a PD matrix (rare edge case)
                if verbose:
                    _log_message(
                        "[DECOMPOSITION] [WARNING] Cholesky still failed after Higham — "
                        "applying small jitter on projected matrix",
                        logger, verbose,
                    )
                M_psd2, jitter = _make_psd(
                    M_psd,
                    jitter_scale=jitter_scale,
                    max_jitter_ratio=max_jitter_ratio,
                    verbose=verbose,
                    logger=logger,
                )
                L = np.linalg.cholesky(M_psd2)
                if verbose:
                    _log_message(
                        f"[DECOMPOSITION] Cholesky successful after Higham + jitter ({jitter:.3e})",
                        logger, verbose,
                    )
                return L
            if verbose:
                _log_message(
                    f"[DECOMPOSITION] Cholesky successful after Higham "
                    f"({psd_info['iterations']} iter, "
                    f"Frobenius error={psd_info['relative_frobenius_error']*100:.4f}%)",
                    logger, verbose,
                )
        else:
            if verbose:
                _log_message("[DECOMPOSITION] Matrix not PD → applying jitter", logger, verbose)
            M_psd, jitter = _make_psd(
                M,
                jitter_scale=jitter_scale,
                max_jitter_ratio=max_jitter_ratio,
                verbose=verbose,
                logger=logger,
            )
            L = np.linalg.cholesky(M_psd)
            if verbose:
                _log_message(f"[DECOMPOSITION] Cholesky successful with jitter {jitter:.6e}", logger, verbose)

        return L

def eigen_decomposition(
    cov_obj: CovarianceMatrixProtocol = None,
    *,
    space: str = "log",
    clip_negatives: bool = True,
    psd_method: Optional[str] = None,
    verbose: bool = True,
    logger = None,
    matrix: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition with PSD correction.

    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol, optional
        Object containing covariance matrix data
    space : str
        "linear" or "log" space for decomposition
    clip_negatives : bool
        Deprecated. Use *psd_method* instead. Kept for backward compatibility.
    psd_method : str or None
        PSD correction method: "higham" (default), "clip", or "none".
        If None, resolved from *clip_negatives*: True → "higham", False → "none".
    verbose : bool
        Whether to log progress
    logger : optional
        Logger instance for output
    matrix : np.ndarray, optional
        If provided, decompose this matrix directly instead of extracting
        from *cov_obj*.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors
    """
    if psd_method is None:
        psd_method = "higham" if clip_negatives else "none"

    if matrix is not None:
        M = matrix
    else:
        M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)

    if psd_method == "higham":
        M, _info = nearest_psd_higham(M, preserve_diagonal=True, verbose=verbose, logger=logger)

    n = M.shape[0]
    if verbose:
        _log_message(f"[DECOMPOSITION] Eigendecomposition in {space} space ({n}×{n})", logger, verbose)

    eigvals, eigvecs = _robust_eigh(M, label="eigen decomposition", verbose=verbose, logger=logger)

    if psd_method == "clip":
        n_negative = np.sum(eigvals < 0)
        if n_negative > 0:
            min_eigval = np.min(eigvals)
            if verbose:
                _log_message(f"[COV] [EIGEN] Clipped {n_negative} negative eigenvalues (min={min_eigval:.3e})", logger, verbose)
            eigvals = np.clip(eigvals, 0.0, None)
        elif verbose:
            _log_message("[COV] [EIGEN] No negative eigenvalues found - no clipping applied", logger, verbose)

    return eigvals, eigvecs

def svd_decomposition(
    cov_obj: CovarianceMatrixProtocol = None,
    *,
    space: str = "log",
    clip_negatives: bool = True,
    psd_method: Optional[str] = None,
    verbose: bool = True,
    full_matrices: bool = False,
    logger = None,
    matrix: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD with PSD pre-processing.

    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol, optional
        Object containing covariance matrix data
    space : str
        "linear" or "log" space for decomposition
    clip_negatives : bool
        Deprecated. Use *psd_method* instead. Kept for backward compatibility.
    psd_method : str or None
        PSD correction method: "higham" (default), "clip", or "none".
        If None, resolved from *clip_negatives*: True → "higham", False → "none".
    verbose : bool
        Whether to log progress
    full_matrices : bool
        Whether to return full-sized U and V matrices
    logger : optional
        Logger instance for output
    matrix : np.ndarray, optional
        If provided, decompose this matrix directly instead of extracting
        from *cov_obj*.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        U, singular values, V^T matrices
    """
    if psd_method is None:
        psd_method = "higham" if clip_negatives else "none"

    if matrix is not None:
        M = matrix
    else:
        M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)

    n = M.shape[0]
    if verbose:
        _log_message(f"[DECOMPOSITION] SVD in {space} space ({n}×{n})", logger, verbose)

    if psd_method == "higham":
        M, _info = nearest_psd_higham(M, preserve_diagonal=True, verbose=verbose, logger=logger)
    elif psd_method == "clip":
        eigvals, eigvecs = _robust_eigh(M, label="SVD clip", verbose=verbose, logger=logger)
        n_negative = np.sum(eigvals < 0)

        if n_negative > 0:
            min_eigval = np.min(eigvals)
            if verbose:
                _log_message(f"[COV] [SVD] Clipped {n_negative} negative eigenvalues before SVD (min={min_eigval:.3e})", logger, verbose)
            eigvals_clipped = np.clip(eigvals, 0.0, None)
            M = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        elif verbose:
            _log_message("[COV] [SVD] No negative eigenvalues - applying SVD directly", logger, verbose)

    U, S, Vt = np.linalg.svd(M, full_matrices=full_matrices)

    return U, S, Vt


def compute_correlation(
    cov_obj: CovarianceMatrixProtocol,
    *,
    clip: bool = False,
    force_diagonal: bool = True
) -> np.ndarray:
    """
    Compute correlation matrix from covariance matrix.
    
    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol
        Object containing covariance matrix data
    clip : bool
        Whether to clip correlations to [-1, 1] range
    force_diagonal : bool
        Whether to force diagonal elements to 1.0
        
    Returns
    -------
    np.ndarray
        Correlation matrix with optional clipping and diagonal forcing
    """
    cov = cov_obj.covariance_matrix
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)

    # pure division, will give inf/nan where denom==0
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / denom

    # mask all undefined entries
    corr[~np.isfinite(corr)] = np.nan

    if force_diagonal:
        # put ones on the diagonal, even if variance was zero
        np.fill_diagonal(corr, 1.0)

    if clip:
        # clip into [-1,1], leaving nan alone
        corr = np.where(np.isfinite(corr),
                        np.clip(corr, -1.0, 1.0),
                        np.nan)

    return corr


# ----------------------------------------------------------------------
# Decomposition Quality Verification Functions
# ----------------------------------------------------------------------

def _verify_decomposition_quality(
    original_matrix: np.ndarray,
    reconstructed_matrix: np.ndarray,
    method_name: str,
    space: str,
    verbose: bool = True,
    logger = None,
) -> Tuple[float, float, float]:
    """
    Verify the quality of a matrix decomposition by comparing reconstruction.
    
    Parameters
    ----------
    original_matrix : np.ndarray
        Original covariance matrix
    reconstructed_matrix : np.ndarray
        Reconstructed matrix from decomposition
    method_name : str
        Name of decomposition method for logging
    space : str
        Space ("linear" or "log") for logging
    verbose : bool
        Whether to log results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[float, float, float]
        Relative Frobenius error (%), max diagonal error (%), max off-diagonal error (%)
    """
    # Mask non-finite entries (NaN/Inf from log-transform of zero-variance params)
    finite_mask = np.isfinite(original_matrix) & np.isfinite(reconstructed_matrix)
    n_nonfinite = int(np.count_nonzero(~finite_mask))

    orig_clean = np.where(finite_mask, original_matrix, 0.0)
    recon_clean = np.where(finite_mask, reconstructed_matrix, 0.0)
    diff_matrix = recon_clean - orig_clean

    # Relative Frobenius norm error
    frob_orig = np.linalg.norm(orig_clean, ord='fro')
    frob_diff = np.linalg.norm(diff_matrix, ord='fro')
    frob_rel_error = (frob_diff / frob_orig) * 100.0 if frob_orig > 0 else 0.0

    # Diagonal reconstruction errors
    diag_orig = np.diag(original_matrix)
    diag_recon = np.diag(reconstructed_matrix)
    diag_errors = np.abs(np.nan_to_num(diag_recon, nan=0.0) - np.nan_to_num(diag_orig, nan=0.0))

    # Relative diagonal errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_rel_errors = np.where(
            np.isfinite(diag_orig) & (diag_orig != 0),
            (diag_errors / np.abs(diag_orig)) * 100.0,
            0.0
        )
    max_diag_error = np.max(diag_rel_errors)

    # Off-diagonal reconstruction errors
    n = original_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    max_off_diag_error = np.max(np.abs(diff_matrix[off_diag_mask])) / frob_orig * 100.0 if frob_orig > 0 else 0.0

    # Quality assessment: combine Frobenius and diagonal error
    if frob_rel_error < 1e-10 and max_diag_error < 1.0:
        quality = "EXCELLENT"
    elif frob_rel_error < 1e-6 and max_diag_error < 10.0:
        quality = "VERY GOOD"
    elif frob_rel_error < 1e-3 and max_diag_error < 50.0:
        quality = "GOOD"
    elif frob_rel_error < 1e-1 and max_diag_error < 100.0:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    # Log results
    if verbose:
        separator = "-" * 60
        _log_message(f"\n[DECOMPOSITION] [QUALITY] {method_name.upper()} in {space} space\n{separator}", logger, verbose)
        if n_nonfinite > 0:
            _log_message(f"  Non-finite entries masked: {n_nonfinite}", logger, verbose)
        _log_message(f"  Relative Frobenius error: {frob_rel_error:.6f}%", logger, verbose)
        _log_message(f"  Max relative diagonal error: {max_diag_error:.6f}%", logger, verbose)
        _log_message(f"  Max relative off-diagonal error: {max_off_diag_error:.6f}%", logger, verbose)
        _log_message(f"  Quality assessment: {quality}", logger, verbose)
        _log_message(separator, logger, verbose)

    return frob_rel_error, max_diag_error, max_off_diag_error


def verify_cholesky_decomposition(
    original_matrix: np.ndarray,
    L: np.ndarray,
    space: str,
    verbose: bool = True,
    logger = None,
) -> Tuple[float, float, float]:
    """
    Verify Cholesky decomposition quality by reconstructing L @ L.T.
    
    Parameters
    ----------
    original_matrix : np.ndarray
        Original covariance matrix
    L : np.ndarray
        Cholesky factor (lower triangular)
    space : str
        Space ("linear" or "log") for logging
    verbose : bool
        Whether to log results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[float, float, float]
        Relative Frobenius error (%), max diagonal error (%), max off-diagonal error (%)
    """
    reconstructed = L @ L.T
    return _verify_decomposition_quality(
        original_matrix, reconstructed, "Cholesky", space, verbose, logger
    )


def verify_eigen_decomposition(
    original_matrix: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    space: str,
    verbose: bool = True,
    logger = None,
) -> Tuple[float, float, float]:
    """
    Verify eigendecomposition quality by reconstructing V @ Λ @ V.T.
    
    Parameters
    ----------
    original_matrix : np.ndarray
        Original covariance matrix
    eigvals : np.ndarray
        Eigenvalues
    eigvecs : np.ndarray
        Eigenvectors
    space : str
        Space ("linear" or "log") for logging
    verbose : bool
        Whether to log results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[float, float, float]
        Relative Frobenius error (%), max diagonal error (%), max off-diagonal error (%)
    """
    reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return _verify_decomposition_quality(
        original_matrix, reconstructed, "Eigendecomposition", space, verbose, logger
    )


def verify_svd_decomposition(
    original_matrix: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    space: str,
    verbose: bool = True,
    logger = None,
) -> Tuple[float, float, float]:
    """
    Verify SVD quality by reconstructing U @ Σ @ V.T.
    
    Parameters
    ----------
    original_matrix : np.ndarray
        Original covariance matrix
    U : np.ndarray
        Left singular vectors
    S : np.ndarray
        Singular values
    Vt : np.ndarray
        Right singular vectors (transposed)
    space : str
        Space ("linear" or "log") for logging
    verbose : bool
        Whether to log results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[float, float, float]
        Relative Frobenius error (%), max diagonal error (%), max off-diagonal error (%)
    """
    reconstructed = U @ np.diag(S) @ Vt
    return _verify_decomposition_quality(
        original_matrix, reconstructed, "SVD", space, verbose, logger
    )


def verify_pca_decomposition(
    original_matrix: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    k: int,
    space: str,
    verbose: bool = True,
    logger = None,
) -> Tuple[float, float, float]:
    """
    Verify PCA decomposition quality by reconstructing the truncated approximation.
    
    Parameters
    ----------
    original_matrix : np.ndarray
        Original covariance matrix
    eigvals : np.ndarray
        All eigenvalues (sorted descending)
    eigvecs : np.ndarray
        All eigenvectors (sorted by descending eigenvalue)
    k : int
        Number of components kept in truncation
    space : str
        Space ("linear" or "log") for logging
    verbose : bool
        Whether to log results
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[float, float, float]
        Relative Frobenius error (%), max diagonal error (%), max off-diagonal error (%)
    """
    # Reconstruct using only the first k components
    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]
    reconstructed = eigvecs_k @ np.diag(eigvals_k) @ eigvecs_k.T
    
    if verbose:
        # Also report variance captured
        total_var = np.sum(eigvals)
        captured_var = np.sum(eigvals_k)
        var_fraction = captured_var / total_var if total_var > 0 else 0.0
        
        separator = "-" * 60
        msg_header = f"\n[DECOMPOSITION] [QUALITY] PCA in {space} space\n{separator}"
        _log_message(msg_header, logger, verbose)
        
        msg_components = f"  Components used: {k}/{len(eigvals)}"
        _log_message(msg_components, logger, verbose)
        
        msg_variance = f"  Variance captured: {var_fraction:.6f} ({var_fraction*100:.4f}%)"
        _log_message(msg_variance, logger, verbose)
    
    return _verify_decomposition_quality(
        original_matrix, reconstructed, "PCA", space, verbose, logger
    )


def verify_psd_projection(
    original_matrix: np.ndarray,
    projected_matrix: np.ndarray,
    info: Dict,
    verbose: bool = True,
    logger=None,
) -> Tuple[float, float, float]:
    """
    Verify PSD projection quality by comparing original and projected matrices.

    Reports Frobenius distance, maximum diagonal change, and maximum
    correlation change between the original and projected matrices.

    Parameters
    ----------
    original_matrix : np.ndarray
        Original (possibly non-PSD) covariance matrix.
    projected_matrix : np.ndarray
        PSD-projected matrix.
    info : dict
        Info dict returned by ``nearest_psd_higham()``.
    verbose : bool
        Whether to log results.
    logger : optional
        Logger instance for file output.

    Returns
    -------
    Tuple[float, float, float]
        (relative_frobenius_error_pct, max_diagonal_change_pct, max_correlation_change)
    """
    frob_orig = np.linalg.norm(original_matrix, "fro")
    frob_dist = info.get("frobenius_distance", np.linalg.norm(projected_matrix - original_matrix, "fro"))
    rel_frob_pct = (frob_dist / frob_orig * 100.0) if frob_orig > 0 else 0.0

    diag_orig = np.diag(original_matrix)
    diag_proj = np.diag(projected_matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        diag_rel_err = np.where(
            np.abs(diag_orig) > 1e-30,
            np.abs(diag_proj - diag_orig) / np.abs(diag_orig) * 100.0,
            0.0,
        )
    max_diag_pct = float(np.max(diag_rel_err))

    # Correlation change
    std_orig = np.sqrt(np.maximum(diag_orig, 0.0))
    std_proj = np.sqrt(np.maximum(diag_proj, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        outer_orig = np.outer(std_orig, std_orig)
        outer_proj = np.outer(std_proj, std_proj)
        corr_orig = np.where(outer_orig > 1e-30, original_matrix / outer_orig, 0.0)
        corr_proj = np.where(outer_proj > 1e-30, projected_matrix / outer_proj, 0.0)
    np.fill_diagonal(corr_orig, 1.0)
    np.fill_diagonal(corr_proj, 1.0)
    mask = np.triu(np.ones_like(corr_orig, dtype=bool), k=1)
    max_corr_change = float(np.max(np.abs(corr_proj[mask] - corr_orig[mask]))) if mask.any() else 0.0

    if verbose:
        separator = "-" * 60
        _log_message(
            f"\n[PSD] [PROJECTION QUALITY]\n{separator}\n"
            f"  Relative Frobenius error: {rel_frob_pct:.6f}%\n"
            f"  Max diagonal error: {max_diag_pct:.6f}%\n"
            f"  Max off-diagonal correlation change: {max_corr_change:.6f}\n"
            f"  Iterations: {info.get('iterations', '?')}\n"
            f"  Converged: {info.get('converged', '?')}\n"
            f"{separator}",
            logger, verbose,
        )

    return rel_frob_pct, max_diag_pct, max_corr_change

