"""
Shared matrix decomposition methods for covariance matrix classes.

This module provides decomposition functionality that can be used by both
CrossSectionCovariance and LegendreCovariance classes without code duplication.
"""

import numpy as np
from typing import Tuple, Protocol, runtime_checkable


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


def cholesky_decomposition(
    cov_obj: CovarianceMatrixProtocol,
    *,
    space: str = "log",
    jitter_scale: float = 1e-10,
    max_jitter_ratio: float = 1e-3,
    verbose: bool = True,
    logger = None,
) -> np.ndarray:
    """
    Robust Cholesky factor L such that M ≈ L L^T.
    
    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol
        Object containing covariance matrix data
    space : str
        "linear" or "log" space for decomposition
    jitter_scale : float
        Base jitter scale for PSD correction
    max_jitter_ratio : float
        Maximum jitter relative to matrix norm
    verbose : bool
        Whether to log progress
    logger : optional
        Logger instance for output
        
    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor L
    """
    M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)
    
    if verbose:
        _log_message(f"[DECOMPOSITION] Computing Cholesky decomposition in {space} space", logger, verbose)
    
    try:
        L = np.linalg.cholesky(M)
        if verbose:
            _log_message("[DECOMPOSITION] Cholesky decomposition successful", logger, verbose)
        return L
    except np.linalg.LinAlgError:
        if verbose:
            _log_message("[COV] Matrix not positive definite, applying jitter", logger, verbose)
        
        M_psd, jitter = _make_psd(
            M,
            jitter_scale=jitter_scale,
            max_jitter_ratio=max_jitter_ratio,
            verbose=verbose,
            logger=logger
        )
        
        L = np.linalg.cholesky(M_psd)
        if verbose:
            _log_message(f"[DECOMPOSITION] Cholesky decomposition successful with jitter {jitter:.6e}", logger, verbose)
        
        return L

def eigen_decomposition(
    cov_obj: CovarianceMatrixProtocol,
    *,
    space: str = "log",
    clip_negatives: bool = True,
    verbose: bool = True,
    logger = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition with optional clipping instead of jitter.
    
    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol
        Object containing covariance matrix data
    space : str
        "linear" or "log" space for decomposition
    clip_negatives : bool
        Whether to clip negative eigenvalues to zero
    verbose : bool
        Whether to log progress
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors
    """
    M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)
    
    if verbose:
        _log_message(f"[DECOMPOSITION] Computing eigendecomposition in {space} space", logger, verbose)
    
    eigvals, eigvecs = np.linalg.eigh(M)
    
    if clip_negatives:
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
    cov_obj: CovarianceMatrixProtocol,
    *,
    space: str = "log",
    clip_negatives: bool = True,
    verbose: bool = True,
    full_matrices: bool = False,
    logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD with pre-clipping using eigendecomposition.
    
    Parameters
    ----------
    cov_obj : CovarianceMatrixProtocol
        Object containing covariance matrix data
    space : str
        "linear" or "log" space for decomposition
    clip_negatives : bool
        Whether to clip negative eigenvalues before SVD
    verbose : bool
        Whether to log progress
    full_matrices : bool
        Whether to return full-sized U and V matrices
    logger : optional
        Logger instance for output
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        U, singular values, V^T matrices
    """
    M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)
    
    if verbose:
        _log_message(f"[DECOMPOSITION] Computing SVD in {space} space", logger, verbose)
    
    if clip_negatives:
        # Pre-process with eigendecomposition to ensure positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(M)
        n_negative = np.sum(eigvals < 0)
        
        if n_negative > 0:
            min_eigval = np.min(eigvals)
            if verbose:
                _log_message(f"[COV] [SVD] Clipped {n_negative} negative eigenvalues before SVD (min={min_eigval:.3e})", logger, verbose)
            
            # Reconstruct matrix with clipped eigenvalues
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
    # Compute errors
    diff_matrix = reconstructed_matrix - original_matrix
    
    # Relative Frobenius norm error
    frob_orig = np.linalg.norm(original_matrix, ord='fro')
    frob_diff = np.linalg.norm(diff_matrix, ord='fro')
    frob_rel_error = (frob_diff / frob_orig) * 100.0 if frob_orig > 0 else 0.0
    
    # Diagonal reconstruction errors
    diag_orig = np.diag(original_matrix)
    diag_recon = np.diag(reconstructed_matrix)
    diag_errors = np.abs(diag_recon - diag_orig)
    
    # Relative diagonal errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_rel_errors = np.where(
            diag_orig != 0,
            (diag_errors / np.abs(diag_orig)) * 100.0,
            0.0
        )
    max_diag_error = np.max(diag_rel_errors)
    
    # Off-diagonal reconstruction errors
    n = original_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    
    off_diag_orig = original_matrix[off_diag_mask]
    off_diag_diff = diff_matrix[off_diag_mask]
    
    # Max absolute off-diagonal error relative to Frobenius norm of original
    max_off_diag_error = np.max(np.abs(off_diag_diff)) / frob_orig * 100.0 if frob_orig > 0 else 0.0
    
    # Log results
    if verbose:
        separator = "-" * 60
        msg_header = f"\n[DECOMPOSITION] [QUALITY] {method_name.upper()} in {space} space\n{separator}"
        _log_message(msg_header, logger, verbose)
        
        msg_frob = f"  Relative Frobenius error: {frob_rel_error:.6f}%"
        _log_message(msg_frob, logger, verbose)
        
        msg_diag = f"  Max diagonal error: {max_diag_error:.6f}%"
        _log_message(msg_diag, logger, verbose)
        
        msg_off_diag = f"  Max off-diagonal error: {max_off_diag_error:.6f}%"
        _log_message(msg_off_diag, logger, verbose)
        
        # Quality assessment
        if frob_rel_error < 1e-10:
            quality = "EXCELLENT"
        elif frob_rel_error < 1e-6:
            quality = "VERY GOOD"
        elif frob_rel_error < 1e-3:
            quality = "GOOD"
        elif frob_rel_error < 1e-1:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
            
        msg_quality = f"  Quality assessment: {quality}"
        _log_message(msg_quality, logger, verbose)
        
        msg_footer = f"{separator}"
        _log_message(msg_footer, logger, verbose)
    
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

