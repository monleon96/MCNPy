"""
Shared matrix decomposition methods for covariance matrix classes.

This module provides decomposition functionality that can be used by both
CrossSectionCovariance and LegendreCovariance classes without code duplication.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Protocol, runtime_checkable


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


# Threshold for psd_method="auto" mode: if |λ_min|/λ_max < this, use clip
# (single eigendecomp, fast); otherwise fall through to Higham (iterative,
# preserves diagonal). At 1e-2, NJOY/multigroup covariances with shallow
# negatives (typical: ratio 1e-4..5e-3) take the fast path; only genuinely
# indefinite matrices (ratio ≳ few %) pay for Higham.
_PSD_AUTO_THRESHOLD: float = 1e-2

# When Higham hits max_iter with relative_frobenius_error above this, the
# Cholesky path swaps in clip on the original matrix instead of accepting
# the unconverged projection.
_HIGHAM_FALLBACK_FROB_THRESHOLD: float = 1e-2

_VALID_PSD_METHODS = frozenset({"auto", "higham", "clip", "clip_rescale", "none"})


def _validate_psd_method(psd_method: str) -> None:
    """Raise ValueError if psd_method is not a recognized option."""
    if psd_method not in _VALID_PSD_METHODS:
        raise ValueError(
            f"psd_method must be one of {sorted(_VALID_PSD_METHODS)}, "
            f"got {psd_method!r}"
        )


def _resolve_psd_auto(
    M: np.ndarray,
    *,
    verbose: bool = True,
    logger=None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Decide between clip and Higham based on |λ_min|/λ_max ratio.

    Computes one eigendecomposition on a NaN/Inf-cleaned copy of *M*
    (mirroring Higham's internal pre-pass), so the ratio is meaningful
    even for the sparse multigroup covariances that have non-finite
    rows for below-threshold reactions. The eigvals/eigvecs returned
    are from this cleaned matrix, so callers reusing them on the clip
    path produce a finite, PSD result by construction.

    Returns
    -------
    resolved_method : {"clip", "higham"}
    eigvals, eigvecs : eigendecomposition of the cleaned matrix (the
        caller reuses these on the clip path; Higham recomputes its
        own internally).
    """
    if not np.isfinite(M).all():
        n_nonfinite = int(np.sum(~np.isfinite(M)))
        if verbose:
            _log_message(
                f"[PSD] [AUTO] Replacing {n_nonfinite} non-finite entries with 0 "
                "(matches Higham's internal repair)",
                logger, verbose,
            )
        M_clean = np.where(np.isfinite(M), M, 0.0)
    else:
        M_clean = M

    eigvals, eigvecs = _robust_eigh(M_clean, label="auto", verbose=verbose, logger=logger)
    lam_max = max(float(eigvals.max()), 1e-300)
    lam_min_neg = max(-float(eigvals.min()), 0.0)
    ratio = lam_min_neg / lam_max
    resolved = "clip" if ratio < _PSD_AUTO_THRESHOLD else "higham"
    if verbose:
        _log_message(
            f"[PSD] [AUTO] ratio={ratio:.3e} (thresh={_PSD_AUTO_THRESHOLD:.0e}) "
            f"-> {resolved}",
            logger, verbose,
        )
    return resolved, eigvals, eigvecs


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


def flag_threshold_bins(
    cov_mat: np.ndarray,
    mt_thresholds: Dict[int, float],
    param_pairs: Sequence[Tuple[int, int]],
    num_groups: int,
    bins: Sequence[float],
    *,
    space: str = "log",
    min_groups_for_median: int = 3,
    verbose: bool = True,
    logger=None,
    label: str = "",
) -> Tuple[List[int], List[float], List[Dict]]:
    """
    Identify threshold-spanning bins from per-MT reaction thresholds (e.g.,
    extracted from ACE σ(E)) and compute a per-bin replacement variance.

    For each (zaid, mt) entry in ``param_pairs`` whose MT appears in
    ``mt_thresholds``, locate the unique group ``g`` such that
    ``bins[g] <= E_thresh < bins[g+1]``. That bin straddles the reaction
    threshold and is the canonical NJOY artifact location: ⟨σ⟩ is pulled
    down by the below-threshold tail, blowing up the relative uncertainty.

    The replacement target for that bin's diagonal variance is the median of
    the diagonal over the same MT's *other* groups (finite, positive).
    Returns indices and targets only — actual rescaling is performed by
    ``rescale_threshold_bins_congruence``.

    Parameters
    ----------
    cov_mat : (n,n) np.ndarray
        Covariance matrix (linear or log space; matching ``space``).
    mt_thresholds : dict
        ``{mt: E_threshold}`` in the same energy units as ``bins`` (MeV here).
    param_pairs : list of (zaid, mt) tuples
        Maps row/col blocks to (isotope, reaction).
    num_groups : int
        Energy groups per (zaid, mt) block.
    bins : array-like
        Group boundaries, length ``num_groups + 1``.
    space : {"log", "linear"}
        Only used in the log lines (interpretation of σ²).
    min_groups_for_median : int
        Skip a flagged bin when fewer than this many same-MT neighbors are
        usable; the bin remains for the global cap to handle.

    Returns
    -------
    flagged_indices : list of int
        Flat covariance indices for bins flagged as threshold-spanning.
    targets : list of float
        Target variance per flagged index (one-to-one with ``flagged_indices``).
    detection_log : list of dict
        Per-flagged-bin detail (zaid, mt, group, energy_lo, energy_hi,
        E_thresh, current_variance, target_variance, n_groups_used).
    """
    bins_arr = np.asarray(bins, dtype=float)
    diag = np.diag(cov_mat)

    flagged_indices: List[int] = []
    targets: List[float] = []
    detection_log: List[Dict] = []
    skipped_too_few: List[Dict] = []

    # Pre-compute per-isotope finite-positive diagonal pool, used as the
    # tier-3 fallback when an MT has 0 same-MT neighbors with usable data
    # (e.g., a reaction with covariance only above its own threshold and
    # the threshold lies in the highest-defined group).
    isotope_diag_pool: Dict[int, np.ndarray] = {}
    for p_idx, (z2, _mt2) in enumerate(param_pairs):
        block = diag[p_idx * num_groups:(p_idx + 1) * num_groups]
        good = block[np.isfinite(block) & (block > 0)]
        if good.size:
            isotope_diag_pool.setdefault(int(z2), []).append(good)
    isotope_diag_pool = {
        z: np.concatenate(arrs) for z, arrs in isotope_diag_pool.items()
    }

    for pair_idx, (zaid, mt) in enumerate(param_pairs):
        E_thresh = mt_thresholds.get(int(mt))
        if E_thresh is None:
            continue
        # Locate the group containing E_thresh
        # bins[g] <= E_thresh < bins[g+1]  ->  g = searchsorted(bins, E, 'right') - 1
        g = int(np.searchsorted(bins_arr, E_thresh, side="right") - 1)
        if g < 0 or g >= num_groups:
            continue

        flat = pair_idx * num_groups + g
        current_var = float(diag[flat])
        if not np.isfinite(current_var) or current_var <= 0.0:
            continue  # nothing to rescale

        # Collect same-MT diagonal entries excluding the flagged group
        block_start = pair_idx * num_groups
        block_diag = diag[block_start: block_start + num_groups]
        other_mask = np.ones(num_groups, dtype=bool)
        other_mask[g] = False
        other_vals = block_diag[other_mask]
        finite_pos = other_vals[np.isfinite(other_vals) & (other_vals > 0)]

        # Tiered fallback: same-MT median when we have ≥ min_groups, else
        # whatever same-MT we have, else the isotope-wide median. Each tier
        # is logged so the source of "target" is auditable.
        if finite_pos.size >= min_groups_for_median:
            target = float(np.median(finite_pos))
            n_used = int(finite_pos.size)
            target_source = "same_mt_median"
        elif finite_pos.size >= 1:
            target = float(np.median(finite_pos))
            n_used = int(finite_pos.size)
            target_source = "same_mt_few"
        else:
            iso_pool = isotope_diag_pool.get(int(zaid))
            if iso_pool is None or iso_pool.size == 0:
                skipped_too_few.append({
                    "zaid": int(zaid), "mt": int(mt), "group": g,
                    "n_usable": 0,
                })
                continue
            target = float(np.median(iso_pool))
            n_used = int(iso_pool.size)
            target_source = "isotope_median"

        e_lo = float(bins_arr[g]) if g < len(bins_arr) - 1 else float("nan")
        e_hi = float(bins_arr[g + 1]) if g < len(bins_arr) - 1 else float("nan")

        flagged_indices.append(flat)
        targets.append(target)
        detection_log.append({
            "index": flat,
            "zaid": int(zaid),
            "mt": int(mt),
            "group": g,
            "energy_lo": e_lo,
            "energy_hi": e_hi,
            "E_thresh": float(E_thresh),
            "current_variance": current_var,
            "target_variance": target,
            "n_groups_used": n_used,
            "target_source": target_source,
        })

    if verbose or logger is not None:
        ctx = f" [{label}]" if label else ""
        sep = "-" * 60
        if flagged_indices:
            _log_message(f"\n[COVARIANCE] [THRESHOLD-BIN DETECTION]{ctx}\n{sep}", logger, verbose)
            _log_message(
                f"  Flagged {len(flagged_indices)} bin(s) as threshold-spanning "
                f"(using ACE σ(E) thresholds, space={space}):",
                logger, verbose,
            )
            for d in detection_log:
                src = d.get("target_source", "same_mt_median")
                src_label = {
                    "same_mt_median": f"median of {d['n_groups_used']} same-MT groups",
                    "same_mt_few": f"median of {d['n_groups_used']} same-MT group(s) [fallback]",
                    "isotope_median": f"isotope-wide median of {d['n_groups_used']} groups [fallback]",
                }.get(src, f"median of {d['n_groups_used']} groups")
                _log_message(
                    f"    MT={d['mt']}, G={d['group']} "
                    f"[{d['energy_lo']:.2e},{d['energy_hi']:.2e}] MeV: "
                    f"E_thresh={d['E_thresh']:.3e}, "
                    f"σ²={d['current_variance']:.3e}, "
                    f"target={d['target_variance']:.3e} "
                    f"({src_label})",
                    logger, verbose,
                )
            if skipped_too_few:
                _log_message(
                    f"  Skipped {len(skipped_too_few)} bin(s) with fewer than "
                    f"{min_groups_for_median} usable same-MT neighbors:",
                    logger, verbose,
                )
                for s in skipped_too_few:
                    _log_message(
                        f"    MT={s['mt']}, G={s['group']}: only {s['n_usable']} usable",
                        logger, verbose,
                    )
            _log_message(sep, logger, verbose)
        else:
            # No flags — emit a single quiet line so it's visible in the log
            _log_message(
                f"  [COVARIANCE] [THRESHOLD-BIN DETECTION]{ctx}: 0 bins flagged "
                f"({len(mt_thresholds)} MT thresholds checked)",
                logger, verbose,
            )

    return flagged_indices, targets, detection_log


def rescale_threshold_bins_congruence(
    cov_mat: np.ndarray,
    flagged_indices: Sequence[int],
    targets: Sequence[float],
    *,
    param_pairs=None,
    num_groups: int = 0,
    bins=None,
    verbose: bool = True,
    logger=None,
    label: str = "",
) -> Tuple[np.ndarray, Dict]:
    """
    Rescale specific diagonal entries to per-index target variances via a
    congruence transform. One-sided: never inflates a variance.

    For each flagged index ``i`` with target ``t_i``:
        s_i = sqrt(t_i / σ²_i)   if σ²_i > t_i
              1.0                 otherwise
    Then ``Σ' = diag(s) @ Σ @ diag(s)``. PSD-preserving, correlation-preserving.

    This complements ``cap_variance_congruence`` (which applies a single global
    cap): use this for *targeted* rescaling of bins identified as artifacts
    (e.g. NJOY threshold-spanning bins flagged by ``flag_threshold_bins``).

    Parameters
    ----------
    cov_mat : (n,n) np.ndarray
        Covariance matrix. A copy is returned; input is not modified.
    flagged_indices : sequence of int
        Flat covariance indices to (potentially) rescale.
    targets : sequence of float
        Target variance for each flagged index, one-to-one with
        ``flagged_indices``.
    param_pairs, num_groups, bins : optional
        For per-entry logging (zaid/mt/group/energy ranges).
    verbose, logger, label : logging controls.

    Returns
    -------
    cov_rescaled : np.ndarray
        Rescaled covariance.
    info : dict
        ``n_flagged``, ``n_actually_rescaled``, ``rescaled_entries`` (list of
        dicts with index, zaid, mt, group, energy_lo, energy_hi,
        original_variance, target_variance, scale_factor).
    """
    n = cov_mat.shape[0]
    flagged = list(flagged_indices)
    targets = list(targets)
    if len(flagged) != len(targets):
        raise ValueError("flagged_indices and targets must have equal length")

    info: Dict = {
        "n_flagged": len(flagged),
        "n_actually_rescaled": 0,
        "rescaled_entries": [],
    }

    if not flagged:
        return cov_mat.copy(), info

    diag = np.diag(cov_mat)
    s = np.ones(n)
    rescaled_details: List[Dict] = []

    for idx, target in zip(flagged, targets):
        cur = float(diag[idx])
        if not np.isfinite(cur) or cur <= 0.0 or not np.isfinite(target) or target <= 0.0:
            continue
        if cur <= target:
            continue  # one-sided: never inflate
        scale = float(np.sqrt(target / cur))
        s[idx] = scale

        entry = {
            "index": int(idx),
            "original_variance": cur,
            "target_variance": float(target),
            "scale_factor": scale,
        }
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
        rescaled_details.append(entry)

    info["n_actually_rescaled"] = len(rescaled_details)
    info["rescaled_entries"] = rescaled_details

    if not rescaled_details:
        return cov_mat.copy(), info

    cov_rescaled = cov_mat * np.outer(s, s)

    if verbose or logger is not None:
        ctx = f" [{label}]" if label else ""
        sep = "-" * 60
        _log_message(f"\n[COVARIANCE] [THRESHOLD-BIN RESCALE]{ctx}\n{sep}", logger, verbose)
        _log_message(
            f"  Rescaled {len(rescaled_details)}/{len(flagged)} flagged bins "
            f"(others already at or below target):",
            logger, verbose,
        )
        for d in rescaled_details:
            if "mt" in d:
                e_range = ""
                if "energy_lo" in d:
                    e_range = f" [{d['energy_lo']:.2e},{d['energy_hi']:.2e}]"
                _log_message(
                    f"    MT={d['mt']}, G={d['group']}{e_range}: "
                    f"σ²={d['original_variance']:.3e} → {d['target_variance']:.3e}, "
                    f"s={d['scale_factor']:.3f}",
                    logger, verbose,
                )
            else:
                _log_message(
                    f"    index={d['index']}: "
                    f"σ²={d['original_variance']:.3e} → {d['target_variance']:.3e}, "
                    f"s={d['scale_factor']:.3f}",
                    logger, verbose,
                )
        _log_message(sep, logger, verbose)

    return cov_rescaled, info


def flag_outlier_variance_bins(
    cov_mat: np.ndarray,
    param_pairs: Sequence[Tuple[int, int]],
    num_groups: int,
    bins: Sequence[float],
    *,
    outlier_factor: float = 1000.0,
    min_groups_for_median: int = 4,
    skip_indices: Optional[Sequence[int]] = None,
    space: str = "log",
    verbose: bool = True,
    logger=None,
    label: str = "",
) -> Tuple[List[int], List[float], List[Dict]]:
    """
    Flag diagonal entries σ²_i > ``outlier_factor`` × per-MT median(σ²) as
    statistical outliers. Target = per-MT median.

    Catches evaluator-written placeholder/filler variances that aren't near
    a reaction threshold (so ``flag_threshold_bins`` won't catch them) and
    would otherwise saturate the global variance cap. Concrete failure mode:
    JEFF-4.0 Mn-55 MT=856 (lumped n,α ladder) carries σ²_rel ~ 10⁹ in bins
    above ~22 MeV where σ̄ collapses; the data is parsed as already-relative
    so the σ̄-floor in the abs→rel conversion never sees it.

    Same return signature as ``flag_threshold_bins`` so
    ``rescale_threshold_bins_congruence`` consumes the output unchanged.

    ``outlier_factor`` is intentionally large (default 1000) so only
    physically-impossible outliers are caught — bins at 5–10× the median
    represent legitimate heterogeneity in measurement quality and are left
    alone.

    Parameters
    ----------
    cov_mat : (n,n) np.ndarray
    param_pairs : list of (zaid, mt) tuples
    num_groups : int
    bins : array-like, length ``num_groups + 1``
    outlier_factor : float
        Bins with σ² > ``outlier_factor`` × per-MT median are flagged.
    min_groups_for_median : int
        Minimum same-MT bins (excluding the flagged one) required to
        compute the median target. If fewer, the bin is not flagged.
    skip_indices : sequence of int, optional
        Flat indices to exclude from outlier consideration entirely
        (typically those already rescaled by ``flag_threshold_bins``).
    space, verbose, logger, label : same as ``flag_threshold_bins``.
    """
    bins_arr = np.asarray(bins, dtype=float)
    diag = np.diag(cov_mat)

    skip_set: Set[int] = set(int(i) for i in (skip_indices or []))

    flagged_indices: List[int] = []
    targets: List[float] = []
    detection_log: List[Dict] = []

    for pair_idx, (zaid, mt) in enumerate(param_pairs):
        block_start = pair_idx * num_groups
        block_diag = diag[block_start: block_start + num_groups]
        finite_pos = block_diag[np.isfinite(block_diag) & (block_diag > 0)]
        if finite_pos.size < min_groups_for_median:
            continue
        median_var = float(np.median(finite_pos))
        if median_var <= 0.0:
            continue
        threshold = outlier_factor * median_var

        for g in range(num_groups):
            flat = block_start + g
            if flat in skip_set:
                continue
            cur = float(diag[flat])
            if not np.isfinite(cur) or cur <= 0.0 or cur <= threshold:
                continue

            other_mask = np.ones(num_groups, dtype=bool)
            other_mask[g] = False
            other_vals = block_diag[other_mask]
            other_pos = other_vals[np.isfinite(other_vals) & (other_vals > 0)]
            if other_pos.size < min_groups_for_median:
                continue
            target = float(np.median(other_pos))
            if target <= 0.0:
                continue

            e_lo = float(bins_arr[g]) if g < len(bins_arr) - 1 else float("nan")
            e_hi = float(bins_arr[g + 1]) if g < len(bins_arr) - 1 else float("nan")

            flagged_indices.append(flat)
            targets.append(target)
            detection_log.append({
                "index": flat,
                "zaid": int(zaid),
                "mt": int(mt),
                "group": g,
                "energy_lo": e_lo,
                "energy_hi": e_hi,
                "current_variance": cur,
                "target_variance": target,
                "median_variance": median_var,
                "ratio_to_median": cur / median_var,
                "n_groups_used": int(other_pos.size),
            })

    if verbose or logger is not None:
        ctx = f" [{label}]" if label else ""
        sep = "-" * 60
        if flagged_indices:
            _log_message(f"\n[COVARIANCE] [VARIANCE-OUTLIER DETECTION]{ctx}\n{sep}", logger, verbose)
            _log_message(
                f"  Flagged {len(flagged_indices)} bin(s) with σ² > "
                f"{outlier_factor:g} × per-MT median (space={space}):",
                logger, verbose,
            )
            for d in detection_log:
                _log_message(
                    f"    MT={d['mt']}, G={d['group']} "
                    f"[{d['energy_lo']:.2e},{d['energy_hi']:.2e}] MeV: "
                    f"σ²={d['current_variance']:.3e}, "
                    f"median={d['median_variance']:.3e}, "
                    f"ratio={d['ratio_to_median']:.2e}, "
                    f"target={d['target_variance']:.3e}",
                    logger, verbose,
                )
            _log_message(sep, logger, verbose)

    return flagged_indices, targets, detection_log


def nearest_psd_higham(
    A: np.ndarray,
    *,
    preserve_diagonal: bool = True,
    max_iter: Optional[int] = None,
    tol: Optional[float] = None,
    eigval_floor: float = 0.0,
    verbose: bool = True,
    logger=None,
    log_every: int = 50,
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
    max_iter : int, optional
        Maximum number of alternating-projection iterations. ``None``
        (default) auto-selects: 1000 for n ≤ 2000, 200 for n > 2000.
        Sparse multigroup covariances (n > 2000, ~95 % non-finite from
        below-threshold reactions) plateau by iter ~100 and the cholesky
        path then falls back to clip — running 1000 iterations just burns
        wall-time without improving the projection.
    tol : float, optional
        Convergence tolerance on relative Frobenius change between iterations.
        ``None`` (default) auto-selects: 1e-10 for matrices ≤ 1000×1000,
        1e-8 for larger ones — at n ≳ 1000, rel_change ≲ 1e-8 is already
        at float-precision noise from the O(n³) eigendecomposition, so
        a stricter tol just guarantees no convergence.
    eigval_floor : float
        Floor for eigenvalues in the PSD projection step (0.0 for exact PSD,
        small positive for strict PD).
    verbose : bool
        Whether to print diagnostic information.
    logger : optional
        Logger instance for file output.
    log_every : int
        Emit a progress line every N iterations (set to 0 to disable).
        Used to track convergence on long runs where each iteration is
        an O(n³) eigendecomposition.

    Returns
    -------
    Tuple[np.ndarray, dict]
        (X_psd, info) where info contains diagnostic metadata:
        - iterations, converged, frobenius_distance, relative_frobenius_error,
          max_diagonal_change, eigenvalue_range_before, eigenvalue_range_after,
          n_negative_eigenvalues_before
    """
    n = A.shape[0]
    if tol is None:
        tol = 1e-10 if n <= 1000 else 1e-8
    if max_iter is None:
        max_iter = 1000 if n <= 2000 else 200
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
        X = (V * w_clipped[None, :]) @ V.T
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
        X_psd = (V * w_clipped[None, :]) @ V.T

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

        # Progress logging — rel_change is the actionable convergence
        # signal (algorithm stops when it drops below tol). We don't log
        # an eigenvalue here because the cheap option (w from R) reflects
        # the projection target R = Y - D_S, not the iterate Y, and can
        # stay indefinite even as Y converges to PSD.
        if verbose and log_every and (k + 1) % log_every == 0:
            _log_message(
                f"[PSD] [HIGHAM] iter {k+1}/{max_iter}: "
                f"rel_change={rel_change:.2e} (tol={tol:.0e})",
                logger, verbose,
            )

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
        Y = (V_f * w_f[None, :]) @ V_f.T
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


def clip_and_rescale(
    M: np.ndarray,
    *,
    eigval_floor: float = 0.0,
    verbose: bool = False,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip negative eigenvalues, then put the stated diagonal back.

    The gap this closes. ``clip`` rebuilds ``V·clip(Λ,0)·Vᵀ``, which preserves
    the eigenvectors — the reason it is the right projection for these matrices,
    since the near-null direction carries a sum rule — but **not the diagonal**.
    Every clipped negative eigenvalue adds variance, spread over the components
    in proportion to its eigenvector, so a component the evaluation gives almost
    no variance comes out of the projection with some. Downstream that is not
    cosmetic: PFNS divides each drawn delta by its group probability, and the
    lowest groups of a fission spectrum hold ~1e-17 of the total, so a
    negligible absolute gain becomes a large perturbation ratio. The standing
    workaround is a 5σ clamp in ``generate_pfns_samples``.

    Higham is the third option and it is not the answer either, though the
    reason first written here was wrong. "Did not finish four 122×122 bands in
    two minutes" was contention on a loaded box: re-measured quiet, one band
    converges in ~1900 iterations and ~13 s. Three numbers rule it out instead.

    * It preserves the diagonal *in absolute norm* — the post-loop eigenvalue
      clip below runs without restoring it, leaving ``max_diagonal_change``
      at 1.4e-12 against a largest stated variance of 7.4e-5. Negligible, and
      the code says so. But the same 1.4e-12 is **0.55% of the smallest stated
      variances**, and on a covariance whose diagonal spans decades that is the
      sense that matters. ``clip_rescale`` is exact in both senses.
    * It degrades ``C·1`` by ~40x (4.2e-6 → 1.7e-4) where ``clip`` improves it.
    * It costs ~10⁴x ``clip``.

    Middling on both axes for four orders of magnitude more time, so ``clip``
    stays on the PFNS path. :func:`kika.cov.conditioning.predict_psd_repairs`
    measures all three on any given matrix, which is how these numbers were got.

    The fix is one line of algebra. With ``d = sqrt(diag(C₀)/diag(C_clip))``,
    the matrix ``diag(d)·C_clip·diag(d)`` has exactly ``diag(C₀)`` on its
    diagonal, is a **congruence transform** of a PSD matrix and therefore still
    PSD, and costs O(n²) against Higham's iterations. It does not preserve the
    eigenvectors exactly — nothing that changes the diagonal can — but it
    rescales rather than rotates, so a component with zero stated variance gets
    an identically zero row and column, which is stronger than what ``clip``
    gives it.

    **Measured, and it does not do what it was proposed for.** On the four
    Cf-252 MF35 bands the diagonal comes back exactly (max error 1.6e-8 → 1.4e-20)
    and the result stays PSD (λ_min ≈ -5e-20). But the **sum rule does not
    survive it**: the row-sum residual ``max|Σ_j C_ij| / max|C|`` goes from
    4.2e-6 to 2.2e-3, some 500x worse, on every band.

    That is structural rather than a tolerance to tune. ``C·1 ≈ 0`` says the
    all-ones vector is a near-null eigenvector, and clipping keeps it null
    precisely because it preserves eigenvectors. The congruence gives
    ``(D C D)·1 = D C (D 1)``, which is zero only when ``d`` is constant — so
    restoring a non-uniform diagonal and preserving the sum rule are, for these
    matrices, the same choice made two ways.

    The 5σ clamp in ``generate_pfns_samples`` therefore **stays**: the rescale
    roughly halves the draws that reach it (172→113, 220→121, 274→127, 277→124
    over the four bands at 64 samples) and removes none of them. Use this where
    the marginals matter and no sum rule does — MF33 and MF34 — and not on a
    normalised spectrum.

    Not reachable from ``psd_method="auto"``: this is a different numerical
    answer, and the pipelines already running on ``clip`` must not change
    underneath because a new option was added. Ask for it by name.

    Returns
    -------
    (matrix, info)
        ``info`` carries ``n_negative``, ``min_eigenvalue``,
        ``max_diagonal_error_before`` and ``max_diagonal_error_after``, so a
        caller can assert the restoration actually happened.
    """
    M = np.asarray(M, dtype=float)
    original = np.diag(M).copy()

    eigvals, eigvecs = _robust_eigh(
        M, label="clip_rescale", verbose=verbose, logger=logger,
    )
    n_negative = int(np.sum(eigvals < 0))
    clipped = np.maximum(eigvals, eigval_floor)
    projected = (eigvecs * clipped[None, :]) @ eigvecs.T
    projected = (projected + projected.T) * 0.5

    before = np.max(np.abs(np.diag(projected) - original)) if original.size else 0.0

    projectedDiagonal = np.diag(projected)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(np.where(projectedDiagonal > 0,
                                 original / projectedDiagonal, 0.0))
    # A non-positive projected diagonal means the component spans none of the
    # retained subspace; zero is the only rescale that keeps the result PSD,
    # and it states the stronger fact — that direction carries no variance.
    scale = np.where(np.isfinite(scale) & (original > 0), scale, 0.0)

    rescaled = projected * np.outer(scale, scale)
    rescaled = (rescaled + rescaled.T) * 0.5

    after = np.max(np.abs(np.diag(rescaled) - original)) if original.size else 0.0
    if verbose:
        _log_message(
            f"[COV] [CLIP_RESCALE] clipped {n_negative} negative eigenvalues "
            f"(min={float(eigvals.min()):.3e}); max diagonal error "
            f"{before:.3e} → {after:.3e}",
            logger, verbose,
        )

    return rescaled, {
        "n_negative": n_negative,
        "min_eigenvalue": float(eigvals.min()) if eigvals.size else 0.0,
        "max_diagonal_error_before": float(before),
        "max_diagonal_error_after": float(after),
    }


def cholesky_decomposition(
    cov_obj: CovarianceMatrixProtocol = None,
    *,
    space: str = "log",
    psd_method: str = "auto",
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
        PSD enforcement method when the matrix is not already PD:
        - "auto" (default): one eigendecomposition; if |λ_min|/λ_max is
          below the module threshold use clip, otherwise Higham.
        - "higham": iterative diagonal-preserving projection.
        - "clip":  zero negative eigenvalues, reconstruct, then Cholesky.
        - "none":  add diagonal jitter until Cholesky succeeds.
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

    _validate_psd_method(psd_method)

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
        # Resolve "auto" -> clip or higham; reuse the eigendecomp on the clip path
        eigvals_auto = eigvecs_auto = None
        if psd_method == "auto":
            psd_method, eigvals_auto, eigvecs_auto = _resolve_psd_auto(
                M, verbose=verbose, logger=logger,
            )

        if psd_method == "higham":
            if verbose:
                _log_message("[DECOMPOSITION] Matrix not PD → applying Higham projection", logger, verbose)
            M_psd, psd_info = nearest_psd_higham(
                M, preserve_diagonal=True, eigval_floor=1e-14,
                verbose=verbose, logger=logger,
            )

            # Fallback: if Higham hit max_iter with a sizeable Frobenius gap,
            # clip is strictly better — its Frobenius distance is the sum of
            # the squared negative eigenvalues only (typically ≪ 1 %), versus
            # 7–17 % we have seen on large multigroup NJOY covariances when
            # Dykstra plateaus. Diagonal preservation is the price; for
            # sampling that's preferable to a heavily distorted covariance.
            higham_fallback_used = False
            if (not psd_info.get("converged", True) and
                    psd_info.get("relative_frobenius_error", 0.0) > _HIGHAM_FALLBACK_FROB_THRESHOLD):
                higham_fallback_used = True
                if verbose:
                    _log_message(
                        f"[DECOMPOSITION] [FALLBACK] Higham did not converge "
                        f"(Frobenius error="
                        f"{psd_info['relative_frobenius_error']*100:.2f}%"
                        f" > {_HIGHAM_FALLBACK_FROB_THRESHOLD*100:.1f}%)"
                        " → falling back to eigenvalue clip on original matrix",
                        logger, verbose,
                    )
                if eigvals_auto is None:
                    M_clean = np.where(np.isfinite(M), M, 0.0)
                    eigvals_auto, eigvecs_auto = _robust_eigh(
                        M_clean, label="higham fallback clip",
                        verbose=verbose, logger=logger,
                    )
                n_neg_fb = int(np.sum(eigvals_auto < 0))
                eigvals_clipped = np.maximum(eigvals_auto, 1e-14)
                M_psd = eigvecs_auto @ np.diag(eigvals_clipped) @ eigvecs_auto.T
                M_psd = (M_psd + M_psd.T) * 0.5
                if verbose:
                    _log_message(
                        f"[DECOMPOSITION] [FALLBACK] Clipped {n_neg_fb} negative "
                        "eigenvalues on original matrix",
                        logger, verbose,
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
                if higham_fallback_used:
                    _log_message(
                        "[DECOMPOSITION] Cholesky successful after fallback clip "
                        "(Higham unconverged result was discarded)",
                        logger, verbose,
                    )
                else:
                    _log_message(
                        f"[DECOMPOSITION] Cholesky successful after Higham "
                        f"({psd_info['iterations']} iter, "
                        f"Frobenius error={psd_info['relative_frobenius_error']*100:.4f}%)",
                        logger, verbose,
                    )

        elif psd_method == "clip":
            if eigvals_auto is None:
                eigvals_auto, eigvecs_auto = _robust_eigh(
                    M, label="cholesky clip", verbose=verbose, logger=logger,
                )
            n_negative = int(np.sum(eigvals_auto < 0))
            if n_negative > 0 or float(np.min(eigvals_auto)) < 1e-14:
                min_eigval = float(np.min(eigvals_auto))
                if verbose:
                    _log_message(
                        f"[COV] [CHOLESKY] Clipped {n_negative} negative eigenvalues "
                        f"(min={min_eigval:.3e}) before Cholesky (floor=1e-14)",
                        logger, verbose,
                    )
                # Floor to a tiny positive value so Cholesky sees a strictly
                # PD matrix. Pure-zero eigenvalues are mathematically PSD but
                # break the LL^T factorization; mirror Higham's eigval_floor=1e-14.
                eigvals_clipped = np.maximum(eigvals_auto, 1e-14)
                M_psd = eigvecs_auto @ np.diag(eigvals_clipped) @ eigvecs_auto.T
                # Force symmetry — V·diag(λ)·Vᵀ is symmetric in exact arithmetic
                # but accumulates float round-off otherwise.
                M_psd = (M_psd + M_psd.T) * 0.5
            else:
                if verbose:
                    _log_message(
                        "[COV] [CHOLESKY] No negative eigenvalues — Cholesky on input matrix",
                        logger, verbose,
                    )
                M_psd = M
            try:
                L = np.linalg.cholesky(M_psd)
            except np.linalg.LinAlgError:
                if verbose:
                    _log_message(
                        "[DECOMPOSITION] [WARNING] Cholesky failed after clip — "
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
                        f"[DECOMPOSITION] Cholesky successful after clip + jitter ({jitter:.3e})",
                        logger, verbose,
                    )
                return L
            if verbose:
                _log_message("[DECOMPOSITION] Cholesky successful after clip", logger, verbose)

        else:  # psd_method == "none"
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
        PSD correction method: "auto" (default), "higham", "clip", or "none".
        If None, resolved from *clip_negatives*: True → "auto", False → "none".
        "auto" picks clip when |λ_min|/λ_max is below the module threshold,
        otherwise Higham.
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
        psd_method = "auto" if clip_negatives else "none"

    _validate_psd_method(psd_method)

    if matrix is not None:
        M = matrix
    else:
        M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)

    n = M.shape[0]
    if verbose:
        _log_message(f"[DECOMPOSITION] Eigendecomposition in {space} space ({n}×{n})", logger, verbose)

    # Resolve "auto" with one eigendecomposition that we can reuse on the clip path
    eigvals = eigvecs = None
    if psd_method == "auto":
        psd_method, eigvals, eigvecs = _resolve_psd_auto(M, verbose=verbose, logger=logger)

    if psd_method == "higham":
        M, _info = nearest_psd_higham(M, preserve_diagonal=True, verbose=verbose, logger=logger)
        eigvals = eigvecs = None  # M changed — force recompute
    elif psd_method == "clip_rescale":
        M, _info = clip_and_rescale(M, verbose=verbose, logger=logger)
        eigvals = eigvecs = None  # the rescale rotates as well as scales

    if eigvals is None:
        eigvals, eigvecs = _robust_eigh(M, label="eigen decomposition", verbose=verbose, logger=logger)

    if psd_method == "clip":
        n_negative = int(np.sum(eigvals < 0))
        if n_negative > 0:
            min_eigval = float(np.min(eigvals))
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
        PSD correction method: "auto" (default), "higham", "clip", or "none".
        If None, resolved from *clip_negatives*: True → "auto", False → "none".
        "auto" picks clip when |λ_min|/λ_max is below the module threshold,
        otherwise Higham.
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
        psd_method = "auto" if clip_negatives else "none"

    _validate_psd_method(psd_method)

    if matrix is not None:
        M = matrix
    else:
        M = (cov_obj.covariance_matrix if space == "linear" else cov_obj.log_covariance_matrix)

    n = M.shape[0]
    if verbose:
        _log_message(f"[DECOMPOSITION] SVD in {space} space ({n}×{n})", logger, verbose)

    # Resolve "auto" with one eigendecomposition that we reuse on the clip path
    eigvals_auto = eigvecs_auto = None
    if psd_method == "auto":
        psd_method, eigvals_auto, eigvecs_auto = _resolve_psd_auto(M, verbose=verbose, logger=logger)

    if psd_method == "higham":
        M, _info = nearest_psd_higham(M, preserve_diagonal=True, verbose=verbose, logger=logger)
    elif psd_method == "clip_rescale":
        M, _info = clip_and_rescale(M, verbose=verbose, logger=logger)
    elif psd_method == "clip":
        if eigvals_auto is None:
            eigvals_auto, eigvecs_auto = _robust_eigh(M, label="SVD clip", verbose=verbose, logger=logger)
        n_negative = int(np.sum(eigvals_auto < 0))

        if n_negative > 0:
            min_eigval = float(np.min(eigvals_auto))
            if verbose:
                _log_message(f"[COV] [SVD] Clipped {n_negative} negative eigenvalues before SVD (min={min_eigval:.3e})", logger, verbose)
            eigvals_clipped = np.clip(eigvals_auto, 0.0, None)
            M = eigvecs_auto @ np.diag(eigvals_clipped) @ eigvecs_auto.T
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

    # Relative diagonal errors. Floor at 1e-12 (rather than just != 0) so
    # below-threshold reactions with σ²≈machine-noise don't dominate the
    # max — those bins are non-perturbative anyway.
    _DIAG_VAR_FLOOR = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_rel_errors = np.where(
            np.isfinite(diag_orig) & (np.abs(diag_orig) >= _DIAG_VAR_FLOOR),
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

