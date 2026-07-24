"""Dense per-experiment evaluation covariance for the chi^2 pipelines.

Builds the N x N block

    Sigma_eval = Sigma_eval^MF34 + Sigma_eval^MF33

for one experiment, where N is the number of data points (E_j, mu_j) in that
experiment. This replaces the per-row scalar `sigma_eval` previously folded
into the diagonal of the chi^2 covariance, so that cross-energy MF34
correlations (read from the full mat[k_j, k_k] block instead of just the
energy-diagonal mat[k, k]), cross-order MF34 correlations propagated through
shared a_L, and cross-energy MF33 correlations are all carried into the
chi^2.

Sensitivity model (matches numpy.polynomial.legendre.legval applied to
[c_0, c_0*(2L+1)*a_L, ...] in the precompute scripts):

    y(E, mu) = c_0(E) * [P_0(mu) + sum_{L>=1} (2L+1) a_L(E) P_L(mu)]
    dy/da_L(E)  = c_0(E) * (2L+1) * P_L(mu)         (L >= 1)
    dy/dsigma(E) at relative scaling: y(E, mu) * dsigma_rel(E)

For relative MF34 blocks, ENDF stores Cov[a_L/a_L_nom, a_L'/a_L'_nom];
we recover the absolute Cov[a_L, a_L'] by multiplying with the *interpolated*
nominal a_L(E_j), a_L'(E_k) (same convention as the legacy
`propagate_mf34_to_dy`). Off-diagonal energy entries use the same dispatch:
two points in MF34 bins k_j, k_k contribute mat[k_j, k_k]; two points in the
same bin share mat[k, k] (perfect intra-bin correlation, ENDF
piecewise-constant semantics).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import legval


# ── MF34: dense Σ_eval contribution from Legendre coefficient covariance ──────

def _legendre_base_sens(mu: np.ndarray, c0: np.ndarray, L_max: int) -> np.ndarray:
    """base_sens[L-1, j] = c_0(E_j) * (2L+1) * P_L(mu_j) for L = 1..L_max.

    This is dy/da_L(E_j, mu_j) before the relative-a_L scaling; shared by the
    MF34 self block and the MF33↔MF34 cross block so the two use one sensitivity
    definition.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    c0 = np.asarray(c0, dtype=float).ravel()
    N = mu.size
    P_arr = np.zeros((L_max, N), dtype=float)
    for L in range(1, L_max + 1):
        coef = np.zeros(L + 1, dtype=float)
        coef[L] = 1.0
        P_arr[L - 1] = legval(mu, coef)
    twoLp1 = np.arange(3, 2 * L_max + 2, 2, dtype=float)  # 3, 5, 7, ... for L=1..
    return c0[None, :] * twoLp1[:, None] * P_arr  # (L_max, N)


def build_mf34_block(
    mf34,
    e_mev: np.ndarray,
    mu: np.ndarray,
    c0: np.ndarray,
    a_l_per_pt: np.ndarray,
) -> np.ndarray:
    """Dense N×N covariance contribution from MF34 for one experiment.

    Parameters
    ----------
    mf34 : MF34CovMat or None
    e_mev : (N,) energy of each data point in MeV.
    mu : (N,) cos(scattering angle).
    c0 : (N,) c_0(E_j) used by the library at each point.
    a_l_per_pt : (N, L_max) interpolated a_L(E_j) for L = 1..L_max.

    Returns
    -------
    (N, N) ndarray.  Zero if mf34 is None or N == 0.
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if mf34 is None:
        return np.zeros((N, N), dtype=float)

    a_l_per_pt = np.asarray(a_l_per_pt, dtype=float)
    if a_l_per_pt.ndim == 1:
        a_l_per_pt = a_l_per_pt.reshape(N, -1)
    L_max = a_l_per_pt.shape[1]
    if L_max == 0:
        return np.zeros((N, N), dtype=float)

    mu = np.asarray(mu, dtype=float).ravel()
    c0 = np.asarray(c0, dtype=float).ravel()
    e_ev = e_mev * 1e6

    # base_sens[L-1, j] = c_0(E_j) * (2L+1) * P_L(mu_j); a_L scaling added per
    # block when relative.
    base_sens = _legendre_base_sens(mu, c0, L_max)  # (L_max, N)

    sigma = np.zeros((N, N), dtype=float)
    for idx in range(len(mf34.matrices)):
        l_r = int(mf34.l_rows[idx])
        l_c = int(mf34.l_cols[idx])
        if l_r < 1 or l_c < 1 or l_r > L_max or l_c > L_max:
            continue
        grid = np.asarray(mf34.energy_grids[idx], dtype=float)
        mat = np.asarray(mf34.matrices[idx], dtype=float)
        M = mat.shape[0]
        if M == 0:
            continue
        bin_idx = np.clip(
            np.searchsorted(grid, e_ev, side="right") - 1, 0, M - 1,
        )
        block = mat[np.ix_(bin_idx, bin_idx)]  # (N, N), block[j,k] = mat[bin_j, bin_k]

        sens_r = base_sens[l_r - 1].copy()
        sens_c = base_sens[l_c - 1].copy()
        if mf34.is_relative[idx]:
            sens_r *= a_l_per_pt[:, l_r - 1]
            sens_c *= a_l_per_pt[:, l_c - 1]

        sigma += sens_r[:, None] * block * sens_c[None, :]
        if l_r != l_c:
            # ENDF stores one of (l_r, l_c) / (l_c, l_r). The companion block's
            # cov at energy pair (E_j, E_k) is mat[bin_k, bin_j] = block.T[j,k]
            # by symmetry of the joint distribution.
            sigma += sens_c[:, None] * block.T * sens_r[None, :]

    # Symmetrize against rounding asymmetries from float ops.
    return 0.5 * (sigma + sigma.T)


# ── MF33: dense Σ_eval contribution from cross-section magnitude covariance ───

def _mf4_bin_edges_for_points(
    energies_mf4_mev: np.ndarray, e_mev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point (E_lo, E_hi) of the MF4 bin containing each E_j (clamped).

    Mirrors `find_mf4_bin` from precompute_chi2_library_c0.py. Returns arrays
    of length N in MeV.
    """
    grid = np.asarray(energies_mf4_mev, dtype=float)
    e = np.asarray(e_mev, dtype=float).ravel()
    n = grid.size
    e_lo = np.empty_like(e)
    e_hi = np.empty_like(e)
    for j, val in enumerate(e):
        if val <= grid[0]:
            e_lo[j], e_hi[j] = grid[0], grid[1]
        elif val >= grid[-1]:
            e_lo[j], e_hi[j] = grid[-2], grid[-1]
        else:
            i_hi = int(np.searchsorted(grid, val, side="right"))
            e_lo[j], e_hi[j] = grid[i_hi - 1], grid[i_hi]
    return e_lo, e_hi


def _mf33_overlap_weights(
    grid_ev: np.ndarray, e_lo_ev: np.ndarray, e_hi_ev: np.ndarray,
) -> np.ndarray:
    """Bin-overlap weight matrix W of shape (N, M), normalized per row.

    W[j, k] = overlap(MF4 bin j, MF33 bin k) / sum_{k'} overlap(MF4 bin j, MF33 bin k').

    Used to bin-average σ-cov over the MF4 bin containing each data point —
    same length-weighted averaging as `avg_mf33_rel_var_over_bin`, but in 2-D
    for ρ_σ(E_j, E_k) = W[j] @ rel_cov @ W[k]^T.
    """
    grid_ev = np.asarray(grid_ev, dtype=float)
    M = grid_ev.size - 1
    N = e_lo_ev.size
    W = np.zeros((N, M), dtype=float)
    for j in range(N):
        lo = float(e_lo_ev[j])
        hi = float(e_hi_ev[j])
        if hi <= lo or hi <= grid_ev[0] or lo >= grid_ev[-1]:
            continue
        k_first = max(0, int(np.searchsorted(grid_ev, lo, side="right") - 1))
        k_last = min(M - 1, int(np.searchsorted(grid_ev, hi, side="left")))
        for k in range(k_first, k_last + 1):
            bin_lo = max(grid_ev[k], lo)
            bin_hi = min(grid_ev[k + 1], hi)
            if bin_hi > bin_lo:
                W[j, k] = bin_hi - bin_lo
        s = W[j].sum()
        if s > 0:
            W[j] /= s
    return W


def build_mf33_block(
    mf33_grid_ev: Optional[np.ndarray],
    mf33_rel_cov: Optional[np.ndarray],
    energies_mf4_mev: np.ndarray,
    e_mev: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    """Dense N×N covariance contribution from MF33 (cross-section magnitude).

    Returns zero block if mf33 inputs are missing.

    Sigma_eval^MF33[j, k] = y_eval(j) * y_eval(k) * (W[j] @ rel_cov @ W[k]^T)

    where W is the MF4-bin-averaged overlap of each data point with the MF33
    grid (same averaging convention as the legacy diagonal path).
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if mf33_grid_ev is None or mf33_rel_cov is None:
        return np.zeros((N, N), dtype=float)

    e_lo_mev, e_hi_mev = _mf4_bin_edges_for_points(energies_mf4_mev, e_mev)
    W = _mf33_overlap_weights(
        np.asarray(mf33_grid_ev, dtype=float),
        e_lo_mev * 1e6,
        e_hi_mev * 1e6,
    )
    rho = W @ np.asarray(mf33_rel_cov, dtype=float) @ W.T
    y = np.asarray(y_eval, dtype=float).ravel()
    sigma = (y[:, None] * y[None, :]) * rho
    return 0.5 * (sigma + sigma.T)


# ── MF33 ↔ MF34: dense Σ_eval cross contribution (sigma ↔ a_L) ────────────────

def build_mf33_mf34_cross_block(
    cross,
    e_mev: np.ndarray,
    mu: np.ndarray,
    c0: np.ndarray,
    a_l_per_pt: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    """Dense N×N MF33↔MF34 cross contribution (magnitude ↔ shape).

    This is the chi^2 counterpart of the MF34 (L=0, L1) cross blocks written by
    ``create_mf34_from_covariance``.  It is **opt-in and defaults to a zero
    block** when ``cross`` is falsy, so today's Σ_eval numbers are unchanged
    (the factorization assumption Cov(sigma, a_L) = 0 stays in force until a
    joint evaluation provides these blocks).

    Parameters
    ----------
    cross : sequence of dict, or None
        Per-order cross-covariance blocks.  Each entry::

            {"l": L1,                       # Legendre order (>= 1)
             "mag_grid_ev": (N0+1,),        # coarse magnitude (sigma) grid
             "shape_grid_ev": (Nsh+1,),     # a_L shape grid
             "matrix": (N0, Nsh),           # Cov(sigma bin, a_L1 bin)
             "is_relative": bool}           # default True

        Mirrors item-2's LB=6 (L=0, L1) block layout.  Missing/empty → zeros.
    e_mev, mu, c0, a_l_per_pt, y_eval
        Same per-point arrays as :func:`build_mf34_block` /
        :func:`build_mf33_block`.

    Notes
    -----
    Sensitivities: dy/dsigma = ``y_eval`` (magnitude side), dy/da_L1 =
    ``c0*(2L1+1)*P_L1`` (shape side; scaled by the nominal a_L1 for relative
    blocks, matching :func:`build_mf34_block`).  The block is assembled
    symmetrically::

        Σ_cross[j,k] = ydsig(j)·C[j,k]·sensL(k) + sensL(j)·C.T[j,k]·ydsig(k)
    """
    e_mev = np.asarray(e_mev, dtype=float).ravel()
    N = e_mev.size
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if not cross:
        return np.zeros((N, N), dtype=float)

    a_l_per_pt = np.asarray(a_l_per_pt, dtype=float)
    if a_l_per_pt.ndim == 1:
        a_l_per_pt = a_l_per_pt.reshape(N, -1)
    L_max = a_l_per_pt.shape[1]
    if L_max == 0:
        return np.zeros((N, N), dtype=float)

    base_sens = _legendre_base_sens(mu, c0, L_max)  # (L_max, N)
    y = np.asarray(y_eval, dtype=float).ravel()
    e_ev = e_mev * 1e6

    sigma = np.zeros((N, N), dtype=float)
    for blk in cross:
        L = int(blk["l"])
        if L < 1 or L > L_max:
            continue
        mag_grid = np.asarray(blk["mag_grid_ev"], dtype=float)
        shape_grid = np.asarray(blk["shape_grid_ev"], dtype=float)
        mat = np.asarray(blk["matrix"], dtype=float)
        if mat.size == 0:
            continue
        n0, n_sh = mat.shape
        bin0 = np.clip(np.searchsorted(mag_grid, e_ev, side="right") - 1, 0, n0 - 1)
        binL = np.clip(np.searchsorted(shape_grid, e_ev, side="right") - 1, 0, n_sh - 1)
        C = mat[np.ix_(bin0, binL)]  # (N, N): C[j,k] = Cov(sigma bin0_j, a_L binL_k)

        sens_L = base_sens[L - 1].copy()
        if blk.get("is_relative", True):
            sens_L *= a_l_per_pt[:, L - 1]

        sigma += y[:, None] * C * sens_L[None, :]
        sigma += sens_L[:, None] * C.T * y[None, :]

    return 0.5 * (sigma + sigma.T)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_eval_cov_for_groups(
    df,
    libraries: Dict[str, Dict],
    a_l_lookup: Callable[[str, Dict, float], np.ndarray],
    *,
    l_max: int,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
) -> Dict[Tuple, np.ndarray]:
    """Group df by `group_cols`, build one dense Σ_eval per group.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-row chi^2 frame with at least: library, experiment_id, energy_mev,
        mu, c0, y_eval.
    libraries : dict[str, dict]
        Library data keyed by library label (must include the same keys used
        by `a_l_lookup` and the optional MF33 fields `mf33_grid_ev` and
        `mf33_rel_cov`, plus `mf34`).
    a_l_lookup : callable (lib_key, lib_dict, e_mev) -> a_L vector of length l_max
        Interpolates the library's nominal Legendre a_L at one energy.
    l_max : int
        Truncation order.

    Returns
    -------
    dict mapping group key tuple -> (N, N) np.float32 covariance block.
    """
    import pandas as pd  # local import to keep this module import-light

    eval_cov: Dict[Tuple, np.ndarray] = {}
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        lib_key = key[group_cols.index("library")] if "library" in group_cols else None
        if lib_key is None or lib_key not in libraries:
            n = len(sub)
            eval_cov[key] = np.zeros((n, n), dtype=np.float32)
            continue
        lib = libraries[lib_key]

        e_mev = sub["energy_mev"].to_numpy(dtype=float)
        mu = sub["mu"].to_numpy(dtype=float)
        c0 = sub["c0"].to_numpy(dtype=float)
        y_eval = sub["y_eval"].to_numpy(dtype=float)
        N = e_mev.size

        a_l_per_pt = np.zeros((N, l_max), dtype=float)
        for i, e_i in enumerate(e_mev):
            a_l_per_pt[i] = a_l_lookup(lib_key, lib, float(e_i))

        sigma_mf34 = build_mf34_block(lib.get("mf34"), e_mev, mu, c0, a_l_per_pt)
        sigma_mf33 = build_mf33_block(
            lib.get("mf33_grid_ev"), lib.get("mf33_rel_cov"),
            lib.get("energies_mf4_mev"), e_mev, y_eval,
        )
        # Opt-in MF33↔MF34 cross term; None → zero block → numbers unchanged.
        sigma_cross = build_mf33_mf34_cross_block(
            lib.get("mf33_mf34_cross"), e_mev, mu, c0, a_l_per_pt, y_eval,
        )
        eval_cov[key] = (sigma_mf34 + sigma_mf33 + sigma_cross).astype(np.float32)
    return eval_cov


# ── Sidecar I/O ───────────────────────────────────────────────────────────────

_KEY_SEP = "@@"


def _encode_key(key: Tuple) -> str:
    return _KEY_SEP.join(str(p) for p in key)


def _decode_key(s: str) -> Tuple:
    return tuple(s.split(_KEY_SEP))


def save_eval_cov(path: str, eval_cov: Dict[Tuple, np.ndarray]) -> None:
    """Write all per-group blocks to a single compressed .npz file."""
    arrays = {_encode_key(k): np.asarray(v) for k, v in eval_cov.items()}
    np.savez_compressed(path, **arrays)


def load_eval_cov(path: str) -> Dict[Tuple, np.ndarray]:
    """Inverse of save_eval_cov. Returns {(library, experiment_id): array}."""
    with np.load(path) as data:
        return {_decode_key(k): np.asarray(data[k]) for k in data.files}
