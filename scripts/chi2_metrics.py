"""Mahalanobis chi^2 with full per-experiment covariance.

Per-experiment covariance for one (library, experiment) block:

    Sigma = D + u u^T + v v^T + Sigma_eval,
        D          = diag(sigma_stat^2)                            [EXFOR uncorrelated]
        u          = sigma_indep_rel * y_exp                       [EXFOR rank-1 normalization]
        v          = sigma_dep_rel  * y_exp                        [EXFOR rank-1 shape]
        Sigma_eval = dense N x N from MF34 (and MF33 in the library pipeline),
                     built per (library, experiment) by scripts.eval_covariance.

Both `u` and `v` model correlated systematics that share a single draw per
experiment per MC sample (manifest ``correlated: true``). `u` has constant
per-row amplitude (energy-independent normalization, e.g. Kinney's 8.06%);
`v` has per-row amplitude varying with energy (piecewise_E calibration shape,
e.g. Cierjacks's 7-12% ERR-T). Putting either on the diagonal would treat
correlated systematics as independent noise and depress chi^2/N.

Sigma_eval carries cross-energy MF34 correlations (mat[k_j, k_k] off-diagonal),
cross-order MF34 correlations propagated through shared a_L sensitivities,
and cross-energy MF33 correlations (library pipeline only). For an experiment
whose evaluation has no MF34/MF33, the caller passes a zero block (or omits
the entry) and Sigma collapses back to the rank-2 EXFOR structure.

Chi^2 is solved via Cholesky on the dense Sigma per experiment for V4:

    chi^2 = r^T Sigma^{-1} r          (cho_solve)
    z     = L^{-1} r,    L L^T = Sigma  (whitened residuals, sum z^2 = chi^2)

V2 and V3 (no dense Sigma_eval) use the closed-form Woodbury reduction for
rank-2 updates D + UU^T where U = [u, v] is N x 2 — still O(N), safe for
28k-point experiments.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular

# Pipeline σ_stat floor — mirrors precompute_chi2_*_c0.py's K&S c₀ fit
# (`sigma_stat = max(sigma_stat, 0.01·|y|)`) and chi2_all_variants_audit.py's
# `_refloor_D`. Without this floor, V4 χ² inflates significantly because the
# rank-deficient Σ_eval inversion amplifies tiny-D points. Keep the three
# downstream scripts (precompute, cluster, audit) consistent on this floor.
SIGMA_STAT_FLOOR_REL = 0.01
# Tiny floor on D_ii to avoid 0/0 when sigma_stat = 0 and y_exp = 0.
DIAGONAL_FLOOR_REL = 1e-3
DIAGONAL_FLOOR_ABS = 1e-30

REQUIRED_COLUMNS = (
    "library", "experiment_id",
    "y_exp", "y_eval",
    "sigma_exp_stat",
    "sigma_sys_indep_rel", "sigma_sys_dep_rel",
)


# ── Sidecar I/O re-export (single import point for notebooks) ────────────────

from scripts.eval_covariance import load_eval_cov, save_eval_cov  # noqa: F401


# ── Core ──────────────────────────────────────────────────────────────────────

def _components(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (D, u, v, r) per row.

    D carries σ_stat² only — σ_dep no longer goes on the diagonal because it's
    a correlated shape mode, not independent noise. Both u (normalization) and
    v (shape) are rank-1 contributions handled in `_per_group_sigma`.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"chi2_metrics: input frame missing columns {sorted(missing)}")

    y_exp = df["y_exp"].to_numpy()
    sigma_stat = df["sigma_exp_stat"].to_numpy()
    sigma_stat = np.maximum(sigma_stat, SIGMA_STAT_FLOOR_REL * np.abs(y_exp))
    u = df["sigma_sys_indep_rel"].to_numpy() * y_exp
    v = df["sigma_sys_dep_rel"].to_numpy() * y_exp

    D = sigma_stat ** 2
    floor = (DIAGONAL_FLOOR_REL * np.abs(y_exp)) ** 2 + DIAGONAL_FLOOR_ABS
    D = np.maximum(D, floor)
    r = y_exp - df["y_eval"].to_numpy()
    return D, u, v, r


def _select_block(
    eval_cov: Dict[Tuple, np.ndarray], key: Tuple, sub: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Look up a (library, experiment_id) block, slicing by `_eval_pos` if the
    sub-DataFrame is a row-subset of the original parquet."""
    block = eval_cov.get(tuple(str(p) for p in key))
    if block is None or "_eval_pos" not in sub.columns:
        return block
    pos = sub["_eval_pos"].to_numpy(dtype=int)
    if (pos < 0).any():
        return block  # rows without eval_pos predate the sidecar; skip slicing.
    return block[np.ix_(pos, pos)]


def _per_group_sigma(
    D: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    eval_cov_block: Optional[np.ndarray],
) -> np.ndarray:
    """Assemble Sigma = diag(D) + u u^T + v v^T + Sigma_eval for one experiment."""
    n = D.size
    sigma = np.diag(D).astype(float)
    sigma += np.outer(u, u)
    sigma += np.outer(v, v)
    if eval_cov_block is not None and eval_cov_block.size:
        if eval_cov_block.shape != (n, n):
            raise ValueError(
                f"eval_cov block has shape {eval_cov_block.shape}, expected {(n, n)}"
            )
        sigma += np.asarray(eval_cov_block, dtype=float)
    return 0.5 * (sigma + sigma.T)


def _solve(sigma: np.ndarray, r: np.ndarray, label: str = "") -> Tuple[float, np.ndarray]:
    """Return (chi^2, whitened residual z) for one experiment.

    Uses Cholesky; falls back to a small jitter on the diagonal when the block
    is numerically indefinite (warn-only, never silently fixes the matrix).
    """
    try:
        c, low = cho_factor(sigma, lower=True, overwrite_a=False, check_finite=False)
    except np.linalg.LinAlgError:
        trace = float(np.trace(sigma))
        jitter = max(1e-12 * abs(trace) / max(sigma.shape[0], 1), 1e-30)
        warnings.warn(
            f"[{label}] Sigma not PSD under Cholesky; adding jitter={jitter:.3g}",
            RuntimeWarning,
            stacklevel=2,
        )
        c, low = cho_factor(
            sigma + jitter * np.eye(sigma.shape[0]),
            lower=True, overwrite_a=False, check_finite=False,
        )
    chi2 = float(r @ cho_solve((c, low), r, check_finite=False))
    z = solve_triangular(
        c if low else c.T, r, lower=True, check_finite=False,
    )
    return chi2, z


def _rank2_chi2_woodbury(
    D: np.ndarray, u: np.ndarray, v: np.ndarray, r: np.ndarray,
) -> float:
    """Closed-form chi^2 for Σ = diag(D) + u uᵀ + v vᵀ via Woodbury (O(N)).

    Let U = [u, v] (N × 2). Then
        Σ⁻¹ = D⁻¹ − D⁻¹ U M⁻¹ Uᵀ D⁻¹,    M = I₂ + Uᵀ D⁻¹ U
    so
        rᵀ Σ⁻¹ r = rᵀ D⁻¹ r − (Uᵀ D⁻¹ r)ᵀ M⁻¹ (Uᵀ D⁻¹ r).

    Generalises the rank-1 Sherman-Morrison closed form previously used; for
    v ≡ 0 the result equals the rank-1 formula exactly (s_uv = s_vv = 0 ⇒
    M = diag(1 + s_uu, 1), inverse trivial).
    """
    d = np.maximum(D, 1e-300)
    s_uu = float(np.sum(u * u / d))
    s_vv = float(np.sum(v * v / d))
    s_uv = float(np.sum(u * v / d))
    s_ur = float(np.sum(u * r / d))
    s_vr = float(np.sum(v * r / d))
    s_rr = float(np.sum(r * r / d))
    det = (1.0 + s_uu) * (1.0 + s_vv) - s_uv ** 2
    if det <= 0.0:
        # Should not happen for PSD D + UUᵀ; fall back to direct rank-1 if v=0.
        return s_rr - s_ur ** 2 / max(1.0 + s_uu, 1e-300)
    correction = (
        (1.0 + s_vv) * s_ur ** 2
        - 2.0 * s_uv * s_ur * s_vr
        + (1.0 + s_uu) * s_vr ** 2
    ) / det
    return s_rr - correction


def mahalanobis_chi2_per_experiment(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
) -> pd.DataFrame:
    """One row per group with N, chi^2, chi^2_per_N.

    Parameters
    ----------
    df : per-row chi^2 frame (REQUIRED_COLUMNS).
    eval_cov : dict mapping group key tuple -> dense (N, N) Sigma_eval block.
        Missing keys are treated as zero blocks (no MF34/MF33 contribution).
        Keys are matched as `tuple(str(...) for v in group)` to align with
        the on-disk .npz schema produced by scripts.eval_covariance.
    """
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}

    groups = df.groupby(list(group_cols), observed=True, sort=False)
    out_rows = []
    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]

        block = _select_block(eval_cov, key, sub)
        sigma = _per_group_sigma(D, u, v, block)
        label = "/".join(str(p) for p in key)
        chi2, _z = _solve(sigma, r, label=label)
        N = len(sub)
        out_rows.append({
            **{col: val for col, val in zip(group_cols, key)},
            "N": N,
            "chi2": chi2,
            "chi2_per_N": chi2 / N if N > 0 else np.nan,
        })
    return pd.DataFrame(out_rows)


def chi2_per_experiment_variants(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    include_eval_diag: bool = True,
) -> pd.DataFrame:
    """Compute all four chi^2 variants per group in a single pass.

    Variants (Σ definition per experiment):
      V1 textbook diag   : Σ = diag(σ_stat² + (σ_indep·y)² + (σ_dep·y)² + σ_eval_diag²)
      V2 rank-2 only     : Σ = D + u uᵀ + v vᵀ                (D excludes σ_eval)
      V3 rank-2 + diag   : Σ = (D + diag(σ_eval_diag²)) + u uᵀ + v vᵀ
      V4 rank-2 + dense  : Σ = D + u uᵀ + v vᵀ + Σ_eval_block (this is the existing
                                                                `mahalanobis_chi2_per_experiment`)

    For V1 and V3 we read σ_eval_diag from the parquet column of the same name
    (set by precompute_chi2_*_c0.py to √ diag(block)). When `eval_cov` is None
    we still need σ_eval_diag in df for V1/V3 to be meaningful; if the column
    is absent it is treated as zero (degenerates V1→ rank-0 textbook with only
    EXFOR diagonal, V3→V2).

    V4 is computed only if `eval_cov` is provided; otherwise the V4 column is
    NaN.

    Setting `include_eval_diag=False` zeroes σ_eval_diag everywhere, turning V1
    into the "EXFOR-only" variant (σ² = σ_stat² + (σ_indep·y)² + (σ_dep·y)²)
    that removes the per-library evaluator cushion. V3 then collapses to V2
    and V4 is unchanged (the dense block is independent of the diagonal flag).
    """
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    if include_eval_diag:
        sigma_eval_diag = (
            df["sigma_eval_diag"].to_numpy(dtype=float)
            if "sigma_eval_diag" in df.columns
            else np.zeros(len(df), dtype=float)
        )
    else:
        sigma_eval_diag = np.zeros(len(df), dtype=float)

    groups = df.groupby(list(group_cols), observed=True, sort=False)
    out_rows = []
    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]
        sed = sigma_eval_diag[ord_pos]
        N = r.size
        label = "/".join(str(p) for p in key)

        # V1 textbook: diag( D + u² + v² + σ_eval_diag² ). The v² term here
        # encodes a *worst-case* point-by-point inflation that ignores σ_dep's
        # within-experiment correlation; included so V1 vs V2/V3/V4 measures
        # exactly the effect of marginalising the correlated modes.
        v1_denom = np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300)
        v1 = float(np.sum(r ** 2 / v1_denom))

        # V2 rank-2 only: D + uuᵀ + vvᵀ — Woodbury closed form (O(N)).
        v2 = _rank2_chi2_woodbury(D, u, v, r)

        # V3 rank-2 + diag eval: (D + σ_eval_diag²) + uuᵀ + vvᵀ — same form.
        v3 = _rank2_chi2_woodbury(D + sed ** 2, u, v, r)

        # V4 rank-2 + dense eval
        if eval_cov:
            block = _select_block(eval_cov, key, sub)
            sigma_v4 = _per_group_sigma(D, u, v, block)
            v4, _ = _solve(sigma_v4, r, label=f"{label}/V4")
        else:
            v4 = float("nan")

        out_rows.append({
            **{col: val for col, val in zip(group_cols, key)},
            "N": N,
            "chi2_v1": v1,
            "chi2_v2": v2,
            "chi2_v3": v3,
            "chi2_v4": v4,
            "chi2_v1_per_N": v1 / N if N > 0 else np.nan,
            "chi2_v2_per_N": v2 / N if N > 0 else np.nan,
            "chi2_v3_per_N": v3 / N if N > 0 else np.nan,
            "chi2_v4_per_N": v4 / N if N > 0 else np.nan,
        })
    return pd.DataFrame(out_rows)


def whitened_residuals_per_experiment(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
) -> pd.Series:
    """Per-row whitened residual z aligned to df.index.

    Satisfies (z**2).groupby(group_cols).sum() == chi2 from
    mahalanobis_chi2_per_experiment, to numerical precision.
    """
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}

    z_full = np.full(len(df), np.nan, dtype=float)
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]

        block = _select_block(eval_cov, key, sub)
        sigma = _per_group_sigma(D, u, v, block)
        label = "/".join(str(p) for p in key)
        _chi2, z = _solve(sigma, r, label=label)
        z_full[ord_pos] = z
    return pd.Series(z_full, index=df.index, name="z")


def whitened_residuals_per_experiment_variant(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    variant: str = "V4",
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    include_eval_diag: bool = True,
) -> pd.Series:
    """Per-row whitened residual z for one of the four chi^2 variants.

    variant ∈ {"V1","V2","V3","V4"}:
      V1 textbook diag  : z_i = r_i / sqrt(D_i + u_i² + v_i² + σ_eval_diag_i²)
      V2 rank-2 only    : Σ = D + u uᵀ + v vᵀ                (dense Cholesky)
      V3 rank-2 + diag  : Σ = (D + diag σ_eval_diag²) + u uᵀ + v vᵀ
      V4 rank-2 + dense : Σ = D + u uᵀ + v vᵀ + Σ_eval_block (current default)

    sum(z²) per group equals chi²_v{n} from chi2_per_experiment_variants
    to numerical precision.

    Setting `include_eval_diag=False` zeroes σ_eval_diag everywhere (EXFOR-only
    diagonal); affects V1 and V3, leaves V2/V4 unchanged.
    """
    variant = variant.upper()
    if variant not in {"V1", "V2", "V3", "V4"}:
        raise ValueError(f"variant must be V1/V2/V3/V4, got {variant!r}")
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    if include_eval_diag:
        sigma_eval_diag = (
            df["sigma_eval_diag"].to_numpy(dtype=float)
            if "sigma_eval_diag" in df.columns
            else np.zeros(len(df), dtype=float)
        )
    else:
        sigma_eval_diag = np.zeros(len(df), dtype=float)

    z_full = np.full(len(df), np.nan, dtype=float)
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        ord_pos = df.index.get_indexer(sub.index)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]
        sed = sigma_eval_diag[ord_pos]
        label = "/".join(str(p) for p in key) + f"/{variant}"

        if variant == "V1":
            denom = np.sqrt(np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300))
            z_full[ord_pos] = r / denom
            continue

        if variant == "V2":
            D_eff, block = D, None
        elif variant == "V3":
            D_eff, block = D + sed ** 2, None
        else:  # V4
            D_eff, block = D, _select_block(eval_cov, key, sub)
        sigma = _per_group_sigma(D_eff, u, v, block)
        _chi2, z = _solve(sigma, r, label=label)
        z_full[ord_pos] = z
    return pd.Series(z_full, index=df.index, name=f"z_{variant.lower()}")


def aggregate_chi2_by_library(
    per_exp: pd.DataFrame,
    lib_labels: Optional[dict] = None,
    *,
    library_col: str = "library",
) -> pd.DataFrame:
    """Sum chi^2 and N across experiments within each library.

    Returns columns: Library (display label), library_key, N, chi2, chi2_per_N.
    """
    grp = per_exp.groupby(library_col, observed=True).agg(
        N=("N", "sum"), chi2=("chi2", "sum")
    ).reset_index()
    grp["chi2_per_N"] = np.where(grp["N"] > 0, grp["chi2"] / grp["N"], np.nan)
    grp = grp.rename(columns={library_col: "library_key"})
    grp["Library"] = (
        grp["library_key"].map(lib_labels) if lib_labels is not None else grp["library_key"]
    )
    return grp[["Library", "library_key", "N", "chi2", "chi2_per_N"]]


def warn_if_not_psd(
    D: np.ndarray, u: np.ndarray, eval_cov_block: Optional[np.ndarray] = None,
    label: str = "",
) -> None:
    """Warn-only PSD diagnostic for one experiment block. Never auto-repairs."""
    bad_D = (~np.isfinite(D)) | (D <= 0)
    bad_u = ~np.isfinite(u)
    n_bad_D = int(bad_D.sum())
    n_bad_u = int(bad_u.sum())
    bad_eval = False
    if eval_cov_block is not None and eval_cov_block.size:
        eig = np.linalg.eigvalsh(0.5 * (eval_cov_block + eval_cov_block.T))
        bad_eval = bool(eig.min() < -1e-10 * max(abs(eig.max()), 1.0))
    if n_bad_D or n_bad_u or bad_eval:
        prefix = f"[{label}] " if label else ""
        warnings.warn(
            f"{prefix}covariance not PSD: "
            f"{n_bad_D} non-positive D, {n_bad_u} non-finite u, "
            f"{'eval_cov has negative eigenvalue' if bad_eval else ''}. "
            "No auto-repair.",
            RuntimeWarning,
            stacklevel=2,
        )
