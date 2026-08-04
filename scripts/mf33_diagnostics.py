"""Warn-only diagnostics for the Phase-2 elastic magnitude (MF33) channel.

Two independent checks, both warn-only (they never gate the MF33 product — the
pipeline replaces the host MF33 over the whole range regardless, matching the
PSD-check policy):

1. **Folded MF3 comparison** (`fold_host_mf3_at_points` + `folded_c0_comparison_stats`).
   Fold the host (JEFF) MF3 elastic cross section through each experiment's TOF
   energy-resolution kernel and compare it against the DCS-derived magnitude
   ``4*pi*c0``.  Kernel broadening is expected to shrink the low-energy
   disagreement dominated by resonance/bin misalignment.  Reports ratio and
   pull distributions (pulls under both the host MF33 and the DCS-derived
   covariance), per-energy-region stats, and the same stats with each major
   campaign excluded in turn.

2. **Between-experiment c0 spread** (`between_experiment_c0_spread`).  Compares
   the empirical scatter of per-experiment magnitude estimates in each bin
   against the manifest ``sigma_sys`` priors — the same role the tau bands play
   for the statistical channel.

The numeric functions are pure (arrays/frames in, stats out) so they unit-test
without the external Fe-56 tape; the folding reuses
``scripts.tof_parameters`` (``get_tof_parameters`` / ``compute_sigma_E`` /
``fold_xs_over_resolution``).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from kika.utils.numerics import average_over_intervals
from scripts.tof_parameters import (
    get_tof_parameters,
    compute_sigma_E,
    fold_xs_over_resolution,
)

FOUR_PI = 4.0 * np.pi


def load_mf33_sidecar(sidecar_dir: str) -> Dict[str, np.ndarray]:
    """Load the pipeline MF33 sidecar into the ``lib["mf33_*"]`` chi2 fields.

    The Phase-2 run writes ``mf33_relative_covariance.npy`` (fine-grid relative
    covariance), ``mf33_energy_grid_ev.npy`` (grid), and
    ``mf33_c0_nominal.npy``.  ``scripts.eval_covariance.build_mf33_block`` reads
    ``lib["mf33_grid_ev"]`` and ``lib["mf33_rel_cov"]``, so a downstream
    (library-normalized) chi2 uses This_work's own MF33 with::

        lib.update(load_mf33_sidecar(this_work_dir))

    Returns
    -------
    dict
        ``{"mf33_grid_ev", "mf33_rel_cov", "mf33_c0_nominal"}``.
    """
    from pathlib import Path

    d = Path(sidecar_dir)
    return {
        "mf33_grid_ev": np.load(d / "mf33_energy_grid_ev.npy"),
        "mf33_rel_cov": np.load(d / "mf33_relative_covariance.npy"),
        "mf33_c0_nominal": np.load(d / "mf33_c0_nominal.npy"),
    }


def bin_average_xs(
    e_ev: np.ndarray,
    xs_b: np.ndarray,
    grid_ev: np.ndarray,
    n_sub: int = 200,
) -> np.ndarray:
    """Width-weighted average of a pointwise cross section over each grid bin.

    Trapezoid integral of the linearly-interpolated ``xs(E)`` over
    ``[grid_ev[i], grid_ev[i+1]]`` divided by the bin width.  Used to project
    the host MF3 onto the fine bin grid so the relative MF33 can be recentred
    on the shipped (host) central value.

    Parameters
    ----------
    e_ev, xs_b : np.ndarray
        Pointwise cross section (energies in eV, values in barns), ascending.
    grid_ev : np.ndarray
        Bin boundaries in eV (N+1 values).
    n_sub : int, default 200
        Sub-samples per bin for the trapezoid rule (the host grid is dense
        through the resonance region; uniform sub-sampling of the interpolant
        is robust to bins wider than the local point spacing).

    Returns
    -------
    np.ndarray
        Bin-averaged cross section (barns), shape ``(N,)``.

    Notes
    -----
    Domain-named adapter over
    :func:`kika.utils.numerics.average_over_intervals`, which holds the
    implementation.  Kept because it reads better at the diagnostic call sites
    and because the eV/barns units are part of the contract here.
    """
    return average_over_intervals(e_ev, xs_b, grid_ev, n_sub=n_sub)


# --------------------------------------------------------------------------- #
# 1. Folded host-MF3 vs 4*pi*c0
# --------------------------------------------------------------------------- #

def fold_host_mf3_at_points(
    mf3_energy_ev: np.ndarray,
    mf3_xs_b: np.ndarray,
    subentries: Sequence[str],
    energies_mev: Sequence[float],
    tof_cache: Dict[str, Any],
    *,
    n_nodes: int = 12,
    default_flight_path_m: float = 27.037,
    default_time_resolution_ns: float = 5.0,
    default_delta_t_is_fwhm: bool = True,
    min_sigma_E_kev: float = 1.0,
) -> np.ndarray:
    """Fold the host MF3 elastic cross section through each point's TOF kernel.

    For point ``i`` at ``energies_mev[i]`` measured in ``subentries[i]``, the
    TOF resolution ``sigma_E`` is resolved from the cache (falling back to the
    ORELA-like defaults) and the host ``sigma(E)`` is Gauss–Hermite averaged
    over ``N(E_i, sigma_E^2)``.

    Returns
    -------
    np.ndarray
        Folded cross section (barns) per point, shape ``(len(energies_mev),)``.
    """
    mf3_energy_ev = np.asarray(mf3_energy_ev, dtype=float)
    mf3_xs_b = np.asarray(mf3_xs_b, dtype=float)
    out = np.empty(len(energies_mev), dtype=float)
    for i, (sub, e_mev) in enumerate(zip(subentries, energies_mev)):
        tof = get_tof_parameters(
            str(sub), tof_cache,
            default_flight_path_m=default_flight_path_m,
            default_time_resolution_ns=default_time_resolution_ns,
            default_delta_t_is_fwhm=default_delta_t_is_fwhm,
        )
        sigma_e = compute_sigma_E(float(e_mev), tof, min_sigma_E_kev=min_sigma_E_kev)
        out[i] = fold_xs_over_resolution(
            mf3_energy_ev, mf3_xs_b, float(e_mev), sigma_e, n_nodes=n_nodes,
        )
    return out


def _folded_stats_block(
    ratio: np.ndarray,
    pull_dcs: np.ndarray,
    pull_host: np.ndarray,
) -> Dict[str, float]:
    """Summary statistics for one (sub)set of folded comparison points."""
    ratio = np.asarray(ratio, dtype=float)
    finite = np.isfinite(ratio)
    ratio = ratio[finite]
    n = int(ratio.size)
    if n == 0:
        return {"n": 0}
    abs_rel = np.abs(ratio - 1.0)

    def _nan_mean(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    def _nan_std(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(np.std(a)) if a.size else float("nan")

    return {
        "n": n,
        "median_ratio": float(np.median(ratio)),
        "mean_abs_rel_diff": float(np.mean(abs_rel)),
        "frac_within_5pct": float(np.mean(abs_rel <= 0.05)),
        "frac_within_10pct": float(np.mean(abs_rel <= 0.10)),
        "pull_dcs_mean": _nan_mean(pull_dcs[finite]),
        "pull_dcs_std": _nan_std(pull_dcs[finite]),
        "pull_host_mean": _nan_mean(pull_host[finite]),
        "pull_host_std": _nan_std(pull_host[finite]),
    }


def folded_c0_comparison_stats(
    df: pd.DataFrame,
    *,
    campaign_col: str = "campaign",
    energy_col: str = "energy_mev",
    folded_col: str = "sigma_folded_b",
    dcs_col: str = "sigma_el_dcs_b",
    rel_dcs_col: str = "rel_sigma_dcs",
    rel_host_col: str = "rel_sigma_host",
    region_edges_mev: Sequence[float] = (2.5,),
    leave_one_out: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Ratio / pull statistics of the DCS magnitude vs the folded host MF3.

    For each comparison point:

    - ``ratio = sigma_el_dcs / sigma_folded``  (4*pi*c0 over the folded host MF3)
    - ``pull_dcs  = (dcs - folded) / (dcs * rel_dcs)``   — residual in DCS sigmas
    - ``pull_host = (dcs - folded) / (folded * rel_host)`` — residual in host sigmas

    Parameters
    ----------
    df : pd.DataFrame
        One row per compared point with the columns named by the ``*_col`` args.
        ``rel_host_col`` may be absent (host MF33 unknown) → host pulls are NaN.
    region_edges_mev : sequence of float
        Interior energy boundaries (MeV) splitting the ``below/above`` regions,
        e.g. ``(2.5,)`` → ``[min, 2.5)`` and ``[2.5, max]``.
    leave_one_out : iterable of str, optional
        Campaign labels to exclude one-at-a-time; each yields an ``overall``
        stats block computed on the remaining points.

    Returns
    -------
    dict
        ``{"overall": {...}, "regions": {label: {...}}, "leave_one_out":
        {campaign: {...}}}``.
    """
    folded = df[folded_col].to_numpy(dtype=float)
    dcs = df[dcs_col].to_numpy(dtype=float)
    rel_dcs = df[rel_dcs_col].to_numpy(dtype=float)
    if rel_host_col in df.columns:
        rel_host = df[rel_host_col].to_numpy(dtype=float)
    else:
        rel_host = np.full(len(df), np.nan)
    energy = df[energy_col].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dcs / folded
        resid = dcs - folded
        pull_dcs = resid / (dcs * rel_dcs)
        pull_host = resid / (folded * rel_host)

    out: Dict[str, Any] = {
        "overall": _folded_stats_block(ratio, pull_dcs, pull_host),
        "regions": {},
        "leave_one_out": {},
    }

    edges = [-np.inf, *sorted(region_edges_mev), np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (energy >= lo) & (energy < hi)
        if not np.any(m):
            continue
        lo_lbl = "min" if lo == -np.inf else f"{lo:g}"
        hi_lbl = "max" if hi == np.inf else f"{hi:g}"
        out["regions"][f"{lo_lbl}-{hi_lbl}MeV"] = _folded_stats_block(
            ratio[m], pull_dcs[m], pull_host[m],
        )

    if leave_one_out is not None and campaign_col in df.columns:
        camp = df[campaign_col].to_numpy()
        for excl in leave_one_out:
            m = camp != excl
            if not np.any(m):
                continue
            out["leave_one_out"][f"excl_{excl}"] = _folded_stats_block(
                ratio[m], pull_dcs[m], pull_host[m],
            )
    return out


# --------------------------------------------------------------------------- #
# 2. Between-experiment c0 spread vs manifest sigma_sys
# --------------------------------------------------------------------------- #

def between_experiment_c0_spread(
    per_experiment_c0: Dict[Any, Dict[str, float]],
    manifest_sigma_sys: Dict[str, float],
    *,
    min_experiments: int = 2,
    exceed_factor: float = 1.0,
) -> pd.DataFrame:
    """Per-bin empirical scatter of per-experiment c0 vs the manifest priors.

    Warn-only check: if the experiments that overlap a bin disagree on the
    magnitude by more than the manifest ``sigma_sys`` priors lead you to expect,
    the systematic budget may be understated (the same intent as the tau bands
    for the statistical channel).

    Parameters
    ----------
    per_experiment_c0 : dict
        ``{bin_key: {experiment_id: c0_estimate}}`` — each experiment's own
        fixed-shape magnitude in that bin.
    manifest_sigma_sys : dict
        ``{experiment_id: sigma_sys_relative}`` representative scalar priors
        (fractions, not percent).
    min_experiments : int
        Bins with fewer overlapping experiments are skipped (no meaningful
        spread).
    exceed_factor : float
        A bin is flagged when ``empirical_rel_std > exceed_factor * expected``.

    Returns
    -------
    pd.DataFrame
        One row per evaluated bin: ``bin_key, n_exp, mean_c0,
        empirical_rel_std, expected_rel_sys, exceeds``.
    """
    rows = []
    for bin_key, by_exp in per_experiment_c0.items():
        vals = np.array([v for v in by_exp.values() if np.isfinite(v)], dtype=float)
        exps = [e for e, v in by_exp.items() if np.isfinite(v)]
        if vals.size < min_experiments:
            continue
        mean_c0 = float(np.mean(vals))
        emp_rel_std = float(np.std(vals, ddof=1) / mean_c0) if mean_c0 != 0 else np.nan
        sys_vals = np.array(
            [float(manifest_sigma_sys.get(e, 0.0)) for e in exps], dtype=float
        )
        # Expected between-experiment relative scatter under the manifest priors:
        # RMS of the per-experiment systematic uncertainties.
        expected = float(np.sqrt(np.mean(sys_vals ** 2))) if sys_vals.size else 0.0
        exceeds = bool(np.isfinite(emp_rel_std) and emp_rel_std > exceed_factor * expected)
        rows.append({
            "bin_key": bin_key,
            "n_exp": int(vals.size),
            "mean_c0": mean_c0,
            "empirical_rel_std": emp_rel_std,
            "expected_rel_sys": expected,
            "exceeds": exceeds,
        })
    return pd.DataFrame(rows)


def log_folded_comparison(stats: Dict[str, Any], logger=None, *, verbose: bool = False) -> None:
    """Emit the folded-comparison summary as warn-only log lines / file comments.

    Follows the transparent-logging convention: a one-line overall summary
    always, per-region + leave-one-out only when ``verbose``.
    """
    if logger is None:
        return
    ov = stats.get("overall", {})
    if ov.get("n", 0) == 0:
        logger.warning("  [MF33] Folded MF3 comparison: no comparable points.", console=True)
        return
    logger.info(
        "  [MF33] Folded MF3 vs 4*pi*c0: "
        f"n={ov['n']}, median ratio={ov['median_ratio']:.3f}, "
        f"mean |Δ|={100 * ov['mean_abs_rel_diff']:.1f}%, "
        f"within 10%={100 * ov['frac_within_10pct']:.0f}%, "
        f"DCS pull μ={ov['pull_dcs_mean']:.2f} σ={ov['pull_dcs_std']:.2f}"
    )
    if not verbose:
        return
    for label, blk in stats.get("regions", {}).items():
        if blk.get("n", 0):
            logger.info(
                f"    [MF33] region {label}: n={blk['n']}, "
                f"median ratio={blk['median_ratio']:.3f}, "
                f"mean |Δ|={100 * blk['mean_abs_rel_diff']:.1f}%"
            )
    for label, blk in stats.get("leave_one_out", {}).items():
        if blk.get("n", 0):
            logger.info(
                f"    [MF33] {label}: n={blk['n']}, "
                f"median ratio={blk['median_ratio']:.3f}, "
                f"mean |Δ|={100 * blk['mean_abs_rel_diff']:.1f}%"
            )
