"""Measure what changing the TOF delta_t convention does to the binning.

Correcting ``delta_t`` from a sigma to a FWHM divides sigma_E by 2.355.  That is
not a uniform rescale of the results: sigma_E sets the width of the Gaussian
overlap kernel that decides how much each EXFOR datapoint contributes to each
analysis bin, so a narrower kernel means fewer points per bin, lower ``n_eff``,
and potentially bins that stop clearing the fitting quality gates entirely.

This runs only the cheap front of the pipeline — EXFOR load, union grid, bin
construction, overlap weights — under both conventions, and reports the shift.
It deliberately does **not** fit anything: the point is to size the change
before committing to a multi-hour production run.

Usage
-----
::

    python -m scripts.diagnose_resolution_change            # uses pipeline config
    python -m scripts.diagnose_resolution_change --csv out.csv

Read the output before re-running the pipeline with ``DELTA_T_IS_FWHM = True``.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.exfor_utils import (
    EnergyBinInfo,
    build_union_energy_grid,
    compute_energy_bins_with_tof_resolution,
    compute_overlap_weight,
)
from scripts.tof_parameters import (
    compute_sigma_E,
    get_tof_parameters,
    load_tof_parameters_file,
)


def _collect_datasets(exfor_cache) -> List[Dict]:
    """Flatten the EXFOR cache into unique (subentry, energy, n_points) records."""
    seen = set()
    out: List[Dict] = []
    for _energy, entries in exfor_cache.items():
        for df, meta in entries:
            if df is None or len(df) == 0:
                continue
            entry = str(meta.get("entry", "?"))
            subentry = str(meta.get("subentry", "?"))
            e_mev = float(meta.get("exfor_energy_mev", _energy))
            key = (entry, subentry, f"{e_mev:.6f}")
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "subentry_id": f"{entry}{subentry}",
                "energy_mev": e_mev,
                "n_points": int(len(df)),
            })
    return out


def _bin_stats(
    bins: Sequence[EnergyBinInfo],
    datasets: Sequence[Dict],
    sigma_by_dataset: Sequence[float],
    min_weight: float,
) -> pd.DataFrame:
    """Per-bin dataset count, point count and Kish effective sample size.

    Vectorised over the full (bins x datasets) grid — the same CDF difference
    ``compute_overlap_weight`` applies, evaluated in one shot.  A Python double
    loop over ~1700 bins and ~1000 datasets is minutes; this is seconds.
    """
    from scipy.stats import norm

    e_ds = np.array([d["energy_mev"] for d in datasets], dtype=float)
    npts_ds = np.array([d["n_points"] for d in datasets], dtype=float)
    sig_ds = np.asarray(sigma_by_dataset, dtype=float)

    lo = np.array([b.bin_lower_mev for b in bins], dtype=float)[:, None]
    hi = np.array([b.bin_upper_mev for b in bins], dtype=float)[:, None]
    bin_sig = np.array([b.sigma_E_mev for b in bins], dtype=float)[:, None]

    # Per-dataset sigma where available, else the bin's own resolution.
    sig = np.where(sig_ds[None, :] > 0, sig_ds[None, :], bin_sig)
    sig = np.where(sig > 0, sig, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = norm.cdf((hi - e_ds[None, :]) / sig) - norm.cdf((lo - e_ds[None, :]) / sig)
    # Zero resolution degenerates to a hard in/out test.
    hard = (e_ds[None, :] >= lo) & (e_ds[None, :] <= hi)
    w = np.where(np.isfinite(w), w, hard.astype(float))

    keep = w >= min_weight
    wk = np.where(keep, w, 0.0)
    sum_w = wk.sum(axis=1)
    sum_w2 = (wk ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(sum_w2 > 0, sum_w ** 2 / sum_w2, 0.0)

    return pd.DataFrame({
        "energy_mev": [b.energy_mev for b in bins],
        "sigma_E_kev": [b.sigma_E_mev * 1e3 for b in bins],
        "n_datasets": keep.sum(axis=1).astype(int),
        "n_points": (keep * npts_ds[None, :]).sum(axis=1).astype(int),
        "n_eff": n_eff,
    })


def run(config, logger=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stats_sigma_convention, stats_fwhm_convention)."""
    from scripts.exfor_to_endf_sampling_v2 import load_exfor_with_new_api

    print("Loading EXFOR data ...")
    exfor_cache, _ = load_exfor_with_new_api(
        exfor_directory=config["EXFOR_DIRECTORY"],
        db_path=config["EXFOR_DB_PATH"],
        source=config["EXFOR_SOURCE"],
        target_zaid=config["TARGET_ZAIDS"],
        projectile=config["TARGET_PROJECTILE"],
        mt=config["MT_NUMBER"],
        energy_range=(config["ENERGY_MIN_MEV"], config["ENERGY_MAX_MEV"]),
        supplementary_json_files=config["SUPPLEMENTARY_JSON_FILES"],
        exclude_experiments=config["EXCLUDE_EXPERIMENTS"],
    )

    grid_ev = build_union_energy_grid(
        exfor_cache=exfor_cache,
        subentries=[tuple(s) for s in config["UNION_GRID_SUBENTRIES"]],
        energy_min_mev=config["ENERGY_MIN_MEV"],
        energy_max_mev=config["ENERGY_MAX_MEV"],
    )
    datasets = _collect_datasets(exfor_cache)
    print(f"  {len(grid_ev)} grid points, {len(datasets)} unique datasets")

    tof_cache = {}
    if config.get("TOF_PARAMETERS_FILE"):
        try:
            tof_cache = load_tof_parameters_file(config["TOF_PARAMETERS_FILE"])
        except FileNotFoundError:
            print("  (TOF parameters file not reachable; using defaults)")

    results = {}
    for is_fwhm in (False, True):
        bins = compute_energy_bins_with_tof_resolution(
            energies_ev=grid_ev,
            energy_min_mev=config["ENERGY_MIN_MEV"],
            energy_max_mev=config["ENERGY_MAX_MEV"],
            delta_t_ns=config["DELTA_T_NS"],
            flight_path_m=config["FLIGHT_PATH_M"],
            reference_grid_ev=None,
            delta_t_is_fwhm=is_fwhm,
        )
        sigma_by_dataset = [
            compute_sigma_E(
                ds["energy_mev"],
                get_tof_parameters(
                    ds["subentry_id"], tof_cache,
                    config["FLIGHT_PATH_M"], config["DELTA_T_NS"],
                    default_delta_t_is_fwhm=is_fwhm,
                ),
            )
            for ds in datasets
        ]
        results[is_fwhm] = _bin_stats(
            bins, datasets, sigma_by_dataset, config["KW_MC_MIN_WEIGHT"],
        )
    return results[False], results[True]


def report(old: pd.DataFrame, new: pd.DataFrame, config) -> None:
    """Print the comparison, focused on what could stop a bin being fittable."""
    print("\n" + "=" * 74)
    print("TOF convention change:  delta_t as sigma  ->  delta_t as FWHM")
    print("=" * 74)

    print(f"\n{'quantity':<22}{'sigma conv.':>16}{'FWHM conv.':>16}{'ratio':>12}")
    for col, fmt in (("sigma_E_kev", "{:.2f}"), ("n_datasets", "{:.1f}"),
                     ("n_points", "{:.0f}"), ("n_eff", "{:.2f}")):
        a, b = old[col].median(), new[col].median()
        ratio = b / a if a else float("nan")
        print(f"  median {col:<15}{fmt.format(a):>16}{fmt.format(b):>16}{ratio:>12.3f}")

    gates = {
        "MIN_ANGULAR_POINTS": ("n_points", config.get("MIN_ANGULAR_POINTS", 4)),
        "MIN_POINTS_PER_BAND": ("n_points", config.get("MIN_POINTS_PER_BAND", 4)),
        "N_EFF_WARNING_THRESHOLD": ("n_eff", config.get("N_EFF_WARNING_THRESHOLD", 5.0)),
        "TAU_PRIOR_NEFF_THRESHOLD": ("n_eff", config.get("TAU_PRIOR_NEFF_THRESHOLD", 5.0)),
    }
    print(f"\n{'gate':<28}{'below (sigma)':>16}{'below (FWHM)':>16}{'newly below':>14}")
    for name, (col, thr) in gates.items():
        a = int((old[col] < thr).sum())
        b = int((new[col] < thr).sum())
        newly = int(((new[col] < thr) & (old[col] >= thr)).sum())
        print(f"  {name:<26}{a:>16}{b:>16}{newly:>14}")

    lost_old = int((old["n_datasets"] == 0).sum())
    lost_new = int((new["n_datasets"] == 0).sum())
    print(f"\n  bins with NO contributing data: {lost_old} -> {lost_new} "
          f"(of {len(old)})")

    worst = (old["n_eff"] - new["n_eff"]).nlargest(5)
    print("\n  largest n_eff drops:")
    for i in worst.index:
        print(f"    E={old.loc[i, 'energy_mev']:.4f} MeV  "
              f"n_eff {old.loc[i, 'n_eff']:.2f} -> {new.loc[i, 'n_eff']:.2f}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="Write the per-bin table here")
    args = ap.parse_args(argv)

    from scripts.exfor_to_endf_sampling_v2 import _collect_config_constants

    config = _collect_config_constants()
    old, new = run(config)
    report(old, new, config)

    if args.csv:
        merged = old.add_suffix("_sigma").join(new.add_suffix("_fwhm"))
        merged.to_csv(args.csv, index=False)
        print(f"per-bin table written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
