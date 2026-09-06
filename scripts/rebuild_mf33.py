"""Rebuild MF33 from a completed pipeline run, without repeating the Monte Carlo.

The expensive part of the MF33 channel is the two-pass c0 Monte Carlo, and the
pipeline already persists its result as ``mf33_absolute_covariance.npy``.
Everything downstream of that — the host denominator, the recentring, the SNR
guard, the adaptive collapse and the ENDF splice — is seconds of work.  So a run
whose MF33 write failed, or that predates a fix to any of those steps, can be
repaired in place instead of being re-run.

This calls exactly the same :mod:`scripts.mf33_build` functions the pipeline
calls, so a rebuilt section is identical to what a fresh run would emit.

Usage
-----
::

    python -m scripts.rebuild_mf33 --run-dir /path/to/new_test_81 --out-dir /path/out

By default the ENDF files are **copied** to ``--out-dir`` and patched there,
leaving the run directory untouched; pass ``--in-place`` to patch the originals.

Inputs it needs from the run directory
--------------------------------------
``mf33_absolute_covariance.npy``, ``mf33_energy_grid_ev.npy``,
``mf33_c0_nominal.npy``, ``nominal_fits.parquet`` and ``run_metadata.json``.
``mf33_c0_host.npy`` is deliberately **not** reused: a run that predates the
folded-PENDF denominator saved a box average of File 3, which is the wrong
quantity (and is zero inside a resolved resonance range).

Per-bin ``sigma_E`` is taken from ``nominal_fits.parquet`` when present, and
otherwise recomputed with ``compute_energy_resolution_tof`` from the
``FLIGHT_PATH_M`` / ``DELTA_T_NS`` recorded in ``run_metadata.json``.  That is a
closed form in the bin energy and those two constants — the same call the
pipeline makes — so the reconstruction is exact, not an approximation.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from scripts.exfor_utils import EnergyBinInfo, compute_energy_resolution_tof
from scripts.mf33_build import (
    build_mf33_denominator,
    build_mf33_matrices,
    write_mf33_products,
)

_LOG = logging.getLogger("rebuild_mf33")


class _ConsoleLogger:
    """Adapter matching the pipeline DualLogger surface (info/warning accept console=)."""

    def info(self, msg: str, **_kw) -> None:
        _LOG.info(msg)

    def warning(self, msg: str, **_kw) -> None:
        _LOG.warning(msg)

    def error(self, msg: str, **_kw) -> None:
        _LOG.error(msg)


def load_energy_bins(run_dir: Path, grid_ev: np.ndarray, config: dict) -> List[EnergyBinInfo]:
    """Rebuild the EnergyBinInfo list the folding kernel needs.

    ``sigma_E_mev`` comes from ``nominal_fits.parquet`` when the run recorded it;
    older runs get it recomputed from the TOF constants in ``run_metadata.json``.
    """
    fits = pd.read_parquet(run_dir / "nominal_fits.parquet")
    fits = fits[fits["has_data"]] if "has_data" in fits.columns else fits
    fits = fits.reset_index(drop=True)

    n_bins = len(grid_ev) - 1
    if len(fits) != n_bins:
        raise ValueError(
            f"nominal_fits.parquet has {len(fits)} has-data rows but the MF33 grid "
            f"describes {n_bins} bins; they must correspond one-to-one"
        )

    energies_mev = np.asarray(fits["energy_mev"], dtype=float)
    if "sigma_E_mev" in fits.columns and np.all(np.isfinite(fits["sigma_E_mev"])):
        sigma_E = np.asarray(fits["sigma_E_mev"], dtype=float)
        _LOG.info("  sigma_E taken from nominal_fits.parquet")
    else:
        flight_path_m = float(config["FLIGHT_PATH_M"])
        delta_t_ns = float(config["DELTA_T_NS"])
        # Runs predating the DELTA_T_IS_FWHM knob (<= 81) read delta_t as a
        # sigma, so absent the key we must reproduce that, not today's default.
        is_fwhm = bool(config.get("DELTA_T_IS_FWHM", False))
        sigma_E = np.array([
            compute_energy_resolution_tof(
                E_mev=e, delta_t_ns=delta_t_ns, flight_path_m=flight_path_m,
                delta_t_is_fwhm=is_fwhm,
            )
            for e in energies_mev
        ])
        _LOG.info(
            f"  sigma_E recomputed from run_metadata (L={flight_path_m} m, "
            f"dt={delta_t_ns} ns, delta_t_is_fwhm={is_fwhm}): "
            f"{1e3 * sigma_E.min():.1f}-{1e3 * sigma_E.max():.1f} keV"
        )

    return [
        EnergyBinInfo(
            index=int(fits["energy_index"].iloc[i]),
            energy_ev=energies_mev[i] * 1e6,
            energy_mev=float(energies_mev[i]),
            sigma_E_mev=float(sigma_E[i]),
            bin_lower_mev=float(grid_ev[i]) / 1e6,
            bin_upper_mev=float(grid_ev[i + 1]) / 1e6,
        )
        for i in range(n_bins)
    ]


def rebuild(
    run_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    in_place: bool = False,
    njoy_exe: Optional[str] = None,
    endf_file: Optional[str] = None,
    mg_representation: str = "multigroup",
    rebuild_mt1: bool = False,
) -> List[str]:
    """Rebuild and splice MF33 into the run's ENDF products.  Returns paths written."""
    logger = _ConsoleLogger()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    config = meta["config"]

    cov_abs = np.load(run_dir / "mf33_absolute_covariance.npy")
    grid_ev = np.load(run_dir / "mf33_energy_grid_ev.npy")
    c0_dcs = np.load(run_dir / "mf33_c0_nominal.npy")
    _LOG.info(
        f"  Loaded MF33 sidecars: cov {cov_abs.shape}, grid {len(grid_ev)} edges, "
        f"[{grid_ev[0] / 1e6:.4f}, {grid_ev[-1] / 1e6:.4f}] MeV"
    )

    energy_bins = load_energy_bins(run_dir, grid_ev, config)
    mt = int(config.get("MT_NUMBER", 2))
    endf_file = endf_file or config["ENDF_FILE"]
    njoy_exe = njoy_exe or config["ACE_NJOY_EXE"]

    # Resolve and create the destination up front: the collapse writes its
    # boundary-decision CSV there, and a bad --out-dir must fail now rather than
    # after the PENDF fold has already cost minutes.
    if in_place:
        dest = run_dir
    else:
        if out_dir is None:
            raise ValueError("--out-dir is required unless --in-place is given")
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir

    denominator = build_mf33_denominator(
        endf_file,
        energy_bins,
        mt=mt,
        njoy_exe=njoy_exe,
        pendf_tolerance=float(config.get("MF33_PENDF_TOLERANCE", 0.001)),
        pendf_cache_dir=config.get("MF33_PENDF_CACHE_DIR"),
        grid_ev=grid_ev,
        logger=logger,
    )

    products = build_mf33_matrices(
        cov_abs_fine=cov_abs,
        sigma_host_bin=denominator.sigma_host_bin,
        energy_bins=energy_bins,
        grid_fine_ev=grid_ev,
        c0_dcs=c0_dcs,
        regularize_near_zero=bool(config.get("REGULARIZE_NEAR_ZERO_REL_UNC", True)),
        snr_threshold=float(config.get("NEAR_ZERO_SNR_THRESHOLD", 1.0)),
        n_neighbors=int(config.get("NEAR_ZERO_N_NEIGHBORS", 3)),
        rho_min=float(config.get("MF33_MULTIGROUP_RHO_MIN", 0.85)),
        sigma_ratio_max=config.get("MF33_MULTIGROUP_SIGMA_RATIO_MAX", 5.0),
        diagnostics_file=dest / "mf33_boundary_decisions.csv",
        logger=logger,
    )

    # Resolve the two ENDF products.
    fine_src = next(iter(sorted(run_dir.glob("*_nominal.endf"))), None)
    mg_src = next(iter(sorted(run_dir.glob("*_nominal_mg.endf"))), None)
    if fine_src is None and mg_src is None:
        raise FileNotFoundError(f"No *_nominal.endf or *_nominal_mg.endf in {run_dir}")

    if in_place:
        fine_dst, mg_dst = fine_src, mg_src
    else:
        fine_dst = mg_dst = None
        for src, name in ((fine_src, "fine"), (mg_src, "multigroup")):
            if src is None:
                continue
            dst = dest / src.name
            _LOG.info(f"  Copying {name} product -> {dst} ({src.stat().st_size / 1e6:.0f} MB)")
            shutil.copyfile(src, dst)
            if name == "fine":
                fine_dst = dst
            else:
                mg_dst = dst

    # za/awr/mat default to None so the merge inherits them from the host section.
    written = write_mf33_products(
        products,
        fine_endf=str(fine_dst) if fine_dst else None,
        mg_endf=str(mg_dst) if mg_dst else None,
        mt=mt,
        mg_representation=mg_representation,
        rebuild_mt1=rebuild_mt1,
        pendf_path=denominator.pendf_path,
        logger=logger,
    )

    np.save(dest / "mf33_relative_covariance.npy", products.rel_fine)
    np.save(dest / "mf33_c0_host.npy", products.c0_host)
    np.save(dest / "mf33_multigroup_grid_ev.npy", products.multigroup.group_boundaries_ev)
    np.save(
        dest / "mf33_multigroup_relative_covariance.npy",
        products.multigroup.cov_rel_grouped,
    )
    _LOG.info(f"  Refreshed sidecars in {dest} (PENDF: {denominator.pendf_path.name})")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Completed pipeline output directory")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write patched copies (default alongside --in-place)")
    ap.add_argument("--in-place", action="store_true",
                    help="Patch the ENDF files in --run-dir instead of copying")
    ap.add_argument("--njoy-exe", default=None,
                    help="Override the NJOY executable from run_metadata.json")
    ap.add_argument("--endf-file", default=None,
                    help="Override the host ENDF tape from run_metadata.json")
    ap.add_argument("--mg-representation", choices=("multigroup", "fine"),
                    default="multigroup",
                    help="MF33 grid for the _mg file; 'fine' keeps the full "
                         "MF33 there while MF34 stays collapsed")
    ap.add_argument("--rebuild-mt1", action="store_true",
                    help="Also rebuild MF33 MT1 from the partials over the "
                         "analysis window, cross terms zeroed")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    written = rebuild(
        args.run_dir,
        out_dir=args.out_dir,
        in_place=args.in_place,
        njoy_exe=args.njoy_exe,
        endf_file=args.endf_file,
        mg_representation=args.mg_representation,
        rebuild_mt1=args.rebuild_mt1,
    )
    print("\nMF33 written to:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
