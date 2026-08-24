"""Perturb MF3 and MF4 from ONE draw of the joint (sigma, a_l) covariance.

    ENDF (covariance)  ->  joint C  ->  one draw  ->  split
                                                        |-> MF4 on the base tape
                                                        |-> MF3 on the PENDF
                                                             -> NJOY ACER -> ACE

WHAT IS DIFFERENT FROM `combined_perturbation`
----------------------------------------------
`perturb_ENDF_PENDF_combined` draws MF33 and MF34 separately and pairs replica
*i* of each. That is right for every released evaluation and wrong for a tape
that ships ``Cov(sigma, a_l)`` in MF34's a₀ blocks, in two independent ways:

1. the cross term is simply absent from the ensemble; and
2. pairing two *balanced* Sobol sets by index is itself a defect — measured at
   18-21 % too narrow on the PFNS driver — because each set is equidistributed
   over its own cube and index-pairing correlates the two.

Here there is one covariance, one seed, one decomposition and one draw. Both
failure modes are structural rather than fixed: there is no second set to pair.

THE THREE ENSEMBLES
-------------------
``which`` selects a principal submatrix of the same assembled joint:

===========  ============================  =================  =================
which        matrix                        ENDF per replica   PENDF per replica
===========  ============================  =================  =================
``joint``    the whole C, cross included   perturbed MF4      perturbed MF3
``mf34``     the shape block               perturbed MF4      the nominal, shared
``mf33``     the magnitude block           the base tape      perturbed MF3
===========  ============================  =================  =================

A principal submatrix of a PSD matrix is PSD, so restricting introduces
nothing, and ``mf34`` + ``mf33`` is exactly ``joint`` with the cross zeroed —
which makes the downstream comparison a one-variable contrast.

⚑ The frozen half is **not** written 512 times as an identical copy. It would
make the three sets prettier and it would cost ~14 GB (mf34's PENDFs) and
~25 GB (mf33's ENDFs) of duplicate bytes; the share was at 93 % when this was
written. Stage B takes the nominal artefact's path for every pair instead,
which is the same file NJOY would have read.

TWO TAPES, ON PURPOSE
---------------------
``covariance_endf`` is READ once (the 570 MB deliverable). ``base_endf`` is
WRITTEN once per replica and must be the covariance-free tape from
:func:`kika.sampling.base_tape.build_base_tape` — same MF3 and MF4, no
MF31/32/33/34/35. See that module for why.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..endf import read_endf
from .core import draw_samples
from .combined_perturbation import _run_pair_b
from .endf_perturbation import (
    _parameter_mapping_from_index,
    perturb_ENDF_files,
)
from .joint_mf33_mf34 import JOINT_SETS, load_joint_mf33_mf34, restrict_joint
from .pendf_perturbation import perturb_PENDF_files
from .utils import DualLogger, _get_logger, _set_logger

__all__ = ["perturb_joint_mf33_mf34", "load_or_build_joint", "joint_report"]


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_or_build_joint(
    covariance_endf: str,
    *,
    mt: int = 2,
    isotope: int = 26056,
    l_max: int = 6,
    require_cross: bool = True,
    cache_path: Optional[str] = None,
    logger=None,
):
    """:func:`load_joint_mf33_mf34`, with an optional on-disk cache.

    Reading the deliverable costs ~6 minutes and assembling the joint costs
    more; the three ensembles need the same matrix, so it is built once.

    ⚠ **The cache is keyed on the sha256 of the tape**, not on its name and
    size. A name-and-size key has already collided in this project — a run's
    tape and the analytic deliverable were the same name and the same number of
    bytes — and the failure is silent: the wrong covariance, no error.
    """
    if cache_path is None:
        return load_joint_mf33_mf34(
            covariance_endf, mt=mt, isotope=isotope, l_max=l_max,
            require_cross=require_cross, logger=logger)

    digest = _sha256(str(covariance_endf))
    cache = Path(cache_path)
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            if str(z["source_sha256"]) == digest:
                index = json.loads(str(z["index_json"]))
                index["sigma_grid_ev"] = z["sigma_grid_ev"]
                index["triplets"] = [tuple(t) for t in index["triplets"]]
                index["widths"] = {tuple(k): v for k, v in index["widths"]}
                index["grids"] = {tuple(k): np.asarray(v)
                                  for k, v in index["grids"]}
                key = tuple(index.pop("_key"))
                if logger:
                    logger.info(f"  [INFO] [JOINT] cache hit {cache}")
                return [(key, z["joint"])], index
            if logger:
                logger.warning(
                    f"  [WARN] [JOINT] cache {cache} was built from a different "
                    f"tape (sha256 mismatch); rebuilding")

    blocks, index = load_joint_mf33_mf34(
        covariance_endf, mt=mt, isotope=isotope, l_max=l_max,
        require_cross=require_cross, logger=logger)
    (key, joint), = blocks
    serial = dict(index)
    serial["_key"] = list(key)
    serial["sigma_grid_ev"] = None
    serial["widths"] = [[list(k), int(v)] for k, v in index["widths"].items()]
    serial["grids"] = [[list(k), np.asarray(v).tolist()]
                       for k, v in index["grids"].items()]
    serial["triplets"] = [list(t) for t in index["triplets"]]
    serial["a_block_key"] = None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, joint=joint, sigma_grid_ev=np.asarray(index["sigma_grid_ev"]),
        source_sha256=digest,
        index_json=json.dumps(serial, default=str))
    if logger:
        logger.info(f"  [INFO] [JOINT] cache written {cache}")
    return blocks, index


def joint_report(matrix: np.ndarray, index: Dict[str, Any],
                 *, eigen: bool = True) -> Dict[str, Any]:
    """Symmetry, correlation bound and spectrum of the assembled joint.

    Reported, never repaired. A tape read back at seven significant digits is
    allowed a ``max|rho|`` of about 1 + 5e-6 — the ASCII round trip, measured —
    and anything beyond that is the file, not the arithmetic. Deciding what to
    do about it is the evaluator's call and belongs in a log, not in a silent
    projection.
    """
    m = np.asarray(matrix, dtype=float)
    out: Dict[str, Any] = {
        "dimension": int(m.shape[0]),
        "n_sigma": int(index.get("n_sigma", 0)),
        "n_a": int(index.get("n_a", 0)),
        "has_cross": bool(index.get("has_cross", False)),
        "cross_orders": list(index.get("cross_orders", [])),
        "asymmetry": float(np.abs(m - m.T).max()),
        "n_non_finite": int((~np.isfinite(m)).sum()),
    }
    d = np.sqrt(np.clip(np.diag(m), 0.0, None))
    nz = d > 0
    out["n_zero_variance"] = int((~nz).sum())
    if nz.any():
        r = m[np.ix_(nz, nz)] / np.outer(d[nz], d[nz])
        out["max_abs_rho"] = float(np.abs(r).max())
    if eigen:
        w = np.linalg.eigvalsh(0.5 * (m + m.T))
        out.update(
            lam_min=float(w.min()), lam_max=float(w.max()),
            lam_min_over_max=float(w.min() / w.max()) if w.max() else None,
            n_negative=int((w < 0).sum()),
            negative_mass_fraction=float(
                -w[w < 0].sum() / w[w > 0].sum()) if (w > 0).any() else None,
        )
    return out


def _split_factors(factors: np.ndarray, index: Dict[str, Any], which: str):
    """The drawn (n_samples, n) block -> what each applier expects."""
    m0 = int(index.get("n_sigma", 0))
    endf_pre = pendf_pre = None
    if which in ("joint", "mf33"):
        sigma = factors[:, :m0] if which == "joint" else factors
        pendf_pre = {
            "factors": np.ascontiguousarray(sigma),
            "mt_to_param_block": {int(index["mt"]): slice(0, sigma.shape[1])},
            "union_grid": np.asarray(index["sigma_grid_ev"], dtype=float).tolist(),
        }
    if which in ("joint", "mf34"):
        a = factors[:, m0:] if which == "joint" else factors
        param_mapping, energy_grids = _parameter_mapping_from_index(index)
        endf_pre = {
            "factors": np.ascontiguousarray(a),
            "param_mapping": param_mapping,
            "energy_grids": energy_grids,
        }
    return endf_pre, pendf_pre


def perturb_joint_mf33_mf34(
    covariance_endf: str,
    base_endf: str,
    num_samples: int,
    *,
    which: str = "joint",
    mt: int = 2,
    isotope: int = 26056,
    l_max: int = 6,
    # --- the draw ---------------------------------------------------------
    space: str = "linear",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    psd_method: str = "none",
    null_tol: Optional[float] = None,
    seed: Optional[int] = 42,
    joint_cache: Optional[str] = None,
    # --- MF4 application --------------------------------------------------
    enforce_positivity: bool = True,
    positivity_check_points: int = 101,
    energy_ranges: Optional[List[Tuple[float, float]]] = None,
    # --- PENDF / NJOY -----------------------------------------------------
    generate_ace: bool = True,
    pendf_cache_dir: Optional[str] = None,
    pendf_tolerance: float = 1.0e-3,
    pendf_timeout_s: float = 600.0,
    njoy_exe: Optional[str] = None,
    ace_temperatures: Optional[Sequence[float]] = None,
    ace_extensions: Optional[Sequence[str]] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = "NJOY 2016.78",
    xsdir_file: Optional[str] = None,
    keep_njoy_io: bool = True,
    # --- run --------------------------------------------------------------
    output_dir: str = ".",
    nprocs: int = 1,
    dry_run: bool = False,
    verbose_diagnostics: int = 1,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """One ensemble of *num_samples* replicas. Returns the run summary.

    ``space`` must be ``"linear"``. MF34's Legendre coefficients change sign
    routinely, so a multiplicative log-space factor is not meaningful for them
    and ``perturb_ENDF_files`` refuses it — and a joint draw cannot use two
    spaces. The consequence is deliberate and must be reported: the magnitude
    half is drawn linear here, whereas ``perturb_PENDF_files`` alone defaults to
    log. Run the ``mf33`` ensemble both ways if the difference matters.

    ``psd_method`` defaults to ``"none"``, matching ``perturb_ENDF_files``, so
    that the ``mf34`` ensemble reproduces today's draw exactly when the same
    seed is used. The spectrum is reported either way — see :func:`joint_report`
    — so choosing something else is a measured decision rather than a default.
    """
    if which not in JOINT_SETS:
        raise ValueError(f"which must be one of {JOINT_SETS}, got {which!r}")
    if space != "linear":
        raise ValueError(
            "space must be 'linear'. MF34's a_l change sign, so exp(y) is not a "
            "factor of anything there, and one draw cannot be in two spaces."
        )
    if generate_ace and njoy_exe is None:
        raise ValueError("njoy_exe must be given when generate_ace=True")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = os.path.join(output_dir, f"joint_{which}_{timestamp}.log")
    logger = DualLogger(log_file)
    _set_logger(logger)

    logger.info(f"[INFO] [JOINT] set={which} n={num_samples} seed={seed}",
                console=True)
    logger.info(f"[INFO] [JOINT] covariance read from : {covariance_endf}")
    logger.info(f"[INFO] [JOINT] replicas written onto: {base_endf}")

    summary: Dict[str, Any] = {
        "which": which, "num_samples": int(num_samples), "seed": seed,
        "timestamp": timestamp, "log_file": log_file,
        "covariance_endf": str(covariance_endf), "base_endf": str(base_endf),
        "output_dir": os.path.abspath(output_dir),
    }

    # ---- 1. the joint, once ---------------------------------------------
    t0 = time.time()
    blocks, index = load_or_build_joint(
        covariance_endf, mt=mt, isotope=isotope, l_max=l_max,
        require_cross=(which == "joint"), cache_path=joint_cache, logger=logger)
    logger.info(f"  [INFO] [JOINT] assembled in {time.time() - t0:.1f}s: "
                f"{index['n_sigma']} sigma + {index['n_a']} a_l rows, "
                f"cross orders {index['cross_orders']}")
    # The spectrum of the FULL joint is the same object in all three runs (the
    # cache is keyed on the tape's sha256), and eigvalsh on 13 003 rows is
    # minutes and gigabytes. It is taken where it is not redundant: on whatever
    # matrix this run actually samples.
    summary["joint_report_full"] = joint_report(blocks[0][1], index,
                                                eigen=(which == "joint"))
    logger.info(f"  [INFO] [JOINT] full: {summary['joint_report_full']}")

    sub_blocks, sub_index = restrict_joint(blocks, index, which)
    sub_index["mt"] = int(mt)
    if which != "joint":
        summary["joint_report_restricted"] = joint_report(sub_blocks[0][1],
                                                          sub_index)
        logger.info(f"  [INFO] [JOINT] {which}: "
                    f"{summary['joint_report_restricted']}")

    # ---- 2. one draw -----------------------------------------------------
    (sub_key, sub_matrix), = sub_blocks
    samples, diagnostics = draw_samples(
        sub_blocks, int(num_samples), space=space, returns="factors",
        decomposition_method=decomposition_method,
        sampling_method=sampling_method, seed=seed, psd_method=psd_method,
        null_tol=null_tol, dtype=np.float64,
        verbose=verbose_diagnostics > 0, logger=logger)
    factors = samples[sub_key]
    diag = diagnostics[sub_key]
    summary["draw"] = {k: (v if np.isscalar(v) or v is None else str(v))
                       for k, v in diag.items()}
    logger.info(f"  [INFO] [JOINT] drew {factors.shape}, rank {diag['rank']}, "
                f"{diag['n_null']} null, realised cov error "
                f"{diag['realised_covariance_error']:.3e}")

    endf_pre, pendf_pre = _split_factors(factors, sub_index, which)

    # ⚑ Drop the covariance BEFORE anything forks. Both appliers and Stage B
    # build a multiprocessing Pool, and fork gives every worker a copy-on-write
    # view of this process; CPython's refcounts write to the object headers, so
    # pages get copied for arrays nobody reads. With 40 workers and a 1.35 GB
    # joint that is a large amount of memory bought for nothing. The factors are
    # all that is needed from here on -- 512 x 13 003 x 8 = 53 MB.
    del blocks, sub_blocks, sub_matrix, samples
    gc.collect()

    # ---- 3. MF4 ----------------------------------------------------------
    if endf_pre is not None:
        t0 = time.time()
        logger.info("#-- MF4: writing perturbed ENDF tapes -----------------")
        perturb_ENDF_files(
            endf_files=str(base_endf), mt_list=[int(mt)],
            legendre_coeffs=list(range(1, int(l_max) + 1)),
            num_samples=int(num_samples), space=space,
            decomposition_method=decomposition_method,
            psd_method=psd_method, sampling_method=sampling_method,
            enforce_positivity=enforce_positivity,
            positivity_check_points=positivity_check_points,
            output_dir=output_dir, seed=seed, nprocs=nprocs, dry_run=dry_run,
            verbose=verbose_diagnostics > 0, generate_ace=False,
            energy_ranges=energy_ranges, log_file=log_file,
            precomputed=[endf_pre],
        )
        _set_logger(logger)
        logger.info(f"#-- MF4 done ({time.time() - t0:.1f}s) ----------------")

    # ---- 4. MF3 on the PENDF --------------------------------------------
    if pendf_pre is not None:
        t0 = time.time()
        logger.info("#-- MF3: writing perturbed PENDF tapes ----------------")
        summary["pendf"] = perturb_PENDF_files(
            endf_files=str(base_endf), mt_list=[int(mt)],
            num_samples=int(num_samples), output_formats=("pendf",),
            keep_njoy_io=keep_njoy_io, pendf_cache_dir=pendf_cache_dir,
            pendf_tolerance=pendf_tolerance, pendf_timeout_s=pendf_timeout_s,
            space=space, decomposition_method=decomposition_method,
            sampling_method=sampling_method, psd_method=psd_method,
            energy_ranges=energy_ranges,
            njoy_exe=str(njoy_exe) if njoy_exe is not None else None,
            output_dir=output_dir, seed=seed, nprocs=nprocs, dry_run=dry_run,
            verbose_diagnostics=verbose_diagnostics, log_file=log_file,
            precomputed=[pendf_pre],
        )
        _set_logger(logger)
        logger.info(f"#-- MF3 done ({time.time() - t0:.1f}s) ----------------")
        # `perturb_PENDF_files` logs and carries on when RECONR is unavailable,
        # which is right for a multi-isotope run and wrong here: this driver has
        # exactly one isotope, and an ensemble with no perturbed PENDF is not a
        # partial result, it is a run that produced nothing while returning a
        # summary that looks like a run.
        if not dry_run:
            n_ok = sum(int(iso.get("n_pendf_ok", 0))
                       for iso in (summary["pendf"].get("isotopes") or {}).values())
            if n_ok == 0:
                raise RuntimeError(
                    f"the MF3 stage wrote 0 perturbed PENDFs for set '{which}'. "
                    f"The usual cause is NJOY: RECONR has to run before MF3 can "
                    f"be perturbed at all. See {log_file}."
                )

    if dry_run or not generate_ace:
        logger.info(f"[INFO] [JOINT] Stage B skipped "
                    f"({'dry_run' if dry_run else 'generate_ace=False'})",
                    console=True)
        return summary

    # ---- 5. Stage B: ACER over the pairs --------------------------------
    summary.update(_stage_b(
        base_endf=str(base_endf), which=which, num_samples=int(num_samples),
        output_dir=output_dir, njoy_exe=str(njoy_exe),
        ace_temperatures=[float(t) for t in (ace_temperatures or [293.6])],
        ace_extensions=(list(ace_extensions) if ace_extensions else None),
        ace_library_name=ace_library_name, ace_njoy_version=ace_njoy_version,
        xsdir_file=xsdir_file, keep_njoy_io=keep_njoy_io,
        pendf_cache_dir=pendf_cache_dir, pendf_tolerance=pendf_tolerance,
        pendf_timeout_s=pendf_timeout_s, nprocs=nprocs, logger=logger))
    return summary


def _stage_b(*, base_endf, which, num_samples, output_dir, njoy_exe,
             ace_temperatures, ace_extensions, ace_library_name,
             ace_njoy_version, xsdir_file, keep_njoy_io, pendf_cache_dir,
             pendf_tolerance, pendf_timeout_s, nprocs, logger):
    """Pair each replica's (ENDF, PENDF) and run one ACER per pair.

    The frozen family contributes the SAME nominal file to every pair — the
    base tape itself for ``mf33``, and the cached RECONR PENDF for ``mf34``.
    That is the file NJOY would have read anyway; writing 512 byte-identical
    copies of it first would only cost disk.
    """
    from kika.processing.njoy_pendf_cache import get_or_create_pendf

    zaid = int(read_endf(base_endf).zaid)
    base, ext = os.path.splitext(os.path.basename(base_endf))
    endf_root = os.path.join(output_dir, "endf", str(zaid))
    pendf_root = os.path.join(output_dir, "pendf", str(zaid))

    nominal_pendf = None
    if which == "mf33" or which == "mf34":
        cache_dir = Path(pendf_cache_dir) if pendf_cache_dir else None
        if which == "mf34":
            nominal_pendf = str(get_or_create_pendf(
                base_endf, tolerance=pendf_tolerance, njoy_exe=njoy_exe,
                cache_dir=cache_dir, timeout_s=pendf_timeout_s,
                keep_njoy_io_dir=(os.path.join(output_dir, "njoy_files",
                                               "recon", str(zaid))
                                  if keep_njoy_io else None)))
            logger.info(f"  [INFO] [JOINT] frozen MF3: every pair reads the "
                        f"nominal PENDF {nominal_pendf}")
        else:
            logger.info("  [INFO] [JOINT] frozen MF4: every pair reads the "
                        f"base tape {base_endf}")

    pair_args, missing = [], []
    for s in range(num_samples):
        tag = f"{s+1:04d}"
        e = (base_endf if which == "mf33"
             else os.path.join(endf_root, tag, f"{base}_{tag}{ext}"))
        p = (nominal_pendf if which == "mf34"
             else os.path.join(pendf_root, tag, f"{base}_{tag}.pendf"))
        if not os.path.exists(e):
            missing.append((s, f"ENDF missing: {e}")); continue
        if p is None or not os.path.exists(p):
            missing.append((s, f"PENDF missing: {p}")); continue
        pair_args.append({
            "endf_path": e, "pendf_path": p, "sample_index": s, "zaid": zaid,
            "njoy_exe": njoy_exe, "ace_temperatures": ace_temperatures,
            "ace_extensions": ace_extensions,
            "ace_library_name": ace_library_name,
            "ace_njoy_version": ace_njoy_version, "xsdir_file": xsdir_file,
            "output_dir": output_dir, "keep_njoy_io": keep_njoy_io,
        })

    for s, msg in missing[:5]:
        logger.warning(f"  [WARN] [JOINT] sample {s+1:04d}: {msg}")
    if len(missing) > 5:
        logger.warning(f"  [WARN] [JOINT] {len(missing)} samples missing inputs")
    if not pair_args:
        return {"n_attempted": 0, "n_ace_ok": 0,
                "n_missing_inputs": len(missing)}

    logger.info(f"  [INFO] [JOINT] ACER on {len(pair_args)}/{num_samples} pairs"
                + (f" across {min(nprocs, len(pair_args))} workers"
                   if nprocs > 1 else ""))
    if nprocs > 1 and len(pair_args) > 1:
        with Pool(processes=min(nprocs, len(pair_args))) as pool:
            results = pool.map(_run_pair_b, pair_args, chunksize=1)
    else:
        results = [_run_pair_b(a) for a in pair_args]

    n_ok = sum(1 for r in results if r.get("success"))
    errs: Dict[str, int] = {}
    for r in results:
        for e in r.get("errors", []):
            errs[e] = errs.get(e, 0) + 1
    for e, c in sorted(errs.items(), key=lambda kv: -kv[1])[:10]:
        logger.warning(f"  [WARN] [JOINT]   ({c}x) {e}")
    logger.info(f"  [INFO] [JOINT] Stage B {n_ok}/{len(pair_args)} ACEs",
                console=True)
    return {"n_attempted": len(pair_args), "n_ace_ok": n_ok,
            "n_missing_inputs": len(missing), "zaid": zaid,
            "nominal_pendf": nominal_pendf}
