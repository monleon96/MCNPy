"""Prompt fission neutron spectrum (MF5/MT18) perturbation pipeline.

Sister module to ``nubar_perturbation.py`` (MF31 → MF1 nu-bar),
``endf_perturbation.py`` (MF34 → MF4) and ``pendf_perturbation.py`` (MF33 →
PENDF MF3), and deliberately close enough to the first of those that the diff
between them reads.

Pipeline::

    ENDF → MF35 covariance (GNDS CovarianceSuite, one section per band)
         → draw_samples (linear, absolute, float64)
         → per sample: perturb the MF5/MT18 LF=1 tables
         → perturbed ENDF → NJOY → ACE + xsdir

Public entry: :func:`perturb_pfns_files`.

**Prompt only, and that is forced rather than chosen.** MF35 exists for MT18
and nothing else in every library checked; there is no MF35/MT455 anywhere, so
a delayed-spectrum perturbation has nothing to sample from. MF5/MT455 is parsed
and written back untouched.

**The perturbed tape is internally inconsistent, and says so.** Nothing here
perturbs MF35 itself, so the output carries the original covariance beside a
spectrum that has moved. That is the same choice every other pipeline in this
package makes, and like them it is recorded — ``"mf35_unchanged": True`` in the
run summary and in the parquet metadata. A user has to be told, not left to
notice.
"""
from __future__ import annotations

import copy
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from kika.endf.read_endf import read_endf
from kika.endf.writers.endf_writer import ENDFWriter, update_mf1_directory
from kika.sampling.endf_perturbation import _process_njoy_for_sample
from kika.sampling.mf35_sampling import (
    SIGMA_CLAMP,
    band_grids,
    build_pfns_covariance,
    generate_pfns_samples,
    normalisation_drift,
    normalisation_residual,
    perturb_pfns_partial,
    realised_covariance_report,
    row_sum_residual,
)
from kika.sampling.utils import (
    DualLogger,
    _finalize_master_perturbation_matrix,
    _get_logger,
    _initialize_master_perturbation_matrix,
    _merge_isotope_metadata,
    _set_logger,
    _write_isotope_parquet,
)

__all__ = ["perturb_pfns_files"]

# ============================================================================
# PIPELINE CONFIGURATION CONSTANTS
# ============================================================================

# --- Output ----------------------------------------------------------------
GENERATE_ACE: bool = True

# --- Sampling --------------------------------------------------------------
#: **Linear, not log — and this is not an oversight to be tidied up.**
#: ``nubar_perturbation`` uses ``"log"`` because a nu-bar factor is
#: multiplicative and must stay positive. MF35 is the covariance of quantities
#: that sum to one, so ``C·1 ≈ 0`` and a *linear* draw satisfies ``1ᵀδ ≈ 0``
#: automatically — which is exactly what preserves the spectrum's
#: normalisation. A log draw destroys that property outright. Someone will
#: eventually try to make this match nu-bar; this comment is for them.
SAMPLING_SPACE: str = "linear"

DECOMPOSITION_METHOD: str = "svd"       # cholesky is refused; these are rank deficient
SAMPLING_METHOD: str = "sobol"
PSD_METHOD: str = "auto"                # -> clip on this data; higham is far too slow
PROJECTION_METRIC: str = "probability"
NULL_TOL: float = 1e-10

#: See :func:`kika.sampling.mf35_sampling.generate_pfns_samples`. Removes
#: PSD-repair debris from groups holding ~1e-14 of the spectrum.
SIGMA_CLAMP_DEFAULT: Optional[float] = SIGMA_CLAMP

#: Cap on the outgoing points of one MF5 sub-table, or None for no cap.
#:
#: **Off, and now measured rather than assumed.** The worry was that NJOY's
#: ACER reads MF5 into fixed-size buffers while this pipeline grows the MT18
#: section by ~60 %. Measured 2026-08-10: NJOY 2016 processes a perturbed
#: ENDF/B-VIII.1 Cf-252 tape (11 668 → ~18 300 outgoing points) to ACE without
#: complaint, and ``test_njoy_regenerates_ace_from_a_perturbed_pfns_tape``
#: keeps that honest by running the unperturbed tape through first as a
#: baseline.
#:
#: So the cap stays off. Capping drops factor steps, which is a loss of the
#: perturbation itself, and paying that for a problem that does not occur would
#: be the wrong trade. It is implemented and tested for the tape that one day
#: does hit a limit — and it logs what it dropped, because a silent cap reads
#: as "the factor was applied everywhere".
MAX_OUTGOING_POINTS: Optional[int] = None

# --- NJOY ------------------------------------------------------------------
ACE_TEMPERATURES: List[float] = [293.6]
ACE_LIBRARY_NAME: str = "endfb81"
ACE_NJOY_VERSION: str = "NJOY 2016.78"

# --- Runtime ---------------------------------------------------------------
VERBOSE_DIAGNOSTICS: int = 1

#: The MT this pipeline perturbs. MF35 exists for no other.
PFNS_MT: int = 18

_logger: Optional[DualLogger] = None


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------

def _write_perturbed_pfns_endf(endf_file: str, out_endf: str,
                               sections: Dict[int, object]) -> None:
    """Write *out_endf* from *endf_file* with the given MF5 MTs replaced.

    Follows ``_write_perturbed_nubar_endf`` exactly, and for the same reason:
    ``ENDFWriter`` snapshots ``original_lines`` at construction, so a writer
    reused across two replacements would splice the second into the *pre*-first
    text and silently drop the first. One fresh writer per MT,
    ``update_directory=False`` on each, and one directory rebuild at the end
    once the record counts have settled.
    """
    shutil.copyfile(endf_file, out_endf)
    for mt in sorted(sections):
        writer = ENDFWriter(out_endf)
        ok = writer.replace_mt_section(
            sections[mt], mf_number=5,
            output_filepath=out_endf, update_directory=False,
        )
        if not ok:
            raise RuntimeError(f"Failed to replace MF5/MT{mt} in {out_endf}")
    update_mf1_directory(out_endf)


def _parameter_labels(symbol: str, mt: int, grids: Sequence[np.ndarray]) -> List[str]:
    """Column names for the parquet: band-qualified outgoing groups.

    ``_write_isotope_parquet``'s default construction is the outer product of
    the MT list and one shared grid, which cannot express this axis — the bands
    have different orders. Flattening order is band-major, matching
    :func:`_flatten_sample`.
    """
    labels: List[str] = []
    for band, grid in enumerate(grids):
        for group in range(len(grid) - 1):
            labels.append(
                f"{symbol}_MT{mt}_b{band}_{grid[group]:.4e}-{grid[group + 1]:.4e}"
            )
    return labels


def _flatten_sample(samples: Dict[Any, np.ndarray], zaid: int, mt: int,
                    n_bands: int, index: int) -> np.ndarray:
    """One sample's deltas, band-major, as a single row."""
    return np.concatenate([
        samples[(zaid, mt, band)][index] for band in range(n_bands)
    ])


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def perturb_pfns_files(
    endf_files: Union[str, List[str]],
    num_samples: int,
    *,
    mt: int = PFNS_MT,
    generate_ace: bool = GENERATE_ACE,
    decomposition_method: str = DECOMPOSITION_METHOD,
    sampling_method: str = SAMPLING_METHOD,
    psd_method: str = PSD_METHOD,
    projection_metric: str = PROJECTION_METRIC,
    sigma_clamp: Optional[float] = SIGMA_CLAMP_DEFAULT,
    max_outgoing_points: Optional[int] = MAX_OUTGOING_POINTS,
    covariance_override=None,
    njoy_exe: Optional[str] = None,
    ace_temperatures: Optional[Sequence[float]] = None,
    ace_extensions: Optional[Sequence[str]] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = ACE_NJOY_VERSION,
    xsdir_file: Optional[str] = None,
    output_dir: str = ".",
    seed: Optional[int] = None,
    dry_run: bool = False,
    verbose_diagnostics: int = VERBOSE_DIAGNOSTICS,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Perturb MF5/MT18 from its MF35 covariance, optionally regenerating ACE.

    Returns a per-ZAID summary: the bands sampled, their ranks and drift
    budgets, the acceptance-gate numbers of the realised ensemble, the
    projection and renormalisation budgets actually spent, and the NJOY
    outcome. ``dry_run=True`` stops after sampling and reports all of that
    without writing a tape.
    """
    global _logger

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = os.path.join(output_dir, f"pfns_perturbation_{timestamp}.log")
        _logger = DualLogger(log_file, mode="w")
    else:
        _logger = DualLogger(log_file, mode="a")
    _set_logger(_logger)

    if isinstance(endf_files, (str, Path)):
        endf_files = [str(endf_files)]
    endf_files = [str(f) for f in endf_files]

    _logger.info("=" * 78)
    _logger.info(f"PFNS (MF5/MT{mt}) perturbation — {num_samples} sample(s)")
    _logger.info(f"  space={SAMPLING_SPACE}  decomposition={decomposition_method}  "
                 f"sampling={sampling_method}  psd={psd_method}")
    _logger.info(f"  projection metric={projection_metric}  sigma_clamp={sigma_clamp}")
    _logger.info("=" * 78)

    matrix_dir = (None if dry_run
                  else _initialize_master_perturbation_matrix(
                      output_dir, timestamp, num_samples))

    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "num_samples": num_samples,
        "mt": mt,
        "sampling_space": SAMPLING_SPACE,
        "dry_run": bool(dry_run),
        # Stated in the summary because a user must be told rather than notice:
        # the perturbed tape carries its parent's MF35 beside a spectrum that
        # has moved, so it is internally inconsistent.
        "mf35_unchanged": True,
        "mf35_unchanged_note": (
            "MF5/MT18 was perturbed; MF35/MT18 was copied through unchanged. "
            "The perturbed tape therefore states an uncertainty that no longer "
            "describes its own central values. This is expected — nothing here "
            "perturbs a covariance — but it means the output must not be fed "
            "back into a covariance-consuming step as if it were an evaluation."
        ),
        "isotopes": {},
        "errors": [],
    }
    fragments: List[Dict] = []

    for endf_file in endf_files:
        try:
            summary, fragment = _process_one_file(
                endf_file, num_samples, mt=mt,
                decomposition_method=decomposition_method,
                sampling_method=sampling_method, psd_method=psd_method,
                projection_metric=projection_metric, sigma_clamp=sigma_clamp,
                max_outgoing_points=max_outgoing_points,
                covariance_override=covariance_override,
                generate_ace=generate_ace, njoy_exe=njoy_exe,
                ace_temperatures=list(ace_temperatures or ACE_TEMPERATURES),
                ace_extensions=list(ace_extensions) if ace_extensions else None,
                ace_library_name=ace_library_name or ACE_LIBRARY_NAME,
                ace_njoy_version=ace_njoy_version, xsdir_file=xsdir_file,
                output_dir=output_dir, seed=seed, dry_run=dry_run,
                matrix_dir=matrix_dir, verbose_diagnostics=verbose_diagnostics,
            )
            results["isotopes"][str(summary["zaid"])] = summary
            if fragment is not None:
                fragments.append(fragment)
        except Exception as exc:                      # one bad tape, not the run
            _logger.error(f"[PFNS] {endf_file}: {exc}")
            results["errors"].append({"file": endf_file, "error": str(exc)})

    if matrix_dir is not None:
        _merge_isotope_metadata(matrix_dir, fragments)
        _finalize_master_perturbation_matrix(matrix_dir)
        results["matrix_dir"] = matrix_dir

    _logger.info("=" * 78)
    _logger.info(f"PFNS perturbation finished: "
                 f"{len(results['isotopes'])} isotope(s), "
                 f"{len(results['errors'])} error(s)")
    _logger.info("=" * 78)
    return results


def _process_one_file(
    endf_file: str, num_samples: int, *, mt: int,
    decomposition_method: str, sampling_method: str, psd_method: str,
    projection_metric: str, sigma_clamp: Optional[float],
    max_outgoing_points: Optional[int], covariance_override,
    generate_ace: bool, njoy_exe: Optional[str],
    ace_temperatures: List[float], ace_extensions: Optional[List[str]],
    ace_library_name: str, ace_njoy_version: str, xsdir_file: Optional[str],
    output_dir: str, seed: Optional[int], dry_run: bool,
    matrix_dir: Optional[str], verbose_diagnostics: int,
):
    logger = _get_logger()
    started = time.time()

    # Only the three files this pipeline touches. On the 36 MB
    # ENDF/B-VIII.1 U-235 tape a full parse costs minutes for sections
    # nothing here reads.
    endf = read_endf(endf_file, mf_numbers=[1, 5, 35])
    suite, section, bands = build_pfns_covariance(
        endf, mt=mt, covariance_override=covariance_override, logger=logger,
    )
    grids = band_grids(suite)
    zaid = int(round(float(section._za)))
    symbol = _symbol_of(zaid)

    tabulated = section.tabulated_partials()
    if not tabulated:
        raise ValueError(
            f"MF5/MT{mt} of {endf_file} has no LF=1 subsection, so there is "
            f"nothing this pipeline knows how to perturb. Laws present: "
            f"{[p.lf for p in section.partials]}"
        )
    if len(tabulated) > 1:
        logger.warning(
            f"  [PFNS] MF5/MT{mt} has {len(tabulated)} tabulated partials; "
            f"perturbing index {tabulated[0][0]} only and passing the rest "
            f"through untouched"
        )
    partial_index, _ = tabulated[0]

    # The input's own sum-rule residual, measured before anything moves. This
    # is how you find out the file was already off instead of blaming the code.
    residual = normalisation_residual(section, partial_index)
    logger.info(
        f"  [PFNS] {symbol} (ZAID {zaid}): input |∫χ − 1| ≤ "
        f"{residual['max_abs']:.3e} over {residual['n_nodes']} incident node(s), "
        f"worst at {residual['argmax_energy']:.4e} eV"
    )
    if residual["max_abs"] > 1e-6:
        logger.warning(
            f"  [PFNS] {symbol}: the input spectrum is normalised only to "
            f"{residual['max_abs']:.3e}; every budget below is measured against "
            f"that, not against 1"
        )

    band_report = []
    for index, cov_section in enumerate(suite):
        matrix = np.asarray(cov_section.form.matrix)
        band_report.append({
            "band": index,
            "incident_range": [float(bands[index][0]), float(bands[index][1])],
            "n_groups": int(matrix.shape[0]),
            "row_sum_residual": row_sum_residual(matrix),
            "normalisation_drift": normalisation_drift(matrix),
        })
        logger.info(
            f"  [PFNS] band {index}: E = [{bands[index][0]:.3e}, "
            f"{bands[index][1]:.3e}] eV, {matrix.shape[0]} groups, "
            f"|ΣC|/max|C| = {band_report[-1]['row_sum_residual']:.2e}, "
            f"sqrt(1ᵀC1) = {band_report[-1]['normalisation_drift']:.2e}"
        )

    samples, sample_diagnostics = generate_pfns_samples(
        suite, num_samples, isotope=zaid, mt=mt,
        decomposition_method=decomposition_method,
        sampling_method=sampling_method, seed=seed,
        psd_method=psd_method, null_tol=NULL_TOL, sigma_clamp=sigma_clamp,
        verbose=verbose_diagnostics >= 2, logger=logger,
    )

    for index, entry in enumerate(band_report):
        info = sample_diagnostics[(zaid, mt, index)]
        entry.update({
            "rank": info["rank"],
            "n_null": info["n_null"],
            "null_fraction": info["null_fraction"],
            "seed": info["seed"],
            "n_clamped": info["n_clamped"],
            "clamped_fraction": info["clamped_fraction"],
            "realised_covariance_error": info["realised_covariance_error"],
        })
        gate = realised_covariance_report(
            samples[(zaid, mt, index)],
            np.asarray(list(suite)[index].form.matrix), null_tol=NULL_TOL,
        )
        entry["acceptance_gate"] = gate
        logger.info(
            f"  [PFNS] band {index}: rank {info['rank']}/{entry['n_groups']}, "
            f"spectral fidelity {gate['spectral_fidelity_median']:.4f} "
            f"(tol {gate['gate_tolerance']:.4f}, "
            f"{'PASS' if gate['passes_spectral_gate'] else 'FAIL'}), "
            f"null leakage {gate['null_leakage']:.2e}, "
            f"{info['n_clamped']} component(s) clamped"
        )

    summary: Dict[str, Any] = {
        "zaid": zaid,
        "symbol": symbol,
        "file": endf_file,
        "mt": mt,
        "partial_index": partial_index,
        "n_bands": len(bands),
        "bands": band_report,
        "input_normalisation_residual": {
            "max_abs": residual["max_abs"],
            "argmax_energy": residual["argmax_energy"],
            "n_nodes": residual["n_nodes"],
        },
        "covariance_source": {"origin": "tape", "path": os.path.abspath(endf_file)},
        "samples_written": 0,
        "njoy_success": 0,
        "sample_errors": [],
    }

    if dry_run:
        summary["elapsed_s"] = time.time() - started
        logger.info(f"  [PFNS] dry run: no tape written")
        return summary, None

    fragment = None
    if matrix_dir is not None:
        flattened = np.vstack([
            _flatten_sample(samples, zaid, mt, len(bands), index)
            for index in range(num_samples)
        ])
        fragment = _write_isotope_parquet(
            matrix_dir, zaid, flattened, [mt], list(grids[0]),
            verbose=verbose_diagnostics >= 1, logger=logger,
            param_labels=_parameter_labels(symbol, mt, grids),
            extra_metadata={
                "pipeline": "pfns",
                "mf35_unchanged": True,
                "sampling_space": SAMPLING_SPACE,
                "quantity": "absolute group-probability deltas",
                "bands": [b["incident_range"] for b in band_report],
            },
        )

    applied: List[Dict[str, Any]] = []
    for index in range(num_samples):
        try:
            out_endf = os.path.join(
                output_dir, f"{symbol}_pfns_{index + 1:04d}.endf"
            )
            perturbed = copy.deepcopy(section)
            deltas = {
                band: samples[(zaid, mt, band)][index] for band in range(len(bands))
            }
            _, node_diagnostics = perturb_pfns_partial(
                perturbed.partials[partial_index], deltas, bands, grids,
                metric=projection_metric,
                max_outgoing_points=max_outgoing_points,
                logger=logger if verbose_diagnostics >= 2 else None,
            )
            node_diagnostics.pop("per_node", None)
            applied.append(node_diagnostics)

            _write_perturbed_pfns_endf(endf_file, out_endf, {mt: perturbed})
            summary["samples_written"] += 1

            if generate_ace and njoy_exe:
                njoy_result = _process_njoy_for_sample(
                    out_endf=out_endf, sample_index=index, njoy_exe=njoy_exe,
                    temperatures=ace_temperatures, extensions=ace_extensions,
                    library_name=ace_library_name, njoy_version=ace_njoy_version,
                    output_dir=output_dir, xsdir_file=xsdir_file,
                )
                if njoy_result.get("success"):
                    summary["njoy_success"] += 1
                else:
                    summary["sample_errors"].append(
                        {"sample": index, "njoy": njoy_result.get("errors")}
                    )
        except Exception as exc:            # counted, not raised — as nu-bar does
            logger.error(f"  [PFNS] sample {index + 1}: {exc}")
            summary["sample_errors"].append({"sample": index, "error": str(exc)})

    summary["applied"] = _summarise_applied(applied)
    summary["elapsed_s"] = time.time() - started

    budgets = summary["applied"]
    logger.info(
        f"  [PFNS] {symbol}: {summary['samples_written']}/{num_samples} tape(s) "
        f"written in {summary['elapsed_s']:.1f} s; "
        f"max |t| = {budgets.get('max_projection_shift', 0.0):.2e}, "
        f"max |1 − N/N'| = {budgets.get('max_renormalisation_error', 0.0):.2e}, "
        f"max clipped mass = {budgets.get('max_clipped_mass_fraction', 0.0):.2e}, "
        f"max group mass error = {budgets.get('max_group_mass_error', 0.0):.2e}"
    )
    if budgets.get("total_steps_dropped"):
        logger.warning(
            f"  [PFNS] {symbol}: {budgets['total_steps_dropped']} factor step(s) "
            f"were dropped to respect max_outgoing_points — the written "
            f"perturbation is not the one that was sampled in those groups"
        )

    return summary, fragment


def _summarise_applied(applied: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Worst case over samples of every budget the perturbation spent.

    Two renormalisations happen per incident node — the constraint solve and
    the global scalar — and both are reported. Neither meaningfully perturbs
    the realised covariance, but a silent projection is a distribution you no
    longer know the shape of.
    """
    if not applied:
        return {}

    def worst(field: str) -> float:
        return float(max(abs(entry.get(field, 0.0)) for entry in applied))

    def total(field: str) -> int:
        return int(sum(entry.get(field, 0) for entry in applied))

    return {
        "n_samples": len(applied),
        "max_projection_shift": worst("max_projection_shift"),
        "max_sum_error_after_projection": worst("max_sum_error_after_projection"),
        "max_renormalisation_error": worst("max_renormalisation_error"),
        "max_group_mass_error": worst("max_group_mass_error"),
        "max_group_ratio_error": worst("max_group_ratio_error"),
        "max_clipped_mass_fraction": worst("max_clipped_mass_fraction"),
        "max_frac_mass_outside_mf35": worst("max_frac_mass_outside_mf35"),
        "total_clipped": total("total_clipped"),
        "total_groups_frozen": total("total_groups_frozen"),
        "total_outgoing_inserted": total("total_outgoing_inserted"),
        "total_steps_dropped": total("total_steps_dropped"),
        "n_incident_inserted": applied[0].get("n_incident_inserted", 0),
    }


def _symbol_of(zaid: int) -> str:
    from kika.sampling.utils import zaid_to_symbol

    try:
        return zaid_to_symbol(zaid)
    except Exception:                        # a ZAID no table knows is not fatal
        return f"ZA{zaid}"
