"""
MF33-native-grid covariance assembly and per-PENDF MF3 factor application.

Sister module to ``kika.sampling.{ace_perturbation, endf_perturbation}``:
bundles the helpers ``pendf_perturbation.py`` orchestrates. Kept separate so
the ENDF-parsing / NC-LTY=0 resolution / MF3-rewrite logic is unit-testable
without dragging in NJOY or multiprocessing.

Public API
----------
- ``load_mf33_covariance``     — assemble unified covariance on an MF33-derived
                                 union grid (relative form), plus the PENDF MF3
                                 sections and the union grid itself.
- ``apply_factors_to_pendf_mf3`` — write a perturbed PENDF using piecewise-
                                   constant factor application.
- ``extract_mt_param_blocks``    — MT → slice into the flat factors vector.
"""
from __future__ import annotations

import dataclasses
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kika._constants import MT_GROUPS
from kika.endf import read_endf
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.endf.classes.mf import MF
from kika.endf.classes.mf3.mf3mt import MF3MT
from kika.endf.writers.endf_writer import ENDFWriter
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.processing.multigroup import compute_rebin_operator, collapse_covariance
from kika.processing.njoy_pendf_cache import (
    mf33_needs_pendf,
    read_pendf_mf3_sections,
)


# Composite MT → set of partial MTs (per ENDF-6 Appendix B).
# Sourced from kika._constants.MT_GROUPS so the table stays in one place.
_COMPOSITE_PARTIALS: Dict[int, frozenset[int]] = {
    int(c): frozenset(int(p) for p in rng) for c, rng in MT_GROUPS
}


# ---------------------------------------------------------------------------
# Covariance assembly
# ---------------------------------------------------------------------------

def load_mf33_covariance(
    endf_path: str,
    pendf_path: Optional[str],
    mt_list: Sequence[int],
    *,
    energy_unit: str = "eV",
    logger=None,
) -> Tuple[CrossSectionCovariance, Dict[int, MF3MT], List[float], List[int]]:
    """Build a CrossSectionCovariance on the MF33 native union grid.

    Steps:
      1. Parse MF33 from the ENDF tape.
      2. Read MF3 from the PENDF (used for NC-LTY0 resolution and absolute→
         relative conversion when LB-types are absolute).
      3. Call ``MF33MT.to_xs_covmat`` for each requested MT — yields per-block
         matrices on individual native grids, with each block flagged
         ``is_relative``.
      4. Build a union grid across all blocks for the requested MTs.
      5. Project every block to the union grid; convert any absolute block
         to relative using bin-averaged σ(E) from PENDF MF3.
      6. Pack into a fresh ``CrossSectionCovariance`` in shared-grid mode
         (``num_groups = len(union)-1``, ``energy_grid = union``).

    Returns
    -------
    cov : CrossSectionCovariance
        Unified covariance, all blocks relative, on the shared union grid.
    mf3_sections : Dict[int, MF3MT]
        Pointwise σ(E) per MT, read from PENDF.
    union_grid : List[float]
        The MF33 native union grid in ``energy_unit``.
    mts_present : List[int]
        Subset of ``mt_list`` for which MF33 data was available (sorted).
    """
    endf_obj = parse_endf_file(str(endf_path))
    mf33_file = endf_obj.get_file(33)
    if mf33_file is None or not getattr(mf33_file, "sections", None):
        raise RuntimeError(f"ENDF {endf_path} has no MF33 sections")

    mf3_sections: Dict[int, MF3MT] = {}
    if pendf_path is not None:
        mf3_sections = read_pendf_mf3_sections(pendf_path)

    sibling_sections = dict(mf33_file.sections)

    requested_mts = sorted({int(mt) for mt in mt_list})
    mts_present = [mt for mt in requested_mts if mt in mf33_file.sections]
    missing = sorted(set(requested_mts) - set(mts_present))
    if missing and logger is not None:
        logger.warning(
            f"[MF33] MTs {missing}: not in MF33; no perturbation applied for these."
        )
    if not mts_present:
        raise RuntimeError(
            f"None of MTs {requested_mts} are present in MF33 of {endf_path}"
        )

    per_mt_covs: Dict[int, CrossSectionCovariance] = {}
    for mt in mts_present:
        mf33mt = mf33_file.sections[mt]
        per_mt_covs[mt] = mf33mt.to_xs_covmat(
            energy_unit=energy_unit,
            sibling_sections=sibling_sections,
            mf3_sections=mf3_sections if mf3_sections else None,
        )

    union_grid = _build_union_grid(per_mt_covs, mts_present)
    if len(union_grid) < 2:
        raise RuntimeError(
            f"MF33 union grid has fewer than 2 boundaries; cannot build covariance"
        )

    # Bin-averaged σ(E) per MT on the union grid (for absolute→relative).
    bin_xs: Dict[int, np.ndarray] = {}
    if mf3_sections:
        for mt in mts_present:
            if mt in mf3_sections:
                bin_xs[mt] = _bin_average_xs(
                    np.asarray(mf3_sections[mt].energies, dtype=float),
                    np.asarray(mf3_sections[mt].cross_sections, dtype=float),
                    union_grid,
                )

    unified = CrossSectionCovariance(
        num_groups=len(union_grid) - 1,
        energy_grid=list(union_grid),
        energy_unit=energy_unit,
    )

    for mt_row in mts_present:
        cov_src = per_mt_covs[mt_row]
        for i in range(len(cov_src.matrices)):
            mt_col = int(cov_src.reaction_cols[i])
            if mt_col not in mts_present:
                continue
            iso_row = int(cov_src.isotope_rows[i])
            iso_col = int(cov_src.isotope_cols[i])
            native_grid = (
                cov_src.energy_grids[i]
                if i < len(cov_src.energy_grids) and cov_src.energy_grids[i]
                else None
            )
            if native_grid is None or len(native_grid) < 2:
                continue
            mat = np.asarray(cov_src.matrices[i], dtype=float)
            is_rel = (
                bool(cov_src.is_relative[i])
                if i < len(cov_src.is_relative)
                else False
            )

            mat_proj = _project_matrix(mat, list(native_grid), union_grid)
            if not is_rel:
                if mt_row not in bin_xs or mt_col not in bin_xs:
                    if logger is not None:
                        logger.warning(
                            f"[MF33] Cannot convert absolute block "
                            f"({mt_row}↔{mt_col}) to relative — PENDF MF3 missing; "
                            f"skipping block."
                        )
                    continue
                mat_proj = _absolute_to_relative(
                    mat_proj, bin_xs[mt_row], bin_xs[mt_col]
                )

            unified.add_matrix(
                iso_row, mt_row, iso_col, mt_col,
                mat_proj,
                energy_grid=list(union_grid),
                is_relative=True,
            )

    return unified, mf3_sections, list(union_grid), mts_present


# ---------------------------------------------------------------------------
# The model-side source (M1)
# ---------------------------------------------------------------------------

def relativiseAbsoluteSections(suite, binCentral, unionGrid, *, mf: int = 33,
                               logger=None):
    """Restate every absolute MF33/MF31 section of *suite* as a relative one.

    **Why this is not part of the assembly.** ``cross_section_covariance_blocks``
    lifts each section onto the union grid and puts it in its place; what it
    does not do — and should not — is divide by a central value, because the
    central value is not in the covariance file. MF3 lives on the PENDF and the
    nu-bar on MF1, so the conversion needs an input the covariance suite does
    not have and can only be done here, by the caller that read both.

    **The order matters and is the carrier's.** ``load_mf33_covariance``
    projects the block onto the union grid *first* and divides *afterwards*, by
    a σ̄ binned on that same union. Dividing on the native grid and projecting
    the result is a different matrix — projection is a sum over sub-bins and
    division is not linear in the divisor — so this function lifts first too,
    through ``model_blocks._lift_matrix``, which is bit-identical to
    ``_project_matrix`` on the grids these files use (asserted in
    ``test_the_model_source_reproduces_the_carrier.py``).

    ``binCentral`` maps MT → the bin-averaged central values on *unionGrid*, as
    :func:`_bin_average_xs` produces them. A section whose row or column MT has
    no central value **is dropped, with a warning**, which is what
    ``load_mf33_covariance`` does with the same case: a relative covariance is
    undefined where there is no cross section to be relative to.

    Returns a new list of sections; *suite* is not modified.
    """
    from kika.sampling.model_blocks import _endf_mt, _is_endf_mf, _lift_matrix
    from kika._covariance_forms import require_single_matrix

    union = np.asarray(unionGrid, dtype=float)
    kept = []
    for section in getattr(suite, "covarianceSections", suite):
        rowData = section.rowData
        if not _is_endf_mf(rowData, mf):
            kept.append(section)
            continue
        colData = section.columnData if section.columnData is not None else rowData
        form = require_single_matrix(
            getattr(section, "form", None),
            f"MF{mf} covariance section {getattr(section, 'label', '?')!r}",
        )
        if form.isRelative:
            kept.append(section)
            continue

        mtRow, mtCol = _endf_mt(rowData), _endf_mt(colData)
        if mtRow not in binCentral or mtCol not in binCentral:
            if logger is not None:
                logger.warning(
                    f"[MF{mf}] Cannot convert absolute block ({mtRow}<->{mtCol}) "
                    f"to relative -- no central values; skipping block."
                )
            continue

        rowGrid = np.asarray(form.rowGrid, dtype=float)
        colGrid = (rowGrid if form.columnGrid is None
                   else np.asarray(form.columnGrid, dtype=float))
        lifted = (_lift_matrix(rowGrid, union)
                  @ np.asarray(form.matrix, dtype=float)
                  @ _lift_matrix(colGrid, union).T)
        relative = _absolute_to_relative(
            lifted, binCentral[mtRow], binCentral[mtCol])

        kept.append(dataclasses.replace(
            section,
            form=dataclasses.replace(form, matrix=relative, isRelative=True,
                             rowGrid=union.copy(), columnGrid=union.copy()),
        ))
    return kept


def loadCrossSectionBlocks(endfObj, mtList, centralSections, *, mf: int = 33,
                           isotope=None, union: str = "global", logger=None):
    """The MF33 (or MF31) joint covariance, assembled **from the model**.

    The model-side replacement for :func:`load_mf33_covariance` and
    :func:`kika.sampling.mf31_sampling.build_mf31_covariance` as a *source*: it
    returns the same joint matrix in the same layout, as the ``(key, matrix)``
    blocks and the index that :func:`kika.sampling.multigroup_draw.draw_relative_factors`
    takes, without a :class:`~kika.cov.cross_section_covariance.CrossSectionCovariance`
    ever being built.

    ``union="global"`` is the shipped layout and the default, so this moves no
    number — ``docs/library/gnds_endf_conflicts.md`` §10.1 measured it
    bit-identical on the full Fe-56 tape, 5110 x 5110. ``"per-component"`` is
    the improvement and keeps its own before/after.

    ``centralSections`` maps MT → a section carrying ``energies`` and
    ``cross_sections``: PENDF MF3 for MF33, MF1 nu-bar for MF31. It is used for
    two things and only two — the absolute→relative conversion above, and
    nothing else — so passing ``{}`` is legitimate for a file whose sections are
    all relative, and produces a warning for one whose sections are not.

    Returns ``(blocks, index, unionGrid, mtsPresent)``.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite
    from kika.sampling.model_blocks import (_cross_section_entries,
                                            _union_grids,
                                            cross_section_covariance_blocks,
                                            cross_section_covariance_index)

    mfFile = endfObj.get_file(mf)
    if mfFile is None or not getattr(mfFile, "sections", None):
        raise RuntimeError(f"this ENDF object has no MF{mf} sections")

    if mtList is None:
        requested = sorted(int(mt) for mt in mfFile.sections)
    else:
        requested = sorted({int(mt) for mt in mtList})
    mtsPresent = [mt for mt in requested if mt in mfFile.sections]
    missing = sorted(set(requested) - set(mtsPresent))
    if missing and logger is not None:
        logger.warning(
            f"[MF{mf}] MTs {missing}: not in MF{mf}; no perturbation applied "
            f"for these."
        )
    if not mtsPresent:
        raise RuntimeError(f"None of MTs {requested} are present in MF{mf}")

    suite, report = decodeCovarianceSuite(endfObj)
    if logger is not None and not report.isClean:
        logger.info(f"[MF{mf}] covariance decode: {report.summary()}")

    # The union grid, before any conversion: the conversion has to divide by a
    # central value binned on the grid the blocks will end up on, so the grid
    # is settled first. Taken from the entries rather than rebuilt, so it is
    # `assemble_joint`'s own and not a second implementation of it.
    entries = _cross_section_entries(suite, mf=mf, mt=mtsPresent)
    if not entries:
        raise RuntimeError(
            f"MF{mf} decoded to no covariance entries for MTs {mtsPresent}; the "
            f"sections are present on the tape but did not reach the model"
        )
    unions = _union_grids(entries, union=union)

    absolute = _absoluteSections(suite, mf)
    if absolute and union != "global":
        # Under `per-component` each key has its own bins, so there is no single
        # grid to state a converted block on -- and stating it on one key's grid
        # while the assembly expects another is wrong everywhere and visible
        # nowhere. The improvement has to decide what a converted absolute block
        # means before it can be taken, so this refuses rather than guesses.
        raise NotImplementedError(
            f"MF{mf} has {len(absolute)} absolute block(s) {absolute} and "
            f"union={union!r}: the absolute-to-relative conversion divides by a "
            f"central value binned on the grid the block ends up on, and "
            f"per-component gives each component a different one. Use "
            f"union='global' (the shipped layout), or decide what a converted "
            f"block means per component first"
        )

    # One grid in `global` mode by construction -- every key gets the same
    # union -- which is what makes a single `binCentral` well defined.
    unionGrid = [float(e) for e in unions[sorted(unions)[0]]]

    binCentral = {}
    for mt in mtsPresent:
        section = (centralSections or {}).get(mt)
        if section is not None:
            binCentral[mt] = _bin_average_xs(
                np.asarray(section.energies, dtype=float),
                np.asarray(section.cross_sections, dtype=float),
                unionGrid,
            )

    sections = relativiseAbsoluteSections(
        suite, binCentral, unionGrid, mf=mf, logger=logger)

    # Both derived from the same converted sections, and through the shipped
    # entry points rather than a second implementation of them: an index built
    # on one bin structure describing a block built on another is wrong
    # everywhere without being wrong anywhere visible.
    blocks = cross_section_covariance_blocks(
        sections, isotope=isotope, mt=mtsPresent, union=union, mf=mf)
    if not blocks:
        raise RuntimeError(
            f"MF{mf}: every block was dropped by the absolute-to-relative "
            f"conversion, so there is nothing to sample"
        )
    index = cross_section_covariance_index(
        sections, isotope=isotope, mt=mtsPresent, union=union, mf=mf)
    return blocks, index, unionGrid, mtsPresent


def _absoluteSections(suite, mf: int):
    """The ``(row MT, column MT)`` of every MF33/MF31 section stated absolute."""
    from kika._covariance_forms import require_single_matrix
    from kika.sampling.model_blocks import _endf_mt, _is_endf_mf

    found = []
    for section in getattr(suite, "covarianceSections", suite):
        if not _is_endf_mf(section.rowData, mf):
            continue
        try:
            form = require_single_matrix(getattr(section, "form", None), "")
        except Exception:
            continue
        if not form.isRelative:
            col = (section.columnData if section.columnData is not None
                   else section.rowData)
            found.append((_endf_mt(section.rowData), _endf_mt(col)))
    return found


def load_mf33_blocks(
    endf_path: str,
    pendf_path: Optional[str],
    mt_list: Sequence[int],
    *,
    energy_unit: str = "eV",
    isotope=None,
    union: str = "global",
    logger=None,
):
    """:func:`load_mf33_covariance`'s signature, the model as its source.

    Reads the tape once, reads the PENDF for its MF3 -- which the caller needs
    anyway to apply the factors, and which is also what the absolute-to-relative
    conversion divides by -- and returns
    ``(blocks, index, mf3_sections, unionGrid, mtsPresent)``.

    ``energy_unit`` is accepted and must be ``'eV'``: the model states its grids
    in eV and does not carry a second unit to convert from, where
    ``MF33MT.to_xs_covmat`` does. Anything else is refused rather than ignored.
    """
    if energy_unit != "eV":
        raise ValueError(
            f"the model source states energy in eV; energy_unit={energy_unit!r} "
            f"would have to be converted and nothing here knows to"
        )

    endf_obj = parse_endf_file(str(endf_path))
    mf3_sections: Dict[int, MF3MT] = {}
    if pendf_path is not None:
        mf3_sections = read_pendf_mf3_sections(pendf_path)

    blocks, index, unionGrid, mtsPresent = loadCrossSectionBlocks(
        endf_obj, mt_list, mf3_sections, mf=33, isotope=isotope,
        union=union, logger=logger,
    )
    return blocks, index, mf3_sections, unionGrid, mtsPresent


def extractParamBlocks(index) -> Dict[int, slice]:
    """MT → slice into the flat factors vector, from a model-side index.

    The model-side twin of :func:`extract_mt_param_blocks`, which reads the same
    thing off the carrier. Both exist for as long as both sources do.
    """
    (meta,) = index.values()
    stride = int(meta["stride"])
    return {int(mt): slice(i * stride, (i + 1) * stride)
            for i, (_za, mt) in enumerate(meta["pairs"])}


def extract_mt_param_blocks(
    cov: CrossSectionCovariance,
) -> Dict[int, slice]:
    """Return MT → slice into the flat (n_mts × num_groups) factors vector.

    Order matches ``cov._get_param_pairs()`` which is the same order
    ``generate_samples`` uses internally.
    """
    param_pairs = cov._get_param_pairs()
    g = cov.num_groups
    return {int(mt): slice(i * g, (i + 1) * g) for i, (_zaid, mt) in enumerate(param_pairs)}


# ---------------------------------------------------------------------------
# PENDF MF3 rewrite
# ---------------------------------------------------------------------------

def apply_factors_to_pendf_mf3(
    pendf_path: str,
    out_pendf_path: str,
    mf3_sections: Dict[int, MF3MT],
    factors_one_sample: np.ndarray,
    mt_to_param_block: Dict[int, slice],
    bins_native: Sequence[float],
    *,
    logger=None,
) -> Dict[int, Dict[str, float]]:
    """Apply piecewise-constant per-MF33-bin factors to PENDF MF3 σ(E).

    For each MT in ``mt_to_param_block``:

        block = factors_one_sample[mt_to_param_block[mt]]    # (n_groups,)
        For each pointwise PENDF energy E_k:
            g = searchsorted(bins_native, E_k, side='right') - 1
            f = block[g]   if 0 <= g < n_groups   else 1.0
            σ_new[k] = σ_orig[k] * f

    Composite-MT expansion: when a perturbed MT is a composite key in
    :data:`kika._constants.MT_GROUPS` (e.g. MT4 = total inelastic, MT103 =
    (n,p)) and the PENDF lacks that MT, the same factor block is applied to
    every partial in the composite's range that PENDF carries. Partials that
    are *also* perturbed individually are excluded from the expansion to
    avoid double-counting.

    Then writes the modified PENDF to ``out_pendf_path`` by sequential
    ``ENDFWriter.replace_mt_section`` calls (one per target MT). The PENDF
    MF1/MT451 directory is left alone (NJOY MODER reads tapes via tape units).

    Per-target-MT step handling: before applying factors, the pointwise table
    for each target MT is augmented with duplicate-energy points at every
    interior MF33 bin edge that falls inside the PENDF range and isn't
    already a duplicate. Factors are then applied with ENDF-6 step semantics
    so the FIRST of each duplicate pair gets the lower-bin factor and the
    SECOND gets the upper-bin factor — mirroring the boundary-discontinuity
    construction used for MF4 Legendre coefficients in
    ``kika.sampling.endf_perturbation``.

    Returns a per-target-MT diagnostic dict ({mt: {'min_factor', 'max_factor',
    'frac_out_of_coverage', 'n_inserted', 'source_mt'}}) for caller logging.
    ``source_mt`` is the MT whose factor block was applied (== mt for direct,
    == composite for expanded partials).
    """
    bins = np.asarray(bins_native, dtype=float)
    n_groups = bins.size - 1
    diagnostics: Dict[int, Dict[str, float]] = {}

    shutil.copyfile(pendf_path, out_pendf_path)

    perturbed_keys = set(int(k) for k in mt_to_param_block.keys())

    for mt, sl in mt_to_param_block.items():
        block = np.asarray(factors_one_sample[sl], dtype=float)
        if block.size != n_groups:
            raise ValueError(
                f"factor block for MT={mt} has size {block.size}, "
                f"expected n_groups={n_groups}"
            )

        if mt in mf3_sections:
            targets = [(int(mt), mf3_sections[mt])]
            source_label = int(mt)
        elif mt in _COMPOSITE_PARTIALS:
            partials_in_pendf = sorted(
                p for p in _COMPOSITE_PARTIALS[mt]
                if p in mf3_sections and p not in perturbed_keys
            )
            if not partials_in_pendf:
                if logger is not None:
                    logger.warning(
                        f"[PENDF] MT={mt} (composite): no partials available in "
                        f"PENDF MF3 (or all are individually perturbed); skipping."
                    )
                continue
            if logger is not None:
                logger.info(
                    f"[PENDF] MT={mt} (composite): applying factor to PENDF "
                    f"partials {partials_in_pendf}"
                )
            targets = [(p, mf3_sections[p]) for p in partials_in_pendf]
            source_label = int(mt)
        else:
            if logger is not None:
                logger.warning(
                    f"[PENDF] MT={mt}: not in PENDF MF3 and not a known "
                    f"composite (MT_GROUPS keys = {sorted(_COMPOSITE_PARTIALS)}); "
                    f"skipping."
                )
            continue

        for tgt_mt, orig in targets:
            energies_orig = np.asarray(orig.energies, dtype=float)
            xs_orig = np.asarray(orig.cross_sections, dtype=float)
            interp_orig = list(orig.energy_interpolation or [])

            energies_aug, xs_aug, interp_aug = _augment_with_step_duplicates(
                energies_orig, xs_orig, interp_orig, bins,
            )
            n_inserted = int(energies_aug.size - energies_orig.size)

            xs_new, factors, frac_out = perturb_pointwise_xs(
                energies_aug, xs_aug, block, bins,
            )

            diagnostics[tgt_mt] = {
                "min_factor": float(factors.min()),
                "max_factor": float(factors.max()),
                "frac_out_of_coverage": frac_out,
                "n_inserted": n_inserted,
                "source_mt": source_label,
            }

            new_mf3mt = dataclasses.replace(
                orig,
                _energies=list(energies_aug),
                _cross_sections=list(xs_new),
                _interpolation=interp_aug,
                _np=len(energies_aug),
                _nr=len(interp_aug),
            )

            writer = ENDFWriter(out_pendf_path)
            ok = writer.replace_mt_section(
                new_mf3mt,
                mf_number=3,
                output_filepath=out_pendf_path,
                update_directory=False,
            )
            if not ok:
                raise RuntimeError(
                    f"Failed to replace MF3/MT{tgt_mt} in {out_pendf_path}"
                )

    return diagnostics


# ---------------------------------------------------------------------------
# Pure factor-application math (testable without ENDFWriter)
# ---------------------------------------------------------------------------

def perturb_pointwise_xs(
    energies: np.ndarray,
    xs_orig: np.ndarray,
    block: np.ndarray,
    bins_native: np.ndarray,
    *,
    atol: float = 1e-10,
    clamp_top_edge: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply a piecewise-constant per-bin factor block to pointwise σ(E),
    respecting ENDF-6 duplicate-energy step semantics.

    Default rule (non-duplicate points and the SECOND of a duplicate pair):
        g = searchsorted(bins_native, E_k, side='right') - 1   # bin starting at E_k
        f = block[g]   if 0 <= g < n_groups   else 1.0

    Step rule (FIRST of a consecutive duplicate-energy pair, i.e. when
    ``energies[k] == energies[k+1]``):
        g = searchsorted(bins_native, E_k, side='left') - 1    # bin ending at E_k
        f = block[g]   if 0 <= g < n_groups   else 1.0

    Same as the legacy ``side='right'`` rule for arrays without duplicates,
    so the bin assignment for non-duplicate energies is unchanged.

    ``clamp_top_edge`` (opt-in, default off) assigns the LAST bin's factor to a
    point sitting exactly on ``bins_native[-1]`` instead of leaving it outside
    coverage at factor 1.0. Under the default ``side='right'`` rule that point
    lands in the (non-existent) bin starting at the top edge, so it is never
    perturbed — harmless for a PENDF table with thousands of points, but for an
    MF1 nu-bar table whose last point *is* the top of the MF31 grid it turns the
    last bin into a ramp back to the unperturbed value (measured: realised σ
    0.50× the MF31 σ in the last bin of JEFF-4.0 Pu-241 MT456). Kept opt-in so
    the MF33 → PENDF path is bit-for-bit unchanged.

    Returns ``(xs_new, factors_per_point, frac_out_of_coverage)``.
    """
    energies = np.asarray(energies, dtype=float)
    xs_orig = np.asarray(xs_orig, dtype=float)
    block = np.asarray(block, dtype=float)
    bins_native = np.asarray(bins_native, dtype=float)
    n_groups = bins_native.size - 1
    if block.size != n_groups:
        raise ValueError(
            f"block has size {block.size} but bins_native implies {n_groups} groups"
        )

    idx_right = np.searchsorted(bins_native, energies, side="right") - 1
    idx_left = np.searchsorted(bins_native, energies, side="left") - 1

    is_first_dup = np.zeros(energies.size, dtype=bool)
    if energies.size >= 2:
        is_first_dup[:-1] = np.isclose(
            energies[:-1], energies[1:], rtol=0.0, atol=atol,
        )

    idx = np.where(is_first_dup, idx_left, idx_right)
    if clamp_top_edge and n_groups > 0 and energies.size:
        at_top = np.isclose(energies, bins_native[-1], rtol=1e-12, atol=0.0)
        idx = np.where(at_top, n_groups - 1, idx)
    in_cov = (idx >= 0) & (idx < n_groups)
    factors = np.ones_like(energies)
    factors[in_cov] = block[idx[in_cov]]
    xs_new = xs_orig * factors
    frac_out = float(1.0 - in_cov.mean()) if energies.size else 0.0
    return xs_new, factors, frac_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_union_grid(
    per_mt_covs: Dict[int, CrossSectionCovariance],
    mts_present: Sequence[int],
) -> List[float]:
    """Union of all native energy grids across blocks involving requested MTs."""
    pts: set[float] = set()
    requested = set(mts_present)
    for mt_row, cov in per_mt_covs.items():
        for i in range(len(cov.matrices)):
            mt_col = int(cov.reaction_cols[i])
            if mt_col not in requested:
                continue
            grid = (
                cov.energy_grids[i]
                if i < len(cov.energy_grids) and cov.energy_grids[i]
                else None
            )
            if grid is None:
                continue
            pts.update(float(e) for e in grid)
    return sorted(pts)


def _project_matrix(
    matrix: np.ndarray,
    native_grid: Sequence[float],
    union_grid: Sequence[float],
) -> np.ndarray:
    """Piecewise-constant projection of an interval matrix to the union grid."""
    rebin = compute_rebin_operator(
        np.asarray(native_grid, dtype=float),
        np.asarray(union_grid, dtype=float),
    )
    return collapse_covariance(matrix, rebin)


# Inert-bin floor for absolute→relative conversion. Bins whose bin-averaged
# σ̄ falls below ``max(SIGMA_FLOOR_REL × σ̄_max, SIGMA_FLOOR_ABS)`` are
# treated as having zero perturbation (factor = 1, variance = 0). This
# prevents 1/σ̄ → ∞ blow-ups in:
#   - threshold-spanning bins where MF33 covers a region σ has not yet
#     ramped up (e.g. O-16 MT=4 at 6.27–6.45 MeV)
#   - placeholder/filler bins evaluators write at the high-E end of a
#     lumped MT whose σ̄ collapses (e.g. JEFF-4.0 Mn-55 MT=856 above
#     ~22 MeV)
# Physical interpretation: when there is no cross section in a bin, the
# relative covariance is undefined and any multiplicative perturbation is
# meaningless — the correct answer is "do not perturb".
SIGMA_FLOOR_REL = 1.0e-3
SIGMA_FLOOR_ABS = 1.0e-9  # barns


def _absolute_to_relative(
    mat_abs: np.ndarray,
    sigma_row: np.ndarray,
    sigma_col: np.ndarray,
) -> np.ndarray:
    """C_rel[i,j] = C_abs[i,j] / (σ_row[i] * σ_col[j]); inert rows/cols set to 0.

    A row/col is considered inert when its σ̄ falls below the per-MT floor
    ``max(SIGMA_FLOOR_REL × σ̄_max, SIGMA_FLOOR_ABS)``.
    """
    sr_max = float(np.max(sigma_row)) if sigma_row.size else 0.0
    sc_max = float(np.max(sigma_col)) if sigma_col.size else 0.0
    floor_r = max(SIGMA_FLOOR_REL * sr_max, SIGMA_FLOOR_ABS)
    floor_c = max(SIGMA_FLOOR_REL * sc_max, SIGMA_FLOOR_ABS)
    sr = np.where(sigma_row > floor_r, sigma_row, 0.0)
    sc = np.where(sigma_col > floor_c, sigma_col, 0.0)
    inv_sr = np.where(sr > 0, 1.0 / sr, 0.0)
    inv_sc = np.where(sc > 0, 1.0 / sc, 0.0)
    return (mat_abs * inv_sr[:, None]) * inv_sc[None, :]


def _augment_with_step_duplicates(
    energies: np.ndarray,
    xs: np.ndarray,
    interpolation: List[Tuple[int, int]],
    bins_native: np.ndarray,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Insert duplicate-energy points at every interior MF33 bin edge that
    falls strictly inside ``(energies[0], energies[-1])``.

    Per interior MF33 bin edge ``E_b``:

      * 0 existing matches → interpolate ``σ_b`` from the (assumed lin-lin)
        original table and insert two consecutive points ``(E_b, σ_b)``.
        After the duplicate-aware factor pass this becomes a real step
        whenever ``f_lower != f_upper``.
      * 1 existing match    → insert a second duplicate at ``E_b`` carrying
        the same ``σ`` value as the existing point. Same effect on the
        perturbed σ as the 0-match case.
      * ≥2 existing matches → leave alone (PENDF already encodes a step
        there, e.g. a reaction threshold; preserve it).

    NBT values in ``interpolation`` are bumped to account for inserted
    points (INT codes are unchanged). PENDF from RECONR is the typical
    input — almost always a single ``[(NP, 2)]`` region.

    Returns ``(new_energies, new_xs, new_interpolation)``.
    """
    e_orig = np.asarray(energies, dtype=float)
    s_orig = np.asarray(xs, dtype=float)
    bins = np.asarray(bins_native, dtype=float)
    if e_orig.size < 2 or bins.size < 2:
        return e_orig, s_orig, list(interpolation)

    e_min = float(e_orig[0])
    e_max = float(e_orig[-1])
    interior_edges = sorted({
        float(b) for b in bins
        if (b > e_min + atol) and (b < e_max - atol)
    })
    if not interior_edges:
        return e_orig, s_orig, list(interpolation)

    # Plan insertions in *original-array* coordinates so NBT updates and
    # position-shifting stay tractable.
    plan: List[Tuple[int, int, float, float]] = []  # (insert_before_idx, n, sigma_b, E_b)
    for E_b in interior_edges:
        match_count = 0
        first_match: Optional[int] = None
        for i, ev in enumerate(e_orig):
            if abs(ev - E_b) < atol:
                match_count += 1
                if first_match is None:
                    first_match = i
                if match_count >= 2:
                    break
        if match_count >= 2:
            continue
        if match_count == 1:
            assert first_match is not None
            plan.append((first_match + 1, 1, float(s_orig[first_match]), E_b))
        else:
            insert_before = int(np.searchsorted(e_orig, E_b, side="right"))
            sigma_b = float(np.interp(E_b, e_orig, s_orig))
            plan.append((insert_before, 2, sigma_b, E_b))

    if not plan:
        return e_orig, s_orig, list(interpolation)

    e_aug = list(map(float, e_orig))
    s_aug = list(map(float, s_orig))
    # Apply in descending insert position so earlier positions remain valid,
    # and -- within one position -- in descending *energy*.
    #
    # The energy half of that key was missing until 2026-09-06, and its absence
    # produced a non-monotonic table. Two bin edges that fall in the same
    # original interval both get `insert_before` = that interval's index; the
    # sort was stable on position alone, so the lower edge was inserted first
    # and the higher one then landed in front of it. The section that came out
    # was not merely different, it was an invalid TAB1: ENDF-6 requires
    # ascending abscissae, and every consumer downstream -- NJOY's `lunion`
    # first -- reads the result wrong or refuses it.
    #
    # Measured on the Fe-56 evaluation the thesis uses, its own MF33 grid (125
    # edges) against its MF3 grids: 2 disordered insertions in MT1, 2 in MT102
    # and 28 in MT2. Whether it fires in the live pipeline depends on the input:
    # this applier is given a *PENDF*, whose grid is far denser in the resolved
    # range, so two MF33 edges sharing an interval is rarer there than on the
    # ENDF grid measured above -- see `docs/library/library-gaps.md` D31, which
    # records the measurement that is still owed on a real PENDF.
    for pos, n, sigma_b, E_b in sorted(plan, key=lambda p: (p[0], p[3]),
                                       reverse=True):
        if n == 1:
            e_aug.insert(pos, E_b)
            s_aug.insert(pos, sigma_b)
        else:
            e_aug.insert(pos, E_b)
            s_aug.insert(pos, sigma_b)
            e_aug.insert(pos + 1, E_b)
            s_aug.insert(pos + 1, sigma_b)

    # NBT_i is the cumulative 1-based count of the last point of region i.
    # Insertions with insert_before_idx <= NBT_i - 1 (= strictly < NBT_i)
    # shift the original last point of region i to the right by n, so:
    #    new_NBT_i = NBT_i + sum(n for plan_pos < NBT_i)
    new_interp: List[Tuple[int, int]] = []
    for nbt, intc in interpolation:
        added = sum(n for pos, n, _, _ in plan if pos < int(nbt))
        new_interp.append((int(nbt) + int(added), int(intc)))

    return np.asarray(e_aug, dtype=float), np.asarray(s_aug, dtype=float), new_interp


def _bin_average_xs(
    energies: np.ndarray,
    xs: np.ndarray,
    bin_edges: Sequence[float],
    n_quad: int = 64,
) -> np.ndarray:
    """1/E-weighted bin average of σ(E) over each interval of ``bin_edges``.

    Mirrors the approach in ``MF33MT._bin_average_xs`` but stays self-
    contained (linear interpolation between tabulated PENDF points).
    """
    edges = np.asarray(bin_edges, dtype=float)
    out = np.zeros(edges.size - 1, dtype=float)
    if energies.size == 0:
        return out
    e_min = float(energies[0])
    e_max = float(energies[-1])

    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if hi <= e_min or lo >= e_max:
            out[i] = 0.0
            continue
        lo_clip = max(lo, e_min)
        hi_clip = min(hi, e_max)
        if hi_clip <= lo_clip:
            out[i] = 0.0
            continue
        # 1/E-weighted average via log-spaced quadrature, with a linear
        # fallback if the interval starts at 0 eV (rare for ENDF MF3).
        if lo_clip > 0:
            sample_e = np.geomspace(lo_clip, hi_clip, n_quad)
            weights = 1.0 / sample_e
        else:
            sample_e = np.linspace(lo_clip, hi_clip, n_quad)
            weights = np.ones_like(sample_e)
        sample_xs = np.interp(sample_e, energies, xs, left=0.0, right=0.0)
        num = np.trapezoid(weights * sample_xs, sample_e)
        den = np.trapezoid(weights, sample_e)
        out[i] = num / den if den > 0 else 0.0
    return out
