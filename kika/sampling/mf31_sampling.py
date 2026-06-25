"""
MF31 (nu-bar) covariance assembly and per-ENDF MF1 nu-bar factor application.

Sister module to :mod:`kika.sampling.mf33_sampling`. Because the ENDF-6 MF31
formats for MT=452/455/456 are identical to MF33 (manual Ch. 31), the covariance
assembly reuses the MF33 record machinery wholesale — the only differences are:

  * the central values used for absolute→relative conversion and NC-LTY=0
    resolution come from the **MF1 nu-bar(E) curves** (not PENDF MF3 σ(E)); and
  * factors are applied to the MF1 nu-bar tables in the ENDF tape, after which
    the standard NJOY chain regenerates ACE.

Public API
----------
- ``load_mf31_covariance``    — assemble the unified nu-bar covariance on the
                                MF31 union grid (relative), plus the parsed MF1
                                nu-bar sections and the union grid.
- ``perturb_nubar_family``    — apply sampled factors to the whole 452/455/456
                                family on one shared grid, enforcing the sum
                                rule nu = nu_d + nu_p exactly (composite
                                semantics across all MF31 coverage patterns).
- ``apply_factors_to_mf1_nubar`` — single-section primitive (own grid), used by
                                the family routine and available standalone.
- ``extract_mt_param_blocks`` — re-exported from mf33_sampling.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kika._constants import NUBAR_TOTAL_MT, NUBAR_COMPONENT_MTS
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.sampling.mf33_sampling import (
    extract_mt_param_blocks,  # re-exported; identical flatten order
    perturb_pointwise_xs,
    _build_union_grid,
    _project_matrix,
    _absolute_to_relative,
    _bin_average_xs,
)

__all__ = [
    "load_mf31_covariance",
    "perturb_nubar_family",
    "apply_factors_to_mf1_nubar",
    "extract_mt_param_blocks",
]

# nu-bar MTs in the canonical (total, prompt, delayed) family.
_NUBAR_MTS: Tuple[int, ...] = (NUBAR_TOTAL_MT, *NUBAR_COMPONENT_MTS)  # (452, 455, 456)


# ---------------------------------------------------------------------------
# Central-value shim: lets MF33MT.to_xs_covmat treat nu-bar(E) the way it
# treats pointwise σ(E) from PENDF MF3 (needs .energies/.cross_sections for
# this module and .get_cross_section for MF33MT._bin_average_xs).
# ---------------------------------------------------------------------------

class _NubarXS:
    """Tabulated nu-bar(E) presented with the MF3MT-like interface."""

    def __init__(self, energies: Sequence[float], nubar: Sequence[float]):
        self.energies = np.asarray(energies, dtype=float)
        self.cross_sections = np.asarray(nubar, dtype=float)
        self.values = self.cross_sections  # alias used by _bin_average_xs fallback

    def get_cross_section(self, e):
        return np.interp(
            e, self.energies, self.cross_sections,
            left=self.cross_sections[0] if self.cross_sections.size else 0.0,
            right=self.cross_sections[-1] if self.cross_sections.size else 0.0,
        )


def _nubar_central_values(mf1_file) -> Dict[int, _NubarXS]:
    """Build {MT: _NubarXS} from the MF1 nu-bar sections (tabulated, eV)."""
    shim: Dict[int, _NubarXS] = {}
    if mf1_file is None or not getattr(mf1_file, "sections", None):
        return shim
    for mt in _NUBAR_MTS:
        sec = mf1_file.sections.get(mt)
        if sec is None:
            continue
        energies, nubar = _nubar_as_tabulated(sec)
        if energies.size:
            shim[mt] = _NubarXS(energies, nubar)
    return shim


# ---------------------------------------------------------------------------
# Covariance assembly
# ---------------------------------------------------------------------------

def load_mf31_covariance(
    endf_path: str,
    mt_list: Optional[Sequence[int]] = None,
    *,
    energy_unit: str = "eV",
    logger=None,
) -> Tuple[CrossSectionCovariance, Dict[int, object], List[float], List[int]]:
    """Build a CrossSectionCovariance on the MF31 native union grid.

    Mirrors :func:`kika.sampling.mf33_sampling.load_mf33_covariance`, but draws
    central values from MF1 nu-bar instead of PENDF MF3.

    Parameters
    ----------
    endf_path : str
        ENDF tape containing MF31 and MF1 nu-bar sections.
    mt_list : sequence of int, optional
        Restrict to these nu-bar MTs. Default: every MT present in MF31.

    Returns
    -------
    cov : CrossSectionCovariance
        Unified covariance, all blocks relative, on the shared union grid.
    nubar_sections : dict
        {MT: MF1 nu-bar section} for the perturbable MTs (452/455/456 present).
    union_grid : list of float
        MF31 native union grid in ``energy_unit``.
    mts_present : list of int
        Subset of requested MTs for which MF31 data was available (sorted).
    """
    endf_obj = parse_endf_file(str(endf_path))
    mf31_file = endf_obj.get_file(31)
    if mf31_file is None or not getattr(mf31_file, "sections", None):
        raise RuntimeError(f"ENDF {endf_path} has no MF31 sections")
    mf1_file = endf_obj.get_file(1)

    nubar_central = _nubar_central_values(mf1_file)
    sibling_sections = dict(mf31_file.sections)

    if mt_list is None:
        requested_mts = sorted(int(mt) for mt in mf31_file.sections)
    else:
        requested_mts = sorted({int(mt) for mt in mt_list})
    mts_present = [mt for mt in requested_mts if mt in mf31_file.sections]
    missing = sorted(set(requested_mts) - set(mts_present))
    if missing and logger is not None:
        logger.warning(
            f"[MF31] MTs {missing}: not in MF31; no perturbation applied for these."
        )
    if not mts_present:
        raise RuntimeError(
            f"None of MTs {requested_mts} are present in MF31 of {endf_path}"
        )

    per_mt_covs: Dict[int, CrossSectionCovariance] = {}
    for mt in mts_present:
        mf31mt = mf31_file.sections[mt]
        per_mt_covs[mt] = mf31mt.to_xs_covmat(
            energy_unit=energy_unit,
            sibling_sections=sibling_sections,
            mf3_sections=nubar_central if nubar_central else None,
        )

    union_grid = _build_union_grid(per_mt_covs, mts_present)
    if len(union_grid) < 2:
        raise RuntimeError(
            "MF31 union grid has fewer than 2 boundaries; cannot build covariance"
        )

    # Bin-averaged nu-bar per MT on the union grid (for absolute→relative).
    bin_nu: Dict[int, np.ndarray] = {}
    for mt in mts_present:
        if mt in nubar_central:
            bin_nu[mt] = _bin_average_xs(
                nubar_central[mt].energies,
                nubar_central[mt].cross_sections,
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
                if mt_row not in bin_nu or mt_col not in bin_nu:
                    if logger is not None:
                        logger.warning(
                            f"[MF31] Cannot convert absolute block "
                            f"({mt_row}↔{mt_col}) to relative — MF1 nu-bar missing; "
                            f"skipping block."
                        )
                    continue
                mat_proj = _absolute_to_relative(
                    mat_proj, bin_nu[mt_row], bin_nu[mt_col]
                )

            unified.add_matrix(
                iso_row, mt_row, iso_col, mt_col,
                mat_proj,
                energy_grid=list(union_grid),
                is_relative=True,
            )

    nubar_sections = {
        mt: mf1_file.sections[mt]
        for mt in _NUBAR_MTS
        if mf1_file is not None and getattr(mf1_file, "sections", None)
        and mt in mf1_file.sections
    }
    return unified, nubar_sections, list(union_grid), mts_present


# ---------------------------------------------------------------------------
# nu-bar representation helpers
# ---------------------------------------------------------------------------

def _nubar_as_tabulated(section) -> Tuple[np.ndarray, np.ndarray]:
    """Return (energies_eV, nu-bar) for an MF1 nu-bar section.

    For LNU=2 the tabulated arrays are returned directly. For LNU=1 (polynomial)
    empty arrays are returned — the caller reconstructs onto an explicit grid
    (manual Ch. 31 Note 2: the covariance applies to the tabular reconstruction,
    not the polynomial coefficients).
    """
    if getattr(section, "lnu", None) == 2:
        return (
            np.asarray(section.energies, dtype=float),
            np.asarray(section.nubar_values, dtype=float),
        )
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def _reconstruct_polynomial_on_grid(
    coefficients: Sequence[float], grid_eV: Sequence[float]
) -> np.ndarray:
    """Evaluate nu(E) = Σ Cₙ Eⁿ on ``grid_eV`` (E in eV)."""
    e = np.asarray(grid_eV, dtype=float)
    nu = np.zeros_like(e)
    for n, c in enumerate(coefficients):
        nu += float(c) * e ** n
    return nu


def _baseline_energies(section, fallback_grid: np.ndarray) -> np.ndarray:
    """Original tabulated energies for a section, or the fallback grid (LNU=1)."""
    e, _ = _nubar_as_tabulated(section)
    return e if e.size else np.asarray(fallback_grid, dtype=float)


def _set_nubar(section, energies: np.ndarray, nubar: np.ndarray, interp):
    """Return a copy of ``section`` as an LNU=2 table with the given arrays."""
    return dataclasses.replace(
        section,
        _lnu=2,
        _nc=0,
        _coefficients=[],
        _energies=list(energies),
        _nubar=list(nubar),
        _interpolation=list(interp),
        _nr=len(interp),
        _np=len(energies),
    )


# ---------------------------------------------------------------------------
# Single-section primitive (own grid)
# ---------------------------------------------------------------------------

def _lnu1_reconstruction_grid(bins: np.ndarray, per_bin: int = 8) -> np.ndarray:
    """Grid for reconstructing a polynomial nu-bar: each MF31 bin subdivided.

    Polynomial nu-bar is smooth, so a handful of geometric points per bin
    reproduces it well while letting the per-bin factors apply cleanly (every
    bin interior carries native nodes). Manual Ch. 31 Note 2.
    """
    b = np.asarray(bins, dtype=float)
    segs = []
    for i in range(b.size - 1):
        lo, hi = b[i], b[i + 1]
        if lo <= 0:
            lo = min(hi * 1.0e-6, 1.0e-5)
        segs.append(np.geomspace(lo, hi, per_bin + 1))
    return np.unique(np.concatenate(segs))


def _insert_bin_edges(
    energies: np.ndarray, nubar: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Add a single interpolated point at each interior bin edge in range.

    Unlike the MF3 step-duplicate augmentation, this inserts *single* (non-
    duplicate) points so every MF31 bin is represented even for coarse nu-bar
    tables (e.g. a 10-point delayed curve), while keeping nu-bar continuous and
    the table compact. Duplicate energies are deliberately avoided: NJOY ACER
    reads MF1 into fixed-size buffers and chokes on bloated / duplicate-energy
    nu-bar tables.
    """
    e = np.asarray(energies, dtype=float)
    n = np.asarray(nubar, dtype=float)
    if e.size < 2:
        return e, n, [(int(e.size), 2)]
    interior = np.asarray(bins, dtype=float)[1:-1]
    lo, hi = e[0], e[-1]
    add = [
        b for b in interior
        if lo < b < hi and not np.any(np.isclose(e, b, rtol=0.0, atol=0.0))
    ]
    if add:
        add_e = np.asarray(add, dtype=float)
        add_n = np.interp(add_e, e, n)
        e = np.concatenate([e, add_e])
        n = np.concatenate([n, add_n])
        order = np.argsort(e, kind="mergesort")
        e, n = e[order], n[order]
    return e, n, [(int(e.size), 2)]


def apply_factors_to_mf1_nubar(
    section,
    factor_block: np.ndarray,
    bins: Sequence[float],
    *,
    logger=None,
):
    """Return a perturbed copy of one MF1 nu-bar section on its own grid.

    Piecewise per-MF31-bin scaling of the tabulated nu-bar(E): single
    interpolated points are inserted at interior bin edges (so every bin is
    represented even for coarse tables) and the per-bin factor is applied to
    each point. LNU=1 sections are reconstructed onto the MF31 bin grid
    (→ LNU=2) first.

    Returns ``(new_section, diagnostics)``. For redundancy-aware perturbation of
    the whole 452/455/456 family, use :func:`perturb_nubar_family` instead.
    """
    bins_arr = np.asarray(bins, dtype=float)
    n_groups = bins_arr.size - 1
    block = np.asarray(factor_block, dtype=float)
    if block.size != n_groups:
        raise ValueError(
            f"factor block has size {block.size}, expected n_groups={n_groups}"
        )

    if getattr(section, "lnu", None) == 1:
        energies_orig = _lnu1_reconstruction_grid(bins_arr)
        nubar_orig = _reconstruct_polynomial_on_grid(section.coefficients, energies_orig)
        if logger is not None:
            logger.info(
                f"  [MF31] MT{section.number}: LNU=1 polynomial reconstructed "
                f"onto {len(energies_orig)} MF31-subdivided points (→ LNU=2)."
            )
    else:
        energies_orig = np.asarray(section.energies, dtype=float)
        nubar_orig = np.asarray(section.nubar_values, dtype=float)

    energies_aug, nubar_aug, interp_aug = _insert_bin_edges(
        energies_orig, nubar_orig, bins_arr,
    )
    nubar_new, factors, frac_out = perturb_pointwise_xs(
        energies_aug, nubar_aug, block, bins_arr,
    )
    new_section = _set_nubar(section, energies_aug, nubar_new, interp_aug)
    diagnostics = {
        "min_factor": float(factors.min()) if factors.size else 1.0,
        "max_factor": float(factors.max()) if factors.size else 1.0,
        "frac_out_of_coverage": frac_out,
        "n_inserted": int(energies_aug.size - energies_orig.size),
    }
    return new_section, diagnostics


# ---------------------------------------------------------------------------
# Family perturbation with exact sum rule  (nu = nu_d + nu_p)
# ---------------------------------------------------------------------------

def perturb_nubar_family(
    sections: Dict[int, object],
    factor_blocks: Dict[int, np.ndarray],
    bins: Sequence[float],
    *,
    derived_mt: Optional[int] = None,
    logger=None,
) -> Tuple[Dict[int, object], Dict[int, dict]]:
    """Apply sampled factors to the 452/455/456 family, enforcing the sum rule.

    Each member is perturbed on *its own* compact grid (its tabulated energies
    plus single interpolated points at the interior MF31 bin edges). The derived
    total is then built on the union of the perturbed members' grids and
    evaluated as nu_d + nu_p; because no member carries duplicate energies, that
    union sum is exact (a piecewise-linear identity), so the sum rule holds to
    machine precision without bloating the sparse delayed-nu-bar table. Handles
    all MF31 coverage patterns with one rule (mirrors ``UQ.sandwich`` ``nubar_mode``):

      * **components carry covariance** (455 and/or 456): each perturbed by its
        own factor; an unperturbed component stays at baseline; total derived.
      * **only total (452) carries covariance**: total's factor rides onto both
        components, so all three scale together and the sum rule still holds.
      * **all three carry covariance**: components perturbed independently, the
        redundant total is recomputed (its own factor block is discarded).

    Parameters
    ----------
    sections : dict
        Baseline {MT: MF1 nu-bar section} for the MTs present (452/455/456).
    factor_blocks : dict
        {MT: per-bin factor block} for the MTs that have MF31 covariance.
    bins : sequence of float
        MF31 union grid (eV).
    derived_mt : int, optional
        Which family member is the derived redundant one. Default: total (452).

    Returns
    -------
    (out_sections, diagnostics)
        ``out_sections`` is {MT: perturbed MF1 section} for every MT in
        ``sections``; ``diagnostics`` is {MT: {min_factor, max_factor,
        frac_out_of_coverage}} for the directly-perturbed members.
    """
    bins_arr = np.asarray(bins, dtype=float)
    n_groups = bins_arr.size - 1
    derived = NUBAR_TOTAL_MT if derived_mt is None else int(derived_mt)
    present = set(int(mt) for mt in sections)

    components_present = [mt for mt in NUBAR_COMPONENT_MTS if mt in present]
    can_derive = (NUBAR_TOTAL_MT in present) and len(components_present) >= 1

    # Degenerate case: no derivable family (e.g. only total present, no
    # components) → perturb each present member on its own grid directly.
    if not can_derive:
        out: Dict[int, object] = {}
        diags: Dict[int, dict] = {}
        for mt, sec in sections.items():
            if mt in factor_blocks:
                out[mt], diags[mt] = apply_factors_to_mf1_nubar(
                    sec, factor_blocks[mt], bins_arr, logger=logger
                )
            else:
                out[mt] = sec
        return out, diags

    # --- Decide which factor block each member rides -----------------------
    # Total perturbed but a component is not → component rides the total factor
    # ("perturb everything with the total").
    total_block = factor_blocks.get(NUBAR_TOTAL_MT)
    block_for: Dict[int, Optional[np.ndarray]] = {}
    for mt in present:
        if mt == derived:
            continue  # derived is computed from the others, not applied
        if mt in factor_blocks:
            block_for[mt] = np.asarray(factor_blocks[mt], dtype=float)
        elif mt in NUBAR_COMPONENT_MTS and total_block is not None:
            block_for[mt] = np.asarray(total_block, dtype=float)
        else:
            block_for[mt] = None  # baseline (factor 1)

    # --- Perturb each non-derived member on its OWN compact grid -----------
    out = {}
    diags = {}
    for mt in present:
        if mt == derived:
            continue
        block = block_for[mt]
        if block is None:
            out[mt] = sections[mt]  # unperturbed, unchanged
        else:
            if block.size != n_groups:
                raise ValueError(
                    f"factor block for MT={mt} has size {block.size}, "
                    f"expected n_groups={n_groups}"
                )
            out[mt], diags[mt] = apply_factors_to_mf1_nubar(
                sections[mt], block, bins_arr, logger=logger
            )

    # --- Recompute the derived member on the union of the others' grids ----
    # No member has duplicate energies, so the union sum is an exact
    # piecewise-linear identity → the sum rule holds to machine precision.
    if derived in present:
        if derived == NUBAR_TOTAL_MT:
            contributors = components_present
            signs = {mt: +1.0 for mt in contributors}
        else:
            contributors = [NUBAR_TOTAL_MT] + [
                mt for mt in NUBAR_COMPONENT_MTS if mt != derived
            ]
            signs = {mt: (+1.0 if mt == NUBAR_TOTAL_MT else -1.0) for mt in contributors}

        grids = [_baseline_energies(out.get(mt) or sections[mt], bins_arr)
                 for mt in contributors if (mt in out or mt in sections)]
        union_e = np.unique(np.concatenate(grids))
        nu_derived = np.zeros_like(union_e)
        for mt in contributors:
            sec = out.get(mt) or sections.get(mt)
            if sec is None:
                continue
            e, n = _nubar_as_tabulated(sec)
            if e.size == 0:
                continue
            nu_derived = nu_derived + signs[mt] * np.interp(
                union_e, e, n, left=n[0], right=n[-1]
            )
        out[derived] = _set_nubar(
            sections[derived], union_e, nu_derived, [(int(union_e.size), 2)]
        )
        if logger is not None:
            logger.info(
                f"  [MF31] MT{derived}: recomputed from sum rule on "
                f"{union_e.size} points."
            )

    return out, diags
