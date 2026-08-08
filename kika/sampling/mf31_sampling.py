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
    "build_mf31_covariance",
    "perturb_nubar_family",
    "apply_factors_to_mf1_nubar",
    "sum_rule_residual",
    "extract_mt_param_blocks",
]

# nu-bar MTs in the canonical (total, prompt, delayed) family.
_NUBAR_MTS: Tuple[int, ...] = (NUBAR_TOTAL_MT, *NUBAR_COMPONENT_MTS)  # (452, 455, 456)

#: Sub-intervals per MF31 bin used to reconstruct an LNU=1 polynomial nu-bar
#: onto an explicit table (manual Ch. 31 Note 2).
NUBAR_NODES_PER_BIN: int = 8

#: Relative width of the "shoulder" node placed just below each interior MF31
#: bin edge, which is how a duplicate-free lin-lin table approximates the step
#: in the per-bin factor (see :func:`_augment_nubar_grid`). 1e-3 keeps the
#: worst-case per-bin factor error at ~5e-4 while staying three orders of
#: magnitude clear of the ~1e-6 relative resolution of the ENDF 11-character
#: float — a shoulder any tighter would be *written* as a duplicate energy.
NUBAR_STEP_SHOULDER: float = 1.0e-3

#: Two energies closer than this in relative terms are the same energy once
#: written to an ENDF record, so never insert a node that close to an existing
#: one.
_ENDF_ENERGY_RTOL: float = 1.0e-6


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


#: Points used to reconstruct an LNU=1 (polynomial) nu-bar as central values.
#: Exact for the NC=1 constant the manual prescribes for spontaneous fission
#: (§1.3.2, §1.4), and ample for any smooth low-order polynomial.
_LNU1_CENTRAL_POINTS: int = 500


def _nubar_central_values(mf1_file) -> Dict[int, _NubarXS]:
    """Build {MT: _NubarXS} from the MF1 nu-bar sections (tabulated, eV).

    LNU=1 sections are reconstructed onto a dense log grid spanning the
    material's energy range rather than being skipped: without central values
    an absolute MF31 block cannot be converted to relative and an NC LTY=0
    sub-subsection cannot be resolved, so the covariance would be dropped for
    exactly the sections that carry no table.
    """
    shim: Dict[int, _NubarXS] = {}
    if mf1_file is None or not getattr(mf1_file, "sections", None):
        return shim

    mt451 = mf1_file.sections.get(451)
    e_max = float(getattr(mt451, "_emax", None) or 0.0) or 2.0e7

    for mt in _NUBAR_MTS:
        sec = mf1_file.sections.get(mt)
        if sec is None:
            continue
        energies, nubar = _nubar_as_tabulated(sec)
        if not energies.size and getattr(sec, "lnu", None) == 1:
            energies = np.geomspace(1.0e-5, e_max, _LNU1_CENTRAL_POINTS)
            nubar = _reconstruct_polynomial_on_grid(
                _nubar_coefficients(sec), energies
            )
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
    return build_mf31_covariance(
        endf_obj, mt_list, energy_unit=energy_unit, logger=logger
    )


def build_mf31_covariance(
    endf_obj,
    mt_list: Optional[Sequence[int]] = None,
    *,
    energy_unit: str = "eV",
    logger=None,
) -> Tuple[CrossSectionCovariance, Dict[int, object], List[float], List[int]]:
    """Build the MF31 nu-bar covariance from an already-parsed ENDF object.

    Identical to :func:`load_mf31_covariance` but accepts a parsed ``endf_obj``,
    avoiding a re-parse when the caller already holds one (e.g. the kika-app
    ENDF endpoints). Returns the same 4-tuple
    ``(cov, nubar_sections, union_grid, mts_present)``.
    """
    mf31_file = endf_obj.get_file(31)
    if mf31_file is None or not getattr(mf31_file, "sections", None):
        raise RuntimeError("ENDF object has no MF31 sections")
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
            f"None of MTs {requested_mts} are present in MF31"
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


def _nubar_coefficients(section) -> List[float]:
    """LNU=1 polynomial coefficients of a nu-bar section."""
    coeffs = getattr(section, "coefficients", None)
    if coeffs is None:
        coeffs = getattr(section, "_coefficients", None)
    if not coeffs:
        raise ValueError(
            f"MT{getattr(section, 'number', '?')}: LNU=1 section carries no "
            f"polynomial coefficients"
        )
    return list(coeffs)


def _warn_non_linlin(section, logger) -> None:
    """Warn when a section we are about to rewrite is not purely lin-lin.

    The perturbed table is always written as a single INT=2 region (the
    per-bin factor application and the sum-rule union are both lin-lin
    identities), so any other interpolation law in the original is silently
    reinterpreted. Every JEFF-4.0 nu-bar section is lin-lin, but say so if
    that ever stops being true.
    """
    if logger is None:
        return
    laws = {int(intc) for _nbt, intc in (getattr(section, "interpolation", None) or [])}
    if laws - {2}:
        logger.warning(
            f"  [MF31] MT{getattr(section, 'number', '?')}: interpolation laws "
            f"{sorted(laws)} present; the perturbed table is rewritten as a "
            f"single lin-lin (INT=2) region."
        )


def _reconstruct_polynomial_on_grid(
    coefficients: Sequence[float], grid_eV: Sequence[float]
) -> np.ndarray:
    """Evaluate nu(E) = Σ Cₙ Eⁿ on ``grid_eV`` (E in eV)."""
    e = np.asarray(grid_eV, dtype=float)
    nu = np.zeros_like(e)
    for n, c in enumerate(coefficients):
        nu += float(c) * e ** n
    return nu


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

def _lnu1_reconstruction_grid(
    bins: np.ndarray, per_bin: int = NUBAR_NODES_PER_BIN
) -> np.ndarray:
    """Grid for reconstructing a polynomial nu-bar: each MF31 bin subdivided.

    Polynomial nu-bar is smooth, so a handful of geometric points per bin
    reproduces it well while letting the per-bin factors apply cleanly (every
    bin interior carries native nodes). Bins starting at (or below) zero get a
    small positive lower bound so the geometric spacing stays defined.
    Manual Ch. 31 Note 2.
    """
    b = np.asarray(bins, dtype=float)
    segs = []
    for i in range(b.size - 1):
        x0, x1 = b[i], b[i + 1]
        if not (x1 > x0):
            continue
        if x0 <= 0:
            x0 = min(x1 * 1.0e-6, 1.0e-5)
        segs.append(np.geomspace(x0, x1, per_bin + 1))
    if not segs:
        return np.asarray([], dtype=float)
    return np.unique(np.concatenate(segs))


def _augment_nubar_grid(
    energies: np.ndarray,
    nubar: np.ndarray,
    bins: np.ndarray,
    *,
    shoulder: float = NUBAR_STEP_SHOULDER,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Resolve the nu-bar table against the MF31 bin structure.

    The factor block is piecewise constant per MF31 bin, but a duplicate-free
    lin-lin table cannot express a step: between the last node of bin *g* and
    the edge it shares with bin *g+1*, the perturbed curve ramps from
    ``f_g·nu`` to ``f_{g+1}·nu``, so the factor the file actually realises over
    bin *g* is a blend of both. On a table as coarse as JEFF-4.0 Pu-241 MT456
    (6 points for 15 MF31 bins) that ramp spans the whole bin and a worst-case
    alternating ±5 % block cancels outright.

    So two kinds of node are inserted, both *single* (non-duplicate) and both
    valued by ``np.interp`` on the original table — the baseline curve is
    preserved exactly:

      * one at every interior MF31 bin edge inside the table's range, which is
        what puts the factor step at the right energy; and
      * one "shoulder" just below each of those edges, at
        ``E_b · (1 - shoulder)``, which confines the ramp to that sliver.

    Measured on the three JEFF-4.0 actinide tapes with a worst-case alternating
    ±5 % block, the maximum per-bin factor error drops from 0.050 (bin edges
    only) to 5e-4 — and the table grows *less* than it would under a uniform
    subdivision of every bin (Pu-241 MT455: 17 → 39 points, versus 129 for 8
    sub-intervals per bin at ten times the error). That matters because NJOY
    ACER reads MF1 into fixed-size buffers and does not love bloated nu-bar
    tables; duplicate energies are avoided for the same reason.

    Returns ``(energies, nubar, interpolation)`` with a single lin-lin region.
    """
    e = np.asarray(energies, dtype=float)
    n = np.asarray(nubar, dtype=float)
    if e.size < 2:
        return e, n, [(int(e.size), 2)]

    b = np.asarray(bins, dtype=float)
    lo, hi = float(e[0]), float(e[-1])

    candidates: List[float] = []
    for edge in b[1:-1]:
        edge = float(edge)
        # ``edge == hi`` still needs a shoulder: the table's last point sits on
        # a bin boundary and (side='right') takes the *upper* bin's factor, so
        # without one the whole top of that bin ramps towards its neighbour.
        # ``bins[-1]`` is excluded because ``clamp_top_edge`` already pulls a
        # point there back into the last bin — there is no step to resolve.
        if not (lo < edge <= hi):
            continue
        if edge < hi:
            candidates.append(edge)
        shoulder_e = edge * (1.0 - shoulder)
        if shoulder_e > lo:
            candidates.append(shoulder_e)

    if candidates:
        cand = np.unique(np.asarray(candidates, dtype=float))
        cand = cand[(cand > lo) & (cand < hi)]
        if cand.size:
            # A node closer than the ENDF float resolution to one already there
            # would be written as a duplicate energy — the very thing this
            # construction exists to avoid.
            nearest = np.searchsorted(e, cand).clip(1, e.size - 1)
            keep = ~(np.isclose(cand, e[nearest], rtol=_ENDF_ENERGY_RTOL, atol=0.0)
                     | np.isclose(cand, e[nearest - 1], rtol=_ENDF_ENERGY_RTOL, atol=0.0))
            cand = cand[keep]
        if cand.size:
            e = np.concatenate([e, cand])
            n = np.concatenate([n, np.interp(cand, energies, nubar)])
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
    interpolated points are inserted at each interior bin edge and just below it
    (see :func:`_augment_nubar_grid`), then the per-bin factor is applied to
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
        nubar_orig = _reconstruct_polynomial_on_grid(
            _nubar_coefficients(section), energies_orig
        )
        if logger is not None:
            logger.info(
                f"  [MF31] MT{section.number}: LNU=1 polynomial reconstructed "
                f"onto {len(energies_orig)} MF31-subdivided points (→ LNU=2)."
            )
    else:
        energies_orig = np.asarray(section.energies, dtype=float)
        nubar_orig = np.asarray(section.nubar_values, dtype=float)
        _warn_non_linlin(section, logger)

    energies_aug, nubar_aug, interp_aug = _augment_nubar_grid(
        energies_orig, nubar_orig, bins_arr,
    )
    nubar_new, factors, frac_out = perturb_pointwise_xs(
        energies_aug, nubar_aug, block, bins_arr, clamp_top_edge=True,
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
# Sum-rule diagnostics
# ---------------------------------------------------------------------------

def sum_rule_residual(
    sections: Dict[int, object], bins: Sequence[float]
) -> Optional[Dict[str, object]]:
    """How far the *input* evaluation is from nu_452 = nu_455 + nu_456.

    The ENDF-6 manual (§1.2.2) requires the three to be consistent, but
    evaluations miss by a little — typically because the delayed table is far
    coarser than the prompt one, so its lin-lin interpolant does not reproduce
    the nu_d the total was built with. :func:`perturb_nubar_family` rebuilds the
    derived member, which repairs that residual as a side effect; this function
    puts a number on the repair so it is recorded rather than silent.

    Only bins that all three tables actually span are scored. Where the delayed
    table stops short of the prompt one — JEFF-4.0 U-235 tabulates nu_d to
    20 MeV and nu_p to 30 MeV — there is no sum rule to check, and counting the
    missing nu_d as a violation would report a 1e-3 discrepancy that is really
    just absent data. Those bins come back as NaN and are counted separately.

    Returns ``None`` when the family is incomplete (nothing is derived, so
    nothing is repaired). Otherwise a dict with the 1/E-weighted relative
    residual per MF31 bin, its maximum, and the pointwise maximum.
    """
    if not set(_NUBAR_MTS).issubset(set(int(mt) for mt in sections)):
        return None

    tables = {mt: _nubar_as_tabulated(sections[mt]) for mt in _NUBAR_MTS}
    if any(e.size == 0 for e, _ in tables.values()):
        return None

    edges = np.asarray(bins, dtype=float)
    lo = max(float(e[0]) for e, _ in tables.values())
    hi = min(float(e[-1]) for e, _ in tables.values())
    covered = (edges[:-1] >= lo) & (edges[1:] <= hi)

    bar = {mt: _bin_average_xs(e, n, edges) for mt, (e, n) in tables.items()}
    total = bar[NUBAR_TOTAL_MT]
    resid = np.abs(total - sum(bar[mt] for mt in NUBAR_COMPONENT_MTS))
    per_bin = np.full(total.shape, np.nan)
    ok = covered & (total > 0)
    per_bin[ok] = resid[ok] / total[ok]

    # Pointwise, on the union of the three grids restricted to the common range.
    union_e = np.unique(np.concatenate([e for e, _ in tables.values()]))
    union_e = union_e[(union_e >= lo) & (union_e <= hi)]
    if union_e.size:
        vals = {mt: np.interp(union_e, e, n) for mt, (e, n) in tables.items()}
        t = vals[NUBAR_TOTAL_MT]
        pw = np.abs(t - sum(vals[mt] for mt in NUBAR_COMPONENT_MTS))
        max_pointwise = float(np.max(np.where(t > 0, pw / np.where(t > 0, t, 1.0), 0.0)))
    else:
        max_pointwise = 0.0

    scored = np.isfinite(per_bin)
    return {
        "per_bin_rel": per_bin,
        "max_bin_rel": float(np.nanmax(per_bin)) if scored.any() else 0.0,
        "argmax_bin": int(np.nanargmax(per_bin)) if scored.any() else -1,
        "max_pointwise_rel": max_pointwise,
        "n_bins_scored": int(scored.sum()),
        "n_bins_uncovered": int((~scored).sum()),
        "common_range_eV": (lo, hi),
    }


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
    plus single interpolated points at the interior MF31 bin edges and inside
    under-resolved bins). The derived total is then built on the union of the
    perturbed members' grids and evaluated as nu_d + nu_p; because no member
    carries duplicate energies, that union sum is exact (a piecewise-linear
    identity), so the sum rule holds to machine precision without bloating the
    sparse delayed-nu-bar table. Handles all MF31 coverage patterns with one
    rule (mirrors ``UQ.sandwich`` ``nubar_mode``):

      * **components carry covariance** (455 and/or 456): each perturbed by its
        own factor; an unperturbed component stays at baseline; total derived.
      * **only total (452) carries covariance**: total's factor rides onto both
        components, so all three scale together and the sum rule still holds.
      * **all three carry covariance**: components perturbed independently, the
        redundant total is recomputed (its own factor block is discarded).

    Enforcing the sum rule is deliberate: the ENDF-6 manual (§1.2.2) requires
    nu_p + nu_d to be consistent with the total whenever MT=455 is present, and
    requires MT=452 to be LNU=2 in that case. Evaluations do not always comply
    exactly, so rebuilding the total also repairs whatever residual the input
    file carries — a change to the central value that the perturbation did not
    ask for. :func:`sum_rule_residual` measures it so the repair is on record
    rather than silent; for the JEFF-4.0 actinides it is ≤0.15 of the MF31 1σ.

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

    Raises
    ------
    ValueError
        If ``sections`` is empty, if a factor block targets an MT with no MF1
        section, or if a contributor to the derived member carries no usable
        table (an LNU=1 section that was never perturbed) — dropping it would
        silently erase that component from the total.
    """
    bins_arr = np.asarray(bins, dtype=float)
    n_groups = bins_arr.size - 1
    derived = NUBAR_TOTAL_MT if derived_mt is None else int(derived_mt)
    present = set(int(mt) for mt in sections)

    if not present:
        raise ValueError(
            "no MF1 nu-bar sections to perturb (MT 452/455/456 all absent); "
            "writing the tape unchanged would report a perturbation that "
            "never happened"
        )
    orphan_blocks = sorted(int(mt) for mt in factor_blocks if int(mt) not in present)
    if orphan_blocks:
        raise ValueError(
            f"MF31 covariance for MT(s) {orphan_blocks} but no matching MF1 "
            f"nu-bar section (present: {sorted(present)})"
        )

    components_present = [mt for mt in NUBAR_COMPONENT_MTS if mt in present]
    # The family is derivable only when EVERY contributor to the derived member
    # is present. The manual (§1.3.2) makes MT=456 mandatory whenever MT=455
    # exists, so an incomplete family means a malformed tape — deriving anyway
    # would write, for a {452, 455} file, nu_total := nu_delayed.
    if derived == NUBAR_TOTAL_MT:
        can_derive = (NUBAR_TOTAL_MT in present
                      and set(NUBAR_COMPONENT_MTS).issubset(present))
    else:
        required = {NUBAR_TOTAL_MT} | (set(NUBAR_COMPONENT_MTS) - {derived})
        can_derive = derived in present and required.issubset(present)

    # Degenerate case: no derivable family (e.g. only the total is present, or
    # the tape is missing a mandatory member) → perturb each present member on
    # its own grid directly and leave the sum rule alone.
    if not can_derive:
        if logger is not None and len(present) > 1:
            logger.warning(
                f"  [MF31] incomplete nu-bar family {sorted(present)}: MT{derived} "
                f"cannot be derived, perturbing each member directly."
            )
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

        tables: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for mt in contributors:
            sec = out.get(mt, sections.get(mt))
            e, n = (_nubar_as_tabulated(sec) if sec is not None
                    else (np.asarray([]), np.asarray([])))
            if e.size == 0:
                # An LNU=1 contributor that was never perturbed has no table.
                # Skipping it used to drop that component from the total
                # outright (measured: −0.55 % on nu-bar when the delayed part
                # vanished), so refuse instead.
                raise ValueError(
                    f"MT{derived} is derived from MT{mt}, but MT{mt} carries no "
                    f"usable nu-bar table (LNU="
                    f"{getattr(sec, 'lnu', None) if sec is not None else None}); "
                    f"perturb it too, or set derived_mt to a member that is"
                )
            tables[mt] = (e, n)

        union_e = np.unique(np.concatenate([e for e, _ in tables.values()]))
        nu_derived = np.zeros_like(union_e)
        for mt, (e, n) in tables.items():
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
