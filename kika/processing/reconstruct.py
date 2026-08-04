"""
Resonance reconstruction on canonical types.

Reconstructs pointwise cross sections from resonance parameters,
returning ``CrossSection`` objects.  Operates entirely on
``kika.nuclear_data`` types — no ENDF I/O dependency.

Supported formalisms
--------------------
- SLBW  (Single-Level Breit-Wigner)
- MLBW  (Multi-Level Breit-Wigner)
- RM    (Reich-Moore, eliminated capture channel)
"""

from typing import Dict, List, Optional
import warnings

import numpy as np

from kika.nuclear_data.cross_section import CrossSection
from kika.nuclear_data.resonance_parameters import ResonanceParameters
from .resonance_formulas import (
    slbw_cross_sections,
    mlbw_cross_sections,
    reich_moore_cross_sections,
)
from .linearization import linearize

# MT numbers for reaction channels
MT_TOTAL = 1
MT_ELASTIC = 2
MT_FISSION = 18
MT_CAPTURE = 102

_DISPATCH = {
    "SLBW": slbw_cross_sections,
    "MLBW": mlbw_cross_sections,
    "RM": reich_moore_cross_sections,
}


def reconstruct(
    resonance_params: List[ResonanceParameters],
    background: Optional[Dict[int, CrossSection]] = None,
    tolerance: float = 1e-3,
) -> Dict[int, CrossSection]:
    """Reconstruct pointwise cross sections from resonance parameters.

    Parameters
    ----------
    resonance_params : list of ResonanceParameters
        One entry per (isotope, energy range) — as returned by
        ``ResonanceParameters.from_endf()``.
    background : dict mapping MT → CrossSection, optional
        Background (smooth) cross sections.  If provided, added to
        the reconstructed resonance contribution inside the resonance
        region and preserved outside it.
    tolerance : float
        Linearization tolerance (default 0.1 %).

    Returns
    -------
    Dict[int, CrossSection]
        Reconstructed pointwise cross sections keyed by MT number.
        Typically MT 1 (total), 2 (elastic), 18 (fission), 102 (capture).
    """
    if not resonance_params:
        warnings.warn("Empty resonance_params list — nothing to reconstruct")
        return {}

    # Use the first entry for ZA / AWR (they should all match)
    za = resonance_params[0].nuclide_id
    awr = resonance_params[0].metadata.get("awr", 0.0)
    mat = resonance_params[0].metadata.get("mat")

    from kika.nuclear_data.resonance_parameters import UnresolvedResonanceParameters

    # Accumulate callable σ(E) contributions from all ranges
    funcs = []
    res_E_lo = np.inf
    res_E_hi = 0.0
    seed_energies: List[float] = []
    urr_entries = []

    for rp in resonance_params:
        if isinstance(rp, UnresolvedResonanceParameters):
            urr_entries.append(rp)
            continue

        formula = _DISPATCH.get(rp.formalism)
        if formula is None:
            warnings.warn(
                f"Unsupported formalism {rp.formalism!r}, skipping "
                f"range [{rp.energy_range[0]:.6g}, {rp.energy_range[1]:.6g}] eV"
            )
            continue

        el, eh = rp.energy_range
        res_E_lo = min(res_E_lo, el)
        res_E_hi = max(res_E_hi, eh)
        abundance = rp.metadata.get("abundance", 1.0)

        # Collect resonance energies for seeding the adaptive grid
        for lg in rp.l_groups:
            for rec in lg.resonances:
                if el <= rec.energy <= eh:
                    seed_energies.append(rec.energy)
                    # Add points near the resonance (± a few widths)
                    if rp.formalism in ("SLBW", "MLBW"):
                        width = abs(rec.c3) if abs(rec.c3) > 0 else abs(rec.c4)
                    else:
                        width = abs(rec.c3) + abs(rec.c4)
                    if width > 0:
                        for factor in [0.1, 0.5, 1.0, 2.0, 5.0]:
                            e_off = rec.energy + factor * width
                            if el < e_off < eh:
                                seed_energies.append(e_off)
                            e_off = rec.energy - factor * width
                            if el < e_off < eh:
                                seed_energies.append(e_off)

        # Build callable for this range
        def _make_sigma_func(rp, formula, el, eh):
            def sigma_func(E_arr):
                E_arr = np.asarray(E_arr, dtype=float)
                sig_el = np.zeros(len(E_arr), dtype=float)
                sig_cap = np.zeros(len(E_arr), dtype=float)
                sig_fis = np.zeros(len(E_arr), dtype=float)

                mask = (E_arr >= el) & (E_arr <= eh)
                if not np.any(mask):
                    return sig_el, sig_cap, sig_fis

                E_in = E_arr[mask]
                for lg in rp.l_groups:
                    l_val = lg.l
                    awr_l = lg.awri if lg.awri > 0 else rp.metadata.get("awr", 0.0)
                    # Per-l scattering radius where the evaluation gives one
                    # (ENDF's APL); the range-level AP otherwise. Every l used
                    # to get the range value, which for JEFF-4.0 Fe-56 meant
                    # the l=1 hard-sphere phase shift was computed from
                    # AP = 0.5444 fm instead of APL = 0.5002 fm.
                    #
                    # An energy-dependent AP table (NRO=1) still wins, inside
                    # the formula functions: it is a property of the range and
                    # ENDF gives no per-l version of it, so overriding it with
                    # a scalar would change NRO=1 tapes for the worse.
                    ap_l = lg.ap if lg.ap is not None else rp.scattering_radius
                    sel, scap, sfis = formula(
                        E_in, lg.resonances, l_val, rp.spin,
                        ap_l, awr_l,
                        ap_table=rp.scattering_radius_table,
                    )
                    sig_el[mask] += sel
                    sig_cap[mask] += scap
                    sig_fis[mask] += sfis

                return sig_el, sig_cap, sig_fis

            return sigma_func

        funcs.append((abundance, _make_sigma_func(rp, formula, el, eh)))

    if not funcs and not urr_entries:
        warnings.warn("No supported resonance ranges found")
        return {}

    # If we have URR but no resolved ranges, create a minimal grid
    if not funcs:
        first_urr = urr_entries[0]
        res_E_lo = first_urr.energy_range[0]
        res_E_hi = first_urr.energy_range[1]
        energy_grid = np.geomspace(max(res_E_lo, 1e-5), res_E_hi, 200)
        sig_el = np.zeros(len(energy_grid))
        sig_cap = np.zeros(len(energy_grid))
        sig_fis = np.zeros(len(energy_grid))
        sig_tot = np.zeros(len(energy_grid))
    else:
        # Combined σ(E) across all ranges / isotopes
        def total_sigma(E_arr):
            E_arr = np.asarray(E_arr, dtype=float)
            sig_el = np.zeros(len(E_arr), dtype=float)
            sig_cap = np.zeros(len(E_arr), dtype=float)
            sig_fis = np.zeros(len(E_arr), dtype=float)
            for abn, fn in funcs:
                sel, scap, sfis = fn(E_arr)
                sig_el += abn * sel
                sig_cap += abn * scap
                sig_fis += abn * sfis
            return sig_el, sig_cap, sig_fis

        if res_E_lo >= res_E_hi:
            warnings.warn("Invalid resonance energy range")
            return {}

        res_E_lo = max(res_E_lo, 1e-5)

        seed = np.unique(np.array(seed_energies))
        seed = seed[(seed > res_E_lo) & (seed < res_E_hi)]

        def sigma_total_for_linearization(E_arr):
            sel, scap, sfis = total_sigma(E_arr)
            return sel + scap + sfis

        energy_grid = linearize(
            sigma_total_for_linearization,
            res_E_lo, res_E_hi,
            tol=tolerance,
            initial_points=seed,
        )

        sig_el, sig_cap, sig_fis = total_sigma(energy_grid)
        sig_tot = sig_el + sig_cap + sig_fis

        # Add background cross sections if provided
        if background is not None:
            _add_background(energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                            background, res_E_lo, res_E_hi)

            energy_grid, sig_el, sig_cap, sig_fis, sig_tot = _extend_to_full_range(
                energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                background, res_E_lo, res_E_hi,
            )

    # --- URR: unresolved resonance region ---
    from .urr_formulas import urr_cross_sections

    for urr_rp in urr_entries:
        if urr_rp.lssf != 0:
            continue  # LSSF=1: MF3 already correct, skip

        el, eh = urr_rp.energy_range
        abn = urr_rp.metadata.get("abundance", 1.0)
        awr_urr = urr_rp.metadata.get("awr", awr)

        # URR cross sections are smooth — use log-spaced grid
        urr_E = np.geomspace(max(el, 1e-5), eh, 200)

        # Interpolate energy-dependent parameters to the evaluation grid
        urr_l_groups = _interpolate_urr_params(urr_rp, urr_E)

        sel, scap, sfis = urr_cross_sections(
            urr_E, urr_l_groups, urr_rp.spin, urr_rp.scattering_radius, awr_urr
        )

        # Merge URR grid with existing grid
        mask_before = energy_grid < el
        mask_after = energy_grid > eh

        combined_E = np.concatenate([energy_grid[mask_before], urr_E, energy_grid[mask_after]])
        combined_el = np.concatenate([sig_el[mask_before], abn * sel, sig_el[mask_after]])
        combined_cap = np.concatenate([sig_cap[mask_before], abn * scap, sig_cap[mask_after]])
        combined_fis = np.concatenate([sig_fis[mask_before], abn * sfis, sig_fis[mask_after]])

        order = np.argsort(combined_E)
        energy_grid = combined_E[order]
        sig_el = combined_el[order]
        sig_cap = combined_cap[order]
        sig_fis = combined_fis[order]
        sig_tot = sig_el + sig_cap + sig_fis

    # Package as CrossSection objects
    result: Dict[int, CrossSection] = {}

    for mt_num, sigma_arr in [
        (MT_TOTAL, sig_tot),
        (MT_ELASTIC, sig_el),
        (MT_CAPTURE, sig_cap),
        (MT_FISSION, sig_fis),
    ]:
        if mt_num == MT_FISSION and np.all(sigma_arr < 1e-30):
            continue

        result[mt_num] = CrossSection(
            energies=energy_grid.copy(),
            values=sigma_arr.copy(),
            reaction=mt_num,
            nuclide_id=za,
            interpolation="linlin",
            metadata={"mat": mat, "awr": awr, "qm": 0.0, "qi": 0.0, "lr": 0,
                       "interpolation_regions": [(len(energy_grid), 2)]},
        )

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _add_background(energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                    background, res_E_lo, res_E_hi):
    """Add background cross sections (modifies arrays in-place)."""
    from kika.endf.utils import interpolate_1d_endf

    mt_map = {
        MT_ELASTIC: sig_el,
        MT_CAPTURE: sig_cap,
        MT_FISSION: sig_fis,
    }

    for mt_num, sigma_arr in mt_map.items():
        if mt_num in background:
            xs = background[mt_num]
            interp_regions = xs.metadata.get("interpolation_regions",
                                             [(len(xs.energies), 2)])
            bg = interpolate_1d_endf(
                xs.energies, xs.values, interp_regions,
                energy_grid, out_of_range="zero",
            )
            bg = np.asarray(bg)
            sigma_arr += bg

    sig_tot[:] = sig_el + sig_cap + sig_fis


def _extend_to_full_range(energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                           background, res_E_lo, res_E_hi):
    """Extend PENDF beyond resonance region using background data."""
    from kika.endf.utils import interpolate_1d_endf

    all_extra_E = set()
    for mt_num in [MT_ELASTIC, MT_CAPTURE, MT_FISSION, MT_TOTAL]:
        if mt_num in background:
            xs = background[mt_num]
            below = xs.energies[xs.energies < res_E_lo]
            above = xs.energies[xs.energies > res_E_hi]
            all_extra_E.update(below)
            all_extra_E.update(above)

    if not all_extra_E:
        return energy_grid, sig_el, sig_cap, sig_fis, sig_tot

    extra_E = np.sort(np.array(list(all_extra_E)))
    extra_sig_el = np.zeros(len(extra_E))
    extra_sig_cap = np.zeros(len(extra_E))
    extra_sig_fis = np.zeros(len(extra_E))

    for mt_num, arr in [(MT_ELASTIC, extra_sig_el),
                        (MT_CAPTURE, extra_sig_cap),
                        (MT_FISSION, extra_sig_fis)]:
        if mt_num in background:
            xs = background[mt_num]
            interp_regions = xs.metadata.get("interpolation_regions",
                                             [(len(xs.energies), 2)])
            arr[:] = np.asarray(interpolate_1d_endf(
                xs.energies, xs.values, interp_regions,
                extra_E, out_of_range="zero",
            ))

    if MT_TOTAL in background:
        xs = background[MT_TOTAL]
        interp_regions = xs.metadata.get("interpolation_regions",
                                         [(len(xs.energies), 2)])
        extra_sig_tot = np.asarray(interpolate_1d_endf(
            xs.energies, xs.values, interp_regions,
            extra_E, out_of_range="zero",
        ))
    else:
        extra_sig_tot = extra_sig_el + extra_sig_cap + extra_sig_fis

    full_E = np.concatenate([extra_E, energy_grid])
    full_sig_el = np.concatenate([extra_sig_el, sig_el])
    full_sig_cap = np.concatenate([extra_sig_cap, sig_cap])
    full_sig_fis = np.concatenate([extra_sig_fis, sig_fis])
    full_sig_tot = np.concatenate([extra_sig_tot, sig_tot])

    order = np.argsort(full_E, kind="mergesort")
    full_E = full_E[order]
    full_sig_el = full_sig_el[order]
    full_sig_cap = full_sig_cap[order]
    full_sig_fis = full_sig_fis[order]
    full_sig_tot = full_sig_tot[order]

    unique_mask = np.diff(full_E, prepend=-1.0) > 0.0
    return (full_E[unique_mask], full_sig_el[unique_mask],
            full_sig_cap[unique_mask], full_sig_fis[unique_mask],
            full_sig_tot[unique_mask])


def _interpolate_urr_params(rp, E_eval):
    """Interpolate energy-dependent URR parameters to evaluation grid.

    For Case A (energy_grid is None): parameters are already scalars.
    For Cases B, C: linearly interpolate tabulated parameters to E_eval.

    Returns a list of URR_LGroup objects with parameters as arrays of len(E_eval).
    """
    from kika.nuclear_data.resonance_parameters import URR_LGroup, URR_JGroup

    if rp.energy_grid is None:
        return rp.l_groups

    E_tab = rp.energy_grid
    result = []
    for lg in rp.l_groups:
        new_jgs = []
        for jg in lg.j_groups:
            new_jg = URR_JGroup(
                j=jg.j, amun=jg.amun,
                d=np.interp(E_eval, E_tab, np.atleast_1d(jg.d)),
                gn0=np.interp(E_eval, E_tab, np.atleast_1d(jg.gn0)),
                gg=np.interp(E_eval, E_tab, np.atleast_1d(jg.gg)),
                gf=np.interp(E_eval, E_tab, np.atleast_1d(jg.gf)),
                gx=np.interp(E_eval, E_tab, np.atleast_1d(jg.gx)),
            )
            new_jgs.append(new_jg)
        result.append(URR_LGroup(awri=lg.awri, l=lg.l, j_groups=new_jgs))
    return result
