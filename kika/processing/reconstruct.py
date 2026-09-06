"""
Resonance reconstruction on the GNDS model.

Reconstructs pointwise cross sections from a
:class:`~kika.nuclear_data.model.resonances.Resonances` node, returning
``XYs1d`` forms. Operates entirely on ``kika.nuclear_data.model`` types — no
ENDF I/O dependency, and no dependency on the flat classes either.

Supported formalisms
--------------------
- ``BreitWigner``, single-level  (ENDF LRF=1)
- ``BreitWigner``, multi-level   (ENDF LRF=2)
- ``RMatrix``, Reich-Moore       (ENDF LRF=3, eliminated capture channel)
- ``TabulatedWidths``            (the unresolved region)

**Every model import in this module is deferred to call time.** ``import kika``
loads ``kika.processing`` — and therefore this module — but does *not* load
``kika.nuclear_data.model``. Hoisting one of the imports below to module scope
would wake ~3 400 lines of model on every ``import kika``: for the cluster
pipeline, the desktop app and every notebook at once, none of which asked to
reconstruct anything. ``test_layering.py`` asserts it statically, with a
self-test proving the check bites, because the cost is invisible in every
functional test.

**Not the ENDF entry point.** ``kika.endf.processing.reconstruct`` takes an
``MF2MT151`` and gives back ``MF3MT`` sections; it decodes to the model, calls
this, and encodes back. That is the only place ENDF types belong.
"""

from typing import Dict, List, Optional
import warnings

import numpy as np

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

#: §9.1.1's style for a cross section reconstructed from resonance parameters.
#: The forms this module returns carry it as their label, so a suite that stores
#: them can say where they came from instead of presenting them as evaluated.
RECONSTRUCTED_LABEL = "recon"


def _targetSpin(formalism, region) -> float:
    """The target spin I, from the formalism's own ``PoPs`` (GNDS §19.3.1).

    It enters ``g_J = (2J+1) / (2(2I+1))``, so this is arithmetic and not a
    label — which is why a missing spin raises instead of defaulting to zero.
    Zero is a perfectly ordinary spin (every even-even target has it) and would
    be indistinguishable from the default.
    """
    pops = getattr(formalism, "PoPs", None)
    particles = list(getattr(pops, "particles", {}).values()) if pops is not None else []
    spins = [p.spin for p in particles if getattr(p, "spin", None) is not None]
    if len(spins) != 1:
        raise ValueError(
            f"the resonance region [{region[0]:.6g}, {region[1]:.6g}] eV carries "
            f"{len(spins)} target spins in its PoPs; reconstruction needs exactly "
            f"one, because the statistical factor g_J is computed from it. A "
            f"region decoded from ENDF gets it from SPI."
        )
    return spins[0].value


def _toEndfRadius(value):
    """A model radius (fm) → ENDF's units. **The one boundary in this module.**

    ``kika.nuclear_data.model`` states every radius in fm since 2026-08-20
    (``MODEL_RADIUS_UNIT``), and :mod:`kika.processing.resonance_formulas` works
    in ENDF's 10^-12 cm throughout — every one of its signatures says so. The
    conversion lives here, at the edge, rather than inside the formulas: the
    formulas are what the reconstruction goldens are pinned to, and changing a
    unit underneath them would move numbers the thesis depends on. Four call
    sites, and they are every path a radius takes into a formula: the NRO=1
    table, the range's AP, the per-l APL, and the URR's AP.

    The import is deferred like every other model import in this module — see
    the module docstring, and ``test_layering.py``, which asserts it statically.
    """
    from kika.nuclear_data.model.resonances import radiusToEndf

    return radiusToEndf(value)


def _radiusTable(resonances, resolvedCount: int):
    """``(energies, values, (NBT, INT) pairs)`` for NRO=1, or ``None``.

    **In ENDF's radius units, not the model's.** Everything downstream of here
    is :mod:`kika.processing.resonance_formulas`, which works in ENDF units
    throughout and says so in every signature (``ap : float — Scattering radius
    (ENDF units: 10^{-12} cm)``). The model is in fm since 2026-08-20, so the
    conversion happens **at this boundary** rather than inside the formulas.
    That is the whole reason the reconstruction did not move when the model's
    unit changed, and ``test_numeric_goldens`` is what says it did not.

    ENDF writes NRO per *range* while the model holds one scattering radius for
    the file, so a file that tabulates the radius on one range and not on
    another cannot say which range it belonged to. That case is refused rather
    than guessed: applying the table to every range would silently recompute
    the hard-sphere phase shift of a range that never asked for one. No
    evaluation on this machine has NRO=1 at all.
    """
    radius = getattr(resonances, "scatteringRadius", None)
    if radius is None or not radius.isEnergyDependent:
        return None
    if resolvedCount > 1:
        raise ValueError(
            "this evaluation has an energy-dependent scattering radius and "
            f"{resolvedCount} resolved ranges. The model holds one radius for "
            "the file, so it cannot say which range tabulated it, and applying "
            "it to all of them would change the phase shift of ranges that "
            "declared a constant."
        )
    if radius.interpolation is None:
        raise ValueError(
            "the energy-dependent scattering radius carries no interpolation "
            "regions, so the values between its points mean nothing definite"
        )
    return (np.asarray(radius.energies, dtype=float),
            _toEndfRadius(np.asarray(radius.values, dtype=float)),
            list(radius.interpolation))


def _resolvedGroups(formalism):
    """``(spin groups, seed-width function)`` for whichever formalism this is.

    The seed width is what the adaptive grid is pre-loaded around, and it is
    per formalism because the widths are: Breit-Wigner seeds on the total width
    (falling back to the neutron width when a file leaves GT at zero),
    Reich-Moore has no total width and seeds on neutron + capture. Both are the
    rules the record-position version used, expressed in names.
    """
    from kika.nuclear_data.model.resonances import BreitWigner

    if isinstance(formalism, BreitWigner):
        groups = formalism.resonanceParameters.spinGroups
        def energiesAndWidths(group):
            for r in group.resonances:
                yield r.energy, (abs(r.totalWidth) if abs(r.totalWidth) > 0
                                 else abs(r.neutronWidth))
        return groups, energiesAndWidths

    from .resonance_formulas import _channelColumns, _width

    groups = formalism.spinGroups
    def energiesAndWidths(group):
        iN, iG, _, _ = _channelColumns(group)
        for index, energy in enumerate(group.energies):
            row = group.widths[index]
            yield energy, abs(_width(row, iN)) + abs(_width(row, iG))
    return groups, energiesAndWidths


def _formula(formalism):
    """The formula function for a resolved formalism, or ``None`` if unsupported.

    **The approximation name is not the discriminator, and assuming it was is a
    bug this caught.** ENDF's LRF=7 with KRM=3 *is* the Reich-Moore
    approximation and the decoder labels it so — correctly. But an LRF=3 range
    and an LRF=7 range are two different parameterisations of it: LRF=3 blocks
    by l and writes a J on every resonance record, over four fixed channels;
    LRF=7 gives the spin group one J and as many channels as the evaluator
    declared, five of them for Fe-57. ``reich_moore_cross_sections`` implements
    the first. Dispatching on the name alone handed it the second and it raised
    an ``IndexError`` reaching for a per-resonance spin that is not there.

    So the test is structural: per-resonance spins, and channels labelled from
    the Reich-Moore set. Anything else is declined by name in a warning, which
    is where LRF=7 sits until P1b implements it.
    """
    from kika.nuclear_data.model.resonances import (BreitWigner,
                                                    BreitWignerApproximation,
                                                    RMatrix)

    from .resonance_formulas import _RM_CHANNELS

    if isinstance(formalism, BreitWigner):
        if formalism.approximation is BreitWignerApproximation.singleLevel:
            return slbw_cross_sections
        return mlbw_cross_sections

    if isinstance(formalism, RMatrix) and formalism.approximation == "ReichMoore":
        blocked = all(
            len(group.spins) == len(group.energies)
            and all(channel.label in _RM_CHANNELS for channel in group.channels)
            for group in formalism.spinGroups
        )
        return reich_moore_cross_sections if blocked else None

    return None


def _formalismName(formalism) -> str:
    """Enough to tell the two Reich-Moore parameterisations apart in a warning.

    "RMatrix/ReichMoore" alone would name LRF=3 and LRF=7 identically, so the
    channel count goes in: it is what actually differs, and it is the reason
    the second is declined.
    """
    approximation = getattr(formalism, "approximation", None)
    name = f"{type(formalism).__name__}/{getattr(approximation, 'value', approximation)}"
    groups = getattr(formalism, "spinGroups", None) or []
    channels = {len(getattr(group, "channels", ())) for group in groups}
    if channels:
        return f"{name} over {sorted(channels)} channels per spin group"
    return name


def reconstruct(
    resonances,
    background: Optional[Dict[int, object]] = None,
    tolerance: float = 1e-3,
    atomicWeightRatio: Optional[float] = None,
) -> Dict[int, object]:
    """Reconstruct pointwise cross sections from a ``Resonances`` node.

    Parameters
    ----------
    resonances : model.resonances.Resonances
        The evaluation's resonance node — resolved regions, the unresolved
        region, and the scattering radius. As returned by
        ``kika.endf.model_adapter.decodeMF2MT151``.
    background : dict mapping MT → a tabulated 1d form, optional
        Background (smooth) cross sections, as ``XYs1d`` or ``Regions1d``.
        If provided, added to the reconstructed resonance contribution inside
        the resonance region and preserved outside it.
    tolerance : float
        Linearization tolerance (default 0.1 %).
    atomicWeightRatio : float, optional
        The evaluation's AWR, used only for a spin group that declares no AWRI
        of its own. ENDF writes AWRI on every l-block and the file-level AWR is
        the fallback; GNDS would take it from the target's mass in ``PoPs``,
        which the local ``PoPs`` on a resonance formalism does not carry.

    Returns
    -------
    Dict[int, XYs1d]
        Reconstructed pointwise cross sections keyed by MT number, labelled
        ``recon``. Typically MT 1 (total), 2 (elastic), 18 (fission),
        102 (capture).
    """
    from kika.nuclear_data.model.axes import crossSectionAxes
    from kika.nuclear_data.model.functions import XYs1d

    resolved = list(getattr(resonances, "resolved", []) or [])
    unresolved = getattr(resonances, "unresolved", None)

    if not resolved and unresolved is None:
        warnings.warn("The resonances node holds no region — nothing to reconstruct")
        return {}

    ap_table = _radiusTable(resonances, len(resolved))

    # Accumulate callable σ(E) contributions from all ranges
    funcs = []
    res_E_lo = np.inf
    res_E_hi = 0.0
    seed_energies: List[float] = []

    for region in resolved:
        formalism = region.formalism
        formula = _formula(formalism)
        if formula is None:
            # Naming the alternative is the point of the message, not politeness.
            # This module implements three formalisms and is not going to grow a
            # fourth -- a decision, recorded in the GNDS roadmap's "what will not
            # be built": for anything it declines, NJOY's RECONR is the answer,
            # and kika already calls it.
            warnings.warn(
                f"Unsupported formalism {_formalismName(formalism)!r}, skipping "
                f"range [{region.domainMin:.6g}, {region.domainMax:.6g}] eV. "
                f"This reconstructor covers SLBW, MLBW and ENDF's LRF=3 "
                f"Reich-Moore only; use kika.processing.njoy_reconstruct, which "
                f"runs RECONR and handles every formalism ENDF defines."
            )
            continue

        el, eh = region.domainMin, region.domainMax
        res_E_lo = min(res_E_lo, el)
        res_E_hi = max(res_E_hi, eh)

        spi = _targetSpin(formalism, (el, eh))
        # ENDF units from here down -- see `_radiusTable`.
        rangeRadius = _toEndfRadius(formalism.scatteringRadius)
        groups, energiesAndWidths = _resolvedGroups(formalism)

        # Collect resonance energies for seeding the adaptive grid
        for group in groups:
            for energy, width in energiesAndWidths(group):
                if el <= energy <= eh:
                    seed_energies.append(energy)
                    # Add points near the resonance (± a few widths)
                    if width > 0:
                        for factor in [0.1, 0.5, 1.0, 2.0, 5.0]:
                            e_off = energy + factor * width
                            if el < e_off < eh:
                                seed_energies.append(e_off)
                            e_off = energy - factor * width
                            if el < e_off < eh:
                                seed_energies.append(e_off)

        # Build callable for this range
        def _make_sigma_func(groups, formula, el, eh, spi, rangeRadius):
            def sigma_func(E_arr):
                E_arr = np.asarray(E_arr, dtype=float)
                sig_el = np.zeros(len(E_arr), dtype=float)
                sig_cap = np.zeros(len(E_arr), dtype=float)
                sig_fis = np.zeros(len(E_arr), dtype=float)

                mask = (E_arr >= el) & (E_arr <= eh)
                if not np.any(mask):
                    return sig_el, sig_cap, sig_fis

                E_in = E_arr[mask]
                for group in groups:
                    awri = group.atomicWeightRatio
                    awr_l = awri if awri and awri > 0 else atomicWeightRatio
                    if awr_l is None:
                        raise ValueError(
                            "a resonance spin group declares no atomic weight "
                            "ratio and none was supplied; the wave number "
                            "cannot be computed without one"
                        )
                    # Per-l scattering radius where the evaluation gives one
                    # (ENDF's APL, which the model puts on the channel); the
                    # range's own AP otherwise. Every l used to get the range
                    # value, which for JEFF-4.0 Fe-56 meant the l=1 hard-sphere
                    # phase shift was computed from AP = 5.444 fm instead of
                    # APL = 5.002 fm. (Both are 0.5444 and 0.5002 in the ENDF
                    # units this line is in; the model states them in fm.)
                    #
                    # An energy-dependent AP table (NRO=1) still wins, inside
                    # the formula functions: it is a property of the range and
                    # ENDF gives no per-l version of it, so overriding it with
                    # a scalar would change NRO=1 tapes for the worse.
                    ap_l = _groupRadius(group)
                    sel, scap, sfis = formula(
                        E_in, group, spi,
                        ap_l if ap_l is not None else rangeRadius, awr_l,
                        ap_table=ap_table,
                    )
                    sig_el[mask] += sel
                    sig_cap[mask] += scap
                    sig_fis[mask] += sfis

                return sig_el, sig_cap, sig_fis

            return sigma_func

        funcs.append(_make_sigma_func(groups, formula, el, eh, spi, rangeRadius))

    urr_entries = [unresolved] if unresolved is not None else []

    if not funcs and not urr_entries:
        warnings.warn("No supported resonance ranges found")
        return {}

    # If we have URR but no resolved ranges, create a minimal grid
    if not funcs:
        first_urr = urr_entries[0]
        res_E_lo = first_urr.domainMin
        res_E_hi = first_urr.domainMax
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
            for fn in funcs:
                sel, scap, sfis = fn(E_arr)
                sig_el += sel
                sig_cap += scap
                sig_fis += sfis
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

    for region in urr_entries:
        widths = region.tabulatedWidths
        if widths is None or widths.selfShieldingOnly:
            continue  # LSSF=1: MF3 already correct, skip

        el, eh = region.domainMin, region.domainMax

        # URR cross sections are smooth — use log-spaced grid
        urr_E = np.geomspace(max(el, 1e-5), eh, 200)

        # Interpolate energy-dependent parameters to the evaluation grid
        urr_groups = _interpolate_urr_params(widths, urr_E)

        sel, scap, sfis = urr_cross_sections(
            urr_E, urr_groups, _targetSpin(widths, (el, eh)),
            _toEndfRadius(widths.scatteringRadius), atomicWeightRatio,
        )

        # Merge URR grid with existing grid
        mask_before = energy_grid < el
        mask_after = energy_grid > eh

        combined_E = np.concatenate([energy_grid[mask_before], urr_E, energy_grid[mask_after]])
        combined_el = np.concatenate([sig_el[mask_before], sel, sig_el[mask_after]])
        combined_cap = np.concatenate([sig_cap[mask_before], scap, sig_cap[mask_after]])
        combined_fis = np.concatenate([sig_fis[mask_before], sfis, sig_fis[mask_after]])

        order = np.argsort(combined_E)
        energy_grid = combined_E[order]
        sig_el = combined_el[order]
        sig_cap = combined_cap[order]
        sig_fis = combined_fis[order]
        sig_tot = sig_el + sig_cap + sig_fis

    # Package as model forms. One interpolation law for the whole table, which
    # is what an XYs1d is: the grid is the linearization's own, so lin-lin is a
    # fact about it rather than an assumption about it.
    result: Dict[int, object] = {}

    for mt_num, sigma_arr in [
        (MT_TOTAL, sig_tot),
        (MT_ELASTIC, sig_el),
        (MT_CAPTURE, sig_cap),
        (MT_FISSION, sig_fis),
    ]:
        if mt_num == MT_FISSION and np.all(sigma_arr < 1e-30):
            continue

        result[mt_num] = XYs1d(
            xs=energy_grid.copy(),
            ys=sigma_arr.copy(),
            axes=crossSectionAxes(),
            label=RECONSTRUCTED_LABEL,
        )

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _add_background(energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                    background, res_E_lo, res_E_hi):
    """Add background cross sections (modifies arrays in-place)."""
    from .interpolation import interpolate_1d

    mt_map = {
        MT_ELASTIC: sig_el,
        MT_CAPTURE: sig_cap,
        MT_FISSION: sig_fis,
    }

    for mt_num, sigma_arr in mt_map.items():
        if mt_num in background:
            energies, values, interp_regions = _tabulated(background[mt_num])
            bg = interpolate_1d(
                energies, values, interp_regions,
                energy_grid, out_of_range="zero",
            )
            bg = np.asarray(bg)
            sigma_arr += bg

    sig_tot[:] = sig_el + sig_cap + sig_fis


def _extend_to_full_range(energy_grid, sig_el, sig_cap, sig_fis, sig_tot,
                           background, res_E_lo, res_E_hi):
    """Extend PENDF beyond resonance region using background data."""
    from .interpolation import interpolate_1d

    all_extra_E = set()
    for mt_num in [MT_ELASTIC, MT_CAPTURE, MT_FISSION, MT_TOTAL]:
        if mt_num in background:
            energies, _, _ = _tabulated(background[mt_num])
            below = energies[energies < res_E_lo]
            above = energies[energies > res_E_hi]
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
            energies, values, interp_regions = _tabulated(background[mt_num])
            arr[:] = np.asarray(interpolate_1d(
                energies, values, interp_regions,
                extra_E, out_of_range="zero",
            ))

    if MT_TOTAL in background:
        energies, values, interp_regions = _tabulated(background[MT_TOTAL])
        extra_sig_tot = np.asarray(interpolate_1d(
            energies, values, interp_regions,
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


def _groupRadius(group):
    """The l-dependent scattering radius, wherever this formalism keeps it.

    ENDF's APL is per l-block. Breit-Wigner's model spin group *is* the block,
    so it carries the radius directly; under Reich-Moore the block became a spin
    group whose channels each carry it (§19.3.4 puts a radius on a channel), and
    every channel of the group has the same one. ``None`` means the file wrote
    APL = 0, which ENDF defines as "use the range's AP" — so the caller falls
    back, rather than this function inventing the fallback.
    """
    radius = getattr(group, "scatteringRadius", None)
    if radius is None:
        channels = getattr(group, "channels", None)
        radius = channels[0].scatteringRadius if channels else None
    # ENDF units out, because the caller hands this straight to the formulas.
    return _toEndfRadius(radius)


def _tabulated(form):
    """``(energies, values, (NBT, INT) pairs)`` out of a background form.

    ``Regions1d`` is what MF3 decodes to — a table with three interpolation
    laws has three regions — and ``XYs1d`` is what this module produces, so
    both have to be accepted: a caller may reasonably use one reconstruction as
    the background of another.
    """
    if hasattr(form, "toEndfRegions"):
        return form.toEndfRegions()
    xs = np.asarray(form.xs, dtype=float)
    return xs, np.asarray(form.ys, dtype=float), [(xs.size, form.endfInterpolationCode)]


def _interpolate_urr_params(widths, E_eval):
    """The unresolved averages, on the evaluation grid.

    Case A (``energyGrid is None``) is energy-independent and needs nothing.
    Cases B and C tabulate every average against the region's own grid and are
    interpolated onto *E_eval* here, once, so the formula sees arrays and never
    has to know which case it was handed.

    Linear interpolation, which is what the record-position version did and is
    **not** what case C's INT code necessarily says — the interpolation code is
    kept in ENDF provenance and is not read here. Recorded rather than fixed:
    changing it would move numbers, and this increment moves none.
    """
    from kika.nuclear_data.model.resonances import (UnresolvedChannel,
                                                    UnresolvedSpinGroup)

    from .urr_formulas import _levelSpacing, _width as _channelWidth

    if widths.energyGrid is None:
        return widths.spinGroups

    E_tab = widths.energyGrid
    result = []
    for group in widths.spinGroups:
        result.append(UnresolvedSpinGroup(
            L=group.L, J=group.J,
            atomicWeightRatio=group.atomicWeightRatio,
            levelSpacing=np.interp(
                E_eval, E_tab, np.atleast_1d(_levelSpacing(group))),
            channels=[
                UnresolvedChannel(
                    label=channel.label,
                    degreesOfFreedom=channel.degreesOfFreedom,
                    widths=np.interp(
                        E_eval, E_tab, np.atleast_1d(_channelWidth(channel))),
                )
                for channel in group.channels
            ],
        ))
    return result
