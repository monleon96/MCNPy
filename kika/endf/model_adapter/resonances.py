"""MF2/151 → :class:`~kika.nuclear_data.model.resonances.Resonances`.

**Split by formalism, which is the whole point.** ENDF writes every resolved
formalism into the same six columns and lets LRF decide what they mean: under
SLBW and MLBW they are ``GT, GN, GG, GF``, under Reich-Moore ``GN, GG, GFA,
GFB``. kika's flat ``ResonanceRecord`` keeps them as ``c3..c6`` and documents
the table, which is honest but leaves every consumer to re-derive the meaning.
GNDS §19 gives each formalism its own node with the names that formalism
actually uses, and that is what this decoder produces: a ``BreitWigner`` with
named widths, an ``RMatrix`` with widths that belong to *channels*, a
``TabulatedWidths`` for the unresolved region.

**The l-dependent scattering radius.** Phase 1 found MF2/151's APL was being
dropped outright — on the committed Fe-56 slice, AP is 0.5444 fm overall but
0.5002 fm for L=1, which moves elastic by a median 2% and up to 42% locally.
``EnergyRange.scattering_radius_for_l`` already resolves it against the LRF (the
same field is QX under LRF=1/2 and must never be read as a radius), so this
decoder asks it rather than reading C2, and a test asserts the L=1 value
survives into ``SpinGroup.scatteringRadius``.

**There is no encoder, and that is a scope decision, not an oversight.**
``ResonanceParameters`` has no ``to_endf`` — there has never been an MF2 writer
driven from a format-agnostic object — so unlike MF3 and MF4 there is no
existing path to be gated against. Writing one means emitting five formalisms'
record layouts from scratch, which is its own increment with its own risk.
Until it exists, ``MF2MT151.__str__`` is the only MF2 writer, it reproduces the
tape byte for byte, and the model is decode-only. The gate here is
correspondingly element-for-element against ``ResonanceParameters.from_endf``
plus the things that path does not keep.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from kika.nuclear_data.model import (
    BreitWigner,
    BreitWignerApproximation,
    ConversionReport,
    Resonances,
    ResolvedRegion,
    ScatteringRadius,
    UnresolvedRegion,
)
from kika.nuclear_data.model.resonances import (
    Channel,
    RMatrix,
    RMatrixSpinGroup,
    Resonance,
    ResonanceParameters,
    ResonanceReaction,
    SpinGroup,
    TabulatedWidths,
    UnresolvedChannel,
    UnresolvedSpinGroup,
)

__all__ = ["decodeMF2MT151", "LRF_TO_APPROXIMATION"]

#: ENDF LRF → the GNDS ``BreitWigner`` approximation (§19.3.6). LRF=3 and LRF=7
#: are not Breit-Wigner at all and become an ``RMatrix``.
LRF_TO_APPROXIMATION = {
    1: BreitWignerApproximation.singleLevel,
    2: BreitWignerApproximation.multiLevel,
}

#: ENDF LRF=7's KRM → the R-matrix approximation GNDS names.
KRM_TO_APPROXIMATION = {
    1: "SingleLevelBreitWigner",
    2: "MultiLevelBreitWigner",
    3: "ReichMoore",
    4: "Full",
}


def decodeMF2MT151(mf2mt151, report: Optional[ConversionReport] = None):
    """One MF2/151 section → ``(Resonances, report)``.

    Every isotope's every energy range becomes a region. A file with more than
    one isotope gets one flat list of regions and a warning, because GNDS puts
    each isotope in its own ``reactionSuite`` and this decoder produces one.
    """
    report = report if report is not None else ConversionReport()
    resonances = Resonances()

    isotopes = list(getattr(mf2mt151, "isotopes", []) or [])
    if len(isotopes) > 1:
        report.warn(
            f"MF2/151 carries {len(isotopes)} isotopes; GNDS gives each its own "
            f"reactionSuite, so their resonance regions are merged into one list "
            f"here and the per-isotope abundances are not represented"
        )

    for isotope in isotopes:
        for energyRange in isotope.energy_ranges:
            _decodeRange(energyRange, resonances, report)

    if not resonances.resolved and resonances.unresolved is None:
        report.lost("MF2/151 yielded no resonance region")
    return resonances, report


def _decodeRange(energyRange, resonances: Resonances, report: ConversionReport) -> None:
    lru, lrf = energyRange.lru, energyRange.lrf
    parameters = energyRange.parameters

    if energyRange.nro == 1 and energyRange.ap_e is not None:
        # An energy-dependent radius is a property of the evaluation, not of one
        # region, and ENDF gives no per-l version of it — so it wins over the
        # constant, and that precedence is stated here rather than left implicit.
        resonances.scatteringRadius = ScatteringRadius(
            constant=getattr(parameters, "ap", None),
            energies=np.asarray(energyRange.ap_e.energies, dtype=float),
            values=np.asarray(energyRange.ap_e.ap_values, dtype=float),
        )
    elif resonances.scatteringRadius is None and getattr(parameters, "ap", None) is not None:
        resonances.scatteringRadius = ScatteringRadius(constant=parameters.ap)

    if lru == 0:
        # A scattering-radius-only range carries no resonances; the radius above
        # is the whole of its content.
        return

    if lru == 2:
        resonances.unresolved = UnresolvedRegion(
            domainMin=energyRange.el,
            domainMax=energyRange.eh,
            tabulatedWidths=_decodeUnresolved(parameters, report),
        )
        return

    if lru != 1:
        report.unsupportedNode(f"MF2/151 LRU={lru} is not 0, 1 or 2; the range is dropped")
        return

    if lrf in LRF_TO_APPROXIMATION:
        formalism = _decodeBreitWigner(energyRange, parameters, lrf, report)
    elif lrf == 3:
        formalism = _decodeReichMoore(energyRange, parameters, report)
    elif lrf == 7:
        formalism = _decodeRMatrixLimited(parameters, report)
    else:
        report.unsupportedNode(
            f"MF2/151 LRF={lrf} in a resolved range is not one of 1, 2, 3, 7; "
            f"the range is present in the file and absent from the model"
        )
        return

    if formalism is None:
        return

    resonances.resolved.append(ResolvedRegion(
        domainMin=energyRange.el,
        domainMax=energyRange.eh,
        formalism=formalism,
    ))


# ---------------------------------------------------------------------------
# Resolved: Breit-Wigner (LRF=1, 2) and Reich-Moore (LRF=3)
# ---------------------------------------------------------------------------

def _spinGroups(energyRange, parameters, widthsFromFlat) -> List[SpinGroup]:
    groups: List[SpinGroup] = []
    for block in parameters.l_values:
        radius = energyRange.scattering_radius_for_l(block.l)
        groups.append(SpinGroup(
            L=block.l,
            resonances=[widthsFromFlat(r) for r in block.resonances],
            # Only a radius that genuinely differs from the range's own is a
            # per-l radius; scattering_radius_for_l falls back to AP.
            scatteringRadius=radius if radius != parameters.ap else None,
            atomicWeightRatio=block.awri,
        ))
    return groups


def _decodeBreitWigner(energyRange, parameters, lrf: int,
                       report: ConversionReport) -> Optional[BreitWigner]:
    """LRF=1/2. ``c3..c6`` are ``GT, GN, GG, GF``, which is what ``fromFlat`` assumes."""
    from kika.endf.classes.mf2.mf2mt151 import ResolvedResonanceRange

    if not isinstance(parameters, ResolvedResonanceRange):
        report.lost(f"MF2/151 LRF={lrf} range has no resolved parameters")
        return None

    for block in parameters.l_values:
        # Under LRF=1/2 the C2/L2 fields are QX and LRX, the competitive
        # reaction's Q value and its flag — not a radius. GNDS carries the
        # competitive channel's Q on the reaction, which this decoder does not
        # build, so the loss is declared rather than quietly absorbed.
        if block.lrx or block.apl_or_qx:
            report.lost(
                f"MF2/151 LRF={lrf} L={block.l}: QX={block.apl_or_qx} LRX="
                f"{block.lrx} describe a competitive width; the model has no "
                f"competitive channel for a Breit-Wigner region"
            )

    groups = _spinGroups(
        energyRange, parameters,
        lambda r: Resonance.fromFlat(r.energy, r.spin, r.c3, r.c4, r.c5, r.c6),
    )
    return BreitWigner(
        approximation=LRF_TO_APPROXIMATION[lrf],
        resonanceParameters=ResonanceParameters(spinGroups=groups),
        scatteringRadius=parameters.ap,
    )


def _decodeReichMoore(energyRange, parameters, report: ConversionReport) -> Optional[RMatrix]:
    """LRF=3. The four columns are ``GN, GG, GFA, GFB`` — *two* fission widths.

    This is the case ``c3..c6`` cannot name and the reason the model splits by
    formalism at all. Reich-Moore is an R-matrix approximation, so it becomes an
    :class:`RMatrix` whose widths belong to named channels, one spin group per
    l-block.
    """
    from kika.endf.classes.mf2.mf2mt151 import ResolvedResonanceRange

    if not isinstance(parameters, ResolvedResonanceRange):
        report.lost("MF2/151 LRF=3 range has no resolved parameters")
        return None

    reactions = [
        ResonanceReaction(label="elastic", ejectile="n"),
        ResonanceReaction(label="capture", ejectile="photon", eliminated=True),
        ResonanceReaction(label="fissionA"),
        ResonanceReaction(label="fissionB"),
    ]
    channelLabels = ("neutron", "capture", "fissionA", "fissionB")

    spinGroups: List[RMatrixSpinGroup] = []
    for block in parameters.l_values:
        radius = energyRange.scattering_radius_for_l(block.l)
        # ENDF's APL is per l-block; §19.3.4's per-channel scatteringRadius is
        # the closest GNDS home, so every channel of this spin group carries it.
        # `None` here means "this l uses the range's AP", which is a different
        # statement from "this l has a radius that happens to equal AP".
        perL = radius if radius != parameters.ap else None
        channels = [
            Channel(label=label, resonanceReaction=reactions[i].label, L=block.l,
                    columnIndex=i, scatteringRadius=perL)
            for i, label in enumerate(channelLabels)
        ]
        spinGroups.append(RMatrixSpinGroup(
            label=f"L{block.l}",
            channels=channels,
            energies=[r.energy for r in block.resonances],
            widths=[[r.c3, r.c4, r.c5, r.c6] for r in block.resonances],
        ))

    return RMatrix(
        approximation="ReichMoore",
        resonanceReactions=reactions,
        spinGroups=spinGroups,
    )


def _decodeRMatrixLimited(parameters, report: ConversionReport) -> Optional[RMatrix]:
    """LRF=7. Already channel-shaped in the file, so the mapping is direct."""
    from kika.endf.classes.mf2.mf2mt151 import RMatrixLimited

    if not isinstance(parameters, RMatrixLimited):
        report.lost("MF2/151 LRF=7 range has no R-matrix-limited parameters")
        return None

    reactions = [
        ResonanceReaction(label=f"MT{pair.mt}", Q=pair.q)
        for pair in parameters.particle_pairs
    ]

    spinGroups: List[RMatrixSpinGroup] = []
    for index, group in enumerate(parameters.spin_groups):
        channels = [
            Channel(
                label=f"channel{i + 1}",
                # IPP is one-based into the particle-pair list.
                resonanceReaction=(
                    reactions[channel.ipp - 1].label
                    if 0 < channel.ipp <= len(reactions) else f"pair{channel.ipp}"
                ),
                L=channel.l,
                channelSpin=channel.sch,
                columnIndex=i,
            )
            for i, channel in enumerate(group.channels)
        ]
        spinGroups.append(RMatrixSpinGroup(
            label=f"J{group.aj}-{index}",
            spin=group.aj,
            parity=int(group.pj) if group.pj else None,
            channels=channels,
            energies=[r.er for r in group.resonances],
            widths=[list(r.widths) for r in group.resonances],
        ))
        if group.kbk:
            report.unsupportedNode(
                f"MF2/151 LRF=7 spin group {index} carries {group.kbk} background "
                f"R-matrix records (LBK); GNDS §19.3 has nodes for these and this "
                f"decoder does not read them"
            )
        if group.kps:
            report.unsupportedNode(
                f"MF2/151 LRF=7 spin group {index} carries tabulated phase shifts "
                f"(KPS={group.kps}); they are not represented"
            )

    if parameters.ifg:
        report.warn(
            "MF2/151 LRF=7 has IFG=1, so the widths are reduced-width amplitudes "
            "rather than widths in eV; they are stored as written and the flag is "
            "not represented in the model"
        )

    return RMatrix(
        approximation=KRM_TO_APPROXIMATION.get(parameters.krm),
        resonanceReactions=reactions,
        spinGroups=spinGroups,
    )


# ---------------------------------------------------------------------------
# Unresolved (LRU=2)
# ---------------------------------------------------------------------------

def _channel(label: str, value, degreesOfFreedom: float = 1.0) -> UnresolvedChannel:
    """One average width, constant (case A) or tabulated (cases B and C)."""
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        return UnresolvedChannel(label=label, degreesOfFreedom=degreesOfFreedom,
                                 widths=np.asarray(value, dtype=float))
    return UnresolvedChannel(label=label, degreesOfFreedom=degreesOfFreedom,
                             constantWidth=float(value))


def _decodeUnresolved(parameters, report: ConversionReport) -> Optional[TabulatedWidths]:
    from kika.endf.classes.mf2.mf2mt151 import (UnresolvedCaseA, UnresolvedCaseB,
                                                UnresolvedCaseC)

    if parameters is None:
        report.lost("MF2/151 unresolved range has no parameters")
        return None

    spinGroups: List[UnresolvedSpinGroup] = []
    energyGrid = None

    if isinstance(parameters, UnresolvedCaseA):
        for block in parameters.l_values:
            for state in block.j_states:
                spinGroups.append(UnresolvedSpinGroup(
                    L=block.l, J=state.aj, levelSpacing=np.asarray([state.d], dtype=float),
                    atomicWeightRatio=block.awri,
                    channels=[
                        _channel("neutron", state.gn0, state.amun),
                        _channel("capture", state.gg),
                    ],
                ))

    elif isinstance(parameters, UnresolvedCaseB):
        energyGrid = np.asarray(parameters.energies, dtype=float)
        for block in parameters.l_values:
            for state in block.j_states:
                channels = [
                    _channel("neutron", state.gn0, state.amun),
                    _channel("capture", state.gg),
                ]
                if state.gf:
                    channels.append(_channel("fission", state.gf, float(state.muf)))
                spinGroups.append(UnresolvedSpinGroup(
                    L=block.l, J=state.aj, levelSpacing=np.asarray([state.d], dtype=float),
                    atomicWeightRatio=block.awri, channels=channels,
                ))

    elif isinstance(parameters, UnresolvedCaseC):
        for block in parameters.l_values:
            for state in block.j_states:
                points = state.energy_points
                energyGrid = np.array([p.es for p in points], dtype=float)
                spinGroups.append(UnresolvedSpinGroup(
                    L=block.l, J=state.aj,
                    levelSpacing=np.array([p.d for p in points], dtype=float),
                    atomicWeightRatio=block.awri,
                    channels=[
                        _channel("neutron", [p.gn0 for p in points], state.amun),
                        _channel("capture", [p.gg for p in points], state.amug),
                        _channel("fission", [p.gf for p in points], state.amuf),
                        _channel("competitive", [p.gx for p in points], state.amux),
                    ],
                ))
    else:
        report.unsupportedNode(
            f"MF2/151 unresolved parameters of type {type(parameters).__name__} "
            f"are not one of ENDF's cases A, B or C"
        )
        return None

    return TabulatedWidths(
        spinGroups=spinGroups,
        energyGrid=energyGrid,
        scatteringRadius=parameters.ap,
        selfShieldingOnly=bool(parameters.lssf),
    )
