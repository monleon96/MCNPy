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
    EndfProvenance,
    BreitWignerApproximation,
    ConversionReport,
    Nuclide,
    PhysicalQuantity,
    PoPs,
    Resonances,
    ResolvedRegion,
    ScatteringRadius,
    MODEL_RADIUS_UNIT,
    radiusFromEndf,
    radiusToEndf,
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

__all__ = ["decodeMF2MT151", "encodeMF2MT151", "LRF_TO_APPROXIMATION",
           "PARTICLE_PAIR_FIELDS"]

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
    """One MF2/151 section → ``(Resonances, EndfProvenance, report)``.

    Every isotope's every energy range becomes a region. A file with more than
    one isotope gets one flat list of regions and a warning, because GNDS puts
    each isotope in its own ``reactionSuite`` and this decoder produces one.

    The provenance's ``headerFields['regions']`` is a list parallel to the
    decoded regions, holding the per-range ENDF bookkeeping GNDS has no node
    for — ``lrf``, ``nlsc``, ``abn``, and the range's own ``spi``/``ap``. It is
    what lets the phase 3d ``ResonanceParameters`` façade rebuild its
    ``metadata`` without re-reading the section, and what lets
    ``encodeMF2MT151`` write the section back.

    **Every range appears in that list, including LRU=0 and the formalisms the
    model cannot hold.** A ``kind`` key says which is which, and a consumer that
    walks it in step with ``resonances.resolved`` must skip ``radiusOnly``:
    those ranges contribute a scattering radius and no region. Before B1a they
    were dropped before the append, so an isotope's second range silently became
    its first.
    """
    report = report if report is not None else ConversionReport()
    resonances = Resonances()
    regions: List[dict] = []

    isotopes = list(getattr(mf2mt151, "isotopes", []) or [])
    if len(isotopes) > 1:
        report.warn(
            f"MF2/151 carries {len(isotopes)} isotopes; GNDS gives each its own "
            f"reactionSuite, so their resonance regions are merged into one list "
            f"here and the per-isotope abundances are not represented"
        )

    for index, isotope in enumerate(isotopes):
        for energyRange in isotope.energy_ranges:
            _decodeRange(energyRange, resonances, report, regions, isotope, index)

    if not resonances.resolved and resonances.unresolved is None:
        report.lost("MF2/151 yielded no resonance region")

    provenance = EndfProvenance(
        mat=getattr(mf2mt151, "_mat", None),
        awr=getattr(mf2mt151, "atomic_weight_ratio", None),
        za=(int(round(float(mf2mt151.zaid))) if mf2mt151.zaid is not None else None),
        headerFields={
            "nis": getattr(mf2mt151, "_nis", None),
            # Per isotope, in file order. ``lfw`` is what picks URR case A from
            # case B and lives on the isotope record, not on the range, so a
            # range alone cannot say which case it is.
            "isotopes": [
                {
                    "za": getattr(isotope, "za", None),
                    "abn": getattr(isotope, "abn", None),
                    "lfw": getattr(isotope, "lfw", None),
                    "ner": len(isotope.energy_ranges),
                }
                for isotope in isotopes
            ],
            "regions": regions,
        },
    )
    return resonances, provenance, report


def _rangeFields(energyRange, isotope, isotopeIndex: int) -> dict:
    parameters = energyRange.parameters
    return {
        "lru": energyRange.lru,
        "lrf": energyRange.lrf,
        "nro": energyRange.nro,
        "naps": energyRange.naps,
        "el": energyRange.el,
        "eh": energyRange.eh,
        "spi": getattr(parameters, "spi", None),
        "ap": getattr(parameters, "ap", None),
        "nls": getattr(parameters, "nls", None),
        "nlsc": getattr(parameters, "nlsc", None),
        "abn": getattr(isotope, "abn", None),
        "isotope_index": isotopeIndex,
        #: LAD says whether MF4 can be computed from these parameters. Parsed
        #: since D5, carried since B1a — Th-232 is the tape to hand that sets it.
        "lad": getattr(parameters, "lad", None),
    }


def _targetPoPs(spi, za) -> Optional[PoPs]:
    """The target nuclide and its spin, as the formalism's own ``PoPs`` (§19.3.1).

    **Why the formalism and not the suite.** ENDF writes SPI on every energy
    *range*, and reconstruction needs it: the statistical factor is
    ``g_J = (2J+1) / (2(2I+1))``, so a wrong I is a wrong cross section rather
    than a wrong label. GNDS's home for it is a local ``PoPs`` on the resonance
    formalism, and this is not inferred — the shipped ENDF/B-VIII.1 GNDS
    distribution writes exactly that for Fe-56::

        <RMatrix label="eval" approximation="ReichMoore" ...>
          <PoPs name="resolved resonances" version="1.0" format="2.0">
            ... <nucleus id="fe56" index="0">
                  <spin><fraction label="eval" value="0" unit="hbar"/></spin>

    Keeping it per range also keeps a file whose ranges disagree representable,
    which a single suite-level spin would not.

    **The duplication with provenance is deliberate and one-directional.**
    ``headerFields['regions'][i]['spi']`` still holds SPI and is still what
    ``encodeMF2MT151`` writes, so this addition cannot move a byte of any
    round trip. The model copy is what a *calculation* reads, because a
    reconstructor that has to open an ENDF provenance to find a spin is not
    format-agnostic. If the two ever need to disagree, provenance is the file
    and the model is the physics — but nothing edits either today.
    """
    if spi is None:
        return None

    identifier = None
    if za is not None:
        try:
            from kika._utils import zaid_to_symbol
            identifier = zaid_to_symbol(int(round(float(za))))
        except Exception:
            identifier = None
    if identifier is None:
        identifier = "target"

    pops = PoPs(name="resolved resonances")
    zNumber, aNumber = None, None
    if za is not None:
        zaInt = int(round(float(za)))
        zNumber, aNumber = divmod(zaInt, 1000)
    pops.add(Nuclide(
        id=identifier,
        Z=zNumber,
        A=aNumber,
        spin=PhysicalQuantity(float(spi), "hbar"),
    ))
    return pops


def _decodeRange(energyRange, resonances: Resonances, report: ConversionReport,
                 regions: Optional[List[dict]] = None, isotope=None,
                 isotopeIndex: int = 0) -> None:
    lru, lrf = energyRange.lru, energyRange.lrf
    parameters = energyRange.parameters
    fields = _rangeFields(energyRange, isotope, isotopeIndex)

    if energyRange.nro == 1 and energyRange.ap_e is not None:
        # An energy-dependent radius is a property of the evaluation, not of one
        # region, and ENDF gives no per-l version of it — so it wins over the
        # constant, and that precedence is stated here rather than left implicit.
        resonances.scatteringRadius = ScatteringRadius(
            constant=radiusFromEndf(getattr(parameters, "ap", None)),
            energies=np.asarray(energyRange.ap_e.energies, dtype=float),
            values=radiusFromEndf(
                np.asarray(energyRange.ap_e.ap_values, dtype=float)),
            interpolation=list(energyRange.ap_e.interpolation),
            unit=MODEL_RADIUS_UNIT,
        )
    elif resonances.scatteringRadius is None and getattr(parameters, "ap", None) is not None:
        resonances.scatteringRadius = ScatteringRadius(
            constant=radiusFromEndf(parameters.ap), unit=MODEL_RADIUS_UNIT)

    if energyRange.nro == 1 and energyRange.ap_e is not None:
        # Kept for *every* kind of range, not only resolved ones: NRO=1 is a
        # property of the range record and an unresolved range may carry it.
        fields["radius_table"] = (
            np.asarray(energyRange.ap_e.energies, dtype=float),
            np.asarray(energyRange.ap_e.ap_values, dtype=float),
            list(energyRange.ap_e.interpolation),
        )

    def keep(kind: str) -> None:
        if regions is not None:
            regions.append({**fields, "kind": kind})

    if lru == 0:
        # A scattering-radius-only range carries no resonances; the radius above
        # is the whole of its content. It still has to be recorded, or the file
        # cannot be written back with the same number of ranges.
        keep("radiusOnly")
        return

    if lru == 2:
        tabulated = _decodeUnresolved(parameters, report, fields)
        if tabulated is not None and tabulated.PoPs is None:
            tabulated.PoPs = _targetPoPs(
                fields.get("spi"),
                getattr(isotope, "za", None) if isotope is not None else None,
            )
        resonances.unresolved = UnresolvedRegion(
            domainMin=energyRange.el,
            domainMax=energyRange.eh,
            tabulatedWidths=tabulated,
        )
        keep("unresolved")
        return

    if lru != 1:
        report.unsupportedNode(f"MF2/151 LRU={lru} is not 0, 1 or 2; the range is dropped")
        keep("unsupported")
        return

    if lrf in LRF_TO_APPROXIMATION:
        formalism = _decodeBreitWigner(parameters, lrf, report, fields)
    elif lrf == 3:
        formalism = _decodeReichMoore(parameters, report, fields)
    elif lrf == 7:
        formalism = _decodeRMatrixLimited(parameters, report, fields)
    else:
        report.unsupportedNode(
            f"MF2/151 LRF={lrf} in a resolved range is not one of 1, 2, 3, 7; "
            f"the range is present in the file and absent from the model"
        )
        keep("unsupported")
        return

    if formalism is None:
        keep("unsupported")
        return

    # Additive: the encoder still writes SPI from provenance, so this cannot
    # move a byte. It is what makes the range self-sufficient for a calculation.
    if getattr(formalism, "PoPs", None) is None:
        formalism.PoPs = _targetPoPs(
            fields.get("spi"),
            getattr(isotope, "za", None) if isotope is not None else None,
        )

    resonances.resolved.append(ResolvedRegion(
        domainMin=energyRange.el,
        domainMax=energyRange.eh,
        formalism=formalism,
    ))
    keep("resolved")


# ---------------------------------------------------------------------------
# Resolved: Breit-Wigner (LRF=1, 2) and Reich-Moore (LRF=3)
# ---------------------------------------------------------------------------

def _lBlockFields(parameters) -> List[dict]:
    """Per l-block bookkeeping: what the block header carries and the model does not.

    ``lrx`` is a flag and is bookkeeping under every LRF. ``qx`` is the C2 field
    read as a Q value, which is what it means under LRF=1/2 only — under LRF=3
    the same field is APL, a radius, and it lives on the model's channels
    instead. Keeping the raw value here as well would give the encoder two
    sources for one field and let a caller's edit to the model be silently
    ignored, so it deliberately does not.
    """
    return [
        {"l": block.l, "lrx": block.lrx, "qx": block.apl_or_qx,
         "num_resonances": block.num_resonances}
        for block in parameters.l_values
    ]


def _spinGroups(parameters, widthsFromFlat) -> List[SpinGroup]:
    groups: List[SpinGroup] = []
    for block in parameters.l_values:
        groups.append(SpinGroup(
            L=block.l,
            resonances=[widthsFromFlat(r) for r in block.resonances],
            # Under LRF=1/2 the block's C2 field is QX, a Q value, and there is
            # no per-l radius in the file at all — so this is None by the
            # formalism, not by a comparison that happened to come out equal.
            scatteringRadius=None,
            atomicWeightRatio=block.awri,
        ))
    return groups


def _decodeBreitWigner(parameters, lrf: int, report: ConversionReport,
                       fields: Optional[dict] = None) -> Optional[BreitWigner]:
    """LRF=1/2. ``c3..c6`` are ``GT, GN, GG, GF``, which is what ``fromFlat`` assumes."""
    from kika.endf.classes.mf2.mf2mt151 import ResolvedResonanceRange

    if not isinstance(parameters, ResolvedResonanceRange):
        report.lost(f"MF2/151 LRF={lrf} range has no resolved parameters")
        return None

    for block in parameters.l_values:
        # Under LRF=1/2 the C2/L2 fields are QX and LRX, the competitive
        # reaction's Q value and its flag — not a radius. GNDS carries the
        # competitive channel's Q on the reaction, which this decoder does not
        # build, so the loss is declared rather than quietly absorbed. Since
        # B1a the values are carried in provenance, so the section can still be
        # written back; the *model* is what has no home for them.
        if block.lrx or block.apl_or_qx:
            report.lost(
                f"MF2/151 LRF={lrf} L={block.l}: QX={block.apl_or_qx} LRX="
                f"{block.lrx} describe a competitive width; the model has no "
                f"competitive channel for a Breit-Wigner region, so they are "
                f"kept in provenance and written back unchanged"
            )

    if fields is not None:
        fields["l_blocks"] = _lBlockFields(parameters)

    groups = _spinGroups(
        parameters,
        lambda r: Resonance.fromFlat(r.energy, r.spin, r.c3, r.c4, r.c5, r.c6),
    )
    return BreitWigner(
        approximation=LRF_TO_APPROXIMATION[lrf],
        resonanceParameters=ResonanceParameters(spinGroups=groups),
        scatteringRadius=radiusFromEndf(parameters.ap),
        radiusUnit=MODEL_RADIUS_UNIT,
    )


def _decodeReichMoore(parameters, report: ConversionReport,
                      fields: Optional[dict] = None) -> Optional[RMatrix]:
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

    if fields is not None:
        fields["l_blocks"] = _lBlockFields(parameters)

    spinGroups: List[RMatrixSpinGroup] = []
    for block in parameters.l_values:
        # ENDF's APL is per l-block; §19.3.4's per-channel scatteringRadius is
        # the closest GNDS home, so every channel of this spin group carries it.
        #
        # `None` means **the file wrote 0 in the APL field**, which is ENDF's
        # way of saying "this l uses the range's AP". It does *not* mean "the
        # radius came out equal to AP". Those are different statements and the
        # first draft of this decoder conflated them, comparing the *resolved*
        # radius against AP and storing None whenever they matched — so on
        # U-235, Th-232 and Pu-241, which all write APL == AP explicitly, an
        # encoder would put 0.0 where the file has 9.686000-1. Three of the six
        # tapes to hand, and byte identity lost on all three.
        # The zero test is on the **file's** number, before the conversion:
        # multiplying first would work today and would make the sentinel depend
        # on a scale factor, which is one refactor away from being wrong.
        perL = radiusFromEndf(block.apl_or_qx) if block.apl_or_qx else None
        channels = [
            Channel(label=label, resonanceReaction=reactions[i].label, L=block.l,
                    columnIndex=i, scatteringRadius=perL,
                    radiusUnit=MODEL_RADIUS_UNIT)
            for i, label in enumerate(channelLabels)
        ]
        spinGroups.append(RMatrixSpinGroup(
            label=f"L{block.l}",
            channels=channels,
            energies=[r.energy for r in block.resonances],
            # ENDF's LRF=3 groups by l and writes AJ per resonance, so the J
            # belongs to the record and not to the group.
            spins=[r.spin for r in block.resonances],
            widths=[[r.c3, r.c4, r.c5, r.c6] for r in block.resonances],
            atomicWeightRatio=block.awri,
        ))

    return RMatrix(
        approximation="ReichMoore",
        resonanceReactions=reactions,
        spinGroups=spinGroups,
        scatteringRadius=radiusFromEndf(parameters.ap),
        radiusUnit=MODEL_RADIUS_UNIT,
    )


#: The twelve columns of an LRF=7 particle-pair record, in file order. Ten of
#: them had no home before B1a: only ``mt`` and ``q`` reached the model, as a
#: ``ResonanceReaction`` label and Q value.
#:
#: They stay in provenance rather than becoming ``PoPs`` particles, and the
#: reason is that the inverse is ambiguous: ``MA = 0.0`` for the capture pair
#: and a negative ``IB`` are ENDF *encoding conventions*, not masses and spins,
#: so reconstructing these columns from PoPs would guess — and a guess inside
#: the encoder is exactly what the byte-identity gate cannot tolerate.
PARTICLE_PAIR_FIELDS = ("ma", "mb", "za", "zb", "ia", "ib",
                        "q", "pnt", "shf", "mt", "pa", "pb")


def _decodeRMatrixLimited(parameters, report: ConversionReport,
                          fields: Optional[dict] = None) -> Optional[RMatrix]:
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
    groupFields: List[dict] = []
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
                # APT is the true channel radius and APE the effective one; the
                # file writes both and they differ. Stored raw, never collapsed
                # to None on a zero — a channel radius of 0.0 is what some
                # evaluations write and it has to come back out that way.
                scatteringRadius=radiusFromEndf(channel.apt),
                hardSphereRadius=radiusFromEndf(channel.ape),
                radiusUnit=MODEL_RADIUS_UNIT,
                boundaryConditionValue=channel.bnd,
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

        # KBK/KPS: carried verbatim so the section round-trips, *and* still
        # declared unsupported so the report does not imply the model
        # understands them. Both halves matter — carrying them silently would
        # let a caller believe the model can reason about a background R-matrix
        # it merely copies. There is no tape to hand that exercises either.
        perGroup = {"kbk": group.kbk, "kps": group.kps, "pj": group.pj}
        if group.kbk:
            perGroup["background"] = list(group.background)
            report.unsupportedNode(
                f"MF2/151 LRF=7 spin group {index} carries {group.kbk} background "
                f"R-matrix records (LBK); GNDS §19.3 has nodes for these, the "
                f"model has none, and they are kept verbatim in provenance so "
                f"the section can still be written back unchanged"
            )
        if group.kps:
            perGroup["phase_shift"] = group.phase_shift
            report.unsupportedNode(
                f"MF2/151 LRF=7 spin group {index} carries tabulated phase shifts "
                f"(KPS={group.kps}); they are not represented in the model and "
                f"are kept verbatim in provenance"
            )
        groupFields.append(perGroup)

    if fields is not None:
        fields["particle_pairs"] = [
            {name: getattr(pair, name) for name in PARTICLE_PAIR_FIELDS}
            for pair in parameters.particle_pairs
        ]
        fields["spin_groups"] = groupFields
        # KRM is recoverable by inverting KRM_TO_APPROXIMATION, but only while
        # that mapping stays injective. Carrying the number is cheaper than
        # depending on that.
        fields["krm"] = parameters.krm

    if parameters.ifg:
        report.warn(
            "MF2/151 LRF=7 has IFG=1, so the widths are reduced-width amplitudes "
            "in eV^1/2 rather than widths in eV; they are stored as written and "
            "RMatrix.reducedWidthAmplitudes says so"
        )

    return RMatrix(
        approximation=KRM_TO_APPROXIMATION.get(parameters.krm),
        resonanceReactions=reactions,
        spinGroups=spinGroups,
        reducedWidthAmplitudes=bool(parameters.ifg),
        relativisticKinematics=bool(parameters.krl),
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


def _decodeUnresolved(parameters, report: ConversionReport,
                      fields: Optional[dict] = None) -> Optional[TabulatedWidths]:
    from kika.endf.classes.mf2.mf2mt151 import (UnresolvedCaseA, UnresolvedCaseB,
                                                UnresolvedCaseC)

    if parameters is None:
        report.lost("MF2/151 unresolved range has no parameters")
        return None

    spinGroups: List[UnresolvedSpinGroup] = []
    stateFields: List[dict] = []
    energyGrid = None
    case = None

    if isinstance(parameters, UnresolvedCaseA):
        case = "A"
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
                stateFields.append({"l": block.l, "j": state.aj})

    elif isinstance(parameters, UnresolvedCaseB):
        case = "B"
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
                # MUF rides on the fission channel's degreesOfFreedom, and when
                # GF is falsy that channel is never built — so the value had
                # nowhere to go. Case B is untested by any tape to hand, which
                # is exactly why the loss has to be closed rather than argued
                # about from the corpus.
                stateFields.append({"l": block.l, "j": state.aj, "muf": state.muf,
                                    "gf": list(state.gf)})

    elif isinstance(parameters, UnresolvedCaseC):
        case = "C"
        for block in parameters.l_values:
            for state in block.j_states:
                points = state.energy_points
                grid = np.array([p.es for p in points], dtype=float)
                # ``energyGrid`` is one grid for the whole region, so a file
                # whose J-states disagree loses all but the last. The three URR
                # tapes to hand all share one grid, so this is latent rather
                # than observed — recorded per state so the encoder does not
                # depend on that staying true.
                if energyGrid is not None and not np.array_equal(energyGrid, grid):
                    report.lost(
                        f"MF2/151 URR case C: L={block.l} J={state.aj} has its own "
                        f"ES grid of {grid.size} points, which differs from the "
                        f"region's; TabulatedWidths holds one grid, so only "
                        f"provenance keeps this one"
                    )
                energyGrid = grid
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
                # INT varies across the corpus — 2 on U-235 and Th-232, 5 on
                # Pu-241 — so it is not a constant that can be assumed on the
                # way out.
                stateFields.append({"l": block.l, "j": state.aj,
                                    "int_code": state.int_code,
                                    "energy_grid": grid})
    else:
        report.unsupportedNode(
            f"MF2/151 unresolved parameters of type {type(parameters).__name__} "
            f"are not one of ENDF's cases A, B or C"
        )
        return None

    if fields is not None:
        fields["urr_case"] = case
        fields["lssf"] = parameters.lssf
        fields["j_states"] = stateFields

    return TabulatedWidths(
        spinGroups=spinGroups,
        energyGrid=energyGrid,
        scatteringRadius=radiusFromEndf(parameters.ap),
        radiusUnit=MODEL_RADIUS_UNIT,
        selfShieldingOnly=bool(parameters.lssf),
    )


# ---------------------------------------------------------------------------
# Encode: model + provenance → MF2MT151
# ---------------------------------------------------------------------------

def encodeMF2MT151(resonances: Resonances, provenance, report=None):
    """``(Resonances, EndfProvenance)`` → an ``MF2MT151``, ready to be written.

    **This is a tree rebuild, not a record writer, and that is the whole design.**
    ``MF2MT151.__str__`` already emits every formalism's record layout and
    reproduces its source tape byte for byte (``test_mf2_writes_back_what_it_read``,
    and it took two fixes — NPP and LAD — to become true). Emitting records here
    as well would be a second implementation of the same layouts, which is what
    A2 avoided deliberately for MF33/MF34 and what ``library-gaps.md`` says MF32
    should reuse rather than write again.

    **Physics comes from the model, bookkeeping from provenance**, and nothing
    is read from both. That rule is what makes an edit effective: change a
    channel radius on the model and this writes the changed value, because the
    raw APL is deliberately *not* also kept in provenance. The inverse rule
    applies to fields the model has no node for — QX, LRX, LAD, the twelve
    particle-pair columns — which come only from provenance and are written back
    unchanged.

    Raises ``ValueError`` rather than guessing when a range cannot be rebuilt.
    A structurally valid MF2 section with invented parameters is worse than no
    section, because it carries an authority it has not earned.
    """
    from kika.endf.classes.mf2.mf2mt151 import (EnergyDependentScatteringRadius,
                                                EnergyRange, Isotope, MF2MT151,
                                                ScatteringRadiusOnly)

    report = report if report is not None else ConversionReport()
    header = getattr(provenance, "headerFields", None) or {}
    regions = header.get("regions")
    if regions is None:
        raise ValueError(
            "this provenance carries no 'regions' entry, so it did not come from "
            "decodeMF2MT151 and there is nothing to rebuild an MF2/151 from"
        )

    resolved = iter(resonances.resolved)
    ranges: dict = {}

    for fields in regions:
        kind = fields["kind"]
        if kind == "unsupported":
            raise ValueError(
                f"MF2/151 range LRU={fields['lru']} LRF={fields['lrf']} was not "
                f"decoded into the model, so it cannot be written back. The "
                f"section can only be re-emitted by the parser that read it."
            )

        if kind == "radiusOnly":
            parameters = ScatteringRadiusOnly(spi=fields["spi"], ap=fields["ap"])
        elif kind == "unresolved":
            if resonances.unresolved is None:
                raise ValueError(
                    "provenance records an unresolved range and the model holds "
                    "no unresolved region"
                )
            parameters = _encodeUnresolved(resonances.unresolved, fields)
        else:
            region = next(resolved, None)
            if region is None:
                raise ValueError(
                    f"provenance records more resolved ranges than the model has "
                    f"regions ({len(resonances.resolved)})"
                )
            parameters = _encodeResolved(region.formalism, fields, report)

        apE = None
        if "radius_table" in fields:
            energies, values, interpolation = fields["radius_table"]
            apE = EnergyDependentScatteringRadius(
                interpolation=[tuple(pair) for pair in interpolation],
                energies=list(energies), ap_values=list(values),
            )

        ranges.setdefault(fields["isotope_index"], []).append(EnergyRange(
            el=fields["el"], eh=fields["eh"], lru=fields["lru"], lrf=fields["lrf"],
            nro=fields["nro"], naps=fields["naps"], parameters=parameters, ap_e=apE,
        ))

    section = MF2MT151(number=151)
    section._za = provenance.za
    section._awr = provenance.awr
    section._mat = provenance.mat
    section._nis = header.get("nis")
    section._isotopes = [
        Isotope(za=isotope["za"], abn=isotope["abn"], lfw=isotope["lfw"],
                num_energy_ranges=isotope["ner"],
                energy_ranges=ranges.get(index, []))
        for index, isotope in enumerate(header.get("isotopes", []))
    ]
    return section


def _encodeResolved(formalism, fields: dict, report: ConversionReport):
    """One resolved range: ``BreitWigner``/``RMatrix`` + its fields → ENDF parameters."""
    from kika.endf.classes.mf2.mf2mt151 import (LValueBlock, ResolvedResonanceRange,
                                                Resonance as EndfResonance)

    if fields["lrf"] == 7:
        return _encodeRMatrixLimited(formalism, fields)

    if formalism is None:
        raise ValueError(f"MF2/151 LRF={fields['lrf']} range has no formalism to write")

    groups = (formalism.resonanceParameters.spinGroups
              if isinstance(formalism, BreitWigner) else formalism.spinGroups)
    blocks = fields.get("l_blocks") or []
    if len(blocks) != len(groups):
        raise ValueError(
            f"MF2/151 LRF={fields['lrf']}: {len(groups)} spin groups on the model "
            f"and {len(blocks)} l-block records in provenance"
        )

    lValues = []
    for group, block in zip(groups, blocks):
        if isinstance(formalism, BreitWigner):
            # LRF=1/2: C2 is QX, a Q value the model has no node for, so it is
            # written back from provenance exactly as it was read.
            c2 = block["qx"]
            records = [EndfResonance(*resonance.toFlat()) for resonance in group.resonances]
            l = group.L
        else:
            # LRF=3: C2 is APL, which *is* on the model — so a caller who
            # changed the radius gets the changed radius written. ``None``
            # means the file wrote 0 and 0 is what goes back.
            # Back to ENDF's units on the way out. The file-level AP does not
            # need this -- it is written from `fields["ap"]`, which is ENDF's
            # own number kept verbatim -- but APL lives on the model and
            # nowhere else, which is what makes a caller's edit effective.
            c2 = group.channels[0].scatteringRadius if group.channels else None
            c2 = radiusToEndf(c2) if c2 is not None else 0.0
            records = [
                EndfResonance(energy=energy, spin=spin,
                              c3=widths[0], c4=widths[1], c5=widths[2], c6=widths[3])
                for energy, spin, widths in zip(group.energies, group.spins, group.widths)
            ]
            l = group.channels[0].L if group.channels else None

        lValues.append(LValueBlock(
            awri=group.atomicWeightRatio, l=l,
            num_resonances=block["num_resonances"],
            apl_or_qx=c2, lrx=block["lrx"], resonances=records,
        ))

    return ResolvedResonanceRange(
        spi=fields["spi"], ap=fields["ap"], nls=fields["nls"],
        nlsc=fields["nlsc"], lad=fields["lad"], l_values=lValues,
    )


def _encodeRMatrixLimited(formalism: RMatrix, fields: dict):
    """LRF=7. Ten of the twelve particle-pair columns come from provenance."""
    from kika.endf.classes.mf2.mf2mt151 import (RMatrixLimited, RML_Channel,
                                                RML_ParticlePair, RML_Resonance,
                                                RML_SpinGroup)

    pairs = fields.get("particle_pairs")
    if pairs is None:
        raise ValueError(
            "MF2/151 LRF=7: provenance has no particle-pair columns. Ten of the "
            "twelve are ENDF encoding conventions (MA = 0 for a capture pair, a "
            "negative IB) and cannot be reconstructed from the model's PoPs "
            "without guessing, so the section is not written."
        )

    # IPP is one-based into the pair list; the model stores the reaction *label*
    # instead, so the index comes back by lookup rather than being stored twice.
    order = {reaction.label: index + 1
             for index, reaction in enumerate(formalism.resonanceReactions)}

    spinGroups = []
    for group, perGroup in zip(formalism.spinGroups, fields["spin_groups"]):
        channels = []
        for channel in group.channels:
            ipp = order.get(channel.resonanceReaction)
            if ipp is None:
                raise ValueError(
                    f"channel {channel.label!r} names resonance reaction "
                    f"{channel.resonanceReaction!r}, which is not one of "
                    f"{sorted(order)}; IPP cannot be resolved"
                )
            channels.append(RML_Channel(
                ipp=ipp, l=channel.L, sch=channel.channelSpin,
                bnd=channel.boundaryConditionValue,
                ape=radiusToEndf(channel.hardSphereRadius),
                apt=radiusToEndf(channel.scatteringRadius),
            ))

        spinGroups.append(RML_SpinGroup(
            aj=group.spin, pj=perGroup["pj"],
            kbk=perGroup["kbk"], kps=perGroup["kps"],
            channels=channels,
            resonances=[RML_Resonance(er=energy, widths=list(widths))
                        for energy, widths in zip(group.energies, group.widths)],
            # Background R-matrices and phase shifts are held verbatim: the
            # model has no node for either, so they are copied through rather
            # than rebuilt. No tape to hand exercises this.
            background=list(perGroup.get("background", [])),
            phase_shift=perGroup.get("phase_shift"),
        ))

    return RMatrixLimited(
        ifg=int(formalism.reducedWidthAmplitudes),
        krm=fields["krm"],
        krl=int(formalism.relativisticKinematics),
        spi=fields["spi"], ap=fields["ap"],
        particle_pairs=[RML_ParticlePair(**pair) for pair in pairs],
        spin_groups=spinGroups,
    )


def _encodeUnresolved(unresolved: UnresolvedRegion, fields: dict):
    """LRU=2, one of ENDF's three cases. The case is recorded, never re-derived."""
    from kika.endf.classes.mf2.mf2mt151 import (URR_EnergyPoint, URR_JState_CaseA,
                                                URR_JState_CaseB, URR_JState_CaseC,
                                                URR_LValue_CaseA, URR_LValue_CaseB,
                                                URR_LValue_CaseC, UnresolvedCaseA,
                                                UnresolvedCaseB, UnresolvedCaseC)

    widths = unresolved.tabulatedWidths
    case = fields.get("urr_case")
    states = fields.get("j_states") or []
    if case is None:
        raise ValueError("provenance does not record which URR case this range is")
    if len(states) != len(widths.spinGroups):
        raise ValueError(
            f"URR case {case}: {len(widths.spinGroups)} spin groups on the model "
            f"and {len(states)} J-state records in provenance"
        )

    #: (L, awri) → the J-state records of that l-block, in file order.
    blocks: dict = {}
    for group, state in zip(widths.spinGroups, states):
        entry = blocks.setdefault(group.L, {"awri": group.atomicWeightRatio,
                                            "states": []})
        channels = {channel.label: channel for channel in group.channels}
        entry["states"].append((group, state, channels))

    common = dict(spi=fields["spi"], ap=fields["ap"], lssf=fields["lssf"],
                  nls=fields["nls"])

    if case == "A":
        return UnresolvedCaseA(l_values=[
            URR_LValue_CaseA(awri=block["awri"], l=l, j_states=[
                URR_JState_CaseA(
                    d=float(np.asarray(group.levelSpacing)[0]), aj=group.J,
                    amun=channels["neutron"].degreesOfFreedom,
                    gn0=channels["neutron"].constantWidth,
                    gg=channels["capture"].constantWidth,
                )
                for group, _, channels in block["states"]
            ])
            for l, block in blocks.items()
        ], **common)

    if case == "B":
        return UnresolvedCaseB(
            ne=int(np.asarray(widths.energyGrid).size),
            energies=list(np.asarray(widths.energyGrid, dtype=float)),
            l_values=[
                URR_LValue_CaseB(awri=block["awri"], l=l, j_states=[
                    URR_JState_CaseB(
                        d=float(np.asarray(group.levelSpacing)[0]), aj=group.J,
                        amun=channels["neutron"].degreesOfFreedom,
                        gn0=channels["neutron"].constantWidth,
                        gg=channels["capture"].constantWidth,
                        # MUF has nowhere to live on the model when GF is falsy,
                        # because the fission channel is then never built.
                        muf=state["muf"], gf=list(state["gf"]),
                    )
                    for group, state, channels in block["states"]
                ])
                for l, block in blocks.items()
            ], **common)

    if case == "C":
        return UnresolvedCaseC(l_values=[
            URR_LValue_CaseC(awri=block["awri"], l=l, j_states=[
                URR_JState_CaseC(
                    aj=group.J,
                    # INT varies across the corpus (2 on U-235 and Th-232, 5 on
                    # Pu-241), so it is read back and never assumed.
                    int_code=state["int_code"],
                    amux=channels["competitive"].degreesOfFreedom,
                    amun=channels["neutron"].degreesOfFreedom,
                    amug=channels["capture"].degreesOfFreedom,
                    amuf=channels["fission"].degreesOfFreedom,
                    energy_points=[
                        URR_EnergyPoint(es=es, d=d, gx=gx, gn0=gn0, gg=gg, gf=gf)
                        for es, d, gx, gn0, gg, gf in zip(
                            np.asarray(state["energy_grid"], dtype=float),
                            np.asarray(group.levelSpacing, dtype=float),
                            np.asarray(channels["competitive"].widths, dtype=float),
                            np.asarray(channels["neutron"].widths, dtype=float),
                            np.asarray(channels["capture"].widths, dtype=float),
                            np.asarray(channels["fission"].widths, dtype=float),
                        )
                    ],
                )
                for group, state, channels in block["states"]
            ])
            for l, block in blocks.items()
        ], **common)

    raise ValueError(f"URR case {case!r} is not one of ENDF's A, B or C")
