"""MF4 ↔ :class:`~kika.nuclear_data.model.distributions.AngularTwoBody`.

MF4 is four ENDF representations behind one MT number — isotropic (LTT=0 with
LI=1), Legendre coefficients (LTT=1), tabulated P(mu) (LTT=2), and both at once
over adjacent energy ranges (LTT=3). GNDS has a node for each: ``isotropic2d``,
and an ``angularTwoBody`` whose ``angular`` is an ``XYs2d`` of ``Legendre`` or
of ``XYs1d``, or a ``regions2d`` when the file uses more than one region.

**The gate is against the file, not against the flat class.** Everywhere else in
phase 3c the acceptance is "the model reproduces what the flat path produces",
because the flat path is the code being replaced and agreeing with it is the
whole claim. MF4 is the exception, and deliberately:
``AngularDistribution.to_endf()`` **cannot** reproduce an LTT=3 section — it
drops NM and it trims trailing zero coefficients, so 141 lines of the committed
Fe-56 slice come back different. Matching that would mean reproducing two
defects. So this encoder is gated against ``str(section)`` itself, which is a
strictly stronger statement, and the flat path's loss is pinned separately as an
``xfail`` in ``tests/test_angular_round_trip.py``.

**What is kept and where.** The physics — the coefficients, the cosine grids,
the interpolation of every region — is in the model. ENDF's LTT, LI, LCT and NM
are format bookkeeping with no GNDS counterpart and go to
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`'s ``headerFields``,
so the encoder writes them back rather than re-deriving them. LCT is the one
that is *also* physics: it is the frame, so it appears in both places —
``productFrame`` for anything that reads the model, ``headerFields['lct']`` so
the byte-level round trip does not depend on the mapping being invertible.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (
    AngularTwoBody,
    ConversionReport,
    EndfProvenance,
    Frame,
    Function1d,
    Isotropic2d,
    Legendre,
    Regions1d,
    Regions2d,
    XYs2d,
    fromEndfTab2,
    toEndfTab2,
)

__all__ = ["decodeMF4MT", "encodeMF4MT", "LCT_TO_FRAME", "FRAME_TO_LCT"]

#: ENDF LCT (§MF4) ↔ GNDS §3.4.2 ``frame``.
LCT_TO_FRAME = {1: Frame.lab, 2: Frame.centerOfMass}
FRAME_TO_LCT = {frame: lct for lct, frame in LCT_TO_FRAME.items()}


def _za(section) -> Optional[int]:
    """ZA, **rounded**. See :func:`kika.nuclear_data.model.provenance._asEndfInt`."""
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return int(round(float(value))) if value is not None else None


def _headerProvenance(mf4mt) -> EndfProvenance:
    return EndfProvenance(
        mat=getattr(mf4mt, "_mat", None),
        awr=mf4mt.atomic_weight_ratio,
        za=_za(mf4mt),
        headerFields={
            "ltt": mf4mt._ltt,
            "li": mf4mt._li,
            "lct": mf4mt._lct,
            # NM — the highest Legendre order the evaluation used. Only LTT=3
            # writes it, and only in that section's second CONT record.
            "nm": getattr(mf4mt, "_nm", None),
        },
    )


# ---------------------------------------------------------------------------
# The per-energy 1-d functions
# ---------------------------------------------------------------------------

def _legendreAt(energy: float, row: Sequence[float], index: int) -> Legendre:
    """One ENDF Legendre record → a :class:`Legendre`.

    ENDF stores ``a_1 .. a_NL`` and leaves ``a_0 = 1`` implicit (the PDF is
    normalised). GNDS stores every coefficient, so ``a_0`` is written out. The
    row is otherwise copied verbatim — **including a trailing zero**, which is
    not padding: NL is the order the evaluator declared, and the flat path's
    habit of trimming it changes NL on 70 of this tape's 3960 energies.
    """
    coefficients = np.empty(len(row) + 1, dtype=float)
    coefficients[0] = 1.0
    coefficients[1:] = np.asarray(row, dtype=float)
    return Legendre(coefficients=coefficients, outerDomainValue=float(energy), index=index)


def _tabulatedAt(energy: float, cosines, probabilities, regions, index: int):
    """One ENDF tabulated record → a function over mu.

    A :class:`Regions1d` when the record really has more than one interpolation
    region, and the single :class:`XYs1d` inside it when it does not. **One
    region is not a region set**: ``function1ds_inRegions``
    (``gnds.xsd:2143-2147``) puts ``minOccurs="2"`` on its children, so a
    ``regions1d`` wrapping one region is a node the schema rejects — and it is
    what every one of this evaluation's tabulated energies produced.

    This is the 1-d twin of a rule the 2-d side already had: ``fromEndfTab2``
    (``functions/higher.py:240-241``) returns an ``XYs2d`` when NR<=1 and a
    ``Regions2d`` only when there is more than one region. It is also what the
    committed fixtures carry — ``micro_fe56.gnds.xml`` writes the tabulated half
    of its LTT=3 section as plain ``<XYs1d outerDomainValue=...>`` children.

    Both classes answer ``toEndfRegions()``, so ``_tabulatedRecord`` below
    re-emits either without asking which one it got.
    """
    pairs = [(int(nbt), int(code)) for nbt, code in regions]
    mu = np.asarray(cosines, dtype=float)
    if not pairs and mu.size:
        pairs = [(int(mu.size), 2)]
    function = Regions1d.fromEndfRegions(mu, np.asarray(probabilities, dtype=float), pairs)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
    function.outerDomainValue = float(energy)
    function.index = index
    return function


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decodeMF4MT(mf4mt, report: Optional[ConversionReport] = None):
    """One MF4/MT section → ``(distribution, provenance, report)``.

    ``distribution`` is an :class:`Isotropic2d` for LTT=0/LI=1 and an
    :class:`AngularTwoBody` otherwise.
    """
    report = report if report is not None else ConversionReport()
    provenance = _headerProvenance(mf4mt)
    frame = LCT_TO_FRAME.get(mf4mt._lct)
    if frame is None:
        report.warn(
            f"MT{mf4mt.number}: MF4 LCT={mf4mt._lct} is neither 1 (lab) nor 2 "
            f"(centre of mass); the product frame is left as centre of mass"
        )
        frame = Frame.centerOfMass

    ltt = mf4mt._ltt

    if ltt == 0:
        if mf4mt._li != 1:
            report.warn(
                f"MT{mf4mt.number}: MF4 has LTT=0 but LI={mf4mt._li}, so it "
                f"declares no representation and no isotropy; read as isotropic"
            )
        return Isotropic2d(productFrame=frame), provenance, report

    if ltt == 1:
        functions = [
            _legendreAt(energy, row, i)
            for i, (energy, row) in enumerate(
                zip(mf4mt.legendre_energies, mf4mt.legendre_coefficients)
            )
        ]
        angular = fromEndfTab2(functions, mf4mt.energy_interpolation)

    elif ltt == 2:
        functions = [
            _tabulatedAt(energy, mf4mt.cosines[i], mf4mt.probabilities[i],
                         _angularRegions(mf4mt, i), i)
            for i, energy in enumerate(mf4mt.energies)
        ]
        angular = fromEndfTab2(functions, mf4mt.energy_interpolation)

    elif ltt == 3:
        legendre = [
            _legendreAt(energy, row, i)
            for i, (energy, row) in enumerate(
                zip(mf4mt.legendre_energies, mf4mt.legendre_coefficients)
            )
        ]
        tabulated = [
            _tabulatedAt(energy, mf4mt.tabulated_cosines[i],
                         mf4mt.tabulated_probabilities[i],
                         _angularRegions(mf4mt, i), i)
            for i, energy in enumerate(mf4mt.tabulated_energies)
        ]
        angular = Regions2d(function2ds=[
            fromEndfTab2(legendre, getattr(mf4mt, "_interpolation", [])),
            fromEndfTab2(tabulated, getattr(mf4mt, "_tab_interpolation", [])),
        ])

    else:
        report.unsupportedNode(
            f"MT{mf4mt.number}: MF4 LTT={ltt} is not one of 0, 1, 2, 3; the "
            f"angular distribution is absent from this reactionSuite"
        )
        return None, provenance, report

    return AngularTwoBody(angular=angular, productFrame=frame), provenance, report


def _angularRegions(mf4mt, index: int) -> List[Tuple[int, int]]:
    rows = getattr(mf4mt, "_angular_interpolation", []) or []
    return rows[index] if index < len(rows) else []


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _legendreRow(function: Function1d) -> List[float]:
    """``a_1 .. a_NL`` — the model keeps ``a_0``, the file does not."""
    return [float(c) for c in np.asarray(function.coefficients, dtype=float)[1:]]


def _tabulatedRecord(function: Function1d):
    mu, p, pairs = function.toEndfRegions()
    return [float(v) for v in mu], [float(v) for v in p], [(int(a), int(b)) for a, b in pairs]


def _applyHeader(obj, provenance: EndfProvenance) -> None:
    header = provenance.headerFields
    obj._za = float(provenance.za) if provenance.za is not None else None
    obj._awr = provenance.awr
    obj._mat = provenance.mat
    obj._li = header.get("li")
    obj._lct = header.get("lct")


def encodeMF4MT(distribution, provenance: EndfProvenance, mt: int,
                report: Optional[ConversionReport] = None):
    """The inverse of :func:`decodeMF4MT`, byte-identical to the source section.

    Requires the ENDF provenance: LTT, LI, LCT and NM have no GNDS counterpart,
    and re-deriving LTT from the shape of the model would be a guess that
    happens to be right on the tapes tested and wrong on the first one that is
    not.
    """
    report = report if report is not None else ConversionReport()
    if provenance is None or provenance.sourceFormat != "endf":
        raise ValueError(
            "encodeMF4MT needs the EndfProvenance decodeMF4MT produced: LTT, LI "
            "and LCT are not recoverable from the model alone"
        )

    ltt = provenance.headerFields.get("ltt")

    if ltt == 0 or isinstance(distribution, Isotropic2d):
        from kika.endf.classes.mf4.isotropic import MF4MTIsotropic

        obj = MF4MTIsotropic(number=mt)
        _applyHeader(obj, provenance)
        return obj, report

    if not isinstance(distribution, AngularTwoBody) or distribution.angular is None:
        raise ValueError(
            f"MT{mt}: LTT={ltt} needs an angularTwoBody with an angular form, "
            f"got {type(distribution).__name__}"
        )

    if ltt == 1:
        from kika.endf.classes.mf4.polynomial import MF4MTLegendre

        obj = MF4MTLegendre(number=mt)
        _applyHeader(obj, provenance)
        functions, pairs = toEndfTab2(distribution.angular)
        obj._energies = [float(f.outerDomainValue) for f in functions]
        obj._legendre_coeffs = [_legendreRow(f) for f in functions]
        obj._ne = len(functions)
        obj._interpolation = pairs
        obj._nr = len(pairs)
        return obj, report

    if ltt == 2:
        from kika.endf.classes.mf4.tabulated import MF4MTTabulated

        obj = MF4MTTabulated(number=mt)
        _applyHeader(obj, provenance)
        functions, pairs = toEndfTab2(distribution.angular)
        records = [_tabulatedRecord(f) for f in functions]
        obj._energies = [float(f.outerDomainValue) for f in functions]
        obj._cosines = [r[0] for r in records]
        obj._probabilities = [r[1] for r in records]
        obj._angular_interpolation = [r[2] for r in records]
        obj._ne = len(functions)
        obj._interpolation = pairs
        obj._nr = len(pairs)
        return obj, report

    if ltt == 3:
        from kika.endf.classes.mf4.mixed import MF4MTMixed

        angular = distribution.angular
        if not isinstance(angular, Regions2d) or len(angular) != 2:
            raise ValueError(
                f"MT{mt}: LTT=3 is two TAB2 records, so its model form is a "
                f"regions2d with exactly two children; got "
                f"{type(angular).__name__} with {len(angular)}"
            )
        legendrePart, tabulatedPart = angular[0], angular[1]

        obj = MF4MTMixed(number=mt)
        _applyHeader(obj, provenance)
        obj._nm = provenance.headerFields.get("nm")

        legendreFunctions, legendrePairs = toEndfTab2(legendrePart)
        obj._energies = [float(f.outerDomainValue) for f in legendreFunctions]
        obj._legendre_coeffs = [_legendreRow(f) for f in legendreFunctions]
        obj._ne1 = len(legendreFunctions)
        obj._interpolation = legendrePairs
        obj._nr = len(legendrePairs)

        tabulatedFunctions, tabulatedPairs = toEndfTab2(tabulatedPart)
        records = [_tabulatedRecord(f) for f in tabulatedFunctions]
        obj._tabulated_energies = [float(f.outerDomainValue) for f in tabulatedFunctions]
        obj._tabulated_cosines = [r[0] for r in records]
        obj._tabulated_probabilities = [r[1] for r in records]
        obj._angular_interpolation = [r[2] for r in records]
        obj._ne2 = len(tabulatedFunctions)
        obj._tab_interpolation = tabulatedPairs
        obj._nr_tab = len(tabulatedPairs)
        obj._ne = obj._ne1 + obj._ne2
        return obj, report

    raise ValueError(f"MT{mt}: cannot encode MF4 with LTT={ltt}")
