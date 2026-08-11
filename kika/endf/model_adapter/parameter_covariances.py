"""MF32 → §25.3 ``parameterCovariances``.

Separate from :mod:`kika.endf.model_adapter.covariances` for the reason §25.3 is
a separate subsection of the standard: the rows of these matrices are *model
parameters*, not bins of a grid. Nothing about row 47 is recoverable from an
energy axis — it means "the neutron width of the twelfth resonance" or it means
nothing — so the container, the links and the decode all differ, and mixing them
into the cross-section path would eventually collapse a resonance index into an
energy without anything noticing.

**What this module had to measure rather than read.** ENDF-102 §32 leaves three
things ambiguous enough that implementing from the text alone gives a decoder
that is wrong on real tapes, and each was settled against the evaluations on
this machine (the commands are in ``docs/mf32-notes.md``):

1. **The vector is resonance-major.** Every parameter of resonance 1, then every
   parameter of resonance 2. Read parameter-major, Mn-55's block gives resonance
   50 a 67 % uncertainty on GG and resonance 5 a 2.5e-8 one on GN — absurd both
   ways, which is what makes this checkable rather than a matter of taste.
2. **LCOMP=2's matrix does not cover every parameter.** It covers those columns
   carrying a non-zero uncertainty *somewhere in the section*, which is ER, GN
   and GG on both Na-23 and Th-232 — three per resonance, not six. Na-23 makes
   the distinction visible: 69 rows for 65 non-zero uncertainties, so the rule
   is per-column and not per-entry, and a per-entry reading is off by four.
3. **LRF=7 is the exception to (2)** — its NNN counts ``(NCH+1)`` per resonance
   whatever the uncertainties are, zeros included. Cl-35, Cu-63 and W-186 agree.

Every one of those is cross-checked at decode time against NNN, and a section
whose arithmetic disagrees is reported and skipped rather than reshaped into
something that fits.

**What is deliberately not decoded.** LCOMP=1's long-range blocks (§32.2.2.5,
NLRS>0) and LCOMP=1 for LRF=7 (§32.2.2.4). No evaluation on this machine has
either, so a decoder for them would be code no test could reach; they are
reported as unsupported and the rest of the section still converts.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport,
    DataLink,
    ParameterCovariance,
    ParameterCovarianceMatrix,
    ParameterLink,
)

__all__ = ["decodeMF32MT", "resonanceParametersHref",
           "unresolvedParametersHref"]


# ---------------------------------------------------------------------------
# Parameter naming
# ---------------------------------------------------------------------------
#
# The six slots of a File 2 resonance record are positional, and which quantity
# sits in each depends on LRF. `_SLOTS` names them; `_COVERED` gives the order in
# which §32 admits them into a covariance, which is *not* the record order — AJ
# never carries an uncertainty and GT is redundant with GN+GG+GF, so both are
# skipped. MPAR counts down `_COVERED`, so MPAR=3 means ER, GN, GG.

_SLOTS = {
    1: ("ER", "AJ", "GT", "GN", "GG", "GF"),
    2: ("ER", "AJ", "GT", "GN", "GG", "GF"),
    3: ("ER", "AJ", "GN", "GG", "GFA", "GFB"),
}

_COVERED = {
    1: ("ER", "GN", "GG", "GF"),
    2: ("ER", "GN", "GG", "GF"),
    3: ("ER", "GN", "GG", "GFA", "GFB"),
}

#: §32.2.4's average unresolved parameters, in record order and in the order
#: MPAR admits them. AJ is skipped for the same reason as above.
_UNRESOLVED_SLOTS = ("D", "AJ", "GNO", "GG", "GF", "GX")
_UNRESOLVED_COVERED = ("D", "GNO", "GG", "GF", "GX")

#: The four parameters an LCOMP=0 block carries covariances for, and where its
#: twelve variance terms sit relative to them. §32.2.1 orders the terms
#: DE2, DN2, DNDG, DG2, DNDF, DGDF, DF2 and then four null spin terms; the four
#: are null *by construction* and §32.3 procedure 2 says a non-zero one is to be
#: treated as null anyway, so they are dropped rather than carried as zeros.
_LCOMP0_PARAMETERS = ("ER", "GN", "GG", "GF")


def _formalism(lrf: int) -> str:
    """Which model node an LRF's parameters live on.

    §19 splits the resolved region by formalism rather than by record position,
    so LRF is what decides the node — and **LRF=3 is an R-matrix node, not a
    Breit-Wigner one**. Reich-Moore is an R-matrix approximation and
    :func:`kika.endf.model_adapter.resonances._decodeReichMoore` builds an
    ``RMatrix`` for it; a href that said ``BreitWigner`` would point at a node
    that never exists for exactly the evaluations MF32 is most often written
    for (Th-232, Mn-55, Ta-181, Pu-239 are all LRF=3).
    """
    return "BreitWigner" if lrf in (1, 2) else "RMatrix"


def resonanceParametersHref(rangeIndex: int, formalism: str = "BreitWigner") -> str:
    """xPath to the resolved resonance table an MF32 covariance is about.

    Follows the shape of :func:`kika.endf.model_adapter.covariances.reactionHref`
    and carries the same caveat: the path matches what kika's own decoder builds
    (``Resonances.resolved`` is a list here), not a GNDS-mandated spelling.
    """
    return (
        f"/reactionSuite/resonances/resolved[{rangeIndex}]"
        f"/{formalism}/resonanceParameters/table"
    )


def unresolvedParametersHref() -> str:
    """xPath to the unresolved average-parameter table."""
    return "/reactionSuite/resonances/unresolved/tabulatedWidths"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangleToFull(values: Sequence[float], order: int) -> np.ndarray:
    """Expand a row-major upper triangle, diagonal included, to a full matrix."""
    matrix = np.zeros((order, order), dtype=float)
    index = 0
    for row in range(order):
        for column in range(row, order):
            matrix[row, column] = matrix[column, row] = values[index]
            index += 1
    return matrix


def _links(labels: Sequence[str], names: Sequence[str], href: str) -> List[ParameterLink]:
    """One link per resonance, each covering the same ``names`` in order."""
    width = len(names)
    return [
        ParameterLink(label=label, href=href, nParameters=width,
                      matrixStartIndex=index * width,
                      parameterNames=list(names))
        for index, label in enumerate(labels)
    ]


def _covered(lrf: int, report: ConversionReport) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    slots = _SLOTS.get(lrf)
    if slots is None:
        report.unsupportedNode(
            f"MF32: LRF={lrf} has no named parameter slots in this decoder "
            f"(§32 covers LRF=1, 2, 3 through these bodies and LRF=7 through "
            f"its own); the section's matrix is not decoded"
        )
        return None
    return slots, _COVERED[lrf]


# ---------------------------------------------------------------------------
# The five bodies
# ---------------------------------------------------------------------------

def _decodeLCOMP0(body, lrf: int, href: str,
                  report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.1 — 18 numbers a resonance, block-diagonal by resonance.

    Each resonance carries its own 4x4 over (ER, GN, GG, GF) and nothing
    correlates two resonances, which is the whole content of the ENDF/B-V
    format: the matrix this returns is block-diagonal by construction rather
    than by approximation.
    """
    names = _LCOMP0_PARAMETERS
    width = len(names)
    labels: List[str] = []
    blocks: List[np.ndarray] = []
    values: List[float] = []

    for block in body.l_blocks:
        raw = block.values
        count = int(block.n2)
        if len(raw) < 18 * count:
            report.lost(
                f"MF32 LCOMP=0: an L={block.l2} block declares NRS={count} "
                f"but carries {len(raw)} numbers, not {18 * count}"
            )
            return None
        table = np.asarray(raw[:18 * count], dtype=float).reshape(count, 18)
        for row in table:
            er, _aj, _gt, gn, gg, gf = row[:6]
            de2, dn2, dndg, dg2, dndf, dgdf, df2 = row[6:13]
            cell = np.array([
                [de2, 0.0,  0.0,  0.0],
                [0.0, dn2,  dndg, dndf],
                [0.0, dndg, dg2,  dgdf],
                [0.0, dndf, dgdf, df2],
            ], dtype=float)
            blocks.append(cell)
            labels.append(f"L{int(block.l2)}/resonance{len(labels)}")
            values.extend((er, gn, gg, gf))

    if not blocks:
        return None

    order = width * len(blocks)
    matrix = np.zeros((order, order), dtype=float)
    for index, cell in enumerate(blocks):
        start = index * width
        matrix[start:start + width, start:start + width] = cell

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False, parameterValues=np.asarray(values, dtype=float),
    )


def _decodeLCOMP1Block(block, lrf: int, href: str, blockIndex: int,
                       report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.2.1 — one short-range block: NRB resonances, then the triangle."""
    covered = _covered(lrf, report)
    if covered is None:
        return None
    slots, order_names = covered

    mpar = int(block.l1)
    count = int(block.n2)
    if not 1 <= mpar <= len(order_names):
        report.lost(
            f"MF32 LCOMP=1 block {blockIndex}: MPAR={mpar} is outside 1.."
            f"{len(order_names)} for LRF={lrf}"
        )
        return None

    names = order_names[:mpar]
    width = mpar
    npar = width * count
    raw = block.values
    expected = 6 * count + npar * (npar + 1) // 2
    if len(raw) < expected:
        report.lost(
            f"MF32 LCOMP=1 block {blockIndex}: MPAR={mpar} and NRB={count} "
            f"need {expected} numbers, the block carries {len(raw)}"
        )
        return None

    table = np.asarray(raw[:6 * count], dtype=float).reshape(count, 6)
    matrix = _triangleToFull(raw[6 * count:expected], npar)

    columns = [slots.index(name) for name in names]
    values = table[:, columns].reshape(-1)
    labels = [f"block{blockIndex}/resonance{i}" for i in range(count)]

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False, parameterValues=values,
    )


def _decodeLCOMP2(body, lrf: int, href: str,
                  report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.3.1-2 — parameters and uncertainties interleaved, then INTG.

    The covariance is rebuilt as ``D R D``: the file stores the correlations as
    packed integers and the standard deviations beside the parameters, and
    neither half is a covariance on its own.
    """
    covered = _covered(lrf, report)
    if covered is None:
        return None
    slots, order_names = covered

    record = body.parameters
    if record is None:
        return None
    count = int(record.n2)
    raw = record.values
    if count <= 0 or len(raw) < 12 * count:
        report.lost(
            f"MF32 LCOMP=2: NRSA={count} needs {12 * count} numbers, "
            f"the record carries {len(raw)}"
        )
        return None

    table = np.asarray(raw[:12 * count], dtype=float).reshape(count, 12)
    parameters, uncertainties = table[:, :6], table[:, 6:]

    # Which columns the matrix covers: those carrying an uncertainty anywhere in
    # the section. Per column and not per entry -- Na-23 has 69 rows against 65
    # non-zero uncertainties, so a per-entry rule is off by four and every row
    # after the first zero would be misassigned.
    candidates = [slots.index(name) for name in order_names]
    columns = [c for c in candidates if np.any(uncertainties[:, c] != 0.0)]
    if not columns:
        report.lost("MF32 LCOMP=2: no parameter carries an uncertainty")
        return None

    names = tuple(slots[c] for c in columns)
    width = len(columns)
    order = width * count

    if body.correlations is not None and int(body.correlations.nnn) != order:
        report.lost(
            f"MF32 LCOMP=2: the INTG record declares NNN="
            f"{int(body.correlations.nnn)} but {width} parameter(s) "
            f"{'/'.join(names)} over {count} resonances give {order} rows; "
            f"the section is not decoded rather than reshaped to fit"
        )
        return None

    sigma = uncertainties[:, columns].reshape(-1)
    correlation = (body.correlations.correlation_matrix()
                   if body.correlations is not None else np.eye(order))
    matrix = correlation * np.outer(sigma, sigma)

    labels = [f"resonance{i}" for i in range(count)]
    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False,
        parameterValues=parameters[:, columns].reshape(-1),
    )


def _decodeLCOMP2RML(body, href: str,
                     report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.3.3 — R-Matrix Limited: ER plus one width per channel.

    Unlike §32.2.3.2 the matrix covers **every** parameter of every resonance,
    zero uncertainties included: NNN is ``sum (NCH+1) * NRSA`` on Cl-35, Cu-63
    and W-186 alike, and dropping the zero-uncertainty rows here would make the
    order disagree with the file by exactly the number of unassigned channels.

    Each resonance occupies two rows of the LIST padded to a full record, the
    values then the uncertainties, so the stride is a multiple of six rather
    than ``NCH+1``.
    """
    labels: List[str] = []
    names: List[str] = []
    sigma: List[float] = []
    values: List[float] = []
    widths: List[int] = []

    for groupIndex, group in enumerate(body.spin_groups):
        channels = int(group.nch)
        count = int(group.nrsa)
        width = channels + 1
        stride = 6 * ((width + 5) // 6)
        raw = group.resonances.values
        # Equality, not "at least": every tape on this machine has NCH <= 3, so
        # the padding term has never been exercised above one line. If it is
        # wrong for a wider group, NPL is what says so — and a >= test would
        # read the first row of each resonance and silently mean something else.
        if int(group.resonances.n1) != 2 * stride * count or len(raw) < 2 * stride * count:
            report.lost(
                f"MF32 LCOMP=2 LRF=7: spin group {groupIndex} declares "
                f"NCH={channels} NRSA={count}, which needs NPL="
                f"{2 * stride * count} at a stride of {stride}; the record "
                f"declares NPL={int(group.resonances.n1)} and carries "
                f"{len(raw)} numbers"
            )
            return None
        table = np.asarray(raw[:2 * stride * count], dtype=float)
        table = table.reshape(count, 2, stride)
        groupNames = ["ER"] + [f"GAM{c + 1}" for c in range(channels)]
        for resonance in range(count):
            values.extend(table[resonance, 0, :width])
            sigma.extend(table[resonance, 1, :width])
            labels.append(f"spinGroup{groupIndex}/resonance{resonance}")
            names.extend(groupNames)
            widths.append(width)

    if not labels:
        return None

    order = len(sigma)
    if body.correlations is not None and int(body.correlations.nnn) != order:
        report.lost(
            f"MF32 LCOMP=2 LRF=7: the INTG record declares NNN="
            f"{int(body.correlations.nnn)} against sum (NCH+1)*NRSA = {order}"
        )
        return None

    sigmaArray = np.asarray(sigma, dtype=float)
    correlation = (body.correlations.correlation_matrix()
                   if body.correlations is not None else np.eye(order))
    matrix = correlation * np.outer(sigmaArray, sigmaArray)

    links: List[ParameterLink] = []
    start = 0
    for label, width in zip(labels, widths):
        links.append(ParameterLink(
            label=label, href=href, nParameters=width, matrixStartIndex=start,
            parameterNames=names[start:start + width],
        ))
        start += width

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=links, isRelative=False,
        parameterValues=np.asarray(values, dtype=float),
    )


def _decodeUnresolved(body, href: str,
                      report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.4 — average parameters per (L, J), and one **relative** triangle.

    ``NPAR`` is computed rather than read: Th-232 writes 0 in the field §32.2.4
    draws it in, and the triangle it then carries — 120 numbers for MPAR=3 over
    five (L, J) combinations — only makes sense against the computed 15.
    """
    matrixRecord = body.matrix
    if matrixRecord is None:
        report.lost("MF32 LRU=2: no covariance record after the L blocks")
        return None

    mpar = int(matrixRecord.l1)
    if not 1 <= mpar <= len(_UNRESOLVED_COVERED):
        report.lost(
            f"MF32 LRU=2: MPAR={mpar} is outside 1..{len(_UNRESOLVED_COVERED)}"
        )
        return None
    names = _UNRESOLVED_COVERED[:mpar]
    columns = [_UNRESOLVED_SLOTS.index(name) for name in names]

    labels: List[str] = []
    values: List[float] = []
    for block in body.l_blocks:
        raw = block.values
        states = int(block.n2)
        if len(raw) < 6 * states:
            report.lost(
                f"MF32 LRU=2: an L={block.l1} block declares NJS={states} "
                f"but carries {len(raw)} numbers, not {6 * states}"
            )
            return None
        table = np.asarray(raw[:6 * states], dtype=float).reshape(states, 6)
        for state in range(states):
            labels.append(f"L{int(block.l1)}/J{state}")
            values.extend(table[state, columns])

    order = mpar * len(labels)
    triangle = matrixRecord.values
    expected = order * (order + 1) // 2
    if len(triangle) < expected:
        report.lost(
            f"MF32 LRU=2: MPAR={mpar} over {len(labels)} (L, J) states needs "
            f"a triangle of {expected} numbers, the record carries "
            f"{len(triangle)}"
        )
        return None

    return ParameterCovarianceMatrix(
        matrix=_triangleToFull(triangle[:expected], order),
        parameters=_links(labels, names, href),
        isRelative=True,
        parameterValues=np.asarray(values, dtype=float),
    )


# ---------------------------------------------------------------------------
# The descent
# ---------------------------------------------------------------------------

def _rangeLink(href: str, energyRange) -> DataLink:
    """The row link: the parameter table, restricted to this energy range."""
    return DataLink.forIncidentEnergyBand(
        href, float(energyRange.el), float(energyRange.eh),
        ENDF_MFMT="32/151", dimension=1,
    )


def decodeMF32MT(mf32mt, report: Optional[ConversionReport] = None):
    """One MF32/MT151 section → a list of :class:`ParameterCovariance`.

    One per covariance body, which is one per energy range except for LCOMP=1:
    its short-range blocks are independent covariances over disjoint sets of
    resonances, so each becomes its own node rather than being padded into a
    common block-diagonal matrix that would claim the zeros are a statement.
    """
    report = report if report is not None else ConversionReport()
    from .covariances import _sectionProvenance

    provenance = _sectionProvenance(mf32mt)
    covariances: List[ParameterCovariance] = []

    for isotopeIndex, isotope in enumerate(getattr(mf32mt, "isotopes", [])):
        for rangeIndex, energyRange in enumerate(isotope.energy_ranges):
            body = energyRange.body
            if body is None:
                continue
            kind = type(body).__name__
            stem = f"MF32-iso{isotopeIndex}-range{rangeIndex}"

            if energyRange.nro:
                report.unsupportedNode(
                    f"{stem}: NRO={energyRange.nro} gives the scattering radius "
                    f"its own File 33-style covariance (§32.2), which is not "
                    f"decoded; the resonance parameter matrix below is"
                )

            href = (unresolvedParametersHref() if kind == "UnresolvedBody"
                    else resonanceParametersHref(rangeIndex,
                                                 _formalism(energyRange.lrf)))

            if kind == "UnresolvedBody":
                forms = [_decodeUnresolved(body, href, report)]
            elif kind == "LCOMP0Body":
                forms = [_decodeLCOMP0(body, energyRange.lrf, href, report)]
            elif kind == "LCOMP2Body":
                forms = [_decodeLCOMP2(body, energyRange.lrf, href, report)]
            elif kind == "LCOMP2RMLBody":
                forms = [_decodeLCOMP2RML(body, href, report)]
            elif kind == "LCOMP1Body":
                if body.long_range:
                    report.unsupportedNode(
                        f"{stem}: {len(body.long_range)} long-range "
                        f"subsection(s) (§32.2.2.5, NLRS>0) are parsed but not "
                        f"decoded; the short-range blocks are"
                    )
                forms = [
                    _decodeLCOMP1Block(block, energyRange.lrf, href, index, report)
                    for index, block in enumerate(body.short_range)
                ]
            elif kind == "LCOMP1RMLBody":
                report.unsupportedNode(
                    f"{stem}: LCOMP=1 with LRF=7 (§32.2.2.4) is parsed but not "
                    f"decoded — no evaluation on this machine has one, so a "
                    f"decoder for it would be untestable"
                )
                forms = []
            else:
                report.unsupportedNode(f"{stem}: no decoder for a {kind}")
                forms = []

            link = _rangeLink(href, energyRange)
            for formIndex, form in enumerate(forms):
                if form is None:
                    continue
                suffix = f"-block{formIndex}" if len(forms) > 1 else ""
                covariances.append(ParameterCovariance(
                    label=f"{stem}{suffix}",
                    rowData=link,
                    columnData=None,
                    form=form,
                    provenance=provenance,
                ))

    if not covariances:
        report.lost("MF32/MT151: no parameter covariance decoded")
    return covariances, report
