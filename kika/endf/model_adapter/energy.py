"""MF5 ↔ the ``energy`` half of :class:`~kika.nuclear_data.model.distributions.Uncorrelated`.

MF5 is a section header and NK subsections, each an independent law weighted by
``p_k(E)``. **Exactly one of the laws has a model node.** LF=1 is a TAB2 over
incident energy whose nodes are TAB1s in E′, which is GNDS §18.3's ``energy``
child spelled in ENDF; LF=5/7/9/11/12 are the six analytic evaporation and
fission spectra of the same section, and kika models none of them — they are
formulas with named parameters, and tabulating one would put numbers in the file
that the evaluator never wrote. ``MF5PartialRaw`` keeps them as the bytes they
came in as, so there is nothing to fill a node from either.

**NK > 1 is ``weightedFunctionals``, and it is not modelled either** — including
when its NK subsections are all LF=1. A single partial out of a weighted sum is
a *piece* of the distribution, and hanging it on the product as though it were
the distribution would be a false statement that no schema can see. So the model
gets a form only for NK=1 with LF=1, and everything else is declared.

That case is not hypothetical and it is not rare where it occurs. ENDF/B-VIII.1
has **zero** NK>1 sections containing an LF=1 (595 sections measured
2026-08-24: 487 are NK=1/LF=1 and the other 108 are homogeneous LF=5/7/9), so a
reader tested only against it would never meet one. **JEFF-4.0's U-235 states
MF5/MT455 as NK=8 with all eight LF=1** — the per-precursor delayed spectra —
and that is the library the thesis track reads.

**What is kept and where.** The tables — the incident grid, every outgoing grid,
every chi, and the interpolation of each — are the model's. NK, LF, U and
``p_k(E)`` are ENDF bookkeeping with no GNDS counterpart and go to
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`'s ``headerFields``
under an ``"mf5"`` key of their own, beside MF4's flat ``ltt``/``li``/``lct``:
one product carries one provenance, and the two files must not tread on each
other. The verbatim records of a law kika does not model go there too, **as
text**. Keeping the parsed ``MF5PartialRaw`` object instead would put an ENDF
class inside the format-neutral model, which is the accidental format surface
the model exists in order not to have; a list of strings is plain data, and it
is enough to write the section back byte for byte in its original order.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport,
    EndfProvenance,
    Function1d,
    Regions1d,
    energyAxes,
    fromEndfTab2,
    toEndfTab2,
)

__all__ = ["decodeMF5MT", "encodeMF5MT"]


def _za(section) -> Optional[int]:
    """ZA, **rounded**. See :func:`kika.nuclear_data.model.provenance._asEndfInt`."""
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return int(round(float(value))) if value is not None else None


def _verbatimBody(partial, mat, mf, mt) -> List[str]:
    """Columns 1-66 of every record of *partial* after its subsection header.

    Asked of the partial itself rather than read off it, because only
    ``MF5PartialRaw`` stores such a list — a ``MF5PartialTabulated`` holds
    parsed tables. Emitting and slicing gets the same bytes from either, which
    is what makes the provenance uniform over laws kika models and laws it does
    not. The line numbers here are throwaway: ``MF5PartialRaw.emit`` re-stamps
    MAT/MF/MT and the running sequence on the way out, exactly as
    ``MF5MT.__str__`` always did.
    """
    header, _ = partial.emit_header(mat, mf, mt, 1)
    whole, _ = partial.emit(mat, mf, mt, 1)
    return [line[:66] for line in whole[len(header):]]


def _partialRecord(partial, index: int, modelled: bool, mat, mf, mt) -> dict:
    """One subsection's fields, as primitives.

    The tables of the **modelled** partial are not here — they are the model's,
    and the encoder rebuilds the records from them. Every other partial keeps
    its records as the bytes the evaluator wrote, whatever its law: that is
    already what ``MF5PartialRaw`` does for the analytic spectra, and an LF=1
    inside an NK>1 needs exactly the same treatment. It is not a hypothetical —
    JEFF-4.0's U-235 states MF5/MT455 as **NK=8 and all eight LF=1**, the
    per-precursor delayed spectra, and none of them can be the section's
    ``energy`` on its own.
    """
    record = {
        "index": index,
        "lf": int(partial.lf),
        "u": float(partial.u),
        "p_interp": [(int(nbt), int(code)) for nbt, code in partial.p_interp],
        "p_energies": [float(v) for v in partial.p_energies],
        "p_values": [float(v) for v in partial.p_values],
    }
    if not modelled:
        record["raw_lines"] = _verbatimBody(partial, mat, mf, mt)
    return record


def _headerProvenance(mf5mt, modelled: Optional[int]) -> EndfProvenance:
    return EndfProvenance(
        mat=getattr(mf5mt, "_mat", None),
        awr=getattr(mf5mt, "_awr", None),
        za=_za(mf5mt),
        headerFields={
            "mf5": {
                # **MF5's own MAT, ZA and AWR, not the product's.** One
                # `EndfProvenance` per product carries MF4's at the top level,
                # and merging MF5 into it would let one file's header decide
                # the other's. That is not hypothetical: ENDF/B-VIII.1's
                # Ce-140 writes AWR=1.387036+2 in MF5/MT91 and 1.387030+2 in
                # MF4, and Am-243 disagrees with itself the same way in MT18 —
                # two sections out of 595, found by writing whole tapes back
                # and comparing (2026-08-24). Six significant figures apart is
                # still a different byte.
                "mat": getattr(mf5mt, "_mat", None),
                "za": _za(mf5mt),
                "awr": getattr(mf5mt, "_awr", None),
                "nk": mf5mt.num_partials,
                # Which subsection the model form came from, so the encoder puts
                # it back where it was rather than assuming subsection 0.
                "modelled": modelled,
                "partials": [
                    _partialRecord(partial, index, index == modelled,
                                   getattr(mf5mt, "_mat", 0) or 0, 5,
                                   mf5mt.number)
                    for index, partial in enumerate(mf5mt.partials)
                ],
            },
        },
    )


# ---------------------------------------------------------------------------
# The per-incident-energy 1-d functions
# ---------------------------------------------------------------------------

def _spectrumAt(energy: float, outgoing, chi, regions, index: int) -> Function1d:
    """One TAB1 node of an LF=1 → a function over E′.

    A :class:`Regions1d` when the record really has more than one interpolation
    region, and the single :class:`XYs1d` inside it when it does not — the same
    rule, and for the same schema reason, as
    :func:`kika.endf.model_adapter.angular._tabulatedAt`: ``gnds.xsd`` puts
    ``minOccurs="2"`` on the children of ``regions1d``, so a one-region
    ``regions1d`` is a node the schema rejects.

    Both classes answer ``toEndfRegions()``, so the encoder re-emits either
    without asking which one it got.
    """
    pairs = [(int(nbt), int(code)) for nbt, code in regions]
    x = np.asarray(outgoing, dtype=float)
    if not pairs and x.size:
        pairs = [(int(x.size), 2)]
    function = Regions1d.fromEndfRegions(x, np.asarray(chi, dtype=float), pairs)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
    function.outerDomainValue = float(energy)
    function.index = index
    return function


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decodeMF5MT(mf5mt, report: Optional[ConversionReport] = None):
    """One MF5/MT section → ``(energyForm, provenance, report)``.

    ``energyForm`` is an :class:`XYs2d`, a :class:`Regions2d` when the TAB2 has
    more than one region, or ``None`` when the section is NK>1 or a law with no
    model node. The provenance is returned in every case, and in the ``None``
    case it is the *whole* section: that is what lets the encoder write back a
    thing the model never held.
    """
    from kika.endf.classes.mf5.partials import MF5PartialTabulated

    report = report if report is not None else ConversionReport()
    partials = list(mf5mt.partials)
    mt = mf5mt.number

    modelled = None
    if len(partials) == 1 and isinstance(partials[0], MF5PartialTabulated):
        modelled = 0

    provenance = _headerProvenance(mf5mt, modelled)

    if modelled is None:
        if len(partials) > 1:
            laws = ",".join(str(p.lf) for p in partials)
            report.unsupportedNode(
                f"MF5/MT{mt} has NK={len(partials)} subsections (LF=[{laws}]), "
                f"which is GNDS §18.3's weightedFunctionals — a weighted sum of "
                f"laws, and kika has no node for it. The whole energy "
                f"distribution is absent from this reactionSuite, including any "
                f"LF=1 among them: one partial of a weighted sum is not the "
                f"distribution. The section's own bytes are kept, so the tape "
                f"still comes back with it"
            )
        else:
            # An NK=1 that is not LF=1 is one of ENDF-6 §5's parametrised
            # spectra. This used to defer to `MF5MT.report_gaps`, on the
            # grounds that it named the law already -- and that deferral broke
            # the moment the reader learned to decode LF=5/7/9/11, because
            # `report_gaps` reports what the *reader* could not read and this
            # report is about what the *model* did not receive. They were the
            # same list once and are not any more: kika now evaluates these
            # spectra and still has no §18 node to put them in, so a section
            # that is fully read is still fully absent from here. Saying it
            # from the model's own side is the only version that stays true.
            partial = partials[0]
            report.unsupportedNode(
                f"MF5/MT{mt} is NK=1 LF={partial.lf} ({partial.describe()}), "
                f"one of ENDF-6 §5's parametrised spectra. kika "
                f"{'reads' if partial.is_decoded else 'does not read'} it, and "
                f"has no GNDS §18 node to carry it either way, so the energy "
                f"distribution is absent from this reactionSuite. The section's "
                f"own bytes are kept, so the tape still comes back with it"
            )
        return None, provenance, report

    partial = partials[0]
    if partial.p_values and not all(value == 1.0 for value in partial.p_values):
        report.warn(
            f"MF5/MT{mt} has NK=1 and its p(E) is not identically 1, which "
            f"ENDF-6 §5 requires of a single subsection; the model carries the "
            f"law and the weight stays in the provenance, so the round trip is "
            f"unaffected, but the two do not multiply to what the file states"
        )

    # One object, shared by the container and everything under it. Calling
    # `energyAxes()` per region would give distinct objects, and
    # `kika/gnds/encode.py:_axesUnlessNested` tests inheritance by *identity* --
    # so a second object would be read as "this child carries axes of its own".
    axes = energyAxes()

    functions = [
        _spectrumAt(energy, partial.outgoing_grids[k], partial.chi[k],
                    partial.outgoing_interp[k], k)
        for k, energy in enumerate(partial.incident_energies)
    ]
    energy = fromEndfTab2(functions, partial.tab2_interp, axes=axes)
    return energy, provenance, report


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _spectrumRecord(function: Function1d):
    x, y, pairs = function.toEndfRegions()
    return ([float(v) for v in x], [float(v) for v in y],
            [(int(a), int(b)) for a, b in pairs])


def _restorePartial(record: dict, energyForm):
    """One provenance record → the ENDF partial it was read from."""
    from kika.endf.classes.mf5.partials import (MF5PartialRaw,
                                                MF5PartialTabulated)

    common = dict(
        u=record["u"],
        lf=record["lf"],
        p_interp=[(int(a), int(b)) for a, b in record["p_interp"]],
        p_energies=list(record["p_energies"]),
        p_values=list(record["p_values"]),
    )

    if energyForm is None:
        return MF5PartialRaw(raw_lines=list(record.get("raw_lines", [])), **common)

    functions, pairs = toEndfTab2(energyForm)
    records = [_spectrumRecord(function) for function in functions]
    return MF5PartialTabulated(
        tab2_interp=pairs,
        incident_energies=[float(f.outerDomainValue) for f in functions],
        outgoing_grids=[r[0] for r in records],
        chi=[r[1] for r in records],
        outgoing_interp=[r[2] for r in records],
        **common,
    )


def encodeMF5MT(energyForm, provenance: Optional[EndfProvenance], mt: int,
                report: Optional[ConversionReport] = None):
    """The inverse of :func:`decodeMF5MT`, byte-identical to the source section.

    Requires the ENDF provenance. NK, LF, U and ``p_k(E)`` have no GNDS
    counterpart, and the laws kika does not model live there and nowhere else —
    so without it this is not a lossy encode, it is an impossible one.
    """
    from kika.endf.classes.mf5.base import MF5MT

    report = report if report is not None else ConversionReport()
    fields = (provenance.headerFields.get("mf5")
              if provenance is not None and provenance.sourceFormat == "endf"
              else None)
    if fields is None:
        raise ValueError(
            "encodeMF5MT needs the EndfProvenance decodeMF5MT produced: NK, LF, "
            "U and p(E) are not recoverable from the model alone, and neither "
            "are the laws it does not model"
        )

    modelled = fields.get("modelled")
    if (modelled is None) != (energyForm is None):
        raise ValueError(
            f"MT{mt}: the provenance says subsection {modelled!r} carried the "
            f"model form and it was handed "
            f"{'nothing' if energyForm is None else type(energyForm).__name__}"
        )

    section = MF5MT(number=mt)
    # From the MF5 block and not from the provenance's top level, which is
    # MF4's — see `_headerProvenance`. Older provenances have no such keys, so
    # the top level is the fallback rather than the source.
    za = fields.get("za", provenance.za)
    section._za = float(za) if za is not None else None
    section._awr = fields.get("awr", provenance.awr)
    section._mat = fields.get("mat", provenance.mat)
    section._nk = fields["nk"]
    section.partials = [
        _restorePartial(record, energyForm if index == modelled else None)
        for index, record in enumerate(fields["partials"])
    ]
    return section, report
