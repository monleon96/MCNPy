"""Model → a whole ENDF-6 tape. The direction that did not exist.

**What this closes.** ``kika/endf/model_adapter`` has held per-section encoders
for a while — ``encodeMF1MT451``, ``encodeMF2MT151``, ``encodeMF3MT``,
``encodeMF4MT``, ``encodeMF1MT452/455/456``, ``encodeMF31MT``, ``encodeMF33MT``,
``encodeMF34MT``, ``encodeMF35MT`` — each gated byte-exact against the file it
came from. What was missing was never the rendering; it was the **assembly**:
the order sections go in, the SEND/FEND/MEND/TEND bookkeeping, the tape
identification record and the MF1/451 directory. That is what this module is.
``docs/library/gnds_endf_conflicts.md`` §2.8, and §10 for why the sampling pipeline
cares.

**The gate is a fixed point inside the model, not byte identity against the
source tape** (decided 2026-08-13, owner Juan)::

    read(tape) → suite → writeEndfTape(suite, tape') → read(tape') → suite'
    assert suite' == suite

Deliberately weaker than byte identity, and deliberately so: where a field is
padded and whether ``1e-5`` is written ``1.0-5`` carry no information, and
chasing them is a different job from being correct. What the fixed point does
catch is the only thing that matters here — **a quantity that does not survive
the trip**. Its blind spot is stated in §2.8 and is real: anything the model
does not carry is absent from both sides and the comparison passes. So the
fixed point is necessary and not sufficient, and it leans on
:class:`~kika.nuclear_data.model.conversion.ConversionReport` being honest about
what did not come through.

**Sections come only from what the model has.** MF5's analytic spectra are
not in it — LF=5, 7, 9, 11 and 12 are §18.3's six formulas, which the model has
no node for — and neither is MF7 or MF12-15. A tape carrying those comes back
without them, reported loudly rather than written as an empty shell: a tape
missing its energy distributions and not saying so is worse than one that says
so. MF5's LF=1 does come back, and so does **all of MF6**: what the model does
not carry there — LAW=5, and any subsection whose LAW is negative — is kept
verbatim in the reaction's provenance, so the section is re-emitted whole even
where the distribution never reached a node.

**Two limits that no amount of code removes**, both format facts rather than
gaps here: a GNDS reaction may have no MT at all (§2.4), and MF13 is reached
from MF3 by arithmetic rather than by re-encoding (§2.6). Both are reported per
occurrence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..utils import (format_endf_fend_record, format_endf_mend_record,
                     format_endf_tend_record)

__all__ = ["MF_WRITE_ORDER", "TAPE_ID_MAT", "DEFAULT_TAPE_ID",
           "encodeTapeSections", "assembleTape", "writeEndfTape"]

#: The MF numbers an encoder exists for, in the order ENDF-6 puts them on the
#: tape. Ascending, which is also §0.3.2's rule, so the constant is a statement
#: of *coverage* rather than of order: MF7, MF12-15 and MF32 are absent because
#: nothing can write them, not because they sort late.
MF_WRITE_ORDER = (1, 2, 3, 4, 5, 6, 31, 33, 34, 35)

#: The MAT column of a tape identification record. ENDF-6 §0.6.2 fixes it at 1
#: regardless of the material that follows.
TAPE_ID_MAT = 1

#: What goes in a TPID's 66 text columns when the caller names nothing. **The
#: label is not in the model**: ``read_endf`` does not keep the first line of
#: the tape, so a round trip cannot reproduce it and this module does not
#: pretend to. It is reported, not silently invented.
DEFAULT_TAPE_ID = "TAPE WRITTEN BY KIKA FROM A REACTION SUITE"


def _idColumns(mat: int, mf: int, mt: int, ns: int) -> str:
    """Columns 67-80 — MAT, MF, MT, NS — in their ENDF-6 field widths."""
    return f"{mat:>4}{mf:>2}{mt:>3}{ns:>5}"


def _tapeIdRecord(label: str) -> str:
    """The TPID record (§0.6.2): 66 text columns, MAT=1, MF=MT=NS=0."""
    return f"{label[:66]:<66}" + _idColumns(TAPE_ID_MAT, 0, 0, 0)


def _mat(suite, mat: Optional[int]) -> int:
    """The material number to write in every ID field."""
    if mat is not None:
        return int(mat)
    provenance = getattr(suite, "provenance", None)
    recorded = getattr(provenance, "mat", None)
    if recorded is None:
        raise ValueError(
            "this reactionSuite carries no MAT number -- its provenance has "
            "none, which means it did not come from an ENDF tape -- and every "
            "record of an ENDF file is stamped with one. Pass mat= explicitly."
        )
    return int(recorded)


def _mf1Sections(suite, mat, report, label=None):
    """MF1: the 451 header first, then whichever nu-bars the suite carries.

    *label* selects which §9.1 form of each nu-bar is written, with the same
    fall-back and the same report line MF3 has: a realisation that perturbed the
    prompt nu-bar and not the delayed one still writes a whole tape, and the
    report says which members carried the label and which fell back. That is
    what a ``multiplicity`` becoming a
    :class:`~kika.nuclear_data.model.component.Component` bought -- before it,
    a realisation had to *replace* the evaluated form and there was no label for
    this function to be told about.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import (encodeMF1MT451, encodeMF1MT452,
                                 encodeMF1MT455, encodeMF1MT456)
    from ..model_adapter.multiplicity import nubarNode

    label = EVAL_LABEL if label is None else label

    sections = []
    header, report = encodeMF1MT451(suite, mat, report)
    sections.append((1, 451, header))

    # 452, 455, 456 -- ascending, which is the order they sit in on the tape.
    # `nubarNode` rather than a try/except around each encoder: the encoders
    # raise on absence, and absence is the common case (nothing but a fissile
    # material has any of these).
    written, fellBack = [], []
    for mt, encode in ((452, encodeMF1MT452), (455, encodeMF1MT455),
                       (456, encodeMF1MT456)):
        node = nubarNode(suite, mt)
        if node is None:
            continue
        formLabel = label if label in node else EVAL_LABEL
        (written if formLabel == label else fellBack).append(mt)
        section, report = encode(suite, mat, report, label=formLabel)
        sections.append((1, mt, section))

    if label != EVAL_LABEL and (written or fellBack):
        report.warn(
            f"nu-bar written with the {label!r} form where there is one: "
            f"MT {written or 'none'} carry it, MT {fellBack or 'none'} fell back "
            f"to {EVAL_LABEL!r} because they have no {label!r} form"
        )
    return sections, report


def _mf2Sections(suite, mat, report):
    """MF2/151, when the suite has resonances **and** the provenance to write them."""
    from ..model_adapter import encodeMF2MT151

    resonances = getattr(suite, "resonances", None)
    if resonances is None:
        return [], report

    provenance = getattr(resonances, "provenance", None)
    if provenance is None:
        report.lost(
            "the suite carries resonances but no ENDF provenance for them, so "
            "MF2/151 cannot be written: QX, LRX, LAD and the particle-pair "
            "columns have no model node and would have to be invented"
        )
        return [], report

    section = encodeMF2MT151(resonances, provenance, report)
    # `encodeMF2MT151` returns the section alone; the report it was handed is
    # the one it wrote into.
    return [(2, 151, section)], report



def _mf3Bearing(suite):
    """Every reaction that owns an MF3 section, in tape order.

    ``reactions`` and ``sums`` -- §21.1's ``crossSectionSums`` are reactions in
    every respect the ENDF writer cares about, and sorting by MT rather than
    concatenating the two lists is what keeps the sections in the order a tape
    states them, which is the order the round-trip gate compares.
    """
    both = list(suite.reactions) + list(suite.sums)
    return sorted(both, key=lambda r: (r.ENDF_MT is None, r.ENDF_MT or 0))


def _mf3And4And5Sections(suite, mat, report, label=None):
    """MF3 for every reaction with an MT, MF4 and MF5 for what states them.

    **The provenance decides which sections are written, not the model's
    shape.** An `uncorrelated` whose angular half kika *inferred* — the tape
    said MF5 and no MF4 — has no `ltt` in its provenance, and writing an MF4 for
    it would put a section on the tape the source never carried. The same rule
    the MF4 encoder already lives by, applied one level up.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import encodeMF3MT, encodeMF4MT, encodeMF5MT

    label = EVAL_LABEL if label is None else label

    mf3, mf4, mf5 = [], [], []
    written, fellBack = [], []
    # `reactions` and `sums` both, because a tape's MF3 does not distinguish
    # them: §21.1 is a statement about what MT1 *means*, not about whether it is
    # in the file. Reading an evaluation and writing it back has to give the
    # sections back, wherever the model chose to keep them.
    for reaction in _mf3Bearing(suite):
        mt = reaction.ENDF_MT
        if mt is None:
            # §2.4, and it is irreducible: GNDS labels reactions and does not
            # require an MT, so a suite that did not come from ENDF may hold a
            # reaction there is no section number for.
            report.lost(
                f"reaction {reaction.label!r} has no ENDF MT, so it has no MF3 "
                f"section to be written into (gnds_endf_conflicts.md §2.4)"
            )
            continue

        # A reaction the caller did not perturb has no form under the
        # realization's label. Falling back to `eval` keeps the tape complete;
        # refusing would mean a partially perturbed suite could not be written
        # at all, and writing only the perturbed sections would be a tape with
        # holes in it.
        formLabel = label if label in reaction.crossSection else EVAL_LABEL
        (written if formLabel == label else fellBack).append(mt)
        section, report = encodeMF3MT(reaction, mat, report, label=formLabel)
        mf3.append((3, mt, section))

        product = _neutronProduct(reaction)
        form = _evaluatedForm(product, label)
        if form is None and label != EVAL_LABEL:
            form = _evaluatedForm(product, EVAL_LABEL)
        provenance = getattr(product, "provenance", None)
        header = getattr(provenance, "headerFields", None) or {}

        angular = _mf4Form(form)
        if angular is not None and "ltt" in header:
            section, report = encodeMF4MT(angular, provenance, mt, report)
            mf4.append((4, mt, section))

        if "mf5" in header:
            section, report = encodeMF5MT(_mf5Form(form), provenance, mt, report)
            mf5.append((5, mt, section))

    if label != EVAL_LABEL:
        # **A mixed tape has to say it is mixed.** The fallback above is right --
        # a partially perturbed suite must still produce a whole tape -- but the
        # file it produces cannot be told apart from a fully perturbed one by
        # reading it, and for an ensemble that distinction is the traceability.
        # So the report states both halves, and states them even when the
        # fallback did not fire: "0 fell back" is a claim, and its absence is not.
        report.warn(
            f"written with the {label!r} form where there is one: "
            f"MT {written or 'none'} carry it, MT {fellBack or 'none'} fell back "
            f"to {EVAL_LABEL!r} because they have no {label!r} form"
        )

    return mf3 + mf4 + mf5, report


def _neutronProduct(reaction):
    """The ``n`` product of a reaction's output channel, or ``None``."""
    channel = getattr(reaction, "outputChannel", None)
    for product in getattr(channel, "products", None) or ():
        if getattr(product, "pid", None) == "n":
            return product
    return None


def _evaluatedForm(product, label):
    """The evaluated distribution of *product*, when it has one."""
    distribution = getattr(product, "distribution", None)
    if distribution is None:
        return None
    try:
        return distribution[label]
    except (KeyError, TypeError):
        return None


def _mf4Form(form):
    """The part of *form* MF4 states, in the shape ``encodeMF4MT`` takes.

    An `uncorrelated` is one GNDS node built from two ENDF files, so it is
    taken apart here rather than in the encoder — `encodeMF4MT` takes what MF4
    itself carries and would have to learn §18.3 to take anything else.

    **The angular half goes back inside an `angularTwoBody`**, which is not
    ceremony: §18.3 stores the `XYs2d` directly under `<angular>` while §18.2
    wraps it, and the encoder dispatches on the wrapper. An `isotropic2d` needs
    no wrapper — it is a distribution form in its own right.
    """
    from kika.nuclear_data.model import AngularTwoBody, Isotropic2d, Uncorrelated

    if isinstance(form, Uncorrelated):
        if form.angular is None or isinstance(form.angular, Isotropic2d):
            return form.angular
        return AngularTwoBody(angular=form.angular,
                              productFrame=form.productFrame)
    if isinstance(form, (AngularTwoBody, Isotropic2d)):
        return form
    return None


def _mf5Form(form):
    """The part of *form* MF5 states: the energy distribution, or ``None``.

    ``None`` is a real answer and not an absence: a section whose every law
    kika does not model round-trips out of the provenance alone, and
    :func:`~kika.endf.model_adapter.energy.encodeMF5MT` requires being told so.
    """
    from kika.nuclear_data.model import Uncorrelated

    return form.energy if isinstance(form, Uncorrelated) else None



def _mf6Sections(suite, mat, report):
    """MF6 for every reaction whose provenance carries one.

    **The provenance decides, and it has to.** An MF6 section is a list of
    products in the evaluator's order with the evaluator's ``ZAP``/``AWP``/
    ``LIP``, its ``JP`` and its ``LCT``, and none of that is recoverable from a
    channel's products: the model holds the physics and the file holds the
    bookkeeping. So a suite that never read an MF6 writes none, and one that
    did writes exactly the section it read.

    Nothing here overlaps :func:`_mf3And4And5Sections`. That one keys on the
    *product's* provenance (``ltt`` for MF4, ``mf5`` for MF5) and this one on
    the *reaction's*, and an MT stated in File 6 does not restate its
    distributions in Files 4 and 5 — except through a negative LAW, which is a
    pointer rather than a duplicate, and which this adapter deliberately leaves
    to those two passes.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import encodeMF6MT

    sections = []
    for reaction in _mf3Bearing(suite):
        provenance = getattr(reaction, "provenance", None)
        header = getattr(provenance, "headerFields", None) or {}
        if "mf6" not in header:
            continue

        mt = reaction.ENDF_MT
        if mt is None:
            report.lost(
                f"reaction {reaction.label!r} carries an MF6 provenance and no "
                f"ENDF MT, so there is no section number to write it under "
                f"(gnds_endf_conflicts.md §2.4)"
            )
            continue

        forms = {}
        for product in reaction.outputChannel.products:
            form = _evaluatedForm(product, EVAL_LABEL)
            if form is not None:
                forms[product.label or product.pid] = form

        section, report = encodeMF6MT(forms, provenance, mt, report)
        sections.append((6, mt, section))

    return sections, report


def _covarianceSections(suite, mat, report):
    """MF31/33/34/35, one section per (MF, row MT) the covariance suite carries."""
    from ..model_adapter import (encodeMF31MT, encodeMF33MT, encodeMF34MT,
                                 encodeMF35MT)

    covarianceSuite = getattr(suite, "covarianceSuite", None)
    if covarianceSuite is None:
        return [], report

    encoders = {31: encodeMF31MT, 33: encodeMF33MT,
                34: encodeMF34MT, 35: encodeMF35MT}

    # One pass over the sections to learn which (MF, MT) pairs exist, because
    # the encoders take an MT and there is no listing of them anywhere else.
    # MF32 is deliberately absent from `encoders`: it decodes and has no
    # encoder, so it is declared rather than skipped.
    present = set()
    for section in getattr(covarianceSuite, "covarianceSections", ()):
        row = getattr(section, "rowData", None)
        if row is None:
            continue
        mf, mt = getattr(row, "ENDF_MF", None), getattr(row, "ENDF_MT", None)
        if mf is not None and mt is not None:
            present.add((int(mf), int(mt)))

    sections = []
    for mf, mt in sorted(present):
        encode = encoders.get(mf)
        if encode is None:
            report.unsupportedNode(
                f"MF{mf}/MT{mt} is in the covarianceSuite and has no encoder, "
                f"so it is absent from the written tape"
            )
            continue
        section, report = encode(covarianceSuite, mt, mat, report)
        sections.append((mf, mt, section))

    # §25.3 lives in its own container, and it is **not** reachable through the
    # loop above -- `parameterCovariances` are not `covarianceSections`, so a
    # tape whose only covariance is MF32 produced an empty `present` and said
    # nothing at all. `decodeMF32MT` exists and has no inverse
    # (`gnds_endf_conflicts.md` §6.5 closed the *writing* of §25.3 to GNDS, not
    # to ENDF), so this is declared rather than attempted.
    parameters = getattr(covarianceSuite, "parameterCovariances", None) or ()
    if parameters:
        report.unsupportedNode(
            f"{len(parameters)} §25.3 parameter covariance section(s) are in the "
            f"covarianceSuite and there is no MF32 encoder, so the written tape "
            f"has no MF32: the resonance-parameter covariances do not survive "
            f"this trip and the fixed point cannot see it, because they are "
            f"absent from both sides"
        )
    return sections, report


def encodeTapeSections(suite, mat: Optional[int] = None, report=None, *,
                       label: Optional[str] = None
                       ) -> Tuple[List[Tuple[int, int, object]], object, int]:
    """A :class:`ReactionSuite` → ``[(MF, MT, section), …]`` in tape order.

    Returns the sections, the report, and the MAT they were stamped with. The
    report is the part to read: a model missing a file comes back as a tape
    missing that file, and only the report says so.
    """
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()
    mat = _mat(suite, mat)

    sections: List[Tuple[int, int, object]] = []
    for build in (_mf1Sections, _mf2Sections, _mf3And4And5Sections,
                  _mf6Sections, _covarianceSections):
        if build in (_mf1Sections, _mf3And4And5Sections):
            built, report = build(suite, mat, report, label)
        else:
            built, report = build(suite, mat, report)
        sections.extend(built)

    written = {mf for mf, _, _ in sections}
    for mf in sorted(written - set(MF_WRITE_ORDER)):  # pragma: no cover
        raise AssertionError(f"MF{mf} was built and MF_WRITE_ORDER omits it")

    # Ascending (MF, MT) is the tape order, and `sorted` is stable, so the
    # per-file builders above do not have to be.
    sections.sort(key=lambda entry: (entry[0], entry[1]))
    return sections, report, mat


def assembleTape(sections: Sequence[Tuple[int, int, object]], mat: int,
                 tapeId: Optional[str] = None) -> str:
    """``[(MF, MT, section), …]`` → the text of a one-material ENDF tape.

    Each section renders itself, ID columns and trailing SEND included — that is
    what the per-section byte-exact gates test, so this function must not
    re-render anything. What it adds is the record bookkeeping the sections
    cannot know about, because none of them knows what follows it: the FEND
    after the last section of each file, the MEND after the material, and the
    TEND that ends the tape.

    Sequence numbers are **per section** and each section already restarts them
    at 1, which is §0.6.3's rule, so there is nothing to renumber here.
    """
    lines = [_tapeIdRecord(tapeId if tapeId is not None else DEFAULT_TAPE_ID)]

    previousMf = None
    for mf, _mt, section in sections:
        if previousMf is not None and mf != previousMf:
            lines.append(format_endf_fend_record(mat))
        lines.append(str(section).rstrip("\n"))
        previousMf = mf

    if previousMf is not None:
        lines.append(format_endf_fend_record(mat))
    lines.append(format_endf_mend_record())
    lines.append(format_endf_tend_record())
    return "\n".join(lines) + "\n"


def writeEndfTape(suite, path, mat: Optional[int] = None,
                  tapeId: Optional[str] = None, report=None, *,
                  label: Optional[str] = None):
    """Write *suite* out as an ENDF-6 tape. Returns the :class:`ConversionReport`.

    The directory is rebuilt **after** the file is on disk, by
    :func:`~kika.endf.writers.update_directory.update_mf1_directory`, and that
    ordering is not incidental: ``NXC`` entries carry ``NC``, a line count, so
    the only place the true counts exist is the written file. ``encodeMF1MT451``
    writes back the directory it read, which is right for a tape whose sections
    have not changed length and wrong the moment one has.

    ``label`` selects which §9.1 form of each cross section and distribution is
    written; the default is ``'eval'``. A ``realization`` (§9.3) drawn beside
    the evaluation is written by naming its label here. Reactions with no form
    under that label fall back to ``'eval'``, so a tape written from a partially
    perturbed suite carries the perturbed sections and the original ones rather
    than only the first.
    """
    from .update_directory import update_mf1_directory

    sections, report, mat = encodeTapeSections(suite, mat, report, label=label)
    if not sections:
        raise ValueError(
            "this reactionSuite produced no ENDF sections at all, so there is "
            "no tape to write; read the ConversionReport for what was refused"
        )

    path = Path(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(assembleTape(sections, mat, tapeId), newline="\n")

    # Every section written is declared, so the rebuild does not drop one that
    # the source tape's directory happened not to list.
    if not update_mf1_directory(str(path),
                                added_sections={(mf, mt) for mf, mt, _ in sections}):
        report.warn(
            f"the MF1/451 directory of {path.name} could not be rebuilt, so its "
            f"NC line counts are the ones the model was read with and are only "
            f"true if no section changed length"
        )

    if tapeId is None:
        report.lost(
            "the tape identification record was written with kika's own label: "
            "read_endf does not keep the first line of a tape, so the original "
            "cannot be reproduced. Pass tapeId= to set it."
        )
    return report
