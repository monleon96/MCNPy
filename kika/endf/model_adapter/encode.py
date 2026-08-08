""":class:`~kika.nuclear_data.model.suite.ReactionSuite` → ENDF section objects.

The gate for phase 3c is **not** "ENDF -> model -> ENDF is byte-identical". The
writer is patch-in-place, so whatever it is not handed survives verbatim and
that assertion stays true even if the model computes garbage. The gate this
module is written against is stronger and narrower: for every MT,

    str(encodeMF3MT(reaction))  ==  str(CrossSection.from_endf(section).to_endf())

byte for byte. That compares what the model produced against what the flat path
produces from the same input, which is the claim phase 3 actually makes.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kika.nuclear_data.model import EVAL_LABEL, ConversionReport, Reaction

__all__ = ["encodeMF3MT", "encodeMF1MT451"]

#: The MF1/451 header fields an encoder cannot invent. ``temp`` is deliberately
#: not among them: a section may legitimately carry 0 K, and the decoder always
#: records it, so its absence means "not decoded from ENDF" rather than "unset".
_MF1_REQUIRED_FIELDS = (
    "lrp", "lfi", "nlib", "nmod", "elis", "sta", "lis", "liso",
    "nfor", "awi", "emax", "lrel", "nsub", "nver", "ldrv",
)

#: ``EndfProvenance.evaluationInfo`` key → the ``MF1MT451`` attribute it fills.
_MF1_EVALUATION_INFO = (
    ("_zsymam", "material_id"), ("_alab", "laboratory"),
    ("_edate", "eval_date"), ("_auth", "authors"),
    ("_ref", "reference"), ("_ddate", "dist_date"),
    ("_rdate", "revision_date"),
)


def encodeMF3MT(reaction: Reaction, mat: Optional[int] = None,
                report: Optional[ConversionReport] = None):
    """A :class:`Reaction` → an ``MF3MT``.

    Everything comes from the model or from the provenance the decoder kept;
    nothing is recomputed. In particular the ``(NBT, INT)`` pairs are the
    file's own, not a reconstruction from the regions — a reconstruction would
    be correct and would still be a second source of truth.
    """
    from kika.endf.classes.mf3.mf3mt import MF3MT

    report = report if report is not None else ConversionReport()
    provenance = getattr(reaction, "provenance", None)

    if EVAL_LABEL not in reaction.crossSection:
        raise ValueError(
            f"{reaction.label} has no {EVAL_LABEL!r} cross-section form; §16.1.1 "
            f"requires one for an evaluated file, and there is nothing to write"
        )
    form = reaction.crossSection[EVAL_LABEL]
    if not hasattr(form, "toEndfRegions"):
        raise TypeError(
            f"the {EVAL_LABEL!r} form of {reaction.label} is a "
            f"{type(form).__name__}; MF3 needs a tabulated form (regions1d)"
        )

    energies, values, regions = form.toEndfRegions()

    if provenance is not None and provenance.interpolationRegions:
        # Prefer the file's own pairs. They and the reconstructed ones agree --
        # a test asserts it -- but preferring the original means a round trip
        # does not depend on that agreement holding for every tape ever written.
        regions = list(provenance.interpolationRegions)

    q = reaction.outputChannel.Q
    qi = q.value
    qm = getattr(provenance, "qm", None)
    lr = getattr(provenance, "lr", None)

    missing = [name for name, value in (("qm", qm), ("qi", qi), ("lr", lr)) if value is None]
    if missing:
        raise ValueError(
            f"{reaction.label} carries no {'/'.join(missing)}, so an ENDF MF3 "
            f"header cannot be written for it. ACE stores no reaction Q values; "
            f"supply them explicitly or build the reaction from ENDF."
        )

    mt = reaction.ENDF_MT
    if mt is None:
        raise ValueError(
            f"{reaction.label} has no ENDF_MT. §15.1.1 deprecates the attribute "
            f"but ENDF cannot be written without it."
        )

    section = MF3MT(number=int(mt))
    section._za = float(_nuclideId(reaction, provenance))
    section._awr = getattr(provenance, "awr", None) or 0.0
    section._mat = mat if mat is not None else getattr(provenance, "mat", None)
    section._qm = qm
    section._qi = qi
    section._lr = lr
    section._energies = list(energies)
    section._cross_sections = list(values)
    section._np = int(energies.size)
    section._nr = len(regions)
    section._interpolation = [tuple(pair) for pair in regions]
    return section, report


def encodeMF1MT451(source, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None):
    """A :class:`ReactionSuite` (or its :class:`EndfProvenance`) → an ``MF1MT451``.

    The inverse of :func:`~kika.endf.model_adapter.decode.decodeMF1MT451`, and
    written the way that decoder's docstring says it has to be: the whole 451
    header went to provenance verbatim because most of it — ``NLIB``, ``NMOD``,
    ``LDRV``, the NWD comment block — is ENDF bookkeeping with no GNDS
    counterpart, so it is written back unchanged rather than recomputed.

    **On the directory.** ``NXC`` entries carry ``NC``, a *line count*, so a
    directory is only true of the tape it was read from. This writes back the one
    it read, which is right for a round trip and wrong the moment a section
    changes length — at which point
    :func:`kika.endf.writers.update_directory.update_mf1_directory` rebuilds it
    by scanning the written file, the only place the true counts exist.
    Recomputing here would mean guessing at lengths this object cannot see.

    Parameters
    ----------
    source : ReactionSuite or EndfProvenance
        A suite decoded from ENDF (its ``provenance`` is used), or the
        provenance directly.
    mat : int, optional
        MAT number. When *None*, the one the provenance recorded.

    Raises
    ------
    ValueError
        If the provenance carries no ENDF header — an ACE-sourced suite, or one
        built by hand. ACE records four header fields out of nineteen and no
        descriptive text at all, so the section would be mostly invented. The
        same refusal, and the same reason, as :func:`encodeMF3MT` on a missing Q.
    """
    from kika.endf.classes.mf1.mf1mt451 import MF1MT451

    report = report if report is not None else ConversionReport()
    provenance = getattr(source, "provenance", source)

    fields = dict(getattr(provenance, "headerFields", None) or {})
    missing = [name for name in _MF1_REQUIRED_FIELDS if fields.get(name) is None]
    if missing:
        sourceFormat = getattr(provenance, "sourceFormat", "unknown")
        raise ValueError(
            f"provenance (sourceFormat={sourceFormat!r}) carries no "
            f"{'/'.join(missing)}, so an ENDF MF1/451 section cannot be written "
            f"for it. ACE records a handful of header fields and no descriptive "
            f"text, so most of the section would be invented. Decode from ENDF, "
            f"where the header comes from the file."
        )

    text = list(getattr(provenance, "descriptiveText", None) or [])
    directory = [tuple(entry) for entry in getattr(provenance, "directory", None) or []]
    evaluationInfo = getattr(provenance, "evaluationInfo", None) or {}

    mt451 = MF1MT451(number=451)
    mt451._mat = mat if mat is not None else getattr(provenance, "mat", None)
    mt451._za = float(getattr(provenance, "za", None) or 0)
    mt451._awr = getattr(provenance, "awr", None)
    mt451._temp = fields.get("temp")
    for name in _MF1_REQUIRED_FIELDS:
        setattr(mt451, f"_{name}", fields[name])

    # `__str__` re-emits records 5..4+NWD from `_text_lines`, which the parser
    # stores as the whole section including its four header records. Only the
    # slice from 4 is ever read, so the padding is never written.
    mt451._text_lines = [""] * 4 + text
    mt451._nwd = len(text)
    mt451._directory = directory
    mt451._nxc = len(directory)

    # Set from `evaluationInfo` as well, and not instead: `__str__` falls back to
    # these seven when there is no text block, and a section whose text survived
    # should not carry a header that disagrees with it.
    for attr, key in _MF1_EVALUATION_INFO:
        setattr(mt451, attr, evaluationInfo.get(key) or "")

    if not text:
        report.lost(
            "MF1/451 is written with no descriptive records: the provenance kept "
            "none, so the NWD comment block cannot be reproduced"
        )
    return mt451, report


def _nuclideId(reaction: Reaction, provenance) -> int:
    """ZA for the section header, from PoPs where the model has it."""
    za = getattr(provenance, "za", None)
    if za is not None:
        return int(za)
    stored = getattr(reaction, "nuclideId", None)
    return int(stored) if stored is not None else 0
