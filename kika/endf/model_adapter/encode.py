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

__all__ = ["encodeMF3MT"]


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


def _nuclideId(reaction: Reaction, provenance) -> int:
    """ZA for the section header, from PoPs where the model has it."""
    za = getattr(provenance, "za", None)
    if za is not None:
        return int(za)
    stored = getattr(reaction, "nuclideId", None)
    return int(stored) if stored is not None else 0
