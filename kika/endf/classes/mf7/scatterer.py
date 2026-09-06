"""Who the material *is*, for a file whose material is not a nuclide.

kika identifies a tape by mapping its MAT through
:data:`kika._constants.ENDF_MAT_TO_ZAID`. That table's lowest key is 125 and
every TSL MAT is 1-8399, so ``endf.zaid`` and ``endf.isotope`` come back
``None`` for a thermal scattering evaluation. They are left that way on
purpose. There is no nuclide to name — "H in H₂O" is a bound scatterer in a
compound, and MAT 1 with ZA 1001 would collide with free hydrogen's MAT 125 in
a table that is inverted by dict comprehension
(:data:`kika._constants.ZAID_TO_ENDF_MAT`), silently reassigning free hydrogen
to whichever entry was inserted last.

So TSL identity is a separate object, built here.

**The name is inferred, and says so.** ZSYMAM is free text and its convention
varies by evaluator — ``H(H2O)``, ``Be_BeO``, ``Be-Metal``, ``s-CH4``,
``' 4-Be-'`` are all real. :attr:`ThermalScatterer.name_source` records which
rule produced the name, so an inferred one is never mistaken for a stated one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from .composition import MF7MT451, TSLIsotope

#: ``NSUB`` of the thermal-scattering sublibrary. The format's own answer to
#: "is this a TSL evaluation", and better than any MAT range or filename test.
THERMAL_SCATTERING_NSUB = 12

#: ZSYMAM written as ``principal(compound)`` — ``H(H2O)``, ``Be(BeO)``.
_PARENTHESISED = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\(([^()]+)\)$")

#: ZSYMAM written as ``principal_compound`` — ``Be_BeO``, ``H_ZrH``.
_UNDERSCORED = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)_([A-Za-z0-9-]+)$")

#: ZSYMAM already written the way kika names it — ``NinUN``. The first group is
#: lazy so ``SiinSiO2`` splits after ``Si`` rather than after ``S``, and the
#: compound must start with a capital so the ``in`` being matched is the joining
#: word and not two letters inside a name.
_JOINED = re.compile(r"^([A-Z][A-Za-z0-9]*?)in([A-Z][A-Za-z0-9-]*)$")

#: Prefix TSL filenames carry in both libraries: ``tsl-HinH2O.endf``,
#: ``tsl_Be_BeO.txt``.
_FILENAME_PREFIX = re.compile(r"^tsl[-_]", re.IGNORECASE)


@dataclass
class ThermalScatterer:
    """The identity of a thermal scattering evaluation."""

    mat: Optional[int] = None
    #: The ZA field as written. A *pseudo*-ZA for most TSL materials — 126 for
    #: beryllium metal, 127 for Be-in-BeO — so never feed it to a ZA→symbol
    #: conversion. ENDF/B-VIII.1 and JEFF-4.0 do not agree here: JEFF's
    #: ``tsl_4-Be.txt`` writes 4000 where ENDF/B's ``tsl-Be-metal.endf``
    #: writes 126, for the same evaluation.
    za: Optional[float] = None
    awr: Optional[float] = None
    zsymam: Optional[str] = None
    name: Optional[str] = None
    #: ``'zsymam'``, ``'filename'`` or ``'zsymam-verbatim'``.
    name_source: Optional[str] = None
    principal: Optional[str] = None
    compound: Optional[str] = None
    #: Empty when the tape has no MF7/MT451 — 28 of ENDF/B-VIII.1's 114 do not.
    composition: List[TSLIsotope] = field(default_factory=list)

    @property
    def is_elemental(self) -> bool:
        """True when the name carries no compound — beryllium metal, graphite."""
        return self.compound is None

    def __repr__(self) -> str:
        return (f"ThermalScatterer({self.name!r}, MAT={self.mat}, "
                f"ZA={self.za}, {len(self.composition)} isotope(s))")


def _name_from_zsymam(zsymam: str):
    """``('HinH2O', 'H', 'H2O')`` when ZSYMAM names both parts, else ``None``."""
    text = zsymam.strip()
    for pattern in (_PARENTHESISED, _UNDERSCORED, _JOINED):
        match = pattern.match(text)
        if match:
            principal, compound = match.group(1), match.group(2)
            return f"{principal}in{compound}", principal, compound
    return None


def _name_from_filename(source: Union[str, Path]) -> str:
    """``tsl-Be-metal.endf`` → ``Be-metal``; ``tsl_4-Be.txt`` → ``4-Be``."""
    return _FILENAME_PREFIX.sub("", Path(source).stem)


def thermal_scatterer(endf, source: Optional[Union[str, Path]] = None
                      ) -> ThermalScatterer:
    """Build a :class:`ThermalScatterer` from a parsed TSL tape.

    Reads MF1/451 for the label and MF7/451 for the composition, so call
    ``read_endf(path, mf_numbers=[1, 7])`` — or a full read — before this.

    *source* is the file the tape came from. It is optional but worth passing:
    it is the only thing that can name ``tsl-Be-metal.endf`` or ``tsl-s-CH4.endf``,
    whose ZSYMAM (``Be-Metal``, ``s-CH4``) says nothing about a compound.
    """
    mt451 = None
    mf1 = endf.files.get(1)
    if mf1 is not None:
        mt451 = mf1.sections.get(451)

    scatterer = ThermalScatterer(mat=endf.mat)

    if mt451 is not None:
        scatterer.zsymam = (mt451.material_id or "").strip() or None
        # ``MF1MT451`` spells these ``zaid`` and ``atomic_weight_ratio``; MF7's
        # sections spell them ``za`` and ``awr``. Neither name is wrong, but
        # reading MF1's through a ``getattr`` default would silently fall
        # through to the MF7 branch below and look like it worked.
        scatterer.za = mt451.zaid
        scatterer.awr = mt451.atomic_weight_ratio

    mf7 = endf.files.get(7)
    if mf7 is not None:
        composition = mf7.sections.get(451)
        if isinstance(composition, MF7MT451):
            scatterer.composition = composition.isotopes()
        # Every MF7 HEAD repeats ZA and AWR, and on a targeted
        # ``read_endf(path, mf_numbers=[7])`` they are the only copy present.
        if scatterer.za is None:
            for section in mf7.sections.values():
                if section.za is not None:
                    scatterer.za = section.za
                    scatterer.awr = section.awr
                    break

    parsed = _name_from_zsymam(scatterer.zsymam) if scatterer.zsymam else None
    if parsed is not None:
        scatterer.name, scatterer.principal, scatterer.compound = parsed
        scatterer.name_source = "zsymam"
    elif source is not None:
        scatterer.name = _name_from_filename(source)
        scatterer.name_source = "filename"
    elif scatterer.zsymam:
        scatterer.name = scatterer.zsymam
        scatterer.name_source = "zsymam-verbatim"

    return scatterer
