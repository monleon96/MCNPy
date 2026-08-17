"""MF7/MT451: what the thermal scatterer is actually made of (ENDF-6 §7.1).

The section that turns a pseudo-ZA back into nuclides. ``tsl-Be-metal.endf``
announces ``ZA = 126``, which is not ``1000Z + A`` and names nothing; its MT451
says ``ZAI = 4009``, i.e. Be-9, at atom fraction 1.0 with σ_free = 6.154 b.

Present in 86 of ENDF/B-VIII.1's 114 TSL evaluations — the 28 without it are
mostly VIII.0-era carryovers (``tsl-HinH2O``, ``tsl-DinD2O``,
``tsl-crystalline-graphite``, ``tsl-s-CH4`` …), so a consumer must treat the
composition as optional and not as something every tape can be asked for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .base import MF7MT, emit_list
from ...utils import PadStyle, format_endf_send_record

#: Values per isotope in the LIST body. NW is this times the isotope count, and
#: driving the loop from ``NW // 6`` rather than from the NAS field is
#: deliberate — see :class:`ElementComposition`.
VALUES_PER_ISOTOPE = 6


@dataclass
class TSLIsotope:
    """One nuclide of one element of the scatterer.

    ``zai`` is an ``int``, unlike the ZA fields elsewhere in MF7. It is a real
    ``1000Z + A`` here rather than a pseudo-ZA, so it is worth having as one —
    and :func:`~kika.endf.utils.parse_number` returns ``4009.0000000000005``
    for ``4.009000+3``, which compares equal to nothing anyone would write.
    Rounding it here is byte-safe: ``format_endf_number(4009)`` is
    ``" 4.009000+3"``, the field it came from.
    """

    zai: int = 0
    lis: int = 0
    atom_fraction: float = 0.0
    awr: float = 0.0
    sigma_free: float = 0.0

    @property
    def z(self) -> int:
        return self.zai // 1000

    @property
    def a(self) -> int:
        return self.zai % 1000

    def values(self) -> List[float]:
        """The six numbers as the LIST body writes them."""
        return [self.zai, self.lis, self.atom_fraction,
                self.awr, self.sigma_free, 0.0]


@dataclass
class ElementComposition:
    """One element of the compound, and its isotopes.

    ``nas`` is the section's own L1 field. ENDF-6 documents it as the number of
    atoms of this element in the molecule, and sometimes it is — ``tsl-HinC5O2H8``
    writes 8, which is right for methyl methacrylate. But ``tsl-BeinBe2C``
    writes 1 where Be₂C has two. So it is carried through verbatim and **not**
    used to size anything; the isotope count comes from ``NW // 6``, which is
    the only field that cannot be inconsistent with the body that follows it.
    """

    nas: int = 0
    isotopes: List[TSLIsotope] = field(default_factory=list)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        values: List[float] = []
        for isotope in self.isotopes:
            values.extend(isotope.values())
        return emit_list(0.0, 0.0, self.nas, 0, len(self.isotopes), values,
                         mat, mf, mt, line_num, pad=pad.values)


@dataclass
class MF7MT451(MF7MT):
    """One MF7/MT451 section: HEAD, then one LIST per element."""

    elements: List[ElementComposition] = field(default_factory=list)

    @property
    def na(self) -> int:
        """Number of elements described."""
        return len(self.elements)

    def isotopes(self) -> List[TSLIsotope]:
        """Every isotope of every element, flattened."""
        return [isotope for element in self.elements
                for isotope in element.isotopes]

    def report_gaps(self) -> List[str]:
        """Empty: MT451 is decoded in full. See :meth:`MF7MT2.report_gaps`."""
        return []

    def head_fields(self) -> List[int]:
        return [self.na, 0, 0, 0]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        head, line_num = self.emit_head(1)
        lines = [head]
        for element in self.elements:
            element_lines, line_num = element.emit(mat, mf, mt, line_num,
                                                  pad=self.pad)
            lines.extend(element_lines)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MF7MT451({self.na} element(s), "
                f"{len(self.isotopes())} isotope(s))")
