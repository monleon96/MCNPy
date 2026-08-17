"""MF7: thermal neutron scattering law (ENDF-6 §7).

The shared part of the three sections ENDF-6 defines for File 7 — MT2
(elastic), MT4 (incoherent inelastic) and MT451 (the scatterer's composition).
Each lives in its own module beside this one; this is the HEAD record they all
start with and the LIST emitter two of them need.

**MF7 materials are not nuclides.** A TSL evaluation describes a bound
scatterer in a compound — H in H₂O, Be in BeO — so it carries a *pseudo*-ZA
(126 for beryllium metal, 127 for Be-in-BeO, 134 for solid methane) that is not
``1000Z + A`` and must never be fed to a ZA→symbol conversion. The identity of
the material is reconstructed in :mod:`~kika.endf.classes.mf7.scatterer`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    PAD_BLANK,
    PadStyle,
    format_data_values,
    format_endf_data_line,
    format_tab1,
)

#: Every MF7 HEAD record types C1/C2 as floats and the four remaining fields as
#: integers. Named once so the three sections cannot drift apart.
HEAD_FORMATS = [
    ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT,
]


def emit_list(c1: float, c2: float, l1: int, l2: int, n2: int,
              values: Sequence[float], mat: int, mf: int, mt: int,
              line_num: int, pad: str = PAD_BLANK
              ) -> Tuple[List[str], int]:
    """A LIST record: the header, then ``len(values)`` numbers six per line.

    ``n1`` is not a parameter because ENDF-6 defines it as the number of values
    that follow, and computing it from ``values`` is the only way the header and
    the body cannot disagree. ``n2`` stays free: MF7's temperature records put
    0 there while MF7/MT451 puts the isotope count.
    """
    header = format_endf_data_line(
        [c1, c2, l1, l2, len(values), n2], mat, mf, mt, line_num,
        formats=HEAD_FORMATS,
    )
    body, line_num = format_data_values(list(values), mat, mf, mt,
                                        line_num + 1, pad=pad)
    return [header] + body, line_num


@dataclass
class MF7MT(MT):
    """The part every MF7 section shares: the HEAD record's four fields.

    Subclasses add their own discriminator (``LTHR`` for MT2, ``LAT``/``LASYM``
    for MT4, ``NA`` for MT451) and supply it through :meth:`head_fields`.
    """

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _mat: Optional[int] = None
    _mf: int = 7
    #: How this section's source filled the unused fields of a short line.
    #: Read from the tape rather than chosen, so a section written back matches
    #: the dialect it came in as. See :class:`~kika.endf.utils.PadStyle` for why
    #: it is two conventions and not one.
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def za(self) -> Optional[float]:
        """The ZA field as written — a *pseudo*-ZA for most TSL materials."""
        return self._za

    @property
    def awr(self) -> Optional[float]:
        """Mass ratio of the scattering unit to the neutron."""
        return self._awr

    @property
    def mat(self) -> Optional[int]:
        return self._mat

    def head_fields(self) -> List[int]:
        """The four integer fields of this section's HEAD, ``L1 L2 N1 N2``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not know its own HEAD fields"
        )

    def emit_head(self, line_num: int) -> Tuple[str, int]:
        mat = self._mat if self._mat is not None else 0
        line = format_endf_data_line(
            [self._za, self._awr, *self.head_fields()],
            mat, self._mf, self.number, line_num,
            formats=HEAD_FORMATS,
        )
        return line, line_num + 1


@dataclass
class TemperatureTable:
    """A quantity tabulated at T₀ and re-tabulated at ``LT`` more temperatures.

    ENDF-6 writes this shape twice in File 7 — for S(E) under coherent elastic
    scattering, and for S(α) at each β under incoherent inelastic scattering —
    and both times with the same asymmetry: **the T₀ record is a TAB1 carrying
    the x grid, and every further temperature is a LIST carrying values only.**
    The x grid is shared, not repeated. Storing the extra temperatures as pairs
    and writing them back as pairs is the mistake this class exists to make
    impossible.

    ``li[k]`` is the interpolation scheme between temperature ``k`` and ``k+1``,
    read from the LIST record for temperature ``k+1``. It is kept per record
    rather than assumed constant because it is a field the evaluator sets.
    """

    t0: float = 0.0
    lt: int = 0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    #: ``(lt + 1, len(x))`` — row 0 is T₀, stored as lists of rows so a section
    #: read from a tape re-emits the numbers it was given.
    values: List[List[float]] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    li: List[int] = field(default_factory=list)

    @property
    def n_temperatures(self) -> int:
        return len(self.temperatures)

    def row(self, index: int) -> List[float]:
        """Values at temperature *index*."""
        return self.values[index]

    def at_temperature(self, temperature: float) -> List[float]:
        """Values at an exact tabulated temperature.

        Exact rather than interpolated: ``li`` says how ENDF wants neighbouring
        temperatures combined, and silently doing something else here would put
        an unlabelled approximation in front of the caller.
        """
        for index, value in enumerate(self.temperatures):
            if value == temperature:
                return self.values[index]
        raise KeyError(
            f"{temperature} K is not tabulated; have "
            f"{[float(t) for t in self.temperatures]}"
        )

    def emit(self, c2: float, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        """The T₀ TAB1 followed by one LIST per further temperature.

        *c2* is the second header field, which ENDF-6 uses differently in the
        two places this shape appears: 0.0 for MT2's coherent elastic table,
        the block's β for MT4's S(α) tables. It is passed in rather than stored
        so the same record class serves both without carrying a field that is
        meaningless in one of them.
        """
        lines, line_num = format_tab1(
            self.t0, c2, self.lt, 0, self.interp, self.x, self.values[0],
            mat, mf, mt, line_num, pad=pad.pairs,
        )
        for index in range(1, len(self.temperatures)):
            block, line_num = emit_list(
                self.temperatures[index], c2, self.li[index - 1], 0, 0,
                self.values[index], mat, mf, mt, line_num, pad=pad.values,
            )
            lines.extend(block)
        return lines, line_num
