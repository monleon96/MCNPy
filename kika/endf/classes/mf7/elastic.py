"""MF7/MT2: thermal elastic scattering (ENDF-6 §7.2).

``LTHR`` selects which of two representations the section carries, and it is a
genuine three-way choice rather than a flag::

    LTHR = 1   coherent only     Bragg edges S(E), tabulated per temperature
    LTHR = 2   incoherent only   the Debye-Waller integral W(T)
    LTHR = 3   both, in that order

All three occur in ENDF/B-VIII.1: 86 of the 114 TSL evaluations are LTHR=1, 11
are LTHR=3 (``tsl-NinUN*``, ``tsl-ZrinZrH2``, ``tsl-YinYH2``, ``tsl-CinZrC``,
the mixed lithium hydrides), and the rest are LTHR=2. A further 17 have no MT2
at all — ``tsl-HinH2O`` among them — so *absent* is a fourth state and code
that reaches for ``sections[2]`` must expect a ``KeyError``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import MF7MT, TemperatureTable
from ...utils import PadStyle, format_endf_send_record, format_tab1

#: ``LTHR`` values that carry a coherent block, and those that carry an
#: incoherent one. Written as sets because LTHR=3 is in both, and every
#: ``lthr == 1`` test in a reader is a latent bug on the 11 LTHR=3 files.
COHERENT_LTHR = frozenset({1, 3})
INCOHERENT_LTHR = frozenset({2, 3})


@dataclass
class CoherentElastic:
    """Bragg-edge structure: S(E) as a stack of temperatures.

    The energy grid is the set of Bragg edges, and the interpolation is **INT=1
    (histogram)** — S(E) is a staircase that steps at each edge, so interpolating
    it linearly invents scattering between edges where there is none. Every
    coherent evaluation measured writes 1 here; the parser records what it read
    rather than assuming, and :meth:`is_histogram` is how a caller checks.
    """

    table: TemperatureTable = field(default_factory=TemperatureTable)

    @property
    def energies(self) -> List[float]:
        """The Bragg edges, in eV."""
        return self.table.x

    @property
    def temperatures(self) -> List[float]:
        return self.table.temperatures

    @property
    def is_histogram(self) -> bool:
        return all(int(code) == 1 for _, code in self.table.interp)

    def s_at(self, index: int) -> List[float]:
        """S(E) at temperature *index*, one value per Bragg edge."""
        return self.table.row(index)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return self.table.emit(0.0, mat, mf, mt, line_num, pad=pad)

    def describe(self) -> str:
        return (f"coherent: {len(self.energies)} Bragg edges, "
                f"{self.table.n_temperatures} temperature(s)")


@dataclass
class IncoherentElastic:
    """The Debye-Waller integral W(T), with the bound cross section SB.

    One TAB1 and no temperature stack — the temperature dependence *is* the
    x grid here, which is why this is not a :class:`TemperatureTable`.
    """

    sb: float = 0.0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return format_tab1(
            self.sb, 0.0, 0, 0, self.interp, self.temperatures, self.w,
            mat, mf, mt, line_num, pad=pad.pairs,
        )

    def describe(self) -> str:
        return f"incoherent: SB={self.sb}, {len(self.temperatures)} temperature(s)"


@dataclass
class MF7MT2(MF7MT):
    """One MF7/MT2 section: HEAD, then whichever blocks ``LTHR`` announces."""

    _lthr: Optional[int] = None
    coherent: Optional[CoherentElastic] = None
    incoherent: Optional[IncoherentElastic] = None

    @property
    def lthr(self) -> Optional[int]:
        return self._lthr

    @property
    def has_coherent(self) -> bool:
        return self.coherent is not None

    @property
    def has_incoherent(self) -> bool:
        return self.incoherent is not None

    @property
    def temperatures(self) -> List[float]:
        """The temperature grid, from whichever block defines one.

        Coherent first: on an LTHR=3 file the two blocks need not agree, and the
        coherent stack is the one that carries S(E) per temperature.
        """
        if self.coherent is not None:
            return self.coherent.temperatures
        if self.incoherent is not None:
            return self.incoherent.temperatures
        return []

    def report_gaps(self) -> List[str]:
        """Empty: MT2 is decoded in full, for every LTHR this format defines.

        Present so MF7 answers the same question MF5 does. See
        :meth:`kika.endf.classes.mf5.base.MF5MT.report_gaps` for why a section
        that skips nothing still has to say so.
        """
        return []

    def head_fields(self) -> List[int]:
        return [self._lthr or 0, 0, 0, 0]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        head, line_num = self.emit_head(1)
        lines = [head]
        for block in (self.coherent, self.incoherent):
            if block is not None:
                block_lines, line_num = block.emit(mat, mf, mt, line_num,
                                                  pad=self.pad)
                lines.extend(block_lines)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        blocks = ", ".join(
            block.describe()
            for block in (self.coherent, self.incoherent) if block is not None
        )
        return f"MF7MT2(LTHR={self._lthr}, {blocks})"
