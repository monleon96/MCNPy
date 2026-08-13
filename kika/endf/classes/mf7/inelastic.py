"""MF7/MT4: incoherent inelastic scattering, S(α, β, T) (ENDF-6 §7.4).

The largest section kika parses. ``tsl-HinH2O.endf`` writes one MT4 of
1 144 410 records — 317 β × 94 temperatures × 222 α — which is the whole 87 MB
file bar 129 lines.

The record layout is a TAB2 over β whose sub-records are the
:class:`~kika.endf.classes.mf7.base.TemperatureTable` shape: a TAB1 in α at T₀,
then one LIST per further temperature carrying S values with **no α column**.
After the β loop come the effective-temperature tables, and how many of those
there are is not written anywhere — it has to be derived from the B array (see
:attr:`MF7MT4.expected_teff_records`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import MF7MT, TemperatureTable, emit_list
from ...utils import (
    PadStyle,
    format_endf_send_record,
    format_tab1,
    format_tab2,
)

#: ``B(6i+1)`` for a non-principal scatterer: the law used for it instead of a
#: tabulated S(α, β). Only 0.0 brings its own effective-temperature table.
SCT_APPROXIMATION = 0.0
FREE_GAS = 1.0
DIFFUSIVE_MOTION = 2.0


@dataclass
class SecondaryScatterer:
    """One non-principal atom type, decoded from six entries of the B array.

    Six of ENDF/B-VIII.1's 114 TSL evaluations have any: ``tsl-HinH2O``,
    ``tsl-HinCH2``, ``tsl-l-CH4`` and ``tsl-s-CH4`` treat theirs as a free gas,
    while ``tsl-benzene`` and ``tsl-SiO2-beta`` use the short-collision-time
    approximation and therefore carry a second Teff table.
    """

    analytic_flag: float = FREE_GAS
    free_xs: float = 0.0
    awr: float = 0.0
    n_atoms: float = 0.0

    @property
    def needs_teff(self) -> bool:
        """SCT is the one law that needs an effective temperature tabulated."""
        return self.analytic_flag == SCT_APPROXIMATION


@dataclass
class EffectiveTemperature:
    """A ``T → T_eff`` TAB1, used by the short-collision-time approximation."""

    interp: List[Tuple[int, int]] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    teff: List[float] = field(default_factory=list)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return format_tab1(
            0.0, 0.0, 0, 0, self.interp, self.temperatures, self.teff,
            mat, mf, mt, line_num, pad=pad.pairs,
        )


@dataclass
class BetaBlock:
    """S(α) at one β, stacked over temperature."""

    beta: float = 0.0
    table: TemperatureTable = field(default_factory=TemperatureTable)

    @property
    def alphas(self) -> List[float]:
        return self.table.x

    @property
    def temperatures(self) -> List[float]:
        return self.table.temperatures

    def s_at(self, index: int) -> List[float]:
        """S(α) at temperature *index*, one value per α."""
        return self.table.row(index)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return self.table.emit(self.beta, mat, mf, mt, line_num, pad=pad)


@dataclass
class MF7MT4(MF7MT):
    """One MF7/MT4 section.

    **What the B array means, and what it does not.** Measured across the
    ENDF/B-VIII.1 TSL sublibrary, three entries are unambiguous and are the only
    ones given names here:

    ``B(1)``  σ_free of the principal scatterer **times** ``B(6)`` — 40.87 for
              H-in-H₂O and H-in-CH₂ (2 × 20.436), 20.44 for H-in-ZrH (1 ×),
              6.79 for D-in-D₂O (2 × 3.395), 6.15 for Be.
    ``B(3)``  the principal scatterer's AWR — matches MF1/451's AWR field in
              every file checked.
    ``B(6)``  the number of principal atoms in the compound.

    ``B(2)``, ``B(4)`` and ``B(5)`` are stored verbatim and deliberately not
    given accessors. ``B(5)`` is 0.0 everywhere measured; ``B(2)`` takes values
    (197.6 for Be, Be-in-BeO, O-in-BeO *and* H-in-CH₂; 395.3 for H-in-H₂O; 78.0
    for H-in-ZrH; 450.0 for s-CH₄) that fit no single reading of the manual, and
    naming a field after a guess is worse than leaving the caller to read
    ``b[1]`` knowing they are on their own.
    """

    _lat: Optional[int] = None
    _lasym: Optional[int] = None
    _lln: Optional[int] = None
    _ns: Optional[int] = None
    b: List[float] = field(default_factory=list)
    beta_interp: List[Tuple[int, int]] = field(default_factory=list)
    blocks: List[BetaBlock] = field(default_factory=list)
    teff: List[EffectiveTemperature] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Header flags
    # ------------------------------------------------------------------
    @property
    def lat(self) -> Optional[int]:
        """1 if α and β are on the 0.0253 eV basis rather than the tape's T."""
        return self._lat

    @property
    def lasym(self) -> Optional[int]:
        """1 if S(α, β) is asymmetric, so negative β are tabulated explicitly.

        Only the four ortho-/para- H and D evaluations set this.
        """
        return self._lasym

    @property
    def lln(self) -> Optional[int]:
        """1 if the table stores ln S rather than S. 0 in every file measured."""
        return self._lln

    @property
    def ns(self) -> int:
        """Number of non-principal scattering atom types (0-3)."""
        return self._ns if self._ns is not None else 0

    # ------------------------------------------------------------------
    # The B array
    # ------------------------------------------------------------------
    @property
    def free_atom_xs(self) -> Optional[float]:
        """``B(1)``: σ_free of the principal scatterer × :attr:`n_principal_atoms`."""
        return self.b[0] if self.b else None

    @property
    def principal_awr(self) -> Optional[float]:
        """``B(3)``: the principal scatterer's mass ratio."""
        return self.b[2] if len(self.b) > 2 else None

    @property
    def n_principal_atoms(self) -> Optional[float]:
        """``B(6)``: principal atoms in the compound — 2 for H in H₂O."""
        return self.b[5] if len(self.b) > 5 else None

    @property
    def has_tabulated_s(self) -> bool:
        """``B(1) == 0`` means the scatterer is analytic and carries no S table."""
        return bool(self.b) and self.b[0] != 0.0

    def secondary_scatterers(self) -> List[SecondaryScatterer]:
        """The ``NS`` non-principal atom types, six B entries each."""
        result = []
        for index in range(1, self.ns + 1):
            base = 6 * index
            if len(self.b) < base + 6:
                break
            result.append(SecondaryScatterer(
                analytic_flag=self.b[base],
                free_xs=self.b[base + 1],
                awr=self.b[base + 2],
                n_atoms=self.b[base + 5],
            ))
        return result

    @property
    def expected_teff_records(self) -> int:
        """How many effective-temperature TAB1s must follow the β loop.

        Nothing in the file states this. It is one for the principal scatterer
        plus one for each non-principal scatterer using the short-collision-time
        approximation, and getting it wrong does not raise — it walks off the
        end of the section or stops early inside it, either way silently. The
        parser checks the count it derives here against the records it actually
        read, and refuses the section if they disagree.
        """
        return 1 + sum(1 for s in self.secondary_scatterers() if s.needs_teff)

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    @property
    def betas(self) -> List[float]:
        return [block.beta for block in self.blocks]

    @property
    def temperatures(self) -> List[float]:
        """The temperature grid, taken from the first β block.

        ENDF-6 requires every β to share it, and the parser checks that rather
        than trusting it.
        """
        return self.blocks[0].temperatures if self.blocks else []

    def block_at_beta(self, beta: float) -> BetaBlock:
        for block in self.blocks:
            if block.beta == beta:
                return block
        raise KeyError(f"beta {beta} is not tabulated in MF7/MT{self.number}")

    def report_gaps(self) -> List[str]:
        """Empty: MT4 is decoded in full. See :meth:`MF7MT2.report_gaps`."""
        return []

    # ------------------------------------------------------------------
    def head_fields(self) -> List[int]:
        return [0, self._lat or 0, self._lasym or 0, 0]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        head, line_num = self.emit_head(1)
        lines = [head]

        b_lines, line_num = emit_list(
            0.0, 0.0, self._lln or 0, 0, self.ns, self.b,
            mat, mf, mt, line_num, pad=self.pad.values,
        )
        lines.extend(b_lines)

        if self.blocks:
            tab2_lines, line_num = format_tab2(
                0.0, 0.0, 0, 0, self.beta_interp, len(self.blocks),
                mat, mf, mt, line_num,
            )
            lines.extend(tab2_lines)

            for block in self.blocks:
                block_lines, line_num = block.emit(mat, mf, mt, line_num,
                                                  pad=self.pad)
                lines.extend(block_lines)

        for table in self.teff:
            teff_lines, line_num = table.emit(mat, mf, mt, line_num,
                                             pad=self.pad)
            lines.extend(teff_lines)

        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MF7MT4(LAT={self._lat}, LASYM={self._lasym}, NS={self.ns}, "
                f"{len(self.blocks)} betas, "
                f"{len(self.temperatures)} temperatures)")
