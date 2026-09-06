"""
MF32, MT=151 — covariances of the resonance parameters given in File 2.

ENDF-102 §32. The section mirrors File 2's outer shape exactly — HEAD, then a
CONT per isotope, then a CONT per energy range — and then diverges: each range
carries a covariance body whose layout depends on the *compatibility flag*
LCOMP, on LRF, and on whether a scattering-radius uncertainty is present (ISR).

Five bodies exist, and every one of them is on a tape this module was measured
against:

======  =========================================  ================================
LCOMP   §                                          measured on
======  =========================================  ================================
0       32.2.1  ENDF/B-V-compatible, per-L blocks   Cm-244, Am-241 (LRF=2)
1       32.2.2  general, NSRS/NLRS blocks           Mn-55, Ta-181, Pu-239 (LRF=3)
2       32.2.3  compact: uncertainties + INTG       Na-23 (LRF=2), Th-232 (LRF=3)
2       32.2.3.3  compact, R-Matrix Limited         Cl-35, Cu-63, W-186 (LRF=7)
n/a     32.2.4  unresolved region (LRU=2)           Th-232, second range
======  =========================================  ================================

Each body is a small typed view over :class:`~.records.Record` objects that keep
the bytes they were read from; see that module for why. The consequence worth
knowing is that this class is faithful rather than opinionated — where a tape
disagrees with §32, kika reproduces the tape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..mt import MT
from ...utils import format_endf_send_record, parse_intg, format_intg
from .records import DATA_WIDTH, PackedList, Record, _stamp, emit_all


# ======================================================================
# The packed correlation matrix (LCOMP=2)
# ======================================================================

@dataclass
class IntgMatrix:
    """The correlation matrix of an LCOMP=2 body, packed into INTG records.

    §32.2.3 stores each off-diagonal coefficient as a signed integer of NDIGIT
    digits: ``C[i,j]`` is dropped if it rounds to zero, and otherwise mapped to
    the integer ``K`` whose range it falls in. The reverse mapping — the one
    :meth:`correlation_matrix` applies — takes ``K`` to the *centre* of that
    range, so a stored 87 with NDIGIT=2 comes back as 0.875, not 0.87.

    ``n2`` is the sixth field of the control record. §32.2.3 draws it as 0 and
    Na-23, Cl-35, Cu-63 and W-186 write 0; **Th-232 writes NM there instead**.
    It is stored rather than assumed so that the tape comes back as it went in.
    """

    ndigit: int = 2
    nnn: int = 0
    nm: int = 0
    n2: int = 0
    control_raw: str = ""
    lines: List[str] = field(default_factory=list)

    @property
    def entries(self) -> List[Tuple[int, int, List[int]]]:
        """``(II, JJ, [K, ...])`` for each INTG record, decoded on demand."""
        decoded, _ = parse_intg(
            [line.ljust(DATA_WIDTH) for line in self.lines], 0,
            self.ndigit, len(self.lines),
        )
        return decoded

    def correlation_matrix(self) -> np.ndarray:
        """The full ``NNN x NNN`` correlation matrix, unpacked.

        The diagonal is exactly 1.0 and is never stored; everything not named by
        an INTG record is zero. Mirrors the FORTRAN reconstruction printed in
        §32.2.3, including its ``JP >= II`` stop, which is what keeps a record
        whose row runs past the diagonal from writing into the upper triangle.
        """
        matrix = np.zeros((self.nnn, self.nnn), dtype=float)
        np.fill_diagonal(matrix, 1.0)
        factor = 10 ** self.ndigit
        for ii, jj, values in self.entries:
            column = jj - 1
            for packed in values:
                if column >= ii - 1:
                    break
                if packed:
                    sign = 1.0 if packed > 0 else -1.0
                    coefficient = sign * (abs(packed) + 0.5) / factor
                    matrix[ii - 1, column] = coefficient
                    matrix[column, ii - 1] = coefficient
                column += 1
        return matrix

    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        out = [_stamp(self.control_raw, mat, mf, mt, line_num)]
        line_num += 1
        for line in self.lines:
            out.append(_stamp(line, mat, mf, mt, line_num))
            line_num += 1
        return out, line_num

    def __repr__(self) -> str:
        return f"IntgMatrix(NDIGIT={self.ndigit}, NNN={self.nnn}, NM={self.nm})"


# ======================================================================
# Range bodies
# ======================================================================

@dataclass
class _Body:
    """What every covariance body shares: its own control records, in order."""

    control: Optional[Record] = None
    dap: Optional[Record] = None

    @property
    def isr(self) -> int:
        """1 if a scattering-radius uncertainty is present, else 0."""
        return self.control.n2 if self.control is not None else 0


@dataclass
class LCOMP0Body(_Body):
    """§32.2.1 — the ENDF/B-V-compatible format, one LIST per L value.

    Each block carries 18 numbers per resonance: the six File 2 parameters, then
    twelve variance/covariance terms, of which the four involving the resonance
    spin J are null by construction (§32.3 procedure 2 says a non-zero one is to
    be treated as null anyway). Applicable only to LRF=1 and 2.
    """

    l_blocks: List[Record] = field(default_factory=list)

    @property
    def nls(self) -> int:
        return self.control.n1 if self.control is not None else 0

    def emit(self, mat, mf, mt, line_num):
        return emit_all([self.control, self.dap, *self.l_blocks],
                        mat, mf, mt, line_num)


@dataclass
class LCOMP1Body(_Body):
    """§32.2.2 — the general format: NSRS short-range and NLRS long-range blocks.

    Short-range blocks each carry the parameters of a block of resonances
    followed by the upper triangle of their covariance matrix, so this is the
    format that gets large: Ta-181 spends 1 440 750 numbers on a single block.
    Long-range blocks resemble File 33's NI subsubsections and apply one
    covariance pattern to every resonance of a given parameter type in a band.
    """

    counts: Optional[Record] = None
    short_range: List[Record] = field(default_factory=list)
    long_range: List[Record] = field(default_factory=list)

    @property
    def nsrs(self) -> int:
        return self.counts.n1 if self.counts is not None else 0

    @property
    def nlrs(self) -> int:
        return self.counts.n2 if self.counts is not None else 0

    def emit(self, mat, mf, mt, line_num):
        return emit_all(
            [self.control, self.dap, self.counts,
             *self.short_range, *self.long_range],
            mat, mf, mt, line_num,
        )


@dataclass
class LCOMP1RMLBody(_Body):
    """§32.2.2.4 — LCOMP=1 for R-Matrix Limited (LRF=7).

    Differs from :class:`LCOMP1Body` in that a short-range block cannot be
    described by a resonance count alone: LRF=7 lets each spin group carry its
    own number of channels, so a block opens with a CONT giving NJSX and then a
    LIST per spin group, and closes with the covariance triangle. No long-range
    blocks are allowed. Nothing on this machine uses it — see the coverage note
    in ``docs/library/mf32-notes.md``.
    """

    counts: Optional[Record] = None
    blocks: List[List[Record]] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num):
        flat: List[Record] = [self.control, self.dap, self.counts]
        for block in self.blocks:
            flat.extend(block)
        return emit_all(flat, mat, mf, mt, line_num)


@dataclass
class LCOMP2Body(_Body):
    """§32.2.3.1-2 — the compact format for LRF=1, 2 and 3.

    One LIST gives, per resonance, the File 2 parameters followed immediately by
    their uncertainties — twelve numbers a resonance either way, with 0.0 in the
    slots of parameters that have no uncertainty (AJ always; GT for LRF=1 and 2,
    which is redundant). The correlations then follow as INTG records, or not at
    all: Na-23 stops after the uncertainties and is a diagonal covariance.
    """

    parameters: Optional[Record] = None
    correlations: Optional[IntgMatrix] = None

    @property
    def nrsa(self) -> int:
        """Number of resonances carried in the covariance matrix."""
        return self.parameters.n2 if self.parameters is not None else 0

    def emit(self, mat, mf, mt, line_num):
        out, line_num = emit_all([self.control, self.dap, self.parameters],
                                 mat, mf, mt, line_num)
        if self.correlations is not None:
            extra, line_num = self.correlations.emit(mat, mf, mt, line_num)
            out.extend(extra)
        return out, line_num


@dataclass
class RMLSpinGroup:
    """One J-pi group of an LCOMP=2 R-Matrix Limited body: channels, then resonances."""

    channels: Optional[Record] = None
    resonances: Optional[Record] = None

    @property
    def nch(self) -> int:
        return self.channels.n2 if self.channels is not None else 0

    @property
    def nrsa(self) -> int:
        return self.resonances.l2 if self.resonances is not None else 0

    def records(self) -> List[Record]:
        return [r for r in (self.channels, self.resonances) if r is not None]


@dataclass
class LCOMP2RMLBody(_Body):
    """§32.2.3.3 — the compact format for R-Matrix Limited (LRF=7).

    Re-declares File 2's particle pairs and channels, then gives each spin
    group's resonances with their uncertainties interleaved line for line, and
    closes with the INTG correlation matrix. This is the format ENDF/B-VIII.1
    reaches for on light and structural nuclides: Cl-35, Cu-63 and W-186 all use
    it.
    """

    particle_pairs: Optional[Record] = None
    spin_groups: List[RMLSpinGroup] = field(default_factory=list)
    correlations: Optional[IntgMatrix] = None

    @property
    def njs(self) -> int:
        return self.control.n1 if self.control is not None else 0

    @property
    def npp(self) -> int:
        return self.particle_pairs.l1 if self.particle_pairs is not None else 0

    def emit(self, mat, mf, mt, line_num):
        flat: List[Record] = [self.control, self.dap, self.particle_pairs]
        for group in self.spin_groups:
            flat.extend(group.records())
        out, line_num = emit_all(flat, mat, mf, mt, line_num)
        if self.correlations is not None:
            extra, line_num = self.correlations.emit(mat, mf, mt, line_num)
            out.extend(extra)
        return out, line_num


@dataclass
class UnresolvedBody(_Body):
    """§32.2.4 — relative covariances of the average unresolved parameters.

    Deliberately simpler than the resolved formats: no energy dependence, one
    LIST per L value giving the average parameters of each J state, and a single
    relative covariance triangle over all (L, J) combinations. ``dap`` is unused
    here — the unresolved control record has no ISR field.
    """

    l_blocks: List[Record] = field(default_factory=list)
    matrix: Optional[Record] = None

    @property
    def nls(self) -> int:
        return self.control.n1 if self.control is not None else 0

    @property
    def mpar(self) -> int:
        """Average parameters covered, in the order D, GNO, GG, GF, GX."""
        return self.matrix.l1 if self.matrix is not None else 0

    def emit(self, mat, mf, mt, line_num):
        return emit_all([self.control, *self.l_blocks, self.matrix],
                        mat, mf, mt, line_num)


@dataclass
class ScatteringRadiusCovariance:
    """The NRO!=0 preamble: NI subsubsections for the scattering radius.

    §32.2 puts this before the LCOMP record when File 2 gives the scattering
    radius as a function of energy. No tape on this machine exercises it.
    """

    control: Optional[Record] = None
    subsections: List[Record] = field(default_factory=list)

    def emit(self, mat, mf, mt, line_num):
        return emit_all([self.control, *self.subsections], mat, mf, mt, line_num)


# ======================================================================
# The tree
# ======================================================================

@dataclass
class CovEnergyRange:
    """One energy range of one isotope, and the covariance body it carries."""

    el: float = 0.0
    eh: float = 0.0
    lru: int = 0
    lrf: int = 0
    nro: int = 0
    naps: int = 0
    control: Optional[Record] = None
    radius: Optional[ScatteringRadiusCovariance] = None
    body: Any = None

    def emit(self, mat, mf, mt, line_num):
        out, line_num = emit_all([self.control], mat, mf, mt, line_num)
        for part in (self.radius, self.body):
            if part is not None:
                lines, line_num = part.emit(mat, mf, mt, line_num)
                out.extend(lines)
        return out, line_num

    def __repr__(self) -> str:
        kind = type(self.body).__name__ if self.body is not None else "empty"
        return (f"CovEnergyRange({self.el:g}-{self.eh:g} eV, "
                f"LRU={self.lru}, LRF={self.lrf}, {kind})")


@dataclass
class CovIsotope:
    """One isotope's ranges. File 32 need not cover every isotope of File 2."""

    zai: float = 0.0
    abn: float = 0.0
    lfw: int = 0
    control: Optional[Record] = None
    energy_ranges: List[CovEnergyRange] = field(default_factory=list)

    @property
    def num_energy_ranges(self) -> int:
        return len(self.energy_ranges)

    def emit(self, mat, mf, mt, line_num):
        out, line_num = emit_all([self.control], mat, mf, mt, line_num)
        for energy_range in self.energy_ranges:
            lines, line_num = energy_range.emit(mat, mf, mt, line_num)
            out.extend(lines)
        return out, line_num


@dataclass
class MF32MT151(MT):
    """MF32/MT=151: the resonance parameter covariances of a material."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _nis: Optional[int] = None
    _mat: Optional[int] = None
    #: Carried as a field rather than hardcoded in ``__str__``. ``MF33MT`` writes
    #: a literal 33 and so mis-stamps the MF31 sections that share its class;
    #: MF32 has no such twin today, but the cost of not repeating the shape is nil.
    _mf: int = 32
    head: Optional[Record] = None
    isotopes: List[CovIsotope] = field(default_factory=list)

    @property
    def num_isotopes(self) -> int:
        return self._nis if self._nis is not None else len(self.isotopes)

    def energy_ranges(self) -> List[CovEnergyRange]:
        """Every range of every isotope, in file order."""
        return [r for isotope in self.isotopes for r in isotope.energy_ranges]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        line_num = 1
        lines: List[str] = []
        if self.head is not None:
            lines, line_num = self.head.emit(mat, mf, mt, line_num)
        for isotope in self.isotopes:
            isotope_lines, line_num = isotope.emit(mat, mf, mt, line_num)
            lines.extend(isotope_lines)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        bodies = ",".join(
            type(r.body).__name__.replace("Body", "")
            for r in self.energy_ranges()
        )
        return f"MF32MT151({self.number}, NIS={self.num_isotopes}, [{bodies}])"
