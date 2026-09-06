"""MF35: covariances of energy distributions (ENDF-6 §35).

One HEAD record and NK subsections, each a LIST covering one **incident-energy
band** and holding an absolute covariance over an outgoing-energy grid.

**What the matrix is the covariance of, and why it matters here.** Measured on
ENDF/B-VIII.1 U-235, JEFF-4.0 U-235 and ENDF/B-VIII.1 Cf-252: the unweighted row
sums vanish (``max|Σ_j C_ij| / max|C|`` between 1.4e-7 and 3.1e-3) while the
dE-weighted row sums do not, and ``sqrt(diag C)/P`` is a few per cent while
``sqrt(diag C)/(P/dE)`` is order 1e+3. So the entries are the covariance of the
**group-integrated probabilities** ``P_i = ∫_{g_i}^{g_i+1} χ(E→E') dE'``, not of
the spectrum density. ``Σ_i P_i = 1`` is what forces ``C·1 ≈ 0``.

Everything downstream rests on that reading, so
:func:`row_sum_residual` is provided here and the sampler refuses a band that
fails it rather than quietly perturbing something else.

**Only LS=1, LB=7 exists.** All three tapes carry that and nothing else. MF33's
LB decoders are not reused: the record shape is close, but they are welded to
MF33's NC/NI subsection hierarchy, which MF35 does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    format_data_values,
    format_endf_data_line,
    format_endf_send_record,
)


@dataclass
class MF35SubSection:
    """One incident-energy band: ``E1, E2, LS, LB, NT, NE`` and the LIST body.

    The body is NE outgoing-energy boundaries followed by the upper triangle
    (diagonal included) of a symmetric ``(NE-1) × (NE-1)`` matrix, so
    ``NT = NE + NE(NE-1)/2``.
    """

    e1: float = 0.0
    e2: float = 0.0
    ls: int = 1
    lb: int = 7
    nt: int = 0
    ne: int = 0
    boundaries: List[float] = field(default_factory=list)
    upper_triangle: List[float] = field(default_factory=list)
    #: The LIST body exactly as read, kept so the section can be written back
    #: rather than rebuilt from the densified matrix. Mirrors MF33's
    #: ``NISubSubsectionRecord``: the values that reach the writer are the
    #: values that came off the tape, in that order.
    raw_list_values: List[float] = field(default_factory=list)

    @property
    def order(self) -> int:
        """``NE - 1`` — the number of groups, and the matrix dimension."""
        return max(self.ne - 1, 0)

    @staticmethod
    def expected_nt(ne: int) -> int:
        return ne + ne * (ne - 1) // 2

    def matrix(self) -> np.ndarray:
        """Densify the stored upper triangle into a symmetric matrix.

        Fifteen lines, and the one place a transposition or an off-by-one in
        the triangle ordering could hide, so it is unit-tested against a
        hand-built 3×3 rather than only against real tapes.
        """
        n = self.order
        out = np.zeros((n, n), dtype=float)
        values = np.asarray(self.upper_triangle, dtype=float)
        cursor = 0
        for row in range(n):
            width = n - row
            out[row, row:] = values[cursor:cursor + width]
            cursor += width
        return out + np.triu(out, 1).T

    def energy_grid(self) -> np.ndarray:
        return np.asarray(self.boundaries, dtype=float)

    # ------------------------------------------------------------------
    def row_sum_residual(self) -> float:
        """``max_i |Σ_j C_ij| / max|C|`` — how close ``C·1`` is to zero.

        The test of the group-probability reading. Measured between 1.4e-7 and
        3.1e-3 on the three reference tapes; JEFF-4.0 U-235's band 0 is the
        3.1e-3 outlier and sets the tolerance anyone downstream should use.
        """
        matrix = self.matrix()
        if matrix.size == 0:
            return 0.0
        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            return 0.0
        return float(np.max(np.abs(matrix.sum(axis=1))) / scale)

    def normalisation_drift(self) -> float:
        """``sqrt(1ᵀC1)`` — the standard deviation of a draw's sum-rule drift.

        The budget a linear-space draw has to stay inside before the projection
        closes it. Measured 9.5e-7 … 3.2e-5 on the reference tapes: physically
        negligible, ~75× the input tapes' own residual, and therefore something
        to close explicitly rather than assume away.

        ``1ᵀC1`` comes out slightly *negative* on about half the bands — it is
        a near-cancelling sum of ~4e5 terms, so the rounding noise straddles
        zero. The magnitude is reported rather than clamped to 0.0, because a
        band whose budget prints as exactly zero reads as "no drift is
        possible here", which is not what was measured.
        """
        return float(np.sqrt(abs(float(self.matrix().sum()))))

    def emit(self, mat: int, mf: int, mt: int, line_num: int):
        """The LIST record: header then NT values, six per line."""
        values = (self.raw_list_values
                  if self.raw_list_values
                  else list(self.boundaries) + list(self.upper_triangle))
        header = format_endf_data_line(
            [self.e1, self.e2, self.ls, self.lb, len(values), self.ne],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        line_num += 1
        body, line_num = format_data_values(list(values), mat, mf, mt, line_num)
        return [header] + body, line_num


@dataclass
class MF35MT(MT):
    """One MT section of MF35: NK bands of one reaction's spectrum covariance."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _nk: Optional[int] = None
    _mat: Optional[int] = None
    _mf: int = 35
    subsections: List[MF35SubSection] = field(default_factory=list)

    @property
    def num_bands(self) -> int:
        return self._nk if self._nk is not None else len(self.subsections)

    def band_for_incident(self, energy: float) -> Optional[int]:
        """Index of the band containing *energy*, or None.

        Bands are half-open ``[E1, E2)`` so that a band edge — which, measured,
        is always also an MF5 incident node — belongs to exactly one band. The
        last band is closed at the top, otherwise the highest incident energy
        on the tape would fall outside every band.
        """
        last = len(self.subsections) - 1
        for index, band in enumerate(self.subsections):
            if band.e1 <= energy < band.e2:
                return index
            if index == last and energy == band.e2:
                return index
        return None

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        line_num = 1
        lines = [format_endf_data_line(
            [self._za, self._awr, 0, 0, self.num_bands, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )]
        line_num += 1

        for band in self.subsections:
            band_lines, line_num = band.emit(mat, mf, mt, line_num)
            lines.extend(band_lines)

        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        sizes = ",".join(str(b.ne) for b in self.subsections)
        return f"MF35MT({self.number}, NK={self.num_bands}, NE=[{sizes}])"
