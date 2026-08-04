"""
MF1/MT452 — Total average number of neutrons per fission, nu-bar(E).

Two representations:
  LNU=1  polynomial   nu(E) = sum_{n=0}^{NC-1} C_n * E^n
  LNU=2  tabulated    TAB1 record of (E, nu) pairs
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..mt import MT
from ...utils import (
    format_endf_send_record,
    format_endf_data_line,
    format_tab1,
    format_data_values,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
)


@dataclass
class MF1MT452(MT):
    """MT452 section: total number of neutrons per fission."""

    number: int = 452

    # HEAD record
    _za: float = None
    _awr: float = None
    _mat: int = None
    _lnu: int = None  # 1 = polynomial, 2 = tabulated

    # Polynomial representation (LNU=1)
    _nc: int = 0
    _coefficients: List[float] = field(default_factory=list)

    # Tabulated representation (LNU=2)
    _nr: int = 0
    _np: int = 0
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)
    _energies: List[float] = field(default_factory=list)
    _nubar: List[float] = field(default_factory=list)

    num_lines: int = 0

    # --- properties ---

    @property
    def zaid(self) -> float:
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        return self._awr

    @property
    def representation(self) -> str:
        """'polynomial' or 'tabulated'."""
        if self._lnu == 1:
            return "polynomial"
        elif self._lnu == 2:
            return "tabulated"
        return "unknown"

    @property
    def lnu(self) -> int:
        return self._lnu

    @property
    def coefficients(self) -> List[float]:
        return self._coefficients

    @property
    def energies(self) -> List[float]:
        return self._energies

    @property
    def nubar_values(self) -> List[float]:
        return self._nubar

    @property
    def interpolation(self) -> List[Tuple[int, int]]:
        return self._interpolation

    # --- serialization ---

    def __str__(self) -> str:
        """Serialize back to ENDF format (HEAD + LIST/TAB1 + SEND)."""
        mat = self._mat if self._mat is not None else 0
        mf = 1
        mt = self.number
        lines = []
        line_num = 1

        # HEAD record: [ZA, AWR, 0, LNU, 0, 0]
        head = format_endf_data_line(
            [self._za, self._awr, 0, self._lnu, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
        )
        lines.append(head)
        line_num += 1

        if self._lnu == 1:
            # LIST record: [0.0, 0.0, 0, 0, NC, 0 / C1..CNC]
            nc = len(self._coefficients)
            list_head = format_endf_data_line(
                [0.0, 0.0, 0, 0, nc, 0],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
            )
            lines.append(list_head)
            line_num += 1
            data_lines, line_num = format_data_values(
                list(self._coefficients), mat, mf, mt, line_num,
            )
            lines.extend(data_lines)

        elif self._lnu == 2:
            # TAB1 record: [0.0, 0.0, 0, 0, NR, NP / interp / E, nu]
            tab1_lines, line_num = format_tab1(
                0.0, 0.0, 0, 0,
                self._interpolation, self._energies, self._nubar,
                mat, mf, mt, line_num,
            )
            lines.extend(tab1_lines)

        # SEND
        send = format_endf_send_record(mat, mf)
        lines.append(send)

        return "\n".join(lines)

    def __repr__(self):
        if self._lnu == 1:
            return f"MF1MT452(total nubar, polynomial, {len(self._coefficients)} coefficients)"
        elif self._lnu == 2:
            return f"MF1MT452(total nubar, tabulated, {len(self._energies)} points)"
        return f"MF1MT452(total nubar)"
