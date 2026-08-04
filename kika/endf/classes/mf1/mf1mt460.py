"""
MF1/MT460 — Delayed photon data.

Two representations:
  LO=1  discrete    NG TAB1 records (time-dependent multiplicity per photon)
  LO=2  continuous  LIST with NNF precursor-family decay constants
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

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
class MF1MT460(MT):
    """MT460 section: delayed photon data."""

    number: int = 460

    # HEAD record
    _za: float = None
    _awr: float = None
    _mat: int = None
    _lo: int = None   # 1 = discrete, 2 = continuous
    _ng: int = 0      # number of discrete photon groups (LO=1)

    # Discrete representation (LO=1): NG TAB1 records
    _photon_data: List[Dict] = field(default_factory=list)
    # Each dict: {energy, index, interpolation, times, multiplicities}

    # Continuous representation (LO=2): LIST with decay constants
    _nnf: int = 0
    _decay_constants: List[float] = field(default_factory=list)

    num_lines: int = 0

    # --- properties ---

    @property
    def zaid(self) -> float:
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        return self._awr

    @property
    def lo(self) -> int:
        return self._lo

    @property
    def representation(self) -> str:
        if self._lo == 1:
            return "discrete"
        elif self._lo == 2:
            return "continuous"
        return "unknown"

    @property
    def num_photon_groups(self) -> int:
        return self._ng

    @property
    def num_precursor_families(self) -> int:
        return self._nnf

    @property
    def decay_constants(self) -> List[float]:
        return self._decay_constants

    # --- serialization ---

    def __str__(self) -> str:
        """Serialize back to ENDF format."""
        mat = self._mat if self._mat is not None else 0
        mf = 1
        mt = self.number
        lines = []
        line_num = 1

        # HEAD record: [ZA, AWR, LO, 0, NG, 0]
        ng_or_zero = self._ng if self._lo == 1 else 0
        head = format_endf_data_line(
            [self._za, self._awr, self._lo, 0, ng_or_zero, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
        )
        lines.append(head)
        line_num += 1

        if self._lo == 1:
            # NG TAB1 records
            for photon in self._photon_data:
                tab1_lines, line_num = format_tab1(
                    photon["energy"], 0.0, photon["index"], 0,
                    photon["interpolation"], photon["times"], photon["multiplicities"],
                    mat, mf, mt, line_num,
                )
                lines.extend(tab1_lines)

        elif self._lo == 2:
            # LIST: [0, 0, 0, 0, NNF, 0 / lambda values]
            nnf = len(self._decay_constants)
            list_head = format_endf_data_line(
                [0.0, 0.0, 0, 0, nnf, 0],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
            )
            lines.append(list_head)
            line_num += 1
            data_lines, line_num = format_data_values(
                list(self._decay_constants), mat, mf, mt, line_num,
            )
            lines.extend(data_lines)

        # SEND
        send = format_endf_send_record(mat, mf)
        lines.append(send)

        return "\n".join(lines)

    def __repr__(self):
        if self._lo == 1:
            return f"MF1MT460(delayed photons, discrete, {self._ng} groups)"
        elif self._lo == 2:
            return f"MF1MT460(delayed photons, continuous, {self._nnf} families)"
        return f"MF1MT460(delayed photons)"
