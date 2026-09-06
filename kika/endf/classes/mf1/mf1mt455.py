"""
MF1/MT455 — Delayed neutron data: nu-bar_d(E) and precursor decay constants.

Structure depends on two flags:
  LDG  decay-constant energy dependence (0 = constant, 1 = energy-dependent)
  LNU  nubar representation (1 = polynomial, 2 = tabulated)

Record layout:
  HEAD: [ZA, AWR, LDG, LNU, 0, 0]

  Decay-constant block:
    LDG=0: LIST [0, 0, 0, 0, NNF, 0 / lambda_1 .. lambda_NNF]
    LDG=1: TAB2 [0, 0, 0, 0, NR, NE / interp]
           + NE LIST records [0, E_i, 0, 0, NNF*2, 0 / lambda_1, sig_1, ..]

  Nubar block:
    LNU=1: LIST [0, 0, 0, 0, NC, 0 / C_1 .. C_NC]
    LNU=2: TAB1 [0, 0, 0, 0, NR, NP / interp / E, nu_d]
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ..mt import MT
from .mf1mt452 import evaluate_nubar
from ...utils import (
    format_endf_send_record,
    format_endf_data_line,
    format_tab1,
    format_tab2,
    format_data_values,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
)


@dataclass
class MF1MT455(MT):
    """MT455 section: delayed neutron data."""

    number: int = 455

    # HEAD record
    _za: float = None
    _awr: float = None
    _mat: int = None
    _ldg: int = None  # 0 = energy-independent decay constants, 1 = energy-dependent
    _lnu: int = None  # 1 = polynomial, 2 = tabulated

    # Decay constants — LDG=0 (energy-independent)
    _nnf: int = 0                               # number of precursor families
    _decay_constants: List[float] = field(default_factory=list)  # lambda_i values

    # Decay constants — LDG=1 (energy-dependent)
    _decay_nr: int = 0
    _decay_ne: int = 0
    _decay_interp: List[Tuple[int, int]] = field(default_factory=list)
    _decay_energies: List[float] = field(default_factory=list)
    _decay_data: List[List[float]] = field(default_factory=list)  # NE lists of NNF*2 values

    # Nubar — polynomial (LNU=1)
    _nc: int = 0
    _coefficients: List[float] = field(default_factory=list)

    # Nubar — tabulated (LNU=2)
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
    def ldg(self) -> int:
        return self._ldg

    @property
    def lnu(self) -> int:
        return self._lnu

    @property
    def num_precursor_families(self) -> int:
        return self._nnf

    @property
    def decay_constants(self) -> List[float]:
        return self._decay_constants

    @property
    def representation(self) -> str:
        if self._lnu == 1:
            return "polynomial"
        elif self._lnu == 2:
            return "tabulated"
        return "unknown"

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

    # --- methods ---

    def get_nubar(
        self,
        energy: Union[float, ArrayLike],
        out_of_range: str = "hold",
    ) -> Union[float, np.ndarray]:
        """Delayed nu-bar at one or more incident energies. See
        :func:`~kika.endf.classes.mf1.mf1mt452.evaluate_nubar`."""
        return evaluate_nubar(
            self._lnu, self._coefficients, self._energies, self._nubar,
            self._interpolation, energy, out_of_range=out_of_range,
        )

    # --- serialization ---

    def __str__(self) -> str:
        """Serialize back to ENDF format."""
        mat = self._mat if self._mat is not None else 0
        mf = 1
        mt = self.number
        lines = []
        line_num = 1

        # HEAD record: [ZA, AWR, LDG, LNU, 0, 0]
        head = format_endf_data_line(
            [self._za, self._awr, self._ldg, self._lnu, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
        )
        lines.append(head)
        line_num += 1

        # --- Decay-constant block ---
        if self._ldg == 0:
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

        elif self._ldg == 1:
            # TAB2: [0, 0, 0, 0, NR, NE / interp]
            tab2_lines, line_num = format_tab2(
                0.0, 0.0, 0, 0,
                self._decay_interp, self._decay_ne,
                mat, mf, mt, line_num,
            )
            lines.extend(tab2_lines)

            # NE LIST records
            for i, energy in enumerate(self._decay_energies):
                npl = len(self._decay_data[i])
                list_head = format_endf_data_line(
                    [0.0, energy, 0, 0, npl, 0],
                    mat, mf, mt, line_num,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                             ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO],
                )
                lines.append(list_head)
                line_num += 1
                data_lines, line_num = format_data_values(
                    list(self._decay_data[i]), mat, mf, mt, line_num,
                )
                lines.extend(data_lines)

        # --- Nubar block ---
        if self._lnu == 1:
            # LIST: [0, 0, 0, 0, NC, 0 / coefficients]
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
            # TAB1: [0, 0, 0, 0, NR, NP / interp / E, nu_d]
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
        rep = self.representation
        families = self._nnf
        ldg = "energy-independent" if self._ldg == 0 else "energy-dependent"
        if self._lnu == 2:
            pts = len(self._energies)
            return f"MF1MT455(delayed nubar, {rep}, {pts} points, {families} families, {ldg} decay)"
        return f"MF1MT455(delayed nubar, {rep}, {families} families, {ldg} decay)"
