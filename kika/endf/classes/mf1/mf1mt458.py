"""
MF1/MT458 — Components of energy release due to fission.

Nine energy components (each with uncertainty):
  IFC=0: EFR  — kinetic energy of fission fragments
  IFC=1: ENP  — kinetic energy of prompt fission neutrons
  IFC=2: END  — kinetic energy of delayed fission neutrons
  IFC=3: EGP  — total energy from prompt gamma rays
  IFC=4: EGD  — total energy from delayed gamma rays
  IFC=5: EB   — total energy from delayed betas
  IFC=6: ENU  — energy carried away by neutrinos
  IFC=7: ER   — total energy less neutrinos (ER = ET - ENU)
  IFC=8: ET   — total fission energy (pseudo-Q-value)

Two formats:
  LFC=0  polynomial   LIST with 18*(NPLY+1) coefficients
  LFC=1  tabulated    LIST (thermal) + NFC TAB1 records
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

# Component names in order (index 0..8)
COMPONENT_NAMES = ["EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET"]


@dataclass
class MF1MT458(MT):
    """MT458 section: components of energy release due to fission."""

    number: int = 458

    # HEAD record
    _za: float = None
    _awr: float = None
    _mat: int = None
    _lfc: int = None  # 0 = polynomial, 1 = tabulated components
    _nfc: int = 0     # number of tabulated TAB1 records (LFC=1)

    # Polynomial representation (LFC=0)
    _nply: int = 0               # polynomial order
    _coefficients: List[float] = field(default_factory=list)  # flat 18*(NPLY+1) values

    # Tabulated representation (LFC=1)
    _thermal_values: List[float] = field(default_factory=list)  # 18 thermal values
    _tab_components: List[Dict] = field(default_factory=list)   # NFC TAB1 records

    num_lines: int = 0

    # --- properties ---

    @property
    def zaid(self) -> float:
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        return self._awr

    @property
    def lfc(self) -> int:
        return self._lfc

    @property
    def nfc(self) -> int:
        return self._nfc

    @property
    def polynomial_order(self) -> int:
        return self._nply

    def _get_component(self, ifc: int, order: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """Get (value, uncertainty) for component *ifc* at polynomial order *order*."""
        if self._lfc == 0 and self._coefficients:
            base = order * 18 + ifc * 2
            if base + 1 < len(self._coefficients):
                return self._coefficients[base], self._coefficients[base + 1]
        elif self._lfc == 1 and self._thermal_values:
            base = ifc * 2
            if base + 1 < len(self._thermal_values):
                return self._thermal_values[base], self._thermal_values[base + 1]
        return None, None

    @property
    def efr(self) -> Tuple[Optional[float], Optional[float]]:
        """Fission fragment kinetic energy (value, uncertainty) at thermal."""
        return self._get_component(0)

    @property
    def enp(self) -> Tuple[Optional[float], Optional[float]]:
        """Prompt neutron kinetic energy (value, uncertainty) at thermal."""
        return self._get_component(1)

    @property
    def end(self) -> Tuple[Optional[float], Optional[float]]:
        """Delayed neutron kinetic energy (value, uncertainty) at thermal."""
        return self._get_component(2)

    @property
    def egp(self) -> Tuple[Optional[float], Optional[float]]:
        """Prompt gamma energy (value, uncertainty) at thermal."""
        return self._get_component(3)

    @property
    def egd(self) -> Tuple[Optional[float], Optional[float]]:
        """Delayed gamma energy (value, uncertainty) at thermal."""
        return self._get_component(4)

    @property
    def eb(self) -> Tuple[Optional[float], Optional[float]]:
        """Delayed beta energy (value, uncertainty) at thermal."""
        return self._get_component(5)

    @property
    def enu(self) -> Tuple[Optional[float], Optional[float]]:
        """Neutrino energy (value, uncertainty) at thermal."""
        return self._get_component(6)

    @property
    def er(self) -> Tuple[Optional[float], Optional[float]]:
        """Total energy less neutrinos (value, uncertainty) at thermal."""
        return self._get_component(7)

    @property
    def et(self) -> Tuple[Optional[float], Optional[float]]:
        """Total fission energy (value, uncertainty) at thermal."""
        return self._get_component(8)

    # --- serialization ---

    def __str__(self) -> str:
        """Serialize back to ENDF format."""
        mat = self._mat if self._mat is not None else 0
        mf = 1
        mt = self.number
        lines = []
        line_num = 1

        # HEAD record: [ZA, AWR, 0, LFC, 0, NFC]
        head = format_endf_data_line(
            [self._za, self._awr, 0, self._lfc, 0, self._nfc],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
        )
        lines.append(head)
        line_num += 1

        if self._lfc == 0:
            # LIST record: [0.0, 0.0, 0, NPLY, NPL, N2 / coefficients]
            npl = len(self._coefficients)
            n2 = npl // 2  # number of (value, uncertainty) pairs
            list_head = format_endf_data_line(
                [0.0, 0.0, 0, self._nply, npl, n2],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
            )
            lines.append(list_head)
            line_num += 1
            data_lines, line_num = format_data_values(
                list(self._coefficients), mat, mf, mt, line_num,
            )
            lines.extend(data_lines)

        elif self._lfc == 1:
            # LIST record for thermal values
            npl = len(self._thermal_values)
            n2 = npl // 2
            list_head = format_endf_data_line(
                [0.0, 0.0, 0, 0, npl, n2],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
            )
            lines.append(list_head)
            line_num += 1
            data_lines, line_num = format_data_values(
                list(self._thermal_values), mat, mf, mt, line_num,
            )
            lines.extend(data_lines)

            # NFC TAB1 records
            for comp in self._tab_components:
                tab1_lines, line_num = format_tab1(
                    0.0, 0.0, comp["ldrv"], comp["ifc"],
                    comp["interpolation"], comp["energies"], comp["values"],
                    mat, mf, mt, line_num,
                )
                lines.extend(tab1_lines)

        # SEND
        send = format_endf_send_record(mat, mf)
        lines.append(send)

        return "\n".join(lines)

    def __repr__(self):
        if self._lfc == 0:
            return f"MF1MT458(fission energy release, polynomial order {self._nply})"
        elif self._lfc == 1:
            return f"MF1MT458(fission energy release, tabulated, {self._nfc} components)"
        return f"MF1MT458(fission energy release)"
