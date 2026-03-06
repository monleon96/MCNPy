from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ..mt import MT
from ...utils import (
    interpolate_1d_endf,
    format_endf_data_line,
    format_tab1,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
)


@dataclass
class MF3MT(MT):
    """
    MT section in MF3 (Reaction Cross Sections).

    Each MF3/MT section stores a single TAB1 record giving the
    cross section sigma(E) as a function of incident energy E.
    """

    _za: float = None
    _awr: float = None
    _mat: int = None
    _qm: float = 0.0
    _qi: float = 0.0
    _lr: int = 0
    _nr: int = 0
    _np: int = 0
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)
    _energies: List[float] = field(default_factory=list)
    _cross_sections: List[float] = field(default_factory=list)

    # Line count
    num_lines: int = 0

    # --- properties ---

    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z + A)."""
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio (nucleus mass / neutron mass)."""
        return self._awr

    @property
    def q_mass_difference(self) -> float:
        """Mass-difference Q value (eV)."""
        return self._qm

    @property
    def q_reaction(self) -> float:
        """Reaction Q value for the lowest energy state (eV)."""
        return self._qi

    @property
    def breakup_flag(self) -> int:
        """Complex/breakup reaction flag (LR)."""
        return self._lr

    @property
    def energies(self) -> List[float]:
        """Energy grid (eV)."""
        return self._energies

    @property
    def cross_sections(self) -> List[float]:
        """Cross section values (barns)."""
        return self._cross_sections

    @property
    def num_energy_points(self) -> int:
        """Number of energy/cross-section pairs."""
        return self._np or len(self._energies)

    @property
    def num_interpolation_regions(self) -> int:
        """Number of interpolation regions."""
        return self._nr or 0

    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs (NBT, INT)."""
        return self._interpolation

    # --- methods ---

    def get_cross_section(
        self,
        energy: Union[float, ArrayLike],
        out_of_range: str = "zero",
    ) -> Union[float, np.ndarray]:
        """
        Interpolate the cross section at one or more energies using
        the ENDF interpolation law(s) stored in this section.

        Parameters
        ----------
        energy : float or array-like
            Query energy / energies in eV.
        out_of_range : str
            ``'zero'`` (default) or ``'hold'``.

        Returns
        -------
        float or np.ndarray
            Cross section(s) in barns.
        """
        return interpolate_1d_endf(
            self._energies,
            self._cross_sections,
            self._interpolation,
            energy,
            out_of_range=out_of_range,
        )

    def to_plot_data(self, label: Optional[str] = None, **styling_kwargs):
        """
        Create a ``CrossSectionPlotData`` for this section.

        Parameters
        ----------
        label : str, optional
            Plot label.
        **styling_kwargs
            Passed to ``CrossSectionPlotData`` (color, linestyle, …).

        Returns
        -------
        CrossSectionPlotData
        """
        from kika.plotting import CrossSectionPlotData

        energies = np.array(self._energies, dtype=float)
        xs = np.array(self._cross_sections, dtype=float)

        isotope = str(int(self._za)) if self._za is not None else None
        mt = self.number

        if label is None:
            label = f"MT={mt}"
            if isotope:
                label = f"{isotope} {label}"

        return CrossSectionPlotData(
            x=energies,
            y=xs,
            isotope=isotope,
            mt=mt,
            energy_range=(float(energies.min()), float(energies.max())) if len(energies) else None,
            data_source="endf",
            label=label,
            **styling_kwargs,
        )

    def __str__(self) -> str:
        """Serialize back to ENDF format (HEAD + TAB1 + SEND)."""
        mat = self._mat if self._mat is not None else 0
        mf = 3
        mt = self.number
        lines = []
        line_num = 1

        # HEAD record
        head = format_endf_data_line(
            [self._za, self._awr, 0, 0, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO],
        )
        lines.append(head)
        line_num += 1

        # TAB1 record
        tab1_lines, line_num = format_tab1(
            self._qm, self._qi, 0, self._lr,
            self._interpolation, self._energies, self._cross_sections,
            mat, mf, mt, line_num,
        )
        lines.extend(tab1_lines)

        # SEND
        send = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_INT] * 6,
        )
        lines.append(send)

        return "\n".join(lines)
