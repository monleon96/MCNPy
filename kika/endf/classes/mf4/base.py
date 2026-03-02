"""
Classes for MT sections within MF4 (Angular Distributions) in ENDF files.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from ..mt import MT


@dataclass
class MF4MT(MT):
    """
    Base class for MT sections within MF4 (Angular Distributions).
    
    This class provides a common interface for all angular distribution formats.
    """
    _za: float = None    # ZA identifier
    _awr: float = None   # Atomic weight ratio
    _ltt: int = None     # Angular distribution format flag
    _li: int = None      # Flag to specify if angular distributions are isotropic (0=not all isotropic, 1=all isotropic)
    _lct: int = None     # Frame of reference (1=LAB, 2=CM)
    _mat: int = None     # Material identifier
    
    # Line count
    num_lines: int = 0  # Number of lines in this MT section
    
    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z+A)"""
        return self._za
    
    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio"""
        return self._awr
    
    @property
    def type(self) -> str:
        """Angular distribution format flag"""
        dist_type = ''
        if self._ltt == 0: dist_type = 'Isotropic'
        elif self._ltt == 1: dist_type = 'Legendre'
        elif self._ltt == 2: dist_type = 'Tabulated'
        elif self._ltt == 3: dist_type = 'Legendre and Tabulated'
        else:
            raise ValueError(f"Invalid LTT value: {self._ltt}. Expected 0, 1, 2, or 3.")
        return dist_type
    
    @property
    def is_isotropic(self) -> bool:
        """Flag for identical particles (0=not all isotropic, 1=all isotropic)"""
        if self._li == 0:
            return False
        elif self._li == 1:
            return True
        else:
            raise ValueError(f"Invalid value for LI: {self._li}. Expected 0 or 1.")
    
    @property
    def frame(self) -> str:
        """Frame of reference (1=LAB system, 2=CM system)"""
        if self._lct == 1: 
            return "LAB"
        elif self._lct == 2:
            return "CM"
        else:
            raise ValueError(f"Invalid value for LCT: {self._lct}. Expected 1 or 2.")
    
    def to_dense_plot_data(
        self,
        order: int,
        energy_range: Tuple[float, float],
        num_points: int = 1000,
        label: Optional[str] = None,
        **styling_kwargs,
    ):
        """
        Evaluate Legendre coefficients on a dense log-spaced grid and return plot data.

        This method works on any MF4MT subclass that implements
        ``extract_legendre_coefficients()``.

        Parameters
        ----------
        order : int
            Legendre polynomial order to extract.
        energy_range : tuple of float
            ``(e_min, e_max)`` in eV for the dense evaluation grid.
        num_points : int, default 1000
            Number of log-spaced evaluation points.
        label : str, optional
            Plot label.  Auto-generated if *None*.
        **styling_kwargs
            Passed through to ``LegendreCoeffPlotData`` (color, linestyle, …).

        Returns
        -------
        LegendreCoeffPlotData
        """
        from kika.plotting import LegendreCoeffPlotData

        e_min, e_max = energy_range
        dense_e = np.logspace(np.log10(e_min), np.log10(e_max), num_points)

        coeffs_dict = self.extract_legendre_coefficients(
            energy=dense_e,
            max_legendre_order=order,
            out_of_range="zero",
        )
        coeff_vals = coeffs_dict[order]

        isotope = getattr(self, "isotope", None)
        if isotope is None and hasattr(self, "zaid"):
            isotope = str(self.zaid)
        mt = getattr(self, "number", None)

        if label is None:
            label = f"ENDF L={order}"

        return LegendreCoeffPlotData(
            x=dense_e,
            y=coeff_vals,
            order=order,
            isotope=isotope,
            mt=mt,
            energy_range=energy_range,
            label=label,
            plot_type="line",
            **styling_kwargs,
        )

    def __str__(self) -> str:
        """
        Convert the MF4MT object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        
        # Format first line - header
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, 1,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        
        return "\n".join(lines)










