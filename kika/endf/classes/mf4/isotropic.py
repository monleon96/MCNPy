from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple
import numpy as np

from .base import MF4MT


@dataclass
class MF4MTIsotropic(MF4MT):
    """
    MT section in MF4 with isotropic angular distributions (LTT=0).
    
    For isotropic distributions, the probability is constant (1.0) for all angles.
    """
    _ltt: int = 0
        
    def extract_legendre_coefficients(
        self,
        energy: Union[float, np.ndarray],
        max_legendre_order: int = 10,
        *,
        trim: bool = False,
        trim_tol: float = 1e-6,
        out_of_range: str = "zero"
    ) -> Dict[int, Union[float, np.ndarray]]:
        """
        Extract Legendre coefficients for isotropic distribution (LTT=0).
        
        For isotropic: a_0(E) = 1, a_l(E) = 0 for l > 0
        
        Parameters
        ----------
        energy : float or np.ndarray
            Energy point(s) at which to evaluate coefficients
        max_legendre_order : int, optional
            Maximum Legendre order to compute (default: 10)
        trim, trim_tol : bool, float, optional
            Accepted for API consistency with the other MF4 classes, and
            inert here: an isotropic section's a_l>0 are exactly zero, so
            trimming them would drop orders a caller asked for by name.
        out_of_range : str, optional
            Ignored for isotropic distributions since they don't depend on energy data tables.
            Included for API consistency with other MF4 implementations. (default: "zero")
            
        Returns
        -------
        Dict[int, Union[float, np.ndarray]]
            Dictionary mapping Legendre indices to coefficients
        """
        if isinstance(energy, np.ndarray):
            result = {0: np.ones_like(energy, dtype=float)}
            for l in range(1, max_legendre_order + 1):
                result[l] = np.zeros_like(energy, dtype=float)
        else:
            result = {0: 1.0}
            for l in range(1, max_legendre_order + 1):
                result[l] = 0.0
        return result
    
    def to_plot_data(
        self,
        order: int,
        energy_range: Tuple[float, float] = (1e-5, 20e6),
        num_points: int = 100,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for isotropic distribution (LTT=0).
        
        For isotropic distributions:
        - a_0(E) = 1.0 for all energies (constant)
        - a_l(E) = 0.0 for all l > 0 (constant)
        
        Since isotropic distributions have no energy dependence, this method creates
        plot data over a specified energy range showing the constant values.
        
        Parameters
        ----------
        order : int
            Legendre polynomial order to extract (0 gives 1.0, l>0 gives 0.0)
        energy_range : tuple of float, optional
            (E_min, E_max) energy range for the plot (default: 1e-5 to 20 MeV)
        num_points : int, optional
            Number of energy points to generate (default: 100)
        label : str, optional
            Custom label for the plot. If None, auto-generates.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        LegendreCoeffPlotData
            Plot data object showing constant coefficient value
            
        Examples
        --------
        >>> # Isotropic distribution - a_0 = 1 everywhere
        >>> data = mf4_isotropic.to_plot_data(order=0, color='blue')
        >>> builder = PlotBuilder().add_data(data).build()
        >>> 
        >>> # Higher orders are zero for isotropic
        >>> data_l1 = mf4_isotropic.to_plot_data(order=1, color='red')
        >>> # Will show a horizontal line at y=0
        
        Notes
        -----
        Isotropic distributions (LTT=0) have no energy dependence. The returned
        plot data shows the constant coefficient value across the specified energy range.
        """
        from kika.plotting import LegendreCoeffPlotData
        
        # Generate energy grid
        energies = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), num_points)
        
        # Generate coefficient values (constant)
        if order == 0:
            coeffs = np.ones(num_points, dtype=float)
        else:
            coeffs = np.zeros(num_points, dtype=float)
        
        # Get isotope information
        isotope = getattr(self, 'isotope', None)
        if isotope is None and hasattr(self, 'zaid'):
            isotope = str(self.zaid)
        
        mt = getattr(self, 'number', None)
        
        # Auto-generate label if not provided
        if label is None:
            if order == 0:
                label = f'Isotropic (L=0, a_0=1)'
            else:
                label = f'Isotropic (L={order}, a_{order}=0)'
        
        return LegendreCoeffPlotData(
            x=energies,
            y=coeffs,
            order=order,
            isotope=isotope,
            mt=mt,
            energy_range=energy_range,
            label=label,
            **styling_kwargs
        )

    def to_bulk_plot_data(
        self,
        max_order: int = 12,
        energy_range: Tuple[float, float] = (1e-5, 20e6),
        num_points: int = 100
    ) -> Dict[str, Union[List[float], Dict[int, List[float]], int, str]]:
        """
        Extract ALL Legendre orders at once for bulk loading.

        For isotropic distributions:
        - a_0(E) = 1.0 for all energies
        - a_l(E) = 0.0 for all l > 0

        Parameters
        ----------
        max_order : int, optional
            Maximum Legendre order to include (default: 12)
        energy_range : tuple, optional
            (E_min, E_max) energy range for the generated grid (default: 1e-5 to 20 MeV)
        num_points : int, optional
            Number of energy points to generate (default: 100)

        Returns
        -------
        dict
            Dictionary containing:
            - 'energies': list of energy values (eV)
            - 'coefficients_by_order': dict mapping order (int) to list of coefficients
            - 'max_order': maximum order extracted
            - 'isotope': isotope identifier (if available)
            - 'mt': MT reaction number
        """
        # Generate energy grid
        energies = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), num_points)

        coefficients_by_order: Dict[int, List[float]] = {}

        # Order 0 is always 1.0
        coefficients_by_order[0] = [1.0] * num_points

        # All higher orders are 0.0 for isotropic
        for order in range(1, max_order + 1):
            coefficients_by_order[order] = [0.0] * num_points

        return {
            'energies': energies.tolist(),
            'coefficients_by_order': coefficients_by_order,
            'max_order': max_order,
            'isotope': getattr(self, 'isotope', None),
            'mt': getattr(self, 'number', None),
        }

    def __str__(self) -> str:
        """
        Convert the MF4MTIsotropic object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        line_num = 1
        
        # Format first line - header - ZA, AWR as float, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        line_num += 1
        
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, 2,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # SEND record: C1=0.0, C2=0.0, L1=0, L2=0, N1=0, N2=0, MT=0
        end_line = format_endf_data_line(
            [0.0, 0.0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(end_line)
        
        return "\n".join(lines)