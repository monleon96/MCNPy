from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Sequence, Optional
import math
import numpy as np

from .base import MF4MT
from ....endf.utils import (
    interpolate_1d_endf, segment_int_codes, interp_energy_values, project_tabulated_to_legendre
)


@dataclass
class MF4MTTabulated(MF4MT):
    """
    MT section in MF4 with tabulated probability distributions (LTT=2).
    
    Stores tabulated angular distributions for each energy.
    """
    _ltt: int = 2
    _ne: int = None  # Number of energy points
    _nr: int = None  # Number of interpolation regions for energy grid
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Energy interpolation scheme pairs
    
    # Tabulated data storage
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _cosines: List[List[float]] = field(default_factory=list)  # Cosine values for each energy
    _probabilities: List[List[float]] = field(default_factory=list)  # Probability values for each energy
    _angular_interpolation: List[List[Tuple[int, int]]] = field(default_factory=list)  # Angular interpolation schemes
    
    @property
    def num_energy_points(self) -> int:
        """Number of energy points in the grid"""
        return self._ne or len(self._energies)
    
    @property
    def num_interpolation_regions(self) -> int:
        """Number of interpolation regions for the energy grid"""
        return self._nr or 0
    
    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for energy grid (NBT, INT)"""
        return self._interpolation
    
    @property
    def energies(self) -> List[float]:
        """Energy grid for angular distribution data"""
        return self._energies
    
    @property
    def cosines(self) -> List[List[float]]:
        """
        Cosine values (μ) for each energy point.
        
        Returns a list of cosine lists, aligned with the energy grid.
        Each inner list contains the cosine values for one energy point.
        """
        return self._cosines
    
    @property
    def probabilities(self) -> List[List[float]]:
        """
        Probability values f(μ,E) for each energy point and cosine.
        
        Returns a list of probability lists, aligned with the energy grid and cosines.
        Each inner list contains the probability values for one energy point.
        """
        return self._probabilities
    
    @property
    def cosine_interpolation(self) -> List[List[Tuple[int, int]]]:
        """
        Interpolation scheme pairs for angular data at each energy.
        
        Returns a list of interpolation scheme lists, aligned with the energy grid.
        Each inner list contains (NBT, INT) pairs for one energy point.
        """
        return self._angular_interpolation
    

    # ------------------------- core helpers -------------------------
    def _energy_panel_code_for_pair(self, upper_index: int) -> int:
        """
        Return the ENDF INT code for the interval (E[upper_index-1], E[upper_index]).
        upper_index runs from 1 to NE-1 (inclusive).
        """
        ne = len(self._energies)
        pairs = self._interpolation if self._interpolation else [(ne, 2)]
        seg_int = segment_int_codes(ne, pairs)
        return int(seg_int[upper_index - 1])

    def _f_mu_at_energy(self, E: float, mu_points: np.ndarray, out_of_range: str = "zero") -> np.ndarray:
        """
        Evaluate f(μ, E) at requested E using ENDF-correct 2D interpolation:
          1) within each energy table, interpolate in μ using that table's (NBT,INT),
          2) then interpolate in E between the bracketing energies using the energy (NBT,INT).
          
        Parameters
        ----------
        E : float
            Energy at which to evaluate the distribution
        mu_points : np.ndarray
            Cosine values where to evaluate f(μ, E)
        out_of_range : str
            Behavior outside energy grid: 'zero' or 'hold'
        """
        energies = np.asarray(self._energies, dtype=float)
        if energies.size == 0:
            return np.zeros_like(mu_points, dtype=float)

        # locate bracketing energies
        if E <= energies[0]:
            idx0, idx1 = 0, 0
        elif E >= energies[-1]:
            idx0, idx1 = len(energies) - 1, len(energies) - 1
        else:
            idx1 = int(np.searchsorted(energies, E, side="right"))
            idx0 = idx1 - 1

        def f_at_table(i: int) -> np.ndarray:
            mu_i = np.asarray(self._cosines[i], dtype=float)
            f_i = np.asarray(self._probabilities[i], dtype=float)
            ang_pairs = self._angular_interpolation[i] if (i < len(self._angular_interpolation) and self._angular_interpolation[i]) else [(len(mu_i), 2)]
            # angular interpolation (μ) using shared utils with consistent out_of_range
            return interpolate_1d_endf(mu_i, f_i, ang_pairs, mu_points, out_of_range="hold")

        f0 = f_at_table(idx0)
        if idx1 == idx0:
            fE = f0
        else:
            f1 = f_at_table(idx1)
            code = self._energy_panel_code_for_pair(idx1)
            fE = interp_energy_values(energies[idx0], f0, energies[idx1], f1, E, code)

        return fE

    # ------------------------- public API -------------------------
    def extract_legendre_coefficients(
        self,
        energy: Union[float, np.ndarray],
        max_legendre_order: int = 10,
        *,
        quad_order: int = 96,
        out_of_range: str = "zero"
    ) -> Dict[int, Union[float, np.ndarray]]:
        """
        Compute a_ℓ(E) = (2ℓ+1)/2 ∫_{-1}^{1} P_ℓ(μ) f(μ,E) dμ,
        honoring ENDF angular and energy interpolation laws.
        
        Parameters
        ----------
        energy : float or array
            Energy point(s) where to evaluate a_ℓ(E)
        max_legendre_order : int
            Maximum Legendre order to compute
        quad_order : int  
            Quadrature order for Gauss-Legendre integration
        out_of_range : str
            Behavior outside energy grid: 'zero' or 'hold'
            
        Returns
        -------
        Dict[int, Union[float, np.ndarray]]
            Dictionary mapping Legendre order ℓ to coefficient values a_ℓ(E)
        """
        # Handle empty data
        if len(self._energies) == 0:
            scalar_input = np.isscalar(energy)
            energy_array = np.array([energy], dtype=float) if scalar_input else np.array(energy, dtype=float)
            zeros = {ell: (0.0 if scalar_input else np.zeros_like(energy_array)) 
                    for ell in range(max_legendre_order + 1)}
            return zeros

        scalar_input = np.isscalar(energy)
        E_arr = np.array([energy], dtype=float) if scalar_input else np.array(energy, dtype=float)

        # Gauss–Legendre quadrature nodes/weights on [-1,1]
        mu_q, w_q = np.polynomial.legendre.leggauss(quad_order)

        out = {ell: np.empty(E_arr.shape, dtype=float) for ell in range(max_legendre_order + 1)}

        for k, Ereq in enumerate(E_arr):
            f_q = self._f_mu_at_energy(Ereq, mu_q, out_of_range)

            # Use the utility function for projection
            # Note: project_tabulated_to_legendre expects tabulated (mu, f_mu) data,
            # but we already have f evaluated at quadrature points, so we pass them directly
            coeffs = project_tabulated_to_legendre(
                mu=mu_q,
                fmu=f_q,
                max_order=max_legendre_order,
                ang_nbt_int=[(len(mu_q), 2)],  # Linear interpolation (already interpolated)
                quad_order=quad_order
            )

            # Store results
            for ell in range(max_legendre_order + 1):
                out[ell][k] = coeffs[ell]

        # Return appropriate format
        if scalar_input:
            return {ell: float(vals[0]) for ell, vals in out.items()}
        return out

    def get_distribution_at_energy(self, energy: float) -> Tuple[List[float], List[float]]:
        """
        Return the stored tabulated (μ, f) at an exact grid energy, or ([],[]) if not found.
        """
        try:
            i = self._energies.index(energy)
            return (self._cosines[i], self._probabilities[i])
        except (ValueError, IndexError):
            return ([], [])

    def get_distribution_dict(self) -> Dict[float, Tuple[List[float], List[float]]]:
        """
        Map each grid energy to its (μ, f) table.
        """
        return {e: (c, p) for e, c, p in zip(self._energies, self._cosines, self._probabilities)}
    
    def to_plot_data(
        self,
        order: int,
        label: str = None,
        quad_order: int = 96,
        **styling_kwargs
    ):
        """
        Create a PlotData object for tabulated distribution projected to Legendre coefficients.
        
        For tabulated distributions (LTT=2), Legendre coefficients are computed by
        projecting the tabulated f(μ,E) distributions onto Legendre polynomials using
        Gauss-Legendre quadrature.
        
        Parameters
        ----------
        order : int
            Legendre polynomial order to extract
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        quad_order : int, optional
            Quadrature order for Gauss-Legendre integration when projecting
            tabulated distributions to Legendre coefficients (default: 96)
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        LegendreCoeffPlotData
            Plot data object ready to be added to a PlotBuilder
            
        Examples
        --------
        >>> # Project tabulated distribution to Legendre coefficients
        >>> data = mf4_tabulated.to_plot_data(order=1, color='blue')
        >>> builder = PlotBuilder().add_data(data).build()
        >>> 
        >>> # Use higher quadrature order for better accuracy
        >>> data = mf4_tabulated.to_plot_data(order=2, quad_order=128)
        
        Notes
        -----
        Tabulated distributions (LTT=2) store f(μ,E) at discrete (μ, E) points.
        To obtain Legendre coefficients a_ℓ(E), we compute:
        
            a_ℓ(E) = (2ℓ+1)/2 ∫_{-1}^{1} P_ℓ(μ) f(μ,E) dμ
        
        using Gauss-Legendre quadrature. The accuracy depends on the quad_order parameter.
        Higher orders require higher quadrature orders for accurate integration.
        """
        from kika.plotting import LegendreCoeffPlotData
        
        # Get energy grid from tabulated data
        energies = np.array(self._energies, dtype=float)
        
        if len(energies) == 0:
            raise ValueError("No tabulated data available to create plot")
        
        # Extract Legendre coefficients at all energy points using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=order,
            quad_order=quad_order,
            out_of_range="zero"
        )
        
        # Get the coefficient values for the requested order
        coeff_values = coeffs_dict[order]
        
        # Get isotope information
        isotope = getattr(self, 'isotope', None)
        if isotope is None and hasattr(self, 'zaid'):
            isotope = str(self.zaid)
        
        mt = getattr(self, 'number', None)

        # Auto-generate label if not provided
        if label is None:
            from kika._constants import format_plot_label
            label = format_plot_label(isotope=isotope, mt=mt, order=order)

        return LegendreCoeffPlotData(
            x=energies,
            y=coeff_values,
            order=order,
            isotope=isotope,
            mt=mt,
            energy_range=(energies.min(), energies.max()),
            label=label,
            **styling_kwargs
        )

    def to_bulk_plot_data(
        self,
        max_order: int = 12,
        quad_order: int = 96
    ) -> Dict[str, Union[List[float], Dict[int, List[float]], int, str]]:
        """
        Extract ALL Legendre orders at once for bulk loading.

        For tabulated distributions (LTT=2), coefficients are computed by projecting
        f(μ,E) onto Legendre polynomials using Gauss-Legendre quadrature.

        Parameters
        ----------
        max_order : int, optional
            Maximum Legendre order to compute (default: 12)
        quad_order : int, optional
            Quadrature order for Gauss-Legendre integration (default: 96)

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
        energies = np.array(self._energies, dtype=float)

        if len(energies) == 0:
            return {
                'energies': [],
                'coefficients_by_order': {},
                'max_order': 0,
                'isotope': getattr(self, 'isotope', None),
                'mt': getattr(self, 'number', None),
            }

        # Extract all Legendre coefficients at once using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=max_order,
            quad_order=quad_order,
            out_of_range="zero"
        )

        # Convert to the expected format
        coefficients_by_order: Dict[int, List[float]] = {}
        for order in range(max_order + 1):
            if order in coeffs_dict:
                vals = coeffs_dict[order]
                if isinstance(vals, np.ndarray):
                    coefficients_by_order[order] = vals.tolist()
                else:
                    coefficients_by_order[order] = [float(vals)] * len(energies)
            else:
                coefficients_by_order[order] = [0.0] * len(energies)

        return {
            'energies': energies.tolist(),
            'coefficients_by_order': coefficients_by_order,
            'max_order': max_order,
            'isotope': getattr(self, 'isotope', None),
            'mt': getattr(self, 'number', None),
        }

    def __str__(self) -> str:
        """
        Convert the MF4MTTabulated object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
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
        
        # Format second line - First value as float, rest as integers
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with number of interpolation regions and energy points
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, self._ne or 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Format energy interpolation scheme pairs - all as integers
        if self._interpolation and self._nr and self._nr > 0:
            # Process interpolation pairs in groups of 3 (6 values per line)
            remaining_pairs = self._interpolation.copy()
            while remaining_pairs:
                # Take up to 3 pairs for this line
                line_pairs = remaining_pairs[:3]
                remaining_pairs = remaining_pairs[3:]
                
                # Flatten pairs into a list of values
                values = []
                for nbt, interp in line_pairs:
                    values.append(nbt)
                    values.append(interp)
                
                # All interpolation values are integers
                # Format line - pad with blanks instead of zeros
                formats = [ENDF_FORMAT_INT] * len(values)
                if len(values) < 6:
                    values.extend([None] * (6 - len(values)))
                    formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                    
                interp_line = format_endf_data_line(
                    values, mat, mf, mt, line_num, formats=formats
                )
                lines.append(interp_line)
                line_num += 1
        
        # Format each tabulated distribution
        for i, energy in enumerate(self._energies):
            cosines = self._cosines[i]
            probabilities = self._probabilities[i]
            ang_interp = self._angular_interpolation[i] if i < len(self._angular_interpolation) else []
            
            np_val = len(cosines)  # Number of angular points
            nr_ang = len(ang_interp)  # Number of angular interpolation regions
            
            # Format header for this energy
            energy_header = format_endf_data_line(
                [0.0, energy, 0, 0, nr_ang, np_val],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(energy_header)
            line_num += 1
            
            # Format angular interpolation scheme pairs (if any)
            if ang_interp and nr_ang > 0:
                # Process interpolation pairs in groups of 3 (6 values per line)
                remaining_pairs = ang_interp.copy()
                while remaining_pairs:
                    # Take up to 3 pairs for this line
                    line_pairs = remaining_pairs[:3]
                    remaining_pairs = remaining_pairs[3:]
                    
                    # Flatten pairs into a list of values
                    values = []
                    for nbt, interp in line_pairs:
                        values.append(nbt)
                        values.append(interp)
                    
                    formats = [ENDF_FORMAT_INT] * len(values)
                    # Use None for blank fields instead of zeros
                    if len(values) < 6:
                        values.extend([None] * (6 - len(values)))
                        formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                    
                    # Format line
                    ang_interp_line = format_endf_data_line(
                        values, mat, mf, mt, line_num,
                        formats=formats
                    )
                    lines.append(ang_interp_line)
                    line_num += 1
            
            # Format cosine-probability pairs (3 pairs per line)
            pair_idx = 0
            while pair_idx < np_val:
                # Get up to 3 pairs for this line
                line_values = []
                for j in range(3):
                    if pair_idx + j < np_val:
                        line_values.append(cosines[pair_idx + j])
                        line_values.append(probabilities[pair_idx + j])
                pair_idx += 3
                
                # Use None for blank fields instead of zeros
                if len(line_values) < 6:
                    line_values.extend([None] * (6 - len(line_values)))
                
                # Format line
                pair_line = format_endf_data_line(
                    line_values, mat, mf, mt, line_num
                )
                lines.append(pair_line)
                line_num += 1
        
        # SEND record: C1=0.0, C2=0.0, L1=0, L2=0, N1=0, N2=0, MT=0
        end_line = format_endf_data_line(
            [0.0, 0.0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(end_line)
        
        return "\n".join(lines)