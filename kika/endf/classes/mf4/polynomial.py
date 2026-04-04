from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
import numpy as np

from .base import MF4MT
from ....endf.utils import (
    get_interpolation_scheme_name, interpolate_1d_endf,
)


@dataclass
class MF4MTLegendre(MF4MT):
    """
    MT section in MF4 with Legendre expansion coefficients (LTT=1).
    
    Stores Legendre coefficients for each energy.
    """
    _ltt: int = 1
    _nr: int = None  # Number of different interpolation intervals
    _ne: int = None  # Number of energy points
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs (NBT, INT)
    
    # Store energy grid and coefficients separately
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _legendre_coeffs: List[List[float]] = field(default_factory=list)  # Legendre coefficients for each energy
    
    @property
    def num_energy_points(self) -> int:
        """Number of energies in the grid"""
        return self._ne or len(self._energies)
    
    @property
    def num_interpolation_regions(self) -> int:
        """Number of different interpolation intervals"""
        return self._nr or 0
    
    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs (NBT, INT)"""
        return self._interpolation
    
    @property
    def legendre_energies(self) -> List[float]:
        """Energy grid for Legendre coefficients (alias for energies for compatibility)"""
        return self._energies

    @property
    def energies(self) -> List[float]:
        """Energy grid (alias for legendre_energies for API consistency with MF4MTTabulated)"""
        return self._energies

    @property
    def legendre_coefficients(self) -> List[List[float]]:
        """
        Legendre coefficients for each energy point.
        
        Returns a list of coefficient lists, aligned with the energy grid.
        Each inner list contains the coefficients for one energy point.
        """
        return self._legendre_coeffs
    
    def get_coefficients_at_energy(self, energy: float) -> List[float]:
        """
        Get Legendre coefficients at a specific energy.
        
        Args:
            energy: Energy point to retrieve coefficients for
            
        Returns:
            List of Legendre coefficients at that energy
        """
        # Find the index of this energy in the grid
        try:
            index = self._energies.index(energy)
            return self._legendre_coeffs[index]
        except (ValueError, IndexError):
            return []
    
    def get_coefficients_dict(self) -> Dict[float, List[float]]:
        """
        Get Legendre coefficients as a dictionary mapping energies to coefficients.
        
        Returns:
            Dictionary with energies as keys and coefficient lists as values
        """
        return {e: c for e, c in zip(self._energies, self._legendre_coeffs)}
    
    # --- Main API: interpolated Legendre coefficients at E ---
    def to_plot_data(
        self,
        order: int,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for this Legendre coefficient data.

        This is a convenience method to easily convert MF4 data into a plottable format
        using the new plotting infrastructure.

        Parameters
        ----------
        order : int
            Legendre polynomial order to extract
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)

        Returns
        -------
        LegendreCoeffPlotData
            Plot data object ready to be added to a PlotBuilder

        Examples
        --------
        >>> # Method 1: Using this convenience method
        >>> data = mf4_obj.to_plot_data(order=1, color='blue')
        >>> builder = PlotBuilder().add_data(data).build()
        >>>
        """
        from kika.plotting import LegendreCoeffPlotData
        
        # Get energy grid from Legendre data
        energies = np.array(self._energies, dtype=float)
        
        if len(energies) == 0:
            raise ValueError("No Legendre data available to create plot")
        
        # Extract Legendre coefficients at all energy points using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=order,
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
            label = f'Legendre L={order}'
            if isotope:
                label = f'{isotope} {label}'
        
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

    def to_bulk_plot_data(self, max_order: int = None) -> Dict[str, Union[List[float], Dict[int, List[float]], int, str]]:
        """
        Extract ALL Legendre orders at once for bulk loading.

        This method is optimized for frontend applications that need to cache
        all orders and switch between them without additional backend requests.

        Parameters
        ----------
        max_order : int, optional
            Maximum Legendre order to extract. If None, uses the maximum
            available order in the data.

        Returns
        -------
        dict
            Dictionary containing:
            - 'energies': list of energy values (eV)
            - 'coefficients_by_order': dict mapping order (int) to list of coefficients
            - 'max_order': maximum order extracted
            - 'isotope': isotope identifier (if available)
            - 'mt': MT reaction number

        Examples
        --------
        >>> bulk_data = mf4_legendre.to_bulk_plot_data()
        >>> # Access order 2 coefficients
        >>> order_2_coeffs = bulk_data['coefficients_by_order'][2]
        >>> # Plot with frontend filtering - no backend call needed
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

        # Determine max order from data if not specified
        file_max_order = 0
        for coeffs in self._legendre_coeffs:
            if coeffs:
                file_max_order = max(file_max_order, len(coeffs))

        if max_order is None:
            max_order = file_max_order
        else:
            max_order = min(max_order, file_max_order) if file_max_order > 0 else max_order

        # Extract coefficients for all orders
        coefficients_by_order: Dict[int, List[float]] = {}

        # Order 0 is always 1.0 (ENDF convention)
        coefficients_by_order[0] = [1.0] * len(energies)

        # Orders 1 to max_order from stored coefficients
        for order in range(1, max_order + 1):
            order_coeffs = []
            for coeffs in self._legendre_coeffs:
                if coeffs and (order - 1) < len(coeffs):
                    order_coeffs.append(float(coeffs[order - 1]))
                else:
                    order_coeffs.append(0.0)
            coefficients_by_order[order] = order_coeffs

        return {
            'energies': energies.tolist(),
            'coefficients_by_order': coefficients_by_order,
            'max_order': max_order,
            'isotope': getattr(self, 'isotope', None),
            'mt': getattr(self, 'number', None),
        }
    
    def extract_legendre_coefficients(
        self,
        energy: Union[float, np.ndarray],
        max_legendre_order: int = 10,
        *,
        out_of_range: str = "zero"
    ) -> Dict[int, Union[float, np.ndarray]]:
        """
        Return a_ℓ(E) coefficients for ℓ=0..max_legendre_order using ENDF energy interpolation.
        Missing higher orders at a given energy are treated as zeros (ENDF convention).
        
        Parameters
        ----------
        energy : float or array
            Energy point(s) where to evaluate a_ℓ(E)
        max_legendre_order : int
            Maximum Legendre order to compute
        out_of_range : str
            Behavior outside energy grid: 'zero' or 'hold'
            
        Returns
        -------
        Dict[int, Union[float, np.ndarray]]
            Dictionary mapping Legendre order ℓ to coefficient values a_ℓ(E)

        Notes
        -----
        * Uses ENDF energy interpolation with regions defined by self._interpolation
        * Energy interpolation regions (NBT, INT) are taken from self._interpolation
        * Missing coefficients at any energy are filled with zeros per ENDF convention
        """
        energies = np.asarray(self._energies, dtype=float)
        coeff_lists = self._legendre_coeffs  # typically a1..a_NL per energy; a0 implicit
        result: Dict[int, Union[float, np.ndarray]] = {}

        # Handle no data
        if energies.size == 0:
            is_scalar = np.isscalar(energy)
            arr = np.array([energy]) if is_scalar else np.asarray(energy, dtype=float)
            for l in range(max_legendre_order + 1):
                result[l] = float(0.0) if is_scalar else np.zeros_like(arr, dtype=float)
            return result

        # Highest available order in file (excluding a0)
        max_available = 0
        for row in coeff_lists:
            if row:
                max_available = max(max_available, len(row))  # row length corresponds to max L present (without a0)
        Lmax = min(max_legendre_order, max_available)

        x_eval = np.array([energy], dtype=float) if np.isscalar(energy) else np.asarray(energy, dtype=float)

        nE = energies.size

        # l=0: a0 = 1 (normalized PDF)
        y0 = np.ones(nE, dtype=float)
        y0_eval = interpolate_1d_endf(energies, y0, self._interpolation, x_eval, out_of_range=out_of_range)
        result[0] = float(y0_eval[0]) if np.isscalar(energy) else y0_eval

        # l>=1 from file rows (index l-1)
        for l in range(1, Lmax + 1):
            y_l = np.empty(nE, dtype=float)
            for i in range(nE):
                row = coeff_lists[i]
                y_l[i] = row[l - 1] if (l - 1) < len(row) else 0.0
            y_eval = interpolate_1d_endf(energies, y_l, self._interpolation, x_eval, out_of_range=out_of_range)
            result[l] = float(y_eval[0]) if np.isscalar(energy) else y_eval

        # Fill remaining orders beyond available
        for l in range(Lmax + 1, max_legendre_order + 1):
            result[l] = float(0.0) if np.isscalar(energy) else np.zeros_like(x_eval, dtype=float)

        return result

    def __str__(self) -> str:
        """
        Convert the MF4MTLegendre object back to ENDF format string.
        
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
        
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, 2,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with interpolation info and energy count
        # Use actual length instead of stored _ne for safety
        ne_actual = len(self._energies)
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, ne_actual],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Ensure interpolation scheme is set if empty
        interpolation_pairs = self._interpolation
        if not interpolation_pairs and ne_actual > 0:
            interpolation_pairs = [(ne_actual, 2)]  # Default to linear-linear
        
        # Format energy interpolation scheme pairs - all as integers
        if interpolation_pairs and len(interpolation_pairs) > 0:
            # Process interpolation pairs in groups of 3 (6 values per line)
            remaining_pairs = interpolation_pairs.copy()
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
        
        # Format Legendre coefficients section
        for i, (energy, coeffs) in enumerate(zip(self._energies, self._legendre_coeffs)):
            # For each energy point, format header and coefficients
            nl = len(coeffs)  # Number of coefficients
            
            # Format header for this energy
            energy_header = format_endf_data_line(
                [0.0, energy, 0, 0, nl, 0],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO]
            )
            lines.append(energy_header)
            line_num += 1
            
            # Format coefficients (6 per line)
            coef_idx = 0
            while coef_idx < nl:
                # Get up to 6 coefficients for this line
                line_coeffs = coeffs[coef_idx:coef_idx + 6]
                coef_idx += 6
                
                # Pad with None (blank) instead of zeros if needed
                if len(line_coeffs) < 6:
                    line_coeffs.extend([None] * (6 - len(line_coeffs)))
                
                # Format line
                coef_line = format_endf_data_line(
                    line_coeffs, mat, mf, mt, line_num
                )
                lines.append(coef_line)
                line_num += 1
        
        # SEND record: C1=0.0, C2=0.0, L1=0, L2=0, N1=0, N2=0, MT=0
        end_line = format_endf_data_line(
            [0.0, 0.0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
