from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
import numpy as np
from scipy import special  # retained in case you need elsewhere

from .base import MF4MT
from ....endf.utils import (
    get_interpolation_scheme_name, project_tabulated_to_legendre,
    interpolate_1d_endf, auto_trim_legendre_tail, pick_mixed_branch,
)

@dataclass
class MF4MTMixed(MF4MT):
    """
    MT section in MF4 with mixed representation (LTT=3).
    
    Uses Legendre expansions for low energies and tabulated data for high energies.
    """
    _ltt: int = 3
    _ne: int = None  # Number of energies
    _nm: int = None  # Maximum order of Legendre polynomials
    _nr: int = None  # Number of different interpolation intervals for Legendre part
    _ne1: int = None  # Number of energy points for Legendre coefficients
    _ne2: int = None  # Number of energy points for tabulated distributions
    _nr_tab: int = None  # Number of interpolation intervals for tabulated data energies
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs for Legendre (NBT, INT)
    _tab_interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs for tabulated energy grid
    
    # Store energy grid and coefficients separately
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _legendre_coeffs: List[List[float]] = field(default_factory=list)  # Legendre coefficients for each energy
    
    # Store tabulated data
    _tabulated_energies: List[float] = field(default_factory=list)  # Energies for tabulated data
    _tabulated_cosines: List[List[float]] = field(default_factory=list)  # Cosines for each energy
    _tabulated_probabilities: List[List[float]] = field(default_factory=list)  # Probabilities for each energy
    _angular_interpolation: List[List[Tuple[int, int]]] = field(default_factory=list)  # Interpolation schemes for angular data

    # ESSENTIAL PROPERTIES - Energy Grids
    @property
    def legendre_energies(self) -> List[float]:
        """Energy grid for Legendre coefficients"""
        return self._energies

    @property
    def tabulated_energies(self) -> List[float]:
        """Energy grid for tabulated angular distributions"""
        return self._tabulated_energies

    # ESSENTIAL PROPERTIES - Data Access
    @property
    def legendre_coefficients(self) -> List[List[float]]:
        """
        Legendre coefficients for each energy point.
        
        Returns a list of coefficient lists, aligned with the legendre_energies.
        Each inner list contains the coefficients for one energy point.
        """
        return self._legendre_coeffs
    
    @property
    def tabulated_cosines(self) -> List[List[float]]:
        """
        Cosine values (μ) for each tabulated energy point.
        
        Returns a list of cosine lists, aligned with tabulated_energies.
        """
        return self._tabulated_cosines
    
    @property
    def tabulated_probabilities(self) -> List[List[float]]:
        """
        Probability values f(μ,E) for each tabulated energy point.
        
        Returns a list of probability lists, aligned with tabulated_energies and tabulated_cosines.
        """
        return self._tabulated_probabilities

    # ESSENTIAL PROPERTIES - Interpolation Schemes
    @property
    def legendre_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for Legendre energy grid (NBT, INT)"""
        return self._interpolation
    
    @property
    def tabulated_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for tabulated energy grid (NBT, INT)"""
        return self._tab_interpolation
    
    @property
    def angular_interpolation(self) -> List[List[Tuple[int, int]]]:
        """
        Interpolation scheme pairs for angular data at each tabulated energy.
        
        Returns a list aligned with tabulated_energies, where each element
        contains the (NBT, INT) pairs for that energy's angular distribution.
        """
        return self._angular_interpolation

    # CONVENIENCE METHODS - Data Access
    def get_legendre_coefficients(self, energy: float) -> List[float]:
        """Get Legendre coefficients at a specific energy."""
        try:
            index = self._energies.index(energy)
            return self._legendre_coeffs[index]
        except (ValueError, IndexError):
            return []
    
    def get_tabulated_distribution(self, energy: float) -> Tuple[List[float], List[float]]:
        """Get tabulated angular distribution at a specific energy."""
        try:
            index = self._tabulated_energies.index(energy)
            return (self._tabulated_cosines[index], self._tabulated_probabilities[index])
        except (ValueError, IndexError):
            return ([], [])
    
    def to_plot_data(
        self,
        order: int,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for Legendre coefficient data from mixed representation.
        
        This method extracts data from the Legendre portion of the mixed representation
        (LTT=3). The tabulated portion is not used for this plot type.
        
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
        >>> # Extract Legendre coefficients from mixed representation
        >>> data = mf4_mixed.to_plot_data(order=1, color='blue')
        >>> builder = PlotBuilder().add_data(data).build()
        
        Notes
        -----
        Mixed representation (LTT=3) contains both Legendre coefficients at low energies
        and tabulated distributions at high energies. This method uses only the Legendre
        portion. The energy range plotted will correspond to the Legendre energy grid.
        """
        from kika.plotting import LegendreCoeffPlotData
        
        # Get energy grid from Legendre portion of mixed representation
        energies = np.array(self._energies, dtype=float)
        
        if len(energies) == 0:
            raise ValueError("No Legendre data available in mixed representation")
        
        # Extract Legendre coefficients at all energy points using the built-in method
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=order,
            trim=False,  # Don't trim for plotting
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
            label = f'Mixed L={order}'
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

    def to_bulk_plot_data(
        self,
        max_order: int = 12,
        quad_order: int = 64
    ) -> Dict[str, Union[List[float], Dict[int, List[float]], int, str]]:
        """
        Extract ALL Legendre orders at once for bulk loading.

        For mixed representation (LTT=3), this uses the Legendre portion's energy grid
        and extracts coefficients using the built-in method that handles both branches.

        Parameters
        ----------
        max_order : int, optional
            Maximum Legendre order to compute (default: 12)
        quad_order : int, optional
            Quadrature order for projecting tabulated portions (default: 64)

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
        # Use the Legendre energy grid as the primary grid
        energies = np.array(self._energies, dtype=float) if self._energies else np.array([], dtype=float)

        if len(energies) == 0:
            # Fall back to tabulated energies if no Legendre energies
            energies = np.array(self._tabulated_energies, dtype=float) if self._tabulated_energies else np.array([], dtype=float)

        if len(energies) == 0:
            return {
                'energies': [],
                'coefficients_by_order': {},
                'max_order': 0,
                'isotope': getattr(self, 'isotope', None),
                'mt': getattr(self, 'number', None),
            }

        # Extract all Legendre coefficients using the built-in method
        # This handles both Legendre and tabulated portions automatically
        coeffs_dict = self.extract_legendre_coefficients(
            energy=energies,
            max_legendre_order=max_order,
            trim=False,  # Don't trim - we want all orders
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

    # --- CORE: extract coefficients with auto-trim and ENDF interpolation ---

    def extract_legendre_coefficients(
        self,
        energy: Union[float, np.ndarray],
        max_legendre_order: int = 10,
        *,
        trim: bool = True,
        trim_tol: float = 1e-6,
        quad_order: int = 64,
        out_of_range: str = "zero"
    ) -> Dict[int, Union[float, np.ndarray]]:
        """
        Return a_ℓ(E) for ℓ=0..L (L=max_legendre_order initially, then auto-trim optional).

        LTT=3 rules:
          - use the Legendre branch (LTT=1) below/at its max energy,
          - use the tabulated branch (LTT=2) above/at its min energy,
          - if there is a gap, pick the nearest boundary.

        Energy interpolation on each branch respects the declared ENDF (NBT,INT) pairs.

        Parameters
        ----------
        energy : float or array
            Energy point(s) where to evaluate a_ℓ(E)
        max_legendre_order : int
            Initial maximum order to compute before optional auto-trim
        trim : bool
            If True, auto-trim trailing orders by the tail sum rule ∑_{ℓ>L} |a_ℓ| < trim_tol
        trim_tol : float
            Tolerance for auto-trim
        quad_order : int
            Quadrature order for projecting tabulated f(μ|E) to Legendre
        out_of_range : str
            Behavior outside energy grid: 'zero' or 'hold'
            
        Returns
        -------
        Dict[int, Union[float, np.ndarray]]
            Dictionary mapping Legendre order ℓ to coefficient values a_ℓ(E)
        """
        # grids
        E_leg = np.asarray(self._energies, dtype=float) if self._energies else np.array([], dtype=float)
        E_tab = np.asarray(self._tabulated_energies, dtype=float) if self._tabulated_energies else np.array([], dtype=float)

        # handle scalar vs array
        scalar = np.isscalar(energy)
        E_query = np.array([energy], dtype=float) if scalar else np.asarray(energy, dtype=float)
        nE = E_query.size

        # Precompute padded coefficient arrays on each grid up to max_legendre_order
        A_leg = None  # shape (n_leg, L+1), with a0 in column 0
        if E_leg.size > 0:
            n_leg = len(self._legendre_coeffs)
            Lmax = max_legendre_order
            pad = np.zeros((n_leg, Lmax + 1), dtype=float)
            for i, coeffs in enumerate(self._legendre_coeffs):
                # coeffs from file typically contain a1..aNL (a0 implicit)
                pad[i, 0] = 1.0  # a0
                max_from_file = min(len(coeffs), Lmax)
                if max_from_file > 0:
                    pad[i, 1:max_from_file + 1] = np.asarray(coeffs[:max_from_file], dtype=float)
            A_leg = pad  # (n_leg, L+1)

        A_tab = None  # shape (n_tab, L+1)
        if E_tab.size > 0:
            n_tab = len(self._tabulated_energies)
            Lmax = max_legendre_order
            pad = np.zeros((n_tab, Lmax + 1), dtype=float)
            for i in range(n_tab):
                mu_i = self._tabulated_cosines[i] if i < len(self._tabulated_cosines) else []
                f_i = self._tabulated_probabilities[i] if i < len(self._tabulated_probabilities) else []
                ang_interp_i = self._angular_interpolation[i] if i < len(self._angular_interpolation) and self._angular_interpolation[i] else [(len(mu_i), 2)]
                pad[i, :] = project_tabulated_to_legendre(
                    mu=np.asarray(mu_i, dtype=float),
                    fmu=np.asarray(f_i, dtype=float),
                    max_order=Lmax,
                    ang_nbt_int=ang_interp_i,
                    quad_order=quad_order,
                )
            A_tab = pad

        # Classify all query energies by branch at once
        branches = np.array([pick_mixed_branch(float(E), E_leg, E_tab) for E in E_query])
        leg_mask = (branches == "leg") & (A_leg is not None) & (E_leg.size >= 1)
        tab_mask = (branches == "tab") & (A_tab is not None) & (E_tab.size >= 1)
        none_mask = ~(leg_mask | tab_mask)

        # Build result as dict of l → array(E_query)
        result: Dict[int, np.ndarray] = {l: np.zeros(nE, dtype=float) for l in range(max_legendre_order + 1)}

        # Isotropic fallback for points with no data
        if none_mask.any():
            result[0][none_mask] = 1.0

        # Vectorized interpolation per Legendre order (not per energy point)
        leg_interp = self._interpolation or ([(len(E_leg), 2)] if E_leg.size > 0 else [])
        tab_interp = self._tab_interpolation or ([(len(E_tab), 2)] if E_tab.size > 0 else [])

        for l in range(max_legendre_order + 1):
            if leg_mask.any() and A_leg is not None:
                result[l][leg_mask] = interpolate_1d_endf(
                    E_leg, A_leg[:, l], leg_interp, E_query[leg_mask], out_of_range=out_of_range
                )
            if tab_mask.any() and A_tab is not None:
                result[l][tab_mask] = interpolate_1d_endf(
                    E_tab, A_tab[:, l], tab_interp, E_query[tab_mask], out_of_range=out_of_range
                )

        # Cast to expected return types
        typed: Dict[int, Union[float, np.ndarray]] = {}
        for l, arr in result.items():
            typed[l] = float(arr[0]) if scalar else arr

        # Optional auto-trim
        if trim:
            # Allow trimming of trailing coefficients while always keeping at least a0
            typed = auto_trim_legendre_tail(typed, tol=trim_tol, min_order=0)

        return typed
    

    def get_interpolation_summary(self) -> str:
        """
        Get a comprehensive summary of all interpolation schemes used in this mixed representation.
        
        Returns:
            str: A formatted summary showing energy ranges and their interpolation schemes
        """
        summary_lines = []
        summary_lines.append("Mixed Representation (LTT=3) Interpolation Summary:")
        summary_lines.append("=" * 60)
        
        # Legendre coefficients section
        if self._energies and self._interpolation:
            summary_lines.append("\nLegendre Coefficients Section:")
            summary_lines.append("-" * 35)
            
            energy_start = 0
            for i, (nbt, int_code) in enumerate(self._interpolation):
                energy_end_idx = min(nbt - 1, len(self._energies) - 1)
                if energy_end_idx >= 0:
                    start_energy = self._energies[energy_start] if energy_start < len(self._energies) else 0.0
                    end_energy = self._energies[energy_end_idx]
                    interp_name = get_interpolation_scheme_name(int_code)
                    
                    summary_lines.append(
                        f"  Energy range: {start_energy:.3e} to {end_energy:.3e} eV"
                    )
                    summary_lines.append(
                        f"  Interpolation: {interp_name}"
                    )
                    summary_lines.append("")
                    
                    energy_start = nbt
        
        # Tabulated distributions section
        if self._tabulated_energies and self._tab_interpolation:
            summary_lines.append("Tabulated Distributions Section:")
            summary_lines.append("-" * 35)
            
            energy_start = 0
            for i, (nbt, int_code) in enumerate(self._tab_interpolation):
                energy_end_idx = min(nbt - 1, len(self._tabulated_energies) - 1)
                if energy_end_idx >= 0:
                    start_energy = self._tabulated_energies[energy_start] if energy_start < len(self._tabulated_energies) else 0.0
                    end_energy = self._tabulated_energies[energy_end_idx]
                    interp_name = get_interpolation_scheme_name(int_code)
                    
                    summary_lines.append(
                        f"  Energy range: {start_energy:.3e} to {end_energy:.3e} eV"
                    )
                    summary_lines.append(
                        f"  Interpolation: {interp_name}"
                    )
                    summary_lines.append("")
                    
                    energy_start = nbt
        
        # Angular interpolation schemes (brief overview)
        if self._angular_interpolation:
            summary_lines.append("Angular Interpolation Schemes:")
            summary_lines.append("-" * 30)
            
            unique_schemes = set()
            for ang_interp in self._angular_interpolation:
                for nbt, int_code in ang_interp:
                    unique_schemes.add(int_code)
            
            if unique_schemes:
                summary_lines.append("  Angular distributions use:")
                for scheme_code in sorted(unique_schemes):
                    scheme_name = get_interpolation_scheme_name(scheme_code)
                    summary_lines.append(f"    - {scheme_name}")
            else:
                summary_lines.append("  No angular interpolation schemes defined")
        
        return "\n".join(summary_lines)

    def __str__(self) -> str:
        """
        Convert the MF4MTMixed object back to ENDF format string.
        
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
            [0.0, self._awr, self._li, self._lct, 0, self._nm],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with energy boundary and Legendre energy count
        # Use actual length instead of stored values for safety
        ne1_actual = len(self._energies)
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, ne1_actual],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Ensure interpolation scheme is set if empty
        interpolation_pairs = self._interpolation
        if not interpolation_pairs and ne1_actual > 0:
            interpolation_pairs = [(ne1_actual, 2)]  # Default to linear-linear
        
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
        
        # Format tabulated distributions section header
        if self._ne2:
            tab_header = format_endf_data_line(
                [0.0, 0.0, 0, 0, self._nr_tab or 0, self._ne2],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(tab_header)
            line_num += 1
            
            # Format energy interpolation scheme pairs for tabulated data
            if self._tab_interpolation and self._nr_tab and self._nr_tab > 0:
                # Process interpolation pairs in groups of 3 (6 values per line)
                remaining_pairs = self._tab_interpolation.copy()
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
            for i, energy in enumerate(self._tabulated_energies):
                cosines = self._tabulated_cosines[i]
                probabilities = self._tabulated_probabilities[i]
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
