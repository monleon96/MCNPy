"""
MF file for ENDF files.

MF files contain related nuclear data sections grouped by MT numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, TypeVar, Tuple, List, Any, TYPE_CHECKING

from .mt import MT
from .mf1.mf1mt import MT451
from ...cov.legendre_covariance import LegendreCovariance

if TYPE_CHECKING:
    from .mf2.mf2mt151 import MF2MT151
    from .mf3.mf3mt import MF3MT

# Type for any MT section class (MT, MT451, etc.)
MTSection = TypeVar('MTSection', bound=MT)

@dataclass
class MF:
    """
    Data class representing an MF file in ENDF format.
    """
    number: int
    sections: Dict[int, Union[MT, MT451, 'MF2MT151', 'MF3MT']] = field(default_factory=dict)
    num_lines: int = 0  # Number of lines in this MF section
    
    def add_section(self, section: Union[MT, MT451, 'MF2MT151', 'MF3MT']) -> None:
        """
        Add an MT section to this MF file
        
        Args:
            section: The MT section object to add
        """
        self.sections[section.number] = section
    
    def get_section(self, mt_number: int) -> Optional[Union[MT, MT451, 'MF2MT151', 'MF3MT']]:
        """
        Get an MT section by number
        
        Args:
            mt_number: The MT section number to retrieve
            
        Returns:
            The MT section object or None if not found
        """
        return self.sections.get(mt_number)
    
    @property
    def mt(self) -> Dict[int, Union[MT, MT451, 'MF2MT151', 'MF3MT']]:
        """Direct access to MT sections dictionary"""
        return self.sections
    
    def to_plot_data(self, mt: int, **kwargs):
        """
        Create a PlotData object from the specified MT section.
        
        This is a convenience method that delegates to the MT section's to_plot_data method.
        For MF4 (angular distributions), this requires an 'order' parameter.
        
        Note: For uncertainty support at the MF level, use the ENDF.to_plot_data method
        which can combine MF4 (nominal) and MF34 (uncertainties) data.
        
        Parameters
        ----------
        mt : int
            MT section number to extract data from
        **kwargs
            Additional parameters passed to the underlying to_plot_data method.
            For MF4, this should include 'order' (Legendre polynomial order).
            May also include styling parameters (label, color, linestyle, etc.)
            
        Returns
        -------
        PlotData
            Plot data object ready to be added to a PlotBuilder
            
        Raises
        ------
        KeyError
            If the MT section doesn't exist
        AttributeError
            If the MT section doesn't support to_plot_data
            
        Examples
        --------
        >>> # For MF4 data (angular distributions)
        >>> mf4 = endf.mf[4]
        >>> data = mf4.to_plot_data(mt=2, order=1)
        >>> 
        >>> # With styling
        >>> data = mf4.to_plot_data(mt=2, order=1, label='Fe-56 Elastic', color='blue')
        """
        if mt not in self.sections:
            raise KeyError(f"MT section {mt} not found in MF{self.number}")
        
        mt_section = self.sections[mt]
        
        if not hasattr(mt_section, 'to_plot_data'):
            raise AttributeError(
                f"MT section {mt} in MF{self.number} does not support to_plot_data. "
                f"This feature is available for MF3 (cross sections) and MF4 (angular distributions)."
            )
        
        return mt_section.to_plot_data(**kwargs)
    
    def __repr__(self):
        return f"MF({self.number}, {len(self.sections)} sections)"
    
    def __getitem__(self, mt_number: int) -> Union[MT, MT451, 'MF2MT151', 'MF3MT']:
        """Allow accessing MT sections like: mf[451]"""
        if mt_number not in self.sections:
            raise KeyError(f"MT section {mt_number} not found in MF{self.number}")
        return self.sections[mt_number]
        
    def __str__(self) -> str:
        """
        Convert the MF file to an ENDF format string.
        
        Returns:
            A string containing all MT sections in ENDF format, sorted by MT number
        """
        # Import inside the method to avoid circular imports
        from ..utils import format_endf_data_line, ENDF_FORMAT_INT
        
        # Get all MT sections and sort them by MT number
        sorted_mts = sorted(self.sections.items(), key=lambda x: x[0])
        
        # Convert each MT section to a string and join them
        mt_strings = [str(mt) for _, mt in sorted_mts]
        
        # Join all MT sections with newlines
        result = "\n".join(mt_strings)
        
        # Add the required end-of-file marker - a blank data line with MAT and zeros for MF, MT
        # Use the MAT number from the last MT section if available, otherwise default to 0
        mat = 0
        if sorted_mts:
            _, last_mt = sorted_mts[-1]
            mat = getattr(last_mt, "_mat", 0) or 0
        
        # Format end-of-file marker
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, 0, 0, 0,  # MF=0, MT=0 for end of file marker
            formats=[ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        
        # Add end-of-file marker to the result
        result += "\n" + end_line
        
        return result
        
    def to_ang_covmat(self, energy_unit: str = 'eV', mf4_data=None) -> LegendreCovariance:
        """
        Convert MF34 data to an LegendreCovariance object that contains data from all MT sections.

        This method aggregates angular covariance data from all MT sections in this MF file
        (if it's MF34) and returns a combined LegendreCovariance object.

        Parameters
        ----------
        energy_unit : str, optional
            Energy unit for the energy grids: 'eV' (default) or 'MeV'
        mf4_data : optional
            MF4 file object forwarded to each ``MF34MT.to_ang_covmat()`` call
            so that Legendre coefficients can be extracted from MF4 data.

        Returns
        -------
        LegendreCovariance
            Combined object containing data from all MT sections.

        Raises
        ------
        ValueError
            If this method is called on an MF that is not MF34.
        """
        # Check if this is an MF34 file
        if self.number != 34:
            raise ValueError(f"The to_ang_covmat method is only available for MF34, not MF{self.number}")

        # Import here to avoid circular imports
        from ...cov.legendre_covariance import LegendreCovariance

        # Create a new LegendreCovariance object to store the combined data
        combined_ang_covmat = LegendreCovariance(energy_unit=energy_unit)

        # Loop through all MT sections and combine their data
        for mt_number, mt_section in self.sections.items():
            # Get the individual LegendreCovariance for this MT section
            try:
                # The MT section must be an MF34MT object with to_ang_covmat method
                mt_ang_covmat: LegendreCovariance = mt_section.to_ang_covmat(
                    energy_unit=energy_unit, mf4_data=mf4_data
                )
                
                # Add all matrices and their energy grids from this MT section
                for i in range(mt_ang_covmat.num_matrices):
                    combined_ang_covmat.add_matrix(
                        mt_ang_covmat.isotope_rows[i],
                        mt_ang_covmat.reaction_rows[i],
                        mt_ang_covmat.l_rows[i],
                        mt_ang_covmat.isotope_cols[i],
                        mt_ang_covmat.reaction_cols[i],
                        mt_ang_covmat.l_cols[i],
                        mt_ang_covmat.matrices[i],
                        mt_ang_covmat.energy_grids[i],
                        mt_ang_covmat.is_relative[i],
                        mt_ang_covmat.frame[i]
                    )
                # Propagate legendre_coefficients
                combined_ang_covmat.legendre_coefficients.update(
                    mt_ang_covmat.legendre_coefficients
                )
            except (AttributeError, ValueError) as e:
                # Catch potential errors if a section isn't a valid MF34MT or conversion fails
                print(f"Warning: Could not convert MT{mt_number} to LegendreCovariance: {e}")

        return combined_ang_covmat
