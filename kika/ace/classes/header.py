from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Header:
    """Class representing the header section of an ACE file.
    
    The header contains basic information about the nuclear data, including
    identification, temperature, and pointers to other sections.
    
    :ivar format_version: Format version ('legacy' or '2.0.1')
    :type format_version: str, optional
    :ivar zaid: ZA identifier (1000*Z + A)
    :type zaid: int, optional
    :ivar extension: File extension (like ".02c")
    :type extension: str, optional
    :ivar atomic_weight_ratio: Atomic weight ratio to neutron mass
    :type atomic_weight_ratio: float, optional
    :ivar temperature: Temperature in MeV
    :type temperature: float, optional
    :ivar date: Date of the evaluation
    :type date: str, optional
    :ivar comment: Description or comment
    :type comment: str, optional
    :ivar matid: Material identifier
    :type matid: int, optional
    :ivar ace_version: ACE version format string (e.g., "2.0")
    :type ace_version: str, optional
    :ivar source: Evaluation source
    :type source: str, optional
    :ivar comment_line_count: Number of comment lines
    :type comment_line_count: int, optional
    :ivar izaw_array: IZAW array (16 pairs of ZA and AWR)
    :type izaw_array: List[Tuple[int, float]], optional
    :ivar nxs_array: NXS array (16 integers)
    :type nxs_array: List[int], optional
    :ivar jxs_array: JXS array (32 integers)
    :type jxs_array: List[int], optional
    """
    # Header format identification
    format_version: Optional[str] = None  # 'legacy' or '2.0.1'
    
    # Common header components
    zaid: Optional[int] = None  # ZA identifier (atomic number)
    extension: Optional[str] = None  # File extension (e.g., ".02c")
    atomic_weight_ratio: Optional[float] = None  # Atomic weight ratio
    # kT in MeV, NOT kelvin -- the class docstring above has always said MeV and
    # this comment said K. Fe-56 at room temperature reads 2.53e-08 here, which
    # is 293.6 K once divided by Boltzmann's constant, as CrossSection.from_ace
    # and kika.ace.model_adapter both do.
    temperature: Optional[float] = None
    date: Optional[str] = None  # Date of the evaluation
    comment: Optional[str] = None  # Description or comment
    matid: Optional[int] = None  # Material identifier
    
    # 2.0.1 format specific fields
    ace_version: Optional[str] = None  # Version format string (e.g., "2.0")
    source: Optional[str] = None  # Evaluation source
    comment_line_count: Optional[int] = None  # Number of comment lines
    
    # Arrays from header
    izaw_array: Optional[List[Tuple[int, float]]] = None  # IZAW array (16 pairs of ZA and AWR)
    nxs_array: Optional[List[int]] = None  # NXS array (16 integers)
    jxs_array: Optional[List[int]] = None  # JXS array (32 integers)
    
    # NXS array properties - useful for users
    @property
    def xss_block_length(self) -> Optional[int]:
        """Length of second block of data (XSS array). NXS(1).
        
        :returns: Length of the XSS array or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[1] if self.nxs_array and len(self.nxs_array) > 1 else None
    
    @property
    def num_energies(self) -> Optional[int]:
        """Number of energies. NXS(3).
        
        :returns: Number of energy points or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[3] if self.nxs_array and len(self.nxs_array) > 3 else None
    
    @property
    def num_reactions(self) -> Optional[int]:
        """Number of reactions including elastic scattering. NXS(4).
        
        :returns: Number of reactions or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[4] if self.nxs_array and len(self.nxs_array) > 4 else None
    
    @property
    def num_secondary_neutron_reactions(self) -> Optional[int]:
        """Number of reactions having secondary neutrons excluding elastic scattering. NXS(5).
        
        :returns: Number of secondary neutron reactions or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[5] if self.nxs_array and len(self.nxs_array) > 5 else None
    
    @property
    def num_photon_production_reactions(self) -> Optional[int]:
        """Number of photon production reactions. NXS(6).
        
        :returns: Number of photon production reactions or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[6] if self.nxs_array and len(self.nxs_array) > 6 else None
    
    @property
    def num_particle_types(self) -> Optional[int]:
        """Number of particle types for which production data is given. NXS(7).
        
        :returns: Number of particle types or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[7] if self.nxs_array and len(self.nxs_array) > 7 else None
    
    @property
    def num_delayed_neutron_precursors(self) -> Optional[int]:
        """Number of delayed neutron precursor families. NXS(8).
        
        :returns: Number of delayed neutron precursor families or None if not available
        :rtype: int, optional
        """
        return self.nxs_array[8] if self.nxs_array and len(self.nxs_array) > 8 else None
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the Header object.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :returns: Formatted string representation of the Header
        :rtype: str
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'ACE Header Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create a summary table of header information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Header Properties:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add header properties to the table
        if self.format_version:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Format Version", self.format_version, width1=property_col_width, width2=value_col_width)
        if self.zaid is not None:
            zaid_str = f"{self.zaid}"
            if self.extension:
                zaid_str += f" ({self.extension})"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "ZA Identifier", zaid_str, width1=property_col_width, width2=value_col_width)
        if self.atomic_weight_ratio is not None:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Atomic Weight Ratio", self.atomic_weight_ratio, width1=property_col_width, width2=value_col_width)
        if self.temperature is not None:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Temperature", f"{self.temperature} MeV", width1=property_col_width, width2=value_col_width)
        if self.date:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Date", self.date, width1=property_col_width, width2=value_col_width)
        if self.matid:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Material ID", self.matid, width1=property_col_width, width2=value_col_width)
        if self.comment:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Comment", self.comment, width1=property_col_width, width2=value_col_width)
        
        # 2.0.1 format specific fields
        if self.format_version == '2.0.1':
            if self.ace_version:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "ACE Version", self.ace_version, width1=property_col_width, width2=value_col_width)
            if self.source:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Source", self.source, width1=property_col_width, width2=value_col_width)
            if self.comment_line_count is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Comment Line Count", self.comment_line_count, width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Nuclear data properties section - more user-friendly naming
        if self.nxs_array and len(self.nxs_array) > 0:
            info_table += "Basic Information:\n"
            info_table += "-" * header_width + "\n"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Property", "Value", width1=property_col_width, width2=value_col_width)
            info_table += "-" * header_width + "\n"
            
            if self.num_energies is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Number of Energy Points", self.num_energies, width1=property_col_width, width2=value_col_width)
            if self.num_reactions is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Number of Reactions", self.num_reactions, width1=property_col_width, width2=value_col_width)
            if self.num_secondary_neutron_reactions is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Secondary Neutron Reactions", self.num_secondary_neutron_reactions, width1=property_col_width, width2=value_col_width)
            if self.num_photon_production_reactions is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Photon Production Reactions", self.num_photon_production_reactions, width1=property_col_width, width2=value_col_width)
            if self.num_particle_types is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Particle Types", self.num_particle_types, width1=property_col_width, width2=value_col_width)
            if self.num_delayed_neutron_precursors is not None:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Delayed Neutron Precursors", self.num_delayed_neutron_precursors, width1=property_col_width, width2=value_col_width)
            
            info_table += "-" * header_width + "\n"
        
        return header + info_table