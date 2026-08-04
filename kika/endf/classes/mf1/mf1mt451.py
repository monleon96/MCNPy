from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

@dataclass
class MF1MT451:
    """
    Data class for MF1MT451 (General Information) section.
    """
    # Make MF1MT451 follow the same interface as MT
    number: int = 451
    
    # Line count
    num_lines: int = 0  # Number of lines in MF1MT451 section
    
    # Line position tracking
    _start_line: int = None  # Line number in original file where section starts
    _end_line: int = None    # Line number in original file where section ends
    
    # First line values
    _za: float = None  # (Z,A) designation
    _awr: float = None  # Atomic weight ratio
    _lrp: int = None  # Resonance parameter flag
    _lfi: int = None  # Fission flag
    _nlib: int = None  # Library identifier
    _nmod: int = None  # Modification number
    
    # Second line values
    _elis: float = None  # Excitation energy of the target nucleus
    _sta: int = None  # Target stability flag
    _lis: int = None  # State number of the target nucleus
    _liso: int = None  # Isomeric state number
    _nfor: int = None  # Library format
    
    # Third line values
    _awi: float = None  # Mass of the projectile in neutron mass units
    _emax: float = None  # Upper limit of the energy range for evaluation
    _lrel: int = None  # Library release number
    _nsub: int = None  # Sub-library number
    _nver: int = None  # Library version number
    
    # Fourth line values
    _temp: float = None  # Target temperature (Kelvin) for Doppler broadening
    _ldrv: int = None  # Special derived material flag
    _nwd: int = None  # Number of records with descriptive text
    _nxc: int = None  # Number of records in directory
    
    # Text description fields (fifth line onwards)
    _zsymam: str = None  # Material identifier (Z-cc-AM format)
    _alab: str = None  # Laboratory identifier
    _edate: str = None  # Evaluation date
    _auth: str = None  # Author(s)
    
    # Reference information (sixth line)
    _ref: str = None  # Primary reference
    _ddate: str = None  # Distribution date
    _rdate: str = None  # Revision date
    _endate: str = None  # Master file entry date
    
    # Raw text lines storage
    _text_lines: List[str] = field(default_factory=list)
    
    # Identification fields for ENDF format
    _mat: int = None
    _mt: int = 451
    _mf: int = 1
    
    # Additional data can be added as needed
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    # Directory entries for MF/MT sections
    _directory: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (MF, MT, NC, MOD)
    
    # First line properties
    @property
    def zaid(self) -> float:
        """Z,A designation (charge, mass)"""
        return self._za
    
    @property
    def atomic_weight_ratio(self) -> float:
        """Ratio of nucleus mass to neutron mass"""
        return self._awr
    
    @property
    def has_resonance_params(self) -> bool:
        """Whether resonance parameters are given in File 2"""
        return self._lrp > 0
    
    @property
    def can_fission(self) -> bool:
        """Whether this material can fission"""
        return self._lfi == 1
    
    @property
    def library_id(self) -> int:
        """Library identifier"""
        return self._nlib
    
    @property
    def mod_number(self) -> int:
        """Modification number for this material"""
        return self._nmod
    
    # Second line properties
    @property
    def excitation_energy(self) -> float:
        """Excitation energy of the target nucleus"""
        return self._elis
    
    @property
    def is_stable(self) -> bool:
        """Whether the target nucleus is stable"""
        return self._sta == 0
    
    @property
    def state_number(self) -> int:
        """State number of the target nucleus"""
        return self._lis
    
    @property
    def isomer_number(self) -> int:
        """Isomeric state number"""
        return self._liso
    
    @property
    def format_version(self) -> int:
        """Library format version"""
        return self._nfor
    
    # Third line properties
    @property
    def projectile_mass(self) -> float:
        """Mass of the projectile in neutron mass units"""
        return self._awi
    
    @property
    def energy_max(self) -> float:
        """Upper limit of energy range for evaluation"""
        return self._emax
    
    @property
    def library_release(self) -> int:
        """Library release number"""
        return self._lrel
    
    @property
    def sublibrary(self) -> int:
        """Sub-library number"""
        return self._nsub
    
    @property
    def library_version(self) -> int:
        """Library version number"""
        return self._nver
    
    # Fourth line properties
    @property
    def temperature(self) -> float:
        """Target temperature (Kelvin)"""
        return self._temp
    
    @property
    def is_derived(self) -> bool:
        """Whether this is a derived material (True) or primary evaluation (False)"""
        return self._ldrv > 0
    
    @property
    def text_records(self) -> int:
        """Number of records with descriptive text"""
        return self._nwd
    
    @property
    def directory_records(self) -> int:
        """Number of records in directory"""
        return self._nxc
    
    # Text fields properties
    @property
    def material_id(self) -> str:
        """Material identifier (Z-cc-AM format)"""
        return self._zsymam
    
    @property
    def laboratory(self) -> str:
        """Laboratory identifier"""
        return self._alab
    
    @property
    def eval_date(self) -> str:
        """Evaluation date"""
        return self._edate
    
    @property
    def authors(self) -> str:
        """Author(s)"""
        return self._auth
    
    @property
    def reference(self) -> str:
        """Primary reference"""
        return self._ref
    
    @property
    def dist_date(self) -> str:
        """Distribution date"""
        return self._ddate
    
    @property
    def revision_date(self) -> str:
        """Revision date"""
        return self._rdate
    
    @property
    def entry_date(self) -> str:
        """Master file entry date"""
        return self._endate
    
    @property
    def directory(self) -> pd.DataFrame:
        """
        Directory entries as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: MF, MT, lines
            - MF: File number
            - MT: Section number 
            - lines: Number of records in the section
        """
        if not self._directory:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['MF', 'MT', 'lines'])
            
        # Convert the list of tuples to DataFrame with selected columns
        df = pd.DataFrame(self._directory, columns=['MF', 'MT', 'NC', 'MOD'])
        
        # Rename NC to lines and drop MOD column
        df = df.rename(columns={'NC': 'lines'}).drop(columns=['MOD'])
        
        return df
    
    def add_directory_entry(self, mf: int, mt: int, nc: int, mod: int) -> None:
        """Add an entry to the directory"""
        self._directory.append((mf, mt, nc, mod))
    
    def __str__(self) -> str:
        """
        Convert the MF1MT451 object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import (
            format_endf_data_line,
            format_endf_send_record,
            ENDF_FORMAT_FLOAT,
            ENDF_FORMAT_INT,
            ENDF_FORMAT_INT_ZERO,
            ENDF_FORMAT_BLANK,
        )

        mat = self._mat if self._mat is not None else 0
        lines = []
        
        # Format first line - numeric data
        # ZA and AWR as floats, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, self._lrp, self._lfi, self._nlib, self._nmod],
            mat, 1, 451, 1,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        
        # Format second line - numeric data
        # ENDF-6 types record 2 as [ELIS, STA, LIS, LISO, 0, NFOR] with ELIS
        # *and STA* as floats. STA was rendered through the integer format, so
        # a stable nuclide came back as "          0" where the source has
        # " 0.000000+0" — one changed line on every tape kika rewrites.
        line2 = format_endf_data_line(
            [self._elis, self._sta, self._lis, self._liso, 0, self._nfor],
            mat, 1, 451, 2,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        
        # Format third line - numeric data
        # AWI and EMAX as floats, rest as integers with zeros printed
        line3 = format_endf_data_line(
            [self._awi, self._emax, self._lrel, 0, self._nsub, self._nver],
            mat, 1, 451, 3,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line3)
        
        # Format fourth line - numeric data
        # TEMP as float, rest as integers with zeros printed
        line4 = format_endf_data_line(
            [self._temp, 0.0, self._ldrv, 0, self._nwd, self._nxc],
            mat, 1, 451, 4,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line4)
        
        # If we have the original text lines and they haven't been modified,
        # we can use them directly
        if self._text_lines and len(self._text_lines) >= self._nwd + 4:
            for i, line in enumerate(self._text_lines[4:4+self._nwd], start=5):
                # Ensure line has correct MAT, MF, MT and sequence number
                if len(line) >= 66:
                    data_part = line[:66]
                    id_part = f"{mat:4d} 1451{i:5d}"
                    lines.append(data_part + id_part)
                else:
                    # If line is too short, pad it
                    padded_line = line.ljust(66) + f"{mat:4d} 1451{i:5d}"
                    lines.append(padded_line)
        else:
            # Otherwise, reconstruct text lines based on parsed values
            
            # Format fifth line - material description
            if self._zsymam or self._alab or self._edate or self._auth:
                line5 = ""
                line5 += self._zsymam.ljust(11) if self._zsymam else " " * 11
                line5 += self._alab.ljust(11) if self._alab else " " * 11
                line5 += self._edate.ljust(10) if self._edate else " " * 10
                line5 += " "  # Space between fields
                line5 += self._auth.ljust(33) if self._auth else " " * 33
                line5 = line5.ljust(66) + f"{mat:4d} 1451 5"
                lines.append(line5)
            
            # Format sixth line - reference information
            if self._ref or self._ddate or self._rdate or self._endate:
                line6 = " "  # First column is blank
                line6 += self._ref.ljust(21) if self._ref else " " * 21
                line6 += self._ddate.ljust(10) if self._ddate else " " * 10
                line6 += " "  # Space between fields
                line6 += self._rdate.ljust(10) if self._rdate else " " * 10
                line6 += " " * 12  # Space before entry date
                line6 += self._endate.ljust(8) if self._endate else " " * 8
                line6 = line6.ljust(66) + f"{mat:4d} 1451 6"
                lines.append(line6)
            
            # Add placeholder text lines if needed to match NWD
            if self._nwd and self._nwd > 2:  # 2 text lines already added
                for i in range(7, 5 + self._nwd):
                    placeholder = f"{'Text line ' + str(i-4):<66}{mat:4d} 1451{i:5d}"
                    lines.append(placeholder)
        
        # Directory entries - all numbers as integers with zeros printed
        line_num = 5 + (self._nwd or 0)  # Start line number after text records
        
        for mf, mt, nc, mod in self._directory:
            # All values in directory entries are integers with zeros printed
            line = format_endf_data_line(
                [None, None, mf, mt, nc, mod],
                mat, 1, 451, line_num,
                formats=[ENDF_FORMAT_BLANK, ENDF_FORMAT_BLANK, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
            )
            lines.append(line)
            line_num += 1
        
        # SEND record. Two things were wrong here at once: the six fields were
        # blanked, where ENDF-6 wants " 0.000000+0 0.000000+0" then four
        # integer zeros, and the sequence number was the running line counter
        # where the standard requires the literal 99999.
        lines.append(format_endf_send_record(mat, 1))
        
        return "\n".join(lines)
