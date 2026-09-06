"""
ENDF Parser - Functions for parsing Evaluated Nuclear Data Files (ENDF)
"""
import os
from typing import Dict, List, Optional, Tuple
import warnings
import re

from ..classes.endf import ENDF
from ..classes.mf import MF
from ..utils import parse_endf_id
from .parse_mf1 import parse_mf1
from .parse_mf2 import parse_mf2
from .parse_mf3 import parse_mf3
from .parse_mf4 import parse_mf4
from .parse_mf5 import parse_mf5
from .parse_mf6 import parse_mf6
from .parse_mf7 import parse_mf7
from .parse_mf31 import parse_mf31
from .parse_mf32 import parse_mf32
from .parse_mf33 import parse_mf33
from .parse_mf34 import parse_mf34
from .parse_mf35 import parse_mf35
from ...utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)


# Dictionary mapping MF numbers to their parser functions
MF_PARSERS = {
    1: parse_mf1,
    2: parse_mf2,
    3: parse_mf3,
    4: parse_mf4,
    5: parse_mf5,
    6: parse_mf6,
    7: parse_mf7,
    31: parse_mf31,
    32: parse_mf32,
    33: parse_mf33,
    34: parse_mf34,
    35: parse_mf35,
    # Additional parsers will be registered here
}


def scan_mat_number(lines: List[str]) -> Optional[int]:
    """The MAT number of a tape, from the first data line that carries one.

    MAT lives in columns 67-70 of every ENDF record, so any data line will do;
    header and comment lines are skipped by requiring MF > 0 in columns 71-72.

    Extracted from :func:`parse_endf_file` so the targeted-parse path in
    :func:`kika.endf.read_endf.read_endf` can use it too. Without it a
    ``read_endf(f, mf_numbers=[1])`` produced an ENDF whose ``mat`` — and so
    whose ``zaid`` — was None, while a full parse of the same file gave 2631.
    Two parse paths, two answers, for a property of the file itself.
    """
    for line in lines:
        if len(line) < 72:
            continue
        try:
            mf_str = line[70:72].strip()
            if not mf_str or int(mf_str) <= 0:
                continue
            mat_str = line[66:70].strip()
            if not mat_str:
                continue
            mat_candidate = int(mat_str)
            if mat_candidate > 0:  # Valid MAT numbers are positive
                return mat_candidate
        except ValueError:
            continue
    return None


def parse_endf_file(filepath: str) -> ENDF:
    """
    Parse a complete ENDF file.
    
    Args:
        filepath: Path to the ENDF file
        
    Returns:
        ENDF object with parsed data for all MF sections that have registered parsers
    """
    logger.debug(f"Starting to parse ENDF file: {filepath}")
    endf = ENDF()
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        logger.debug(f"Read {len(lines)} lines from file")
        
        mat_number = scan_mat_number(lines)
        if mat_number is not None:
            endf.mat = mat_number
            logger.debug(f"Extracted MAT number: {mat_number}")
        else:
            logger.debug("Could not extract MAT number from file")

        # First, scan the file to identify available MF sections
        available_mf_numbers = _scan_available_mf(lines)
        logger.debug(f"Found MF sections: {available_mf_numbers}")
        
        if not available_mf_numbers:
            warnings.warn(f"No MF sections found in file: {filepath}")
            return endf
        
        # Filter to MF sections that have parsers
        parseable_mfs = [mf for mf in available_mf_numbers if mf in MF_PARSERS]
        skipped_mfs = [mf for mf in available_mf_numbers if mf not in MF_PARSERS]
        
        if not parseable_mfs:
            warnings.warn(f"Found MF sections {available_mf_numbers}, but none have parsers")
            return endf
            
        if skipped_mfs:
            logger.debug(f"Skipping MF sections without parsers: {skipped_mfs}")
            warnings.warn(f"Skipping MF sections without parsers: {skipped_mfs}. Only parsing: {parseable_mfs}")
        
        logger.debug(f"Parsing MF sections: {parseable_mfs}")
        
        # Group lines by MF and MT with position tracking
        mf_mt_groups, line_counts = _group_lines_by_mf_mt_with_positions(lines)
        
        # Parse each MF section that has a registered parser
        for mf_number in parseable_mfs:
            if mf_number in mf_mt_groups:
                logger.debug(f"Parsing MF{mf_number}")
                mt_groups = mf_mt_groups[mf_number]
                
                # Extract just the lines for this MF
                mf_lines = []
                
                # Get all lines for this MF
                for mt_number, mt_lines in mt_groups.items():
                    mf_lines.extend(mt_lines)
                
                # Parse the MF section with line count information
                if mf_lines:
                    logger.debug(f"Parsing MF{mf_number} with {len(mf_lines)} lines")
                    mf = MF_PARSERS[mf_number](mf_lines)
                    if mf_number in line_counts:
                        mf.num_lines = line_counts[mf_number]
                    endf.add_file(mf)
                    logger.debug(f"Successfully parsed MF{mf_number}")
    
    logger.debug(f"Finished parsing ENDF file: {filepath}")
    return endf


def _scan_available_mf(lines: List[str]) -> List[int]:
    """
    Scan ENDF file to identify which MF sections are present.
    
    Args:
        lines: List of all lines from an ENDF file
        
    Returns:
        List of MF section numbers found
    """
    mf_numbers = set()
    
    for line in lines:
        if len(line) >= 72:
            try:
                mf = int(line[70:72])
                if mf > 0:  # Skip end markers (MF=0)
                    mf_numbers.add(mf)
            except ValueError:
                continue
    
    return sorted(list(mf_numbers))


def parse_mf_from_file(filepath: str, mf_number: int) -> Optional[MF]:
    """
    Parse only a specific MF section from an ENDF file.
    
    Args:
        filepath: Path to the ENDF file
        mf_number: MF number to parse
        
    Returns:
        MF object if found and parsed, None otherwise
    """
    logger.debug(f"Parsing MF{mf_number} from file: {filepath}")
    
    if mf_number not in MF_PARSERS:
        raise ValueError(f"No parser available for MF{mf_number}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENDF file not found at {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            logger.debug(f"Read {len(lines)} lines from file")
            
            # First check if the MF is actually in the file
            available_mfs = _scan_available_mf(lines)
            logger.debug(f"Available MF sections in file: {available_mfs}")
            
            if mf_number not in available_mfs:
                logger.debug(f"MF{mf_number} not found in file")
                warnings.warn(f"MF{mf_number} not found in file: {filepath}")
                return None
                
            # Find all MF sections matching the specified MF number
            mf_lines = []
            for line in lines:
                if len(line) >= 72:
                    try:
                        current_mf = int(line[70:72])
                        if current_mf == mf_number:
                            mf_lines.append(line)
                    except ValueError:
                        pass
            
            # Parse the MF section if found
            if mf_lines:
                logger.debug(f"Found {len(mf_lines)} lines for MF{mf_number}, parsing...")
                result = MF_PARSERS[mf_number](mf_lines)
                logger.debug(f"Successfully parsed MF{mf_number}")
                return result
            
            # Fallback to the group_by method
            logger.debug(f"Direct line matching failed, trying grouping method")
            mf_mt_groups = _group_lines_by_mf_mt(lines)
            
            # Check if the requested MF is in the file
            if mf_number in mf_mt_groups:
                # Extract just the lines for this MF
                mf_lines = []
                for mt_lines in mf_mt_groups[mf_number].values():
                    mf_lines.extend(mt_lines)
                
                # Parse the MF section
                if mf_lines:
                    logger.debug(f"Found {len(mf_lines)} lines for MF{mf_number} via grouping, parsing...")
                    result = MF_PARSERS[mf_number](mf_lines)
                    logger.debug(f"Successfully parsed MF{mf_number}")
                    return result
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise RuntimeError(f"Error parsing file: {e}")
    
    logger.debug(f"MF{mf_number} not found or could not be parsed")
    return None


def _group_lines_by_mf_mt(lines: List[str]) -> Dict[int, Dict[int, List[str]]]:
    """
    Group ENDF file lines by MF and MT numbers.
    
    Args:
        lines: List of all lines from an ENDF file
        
    Returns:
        Nested dictionary: {mf_number: {mt_number: [lines]}}
    """
    result: Dict[int, Dict[int, List[str]]] = {}
    current_mf = None
    current_mt = None
    current_lines: List[str] = []
    
    for line in lines:
        # Parse identification fields
        mat, mf, mt = parse_endf_id(line)
        
        # Handle end markers or section changes
        if mf == 0 and mt == 0:
            # End of material or file - save current section if any
            if current_mf is not None and current_mt is not None and current_lines:
                if current_mf not in result:
                    result[current_mf] = {}
                if current_mt not in result[current_mf]:
                    result[current_mf][current_mt] = []
                result[current_mf][current_mt].extend(current_lines)
                current_lines = []
            continue  # Don't break, just continue to next line
        
        if mf == 0 or mt == 0:
            # Skip end-of-section markers but don't break
            continue
            
        if current_mf is None or current_mt is None:
            # First section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        elif current_mf != mf or current_mt != mt:
            # Section change
            if current_mf not in result:
                result[current_mf] = {}
            if current_mt not in result[current_mf]:
                result[current_mf][current_mt] = []
            result[current_mf][current_mt].extend(current_lines)
            
            # Start new section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        else:
            # Continue current section
            current_lines.append(line)
    
    # Add the last section if needed
    if current_mf is not None and current_mt is not None and current_lines:
        if current_mf not in result:
            result[current_mf] = {}
        if current_mt not in result[current_mf]:
            result[current_mf][current_mt] = []
        result[current_mf][current_mt].extend(current_lines)
    
    return result


def _group_lines_by_mf_mt_with_positions(lines: List[str]) -> Tuple[Dict[int, Dict[int, List[str]]], Dict[int, int]]:
    """
    Group ENDF file lines by MF and MT numbers and track line counts.
    
    Args:
        lines: List of all lines from an ENDF file
        
    Returns:
        Tuple of:
            - Nested dictionary: {mf_number: {mt_number: [lines]}}
            - Dictionary mapping MF numbers to line counts
    """
    result: Dict[int, Dict[int, List[str]]] = {}
    line_counts: Dict[int, int] = {}
    current_mf = None
    current_mt = None
    current_lines: List[str] = []
    
    for i, line in enumerate(lines):
        # Parse identification fields
        mat, mf, mt = parse_endf_id(line)
        
        # Handle end markers or section changes
        if mf == 0 and mt == 0:
            # End of material or file - save current section if any
            if current_mf is not None and current_mt is not None and current_lines:
                if current_mf not in result:
                    result[current_mf] = {}
                if current_mt not in result[current_mf]:
                    result[current_mf][current_mt] = []
                result[current_mf][current_mt].extend(current_lines)
                if current_mf not in line_counts:
                    line_counts[current_mf] = len(current_lines)
                else:
                    line_counts[current_mf] += len(current_lines)
                current_lines = []
            continue  # Don't break, just continue to next line
        
        if mf == 0 or mt == 0:
            # Skip other end markers without breaking
            continue
            
        if current_mf is None or current_mt is None:
            # First section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        elif current_mf != mf or current_mt != mt:
            # Section change
            if current_mf not in result:
                result[current_mf] = {}
            if current_mt not in result[current_mf]:
                result[current_mf][current_mt] = []
            result[current_mf][current_mt].extend(current_lines)
            if current_mf not in line_counts:
                line_counts[current_mf] = len(current_lines)
            else:
                line_counts[current_mf] += len(current_lines)
            
            # Start new section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        else:
            # Continue current section
            current_lines.append(line)
    
    # Add the last section if needed
    if current_mf is not None and current_mt is not None and current_lines:
        if current_mf not in result:
            result[current_mf] = {}
        if current_mt not in result[current_mf]:
            result[current_mf][current_mt] = []
        result[current_mf][current_mt].extend(current_lines)
        if current_mf not in line_counts:
            line_counts[current_mf] = len(current_lines)
        else:
            line_counts[current_mf] += len(current_lines)
    
    return result, line_counts
