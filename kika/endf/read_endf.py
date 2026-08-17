"""
High-level functions for reading ENDF files with selective parsing.
"""
import os
from typing import List, Optional, Union

from .classes.endf import ENDF
from .parsers.parse_endf import (
    parse_endf_file,
    parse_mf_from_file,
    scan_mat_number,
    MF_PARSERS,
)
from .classes.mf1.mf1mt451 import MF1MT451
from .classes.mf2.mf2mt151 import MF2MT151
from .classes.mf3.mf3mt import MF3MT
from .classes.mf4.base import MF4MT


def read_endf(filepath: str, mf_numbers: Optional[Union[int, List[int]]] = None) -> ENDF:
    """
    Read and parse an ENDF file, optionally selecting specific MF sections.
    
    Args:
        filepath: Path to the ENDF file
        mf_numbers: Optional MF number or list of MF numbers to parse 
                   (None = all MF sections with registered parsers)
                   
    Returns:
        ENDF object with parsed data
        
    Notes:
        Parsers are available for MF1, MF2, MF3, MF4, MF5, MF7, MF31, MF32,
        MF33, MF34 and MF35 — the registry is
        :data:`kika.endf.parsers.parse_endf.MF_PARSERS`, which is the list this
        note has drifted from before. Other MF sections are skipped with a
        warning.

    Examples:
        # Parse all MF sections with registered parsers
        endf = read_endf("path/to/file")
        
        # Parse only MF1 and MF4
        endf = read_endf("path/to/file", mf_numbers=[1, 4])
        
        # Parse only MF4
        endf = read_endf("path/to/file", mf_numbers=4)
    """
    # Normalize mf_numbers input
    if mf_numbers is not None and isinstance(mf_numbers, int):
        mf_numbers = [mf_numbers]
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENDF file not found: {filepath}")
        
    # If no filtering, parse entire file with all available parsers
    if mf_numbers is None:
        return parse_endf_file(filepath)
    
    # Create empty ENDF object
    endf = ENDF()

    # MAT is a property of the tape, not of the sections asked for, so a
    # targeted parse must report the same one a full parse does. It did not:
    # only parse_endf_file set it, so read_endf(f, mf_numbers=[1]).zaid was
    # None while read_endf(f).zaid was 26056 — and kika/sampling built its
    # output directory from the former, writing perturbed samples under
    # endf/unknown/ where the pairing stage looked under endf/26056/.
    with open(filepath, 'r') as f:
        endf.mat = scan_mat_number(f.readlines())

    # Parse each requested MF section
    for mf_number in mf_numbers:
        mf = parse_mf_from_file(filepath, mf_number)
        if mf is None:
            continue

        # Add the MF to the ENDF object
        endf.add_file(mf)

    return endf


def read_mt451(filepath: str) -> Optional['MF1MT451']:
    """
    Read only the MT451 (General Information) section from an ENDF file.
    
    Args:
        filepath: Path to the ENDF file
        
    Returns:
        MT451 object if found, None otherwise
    """
    # Parse MF1
    mf1 = parse_mf_from_file(filepath, 1)
    if mf1 and 451 in mf1.sections:
        return mf1.sections[451]
    return None


def read_mf3_mt(filepath: str, mt_number: int) -> Optional['MF3MT']:
    """
    Read a specific MT section from MF3 (Reaction Cross Sections) in an ENDF file.

    Args:
        filepath: Path to the ENDF file
        mt_number: MT section number to read (e.g. 1=total, 2=elastic, 18=fission)

    Returns:
        MF3MT object if found, None otherwise
    """
    mf3 = parse_mf_from_file(filepath, 3)
    if mf3 and mt_number in mf3.sections:
        return mf3.sections[mt_number]
    return None


def read_mf2(filepath: str) -> Optional['MF2MT151']:
    """
    Read the MF2/MT151 (Resonance Parameters) section from an ENDF file.

    Args:
        filepath: Path to the ENDF file

    Returns:
        MF2MT151 object if found, None otherwise
    """
    mf2 = parse_mf_from_file(filepath, 2)
    if mf2 and 151 in mf2.sections:
        return mf2.sections[151]
    return None


def read_mf4_mt(filepath: str, mt_number: int) -> Optional['MF4MT']:
    """
    Read a specific MT section from MF4 in an ENDF file.

    Args:
        filepath: Path to the ENDF file
        mt_number: MT section number to read

    Returns:
        MF4MT object if found, None otherwise
    """
    # Parse MF4
    mf4 = parse_mf_from_file(filepath, 4)
    if mf4 and mt_number in mf4.sections:
        return mf4.sections[mt_number]
    return None


def read_mf7_mt(filepath: str, mt_number: int):
    """
    Read a specific MT section from MF7 (thermal scattering law) in an ENDF file.

    Args:
        filepath: Path to the ENDF file
        mt_number: 2 (elastic), 4 (incoherent inelastic) or 451 (composition)

    Returns:
        MF7MT2, MF7MT4 or MF7MT451 if found, None otherwise

    Notes:
        MF7 lives only in the thermal scattering sublibrary (NSUB=12) — no
        neutron evaluation in ENDF/B-VIII.1 carries one — so *filepath* is
        expected to be a TSL tape such as ``tsl-HinH2O.endf``. These are large:
        the MF7/MT4 of ``tsl-HinH2O.endf`` is 1.14 million records and takes
        about 35 s to decode.
    """
    mf7 = parse_mf_from_file(filepath, 7)
    if mf7 and mt_number in mf7.sections:
        return mf7.sections[mt_number]
    return None
