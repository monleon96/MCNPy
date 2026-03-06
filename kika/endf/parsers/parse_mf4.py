"""
Parser for MF4 (Angular Distributions) sections in ENDF files.

MF4 contains angular distribution data for neutron-induced reactions.
Different formats are used based on the LTT flag:
- LTT=0: All angular distributions are isotropic
- LTT=1: Data given as Legendre expansion coefficients
- LTT=2: Data given as tabulated probability distributions
- LTT=3: Mixed representation (Legendre + tabulated)
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf4.base import MF4MT
from ..classes.mf4.isotropic import MF4MTIsotropic
from ..classes.mf4.polynomial import MF4MTLegendre
from ..classes.mf4.tabulated import MF4MTTabulated
from ..classes.mf4.mixed import MF4MTMixed
from ..utils import (
    parse_line, parse_endf_id, group_lines_by_mt_with_positions,
    parse_interp_pairs, parse_data_pairs, parse_data_values,
)
from ...utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)


def parse_mf4(lines: List[str]) -> MF:
    """
    Parse MF4 (Angular Distributions) data.

    Args:
        lines: List of string lines from the MF4 section

    Returns:
        MF object with parsed MF4 data
    """
    logger.debug(f"Parsing MF4 with {len(lines)} lines")
    mf = MF(number=4)

    # Record number of lines
    mf.num_lines = len(lines)

    # Group lines by MT sections with line counting
    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    # Parse each MT section
    for mt, mt_lines in mt_groups.items():
        # Skip MT=0 since these are section end markers, not actual data
        if mt == 0:
            logger.debug("Ignoring MT=0 marker lines (end of section indicators)")
            continue

        try:
            logger.debug(f"Parsing MT{mt} with {len(mt_lines)} lines")
            mt_section = parse_mf4_mt(mt_lines, mt)
            mf.add_section(mt_section)

            # Add line count information if available
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF4: {e}")

    logger.debug("Finished parsing MF4")
    return mf

def parse_mf4_mt(lines: List[str], mt: int) -> MF4MT:
    """
    Parse a MT section from MF4, determining the appropriate format based on LTT.

    Args:
        lines: List of string lines from the MT section
        mt: The MT section number

    Returns:
        MF4MT object of the appropriate type based on LTT
    """
    logger.debug(f"Parsing MF4 MT{mt} with {len(lines)} lines")

    # Need at least one line for the header
    if not lines:
        raise ValueError(f"No data provided for MT{mt} section in MF4")

    # Parse the header line
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    # C3 is zero (unused)
    ltt = header.get("C4")
    # C5 and C6 are zero (unused)

    logger.debug(f"MT{mt} header: ZA={za}, AWR={awr}, LTT={ltt}")

    # Get material number
    mat, mf, mt = parse_endf_id(lines[0])

    # Check which format we're dealing with and parse accordingly
    if ltt == 0:
        logger.debug(f"MT{mt} using isotropic format (LTT=0)")
        return _parse_mf4_mt_isotropic(lines, za, awr, mt, mat)
    elif ltt == 1:
        logger.debug(f"MT{mt} using Legendre expansion format (LTT=1)")
        return _parse_mf4_mt_legendre(lines, za, awr, mt, mat)
    elif ltt == 2:
        logger.debug(f"MT{mt} using tabulated format (LTT=2)")
        return _parse_mf4_mt_tabulated(lines, za, awr, mt, mat)
    elif ltt == 3:
        logger.debug(f"MT{mt} using mixed format (LTT=3)")
        return _parse_mf4_mt_mixed(lines, za, awr, mt, mat)
    else:
        raise ValueError(f"Unknown LTT value: {ltt} in MT{mt}")


def _parse_mf4_mt_isotropic(lines: List[str], za: float, awr: float,
                          mt: int, mat: int) -> MF4MTIsotropic:
    """
    Parse MT section with isotropic angular distributions (LTT=0).

    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number

    Returns:
        MF4MTIsotropic object
    """
    # For isotropic, we just need the header and any energy points
    # Parse the second line for NE (number of energies)
    if len(lines) < 2:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=0)")

    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    # C1 is 0.0 (unused)
    # C2 AWR already parsed
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)
    # C5 is 0 (unused)
    # C6 is 0 (unused)

    # Create the MT section object
    mt_section = MF4MTIsotropic(
        number=mt,
        _za=za,
        _awr=awr,
        _li=li,
        _lct=lct,
        _mat=mat
    )

    # The energy grid will be parsed once specifications are provided

    return mt_section


def _parse_mf4_mt_legendre(lines: List[str], za: float, awr: float,
                         mt: int, mat: int) -> MF4MTLegendre:
    """
    Parse MT section with Legendre expansions (LTT=1).

    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number

    Returns:
        MF4MTLegendre object with coefficients for each energy
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=1)")

    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)

    # Parse the third line with NR and NE
    line3 = parse_line(lines[2])
    nr = line3.get("C5")    # Number of interpolation regions
    ne = line3.get("C6")    # Number of energy points

    # Create the MT section object
    mt_section = MF4MTLegendre(
        number=mt,
        _za=za,
        _awr=awr,
        _li=li,
        _lct=lct,
        _nr=nr,
        _ne=ne,
        _mat=mat
    )

    # Parse the interpolation scheme pairs
    current_line = 3
    if nr and nr > 0:
        interpolation_pairs, current_line = parse_interp_pairs(lines, current_line, nr)
        mt_section._interpolation = interpolation_pairs

    # Parse Legendre coefficients for each energy point
    energies, legendre_coeffs, current_line = _parse_legendre_blocks(
        lines, current_line, ne
    )

    mt_section._energies = energies
    mt_section._legendre_coeffs = legendre_coeffs

    return mt_section


def _parse_legendre_blocks(lines, start, ne):
    """
    Parse NE energy-point Legendre coefficient LIST records.

    Returns (energies, legendre_coeffs, next_line_idx).
    """
    energies = []
    legendre_coeffs = []
    idx = start

    for _ in range(ne):
        if idx >= len(lines):
            break

        header_line = parse_line(lines[idx])
        idx += 1

        energy = header_line.get("C2")
        nl = header_line.get("C5")

        if energy is None:
            continue

        energies.append(energy)
        coeffs, idx = parse_data_values(lines, idx, nl)
        legendre_coeffs.append(coeffs)

    return energies, legendre_coeffs, idx


def _parse_tabulated_blocks(lines, start, ne):
    """
    Parse NE energy-point tabulated angular distribution TAB1-like records.

    Each record has a header line (with NR_ang, NP), interpolation pairs,
    then NP (mu, f) data pairs.

    Returns (energies, cosines, probabilities, angular_interpolation, next_line_idx).
    """
    energies = []
    cosines = []
    probabilities = []
    angular_interpolation = []
    idx = start

    for _ in range(ne):
        if idx >= len(lines):
            break

        point_header = parse_line(lines[idx])
        idx += 1

        energy = point_header.get("C2")
        nr_ang = point_header.get("C5")
        np_val = point_header.get("C6")

        if energy is None:
            continue

        energies.append(energy)

        # Angular interpolation pairs
        ang_interp_pairs = []
        if nr_ang and nr_ang > 0:
            ang_interp_pairs, idx = parse_interp_pairs(lines, idx, nr_ang)
        angular_interpolation.append(ang_interp_pairs)

        # Cosine / probability data pairs
        energy_cosines, energy_probabilities, idx = parse_data_pairs(lines, idx, np_val)
        cosines.append(energy_cosines)
        probabilities.append(energy_probabilities)

    return energies, cosines, probabilities, angular_interpolation, idx


def _parse_mf4_mt_tabulated(lines: List[str], za: float, awr: float,
                          mt: int, mat: int) -> MF4MTTabulated:
    """
    Parse MT section with tabulated distributions (LTT=2).

    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number

    Returns:
        MF4MTTabulated object with probability tables for each energy
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=2)")

    line2 = parse_line(lines[1])
    li = line2.get("C3")
    lct = line2.get("C4")

    line3 = parse_line(lines[2])
    nr = line3.get("C5")
    ne = line3.get("C6")

    mt_section = MF4MTTabulated(
        number=mt,
        _za=za,
        _awr=awr,
        _li=li,
        _lct=lct,
        _nr=nr,
        _ne=ne,
        _mat=mat
    )

    current_line = 3
    if nr and nr > 0:
        interpolation_pairs, current_line = parse_interp_pairs(lines, current_line, nr)
        mt_section._interpolation = interpolation_pairs

    energies, cosines_list, probs_list, ang_interp, _ = _parse_tabulated_blocks(
        lines, current_line, ne
    )

    mt_section._energies = energies
    mt_section._cosines = cosines_list
    mt_section._probabilities = probs_list
    mt_section._angular_interpolation = ang_interp

    return mt_section


def _parse_mf4_mt_mixed(lines: List[str], za: float, awr: float,
                      mt: int, mat: int) -> MF4MTMixed:
    """
    Parse MT section with mixed representation (LTT=3).

    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number

    Returns:
        MF4MTMixed object with Legendre and tabulated components
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=3)")

    line2 = parse_line(lines[1])
    li = line2.get("C3")
    lct = line2.get("C4")
    nm = line2.get("C6")

    # ======================================
    # Parsing of Legendre coefficients
    # ======================================

    line3 = parse_line(lines[2])
    nr = line3.get("C5")
    ne1 = line3.get("C6")

    mt_section = MF4MTMixed(
        number=mt,
        _za=za,
        _awr=awr,
        _li=li,
        _lct=lct,
        _nm=nm,
        _nr=nr,
        _ne1=ne1,
        _mat=mat
    )

    current_line = 3
    if nr and nr > 0:
        interpolation_pairs, current_line = parse_interp_pairs(lines, current_line, nr)
        mt_section._interpolation = interpolation_pairs

    energies, legendre_coeffs, current_line = _parse_legendre_blocks(
        lines, current_line, ne1
    )
    mt_section._energies = energies
    mt_section._legendre_coeffs = legendre_coeffs

    # ======================================
    # Now parse the tabulated distribution part
    # ======================================

    tab_header_line = parse_line(lines[current_line])
    current_line += 1

    nr_tab = tab_header_line.get("C5")
    ne2 = tab_header_line.get("C6")

    mt_section._ne2 = ne2
    mt_section._nr_tab = nr_tab

    if not ne2 or ne2 <= 0:
        return mt_section

    # Tabulated energy interpolation pairs
    tab_interpolation_pairs = []
    if nr_tab and nr_tab > 0:
        tab_interpolation_pairs, current_line = parse_interp_pairs(lines, current_line, nr_tab)
        mt_section._tab_interpolation = tab_interpolation_pairs

    # Tabulated distributions
    tab_energies, tab_cosines, tab_probs, ang_interp, _ = _parse_tabulated_blocks(
        lines, current_line, ne2
    )

    mt_section._tabulated_energies = tab_energies
    mt_section._tabulated_cosines = tab_cosines
    mt_section._tabulated_probabilities = tab_probs
    mt_section._angular_interpolation = ang_interp

    return mt_section
