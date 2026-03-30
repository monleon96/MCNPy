"""
Parser functions for MF1 sections in ENDF files.
"""
from typing import Dict, List, Tuple

from ..classes.mf import MF
from ..classes.mf1.mf1mt451 import MF1MT451
from ..classes.mf1.mf1mt452 import MF1MT452
from ..classes.mf1.mf1mt455 import MF1MT455
from ..classes.mf1.mf1mt456 import MF1MT456
from ..classes.mf1.mf1mt458 import MF1MT458
from ..classes.mf1.mf1mt460 import MF1MT460
from ..utils import (
    parse_line, parse_endf_id, group_lines_by_mt_with_positions,
    parse_tab1, parse_tab2, parse_data_values,
)
from ...utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)


def parse_mf1(lines: List[str], file_offset: int = 0) -> MF:
    """
    Parse MF1 (General Information) data.

    Args:
        lines: List of string lines from the MF1 section
        file_offset: Line number offset from the start of the original file

    Returns:
        MF object with parsed MF1 data
    """
    logger.debug(f"Parsing MF1 with {len(lines)} lines")
    mf = MF(number=1)

    # Record number of lines
    mf.num_lines = len(lines)

    # Group lines by MT sections with position tracking
    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    # Parse MT451 (General Information)
    if 451 in mt_groups:
        logger.debug(f"Parsing MT451 with {len(mt_groups[451])} lines")
        mt451 = parse_mt451(mt_groups[451])
        if 451 in line_counts:
            mt451.num_lines = line_counts[451]
        mf.add_section(mt451)
        logger.debug("Successfully parsed MT451")
    else:
        logger.warning("No MT451 section found in MF1")

    # Parse MT452 (Total nubar)
    if 452 in mt_groups:
        logger.debug(f"Parsing MT452 with {len(mt_groups[452])} lines")
        try:
            mt452 = parse_mt452(mt_groups[452])
            if 452 in line_counts:
                mt452.num_lines = line_counts[452]
            mf.add_section(mt452)
            logger.debug("Successfully parsed MT452")
        except Exception as e:
            logger.warning(f"Error parsing MT452: {e}")

    # Parse MT455 (Delayed nubar)
    if 455 in mt_groups:
        logger.debug(f"Parsing MT455 with {len(mt_groups[455])} lines")
        try:
            mt455 = parse_mt455(mt_groups[455])
            if 455 in line_counts:
                mt455.num_lines = line_counts[455]
            mf.add_section(mt455)
            logger.debug("Successfully parsed MT455")
        except Exception as e:
            logger.warning(f"Error parsing MT455: {e}")

    # Parse MT456 (Prompt nubar)
    if 456 in mt_groups:
        logger.debug(f"Parsing MT456 with {len(mt_groups[456])} lines")
        try:
            mt456 = parse_mt456(mt_groups[456])
            if 456 in line_counts:
                mt456.num_lines = line_counts[456]
            mf.add_section(mt456)
            logger.debug("Successfully parsed MT456")
        except Exception as e:
            logger.warning(f"Error parsing MT456: {e}")

    # Parse MT458 (Fission energy release)
    if 458 in mt_groups:
        logger.debug(f"Parsing MT458 with {len(mt_groups[458])} lines")
        try:
            mt458 = parse_mt458(mt_groups[458])
            if 458 in line_counts:
                mt458.num_lines = line_counts[458]
            mf.add_section(mt458)
            logger.debug("Successfully parsed MT458")
        except Exception as e:
            logger.warning(f"Error parsing MT458: {e}")

    # Parse MT460 (Delayed photon data)
    if 460 in mt_groups:
        logger.debug(f"Parsing MT460 with {len(mt_groups[460])} lines")
        try:
            mt460 = parse_mt460(mt_groups[460])
            if 460 in line_counts:
                mt460.num_lines = line_counts[460]
            mf.add_section(mt460)
            logger.debug("Successfully parsed MT460")
        except Exception as e:
            logger.warning(f"Error parsing MT460: {e}")

    logger.debug("Finished parsing MF1")
    return mf


def parse_mt451(lines: List[str]) -> MF1MT451:
    """
    Parse MT451 (General Information) data.

    Args:
        lines: List of string lines from the MT451 section

    Returns:
        MT451 object with parsed data
    """
    logger.debug(f"Parsing MT451 with {len(lines)} lines")
    mt451 = MF1MT451()

    if len(lines) < 4:  # Need at least 4 lines for basic metadata
        logger.warning(f"Insufficient lines for MT451: {len(lines)} < 4")
        return mt451

    # Parse first line - numeric data
    line1 = parse_line(lines[0])
    mt451._za = int(round(float(line1.get("C1"))))
    mt451._awr = line1.get("C2")
    mt451._lrp = line1.get("C3")
    mt451._lfi = line1.get("C4")
    mt451._nlib = line1.get("C5")
    mt451._nmod = line1.get("C6")
    logger.debug(f"MT451 header: ZA={mt451._za}, AWR={mt451._awr}")

    # Parse second line - numeric data
    line2 = parse_line(lines[1])
    mt451._elis = line2.get("C1")
    mt451._sta = line2.get("C2")
    mt451._lis = line2.get("C3")
    mt451._liso = line2.get("C4")
    # Skip C5 as it's a placeholder (0)
    mt451._nfor = line2.get("C6")

    # Parse third line - numeric data
    line3 = parse_line(lines[2])
    mt451._awi = line3.get("C1")
    mt451._emax = line3.get("C2")
    mt451._lrel = line3.get("C3")
    # Skip C4 as it's a placeholder (0)
    mt451._nsub = line3.get("C5")
    mt451._nver = line3.get("C6")

    # Parse fourth line - numeric data
    line4 = parse_line(lines[3])
    mt451._temp = line4.get("C1")
    # Skip C2 as it's a placeholder (0.0)
    mt451._ldrv = line4.get("C3")
    # Skip C4 as it's a placeholder (0)
    mt451._nwd = line4.get("C5")
    mt451._nxc = line4.get("C6")

    logger.debug(f"MT451 metadata: NWD={mt451._nwd}, NXC={mt451._nxc}")

    # Get MAT number from first line
    mat, mf, mt = parse_endf_id(lines[0])
    mt451._mat = mat

    # Determine how many lines are in the MT451 section
    nwd = mt451._nwd if mt451._nwd is not None else 0
    nxc = mt451._nxc if mt451._nxc is not None else 0

    # Store the raw text section (NWD lines after the 4 header lines)
    if len(lines) >= 4 + nwd:
        mt451._text_lines = lines[:4+nwd]  # Include the 4 header lines + text lines
        logger.debug(f"Stored {len(mt451._text_lines)} text lines")

    # Extract basic text fields for convenience (if lines are available)
    if len(lines) >= 5:
        line = lines[4]
        if len(line) >= 66:
            mt451._zsymam = line[:11].strip()    # cols 1-11
            mt451._alab = line[11:22].strip()    # cols 12-22
            mt451._edate = line[22:32].strip()   # cols 23-32
            mt451._auth = line[33:66].strip()    # cols 34-66
            logger.debug(f"Extracted isotope info: {mt451._zsymam} {mt451._alab}")

    if len(lines) >= 6:
        line = lines[5]
        if len(line) >= 66:
            mt451._ref = line[1:22].strip()      # cols 2-22
            mt451._ddate = line[22:32].strip()   # cols 23-32
            mt451._rdate = line[33:43].strip()   # cols 34-43
            mt451._endate = line[55:63].strip()  # cols 56-63
            logger.debug(f"Extracted reference and dates: {mt451._ref}")

    # Parse directory entries
    dir_start = 4 + nwd  # Directory starts after header + text lines
    dir_entries_parsed = 0
    for i in range(dir_start, min(dir_start + nxc, len(lines))):
        line_data = parse_line(lines[i])
        mf_val = line_data.get("C3")
        mt_val = line_data.get("C4")
        nc_val = line_data.get("C5")
        mod_val = line_data.get("C6")

        # If we hit end marker, break
        if mt_val == 0:
            break

        # Add valid entries to directory
        if mf_val is not None and mt_val is not None and nc_val is not None and mod_val is not None:
            mt451.add_directory_entry(mf_val, mt_val, nc_val, mod_val)
            dir_entries_parsed += 1

    logger.debug(f"Parsed {dir_entries_parsed} directory entries")
    logger.debug("Finished parsing MT451")
    return mt451


# ---------------------------------------------------------------------------
#  Nubar parsers (MT452, MT455, MT456)
# ---------------------------------------------------------------------------

def _parse_nubar_section(lines: List[str], cls, default_mt: int):
    """
    Shared parser for MT452 and MT456 (identical record structure).

    HEAD: [ZA, AWR, 0, LNU, 0, 0]
    LNU=1: LIST [0, 0, 0, 0, NC, 0 / C1..CNC]
    LNU=2: TAB1 [0, 0, 0, 0, NR, NP / interp / E, nu]
    """
    if not lines:
        raise ValueError(f"No data for MT{default_mt}")

    head = parse_line(lines[0])
    mat, _, _ = parse_endf_id(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    lnu = int(head.get("C4", 0) or 0)

    obj = cls()
    obj._za = za
    obj._awr = awr
    obj._mat = mat
    obj._lnu = lnu

    if lnu == 1:
        # LIST record at line 1
        list_hdr = parse_line(lines[1])
        nc = int(list_hdr.get("C5", 0) or 0)
        obj._nc = nc
        coefficients, _ = parse_data_values(lines, 2, nc)
        obj._coefficients = coefficients

    elif lnu == 2:
        # TAB1 record at line 1
        tab1_hdr, interp_pairs, energies, nubar, _ = parse_tab1(lines, 1)
        obj._nr = len(interp_pairs)
        obj._np = len(energies)
        obj._interpolation = interp_pairs
        obj._energies = energies
        obj._nubar = nubar

    return obj


def parse_mt452(lines: List[str]) -> MF1MT452:
    """Parse MT452 (total nubar)."""
    return _parse_nubar_section(lines, MF1MT452, 452)


def parse_mt456(lines: List[str]) -> MF1MT456:
    """Parse MT456 (prompt nubar)."""
    return _parse_nubar_section(lines, MF1MT456, 456)


def parse_mt455(lines: List[str]) -> MF1MT455:
    """
    Parse MT455 (delayed nubar).

    HEAD: [ZA, AWR, LDG, LNU, 0, 0]
    Decay block (LDG-dependent) then nubar block (LNU-dependent).
    """
    if not lines:
        raise ValueError("No data for MT455")

    head = parse_line(lines[0])
    mat, _, _ = parse_endf_id(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    ldg = int(head.get("C3", 0) or 0)
    lnu = int(head.get("C4", 0) or 0)

    obj = MF1MT455()
    obj._za = za
    obj._awr = awr
    obj._mat = mat
    obj._ldg = ldg
    obj._lnu = lnu

    idx = 1  # current line index (after HEAD)

    # --- Decay-constant block ---
    if ldg == 0:
        # LIST: header at idx, then data
        list_hdr = parse_line(lines[idx])
        nnf = int(list_hdr.get("C5", 0) or 0)
        obj._nnf = nnf
        idx += 1
        decay_constants, idx = parse_data_values(lines, idx, nnf)
        obj._decay_constants = decay_constants

    elif ldg == 1:
        # TAB2 header + NE LIST records
        tab2_hdr, decay_interp, idx = parse_tab2(lines, idx)
        ne = int(tab2_hdr.get("C6", 0) or 0)
        obj._decay_nr = len(decay_interp)
        obj._decay_ne = ne
        obj._decay_interp = decay_interp

        for _ in range(ne):
            list_hdr = parse_line(lines[idx])
            energy = list_hdr.get("C2", 0.0)
            npl = int(list_hdr.get("C5", 0) or 0)
            idx += 1
            values, idx = parse_data_values(lines, idx, npl)
            obj._decay_energies.append(energy)
            obj._decay_data.append(values)

        # Extract NNF from first LIST record
        if obj._decay_data:
            obj._nnf = len(obj._decay_data[0]) // 2

    # --- Nubar block ---
    if lnu == 1:
        list_hdr = parse_line(lines[idx])
        nc = int(list_hdr.get("C5", 0) or 0)
        obj._nc = nc
        idx += 1
        coefficients, idx = parse_data_values(lines, idx, nc)
        obj._coefficients = coefficients

    elif lnu == 2:
        tab1_hdr, interp_pairs, energies, nubar, idx = parse_tab1(lines, idx)
        obj._nr = len(interp_pairs)
        obj._np = len(energies)
        obj._interpolation = interp_pairs
        obj._energies = energies
        obj._nubar = nubar

    return obj


def parse_mt458(lines: List[str]) -> MF1MT458:
    """
    Parse MT458 (fission energy release).

    HEAD: [ZA, AWR, 0, LFC, 0, NFC]
    LFC=0: LIST [0, 0, 0, NPLY, NPL, N2 / coefficients]
    LFC=1: LIST (thermal) + NFC TAB1 records
    """
    if not lines:
        raise ValueError("No data for MT458")

    head = parse_line(lines[0])
    mat, _, _ = parse_endf_id(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    lfc = int(head.get("C4", 0) or 0)
    nfc = int(head.get("C6", 0) or 0)

    obj = MF1MT458()
    obj._za = za
    obj._awr = awr
    obj._mat = mat
    obj._lfc = lfc
    obj._nfc = nfc

    idx = 1  # after HEAD

    if lfc == 0:
        # LIST record: polynomial coefficients
        list_hdr = parse_line(lines[idx])
        nply = int(list_hdr.get("C4", 0) or 0)
        npl = int(list_hdr.get("C5", 0) or 0)
        obj._nply = nply
        idx += 1
        coefficients, idx = parse_data_values(lines, idx, npl)
        obj._coefficients = coefficients

    elif lfc == 1:
        # LIST record: thermal values
        list_hdr = parse_line(lines[idx])
        npl = int(list_hdr.get("C5", 0) or 0)
        idx += 1
        thermal_values, idx = parse_data_values(lines, idx, npl)
        obj._thermal_values = thermal_values

        # NFC TAB1 records
        for _ in range(nfc):
            tab1_hdr, interp_pairs, energies, values, idx = parse_tab1(lines, idx)
            ldrv = int(tab1_hdr.get("C3", 0) or 0)
            ifc = int(tab1_hdr.get("C4", 0) or 0)
            obj._tab_components.append({
                "ldrv": ldrv,
                "ifc": ifc,
                "interpolation": interp_pairs,
                "energies": energies,
                "values": values,
            })

    return obj


def parse_mt460(lines: List[str]) -> MF1MT460:
    """
    Parse MT460 (delayed photon data).

    HEAD: [ZA, AWR, LO, 0, NG, 0]
    LO=1: NG TAB1 records
    LO=2: LIST with NNF decay constants
    """
    if not lines:
        raise ValueError("No data for MT460")

    head = parse_line(lines[0])
    mat, _, _ = parse_endf_id(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    lo = int(head.get("C3", 0) or 0)
    ng = int(head.get("C5", 0) or 0)

    obj = MF1MT460()
    obj._za = za
    obj._awr = awr
    obj._mat = mat
    obj._lo = lo
    obj._ng = ng

    idx = 1  # after HEAD

    if lo == 1:
        # NG TAB1 records
        for _ in range(ng):
            tab1_hdr, interp_pairs, times, multiplicities, idx = parse_tab1(lines, idx)
            energy = tab1_hdr.get("C1", 0.0)
            photon_index = int(tab1_hdr.get("C3", 0) or 0)
            obj._photon_data.append({
                "energy": energy,
                "index": photon_index,
                "interpolation": interp_pairs,
                "times": times,
                "multiplicities": multiplicities,
            })

    elif lo == 2:
        # LIST: decay constants
        list_hdr = parse_line(lines[idx])
        nnf = int(list_hdr.get("C5", 0) or 0)
        obj._nnf = nnf
        idx += 1
        decay_constants, idx = parse_data_values(lines, idx, nnf)
        obj._decay_constants = decay_constants

    return obj
