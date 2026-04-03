"""
Parser for MF34 (Angular Distribution Covariances) sections in ENDF files.

MF34 contains covariance data for angular distributions of secondary particles.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf34.mf34 import MF34MT, Subsection, SubSubsection, SubSubsectionRecord
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions
from ...utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)
from ...utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)


def parse_mf34(lines: List[str]) -> MF:
    """
    Parse MF34 (Angular Distribution Covariances) data.
    
    Args:
        lines: List of string lines from the MF34 section
        
    Returns:
        MF object with parsed MF34 data
    """
    logger.debug(f"Parsing MF34 with {len(lines)} lines")
    mf = MF(number=34)
    
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
            mt_section = parse_mf34_mt(mt_lines, mt)
            mf.add_section(mt_section)
            
            # Add line count information if available
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF34: {e}")
    
    logger.debug("Finished parsing MF34")
    return mf


def parse_mf34_mt(lines: List[str], mt: int) -> MF34MT:
    """
    Parse a MT section from MF34.
    
    Args:
        lines: List of string lines from the MT section
        mt: The MT section number
        
    Returns:
        MF34MT object with parsed data
    """
    # Need at least two lines for header and first subsection
    if len(lines) < 2:
        raise ValueError(f"Insufficient data provided for MT{mt} section in MF34")
    
    # Parse the header line
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    # C3 is zero (unused)
    ltt = header.get("C4")
    # C5 is zero (unused)
    nmt1 = header.get("C6")  # Number of subsections
    

    logger.debug(f"Parsing MF34 MT{mt} with ZA={za}, AWR={awr}, LTT={ltt}, NMT1={nmt1}")


    # Get material number
    mat, mf, mt = parse_endf_id(lines[0])
    
    # Create the MT section object
    mt_section = MF34MT(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _ltt=ltt,
        _nmt1=nmt1,
        _mat=mat,
        _mf=mf
    )
    
    # Parse each subsection
    current_line = 1  # Start with the first subsection header
    
    for subsection_idx in range(nmt1 if nmt1 else 0):
        if current_line >= len(lines):
            break
            
        # Parse subsection header (second line of first subsection or subsequent subsections)
        subsec_line = parse_line(lines[current_line])
        current_line += 1
        
        # C1 and C2 are 0.0 (unused)
        mat1 = subsec_line.get("C3")  # MAT1 = 0.0 typically
        mt1 = subsec_line.get("C4")
        nl = subsec_line.get("C5")
        nl1 = subsec_line.get("C6")
        
        logger.debug(f"Parsing subsection {subsection_idx+1}/{nmt1} with MAT1={mat1}, MT1={mt1}, NL={nl}, NL1={nl1}")

        # Create subsection
        subsection = Subsection(
            mt1=mt1,
            nl=nl,
            nl1=nl1,
            mat1=mat1
        )
        
        # Calculate number of sub-subsections
        if mt1 == mt:
            # Only upper triangle is given when MT1=MT to avoid redundancy
            nss = (nl * (nl + 1)) // 2
        else:
            nss = nl * nl1
        
        logger.debug(f"Processing {nss} sub-subsections for this subsection")
        
        # Parse sub-subsections
        for sub_idx in range(nss):
            if current_line >= len(lines):
                break
                
            # Parse sub-subsection header
            sub_header = parse_line(lines[current_line])
            current_line += 1
            
            # C1 and C2 are 0.0 (unused)
            l = sub_header.get("C3")
            l1 = sub_header.get("C4")
            lct = sub_header.get("C5")  # System frame (0=same as MF4, 1=LAB, 2=CM)
            ni = sub_header.get("C6")  # Number of LIST records
            
            logger.debug(f"Parsing sub-subsection {sub_idx+1}/{nss} with L={l}, L1={l1}, LCT={lct}, NI={ni}")

            # Create sub-subsection
            sub_subsection = SubSubsection(
                l=l,
                l1=l1,
                lct=lct,
                ni=ni
            )
            
            # Parse LIST records
            for list_idx in range(ni):
                if current_line >= len(lines):
                    break
                    
                # Parse LIST record header
                list_header = parse_line(lines[current_line])
                current_line += 1
                
                # For LB=5: C1=0.0, C2=0.0, C3=LS, C4=LB, C5=NT, C6=NE
                # For LB=0,1,2: C1=0.0, C2=0.0, C3=LT, C4=LB, C5=NT, C6=NP
                # For LB=6: C1=0.0, C2=0.0, C3=0, C4=LB, C5=NT, C6=NER
                # C1 and C2 are 0.0 (unused)
                c3 = list_header.get("C3")  # Flag LS for LB=5, LT for LB=0..2, 0 for LB=6
                lb = list_header.get("C4")
                nt = list_header.get("C5")  # Total number of items in the LIST 
                                            #   - for LB=0,1,2: NT=NP*2 
                                            #   - for LB=5: if LS=0 (asymmetric) NT=NE*(NE+1)+1, if LS=1 (symmetric) NT=[NE*(NE+1)]/2
                                            #   - for LB=6: NT=1 + NER*NEC  (NER=row energies, NEC=col energies)
                c6 = list_header.get("C6")  # Total number of items in the arrays
                                            #   - for LB=0,1,2: c6=NP, number of (E,F) pairs
                                            #   - for LB=5: c6=NE, number of energy entries (defining NE-1 intervals)
                                            #   - for LB=6: c6=NER, number of row energies, hence NEC=(NT-1)/NER
                
                # Determine what C3 and C6 represent based on LB
                if lb in (0, 1, 2):
                    lt = c3     # if LT=0, table contains a single (E,F) table;
                                # if LT>0, table contains two (E,F) tables, the first array has (NP-LT) pairs (only for LB=3,4 - not in MF34)
                    np = c6  # Number of (E,F) pairs
                    logger.debug(f"Parsing LIST record {list_idx+1}/{ni} with LT={lt}, LB={lb}, NT={nt}, NP={np}")
                elif lb == 5:
                    ls = c3  # Flag for symmetric matrix for LB=5 (1=symmetric, 0=asymmetric)
                    ne = c6  # Number of energy entries
                    logger.debug(f"Parsing LIST record {list_idx+1}/{ni} with LS={ls}, LB={lb}, NT={nt}, NE={ne}")
                elif lb == 6:
                    # C3 should be 0 for LB=6
                    ner = c6  # Number of row energies
                    logger.debug(f"Parsing LIST record {list_idx+1}/{ni} with LB={lb}, NT={nt}, NER={ner}")
                else:
                    logger.debug(f"Parsing LIST record {list_idx+1}/{ni} with LB={lb}, NT={nt}")

                # Create LIST record
                list_record = SubSubsectionRecord(lb=lb, nt=nt)

                if lb in (0, 1, 2):
                    # Header fields for LB=0..2 (LIST):
                    # C3=LT, C4=LB, C5=NT (#floats to follow), C6=NP (#(E,F) pairs across table[s])
                    list_record.lt = lt 
                    list_record.lb = lb
                    list_record.nt = nt
                    list_record.np = np

                    values_to_read = nt
                    all_values = []
                    while len(all_values) < values_to_read and current_line < len(lines):
                        vl = parse_line(lines[current_line]); current_line += 1
                        for i in range(1, 7):
                            if len(all_values) < values_to_read:
                                v = vl.get(f"C{i}")
                                if v is not None:
                                    all_values.append(v)
                    # Preserve raw floats exactly for round-trip
                    list_record.raw_list_values = all_values[:]

                    # Decode only the common LT==0 case (single [E,F] table)
                    if (list_record.lt or 0) == 0:
                        # one (E_k, F_k) table; NP==NE
                        ne = int(list_record.np or 0)
                        list_record.ne = ne
                        for i in range(ne):
                            list_record.e_table_k.append(all_values[2*i])
                            list_record.f_table_k.append(all_values[2*i+1])
                    else:
                        # LT>0 rare two-table variant: we keep raw values for exact __str__ output
                        # and skip decoding until support is implemented.
                        logger.info(f"LB={lb} with LT={list_record.lt}>0 encountered: preserving raw LIST values for exact round-trip; skipping decode.")


                elif lb == 5:
                    # For LB=5: LS in C3, NE in C6
                    list_record.ls = ls
                    list_record.ne = ne
                    
                    logger.debug(f"LB=5 processing: LS={ls}, NE={ne}")
                    
                    # Calculate number of matrix elements based on LS and NE for LB=5
                    # Matrix is (NE-1) x (NE-1) for intervals
                    m = ne - 1  # Number of intervals = matrix dimension
                    if ne < 2:
                        raise ValueError(f"LB=5 requires NE >= 2 to define intervals, got NE={ne}")
                    
                    if ls == 0:  # Asymmetric matrix
                        num_matrix_elements = m * m  # (NE-1)²
                        expected_nt = ne + num_matrix_elements
                        logger.debug(f"LB=5 asymmetric matrix: {m}x{m} = {num_matrix_elements} elements")
                    elif ls == 1:  # Symmetric matrix - upper triangle including diagonal
                        num_matrix_elements = m * (m + 1) // 2  # (NE-1)(NE-1+1)/2
                        expected_nt = ne + num_matrix_elements
                        logger.debug(f"LB=5 symmetric matrix: {m}x{m}, upper triangle = {num_matrix_elements} elements")
                    else:
                        raise ValueError(f"LB=5 unknown LS value: {ls}. Should be 0 or 1")

                    # Total values to read: energy grid + matrix elements
                    values_to_read = ne + num_matrix_elements
                    logger.debug(f"LB=5 total values to read: {ne} (energies) + {num_matrix_elements} (matrix) = {values_to_read}")
                    
                    # Validate NT field
                    if nt != expected_nt:
                        logger.warning(f"LB=5 NT mismatch! Expected {expected_nt}, got {nt}. Proceeding with NT={nt}")
                        values_to_read = nt  # Use actual NT for reading
                    
                    # Read all the data values following the LIST header
                    all_values = []
                    remaining_values = nt
                    
                    while remaining_values > 0 and current_line < len(lines):
                        data_line = parse_line(lines[current_line])
                        current_line += 1
                        
                        # Extract all 6 values from this line
                        line_values = [
                            data_line.get("C1"), data_line.get("C2"), data_line.get("C3"),
                            data_line.get("C4"), data_line.get("C5"), data_line.get("C6")
                        ]
                        
                        # Only take the values we actually need
                        values_from_this_line = min(6, remaining_values)
                        all_values.extend(line_values[:values_from_this_line])
                        remaining_values -= values_from_this_line
                    
                    logger.debug(f"LB=5 read {len(all_values)} total values")
                    
                    # Split into energies and matrix values
                    energies = all_values[:ne]
                    matrix_values = all_values[ne:]
                    
                    logger.debug(f"LB=5 split into {len(energies)} energies and {len(matrix_values)} matrix values")
                    
                    # Store in record
                    list_record.energies = energies
                    list_record.matrix = matrix_values
                    
                    logger.debug(f"LB=5 stored {len(list_record.energies)} energies and {len(list_record.matrix)} matrix values")
                
                elif lb == 6:
                    # Rectangular matrix: NT = 1 + NER*NEC = NER + NEC + (NER-1)*(NEC-1)
                    list_record.ne = ner
                    
                    # Strict validation: (NT - 1) must be divisible by NER
                    if (nt - 1) % ner != 0:
                        raise ValueError(f"LB=6 invalid NT={nt} for NER={ner}: (NT-1)={nt-1} must be divisible by NER")
                    
                    nec = (nt - 1) // ner
                    
                    if ner < 2 or nec < 2:
                        raise ValueError(f"LB=6 requires NER >= 2 and NEC >= 2 to define intervals, got NER={ner}, NEC={nec}")
                    
                    # Calculate expected matrix size
                    r = ner - 1  # Number of row intervals
                    c = nec - 1  # Number of column intervals
                    expected_matrix_size = r * c
                    expected_total_size = ner + nec + expected_matrix_size
                    
                    if nt != expected_total_size:
                        logger.warning(f"LB=6 NT mismatch! Expected {expected_total_size} (NER={ner} + NEC={nec} + matrix={expected_matrix_size}), got {nt}")
                    
                    values_to_read = nt
                    all_values = []
                    while len(all_values) < values_to_read and current_line < len(lines):
                        vl = parse_line(lines[current_line]); current_line += 1
                        for i in range(1,7):
                            if len(all_values) < values_to_read:
                                v = vl.get(f"C{i}")
                                if v is not None:
                                    all_values.append(v)
                    
                    # Split row energies, col energies, matrix
                    list_record.row_energies = all_values[:ner]
                    list_record.col_energies = all_values[ner:ner+nec]
                    list_record.rect_matrix   = all_values[ner+nec:]
                    
                    # Validate matrix size
                    if len(list_record.rect_matrix) != expected_matrix_size:
                        raise ValueError(f"LB=6 rect_matrix size mismatch! Expected {expected_matrix_size} elements for (NER-1)*(NEC-1)=({ner}-1)*({nec}-1), got {len(list_record.rect_matrix)}")
                    
                    logger.debug(f"Parsed LB=6 record with NER={ner}, NEC={nec}, matrix size={len(list_record.rect_matrix)}")

                else:
                    raise ValueError(f"Unsupported LB value: {lb}")
                
                # Add the list record to the sub-subsection
                sub_subsection.records.append(list_record)
            
            # Add the sub-subsection to the subsection
            subsection.sub_subsections.append(sub_subsection)
        
        # Add the subsection to the MT section
        mt_section.add_subsection(subsection)
    
    logger.debug(f"Finished parsing MF34 MT{mt}")
    return mt_section