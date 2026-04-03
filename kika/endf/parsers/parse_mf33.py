"""
Parser for MF33 (Cross-Section Covariances) sections in ENDF files.

MF33 contains covariance data for neutron cross sections given in File 3.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf33.mf33 import MF33MT, Subsection, NCSubSubsection, NISubSubsectionRecord
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf33(lines: List[str]) -> MF:
    """
    Parse MF33 (Cross-Section Covariances) data.

    Args:
        lines: List of string lines from the MF33 section

    Returns:
        MF object with parsed MF33 data
    """
    logger.debug(f"Parsing MF33 with {len(lines)} lines")
    mf = MF(number=33)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            logger.debug("Ignoring MT=0 marker lines")
            continue

        try:
            logger.debug(f"Parsing MT{mt} with {len(mt_lines)} lines")
            mt_section = parse_mf33_mt(mt_lines, mt)
            mf.add_section(mt_section)
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF33: {e}")

    logger.debug("Finished parsing MF33")
    return mf


def _read_list_values(lines: List[str], current_line: int, count: int):
    """Read `count` float values from consecutive ENDF data lines.

    Returns (values, new_current_line).
    """
    all_values = []
    while len(all_values) < count and current_line < len(lines):
        vl = parse_line(lines[current_line])
        current_line += 1
        for i in range(1, 7):
            if len(all_values) < count:
                v = vl.get(f"C{i}")
                if v is not None:
                    all_values.append(v)
    return all_values, current_line


def parse_mf33_mt(lines: List[str], mt: int) -> MF33MT:
    """
    Parse a MT section from MF33.

    Args:
        lines: List of string lines from the MT section
        mt: The MT section number

    Returns:
        MF33MT object with parsed data
    """
    if len(lines) < 1:
        raise ValueError(f"Insufficient data provided for MT{mt} section in MF33")

    # Parse HEAD line: ZA, AWR, 0, MTL, 0, NL
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    mtl = header.get("C4")
    nl = header.get("C6")

    mat, mf_num, mt_num = parse_endf_id(lines[0])

    logger.debug(f"Parsing MF33 MT{mt}: ZA={za}, AWR={awr}, MTL={mtl}, NL={nl}")

    mt_section = MF33MT(
        number=mt,
        _za=za,
        _awr=awr,
        _mtl=mtl,
        _nl=nl,
        _mat=mat,
        _mf=33,
    )

    # Lumped component: MTL != 0 means no subsections
    if mtl and mtl != 0:
        logger.debug(f"MT{mt} is a lumped component of MT{mtl}, no subsections")
        return mt_section

    current_line = 1

    for subsec_idx in range(nl if nl else 0):
        if current_line >= len(lines):
            break

        # Subsection CONT: XMF1, XLFS1, MAT1, MT1, NC, NI
        cont = parse_line(lines[current_line])
        current_line += 1

        xmf1 = cont.get("C1")
        xlfs1 = cont.get("C2")
        mat1 = cont.get("C3")
        mt1 = cont.get("C4")
        nc = cont.get("C5")
        ni = cont.get("C6")

        logger.debug(
            f"Subsection {subsec_idx + 1}/{nl}: XMF1={xmf1}, XLFS1={xlfs1}, "
            f"MAT1={mat1}, MT1={mt1}, NC={nc}, NI={ni}"
        )

        subsection = Subsection(
            xmf1=xmf1, xlfs1=xlfs1, mat1=mat1, mt1=mt1, nc=nc, ni=ni
        )

        # --- NC-type sub-subsections ---
        for nc_idx in range(nc if nc else 0):
            if current_line >= len(lines):
                break

            # CONT: 0.0, 0.0, 0, LTY, 0, 0
            nc_cont = parse_line(lines[current_line])
            current_line += 1
            lty = nc_cont.get("C4")

            logger.debug(f"NC sub-subsection {nc_idx + 1}/{nc}: LTY={lty}")

            nc_record = NCSubSubsection(lty=lty)

            if current_line >= len(lines):
                break

            # LIST header
            list_hdr = parse_line(lines[current_line])
            current_line += 1

            nc_record.e1 = list_hdr.get("C1")
            nc_record.e2 = list_hdr.get("C2")

            if lty == 0:
                # LIST: E1, E2, 0, 0, 2*NCI, NCI / {CI, XMTI}
                nci = list_hdr.get("C6")
                nt_nc = list_hdr.get("C5")  # 2*NCI
                nc_record.nci = nci

                values, current_line = _read_list_values(
                    lines, current_line, nt_nc if nt_nc else 0
                )
                for i in range(0, len(values), 2):
                    nc_record.ci.append(values[i])
                    if i + 1 < len(values):
                        nc_record.xmti.append(values[i + 1])

                logger.debug(
                    f"LTY=0: E1={nc_record.e1}, E2={nc_record.e2}, "
                    f"NCI={nci}, pairs={len(nc_record.ci)}"
                )

            elif lty in (1, 2, 3):
                # LIST: E1, E2, MATS, MTS, 2*NEI+2, NEI / (XMFS, XLFSS), {EI, WEI}
                nc_record.mats = list_hdr.get("C3")
                nc_record.mts = list_hdr.get("C4")
                nei = list_hdr.get("C6")
                nt_nc = list_hdr.get("C5")  # 2*NEI + 2
                nc_record.nei = nei

                values, current_line = _read_list_values(
                    lines, current_line, nt_nc if nt_nc else 0
                )

                if len(values) >= 2:
                    nc_record.xmfs = values[0]
                    nc_record.xlfss = values[1]
                    for i in range(2, len(values), 2):
                        nc_record.ei.append(values[i])
                        if i + 1 < len(values):
                            nc_record.wei.append(values[i + 1])

                logger.debug(
                    f"LTY={lty}: E1={nc_record.e1}, E2={nc_record.e2}, "
                    f"MATS={nc_record.mats}, MTS={nc_record.mts}, NEI={nei}"
                )
            else:
                logger.warning(f"Unknown LTY={lty} in NC sub-subsection, skipping")

            subsection.nc_records.append(nc_record)

        # --- NI-type sub-subsections ---
        for ni_idx in range(ni if ni else 0):
            if current_line >= len(lines):
                break

            # LIST header: 0.0, 0.0, LT/LS, LB, NT, NP/NE
            list_hdr = parse_line(lines[current_line])
            current_line += 1

            c3 = list_hdr.get("C3")
            lb = list_hdr.get("C4")
            nt = list_hdr.get("C5")
            c6 = list_hdr.get("C6")

            logger.debug(
                f"NI sub-subsection {ni_idx + 1}/{ni}: LB={lb}, NT={nt}, C3={c3}, C6={c6}"
            )

            ni_record = NISubSubsectionRecord(lb=lb, nt=nt)

            if lb in (0, 1, 2, 8, 9):
                # Single E-table: LT=0, NP pairs
                ni_record.lt = c3
                ni_record.np = c6
                ni_record.ne = c6

                values, current_line = _read_list_values(lines, current_line, nt if nt else 0)
                ni_record.raw_list_values = values[:]

                ne = int(c6 or 0)
                for i in range(ne):
                    ni_record.e_table_k.append(values[2 * i])
                    ni_record.f_table_k.append(values[2 * i + 1])

            elif lb in (3, 4):
                # Two E-tables: LT pairs in second table, (NP-LT) pairs in first
                lt = c3
                np_total = c6  # Total pairs across both tables
                ni_record.lt = lt
                ni_record.np = np_total

                values, current_line = _read_list_values(lines, current_line, nt if nt else 0)
                ni_record.raw_list_values = values[:]

                np_first = int((np_total or 0) - (lt or 0))
                np_second = int(lt or 0)

                # First table: (NP - LT) pairs
                for i in range(np_first):
                    ni_record.e_table_k.append(values[2 * i])
                    ni_record.f_table_k.append(values[2 * i + 1])

                # Second table: LT pairs
                offset = 2 * np_first
                for i in range(np_second):
                    ni_record.e_table_l.append(values[offset + 2 * i])
                    ni_record.f_table_l.append(values[offset + 2 * i + 1])

                logger.debug(
                    f"LB={lb}: first table {np_first} pairs, second table {np_second} pairs"
                )

            elif lb == 5:
                # Symmetric/asymmetric matrix: LS in C3, NE in C6
                ls = c3
                ne = c6
                ni_record.ls = ls
                ni_record.ne = ne

                if ne and ne < 2:
                    raise ValueError(f"LB=5 requires NE >= 2, got NE={ne}")

                m = int(ne or 0) - 1
                if ls == 0:
                    num_matrix_elements = m * m
                elif ls == 1:
                    num_matrix_elements = m * (m + 1) // 2
                else:
                    raise ValueError(f"LB=5 unknown LS value: {ls}")

                expected_nt = int(ne or 0) + num_matrix_elements
                values_to_read = nt if nt else expected_nt
                if nt and nt != expected_nt:
                    logger.warning(
                        f"LB=5 NT mismatch: expected {expected_nt}, got {nt}. Using NT={nt}"
                    )

                values, current_line = _read_list_values(
                    lines, current_line, int(values_to_read)
                )

                ne_int = int(ne or 0)
                ni_record.energies = values[:ne_int]
                ni_record.matrix = values[ne_int:]

                logger.debug(
                    f"LB=5 LS={ls}: {ne_int} energies, {len(ni_record.matrix)} matrix values"
                )

            elif lb == 6:
                # Rectangular matrix: NER in C6
                ner = c6
                ni_record.ne = ner

                ner_int = int(ner or 0)
                if nt and (int(nt) - 1) % ner_int != 0:
                    raise ValueError(
                        f"LB=6 invalid NT={nt} for NER={ner}: "
                        f"(NT-1)={int(nt)-1} not divisible by NER"
                    )

                nec = (int(nt or 0) - 1) // ner_int if ner_int else 0

                if ner_int < 2 or nec < 2:
                    raise ValueError(
                        f"LB=6 requires NER >= 2 and NEC >= 2, got NER={ner_int}, NEC={nec}"
                    )

                values, current_line = _read_list_values(
                    lines, current_line, int(nt or 0)
                )

                ni_record.row_energies = values[:ner_int]
                ni_record.col_energies = values[ner_int:ner_int + nec]
                ni_record.rect_matrix = values[ner_int + nec:]

                expected_matrix_size = (ner_int - 1) * (nec - 1)
                if len(ni_record.rect_matrix) != expected_matrix_size:
                    raise ValueError(
                        f"LB=6 matrix size mismatch: expected {expected_matrix_size}, "
                        f"got {len(ni_record.rect_matrix)}"
                    )

                logger.debug(
                    f"LB=6: NER={ner_int}, NEC={nec}, "
                    f"matrix {ner_int-1}x{nec-1}={len(ni_record.rect_matrix)} values"
                )

            else:
                raise ValueError(f"Unsupported LB value: {lb}")

            subsection.ni_records.append(ni_record)

        mt_section.add_subsection(subsection)

    logger.debug(f"Finished parsing MF33 MT{mt}")
    return mt_section
