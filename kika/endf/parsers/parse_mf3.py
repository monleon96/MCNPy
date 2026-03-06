"""
Parser for MF3 (Reaction Cross Sections) sections in ENDF files.

Each MT section in MF3 consists of a HEAD record followed by a single TAB1
record giving sigma(E).
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf3.mf3mt import MF3MT
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions, parse_tab1
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf3(lines: List[str]) -> MF:
    """
    Parse MF3 (Reaction Cross Sections) data.

    Parameters
    ----------
    lines : list of str
        Lines belonging to MF3 (all MT sections concatenated).

    Returns
    -------
    MF
    """
    logger.debug(f"Parsing MF3 with {len(lines)} lines")
    mf = MF(number=3)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            logger.debug(f"Parsing MT{mt} with {len(mt_lines)} lines")
            mt_section = parse_mf3_mt(mt_lines, mt)
            mf.add_section(mt_section)
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF3: {e}")

    logger.debug("Finished parsing MF3")
    return mf


def parse_mf3_mt(lines: List[str], mt: int) -> MF3MT:
    """
    Parse a single MT section from MF3.

    Parameters
    ----------
    lines : list of str
        Lines for this MT section (HEAD + TAB1).
    mt : int
        MT number.

    Returns
    -------
    MF3MT
    """
    if not lines:
        raise ValueError(f"No data provided for MT{mt} section in MF3")

    # HEAD record (line 0)
    head = parse_line(lines[0])
    za = head.get("C1")
    awr = head.get("C2")
    mat, _, _ = parse_endf_id(lines[0])

    # TAB1 record (line 1+)
    tab1_hdr, interp_pairs, energies, cross_sections, _ = parse_tab1(lines, 1)
    qm = tab1_hdr.get("C1", 0.0) or 0.0
    qi = tab1_hdr.get("C2", 0.0) or 0.0
    lr = int(tab1_hdr.get("C4", 0) or 0)
    nr = int(tab1_hdr.get("C5", 0) or 0)
    np_count = int(tab1_hdr.get("C6", 0) or 0)

    return MF3MT(
        number=mt,
        _za=za,
        _awr=awr,
        _mat=mat,
        _qm=qm,
        _qi=qi,
        _lr=lr,
        _nr=nr,
        _np=np_count,
        _interpolation=interp_pairs,
        _energies=energies,
        _cross_sections=cross_sections,
    )
