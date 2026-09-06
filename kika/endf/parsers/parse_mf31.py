"""
Parser for MF31 (Covariances of the Average Number of Neutrons per Fission)
sections in ENDF files.

MF31 holds the covariances of nu-bar (MT=452 total, MT=455 delayed, MT=456
prompt), given in File 1. Per the ENDF-6 manual (Ch. 31), the particle-induced
formats for MT=452/455/456 are *directly analogous to those of File 33*, and
all File 33 procedures apply. This parser therefore reuses the MF33 record
machinery verbatim (``parse_mf33_mt`` decodes the same NC/NI sub-subsections
and LB-types); only the file number differs.
"""
from typing import List

from ..classes.mf import MF
from .parse_mf33 import parse_mf33_mt
from ..utils import group_lines_by_mt_with_positions
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf31(lines: List[str]) -> MF:
    """
    Parse MF31 (nu-bar covariances) data.

    Args:
        lines: List of string lines from the MF31 section

    Returns:
        MF object (number=31) holding one MF33MT-shaped section per MT.
    """
    logger.debug(f"Parsing MF31 with {len(lines)} lines")
    mf = MF(number=31)
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
            # Record machinery is shared with MF33; tag the section as MF31
            # so downstream code and serialization report the right file.
            mt_section._mf = 31
            mf.add_section(mt_section)
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF31: {e}")

    logger.debug("Finished parsing MF31")
    return mf
