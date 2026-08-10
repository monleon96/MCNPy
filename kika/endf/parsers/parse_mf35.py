"""Parser for MF35 (covariances of energy distributions).

One HEAD record, then NK LIST records — one per incident-energy band. Simpler
than MF33 because MF35 has no NC/NI hierarchy and no cross-material blocks:
every subsection is a self-contained absolute covariance over its own outgoing
grid.
"""
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf35.mf35 import MF35MT, MF35SubSection
from ..utils import (
    group_lines_by_mt_with_positions,
    parse_data_values,
    parse_line,
)
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf35(lines: List[str]) -> MF:
    """Parse MF35 into an :class:`MF` of :class:`MF35MT`."""
    logger.debug(f"Parsing MF35 with {len(lines)} lines")
    mf = MF(number=35)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf35_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF35: {exc}")

    return mf


def parse_mf35_mt(lines: List[str], mt: int) -> MF35MT:
    """Parse one MT section of MF35: HEAD, then NK band LIST records."""
    head = parse_line(lines[0])
    section = MF35MT(
        number=mt,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _nk=int(head.get("C5") or 0),
        _mat=head.get("MAT"),
    )

    idx = 1
    for band_index in range(section._nk):
        if idx >= len(lines):
            logger.warning(
                f"MF35/MT{mt} declares NK={section._nk} but ran out of lines "
                f"after {band_index} band(s)"
            )
            break
        band, idx = _parse_lb7(lines, idx, mt, band_index)
        section.subsections.append(band)

    return section


def _parse_lb7(lines: List[str], idx: int, mt: int,
               band_index: int) -> Tuple[MF35SubSection, int]:
    """One ``LS=1, LB=7`` LIST record.

    Refuses anything else by name. MF35 admits only this form in practice, and
    a silently mis-split body would produce a plausible-looking matrix of the
    wrong quantity — the failure this whole module is arranged to prevent.
    """
    header = parse_line(lines[idx])
    idx += 1

    ls = int(header.get("C3") or 0)
    lb = int(header.get("C4") or 0)
    nt = int(header.get("C5") or 0)
    ne = int(header.get("C6") or 0)

    if (ls, lb) != (1, 7):
        raise ValueError(
            f"MF35/MT{mt} band {band_index} is LS={ls}, LB={lb}; this parser "
            f"handles LS=1, LB=7 only (the only form the reference "
            f"evaluations use)"
        )

    expected = MF35SubSection.expected_nt(ne)
    if nt != expected:
        raise ValueError(
            f"MF35/MT{mt} band {band_index} declares NT={nt} for NE={ne}, but "
            f"an LS=1/LB=7 body is NE boundaries plus the upper triangle of a "
            f"(NE-1)² matrix, so NT must be {expected}"
        )

    values, idx = parse_data_values(lines, idx, nt)
    if len(values) != nt:
        raise ValueError(
            f"MF35/MT{mt} band {band_index} declares NT={nt} but only "
            f"{len(values)} values could be read"
        )

    return MF35SubSection(
        e1=header.get("C1"),
        e2=header.get("C2"),
        ls=ls, lb=lb, nt=nt, ne=ne,
        boundaries=list(values[:ne]),
        upper_triangle=list(values[ne:]),
        raw_list_values=list(values),
    ), idx
