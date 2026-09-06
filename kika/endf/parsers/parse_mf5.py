"""Parser for MF5 (energy distributions of secondary neutrons).

Follows :mod:`~kika.endf.parsers.parse_mf4` in shape: a file-level function that
groups by MT and logs per-section failures rather than losing the whole file to
one bad section, and a per-MT function under it.

**The subsection header is a TAB1.** ENDF-6 §5.1 writes ``U`` in C1 and ``LF``
in L2 of the same record that carries ``p_k(E)``, so
:func:`~kika.endf.utils.parse_tab1` returns the law, the weight and the
interpolation in one call. And :func:`~kika.endf.utils.parse_tab2` deliberately
does not consume the records under it, so the NE sub-TAB1s of an LF=1 partial
are read by looping here — the same arrangement ``_parse_tabulated_blocks``
uses in MF4.
"""
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf5.analytic import ANALYTIC_LAWS, ANALYTIC_RECORDS
from ..classes.mf5.base import MF5MT
from ..classes.mf5.partials import (
    TAB1_RECORDS_AFTER_HEADER,
    MF5Partial,
    MF5PartialRaw,
    MF5PartialTabulated,
)
from ..utils import group_lines_by_mt_with_positions, parse_line, parse_tab1, parse_tab2
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def parse_mf5(lines: List[str]) -> MF:
    """Parse MF5 (energy distributions) into an :class:`MF` of :class:`MF5MT`."""
    logger.debug(f"Parsing MF5 with {len(lines)} lines")
    mf = MF(number=5)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf5_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF5: {exc}")

    return mf


def parse_mf5_mt(lines: List[str], mt: int) -> MF5MT:
    """Parse one MT section of MF5: HEAD, then NK subsections."""
    head = parse_line(lines[0])
    section = MF5MT(
        number=mt,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _nk=int(head.get("C5") or 0),
        _mat=head.get("MAT"),
    )

    idx = 1
    for partial_index in range(section._nk):
        if idx >= len(lines):
            logger.warning(
                f"MF5/MT{mt} declares NK={section._nk} but ran out of lines "
                f"after {partial_index} subsection(s)"
            )
            break
        partial, idx = _parse_partial(lines, idx, mt, partial_index)
        section.partials.append(partial)

    return section


def _parse_partial(lines: List[str], idx: int, mt: int,
                   partial_index: int) -> Tuple[MF5Partial, int]:
    """One subsection: the header TAB1, then whatever its LF says follows."""
    header, p_interp, p_energies, p_values, idx = parse_tab1(lines, idx)
    u = header.get("C1")
    lf = int(header.get("C4") or 0)

    if lf == 1:
        return _parse_lf1(lines, idx, u, p_interp, p_energies, p_values)

    body_start = idx
    records, idx = _read_law_records(lines, idx, lf, mt, partial_index)
    common = dict(
        u=u, lf=lf,
        p_interp=list(p_interp), p_energies=list(p_energies),
        p_values=list(p_values),
        raw_lines=[line[:66] for line in lines[body_start:idx]],
    )

    law_class = ANALYTIC_LAWS.get(lf)
    fields = _law_fields(lf, records) if law_class is not None else None
    if fields is None:
        # Either a law nothing decodes (LF=12), or one whose records ran out
        # mid-subsection. Both keep their bytes: a half-populated analytic
        # partial would answer questions about a spectrum it does not have.
        if law_class is not None:
            logger.warning(
                f"MF5/MT{mt} subsection {partial_index} declares LF={lf} but "
                f"carries {len(records)} of the "
                f"{TAB1_RECORDS_AFTER_HEADER[lf]} expected TAB1 record(s); "
                f"kept verbatim"
            )
        return MF5PartialRaw(**common), idx
    return law_class(**common, **fields), idx


def _law_fields(lf: int, records):
    """Map the walked TAB1 records onto the law class's field names.

    ``None`` when the count does not match what the law declares -- the caller
    falls back to keeping the bytes.
    """
    names = ANALYTIC_RECORDS.get(lf)
    if names is None or len(records) != len(names):
        return None
    fields = {}
    for (interp_name, x_name, y_name), (interp, x, y) in zip(names, records):
        fields[interp_name] = list(interp)
        fields[x_name] = list(x)
        fields[y_name] = list(y)
    return fields


def _parse_lf1(lines: List[str], idx: int, u: float, p_interp,
               p_energies, p_values) -> Tuple[MF5PartialTabulated, int]:
    """LF=1: a TAB2 over incident energy, then NE outgoing TAB1 records."""
    tab2_header, tab2_interp, idx = parse_tab2(lines, idx)
    ne = int(tab2_header.get("C6") or 0)

    incident_energies: List[float] = []
    outgoing_grids: List[List[float]] = []
    chi: List[List[float]] = []
    outgoing_interp: List[List[Tuple[int, int]]] = []

    for _ in range(ne):
        if idx >= len(lines):
            break
        sub_header, sub_interp, x_data, y_data, idx = parse_tab1(lines, idx)
        incident_energies.append(sub_header.get("C2"))
        outgoing_interp.append(list(sub_interp))
        outgoing_grids.append(list(x_data))
        chi.append(list(y_data))

    return MF5PartialTabulated(
        u=u, lf=1,
        p_interp=list(p_interp), p_energies=list(p_energies), p_values=list(p_values),
        tab2_interp=list(tab2_interp),
        incident_energies=incident_energies,
        outgoing_grids=outgoing_grids,
        chi=chi,
        outgoing_interp=outgoing_interp,
    ), idx


def _read_law_records(lines: List[str], idx: int, lf: int, mt: int,
                      partial_index: int) -> Tuple[List[Tuple], int]:
    """Walk the TAB1 records of a law, keeping what they held.

    This used to be ``_skip_lf`` and threw the records away. Walking already
    *parses* every one of them to find its end, so decoding LF=5/7/9/11 costs
    nothing here beyond returning what was read — the walk itself is unchanged,
    and so is the position it leaves ``idx`` at.

    Raises on an LF it has never been told the shape of, naming the MT and the
    subsection — a wrong record count would silently swallow the *next*
    subsection, so guessing is worse than stopping. The file-level loop catches
    this and logs it against the one MT.
    """
    try:
        n_records = TAB1_RECORDS_AFTER_HEADER[lf]
    except KeyError:
        raise ValueError(
            f"MF5/MT{mt} subsection {partial_index} uses LF={lf}, which this "
            f"parser has no record layout for (known: "
            f"{sorted(TAB1_RECORDS_AFTER_HEADER)} plus LF=1)"
        ) from None

    records: List[Tuple] = []
    for _ in range(n_records):
        if idx >= len(lines):
            break
        _, interp, x_data, y_data, idx = parse_tab1(lines, idx)
        records.append((interp, x_data, y_data))
    return records, idx
