"""Parser for MF6 (energy-angle distributions of reaction products).

Follows :mod:`~kika.endf.parsers.parse_mf5` in shape — a file-level function
that groups by MT and logs per-section failures rather than losing the whole
file to one bad section — and :mod:`~kika.endf.parsers.parse_mf7` in discipline:
:func:`_check_consumed` asserts that every record of the section was walked.

**Why the walk has to be exact, and why there is no verbatim fallback.** MF5 can
step over a law it does not decode using a table of record counts, because every
MF5 law body is a fixed number of TAB1s. MF6's bodies are data-dependent — how
many records a LAW=1 body occupies is written inside its own TAB2 — so a law
this module does not understand cannot be stepped over at all, and a guessed
length silently swallows the product after it. An unknown LAW therefore raises,
naming the MT and the product, and the file-level loop charges it to that one MT.

**The walk was verified before this parser was written.** A standalone record
walker built from the same layout was run over the MF6 of twelve ENDF/B-VIII.1
tapes and landed exactly on the SEND record of 633 sections out of 633. The
measured law census, and the negative-LAW rule that census turned up, are in
``docs/library/mf6_notes.md`` in the workspace repo.
"""
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf6.base import MF6MT
from ..classes.mf6.laws import (
    NO_BODY_LAWS,
    LawSevenAngle,
    LawSevenEnergy,
    MF6Law,
    MF6LawChargedElastic,
    MF6LawContinuum,
    MF6LawElsewhere,
    MF6LawLabAngleEnergy,
    MF6LawNoBody,
    MF6LawPhaseSpace,
    MF6LawTwoBody,
)
from ..classes.mf6.products import MF6Product
from ..utils import (
    PaddingProbe,
    group_lines_by_mt_with_positions,
    parse_data_values,
    parse_line,
    parse_tab1,
    parse_tab2,
)
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)

#: LAW values this module has a record layout for. Everything else raises.
KNOWN_LAWS = (0, 1, 2, 3, 4, 5, 6, 7)


def parse_mf6(lines: List[str]) -> MF:
    """Parse MF6 into an :class:`MF` of :class:`MF6MT` sections."""
    logger.debug(f"Parsing MF6 with {len(lines)} lines")
    mf = MF(number=6)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf6_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF6: {exc}")

    return mf


def parse_mf6_mt(lines: List[str], mt: int) -> MF6MT:
    """Parse one MT section of MF6: HEAD, then NK product subsections."""
    head = parse_line(lines[0])
    section = MF6MT(
        number=mt,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _jp=int(head.get("C3") or 0),
        _lct=int(head.get("C4") or 0),
        _nk=int(head.get("C5") or 0),
        _mat=head.get("MAT"),
    )

    probe = PaddingProbe()
    idx = 1
    for index in range(section._nk):
        if idx >= len(lines):
            logger.warning(
                f"MF6/MT{mt} declares NK={section._nk} but ran out of lines "
                f"after {index} product(s)"
            )
            break
        product, idx = _parse_product(lines, idx, mt, index, probe)
        section.products.append(product)

    section.pad = probe.resolve()
    _check_consumed(idx, lines, mt)
    return section


def _parse_product(lines: List[str], idx: int, mt: int, index: int,
                   probe: PaddingProbe) -> Tuple[MF6Product, int]:
    """One product: the header TAB1 carrying y(E), then whatever its LAW says."""
    header, y_interp, y_energies, y_values, idx = parse_tab1(lines, idx)
    probe.observe_pairs(lines, idx, len(y_energies))
    # The interpolation record sits between the header and the x/y body, so its
    # own last line is len(y)-derived lines back from where the TAB1 ended.
    probe.observe_interp(lines, idx - _pair_lines(len(y_energies)), len(y_interp))

    law = int(header.get("C4") or 0)
    law_data, idx = _parse_law(lines, idx, law, mt, index, probe)

    return MF6Product(
        zap=header.get("C1"),
        awp=header.get("C2"),
        lip=int(header.get("C3") or 0),
        y_interp=list(y_interp),
        y_energies=list(y_energies),
        y_values=list(y_values),
        law_data=law_data,
    ), idx


def _parse_law(lines: List[str], idx: int, law: int, mt: int, index: int,
               probe: PaddingProbe) -> Tuple[MF6Law, int]:
    """Dispatch on LAW, and refuse rather than guess at one we do not know."""
    if law < 0:
        # A negative LAW defers the distribution to File |LAW| and writes no
        # records of its own. Measured on ENDF/B-VIII.1 for -5 and -15; the
        # rule is general, and _check_consumed is what would notice if a
        # negative law ever did carry a body.
        return MF6LawElsewhere(law=law), idx
    if law in NO_BODY_LAWS:
        return MF6LawNoBody(law=law), idx
    if law == 1:
        return _parse_law1(lines, idx, probe)
    if law == 2:
        return _parse_law2(lines, idx, probe)
    if law == 5:
        return _parse_law5(lines, idx, probe)
    if law == 6:
        return _parse_law6(lines, idx)
    if law == 7:
        return _parse_law7(lines, idx, probe)
    raise ValueError(
        f"MF6/MT{mt} product {index} uses LAW={law}, which this parser has no "
        f"record layout for (known: {list(KNOWN_LAWS)} and any negative LAW). "
        f"A guessed length would swallow the products after it, so the section "
        f"is refused instead"
    )


def _pair_lines(n_pairs: int) -> int:
    """How many body lines ``n_pairs`` (x, y) points occupy, three per line."""
    return -(-n_pairs // 3)


def _parse_list(lines: List[str], idx: int) -> Tuple[dict, List[float], int]:
    """A LIST record: the CONT header, then its NW values."""
    header = parse_line(lines[idx])
    nw = int(header.get("C5") or 0)
    values, idx = parse_data_values(lines, idx + 1, nw)
    return header, values, idx


def _parse_law1(lines, idx, probe) -> Tuple[MF6LawContinuum, int]:
    """LAW=1: a TAB2 over incident energy, each node a LIST of E' and shape."""
    tab2_header, tab2_interp, idx = parse_tab2(lines, idx)
    probe.observe_interp(lines, idx, len(tab2_interp))
    ne = int(tab2_header.get("C6") or 0)

    body = MF6LawContinuum(
        lang=int(tab2_header.get("C3") or 0),
        lep=int(tab2_header.get("C4") or 0),
        tab2_interp=list(tab2_interp),
    )
    for _ in range(ne):
        if idx >= len(lines):
            break
        header, values, idx = _parse_list(lines, idx)
        probe.observe_values(lines, idx, len(values))
        body.incident_energies.append(header.get("C2"))
        body.nd.append(int(header.get("C3") or 0))
        body.na.append(int(header.get("C4") or 0))
        body.values.append(values)
    return body, idx


def _parse_law2(lines, idx, probe) -> Tuple[MF6LawTwoBody, int]:
    """LAW=2: a TAB2 over incident energy, each node an angular LIST."""
    tab2_header, tab2_interp, idx = parse_tab2(lines, idx)
    probe.observe_interp(lines, idx, len(tab2_interp))
    ne = int(tab2_header.get("C6") or 0)

    body = MF6LawTwoBody(tab2_interp=list(tab2_interp))
    for _ in range(ne):
        if idx >= len(lines):
            break
        header, values, idx = _parse_list(lines, idx)
        probe.observe_values(lines, idx, len(values))
        body.incident_energies.append(header.get("C2"))
        body.lang.append(int(header.get("C3") or 0))
        body.values.append(values)
    return body, idx


def _parse_law5(lines, idx, probe) -> Tuple[MF6LawChargedElastic, int]:
    """LAW=5: charged-particle elastic. SPI and LIDP ride on the TAB2 header."""
    tab2_header, tab2_interp, idx = parse_tab2(lines, idx)
    probe.observe_interp(lines, idx, len(tab2_interp))
    ne = int(tab2_header.get("C6") or 0)

    body = MF6LawChargedElastic(
        spi=tab2_header.get("C1"),
        lidp=int(tab2_header.get("C3") or 0),
        tab2_interp=list(tab2_interp),
    )
    for _ in range(ne):
        if idx >= len(lines):
            break
        header, values, idx = _parse_list(lines, idx)
        probe.observe_values(lines, idx, len(values))
        body.incident_energies.append(header.get("C2"))
        body.ltp.append(int(header.get("C3") or 0))
        body.nl_values.append(int(header.get("C6") or 0))
        body.values.append(values)
    return body, idx


def _parse_law6(lines, idx) -> Tuple[MF6LawPhaseSpace, int]:
    """LAW=6: one CONT record, and that is the whole distribution."""
    header = parse_line(lines[idx])
    return MF6LawPhaseSpace(
        apsx=header.get("C1"),
        npsx=int(header.get("C6") or 0),
    ), idx + 1


def _parse_law7(lines, idx, probe) -> Tuple[MF6LawLabAngleEnergy, int]:
    """LAW=7: TAB2 over E, then per E a TAB2 over mu, then per mu a TAB1."""
    tab2_header, tab2_interp, idx = parse_tab2(lines, idx)
    probe.observe_interp(lines, idx, len(tab2_interp))
    ne = int(tab2_header.get("C6") or 0)

    body = MF6LawLabAngleEnergy(tab2_interp=list(tab2_interp))
    for _ in range(ne):
        if idx >= len(lines):
            break
        mu_header, mu_interp, idx = parse_tab2(lines, idx)
        block = LawSevenEnergy(
            energy=mu_header.get("C2"),
            mu_interp=list(mu_interp),
        )
        for _ in range(int(mu_header.get("C6") or 0)):
            if idx >= len(lines):
                break
            header, interp, x_data, y_data, idx = parse_tab1(lines, idx)
            probe.observe_pairs(lines, idx, len(x_data))
            block.angles.append(LawSevenAngle(
                mu=header.get("C2"),
                interp=list(interp),
                e_out=list(x_data),
                f=list(y_data),
            ))
        body.blocks.append(block)
    return body, idx


def _check_consumed(idx: int, lines: List[str], mt: int) -> None:
    """Every record of the section must have been read.

    A section that parses without reaching its last line has mis-walked a law
    body earlier, and the leftover count is the only evidence of it — the values
    read are all plausible, and the section re-emits at the wrong length.
    """
    if idx != len(lines):
        raise ValueError(
            f"MF6/MT{mt}: parsed {idx} of {len(lines)} records; "
            f"{len(lines) - idx} left over, so the record walk is wrong"
        )
