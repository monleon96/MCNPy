"""Parser for MF7 (thermal neutron scattering law).

Follows :mod:`~kika.endf.parsers.parse_mf5` in shape: a file-level function that
groups by MT and logs per-section failures rather than losing the whole file to
one bad section, and per-MT functions under it.

**File 7 defines exactly three sections** — MT2, MT4 and MT451 — so an MT this
module does not recognise is refused rather than guessed at. Unlike MF5, where
an unknown LF costs one subsection, MF7 has no repeating outer structure to
resynchronise on: a wrong record count runs to the end of the section.

**Two counts are not written in the file and have to be derived**, and both are
checked rather than trusted:

* how many effective-temperature tables follow MT4's β loop — derived from the
  B array by :attr:`~kika.endf.classes.mf7.inelastic.MF7MT4.expected_teff_records`;
* that every record of the section was consumed. A section that parses without
  reaching its last line has mis-walked something earlier, and the leftover
  count is the only evidence of it.
"""
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf7.base import TemperatureTable
from ..classes.mf7.composition import (
    VALUES_PER_ISOTOPE,
    ElementComposition,
    MF7MT451,
    TSLIsotope,
)
from ..classes.mf7.elastic import (
    COHERENT_LTHR,
    INCOHERENT_LTHR,
    CoherentElastic,
    IncoherentElastic,
    MF7MT2,
)
from ..classes.mf7.inelastic import (
    BetaBlock,
    EffectiveTemperature,
    MF7MT4,
)
from ..utils import (
    PaddingProbe as _PaddingProbe,
    group_lines_by_mt_with_positions,
    parse_data_values,
    parse_line,
    parse_tab1,
    parse_tab2,
)
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)

#: The only MT numbers ENDF-6 defines for File 7.
KNOWN_MT = (2, 4, 451)

def parse_mf7(lines: List[str]) -> MF:
    """Parse MF7 (thermal scattering) into an :class:`MF` of MF7 sections."""
    logger.debug(f"Parsing MF7 with {len(lines)} lines")
    mf = MF(number=7)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf7_mt(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
            logger.debug(f"Successfully parsed MT{mt}")
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF7: {exc}")

    return mf


def parse_mf7_mt(lines: List[str], mt: int):
    """Parse one MT section of MF7, dispatching on the MT number."""
    if mt == 2:
        return parse_mf7_mt2(lines)
    if mt == 4:
        return parse_mf7_mt4(lines)
    if mt == 451:
        return parse_mf7_mt451(lines)
    raise ValueError(
        f"MF7/MT{mt} is not a section ENDF-6 defines for File 7 "
        f"(known: {list(KNOWN_MT)})"
    )


# ---------------------------------------------------------------------------
# Shared records
# ---------------------------------------------------------------------------

def _head(lines: List[str]) -> dict:
    return parse_line(lines[0])


def _check_consumed(idx: int, lines: List[str], mt: int) -> None:
    """Every record of the section must have been read."""
    if idx != len(lines):
        raise ValueError(
            f"MF7/MT{mt}: parsed {idx} of {len(lines)} records; "
            f"{len(lines) - idx} left over, so the record walk is wrong"
        )


def _parse_temperature_table(lines: List[str], idx: int, mt: int, what: str,
                             probe: "_PaddingProbe"
                             ) -> Tuple[TemperatureTable, int]:
    """A TAB1 at T₀ followed by ``LT`` LIST records sharing its x grid.

    The LIST records carry values only — re-reading them as x/y pairs would
    consume twice the records and silently swallow whatever follows.
    """
    header, interp, x_data, y_data, idx = parse_tab1(lines, idx)
    probe.observe_pairs(lines, idx, len(x_data))
    lt = int(header.get("C3") or 0)

    table = TemperatureTable(
        t0=header.get("C1"),
        lt=lt,
        interp=list(interp),
        x=list(x_data),
        values=[list(y_data)],
        temperatures=[header.get("C1")],
        li=[],
    )

    for step in range(lt):
        if idx >= len(lines):
            raise ValueError(
                f"MF7/MT{mt} {what}: LT={lt} but the section ended after "
                f"{step} extra temperature(s)"
            )
        list_header = parse_line(lines[idx])
        n_values = int(list_header.get("C5") or 0)
        if n_values != len(table.x):
            raise ValueError(
                f"MF7/MT{mt} {what}: temperature {list_header.get('C1')} K "
                f"lists {n_values} values for a grid of {len(table.x)}"
            )
        values, idx = parse_data_values(lines, idx + 1, n_values)
        probe.observe_values(lines, idx, n_values)
        table.temperatures.append(list_header.get("C1"))
        table.li.append(int(list_header.get("C3") or 0))
        table.values.append(list(values))

    return table, idx


# ---------------------------------------------------------------------------
# MT2 — elastic
# ---------------------------------------------------------------------------

def parse_mf7_mt2(lines: List[str]) -> MF7MT2:
    """Parse MF7/MT2, whose LTHR selects coherent, incoherent or both."""
    head = _head(lines)
    lthr = int(head.get("C3") or 0)

    if lthr not in COHERENT_LTHR and lthr not in INCOHERENT_LTHR:
        raise ValueError(
            f"MF7/MT2 has LTHR={lthr}; ENDF-6 defines 1 (coherent), "
            f"2 (incoherent) and 3 (both)"
        )

    section = MF7MT2(
        number=2,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _mat=head.get("MAT"),
        _lthr=lthr,
    )

    probe = _PaddingProbe()
    idx = 1
    if lthr in COHERENT_LTHR:
        table, idx = _parse_temperature_table(
            lines, idx, 2, "coherent elastic", probe)
        section.coherent = CoherentElastic(table=table)

    if lthr in INCOHERENT_LTHR:
        header, interp, x_data, y_data, idx = parse_tab1(lines, idx)
        probe.observe_pairs(lines, idx, len(x_data))
        section.incoherent = IncoherentElastic(
            sb=header.get("C1"),
            interp=list(interp),
            temperatures=list(x_data),
            w=list(y_data),
        )

    section.pad = probe.resolve()
    _check_consumed(idx, lines, 2)
    return section


# ---------------------------------------------------------------------------
# MT4 — incoherent inelastic
# ---------------------------------------------------------------------------

def parse_mf7_mt4(lines: List[str]) -> MF7MT4:
    """Parse MF7/MT4: the B array, the β loop, then the Teff tables."""
    head = _head(lines)

    section = MF7MT4(
        number=4,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _mat=head.get("MAT"),
        _lat=int(head.get("C4") or 0),
        _lasym=int(head.get("C5") or 0),
    )

    b_header = parse_line(lines[1])
    section._lln = int(b_header.get("C3") or 0)
    ni = int(b_header.get("C5") or 0)
    section._ns = int(b_header.get("C6") or 0)

    # Checked *before* the B array is read, not after. NI is what says how many
    # records to consume, so a wrong one has already put the cursor in the wrong
    # place by the time the values are in hand.
    expected_ni = VALUES_PER_ISOTOPE * (section.ns + 1)
    if ni != expected_ni:
        raise ValueError(
            f"MF7/MT4 declares NI={ni} with NS={section.ns}; ENDF-6 requires "
            f"NI = 6(NS+1) = {expected_ni}"
        )

    probe = _PaddingProbe()
    b_values, idx = parse_data_values(lines, 2, ni)
    probe.observe_values(lines, idx, ni)
    section.b = list(b_values)

    if section.has_tabulated_s:
        tab2_header, beta_interp, idx = parse_tab2(lines, idx)
        section.beta_interp = list(beta_interp)
        nb = int(tab2_header.get("C6") or 0)

        for index in range(nb):
            if idx >= len(lines):
                raise ValueError(
                    f"MF7/MT4 declares NB={nb} betas but the section ended "
                    f"after {index}"
                )
            beta = parse_line(lines[idx]).get("C2")
            table, idx = _parse_temperature_table(
                lines, idx, 4, f"beta {beta}", probe)
            section.blocks.append(BetaBlock(beta=beta, table=table))

        _check_shared_temperature_grid(section)

    for _ in range(section.expected_teff_records):
        if idx >= len(lines):
            raise ValueError(
                f"MF7/MT4 needs {section.expected_teff_records} "
                f"effective-temperature table(s) for its B array but the "
                f"section ended after {len(section.teff)}"
            )
        header, interp, x_data, y_data, idx = parse_tab1(lines, idx)
        probe.observe_pairs(lines, idx, len(x_data))
        section.teff.append(EffectiveTemperature(
            interp=list(interp),
            temperatures=list(x_data),
            teff=list(y_data),
        ))

    section.pad = probe.resolve()
    _check_consumed(idx, lines, 4)
    return section


def _check_shared_temperature_grid(section: MF7MT4) -> None:
    """Every β must be tabulated at the same temperatures.

    ENDF-6 requires it and :attr:`MF7MT4.temperatures` reports the first block's
    grid as if it were the section's, so this is what makes that true rather
    than assumed. Only the first offending β is named — on a 1325-β file, a
    list of all of them is not a better error message.
    """
    if not section.blocks:
        return
    reference = section.blocks[0].temperatures
    for block in section.blocks[1:]:
        if block.temperatures != reference:
            raise ValueError(
                f"MF7/MT4: beta {block.beta} is tabulated at "
                f"{len(block.temperatures)} temperature(s) but beta "
                f"{section.blocks[0].beta} at {len(reference)}"
            )


# ---------------------------------------------------------------------------
# MT451 — composition
# ---------------------------------------------------------------------------

def parse_mf7_mt451(lines: List[str]) -> MF7MT451:
    """Parse MF7/MT451: one LIST of isotopes per element of the compound."""
    head = _head(lines)
    na = int(head.get("C3") or 0)

    section = MF7MT451(
        number=451,
        _za=head.get("C1"),
        _awr=head.get("C2"),
        _mat=head.get("MAT"),
    )

    probe = _PaddingProbe()
    idx = 1
    for index in range(na):
        if idx >= len(lines):
            raise ValueError(
                f"MF7/MT451 declares NA={na} element(s) but the section ended "
                f"after {index}"
            )
        list_header = parse_line(lines[idx])
        nw = int(list_header.get("C5") or 0)
        values, idx = parse_data_values(lines, idx + 1, nw)
        probe.observe_values(lines, idx, nw)

        if nw % VALUES_PER_ISOTOPE:
            raise ValueError(
                f"MF7/MT451 element {index} has NW={nw}, not a multiple of "
                f"{VALUES_PER_ISOTOPE}"
            )

        element = ElementComposition(nas=int(list_header.get("C3") or 0))
        for start in range(0, nw, VALUES_PER_ISOTOPE):
            zai, lis, fraction, awr, sigma_free, _ = values[
                start:start + VALUES_PER_ISOTOPE]
            element.isotopes.append(TSLIsotope(
                zai=int(round(zai or 0)),
                lis=int(lis or 0),
                atom_fraction=fraction,
                awr=awr,
                sigma_free=sigma_free,
            ))
        section.elements.append(element)

    section.pad = probe.resolve()
    _check_consumed(idx, lines, 451)
    return section
