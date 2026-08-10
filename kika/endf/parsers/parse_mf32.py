"""
Parser for MF32 — covariances of the resonance parameters given in File 2.

ENDF-102 §32. The walk is a straight descent: HEAD, then NIS isotope records,
then NER energy ranges each, and for each range one covariance body chosen by
(LRU, LRF, LCOMP). What makes it worth reading carefully is that the LCOMP flag
sits in a *different field* depending on LRF, and that the record following the
LCOMP flag depends on ISR and LRF together — a CONT for Breit-Wigner, a LIST for
Reich-Moore and for R-Matrix Limited.

The parser decodes control records and keeps the bulk as text; see
``kika.endf.classes.mf32.records`` for why. It also does not repair what it
reads: a field the manual draws as zero and a tape writes as something else is
carried through, because ``str(parse(s)) == s`` is this parser's gate.
"""
from typing import List, Optional, Tuple

from ..classes.mf import MF
from ..classes.mf32.mf32mt151 import (
    CovEnergyRange,
    CovIsotope,
    IntgMatrix,
    LCOMP0Body,
    LCOMP1Body,
    LCOMP1RMLBody,
    LCOMP2Body,
    LCOMP2RMLBody,
    MF32MT151,
    RMLSpinGroup,
    ScatteringRadiusCovariance,
    UnresolvedBody,
)
from ..classes.mf32.records import DATA_WIDTH, Record
from ..utils import group_lines_by_mt_with_positions, parse_endf_id, parse_line
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


# ----------------------------------------------------------------------
# Scattering radius uncertainty (ISR=1)
# ----------------------------------------------------------------------

def _parse_dap(lines: List[str], idx: int, lrf: int) -> Tuple[Record, int]:
    """The scattering-radius uncertainty record, whose shape depends on LRF.

    §32.2.1 and §32.2.3.1 give a bare CONT carrying DAP in C2 for the
    Breit-Wigner formalisms; §32.2.2.2 and §32.2.3.2 give a LIST of MLS values
    for Reich-Moore, one per L value plus a default; §32.2.2.4 and §32.2.3.3
    give a LIST of NJCH values for R-Matrix Limited, one per channel. Reading
    the Breit-Wigner form as a LIST would swallow the record that follows it.
    """
    return Record.read(lines, idx, with_body=lrf in (3, 7))


# ----------------------------------------------------------------------
# The five covariance bodies
# ----------------------------------------------------------------------

def _parse_lcomp0(lines: List[str], idx: int, control: Record,
                  lrf: int) -> Tuple[LCOMP0Body, int]:
    """§32.2.1 — NLS LIST records, one per L value, 18 numbers a resonance."""
    body = LCOMP0Body(control=control)
    if control.n2:
        body.dap, idx = _parse_dap(lines, idx, lrf)
    for _ in range(control.n1):
        if idx >= len(lines):
            break
        block, idx = Record.read(lines, idx, with_body=True)
        body.l_blocks.append(block)
    return body, idx


def _parse_lcomp1(lines: List[str], idx: int, control: Record,
                  lrf: int) -> Tuple[LCOMP1Body, int]:
    """§32.2.2 — a CONT giving NSRS and NLRS, then that many blocks of each."""
    body = LCOMP1Body(control=control)
    if control.n2:
        body.dap, idx = _parse_dap(lines, idx, lrf)
    body.counts, idx = Record.read(lines, idx)
    for _ in range(body.counts.n1):
        if idx >= len(lines):
            break
        block, idx = Record.read(lines, idx, with_body=True)
        body.short_range.append(block)
    for _ in range(body.counts.n2):
        if idx >= len(lines):
            break
        block, idx = Record.read(lines, idx, with_body=True)
        body.long_range.append(block)
    return body, idx


def _parse_lcomp1_rml(lines: List[str], idx: int,
                      control: Record) -> Tuple[LCOMP1RMLBody, int]:
    """§32.2.2.4 — LCOMP=1 for LRF=7. No long-range blocks are allowed.

    A short-range block cannot state its size in one number: it opens with a
    CONT giving NJSX, carries a LIST per spin group, and closes with the
    covariance triangle. NLRS is read and required to be zero rather than
    assumed, so a tape that breaks the rule is reported instead of mis-parsed.
    """
    body = LCOMP1RMLBody(control=control)
    if control.n2:
        body.dap, idx = _parse_dap(lines, idx, 7)
    body.counts, idx = Record.read(lines, idx)
    if body.counts.n2:
        raise ValueError(
            f"MF32 LCOMP=1 LRF=7 declares NLRS={body.counts.n2}; §32.2.2.4 "
            "allows no long-range subsections for R-Matrix Limited"
        )
    for _ in range(body.counts.n1):
        if idx >= len(lines):
            break
        njsx_record, idx = Record.read(lines, idx)
        block: List[Record] = [njsx_record]
        for _group in range(njsx_record.l1):
            if idx >= len(lines):
                break
            group_record, idx = Record.read(lines, idx, with_body=True)
            block.append(group_record)
        matrix, idx = Record.read(lines, idx, with_body=True)
        block.append(matrix)
        body.blocks.append(block)
    return body, idx


def _parse_intg_block(lines: List[str], idx: int) -> Tuple[Optional[IntgMatrix], int]:
    """The INTG control record and the NM records after it, if present.

    An LCOMP=2 body may stop after the uncertainties, which is a diagonal
    covariance and is what Na-23 does. The caller cannot tell in advance, so the
    absence of any further line is the signal.
    """
    if idx >= len(lines):
        return None, idx
    control, idx = Record.read(lines, idx)
    matrix = IntgMatrix(
        ndigit=control.l1, nnn=control.l2, nm=control.n1, n2=control.n2,
        control_raw=control.raw,
        lines=[line[:DATA_WIDTH] for line in lines[idx:idx + control.n1]],
    )
    return matrix, idx + control.n1


def _parse_lcomp2(lines: List[str], idx: int, control: Record,
                  lrf: int) -> Tuple[LCOMP2Body, int]:
    """§32.2.3.1-2 — one LIST of parameters-plus-uncertainties, then INTG.

    *lrf* has to come from the range record rather than from *control*: LRF=3
    puts LAD in the field where LRF=1 and 2 put a zero, so this record cannot
    say which formalism it belongs to, and the DAP record's shape depends on it.
    """
    body = LCOMP2Body(control=control)
    if control.n2:
        body.dap, idx = _parse_dap(lines, idx, lrf)
    body.parameters, idx = Record.read(lines, idx, with_body=True)
    body.correlations, idx = _parse_intg_block(lines, idx)
    return body, idx


def _parse_lcomp2_rml(lines: List[str], idx: int,
                      control: Record) -> Tuple[LCOMP2RMLBody, int]:
    """§32.2.3.3 — particle pairs, then NJS spin groups, then INTG.

    The number of spin groups to read comes from NJS on the control record, not
    from NJSX on the particle-pair record: NJSX counts the groups *represented
    in the covariance matrix*, and Cl-35 has NJS=8 against NJSX=7. Reading NJSX
    groups there would leave one group's two LIST records to be mistaken for
    the INTG control record.
    """
    body = LCOMP2RMLBody(control=control)
    if control.n2:
        body.dap, idx = _parse_dap(lines, idx, 7)
    body.particle_pairs, idx = Record.read(lines, idx, with_body=True)
    for _ in range(control.n1):
        if idx >= len(lines):
            break
        channels, idx = Record.read(lines, idx, with_body=True)
        resonances, idx = Record.read(lines, idx, with_body=True)
        body.spin_groups.append(
            RMLSpinGroup(channels=channels, resonances=resonances))
    body.correlations, idx = _parse_intg_block(lines, idx)
    return body, idx


def _parse_unresolved(lines: List[str], idx: int,
                      control: Record) -> Tuple[UnresolvedBody, int]:
    """§32.2.4 — NLS LIST records of average parameters, then one triangle."""
    body = UnresolvedBody(control=control)
    for _ in range(control.n1):
        if idx >= len(lines):
            break
        block, idx = Record.read(lines, idx, with_body=True)
        body.l_blocks.append(block)
    if idx < len(lines):
        body.matrix, idx = Record.read(lines, idx, with_body=True)
    return body, idx


# ----------------------------------------------------------------------
# The descent
# ----------------------------------------------------------------------

def _parse_energy_range(lines: List[str], idx: int) -> Tuple[CovEnergyRange, int]:
    """One range: its CONT, an optional NRO preamble, and one covariance body."""
    control, idx = Record.read(lines, idx)
    energy_range = CovEnergyRange(
        el=control.c1, eh=control.c2, lru=control.l1, lrf=control.l2,
        nro=control.n1, naps=control.n2, control=control,
    )
    lrf = energy_range.lrf

    if energy_range.nro:
        # §32.2: NI subsubsections for the energy-dependent radius, in File 33's
        # own format. Untested — no tape on this machine has one.
        radius_control, idx = Record.read(lines, idx)
        radius = ScatteringRadiusCovariance(control=radius_control)
        for _ in range(radius_control.n2):
            block, idx = Record.read(lines, idx, with_body=True)
            radius.subsections.append(block)
        energy_range.radius = radius

    body_control, idx = Record.read(lines, idx)

    if energy_range.lru == 2:
        energy_range.body, idx = _parse_unresolved(lines, idx, body_control)
        return energy_range, idx

    lcomp = body_control.l2
    if lrf == 7:
        if lcomp == 1:
            energy_range.body, idx = _parse_lcomp1_rml(lines, idx, body_control)
        elif lcomp == 2:
            energy_range.body, idx = _parse_lcomp2_rml(lines, idx, body_control)
        else:
            raise ValueError(
                f"MF32 LRF=7 declares LCOMP={lcomp}; §32.2 allows 1 or 2"
            )
    elif lcomp == 0:
        energy_range.body, idx = _parse_lcomp0(lines, idx, body_control, lrf)
    elif lcomp == 1:
        energy_range.body, idx = _parse_lcomp1(lines, idx, body_control, lrf)
    elif lcomp == 2:
        energy_range.body, idx = _parse_lcomp2(lines, idx, body_control, lrf)
    else:
        raise ValueError(
            f"MF32 declares LCOMP={lcomp} for LRF={lrf}; §32.2 allows 0, 1 or 2"
        )
    return energy_range, idx


def parse_mf32_mt151(lines: List[str], mt: int = 151) -> MF32MT151:
    """Parse the MF32/MT=151 section of a tape."""
    if not lines:
        raise ValueError("Empty MF32 MT151 section")

    head, idx = Record.read(lines, 0)
    mat, _, _ = parse_endf_id(lines[0])
    section = MF32MT151(
        number=mt, _za=head.c1, _awr=head.c2, _nis=head.n1,
        _mat=mat, _mf=32, head=head,
    )

    for _ in range(head.n1):
        if idx >= len(lines):
            break
        isotope_control, idx = Record.read(lines, idx)
        isotope = CovIsotope(
            zai=isotope_control.c1, abn=isotope_control.c2,
            lfw=isotope_control.l2, control=isotope_control,
        )
        for _range in range(isotope_control.n1):
            if idx >= len(lines):
                break
            energy_range, idx = _parse_energy_range(lines, idx)
            isotope.energy_ranges.append(energy_range)
        section.isotopes.append(isotope)

    return section


def parse_mf32(lines: List[str]) -> MF:
    """Parse all MT sections within an MF32 block. Only MT=151 is defined."""
    logger.debug(f"Parsing MF32 with {len(lines)} lines")
    mf = MF(number=32)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            section = parse_mf32_mt151(mt_lines, mt)
            mf.add_section(section)
            if mt in line_counts:
                section.num_lines = line_counts[mt]
        except Exception as exc:
            logger.warning(f"Error parsing MT{mt} in MF32: {exc}")

    return mf
