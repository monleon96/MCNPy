"""
Parser for MF34 (Angular Distribution Covariances) sections in ENDF files.

MF34 stores covariance data for the Legendre coefficients of secondary
particle angular distributions described in MF4.  Each MT section contains
NMT1 subsections (one per cross-correlated reaction MT1); each subsection
contains a set of sub-subsections, one per (L, L1) Legendre-index pair; and
each sub-subsection holds NI LIST records (LB=0..2, 5, or 6).
"""
from typing import List, Tuple

from ..classes.mf import MF
from ..classes.mf34.mf34 import MF34MT, Subsection, SubSubsection, SubSubsectionRecord
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


# ---- helpers ---------------------------------------------------------------


def _l_min_for_ltt(ltt) -> int:
    """Lowest Legendre index L allowed by the given LTT representation.

    LTT=2 (coefficients start at a_0) and LTT=3 ("if either L or L1=0 anywhere
    in the Section", ENDF-6 manual Sec. 34.2) both admit a_0; every other value
    starts at a_1.
    """
    return 0 if int(ltt or 1) in (2, 3) else 1


def _expected_subsubsection_count(nl, nl1, mt: int, mt1, ltt) -> int:
    """Number of (L, L1) sub-subsections per ENDF-6 MF34 conventions.

    For MT1 == MT only the upper triangle is stored (count = NL*(NL+1)/2);
    otherwise every (L, L1) pair is stored (count = NL * NL1).  ENDF-6 manual
    Sec. 34.2, which defines NL as the *number* of Legendre coefficients
    carried -- "NL Number of Legendre coefficients for which covariance data
    are given ... (The first coefficient is a0 if LTT=3, a1 if LTT=1)" -- not
    the highest index.  With l_min = 1 the two readings coincide, which is why
    LTT=1 files are unaffected by this distinction; they differ only when a_0
    is present, where the count is max_order + 1.

    ``ltt`` is retained in the signature for callers and for symmetry with
    :func:`_l_min_for_ltt`; the count itself depends only on NL/NL1.
    """
    n = max(0, int(nl or 0))
    n1 = max(0, int(nl1 or 0))
    if mt1 == mt:
        return n * (n + 1) // 2
    return n * n1


def _read_floats(lines: List[str], idx: int, n: int) -> Tuple[List[float], int]:
    """Read N scalar floats from the body of a LIST record (6 per line)."""
    values: List[float] = []
    while len(values) < n and idx < len(lines):
        ld = parse_line(lines[idx])
        idx += 1
        for i in range(1, 7):
            if len(values) >= n:
                break
            v = ld.get(f"C{i}")
            if v is not None:
                values.append(v)
    return values, idx


# ---- LIST record dispatch --------------------------------------------------


def _parse_list_record(lines: List[str], idx: int) -> Tuple[SubSubsectionRecord, int]:
    """Parse a single LIST record (header + body); return record and next index."""
    header = parse_line(lines[idx])
    idx += 1

    c3 = header.get("C3")
    lb = header.get("C4")
    nt = int(header.get("C5") or 0)
    c6 = header.get("C6")

    record = SubSubsectionRecord(lb=lb, nt=nt)

    if lb in (0, 1, 2):
        # Header for LB=0..2: C3=LT, C5=NT, C6=NP
        record.lt = c3
        record.np = c6

        all_values, idx = _read_floats(lines, idx, nt)
        record.raw_list_values = all_values[:]

        # Decode the common LT==0 case (single (E,F) table); preserve raw
        # floats for the rare LT>0 two-table form so it round-trips exactly.
        if (record.lt or 0) == 0:
            ne = int(record.np or 0)
            record.ne = ne
            for i in range(ne):
                record.e_table_k.append(all_values[2 * i])
                record.f_table_k.append(all_values[2 * i + 1])
        else:
            logger.info(
                f"LB={lb} with LT={record.lt} (>0): preserving raw LIST values "
                f"for round-trip; per-table decoding not implemented."
            )

    elif lb == 5:
        # Header for LB=5: C3=LS, C5=NT, C6=NE
        ls = c3
        ne = int(c6 or 0)
        if ne < 2:
            raise ValueError(f"LB=5 requires NE >= 2 to define intervals, got NE={ne}")
        m = ne - 1
        if ls == 1:
            n_matrix = m * (m + 1) // 2
        elif ls == 0:
            n_matrix = m * m
        else:
            raise ValueError(f"LB=5 unsupported LS value: {ls}; expected 0 or 1")

        record.ls = ls
        record.ne = ne

        if nt != ne + n_matrix:
            logger.warning(
                f"LB=5 LS={ls}: NT={nt} != NE + matrix size ({ne + n_matrix}); "
                f"reading NT values"
            )
        all_values, idx = _read_floats(lines, idx, nt)
        record.energies = all_values[:ne]
        record.matrix = all_values[ne:]

    elif lb == 6:
        # Header for LB=6: C5=NT, C6=NER (number of row energies)
        ner = int(c6 or 0)
        if ner < 2:
            raise ValueError(f"LB=6 requires NER >= 2, got NER={ner}")
        if (nt - 1) % ner != 0:
            raise ValueError(
                f"LB=6 invalid NT={nt} for NER={ner}: (NT-1)={nt - 1} "
                f"must be divisible by NER"
            )
        nec = (nt - 1) // ner
        if nec < 2:
            raise ValueError(f"LB=6 requires NEC >= 2, got NEC={nec}")

        record.ne = ner

        all_values, idx = _read_floats(lines, idx, nt)
        record.row_energies = all_values[:ner]
        record.col_energies = all_values[ner:ner + nec]
        record.rect_matrix = all_values[ner + nec:]

        expected = (ner - 1) * (nec - 1)
        if len(record.rect_matrix) != expected:
            raise ValueError(
                f"LB=6 matrix size mismatch: expected {expected} elements for "
                f"(NER-1)*(NEC-1)=({ner}-1)*({nec}-1), got {len(record.rect_matrix)}"
            )

    else:
        raise ValueError(f"Unsupported LB value: {lb}")

    return record, idx


# ---- sub-subsection / subsection / MT --------------------------------------


def _parse_subsubsection(lines: List[str], idx: int) -> Tuple[SubSubsection, int]:
    """Parse a sub-subsection (header + NI LIST records)."""
    header = parse_line(lines[idx])
    idx += 1

    sub_subsec = SubSubsection(
        l=header.get("C3"),
        l1=header.get("C4"),
        lct=header.get("C5"),
        ni=int(header.get("C6") or 0),
    )

    for _ in range(sub_subsec.ni):
        if idx >= len(lines):
            break
        record, idx = _parse_list_record(lines, idx)
        sub_subsec.records.append(record)

    return sub_subsec, idx


def parse_mf34_mt(lines: List[str], mt: int) -> MF34MT:
    """Parse one MT section of MF34."""
    if len(lines) < 2:
        raise ValueError(f"Insufficient data for MT{mt} section in MF34")

    # MT-section header: ZA, AWR, 0, LTT, 0, NMT1
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    ltt = header.get("C4")
    nmt1 = int(header.get("C6") or 0)
    mat, _, _ = parse_endf_id(lines[0])

    logger.debug(
        f"Parsing MF34 MT{mt}: ZA={za}, AWR={awr}, LTT={ltt}, NMT1={nmt1}"
    )

    mt_section = MF34MT(
        number=mt, _za=za, _awr=awr, _ltt=ltt,
        _nmt1=nmt1, _mat=mat, _mf=34,
    )

    idx = 1
    for sub_idx in range(nmt1):
        if idx >= len(lines):
            break

        # Subsection header: 0, 0, MAT1, MT1, NL, NL1
        sub_header = parse_line(lines[idx])
        idx += 1
        mat1 = sub_header.get("C3")
        mt1 = sub_header.get("C4")
        nl = sub_header.get("C5")
        nl1 = sub_header.get("C6")

        subsection = Subsection(mt1=mt1, nl=nl, nl1=nl1, mat1=mat1)
        nss = _expected_subsubsection_count(nl, nl1, mt, mt1, ltt)
        logger.debug(
            f"Subsection {sub_idx + 1}/{nmt1}: MAT1={mat1}, MT1={mt1}, "
            f"NL={nl}, NL1={nl1} -> {nss} sub-subsections"
        )

        for _ in range(nss):
            if idx >= len(lines):
                break
            sub_subsec, idx = _parse_subsubsection(lines, idx)
            subsection.sub_subsections.append(sub_subsec)

        mt_section.add_subsection(subsection)

    return mt_section


def parse_mf34(lines: List[str]) -> MF:
    """Parse all MT sections within an MF34 block."""
    logger.debug(f"Parsing MF34 with {len(lines)} lines")
    mf = MF(number=34)
    mf.num_lines = len(lines)

    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    logger.debug(f"Found MT sections: {list(mt_groups.keys())}")

    for mt, mt_lines in mt_groups.items():
        if mt == 0:
            continue
        try:
            mt_section = parse_mf34_mt(mt_lines, mt)
            mf.add_section(mt_section)
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
        except Exception as e:
            logger.warning(f"Error parsing MT{mt} in MF34: {e}")

    return mf
