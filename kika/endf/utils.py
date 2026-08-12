"""
Utility functions for ENDF file parsing and writing.

Contains helper functions for handling the specific formatting requirements of ENDF files.
"""
import re
from typing import Dict,Union, List, Optional, Tuple, Sequence, Any
from .classes.mt import MT
from .classes.mf1.mf1mt451 import MF1MT451
from .classes.mf import MF
import numpy as np
import math
from numpy.typing import ArrayLike

#: The fixed-width record grammar, re-exported from :mod:`kika._records`.
#:
#: It moved out of this module in phase 4's P4 because it is not ENDF's:
#: COVERX/COVFIL/BOXER share the same 11-column convention, and
#: ``kika/cov/parse_covmat.py`` importing it from here was the one import-time
#: format dependency in ``kika/cov``. These names stay importable from
#: ``kika.endf.utils`` -- ``test_library_export_surface.py`` pins
#: ``parse_endf_id`` here as public surface, and every parser, writer and MF
#: class in this package reaches them by this path.
from .._records import (
    ENDF_FORMAT_BLANK,
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    ENDF_FORMAT_INT_ZERO,
    ENDF_FORMAT_PRESERVE,
    MAX_SEQUENCE_NUMBER,
    format_endf_data_line,
    format_endf_id_columns,
    format_endf_number,
    parse_endf_id,
    parse_line,
    parse_number,
)




#: Field types of a termination record's data part.
#:
#: ENDF-6 types the first two fields of SEND/FEND/MEND/TEND as floats (C1, C2)
#: and the remaining four as integers, so a terminator reads
#: ``" 0.000000+0 0.000000+0          0          0          0          0"``.
#: Nine emitters used to build this by hand with ``[ENDF_FORMAT_INT] * 6``,
#: rendering the two float fields as right-aligned integer zeros.
_TERMINATION_FORMATS = [
    ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT,
]

#: Sequence number ENDF-6 requires on a SEND record, verbatim.
SEND_SEQUENCE_NUMBER = 99999


def format_endf_send_record(mat: int, mf: int) -> str:
    """The SEND record closing a section: MT=0, sequence number 99999."""
    return format_endf_data_line(
        [0.0, 0.0, 0, 0, 0, 0], mat, mf, 0, SEND_SEQUENCE_NUMBER,
        formats=_TERMINATION_FORMATS,
    )


def format_endf_fend_record(mat: int) -> str:
    """The FEND record closing a file: MF=0, MT=0, sequence number 0."""
    return format_endf_data_line(
        [0.0, 0.0, 0, 0, 0, 0], mat, 0, 0, 0, formats=_TERMINATION_FORMATS,
    )


def format_endf_mend_record() -> str:
    """The MEND record closing a material: MAT=0."""
    return format_endf_data_line(
        [0.0, 0.0, 0, 0, 0, 0], 0, 0, 0, 0, formats=_TERMINATION_FORMATS,
    )


def format_endf_tend_record() -> str:
    """The TEND record closing a tape: MAT=-1."""
    return format_endf_data_line(
        [0.0, 0.0, 0, 0, 0, 0], -1, 0, 0, 0, formats=_TERMINATION_FORMATS,
    )




def group_lines_by_mt_with_positions(lines: List[str]) -> Tuple[Dict[int, List[str]], Dict[int, int]]:
    """
    Group lines by MT numbers and track their line counts.
    
    Args:
        lines: List of string lines
        
    Returns:
        Tuple of:
            - Dictionary mapping MT numbers to lists of lines
            - Dictionary mapping MT numbers to line counts
    """
    result: Dict[int, List[str]] = {}
    line_counts: Dict[int, int] = {}
    current_mt = None
    current_lines: List[str] = []
    
    for i, line in enumerate(lines):
        # Parse MT number from the line
        try:
            _, _, mt = parse_endf_id(line)
            
            # Skip MT=0 as a data section (it's a marker)
            if mt == 0:
                # If we were collecting a section, finalize it before the MT=0 marker
                if current_mt is not None and current_lines:
                    result[current_mt] = current_lines
                    line_counts[current_mt] = len(current_lines)
                    current_mt = None
                    current_lines = []
                continue
            
            # Handle section changes
            if current_mt is None:
                # Start a new section
                current_mt = mt
                current_lines = [line]
            elif mt != current_mt:
                # Complete the previous section
                result[current_mt] = current_lines
                line_counts[current_mt] = len(current_lines)
                
                # Start a new section
                current_mt = mt
                current_lines = [line]
            else:
                # Continue current section
                current_lines.append(line)
        except Exception:
            # If we can't parse the line, just add it to the current section if we have one
            if current_mt is not None:
                current_lines.append(line)
    
    # Add the last section if needed
    if current_mt is not None and current_lines:
        result[current_mt] = current_lines
        line_counts[current_mt] = len(current_lines)
    
    return result, line_counts


# Interpolation scheme codes and their meanings based on ENDF format specification
INTERPOLATION_SCHEMES = {
    1: "constant in x (histogram)",
    2: "linear-linear",
    3: "linear-log",
    4: "log-linear",
    5: "log-log",
    6: "special one-dimensional interpolation for charged-particle cross sections",
    11: "method of corresponding points (interpolation law 1)",
    12: "method of corresponding points (interpolation law 2)",
    13: "method of corresponding points (interpolation law 3)",
    14: "method of corresponding points (interpolation law 4)",
    15: "method of corresponding points (interpolation law 5)",
    21: "unit base interpolation (interpolation law 1)",
    22: "unit base interpolation (interpolation law 2)",
    23: "unit base interpolation (interpolation law 3)",
    24: "unit base interpolation (interpolation law 4)",
    25: "unit base interpolation (interpolation law 5)"
}

def get_interpolation_scheme_name(scheme_code):
    """
    Get the descriptive name of an interpolation scheme based on its code.
    
    Parameters:
        scheme_code (int): The interpolation scheme code (INT in ENDF format)
        
    Returns:
        str: The descriptive name of the interpolation scheme
    """
    return INTERPOLATION_SCHEMES.get(scheme_code, f"Unknown scheme ({scheme_code})")

def describe_interpolation_region(nbt, int_code):
    """
    Generate a descriptive string for an interpolation region.
    
    Parameters:
        nbt (int): The NBT value indicating the upper bound of points for this interpolation
        int_code (int): The interpolation scheme code (INT)
        
    Returns:
        str: A descriptive string for this interpolation region
    """
    scheme_name = get_interpolation_scheme_name(int_code)
    return f"Points up to {nbt} use {scheme_name}"



# Moved to kika/processing/interpolation.py by phase 2 of the GNDS roadmap:
# interpolation law codes 1-5 are shared with GNDS (§3.4.4) and are not
# ENDF-specific. These are *live* re-exports -- eight call sites in kika/endf
# import interpolate_1d_endf -- not shims awaiting deletion. The private names
# come along because kika/endf/tests reaches for them.
from kika.processing.interpolation import (
    interpolate_1d as interpolate_1d_endf,
    _regionize,
    _base_int_code,
    _interp_pair,
    _interp_pair_vec,
)



def parse_interp_pairs(lines, start, nr):
    """
    Read NR interpolation (NBT, INT) pairs from ENDF lines.

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the first line containing pairs.
    nr : int
        Number of (NBT, INT) pairs to read.

    Returns
    -------
    pairs : list of tuple(int, int)
    next_idx : int
        Index of the next line after all pairs have been read.
    """
    pairs = []
    idx = start
    remaining = nr
    while remaining > 0 and idx < len(lines):
        ld = parse_line(lines[idx])
        n_this_line = min(3, remaining)
        for i in range(n_this_line):
            nbt = ld.get(f"C{i * 2 + 1}")
            interp = ld.get(f"C{i * 2 + 2}")
            if nbt is not None and interp is not None:
                pairs.append((int(nbt), int(interp)))
        remaining -= n_this_line
        idx += 1
    return pairs, idx


def parse_data_pairs(lines, start, np_count):
    """
    Read NP x/y data pairs (3 pairs per line) from ENDF lines.

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the first data line.
    np_count : int
        Number of (x, y) pairs to read.

    Returns
    -------
    x_list : list of float
    y_list : list of float
    next_idx : int
    """
    x_list = []
    y_list = []
    idx = start
    num_lines_needed = (np_count + 2) // 3
    for _ in range(num_lines_needed):
        if idx >= len(lines):
            break
        ld = parse_line(lines[idx])
        idx += 1
        for i in range(3):
            if len(x_list) >= np_count:
                break
            x = ld.get(f"C{i * 2 + 1}")
            y = ld.get(f"C{i * 2 + 2}")
            if x is not None and y is not None:
                x_list.append(x)
                y_list.append(y)
    return x_list, y_list, idx


def format_data_values(values, mat, mf, mt, start_line, formats=None):
    """
    Format N scalar values into ENDF LIST-record body lines (6 values per line).

    Counterpart of :func:`parse_data_values`.

    Parameters
    ----------
    values : list of float/int
        Scalar values to write.
    mat, mf, mt : int
        ENDF identification numbers.
    start_line : int
        Starting line sequence number.
    formats : list of str, optional
        Per-value format codes (ENDF_FORMAT_*).  If *None*, all values
        are written as floats.

    Returns
    -------
    lines : list of str
    next_line_num : int
    """
    result_lines = []
    line_num = start_line
    n = len(values)
    i = 0
    while i < n:
        chunk = values[i:i + 6]
        if formats is not None:
            fmts = formats[i:i + 6]
            # Pad with ENDF_FORMAT_BLANK for trailing positions
            while len(fmts) < len(chunk):
                fmts.append(ENDF_FORMAT_FLOAT)
        else:
            fmts = [ENDF_FORMAT_FLOAT] * len(chunk)
        # Pad chunk to 6 values
        while len(chunk) < 6:
            chunk.append(None)
            fmts.append(ENDF_FORMAT_BLANK)
        result_lines.append(format_endf_data_line(chunk, mat, mf, mt, line_num, formats=fmts))
        line_num += 1
        i += 6
    return result_lines, line_num


def parse_data_values(lines, start, n_values):
    """
    Read N scalar values (6 per line) from ENDF lines (LIST record body).

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the first data line.
    n_values : int
        Number of scalar values to read.

    Returns
    -------
    values : list of float
    next_idx : int
    """
    values = []
    idx = start
    num_lines_needed = (n_values + 5) // 6
    for _ in range(num_lines_needed):
        if idx >= len(lines):
            break
        ld = parse_line(lines[idx])
        idx += 1
        for i in range(1, 7):
            if len(values) >= n_values:
                break
            v = ld.get(f"C{i}")
            if v is not None:
                values.append(v)
    return values, idx


# ----------------------------------------------------------------------
# INTG records — the packed correlation matrix of an MF32 LCOMP=2 subsection
# ----------------------------------------------------------------------

#: NDIGIT → (NROW, field width, separator width after JJ).
#:
#: ENDF-102 §32.2.3 gives these as five Fortran FORMAT statements rather than a
#: formula, and the widths are exactly what makes each line land on 66 columns:
#:
#:   NDIGIT=2  (I5, I5, 1X, 18I3, 1X)   10 + 1 + 54 + 1 = 66
#:   NDIGIT=3  (I5, I5, 1X, 13I4, 3X)   10 + 1 + 52 + 3 = 66
#:   NDIGIT=4  (I5, I5, 1X, 11I5)       10 + 1 + 55     = 66
#:   NDIGIT=5  (I5, I5, 1X,  9I6, 1X)   10 + 1 + 54 + 1 = 66
#:   NDIGIT=6  (I5, I5,      8I7)       10 +      56    = 66
#:
#: NDIGIT=6 is the only one with no ``1X`` after JJ, which is why the five cases
#: are a table and not an arithmetic expression. Getting that wrong shifts every
#: field of an NDIGIT=6 record by one column, and no tape on this machine uses
#: NDIGIT=6 to catch it — see ``docs/mf32-notes.md``.
_INTG_LAYOUT: Dict[int, Tuple[int, int, int]] = {
    2: (18, 3, 1),
    3: (13, 4, 1),
    4: (11, 5, 1),
    5: (9, 6, 1),
    6: (8, 7, 0),
}


def intg_row_length(ndigit: int) -> int:
    """How many correlation coefficients one INTG record of *ndigit* holds."""
    try:
        return _INTG_LAYOUT[ndigit][0]
    except KeyError:
        raise ValueError(
            f"NDIGIT={ndigit} is not an ENDF-6 INTG width; §32.2.3 allows 2-6"
        ) from None


def parse_intg(lines: List[str], start: int, ndigit: int,
               nm: int) -> Tuple[List[Tuple[int, int, List[int]]], int]:
    """
    Read *nm* INTG records — the packed correlation matrix of MF32 LCOMP=2.

    Each record locates itself with ``(II, JJ)`` and then carries up to NROW
    integers standing for the correlation coefficients ``C[II,JJ]``,
    ``C[II,JJ+1]``, … A coefficient that mapped to zero may be written either as
    a blank field or as an explicit ``0``; both read back as ``0`` here, so this
    function is **not** enough on its own to rewrite a tape byte-identically.
    The caller keeps the raw text for that — see
    :class:`~kika.endf.classes.mf32.mf32mt151.IntgMatrix`.

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the first INTG record (the CONT carrying NDIGIT/NNN/NM is the
        caller's to consume).
    ndigit : int
        Number of digits of the packed integers, 2-6.
    nm : int
        Number of INTG records to read.

    Returns
    -------
    entries : list of (ii, jj, values)
    next_idx : int
    """
    nrow, width, sep = _INTG_LAYOUT[ndigit] if ndigit in _INTG_LAYOUT else (
        intg_row_length(ndigit), 0, 0)

    entries: List[Tuple[int, int, List[int]]] = []
    idx = start
    for _ in range(nm):
        if idx >= len(lines):
            break
        line = lines[idx]
        idx += 1
        ii = int(line[0:5]) if line[0:5].strip() else 0
        jj = int(line[5:10]) if line[5:10].strip() else 0
        values: List[int] = []
        offset = 10 + sep
        for n in range(nrow):
            field = line[offset + n * width: offset + (n + 1) * width]
            values.append(int(field) if field.strip() else 0)
        entries.append((ii, jj, values))
    return entries, idx


def format_intg(entries: Sequence[Tuple[int, int, Sequence[int]]], ndigit: int,
                mat: int, mf: int, mt: int,
                start_line: int) -> Tuple[List[str], int]:
    """
    Write INTG records. Counterpart of :func:`parse_intg`.

    Zeros are written as blank fields, which is what every evaluation measured
    for ``docs/mf32-notes.md`` does; §32.2.3 permits an explicit ``0`` too, so a
    tape written this way may differ from its source in whitespace alone. That
    is why the round-trip path re-emits stored text instead of calling this —
    this function is for covariance matrices kika *builds*, not ones it read.
    """
    nrow, width, sep = _INTG_LAYOUT[ndigit] if ndigit in _INTG_LAYOUT else (
        intg_row_length(ndigit), 0, 0)

    result_lines: List[str] = []
    line_num = start_line
    for ii, jj, values in entries:
        fields = "".join(
            (f"{int(v):{width}d}" if v else " " * width)
            for v in list(values)[:nrow]
        )
        body = f"{int(ii):5d}{int(jj):5d}{' ' * sep}{fields}".ljust(66)
        result_lines.append(f"{body}{mat:4d}{mf:2d}{mt:3d}{line_num:5d}")
        line_num += 1
    return result_lines, line_num


def parse_tab1(lines, start):
    """
    Parse a complete TAB1 record starting at *start*.

    A TAB1 record consists of:
      - A header line with C1..C6
      - NR interpolation (NBT, INT) pairs
      - NP x/y data pairs

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the TAB1 header line.

    Returns
    -------
    header : dict
        Parsed header fields (C1..C6, MAT, MF, MT).
    interp_pairs : list of tuple(int, int)
    x_data : list of float
    y_data : list of float
    next_idx : int
    """
    header = parse_line(lines[start])
    nr = int(header.get("C5", 0) or 0)
    np_count = int(header.get("C6", 0) or 0)
    idx = start + 1
    interp_pairs = []
    if nr > 0:
        interp_pairs, idx = parse_interp_pairs(lines, idx, nr)
    x_data, y_data, idx = parse_data_pairs(lines, idx, np_count)
    return header, interp_pairs, x_data, y_data, idx


def format_interp_pairs(pairs, mat, mf, mt, start_line):
    """
    Format NR interpolation (NBT, INT) pairs into ENDF lines.

    Returns
    -------
    lines : list of str
    next_line_num : int
    """
    result_lines = []
    line_num = start_line
    remaining = list(pairs)
    while remaining:
        chunk = remaining[:3]
        remaining = remaining[3:]
        values = []
        fmts = []
        for nbt, interp in chunk:
            values.extend([nbt, interp])
            fmts.extend([ENDF_FORMAT_INT, ENDF_FORMAT_INT])
        while len(values) < 6:
            values.append(None)
            fmts.append(ENDF_FORMAT_BLANK)
        result_lines.append(format_endf_data_line(values, mat, mf, mt, line_num, formats=fmts))
        line_num += 1
    return result_lines, line_num


def format_data_pairs(x_data, y_data, mat, mf, mt, start_line):
    """
    Format NP x/y data pairs into ENDF lines (3 pairs per line).

    Returns
    -------
    lines : list of str
    next_line_num : int
    """
    result_lines = []
    line_num = start_line
    n = len(x_data)
    i = 0
    while i < n:
        values = []
        for j in range(3):
            if i + j < n:
                values.extend([x_data[i + j], y_data[i + j]])
            else:
                values.extend([None, None])
        fmts = [ENDF_FORMAT_FLOAT if v is not None else ENDF_FORMAT_BLANK for v in values]
        result_lines.append(format_endf_data_line(values, mat, mf, mt, line_num, formats=fmts))
        line_num += 1
        i += 3
    return result_lines, line_num


def format_tab1(c1, c2, l1, l2, interp_pairs, x_data, y_data, mat, mf, mt, start_line):
    """
    Format a complete TAB1 record to ENDF lines.

    Parameters
    ----------
    c1, c2 : float
        Header float fields.
    l1, l2 : int
        Header integer fields (positions C3, C4).
    interp_pairs : list of tuple(int, int)
    x_data, y_data : list of float
    mat, mf, mt : int
    start_line : int

    Returns
    -------
    lines : list of str
    next_line_num : int
    """
    nr = len(interp_pairs)
    np_count = len(x_data)
    line_num = start_line
    result_lines = []

    # Header line
    header = format_endf_data_line(
        [c1, c2, l1, l2, nr, np_count],
        mat, mf, mt, line_num,
        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    )
    result_lines.append(header)
    line_num += 1

    # Interpolation pairs
    if nr > 0:
        ip_lines, line_num = format_interp_pairs(interp_pairs, mat, mf, mt, line_num)
        result_lines.extend(ip_lines)

    # Data pairs
    dp_lines, line_num = format_data_pairs(x_data, y_data, mat, mf, mt, line_num)
    result_lines.extend(dp_lines)

    return result_lines, line_num


def parse_tab2(lines, start):
    """
    Parse a TAB2 record starting at *start*.

    A TAB2 record consists of a header line (C1..C6) followed by NR
    interpolation (NBT, INT) pairs.  Unlike TAB1, it carries no x/y
    data — NZ (C6) indicates how many sub-records follow.

    Parameters
    ----------
    lines : list of str
        ENDF lines.
    start : int
        Index of the TAB2 header line.

    Returns
    -------
    header : dict
        Parsed header fields (C1..C6, MAT, MF, MT).
    interp_pairs : list of tuple(int, int)
    next_idx : int
    """
    header = parse_line(lines[start])
    nr = int(header.get("C5", 0) or 0)
    idx = start + 1
    interp_pairs = []
    if nr > 0:
        interp_pairs, idx = parse_interp_pairs(lines, idx, nr)
    return header, interp_pairs, idx


def format_tab2(c1, c2, l1, l2, interp_pairs, nz, mat, mf, mt, start_line):
    """
    Format a TAB2 record to ENDF lines.

    Parameters
    ----------
    c1, c2 : float
        Header float fields.
    l1, l2 : int
        Header integer fields (positions C3, C4).
    interp_pairs : list of tuple(int, int)
        Interpolation (NBT, INT) pairs.
    nz : int
        Number of sub-records (written to C6).
    mat, mf, mt : int
    start_line : int

    Returns
    -------
    lines : list of str
    next_line_num : int
    """
    nr = len(interp_pairs)
    line_num = start_line
    result_lines = []

    header = format_endf_data_line(
        [c1, c2, l1, l2, nr, nz],
        mat, mf, mt, line_num,
        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO,
                 ENDF_FORMAT_INT, ENDF_FORMAT_INT],
    )
    result_lines.append(header)
    line_num += 1

    if nr > 0:
        ip_lines, line_num = format_interp_pairs(interp_pairs, mat, mf, mt, line_num)
        result_lines.extend(ip_lines)

    return result_lines, line_num


def project_tabulated_to_legendre(
    mu: ArrayLike,
    fmu: ArrayLike,
    max_order: int,
    ang_nbt_int: Optional[Sequence[Tuple[int, int]]] = None,
    quad_order: int = 64,
) -> np.ndarray:
    """
    Compute Legendre coefficients a_l up to max_order from tabulated f(μ) on μ∈[-1,1].
    Uses Gauss–Legendre quadrature on an ENDF-interpolated f(μ) (respects angular INT codes).

    Conventions:
    - Angular PDF is represented as f(μ) = 1/2 Σ_{l=0}^L (2l+1) a_l P_l(μ)
    - With this convention, coefficients are: a_l = ∫_{-1}^{1} f(μ) P_l(μ) dμ
    """
    mu = np.asarray(mu, dtype=float)
    fmu = np.asarray(fmu, dtype=float)
    if mu.size == 0 or fmu.size == 0:
        return np.zeros(max_order + 1, dtype=float)

    # GL nodes/weights
    mu_q, w_q = np.polynomial.legendre.leggauss(quad_order)
    # Interpolate f to GL nodes with ENDF angular interpolation (default linear)
    f_q = interpolate_1d_endf(mu, fmu, ang_nbt_int or [(len(mu), 2)], mu_q, out_of_range="hold")

    # Normalize on [-1,1] using the same quadrature
    norm = float(np.sum(f_q * w_q))
    if abs(norm) > 1e-15:
        f_q = f_q / norm

    # Project: a_l = ∫ f(μ) P_l(μ) dμ under the convention used elsewhere (a0 ≈ 1)
    coeffs = np.zeros(max_order + 1, dtype=float)
    for l in range(max_order + 1):
        P_l = np.polynomial.legendre.legval(mu_q, [0] * l + [1])  # evaluate P_l(μ)
        coeffs[l] = float(np.sum(P_l * f_q * w_q))
    return coeffs


def auto_trim_legendre_tail(
    coeffs_by_l: Dict[int, Union[float, np.ndarray]],
    tol: float = 1e-6,
    min_order: int = 0
) -> Dict[int, Union[float, np.ndarray]]:
    """
    Auto-trim to smallest L such that sum_{ℓ>L} |a_ℓ| < tol.
    If values are arrays over energies, use a *global* L that satisfies the condition for all energies,
    so dictionary keys remain consistent.
    
    Parameters
    ----------
    coeffs_by_l : dict
        Dictionary mapping order l to coefficient values
    tol : float
        Tolerance for trimming
    min_order : int
        Minimum order to keep (ensures at least orders 0 to min_order are returned)
    """
    if not coeffs_by_l:
        return coeffs_by_l
        
    # collect in order
    max_l = max(coeffs_by_l) if coeffs_by_l else 0
    arrs = [np.atleast_1d(coeffs_by_l.get(l, 0.0)) for l in range(max_l + 1)]
    A = np.vstack(arrs)  # shape: (L+1, nE)
    absA = np.abs(A)
    # tail sums S_l = sum_{j>l} |a_j|
    tail = np.flipud(np.cumsum(np.flipud(absA), axis=0))  # S_l includes |a_l|; we want >l, so shift
    tail_gt = np.vstack([tail[1:, :], np.zeros((1, tail.shape[1]))])  # shift up
    # For each energy (column), find smallest L with tail_gt[L] < tol
    per_energy_L = [int(np.argmax(tail_gt[:, j] < tol)) for j in range(tail_gt.shape[1])]
    L_global = max(per_energy_L) if per_energy_L else max_l
    
    # Ensure we keep at least up to min_order
    L_global = max(L_global, min_order)
    
    return {l: coeffs_by_l[l] for l in range(L_global + 1)}


def pick_mixed_branch(E: float, E_leg: np.ndarray, E_tab: np.ndarray) -> str:
    """
    For LTT=3, decide which branch to use at energy E.
    - If within Legendre range: 'leg'
    - If within tabulated range: 'tab'
    - If between disjoint ranges: pick the closer boundary
    """
    has_leg = E_leg.size > 0
    has_tab = E_tab.size > 0
    if not has_leg and not has_tab:
        return "none"
    if has_leg and (E <= E_leg.max()):
        return "leg"
    if has_tab and (E >= E_tab.min()):
        return "tab"
    if has_leg and has_tab:
        return "leg" if abs(E - E_leg.max()) <= abs(E - E_tab.min()) else "tab"
    return "leg" if has_leg else "tab"


def segment_int_codes(ne: int, nbt_int_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Build an array of length (ne-1) with the INT code for each energy interval [k, k+1].
    ENDF NBT's are 1-based indices of the *last* point in the region.
    """
    if not nbt_int_pairs:
        nbt_int_pairs = [(ne, 2)]  # default linear across full grid

    seg = np.full(ne - 1, 2, dtype=int)
    start = 0
    for nbt, ic in nbt_int_pairs:
        # region covers points [start ... end], so intervals [start ... end-1]
        end = max(0, min(nbt - 1, ne - 1))
        if end > start:
            seg[start:end] = ic
        start = max(0, min(nbt, ne - 1))
        if start >= ne - 1:
            break
    return seg


def interp_energy_values(E0: float, f0: np.ndarray,
                          E1: float, f1: np.ndarray,
                          E: float, int_code: int) -> np.ndarray:
    """
    Vectorized interpolation of y(E) between (E0,f0) and (E1,f1) under ENDF INT code (1..5).
    Falls back to linear where logs are invalid.
    """
    if E0 == E1:
        return np.array(f0, dtype=float, copy=True)

    t = (E - E0) / (E1 - E0)
    code = int_code % 10 if int_code >= 10 else int_code
    code = 5 if code == 0 else code  # 10,20 → 0 → use 5

    # default lin-lin
    if code == 1:
        return np.array(f0, dtype=float, copy=True)  # histogram in E: hold left
    if code == 2:
        return (1.0 - t) * np.asarray(f0, dtype=float) + t * np.asarray(f1, dtype=float)

    # helpers
    f0 = np.asarray(f0, dtype=float)
    f1 = np.asarray(f1, dtype=float)

    # lin-log (y linear in ln E)
    if code == 3:
        if E0 <= 0 or E1 <= 0 or E <= 0:
            return (1.0 - t) * f0 + t * f1
        le0, le1, le = math.log(E0), math.log(E1), math.log(E)
        tt = (le - le0) / (le1 - le0)
        return (1.0 - tt) * f0 + tt * f1

    # log-lin (ln y linear in E)
    if code == 4:
        mask = (f0 > 0.0) & (f1 > 0.0)
        out = (1.0 - t) * f0 + t * f1
        if np.any(mask):
            ln_y = (1.0 - t) * np.log(f0[mask]) + t * np.log(f1[mask])
            out[mask] = np.exp(ln_y)
        return out

    # log-log (ln y linear in ln E)
    if code == 5:
        if E0 <= 0 or E1 <= 0 or E <= 0:
            return (1.0 - t) * f0 + t * f1
        le0, le1, le = math.log(E0), math.log(E1), math.log(E)
        tt = (le - le0) / (le1 - le0)
        mask = (f0 > 0.0) & (f1 > 0.0)
        out = (1.0 - t) * f0 + t * f1
        if np.any(mask):
            ln_y = (1.0 - tt) * np.log(f0[mask]) + tt * np.log(f1[mask])
            out[mask] = np.exp(ln_y)
        return out

    # fallback
    return (1.0 - t) * f0 + t * f1
