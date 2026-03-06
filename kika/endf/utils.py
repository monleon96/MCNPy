"""
Utility functions for ENDF file parsing and writing.

Contains helper functions for handling the specific formatting requirements of ENDF files.
"""
import re
from typing import Dict,Union, List, Optional, Tuple, Sequence, Any
from .classes.mt import MT
from .classes.mf1.mf1mt import MT451
from .classes.mf import MF
import numpy as np
import math
from numpy.typing import ArrayLike

def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.

    The output is an 11-character field made up as follows:
      - The first character is '-' if the number is negative or a blank if positive.
      - The number is written in scientific notation without an 'E'.
      - When the exponent (after normalization) has only one digit (|exponent| < 10),
        the mantissa is printed with 6 decimal digits and the exponent with one digit.
      - When the exponent has two digits (|exponent| >= 10), the mantissa is printed with 5 decimal digits and the exponent with two digits.
      
    For example:
      - A number like -3.14159e-1 will be formatted as "-3.141590-1".
      - A number like 1.234567e+5 will be formatted as " 1.234567+5".
      - A number like 1.0e10 will be formatted as " 1.00000+10".

    Args:
        value: The number to be formatted. If None, returns a blank field.
        width: The total field width (default is 11 characters).

    Returns:
        A string representing the formatted number in ENDF style.
    """
    if value is None:
        return " " * width

    # Special handling for zero: use exponent 0 (one-digit) and 6 decimal places.
    if value == 0:
        return " 0.000000+0"

    sign_char = "-" if value < 0 else " "
    abs_val = abs(value)
    exponent = int(math.floor(math.log10(abs_val)))
    if abs(exponent) > 99:
        return " 0.000000+0"
    mantissa = abs_val / (10 ** exponent)

    # Select the number of decimals based on the exponent.
    # Use 6 decimals if |exponent| < 10, else use 5 decimals.
    # Adjust the mantissa if rounding would push it to 10 or more.
    prec = 6 if abs(exponent) < 10 else 5
    mantissa_str = f"{mantissa:1.{prec}f}"
    # Rounding overflow: e.g. 9.9999999 -> "10.000000" (length > prec + 2)
    if len(mantissa_str) > prec + 2:
        mantissa /= 10.0
        exponent += 1
        prec = 6 if abs(exponent) < 10 else 5
        mantissa_str = f"{mantissa:1.{prec}f}"

    # Format the exponent: one digit if |exponent| < 10, two digits otherwise.
    if abs(exponent) < 10:
        exp_str = f"{abs(exponent):d}"
    else:
        exp_str = f"{abs(exponent):02d}"
    exp_sign = '+' if exponent >= 0 else '-'

    formatted = f"{sign_char}{mantissa_str}{exp_sign}{exp_str}"
    return formatted.rjust(width)


# Format constants for ENDF data types
ENDF_FORMAT_FLOAT = 'float'       # Scientific notation (e.g., " 1.234567+5")
ENDF_FORMAT_INT = 'int'           # Integer format (e.g., "         11")
ENDF_FORMAT_INT_ZERO = 'int_zero' # Integer with zero rendered as 0 (not blank)
ENDF_FORMAT_BLANK = 'blank'       # Blank field
ENDF_FORMAT_PRESERVE = 'preserve' # Use value's own type to determine format


def format_endf_data_line(values: Sequence[Union[int, float, None]], 
                         mat: int, mf: int, mt: int, line_num: int = 0,
                         formats: Optional[List[str]] = None) -> str:
    """
    Format a complete ENDF line with both data and identification parts.
    
    Args:
        values: Sequence of up to 6 numeric values for the data part
        mat: Material number
        mf: File number
        mt: Section number
        line_num: Line sequence number (optional)
        formats: Optional list of format types for each value (ENDF_FORMAT_*)
        
    Returns:
        Formatted 80-character ENDF line
    """
    # Format the data part (columns 1-66)
    parts = []

    # Apply formats if provided, otherwise use default formatting
    if formats:
        # Make sure formats list matches values length
        format_list = formats + [ENDF_FORMAT_PRESERVE] * (len(values) - len(formats))
        format_list = format_list[:len(values)]

        for value, fmt in zip(values, format_list):
            if fmt == ENDF_FORMAT_INT and value is not None:
                parts.append(f"{int(value):11d}")
            elif fmt == ENDF_FORMAT_INT_ZERO and value is not None:
                parts.append(f"{int(value):11d}")
            elif fmt == ENDF_FORMAT_BLANK or value is None:
                parts.append("           ")
            else:
                parts.append(format_endf_number(value))
    else:
        for value in values[:6]:
            parts.append(format_endf_number(value))

    # Pad to 66 characters if needed
    data_part = ''.join(parts).ljust(66)
    
    # Format the identification part (columns 67-80)
    id_part = f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}"
    
    return data_part + id_part


def parse_number(text: str) -> Union[float, int, None]:
    """
    Parse an ENDF-formatted number.
    
    ENDF uses a special format where numbers can be written in forms like:
    "1.234+5" meaning 1.234×10^5
    
    Args:
        text: The text representation of the number
        
    Returns:
        Parsed number as float or int, or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None
    
    try:
        # Try standard float parsing first
        value = float(text)
        # Return as int if it's a whole number
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        # Handle ENDF-specific format where "+" or "-" might be used instead of "E"
        # For example, "1.234+5" instead of "1.234E+5"
        match = re.search(r'([-+]?\d*\.\d*)([+-]\d+)', text)
        if match:
            try:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                value = mantissa * (10 ** exponent)
                if value.is_integer():
                    return int(value)
                return value
            except (ValueError, IndexError):
                pass
                
        # If all parsing fails
        return None


def parse_line(line: str) -> Dict[str, Any]:
    """
    Parse a standard ENDF record line into its components.
    
    Args:
        line: An 80-character ENDF line
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    # Parse data fields (columns 1-66)
    if len(line) >= 66:
        data_part = line[:66]
        # ENDF format typically has 6 fields of 11 characters each
        for i in range(6):
            field_name = f"C{i+1}"
            start = i * 11
            end = start + 11
            if end <= len(data_part):
                field_value = data_part[start:end].strip()
                result[field_name] = parse_number(field_value)
    
    # Parse identification fields (columns 67-80)
    if len(line) >= 75:
        result["MAT"] = int(line[66:70]) if line[66:70].strip() else None
        result["MF"] = int(line[70:72]) if line[70:72].strip() else None
        result["MT"] = int(line[72:75]) if line[72:75].strip() else None
        
    if len(line) >= 80:
        result["SEQ"] = int(line[75:80]) if line[75:80].strip() else None
    
    return result


def parse_endf_id(line: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse the identification fields from an ENDF line.
    
    ENDF format specifies:
    - Columns 67-70 (0-indexed: 66-69): MAT number
    - Columns 71-72 (0-indexed: 70-71): MF number
    - Columns 73-75 (0-indexed: 72-74): MT number
    
    Args:
        line: A line from an ENDF file
        
    Returns:
        Tuple of (MAT, MF, MT) numbers
    """
    if len(line) < 75:
        return None, None, None
    
    try:
        # ENDF format has specific columns for MAT, MF, MT
        mat_str = line[66:70].strip()
        mf_str = line[70:72].strip()
        mt_str = line[72:75].strip()
        
        # Convert to integers, handling empty strings
        mat = int(mat_str) if mat_str else None
        mf = int(mf_str) if mf_str else None
        mt = int(mt_str) if mt_str else None
        
        return mat, mf, mt
    except ValueError as e:
        # This might happen if the fields contain non-numeric data
        return None, None, None


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



def _regionize(nbt_int_pairs: Sequence[Tuple[int, int]], np_len: int) -> List[Tuple[int, int, int]]:
    """
    Convert ENDF (NBT, INT) pairs into 0-based [start_idx, end_idx, INT] regions.
    NBT is 1-based index of the *last* point in each region in ENDF.
    """
    if not nbt_int_pairs:
        # single region with default linear
        return [(0, np_len - 1, 2)]
    regions: List[Tuple[int, int, int]] = []
    start = 0
    for nbt, int_code in nbt_int_pairs:
        end = min(max(nbt - 1, 0), np_len - 1)
        if end >= start:
            regions.append((start, end, int_code))
        start = min(max(nbt, 0), np_len)  # next region starts at nbt (1-based → 0-based)
        if start >= np_len:
            break
    # Guard if list does not cover the tail
    if regions and regions[-1][1] < np_len - 1:
        regions.append((regions[-1][1], np_len - 1, regions[-1][2]))
    if not regions:
        regions = [(0, np_len - 1, 2)]
    return regions


def _base_int_code(int_code: int) -> int:
    """Map 11–15 → 1–5 and 21–25 → 1–5 for 1-D use."""
    if int_code >= 10:
        return int_code % 10 if int_code % 10 != 0 else 5
    return int_code


def _interp_pair(x: float, x1: float, y1: float, x2: float, y2: float, int_code: int) -> float:
    """
    Interpolate y(x) between (x1,y1) and (x2,y2) using ENDF INT code semantics (1–5).
    For INT=6 or unsupported codes → fall back to linear-linear.
    """
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    code = _base_int_code(int_code)
    if code == 1:  # histogram/constant
        return y1
    elif code == 2:  # lin-lin
        return (1.0 - t) * y1 + t * y2
    elif code == 3:  # lin-log (y linear in ln x)
        if x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        return (1.0 - tt) * y1 + tt * y2
    elif code == 4:  # log-lin (ln y linear in x)
        if y1 <= 0 or y2 <= 0:
            return (1.0 - t) * y1 + t * y2
        ln_y = (1.0 - t) * math.log(y1) + t * math.log(y2)
        return math.exp(ln_y)
    elif code == 5:  # log-log (ln y linear in ln x)
        if y1 <= 0 or y2 <= 0 or x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        ln_y = (1.0 - tt) * math.log(y1) + tt * math.log(y2)
        return math.exp(ln_y)
    else:  # fallback for INT=6 etc.
        return (1.0 - t) * y1 + t * y2


def _interp_pair_vec(
    xq: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray, int_code: int,
) -> np.ndarray:
    """Vectorized interpolation between paired arrays using ENDF INT code."""
    out = np.empty_like(xq, dtype=float)
    same = x1 == x2
    if same.all():
        return y1.copy()
    diff = ~same
    dx = np.where(diff, x2 - x1, 1.0)
    t = (xq - x1) / dx
    code = _base_int_code(int_code)
    if code == 1:
        out[:] = y1
    elif code == 2:
        out[:] = (1.0 - t) * y1 + t * y2
    elif code == 3:
        safe = diff & (x1 > 0) & (x2 > 0) & (xq > 0)
        out[:] = (1.0 - t) * y1 + t * y2  # fallback
        if safe.any():
            lx1 = np.log(x1[safe]); lx2 = np.log(x2[safe]); lx = np.log(xq[safe])
            tt = (lx - lx1) / (lx2 - lx1)
            out[safe] = (1.0 - tt) * y1[safe] + tt * y2[safe]
    elif code == 4:
        safe = diff & (y1 > 0) & (y2 > 0)
        out[:] = (1.0 - t) * y1 + t * y2
        if safe.any():
            ln_y = (1.0 - t[safe]) * np.log(y1[safe]) + t[safe] * np.log(y2[safe])
            out[safe] = np.exp(ln_y)
    elif code == 5:
        safe = diff & (x1 > 0) & (x2 > 0) & (xq > 0) & (y1 > 0) & (y2 > 0)
        out[:] = (1.0 - t) * y1 + t * y2
        if safe.any():
            lx1 = np.log(x1[safe]); lx2 = np.log(x2[safe]); lx = np.log(xq[safe])
            tt = (lx - lx1) / (lx2 - lx1)
            ln_y = (1.0 - tt) * np.log(y1[safe]) + tt * np.log(y2[safe])
            out[safe] = np.exp(ln_y)
    else:
        out[:] = (1.0 - t) * y1 + t * y2
    if same.any():
        out[same] = y1[same]
    return out


def interpolate_1d_endf(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    nbt_int_pairs: Sequence[Tuple[int, int]],
    xq: Union[float, ArrayLike],
    out_of_range: str = "zero",
) -> Union[float, np.ndarray]:
    """
    ENDF one-dimensional interpolation using (NBT, INT) regions (Table of INT codes).
    - out_of_range: 'zero'  → return 0 outside grid
                    'hold'  → hold edge value
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    scalar = np.ndim(xq) == 0
    if x.size == 0:
        return 0.0 if scalar else np.zeros_like(np.asarray(xq, dtype=float))
    regions = _regionize(nbt_int_pairs, len(x))
    xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
    n = xq_arr.size

    # Vectorized interval lookup
    k = np.searchsorted(x, xq_arr, side="right") - 1
    np.clip(k, 0, len(x) - 2, out=k)

    # Out-of-range masks
    lo_mask = xq_arr < x[0]
    hi_mask = xq_arr > x[-1]
    in_mask = ~(lo_mask | hi_mask)

    out = np.empty(n, dtype=float)
    if out_of_range == "zero":
        out[lo_mask] = 0.0
        out[hi_mask] = 0.0
    else:
        out[lo_mask] = y[0]
        out[hi_mask] = y[-1]

    if not in_mask.any():
        return float(out[0]) if scalar else out

    # In-range points
    k_in = k[in_mask]
    xq_in = xq_arr[in_mask]

    # Fast path: single region
    if len(regions) == 1:
        _, _, ic = regions[0]
        base_ic = _base_int_code(ic)
        if base_ic == 2:
            # Pure linear-linear → numpy builtin
            out[in_mask] = np.interp(xq_in, x, y)
        else:
            out[in_mask] = _interp_pair_vec(
                xq_in, x[k_in], y[k_in], x[k_in + 1], y[k_in + 1], ic
            )
    else:
        # Assign INT codes per query point from regions
        int_codes = np.full(k_in.size, 2, dtype=int)
        for start, end, ic in regions:
            rmask = (k_in + 1 >= start) & (k_in + 1 <= end)
            int_codes[rmask] = ic
        unique_codes = np.unique(int_codes)
        if unique_codes.size == 1:
            ic = int(unique_codes[0])
            if _base_int_code(ic) == 2:
                out[in_mask] = np.interp(xq_in, x, y)
            else:
                out[in_mask] = _interp_pair_vec(
                    xq_in, x[k_in], y[k_in], x[k_in + 1], y[k_in + 1], ic
                )
        else:
            result_in = np.empty(k_in.size, dtype=float)
            for ic in unique_codes:
                cm = int_codes == ic
                ki = k_in[cm]
                result_in[cm] = _interp_pair_vec(
                    xq_in[cm], x[ki], y[ki], x[ki + 1], y[ki + 1], int(ic)
                )
            out[in_mask] = result_in

    return float(out[0]) if scalar else out


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
