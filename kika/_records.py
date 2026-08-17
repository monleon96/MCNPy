"""The fixed-width record grammar the nuclear-data formats share.

ENDF-6 writes 11-character numeric fields, six to a line, followed by a
14-character identification field. **That convention is not ENDF's property.**
COVERX, COVFIL and BOXER use the same one, which is why
``kika/cov/parse_covmat.py`` used to import all of this from
``kika.endf.utils`` -- the single import in ``kika/cov`` that ran at *import*
time rather than at call time, and the one the layering ratchet called "the
genuine leak".

Moved here in phase 4's P4, on the precedent phase 2 set with
``interpolate_1d_endf`` -> ``kika.processing.interpolation``: the interpolation
laws are not ENDF's property either, and the fix for a real shared dependency
is to put the shared thing where both callers can reach it without one
importing the other.

``kika.endf.utils`` re-exports every name below, and that is **not** tidiness:
``kika/tests/test_library_export_surface.py`` pins
``("kika.endf.utils", "parse_endf_id")`` as public surface, and the MF1/MF2/MF4
classes, the parsers, the writers and ``scripts/build_group_cross.py`` all
import from there.

What deliberately stayed behind in ``kika.endf.utils``: the SEND/FEND/MEND/TEND
emitters and ``_TERMINATION_FORMATS``. A terminator record is an ENDF-6
structural statement, not a fixed-width convention -- COVERX has no SEND.

This module is a leaf. It imports nothing from kika.
"""
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _mantissa_precision(exponent: int) -> int:
    """Decimal places the mantissa gets, so the field stays 11 characters wide."""
    if abs(exponent) < 10:
        return 6
    if abs(exponent) < 100:
        return 5
    return 4


def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.

    The output is an 11-character field made up as follows:
      - The first character is '-' if the number is negative or a blank if positive.
      - The number is written in scientific notation without an 'E'.
      - When the exponent (after normalization) has only one digit (|exponent| < 10),
        the mantissa is printed with 6 decimal digits and the exponent with one digit.
      - When the exponent has two digits (|exponent| >= 10), the mantissa is printed with 5 decimal digits and the exponent with two digits.
      - When the exponent has three digits (|exponent| >= 100), the mantissa is
        printed with 4 decimal digits and the exponent with three digits.

    For example:
      - A number like -3.14159e-1 will be formatted as "-3.141590-1".
      - A number like 1.234567e+5 will be formatted as " 1.234567+5".
      - A number like 1.0e10 will be formatted as " 1.00000+10".
      - A number like 1.5963e-100 will be formatted as " 1.5963-100".

    The three-digit case is not hypothetical and used to be **written as zero**.
    ``tsl-ortho-H.endf`` tabulates S(α, β) down to 1e-100 and below — 1 403 of
    its MF7/MT4 records carry such a value — and every one of them came back
    from this function as ``" 0.000000+0"``. Silently: nothing warned, and the
    line stayed the right width, so the loss was invisible to a caller that did
    not diff against the source. Three digits is also the end of the ladder,
    since a finite double cannot exceed 1e308.

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

    if not math.isfinite(value):
        raise ValueError(f"Cannot format non-finite ENDF value: {value}")

    sign_char = "-" if value < 0 else " "
    abs_val = abs(value)
    exponent = int(math.floor(math.log10(abs_val)))
    mantissa = abs_val / (10 ** exponent)

    # Select the number of decimals so that sign + mantissa + exponent sign +
    # exponent digits stays exactly 11 characters: one decimal is given up for
    # each extra digit the exponent needs.
    prec = _mantissa_precision(exponent)
    mantissa_str = f"{mantissa:1.{prec}f}"
    # Rounding overflow: e.g. 9.9999999 -> "10.000000" (length > prec + 2)
    if len(mantissa_str) > prec + 2:
        mantissa /= 10.0
        exponent += 1
        prec = _mantissa_precision(exponent)
        mantissa_str = f"{mantissa:1.{prec}f}"

    exp_str = f"{abs(exponent):d}" if abs(exponent) < 10 else (
        f"{abs(exponent):02d}" if abs(exponent) < 100
        else f"{abs(exponent):03d}"
    )
    exp_sign = '+' if exponent >= 0 else '-'

    formatted = f"{sign_char}{mantissa_str}{exp_sign}{exp_str}"
    return formatted.rjust(width)


# Format constants for ENDF data types
ENDF_FORMAT_FLOAT = 'float'       # Scientific notation (e.g., " 1.234567+5")
ENDF_FORMAT_INT = 'int'           # Integer format (e.g., "         11")
ENDF_FORMAT_BLANK = 'blank'       # Blank field
ENDF_FORMAT_PRESERVE = 'preserve' # Use value's own type to determine format

#: Alias for :data:`ENDF_FORMAT_INT`, kept for the ~30 call sites that use it.
#:
#: It read "integer with zero rendered as 0 (not blank)", implying a contrast
#: with ENDF_FORMAT_INT. There has never been one: ``format_endf_data_line``
#: gave both constants the same branch, and neither has ever blanked a zero —
#: blanking is what ENDF_FORMAT_BLANK does. Making it an alias states that
#: outright, rather than leaving two names that look like a choice.
#:
#: Do not add the promised behaviour instead. Roughly forty call sites pass
#: ENDF_FORMAT_INT for fields whose zeros must be written as 0, and giving the
#: name real meaning would move every one of them.
ENDF_FORMAT_INT_ZERO = ENDF_FORMAT_INT


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
                # ENDF_FORMAT_INT_ZERO is an alias for this, so one branch
                # serves both — as it always did, in two identical copies.
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

    return data_part + format_endf_id_columns(mat, mf, mt, line_num)


#: Highest sequence number the five-character NS field can hold. ENDF-6 counts
#: records from 1 within a section and wraps here rather than widening: the
#: record after 99999 is 1 again, not 100000.
MAX_SEQUENCE_NUMBER = 99999


def format_endf_id_columns(mat: int, mf: int, mt: int, line_num: int) -> str:
    """Columns 67-80: MAT, MF, MT and the wrapped sequence number.

    The wrap is not cosmetic. ``f"{100000:5d}"`` is six characters wide, so a
    section long enough to reach it used to be written 81 columns wide from that
    line on. Only two sections on this machine are long enough to show it —
    Ta-181's MF32 at 240 131 lines and Pu-239's at 190 445 — and both wrap
    99999 → 1, which is ``((n - 1) % 99999) + 1``.

    SEND, FEND, MEND and TEND pass their own fixed numbers through here
    unchanged: 99999 maps to itself and 0 stays 0.
    """
    if line_num > MAX_SEQUENCE_NUMBER:
        line_num = ((line_num - 1) % MAX_SEQUENCE_NUMBER) + 1
    return f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}"


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
