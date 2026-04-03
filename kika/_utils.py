import re

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, BOLTZMANN_CONSTANT, K_TO_SUFFIX, SYMBOL_TO_ATOMIC_NUMBER

def symbol_to_zaid(symbol: str) -> int:
    """
    Convert an element-mass symbol to ZAID (e.g., Fe56 -> 26056, Fe -> 26000 for natural)

    Supports metastable suffixes: "Co58m" -> 1027058, "Co58m2" -> 2027058.
    The returned ZAID uses extended IZZZAAA format when isomeric_state > 0.

    Parameters
    ----------
    symbol : str
        Element symbol with optional mass number and metastable suffix
        (e.g., "Fe56", "Fe", "U235", "H", "Co58m", "Co58m2")

    Returns
    -------
    int
        ZAID identifier (ZZAAA or IZZZAAA format)

    Raises
    ------
    ValueError
        If the element symbol is not recognized or format is invalid

    Examples
    --------
    >>> symbol_to_zaid("Fe56")
    26056
    >>> symbol_to_zaid("Fe")
    26000
    >>> symbol_to_zaid("U235")
    92235
    >>> symbol_to_zaid("Co58m")
    1027058
    """
    if not symbol or len(symbol) < 1:
        raise ValueError(f"Invalid symbol: '{symbol}'. Must be at least 1 character.")

    # Extract metastable suffix (e.g., "Co58m" or "Co58m2")
    # Lookbehind ensures 'm' follows a digit, avoiding false matches on element names like "Am"
    meta_match = re.search(r'(?<=\d)m(\d*)$', symbol)
    isomeric_state = 0
    if meta_match:
        isomeric_state = int(meta_match.group(1)) if meta_match.group(1) else 1
        symbol = symbol[:meta_match.start()]

    # Try to extract element symbol (1-2 characters, case-insensitive)
    element_symbol = None
    mass_number = None

    for elem_len in (2, 1):  # Try 2-char first, then 1-char
        potential_symbol = symbol[:elem_len]
        # Check if it matches a known element (case-insensitive)
        for known_symbol, atomic_num in SYMBOL_TO_ATOMIC_NUMBER.items():
            if potential_symbol.lower() == known_symbol.lower():
                element_symbol = known_symbol
                potential_mass = symbol[elem_len:]
                if potential_mass:
                    try:
                        mass_number = int(potential_mass)
                    except ValueError:
                        raise ValueError(f"Invalid mass number in symbol '{symbol}': '{potential_mass}' is not an integer.")
                else:
                    mass_number = 0  # Natural element
                break

        if element_symbol:
            break

    if element_symbol is None:
        raise ValueError(f"Unknown element symbol: '{symbol}'")

    atomic_number = SYMBOL_TO_ATOMIC_NUMBER[element_symbol]

    # Construct ZAID: ZZAAA format (or IZZZAAA for metastable)
    if mass_number is None:
        mass_number = 0  # Natural element

    if mass_number < 0 or mass_number > 999:
        raise ValueError(f"Mass number must be between 0 and 999, got {mass_number}")

    zaid = atomic_number * 1000 + mass_number
    if isomeric_state > 0:
        zaid = isomeric_state * 1000000 + zaid
    return zaid


def zaid_to_symbol(zaid: int) -> str:
    """
    Convert a ZAID to element-mass symbol (e.g., 26056 -> Fe56)
    For natural elements (mass number 0), returns just the element symbol (e.g., 26000 -> Fe)

    Supports extended IZZZAAA format used in SCALE COVERX files, where I is the
    isomeric state (0=ground, 1=1st metastable, etc.). Example: 1027058 -> Co58m.

    Parameters
    ----------
    zaid : int
        ZAID identifier (ZZAAA or IZZZAAA format)

    Returns
    -------
    str
        Element symbol with mass number (e.g., "Fe56", "Co58m") or just element (e.g., "Fe" for natural)
    """
    z = zaid // 1000
    a = zaid % 1000

    # Handle extended ZAID with isomeric state prefix (IZZZAAA)
    # SCALE COVERX files use this format: e.g. 1027058 = Co-58m
    isomeric_state = 0
    if z > 118:
        isomeric_state = zaid // 1000000
        z = (zaid % 1000000) // 1000
        a = zaid % 1000

    if z in ATOMIC_NUMBER_TO_SYMBOL:
        symbol = ATOMIC_NUMBER_TO_SYMBOL[z]
        if a == 0:
            base = symbol
        else:
            base = f"{symbol}{a}"
        if isomeric_state > 0:
            base += "m" if isomeric_state == 1 else f"m{isomeric_state}"
        return base
    return f"{zaid}"  # Fallback if conversion fails

def kelvin_to_MeV(temp: float) -> float:
    """
    Convert temperature in Kelvin to MeV.
    
    Parameters
    ----------
    temp : float
        Temperature in Kelvin
    
    Returns
    -------
    float
        Temperature in MeV
    """
    return temp * BOLTZMANN_CONSTANT

def MeV_to_kelvin(temp: float) -> float:
    """
    Convert temperature in MeV to Kelvin.
    
    Parameters
    ----------
    temp : float
        Temperature in MeV
    
    Returns
    -------
    float
        Temperature in Kelvin
    """
    return round(temp / BOLTZMANN_CONSTANT, 2)


def temperature_to_suffix(temp_K: float) -> str:
    """
    Convert temperature in Kelvin to MCNP suffix based on K_TO_SUFFIX mapping.
    
    Parameters
    ----------
    temp_K : float
        Temperature in Kelvin
    
    Returns
    -------
    str
        MCNP suffix (e.g., ".02", ".03", etc.)
        
    Examples
    --------
    >>> temperature_to_suffix(293.6)
    '.02'
    >>> temperature_to_suffix(300)
    '.03'
    """
    # Find the closest temperature in K_TO_SUFFIX
    closest_temp = min(K_TO_SUFFIX.keys(), key=lambda k: abs(k - temp_K))
    return K_TO_SUFFIX[closest_temp]


def add_repr_method(method_name, description, buffer=None, method_col_width=30, desc_col_width=45, 
                  align_method="<", align_desc="<"):
    """
    Format a method name and description as a row in a two-column table with text wrapping.
    
    This utility function is designed to be used in __repr__ methods to create consistent
    method documentation sections across different classes.
    
    Parameters
    ----------
    method_name : str
        The name of the method to document
    description : str
        The description of what the method does
    buffer : str, optional
        An existing string buffer to append to. If None, a new string is returned.
    method_col_width : int, optional
        Width of the method name column, default 30
    desc_col_width : int, optional
        Width of the description column, default 45
    align_method : str, optional
        Alignment for method column ("<" for left, "^" for center, ">" for right), default "<"
    align_desc : str, optional
        Alignment for description column, default "<"
        
    Returns
    -------
    str
        The formatted string with the method and description properly formatted and wrapped
        
    Examples
    --------
    >>> buffer = ""
    >>> buffer = add_repr_method(".my_method()", "This is what the method does", buffer)
    >>> buffer = add_repr_method(".another_method(param)", "This has a very long description that will be automatically wrapped to the next line", buffer)
    >>> print(buffer)
    .my_method()                    This is what the method does
    .another_method(param)          This has a very long description that will 
                                    be automatically wrapped to the next line
    """
    result = ""
    
    # Format the first line with method name and first part of description
    if len(description) <= desc_col_width:
        result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
            method_name, description, 
            align=align_method, width1=method_col_width, 
            align2=align_desc, width2=desc_col_width)
    else:
        # Find a good break point for the first line (at a space)
        break_point = desc_col_width
        if ' ' in description[:desc_col_width]:
            # Find the last space within the width limit
            last_space = description[:desc_col_width].rstrip().rfind(' ')
            if last_space > 0:
                break_point = last_space + 1
        
        first_part = description[:break_point].rstrip()
        result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
            method_name, first_part, 
            align=align_method, width1=method_col_width, 
            align2=align_desc, width2=desc_col_width)
        
        # Process remaining text with improved word-aware wrapping
        remaining = description[break_point:].lstrip()
        while remaining:
            # Find a good break point for this chunk
            if len(remaining) <= desc_col_width:
                chunk = remaining
                remaining = ""
            else:
                chunk_end = desc_col_width
                if ' ' in remaining[:desc_col_width]:
                    # Find the last space within the width limit
                    last_space = remaining[:desc_col_width].rstrip().rfind(' ')
                    if last_space > 0:
                        chunk_end = last_space + 1
                
                chunk = remaining[:chunk_end].rstrip()
                remaining = remaining[chunk_end:].lstrip()
            
            # Add the continuation line with empty method column
            result += "{:{align}{width1}} {:{align2}{width2}}\n".format(
                "", chunk, 
                align=align_method, width1=method_col_width, 
                align2=align_desc, width2=desc_col_width)
    
    # Either append to the buffer or return the result
    if buffer is not None:
        return buffer + result
    else:
        return result


def create_repr_section(title, methods_dict, total_width=85, method_col_width=30, desc_col_width=None):
    """
    Create a complete representation section for methods with proper formatting.
    
    This creates a titled section with a header, method table, and bottom border.
    
    Parameters
    ----------
    title : str
        The title for the section, e.g. "Available Methods:"
    methods_dict : dict
        Dictionary mapping method names to their descriptions
    total_width : int, optional
        Total width of the table, default 85
    method_col_width : int, optional
        Width of the method name column, default 30
    desc_col_width : int, optional
        Width of the description column, calculated from total_width if None
        
    Returns
    -------
    str
        A complete formatted section with all methods
        
    Examples
    --------
    >>> methods = {
    ...     ".method1()": "Description of method 1",
    ...     ".method2(param)": "Description of method 2 with parameters"
    ... }
    >>> print(create_repr_section("Available Methods:", methods))
    Available Methods:
    -----------------------------
    Method                         Description
    -----------------------------
    .method1()                     Description of method 1
    .method2(param)                Description of method 2 with parameters
    -----------------------------
    """
    if desc_col_width is None:
        # Calculate description column width based on total width
        desc_col_width = total_width - method_col_width - 3  # -3 for spacing and formatting
    
    # Create the section header
    section = f"{title}\n"
    section += "-" * total_width + "\n"
    
    # Add table header
    section += "{:<{width1}} {:<{width2}}\n".format(
        "Method", "Description", width1=method_col_width, width2=desc_col_width)
    section += "-" * total_width + "\n"
    
    # Add each method
    for method, description in methods_dict.items():
        section = add_repr_method(method, description, section, method_col_width, desc_col_width)
    
    # Add bottom border
    section += "-" * total_width + "\n"
    
    return section
