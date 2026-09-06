import os
import io
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.xss import XssEntry
from kika.ace.writers.write_header import write_header

def write_ace(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    # Verify header and XSS data exist
    if ace.header is None:
        raise ValueError("Ace object must have a header")
    if ace.xss_data is None or len(ace.xss_data) == 0:
        raise ValueError("Ace object must have XSS data")
    
    # Determine file path if not provided
    if filepath is None:
        if ace.filename is not None:
            base_path, extension = os.path.splitext(ace.filename)
            filepath = f"{base_path}_recon{extension}"
            print(f"Using modified original filename: {filepath}")
        else:
            zaid = getattr(ace.header, 'zaid', None) or "unknown"
            temp = getattr(ace.header, 'temperature', None) or "0K"
            filepath = f"{zaid}_{int(temp)}K.ace"
            print(f"No filepath provided. Writing to {filepath} in the current directory.")
    
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite is False")
    
    # Work on a local view: wrap only what is not already an XssEntry, and never
    # assign back to ace.xss_data. read_xss seeds index 0 with a bare 0 as the
    # FORTRAN 1-based placeholder, so something has to cover it before the index
    # checks below — but wrapping the whole list, as this used to, re-wrapped
    # every real entry into XssEntry(value=XssEntry(...)) on the caller's object.
    entries = [
        entry if isinstance(entry, XssEntry) else XssEntry(index=i, value=entry)
        for i, entry in enumerate(ace.xss_data)
    ]

    xss_length = len(entries)
    indices = [entry.index for entry in entries]

    # Validate indices
    if any(idx is None for idx in indices):
        raise ValueError("Found XSS entry with no index")
    if min(indices) < 0 or max(indices) >= xss_length:
        raise ValueError(f"Invalid XSS indices (valid range: 0 to {xss_length-1})")
    if len(set(indices)) != xss_length:
        missing = set(range(xss_length)) - set(indices)
        raise ValueError(f"Missing XSS indices: {sorted(missing)}")
    
    # If the data is not already sorted, sort it by index
    if any(entries[i].index > entries[i+1].index for i in range(xss_length - 1)):
        sorted_xss = sorted(entries, key=lambda entry: entry.index)
    else:
        sorted_xss = entries

    # Skip the 0th element if there are multiple entries
    start_idx = 1 if len(sorted_xss) > 1 else 0

    # Extract the numeric values and convert to a NumPy array for fast processing
    values_list = [entry.value for entry in sorted_xss[start_idx:]]
    values = np.array(values_list, dtype=float)  # using float for uniformity

    # Determine the formatting string: use integer formatting if all values are integers
    if np.all(np.abs(values - np.rint(values)) < 1e-12):
        values = values.astype(np.int64)
        fmt = "%20d"
    else:
        fmt = "%20.11E"

    try:
        # Open file with a large buffer and write header and data in chunks
        with open(filepath, 'w', buffering=1024*1024) as f:
            header_str = write_header(ace.header)
            f.write(header_str + "\n")
            
            n = values.shape[0]
            n_full_rows = n // 4
            remainder = n % 4
            
            # Reshape and write full rows (each with 4 values) using NumPy's fast formatting
            if n_full_rows > 0:
                full_rows = values[:n_full_rows*4].reshape(-1, 4)
                row_fmt = fmt * 4  # Concatenate the format string for 4 columns
                np.savetxt(f, full_rows, fmt=row_fmt, newline='\n')
            
            # Write remaining values (if any) formatted and padded to 80 characters
            if remainder:
                rem_line = ''.join(fmt % val for val in values[n_full_rows*4:])
                f.write(rem_line.ljust(80) + "\n")
            
            return f"Success! ACE file written to: {filepath}"
    except Exception as e:
        raise IOError(f"Error writing ACE file: {e}")

def write_ace_ascii(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    """
    Write an Ace object to a formatted ASCII file.
    This is a convenience wrapper around write_ace.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to write
    filepath : str, optional
        The file path to write to. If None, will use the original filename 
        from which the ACE data was read, or create a name based on ZAID and temperature.
    overwrite : bool, optional
        Whether to overwrite an existing file, defaults to False
        
    Returns
    -------
    str
        A success message including the path to the written file
    """
    return write_ace(ace, filepath, overwrite)
