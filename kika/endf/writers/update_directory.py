"""
MF1/MT451 directory update utility.

After modifying ENDF files (replacing/adding/removing MF or MT sections),
the MF1/MT451 directory must be updated to reflect the new line counts.
This module provides a standalone function to scan an ENDF file and
rebuild the directory from scratch.
"""
from typing import Dict, List, Optional, Set, Tuple
from collections import OrderedDict

from ..parsers.parse_mf1 import parse_mt451
from ..utils import parse_endf_id
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def update_mf1_directory(filepath: str, added_sections: Optional[Set[Tuple[int, int]]] = None) -> bool:
    """
    Scan an ENDF file and rebuild the MF1/MT451 directory to match actual content.

    This function reads the file, counts data lines per (MF, MT) section,
    rebuilds the directory entries with correct NC (line count) values,
    and rewrites the MF1/MT451 section in place.

    Only sections that were already in the original directory (or are
    explicitly listed in *added_sections*) are included.  This prevents
    orphan data that happens to exist in the file from being promoted
    into the directory.

    Parameters
    ----------
    filepath : str
        Path to the ENDF file to update.
    added_sections : set of (MF, MT) tuples, optional
        Sections that were intentionally added and should appear in the
        directory even if they were not present before.

    Returns
    -------
    bool
        True if the directory was successfully updated, False if MF1/MT451
        was not found or an error occurred.
    """
    try:
        with open(filepath, 'r', newline='') as f:
            lines = f.readlines()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Cannot read file for directory update: {e}")
        return False

    # --- Step 1: Find MF1/MT451 boundaries ---
    mt451_start = None
    mt451_end = None  # exclusive (first line AFTER MT451 + its SEND record)

    for i, line in enumerate(lines):
        mat, mf, mt = parse_endf_id(line)
        if mf == 1 and mt == 451:
            if mt451_start is None:
                mt451_start = i
        elif mt451_start is not None and mt451_end is None:
            if mf == 1 and mt == 0:
                # SEND record(s) for MT451 — consume all consecutive
                # MF=1/MT=0 lines so the serialized MT451 (which
                # includes its own SEND) replaces them cleanly.
                j = i
                while j < len(lines):
                    _, mf_j, mt_j = parse_endf_id(lines[j])
                    if mf_j == 1 and mt_j == 0:
                        j += 1
                    else:
                        break
                mt451_end = j
            else:
                mt451_end = i
            break

    if mt451_start is None:
        logger.warning("No MF1/MT451 section found — cannot update directory")
        return False
    if mt451_end is None:
        mt451_end = len(lines)

    # --- Step 2: Parse existing MT451 to preserve header and text ---
    mt451_lines = [line.rstrip('\n').rstrip('\r') for line in lines[mt451_start:mt451_end]]
    mt451 = parse_mt451(mt451_lines)

    nwd = mt451._nwd if mt451._nwd is not None else 0

    # --- Step 3: Build the set of (MF, MT) pairs allowed in the directory ---
    # Start from what was already in the old directory, plus any sections
    # the caller explicitly added.  This prevents orphan data from being
    # promoted into the directory.
    old_keys: Set[Tuple[int, int]] = set()
    for mf_val, mt_val, nc_val, mod_val in mt451._directory:
        old_keys.add((int(mf_val), int(mt_val)))
    allowed_keys = old_keys | (added_sections or set())

    # --- Step 4: Count data lines only for allowed (MF, MT) pairs ---
    line_counts: Dict[Tuple[int, int], int] = OrderedDict()

    for line in lines:
        mat, mf, mt = parse_endf_id(line)
        if mf is not None and mt is not None and mf > 0 and mt > 0:
            key = (mf, mt)
            if key in allowed_keys:
                line_counts[key] = line_counts.get(key, 0) + 1

    # Remove entries that no longer have data in the file
    sorted_keys = sorted(k for k in line_counts.keys() if line_counts[k] > 0)

    # --- Step 5: Build old MOD lookup ---
    old_mod: Dict[Tuple[int, int], int] = {}
    for mf_val, mt_val, nc_val, mod_val in mt451._directory:
        old_mod[(int(mf_val), int(mt_val))] = int(mod_val)

    # --- Step 6: Build new directory ---
    nxc_new = len(sorted_keys)

    new_directory: List[Tuple[int, int, int, int]] = []
    for mf_val, mt_val in sorted_keys:
        if mf_val == 1 and mt_val == 451:
            # Self-referential: NC = 4 header + NWD text + NXC directory entries
            nc = 4 + nwd + nxc_new
        else:
            nc = line_counts[(mf_val, mt_val)]
        mod = old_mod.get((mf_val, mt_val), 0)
        new_directory.append((mf_val, mt_val, nc, mod))

    # --- Step 7: Update MT451 fields ---
    mt451._nxc = nxc_new
    mt451._directory = new_directory

    # --- Step 8: Serialize and replace in file ---
    # Reuse the terminator the file already uses. `lines` came from a
    # newline='' read so they carry their own, and hardcoding '\n' here
    # rewrote the MF1/451 block of a CRLF tape with bare LF — 911 lines of a
    # JEFF-4.0 Fe-56 tape silently changed by an operation that is supposed to
    # be a no-op when the directory already agrees with the content.
    replaced = lines[mt451_start:mt451_end] or lines
    terminator = '\r\n' if replaced and replaced[0].endswith('\r\n') else '\n'
    new_mt451_str = str(mt451)
    new_mt451_lines = [line + terminator for line in new_mt451_str.split('\n') if line]

    new_file_lines = lines[:mt451_start] + new_mt451_lines + lines[mt451_end:]

    with open(filepath, 'w', newline='') as f:
        f.writelines(new_file_lines)

    logger.debug(
        f"Updated MF1/MT451 directory: {nxc_new} entries "
        f"(was {len(mt451._directory)} before rebuild)"
    )
    return True
