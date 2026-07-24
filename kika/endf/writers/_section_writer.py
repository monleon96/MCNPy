"""Generic ENDF MF-section insert-or-replace file writer.

Extracted from the MF34 writer so MF33 — and any future MFxx covariance
section that serializes via ``str(section)`` — can reuse the same
insert-before-MEND / replace-existing logic and MF1/451 directory refresh
instead of duplicating it.

A "section" here is any object that exposes ``_mf`` (file number), ``number``
(MT number), ``_mat`` (material number), and a ``__str__`` that emits the
section body without the trailing FEND marker (both ``MF34MT`` and ``MF33MT``
satisfy this).
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def _find_mf_boundaries(
    lines: List[str], mf_number: int
) -> Tuple[Optional[int], Optional[int]]:
    """Locate the given MF block; return (start_idx, end_idx) or (None, None).

    ``end_idx`` is one past the last line of the block (slice-friendly).
    """
    start = None
    end = None
    for i, line in enumerate(lines):
        if len(line) >= 75:
            try:
                mf = int(line[70:72].strip() or '0')
            except ValueError:
                continue
            if mf == mf_number:
                if start is None:
                    start = i
                end = i + 1
    return start, end


def _parse_mf_mt(line: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse the (MF, MT) id columns of an ENDF line; (None, None) if unparsable."""
    if len(line) < 75:
        return None, None
    try:
        return (
            int(line[70:72].strip() or '0'),
            int(line[72:75].strip() or '0'),
        )
    except ValueError:
        return None, None


def _find_mt_section_boundaries(
    lines: List[str], mf_number: int, mt_number: int
) -> Tuple[Optional[int], Optional[int]]:
    """Locate one (MF, MT) section incl. its trailing SEND; (start, end) or (None, None).

    ``end`` is one past the section's SEND marker (slice-friendly).  Sections of
    one MT are contiguous in a valid ENDF file, so the scan stops at the first
    non-matching line after the section (including that line when it is the
    section's SEND: same MF, MT=0).
    """
    start = None
    end = None
    for i, line in enumerate(lines):
        mf, mt = _parse_mf_mt(line)
        if mf is None:
            continue
        if mf == mf_number and mt == mt_number:
            if start is None:
                start = i
            end = i + 1
        elif start is not None:
            if mf == mf_number and mt == 0:
                end = i + 1  # the section's SEND marker
            break
    return start, end


def _find_mt_insertion_point(
    lines: List[str], mf_number: int, mt_number: int, block_start: int, block_end: int
) -> int:
    """Index where a new MT section keeps ascending MT order inside an MF block.

    Returns the HEAD-line index of the first existing section with MT greater
    than ``mt_number``, or ``block_end`` (just before the block's FEND) when the
    new section belongs last.
    """
    current_mt: Optional[int] = None
    for i in range(block_start, block_end):
        mf, mt = _parse_mf_mt(lines[i])
        if mf != mf_number or mt is None:
            continue
        if mt == 0:
            current_mt = None
        elif mt != current_mt:
            if mt > mt_number:
                return i
            current_mt = mt
    return block_end


def _find_mend_marker(lines: List[str]) -> int:
    """Find the line index of the MEND marker (MAT=0, MF=0, MT=0)."""
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if len(line) >= 75:
            try:
                mat = int(line[66:70].strip() or '0')
                mf = int(line[70:72].strip() or '0')
                mt = int(line[72:75].strip() or '0')
                if mat == 0 and mf == 0 and mt == 0:
                    return i
            except ValueError:
                continue
    return len(lines)


def write_mf_section_to_file(
    source_endf: str,
    section,
    output_path: str,
    *,
    replace_existing: bool = True,
    update_directory: bool = True,
) -> str:
    """Insert or replace one (MF, MT) section in an ENDF file.

    Uses ``source_endf`` as a template.  When the file already has a block for
    ``section._mf``, only the target MT section is replaced (or inserted in
    ascending MT order) — sibling MT sections and the block FEND survive
    byte-identically.  When the MF block is absent entirely, the section plus a
    fresh FEND are inserted before the MEND marker.  Optionally refreshes the
    MF1/MT451 directory (which recounts every section from the file, so a
    multi-MT block stays consistent).

    Parameters
    ----------
    source_endf : str
        Path to the source ENDF file used as a template.
    section : object
        Section object exposing ``_mf``, ``number``, ``_mat`` and ``__str__``.
    output_path : str
        Destination ENDF file path.
    replace_existing : bool, default True
        Replace an existing section for this (MF, MT).  Raises
        ``FileExistsError`` if False and the section is already present.
    update_directory : bool, default True
        Refresh the MF1/MT451 directory after writing.
    """
    mf_number = int(section._mf)
    mt_number = int(section.number)

    with open(source_endf, 'r') as f:
        lines = f.readlines()

    block_start, block_end = _find_mf_boundaries(lines, mf_number)
    has_block = block_start is not None

    # ``str(section)`` emits the section body incl. its own SEND, no FEND.
    content = str(section)
    section_lines = [line + '\n' for line in content.split('\n') if line.strip()]

    if has_block:
        # Splice only the target (MF, MT) section; sibling MT sections and the
        # block's FEND are preserved byte-identically.
        sec_start, sec_end = _find_mt_section_boundaries(lines, mf_number, mt_number)
        if sec_start is not None:
            if not replace_existing:
                raise FileExistsError(
                    f"MF{mf_number} MT{mt_number} already exists in {source_endf}. "
                    f"Set replace_existing=True to replace it."
                )
            new_lines = lines[:sec_start] + section_lines + lines[sec_end:]
        else:
            insert_idx = _find_mt_insertion_point(
                lines, mf_number, mt_number, block_start, block_end
            )
            new_lines = lines[:insert_idx] + section_lines + lines[insert_idx:]
    else:
        # No block for this MF yet: insert section + fresh FEND before MEND.
        from ..utils import format_endf_data_line, ENDF_FORMAT_INT
        mat_num = section._mat or 0
        fend_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0], mat_num, 0, 0, 0,
            formats=[ENDF_FORMAT_INT] * 6
        ) + '\n'
        insert_idx = _find_mend_marker(lines)
        new_lines = (
            lines[:insert_idx] + section_lines + [fend_line] + lines[insert_idx:]
        )

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(output_path, added_sections={(mf_number, mt_number)})

    return output_path
