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
    """Insert or replace an MF section in an ENDF file.

    Uses ``source_endf`` as a template; either replaces the existing block for
    ``section._mf`` or inserts a new one immediately before the MEND marker,
    appends the FEND marker, and (optionally) refreshes the MF1/MT451
    directory.

    Parameters
    ----------
    source_endf : str
        Path to the source ENDF file used as a template.
    section : object
        Section object exposing ``_mf``, ``number``, ``_mat`` and ``__str__``.
    output_path : str
        Destination ENDF file path.
    replace_existing : bool, default True
        Replace any existing block for this MF.  Raises ``FileExistsError`` if
        False and the section is already present.
    update_directory : bool, default True
        Refresh the MF1/MT451 directory after writing.
    """
    mf_number = int(section._mf)
    mt_number = int(section.number)

    with open(source_endf, 'r') as f:
        lines = f.readlines()

    start, end = _find_mf_boundaries(lines, mf_number)
    has_section = start is not None

    if has_section and not replace_existing:
        raise FileExistsError(
            f"MF{mf_number} already exists in {source_endf}. "
            f"Set replace_existing=True to replace it."
        )

    content = str(section)
    section_lines = [line + '\n' for line in content.split('\n') if line.strip()]

    from ..utils import format_endf_data_line, ENDF_FORMAT_INT
    mat_num = section._mat or 0
    fend_line = format_endf_data_line(
        [0, 0, 0, 0, 0, 0], mat_num, 0, 0, 0,
        formats=[ENDF_FORMAT_INT] * 6
    ) + '\n'
    section_lines.append(fend_line)

    if has_section:
        skip_end = end
        if skip_end < len(lines) and len(lines[skip_end]) >= 75:
            try:
                old_mf = int(lines[skip_end][70:72].strip() or '0')
                old_mt = int(lines[skip_end][72:75].strip() or '0')
                if old_mf == 0 and old_mt == 0:
                    skip_end += 1
            except ValueError:
                pass
        new_lines = lines[:start] + section_lines + lines[skip_end:]
    else:
        insert_idx = _find_mend_marker(lines)
        new_lines = lines[:insert_idx] + section_lines + lines[insert_idx:]

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(output_path, added_sections={(mf_number, mt_number)})

    return output_path
