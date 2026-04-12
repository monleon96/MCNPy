"""
ENDF section operations (remove sections).
"""
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils import parse_endf_id
from .update_directory import update_mf1_directory
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def remove_sections(
    content: str,
    sections: List[Tuple[int, Optional[int]]],
) -> Tuple[str, int]:
    """Remove MF/MT sections from ENDF content.

    Parameters
    ----------
    content : str
        Raw ENDF file content.
    sections : list of (MF, MT) tuples
        Sections to remove. MT=None means remove entire MF.

    Returns
    -------
    (modified_content, sections_removed_count)
    """
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Build set of specific (mf, mt) pairs to remove
    remove_specific: set = set()  # (mf, mt) pairs
    remove_whole_mf: set = set()  # mf values where MT=None

    for mf, mt in sections:
        if mt is None:
            remove_whole_mf.add(mf)
        else:
            remove_specific.add((mf, mt))

    # Protect MF1/MT451
    remove_specific.discard((1, 451))

    lines = content.splitlines(keepends=True)

    # Pre-pass: parse each line once and record the earliest index of a
    # surviving data line per MF. This replaces an inner kept_lines scan
    # (previously O(N²) across all SEND records) with an O(N) lookup.
    parsed: List[Optional[Tuple[int, int]]] = [None] * len(lines)
    first_surviving_idx: dict = {}
    for i, line in enumerate(lines):
        if len(line.rstrip('\n')) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf is None or mt is None:
            continue
        parsed[i] = (mf, mt)
        if mf > 0 and mt > 0:
            is_protected = (mf == 1 and mt == 451)
            will_survive = is_protected or not (
                mf in remove_whole_mf or (mf, mt) in remove_specific
            )
            if will_survive and mf not in first_surviving_idx:
                first_surviving_idx[mf] = i

    kept_lines = []
    sections_removed = set()

    for i, line in enumerate(lines):
        p = parsed[i]
        if p is None:
            kept_lines.append(line)
            continue
        mf, mt = p

        # Protect MF1/MT451
        if mf == 1 and mt == 451:
            kept_lines.append(line)
            continue

        should_remove = False

        if mf > 0 and mt > 0:
            # Data line
            if mf in remove_whole_mf or (mf, mt) in remove_specific:
                should_remove = True
                sections_removed.add((mf, mt))
        elif mf > 0 and mt == 0:
            # SEND line — drop if its MF is being fully removed, or if
            # no surviving data line for this MF appears before it
            # (matches the original kept_lines-scan semantics).
            if mf in remove_whole_mf:
                should_remove = True
            else:
                earliest = first_surviving_idx.get(mf)
                if earliest is None or earliest >= i:
                    should_remove = True

        if not should_remove:
            kept_lines.append(line)

    if not sections_removed:
        return content, 0

    modified_content = "".join(kept_lines)

    # Write to temp file and update MF1/MT451 directory
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".endf", delete=False, encoding="utf-8", newline=""
        ) as tmp:
            tmp.write(modified_content)
            tmp_path = tmp.name

        update_mf1_directory(tmp_path)

        with open(tmp_path, "r", encoding="utf-8", newline="") as f:
            modified_content = f.read()
    except Exception as e:
        logger.warning(f"Directory update after removal failed (non-fatal): {e}")
    finally:
        if tmp_path:
            p = Path(tmp_path)
            if p.exists():
                p.unlink()

    return modified_content, len(sections_removed)
