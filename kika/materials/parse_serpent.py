"""Parse Serpent material definitions from input files.

Serpent material format::

    mat <name> <density> [rgb R G B] [tmp T] [moder lib ZA] ...
        <zaid>.<lib> <fraction>
        <zaid>.<lib> <fraction>
        ...

Density convention:
    - Negative value: mass density in g/cc
    - Positive value: atomic density in atoms/b-cm
    - ``sum``: density calculated from isotope fractions
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .material import Material, MaterialCollection
from .parse_mcnp import _parse_nuclide_token


# Keywords that consume one or more arguments on the mat line
_SINGLE_ARG_KEYWORDS = {"tmp", "tms", "ettm", "burn", "vol", "coeft"}
_DOUBLE_ARG_KEYWORDS = {"tft", "fix"}
_TRIPLE_ARG_KEYWORDS = {"rgb"}
# moder consumes pairs: name ZA [name ZA ...]
_MODER_KEYWORD = "moder"
_COEFS_KEYWORD = "coefs"

# Keywords that indicate we're still on the mat header (not nuclide data)
_ALL_KEYWORDS = (
    _SINGLE_ARG_KEYWORDS | _DOUBLE_ARG_KEYWORDS | _TRIPLE_ARG_KEYWORDS
    | {_MODER_KEYWORD, _COEFS_KEYWORD}
)


def _strip_comment(line: str) -> str:
    """Remove Serpent ``%`` comments from a line."""
    idx = line.find("%")
    if idx >= 0:
        return line[:idx]
    return line


def _is_nuclide_line(tokens: List[str]) -> bool:
    """Check if tokens represent a nuclide entry (zaid fraction)."""
    if len(tokens) < 2:
        return False
    # First token should look like a ZAID (digits, optionally with .lib suffix)
    first = tokens[0]
    zaid_part = first.split(".")[0] if "." in first else first
    try:
        int(zaid_part)
    except ValueError:
        return False
    # Second token should be a number
    try:
        float(tokens[1])
    except ValueError:
        return False
    return True


def _read_serpent_material(
    lines: List[str], start_index: int, mat_id: int
) -> Tuple[Material, int]:
    """Parse a single Serpent material card starting at *start_index*.

    Parameters
    ----------
    lines : list of str
        All lines from the input file (raw, with newlines).
    start_index : int
        Index of the line containing the ``mat`` keyword.
    mat_id : int
        Numeric ID to assign to the Material object.

    Returns
    -------
    tuple of (Material, int)
        Parsed Material and the index of the first line *after* this material.
    """
    material = Material(id=mat_id)

    # Collect the mat header line (may span only one line in Serpent)
    header_line = _strip_comment(lines[start_index]).strip()
    tokens = header_line.split()

    # tokens[0] == "mat", tokens[1] == name, tokens[2] == density
    mat_name = tokens[1]
    material.name = mat_name

    # Parse density
    density_token = tokens[2]
    if density_token.lower() == "sum":
        material.metadata["serpent_density_sum"] = True
    else:
        density_val = float(density_token)
        if density_val < 0:
            material.set_density(abs(density_val), "g/cc")
        else:
            material.set_density(density_val, "atoms/b-cm")

    # Parse optional modifiers from the rest of the header tokens
    idx = 3
    while idx < len(tokens):
        tok = tokens[idx].lower()

        if tok in _SINGLE_ARG_KEYWORDS:
            if idx + 1 < len(tokens):
                val = float(tokens[idx + 1])
                if tok == "tmp":
                    material.set_temperature(val)
                elif tok in ("tms", "ettm"):
                    material.metadata["tms"] = val
                elif tok == "burn":
                    material.metadata["burn"] = int(val)
                elif tok == "vol":
                    material.metadata["vol"] = val
                elif tok == "coeft":
                    material.metadata["coeft"] = val
                idx += 2
            else:
                idx += 1
            continue

        if tok in _DOUBLE_ARG_KEYWORDS:
            if idx + 2 < len(tokens):
                val1 = tokens[idx + 1]
                val2 = float(tokens[idx + 2])
                if tok == "tft":
                    material.metadata["tft"] = (float(val1), val2)
                elif tok == "fix":
                    material.metadata["fix"] = (val1, val2)
                idx += 3
            else:
                idx += 1
            continue

        if tok in _TRIPLE_ARG_KEYWORDS:
            if idx + 3 < len(tokens):
                if tok == "rgb":
                    r, g, b = int(tokens[idx + 1]), int(tokens[idx + 2]), int(tokens[idx + 3])
                    material.metadata["rgb"] = (r, g, b)
                idx += 4
            else:
                idx += 1
            continue

        if tok == _MODER_KEYWORD:
            moder_list = material.metadata.get("moder", [])
            if idx + 2 < len(tokens):
                lib_name = tokens[idx + 1]
                za = int(tokens[idx + 2])
                moder_list.append((lib_name, za))
                material.metadata["moder"] = moder_list
                idx += 3
            else:
                idx += 1
            continue

        if tok == _COEFS_KEYWORD:
            if idx + 1 < len(tokens):
                material.metadata["coefs"] = tokens[idx + 1]
                idx += 2
            else:
                idx += 1
            continue

        # Unknown token on the mat line — skip
        idx += 1

    # Now read nuclide lines
    i = start_index + 1
    inferred_fraction_type: Optional[str] = None

    while i < len(lines):
        raw = _strip_comment(lines[i]).strip()

        if not raw:
            i += 1
            continue

        line_tokens = raw.split()

        # Stop if we hit another card keyword
        if line_tokens[0].lower() in ("mat", "surf", "cell", "det", "src",
                                       "set", "ene", "sens", "plot", "mesh",
                                       "pin", "lat", "nest", "pbed", "umsh",
                                       "trans", "utrans", "strans",
                                       "ifc", "div", "dep", "include",
                                       "therm", "branch", "coef"):
            break

        if not _is_nuclide_line(line_tokens):
            # Could be continuation of unknown card, skip
            i += 1
            continue

        # Parse nuclide entries (possibly multiple per line)
        j = 0
        while j + 1 < len(line_tokens):
            try:
                zaid, lib_suffix = _parse_nuclide_token(line_tokens[j])
            except ValueError:
                j += 1
                continue

            try:
                raw_fraction = float(line_tokens[j + 1])
            except ValueError:
                j += 1
                continue

            current_type = "wo" if raw_fraction < 0.0 else "ao"
            if inferred_fraction_type is None:
                inferred_fraction_type = current_type
            elif inferred_fraction_type != current_type:
                raise ValueError(
                    f"Material '{mat_name}' has mixed atomic/weight fractions"
                )

            material.add_nuclide(
                nuclide=zaid,
                fraction=abs(raw_fraction),
                fraction_type=current_type,
                library=lib_suffix,
            )
            j += 2

        i += 1

    material.fraction_type = inferred_fraction_type or "ao"
    return material, i


def _find_material_spans(
    lines: List[str],
) -> List[Tuple[str, int, int]]:
    """Find (name, start_line, end_line) spans of all materials in the file.

    Used by ``write_to_serpent`` for targeted replacement.
    """
    spans: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue
        tokens = stripped.split()
        if tokens[0].lower() == "mat" and len(tokens) >= 3:
            mat_name = tokens[1]
            start = i
            # Find where this material ends
            _, end = _read_serpent_material(lines, i, mat_id=0)
            spans.append((mat_name, start, end))
            i = end
        else:
            i += 1
    return spans


def read_serpent_materials(
    source: Union[str, Path, List[str]],
) -> MaterialCollection:
    """Parse all materials from a Serpent input file.

    Parameters
    ----------
    source : str, Path, or list of str
        File path or list of lines to parse.

    Returns
    -------
    MaterialCollection
        Collection of parsed materials with auto-assigned numeric IDs.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r") as f:
            lines = f.readlines()
    else:
        lines = list(source)

    collection = MaterialCollection()
    mat_id = 1
    i = 0

    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        tokens = stripped.split()
        if tokens[0].lower() == "mat" and len(tokens) >= 3:
            material, next_i = _read_serpent_material(lines, i, mat_id)
            collection.add_material(material)
            mat_id += 1
            i = next_i
        else:
            i += 1

    return collection
