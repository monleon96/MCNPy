import re
from typing import List, Optional, Tuple

from .material import Material


def _parse_nuclide_token(token: str) -> Tuple[int, Optional[str]]:
    """Return (zaid, library_suffix) parsed from token like '1001.80c' or '1001'."""
    if "." in token:
        nuclide_parts = token.split(".")
        return int(nuclide_parts[0]), nuclide_parts[1]
    return int(token), None


def _parse_kika_name_comment(line: str) -> Optional[str]:
    """Parse a KIKA material name comment line.

    Expected format: 'c KIKA_MAT_NAME: Material Name Here'

    Parameters
    ----------
    line : str
        Comment line to parse.

    Returns
    -------
    str or None
        Material name if found, None otherwise.
    """
    stripped = line.strip()
    if stripped.lower().startswith("c kika_mat_name:"):
        # Extract the name after the marker
        name_part = stripped[16:].strip()  # len("c KIKA_MAT_NAME:") = 16
        if name_part:
            return name_part
    return None


def _parse_kika_density_comment(line: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a KIKA density comment from a material line.

    Expected format in inline comment: '$ KIKA_DENSITY: X.XXe+XX g/cc, Y.YYe+YY atoms/b-cm'
    or: '$ KIKA_DENSITY: X.XXe+XX g/cc'

    Parameters
    ----------
    line : str
        Line containing the density comment.

    Returns
    -------
    tuple of (float or None, float or None)
        (mass_density_gcc, atomic_density_abc) or (None, None) if not found.
    """
    if "KIKA_DENSITY:" not in line:
        return None, None

    # Find the KIKA_DENSITY part
    match = re.search(r'KIKA_DENSITY:\s*([\d.eE+-]+)\s*g/cc(?:,\s*([\d.eE+-]+)\s*atoms/b-cm)?', line)
    if match:
        mass_density = float(match.group(1))
        atomic_density = float(match.group(2)) if match.group(2) else None
        return mass_density, atomic_density

    return None, None


def read_material(lines: List[str], start_index: int) -> Tuple[Optional[Material], int]:
    """Read and parse a material card from MCNP input lines.

    Handles KIKA-formatted comments for material name and density:
    - 'c KIKA_MAT_NAME: <name>' comment before material card
    - '$ KIKA_DENSITY: <mass> g/cc, <atomic> atoms/b-cm' inline comment
    - '$ <SYMBOL>' inline comments after nuclide fractions
    """
    i = start_index

    # Check for KIKA material name comment in preceding lines
    material_name: Optional[str] = None

    # Look backwards for a KIKA_MAT_NAME comment (at most 3 lines back)
    for back_offset in range(1, min(4, start_index + 1)):
        prev_idx = start_index - back_offset
        if prev_idx < 0:
            break
        prev_line = lines[prev_idx].strip()
        # Stop if we hit another material or non-comment content
        if prev_line and not prev_line.lower().startswith('c ') and not prev_line.lower().startswith('c\t'):
            break
        name = _parse_kika_name_comment(prev_line)
        if name is not None:
            material_name = name
            break

    line = lines[i].strip()
    card_lines: List[str] = []
    original_first_line = line  # Keep for density parsing

    while line.endswith("&"):
        card_lines.append(line.rstrip("&").strip())
        i += 1
        if i < len(lines):
            line = lines[i].strip()
        else:
            break
    card_lines.append(line)
    full_line = " ".join(card_lines)

    # Parse density from the first line before stripping comments
    mass_density, atomic_density = _parse_kika_density_comment(original_first_line)

    if "$" in full_line:
        full_line = full_line.split("$")[0].strip()

    tokens = full_line.split()
    match = re.match(r"^m(\d+)", tokens[0])
    if not match:
        return None, i + 1

    material_id = int(match.group(1))
    material_obj = Material(id=material_id)

    # Set name if found
    if material_name:
        material_obj.name = material_name

    # Set density if found
    if mass_density is not None:
        material_obj.set_density(mass_density, "g/cc")

    inferred_fraction_type: Optional[str] = None
    idx = 1
    while idx < len(tokens):
        tok = tokens[idx]
        lower_tok = tok.lower()

        if lower_tok.startswith("nlib="):
            material_obj.libs["nlib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("plib="):
            material_obj.libs["plib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("ylib="):
            material_obj.libs["ylib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("lib="):
            lib_value = tok.split("=", 1)[1]
            lib_key = material_obj._infer_lib_key_from_suffix(lib_value)
            if lib_key is not None:
                material_obj.libs[lib_key] = lib_value
            idx += 1
            continue

        # At this point we should be reading ZAID/fraction pairs.
        nuclide_spec = tok
        try:
            zaid, specific_lib = _parse_nuclide_token(nuclide_spec)
        except ValueError:
            idx += 1
            continue

        if idx + 1 >= len(tokens):
            break

        try:
            raw_fraction = float(tokens[idx + 1])
        except ValueError:
            idx += 1
            continue

        current_type = "wo" if raw_fraction < 0.0 else "ao"
        if inferred_fraction_type is None:
            inferred_fraction_type = current_type
        elif inferred_fraction_type != current_type:
            raise ValueError(f"Material {material_id} has mixed atomic/weight fractions")

        material_obj.add_nuclide(nuclide=zaid, fraction=abs(raw_fraction),
                                  fraction_type=current_type, library=specific_lib)
        idx += 2

    i += 1
    last_data_line = i  # Track the last line with actual material data

    while i < len(lines):
        line = lines[i].strip()

        if line.lower().startswith("c ") or line.lower() == "c" or not line:
            i += 1
            continue

        if "$" in line:
            line = line.split("$")[0].strip()

        if not line:
            i += 1
            continue

        if not re.match(r"^\s*\d+", line):
            break

        if re.match(r"^m\d+", line):
            break

        parts = line.split()
        idx = 0
        has_data = False  # Track if this line has material data

        while idx < len(parts):
            nuclide_spec = parts[idx]
            try:
                zaid, specific_lib = _parse_nuclide_token(nuclide_spec)
            except ValueError:
                idx += 1
                continue

            if idx + 1 >= len(parts):
                break

            try:
                raw_fraction = float(parts[idx + 1])
            except ValueError:
                idx += 1
                continue

            current_type = "wo" if raw_fraction < 0.0 else "ao"
            if inferred_fraction_type is None:
                inferred_fraction_type = current_type
            elif inferred_fraction_type != current_type:
                raise ValueError(f"Material {material_id} has mixed atomic/weight fractions")

            material_obj.add_nuclide(nuclide=zaid, fraction=abs(raw_fraction),
                                      fraction_type=current_type, library=specific_lib)
            has_data = True
            idx += 2

        i += 1
        if has_data:
            last_data_line = i  # Update to current position after processing data line

    material_obj.fraction_type = inferred_fraction_type or "ao"
    return material_obj, last_data_line
