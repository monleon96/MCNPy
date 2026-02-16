"""RECONR — reconstruct, linearise and unionize cross-section data."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    npend = p.get("npend", "")

    isotope = p.get("mat", "U235")
    mat = get_mat_number(isotope)

    try:
        tempr = float(p["tempr"]) if p.get("tempr") is not None else 0.0
    except (ValueError, TypeError):
        tempr = 0.0

    label = f"reconstructed data for {isotope} @ {tempr} K"

    tolerance = float(p.get("err", "0.001"))
    user_errmax = p.get("errmax")
    user_errint = p.get("errint")

    # Build Card 4 with dependencies
    card4_parts = [str(tolerance)]
    if user_errint is not None:
        if user_errmax is None:
            user_errmax = 10 * tolerance
        card4_parts.extend([str(p.get("tempr", 0.0)), str(user_errmax), str(user_errint)])
    elif user_errmax is not None:
        card4_parts.extend([str(p.get("tempr", 0.0)), str(user_errmax)])
    elif p.get("tempr") is not None:
        card4_parts.append(str(tempr))

    card4_line = " ".join(card4_parts) + " /"

    return [
        "-- reconstruct, linearise and unionize data",
        "reconr",
        f"{nendf} {npend}",
        f"'{label}'/",
        f"{mat} /",
        card4_line,
        "0 /",
    ]


def parse(card_lines: list[str]) -> dict:
    """Parse RECONR card lines into a parameter dict.

    Handles single and multi-material decks.  The first material's
    parameters are promoted to top-level keys for generator compatibility.
    """
    # Card 1: nendf npend
    c1 = parse_card_values(card_lines[0])
    # Card 2: 'label'/
    label = parse_quoted_string(card_lines[1])

    # Cards 3+4 repeated per material, terminated by 0 /
    # Card 3: mat [ncards] /   (ncards = number of user title cards after Card 4)
    # Card 4: err [tempr] [errmax] [errint] /
    # Cards 5..5+ncards-1: user title cards (quoted strings), skipped
    materials: list[dict] = []
    idx = 2
    while idx < len(card_lines):
        c3 = parse_card_values(card_lines[idx])
        mat = int(c3[0])
        if mat == 0:
            break
        ncards = int(c3[1]) if len(c3) > 1 else 0
        idx += 1
        c4 = parse_card_values(card_lines[idx])
        mat_info: dict = {"mat": mat, "err": float(c4[0])}
        if len(c4) > 1:
            mat_info["tempr"] = float(c4[1])
        if len(c4) > 2:
            mat_info["errmax"] = float(c4[2])
        if len(c4) > 3:
            mat_info["errint"] = float(c4[3])
        materials.append(mat_info)
        idx += 1
        # Skip ncards user title lines
        idx += ncards

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "label": label,
    }
    if materials:
        result.update(materials[0])
    if len(materials) > 1:
        result["materials"] = materials
    return result
