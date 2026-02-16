"""GAMINR — compute multigroup photoatomic cross sections."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    npend = p.get("npend", "")
    ngam1 = p.get("ngam1", 0)
    ngam2 = p.get("ngam2", "")

    mat_str = p.get("matb", "U235")
    matb = get_mat_number(mat_str)

    igg = int(p["igg"])
    iwt = int(p["iwt"])
    lord = int(p.get("lord", 0))

    # Validate unsupported iwt values
    if iwt == 1:
        raise ValueError(
            "iwt=1 (tabulated TAB1 weight function) is not supported. "
            "Use iwt=2 (constant) or iwt=3 (1/E + roll-offs)."
        )

    title = p.get("title", f"photoatomic data for {mat_str}")

    # Card 2: matb igg iwt lord [iprint]
    iprint = parse_iprint(p.get("iprint"))

    card2_parts = [str(matb), str(igg), str(iwt), str(lord)]
    if iprint is not None:
        card2_parts.append(str(iprint))

    lines: Lines = [
        "-- compute multigroup photoatomic cross sections",
        "gaminr",
        f"{nendf} {npend} {ngam1} {ngam2}",
        " ".join(card2_parts) + " /",
        f"'{title}'/",
    ]

    # Card 4: custom gamma group boundaries (only when igg=1)
    if igg == 1:
        egg = p.get("egg")
        if egg is None:
            raise ValueError(
                "igg=1 requires custom gamma group boundaries ('egg')."
            )
        if isinstance(egg, list):
            egg_values = [str(e) for e in egg]
        else:
            egg_values = str(egg).split()
        ngg = len(egg_values) - 1
        lines.append(f"{ngg} /")
        lines.append(" ".join(egg_values) + " /")

    # Card 6: reaction list
    reactions = p.get("reactions")
    if reactions:
        for rxn in reactions:
            mfd = rxn[0]
            parts = [str(mfd)]
            if len(rxn) > 1:
                mtd = rxn[1]
                parts.append(str(mtd))
                if len(rxn) > 2:
                    mtname = rxn[2]
                    parts.append(f"'{mtname}'")
            lines.append(" ".join(parts) + " /")
        lines.append("0 /")
    else:
        # Auto: process all reactions
        lines.append("-1 /")

    # Card 7: material terminator
    lines.append("0 /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse GAMINR card lines into a parameter dict."""
    # Card 1: nendf npend ngam1 ngam2
    c1 = parse_card_values(card_lines[0])
    # Card 2: matb igg iwt lord [iprint]
    c2 = parse_card_values(card_lines[1])
    # Card 3: 'title'/
    title = parse_quoted_string(card_lines[2])

    igg = int(c2[1])

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "ngam1": int(c1[2]),
        "ngam2": int(c1[3]),
        "matb": int(c2[0]),
        "igg": igg,
        "iwt": int(c2[2]),
        "lord": int(c2[3]),
        "title": title,
    }
    if len(c2) > 4:
        result["iprint"] = int(c2[4])

    idx = 3

    # Card 4: custom gamma group boundaries (igg=1)
    if igg == 1:
        ngg_vals = parse_card_values(card_lines[idx])
        result["ngg"] = int(ngg_vals[0])
        idx += 1
        result["egg"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1

    # Card 6: reactions (terminated by 0 / or -1 / for auto)
    reactions: list[list] = []
    while idx < len(card_lines):
        vals = parse_card_values(card_lines[idx])
        mfd = int(vals[0])
        idx += 1
        if mfd == 0:
            # Reaction terminator — next 0 / is material terminator
            break
        if mfd == -1:
            # Auto mode — no explicit reactions
            break
        rxn: list = [mfd]
        if len(vals) > 1:
            rxn.append(int(vals[1]))
            # Check for optional quoted name
            line_stripped = card_lines[idx - 1].strip()
            if "'" in line_stripped:
                mtname = parse_quoted_string(line_stripped)
                rxn.append(mtname)
        reactions.append(rxn)

    if reactions:
        result["reactions"] = reactions
    # Remaining 0 / is the material terminator — skipped
    return result
