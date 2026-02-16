"""RESXSR — generate resonance cross section files (RESXS format)."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nout = p.get("nout", "")

    materials = p.get("materials", [])
    nmat = len(materials)
    comments = p.get("comments", [])
    nholl = len(comments)

    maxt = int(p.get("maxt", 1))
    efirst = p.get("efirst", 4.0)
    elast = p.get("elast", 500.0)
    eps = p.get("eps", 0.001)

    huse = p.get("huse", "NJOY")
    ivers = int(p.get("ivers", 0))

    lines: Lines = [
        "-- generate resonance cross section file",
        "resxsr",
        # Card 1
        f"{nout} /",
        # Card 2
        f"{nmat} {maxt} {nholl} {efirst} {elast} {eps} /",
        # Card 3
        f"'{huse}' {ivers} /",
    ]

    # Card 4: hollerith comment lines (nholl times)
    for comment in comments:
        lines.append(f"'{comment}'/")

    # Card 5: material entries (nmat times)
    for m in materials:
        name = m.get("name", m.get("isotope", ""))
        mat = m.get("mat")
        if mat is None:
            mat = get_mat_number(name)
        unit = m.get("unit", "")
        lines.append(f"'{name}' {mat} {unit} /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse RESXSR card lines into a parameter dict."""
    # Card 1: nout
    c1 = parse_card_values(card_lines[0])
    nout = int(c1[0])

    # Card 2: nmat maxt nholl efirst elast eps
    c2 = parse_card_values(card_lines[1])
    nmat = int(c2[0])
    maxt = int(c2[1])
    nholl = int(c2[2])
    efirst = float(c2[3])
    elast = float(c2[4])
    eps = float(c2[5])

    # Card 3: 'huse' ivers
    huse = parse_quoted_string(card_lines[2])
    c3_rest = parse_card_values(card_lines[2].split("'")[-1])
    ivers = int(c3_rest[0]) if c3_rest else 0

    idx = 3

    # Card 4: hollerith comments (nholl times)
    comments: list[str] = []
    for _ in range(nholl):
        comments.append(parse_quoted_string(card_lines[idx]))
        idx += 1

    # Card 5: materials (nmat times)
    materials: list[dict] = []
    for _ in range(nmat):
        name = parse_quoted_string(card_lines[idx])
        rest = parse_card_values(card_lines[idx].split("'")[-1])
        mat = int(rest[0])
        unit = int(rest[1]) if len(rest) > 1 else 0
        materials.append({"name": name, "mat": mat, "unit": unit})
        idx += 1

    return {
        "nout": nout,
        "nmat": nmat,
        "maxt": maxt,
        "nholl": nholl,
        "efirst": efirst,
        "elast": elast,
        "eps": eps,
        "huse": huse,
        "ivers": ivers,
        "comments": comments,
        "materials": materials,
    }
