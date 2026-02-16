"""MATXSR — format multigroup cross sections into a MATXS library."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def _parse_word_list(line: str) -> list[str]:
    """Split unquoted space-separated names from a card line (strip ``/``)."""
    line = line.strip()
    if line.endswith("/"):
        line = line[:-1].strip()
    return line.split()


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    ngen1 = p["ngen1"]
    ngen2 = int(p.get("ngen2", 0))
    nmatx = p["nmatx"]
    ngen3 = int(p.get("ngen3", 0))
    ngen4 = int(p.get("ngen4", 0))
    ngen5 = int(p.get("ngen5", 0))
    ngen6 = int(p.get("ngen6", 0))
    ngen7 = int(p.get("ngen7", 0))
    ngen8 = int(p.get("ngen8", 0))

    ivers = int(p.get("ivers", 0))
    huse = p.get("huse", "")

    hsetid = p.get("hsetid", ["matxs library"])
    particles = p["particles"]
    data_types = p["data_types"]
    materials = p["materials"]

    npart = len(particles)
    ntype = len(data_types)
    nholl = len(hsetid)
    nmat = len(materials)

    # Card 1: tape units with trailing zeros trimmed
    tapes = [ngen1, ngen2, nmatx, ngen3, ngen4, ngen5, ngen6, ngen7, ngen8]
    while len(tapes) > 3 and tapes[-1] == 0:
        tapes.pop()
    card1 = " ".join(str(t) for t in tapes) + " /"

    # Card 2: ivers 'huse'/
    card2 = f"{ivers} '{huse}'/"

    # Card 3: npart ntype nholl nmat /
    card3 = f"{npart} {ntype} {nholl} {nmat} /"

    lines: Lines = [
        "-- format multigroup cross sections into matxs library",
        "matxsr",
        card1,
        card2,
        card3,
    ]

    # Card 4: 'hsetid'/ — repeated nholl times
    for h in hsetid:
        lines.append(f"'{h}'/")

    # Card 5: particle names (unquoted)
    lines.append(" ".join(pt["name"] for pt in particles) + " /")

    # Card 6: ngrp per particle
    lines.append(" ".join(str(pt["ngrp"]) for pt in particles) + " /")

    # Card 7: data type names (unquoted)
    lines.append(" ".join(dt["name"] for dt in data_types) + " /")

    # Card 8: jinp per data type
    lines.append(" ".join(str(dt["jinp"]) for dt in data_types) + " /")

    # Card 9: joutp per data type
    lines.append(" ".join(str(dt["joutp"]) for dt in data_types) + " /")

    # Card 10: hmat matno [matgg] / — repeated nmat times
    for mat in materials:
        hmat = mat["hmat"]
        matno = mat["matno"]
        if isinstance(matno, str):
            matno = get_mat_number(matno)
        matgg = mat.get("matgg", 0)
        if matgg:
            lines.append(f"{hmat} {matno} {matgg} /")
        else:
            lines.append(f"{hmat} {matno} /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse MATXSR card lines into a parameter dict."""
    # Card 1: ngen1 ngen2 nmatx [ngen3..ngen8]
    c1 = parse_card_values(card_lines[0])
    result: dict = {
        "ngen1": int(c1[0]),
        "ngen2": int(c1[1]) if len(c1) > 1 else 0,
        "nmatx": int(c1[2]) if len(c1) > 2 else 0,
    }
    gen_keys = ["ngen3", "ngen4", "ngen5", "ngen6", "ngen7", "ngen8"]
    for i, key in enumerate(gen_keys):
        if len(c1) > 3 + i:
            result[key] = int(c1[3 + i])

    idx = 1

    # Card 2: ivers 'huse'/
    card2_line = card_lines[idx].strip()
    quote_pos = card2_line.index("'")
    ivers_str = card2_line[:quote_pos].strip()
    result["ivers"] = int(ivers_str) if ivers_str else 0
    result["huse"] = parse_quoted_string(card2_line)
    idx += 1

    # Card 3: npart ntype nholl nmat
    c3 = parse_card_values(card_lines[idx])
    npart = int(c3[0])
    ntype = int(c3[1])
    nholl = int(c3[2])
    nmat = int(c3[3])
    idx += 1

    # Card 4: 'hsetid'/ — repeated nholl times
    hsetid = []
    for _ in range(nholl):
        hsetid.append(parse_quoted_string(card_lines[idx]))
        idx += 1
    result["hsetid"] = hsetid

    # Card 5: particle names (unquoted)
    pnames = _parse_word_list(card_lines[idx])
    idx += 1

    # Card 6: ngrp per particle
    ngrp_vals = parse_card_values(card_lines[idx])
    idx += 1

    particles = []
    for i in range(npart):
        particles.append({"name": pnames[i], "ngrp": int(ngrp_vals[i])})
    result["particles"] = particles

    # Card 7: data type names (unquoted)
    tnames = _parse_word_list(card_lines[idx])
    idx += 1

    # Card 8: jinp per data type
    jinp_vals = parse_card_values(card_lines[idx])
    idx += 1

    # Card 9: joutp per data type
    joutp_vals = parse_card_values(card_lines[idx])
    idx += 1

    data_types = []
    for j in range(ntype):
        data_types.append({
            "name": tnames[j],
            "jinp": int(jinp_vals[j]),
            "joutp": int(joutp_vals[j]),
        })
    result["data_types"] = data_types

    # Card 10: hmat matno [matgg] / — repeated nmat times
    materials = []
    for _ in range(nmat):
        vals = parse_card_values(card_lines[idx])
        mat_entry: dict = {"hmat": vals[0], "matno": int(vals[1])}
        if len(vals) > 2:
            mat_entry["matgg"] = int(vals[2])
        materials.append(mat_entry)
        idx += 1
    result["materials"] = materials

    return result
