"""BROADR — Doppler-broaden cross-section data."""

from __future__ import annotations

from ._base import Lines, parse_card_values


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    nin = p.get("nin", "")
    nout = p.get("nout", "")

    mat_str = p.get("mat", "U235")
    mat_num = get_mat_number(mat_str)
    temp2_str = str(p.get("temp2", ""))
    temps = temp2_str.split()
    ntemp2 = len(temps)

    errthn = float(p.get("errthn", "0.001"))
    user_thnmax = p.get("thnmax")
    user_errmax = p.get("errmax")
    user_errint = p.get("errint")

    # Card 2 new optional fields
    user_istart = p.get("istart")
    user_istrap = p.get("istrap")
    user_temp1 = p.get("temp1")

    option_map = {"no": 0, "yes": 1}

    # Build Card 2 with positional dependencies
    card2_parts = [str(mat_num), str(ntemp2)]
    if user_temp1 is not None:
        istart_val = option_map.get(user_istart, 0) if user_istart is not None else 0
        istrap_val = option_map.get(user_istrap, 0) if user_istrap is not None else 0
        card2_parts.extend([str(istart_val), str(istrap_val), str(user_temp1)])
    elif user_istrap is not None:
        istart_val = option_map.get(user_istart, 0) if user_istart is not None else 0
        istrap_val = option_map[user_istrap]
        card2_parts.extend([str(istart_val), str(istrap_val)])
    elif user_istart is not None:
        istart_val = option_map[user_istart]
        card2_parts.append(str(istart_val))
    card2_line = " ".join(card2_parts) + " /"

    # Build Card 3 with dependencies
    card3_parts = [str(errthn)]
    if user_errint is not None:
        if user_errmax is None:
            user_errmax = 10 * errthn
        if user_thnmax is None:
            user_thnmax = 1
        card3_parts.extend([str(user_thnmax), str(user_errmax), str(user_errint)])
    elif user_errmax is not None:
        if user_thnmax is None:
            user_thnmax = 1
        card3_parts.extend([str(user_thnmax), str(user_errmax)])
    elif user_thnmax is not None:
        card3_parts.append(str(user_thnmax))

    card3_line = " ".join(card3_parts) + " /"

    lines = [
        "-- calculate doppler broadening",
        "broadr",
        f"{nendf} {nin} {nout}",
        card2_line,
        card3_line,
    ]
    if ntemp2 > 0:
        lines.append(" ".join(temps) + " /")
    lines.append("0 /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse BROADR card lines into a parameter dict."""
    # Card 1: nendf nin nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: mat ntemp2 [istart] [istrap] [temp1]
    c2 = parse_card_values(card_lines[1])
    # Card 3: errthn [thnmax] [errmax] [errint]
    c3 = parse_card_values(card_lines[2])

    ntemp2 = int(c2[1])

    result: dict = {
        "nendf": int(c1[0]),
        "nin": int(c1[1]),
        "nout": int(c1[2]),
        "mat": int(c2[0]),
        "ntemp2": ntemp2,
        "errthn": float(c3[0]),
    }
    if len(c2) > 2:
        result["istart"] = int(c2[2])
    if len(c2) > 3:
        result["istrap"] = int(c2[3])
    if len(c2) > 4:
        result["temp1"] = float(c2[4])
    if len(c3) > 1:
        result["thnmax"] = float(c3[1])
    if len(c3) > 2:
        result["errmax"] = float(c3[2])
    if len(c3) > 3:
        result["errint"] = float(c3[3])

    # Card 4: temperatures (if ntemp2 > 0)
    idx = 3
    if ntemp2 > 0:
        result["temperatures"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1
    # Card 5: 0 / (terminator — skipped)
    return result
