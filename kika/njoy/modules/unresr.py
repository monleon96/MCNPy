"""UNRESR — calculate self-shielding in the unresolved range."""

from __future__ import annotations

from ._base import Lines, parse_iprint


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    nin = p.get("nin", "")
    nout = p.get("nout", "")

    mat_str = p.get("matd", "U235")
    mat_num = get_mat_number(mat_str)

    temp_str = str(p.get("temp", ""))
    temps = temp_str.split()
    ntemp = len(temps)

    sigz_str = str(p.get("sigz", ""))
    sigz_values = sigz_str.split()
    nsigz = len(sigz_values)

    iprint = parse_iprint(p.get("iprint"))

    card2_parts = [str(mat_num), str(ntemp), str(nsigz)]

    if p.get("iprint") is not None:
        card2_parts.append(str(iprint))

    lines = [
        "-- calculate self-shielding",
        "unresr",
        f"{nendf} {nin} {nout}",
        " ".join(card2_parts) + " /",
    ]
    if ntemp > 0:
        lines.append(" ".join(temps) + " /")
    if nsigz > 0:
        lines.append(" ".join(sigz_values) + " /")
    lines.append("0 /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse UNRESR card lines into a parameter dict."""
    from ._base import parse_card_values

    # Card 1: nendf nin nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: matd ntemp nsigz [iprint]
    c2 = parse_card_values(card_lines[1])

    ntemp = int(c2[1])
    nsigz = int(c2[2])

    result: dict = {
        "nendf": int(c1[0]),
        "nin": int(c1[1]),
        "nout": int(c1[2]),
        "matd": int(c2[0]),
        "ntemp": ntemp,
        "nsigz": nsigz,
    }
    if len(c2) > 3:
        result["iprint"] = int(c2[3])

    idx = 2
    if ntemp > 0:
        result["temperatures"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1
    if nsigz > 0:
        result["sigz"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1
    # 0 / terminator — skipped
    return result
