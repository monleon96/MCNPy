"""HEATR — calculate heating values (KERMA factors)."""

from __future__ import annotations

from ._base import Lines, parse_iprint


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    nin = p.get("nin", "")
    nout = p.get("nout", "")
    nplot = p.get("nplot") or "0"

    mat_str = p.get("matd", "U235")
    mat_num = get_mat_number(mat_str)
    mtk_str = str(p.get("mtk", ""))
    mtk_list = mtk_str.split()
    npk = len(mtk_list)

    user_ed = p.get("ed")
    iprint = parse_iprint(p.get("iprint"))
    ntemp = p.get("ntemp", 0)

    user_local = p.get("local")
    if user_local == "Transported":
        local = 0
    elif user_local == "Deposited":
        local = 1
    else:
        local = None

    card2_parts = [str(mat_num), str(npk)]

    if user_ed is not None:
        card2_parts.extend([
            "0",  # nqa
            str(ntemp),
            str(local if local is not None else 0),
            str(iprint if iprint is not None else 0),
            str(user_ed),
        ])
    elif p.get("iprint") is not None:
        card2_parts.extend([
            "0",
            str(ntemp),
            str(local if local is not None else 0),
            str(iprint),
        ])
    elif user_local is not None:
        card2_parts.extend([
            "0",
            str(ntemp),
            str(local),
        ])
    elif ntemp:
        card2_parts.extend([
            "0",
            str(ntemp),
        ])

    lines = [
        "-- calculate heating values",
        "heatr",
        f"{nendf} {nin} {nout} {nplot}",
        " ".join(card2_parts) + " /",
    ]
    if npk > 0:
        lines.append(" ".join(mtk_list) + " /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse HEATR card lines into a parameter dict."""
    from ._base import parse_card_values

    # Card 1: nendf nin nout nplot
    c1 = parse_card_values(card_lines[0])
    # Card 2: matd npk [nqa] [ntemp] [local] [iprint] [ed]
    c2 = parse_card_values(card_lines[1])

    npk = int(c2[1])

    result: dict = {
        "nendf": int(c1[0]),
        "nin": int(c1[1]),
        "nout": int(c1[2]),
        "nplot": int(c1[3]),
        "matd": int(c2[0]),
        "npk": npk,
    }
    if len(c2) > 2:
        result["nqa"] = int(c2[2])
    if len(c2) > 3:
        result["ntemp"] = int(c2[3])
    if len(c2) > 4:
        result["local"] = int(c2[4])
    if len(c2) > 5:
        result["iprint"] = int(c2[5])
    if len(c2) > 6:
        result["ed"] = float(c2[6])

    # Card 3: MT list (if npk > 0)
    if npk > 0 and len(card_lines) > 2:
        result["mtk"] = [int(v) for v in parse_card_values(card_lines[2])]

    return result
