"""ERRORR — produce multigroup covariance data from ENDF error files."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    npend = p.get("npend", "")
    ngout = p.get("ngout", 0)
    nout = p.get("nout", "")

    mat_str = p.get("matd", "U235")
    matd = get_mat_number(mat_str)

    ign = int(p.get("ign", 1))
    iwt = int(p.get("iwt", 6))
    irelco = int(p.get("irelco", 1))

    iprint = parse_iprint(p.get("iprint"))
    iprint_val = iprint if iprint is not None else 1

    mprint = int(p.get("mprint", 0))
    tempin = float(p.get("tempin", 300.0))

    # Card 7 parameters
    iread = int(p.get("iread", 0))
    mfcov = int(p.get("mfcov", 33))
    irespr = p.get("irespr")
    legord = p.get("legord")
    ifissp = p.get("ifissp")
    efmean = p.get("efmean")
    dap = p.get("dap")

    # Custom energy group boundaries force ign=1
    egn = p.get("egn")
    if egn is not None:
        ign = 1

    # Build Card 7 with positional dependencies
    card7_parts = [str(iread), str(mfcov)]
    if dap is not None:
        card7_parts.extend([
            str(int(irespr) if irespr is not None else 1),
            str(int(legord) if legord is not None else 0),
            str(int(ifissp) if ifissp is not None else 0),
            str(float(efmean) if efmean is not None else 0.0),
            str(dap),
        ])
    elif efmean is not None:
        card7_parts.extend([
            str(int(irespr) if irespr is not None else 1),
            str(int(legord) if legord is not None else 0),
            str(int(ifissp) if ifissp is not None else 0),
            str(efmean),
        ])
    elif ifissp is not None:
        card7_parts.extend([
            str(int(irespr) if irespr is not None else 1),
            str(int(legord) if legord is not None else 0),
            str(ifissp),
        ])
    elif legord is not None:
        card7_parts.extend([
            str(int(irespr) if irespr is not None else 1),
            str(legord),
        ])
    elif irespr is not None:
        card7_parts.append(str(int(irespr)))

    lines: Lines = [
        "-- produce covariance data",
        "errorr",
        f"{nendf} {npend} {ngout} {nout} /",
        f"{matd} {ign} {iwt} {iprint_val} {irelco} /",
        f"{mprint} {tempin} /",
        " ".join(card7_parts) + " /",
    ]

    # Card 12a/12b: custom energy group boundaries (only when ign=1 or 19)
    if ign in (1, 19):
        if egn is None:
            raise ValueError(
                "ign=1 requires custom neutron group boundaries ('egn')."
            )
        if isinstance(egn, list):
            egn_values = [str(e) for e in egn]
        else:
            egn_values = str(egn).split()
        ngn = len(egn_values) - 1
        lines.append(f"{ngn} /")
        lines.append(" ".join(egn_values) + " /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse ERRORR card lines into a parameter dict."""
    # Card 1: nendf npend ngout nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: matd ign iwt iprint irelco
    c2 = parse_card_values(card_lines[1])
    # Card 5: mprint tempin
    c5 = parse_card_values(card_lines[2])
    # Card 7: iread mfcov [irespr] [legord] [ifissp] [efmean] [dap]
    c7 = parse_card_values(card_lines[3])

    ign = int(c2[1])

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "ngout": int(c1[2]),
        "nout": int(c1[3]),
        "matd": int(c2[0]),
        "ign": ign,
        "iwt": int(c2[2]),
        "iprint": int(c2[3]),
        "irelco": int(c2[4]),
        "mprint": int(c5[0]),
        "tempin": float(c5[1]),
        "iread": int(c7[0]),
        "mfcov": int(c7[1]),
    }
    if len(c7) > 2:
        result["irespr"] = int(c7[2])
    if len(c7) > 3:
        result["legord"] = int(c7[3])
    if len(c7) > 4:
        result["ifissp"] = int(c7[4])
    if len(c7) > 5:
        result["efmean"] = float(c7[5])
    if len(c7) > 6:
        result["dap"] = float(c7[6])

    # Cards 12a/12b: custom energy boundaries (ign=1 or 19)
    idx = 4
    if ign in (1, 19) and idx < len(card_lines):
        ngn_vals = parse_card_values(card_lines[idx])
        result["ngn"] = int(ngn_vals[0])
        idx += 1
        if idx < len(card_lines):
            result["egn"] = [float(v) for v in parse_card_values(card_lines[idx])]

    return result
