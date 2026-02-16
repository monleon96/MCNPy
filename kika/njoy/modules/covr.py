"""COVR — Post-process ERRORR covariance data."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    nin = p["nin"]
    nout = p.get("nout", 0)
    cases = p.get("cases") or []
    ncase = len(cases)

    if nout > 0:
        # ── Library mode ──
        matype = p.get("matype", 3)
        hlibid = p.get("hlibid", "")
        hdescr = p.get("hdescr", "")

        lines: Lines = [
            "-- produce covariance library",
            "covr",
            f"{nin} {nout} /",
            f"{matype} {ncase} /",
            f"'{hlibid}'/",
            f"'{hdescr}'/",
        ]
        for case in cases:
            lines.append(" ".join(str(v) for v in case) + " /")
    else:
        # ── Plot mode ──
        nplot = p.get("nplot", 0)
        icolor = p.get("icolor", 0)
        epmin = p.get("epmin", 0.0)
        irelco = p.get("irelco", 1)
        noleg = p.get("noleg", 0)
        nstart = p.get("nstart", 1)
        ndiv = p.get("ndiv", 1)

        lines = [
            "-- produce covariance plots",
            "covr",
            f"{nin} 0 {nplot} /",
            f"{icolor} /",
        ]

        # Card 2b: custom colour boundaries (only when icolor=2)
        if icolor == 2:
            tlev = p.get("tlev") or []
            nlev = len(tlev)
            vals = " ".join(str(v) for v in tlev)
            lines.append(f"{nlev} {vals} /")

        lines.append(f"{epmin} /")
        lines.append(f"{irelco} {ncase} {noleg} {nstart} {ndiv} /")
        for case in cases:
            lines.append(" ".join(str(v) for v in case) + " /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse COVR card lines into a parameter dict."""
    # Card 1 determines mode: library (nout>0) vs plot (nout==0)
    c1 = parse_card_values(card_lines[0])
    nin = int(c1[0])
    nout = int(c1[1])

    if nout > 0:
        return _parse_library(card_lines, nin, nout)
    return _parse_plot(card_lines, nin, int(c1[2]) if len(c1) > 2 else 0)


def _parse_library(card_lines: list[str], nin: int, nout: int) -> dict:
    """Parse COVR library mode."""
    # Card 2: matype ncase
    c2 = parse_card_values(card_lines[1])
    matype = int(c2[0])
    ncase = int(c2[1])
    # Card 3: 'hlibid'/
    hlibid = parse_quoted_string(card_lines[2])
    # Card 4: 'hdescr'/
    hdescr = parse_quoted_string(card_lines[3])
    # Cards 5+: cases
    cases = []
    for i in range(ncase):
        cases.append([int(v) for v in parse_card_values(card_lines[4 + i])])
    return {
        "nin": nin,
        "nout": nout,
        "matype": matype,
        "hlibid": hlibid,
        "hdescr": hdescr,
        "cases": cases,
    }


def _parse_plot(card_lines: list[str], nin: int, nplot: int) -> dict:
    """Parse COVR plot mode."""
    # Card 2: icolor
    c2 = parse_card_values(card_lines[1])
    icolor = int(c2[0])

    idx = 2
    result: dict = {
        "nin": nin,
        "nout": 0,
        "nplot": nplot,
        "icolor": icolor,
    }

    # Card 2b: custom colour boundaries (icolor=2)
    if icolor == 2:
        c2b = parse_card_values(card_lines[idx])
        nlev = int(c2b[0])
        result["tlev"] = [float(v) for v in c2b[1:]]
        idx += 1

    # Card 3: epmin
    result["epmin"] = float(parse_card_values(card_lines[idx])[0])
    idx += 1

    # Card 4: irelco ncase noleg nstart ndiv
    c4 = parse_card_values(card_lines[idx])
    irelco = int(c4[0])
    ncase = int(c4[1])
    result["irelco"] = irelco
    result["noleg"] = int(c4[2]) if len(c4) > 2 else 0
    result["nstart"] = int(c4[3]) if len(c4) > 3 else 1
    result["ndiv"] = int(c4[4]) if len(c4) > 4 else 1
    idx += 1

    # Cases
    cases = []
    for i in range(ncase):
        cases.append([int(v) for v in parse_card_values(card_lines[idx + i])])
    result["cases"] = cases
    return result
