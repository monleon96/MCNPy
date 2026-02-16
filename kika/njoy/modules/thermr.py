"""THERMR — generate thermal scattering cross sections."""

from __future__ import annotations

from ._base import Lines, parse_iprint


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    nin = p.get("nin", "")
    nout = p.get("nout", "")

    matde = int(p.get("matde", 0))
    matdp_str = p.get("matdp", "U235")
    matdp = get_mat_number(matdp_str)

    nbin = int(p.get("nbin", 8))

    temp_str = str(p.get("temperatures", "293.6"))
    temps = temp_str.split()
    ntemp = len(temps)

    iin = int(p.get("iin", 2))
    icoh = int(p.get("icoh", 0))

    user_iprint = parse_iprint(p.get("iprint"))

    # Optional trailing Card 2 parameters: iform, natom, mtref, iprint
    user_iform = p.get("iform")
    user_natom = p.get("natom")
    user_mtref = p.get("mtref")

    card2_parts = [str(matde), str(matdp), str(nbin), str(ntemp),
                   str(iin), str(icoh)]

    if user_iprint is not None:
        card2_parts.extend([
            str(int(user_iform) if user_iform is not None else 0),
            str(int(user_natom) if user_natom is not None else 1),
            str(int(user_mtref) if user_mtref is not None else 222),
            str(user_iprint),
        ])
    elif user_mtref is not None:
        card2_parts.extend([
            str(int(user_iform) if user_iform is not None else 0),
            str(int(user_natom) if user_natom is not None else 1),
            str(int(user_mtref)),
        ])
    elif user_natom is not None:
        card2_parts.extend([
            str(int(user_iform) if user_iform is not None else 0),
            str(int(user_natom)),
        ])
    elif user_iform is not None:
        card2_parts.append(str(int(user_iform)))

    tol = p.get("tol", 0.01)
    emax = p.get("emax", 4.0)

    lines = [
        "-- generate thermal scattering cross sections",
        "thermr",
        f"{nendf} {nin} {nout}",
        " ".join(card2_parts) + " /",
        " ".join(temps) + " /",
        f"{tol} {emax} /",
    ]

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse THERMR card lines into a parameter dict."""
    from ._base import parse_card_values

    # Card 1: nendf nin nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: matde matdp nbin ntemp iin icoh [iform] [natom] [mtref] [iprint]
    c2 = parse_card_values(card_lines[1])

    ntemp = int(c2[3])

    result: dict = {
        "nendf": int(c1[0]),
        "nin": int(c1[1]),
        "nout": int(c1[2]),
        "matde": int(c2[0]),
        "matdp": int(c2[1]),
        "nbin": int(c2[2]),
        "ntemp": ntemp,
        "iin": int(c2[4]),
        "icoh": int(c2[5]),
    }
    if len(c2) > 6:
        result["iform"] = int(c2[6])
    if len(c2) > 7:
        result["natom"] = int(c2[7])
    if len(c2) > 8:
        result["mtref"] = int(c2[8])
    if len(c2) > 9:
        result["iprint"] = int(c2[9])

    # Card 3: temperatures
    if ntemp > 0:
        result["temperatures"] = [float(v) for v in parse_card_values(card_lines[2])]

    # Card 4: tol emax
    c4 = parse_card_values(card_lines[3])
    result["tol"] = float(c4[0])
    if len(c4) > 1:
        result["emax"] = float(c4[1])

    return result
