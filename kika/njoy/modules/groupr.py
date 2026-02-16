"""GROUPR — compute multigroup cross sections and scattering matrices."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    npend = p.get("npend", "")
    ngout1 = p.get("ngout1", 0)
    ngout2 = p.get("ngout2", "")

    mat_str = p.get("matb", "U235")
    matb = get_mat_number(mat_str)

    ign = int(p["ign"])
    igg = int(p.get("igg", 0))
    iwt = int(p["iwt"])
    lord = int(p.get("lord", 0))

    # Validate unsupported iwt values
    if iwt == 0:
        raise ValueError(
            "iwt=0 (read flux from input tape) is not supported."
        )
    if iwt < 0:
        raise ValueError(
            "iwt<0 (flux calculator) is not supported."
        )
    if iwt == 1:
        raise ValueError(
            "iwt=1 (tabulated TAB1 weight function) is not supported. "
            "Use iwt=2..12 for predefined/analytic weight functions."
        )

    # Temperatures and sigma-zero values
    temp_str = str(p.get("temperatures", "293.6"))
    temps = temp_str.split()
    ntemp = len(temps)

    sigz_str = str(p.get("sigz", "1e10"))
    sigz_values = sigz_str.split()
    nsigz = len(sigz_values)

    title = p.get("title", f"multigroup data for {mat_str}")

    # Card 2: matb ign igg iwt lord ntemp nsigz [iprint] [ismooth]
    iprint = parse_iprint(p.get("iprint"))
    ismooth = p.get("ismooth")

    card2_parts = [str(matb), str(ign), str(igg), str(iwt),
                   str(lord), str(ntemp), str(nsigz)]
    if ismooth is not None:
        card2_parts.append(str(iprint if iprint is not None else 1))
        card2_parts.append(str(ismooth))
    elif iprint is not None:
        card2_parts.append(str(iprint))

    lines: Lines = [
        "-- compute multigroup cross sections",
        "groupr",
        f"{nendf} {npend} {ngout1} {ngout2}",
        " ".join(card2_parts) + " /",
        f"'{title}'/",
    ]

    # Card 4: temperatures
    lines.append(" ".join(temps) + " /")

    # Card 5: sigma-zero values
    lines.append(" ".join(sigz_values) + " /")

    # Card 6a/6b: custom neutron group boundaries (only when ign=1)
    if ign == 1:
        egn = p.get("egn")
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

    # Card 7a/7b: custom photon group boundaries (only when igg=1)
    if igg == 1:
        egg = p.get("egg")
        if egg is None:
            raise ValueError(
                "igg=1 requires custom photon group boundaries ('egg')."
            )
        if isinstance(egg, list):
            egg_values = [str(e) for e in egg]
        else:
            egg_values = str(egg).split()
        ngg = len(egg_values) - 1
        lines.append(f"{ngg} /")
        lines.append(" ".join(egg_values) + " /")

    # Card 8c: analytic flux params (only when |iwt|=4)
    if abs(iwt) == 4:
        eb = p.get("eb")
        tb = p.get("tb")
        ec = p.get("ec")
        tc = p.get("tc")
        if any(v is None for v in (eb, tb, ec, tc)):
            raise ValueError(
                "iwt=4 requires analytic flux parameters: eb, tb, ec, tc."
            )
        lines.append(f"{eb} {tb} {ec} {tc} /")

    # Card 9: reaction list
    reactions = p.get("reactions")
    if reactions:
        for rxn in reactions:
            mfd, mtd = rxn[0], rxn[1]
            mtname = rxn[2] if len(rxn) > 2 else None
            if mtname is not None:
                lines.append(f"{mfd} {mtd} '{mtname}' /")
            else:
                lines.append(f"{mfd} {mtd} /")
    lines.append("0 /")

    # Card 10: material terminator
    lines.append("0 /")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse GROUPR card lines into a parameter dict."""
    # Card 1: nendf npend ngout1 ngout2
    c1 = parse_card_values(card_lines[0])
    # Card 2: matb ign igg iwt lord ntemp nsigz [iprint] [ismooth]
    c2 = parse_card_values(card_lines[1])
    # Card 3: 'title'/
    title = parse_quoted_string(card_lines[2])

    ign = int(c2[1])
    igg = int(c2[2])
    iwt = int(c2[3])
    ntemp = int(c2[5])
    nsigz = int(c2[6])

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "ngout1": int(c1[2]),
        "ngout2": int(c1[3]),
        "matb": int(c2[0]),
        "ign": ign,
        "igg": igg,
        "iwt": iwt,
        "lord": int(c2[4]),
        "ntemp": ntemp,
        "nsigz": nsigz,
        "title": title,
    }
    if len(c2) > 7:
        result["iprint"] = int(c2[7])
    if len(c2) > 8:
        result["ismooth"] = int(c2[8])

    idx = 3
    # Card 4: temperatures
    result["temperatures"] = [float(v) for v in parse_card_values(card_lines[idx])]
    idx += 1
    # Card 5: sigma-zero values
    result["sigz"] = [float(v) for v in parse_card_values(card_lines[idx])]
    idx += 1

    # Card 6a/6b: custom neutron group boundaries (ign=1)
    if ign == 1:
        ngn_vals = parse_card_values(card_lines[idx])
        result["ngn"] = int(ngn_vals[0])
        idx += 1
        result["egn"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1

    # Card 7a/7b: custom photon group boundaries (igg=1)
    if igg == 1:
        ngg_vals = parse_card_values(card_lines[idx])
        result["ngg"] = int(ngg_vals[0])
        idx += 1
        result["egg"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1

    # Card 8c: analytic flux params (|iwt|=4)
    if abs(iwt) == 4:
        c8 = parse_card_values(card_lines[idx])
        result["eb"] = float(c8[0])
        result["tb"] = float(c8[1])
        result["ec"] = float(c8[2])
        result["tc"] = float(c8[3])
        idx += 1

    # Card 9: reactions (terminated by 0 /)
    reactions: list[list] = []
    while idx < len(card_lines):
        vals = parse_card_values(card_lines[idx])
        idx += 1
        if int(vals[0]) == 0:
            # Check if this is the reaction terminator or material terminator
            # First 0 / terminates reactions, second terminates materials
            break
        mfd = int(vals[0])
        mtd = int(vals[1])
        rxn: list = [mfd, mtd]
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
