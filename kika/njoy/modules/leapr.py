"""LEAPR — prepare thermal scattering law S(α,β) in ENDF-6 format."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    """Generate LEAPR input lines from parameter dict."""
    nout = p.get("nout", 24)
    title = p.get("title", "thermal scattering law via leapr")

    # Derive counts from list lengths
    alphas = p["alphas"]
    betas = p["betas"]
    temperatures = p["temperatures"]
    nalpha = len(alphas)
    nbeta = len(betas)
    ntempr = len(temperatures)

    iprint = parse_iprint(p.get("iprint"))
    if iprint is None:
        iprint = 0
    nphon = p.get("nphon", 100)
    nsk = p.get("nsk", 0)
    nss = p.get("nss", 0)
    b7 = p.get("b7", 0)
    lat = p.get("lat", 1)

    lines: Lines = [
        "-- prepare thermal scattering law",
        "leapr",
        f"{nout} /",
        f"'{title}'/",
    ]

    # Card 3: ntempr, iprint, nphon (positional cascade)
    lines.append(_card3(ntempr, iprint, nphon))
    # Card 4: mat, za, isabt, ilog, smin
    lines.append(_card4(p))
    # Card 5: awr, spr, npr, iel, ncold, nsk
    lines.append(_card5(p))
    # Card 6: nss, b7, aws, sps, mss
    lines.append(_card6(p))
    # Card 7: nalpha, nbeta, lat
    lines.append(_card7(nalpha, nbeta, lat))
    # Card 8: alphas
    lines.append(_float_list(alphas) + " /")
    # Card 9: betas
    lines.append(_float_list(betas) + " /")

    # Temperature blocks for principal scatterer
    for t_block in temperatures:
        _emit_temperature_block(lines, t_block, nsk)

    # Secondary scatterer temperature blocks (nss > 0 and b7 == 0)
    if nss > 0 and b7 == 0:
        for t_block in p.get("secondary_temperatures", []):
            _emit_temperature_block(lines, t_block, 0)

    return lines


# ------------------------------------------------------------------
# Card builders
# ------------------------------------------------------------------


def _card3(ntempr: int, iprint: int, nphon: int) -> str:
    """Card 3: ntempr, iprint, nphon (positional cascade)."""
    parts = [str(ntempr)]
    if nphon != 100:
        parts.extend([str(iprint), str(nphon)])
    elif iprint != 0:
        parts.append(str(iprint))
    return " ".join(parts) + " /"


def _card4(p: dict) -> str:
    """Card 4: mat, za, isabt, ilog, smin (positional cascade)."""
    mat = p["mat"]
    za = p["za"]
    isabt = p.get("isabt", 0)
    ilog = p.get("ilog", 0)
    smin = p.get("smin")

    parts = [str(mat), str(za)]
    if smin is not None:
        parts.extend([str(isabt), str(ilog), str(smin)])
    elif ilog != 0:
        parts.extend([str(isabt), str(ilog)])
    elif isabt != 0:
        parts.append(str(isabt))
    return " ".join(parts) + " /"


def _card5(p: dict) -> str:
    """Card 5: awr, spr, npr, iel, ncold, nsk (positional cascade)."""
    awr = p["awr"]
    spr = p["spr"]
    npr = p.get("npr", 1)
    iel = p.get("iel", 0)
    ncold = p.get("ncold", 0)
    nsk = p.get("nsk", 0)

    parts = [str(awr), str(spr)]
    if nsk != 0:
        parts.extend([str(npr), str(iel), str(ncold), str(nsk)])
    elif ncold != 0:
        parts.extend([str(npr), str(iel), str(ncold)])
    elif iel != 0:
        parts.extend([str(npr), str(iel)])
    elif npr != 1:
        parts.append(str(npr))
    return " ".join(parts) + " /"


def _card6(p: dict) -> str:
    """Card 6: nss, b7, aws, sps, mss (positional cascade)."""
    nss = p.get("nss", 0)
    b7 = p.get("b7", 0)
    aws = p.get("aws", 0.0)
    sps = p.get("sps", 0.0)
    mss = p.get("mss", 0)

    parts = [str(nss)]
    if mss != 0:
        parts.extend([str(b7), str(aws), str(sps), str(mss)])
    elif sps != 0.0:
        parts.extend([str(b7), str(aws), str(sps)])
    elif aws != 0.0:
        parts.extend([str(b7), str(aws)])
    elif b7 != 0:
        parts.append(str(b7))
    return " ".join(parts) + " /"


def _card7(nalpha: int, nbeta: int, lat: int) -> str:
    """Card 7: nalpha, nbeta, lat."""
    parts = [str(nalpha), str(nbeta)]
    if lat != 1:
        parts.append(str(lat))
    return " ".join(parts) + " /"


# ------------------------------------------------------------------
# Temperature block
# ------------------------------------------------------------------


def _emit_temperature_block(lines: Lines, t: dict, nsk: int) -> None:
    """Emit cards 10–19 for one temperature point."""
    temp = t["temp"]
    lines.append(f"{temp} /")  # Card 10

    if temp < 0:
        return  # negative → reuse previous spectrum

    # Card 11: delta, ni
    lines.append(f"{t['delta']} {t['ni']} /")

    # Card 12: rho values
    lines.append(_float_list(t["rho"]) + " /")

    # Card 13: twt, c, tbeta
    lines.append(f"{t['twt']} {t['c']} {t['tbeta']} /")

    # Card 14: nd
    nd = t.get("nd", 0)
    lines.append(f"{nd} /")

    # Cards 15–16: discrete oscillator energies and weights
    if nd > 0:
        lines.append(_float_list(t["energies"]) + " /")  # Card 15
        lines.append(_float_list(t["weights"]) + " /")   # Card 16

    # Cards 17–19: pair correlation (principal scatterer only)
    if nsk != 0:
        nka = t.get("nka", 0)
        dka = t.get("dka", 0.0)
        lines.append(f"{nka} {dka} /")  # Card 17
        if nka > 0:
            lines.append(_float_list(t["kappas"]) + " /")  # Card 18
        if nsk == 2:
            lines.append(f"{t.get('cfrac', 0.0)} /")  # Card 19


def _float_list(values: list) -> str:
    """Format a list of numbers as a space-separated string."""
    return " ".join(str(v) for v in values)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def parse(card_lines: list[str]) -> dict:
    """Parse LEAPR card lines into a parameter dict."""
    idx = 0

    # Card 1: nout
    c1 = parse_card_values(card_lines[idx]); idx += 1
    nout = int(c1[0])

    # Card 2: 'title'/
    title = parse_quoted_string(card_lines[idx]); idx += 1

    # Card 3: ntempr [iprint] [nphon]
    c3 = parse_card_values(card_lines[idx]); idx += 1
    ntempr = int(c3[0])
    iprint = int(c3[1]) if len(c3) > 1 else 0
    nphon = int(c3[2]) if len(c3) > 2 else 100

    # Card 4: mat za [isabt] [ilog] [smin]
    c4 = parse_card_values(card_lines[idx]); idx += 1
    mat = int(c4[0])
    za = int(c4[1])
    isabt = int(c4[2]) if len(c4) > 2 else 0
    ilog = int(c4[3]) if len(c4) > 3 else 0
    smin = float(c4[4]) if len(c4) > 4 else None

    # Card 5: awr spr [npr] [iel] [ncold] [nsk]
    c5 = parse_card_values(card_lines[idx]); idx += 1
    awr = float(c5[0])
    spr = float(c5[1])
    npr = int(c5[2]) if len(c5) > 2 else 1
    iel = int(c5[3]) if len(c5) > 3 else 0
    ncold = int(c5[4]) if len(c5) > 4 else 0
    nsk = int(c5[5]) if len(c5) > 5 else 0

    # Card 6: nss [b7] [aws] [sps] [mss]
    c6 = parse_card_values(card_lines[idx]); idx += 1
    nss = int(c6[0])
    b7 = int(c6[1]) if len(c6) > 1 else 0
    aws = float(c6[2]) if len(c6) > 2 else 0.0
    sps = float(c6[3]) if len(c6) > 3 else 0.0
    mss = int(c6[4]) if len(c6) > 4 else 0

    # Card 7: nalpha nbeta [lat]
    c7 = parse_card_values(card_lines[idx]); idx += 1
    nalpha = int(c7[0])
    nbeta = int(c7[1])
    lat = int(c7[2]) if len(c7) > 2 else 1

    # Card 8: alphas
    alphas = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1
    # Card 9: betas
    betas = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1

    # Temperature blocks for principal scatterer
    temperatures = []
    for _ in range(ntempr):
        t_block, idx = _parse_temperature_block(card_lines, idx, nsk)
        temperatures.append(t_block)

    result: dict = {
        "nout": nout,
        "title": title,
        "ntempr": ntempr,
        "iprint": iprint,
        "nphon": nphon,
        "mat": mat,
        "za": za,
        "isabt": isabt,
        "ilog": ilog,
        "awr": awr,
        "spr": spr,
        "npr": npr,
        "iel": iel,
        "ncold": ncold,
        "nsk": nsk,
        "nss": nss,
        "b7": b7,
        "aws": aws,
        "sps": sps,
        "mss": mss,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "lat": lat,
        "alphas": alphas,
        "betas": betas,
        "temperatures": temperatures,
    }
    if smin is not None:
        result["smin"] = smin

    # Secondary scatterer temperature blocks (nss > 0 and b7 == 0)
    if nss > 0 and b7 == 0:
        sec_temps = []
        while idx < len(card_lines):
            t_block, idx = _parse_temperature_block(card_lines, idx, 0)
            sec_temps.append(t_block)
        if sec_temps:
            result["secondary_temperatures"] = sec_temps

    return result


def _parse_temperature_block(
    card_lines: list[str], idx: int, nsk: int
) -> tuple[dict, int]:
    """Parse one temperature block, returning (block_dict, next_idx)."""
    # Card 10: temp
    c10 = parse_card_values(card_lines[idx]); idx += 1
    temp = float(c10[0])
    block: dict = {"temp": temp}

    if temp < 0:
        return block, idx  # negative → reuse previous spectrum

    # Card 11: delta ni
    c11 = parse_card_values(card_lines[idx]); idx += 1
    block["delta"] = float(c11[0])
    block["ni"] = int(c11[1])

    # Card 12: rho values
    block["rho"] = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1

    # Card 13: twt c tbeta
    c13 = parse_card_values(card_lines[idx]); idx += 1
    block["twt"] = float(c13[0])
    block["c"] = float(c13[1])
    block["tbeta"] = float(c13[2])

    # Card 14: nd
    c14 = parse_card_values(card_lines[idx]); idx += 1
    nd = int(c14[0])
    block["nd"] = nd

    # Cards 15-16: discrete oscillator energies and weights
    if nd > 0:
        block["energies"] = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1
        block["weights"] = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1

    # Cards 17-19: pair correlation (principal scatterer only)
    if nsk != 0:
        c17 = parse_card_values(card_lines[idx]); idx += 1
        nka = int(c17[0])
        dka = float(c17[1])
        block["nka"] = nka
        block["dka"] = dka
        if nka > 0:
            block["kappas"] = [float(v) for v in parse_card_values(card_lines[idx])]; idx += 1
        if nsk == 2:
            c19 = parse_card_values(card_lines[idx]); idx += 1
            block["cfrac"] = float(c19[0])

    return block, idx
