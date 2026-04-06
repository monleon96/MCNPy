"""CCCCR — produce CCCC interface files (ISOTXS, BRKOXS, DLAYXS)."""

from __future__ import annotations

import re

from ._base import Lines, parse_card_values, parse_multiline_card


def _parse_star_string(line: str) -> str:
    """Extract text between ``*`` delimiters from a Hollerith field."""
    m = re.search(r"\*([^*]*)\*", line)
    return m.group(1) if m else ""


def _parse_all_star_strings(line: str) -> list[str]:
    """Extract all ``*...*`` delimited strings from a line."""
    return re.findall(r"\*([^*]*)\*", line)


def generate(p: dict) -> Lines:
    nin = p["nin"]
    nisot = int(p.get("nisot", 0))
    nbrks = int(p.get("nbrks", 0))
    ndlay = int(p.get("ndlay", 0))

    lprint = int(p.get("lprint", 0))
    ivers = int(p.get("ivers", 0))
    huse = str(p.get("huse", ""))
    hsetid = str(p.get("hsetid", ""))

    ngroup = int(p["ngroup"])
    nggrup = int(p.get("nggrup", 0))
    maxord = int(p.get("maxord", 0))
    ifopt = int(p.get("ifopt", 1))

    isotopes = p.get("isotopes") or []
    niso = len(isotopes)

    lines: Lines = [
        "-- produce CCCC interface files",
        "ccccr",
    ]

    # Card 1: nin nisot nbrks ndlay
    lines.append(f"{nin} {nisot} {nbrks} {ndlay}")

    # Card 2: lprint ivers *huse*/
    if huse:
        lines.append(f"{lprint} {ivers} *{huse}*/")
    else:
        lines.append(f"{lprint} {ivers} /")

    # Card 3: *hsetid*/
    if hsetid:
        lines.append(f"*{hsetid}*/")
    else:
        lines.append("/")

    # Card 4: ngroup nggrup niso maxord ifopt
    lines.append(f"{ngroup} {nggrup} {niso} {maxord} {ifopt}")

    # Card 5 (niso times): *hisnm* *habsid* *hident* *hmat* imat xspo
    for iso in isotopes:
        hisnm = iso["hisnm"]
        habsid = iso["habsid"]
        hident = iso["hident"]
        hmat = iso["hmat"]
        imat = int(iso["imat"])
        xspo = iso["xspo"]
        lines.append(f"*{hisnm}* *{habsid}* *{hident}* *{hmat}* {imat} {xspo}")

    # CISOTX cards (if nisot > 0)
    if nisot > 0:
        nsblok = int(p.get("nsblok", 1))
        maxup = int(p.get("maxup", 0))
        maxdn = int(p.get("maxdn", ngroup))
        ichix = int(p.get("ichix", -1))

        # CISOTX Card 1: nsblok maxup maxdn ichix
        lines.append(f"{nsblok} {maxup} {maxdn} {ichix}")

        # CISOTX Card 2 (ichix=1): spec(1..ngroup) — fission spectrum
        # CISOTX Card 3 (ichix>1): spec(1..ngroup) — spectrum assignment
        if ichix >= 1:
            spec = p.get("spec", [])
            lines.append(" ".join(str(v) for v in spec) + " /")

        # CISOTX Card 4 (niso times): kbr amass efiss ecapt temp sigpot adens
        isotxs_params = p.get("isotxs_params", [])
        for ip in isotxs_params:
            kbr = int(ip["kbr"])
            amass = ip["amass"]
            efiss = ip["efiss"]
            ecapt = ip["ecapt"]
            temp = ip["temp"]
            sigpot = ip["sigpot"]
            adens = ip["adens"]
            lines.append(f"{kbr} {amass} {efiss} {ecapt} {temp} {sigpot} {adens}")

    # CBRKOXS cards (if nbrks > 0)
    if nbrks > 0:
        nti = int(p.get("nti", 0))
        nzi = int(p.get("nzi", 0))

        # CBRKOXS Card 1: nti nzi
        lines.append(f"{nti} {nzi}")

        # CBRKOXS Card 2 (if nti > 0): atem(1..nti)
        if nti > 0:
            atem = p.get("atem", [])
            lines.append(" ".join(str(v) for v in atem) + " /")

        # CBRKOXS Card 3 (if nzi > 0): asig(1..nzi)
        if nzi > 0:
            asig = p.get("asig", [])
            lines.append(" ".join(str(v) for v in asig) + " /")

    # CDLAYXS: no additional input required

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse CCCCR card lines into a parameter dict."""
    idx = 0

    # Card 1: nin nisot nbrks ndlay
    c1 = parse_card_values(card_lines[idx])
    lm: dict[str, list[int]] = {
        "nin": [idx], "nisot": [idx], "nbrks": [idx], "ndlay": [idx],
    }
    idx += 1

    # Card 2: lprint ivers *huse*/
    line2 = card_lines[idx]
    huse = ""
    if "*" in line2:
        huse = _parse_star_string(line2)
        cleaned = re.sub(r"\*[^*]*\*/?\s*$", "", line2).strip()
        c2 = cleaned.split()
    else:
        c2 = parse_card_values(line2)
    lprint = int(c2[0])
    ivers = int(c2[1]) if len(c2) > 1 else 0
    lm["lprint"] = [idx]; lm["ivers"] = [idx]; lm["huse"] = [idx]
    idx += 1

    nin = int(c1[0])
    nisot = int(c1[1])
    nbrks = int(c1[2])
    ndlay = int(c1[3])

    # Card 3: *hsetid*/
    line3 = card_lines[idx]
    hsetid = ""
    if "*" in line3:
        hsetid = _parse_star_string(line3)
    lm["hsetid"] = [idx]
    idx += 1

    # Card 4: ngroup nggrup niso maxord ifopt
    c4 = parse_card_values(card_lines[idx])
    ngroup = int(c4[0])
    nggrup = int(c4[1])
    niso = int(c4[2])
    maxord = int(c4[3])
    ifopt = int(c4[4])
    lm["ngroup"] = [idx]; lm["nggrup"] = [idx]; lm["maxord"] = [idx]; lm["ifopt"] = [idx]
    idx += 1

    result: dict = {
        "nin": nin,
        "nisot": nisot,
        "nbrks": nbrks,
        "ndlay": ndlay,
        "lprint": lprint,
        "ivers": ivers,
        "huse": huse,
        "hsetid": hsetid,
        "ngroup": ngroup,
        "nggrup": nggrup,
        "maxord": maxord,
        "ifopt": ifopt,
    }

    # Card 5 (niso times): *hisnm* *habsid* *hident* *hmat* imat xspo
    iso_start = idx
    isotopes = []
    for _ in range(niso):
        line = card_lines[idx]; idx += 1
        stars = _parse_all_star_strings(line)
        cleaned = re.sub(r"\*[^*]*\*", "", line).strip()
        nums = cleaned.split()
        isotopes.append({
            "hisnm": stars[0] if len(stars) > 0 else "",
            "habsid": stars[1] if len(stars) > 1 else "",
            "hident": stars[2] if len(stars) > 2 else "",
            "hmat": stars[3] if len(stars) > 3 else "",
            "imat": int(nums[0]),
            "xspo": float(nums[1]),
        })
    result["isotopes"] = isotopes
    if niso > 0:
        lm["isotopes"] = list(range(iso_start, idx))

    # CISOTX cards (if nisot > 0)
    if nisot > 0:
        c_iso1 = parse_card_values(card_lines[idx])
        nsblok = int(c_iso1[0])
        maxup = int(c_iso1[1])
        maxdn = int(c_iso1[2])
        ichix = int(c_iso1[3])
        result["nsblok"] = nsblok
        result["maxup"] = maxup
        result["maxdn"] = maxdn
        result["ichix"] = ichix
        lm["nsblok"] = [idx]; lm["maxup"] = [idx]; lm["maxdn"] = [idx]; lm["ichix"] = [idx]
        idx += 1

        if ichix >= 1:
            spec_vals, spec_lines = parse_multiline_card(card_lines, idx)
            result["spec"] = [float(v) for v in spec_vals]
            lm["spec"] = list(range(idx, idx + spec_lines))
            idx += spec_lines

        isotxs_start = idx
        isotxs_params = []
        for _ in range(niso):
            c_ip = parse_card_values(card_lines[idx]); idx += 1
            isotxs_params.append({
                "kbr": int(c_ip[0]),
                "amass": float(c_ip[1]),
                "efiss": float(c_ip[2]),
                "ecapt": float(c_ip[3]),
                "temp": float(c_ip[4]),
                "sigpot": float(c_ip[5]),
                "adens": float(c_ip[6]),
            })
        result["isotxs_params"] = isotxs_params
        if niso > 0:
            lm["isotxs_params"] = list(range(isotxs_start, idx))

    # CBRKOXS cards (if nbrks > 0)
    if nbrks > 0:
        c_brk1 = parse_card_values(card_lines[idx])
        nti = int(c_brk1[0])
        nzi = int(c_brk1[1])
        result["nti"] = nti
        result["nzi"] = nzi
        lm["nti"] = [idx]; lm["nzi"] = [idx]
        idx += 1

        if nti > 0:
            atem_vals, atem_lines = parse_multiline_card(card_lines, idx)
            result["atem"] = [float(v) for v in atem_vals]
            lm["atem"] = list(range(idx, idx + atem_lines))
            idx += atem_lines

        if nzi > 0:
            asig_vals, asig_lines = parse_multiline_card(card_lines, idx)
            result["asig"] = [float(v) for v in asig_vals]
            lm["asig"] = list(range(idx, idx + asig_lines))
            idx += asig_lines

    # CDLAYXS: no additional cards

    result["_line_map"] = lm
    return result
