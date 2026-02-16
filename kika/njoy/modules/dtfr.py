"""DTFR — prepare DTF-IV transport tables from GROUPR output."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values, parse_quoted_string


def _extract_quoted_strings(line: str) -> list[str]:
    """Extract all single-quoted strings from a line."""
    strings: list[str] = []
    start = 0
    while True:
        try:
            q1 = line.index("'", start)
            q2 = line.index("'", q1 + 1)
            strings.append(line[q1 + 1 : q2])
            start = q2 + 1
        except ValueError:
            break
    return strings


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    # Card 1: nin nout npend nplot
    nin = p.get("nin", "")
    nout = p.get("nout", "")
    npend = p.get("npend", 0)
    nplot = p.get("nplot", 0)

    # Card 2: iprint ifilm iedit
    iprint = parse_iprint(p.get("iprint")) or 0
    ifilm = int(p.get("ifilm", 0))
    iedit = int(p.get("iedit", 0))

    lines: Lines = [
        "-- prepare DTF transport tables",
        "dtfr",
        f"{nin} {nout} {npend} {nplot}",
        f"{iprint} {ifilm} {iedit} /",
    ]

    nlmax = int(p["nlmax"])
    ng = int(p["ng"])

    if iedit == 0:
        # Card 3: nlmax ng iptotl ipingp itabl ned ntherm
        edit_names: list[str] = p.get("edit_names", [])
        edits: list[list] = p.get("edits", [])
        ned = len(edits)

        # iptotl: derive from edit_names if not explicit
        if "iptotl" in p:
            iptotl = int(p["iptotl"])
        else:
            iptotl = len(edit_names) + 3

        # Validate edit_names count
        expected = iptotl - 3
        if len(edit_names) != expected:
            raise ValueError(
                f"Expected {expected} edit names (iptotl - 3), "
                f"got {len(edit_names)}"
            )

        ipingp = int(p.get("ipingp", iptotl + 1))
        itabl = int(p.get("itabl", iptotl + nlmax * ng))
        ntherm = int(p.get("ntherm", 0))

        lines.append(
            f"{nlmax} {ng} {iptotl} {ipingp} {itabl} {ned} {ntherm} /"
        )

        # Card 3a (if ntherm > 0): mti mtc nlc
        if ntherm > 0:
            mti = int(p["mti"])
            mtc = int(p.get("mtc", 0))
            nlc = int(p.get("nlc", 0))
            lines.append(f"{mti} {mtc} {nlc} /")

        # Card 4: iptotl-3 edit names (quoted strings)
        if edit_names:
            names_str = " ".join(f"'{n}'" for n in edit_names)
            lines.append(f"{names_str} /")

        # Card 5: ned triplets of jpos mt mult
        for triplet in edits:
            jpos, mt, mult = int(triplet[0]), int(triplet[1]), int(triplet[2])
            lines.append(f"{jpos} {mt} {mult} /")
    else:
        # Card 6 (iedit=1, CLAW): nlmax ng
        lines.append(f"{nlmax} {ng} /")

    # Card 7: nptabl ngp (gamma tables)
    nptabl = int(p.get("nptabl", 0))
    ngp = int(p.get("ngp", 0))
    lines.append(f"{nptabl} {ngp} /")

    # Card 8: materials (repeating, terminated by /)
    materials: list[dict] = p.get("materials", [])
    for mat_entry in materials:
        name = mat_entry.get("name", "")
        isotope = mat_entry.get("isotope", mat_entry.get("mat", ""))
        if isinstance(isotope, str) and not isotope.isdigit():
            mat_num = get_mat_number(isotope)
        else:
            mat_num = int(isotope)
        jsigz = int(mat_entry.get("jsigz", 1))
        dtemp = mat_entry.get("dtemp", 293.6)
        lines.append(f"'{name}' {mat_num} {jsigz} {dtemp} /")
    lines.append("/")

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse DTFR card lines into a parameter dict."""
    # Card 1: nin nout npend nplot
    c1 = parse_card_values(card_lines[0])
    # Card 2: iprint ifilm iedit
    c2 = parse_card_values(card_lines[1])

    iedit = int(c2[2]) if len(c2) > 2 else 0

    result: dict = {
        "nin": int(c1[0]),
        "nout": int(c1[1]),
        "npend": int(c1[2]) if len(c1) > 2 else 0,
        "nplot": int(c1[3]) if len(c1) > 3 else 0,
        "iprint": int(c2[0]) if len(c2) > 0 else 0,
        "ifilm": int(c2[1]) if len(c2) > 1 else 0,
        "iedit": iedit,
    }

    idx = 2

    if iedit == 0:
        # Card 3: nlmax ng iptotl ipingp itabl ned ntherm
        c3 = parse_card_values(card_lines[idx])
        idx += 1
        nlmax = int(c3[0])
        ng = int(c3[1])
        iptotl = int(c3[2])
        ipingp = int(c3[3]) if len(c3) > 3 else 0
        itabl = int(c3[4]) if len(c3) > 4 else 0
        ned = int(c3[5]) if len(c3) > 5 else 0
        ntherm = int(c3[6]) if len(c3) > 6 else 0

        result.update({
            "nlmax": nlmax,
            "ng": ng,
            "iptotl": iptotl,
            "ipingp": ipingp,
            "itabl": itabl,
            "ntherm": ntherm,
        })

        # Card 3a (if ntherm > 0): mti mtc nlc
        if ntherm > 0:
            c3a = parse_card_values(card_lines[idx])
            idx += 1
            result["mti"] = int(c3a[0])
            result["mtc"] = int(c3a[1]) if len(c3a) > 1 else 0
            result["nlc"] = int(c3a[2]) if len(c3a) > 2 else 0

        # Card 4: iptotl-3 edit names
        num_names = iptotl - 3
        if num_names > 0:
            result["edit_names"] = _extract_quoted_strings(card_lines[idx])
            idx += 1

        # Card 5: ned triplets of jpos mt mult
        edits: list[list[int]] = []
        for _ in range(ned):
            c5 = parse_card_values(card_lines[idx])
            idx += 1
            edits.append([int(c5[0]), int(c5[1]), int(c5[2])])
        if edits:
            result["edits"] = edits
    else:
        # Card 6 (iedit=1, CLAW): nlmax ng
        c6 = parse_card_values(card_lines[idx])
        idx += 1
        result["nlmax"] = int(c6[0])
        result["ng"] = int(c6[1])

    # Card 7: nptabl ngp
    c7 = parse_card_values(card_lines[idx])
    idx += 1
    result["nptabl"] = int(c7[0])
    result["ngp"] = int(c7[1]) if len(c7) > 1 else 0

    # Card 8: materials (terminated by bare /)
    materials: list[dict] = []
    while idx < len(card_lines):
        line = card_lines[idx].strip()
        idx += 1
        # Bare / terminates material list
        if line == "/":
            break
        # Parse: 'hisnam' mat jsigz dtemp /
        name = parse_quoted_string(line) if "'" in line else ""
        # Remove quoted portion and trailing /
        rest = line
        if "'" in rest:
            last_quote = rest.rindex("'")
            rest = rest[last_quote + 1 :]
        if rest.strip().endswith("/"):
            rest = rest.strip()[:-1]
        vals = rest.split()
        mat_entry: dict = {"name": name}
        if len(vals) > 0:
            mat_entry["mat"] = int(vals[0])
        if len(vals) > 1:
            mat_entry["jsigz"] = int(vals[1])
        if len(vals) > 2:
            mat_entry["dtemp"] = float(vals[2])
        materials.append(mat_entry)

    if materials:
        result["materials"] = materials

    return result
