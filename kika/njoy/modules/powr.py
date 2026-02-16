"""POWR — format multigroup data for EPRI-CELL and EPRI-CPM codes."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    ngendf = p.get("ngendf", "")
    nout = p.get("nout", "")

    lib = int(p.get("lib", 1))
    iprint = parse_iprint(p.get("iprint"))
    if iprint is None:
        iprint = 0
    iclaps = int(p.get("iclaps", 0))

    lines: Lines = [
        "-- format data for EPRI codes",
        "powr",
        f"{ngendf} {nout}",
        f"{lib} {iprint} {iclaps} /",
    ]

    if lib == 1:
        _emit_fast(lines, p, get_mat_number)
    elif lib == 2:
        _emit_thermal(lines, p, get_mat_number)
    elif lib == 3:
        _emit_cpm(lines, p, get_mat_number)
    else:
        raise ValueError(f"POWR lib={lib} is not valid (expected 1, 2, or 3)")

    return lines


# ------------------------------------------------------------------
# lib=1  Fast library
# ------------------------------------------------------------------

def _emit_fast(lines: Lines, p: dict, get_mat: callable) -> None:
    materials = p.get("materials", [])
    for mat in materials:
        matd_raw = mat.get("matd")
        if matd_raw is None:
            raise ValueError("Each fast material requires 'matd'")
        matd = get_mat(matd_raw)

        rtemp = mat.get("rtemp", "")
        iff = mat.get("iff", "")
        nsgz = mat.get("nsgz", "")
        izref = mat.get("izref", "")

        # Card 3: matd [rtemp] [iff] [nsgz] [izref]
        card3_parts = [str(matd)]
        if rtemp != "":
            card3_parts.append(str(rtemp))
        if iff != "":
            card3_parts.append(str(iff))
        if nsgz != "":
            card3_parts.append(str(nsgz))
        if izref != "":
            card3_parts.append(str(izref))
        lines.append(" ".join(card3_parts) + " /")

        # Card 4: nuclide description (only when matd > 0)
        if matd > 0:
            word = mat.get("word", "")
            lines.append(f"*{word}*/")

        # Card 5: fission spectrum title (only when matd > 0)
        if matd > 0:
            fsn = mat.get("fsn", "")
            lines.append(f"*{fsn}*/")

    # Terminator
    lines.append("0 /")


# ------------------------------------------------------------------
# lib=2  Thermal library
# ------------------------------------------------------------------

def _emit_thermal(lines: Lines, p: dict, get_mat: callable) -> None:
    materials = p.get("materials", [])
    for mat in materials:
        matd_raw = mat.get("matd")
        if matd_raw is None:
            raise ValueError("Each thermal material requires 'matd'")
        matd = get_mat(matd_raw)

        idtemp = mat.get("idtemp", "")
        name = mat.get("name", "")

        # Card 3: matd [idtemp] *name*/
        card3_parts = [str(matd)]
        if idtemp != "":
            card3_parts.append(str(idtemp))
        card3_str = " ".join(card3_parts) + f" *{name}*/"
        lines.append(card3_str)

        if matd > 0:
            # Card 4: itrc mti mtc
            itrc = mat.get("itrc", 0)
            mti = mat.get("mti", 0)
            mtc = mat.get("mtc", 0)
            lines.append(f"{itrc} {mti} {mtc} /")

            # Card 5: xi alpha mubar nu kf kc lambda sigma_s
            xi = mat.get("xi", 0.0)
            alpha = mat.get("alpha", 0.0)
            mubar = mat.get("mubar", 0.0)
            nu = mat.get("nu", 0.0)
            kf = mat.get("kf", 0)
            kc = mat.get("kc", 0)
            lam = mat.get("lambda", 0.0)
            sigma_s = mat.get("sigma_s", 0.0)
            lines.append(
                f"{xi} {alpha} {mubar} {nu} {kf} {kc} {lam} {sigma_s} /"
            )

    # Terminator
    lines.append("0 /")


# ------------------------------------------------------------------
# lib=3  CPM library
# ------------------------------------------------------------------

def _emit_cpm(lines: Lines, p: dict, get_mat: callable) -> None:
    nlib = int(p.get("nlib", 1))
    idat = int(p.get("idat", 0))
    newmat = int(p.get("newmat", 1))
    iopt = int(p.get("iopt", 0))
    mode = int(p.get("mode", 0))
    if5 = int(p.get("if5", 0))
    if4 = int(p.get("if4", 0))

    # Card 3: nlib idat newmat iopt mode if5 if4
    lines.append(f"{nlib} {idat} {newmat} {iopt} {mode} {if5} {if4} /")

    # Card 4: MAT numbers (only when iopt=0)
    if iopt == 0:
        mat_list = p.get("mat_list", [])
        mat_nums = [str(get_mat(m)) for m in mat_list]
        if mat_nums:
            lines.append(" ".join(mat_nums) + " /")
        else:
            lines.append("/")

    # Card 5: nuclide parameters (repeated per nuclide)
    nuclides = p.get("nuclides", [])
    for nuc in nuclides:
        nina = int(nuc.get("nina", 0))
        ntemp = int(nuc.get("ntemp", 0))
        nsigz = int(nuc.get("nsigz", 0))
        sgref = nuc.get("sgref", 1e10)
        ires = nuc.get("ires", 0)
        sigp = nuc.get("sigp", 0.0)
        mti = nuc.get("mti", 0)
        mtc = nuc.get("mtc", 0)
        ip1opt = nuc.get("ip1opt", 0)
        inorf = nuc.get("inorf", 0)
        pos = nuc.get("pos", 0.0)
        posr = nuc.get("posr", 0.0)

        card5_parts = [
            str(nina), str(ntemp), str(nsigz), str(sgref),
            str(ires), str(sigp), str(mti), str(mtc),
            str(ip1opt), str(inorf), str(pos), str(posr),
        ]
        lines.append(" ".join(card5_parts) + " /")

        # Card 10: lambda values (required when nina=0/3 or ires=1)
        if nina in (0, 3) or ires == 1:
            lambdas = nuc.get("lambdas", [])
            if isinstance(lambdas, list):
                lam_strs = [str(v) for v in lambdas]
            else:
                lam_strs = str(lambdas).split()
            if lam_strs:
                lines.append(" ".join(lam_strs) + " /")
            else:
                lines.append("/")

    # Cards 6-9: burnup data (only when if5 > 0)
    if if5 > 0:
        burnup_data = p.get("burnup_data", [])
        for bd in burnup_data:
            lines.append(str(bd) + " /")


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def parse(card_lines: list[str]) -> dict:
    """Parse POWR card lines into a parameter dict."""
    # Card 1: ngendf nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: lib iprint iclaps
    c2 = parse_card_values(card_lines[1])

    lib = int(c2[0])
    result: dict = {
        "ngendf": int(c1[0]),
        "nout": int(c1[1]),
        "lib": lib,
        "iprint": int(c2[1]) if len(c2) > 1 else 0,
        "iclaps": int(c2[2]) if len(c2) > 2 else 0,
    }

    if lib == 1:
        _parse_fast(result, card_lines, 2)
    elif lib == 2:
        _parse_thermal(result, card_lines, 2)
    elif lib == 3:
        _parse_cpm(result, card_lines, 2)

    return result


def _parse_fast(result: dict, card_lines: list[str], offset: int) -> None:
    """Parse lib=1 (fast) material cards."""
    materials: list[dict] = []
    idx = offset
    while idx < len(card_lines):
        c = parse_card_values(card_lines[idx])
        matd = int(c[0])
        if matd == 0:
            break
        mat: dict = {"matd": matd}
        if len(c) > 1:
            mat["rtemp"] = float(c[1])
        if len(c) > 2:
            mat["iff"] = int(c[2])
        if len(c) > 3:
            mat["nsgz"] = int(c[3])
        if len(c) > 4:
            mat["izref"] = int(c[4])
        idx += 1

        if matd > 0:
            # Card 4: *word*/
            mat["word"] = _parse_star_string(card_lines[idx])
            idx += 1
            # Card 5: *fsn*/
            mat["fsn"] = _parse_star_string(card_lines[idx])
            idx += 1

        materials.append(mat)

    result["materials"] = materials


def _parse_thermal(result: dict, card_lines: list[str], offset: int) -> None:
    """Parse lib=2 (thermal) material cards."""
    materials: list[dict] = []
    idx = offset
    while idx < len(card_lines):
        line = card_lines[idx]
        # Check for terminator before parsing
        c = parse_card_values(line)
        if int(c[0]) == 0:
            break

        # Card 3 contains matd [idtemp] *name*/
        mat: dict = {"matd": int(c[0])}
        matd = int(c[0])

        # Parse idtemp from numeric values before the star-delimited string
        if len(c) > 1 and "*" not in c[1]:
            mat["idtemp"] = int(c[1])

        # Extract the star-delimited name
        mat["name"] = _parse_star_string(line)
        idx += 1

        if matd > 0:
            # Card 4: itrc mti mtc
            c4 = parse_card_values(card_lines[idx])
            mat["itrc"] = int(c4[0])
            if len(c4) > 1:
                mat["mti"] = int(c4[1])
            if len(c4) > 2:
                mat["mtc"] = int(c4[2])
            idx += 1

            # Card 5: xi alpha mubar nu kf kc lambda sigma_s
            c5 = parse_card_values(card_lines[idx])
            if len(c5) > 0:
                mat["xi"] = float(c5[0])
            if len(c5) > 1:
                mat["alpha"] = float(c5[1])
            if len(c5) > 2:
                mat["mubar"] = float(c5[2])
            if len(c5) > 3:
                mat["nu"] = float(c5[3])
            if len(c5) > 4:
                mat["kf"] = int(c5[4])
            if len(c5) > 5:
                mat["kc"] = int(c5[5])
            if len(c5) > 6:
                mat["lambda"] = float(c5[6])
            if len(c5) > 7:
                mat["sigma_s"] = float(c5[7])
            idx += 1

        materials.append(mat)

    result["materials"] = materials


def _parse_cpm(result: dict, card_lines: list[str], offset: int) -> None:
    """Parse lib=3 (CPM) cards."""
    # Card 3: nlib idat newmat iopt mode if5 if4
    c3 = parse_card_values(card_lines[offset])
    nlib = int(c3[0])
    iopt = int(c3[3]) if len(c3) > 3 else 0
    if5 = int(c3[5]) if len(c3) > 5 else 0

    result["nlib"] = nlib
    result["idat"] = int(c3[1]) if len(c3) > 1 else 0
    result["newmat"] = int(c3[2]) if len(c3) > 2 else 1
    result["iopt"] = iopt
    result["mode"] = int(c3[4]) if len(c3) > 4 else 0
    result["if5"] = if5
    result["if4"] = int(c3[6]) if len(c3) > 6 else 0

    idx = offset + 1

    # Card 4: MAT numbers (only when iopt=0)
    if iopt == 0 and idx < len(card_lines):
        c4 = parse_card_values(card_lines[idx])
        result["mat_list"] = [int(v) for v in c4] if c4 else []
        idx += 1

    # Card 5: nuclide parameters (repeated)
    nuclides: list[dict] = []
    nmat = len(result.get("mat_list", []))
    for _ in range(nmat):
        if idx >= len(card_lines):
            break
        c5 = parse_card_values(card_lines[idx])
        nuc: dict = {}
        if len(c5) > 0:
            nuc["nina"] = int(c5[0])
        if len(c5) > 1:
            nuc["ntemp"] = int(c5[1])
        if len(c5) > 2:
            nuc["nsigz"] = int(c5[2])
        if len(c5) > 3:
            nuc["sgref"] = float(c5[3])
        if len(c5) > 4:
            nuc["ires"] = int(c5[4])
        if len(c5) > 5:
            nuc["sigp"] = float(c5[5])
        if len(c5) > 6:
            nuc["mti"] = int(c5[6])
        if len(c5) > 7:
            nuc["mtc"] = int(c5[7])
        if len(c5) > 8:
            nuc["ip1opt"] = int(c5[8])
        if len(c5) > 9:
            nuc["inorf"] = int(c5[9])
        if len(c5) > 10:
            nuc["pos"] = float(c5[10])
        if len(c5) > 11:
            nuc["posr"] = float(c5[11])
        idx += 1

        nina = nuc.get("nina", 0)
        ires = nuc.get("ires", 0)

        # Card 10: lambda values (for nina=0/3 or ires=1)
        if (nina in (0, 3) or ires == 1) and idx < len(card_lines):
            lam_vals = parse_card_values(card_lines[idx])
            nuc["lambdas"] = [float(v) for v in lam_vals]
            idx += 1

        nuclides.append(nuc)

    if nuclides:
        result["nuclides"] = nuclides


def _parse_star_string(line: str) -> str:
    """Extract text between ``*`` delimiters from a ``*text*/`` card."""
    line = line.strip()
    start = line.index("*") + 1
    end = line.index("*", start)
    return line[start:end]
