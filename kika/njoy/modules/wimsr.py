"""WIMSR — prepare WIMS-D/WIMS-E library data from GENDF tapes."""

from __future__ import annotations

from ._base import Lines, parse_iprint, parse_card_values


def generate(p: dict) -> Lines:
    from ..isotopes import get_mat_number

    ngendf = p.get("ngendf", "")
    nout = p.get("nout", "")

    iprint = parse_iprint(p.get("iprint"))
    iverw = int(p.get("iverw", 4))
    igroup = int(p.get("igroup", 0))

    mat_str = p.get("mat", "U235")
    mat = get_mat_number(mat_str)

    rdfid = p.get("rdfid")
    if rdfid is None:
        rdfid = float(mat)

    iburn = int(p.get("iburn", 0))

    # Card 4 parameters
    ntemp = int(p.get("ntemp", 0))
    nsigz = int(p.get("nsigz", 0))
    sgref = p.get("sgref", 1e10)
    ires = int(p.get("ires", 0))
    sigp = p.get("sigp", 0.0)
    mti = int(p.get("mti", 0))
    mtc = int(p.get("mtc", 0))
    ip1opt = int(p.get("ip1opt", 1))
    inorf = int(p.get("inorf", 0))
    isof = int(p.get("isof", 0))
    ifprod = int(p.get("ifprod", 0))
    jp1 = int(p.get("jp1", 0))

    lines: Lines = [
        "-- prepare WIMS library data",
        "wimsr",
    ]

    # Card 1: ngendf nout
    lines.append(f"{ngendf} {nout}")

    # Card 2: iprint iverw igroup
    card2_parts = []
    if iprint is not None:
        card2_parts.append(str(iprint))
    else:
        card2_parts.append("0")
    card2_parts.append(str(iverw))
    card2_parts.append(str(igroup))
    lines.append(" ".join(card2_parts) + " /")

    # Card 2a: ngnd nfg nrg [igref] (only when igroup=9)
    if igroup == 9:
        ngnd = p.get("ngnd")
        nfg = p.get("nfg")
        nrg = p.get("nrg")
        igref = p.get("igref")
        if ngnd is None or nfg is None or nrg is None:
            raise ValueError(
                "igroup=9 requires custom group parameters: ngnd, nfg, nrg."
            )
        card2a_parts = [str(ngnd), str(nfg), str(nrg)]
        if igref is not None:
            card2a_parts.append(str(igref))
        lines.append(" ".join(card2a_parts) + " /")

    # Card 3: mat nfid(=0) rdfid iburn
    lines.append(f"{mat} 0 {rdfid} {iburn} /")

    # Card 4: ntemp nsigz sgref ires sigp mti mtc ip1opt inorf isof ifprod jp1
    card4_parts = [
        str(ntemp), str(nsigz), str(sgref), str(ires), str(sigp),
        str(mti), str(mtc), str(ip1opt), str(inorf), str(isof),
        str(ifprod), str(jp1),
    ]
    lines.append(" ".join(card4_parts) + " /")

    # Card 5: ntis efiss (only when iburn > 0)
    if iburn > 0:
        burnup = p.get("burnup", {})
        ntis = int(burnup.get("ntis", 0))
        efiss = burnup.get("efiss", 0.0)
        lines.append(f"{ntis} {efiss} /")

        # Card 6a: capture product (identa, yield)
        capture = burnup.get("capture")
        if capture is not None:
            lines.append(f"{capture[0]} {capture[1]} /")

        # Card 6b: decay product (identa, lambda)
        decay = burnup.get("decay")
        if decay is not None:
            lines.append(f"{decay[0]} {decay[1]} /")

        # Card 6c: fission products (repeated ntis-2 times)
        fission_products = burnup.get("fission_products", [])
        for fp in fission_products:
            lines.append(f"{fp[0]} {fp[1]} /")

    # Card 7: Goldstein lambdas
    lambdas = p.get("lambdas")
    if lambdas is not None:
        if isinstance(lambdas, list):
            lambda_strs = [str(v) for v in lambdas]
        else:
            lambda_strs = str(lambdas).split()
        lines.append(" ".join(lambda_strs) + " /")
    else:
        # Default: 13 values of 0.0 for 69-group structure
        lines.append(" ".join(["0.0"] * 13) + " /")

    # Card 8: current spectrum values (only when jp1 > 0)
    if jp1 > 0:
        p1flx = p.get("p1flx")
        if p1flx is None:
            raise ValueError("jp1 > 0 requires 'p1flx' (current spectrum values).")
        if isinstance(p1flx, list):
            p1flx_strs = [str(v) for v in p1flx]
        else:
            p1flx_strs = str(p1flx).split()
        lines.append(" ".join(p1flx_strs) + " /")

    return lines


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def parse(card_lines: list[str]) -> dict:
    """Parse WIMSR card lines into a parameter dict."""
    # Card 1: ngendf nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: iprint iverw igroup
    c2 = parse_card_values(card_lines[1])

    iprint = int(c2[0])
    iverw = int(c2[1])
    igroup = int(c2[2])

    result: dict = {
        "ngendf": int(c1[0]),
        "nout": int(c1[1]),
        "iprint": iprint,
        "iverw": iverw,
        "igroup": igroup,
    }

    idx = 2

    # Card 2a: ngnd nfg nrg [igref] (only when igroup=9)
    if igroup == 9:
        c2a = parse_card_values(card_lines[idx])
        result["ngnd"] = int(c2a[0])
        result["nfg"] = int(c2a[1])
        result["nrg"] = int(c2a[2])
        if len(c2a) > 3:
            result["igref"] = int(c2a[3])
        idx += 1

    # Card 3: mat nfid rdfid iburn
    c3 = parse_card_values(card_lines[idx])
    mat = int(c3[0])
    iburn = int(c3[3])
    result["mat"] = mat
    result["rdfid"] = float(c3[2])
    result["iburn"] = iburn
    idx += 1

    # Card 4: ntemp nsigz sgref ires sigp mti mtc ip1opt inorf isof ifprod jp1
    c4 = parse_card_values(card_lines[idx])
    result["ntemp"] = int(c4[0])
    result["nsigz"] = int(c4[1])
    result["sgref"] = float(c4[2])
    result["ires"] = int(c4[3])
    result["sigp"] = float(c4[4])
    result["mti"] = int(c4[5])
    result["mtc"] = int(c4[6])
    result["ip1opt"] = int(c4[7])
    result["inorf"] = int(c4[8])
    result["isof"] = int(c4[9])
    result["ifprod"] = int(c4[10])
    jp1 = int(c4[11])
    result["jp1"] = jp1
    idx += 1

    # Card 5: ntis efiss (only when iburn > 0)
    if iburn > 0:
        c5 = parse_card_values(card_lines[idx])
        burnup: dict = {
            "ntis": int(c5[0]),
            "efiss": float(c5[1]),
        }
        ntis = int(c5[0])
        idx += 1

        # Card 6a: capture product
        if idx < len(card_lines):
            c6a = parse_card_values(card_lines[idx])
            burnup["capture"] = (int(c6a[0]), float(c6a[1]))
            idx += 1

        # Card 6b: decay product
        if idx < len(card_lines):
            c6b = parse_card_values(card_lines[idx])
            burnup["decay"] = (int(c6b[0]), float(c6b[1]))
            idx += 1

        # Card 6c: fission products (ntis - 2 entries)
        fission_products = []
        for _ in range(ntis - 2):
            if idx < len(card_lines):
                c6c = parse_card_values(card_lines[idx])
                fission_products.append((int(c6c[0]), float(c6c[1])))
                idx += 1
        if fission_products:
            burnup["fission_products"] = fission_products

        result["burnup"] = burnup

    # Card 7: lambdas
    if idx < len(card_lines):
        result["lambdas"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1

    # Card 8: current spectrum values (only when jp1 > 0)
    if jp1 > 0 and idx < len(card_lines):
        result["p1flx"] = [float(v) for v in parse_card_values(card_lines[idx])]
        idx += 1

    return result
