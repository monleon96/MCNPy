"""ACER — generate ACE-format data for Monte Carlo codes."""

from __future__ import annotations

from ._base import Lines, _get, parse_iprint, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    import sys
    print(f"[ACER DEBUG] __file__ = {__file__}", file=sys.stderr, flush=True)
    print(f"[ACER DEBUG] iopt raw = {p.get('iopt')!r}", file=sys.stderr, flush=True)

    from ..isotopes import get_mat_number

    nendf = p.get("nendf", "")
    npend = p.get("npend", "")
    ngend = p.get("ngend", "")
    nace = p.get("nace", "")
    ndir = p.get("ndir", "")

    iprint = parse_iprint(p.get("iprint"))
    if iprint is None:
        iprint = 1  # default

    itype = _get(p, "itype", 1)
    suff = float(_get(p, "suff", 0.0))
    iopt = _get(p, "iopt", "1")
    abs_iopt = abs(int(iopt))

    print(f"[ACER DEBUG] abs_iopt={abs_iopt}, type={type(abs_iopt)}", file=sys.stderr, flush=True)

    suff_trunc = int(suff * 100) / 100 if suff > 0 else suff

    # nxtra iz,aw pairs
    nxtra_pairs = p.get("nxtra", [])
    nxtra_count = len(nxtra_pairs) if nxtra_pairs else 0

    # Resolve isotope/MAT only for modes that need it
    if abs_iopt not in (7, 8):
        isotope = p.get("matd", "U235")
        mat_num = get_mat_number(isotope)
        tempd = p.get("tempd", "")
        hk = f"{isotope} @ {tempd} K ACE data"
    else:
        isotope = None
        mat_num = None
        tempd = None
        hk = "ACE data"

    # Card 1 & 2
    lines: Lines = [
        "-- generate ACE file",
        "acer",
        f"{nendf} {npend} {ngend} {nace} {ndir}",
    ]

    # Card 2: iopt iprint itype suff [nxtra]
    if nxtra_count > 0:
        lines.append(f"{iopt} {iprint} {itype} {suff_trunc:.2f} {nxtra_count} /")
    else:
        lines.append(f"{iopt} {iprint} {itype} {suff_trunc:.2f} /")

    # Card 3: hk
    lines.append(f"'{hk}' /")

    # Card 4: iz,aw pairs (when nxtra > 0)
    if nxtra_count > 0:
        tokens = []
        for pair in nxtra_pairs:
            tokens.append(str(pair[0]))
            tokens.append(str(pair[1]))
        lines.append(" ".join(tokens) + " /")

    # Branch by abs_iopt
    if abs_iopt == 1:
        _emit_fast(lines, p, mat_num, tempd)
    elif abs_iopt == 2:
        _emit_thermal(lines, p, mat_num, tempd)
    elif abs_iopt == 3:
        lines.append(f"{mat_num} {tempd} /")
    elif abs_iopt == 4:
        lines.append(f"{mat_num} /")
    elif abs_iopt == 5:
        lines.append(f"{mat_num} /")
    elif abs_iopt in (7, 8):
        pass  # no data cards beyond 1-3
    else:
        raise ValueError(f"ACER iopt={iopt} is not yet implemented")

    return lines


def _emit_fast(lines: Lines, p: dict, mat_num: int, tempd) -> None:
    """Append Card 5, 6, 7 for fast data (iopt=1)."""
    # Card 5: matd tempd
    lines.append(f"{mat_num} {tempd} /")

    # Card 6: newfor iopp ismooth
    newfor = _get(p, "newfor", 1)
    iopp = _get(p, "iopp", 1)
    ismooth = _get(p, "ismooth", 1)

    if ismooth != 1:
        lines.append(f"{newfor} {iopp} {ismooth} /")
    elif iopp != 1:
        lines.append(f"{newfor} {iopp} /")
    elif newfor != 1:
        lines.append(f"{newfor} /")
    else:
        lines.append("/")

    # Card 7: thin1 [thin2] [thin3]
    thin1 = p.get("thin1")
    if thin1 is not None:
        thin_parts = [str(thin1)]
        thin2 = p.get("thin2")
        thin3 = p.get("thin3")
        if thin2 is not None:
            thin_parts.append(str(thin2))
        if thin3 is not None:
            thin_parts.append(str(thin3))
        lines.append(" ".join(thin_parts) + " /")
    else:
        lines.append("/")


def _emit_thermal(lines: Lines, p: dict, mat_num: int, tempd) -> None:
    """Append Card 8, 8a, and 9 for thermal scattering (iopt=2)."""
    tname = p.get("tname")
    if tname is None:
        raise ValueError("ACER iopt=2 requires 'tname' (thermal ZAID name)")

    iza_raw = p.get("iza")
    if iza_raw is None:
        raise ValueError("ACER iopt=2 requires 'iza' (moderator ZA values)")

    # Accept list, space-separated string, or single value
    if isinstance(iza_raw, str):
        iza_list = iza_raw.split()
    elif isinstance(iza_raw, (list, tuple)):
        iza_list = [str(v) for v in iza_raw]
    else:
        iza_list = [str(iza_raw)]

    nza = len(iza_list)

    mti = p.get("mti")
    if mti is None:
        raise ValueError("ACER iopt=2 requires 'mti' (inelastic MT number)")

    # Card 8: matd tempd tname nza
    lines.append(f"{mat_num} {tempd} '{tname}' {nza} /")

    # Card 8a: iza values
    lines.append(" ".join(iza_list) + " /")

    # Card 9: mti nbint mte ielas nmix emax iwt
    # Use positional cascade — only emit trailing params when non-default
    nbint = _get(p, "nbint", 16)
    mte = _get(p, "mte", 0)
    ielas = _get(p, "ielas", 0)
    nmix = _get(p, "nmix", 1)
    emax = _get(p, "emax", 1000.0)
    iwt = _get(p, "iwt", 2)

    card9_parts = [str(mti)]

    if iwt != 2:
        card9_parts.extend([
            str(nbint), str(mte), str(ielas),
            str(nmix), str(emax), str(iwt),
        ])
    elif emax != 1000.0:
        card9_parts.extend([
            str(nbint), str(mte), str(ielas),
            str(nmix), str(emax),
        ])
    elif nmix != 1:
        card9_parts.extend([
            str(nbint), str(mte), str(ielas), str(nmix),
        ])
    elif ielas != 0:
        card9_parts.extend([
            str(nbint), str(mte), str(ielas),
        ])
    elif mte != 0:
        card9_parts.extend([str(nbint), str(mte)])
    elif nbint != 16:
        card9_parts.append(str(nbint))

    lines.append(" ".join(card9_parts) + " /")


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def parse(card_lines: list[str]) -> dict:
    """Parse ACER card lines into a parameter dict."""
    # Card 1: nendf npend ngend nace ndir
    c1 = parse_card_values(card_lines[0])
    # Card 2: iopt iprint itype suff [nxtra]
    c2 = parse_card_values(card_lines[1])
    # Card 3: 'hk' /
    hk = parse_quoted_string(card_lines[2])

    iopt = int(c2[0])
    abs_iopt = abs(iopt)
    nxtra = int(c2[4]) if len(c2) > 4 else 0

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "ngend": int(c1[2]),
        "nace": int(c1[3]),
        "ndir": int(c1[4]),
        "iopt": iopt,
        "iprint": int(c2[1]),
        "itype": int(c2[2]),
        "suff": float(c2[3]),
        "hk": hk,
    }

    # Card 4: iz,aw pairs when nxtra > 0
    offset = 3  # next card index after Card 3
    if nxtra > 0:
        c4 = parse_card_values(card_lines[offset])
        pairs = []
        for i in range(0, nxtra * 2, 2):
            pairs.append((int(c4[i]), float(c4[i + 1])))
        result["nxtra"] = pairs
        offset += 1

    if abs_iopt == 1:
        _parse_fast(result, card_lines, offset)
    elif abs_iopt == 2:
        _parse_thermal(result, card_lines, offset)
    elif abs_iopt == 3:
        _parse_dosimetry(result, card_lines, offset)
    elif abs_iopt == 4:
        _parse_photoatomic(result, card_lines, offset)
    elif abs_iopt == 5:
        _parse_photonuclear(result, card_lines, offset)
    # iopt 7/8: no additional cards to parse

    return result


def _parse_fast(result: dict, card_lines: list[str], offset: int = 3) -> None:
    """Parse ACER iopt=1 (fast) cards."""
    # Card 5: matd tempd
    c5 = parse_card_values(card_lines[offset])
    result["matd"] = int(c5[0])
    result["tempd"] = float(c5[1])

    # Card 6: newfor iopp ismooth (may be defaulted "/")
    if offset + 1 < len(card_lines):
        c6 = parse_card_values(card_lines[offset + 1])
        if len(c6) > 0:
            result["newfor"] = int(c6[0])
        if len(c6) > 1:
            result["iopp"] = int(c6[1])
        if len(c6) > 2:
            result["ismooth"] = int(c6[2])

    # Card 7: thin1 [thin2] [thin3] (may be defaulted "/")
    if offset + 2 < len(card_lines):
        c7 = parse_card_values(card_lines[offset + 2])
        if len(c7) > 0:
            result["thin1"] = float(c7[0])
        if len(c7) > 1:
            result["thin2"] = float(c7[1])
        if len(c7) > 2:
            result["thin3"] = float(c7[2])


def _parse_thermal(result: dict, card_lines: list[str], offset: int = 3) -> None:
    """Parse ACER iopt=2 (thermal) cards."""
    # Card 8: matd tempd 'tname' nza
    line8 = card_lines[offset].strip()
    # Extract tname from quotes
    tname = parse_quoted_string(line8)
    # Parse numeric values (before and after the quoted string)
    before_quote = line8[: line8.index("'")].strip()
    after_quote = line8[line8.rindex("'") + 1 :].strip()
    if after_quote.endswith("/"):
        after_quote = after_quote[:-1].strip()
    before_vals = before_quote.split()
    after_vals = after_quote.split() if after_quote else []

    result["matd"] = int(before_vals[0])
    result["tempd"] = float(before_vals[1])
    result["tname"] = tname
    nza = int(after_vals[0]) if after_vals else 1

    # Card 8a: iza values
    result["iza"] = [int(v) for v in parse_card_values(card_lines[offset + 1])]

    # Card 9: mti [nbint] [mte] [ielas] [nmix] [emax] [iwt]
    c9 = parse_card_values(card_lines[offset + 2])
    result["mti"] = int(c9[0])
    if len(c9) > 1:
        result["nbint"] = int(c9[1])
    if len(c9) > 2:
        result["mte"] = int(c9[2])
    if len(c9) > 3:
        result["ielas"] = int(c9[3])
    if len(c9) > 4:
        result["nmix"] = int(c9[4])
    if len(c9) > 5:
        result["emax"] = float(c9[5])
    if len(c9) > 6:
        result["iwt"] = int(c9[6])


def _parse_dosimetry(result: dict, card_lines: list[str], offset: int = 3) -> None:
    """Parse ACER iopt=3 (dosimetry) cards."""
    c = parse_card_values(card_lines[offset])
    result["matd"] = int(c[0])
    result["tempd"] = float(c[1])


def _parse_photoatomic(result: dict, card_lines: list[str], offset: int = 3) -> None:
    """Parse ACER iopt=4 (photo-atomic) cards."""
    c = parse_card_values(card_lines[offset])
    result["matd"] = int(c[0])


def _parse_photonuclear(result: dict, card_lines: list[str], offset: int = 3) -> None:
    """Parse ACER iopt=5 (photo-nuclear) cards."""
    c = parse_card_values(card_lines[offset])
    result["matd"] = int(c[0])
