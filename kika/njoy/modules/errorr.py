"""ERRORR — produce multigroup covariance data from ENDF error files."""

from __future__ import annotations

from ._base import Lines, resolve_mat, parse_iprint, fmt_float, parse_card_values, format_multiline_card, parse_multiline_card


def generate(p: dict) -> Lines:
    nendf = p.get("nendf", "")
    npend = p.get("npend", "")
    ngout = p.get("ngout", 0)
    nout = p.get("nout", "")

    matd = resolve_mat(p, "matd")

    ign = int(p.get("ign", 1))
    iwt = int(p.get("iwt", 6))
    irelco = int(p.get("irelco", 1))

    iprint = parse_iprint(p.get("iprint"))
    iprint_val = iprint if iprint is not None else 1

    mprint = int(p.get("mprint", 0))
    raw_tempin = p.get("tempin")
    if raw_tempin is None or str(raw_tempin).strip() == "":
        tempin = "{TEMP}"
    else:
        tempin = fmt_float(raw_tempin)

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
            fmt_float(float(efmean) if efmean is not None else 0.0),
            fmt_float(dap),
        ])
    elif efmean is not None:
        card7_parts.extend([
            str(int(irespr) if irespr is not None else 1),
            str(int(legord) if legord is not None else 0),
            str(int(ifissp) if ifissp is not None else 0),
            fmt_float(efmean),
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
    ]

    # Card 5a: analytic flux params (only when |iwt|=4)
    if abs(iwt) == 4:
        eb = p.get("eb")
        tb = p.get("tb")
        ec = p.get("ec")
        tc = p.get("tc")
        if any(v is None for v in (eb, tb, ec, tc)):
            raise ValueError(
                "iwt=4 requires analytic flux parameters: eb, tb, ec, tc."
            )
        lines.append(f"{fmt_float(eb)} {fmt_float(tb)} {fmt_float(ec)} {fmt_float(tc)} /")

    lines.append(" ".join(card7_parts) + " /")

    # Cards 8/8a (iread=1) or Card 10 (iread=2): manual MT selection.
    # nek=0 is hardcoded — the energy-dependent xsec correlation matrix
    # (Cards 8b/9) is not exposed.
    if iread == 1:
        mts_raw = p.get("mts")
        if mts_raw is None or (isinstance(mts_raw, str) and not mts_raw.strip()):
            mts_list: list[str] = []
        elif isinstance(mts_raw, list):
            mts_list = [str(int(v)) for v in mts_raw]
        else:
            mts_list = str(mts_raw).split()
        if not mts_list:
            raise ValueError(
                "ERRORR iread=1 requires a non-empty 'mts' list of MT numbers."
            )
        nmt = len(mts_list)
        lines.append(f"{nmt} 0 /")
        lines.append(" ".join(mts_list) + " /")
    elif iread == 2:
        pairs = p.get("mat1mt1_pairs") or []
        for entry in pairs:
            if isinstance(entry, (list, tuple)):
                m1, t1 = entry[0], entry[1]
            else:
                parts = str(entry).split()
                if len(parts) < 2:
                    continue
                m1, t1 = parts[0], parts[1]
            lines.append(f"{int(m1)} {int(t1)} /")
        lines.append("0 0 /")

    # Card 12a/12b: custom energy group boundaries (only when ign=1 or 19)
    if ign in (1, 19):
        if egn is None or (isinstance(egn, str) and not egn.strip()):
            lines.append("{NGN} /")
            lines.append("{EGN} /")
        else:
            if isinstance(egn, list):
                egn_values = [str(e) for e in egn]
            else:
                egn_values = str(egn).split()
            ngn = len(egn_values) - 1
            lines.append(f"{ngn} /")
            lines.extend(format_multiline_card(egn_values))

    return lines


def parse(card_lines: list[str]) -> dict:
    """Parse ERRORR card lines into a parameter dict."""
    # Card 1: nendf npend ngout nout
    c1 = parse_card_values(card_lines[0])
    # Card 2: matd ign iwt iprint irelco
    c2 = parse_card_values(card_lines[1])
    # Card 5: mprint tempin
    c5 = parse_card_values(card_lines[2])

    ign = int(c2[1])
    iwt = int(c2[2])

    # Card 5a: eb tb ec tc — present only when |iwt|=4
    card5a_idx: int | None = None
    c7_idx = 3
    if abs(iwt) == 4:
        card5a_idx = 3
        c7_idx = 4

    # Card 7: iread mfcov [irespr] [legord] [ifissp] [efmean] [dap]
    c7 = parse_card_values(card_lines[c7_idx])

    result: dict = {
        "nendf": int(c1[0]),
        "npend": int(c1[1]),
        "ngout": int(c1[2]),
        "nout": int(c1[3]),
        "matd": int(c2[0]),
        "ign": ign,
        "iwt": iwt,
        "iprint": int(c2[3]),
        "irelco": int(c2[4]),
        "mprint": int(c5[0]),
        "tempin": float(c5[1]),
        "iread": int(c7[0]),
        "mfcov": int(c7[1]),
    }

    if card5a_idx is not None:
        c5a = parse_card_values(card_lines[card5a_idx])
        result["eb"] = float(c5a[0])
        result["tb"] = float(c5a[1])
        result["ec"] = float(c5a[2])
        result["tc"] = float(c5a[3])

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

    # Build line map
    lm: dict[str, list[int]] = {
        "nendf": [0], "npend": [0], "ngout": [0], "nout": [0],
        "matd": [1], "ign": [1], "iwt": [1], "iprint": [1], "irelco": [1],
        "mprint": [2], "tempin": [2],
        "iread": [c7_idx], "mfcov": [c7_idx], "irespr": [c7_idx], "legord": [c7_idx],
        "ifissp": [c7_idx], "efmean": [c7_idx], "dap": [c7_idx],
    }
    if card5a_idx is not None:
        lm["eb"] = [card5a_idx]
        lm["tb"] = [card5a_idx]
        lm["ec"] = [card5a_idx]
        lm["tc"] = [card5a_idx]

    idx = c7_idx + 1

    # Cards 8/8a (iread=1) or Card 10 (iread=2): manual MT selection
    iread_val = int(c7[0])
    if iread_val == 1 and idx < len(card_lines):
        c8 = parse_card_values(card_lines[idx])
        nmt = int(c8[0]) if c8 else 0
        nek = int(c8[1]) if len(c8) > 1 else 0
        c8_idx = idx
        idx += 1
        if idx < len(card_lines) and nmt > 0:
            mts_vals, mts_consumed = parse_multiline_card(card_lines, idx)
            result["mts"] = [int(float(v)) for v in mts_vals[:nmt]]
            lm["mts"] = list(range(c8_idx, idx + mts_consumed))
            idx += mts_consumed
        if nek > 0:
            # Skip Card 8b (nek+1 energy bounds) and Card 9 (akxy matrix)
            if idx < len(card_lines):
                _, eb_consumed = parse_multiline_card(card_lines, idx)
                idx += eb_consumed
            for _ in range(nmt * nek):
                if idx >= len(card_lines):
                    break
                _, row_consumed = parse_multiline_card(card_lines, idx)
                idx += row_consumed
            result["_iread1_nek_skipped"] = True
    elif iread_val == 2 and idx < len(card_lines):
        pairs: list[list[int]] = []
        pair_lines: list[int] = []
        while idx < len(card_lines):
            toks = parse_card_values(card_lines[idx])
            if len(toks) < 2:
                break
            m1 = int(float(toks[0]))
            t1 = int(float(toks[1]))
            pair_lines.append(idx)
            idx += 1
            if m1 == 0:
                break
            pairs.append([m1, t1])
        if pairs:
            result["mat1mt1_pairs"] = pairs
            lm["mat1mt1_pairs"] = pair_lines

    # Cards 12a/12b: custom energy boundaries (ign=1 or 19, may span multiple lines)
    if ign in (1, 19) and idx < len(card_lines):
        ngn_vals = parse_card_values(card_lines[idx])
        result["ngn"] = int(ngn_vals[0])
        lm["ngn"] = [idx]
        idx += 1
        if idx < len(card_lines):
            egn_vals, egn_lines = parse_multiline_card(card_lines, idx)
            result["egn"] = [float(v) for v in egn_vals]
            lm["egn"] = list(range(idx, idx + egn_lines))

    result["_line_map"] = lm
    return result
