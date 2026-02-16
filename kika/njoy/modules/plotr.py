"""PLOTR — generate plot commands from ENDF/PENDF/GENDF tape data."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    nplt = p.get("nplt", 0)
    nplt0 = p.get("nplt0", 0)
    plot_def = p.get("plot_def")

    lines: Lines = [
        "-- generate plot commands",
        "plotr",
        f"{nplt} {nplt0} /",
    ]

    if plot_def is not None:
        lines.extend(_generate_plot_cards(plot_def))

    return lines


def _generate_plot_cards(plot_def) -> Lines:
    """Emit the full sequence of PLOTR inline plot commands."""
    from ..plotr_defs import PlotrJob
    from ..viewr_defs import LegendStyle

    lines: Lines = []
    page = plot_def.page

    # Card 1 — page setup (emitted once)
    lines.append(
        f"{int(page.orientation)} {int(page.font_style)}"
        f" {page.size} {int(page.page_color)} /"
    )

    for plot in plot_def.plots:
        _emit_plot(lines, plot)

    # Terminator
    lines.append("99 /")
    return lines


def _emit_plot(lines: Lines, plot) -> Lines:
    """Emit cards for one PlotrPlot (axes + all its curves)."""
    from ..viewr_defs import LegendStyle

    for i, curve in enumerate(plot.curves):
        is_first = i == 0

        if is_first:
            iplot = 1 if plot.new_page else -1
        else:
            iplot = len(plot.curves) - i

        # Card 2 — iplot iwcol factx facty xll yll ww wh wr
        lines.append(
            f"{iplot} {int(plot.window_color)}"
            f" {plot.factx} {plot.facty}"
            f" {plot.xll} {plot.yll} {plot.ww} {plot.wh} {plot.wr} /"
        )

        if is_first:
            # Card 3/3a — titles
            lines.append(f"'{plot.title1}'/")
            lines.append(f"'{plot.title2}'/")

            # Card 4 — itype jtype igrid ileg [xtag ytag]
            itype = int(plot.plot_type)
            jtype = plot.alt_axis_type
            igrid = int(plot.grid)
            ileg = int(plot.legend_style)
            card4 = f"{itype} {jtype} {igrid} {ileg}"
            if ileg == int(LegendStyle.LEGEND_BOX) and (plot.xtag != 0.0 or plot.ytag != 0.0):
                card4 += f" {plot.xtag} {plot.ytag}"
            lines.append(f"{card4} /")

            # Card 5/5a — x-axis range + label
            xr = plot.x_range
            lines.append(f"{xr.min} {xr.max} {xr.step} /")
            lines.append(f"'{plot.x_label}'/")

            # Card 6/6a — y-axis range + label
            yr = plot.y_range
            lines.append(f"{yr.min} {yr.max} {yr.step} /")
            lines.append(f"'{plot.y_label}'/")

            # Card 7/7a — z/alt-y range + label (only if jtype != 0)
            if jtype != 0:
                zr = plot.z_range if plot.z_range is not None else _default_range()
                lines.append(f"{zr.min} {zr.max} {zr.step} /")
                lines.append(f"'{plot.z_label}'/")

        # Card 8 — ENDF source
        src = curve.source
        lines.append(
            f"{src.iverf} {src.nin} {src.matd} {src.mfd} {src.mtd}"
            f" {src.temper} {src.nth} {src.ntp} {src.nkh} /"
        )

        # Card 9 — curve style (2D only, itype > 0)
        itype = int(plot.plot_type)
        if itype > 0:
            s = curve.style
            lines.append(
                f"{s.icon} {int(s.isym)} {int(s.idash)}"
                f" {int(s.iccol)} {s.ithick} {s.ishade} /"
            )

        # Card 10 — legend text (if legend_style != NONE)
        ileg = int(plot.legend_style)
        if ileg != 0:
            lines.append(f"'{curve.legend}'/")

        # Card 10a — tag position (if TAG_LABELS)
        if ileg == int(LegendStyle.TAG_LABELS):
            lines.append("0.0 0.0 0.0 /")

        # Cards 12/13 — user data (only when iverf == 0)
        if src.iverf == 0:
            lines.append(f"{curve.nform} /")
            if curve.nform == 0:
                for pt in curve.data_2d:
                    has_errors = (
                        pt.yerr_upper != 0.0
                        or pt.yerr_lower != 0.0
                        or pt.xerr_upper != 0.0
                        or pt.xerr_lower != 0.0
                    )
                    if has_errors:
                        lines.append(
                            f"{pt.x} {pt.y}"
                            f" {pt.yerr_upper} {pt.yerr_lower}"
                            f" {pt.xerr_upper} {pt.xerr_lower}"
                        )
                    else:
                        lines.append(f"{pt.x} {pt.y}")
                lines.append("/")

    # Card 2 terminator
    lines.append("99 /")
    return lines


def _default_range():
    from ..viewr_defs import AxisRange
    return AxisRange()


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def parse(card_lines: list[str]) -> dict:
    """Parse PLOTR card lines into a parameter dict."""
    # Card 0: nplt nplt0
    c0 = parse_card_values(card_lines[0])
    nplt = int(c0[0])
    nplt0 = int(c0[1]) if len(c0) > 1 else 0

    result: dict = {"nplt": nplt, "nplt0": nplt0}

    if len(card_lines) <= 1:
        return result

    idx = 1

    # Card 1: orientation font_style size page_color
    c1 = parse_card_values(card_lines[idx]); idx += 1
    result["page_setup"] = {
        "orientation": int(c1[0]),
        "font_style": int(c1[1]),
        "size": float(c1[2]),
        "page_color": int(c1[3]),
    }

    # Parse plots until outer 99 / terminator
    plots: list[dict] = []
    current_plot: dict | None = None
    itype = 1
    ileg = 0

    while idx < len(card_lines):
        vals = parse_card_values(card_lines[idx])
        if vals and vals[0] == "99":
            idx += 1
            # Check if this is the inner terminator (still more lines) or outer
            if idx < len(card_lines):
                # There might be another 99 / (outer terminator)
                vals2 = parse_card_values(card_lines[idx])
                if vals2 and vals2[0] == "99":
                    idx += 1
            break

        # Card 2: iplot iwcol factx facty xll yll ww wh wr
        c2 = vals
        iplot_val = int(c2[0])
        idx += 1

        curve: dict = {
            "iplot": iplot_val,
            "iwcol": int(c2[1]) if len(c2) > 1 else 0,
            "factx": float(c2[2]) if len(c2) > 2 else 1.0,
            "facty": float(c2[3]) if len(c2) > 3 else 1.0,
            "xll": float(c2[4]) if len(c2) > 4 else 0.0,
            "yll": float(c2[5]) if len(c2) > 5 else 0.0,
            "ww": float(c2[6]) if len(c2) > 6 else 0.0,
            "wh": float(c2[7]) if len(c2) > 7 else 0.0,
            "wr": float(c2[8]) if len(c2) > 8 else 0.0,
        }

        # Detect first curve vs additional: first curve is followed by a title card
        next_line = card_lines[idx].strip() if idx < len(card_lines) else ""
        is_first = "'" in next_line

        if is_first:
            current_plot = {"curves": []}
            plots.append(current_plot)

            # Card 3/3a — titles
            current_plot["title1"] = parse_quoted_string(card_lines[idx]); idx += 1
            current_plot["title2"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 4: itype jtype igrid ileg [xtag ytag]
            c4 = parse_card_values(card_lines[idx]); idx += 1
            itype = int(c4[0])
            jtype = int(c4[1])
            igrid = int(c4[2])
            ileg = int(c4[3])
            current_plot["itype"] = itype
            current_plot["jtype"] = jtype
            current_plot["igrid"] = igrid
            current_plot["ileg"] = ileg
            if len(c4) >= 6:
                current_plot["xtag"] = float(c4[4])
                current_plot["ytag"] = float(c4[5])

            # Card 5/5a — x-axis range + label
            c5 = parse_card_values(card_lines[idx]); idx += 1
            current_plot["x_range"] = [float(v) for v in c5[:3]]
            current_plot["x_label"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 6/6a — y-axis range + label
            c6 = parse_card_values(card_lines[idx]); idx += 1
            current_plot["y_range"] = [float(v) for v in c6[:3]]
            current_plot["y_label"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 7/7a — z-axis (only if jtype != 0)
            if jtype != 0:
                c7 = parse_card_values(card_lines[idx]); idx += 1
                current_plot["z_range"] = [float(v) for v in c7[:3]]
                current_plot["z_label"] = parse_quoted_string(card_lines[idx]); idx += 1
        else:
            itype = current_plot["itype"]
            ileg = current_plot.get("ileg", 0)

        # Card 8: iverf nin matd mfd mtd temper nth ntp nkh
        c8 = parse_card_values(card_lines[idx]); idx += 1
        source = {
            "iverf": int(c8[0]),
            "nin": int(c8[1]) if len(c8) > 1 else 0,
            "matd": int(c8[2]) if len(c8) > 2 else 0,
            "mfd": int(c8[3]) if len(c8) > 3 else 0,
            "mtd": int(c8[4]) if len(c8) > 4 else 0,
            "temper": float(c8[5]) if len(c8) > 5 else 0.0,
            "nth": int(c8[6]) if len(c8) > 6 else 0,
            "ntp": int(c8[7]) if len(c8) > 7 else 0,
            "nkh": int(c8[8]) if len(c8) > 8 else 0,
        }
        curve["source"] = source
        iverf = source["iverf"]

        # Card 9 — curve style (2D only, itype > 0)
        if itype > 0:
            c9 = parse_card_values(card_lines[idx]); idx += 1
            curve["style"] = {
                "icon": int(c9[0]),
                "isym": int(c9[1]),
                "idash": int(c9[2]),
                "iccol": int(c9[3]),
                "ithick": int(c9[4]),
                "ishade": int(c9[5]),
            }

        # Card 10 — legend text (if ileg != 0)
        if ileg != 0:
            curve["legend"] = parse_quoted_string(card_lines[idx]); idx += 1

        # Card 10a — tag position (if TAG_LABELS)
        if ileg == 2:
            c10a = parse_card_values(card_lines[idx]); idx += 1
            curve["xtag_curve"] = float(c10a[0])
            curve["ytag_curve"] = float(c10a[1])
            if len(c10a) > 2:
                curve["xpoint"] = float(c10a[2])

        # Cards 12/13 — user data (only when iverf == 0)
        if iverf == 0:
            c12 = parse_card_values(card_lines[idx]); idx += 1
            nform = int(c12[0])
            curve["nform"] = nform

            if nform == 0:
                data_2d: list[list[float]] = []
                while idx < len(card_lines):
                    stripped = card_lines[idx].strip()
                    if stripped == "/":
                        idx += 1
                        break
                    pts = parse_card_values(card_lines[idx])
                    data_2d.append([float(v) for v in pts])
                    idx += 1
                curve["data_2d"] = data_2d

        current_plot["curves"].append(curve)

    result["plots"] = plots
    return result
