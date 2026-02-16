"""VIEWR — produce publication-quality plots."""

from __future__ import annotations

from ._base import Lines, parse_card_values, parse_quoted_string


def generate(p: dict) -> Lines:
    nin = p.get("infile", "")
    nps = p.get("nps", "")
    plot_def = p.get("plot_def")

    if plot_def is not None:
        nin = 5  # force stdin for inline plot commands

    lines: Lines = [
        "-- produce plots",
        "viewr",
        f"{nin} {nps}",
    ]

    if plot_def is not None:
        lines.extend(_generate_plot_cards(plot_def))

    return lines


def _generate_plot_cards(plot_def) -> Lines:
    """Emit the full sequence of VIEWR inline plot commands."""
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
    """Emit cards for one Plot (axes + all its curves)."""
    from ..viewr_defs import LegendStyle

    for i, curve in enumerate(plot.curves):
        is_first = i == 0

        if is_first:
            # iplot: 1 = new page, -1 = subplot on same page
            iplot = 1 if plot.new_page else -1
        else:
            # Additional curve on same axes
            iplot = len(plot.curves) - i

        # Card 2 — iplot, iwcol, factx, facty, xll, yll, ww, wh
        lines.append(
            f"{iplot} {int(plot.window_color)}"
            f" {plot.factx} {plot.facty}"
            f" {plot.xll} {plot.yll} {plot.ww} {plot.wh} /"
        )

        if is_first:
            # Cards 3/3a — titles
            lines.append(f"'{plot.title1}'/")
            lines.append(f"'{plot.title2}'/")

            # Card 4 — axis setup
            itype = int(plot.plot_type)
            jtype = plot.alt_axis_type
            igrid = int(plot.grid)
            ileg = int(plot.legend_style)
            lines.append(f"{itype} {jtype} {igrid} {ileg} /")

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

        # Card 8 — dummy (always present)
        lines.append("0 /")

        # Card 9 — curve style (2D only, itype >= 0)
        if int(plot.plot_type) > 0:
            s = curve.style
            lines.append(
                f"{s.icon} {int(s.isym)} {int(s.idash)}"
                f" {int(s.iccol)} {s.ithick} {s.ishade} /"
            )

        # Card 10 — legend text (if legend_style != NONE)
        if plot.legend_style != LegendStyle.NONE:
            lines.append(f"'{curve.legend}'/")

        # Card 10a — tag position (if TAG_LABELS)
        if plot.legend_style == LegendStyle.TAG_LABELS:
            lines.append(f"{curve.xtag} {curve.ytag} /")

        # Card 11 — 3D viewpoint (if itype < 0, i.e. 3D)
        if int(plot.plot_type) < 0 and curve.view3d is not None:
            v = curve.view3d
            lines.append(
                f"{v.xv} {v.yv} {v.zv} {v.x3} {v.y3} {v.z3} /"
            )

        # Card 12 — data format
        lines.append(f"{curve.nform} /")

        # Card 13 — 2D data (nform=0)
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

        # Cards 14/14a — 3D family data (nform=1)
        elif curve.nform == 1:
            for x_val, yz_pairs in curve.data_3d:
                lines.append(f"{x_val} /")
                for y_val, z_val in yz_pairs:
                    lines.append(f"{y_val} {z_val}")
                lines.append("/")
            # Empty family terminates 3D data
            lines.append("/")

    return lines


def _default_range():
    """Return a zero-initialised AxisRange without importing at module level."""
    from ..viewr_defs import AxisRange

    return AxisRange()


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def parse(card_lines: list[str]) -> dict:
    """Parse VIEWR card lines into a parameter dict."""
    # Card 0: infile nps
    c0 = parse_card_values(card_lines[0])
    infile = int(c0[0])
    nps = int(c0[1])

    result: dict = {"infile": infile, "nps": nps}

    if infile != 5 or len(card_lines) <= 1:
        return result

    # Inline plot cards — parse page setup and plot data
    idx = 1

    # Card 1: orientation font_style size page_color
    c1 = parse_card_values(card_lines[idx]); idx += 1
    result["page_setup"] = {
        "orientation": int(c1[0]),
        "font_style": int(c1[1]),
        "size": float(c1[2]),
        "page_color": int(c1[3]),
    }

    # Parse plots until 99 / terminator
    plots: list[dict] = []
    current_plot: dict | None = None
    itype = 1  # track from Card 4

    while idx < len(card_lines):
        line = card_lines[idx].strip()

        # Check for page terminator
        vals = parse_card_values(card_lines[idx])
        if vals and vals[0] == "99":
            break

        # Card 2: iplot iwcol factx facty xll yll ww wh
        c2 = vals
        iplot = int(c2[0])
        idx += 1

        curve: dict = {
            "iplot": iplot,
            "iwcol": int(c2[1]),
            "factx": float(c2[2]),
            "facty": float(c2[3]),
            "xll": float(c2[4]),
            "yll": float(c2[5]),
            "ww": float(c2[6]),
            "wh": float(c2[7]),
        }

        # Distinguish new-plot first curve from additional curve:
        # - A first curve is followed by a quoted title card ('...'/),
        # - An additional curve is followed by the dummy card (0 /).
        next_line = card_lines[idx].strip() if idx < len(card_lines) else ""
        is_first = "'" in next_line

        if is_first:
            # Start a new plot
            current_plot = {"curves": []}
            plots.append(current_plot)

            # Card 3/3a: titles
            current_plot["title1"] = parse_quoted_string(card_lines[idx]); idx += 1
            current_plot["title2"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 4: itype jtype igrid ileg
            c4 = parse_card_values(card_lines[idx]); idx += 1
            itype = int(c4[0])
            jtype = int(c4[1])
            igrid = int(c4[2])
            ileg = int(c4[3])
            current_plot["itype"] = itype
            current_plot["jtype"] = jtype
            current_plot["igrid"] = igrid
            current_plot["ileg"] = ileg

            # Card 5/5a: x-axis range + label
            c5 = parse_card_values(card_lines[idx]); idx += 1
            current_plot["x_range"] = [float(v) for v in c5[:3]]
            current_plot["x_label"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 6/6a: y-axis range + label
            c6 = parse_card_values(card_lines[idx]); idx += 1
            current_plot["y_range"] = [float(v) for v in c6[:3]]
            current_plot["y_label"] = parse_quoted_string(card_lines[idx]); idx += 1

            # Card 7/7a: z-axis (only if jtype != 0)
            if jtype != 0:
                c7 = parse_card_values(card_lines[idx]); idx += 1
                current_plot["z_range"] = [float(v) for v in c7[:3]]
                current_plot["z_label"] = parse_quoted_string(card_lines[idx]); idx += 1
        else:
            # Use current_plot's itype/ileg
            itype = current_plot["itype"]
            ileg = current_plot.get("ileg", 0)

        # Card 8: dummy (0 /)
        idx += 1

        # Card 9: curve style (2D only, itype > 0)
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

        # Card 10: legend text (if ileg != 0)
        if ileg != 0:
            curve["legend"] = parse_quoted_string(card_lines[idx]); idx += 1

        # Card 10a: tag position (if ileg == 2, TAG_LABELS)
        if ileg == 2:
            c10a = parse_card_values(card_lines[idx]); idx += 1
            curve["xtag"] = float(c10a[0])
            curve["ytag"] = float(c10a[1])

        # Card 11: 3D viewpoint (if itype < 0)
        if itype < 0:
            c11 = parse_card_values(card_lines[idx]); idx += 1
            curve["view3d"] = {
                "xv": float(c11[0]),
                "yv": float(c11[1]),
                "zv": float(c11[2]),
                "x3": float(c11[3]),
                "y3": float(c11[4]),
                "z3": float(c11[5]),
            }

        # Card 12: nform
        c12 = parse_card_values(card_lines[idx]); idx += 1
        nform = int(c12[0])
        curve["nform"] = nform

        # Card 13: data
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
        elif nform == 1:
            data_3d: list[tuple] = []
            while idx < len(card_lines):
                stripped = card_lines[idx].strip()
                if stripped == "/":
                    idx += 1
                    break  # empty family terminates
                # x_val /
                x_val = float(parse_card_values(card_lines[idx])[0]); idx += 1
                yz_pairs: list[list[float]] = []
                while idx < len(card_lines):
                    stripped2 = card_lines[idx].strip()
                    if stripped2 == "/":
                        idx += 1
                        break
                    yz = parse_card_values(card_lines[idx])
                    yz_pairs.append([float(v) for v in yz])
                    idx += 1
                data_3d.append((x_val, yz_pairs))
            curve["data_3d"] = data_3d

        current_plot["curves"].append(curve)

    result["plots"] = plots
    return result
