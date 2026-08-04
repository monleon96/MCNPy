"""
Plots for SINBAD shielding benchmarks.

Deliberately few, and each one answers a question a user actually asks. The
C/E plot carries the experimental uncertainty as declared by the entry, which
for a legacy benchmark usually means an unresolved total -- an assumption the
package makes visible rather than one it hides. See
:meth:`kika.sinbad.SinbadBenchmark.unresolved`.

An entry with several measurement systems is drawn as small multiples, one
panel per system, rather than as one axis carrying every foil at once: the
reaction rates of five different reactions share no scale, and C/E curves that
belong to different detectors are not a single series.
"""

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt

# kika light palette, first slots. Colourblind-safe and validated for the
# all-pairs case, which is what a scatter/line chart of several libraries needs.
_PALETTE = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9",
]

_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e6e5e1"
_AXIS = "#c9c8c3"

#: Sequential single-hue ramp, light to dark. Depth is an ordered magnitude,
#: not an identity, so it gets a ramp rather than categorical hues -- a rainbow
#: over fourteen positions would imply distinctions that are not there.
_DEPTH_RAMP = [
    "#cfe3f2", "#a9cce7", "#82b4db", "#5b9bcf", "#3a82bf",
    "#1f6aa8", "#0f5490", "#0a3f73", "#062c55", "#031c39",
]

__all__ = [
    "plot_ce",
    "plot_sensitivity",
    "plot_sensitivity_depth",
    "plot_uncertainty_budget",
]


def _style(ax) -> None:
    ax.grid(True, lw=0.5, color=_GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED)


def _colours(libraries) -> dict:
    """Colour follows the library, never its rank in a filtered subset."""
    return {lib: _PALETTE[i % len(_PALETTE)] for i, lib in enumerate(libraries)}


def _draw_ce_panel(ax, df, colours, uncertainty, direct_labels) -> None:
    ax.axhline(1.0, color=_AXIS, lw=1, zorder=1)
    ends = []
    for lib in sorted(df["library"].unique()):
        sub = df[df["library"] == lib].sort_values("depth_cm")
        err = (sub["ce"] * sub["exp_rel_unc"]) if uncertainty else None
        ax.errorbar(
            sub["depth_cm"], sub["ce"], yerr=err,
            lw=2, marker="o", ms=6, capsize=3, color=colours[lib],
            label=lib, zorder=3, markeredgecolor="white", markeredgewidth=1.2,
        )
        ends.append([sub["ce"].iloc[-1], sub["depth_cm"].iloc[-1], lib])

    if not direct_labels:
        return
    # Libraries can land on top of each other at the deepest position; push the
    # direct labels apart so both stay readable.
    span = df["ce"].max() - df["ce"].min()
    gap = max(span * 0.09, 1e-6)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y_lab, x_end, lib in ends:
        ax.annotate(
            lib, (x_end, y_lab), color=colours[lib], fontsize=10, fontweight="bold",
            xytext=(9, 0), textcoords="offset points", va="center",
        )
    ax.set_xlim(right=df["depth_cm"].max() * 1.34)


def plot_ce(
    benchmark,
    system: Optional[str] = None,
    uncertainty: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """
    Plot C/E against detector depth, one line per nuclear data library.

    With one measurement system, or when ``system`` selects one, this is a
    single axis with direct labels. With several it becomes small multiples,
    one panel per system, sharing a legend.

    Parameters
    ----------
    benchmark : SinbadBenchmark
        The benchmark to plot.
    system : str, optional
        Restrict to one measurement system -- an identifier (``"FOIL-AL27"``),
        a target nuclide (``"Al27"``), or any unambiguous fragment.
    uncertainty : bool, default True
        Draw the experimental uncertainty as error bars.
    figsize : tuple of float, optional
        Figure size. Defaults to a size appropriate to the panel count.
    title : str, optional
        Figure title. Defaults to the benchmark identifier.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis. Only valid for the single-panel case.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray of Axes
    """
    df = benchmark.ce(system=system)
    libraries = sorted(df["library"].unique())
    colours = _colours(benchmark.libraries)
    systems = list(dict.fromkeys(df["system"]))

    caption = ("error bars: experimental uncertainty as declared by the entry"
               if uncertainty else None)

    # -- single panel ----------------------------------------------------
    if len(systems) == 1 or ax is not None:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (7.6, 4.6))
        _draw_ce_panel(ax, df, colours, uncertainty, direct_labels=True)
        label = benchmark.system(systems[0]).reaction if len(systems) == 1 else ""
        ax.set_xlabel("detector depth (cm)", color=_MUTED)
        ax.set_ylabel("C/E", color=_MUTED)
        ax.set_title(
            title or f"{benchmark.id} · {label}",
            color=_INK, loc="left", fontweight="bold", pad=24,
        )
        if caption:
            ax.text(0, 1.02, caption, transform=ax.transAxes,
                    fontsize=8.5, color=_MUTED)
        _style(ax)
        ax.legend(frameon=False, loc="lower left", labelcolor=_MUTED)
        if show:
            plt.show()
        return ax

    # -- small multiples -------------------------------------------------
    ncols = min(3, len(systems))
    nrows = math.ceil(len(systems) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (4.6 * ncols, 3.3 * nrows),
        sharex=True, squeeze=False,
    )
    flat = axes.ravel()
    for panel, sys_id in zip(flat, systems):
        _draw_ce_panel(panel, df[df["system"] == sys_id], colours,
                       uncertainty, direct_labels=False)
        panel.set_title(benchmark.system(sys_id).reaction,
                        color=_INK, loc="left", fontsize=10.5, fontweight="bold")
        _style(panel)
    for panel in flat[len(systems):]:
        panel.set_visible(False)
    # The bottom row is ragged when the panel count is not a multiple of ncols;
    # label the lowest *visible* panel of each column, not the lowest slot.
    for col in range(ncols):
        for row in range(nrows - 1, -1, -1):
            if axes[row][col].get_visible():
                axes[row][col].set_xlabel("detector depth (cm)", color=_MUTED)
                axes[row][col].tick_params(labelbottom=True)
                break
    for row in axes:
        row[0].set_ylabel("C/E", color=_MUTED)

    fig.suptitle(title or f"{benchmark.id} · C/E by activation foil",
                 color=_INK, x=0.01, ha="left", fontweight="bold")
    if caption:
        fig.text(0.01, 0.945, caption, fontsize=8.5, color=_MUTED)
    handles = [
        plt.Line2D([], [], color=colours[lib], lw=2, marker="o", label=lib)
        for lib in libraries
    ]
    fig.legend(handles=handles, frameon=False, ncol=len(libraries),
               loc="lower left", bbox_to_anchor=(0.01, 0.0), labelcolor=_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    if show:
        plt.show()
    return axes


def _ramp(n: int) -> list:
    """``n`` steps of the sequential ramp, spread across its full range."""
    if n <= 1:
        return [_DEPTH_RAMP[-1]]
    last = len(_DEPTH_RAMP) - 1
    return [_DEPTH_RAMP[round(i * last / (n - 1))] for i in range(n)]


def plot_sensitivity_depth(
    benchmark,
    system: str,
    mt: int = 4,
    figsize: Tuple[float, float] = (8.0, 4.6),
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """
    One reaction's sensitivity profile at every measured depth.

    The question a set of profiles down a shield answers is how the response
    stops being sensitive to the same energies as it goes deeper. Drawing them
    on one axis with a light-to-dark ramp shows that directly; one panel per
    position would not, because the comparison *is* the point.

    Parameters
    ----------
    benchmark : SinbadBenchmark
        Entry to read the profiles from. Requires ``h5py`` for HDF5 payloads.
    system : str
        Measurement system id, target nuclide, or an unambiguous fragment.
    mt : int, default 4
        MT number of the reaction to draw. 4 is inelastic scattering, which is
        what governs deep penetration in iron.
    figsize : tuple of float, default (8.0, 4.6)
        Figure size, used only when ``ax`` is None.
    title : str, optional
        Plot title. Defaults to the system and reaction.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis instead of creating a figure.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If no profile in the system carries the requested MT.
    """
    sets = benchmark.sensitivities(system)
    sets = [s for s in sets if mt in s.mts]
    if not sets:
        available = sorted({m for s in benchmark.sensitivities(system) for m in s.mts})
        raise KeyError(f"no profile with MT={mt} for {system!r}; available: {available}")

    depth = {m.id: m.depth_cm for m in benchmark.measurements()}
    sets.sort(key=lambda s: (depth.get(s.measurement, float("inf")), s.position))
    colours = _ramp(len(sets))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    label = sets[0].reactions[sets[0].mts.index(mt)]
    lo, hi = None, None
    for s, colour in zip(sets, colours):
        df = s.to_dataframe()
        sub = df[df["mt"] == mt]
        d = depth.get(s.measurement)
        ax.step(
            sub["e_mid"], sub["sensitivity"], where="mid", lw=2, color=colour,
            label=f"{s.position}" + (f" · {d:.0f} cm" if d is not None else ""),
            zorder=3,
        )
        # A threshold reaction is flat zero over most of a 12-decade grid.
        # Spending the axis on that hides the part anyone is looking at, so the
        # range follows the support of the data rather than the grid.
        live = sub[sub["sensitivity"].abs() > 1e-12]
        if not live.empty:
            lo = live["e_low"].min() if lo is None else min(lo, live["e_low"].min())
            hi = live["e_high"].max() if hi is None else max(hi, live["e_high"].max())

    ax.axhline(0.0, color=_AXIS, lw=1, zorder=1)
    ax.set_xscale("log")
    if lo and hi:
        ax.set_xlim(lo * 0.7, hi * 1.4)
    ax.set_xlabel("energy (MeV)", color=_MUTED)
    ax.set_ylabel(f"sensitivity to {sets[0].target_nuclide} {label}", color=_MUTED)
    ax.set_title(
        title or f"{system} · {sets[0].target_nuclide} {label} · by depth",
        color=_INK, loc="left", fontweight="bold", pad=24,
    )
    ax.text(
        0, 1.02,
        f"{sets[0].convention} · {sets[0].nuclear_data_library}",
        transform=ax.transAxes, fontsize=8.5, color=_MUTED,
    )
    _style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8.5, labelcolor=_MUTED,
              loc="lower left")

    if show:
        plt.show()
    return ax


def plot_uncertainty_budget(
    benchmark,
    figsize: Tuple[float, float] = (7.6, 3.8),
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """
    Plot what the declared correlation structure is worth.

    Paired bars per scope: the uncertainty on the mean C/E computed from the
    full covariance, against the same quantity computed from its diagonal. The
    gap is what an analysis that treats the points as independent throws away.

    Parameters
    ----------
    benchmark : SinbadBenchmark
        The benchmark to plot.
    figsize : tuple of float, default (7.6, 3.8)
        Figure size, used only when ``ax`` is None.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis instead of creating a figure.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = benchmark.uncertainty_budget()
    labels = [
        s if s == "whole entry" else s.replace("FOIL-", "")
        for s in df["scope"]
    ]
    y = range(len(df))
    height = 0.38

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # 2 px of surface between the paired bars, per the mark spec.
    ax.barh([i + height / 2 + 0.01 for i in y], 100 * df["full"], height,
            color=_PALETTE[0], label="with declared correlations",
            edgecolor="white", linewidth=1)
    ax.barh([i - height / 2 - 0.01 for i in y], 100 * df["diagonal_only"], height,
            color=_PALETTE[1], label="diagonal only", edgecolor="white", linewidth=1)

    for i, row in df.iterrows():
        ax.annotate(f"{row['factor']:.1f}x", (100 * row["full"], i),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=_MUTED, fontweight="bold")

    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("uncertainty on the mean C/E (%)", color=_MUTED)
    ax.set_xlim(right=100 * df["full"].max() * 1.22)
    ax.set_title(title or f"{benchmark.id} · cost of ignoring the correlations",
                 color=_INK, loc="left", fontweight="bold", pad=24)
    ax.text(0, 1.02,
            "factor by which independence understates the aggregate",
            transform=ax.transAxes, fontsize=8.5, color=_MUTED)
    _style(ax)
    ax.grid(axis="y", visible=False)
    # Upper right: the "whole entry" row is the shortest pair, so that corner is
    # the only one guaranteed clear of the bars and their factor labels.
    ax.legend(frameon=False, loc="upper right", labelcolor=_MUTED)

    if show:
        plt.show()
    return ax


def plot_sensitivity(
    sensitivity,
    figsize: Tuple[float, float] = (7.6, 4.2),
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """
    Plot a sensitivity set, one step curve per reaction.

    Parameters
    ----------
    sensitivity : SensitivitySet
        The set to plot. Requires ``h5py`` if the package stores arrays as HDF5.
    figsize : tuple of float, default (7.6, 4.2)
        Figure size, used only when ``ax`` is None.
    title : str, optional
        Plot title. Defaults to the set identifier and target nuclide.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis instead of creating a figure.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = sensitivity.to_dataframe()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, reaction in enumerate(sensitivity.reactions):
        sub = df[df["reaction"] == reaction]
        ax.step(
            sub["e_mid"], sub["sensitivity"], where="mid",
            lw=2, color=_PALETTE[i % len(_PALETTE)], label=reaction,
        )

    ax.set_xscale("log")
    ax.set_xlabel("energy (MeV)", color=_MUTED)
    ax.set_ylabel("sensitivity coefficient", color=_MUTED)
    ax.set_title(
        title or f"{sensitivity.id} · {sensitivity.target_nuclide}",
        color=_INK, loc="left", fontweight="bold", pad=24,
    )
    ax.text(
        0, 1.02, sensitivity.convention,
        transform=ax.transAxes, fontsize=8.5, color=_MUTED,
    )
    _style(ax)
    ax.legend(frameon=False, loc="lower left", labelcolor=_MUTED)

    # A placeholder that cannot be told apart from data is worse than none.
    if sensitivity.data_origin == "syntheticDemo":
        ax.text(
            0.30, 0.55, "SYNTHETIC — NOT PHYSICS", transform=ax.transAxes,
            ha="center", va="center", fontsize=15, color="#e34948",
            alpha=0.38, fontweight="bold",
        )

    if show:
        plt.show()
    return ax
