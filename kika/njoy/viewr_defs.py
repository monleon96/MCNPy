"""VIEWR data model — dataclasses and enums for custom plot definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ── Enums (IntEnum so they serialize as NJOY integer values) ─────────


class Orientation(IntEnum):
    PORTRAIT = 0
    LANDSCAPE = 1


class FontStyle(IntEnum):
    ROMAN = 1
    SWISS = 2


class PlotType(IntEnum):
    LIN_LIN = 1
    LIN_LOG = 2
    LOG_LIN = 3
    LOG_LOG = 4


class GridStyle(IntEnum):
    NONE = 0
    GRID_LINES = 1
    TIC_MARKS_OUTSIDE = 2
    TIC_MARKS_INSIDE = 3


class LegendStyle(IntEnum):
    NONE = 0
    LEGEND_BOX = 1
    TAG_LABELS = 2


class CurveColor(IntEnum):
    BLACK = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    MAGENTA = 4
    CYAN = 5
    BROWN = 6
    PURPLE = 7
    ORANGE = 8


class PageColor(IntEnum):
    WHITE = 0
    VERY_PALE_GREEN = 1
    VERY_PALE_CYAN = 2
    VERY_PALE_YELLOW = 3
    VERY_PALE_MAGENTA = 4
    VERY_PALE_SALMON = 5
    VERY_PALE_GREY = 6
    VERY_PALE_BLUE = 7


class SymbolType(IntEnum):
    SQUARE = 0
    OCTAGON = 1
    TRIANGLE_UP = 2
    TRIANGLE_DOWN = 3
    DIAMOND = 4
    CROSS = 5
    X_MARK = 6
    STAR1 = 7
    STAR2 = 8
    FILLED_SQUARE = 9
    FILLED_OCTAGON = 10
    FILLED_TRIANGLE_UP = 11
    FILLED_TRIANGLE_DOWN = 12
    FILLED_DIAMOND = 13
    PLUS_SQUARE = 14
    PLUS_OCTAGON = 15
    PLUS_TRIANGLE_UP = 16
    PLUS_TRIANGLE_DOWN = 17
    PLUS_DIAMOND = 18
    HOURGLASS = 19
    FILLED_HOURGLASS = 20
    BOWTIE = 21
    FILLED_BOWTIE = 22
    EXED_SQUARE = 23
    EXED_DIAMOND = 24


class DashType(IntEnum):
    SOLID = 0
    DASHED = 1
    CHAIN_DASH = 2
    CHAIN_DOT = 3
    DOT = 4
    INVISIBLE = 5


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class PageSetup:
    """Card 1 — page-level settings (emitted once per ViewrPlot)."""

    orientation: Orientation = Orientation.PORTRAIT
    font_style: FontStyle = FontStyle.SWISS
    size: float = 1.0
    page_color: PageColor = PageColor.WHITE


@dataclass
class AxisRange:
    """Axis range specification (Cards 5/6/7)."""

    min: float = 0.0
    max: float = 0.0
    step: float = 0.0


@dataclass
class CurveStyle:
    """Card 9 — per-curve visual style (2D only)."""

    icon: int = 0
    """0=line only, positive=symbol+line, negative=symbol only."""
    isym: SymbolType = SymbolType.SQUARE
    idash: DashType = DashType.SOLID
    iccol: CurveColor = CurveColor.BLACK
    ithick: int = 1
    ishade: int = 0
    """0=open, 1-7 = cross-hatch patterns, 100+ = color fill."""


@dataclass
class DataPoint2D:
    """A single 2D data point (Card 13).

    At minimum provide *x* and *y*.  Optional error-bar fields default
    to zero (no error bar).
    """

    x: float = 0.0
    y: float = 0.0
    yerr_upper: float = 0.0
    yerr_lower: float = 0.0
    xerr_upper: float = 0.0
    xerr_lower: float = 0.0


@dataclass
class View3D:
    """Card 11 — 3D viewpoint parameters."""

    xv: float = 15.0
    yv: float = -15.0
    zv: float = 15.0
    x3: float = 2.5
    y3: float = 6.5
    z3: float = 2.5


@dataclass
class Curve:
    """A single curve within a Plot."""

    style: CurveStyle = field(default_factory=CurveStyle)
    legend: str = ""
    """Legend label (Card 10) for LEGEND_BOX or TAG_LABELS mode."""
    xtag: float = 0.0
    """Tag x-position (Card 10a) — used only with TAG_LABELS."""
    ytag: float = 0.0
    """Tag y-position (Card 10a) — used only with TAG_LABELS."""
    view3d: View3D | None = None
    """3D viewpoint — set only for 3D plots (negative PlotType)."""
    nform: int = 0
    """Data format: 0 = free-format 2D (Card 13), 1 = 3D families (Card 14)."""
    data_2d: list[DataPoint2D] = field(default_factory=list)
    """2D data points (used when nform=0)."""
    data_3d: list[tuple[float, list[tuple[float, float]]]] = field(
        default_factory=list
    )
    """3D family data (used when nform=1).

    Each element is ``(x_value, [(y1, z1), (y2, z2), ...])``.
    A family is terminated by an empty list.
    """


@dataclass
class Plot:
    """A single plot (one set of axes, possibly with multiple curves)."""

    title1: str = ""
    title2: str = ""
    plot_type: PlotType = PlotType.LIN_LIN
    alt_axis_type: int = 0
    """jtype: 0=none, 1=lin alt-y, 2=log alt-y.  Negative for 3D z-axis."""
    grid: GridStyle = GridStyle.TIC_MARKS_INSIDE
    legend_style: LegendStyle = LegendStyle.NONE

    x_range: AxisRange = field(default_factory=AxisRange)
    x_label: str = ""
    y_range: AxisRange = field(default_factory=AxisRange)
    y_label: str = ""
    z_range: AxisRange | None = None
    """Set for alt-y or 3D z-axis (Card 7/7a). None = omit."""
    z_label: str = ""

    # Window / subplot positioning
    new_page: bool = True
    """True → iplot=1 (new page); False → iplot=-1 (subplot on same page)."""
    window_color: PageColor = PageColor.WHITE
    factx: float = 1.0
    facty: float = 1.0
    xll: float = 0.0
    yll: float = 0.0
    ww: float = 0.0
    wh: float = 0.0

    curves: list[Curve] = field(default_factory=list)


@dataclass
class ViewrPlot:
    """Top-level VIEWR custom plot definition.

    Pass to ``InputDeck.viewr(plot_def=...)`` to emit inline plot
    commands.
    """

    page: PageSetup = field(default_factory=PageSetup)
    plots: list[Plot] = field(default_factory=list)
