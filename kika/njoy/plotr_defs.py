"""PLOTR data model — dataclasses for tape-driven plot definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .viewr_defs import (
    Orientation,
    FontStyle,
    PlotType,
    GridStyle,
    LegendStyle,
    CurveColor,
    PageColor,
    SymbolType,
    DashType,
    PageSetup,
    AxisRange,
    CurveStyle,
    DataPoint2D,
)


@dataclass
class EndfSource:
    """Card 8 — ENDF/PENDF/GENDF tape source for a curve.

    When *iverf* is 0, user-supplied data is used instead of tape data.
    """

    iverf: int = 1
    """ENDF version: 0=user data, 1=ENDF-6 format (default)."""
    nin: int = 0
    """Input tape unit number (negative = binary)."""
    matd: int = 0
    """MAT number."""
    mfd: int = 0
    """MF (file) number."""
    mtd: int = 0
    """MT (reaction) number."""
    temper: float = 0.0
    """Temperature for Doppler-broadened data (default 0.0 = first temp)."""
    nth: int = 0
    """Thermal index (0 = default)."""
    ntp: int = 0
    """Sub-particle index (0 = default)."""
    nkh: int = 0
    """Hollerith index (0 = default)."""


@dataclass
class PlotrCurve:
    """A single curve within a PLOTR plot."""

    source: EndfSource = field(default_factory=EndfSource)
    """Card 8 — tape source for this curve."""
    style: CurveStyle = field(default_factory=CurveStyle)
    """Card 9 — visual style (2D only)."""
    legend: str = ""
    """Legend label (Card 10)."""
    data_2d: list[DataPoint2D] = field(default_factory=list)
    """User-supplied 2D data points (used when source.iverf == 0)."""
    nform: int = 0
    """Data format for user data: 0 = x,y[,errors] (Card 13)."""


@dataclass
class PlotrPlot:
    """A single plot (axes + curves) within a PLOTR job."""

    title1: str = ""
    title2: str = ""
    plot_type: PlotType = PlotType.LIN_LIN
    alt_axis_type: int = 0
    """jtype: 0=none, 1=lin alt-y, 2=log alt-y."""
    grid: GridStyle = GridStyle.TIC_MARKS_INSIDE
    legend_style: LegendStyle = LegendStyle.NONE
    xtag: float = 0.0
    """Card 4 legend box x-position (used when legend_style == LEGEND_BOX)."""
    ytag: float = 0.0
    """Card 4 legend box y-position (used when legend_style == LEGEND_BOX)."""

    x_range: AxisRange = field(default_factory=AxisRange)
    x_label: str = ""
    y_range: AxisRange = field(default_factory=AxisRange)
    y_label: str = ""
    z_range: AxisRange | None = None
    z_label: str = ""

    # Window / subplot positioning
    new_page: bool = True
    """True -> iplot=1 (new page); False -> iplot=-1 (subplot)."""
    window_color: PageColor = PageColor.WHITE
    factx: float = 1.0
    facty: float = 1.0
    xll: float = 0.0
    yll: float = 0.0
    ww: float = 0.0
    wh: float = 0.0
    wr: float = 0.0
    """Window rotation angle in degrees (PLOTR-specific)."""

    curves: list[PlotrCurve] = field(default_factory=list)


@dataclass
class PlotrJob:
    """Top-level PLOTR plot definition.

    Pass to ``InputDeck.plotr(plot_def=...)`` to emit inline plot
    commands after the module card line.
    """

    nplt: int = 0
    """Output plot-command tape unit (filled by deck builder)."""
    nplt0: int = 0
    """Input plot-command tape unit (0 = none)."""
    page: PageSetup = field(default_factory=PageSetup)
    plots: list[PlotrPlot] = field(default_factory=list)
