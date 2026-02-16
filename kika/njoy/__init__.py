"""kika.njoy — NJOY input-deck generation and execution."""

from .run_njoy import run_njoy
from .deck import InputDeck
from .reader import read_deck, read_deck_file, ParsedDeck, ParsedModule
from .isotopes import load_isotopes, load_module_definitions, get_mat_number
from .viewr_defs import (
    ViewrPlot,
    PageSetup,
    Plot,
    Curve,
    DataPoint2D,
    CurveStyle,
    AxisRange,
    View3D,
    Orientation,
    FontStyle,
    PlotType,
    GridStyle,
    LegendStyle,
    CurveColor,
    PageColor,
    SymbolType,
    DashType,
)
from .plotr_defs import (
    PlotrJob,
    PlotrPlot,
    PlotrCurve,
    EndfSource,
)
