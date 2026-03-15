"""
Plotting infrastructure for KIKA.

This module provides a flexible, object-oriented approach to creating plots
by separating data representation from visual styling and plot composition.
"""

from .plot_data import (
    PlotData,
    LegendreCoeffPlotData,
    LegendreUncertaintyPlotData,
    AngularDistributionPlotData,
    CrossSectionPlotData,
    DifferencePlotData,
    MultigroupCrossSectionPlotData,
    MultigroupXSPlotData,
    MultigroupUncertaintyPlotData,
    UncertaintyBand,
    HeatmapPlotData,
    CovarianceHeatmapData,
    LegendreHeatmapData,
    MF34HeatmapData,
)
from .plot_builder import PlotBuilder
from .heatmap_builder import HeatmapBuilder
from .comparison import (
    ComparisonBuilder,
    ComparisonResult,
    compute_difference,
    interpolate_to_grid,
)

__all__ = [
    'PlotData',
    'LegendreCoeffPlotData',
    'LegendreUncertaintyPlotData',
    'AngularDistributionPlotData',
    'CrossSectionPlotData',
    'DifferencePlotData',
    'MultigroupCrossSectionPlotData',
    'MultigroupXSPlotData',
    'MultigroupUncertaintyPlotData',
    'UncertaintyBand',
    'HeatmapPlotData',
    'CovarianceHeatmapData',
    'LegendreHeatmapData',
    'MF34HeatmapData',
    'PlotBuilder',
    'HeatmapBuilder',
    'ComparisonBuilder',
    'ComparisonResult',
    'compute_difference',
    'interpolate_to_grid',
]
