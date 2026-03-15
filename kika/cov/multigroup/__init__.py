# Multigroup covariance analysis package

from .mg_legendre_covariance import MultigroupLegendreCovariance, MGMF34CovMat
from .collapse import collapse_to_multigroup, MF34_to_MG
from .plotting_mg import (
    plot_mg_legendre_coefficients,
    plot_mg_vs_endf_comparison,
    plot_mg_vs_endf_uncertainties_comparison,
    plot_mg_covariance_heatmap
)

__all__ = [
    'MultigroupLegendreCovariance',
    'MGMF34CovMat',
    'collapse_to_multigroup',
    'MF34_to_MG',
    'plot_mg_legendre_coefficients',
    'plot_mg_vs_endf_comparison',
    'plot_mg_vs_endf_uncertainties_comparison',
    'plot_mg_covariance_heatmap'
]