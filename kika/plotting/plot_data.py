"""
Base classes for representing plottable data.

These classes encapsulate the data and basic styling information for plot elements,
separating data representation from the actual plotting logic.
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PlotData:
    """
    Base class for plottable data.
    
    This encapsulates the data to plot along with styling information,
    allowing flexible composition of plots without recreating plotting logic.
    
    Attributes
    ----------
    x : array-like
        X-axis data
    y : array-like
        Y-axis data
    label : str, optional
        Label for legend
    color : str or tuple, optional
        Line/marker color
    linestyle : str, optional
        Line style ('-', '--', ':', '-.', etc.)
    linewidth : float, optional
        Line width
    marker : str, optional
        Marker style ('o', 's', '^', etc.)
    markersize : float, optional
        Marker size
    alpha : float, optional
        Transparency (0-1)
    plot_type : str
        Type of plot: 'line', 'step', 'scatter', 'errorbar'
    drawstyle : str, optional
        Matplotlib drawstyle for line plots. If set to 'steps-pre', 'steps-post',
        or 'steps-mid', the plot will use step rendering even if plot_type is 'line'.
        This provides an alternative way to specify step plots that integrates with
        frontend plotting libraries (e.g., Plotly's line_shape parameter).
    metadata : dict
        Additional metadata about the data (e.g., isotope, MT, order)
    """
    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    color: Optional[Union[str, Tuple]] = None
    linestyle: Optional[str] = '-'
    linewidth: Optional[float] = None
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    plot_type: str = 'line'
    drawstyle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and convert data to numpy arrays."""
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have the same length. Got x: {len(self.x)}, y: {len(self.y)}")
    
    def apply_styling(self, **kwargs) -> 'PlotData':
        """
        Create a copy with updated styling.
        
        Parameters
        ----------
        **kwargs : dict
            Styling attributes to update (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        PlotData
            New PlotData object with updated styling
        """
        # Create a copy of the current object
        import copy
        new_data = copy.copy(self)
        
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(new_data, key):
                setattr(new_data, key, value)
            else:
                raise ValueError(f"Unknown styling attribute: {key}")
        
        return new_data
    
    def get_plot_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for matplotlib plotting functions.
        
        Returns
        -------
        dict
            Dictionary of kwargs suitable for ax.plot(), ax.step(), etc.
        """
        kwargs = {}
        
        if self.label is not None:
            kwargs['label'] = self.label
        if self.color is not None:
            kwargs['color'] = self.color
        if self.linestyle is not None:
            kwargs['linestyle'] = self.linestyle
        if self.linewidth is not None:
            kwargs['linewidth'] = self.linewidth
        if self.marker is not None:
            kwargs['marker'] = self.marker
        if self.markersize is not None:
            kwargs['markersize'] = self.markersize
        if self.alpha is not None:
            kwargs['alpha'] = self.alpha
            
        return kwargs


@dataclass
class LegendreCoeffPlotData(PlotData):
    """
    Plottable data for Legendre coefficients.
    
    Additional Attributes
    ---------------------
    order : int
        Legendre polynomial order
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    energy_range : tuple, optional
        (min, max) energy range for this data
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    energy_range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['energy_range'] = self.energy_range
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class LegendreUncertaintyPlotData(PlotData):
    """
    Plottable data for Legendre coefficient uncertainties.
    
    Additional Attributes
    ---------------------
    order : int
        Legendre polynomial order
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    uncertainty_type : str
        'relative' or 'absolute'
    sigma : float
        Number of sigma levels (1.0 = 1σ, 2.0 = 2σ)
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    sigma: float = 1.0
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class AngularDistributionPlotData(PlotData):
    """
    Plottable data for angular distributions.

    Additional Attributes
    ---------------------
    energy : float
        Energy at which this distribution is defined
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    distribution_type : str, optional
        Type of distribution: 'legendre', 'tabulated', 'mixed'
    """
    energy: float = 0.0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    distribution_type: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['energy'] = self.energy
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['distribution_type'] = self.distribution_type

        # Default label if not provided
        if self.label is None and self.energy is not None:
            self.label = f"E = {self.energy:.2e} MeV"


@dataclass
class CrossSectionPlotData(PlotData):
    """
    Plottable data for cross sections (energy-dependent).

    This class represents cross section data as a function of energy,
    typically from EXFOR experimental data or ENDF evaluations.

    Additional Attributes
    ---------------------
    isotope : str, optional
        Isotope identifier (e.g., 'Fe56')
    mt : int, optional
        MT reaction number (e.g., 1 for total, 2 for elastic)
    energy_range : Tuple[float, float], optional
        (min, max) energy range for this data in MeV
    data_source : str, optional
        Source of data: 'exfor', 'endf', 'ace', etc.
    """
    isotope: Optional[str] = None
    mt: Optional[int] = None
    energy_range: Optional[Tuple[float, float]] = None
    data_source: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['energy_range'] = self.energy_range
        self.metadata['data_source'] = self.data_source

        # Default label if not provided
        if self.label is None and self.isotope and self.mt is not None:
            self.label = f"{self.isotope} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


class UncertaintyBand:
    """
    Represents an uncertainty band for a plot.
    
    This can store either absolute bounds or relative uncertainties.
    When relative uncertainties are provided, they will be converted to
    absolute bounds by PlotBuilder using the nominal y values.
    
    Attributes
    ----------
    x : array-like
        X-axis data (same as the main plot data)
    y_lower : array-like, optional
        Lower bound of uncertainty (absolute values)
    y_upper : array-like, optional
        Upper bound of uncertainty (absolute values)
    relative_uncertainty : array-like, optional
        Relative uncertainties (fractional, e.g., 0.05 for 5%)
        If provided, y_lower and y_upper should be None
    sigma : float
        Number of sigma levels (default: 1.0)
    color : str or tuple, optional
        Fill color
    alpha : float
        Transparency (default: 0.2)
    label : str, optional
        Label for legend
    style : str
        Rendering style: ``'band'`` for ``fill_between`` (default) or
        ``'errorbar'`` for vertical error bars via ``ax.errorbar()``.
    capsize : float
        Cap size for error bars (default: 2.5). Only used when
        ``style='errorbar'``.
    elinewidth : float
        Line width for error bars (default: 1.5). Only used when
        ``style='errorbar'``.
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        relative_uncertainty: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        color: Optional[Union[str, Tuple]] = None,
        alpha: float = 0.2,
        label: Optional[str] = None,
        style: str = 'band',
        capsize: float = 2.5,
        elinewidth: float = 1.5,
    ):
        self.x = np.asarray(x)
        self.sigma = sigma
        self.color = color
        self.alpha = alpha
        self.label = label

        if style not in ('band', 'errorbar'):
            raise ValueError(f"style must be 'band' or 'errorbar', got '{style}'")
        self.style = style
        self.capsize = capsize
        self.elinewidth = elinewidth

        # Store either absolute or relative uncertainties
        if relative_uncertainty is not None:
            if y_lower is not None or y_upper is not None:
                raise ValueError("Cannot specify both relative_uncertainty and y_lower/y_upper")
            self.relative_uncertainty = np.asarray(relative_uncertainty)
            self.y_lower = None
            self.y_upper = None
            # Validate
            if len(self.x) != len(self.relative_uncertainty):
                raise ValueError("x and relative_uncertainty must have the same length")
        elif y_lower is not None and y_upper is not None:
            self.y_lower = np.asarray(y_lower)
            self.y_upper = np.asarray(y_upper)
            self.relative_uncertainty = None
            # Validate
            if not (len(self.x) == len(self.y_lower) == len(self.y_upper)):
                raise ValueError("x, y_lower, and y_upper must have the same length")
        else:
            raise ValueError("Must provide either (y_lower, y_upper) or relative_uncertainty")
    
    def is_relative(self) -> bool:
        """Check if this band stores relative uncertainties."""
        return self.relative_uncertainty is not None
    
    def to_absolute(self, y_nominal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert relative uncertainties to absolute bounds.
        
        Parameters
        ----------
        y_nominal : array-like
            Nominal y values to apply uncertainties to
            
        Returns
        -------
        tuple of (y_lower, y_upper)
            Absolute uncertainty bounds
        """
        if not self.is_relative():
            return self.y_lower, self.y_upper
        
        y_nominal = np.asarray(y_nominal)
        if len(y_nominal) != len(self.relative_uncertainty):
            raise ValueError("y_nominal must have same length as relative_uncertainty")
        
        y_lower = y_nominal * (1.0 - self.sigma * self.relative_uncertainty)
        y_upper = y_nominal * (1.0 + self.sigma * self.relative_uncertainty)
        
        return y_lower, y_upper
    
    def get_fill_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for ax.fill_between()."""
        kwargs = {'alpha': self.alpha}

        if self.color is not None:
            kwargs['color'] = self.color
        if self.label is not None:
            kwargs['label'] = self.label

        return kwargs

    def get_errorbar_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for ax.errorbar() when style='errorbar'."""
        kwargs = {
            'fmt': ' ',
            'capsize': self.capsize,
            'elinewidth': self.elinewidth,
        }
        if self.color is not None:
            kwargs['ecolor'] = self.color
        if self.label is not None:
            kwargs['label'] = self.label
        return kwargs


@dataclass
class MultigroupXSPlotData(PlotData):
    """
    Plottable data for multigroup cross sections.
    
    Additional Attributes
    ---------------------
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for cross sections
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class MultigroupUncertaintyPlotData(PlotData):
    """
    Plottable data for multigroup cross section uncertainties.
    
    Additional Attributes
    ---------------------
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    uncertainty_type : str
        'relative' or 'absolute'
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class HeatmapPlotData:
    """
    Base class for 2D heatmap data.
    
    This encapsulates data for heatmap/image plots including matrix data,
    colormap configuration, and colorbar settings.
    
    Attributes
    ----------
    matrix_data : np.ndarray
        2D matrix to display
    extent : tuple of float, optional
        (xmin, xmax, ymin, ymax) for imshow extent parameter
    x_edges : np.ndarray, optional
        Bin edges for x-axis (for pcolormesh, preferred over imshow)
    y_edges : np.ndarray, optional
        Bin edges for y-axis (for pcolormesh)
    cmap : str or colormap
        Colormap name or instance
    norm : matplotlib.colors.Normalize, optional
        Custom normalization (e.g., TwoSlopeNorm for diverging colormaps).
        If not provided, auto-scaling is used.
    colorbar_label : str, optional
        Label for the colorbar
    colorbar_position : str
        Position for colorbar: 'right', 'bottom', 'top', 'left'
    mask_value : float, optional
        Value to mask (e.g., 0.0 for correlation matrices)
    mask_color : str
        Color for masked regions
    label : str, optional
        Label for plot identification
    metadata : dict
        Additional metadata
    """
    matrix_data: np.ndarray
    extent: Optional[Tuple[float, float, float, float]] = None
    x_edges: Optional[np.ndarray] = None
    y_edges: Optional[np.ndarray] = None
    cmap: Union[str, Any] = "viridis"
    norm: Optional[Any] = None
    colorbar_label: Optional[str] = None
    colorbar_position: str = "right"
    mask_value: Optional[float] = None
    mask_color: str = "#F0F0F0"
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate heatmap data."""
        self.matrix_data = np.asarray(self.matrix_data)
        if self.matrix_data.ndim != 2:
            raise ValueError(f"matrix_data must be 2D array, got shape {self.matrix_data.shape}")
        
        if self.x_edges is not None:
            self.x_edges = np.asarray(self.x_edges)
        if self.y_edges is not None:
            self.y_edges = np.asarray(self.y_edges)
    
    def get_masked_data(self) -> np.ndarray:
        """
        Get masked array with mask_value and NaN values masked.

        Returns
        -------
        np.ma.MaskedArray
            Masked version of data where NaN values and mask_value (if set) are masked
        """
        # Always mask NaN values so they render as grey via set_bad()
        mask = ~np.isfinite(self.matrix_data)
        if self.mask_value is not None:
            mask = mask | (self.matrix_data == self.mask_value)
        return np.ma.masked_where(mask, self.matrix_data)


@dataclass
class CovarianceHeatmapData(HeatmapPlotData):
    """
    Covariance/correlation matrix heatmap data.

    Additional Attributes
    ---------------------
    matrix_type : str
        'cov' or 'corr'
    zaid : int, List[int], or None
        Isotope identifier(s). Single ZAID for single-isotope heatmaps,
        list of ZAIDs for multi-isotope heatmaps.
    block_info : dict, optional
        Information about matrix block structure:
        - For single-isotope:
            - 'mts': list of MT numbers
            - 'G': number of energy groups per MT
            - 'ranges': list of (start, end) indices for each MT block
            - 'energy_ranges': dict mapping MT -> (start, end) coordinates
        - For multi-isotope (when 'is_multi_isotope' is True):
            - 'blocks': list of (zaid, mt) tuples
            - 'zaids': list of ZAIDs
            - 'mts': list of MT numbers
            - 'G': number of energy groups per block
            - 'ranges': dict mapping (zaid, mt) -> (start, end) indices
            - 'energy_ranges': dict mapping (zaid, mt) -> (start, end) coordinates
            - 'is_multi_isotope': True
    uncertainty_data : dict, optional
        Uncertainty values for optional panels above heatmap.
        - For single-isotope: {mt: sigma_percent_array}
        - For multi-isotope: {(zaid, mt): sigma_percent_array}
    energy_grid : np.ndarray, optional
        Energy bin boundaries (length G+1) for automatic energy tick display
    mt_labels : list of str, optional
        Custom labels for blocks. For multi-isotope, typically "Symbol-MT#" format.
    is_diagonal : bool
        Whether this represents diagonal blocks (vs off-diagonal)
    """
    matrix_type: str = "corr"
    zaid: Optional[Union[int, List[int]]] = None
    block_info: Optional[Dict[str, Any]] = None
    uncertainty_data: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None
    energy_grid: Optional[np.ndarray] = None
    mt_labels: Optional[List[str]] = None
    is_diagonal: bool = True
    scale: str = "log"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Auto-configure for correlation matrices with diverging colormap
        if self.matrix_type == "corr" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm
            # Use masked data to exclude mask_value from absmax calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                # compressed() returns only non-masked values
                valid_data = masked_data.compressed()
                absmax = np.nanmax(np.abs(valid_data)) if len(valid_data) > 0 else 1.0
            else:
                absmax = np.nanmax(np.abs(masked_data))
            if absmax < 1e-10:
                absmax = 1.0  # Avoid division by zero
            self.norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

        # Auto-configure for covariance matrices with proper normalization
        # This ensures masked/NaN values are properly displayed as grey
        if self.matrix_type == "cov" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm, Normalize
            # Use masked data to exclude NaN values from range calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                valid_data = masked_data.compressed()
                if len(valid_data) > 0:
                    vmin = np.nanmin(valid_data)
                    vmax = np.nanmax(valid_data)
                else:
                    vmin, vmax = -1.0, 1.0
            else:
                vmin = np.nanmin(masked_data)
                vmax = np.nanmax(masked_data)

            # Use TwoSlopeNorm if data spans zero, otherwise use regular Normalize
            if vmin < 0 < vmax:
                self.norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                self.norm = Normalize(vmin=vmin, vmax=vmax)

        if self.matrix_type == "corr" and isinstance(self.cmap, str) and self.cmap == "viridis":
            self.cmap = "RdYlGn"  # Diverging colormap for correlations
        
        # Auto-generate colorbar label if not provided
        if self.colorbar_label is None:
            self.colorbar_label = "Correlation" if self.matrix_type == "corr" else "Covariance"
        
        # Store metadata
        self.metadata['matrix_type'] = self.matrix_type
        self.metadata['zaid'] = self.zaid
        self.metadata['is_diagonal'] = self.is_diagonal
        self.metadata['scale'] = self.scale
        
        # Convert energy_grid if provided
        if self.energy_grid is not None:
            self.energy_grid = np.asarray(self.energy_grid)


@dataclass
class MF34HeatmapData(HeatmapPlotData):
    """
    Angular distribution (MF34) covariance heatmap data.
    
    These heatmaps can be more complex because different Legendre coefficients
    may have different energy group structures.
    
    Additional Attributes
    ---------------------
    isotope : int
        Isotope identifier (ZAID)
    mt : int
        MT reaction number
    legendre_coeffs : list of int
        Legendre polynomial orders included
    matrix_type : str
        'cov' or 'corr'
    scale : str
        Energy axis scale: 'log' or 'linear'
    block_info : dict, optional
        Per-Legendre block structure information:
        - 'legendre_coeffs': list of L values
        - 'G_per_L': dict {L: G_L} of energy groups per L
        - 'ranges': dict {L: (start, end)} of index ranges
    uncertainty_data : dict, optional
        Uncertainty values for optional panels
        Format: {L: sigma_percent_array[G_L]}
    energy_grids : dict, optional
        Energy bin boundaries per Legendre coefficient for automatic energy tick display
        Format: {L: energy_grid[G_L+1]}
    is_diagonal : bool
        Whether diagonal blocks (L vs L) or off-diagonal (L1 vs L2)
    """
    isotope: int = 0
    mt: int = 0
    legendre_coeffs: List[int] = field(default_factory=list)
    matrix_type: str = "corr"
    scale: str = "log"
    block_info: Optional[Dict[str, Any]] = None
    uncertainty_data: Optional[Dict[int, np.ndarray]] = None
    energy_grids: Optional[Dict[int, np.ndarray]] = None
    is_diagonal: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        
        # Auto-configure for correlation matrices
        if self.matrix_type == "corr" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm
            # Use masked data to exclude mask_value from absmax calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                # compressed() returns only non-masked values
                valid_data = masked_data.compressed()
                absmax = np.nanmax(np.abs(valid_data)) if len(valid_data) > 0 else 1.0
            else:
                absmax = np.nanmax(np.abs(masked_data))
            if absmax < 1e-10:
                absmax = 1.0
            self.norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

        # Auto-configure for covariance matrices with proper normalization
        # This ensures masked/NaN values are properly displayed as grey
        if self.matrix_type == "cov" and self.norm is None:
            from matplotlib.colors import TwoSlopeNorm, Normalize
            # Use masked data to exclude NaN values from range calculation
            masked_data = self.get_masked_data()
            if isinstance(masked_data, np.ma.MaskedArray):
                valid_data = masked_data.compressed()
                if len(valid_data) > 0:
                    vmin = np.nanmin(valid_data)
                    vmax = np.nanmax(valid_data)
                else:
                    vmin, vmax = -1.0, 1.0
            else:
                vmin = np.nanmin(masked_data)
                vmax = np.nanmax(masked_data)

            # Use TwoSlopeNorm if data spans zero, otherwise use regular Normalize
            if vmin < 0 < vmax:
                self.norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                self.norm = Normalize(vmin=vmin, vmax=vmax)

        if self.matrix_type == "corr" and isinstance(self.cmap, str) and self.cmap == "viridis":
            self.cmap = "RdYlGn"

        # Auto-generate colorbar label
        if self.colorbar_label is None:
            self.colorbar_label = "Correlation" if self.matrix_type == "corr" else "Covariance"

        # Store metadata
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['legendre_coeffs'] = self.legendre_coeffs
        self.metadata['matrix_type'] = self.matrix_type
        self.metadata['scale'] = self.scale
        self.metadata['is_diagonal'] = self.is_diagonal
        
        # Convert energy_grids if provided
        if self.energy_grids is not None:
            self.energy_grids = {L: np.asarray(grid) for L, grid in self.energy_grids.items()}
