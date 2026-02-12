"""
PlotBuilder class for composing and rendering plots.

This module provides the PlotBuilder class that takes PlotData objects
and creates publication-quality plots with consistent styling.
"""

from typing import List, Optional, Tuple, Union, Dict, Any

# Sentinel value to distinguish "not set" from "explicitly set to None"
_NOT_SET = object()
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .plot_data import (
    PlotData,
    UncertaintyBand,
    LegendreUncertaintyPlotData,
    MultigroupUncertaintyPlotData,
    HeatmapPlotData,
    CovarianceHeatmapData,
    MF34HeatmapData
)
from .styles import (
    _get_color_palette,
    _get_linestyles,
    _apply_style_to_rcparams,
    _adjust_figsize_for_notebook,
    _adjust_dpi_for_notebook,
    format_energy_axis_ticks
)
from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _setup_notebook_backend,
    _configure_figure_interactivity
)


class PlotBuilder:
    """
    Builder class for creating plots from PlotData objects.
    
    This class handles the composition of multiple plot elements and applies
    consistent styling and formatting.
    
    Examples
    --------
    >>> # Create plot data objects
    >>> data1 = LegendreCoeffPlotData(x=energies1, y=coeffs1, order=1, isotope='U235')
    >>> data2 = LegendreCoeffPlotData(x=energies2, y=coeffs2, order=1, isotope='U238')
    >>> 
    >>> # Build the plot
    >>> builder = PlotBuilder(style='light', figsize=(10, 6))
    >>> builder.add_data(data1, color='blue')
    >>> builder.add_data(data2, color='red')
    >>> builder.set_labels(
    ...     title='Elastic Scattering Legendre Coefficients',
    ...     x_label='Energy (eV)',
    ...     y_label='Coefficient Value'
    ... )
    >>> fig = builder.build()
    >>> fig.show()
    """
    
    def __init__(
        self,
        style: str = 'light',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        ax: Optional[plt.Axes] = None,
        projection: Optional[str] = None,
        font_family: str = 'serif',
        notebook_mode: Optional[bool] = None,
        interactive: Optional[bool] = None,
    ):
        """
        Initialize the PlotBuilder.
        
        Parameters
        ----------
        style : str
            Plot style: 'light' (default, publication-quality) or 'dark'
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Dots per inch for figure resolution
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure and axes.
        projection : str, optional
            Projection type (e.g., '3d' for 3D plots)
        font_family : str
            Font family for text elements (default: 'serif')
        notebook_mode : bool, optional
            Force notebook mode (auto-detected if None)
        interactive : bool, optional
            Force interactive mode (auto-detected if None)
        """
        if style not in ('light', 'dark'):
            raise ValueError(f"Invalid style '{style}'. Must be 'light' or 'dark'.")
        
        self.style = style
        self.projection = projection
        self.font_family = font_family
        
        # Auto-detect notebook and interactive mode
        if notebook_mode is None:
            notebook_mode = _is_notebook()
        if interactive is None:
            interactive = notebook_mode and _detect_interactive_backend()
        
        self._notebook_mode = notebook_mode
        self._interactive = interactive
        
        # Adjust settings for notebook environment
        # Keep a record of the user-requested size/dpi before any notebook tweaks
        self._figsize_user = figsize
        self._dpi_user = dpi

        if notebook_mode:
            figsize = _adjust_figsize_for_notebook(figsize, interactive)
            dpi = _adjust_dpi_for_notebook(dpi, interactive)
            
            # Setup interactive backend if needed
            if interactive:
                backend_success, backend_msg = _setup_notebook_backend()
                self._backend_info = backend_msg
                self._interactive = backend_success
            else:
                self._backend_info = f"Notebook mode, backend: {plt.get_backend()}"
        else:
            self._backend_info = f"Non-notebook environment, backend: {plt.get_backend()}"
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Get color palette and linestyles for this style
        self._colors = _get_color_palette(style)
        self._linestyles = _get_linestyles()
        
        # Storage for plot elements
        self._data_list: List[PlotData] = []
        self._uncertainty_bands: List[Tuple[UncertaintyBand, int]] = []  # (band, data_index)
        self._custom_styling: List[Dict[str, Any]] = []  # Per-data styling overrides
        
        # Storage for heatmap data (alternative to line plots)
        self._heatmap_data: Optional['HeatmapPlotData'] = None
        self._heatmap_show_uncertainties: bool = True
        self._heatmap_styling_overrides: Dict[str, Any] = {}
        
        # Plot configuration
        self._x_label: Optional[str] = None
        self._y_label: Optional[str] = None
        self._title = _NOT_SET  # Use sentinel to distinguish "not set" from "explicitly None"
        self._legend_loc: str = 'best'
        self._legend_outside: bool = False
        self._use_log_x: bool = False
        self._use_log_y: bool = False
        self._x_lim: Optional[Tuple[float, float]] = None
        self._y_lim: Optional[Tuple[float, float]] = None
        self._grid: bool = True
        self._grid_alpha: float = 0.3  # Alpha (transparency) for major grid
        self._show_minor_grid: bool = False  # Whether to show minor grid
        self._minor_grid_alpha: float = 0.15  # Alpha for minor grid
        
        # Font size configuration (will override style defaults if set)
        self._title_fontsize: Optional[float] = None
        self._label_fontsize: Optional[float] = None
        self._tick_labelsize: Optional[float] = None
        self._legend_fontsize: Optional[float] = None
        
        # Setup figure and axes
        if ax is not None:
            # Use provided axes
            self.fig = ax.figure
            self.ax = ax
        else:
            # Apply style to matplotlib rcParams
            _apply_style_to_rcparams(
                style=style,
                notebook_mode=notebook_mode,
                figsize=figsize,
                dpi=dpi,
                font_family=font_family,
                projection=projection
            )
            
            # Create figure and axes
            if projection is not None:
                self.fig = plt.figure(figsize=figsize, dpi=dpi)
                self.ax = self.fig.add_subplot(111, projection=projection)
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Configure figure interactivity
            _configure_figure_interactivity(self.fig, interactive)
    
    def add_data(
        self,
        data: Union[PlotData, Tuple[PlotData, Optional[Union[UncertaintyBand, PlotData]]]],
        uncertainty: Optional[Union[UncertaintyBand, PlotData]] = None,
        **styling_overrides
    ) -> 'PlotBuilder':
        """
        Add a PlotData object to the plot.
        
        Parameters
        ----------
        data : PlotData or tuple
            The data to plot. Can be either:
            - A PlotData object
            - A tuple (PlotData, UncertaintyBand) as returned by to_plot_data with uncertainty=True
            If a tuple is provided and uncertainty is None, the UncertaintyBand from the tuple will be used.
        uncertainty : UncertaintyBand or PlotData, optional
            Uncertainty to plot with this data. Can be either:
            - UncertaintyBand object (will be plotted as shaded region)
            - PlotData object with uncertainty values (will be converted to band)
            If data is a tuple and this parameter is provided, this parameter takes precedence.
        **styling_overrides
            Styling overrides for this specific data (color, linestyle, etc.)
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        # Check for conflicts with heatmap data
        if self._heatmap_data is not None:
            raise ValueError(
                "Cannot mix line plots and heatmaps. "
                "add_data() cannot be called after add_heatmap(). "
                "Create separate PlotBuilder instances for heatmaps and line plots."
            )
        
        # Handle tuple input from to_plot_data(uncertainty=True)
        if isinstance(data, tuple):
            if len(data) == 2:
                plot_data, tuple_uncertainty = data
                # Use the uncertainty from tuple if no explicit uncertainty is provided
                if uncertainty is None:
                    uncertainty = tuple_uncertainty
                data = plot_data
            else:
                raise ValueError(f"Expected tuple of length 2, got {len(data)}")
        
        self._data_list.append(data)
        self._custom_styling.append(styling_overrides)
        
        if uncertainty is not None:
            # Convert PlotData to UncertaintyBand if needed
            if isinstance(uncertainty, PlotData):
                uncertainty = self._convert_plotdata_to_band(uncertainty, data)
            
            data_index = len(self._data_list) - 1
            self._uncertainty_bands.append((uncertainty, data_index))
        
        return self
    
    def add_multiple(
        self,
        data_list: List[PlotData],
        colors: Optional[List[Union[str, Tuple]]] = None,
        linestyles: Optional[List[str]] = None,
        **common_styling
    ) -> 'PlotBuilder':
        """
        Add multiple PlotData objects at once.
        
        Parameters
        ----------
        data_list : list of PlotData
            List of data objects to plot
        colors : list of str/tuple, optional
            Colors for each data object. If None, uses automatic color cycling.
        linestyles : list of str, optional
            Line styles for each data object. If None, uses solid lines.
        **common_styling
            Styling applied to all data objects
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        for i, data in enumerate(data_list):
            styling = common_styling.copy()
            
            if colors is not None and i < len(colors):
                styling['color'] = colors[i]
            if linestyles is not None and i < len(linestyles):
                styling['linestyle'] = linestyles[i]
            
            self.add_data(data, **styling)
        
        return self
    
    def set_labels(
        self,
        title = _NOT_SET,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> 'PlotBuilder':
        """
        Set plot labels.

        Parameters
        ----------
        title : str, optional
            Plot title. Use a string to set a custom title, or pass None or ""
            to explicitly hide the title. If not provided, the default title
            (e.g., from heatmap_data.label) will be used.
        x_label : str, optional
            X-axis label
        y_label : str, optional
            Y-axis label

        Returns
        -------
        PlotBuilder
            Self for method chaining

        Examples
        --------
        >>> builder.set_labels(title="My Custom Title")  # Custom title
        >>> builder.set_labels(title="")  # Hide title
        >>> builder.set_labels(title=None)  # Hide title
        """
        if title is not _NOT_SET:
            self._title = title
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label

        return self
    
    def set_scales(
        self,
        log_x: bool = False,
        log_y: bool = False
    ) -> 'PlotBuilder':
        """
        Set axis scales.
        
        Parameters
        ----------
        log_x : bool
            Use logarithmic scale for x-axis
        log_y : bool
            Use logarithmic scale for y-axis
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._use_log_x = log_x
        self._use_log_y = log_y
        return self
    
    def set_limits(
        self,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None
    ) -> 'PlotBuilder':
        """
        Set axis limits.
        
        For covariance/correlation heatmaps (symmetric matrices), if only one limit
        is provided, it will be applied to both axes to maintain symmetry.
        
        Parameters
        ----------
        x_lim : tuple, optional
            (min, max) for x-axis. For symmetric matrices, also applied to y-axis if y_lim is None.
        y_lim : tuple, optional
            (min, max) for y-axis. For symmetric matrices, also applied to x-axis if x_lim is None.
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._x_lim = x_lim
        self._y_lim = y_lim
        return self
    
    def set_legend(self, loc: str = 'best', outside: bool = False) -> 'PlotBuilder':
        """
        Set legend location.

        Parameters
        ----------
        loc : str
            Legend location
        outside : bool
            If True, place legend outside the plot area to the right

        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._legend_loc = loc
        self._legend_outside = outside
        return self
    
    def set_grid(
        self, 
        grid: bool = True,
        alpha: float = 0.3,
        show_minor: bool = False,
        minor_alpha: float = 0.15
    ) -> 'PlotBuilder':
        """
        Configure grid display settings.
        
        Parameters
        ----------
        grid : bool
            Whether to show major grid
        alpha : float
            Alpha (transparency) for major grid lines. Range: 0.0-1.0
        show_minor : bool
            Whether to show minor grid lines
        minor_alpha : float
            Alpha (transparency) for minor grid lines. Range: 0.0-1.0
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._grid = grid
        self._grid_alpha = alpha
        self._show_minor_grid = show_minor
        self._minor_grid_alpha = minor_alpha
        return self
    
    def set_tick_params(
        self,
        max_ticks_x: Optional[int] = None,
        max_ticks_y: Optional[int] = None,
        rotate_x: Optional[float] = None,
        rotate_y: Optional[float] = None,
        auto_rotate: bool = False
    ) -> 'PlotBuilder':
        """
        Configure tick parameters to avoid overlapping labels.
        
        This method helps prevent tick label overlap, especially useful for
        small figures or dense data.
        
        Parameters
        ----------
        max_ticks_x : int, optional
            Maximum number of ticks on x-axis. If None, matplotlib default.
        max_ticks_y : int, optional
            Maximum number of ticks on y-axis. If None, matplotlib default.
        rotate_x : float, optional
            Rotation angle for x-axis tick labels (degrees). 
            Common values: 45, 90 for rotated text.
        rotate_y : float, optional
            Rotation angle for y-axis tick labels (degrees).
        auto_rotate : bool, optional
            If True, automatically rotate x-axis labels by 45° for better spacing.
            Default is False.
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> # Limit to 8 ticks on x-axis and rotate labels
        >>> builder.set_tick_params(max_ticks_x=8, rotate_x=45)
        >>> 
        >>> # Auto-rotate x-axis labels for better spacing
        >>> builder.set_tick_params(auto_rotate=True)
        >>> 
        >>> # Control both axes
        >>> builder.set_tick_params(max_ticks_x=6, max_ticks_y=8)
        """
        # Store tick parameters
        if not hasattr(self, '_tick_params'):
            self._tick_params = {}
        
        if max_ticks_x is not None:
            self._tick_params['max_ticks_x'] = max_ticks_x
        if max_ticks_y is not None:
            self._tick_params['max_ticks_y'] = max_ticks_y
        if rotate_x is not None:
            self._tick_params['rotate_x'] = rotate_x
        if rotate_y is not None:
            self._tick_params['rotate_y'] = rotate_y
        if auto_rotate:
            self._tick_params['rotate_x'] = 45
        
        return self
    
    def set_font_sizes(
        self,
        title: Optional[float] = None,
        labels: Optional[float] = None,
        ticks: Optional[float] = None,
        legend: Optional[float] = None
    ) -> 'PlotBuilder':
        """
        Set font sizes for plot elements.
        
        This method allows fine-grained control over font sizes, overriding
        the defaults set by the style parameter.
        
        Parameters
        ----------
        title : float, optional
            Font size for the plot title
        labels : float, optional
            Font size for axis labels (x and y)
        ticks : float, optional
            Font size for tick labels
        legend : float, optional
            Font size for legend text
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> builder = PlotBuilder(style='paper')
        >>> builder.add_data(data)
        >>> builder.set_labels(title='My Plot', x_label='Energy', y_label='Value')
        >>> builder.set_font_sizes(title=18, labels=14, ticks=12, legend=11)
        >>> fig = builder.build()
        """
        if title is not None:
            self._title_fontsize = title
        if labels is not None:
            self._label_fontsize = labels
        if ticks is not None:
            self._tick_labelsize = ticks
        if legend is not None:
            self._legend_fontsize = legend
        
        return self
    
    def add_heatmap(
        self,
        heatmap_data: 'HeatmapPlotData',
        show_uncertainties: bool = True,
        **styling_overrides
    ) -> 'PlotBuilder':
        """
        Add heatmap data to be rendered when build() is called.

        .. deprecated::
            This method is deprecated. Use HeatmapBuilder instead for more
            control over heatmap formatting options.

        Parameters
        ----------
        heatmap_data : HeatmapPlotData
            Heatmap data object (CovarianceHeatmapData or MF34HeatmapData)
        show_uncertainties : bool, default True
            Whether to show uncertainty panels above the heatmap
        **styling_overrides
            Styling overrides for heatmap rendering. Common options:
            - cmap : str or Colormap - Override colormap (e.g., 'viridis', 'RdBu_r')
            - norm : Normalize - Custom normalization
            - colorbar_label : str - Override colorbar label

        Returns
        -------
        PlotBuilder
            Self for method chaining (returns HeatmapBuilder internally)

        Notes
        -----
        This method is deprecated. Please use HeatmapBuilder directly for
        more control over formatting:

        >>> from kika.plotting import HeatmapBuilder
        >>> fig = HeatmapBuilder().add_heatmap(
        ...     heatmap_data,
        ...     show_energy_ticks=True,
        ...     show_block_labels=True,
        ...     energy_tick_fontsize=10
        ... ).build()
        """
        import warnings
        warnings.warn(
            "PlotBuilder.add_heatmap() is deprecated. Use HeatmapBuilder instead "
            "for more control over heatmap formatting options.",
            DeprecationWarning,
            stacklevel=2
        )

        # Check for conflicts with line plot data
        if self._data_list:
            raise ValueError(
                "Cannot mix line plots and heatmaps. "
                "add_heatmap() cannot be called after add_data(). "
                "Create separate PlotBuilder instances for heatmaps and line plots."
            )

        # Store heatmap data for rendering in build()
        self._heatmap_data = heatmap_data
        self._heatmap_show_uncertainties = show_uncertainties
        self._heatmap_styling_overrides = styling_overrides

        return self
    
    def _build_heatmap(self) -> plt.Figure:
        """
        Internal method to render heatmap. Called by build() when heatmap data is present.

        This method delegates to HeatmapBuilder for the actual rendering.

        Returns
        -------
        matplotlib.figure.Figure
            The completed heatmap figure
        """
        from .heatmap_builder import HeatmapBuilder

        # Create HeatmapBuilder with same settings
        figsize = getattr(self, "_figsize_user", None) or self.figsize
        dpi = getattr(self, "_dpi_user", None) or self.dpi

        builder = HeatmapBuilder(
            style=self.style,
            figsize=figsize,
            dpi=dpi,
            font_family=self.font_family,
            notebook_mode=self._notebook_mode,
            interactive=self._interactive,
        )

        # Transfer title if explicitly set (including None or empty string to hide)
        if self._title is not _NOT_SET:
            builder.set_labels(title=self._title)

        # Transfer limits if set
        if self._x_lim is not None or self._y_lim is not None:
            builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)

        # Transfer font sizes if set
        if self._title_fontsize is not None or self._tick_labelsize is not None:
            builder.set_font_sizes(
                title=self._title_fontsize,
                ticks=self._tick_labelsize
            )

        # Add heatmap data and build
        builder.add_heatmap(
            self._heatmap_data,
            show_uncertainties=self._heatmap_show_uncertainties,
            **self._heatmap_styling_overrides
        )

        return builder.build()

    # NOTE: The following heatmap helper methods are deprecated.
    # Heatmap functionality has been moved to HeatmapBuilder class in heatmap_builder.py.
    # These methods are kept for backward compatibility but will not be called
    # since _build_heatmap() now delegates to HeatmapBuilder.

    def _setup_mf34_heatmap_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        data: 'MF34HeatmapData',
        full_xlim: Tuple[float, float],
        full_ylim: Tuple[float, float]
    ) -> None:
        """
        Setup ticks for MF34 angular distribution heatmap with Legendre labels and energy axes.
        
        Primary axes show Legendre order labels using figure-fixed positioning.
        Secondary axes (top/right) show energy tick marks.
        """
        import numpy as np

        block_info = data.block_info or {}
        legendre_list = block_info.get('legendre_coeffs', data.legendre_coeffs)
        ranges_idx = block_info.get('ranges', {}) or {}
        energy_ranges = block_info.get('energy_ranges', {}) or {}
        legendre_labels = [str(l) for l in legendre_list]  # Format as 1, 2, etc.

        # Legendre labels using figure-fixed positioning (like MT labels)
        if data.is_diagonal:
            centers = []
            labels = []
            for i, l_val in enumerate(legendre_list):
                rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                if rng:
                    centers.append((rng[0] + rng[1]) * 0.5)
                    labels.append(legendre_labels[i])
            if centers:
                # Use figure-fixed labels instead of axis ticks
                ax.set_xticks([])
                ax.set_yticks([])
                self._add_block_labels_figure_fixed(
                    fig, ax, centers, labels, full_xlim, full_ylim,
                    x_axis_label="Legendre Order", y_axis_label="Legendre Order"
                )
        elif len(legendre_list) >= 2:
            # For off-diagonal blocks, still use figure-fixed approach
            row_l = legendre_list[0]
            col_l = legendre_list[1] if len(legendre_list) > 1 else legendre_list[0]
            x_rng = energy_ranges.get(col_l) or ranges_idx.get(col_l)
            y_rng = energy_ranges.get(row_l) or ranges_idx.get(row_l)
            
            ax.set_xticks([])
            ax.set_yticks([])

            if x_rng:
                x_centers = [(x_rng[0] + x_rng[1]) * 0.5]
                x_labels = [str(col_l)]
                self._add_block_labels_figure_fixed(
                    fig, ax, x_centers, x_labels, full_xlim, full_ylim,
                    x_axis_label="Legendre Order", y_axis_label="Legendre Order"
                )
            if y_rng:
                y_centers = [(y_rng[0] + y_rng[1]) * 0.5]
                y_labels = [str(row_l)]
                self._add_block_labels_figure_fixed(
                    fig, ax, y_centers, y_labels, full_xlim, full_ylim,
                    x_axis_label="Legendre Order", y_axis_label="Legendre Order"
                )

        # Add energy ticks on secondary axes if energy grids available
        if data.energy_grids is not None and len(data.energy_grids) > 0:
            # Use multi-block energy ticks if there are multiple Legendre orders with coordinate ranges
            if len(legendre_list) > 1 and energy_ranges:
                # Build block_ranges dict from energy_ranges
                block_ranges_dict = {}
                for l_val in legendre_list:
                    rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                    if rng is not None:
                        block_ranges_dict[l_val] = tuple(rng)
                if block_ranges_dict:
                    self._add_multi_block_energy_ticks(ax, data.energy_grids, block_ranges_dict, data.scale)
                else:
                    first_grid = next(iter(data.energy_grids.values()))
                    self._add_energy_ticks(ax, first_grid, data.scale)
            else:
                # Single block - use first available energy grid for tick placement
                first_grid = next(iter(data.energy_grids.values()))
                self._add_energy_ticks(ax, first_grid, data.scale)
    
    def _add_energy_ticks(self, ax: plt.Axes, energy_grid: np.ndarray, scale: str = 'log') -> None:
        """
        Add energy tick marks on secondary (top/right) axes in transformed coordinates.
        
        The heatmap is displayed in transformed coordinate space. For log scale, coordinates
        are log10(energy) - log10(e_min), shifted to start at 0. Energy ticks must account
        for this transformation.
        
        Parameters
        ----------
        ax : plt.Axes
            Primary heatmap axes (already in transformed coordinates)
        energy_grid : np.ndarray
            Energy bin boundaries (in original eV units)
        scale : str
            'log' or 'linear' scaling
        """
        import numpy as np
        from matplotlib.ticker import FuncFormatter, FixedLocator
        
        # Create secondary axes for energy ticks
        ax_top = ax.twiny()
        ax_right = ax.twinx()
        
        # Get the axes limits (these are in transformed space)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if scale == 'log':
            # Energy grid is in eV, heatmap coordinates are log10(eV) - log10(e_min)
            e_min, e_max = energy_grid.min(), energy_grid.max()
            log_e_min = np.log10(np.maximum(e_min, 1e-300))
            log_e_max = np.log10(e_max)
            
            # Heatmap coordinates are shifted: coord = log10(E) - log10(e_min)
            # So to place a tick at energy E, use coordinate: log10(E) - log10(e_min)
            
            # Find decade boundaries within data range
            decade_min = int(np.floor(log_e_min))
            decade_max = int(np.ceil(log_e_max))
            
            # Major ticks at every decade (1e+00, 1e+01, 1e+02, etc.)
            major_tick_energies = [10**d for d in range(decade_min, decade_max + 1)]
            major_tick_positions = [np.log10(e) - log_e_min for e in major_tick_energies 
                                   if e_min <= e <= e_max]
            
            # Minor ticks at 2e+XX, 3e+XX, ..., 9e+XX within each decade
            minor_tick_positions = []
            for decade in range(decade_min, decade_max + 1):
                for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
                    e = multiplier * 10**decade
                    if e_min <= e <= e_max:
                        pos = np.log10(e) - log_e_min
                        minor_tick_positions.append(pos)
            
            # Format function for scientific notation without units
            # The tick position is in shifted log space, so we need to un-shift it
            def format_energy_log(shifted_log_val, pos=None):
                """Format shifted log10(energy) as scientific notation."""
                # Un-shift: log10(E) = shifted_log_val + log10(e_min)
                log_val = shifted_log_val + log_e_min
                energy = 10**log_val
                
                # Use scientific notation format
                exponent = int(np.round(log_val))
                mantissa = energy / (10**exponent)
                
                # Clean format: "1e+01", "2e+00", etc.
                if abs(mantissa - 1.0) < 0.1:  # It's a clean decade
                    return f'1e{exponent:+03d}'
                else:
                    # For non-decade ticks (2e+01, 3e+01, etc.)
                    return f'{int(np.round(mantissa))}e{exponent:+03d}'
            
            formatter = FuncFormatter(format_energy_log)
            
            # Set major ticks on top axis
            ax_top.set_xlim(xlim)
            ax_top.xaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax_top.xaxis.set_major_formatter(formatter)
            ax_top.xaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax_top.tick_params(axis='x', which='major', labelsize=8, rotation=30, length=6, pad=2)
            # Adjust label alignment to shift right
            for label in ax_top.get_xticklabels():
                label.set_ha('left')
            ax_top.tick_params(axis='x', which='minor', length=3)
            
            # Set major ticks on right axis
            ax_right.set_ylim(ylim)
            ax_right.yaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax_right.yaxis.set_major_formatter(formatter)
            ax_right.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax_right.tick_params(axis='y', which='major', labelsize=8, length=6)
            ax_right.tick_params(axis='y', which='minor', length=3)
            
        else:
            # Linear scale: coordinates are not transformed (just the raw energy values)
            from matplotlib.ticker import MaxNLocator
            
            # Use linear spacing
            locator = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
            tick_positions = locator.tick_values(energy_grid.min(), energy_grid.max())
            tick_positions = tick_positions[(tick_positions >= energy_grid.min()) & 
                                           (tick_positions <= energy_grid.max())]
            
            # Format function for scientific notation without units
            def format_energy_linear(val, pos=None):
                if val == 0:
                    return '0'
                exponent = int(np.floor(np.log10(abs(val))))
                mantissa = val / (10**exponent)
                if abs(mantissa - 1.0) < 0.1:
                    return f'1e{exponent:+03d}'
                else:
                    return f'{int(np.round(mantissa))}e{exponent:+03d}'
            
            formatter = FuncFormatter(format_energy_linear)
            
            ax_top.set_xlim(xlim)
            ax_top.set_xticks(tick_positions)
            ax_top.xaxis.set_major_formatter(formatter)
            ax_top.tick_params(axis='x', labelsize=10, rotation=30, pad=2)
            # Adjust label alignment to shift right
            for label in ax_top.get_xticklabels():
                label.set_ha('left')
            
            ax_right.set_ylim(ylim)
            ax_right.set_yticks(tick_positions)
            ax_right.yaxis.set_major_formatter(formatter)
            ax_right.tick_params(axis='y', labelsize=10)

    def _add_multi_block_energy_ticks(
        self,
        ax: plt.Axes,
        energy_grids: Dict[Any, np.ndarray],
        block_ranges: Dict[Any, Tuple[float, float]],
        scale: str = 'log'
    ) -> None:
        """
        Add energy tick marks for multiple blocks on secondary (top/right) axes.

        Each block gets its own energy ticks within its coordinate range.

        Parameters
        ----------
        ax : plt.Axes
            Primary heatmap axes
        energy_grids : dict
            {key: energy_grid} where key is MT or L, energy_grid is bin boundaries in eV
        block_ranges : dict
            {key: (start, end)} coordinate ranges for each block
        scale : str
            'log' or 'linear' scaling
        """
        import numpy as np
        from matplotlib.ticker import FuncFormatter, FixedLocator

        ax_top = ax.twiny()
        ax_right = ax.twinx()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Collect all tick positions and labels across blocks
        all_major_x_positions = []
        all_minor_x_positions = []
        all_major_y_positions = []
        all_minor_y_positions = []
        tick_labels_x = {}  # position -> label
        tick_labels_y = {}

        # Sort keys to ensure consistent ordering
        sorted_keys = sorted(block_ranges.keys())

        for key in sorted_keys:
            if key not in energy_grids or key not in block_ranges:
                continue

            energy_grid = np.asarray(energy_grids[key], dtype=float)
            coord_start, coord_end = block_ranges[key]
            coord_span = coord_end - coord_start

            if len(energy_grid) < 2 or coord_span <= 0:
                continue

            e_min, e_max = energy_grid.min(), energy_grid.max()

            if scale == 'log':
                log_e_min = np.log10(np.maximum(e_min, 1e-300))
                log_e_max = np.log10(e_max)
                log_span = log_e_max - log_e_min

                if log_span <= 0:
                    continue

                # Find decade boundaries
                decade_min = int(np.floor(log_e_min))
                decade_max = int(np.ceil(log_e_max))

                # Major ticks at every decade
                for decade in range(decade_min, decade_max + 1):
                    e = 10**decade
                    if e_min <= e <= e_max:
                        # Map energy to coordinate position within this block
                        frac = (np.log10(e) - log_e_min) / log_span
                        pos = coord_start + frac * coord_span

                        # Add to x and y positions
                        all_major_x_positions.append(pos)
                        all_major_y_positions.append(pos)

                        # Label format
                        label = f'1e{decade:+03d}'
                        tick_labels_x[pos] = label
                        tick_labels_y[pos] = label

                # Minor ticks at 2-9 within each decade
                for decade in range(decade_min, decade_max + 1):
                    for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                        e = mult * 10**decade
                        if e_min <= e <= e_max:
                            frac = (np.log10(e) - log_e_min) / log_span
                            pos = coord_start + frac * coord_span
                            all_minor_x_positions.append(pos)
                            all_minor_y_positions.append(pos)

            else:  # linear
                from matplotlib.ticker import MaxNLocator

                locator = MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
                tick_energies = locator.tick_values(e_min, e_max)
                tick_energies = tick_energies[(tick_energies >= e_min) & (tick_energies <= e_max)]

                energy_span = e_max - e_min
                if energy_span <= 0:
                    continue

                for e in tick_energies:
                    frac = (e - e_min) / energy_span
                    pos = coord_start + frac * coord_span

                    all_major_x_positions.append(pos)
                    all_major_y_positions.append(pos)

                    # Format label
                    if e == 0:
                        label = '0'
                    else:
                        exponent = int(np.floor(np.log10(abs(e))))
                        mantissa = e / (10**exponent)
                        if abs(mantissa - 1.0) < 0.1:
                            label = f'1e{exponent:+03d}'
                        else:
                            label = f'{int(np.round(mantissa))}e{exponent:+03d}'

                    tick_labels_x[pos] = label
                    tick_labels_y[pos] = label

        # Set up top axis with x ticks
        ax_top.set_xlim(xlim)
        if all_major_x_positions:
            ax_top.xaxis.set_major_locator(FixedLocator(all_major_x_positions))
            # Create formatter that looks up labels
            def format_x(val, pos=None):
                return tick_labels_x.get(val, '')
            ax_top.xaxis.set_major_formatter(FuncFormatter(format_x))
        if all_minor_x_positions:
            ax_top.xaxis.set_minor_locator(FixedLocator(all_minor_x_positions))
        ax_top.tick_params(axis='x', which='major', labelsize=8, rotation=30, length=6, pad=2)
        # Adjust label alignment to shift right
        for label in ax_top.get_xticklabels():
            label.set_ha('left')
        ax_top.tick_params(axis='x', which='minor', length=3)

        # Set up right axis with y ticks
        ax_right.set_ylim(ylim)
        if all_major_y_positions:
            ax_right.yaxis.set_major_locator(FixedLocator(all_major_y_positions))
            def format_y(val, pos=None):
                return tick_labels_y.get(val, '')
            ax_right.yaxis.set_major_formatter(FuncFormatter(format_y))
        if all_minor_y_positions:
            ax_right.yaxis.set_minor_locator(FixedLocator(all_minor_y_positions))
        ax_right.tick_params(axis='y', which='major', labelsize=8, length=6)
        ax_right.tick_params(axis='y', which='minor', length=3)

    def _add_uncertainty_energy_ticks(
        self,
        ax: plt.Axes,
        energy_grid: np.ndarray,
        scale: str = 'log',
        xlim: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Add energy tick marks on bottom of uncertainty panel.

        Parameters
        ----------
        ax : plt.Axes
            Uncertainty panel axes (uses local coordinates starting at 0)
        energy_grid : np.ndarray
            Energy bin boundaries (in original eV units)
        scale : str
            'log' or 'linear' scaling
        xlim : tuple, optional
            X-axis limits (start, end) in transformed local coordinates
        """
        import numpy as np
        from matplotlib.ticker import FuncFormatter, FixedLocator

        if xlim is None:
            xlim = ax.get_xlim()

        if scale == 'log':
            e_min, e_max = energy_grid.min(), energy_grid.max()
            log_e_min = np.log10(np.maximum(e_min, 1e-300))
            log_e_max = np.log10(e_max)

            # Find decade boundaries
            decade_min = int(np.floor(log_e_min))
            decade_max = int(np.ceil(log_e_max))

            # Major ticks at every decade
            major_tick_energies = [10**d for d in range(decade_min, decade_max + 1)]
            major_tick_positions = [np.log10(e) - log_e_min for e in major_tick_energies
                                   if e_min <= e <= e_max]

            # Minor ticks at 2e+XX, 3e+XX, ..., 9e+XX
            minor_tick_positions = []
            for decade in range(decade_min, decade_max + 1):
                for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
                    e = multiplier * 10**decade
                    if e_min <= e <= e_max:
                        pos = np.log10(e) - log_e_min
                        minor_tick_positions.append(pos)

            # Format function for scientific notation
            def format_energy_log(shifted_log_val, pos=None):
                log_val = shifted_log_val + log_e_min
                energy = 10**log_val
                exponent = int(np.round(log_val))
                mantissa = energy / (10**exponent)
                if abs(mantissa - 1.0) < 0.1:
                    return f'1e{exponent:+03d}'
                else:
                    return f'{int(np.round(mantissa))}e{exponent:+03d}'

            formatter = FuncFormatter(format_energy_log)

            # Set ticks on bottom axis
            ax.set_xlim(xlim)
            ax.xaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax.tick_params(axis='x', which='major', length=4,
                          bottom=True, top=False, labelbottom=False, labeltop=False, direction='in')
            ax.tick_params(axis='x', which='minor', length=2,
                          bottom=True, top=False, direction='in')
        else:
            # Linear scale
            from matplotlib.ticker import MaxNLocator

            locator = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
            tick_positions = locator.tick_values(energy_grid.min(), energy_grid.max())
            tick_positions = tick_positions[(tick_positions >= energy_grid.min()) &
                                           (tick_positions <= energy_grid.max())]

            def format_energy_linear(val, pos=None):
                if val == 0:
                    return '0'
                exponent = int(np.floor(np.log10(abs(val))))
                mantissa = val / (10**exponent)
                if abs(mantissa - 1.0) < 0.1:
                    return f'1e{exponent:+03d}'
                else:
                    return f'{int(np.round(mantissa))}e{exponent:+03d}'

            formatter = FuncFormatter(format_energy_linear)

            ax.set_xlim(xlim)
            ax.set_xticks(tick_positions)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.tick_params(axis='x', length=4,
                          bottom=True, top=False, labelbottom=False, labeltop=False, direction='in')
            ax.tick_params(axis='x', which='minor', length=2,
                          bottom=True, top=False, direction='in')

    def _draw_block_boundaries(
        self,
        ax: plt.Axes,
        heatmap_data: 'HeatmapPlotData'
    ) -> None:
        """Draw light grid lines separating MT/L submatrices."""
        from .plot_data import CovarianceHeatmapData, MF34HeatmapData

        ranges_x = []
        ranges_y = []

        if isinstance(heatmap_data, CovarianceHeatmapData):
            info = heatmap_data.block_info or {}
            mts = info.get('mts', [])
            energy_ranges = info.get('energy_ranges', {}) or {}
            ranges_fallback = info.get('ranges', []) or []

            if heatmap_data.is_diagonal:
                for i, m in enumerate(mts):
                    rng = energy_ranges.get(m)
                    if rng is None and i < len(ranges_fallback):
                        rng = ranges_fallback[i]
                    if rng is not None:
                        ranges_x.append(tuple(rng))
                        ranges_y.append(tuple(rng))
            elif len(mts) >= 2:
                col_rng = energy_ranges.get(mts[1]) or (ranges_fallback[1] if len(ranges_fallback) > 1 else None)
                row_rng = energy_ranges.get(mts[0]) or (ranges_fallback[0] if ranges_fallback else None)
                if col_rng is not None:
                    ranges_x.append(tuple(col_rng))
                if row_rng is not None:
                    ranges_y.append(tuple(row_rng))

        elif isinstance(heatmap_data, MF34HeatmapData):
            info = heatmap_data.block_info or {}
            legendre_list = info.get('legendre_coeffs', heatmap_data.legendre_coeffs)
            energy_ranges = info.get('energy_ranges', {}) or {}
            ranges_idx = info.get('ranges', {}) or {}

            if heatmap_data.is_diagonal and len(legendre_list) > 1:
                for l_val in legendre_list:
                    rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                    if rng is not None:
                        ranges_x.append(tuple(rng))
                        ranges_y.append(tuple(rng))
            elif len(legendre_list) >= 2:
                col_l = legendre_list[1]
                row_l = legendre_list[0]
                col_rng = energy_ranges.get(col_l) or ranges_idx.get(col_l)
                row_rng = energy_ranges.get(row_l) or ranges_idx.get(row_l)
                if col_rng is not None:
                    ranges_x.append(tuple(col_rng))
                if row_rng is not None:
                    ranges_y.append(tuple(row_rng))

        if len(ranges_x) <= 1 and len(ranges_y) <= 1:
            return

        def _boundary_positions(ranges):
            sorted_ranges = sorted(ranges, key=lambda r: r[0])
            return [r[0] for r in sorted_ranges[1:]]

        for x in _boundary_positions(ranges_x):
            ax.axvline(x, color="#404040", lw=1.0, alpha=0.8, zorder=5)
        for y in _boundary_positions(ranges_y):
            ax.axhline(y, color="#404040", lw=1.0, alpha=0.8, zorder=5)
    
    def _draw_uncertainty_panels(
        self,
        uncertainty_axes: List[plt.Axes],
        heatmap_data: 'HeatmapPlotData',
        ax_heatmap: plt.Axes,
        x_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Draw uncertainty panels above heatmap.
        
        Uses step plotting with 'post' stepping for proper bin coverage,
        adds internal grid lines at 10% intervals with right-aligned labels.
        
        Parameters
        ----------
        x_limits : tuple, optional
            X-axis limits (start, end) in transformed coordinates to match heatmap zoom
        """
        from .plot_data import CovarianceHeatmapData, MF34HeatmapData
        import numpy as np
        
        uncertainty_data = heatmap_data.uncertainty_data
        if not uncertainty_data:
            return
        
        background_color = "#F5F5F5"
        tick_grey = "#707070"
        
        # Helper function to draw internal y-grid
        def _nice_ylim_and_step(max_val: float) -> tuple[float, float]:
            """Return (top, step) for readable grids with ~5-8 ticks."""
            if not np.isfinite(max_val) or max_val <= 0:
                return 1.0, 0.2
            base = 10.0 ** np.floor(np.log10(max_val))
            for mult in (1.0, 2.0, 5.0, 10.0):
                step = mult * base
                if max_val / step <= 8.0:
                    break
            top = np.ceil(max_val / step) * step
            return float(top), float(step)

        def _draw_ygrid_inside(ax_u, xr, ymax):
            """Draw grid lines with adaptive spacing and left-aligned labels inside uncertainty panel."""
            top, step = _nice_ylim_and_step(ymax)
            grid_vals = np.arange(0.0, top + 1e-9, step)
            for y in grid_vals:
                ax_u.axhline(y, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)
            x_label = xr[0] + 0.02 * (xr[1] - xr[0])
            for y in grid_vals:
                lbl = f"{y:g}" if step < 1 else f"{int(y)}"
                ax_u.text(x_label, y, lbl, ha="left", va="center",
                         color=tick_grey, fontsize=8, alpha=0.9, zorder=2)
        
        # Get sorted keys (MTs or Legendre orders)
        keys = sorted(uncertainty_data.keys())
        
        if isinstance(heatmap_data, MF34HeatmapData):
            # MF34 uncertainty panels
            # Use transformed energy coordinates matching the heatmap
            edges_map = {}
            if heatmap_data.block_info:
                edges_map = heatmap_data.block_info.get("edges_transformed", {}) or {}

            def _edges_for_L(L_val: int) -> Optional[np.ndarray]:
                if L_val in edges_map:
                    return np.asarray(edges_map[L_val], dtype=float)
                raw = heatmap_data.energy_grids.get(L_val)
                if raw is None:
                    return None
                if heatmap_data.scale == "log":
                    transformed = np.log10(np.maximum(raw, 1e-300))
                else:
                    transformed = np.asarray(raw, dtype=float)
                return transformed - transformed[0]

            if heatmap_data.is_diagonal and len(keys) > 1:
                # Multiple diagonal blocks - each panel shows one L using global coords
                for i, (ax_u, L) in enumerate(zip(uncertainty_axes, keys)):
                    sigma_pct = uncertainty_data[L]
                    xs_edges = _edges_for_L(L)
                    if xs_edges is None or xs_edges.size == 0:
                        continue

                    ax_u.set_facecolor(background_color)
                    ax_u.grid(False)

                    # Use local coordinates so each panel width matches its own block
                    xs_local = xs_edges - xs_edges[0]
                    sigma_vals = sigma_pct if sigma_pct.size == xs_local.size - 1 else sigma_pct
                    if sigma_vals.size > 0:
                        sigma_ext = np.append(sigma_vals, sigma_vals[-1])
                    else:
                        sigma_ext = sigma_vals

                    if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                        ax_u.step(xs_local, sigma_ext, where='post',
                                  linewidth=1.4, color=f"C{i}", zorder=3)
                        y_max = float(np.nanmax(sigma_ext))
                    else:
                        y_max = 5.0

                    y_max = max(y_max, 1.0)
                    top, _ = _nice_ylim_and_step(y_max * 1.05)
                    ax_u.set_ylim(0.0, top)
                    
                    # Apply x_limits if provided (to match heatmap zoom)
                    if x_limits is not None:
                        ax_u.set_xlim(x_limits)
                    else:
                        ax_u.set_xlim(0.0, xs_local[-1])

                    # Add energy ticks to uncertainty panel
                    raw_grid = heatmap_data.energy_grids.get(L) if heatmap_data.energy_grids else None
                    if raw_grid is not None:
                        current_xlim = ax_u.get_xlim()
                        self._add_uncertainty_energy_ticks(ax_u, raw_grid, heatmap_data.scale,
                                                         xlim=current_xlim)
                    else:
                        ax_u.set_xticks([])
                    ax_u.set_yticks([])

                    _draw_ygrid_inside(ax_u, (0.0, xs_local[-1]), ax_u.get_ylim()[1])

                    if i == 0:
                        ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

                    for side in ('left', 'right', 'top', 'bottom'):
                        ax_u.spines[side].set_visible(False)
            else:
                # Single block (one L or off-diagonal)
                ax_u = uncertainty_axes[0]
                L = keys[0]
                sigma_pct = uncertainty_data[L]
                xs_edges = _edges_for_L(L)
                if xs_edges is None or xs_edges.size == 0:
                    return

                ax_u.set_facecolor(background_color)
                ax_u.grid(False)

                xs_local = xs_edges - xs_edges[0]
                sigma_ext = np.append(sigma_pct, sigma_pct[-1]) if sigma_pct.size > 0 else sigma_pct

                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    ax_u.step(xs_local, sigma_ext, where='post',
                              linewidth=1.4, color='C0', zorder=3)
                    y_max = float(np.nanmax(sigma_ext))
                else:
                    y_max = 5.0

                y_max = max(y_max, 1.0)
                top, _ = _nice_ylim_and_step(y_max * 1.05)
                ax_u.set_ylim(0.0, top)

                # Apply x_limits if provided (to match heatmap zoom)
                if x_limits is not None:
                    ax_u.set_xlim(x_limits)
                else:
                    ax_u.set_xlim(0.0, xs_local[-1])

                # Add energy ticks to uncertainty panel
                raw_grid = heatmap_data.energy_grids.get(L) if heatmap_data.energy_grids else None
                if raw_grid is not None:
                    current_xlim = ax_u.get_xlim()
                    self._add_uncertainty_energy_ticks(ax_u, raw_grid, heatmap_data.scale,
                                                     xlim=current_xlim)
                else:
                    ax_u.set_xticks([])
                ax_u.set_yticks([])

                _draw_ygrid_inside(ax_u, (0.0, xs_local[-1]), ax_u.get_ylim()[1])

                ax_u.set_ylabel('Unc. (%)', fontsize=10)

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_u.spines[side].set_visible(False)

        elif isinstance(heatmap_data, CovarianceHeatmapData):
            # CovMat uncertainty panels
            energy_grid = heatmap_data.energy_grid

            if energy_grid is not None:
                if heatmap_data.scale == "log":
                    edges_tx = np.log10(np.maximum(energy_grid, 1e-300))
                else:
                    edges_tx = np.asarray(energy_grid, dtype=float)
                edges_tx = edges_tx - edges_tx[0]
            else:
                edges_tx = None
            
            for i, (ax_unc, key) in enumerate(zip(uncertainty_axes, keys)):
                sigma_pct = uncertainty_data[key]

                if edges_tx is not None and len(edges_tx) == len(sigma_pct) + 1:
                    xs_edges = edges_tx
                else:
                    xs_edges = np.arange(len(sigma_pct) + 1, dtype=float)

                sigma_ext = np.append(sigma_pct, sigma_pct[-1]) if sigma_pct.size > 0 else sigma_pct
                xs_plot = xs_edges

                ax_unc.set_facecolor(background_color)
                ax_unc.grid(False)

                color = self._colors[i % len(self._colors)] if getattr(self, "_colors", None) else f"C{i}"
                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    ax_unc.step(xs_plot, sigma_ext, where='post',
                                linewidth=1.4, color=color, zorder=3)
                    y_max = float(np.nanmax(sigma_ext))
                else:
                    y_max = 5.0

                y_max = max(y_max, 1.0)
                top, _ = _nice_ylim_and_step(y_max * 1.05)
                ax_unc.set_ylim(0.0, top)
                
                # Apply x_limits if provided (to match heatmap zoom)
                if x_limits is not None:
                    ax_unc.set_xlim(x_limits)
                else:
                    ax_unc.set_xlim(xs_plot[0], xs_plot[-1])

                # Add energy ticks to uncertainty panel
                if energy_grid is not None:
                    current_xlim = ax_unc.get_xlim()
                    self._add_uncertainty_energy_ticks(ax_unc, energy_grid, heatmap_data.scale,
                                                     xlim=current_xlim)
                else:
                    ax_unc.set_xticks([])
                ax_unc.set_yticks([])

                if xs_edges.size >= 2:
                    _draw_ygrid_inside(ax_unc, (xs_edges[0], xs_edges[-1]), ax_unc.get_ylim()[1])

                if i == 0:
                    ax_unc.set_ylabel('Unc. (%)', fontsize=10, color='black')

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_unc.spines[side].set_visible(False)
    
    def _add_block_labels_figure_fixed(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        centers: List[float],
        labels: List[str],
        full_xlim: Tuple[float, float],
        full_ylim: Tuple[float, float],
        *,
        pad_frac_x: float = 0.000,  # Reduced to bring bottom labels closer to plot
        pad_frac_y: float = 0.012,  # Reduced to bring left labels closer to plot
        fontsize: Optional[float] = None,
        x_axis_label: Optional[str] = None,  # Outer label for x-axis (e.g., "MT Number")
        y_axis_label: Optional[str] = None,  # Outer label for y-axis (e.g., "Legendre Order")
    ) -> None:
        """
        Place block labels relative to the figure (not axis ticks), so they do not shift
        when set_xlim/set_ylim is used.

        Positions are computed relative to the FULL data extent (full_xlim/full_ylim).
        Bottom labels are placed further down to appear below energy tick labels.
        Left labels are placed further left to avoid overlap with y-axis tick labels.

        Optionally adds outer axis labels (x_axis_label, y_axis_label) centered on each axis,
        positioned further out than the block number labels.
        """
        ax_pos = ax.get_position()

        x0, x1 = float(full_xlim[0]), float(full_xlim[1])
        y0, y1 = float(full_ylim[0]), float(full_ylim[1])
        dx = (x1 - x0) if (x1 > x0) else 1.0
        dy = (y1 - y0) if (y1 > y0) else 1.0

        fs = fontsize if fontsize is not None else (self._tick_labelsize or 11)
        outer_label_fs = fs  # Same font size for outer labels

        # Bottom labels (x direction) - placed further down to swap with energy ticks
        y_text = ax_pos.y0 - pad_frac_x
        for c, lab in zip(centers, labels):
            frac = (float(c) - x0) / dx
            frac = min(1.0, max(0.0, frac))
            x_text = ax_pos.x0 + frac * ax_pos.width
            fig.text(x_text, y_text, lab, ha="center", va="top", fontsize=fs)

        # Left labels (y direction) - placed further left for more separation
        # INVERT because heatmap y-axis is flipped (origin='upper' / set_ylim(high, low))
        x_text = ax_pos.x0 - pad_frac_y
        for c, lab in zip(centers, labels):
            frac = (float(c) - y0) / dy
            frac = min(1.0, max(0.0, frac))
            frac = 1.0 - frac  # Invert for y-axis
            y_text = ax_pos.y0 + frac * ax_pos.height
            fig.text(x_text, y_text, lab, ha="right", va="center", fontsize=fs)

        # Outer axis labels (further out, centered on each axis)
        outer_offset = 0.025  # Additional offset for outer labels

        if x_axis_label:
            # Bottom outer label - centered horizontally, below the block labels
            x_center = ax_pos.x0 + ax_pos.width / 2
            y_outer = ax_pos.y0 - pad_frac_x - outer_offset
            fig.text(x_center, y_outer, x_axis_label, ha="center", va="top",
                    fontsize=outer_label_fs)

        if y_axis_label:
            # Left outer label - centered vertically, left of the block labels, rotated 90°
            x_outer = ax_pos.x0 - pad_frac_y - outer_offset
            y_center = ax_pos.y0 + ax_pos.height / 2
            fig.text(x_outer, y_center, y_axis_label, ha="right", va="center",
                    fontsize=outer_label_fs, rotation=90)

    def build(self, show: bool = False) -> plt.Figure:
        """
        Build and return the figure.
        
        Parameters
        ----------
        show : bool
            Whether to display the figure immediately
            
        Returns
        -------
        matplotlib.figure.Figure
            The completed figure
        """
        # Check if we're building a heatmap or line plot
        if self._heatmap_data is not None:
            # Build heatmap
            fig = self._build_heatmap()
            if show:
                plt.show()
            return fig
        
        # Otherwise, build line plot (existing logic)
        # Get default colors
        default_colors = self._colors
        
        # Plot uncertainty bands first (so they appear behind the lines)
        for band, data_idx in self._uncertainty_bands:
            # Check if the associated data is a step plot (for multigroup/uncertainty data)
            data = self._data_list[data_idx] if data_idx < len(self._data_list) else None

            # Resolve color from associated data when not set on the band
            resolved_color = band.color
            if resolved_color is None and data_idx < len(self._data_list):
                if 'color' in self._custom_styling[data_idx]:
                    resolved_color = self._custom_styling[data_idx]['color']
                elif self._data_list[data_idx].color is not None:
                    resolved_color = self._data_list[data_idx].color

            # Convert relative uncertainties to absolute if needed
            if band.is_relative():
                if data is None:
                    raise ValueError("Cannot convert relative uncertainty to absolute without associated data")
                y_lower, y_upper = band.to_absolute(data.y)
            else:
                y_lower, y_upper = band.y_lower, band.y_upper

            if band.style == 'errorbar':
                # Render as vertical error bars
                eb_kwargs = band.get_errorbar_kwargs()
                if resolved_color is not None and 'ecolor' not in eb_kwargs:
                    eb_kwargs['ecolor'] = resolved_color

                if data and data.plot_type == 'step':
                    # For step data the x array has n+1 bin edges and y has
                    # n+1 values (last repeated).  Place errorbars at bin
                    # midpoints using only the first n values — matching
                    # Coefficients._plot_on_ax() behaviour.
                    x = np.asarray(band.x)
                    x_mid = (x[:-1] + x[1:]) / 2.0
                    n = len(x_mid)
                    y_nom = data.y[:n]
                    yerr_lo = np.abs(y_nom - y_lower[:n])
                    yerr_hi = np.abs(y_upper[:n] - y_nom)
                    self.ax.errorbar(x_mid, y_nom, yerr=[yerr_lo, yerr_hi], **eb_kwargs)
                else:
                    y_nom = data.y if data is not None else (y_lower + y_upper) / 2.0
                    yerr_lo = np.abs(y_nom - y_lower)
                    yerr_hi = np.abs(y_upper - y_nom)
                    self.ax.errorbar(band.x, y_nom, yerr=[yerr_lo, yerr_hi], **eb_kwargs)
            else:
                # Default: shaded band via fill_between
                fill_kwargs = band.get_fill_kwargs()
                if resolved_color is not None and 'color' not in fill_kwargs:
                    fill_kwargs['color'] = resolved_color

                if data and data.plot_type == 'step':
                    self.ax.fill_between(band.x, y_lower, y_upper, step='post', **fill_kwargs)
                else:
                    self.ax.fill_between(band.x, y_lower, y_upper, **fill_kwargs)
        
        # Plot each data object
        for i, (data, styling_overrides) in enumerate(zip(self._data_list, self._custom_styling)):
            # Merge styling: data defaults < custom overrides
            plot_kwargs = data.get_plot_kwargs()
            plot_kwargs.update(styling_overrides)
            
            # Auto-assign color if not specified
            if 'color' not in plot_kwargs and default_colors is not None:
                plot_kwargs['color'] = default_colors[i % len(default_colors)]
            
            # Determine effective plot type - drawstyle can override plot_type
            # This allows frontend to specify step plots via drawstyle parameter
            effective_plot_type = data.plot_type
            step_where = getattr(data, 'step_where', 'post')  # default step direction
            
            if data.drawstyle is not None and 'steps' in data.drawstyle:
                effective_plot_type = 'step'
                # Extract step direction from drawstyle (steps-pre, steps-post, steps-mid)
                if 'pre' in data.drawstyle:
                    step_where = 'pre'
                elif 'mid' in data.drawstyle:
                    step_where = 'mid'
                else:
                    step_where = 'post'
            
            # Plot based on effective plot_type
            if effective_plot_type == 'line':
                self.ax.plot(data.x, data.y, **plot_kwargs)
            
            elif effective_plot_type == 'step':
                # Handle step plots (common for uncertainties and multigroup data)
                # step_where was already computed above from drawstyle or data.step_where
                
                # Check if markers are requested
                marker = plot_kwargs.get('marker', None)
                if marker is not None and marker != '':
                    # For step plots with markers, we need to:
                    # 1. Plot the step line without markers
                    # 2. Plot markers separately at segment midpoints
                    
                    # Extract marker properties
                    markersize = plot_kwargs.pop('markersize', None)
                    markerfacecolor = plot_kwargs.get('color', None)
                    markeredgecolor = plot_kwargs.get('color', None)
                    
                    # Plot step line without marker
                    marker_backup = plot_kwargs.pop('marker')
                    self.ax.step(data.x, data.y, where=step_where, **plot_kwargs)
                    
                    # Calculate midpoints for markers
                    # For 'post' step: each segment spans from x[i] to x[i+1] at height y[i]
                    # For 'pre' step: each segment spans from x[i-1] to x[i] at height y[i]
                    # For 'mid' step: segments are centered at x[i]
                    
                    if step_where == 'post' and len(data.x) > 1:
                        # Midpoints between consecutive x values
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[:-1]  # Heights correspond to the left point
                    elif step_where == 'pre' and len(data.x) > 1:
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[1:]  # Heights correspond to the right point
                    elif step_where == 'mid':
                        x_mid = data.x
                        y_mid = data.y
                    else:
                        # Fallback: use original points
                        x_mid = data.x
                        y_mid = data.y
                    
                    # Plot markers at midpoints
                    marker_kwargs = {
                        'marker': marker_backup,
                        'linestyle': 'none',
                        'color': markerfacecolor,
                        'markeredgecolor': markeredgecolor,
                    }
                    if markersize is not None:
                        marker_kwargs['markersize'] = markersize
                    if 'alpha' in plot_kwargs:
                        marker_kwargs['alpha'] = plot_kwargs['alpha']
                    
                    self.ax.plot(x_mid, y_mid, **marker_kwargs)
                else:
                    # No markers, just plot the step normally
                    self.ax.step(data.x, data.y, where=step_where, **plot_kwargs)
            
            elif data.plot_type == 'scatter':
                # Convert markersize to s for scatter plots
                if 'markersize' in plot_kwargs:
                    plot_kwargs['s'] = plot_kwargs.pop('markersize')
                self.ax.scatter(data.x, data.y, **plot_kwargs)
            
            elif data.plot_type == 'errorbar':
                # Extract error bar specific kwargs
                yerr = plot_kwargs.pop('yerr', None)
                xerr = plot_kwargs.pop('xerr', None)
                # Also check metadata for yerr/xerr (used by EXFOR data)
                if yerr is None and 'yerr' in data.metadata:
                    yerr = data.metadata['yerr']
                if xerr is None and 'xerr' in data.metadata:
                    xerr = data.metadata['xerr']
                # Get capsize from metadata if available
                capsize = data.metadata.get('capsize', 2)
                self.ax.errorbar(data.x, data.y, yerr=yerr, xerr=xerr, capsize=capsize, **plot_kwargs)
            
            else:
                raise ValueError(f"Unknown plot_type: {data.plot_type}")
        
        # Apply scales
        if self._use_log_x:
            self.ax.set_xscale('log')
        if self._use_log_y:
            self.ax.set_yscale('log')
        
        # Set tight limits by default (no margins) if limits are not specified
        if self._x_lim is None and self._data_list:
            # Find the data range across all datasets
            x_min = min(np.min(data.x) for data in self._data_list if len(data.x) > 0)
            x_max = max(np.max(data.x) for data in self._data_list if len(data.x) > 0)
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                self.ax.set_xlim(x_min, x_max)
        else:
            # Apply user-specified limits (validate for log scale)
            if self._x_lim is not None:
                x_lim = self._x_lim
                if self._use_log_x:
                    # For log scale, ensure limits are positive
                    x_lim = (max(x_lim[0], 1e-10), max(x_lim[1], 1e-10))
                self.ax.set_xlim(x_lim)

        if self._y_lim is None and self._data_list:
            # Find the data range across all datasets (including uncertainty bands)
            y_values = []
            for data in self._data_list:
                if len(data.y) > 0:
                    y_values.extend(data.y)
            # Also include uncertainty band values
            for band, data_idx in self._uncertainty_bands:
                if band.is_relative():
                    # Need to convert to absolute first
                    if data_idx < len(self._data_list):
                        data = self._data_list[data_idx]
                        y_lower, y_upper = band.to_absolute(data.y)
                        if len(y_lower) > 0:
                            y_values.extend(y_lower)
                        if len(y_upper) > 0:
                            y_values.extend(y_upper)
                else:
                    # Already absolute
                    if band.y_lower is not None and len(band.y_lower) > 0:
                        y_values.extend(band.y_lower)
                    if band.y_upper is not None and len(band.y_upper) > 0:
                        y_values.extend(band.y_upper)
            
            if y_values:
                y_arr = np.array(y_values)
                y_arr = y_arr[np.isfinite(y_arr)]  # Remove inf/nan
                if len(y_arr) > 0:
                    y_min, y_max = np.min(y_arr), np.max(y_arr)
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                        # Add small padding for y-axis (5% on each side)
                        if self._use_log_y and y_min > 0:
                            # For log scale, use multiplicative padding
                            log_range = np.log10(y_max) - np.log10(y_min)
                            padding = 0.05 * log_range
                            y_min = 10 ** (np.log10(y_min) - padding)
                            y_max = 10 ** (np.log10(y_max) + padding)
                        else:
                            # For linear scale, use additive padding
                            padding = 0.05 * (y_max - y_min)
                            y_min = y_min - padding
                            y_max = y_max + padding
                        self.ax.set_ylim(y_min, y_max)
        else:
            # Apply user-specified limits (validate for log scale)
            if self._y_lim is not None:
                y_lim = self._y_lim
                if self._use_log_y:
                    # For log scale, ensure limits are positive
                    y_lim = (max(y_lim[0], 1e-10), max(y_lim[1], 1e-10))
                self.ax.set_ylim(y_lim)

        # Apply axis labels
        if self._x_label is not None:
            if self._label_fontsize is not None:
                self.ax.set_xlabel(self._x_label, fontsize=self._label_fontsize)
            else:
                self.ax.set_xlabel(self._x_label)
        elif self._use_log_x:
            # Auto-label for energy axis if log scale is used
            is_energy_axis = self._x_label is None or 'energy' in self._x_label.lower()
            if is_energy_axis:
                self.ax.set_xlabel("Energy (MeV)")
        else:
            # Auto-label for group index if not log scale and no label
            if self._x_label is None:
                self.ax.set_xlabel("Energy-group index")
                self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        
        if self._y_label is not None:
            if self._label_fontsize is not None:
                self.ax.set_ylabel(self._y_label, fontsize=self._label_fontsize)
            else:
                self.ax.set_ylabel(self._y_label)
        
        # Apply title (only if explicitly set to a non-empty string)
        if self._title is not _NOT_SET and self._title:
            if self._title_fontsize is not None:
                self.ax.set_title(self._title, fontsize=self._title_fontsize)
            else:
                self.ax.set_title(self._title)
        
        # Format energy axis ticks if using log scale on x-axis
        if self._use_log_x:
            format_energy_axis_ticks(self.ax)
        
        # Apply tick label size
        if self._tick_labelsize is not None:
            self.ax.tick_params(axis='both', which='major', labelsize=self._tick_labelsize)
        
        # Add legend if there are labeled artists
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            legend_kwargs = {'loc': self._legend_loc, 'framealpha': 0.9}

            if self._legend_outside:
                legend_kwargs['bbox_to_anchor'] = (1.02, 1)
                legend_kwargs['loc'] = 'upper left'

            if self.style == 'light':
                # For light style, use framed legend with black edge (publication style)
                legend_kwargs.update({
                    'frameon': True,
                    'fancybox': False,
                    'edgecolor': 'black'
                })
            else:
                # For dark style, use default fancybox
                legend_kwargs['fancybox'] = True

            if self._legend_fontsize is not None:
                legend_kwargs['fontsize'] = self._legend_fontsize

            self.ax.legend(**legend_kwargs)

            if self._legend_outside:
                self.fig.subplots_adjust(right=0.78)
        
        # Apply tick parameters if specified
        if hasattr(self, '_tick_params') and self._tick_params:
            from matplotlib.ticker import MaxNLocator
            
            # Limit number of ticks
            if 'max_ticks_x' in self._tick_params:
                max_x = self._tick_params['max_ticks_x']
                if self._use_log_x:
                    # For log scale, use LogLocator with numticks parameter
                    from matplotlib.ticker import LogLocator
                    self.ax.xaxis.set_major_locator(LogLocator(numticks=max_x))
                else:
                    # For linear scale, use MaxNLocator
                    self.ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x, integer=False))
            
            if 'max_ticks_y' in self._tick_params:
                max_y = self._tick_params['max_ticks_y']
                if self._use_log_y:
                    from matplotlib.ticker import LogLocator
                    self.ax.yaxis.set_major_locator(LogLocator(numticks=max_y))
                else:
                    self.ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y, integer=False))
            
            # Rotate tick labels
            if 'rotate_x' in self._tick_params:
                rotation = self._tick_params['rotate_x']
                # Also adjust horizontal alignment for better appearance
                ha = 'right' if rotation > 0 else 'center'
                self.ax.tick_params(axis='x', rotation=rotation)
                for label in self.ax.get_xticklabels():
                    label.set_rotation(rotation)
                    label.set_ha(ha)
            
            if 'rotate_y' in self._tick_params:
                rotation = self._tick_params['rotate_y']
                self.ax.tick_params(axis='y', rotation=rotation)
                for label in self.ax.get_yticklabels():
                    label.set_rotation(rotation)
        
        # Apply tight layout to prevent label cutoff (especially after rotation)
        # Only if constrained_layout is not already being used
        if hasattr(self, '_tick_params') and self._tick_params:
            try:
                # Check if constrained_layout is active
                if not self.fig.get_constrained_layout():
                    self.fig.tight_layout()
            except:
                # Fallback if constrained_layout check fails
                try:
                    self.fig.tight_layout()
                except:
                    pass  # If tight_layout fails, just continue
        
        # Apply grid configuration
        if self._grid:
            # Major grid
            self.ax.grid(True, which='major', linestyle='--', alpha=self._grid_alpha)

            # Minor grid (only when explicitly requested)
            if self._show_minor_grid:
                self.ax.minorticks_on()
                self.ax.grid(True, which='minor', linestyle=':', alpha=self._minor_grid_alpha, linewidth=0.5)
        else:
            self.ax.grid(False)
        
        # Finalize and display
        if show:
            if self._notebook_mode:
                # In notebooks, the plot should auto-display
                # Only call show() if using inline backend
                if plt.get_backend() == 'module://matplotlib_inline.backend_inline':
                    plt.show()
            else:
                plt.show()
        
        return self.fig
    
    def _convert_plotdata_to_band(
        self,
        uncertainty_data: PlotData,
        nominal_data: PlotData
    ) -> UncertaintyBand:
        """
        Convert uncertainty PlotData to UncertaintyBand.
        
        Parameters
        ----------
        uncertainty_data : PlotData
            PlotData containing uncertainty values (MultigroupUncertaintyPlotData or LegendreUncertaintyPlotData)
        nominal_data : PlotData
            The nominal data that these uncertainties correspond to
            
        Returns
        -------
        UncertaintyBand
            Converted uncertainty band object
        """
        # Check if this is a recognized uncertainty PlotData type
        if isinstance(uncertainty_data, (MultigroupUncertaintyPlotData, LegendreUncertaintyPlotData)):
            # These PlotData types now store uncertainties at bin edges (n+1 points)
            # to properly support step plots with where='post'
            
            uncertainty_type = getattr(uncertainty_data, 'uncertainty_type', 'relative')
            
            if uncertainty_type == 'relative':
                # y values are in percentage - convert to fractional (0.05 for 5%)
                rel_unc = np.array(uncertainty_data.y) / 100.0
                unc_x = np.array(uncertainty_data.x)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate uncertainty to match nominal grid
                    # This happens when ENDF MF34 (sparse) is combined with MF4 (dense)
                    rel_unc = np.interp(nom_x, unc_x, rel_unc)
                elif len(nominal_data.y) == len(rel_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case - G vs G+1 points)
                    # Extend uncertainties to match nominal data
                    rel_unc = np.append(rel_unc, rel_unc[-1])
                
                # Use nominal data's x values (energy bin boundaries)
                return UncertaintyBand(
                    x=nominal_data.x,
                    relative_uncertainty=rel_unc,
                    sigma=1.0,  # Uncertainties are already at the desired sigma level
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
            else:  # absolute
                # y values are absolute uncertainties
                abs_unc = np.array(uncertainty_data.y)
                unc_x = np.array(uncertainty_data.x)
                nominal_y = np.array(nominal_data.y)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate
                    abs_unc = np.interp(nom_x, unc_x, abs_unc)
                elif len(nominal_y) == len(abs_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case)
                    abs_unc = np.append(abs_unc, abs_unc[-1])
                
                # Convert to bounds
                y_lower = nominal_y - abs_unc
                y_upper = nominal_y + abs_unc
                
                return UncertaintyBand(
                    x=nominal_data.x,
                    y_lower=y_lower,
                    y_upper=y_upper,
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
        else:
            # For other PlotData types, assume y values are relative uncertainties in fractional form
            rel_unc = np.array(uncertainty_data.y)
            
            # Check if we need to extend
            if len(nominal_data.y) == len(rel_unc) + 1:
                rel_unc = np.append(rel_unc, rel_unc[-1])
            
            return UncertaintyBand(
                x=nominal_data.x,
                relative_uncertainty=rel_unc,
                sigma=1.0,
                color=getattr(uncertainty_data, 'color', None),
                alpha=0.2
            )
    
    def clear(self) -> 'PlotBuilder':
        """
        Clear all data and reset to initial state.
        
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._data_list.clear()
        self._uncertainty_bands.clear()
        self._custom_styling.clear()
        
        if self.ax is not None:
            self.ax.clear()
        
        return self
