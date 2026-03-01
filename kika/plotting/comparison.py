"""
Comparison utilities for plotting cross sections and other nuclear data.

This module provides:
- Grid interpolation between datasets with different x-grids
- Difference/ratio computation (absolute and relative)
- ComparisonBuilder for dual-subplot comparison figures

Examples
--------
>>> # Quick difference check using utility functions
>>> from kika.plotting import compute_difference, PlotBuilder
>>> result = compute_difference(ref_data, cmp_data, mode='relative')
>>> fig = PlotBuilder().add_data(result.difference).build()

>>> # Full dual-panel comparison figure
>>> from kika.plotting import ComparisonBuilder
>>> fig = (ComparisonBuilder()
...     .set_reference(ref_data)
...     .add_comparison(cmp_data)
...     .set_difference_panel(mode='relative')
...     .set_scales(log_x=True, log_y=True)
...     .build())
"""

from typing import Optional, Tuple, List, Literal
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from .plot_data import PlotData, DifferencePlotData
from .plot_builder import PlotBuilder, _NOT_SET
from .styles import (
    _get_color_palette,
    _apply_style_to_rcparams,
    format_energy_axis_ticks,
)
from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _configure_figure_interactivity,
)


# ---------------------------------------------------------------------------
# Auto-interpolation defaults by PlotData subclass
# ---------------------------------------------------------------------------

_INTERPOLATION_DEFAULTS = {
    'CrossSectionPlotData': 'log-log',
    'AngularDistributionPlotData': 'lin-lin',
    'LegendreCoeffPlotData': 'log-log',
    'MultigroupXSPlotData': 'lin-lin',
    'MultigroupUncertaintyPlotData': 'lin-lin',
}


# ---------------------------------------------------------------------------
# Interpolation utility
# ---------------------------------------------------------------------------

def interpolate_to_grid(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    method: str = 'log-log',
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Interpolate ``(x_source, y_source)`` onto *x_target*.

    Parameters
    ----------
    x_target : array-like
        Target x-grid to interpolate onto.
    x_source : array-like
        Source x-grid (must be monotonically increasing).
    y_source : array-like
        Source y-values corresponding to *x_source*.
    method : str
        Interpolation space:

        * ``'log-log'`` – log-space in both axes (standard for cross
          sections that span many orders of magnitude).
        * ``'lin-lin'`` – linear interpolation in linear space.
        * ``'log-lin'`` – log x, linear y.
        * ``'lin-log'`` – linear x, log y.
    fill_value : float
        Value assigned to target points outside the source range.
        Default ``np.nan`` so out-of-range points are clearly marked.

    Returns
    -------
    np.ndarray
        Interpolated y-values on *x_target*.  Points outside the source
        range are set to *fill_value*.
    """
    x_src = np.asarray(x_source, dtype=float)
    y_src = np.asarray(y_source, dtype=float)
    x_tgt = np.asarray(x_target, dtype=float)

    if len(x_src) != len(y_src):
        raise ValueError(
            f"x_source and y_source must have the same length. "
            f"Got {len(x_src)} and {len(y_src)}"
        )

    # Points inside the source range
    in_range = (x_tgt >= x_src[0]) & (x_tgt <= x_src[-1])
    result = np.full_like(x_tgt, fill_value, dtype=float)

    if not np.any(in_range):
        return result

    x_in = x_tgt[in_range]

    log_x = method in ('log-log', 'log-lin')
    log_y = method in ('log-log', 'lin-log')

    # --- x transform ---
    if log_x:
        if np.any(x_src <= 0) or np.any(x_in <= 0):
            raise ValueError(
                f"method='{method}' requires positive x values. "
                f"Source x range: [{x_src.min()}, {x_src.max()}]"
            )
        xi = np.log(x_in)
        xs = np.log(x_src)
    else:
        xi = x_in
        xs = x_src

    # --- y transform + interpolation ---
    if log_y:
        safe = y_src > 0
        if np.all(safe):
            ys = np.log(y_src)
            yi = np.interp(xi, xs, ys)
            result[in_range] = np.exp(yi)
        else:
            # Fallback to linear when some y values are non-positive
            yi = np.interp(xi, xs, y_src)
            result[in_range] = yi
    else:
        yi = np.interp(xi, xs, y_src)
        result[in_range] = yi

    return result


# ---------------------------------------------------------------------------
# Difference computation
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """
    Result of comparing two datasets.

    Attributes
    ----------
    reference : PlotData
        Reference data on the common grid.
    comparison : PlotData
        Comparison data on the common grid.
    difference : DifferencePlotData
        Computed difference (relative or absolute).
    common_x : np.ndarray
        Common x-grid used for the comparison.
    valid_mask : np.ndarray
        Boolean mask – ``True`` where both datasets have finite values.
    """
    reference: PlotData
    comparison: PlotData
    difference: DifferencePlotData
    common_x: np.ndarray
    valid_mask: np.ndarray


def compute_difference(
    reference: PlotData,
    comparison: PlotData,
    mode: Literal['relative', 'absolute'] = 'relative',
    interpolation: Optional[str] = 'log-log',
    grid: Literal['reference', 'comparison', 'union'] = 'reference',
    relative_in_percent: bool = True,
) -> ComparisonResult:
    """
    Compute the difference between two :class:`PlotData` datasets.

    Handles different x-grids by interpolating onto a common grid.

    Parameters
    ----------
    reference : PlotData
        Baseline dataset.
    comparison : PlotData
        Dataset to compare against the baseline.
    mode : {'relative', 'absolute'}
        * ``'relative'``: ``(comparison - reference) / reference``
        * ``'absolute'``: ``comparison - reference``
    interpolation : str
        Interpolation method passed to :func:`interpolate_to_grid`.
    grid : {'reference', 'comparison', 'union'}
        Which grid to use as common grid.

        * ``'reference'`` (default): interpolate comparison onto the
          reference grid.  Standard practice — the reference is left
          unmodified.
        * ``'comparison'``: interpolate reference onto comparison grid.
        * ``'union'``: sorted union of both grids (both are interpolated).
    relative_in_percent : bool
        If ``True`` and *mode* is ``'relative'``, multiply by 100.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If the datasets have no overlapping x-range.
    """
    x_ref = np.asarray(reference.x, dtype=float)
    y_ref = np.asarray(reference.y, dtype=float)
    x_cmp = np.asarray(comparison.x, dtype=float)
    y_cmp = np.asarray(comparison.y, dtype=float)

    # Overlapping range
    x_lo = max(x_ref[0], x_cmp[0])
    x_hi = min(x_ref[-1], x_cmp[-1])

    if x_lo >= x_hi:
        raise ValueError(
            f"No overlapping x-range. "
            f"Reference: [{x_ref[0]:.6e}, {x_ref[-1]:.6e}], "
            f"Comparison: [{x_cmp[0]:.6e}, {x_cmp[-1]:.6e}]"
        )

    # Build common grid
    if grid == 'reference':
        common_x = x_ref[(x_ref >= x_lo) & (x_ref <= x_hi)]
    elif grid == 'comparison':
        common_x = x_cmp[(x_cmp >= x_lo) & (x_cmp <= x_hi)]
    elif grid == 'union':
        both = np.concatenate([
            x_ref[(x_ref >= x_lo) & (x_ref <= x_hi)],
            x_cmp[(x_cmp >= x_lo) & (x_cmp <= x_hi)],
        ])
        common_x = np.unique(both)
    else:
        raise ValueError(f"Unknown grid option: {grid!r}")

    # Interpolate both onto common grid
    y_ref_interp = interpolate_to_grid(
        common_x, x_ref, y_ref, method=interpolation,
    )
    y_cmp_interp = interpolate_to_grid(
        common_x, x_cmp, y_cmp, method=interpolation,
    )

    # Valid mask: both must be finite
    valid = np.isfinite(y_ref_interp) & np.isfinite(y_cmp_interp)

    # Compute difference
    diff = np.full_like(common_x, np.nan)
    if mode == 'relative':
        nonzero = valid & (np.abs(y_ref_interp) > 0)
        diff[nonzero] = (
            (y_cmp_interp[nonzero] - y_ref_interp[nonzero])
            / y_ref_interp[nonzero]
        )
        if relative_in_percent:
            diff *= 100.0
    elif mode == 'absolute':
        diff[valid] = y_cmp_interp[valid] - y_ref_interp[valid]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Build output PlotData objects preserving original styling
    ref_out = PlotData(
        x=common_x, y=y_ref_interp,
        label=reference.label,
        color=reference.color,
        linestyle=reference.linestyle,
        linewidth=reference.linewidth,
        plot_type=reference.plot_type,
        metadata={**reference.metadata, 'interpolated': grid != 'reference'},
    )
    cmp_out = PlotData(
        x=common_x, y=y_cmp_interp,
        label=comparison.label,
        color=comparison.color,
        linestyle=comparison.linestyle,
        linewidth=comparison.linewidth,
        plot_type=comparison.plot_type,
        metadata={**comparison.metadata, 'interpolated': grid != 'comparison'},
    )

    diff_out = DifferencePlotData(
        x=common_x,
        y=diff,
        difference_type=mode,
        reference_label=reference.label,
        comparison_label=comparison.label,
    )

    return ComparisonResult(
        reference=ref_out,
        comparison=cmp_out,
        difference=diff_out,
        common_x=common_x,
        valid_mask=valid,
    )


# ---------------------------------------------------------------------------
# ComparisonBuilder
# ---------------------------------------------------------------------------

class ComparisonBuilder:
    """
    Builder for comparison plots with an optional difference panel.

    Creates figures in two layouts:

    1. **Single-panel** – overlay of reference + comparisons.
    2. **Dual-panel** – main overlay on top, difference panel below
       (enabled via :meth:`set_difference_panel`).

    Examples
    --------
    >>> fig = (ComparisonBuilder()
    ...     .set_reference(ref_data)
    ...     .add_comparison(cmp_data)
    ...     .set_difference_panel(mode='relative')
    ...     .set_labels(title='Elastic XS', x_label='Energy (MeV)',
    ...                 y_label='Cross Section (b)')
    ...     .set_scales(log_x=True, log_y=True)
    ...     .build())
    """

    def __init__(
        self,
        style: str = 'light',
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 100,
        font_family: str = 'serif',
        notebook_mode: Optional[bool] = None,
        interactive: Optional[bool] = None,
        interpolation: Optional[str] = None,
        grid_strategy: Literal['reference', 'comparison', 'union'] = 'reference',
    ):
        self._style = style
        self._figsize = figsize
        self._dpi = dpi
        self._font_family = font_family
        self._notebook_mode = notebook_mode
        self._interactive = interactive
        self._interpolation = interpolation
        self._grid_strategy = grid_strategy

        # Data
        self._reference: Optional[PlotData] = None
        self._reference_styling: dict = {}
        self._comparisons: List[Tuple[PlotData, dict]] = []

        # Difference-panel config
        self._show_diff_panel: bool = False
        self._diff_only: bool = False
        self._diff_mode: str = 'relative'
        self._diff_y_label: Optional[str] = None
        self._diff_y_lim: Optional[Tuple[float, float]] = None
        self._relative_in_percent: bool = True
        self._height_ratios: Tuple[float, float] = (3.0, 1.0)
        self._zero_line: bool = True
        self._diff_log_y: bool = False

        # Shared settings forwarded to PlotBuilder instances
        self._title = _NOT_SET
        self._x_label: Optional[str] = None
        self._y_label: Optional[str] = None
        self._use_log_x: bool = False
        self._use_log_y: bool = False
        self._x_lim: Optional[Tuple[float, float]] = None
        self._y_lim: Optional[Tuple[float, float]] = None
        self._legend_loc: str = 'best'
        self._legend_ncol: Optional[int] = None
        self._grid: bool = True

    # ---- fluent API -------------------------------------------------------

    def set_reference(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Set the reference (baseline) dataset."""
        self._reference = data
        self._reference_styling = styling
        return self

    def add_comparison(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add a comparison dataset."""
        self._comparisons.append((data, styling))
        return self

    def set_difference_panel(
        self,
        mode: Literal['relative', 'absolute'] = _NOT_SET,
        y_label: Optional[str] = _NOT_SET,
        y_lim: Optional[Tuple[float, float]] = _NOT_SET,
        height_ratios: Tuple[float, float] = _NOT_SET,
        relative_in_percent: bool = _NOT_SET,
        zero_line: bool = _NOT_SET,
        only: bool = _NOT_SET,
        log_y: bool = _NOT_SET,
    ) -> 'ComparisonBuilder':
        """
        Enable and configure the difference sub-panel.

        Can be called multiple times — only the parameters you pass are
        updated; everything else keeps its previous value.

        Parameters
        ----------
        mode : {'relative', 'absolute'}
            Difference type (default ``'relative'``).
        y_label : str, optional
            Custom y-axis label for the diff panel.
        y_lim : tuple, optional
            ``(min, max)`` y-axis limits for the diff panel.
        height_ratios : tuple
            ``(main, diff)`` height ratio (default ``(3.0, 1.0)``).
        relative_in_percent : bool
            Multiply relative differences by 100 (default ``True``).
        zero_line : bool
            Draw a dashed zero reference line (default ``True``).
        only : bool
            If ``True``, show *only* the difference panel (no main overlay).
        log_y : bool
            Use logarithmic y-axis on the difference panel (default ``False``).
        """
        self._show_diff_panel = True
        if mode is not _NOT_SET:
            self._diff_mode = mode
        if y_label is not _NOT_SET:
            self._diff_y_label = y_label
        if y_lim is not _NOT_SET:
            self._diff_y_lim = y_lim
        if height_ratios is not _NOT_SET:
            self._height_ratios = height_ratios
        if relative_in_percent is not _NOT_SET:
            self._relative_in_percent = relative_in_percent
        if zero_line is not _NOT_SET:
            self._zero_line = zero_line
        if only is not _NOT_SET:
            self._diff_only = only
        if log_y is not _NOT_SET:
            self._diff_log_y = log_y
        return self

    def set_labels(
        self,
        title=_NOT_SET,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> 'ComparisonBuilder':
        """Set plot labels."""
        if title is not _NOT_SET:
            self._title = title
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label
        return self

    def set_scales(
        self, log_x: bool = False, log_y: bool = False,
    ) -> 'ComparisonBuilder':
        """Set axis scales."""
        self._use_log_x = log_x
        self._use_log_y = log_y
        return self

    def set_limits(
        self,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
    ) -> 'ComparisonBuilder':
        """Set axis limits for the main panel."""
        self._x_lim = x_lim
        self._y_lim = y_lim
        return self

    def set_legend(
        self, loc: str = 'best', ncol: Optional[int] = None,
    ) -> 'ComparisonBuilder':
        """Set legend placement."""
        self._legend_loc = loc
        self._legend_ncol = ncol
        return self

    def set_grid(self, grid: bool = True) -> 'ComparisonBuilder':
        """Enable or disable grid."""
        self._grid = grid
        return self

    # ---- interpolation inference ------------------------------------------

    def _infer_interpolation(self, data: PlotData) -> str:
        """Infer interpolation method from the PlotData subclass type."""
        class_name = type(data).__name__
        return _INTERPOLATION_DEFAULTS.get(class_name, 'log-log')

    # ---- build ------------------------------------------------------------

    def build(self, show: bool = False) -> plt.Figure:
        """
        Build and return the comparison figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._reference is None:
            raise ValueError(
                "No reference data set. Call set_reference() first."
            )
        if not self._comparisons:
            raise ValueError(
                "No comparison data. Call add_comparison() at least once."
            )

        # Resolve interpolation: explicit value wins, otherwise infer
        interpolation = self._interpolation
        if interpolation is None:
            interpolation = self._infer_interpolation(self._reference)

        # Pre-compute differences if the panel is requested
        results: List[ComparisonResult] = []
        if self._show_diff_panel:
            for cmp_data, _ in self._comparisons:
                result = compute_difference(
                    reference=self._reference,
                    comparison=cmp_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid=self._grid_strategy,
                    relative_in_percent=self._relative_in_percent,
                )
                results.append(result)

        if self._show_diff_panel and self._diff_only:
            return self._build_diff_only_panel(results, show)
        elif self._show_diff_panel:
            return self._build_dual_panel(results, show)
        else:
            return self._build_single_panel(show)

    # ---- internal ---------------------------------------------------------

    def _resolve_diff_y_label(self) -> str:
        """Build the default y-axis label for difference panels."""
        if self._diff_y_label is not None:
            return self._diff_y_label
        if self._diff_mode == 'relative':
            return (
                'Relative Diff (%)'
                if self._relative_in_percent
                else 'Relative Diff'
            )
        return 'Absolute Diff'

    def _build_diff_only_panel(
        self, results: List[ComparisonResult], show: bool,
    ) -> plt.Figure:
        """Single-panel figure showing only difference curves."""
        builder = PlotBuilder(
            style=self._style,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
            notebook_mode=self._notebook_mode,
            interactive=self._interactive,
        )

        colors = _get_color_palette(self._style)
        for i, (result, (cmp_data, _)) in enumerate(
            zip(results, self._comparisons)
        ):
            diff_data = result.difference
            # Use the short comparison label, not the verbose auto-label
            diff_data.label = cmp_data.label
            color_idx = (i + 1) % len(colors)
            builder.add_data(diff_data, color=colors[color_idx])

        builder.set_labels(
            title=self._title,
            x_label=self._x_label,
            y_label=self._resolve_diff_y_label(),
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        builder.set_limits(x_lim=self._x_lim, y_lim=self._diff_y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(grid=self._grid)

        fig = builder.build(show=False)
        ax = fig.axes[0]

        # Zero reference line
        if self._zero_line:
            ax.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        # Reference annotation
        ref_label = self._reference.label or 'reference'
        ax.text(
            0.02, 0.97, f'ref: {ref_label}',
            transform=ax.transAxes,
            fontsize=11, va='top', ha='left',
            fontstyle='italic', alpha=0.7,
        )

        if show:
            plt.show()

        return fig

    def _build_single_panel(self, show: bool) -> plt.Figure:
        """Single-panel overlay plot."""
        builder = PlotBuilder(
            style=self._style,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
            notebook_mode=self._notebook_mode,
            interactive=self._interactive,
        )

        builder.add_data(self._reference, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            builder.add_data(cmp_data, **styling)

        builder.set_labels(
            title=self._title, x_label=self._x_label, y_label=self._y_label,
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(grid=self._grid)

        return builder.build(show=show)

    def _build_dual_panel(
        self, results: List[ComparisonResult], show: bool,
    ) -> plt.Figure:
        """Dual-panel figure: main overlay + difference panel."""
        # Resolve notebook / interactive settings
        notebook = (
            self._notebook_mode if self._notebook_mode is not None
            else _is_notebook()
        )
        interactive = self._interactive
        if interactive is None and notebook:
            interactive = _detect_interactive_backend()

        # Apply global style
        _apply_style_to_rcparams(
            style=self._style,
            notebook_mode=notebook,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
        )

        fig, (ax_main, ax_diff) = plt.subplots(
            nrows=2, ncols=1,
            figsize=self._figsize,
            dpi=self._dpi,
            gridspec_kw={
                'height_ratios': list(self._height_ratios),
                'hspace': 0.05,
            },
            sharex=True,
        )

        if notebook and interactive:
            _configure_figure_interactivity(fig, interactive)

        # --- Main panel via PlotBuilder on existing axes ---
        main_builder = PlotBuilder(
            style=self._style, ax=ax_main,
            font_family=self._font_family,
            notebook_mode=notebook,
        )
        main_builder.add_data(self._reference, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            main_builder.add_data(cmp_data, **styling)
        main_builder.set_labels(title=self._title, y_label=self._y_label)
        main_builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        main_builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        main_builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        main_builder.set_grid(grid=self._grid)
        main_builder.build()

        # Hide x-axis labels on main panel (shared with diff panel)
        ax_main.set_xlabel('')
        ax_main.tick_params(axis='x', labelbottom=False)

        # --- Difference panel ---
        diff_builder = PlotBuilder(
            style=self._style, ax=ax_diff,
            font_family=self._font_family,
            notebook_mode=notebook,
        )

        colors = _get_color_palette(self._style)
        for i, result in enumerate(results):
            # Comparison colors start at index 1 (index 0 is reference)
            color_idx = (i + 1) % len(colors)
            diff_styling = {'color': colors[color_idx]}
            diff_data = result.difference
            # Suppress verbose diff labels — colors match the main panel
            diff_data.label = None
            diff_builder.add_data(diff_data, **diff_styling)

        diff_builder.set_labels(
            y_label=self._resolve_diff_y_label(), x_label=self._x_label,
        )
        diff_builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        diff_builder.set_limits(x_lim=self._x_lim, y_lim=self._diff_y_lim)
        diff_builder.set_grid(grid=self._grid)
        diff_builder.build()

        # Zero reference line
        if self._zero_line:
            ax_diff.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        # Reference annotation — colors match the main panel legend
        ref_label = self._reference.label or 'reference'
        ax_diff.text(
            0.02, 0.97, f'ref: {ref_label}',
            transform=ax_diff.transAxes,
            fontsize=11, va='top', ha='left',
            fontstyle='italic', alpha=0.7,
        )

        if show:
            plt.show()

        return fig
