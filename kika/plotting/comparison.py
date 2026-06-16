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
    'MultigroupCrossSectionPlotData': 'lin-lin',
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
        self._overlays: List[Tuple[PlotData, dict]] = []
        self._scatter_overlays: List[Tuple[PlotData, dict]] = []

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
        self._grid_alpha: float = 0.3
        self._show_minor_grid: bool = False
        self._minor_grid_alpha: float = 0.15

        # Resonance-region group-average overlay. When set, each
        # PlotData whose metadata contains a 'group_average_overlay'
        # dict (edges, xs, bounds_used, weighting) participates in
        # the averaged rendering:
        #   'pointwise' — no overlay drawn; diff panel still shows
        #       the averaged bin trace inside [bounds_used].
        #   'average'   — pointwise masked inside [bounds_used] on
        #       the main panel; averaged step overlay drawn in that
        #       range.
        #   'both'      — pointwise drawn across the full range, with
        #       averaged step overlay on top inside [bounds_used].
        self._main_display: Literal['pointwise', 'average', 'both'] = 'both'
        # "ref: <label>" annotation on the diff and diff-only panels.
        self._show_reference_label: bool = True
        self._reference_label_fontsize: float = 11

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

    def add_overlay(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add overlay data to the main panel only.

        Overlays are rendered on the main panel but are NOT included in
        difference computations.
        """
        self._overlays.append((data, styling))
        return self

    def add_scatter_overlay(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add scatter overlay data (e.g., EXFOR) to both main and diff panels.

        Scatter overlays are rendered on the main panel AND their difference
        against the reference is shown as scatter points in the diff panel.
        """
        self._scatter_overlays.append((data, styling))
        return self

    def set_difference_panel(
        self,
        mode: Literal['relative', 'absolute'] = _NOT_SET,
        y_label: Optional[str] = _NOT_SET,
        y_lim: Optional[Tuple[Optional[float], Optional[float]]] = _NOT_SET,
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
        x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
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

    def set_grid(
        self,
        grid: bool = True,
        alpha: float = 0.3,
        show_minor: bool = False,
        minor_alpha: float = 0.15,
    ) -> 'ComparisonBuilder':
        """Configure grid display settings for reference and comparison panels."""
        self._grid = grid
        self._grid_alpha = alpha
        self._show_minor_grid = show_minor
        self._minor_grid_alpha = minor_alpha
        return self

    def set_group_average(
        self,
        main_display: Literal['pointwise', 'average', 'both'] = 'both',
    ) -> 'ComparisonBuilder':
        """Configure resonance-region group-average rendering.

        The overlay data itself is attached per-series via
        ``PlotData.metadata['group_average_overlay']`` (a dict with keys
        ``edges``, ``xs``, ``bounds_used``, ``weighting``). The
        ``main_display`` argument controls what the main panel shows
        inside the averaged energy window:

        * ``'pointwise'`` — only pointwise curves, no averaged overlay.
        * ``'average'`` — pointwise masked inside the window, averaged
          step overlay rendered there; pointwise continues outside.
        * ``'both'`` — pointwise drawn across the full range, averaged
          step overlay drawn on top inside the window.

        The diff panel always renders the bin-averaged diff trace inside
        the window when both reference and comparison carry overlays —
        that is the whole point of this comparison mode.
        """
        self._main_display = main_display
        return self

    def set_reference_label(
        self, show: bool = True, fontsize: Optional[float] = None
    ) -> 'ComparisonBuilder':
        """Toggle the 'ref: <label>' annotation on the diff panel.

        ``fontsize`` scales the annotation; pass the legend fontsize so the
        label tracks it. ``None`` keeps the current value.
        """
        self._show_reference_label = show
        if fontsize is not None:
            self._reference_label_fontsize = fontsize
        return self

    # ---- interpolation inference ------------------------------------------

    def _infer_interpolation(self, data: PlotData) -> str:
        """Infer interpolation method from the PlotData subclass type."""
        class_name = type(data).__name__
        return _INTERPOLATION_DEFAULTS.get(class_name, 'log-log')

    # ---- group-average overlay helpers ------------------------------------

    @staticmethod
    def _overlay_from(data: PlotData) -> Optional[dict]:
        """Return the ``group_average_overlay`` payload from a series, or None."""
        overlay = data.metadata.get('group_average_overlay') if data.metadata else None
        if not overlay:
            return None
        edges = overlay.get('edges')
        xs = overlay.get('xs')
        bounds = overlay.get('bounds_used')
        if edges is None or xs is None or bounds is None:
            return None
        return overlay

    def _draw_main_overlay(
        self, ax, data: PlotData, color: Optional[str],
    ) -> bool:
        """Draw the dashed step-post overlay on the main panel.

        Returns True if an overlay was actually drawn (so the caller
        can refresh the legend to pick up the new labeled line).
        """
        if self._main_display == 'pointwise':
            return False
        overlay = self._overlay_from(data)
        if overlay is None:
            return False
        edges = np.asarray(overlay['edges'], dtype=float)
        xs = np.asarray(overlay['xs'], dtype=float)
        if edges.size < 2 or xs.size == 0:
            return False
        # steps-post needs one extra y to match edges length; repeat last.
        y_step = np.concatenate([xs, xs[-1:]])
        line_color = color or data.color
        series_label = data.label or ''
        # In 'average' mode the pointwise trace's legend entry is
        # suppressed (see _pointwise_mask_for_main), so the step trace
        # represents the whole series and reuses the original label
        # without an "(avg)" suffix. In 'both' mode the pointwise still
        # appears in the legend, so the suffix distinguishes the two.
        if self._main_display == 'average':
            avg_label = series_label or None
        else:
            avg_label = f'{series_label} (avg)' if series_label else None
        ax.plot(
            edges, y_step,
            drawstyle='steps-post',
            linestyle='--',
            linewidth=(data.linewidth or 1.5),
            color=line_color,
            label=avg_label,
            alpha=0.95,
        )
        return True

    def _pointwise_mask_for_main(self, data: PlotData) -> Optional[PlotData]:
        """Return a copy of ``data`` with pointwise y masked inside
        the averaging window, or None when no mask is needed.

        Used in ``main_display='average'`` mode so the only thing
        rendered inside [Elow, Ehigh] is the averaged step overlay.
        The first and last masked y-values are bridged to the averaged
        step's leading and trailing bin values so the pointwise line
        visually meets the step trace at the boundaries instead of
        dropping out into a NaN gap.

        The returned copy also has ``label = None`` so the pointwise
        trace is excluded from the legend — the averaged step trace
        added by :meth:`_draw_main_overlay` carries the series label
        instead. Without this the legend lists both lines and
        ``'average'`` looks indistinguishable from ``'both'``.
        """
        if self._main_display != 'average':
            return None
        overlay = self._overlay_from(data)
        if overlay is None:
            return None

        import copy as _copy
        masked = _copy.copy(data)
        masked.metadata = dict(data.metadata)
        masked.metadata.pop('group_average_overlay', None)
        masked.label = None

        lo, hi = float(overlay['bounds_used'][0]), float(overlay['bounds_used'][1])
        xs = np.asarray(overlay.get('xs', []), dtype=float)
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)
        in_range = (x >= lo) & (x <= hi)
        if not np.any(in_range):
            # No points to mask, but we still return the copy so the
            # label suppression takes effect (avoids a duplicate legend
            # entry when the step trace draws with the series label).
            masked.x = x
            masked.y = y
            return masked
        idx = np.where(in_range)[0]
        first_in, last_in = int(idx[0]), int(idx[-1])
        new_y = y.copy()
        new_y[first_in:last_in + 1] = np.nan
        # Bridge to the step trace's leading / trailing values.
        if xs.size > 0 and np.isfinite(xs[0]):
            new_y[first_in] = float(xs[0])
        if last_in > first_in and xs.size > 0 and np.isfinite(xs[-1]):
            new_y[last_in] = float(xs[-1])

        masked.x = x
        masked.y = new_y
        return masked

    @staticmethod
    def _mask_pointwise_in_range(
        diff_data: DifferencePlotData, lo: float, hi: float,
        bridge_values: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Replace diff values inside [lo, hi] with NaN so the averaged
        step trace can occupy that region without visual overlap.

        ``bridge_values=(left, right)`` overrides the first/last masked
        y so the pointwise line has real (non-NaN) endpoints at the
        range boundaries. Without bridges, matplotlib drops the last
        segment before NaN and the first segment after NaN, leaving a
        visible gap between the pointwise and the step trace. The
        bridge values should be the averaged-step's leading and
        trailing bin values so the pointwise line visually meets the
        step exactly at the boundary.
        """
        x = np.asarray(diff_data.x, dtype=float)
        y = np.asarray(diff_data.y, dtype=float)
        in_range = (x >= lo) & (x <= hi)
        if not np.any(in_range):
            return
        idx = np.where(in_range)[0]
        first_in, last_in = int(idx[0]), int(idx[-1])
        y = y.copy()
        y[first_in:last_in + 1] = np.nan
        if bridge_values is not None:
            left, right = bridge_values
            if np.isfinite(left):
                y[first_in] = float(left)
            if last_in > first_in and np.isfinite(right):
                y[last_in] = float(right)
        diff_data.y = y

    def _compute_overlay_diff(
        self, ref_overlay: dict, cmp_overlay: dict,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
        """Return (edges, diff_values, bounds) for bin-averaged diff, or None.

        Both series are expected to share edges because the frontend
        uses a single Elow/Ehigh/nBins/weighting config. If edges
        mismatch (defensive guard), return None so the caller can
        silently fall back to pointwise-only diff in that range.
        """
        ref_edges = np.asarray(ref_overlay['edges'], dtype=float)
        cmp_edges = np.asarray(cmp_overlay['edges'], dtype=float)
        if ref_edges.shape != cmp_edges.shape or not np.allclose(ref_edges, cmp_edges):
            return None
        ref_xs = np.asarray(ref_overlay['xs'], dtype=float)
        cmp_xs = np.asarray(cmp_overlay['xs'], dtype=float)
        if ref_xs.shape != cmp_xs.shape:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            if self._diff_mode == 'relative':
                nonzero = np.abs(ref_xs) > 0
                diff = np.full_like(ref_xs, np.nan)
                diff[nonzero] = (cmp_xs[nonzero] - ref_xs[nonzero]) / ref_xs[nonzero]
                if self._relative_in_percent:
                    diff *= 100.0
            else:
                diff = cmp_xs - ref_xs
        lo = float(min(ref_overlay['bounds_used'][0], cmp_overlay['bounds_used'][0]))
        hi = float(max(ref_overlay['bounds_used'][1], cmp_overlay['bounds_used'][1]))
        return ref_edges, diff, (lo, hi)

    def _draw_overlay_diff(
        self, ax, edges: np.ndarray, diff_values: np.ndarray, color: Optional[str],
        linewidth: float,
    ) -> None:
        """Render a bin-averaged step-post diff trace on the diff panel.

        Drawn solid (distinct from the dashed main-panel overlay) so the
        diff panel reads as a continuous diff curve with the averaged
        segment visually stitched into the pointwise segments.
        """
        if edges.size < 2 or diff_values.size == 0:
            return
        y_step = np.concatenate([diff_values, diff_values[-1:]])
        ax.plot(
            edges, y_step,
            drawstyle='steps-post',
            linestyle='-',
            linewidth=linewidth or 1.5,
            color=color,
            label=None,
            alpha=1.0,
        )

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
        if not self._comparisons and not self._scatter_overlays:
            raise ValueError(
                "No comparison or scatter overlay data. "
                "Call add_comparison() or add_scatter_overlay() at least once."
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
            return self._build_diff_only_panel(results, show, interpolation)
        elif self._show_diff_panel:
            return self._build_dual_panel(results, show, interpolation)
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
        interpolation: str = 'log-log',
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
        overlay_diff_draws: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        ref_overlay = self._overlay_from(self._reference)
        for i, (result, (cmp_data, _)) in enumerate(
            zip(results, self._comparisons)
        ):
            diff_data = result.difference
            # Use per-series diff_label from metadata if provided:
            #   not present → use comparison label (default)
            #   empty string → suppress label (None)
            #   non-empty → use as-is
            raw_diff_label = cmp_data.metadata.get('diff_label')
            if raw_diff_label is None:
                diff_data.label = cmp_data.label
            else:
                diff_data.label = raw_diff_label or None
            color_idx = (i + 1) % len(colors)
            diff_color = cmp_data.color if cmp_data.color else colors[color_idx]

            cmp_overlay = self._overlay_from(cmp_data)
            if ref_overlay is not None and cmp_overlay is not None:
                overlay_diff = self._compute_overlay_diff(ref_overlay, cmp_overlay)
                if overlay_diff is not None:
                    edges, diff_values, (lo, hi) = overlay_diff
                    bridge = (
                        float(diff_values[0]) if diff_values.size > 0 else float('nan'),
                        float(diff_values[-1]) if diff_values.size > 0 else float('nan'),
                    )
                    self._mask_pointwise_in_range(diff_data, lo, hi, bridge_values=bridge)
                    overlay_diff_draws.append(
                        (edges, diff_values, diff_color, cmp_data.linewidth or 1.5)
                    )

            builder.add_data(
                diff_data, color=diff_color, linewidth=cmp_data.linewidth or 1.5
            )

        # Add scatter overlays to diff-only panel
        for ovl_data, ovl_styling in self._scatter_overlays:
            try:
                ovl_result = compute_difference(
                    reference=self._reference,
                    comparison=ovl_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid='comparison',
                    relative_in_percent=self._relative_in_percent,
                )
                ovl_diff = ovl_result.difference
                ovl_diff.label = ovl_data.label
                ovl_diff_styling = {
                    'color': ovl_data.color or ovl_styling.get('color'),
                    'marker': ovl_data.marker or ovl_styling.get('marker', 'o'),
                    'markersize': ovl_data.markersize or ovl_styling.get('markersize', 5),
                    'linestyle': 'none',
                }
                builder.add_data(ovl_diff, **ovl_diff_styling)
            except Exception:
                pass

        builder.set_labels(
            title=self._title,
            x_label=self._x_label,
            y_label=self._resolve_diff_y_label(),
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        builder.set_limits(x_lim=self._x_lim, y_lim=self._diff_y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
        )

        fig = builder.build(show=False)
        ax = fig.axes[0]

        # Group-average bin-diff traces (step-post) on top of the
        # masked pointwise diff.
        for edges, diff_values, color, lw in overlay_diff_draws:
            self._draw_overlay_diff(ax, edges, diff_values, color, lw)

        # Zero reference line
        if self._zero_line:
            ax.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        # Reference annotation (diff-only panel)
        if self._show_reference_label:
            ref_label = self._reference.label or 'reference'
            ax.text(
                0.02, 0.97, f'ref: {ref_label}',
                transform=ax.transAxes,
                fontsize=self._reference_label_fontsize, va='top', ha='left',
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

        ref_for_main = self._pointwise_mask_for_main(self._reference) or self._reference
        builder.add_data(ref_for_main, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            cmp_for_main = self._pointwise_mask_for_main(cmp_data) or cmp_data
            builder.add_data(cmp_for_main, **styling)
        for ovl_data, ovl_styling in self._overlays:
            builder.add_data(ovl_data, **ovl_styling)
        for ovl_data, ovl_styling in self._scatter_overlays:
            builder.add_data(ovl_data, **ovl_styling)

        builder.set_labels(
            title=self._title, x_label=self._x_label, y_label=self._y_label,
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
        )

        fig = builder.build(show=False)
        ax = fig.axes[0]
        overlay_drawn = self._draw_main_overlay(ax, self._reference, self._reference.color)
        for cmp_data, _styling in self._comparisons:
            if self._draw_main_overlay(ax, cmp_data, cmp_data.color):
                overlay_drawn = True
        if overlay_drawn:
            _existing_legend = ax.get_legend()
            if _existing_legend is not None:
                _existing_legend.remove()
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_kwargs = {'loc': self._legend_loc}
                if self._legend_ncol:
                    legend_kwargs['ncol'] = self._legend_ncol
                ax.legend(handles, labels, **legend_kwargs)

        if show:
            plt.show()
        return fig

    def _build_dual_panel(
        self, results: List[ComparisonResult], show: bool,
        interpolation: str = 'log-log',
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
        # In 'average' display mode, add masked copies of reference and
        # comparisons so the pointwise trace is blank inside the
        # averaging window — the step overlay fills that region.
        ref_for_main = self._pointwise_mask_for_main(self._reference) or self._reference
        main_builder.add_data(ref_for_main, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            cmp_for_main = self._pointwise_mask_for_main(cmp_data) or cmp_data
            main_builder.add_data(cmp_for_main, **styling)
        for ovl_data, ovl_styling in self._overlays:
            main_builder.add_data(ovl_data, **ovl_styling)
        for ovl_data, ovl_styling in self._scatter_overlays:
            main_builder.add_data(ovl_data, **ovl_styling)
        main_builder.set_labels(title=self._title, y_label=self._y_label)
        main_builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        main_builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        main_builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        main_builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
        )
        main_builder.build()

        # Group-average overlays on main panel (dashed step-post, same
        # color as the pointwise trace, labeled "<series> (avg)").
        overlay_drawn = False
        if self._draw_main_overlay(ax_main, self._reference, self._reference.color):
            overlay_drawn = True
        for cmp_data, _styling in self._comparisons:
            if self._draw_main_overlay(ax_main, cmp_data, cmp_data.color):
                overlay_drawn = True

        # Refresh the legend so the overlay entries appear.
        if overlay_drawn:
            _existing_legend = ax_main.get_legend()
            if _existing_legend is not None:
                _existing_legend.remove()
            handles, labels = ax_main.get_legend_handles_labels()
            if handles:
                legend_kwargs = {'loc': self._legend_loc}
                if self._legend_ncol:
                    legend_kwargs['ncol'] = self._legend_ncol
                ax_main.legend(handles, labels, **legend_kwargs)

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
        # Track overlay diffs to draw after the builder renders the
        # pointwise traces — this way the step trace sits on top of
        # the masked pointwise and shares the same axis limits.
        overlay_diff_draws: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        ref_overlay = self._overlay_from(self._reference)
        for i, (result, (cmp_data, _)) in enumerate(
            zip(results, self._comparisons)
        ):
            # Use the comparison series' own color if specified; fall back to palette
            color_idx = (i + 1) % len(colors)
            diff_color = cmp_data.color if cmp_data.color else colors[color_idx]
            diff_styling = {'color': diff_color, 'linewidth': cmp_data.linewidth or 1.5}
            diff_data = result.difference
            # In dual-panel mode: labels are suppressed by default (colors match main panel)
            # but a per-series diff_label in metadata overrides this
            raw_diff_label = cmp_data.metadata.get('diff_label')
            if raw_diff_label is None:
                diff_data.label = None  # Default: suppress in dual-panel
            else:
                diff_data.label = raw_diff_label or None

            # If both reference and this comparison carry group-average
            # overlays, compute the bin-averaged diff, mask the
            # pointwise diff inside the bounds window, and queue the
            # averaged-step trace to draw over the masked region.
            cmp_overlay = self._overlay_from(cmp_data)
            if ref_overlay is not None and cmp_overlay is not None:
                overlay_diff = self._compute_overlay_diff(ref_overlay, cmp_overlay)
                if overlay_diff is not None:
                    edges, diff_values, (lo, hi) = overlay_diff
                    bridge = (
                        float(diff_values[0]) if diff_values.size > 0 else float('nan'),
                        float(diff_values[-1]) if diff_values.size > 0 else float('nan'),
                    )
                    self._mask_pointwise_in_range(diff_data, lo, hi, bridge_values=bridge)
                    overlay_diff_draws.append(
                        (edges, diff_values, diff_color, cmp_data.linewidth or 1.5)
                    )

            diff_builder.add_data(diff_data, **diff_styling)

        # Add scatter overlays to diff panel (e.g., EXFOR experimental data)
        for ovl_data, ovl_styling in self._scatter_overlays:
            try:
                ovl_result = compute_difference(
                    reference=self._reference,
                    comparison=ovl_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid='comparison',  # Keep scatter x-points
                    relative_in_percent=self._relative_in_percent,
                )
                ovl_diff = ovl_result.difference
                ovl_diff.label = None  # Suppress legend in diff panel
                # Preserve scatter marker style
                ovl_diff_styling = {
                    'color': ovl_data.color or ovl_styling.get('color'),
                    'marker': ovl_data.marker or ovl_styling.get('marker', 'o'),
                    'markersize': ovl_data.markersize or ovl_styling.get('markersize', 5),
                    'linestyle': 'none',
                }
                diff_builder.add_data(ovl_diff, **ovl_diff_styling)
            except Exception:
                pass  # Skip if interpolation fails for this overlay

        diff_builder.set_labels(
            y_label=self._resolve_diff_y_label(), x_label=self._x_label,
        )
        diff_builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        diff_builder.set_limits(x_lim=self._x_lim, y_lim=self._diff_y_lim)
        diff_builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
        )
        diff_builder.build()

        # Group-average bin-diff traces (step-post) rendered after the
        # pointwise diff so they sit on top of the NaN-masked gaps.
        for edges, diff_values, color, lw in overlay_diff_draws:
            self._draw_overlay_diff(ax_diff, edges, diff_values, color, lw)

        # Remove legend from diff panel — colors match the main panel
        _diff_legend = ax_diff.get_legend()
        if _diff_legend:
            _diff_legend.remove()

        # Zero reference line
        if self._zero_line:
            ax_diff.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        # Reference annotation — colors match the main panel legend
        if self._show_reference_label:
            ref_label = self._reference.label or 'reference'
            ax_diff.text(
                0.02, 0.97, f'ref: {ref_label}',
                transform=ax_diff.transAxes,
                fontsize=self._reference_label_fontsize, va='top', ha='left',
                fontstyle='italic', alpha=0.7,
            )

        if show:
            plt.show()

        return fig
