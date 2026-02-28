"""
Plotting styles for kika.

This module defines the available plotting styles ('light' and 'dark')
and provides functions to apply them to matplotlib figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Optional, Tuple

from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _setup_notebook_backend,
    _configure_figure_interactivity
)


def _get_color_palette(style: str) -> list[str]:
    """
    Get the color palette for a given style.
    
    Parameters
    ----------
    style : str
        Style name ('light' or 'dark')
        
    Returns
    -------
    list of str
        List of color hex codes
    """
    if style == 'light':
        # Color-blind friendly palette that works well in print
        return [
            '#0173B2', '#DE8F05', '#029E73', '#D55E00',
            '#CC78BC', '#CA9161', '#FBAFE4', '#949494',
            '#ECE133', '#56B4E9'
        ]
    else:  # 'dark' or default
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        return prop_cycle.by_key()['color']


def _get_linestyles() -> list:
    """
    Get the default linestyle cycle.
    
    Returns
    -------
    list
        List of linestyles for cycling
    """
    return ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def _apply_style_to_rcparams(
    style: str,
    notebook_mode: bool,
    figsize: Tuple[float, float],
    dpi: int,
    font_family: str,
    projection: Optional[str] = None
) -> None:
    """
    Apply style-specific settings to matplotlib rcParams.
    
    Parameters
    ----------
    style : str
        Style name ('light' or 'dark')
    notebook_mode : bool
        Whether running in notebook mode
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Dots per inch for figure resolution
    font_family : str
        Font family for text elements
    projection : str, optional
        Matplotlib projection type (e.g., '3d' for 3D plots)
    """
    # Reset matplotlib settings to avoid style contamination
    plt.close('all')
    mpl.rcdefaults()
    plt.style.use('default')
    
    # For 3D plots, avoid constrained_layout (it doesn't work well with 3D)
    # Enable for all 2D plots to prevent label overlap and cutoff
    use_constrained_layout = (projection != '3d')
    
    # Base settings (common to all styles)
    base_settings = {
        'font.family': font_family,
        'font.size': 11 if notebook_mode else 12,
        'axes.labelsize': 12 if notebook_mode else 14,
        'axes.titlesize': 13 if notebook_mode else 14,
        'xtick.labelsize': 10 if notebook_mode else 12,
        'ytick.labelsize': 10 if notebook_mode else 12,
        'legend.fontsize': 10 if notebook_mode else 12,
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.linewidth': 1.0 if notebook_mode else 1.2,
        'lines.linewidth': 1.0 if notebook_mode else 1.2,
        'lines.markersize': 6 if notebook_mode else 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.major.width': 1.0 if notebook_mode else 1.2,
        'ytick.major.width': 1.0 if notebook_mode else 1.2,
        'xtick.minor.width': 0.8 if notebook_mode else 1.0,
        'ytick.minor.width': 0.8 if notebook_mode else 1.0,
        'xtick.major.size': 4.0 if notebook_mode else 5.0,
        'ytick.major.size': 4.0 if notebook_mode else 5.0,
        'xtick.minor.size': 2.5 if notebook_mode else 3.0,
        'ytick.minor.size': 2.5 if notebook_mode else 3.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.constrained_layout.use': use_constrained_layout,
    }
    
    # Style-specific settings
    if style == 'dark':
        style_settings = {
            'axes.facecolor': 'black',
            'figure.facecolor': 'black',
            'savefig.facecolor': 'black',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
        }
    else:  # 'light' (default)
        style_settings = {
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
        }
    
    # Apply all settings
    plt.rcParams.update(base_settings)
    plt.rcParams.update(style_settings)


def _adjust_figsize_for_notebook(
    figsize: Tuple[float, float],
    interactive: bool
) -> Tuple[float, float]:
    """
    Adjust figure size for notebook environments.
    
    Parameters
    ----------
    figsize : tuple
        Original figure size (width, height) in inches
    interactive : bool
        Whether using interactive backend
        
    Returns
    -------
    tuple
        Adjusted figure size
    """
    # More aggressive figure size reduction for notebooks
    if figsize[0] > 10 or figsize[1] > 7:
        scale = min(8/figsize[0], 5/figsize[1])
        return (figsize[0] * scale, figsize[1] * scale)
    elif figsize[0] > 8 or figsize[1] > 6:
        scale = min(7/figsize[0], 5/figsize[1])
        return (figsize[0] * scale, figsize[1] * scale)
    return figsize


def _adjust_dpi_for_notebook(dpi: int, interactive: bool) -> int:
    """
    Adjust DPI for notebook environments.
    
    Parameters
    ----------
    dpi : int
        Original DPI value
    interactive : bool
        Whether using interactive backend
        
    Returns
    -------
    int
        Adjusted DPI value
    """
    if dpi > 120:
        return 90 if interactive else 120
    return dpi


def format_energy_axis_ticks(ax: plt.Axes) -> None:
    """
    Format the ticks on a log-scale energy axis to ensure readability.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes with log-scale energy x-axis
    """
    # Get current x-axis limits
    xmin, xmax = ax.get_xlim()
    
    # Determine appropriate major tick locations based on range
    if xmax / xmin > 1e6:
        # Very wide energy range - use order of magnitude ticks
        major_locator = mpl.ticker.LogLocator(base=10.0, numticks=10)
    else:
        # Narrower range - use more frequent ticks
        major_locator = mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
    
    # Set major ticks
    ax.xaxis.set_major_locator(major_locator)
    
    # Use a formatter that shows enough precision for energy values
    formatter = mpl.ticker.FuncFormatter(lambda x, pos: f"{x:.3g}")
    ax.xaxis.set_major_formatter(formatter)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))


def format_axes(
    ax: plt.Axes,
    style: str = 'light',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    use_log_scale: bool = False,
    is_energy_axis: bool = False,
    use_y_log_scale: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> plt.Axes:
    """
    Format axes with labels, title, legend, scales, and limits.
    
    This is a centralized function to apply consistent formatting to plot axes.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to format
    style : str
        Style name ('light' or 'dark')
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    title : str, optional
        Plot title (suppressed for 'paper' and 'publication' styles)
    legend_loc : str, optional
        Legend location ('best', 'upper right', etc.)
    use_log_scale : bool
        Whether to use logarithmic scale for x-axis
    is_energy_axis : bool
        Whether x-axis is an energy axis (applies special tick formatting)
    use_y_log_scale : bool
        Whether to use logarithmic scale for y-axis
    x_min, x_max : float, optional
        X-axis limits
    y_min, y_max : float, optional
        Y-axis limits
        
    Returns
    -------
    plt.Axes
        Formatted axes object
    """
    # Set axis scales
    if use_log_scale:
        ax.set_xscale('log')
        if is_energy_axis:
            format_energy_axis_ticks(ax)
    
    if use_y_log_scale:
        ax.set_yscale('log')
    
    # Set labels
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    
    # Set title (suppress for paper/publication styles)
    if title and style not in ('paper', 'publication'):
        ax.set_title(title)
    
    # Set limits
    if x_min is not None or x_max is not None:
        current_xlim = ax.get_xlim()
        ax.set_xlim(
            x_min if x_min is not None else current_xlim[0],
            x_max if x_max is not None else current_xlim[1]
        )
    
    if y_min is not None or y_max is not None:
        current_ylim = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_ylim[0],
            y_max if y_max is not None else current_ylim[1]
        )
    
    # Add legend if requested
    if legend_loc:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Style-specific legend formatting
            if style == 'dark':
                ax.legend(
                    loc=legend_loc,
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    facecolor='black',
                    edgecolor='white'
                )
            else:
                ax.legend(
                    loc=legend_loc,
                    frameon=True,
                    fancybox=True,
                    shadow=False
                )
    
    # Grid configuration
    ax.grid(True, alpha=0.3, linestyle='--', which='major')
    if use_log_scale or use_y_log_scale:
        ax.grid(True, alpha=0.15, linestyle=':', which='minor')
    
    return ax


def setup_plot_style(style: str = 'light', figsize: Tuple[float, float] = (8, 6), ax: Optional[plt.Axes] = None):
    """
    Legacy function for backward compatibility with old plotting code.
    
    This function creates a figure and axes with the specified style settings.
    For new code, use PlotBuilder instead.
    
    Parameters
    ----------
    style : str
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple
        Figure size in inches (width, height)
    ax : plt.Axes, optional
        Existing axes to use. If provided, no new figure is created.
        
    Returns
    -------
    dict
        Dictionary with 'fig' and 'ax' keys
    """
    if ax is not None:
        return {'fig': ax.figure, 'ax': ax}
    
    # Apply style using the internal function with default parameters
    _apply_style_to_rcparams(
        style=style,
        notebook_mode=_is_notebook(),
        figsize=figsize,
        dpi=100,
        font_family='sans-serif'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    return {'fig': fig, 'ax': ax}


def finalize_plot(fig: plt.Figure, ax: plt.Axes, show: bool = True):
    """
    Legacy function for backward compatibility with old plotting code.
    
    Finalizes a plot by adjusting layout and optionally displaying it.
    For new code, use PlotBuilder instead.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to finalize
    ax : plt.Axes
        Axes to finalize
    show : bool
        Whether to call plt.show()
    """
    fig.tight_layout()
    if show:
        plt.show()
