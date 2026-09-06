"""
Plotting functions for multigroup covariance data and Legendre coefficients.

This module provides plotting capabilities for MultigroupLegendreCovariance objects,
including visualization of multigroup Legendre coefficients and comparison
with continuous ENDF data.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import TwoSlopeNorm
from typing import List, Optional, Union, Tuple, Dict, Any, TYPE_CHECKING
from kika.plotting.styles import setup_plot_style, format_axes
from ..legendre_covariance import LegendreCovariance

if TYPE_CHECKING:
    from .mg_legendre_covariance import MultigroupLegendreCovariance


def _generate_unique_labels(mg_list, labels):
    """
    Generate unique labels for multigroup objects, handling duplicates properly.
    
    Parameters
    ----------
    mg_list : list
        List of MultigroupLegendreCovariance objects
    labels : None, str, or list of str
        Label specification
        
    Returns
    -------
    list of str
        Unique labels for each object
    """
    if labels is None:
        # Generate default labels based on isotope info if available
        mg_labels = [f"MG Data {i+1}" for i in range(len(mg_list))]
    elif isinstance(labels, str):
        # Single string - create unique labels with indices
        if len(mg_list) == 1:
            mg_labels = [labels]
        else:
            mg_labels = [f"{labels} {i+1}" for i in range(len(mg_list))]
    else:
        # List of labels - pad with defaults if needed
        mg_labels = list(labels)
        while len(mg_labels) < len(mg_list):
            mg_labels.append(f"MG Data {len(mg_labels)+1}")
    
    return mg_labels


def _get_energy_group_centers(energy_grid: np.ndarray) -> np.ndarray:
    """
    Calculate energy group centers from group boundaries.
    
    Parameters
    ----------
    energy_grid : np.ndarray
        Energy group boundaries (length n+1 for n groups)
        
    Returns
    -------
    np.ndarray
        Energy group centers (length n)
    """
    return np.sqrt(energy_grid[:-1] * energy_grid[1:])  # Geometric mean


def plot_mg_legendre_coefficients(
    mg_covmat: Union["MultigroupLegendreCovariance", List["MultigroupLegendreCovariance"]],
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (10, 6),
    legend_loc: str = 'best',
    marker: bool = True,
    uncertainty: bool = False,
    sigma: float = 1.0,
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot multigroup Legendre coefficients from MultigroupLegendreCovariance object(s).
    
    This function plots the group-averaged Legendre coefficients as a function
    of energy group centers, optionally including uncertainty bands.
    
    Parameters
    ----------
    mg_covmat : MultigroupLegendreCovariance or list of MultigroupLegendreCovariance
        Multigroup covariance matrix object(s) containing Legendre coefficients
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    energy_range : tuple(float, float), optional
        If provided, (emin, emax) limits for x-axis. Values are clamped to data range.
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    marker : bool
        Whether to include markers on the plot lines
    uncertainty : bool
        Whether to include uncertainty bands if available
    sigma : float
        Number of sigma levels for uncertainty bands
    labels : str or list of str, optional
        Labels for each dataset
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
    """
    # Convert single object to list for uniform processing
    if not isinstance(mg_covmat, list):
        mg_list = [mg_covmat]
    else:
        mg_list = mg_covmat
    
    # Generate labels
    mg_labels = _generate_unique_labels(mg_list, labels)
    
    # Validate inputs
    for i, mg_obj in enumerate(mg_list):
        if mg_obj.num_groups == 0:
            raise ValueError(f"MultigroupLegendreCovariance object {i+1} has no energy groups")
    
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Determine which orders to plot
    first_mg = mg_list[0]
    available_orders = []
    for key in first_mg.legendre_coefficients.keys():
        iso_key, mt_key, l_key = key
        if iso_key == isotope and mt_key == mt:
            available_orders.append(l_key)
    
    if not available_orders:
        ax.text(0.5, 0.5, f'No Legendre coefficient data available\nfor isotope {isotope}, MT {mt}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"No Data Found for Isotope {isotope}, MT {mt}")
        return fig
    
    available_orders = sorted(set(available_orders))
    
    if orders is None:
        orders = available_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_orders]
    
    if not orders:
        ax.text(0.5, 0.5, f'Requested orders not available\nAvailable orders: {available_orders}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Requested Orders Not Available")
        return fig
    
    from kika._utils import zaid_to_symbol

    # Track all energy boundaries for applying global limits
    all_energy_bounds: List[float] = []

    # Plot data for each multigroup object
    for mg_idx, (mg_obj, mg_label) in enumerate(zip(mg_list, mg_labels)):
        # Energy group boundaries (length n+1) and coefficients (length n)
        energy_grid = np.asarray(mg_obj.energy_grid, dtype=float)
        if energy_grid.ndim != 1 or energy_grid.size < 2:
            continue
        all_energy_bounds.extend(energy_grid.tolist())

        # Plot each requested order
        for order_idx, order in enumerate(orders):
            key = (isotope, mt, order)
            if key not in mg_obj.legendre_coefficients:
                continue

            coeffs = np.asarray(mg_obj.legendre_coefficients[key], dtype=float)
            if coeffs.size != energy_grid.size - 1:
                # Size mismatch; skip to avoid plotting errors
                continue

            # Color by order (primary) then dataset variation if many datasets
            base_color_idx = order_idx % len(colors)
            # Slight variation for different datasets of same order (cycle linestyle only)
            color = colors[base_color_idx]

            linestyle = '-' if len(mg_list) == 1 else ['-', '--', '-.', ':'][mg_idx % 4]

            # Build isotope symbol using ZAID conversion (isotope is ZAID)
            try:
                isotope_symbol = zaid_to_symbol(isotope)
            except Exception:
                isotope_symbol = f"Isotope {isotope}"

            # Base label (without uncertainty)
            base_label = f"{isotope_symbol} - $a_{{{order}}}$"
            # If uncertainties requested: include sigma text on the LINE label (not on fill)
            if uncertainty and getattr(mg_obj, 'relative_matrices', None):
                if sigma == 1.0:
                    base_label += " ±1σ"
                else:
                    base_label += f" ±{sigma}σ"
            line_label = base_label if len(mg_list) == 1 else f"{base_label} ({mg_label})"

            # Prepare step arrays: for where='post', need y length n+1
            coeff_steps = np.append(coeffs, coeffs[-1])

            # Plot step line
            line_kwargs = {
                'color': color,
                'linestyle': linestyle,
                'label': line_label,
                'where': 'post',
                'zorder': 3
            }
            if marker:
                # Plot markers at bin centers separately (optional)
                centers = np.sqrt(energy_grid[:-1] * energy_grid[1:])
                ax.plot(centers, coeffs, 'o', color=color, markersize=4, zorder=4)
            ax.step(energy_grid, coeff_steps, **line_kwargs)

            # Uncertainty bands (no label; legend handled by line label above)
            if uncertainty and getattr(mg_obj, 'relative_matrices', None):
                _plot_mg_uncertainty_bands(
                    ax=ax,
                    mg_obj=mg_obj,
                    isotope=isotope,
                    mt=mt,
                    order=order,
                    energy_grid=energy_grid,
                    coeffs=coeffs,
                    sigma=sigma,
                    color=color,
                    label=None
                )

    # Apply energy range limits if specified
    if energy_range is not None and all_energy_bounds:
        data_min = min(all_energy_bounds)
        data_max = max(all_energy_bounds)
        e_min = max(energy_range[0], data_min)
        e_max = min(energy_range[1], data_max)
        if e_min < e_max:
            ax.set_xlim(e_min, e_max)
    
    # Create an improved title with isotope and reaction information
    title_parts = []
    
    # Add reaction information (import constants if available)
    try:
        from kika._constants import MT_TO_REACTION
        if mt in MT_TO_REACTION:
            reaction_name = MT_TO_REACTION[mt]
            title_parts.append(f"MT={mt} {reaction_name}")
        else:
            title_parts.append(f"MT={mt}")
    except ImportError:
        title_parts.append(f"MT={mt}")
    
    # Add isotope information (convert ZAID to symbol)
    try:
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"
    title_parts.append(isotope_symbol)
    
    # Add "Multigroup Legendre Coefficients" at the end
    title_parts.append("Multigroup Legendre Coefficients")
    
    title = " - ".join(title_parts)
    
    # Format axes using the same style as continuous plotting
    ax = format_axes(
        ax,
        style=style,
        use_log_scale=True,
        is_energy_axis=True,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=title,
        legend_loc=legend_loc
    )

    return fig


def _plot_mg_uncertainty_bands(
    ax,
    mg_obj,
    isotope: int,
    mt: int,
    order: int,
    energy_grid: np.ndarray,
    coeffs: np.ndarray,
    sigma: float,
    color,
    label: Optional[str] = None,
):
    """Plot step-style uncertainty bands for a given order.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes
    mg_obj : MultigroupLegendreCovariance
        Multigroup covariance object
    isotope, mt, order : int
        Selection identifiers
    energy_grid : np.ndarray
        Bin boundaries (length n+1)
    coeffs : np.ndarray
        Coefficient values (length n)
    sigma : float
        Sigma multiplier
    color : matplotlib color
        Band color (matches line color)
    label : str, optional
        Legend label for the band
    """
    try:
        # Locate matching diagonal covariance block
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(
            zip(
                mg_obj.isotope_rows,
                mg_obj.reaction_rows,
                mg_obj.l_rows,
                mg_obj.isotope_cols,
                mg_obj.reaction_cols,
                mg_obj.l_cols,
            )
        ):
            if (
                iso_r == isotope
                and mt_r == mt
                and l_r == order
                and iso_c == isotope
                and mt_c == mt
                and l_c == order
            ):
                cov_matrix = mg_obj.relative_matrices[i]
                diag = np.diag(cov_matrix)
                if diag.size != coeffs.size:
                    break
                rel_unc = np.sqrt(diag) * sigma
                abs_unc = rel_unc * np.abs(coeffs)
                upper = coeffs + abs_unc
                lower = coeffs - abs_unc
                # Extend for step representation
                upper_steps = np.append(upper, upper[-1])
                lower_steps = np.append(lower, lower[-1])
                ax.fill_between(
                    energy_grid,
                    lower_steps,
                    upper_steps,
                    step='post',
                    alpha=0.18,
                    color=color,
                    edgecolor='none',
                    label=label,
                    zorder=2,
                )
                break
    except Exception:
        # Silent fail – uncertainty bands are optional
        pass


def plot_mg_vs_endf_comparison(
    mg_covmat: "MultigroupLegendreCovariance",
    endf: object,
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (12, 8),
    legend_loc: str = 'best',
    mg_marker: bool = True,
    endf_native: bool = False,
    uncertainty: bool = False,
    sigma: float = 1.0,
    **kwargs
) -> plt.Figure:
    """
    Compare multigroup Legendre coefficients with ENDF data.
    
    This function plots both the multigroup averaged coefficients and the
    coefficients from the original ENDF data for comparison.
    
    Parameters
    ----------
    mg_covmat : MultigroupLegendreCovariance
        Multigroup covariance matrix object
    endf : ENDF object
        Original ENDF data object containing MF4 data
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range for plotting ENDF data
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    mg_marker : bool
        Whether to include markers for multigroup data
    endf_native : bool
        If True, plot the ENDF data (and its uncertainty band) using the native
        Legendre coefficient energy mesh from the MF4 file instead of a dense interpolated grid.
        This mirrors the appearance of plot_legendre_coefficients_from_endf and ensures that
        the uncertainty band is centered exactly on the plotted line. If False (default), a
        dense log-spaced energy grid is used for the line for smooth appearance; when
        include_uncertainties=True the band will still be computed on native energies and may
        appear slightly coarser or visually offset if interpolation introduces differences.
    uncertainty : bool
        If True and covariance data (MF34 / multigroup relative matrices) are available, show ±σ bands.
    sigma : float
        Number of sigma for uncertainty bands (default 1.0)
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
    """
    warnings.warn(
        "plot_mg_vs_endf_uncertainties_comparison is deprecated; prefer building PlotBuilder figures from to_plot_data outputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    warnings.warn(
        "plot_mg_vs_endf_comparison is deprecated; prefer building PlotBuilder figures from to_plot_data outputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Validate inputs
    if 4 not in endf.mf:
        raise ValueError("ENDF object does not contain MF4 data")
    if mt not in endf.mf[4].mt:
        raise ValueError(f"MT {mt} not found in ENDF MF4 data")
    
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    # If only one order is requested we will later use two distinct colors (if available)
    
    # Multigroup energy grid (boundaries) and centers
    mg_energy_grid = np.asarray(mg_covmat.energy_grid, dtype=float)
    mg_energy_centers = _get_energy_group_centers(mg_energy_grid)
    
    # Determine which orders to plot
    available_mg_orders = []
    for key in mg_covmat.legendre_coefficients.keys():
        iso_key, mt_key, l_key = key
        if iso_key == isotope and mt_key == mt:
            available_mg_orders.append(l_key)
    
    available_mg_orders = sorted(set(available_mg_orders))
    
    if orders is None:
        orders = available_mg_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_mg_orders]
    
    if not orders:
        ax.text(0.5, 0.5, f'No matching orders available\nMG orders: {available_mg_orders}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Get continuous ENDF data
    mf4_data = endf.mf[4].mt[mt]
    
    # Check if ENDF data has extract_legendre_coefficients method
    if not hasattr(mf4_data, 'extract_legendre_coefficients'):
        ax.text(0.5, 0.5, 'ENDF data does not support coefficient extraction\n'
                          'Object type may not have Legendre coefficients', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create energy grid for ENDF plotting
    if energy_range is None:
        energy_range = (mg_covmat.energy_grid[0], mg_covmat.energy_grid[-1])
    
    # Prepare dense energy grid (may be overridden by native usage)
    log_e_min = np.log10(energy_range[0])
    log_e_max = np.log10(energy_range[1])
    dense_endf_energies = np.logspace(log_e_min, log_e_max, 1000)
    
    from kika._utils import zaid_to_symbol
    try:
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"

    # Attempt to access MF34 covariance for ENDF uncertainties once if needed
    mf34_covmat = None
    if uncertainty and 34 in endf.mf and mt in endf.mf[34].mt:
        try:
            mf34_covmat = endf.mf[34].mt[mt].to_ang_covmat()
        except Exception:
            mf34_covmat = None  # Silent fail
    # The band plotter, which works on native ENDF energies. It lives in
    # kika/plotting since phase 4's P4: it draws a LegendreCovariance and names
    # no ENDF type, so reaching it in the format package meant reaching through
    # a leading underscore across a package boundary.
    if uncertainty and mf34_covmat is not None:
        try:
            from kika.plotting.covariance import plot_legendre_uncertainty_bands as _plot_endf_uncertainty_bands
        except Exception:
            _plot_endf_uncertainty_bands = None

    # Flag for single-order special formatting (distinct colors for MG vs ENDF)
    single_order_mode = len(orders) == 1

    # Simple color adjustment helper (only used if we need a second color but palette has one)
    from matplotlib import colors as mcolors
    def _lighten_color(c, factor=0.6):
        try:
            r, g, b = mcolors.to_rgb(c)
            return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b
        except Exception:
            return c

    # Pre-compute colors for single-order case
    if single_order_mode:
        mg_color = colors[0]
        if len(colors) > 1:
            endf_color = colors[1]
        else:
            endf_color = _lighten_color(mg_color, 0.4)

    # Plot each order (multigroup as step, ENDF as solid; special case if single order)
    for order_idx, order in enumerate(orders):
        if not single_order_mode:
            color = colors[order_idx % len(colors)]  # same color for MG & ENDF (original behavior)
            mg_color = color
            endf_color = color

        mg_key = (isotope, mt, order)
        if mg_key in mg_covmat.legendre_coefficients:
            mg_coeffs = np.asarray(mg_covmat.legendre_coefficients[mg_key], dtype=float)
            if mg_coeffs.size == mg_energy_grid.size - 1:
                mg_steps = np.append(mg_coeffs, mg_coeffs[-1])
                # Create label with uncertainty notation if uncertainties are enabled
                line_label = f"MG {isotope_symbol} - $a_{{{order}}}$"
                if uncertainty:
                    line_label += f" ±{sigma}σ"
                ax.step(
                    mg_energy_grid,
                    mg_steps,
                    where='post',
                    color=mg_color,
                    linewidth=2,
                    label=line_label,
                    zorder=3
                )
                if mg_marker:
                    centers = mg_energy_centers
                    ax.plot(centers, mg_coeffs, 'o', color=mg_color, markersize=4, zorder=4)
                # MG uncertainties from relative matrices (diagonal) - using dashed outline
                if uncertainty:
                    try:
                        # Find matching relative matrix for diagonal uncertainties
                        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
                                mg_covmat.isotope_rows,
                                mg_covmat.reaction_rows,
                                mg_covmat.l_rows,
                                mg_covmat.isotope_cols,
                                mg_covmat.reaction_cols,
                                mg_covmat.l_cols)):
                            if (iso_r == isotope and mt_r == mt and l_r == order and
                                    iso_c == isotope and mt_c == mt and l_c == order):
                                matrix = mg_covmat.relative_matrices[i]
                                diag = np.diag(matrix)
                                if diag.size == mg_coeffs.size:
                                    rel_unc = np.sqrt(diag) * sigma
                                    abs_unc = rel_unc * np.abs(mg_coeffs)
                                    upper = mg_coeffs + abs_unc
                                    lower = mg_coeffs - abs_unc
                                    upper_steps = np.append(upper, upper[-1])
                                    lower_steps = np.append(lower, lower[-1])
                                    # Plot upper and lower bounds as thin dashed lines
                                    ax.step(
                                        mg_energy_grid,
                                        upper_steps,
                                        where='post',
                                        linestyle='--',
                                        linewidth=0.8,
                                        color=mg_color,
                                        alpha=0.7,
                                        zorder=2,
                                    )
                                    ax.step(
                                        mg_energy_grid,
                                        lower_steps,
                                        where='post',
                                        linestyle='--',
                                        linewidth=0.8,
                                        color=mg_color,
                                        alpha=0.7,
                                        zorder=2,
                                    )
                                break
                    except Exception:
                        pass

        # Always plot ENDF line for ENDF data
        # Decide whether to use native energies (recommended when uncertainties shown)
        use_native_for_line = endf_native or (uncertainty and mf34_covmat is not None)
        try:
            if use_native_for_line:
                # Build coefficient arrays directly from native MF4 data
                # Note: legendre_coefficients typically store a1..aNL per energy (a0 implicit)
                native_energies = getattr(mf4_data, 'legendre_energies', [])
                native_coeff_lists = getattr(mf4_data, 'legendre_coefficients', [])
                energy_values_native: List[float] = []
                coeff_values_native: List[float] = []
                for j, coeff_list in enumerate(native_coeff_lists):
                    if order == 0:
                        coeff_values_native.append(1.0)
                        energy_values_native.append(native_energies[j])
                    elif (order - 1) < len(coeff_list):
                        coeff_values_native.append(coeff_list[order - 1])
                        energy_values_native.append(native_energies[j])
                if energy_values_native:
                    # Label only in single-order mode (to differentiate colors)
                    endf_label = None
                    if single_order_mode:
                        endf_label = f"ENDF {isotope_symbol} - $a_{{{order}}}$"
                    ax.plot(
                        energy_values_native,
                        coeff_values_native,
                        linestyle='-',  # Always solid line
                        color=endf_color,
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=2,
                        label=endf_label
                    )
                    # Uncertainty band exactly aligned with line
                    if uncertainty and mf34_covmat is not None and '_plot_endf_uncertainty_bands' in locals() and _plot_endf_uncertainty_bands is not None:
                        try:
                            isotope_id = int(getattr(mf4_data, 'zaid', isotope))
                            _plot_endf_uncertainty_bands(
                                ax,
                                energy_values_native,
                                coeff_values_native,
                                mf34_covmat,
                                isotope_id,
                                mt,
                                order,
                                sigma,
                                endf_color,
                                alpha=0.15
                            )
                        except Exception:
                            pass
            else:
                # Dense interpolation path (no or deferred uncertainties)
                endf_coeffs_dict = mf4_data.extract_legendre_coefficients(
                    dense_endf_energies,
                    max_legendre_order=max(orders),
                    out_of_range="hold"
                )
                if order in endf_coeffs_dict:
                    endf_coeffs = endf_coeffs_dict[order]
                    endf_label = None
                    if single_order_mode:
                        endf_label = f"ENDF {isotope_symbol} - $a_{{{order}}}$"
                    ax.plot(
                        dense_endf_energies,
                        endf_coeffs,
                        linestyle='-',  # Always solid line
                        color=endf_color,
                        linewidth=1.4,
                        alpha=0.75,
                        zorder=2,
                        label=endf_label
                    )
                    # If uncertainties requested, still draw band using native energies (may look coarser)
                    if uncertainty and mf34_covmat is not None and '_plot_endf_uncertainty_bands' in locals() and _plot_endf_uncertainty_bands is not None:
                        try:
                            native_energies = getattr(mf4_data, 'legendre_energies', [])
                            native_coeff_lists = getattr(mf4_data, 'legendre_coefficients', [])
                            coeff_values_native = []
                            energy_values_native = []
                            for j, coeff_list in enumerate(native_coeff_lists):
                                if order == 0:
                                    coeff_values_native.append(1.0)
                                    energy_values_native.append(native_energies[j])
                                elif (order - 1) < len(coeff_list):
                                    coeff_values_native.append(coeff_list[order - 1])
                                    energy_values_native.append(native_energies[j])
                            if energy_values_native:
                                isotope_id = int(getattr(mf4_data, 'zaid', isotope))
                                _plot_endf_uncertainty_bands(
                                    ax,
                                    energy_values_native,
                                    coeff_values_native,
                                    mf34_covmat,
                                    isotope_id,
                                    mt,
                                    order,
                                    sigma,
                                    endf_color,
                                    alpha=0.15
                                )
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning: Could not extract ENDF coefficients for order {order}: {e}")
    
    # Format the plot
    # Improved title with reaction name if available
    title_parts = []
    try:
        from kika._constants import MT_TO_REACTION
        if mt in MT_TO_REACTION:
            title_parts.append(f"MT={mt} {MT_TO_REACTION[mt]}")
        else:
            title_parts.append(f"MT={mt}")
    except ImportError:
        title_parts.append(f"MT={mt}")
    title_parts.append(isotope_symbol)
    title_parts.append("MG vs ENDF Legendre Coefficients")
    title = " - ".join(title_parts)

    ax = format_axes(
        ax,
        style=style,
        use_log_scale=True,
        is_energy_axis=True,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=title,
        legend_loc=legend_loc
    )
    ax.set_xlim(energy_range)
    
    # Manually compute y-limits based on visible data
    if energy_range is not None:
        e_min, e_max = energy_range
        y_values_in_range = []
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            # Filter to visible x-range
            mask = (xdata >= e_min) & (xdata <= e_max)
            if np.any(mask):
                y_values_in_range.extend(ydata[mask])
        
        if y_values_in_range:
            y_min = np.min(y_values_in_range)
            y_max = np.max(y_values_in_range)
            # Add 5% margin
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    return fig


def plot_mg_vs_endf_uncertainties_comparison(
    mg_covmat: "MultigroupLegendreCovariance",
    endf_data: Union["LegendreCovariance", object],
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (10, 6),
    legend_loc: str = 'best',
    mg_marker: bool = True,
    uncertainty_type: str = "relative",
    **kwargs
) -> plt.Figure:
    """
    Compare multigroup uncertainties with ENDF uncertainties for Legendre coefficients.
    
    This function plots both the multigroup uncertainties and the uncertainties
    from the original ENDF MF34 covariance data for comparison using step plots.
    
    Parameters
    ----------
    mg_covmat : MultigroupLegendreCovariance
        Multigroup covariance matrix object
    endf_data : LegendreCovariance or ENDF object
        Either:
        - LegendreCovariance: Original ENDF MF34 covariance data object
        - ENDF object: ENDF object containing MF34 data (will extract LegendreCovariance automatically)
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range for plotting. If None, uses full multigroup range
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    mg_marker : bool
        Whether to include markers for multigroup data
    uncertainty_type : str
        Type of uncertainty to plot: "relative" (%) or "absolute"
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
    """
    # Handle endf_data parameter - convert to LegendreCovariance if needed and capture MF4 for absolute uncertainties
    endf_mf4_mt = None
    if hasattr(endf_data, 'mf') and 34 in endf_data.mf:
        # This is an ENDF object
        if mt in endf_data.mf[34].mt:
            endf_mf34_covmat = endf_data.mf[34].mt[mt].to_ang_covmat()
        else:
            raise ValueError(f"MT {mt} not found in ENDF MF34 data")
        # Capture MF4 MT object (may not exist for some files)
        if 4 in endf_data.mf and mt in endf_data.mf[4].mt:
            endf_mf4_mt = endf_data.mf[4].mt[mt]
    else:
        # Assume this is already a LegendreCovariance object (no MF4 info for absolute uncertainties)
        endf_mf34_covmat = endf_data
    
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Multigroup energy grid (boundaries) and centers
    mg_energy_grid = np.asarray(mg_covmat.energy_grid, dtype=float)
    mg_energy_centers = _get_energy_group_centers(mg_energy_grid)
    
    # Determine which orders to plot
    available_mg_orders = []
    for key in mg_covmat.legendre_coefficients.keys():
        iso_key, mt_key, l_key = key
        if iso_key == isotope and mt_key == mt:
            available_mg_orders.append(l_key)
    
    available_mg_orders = sorted(set(available_mg_orders))
    
    if orders is None:
        orders = available_mg_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_mg_orders]
    
    if not orders:
        ax.text(0.5, 0.5, f'No matching orders available\nMG orders: {available_mg_orders}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Set energy range
    user_supplied_energy_range = energy_range is not None

    # Pre-scan ENDF covariance data to collect true bin boundaries for requested orders
    # so we can extend default energy range to include full ENDF coverage.
    def _find_endf_boundaries_for_order(l_order: int) -> Optional[np.ndarray]:
        """Return the raw energy bin boundaries (as stored in MF34) for a diagonal block."""
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, grid) in enumerate(zip(
            endf_mf34_covmat.isotope_rows, endf_mf34_covmat.reaction_rows, endf_mf34_covmat.l_rows,
            endf_mf34_covmat.isotope_cols, endf_mf34_covmat.reaction_cols, endf_mf34_covmat.l_cols,
            endf_mf34_covmat.energy_grids
        )):
            if (iso_r == isotope and iso_c == isotope and mt_r == mt and mt_c == mt and l_r == l_order and l_c == l_order):
                try:
                    return np.asarray(grid, dtype=float)
                except Exception:
                    return None
        return None

    mg_min = float(mg_covmat.energy_grid[0]) if len(mg_covmat.energy_grid) else None
    mg_max = float(mg_covmat.energy_grid[-1]) if len(mg_covmat.energy_grid) else None
    endf_min = None
    endf_max = None
    # We'll gather ENDF boundary extrema for the selected orders (after filtering below)
    
    from kika._utils import zaid_to_symbol
    try:
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"

    # Determine if single order for special formatting
    single_order_mode = len(orders) == 1
    from matplotlib import colors as mcolors
    def _lighten_color(c, factor=0.6):
        try:
            r, g, b = mcolors.to_rgb(c)
            return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b
        except Exception:
            return c
    if single_order_mode:
        mg_color = colors[0]
        if len(colors) > 1:
            endf_color = colors[1]
        else:
            endf_color = _lighten_color(mg_color, 0.4)

    # After determining single_order_mode we can finalize default energy range if not user supplied.
    # (We need the list of orders first; orders filtering happened above.)
    if not user_supplied_energy_range:
        for order in orders:
            boundaries = _find_endf_boundaries_for_order(order)
            if boundaries is not None and len(boundaries) > 0:
                bmin = float(boundaries[0]); bmax = float(boundaries[-1])
                endf_min = bmin if endf_min is None else min(endf_min, bmin)
                endf_max = bmax if endf_max is None else max(endf_max, bmax)
        # Compute union of MG and ENDF ranges (ignore None cases gracefully)
        all_mins = [x for x in [mg_min, endf_min] if x is not None]
        all_maxs = [x for x in [mg_max, endf_max] if x is not None]
        if all_mins and all_maxs:
            energy_range = (min(all_mins), max(all_maxs))
        elif mg_min is not None and mg_max is not None:
            energy_range = (mg_min, mg_max)
        elif endf_min is not None and endf_max is not None:
            energy_range = (endf_min, endf_max)
        else:
            energy_range = (1e-5, 1.0)  # fallback minimal sensible range

    # Plot each order
    for order_idx, order in enumerate(orders):
        if not single_order_mode:
            color = colors[order_idx % len(colors)]
            mg_color = color
            endf_color = color

    # Plot multigroup uncertainties (dashed step plot unless single-order mode where solid retained)
        mg_key = (isotope, mt, order)
        if mg_key in mg_covmat.legendre_coefficients:
            # Get multigroup uncertainties from diagonal covariance matrix
            mg_uncertainties = None
            for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
                    mg_covmat.isotope_rows,
                    mg_covmat.reaction_rows,
                    mg_covmat.l_rows,
                    mg_covmat.isotope_cols,
                    mg_covmat.reaction_cols,
                    mg_covmat.l_cols)):
                if (iso_r == isotope and mt_r == mt and l_r == order and
                        iso_c == isotope and mt_c == mt and l_c == order):
                    matrix = mg_covmat.relative_matrices[i]
                    diag = np.diag(matrix)
                    mg_uncertainties = np.sqrt(diag)  # Relative uncertainties
                    if uncertainty_type == "relative":
                        mg_uncertainties = mg_uncertainties * 100  # Convert to percentage
                    elif uncertainty_type == "absolute":
                        # Convert to absolute using multigroup coefficients
                        mg_coeffs = np.asarray(mg_covmat.legendre_coefficients[mg_key], dtype=float)
                        mg_uncertainties = mg_uncertainties * np.abs(mg_coeffs)
                    break
            
            if mg_uncertainties is not None:
                # Create step plot for multigroup data (dashed)
                mg_steps = np.append(mg_uncertainties, mg_uncertainties[-1])
                ax.step(
                    mg_energy_grid,
                    mg_steps,
                    where='post',
                    color=mg_color,
                    linewidth=2,
                    linestyle='-' if single_order_mode else '--',  # solid only if single order
                    label=(f"MG {isotope_symbol} - L={order}" if single_order_mode else f"MG {isotope_symbol} - L={order}"),
                    zorder=3
                )
                if mg_marker:
                    ax.plot(mg_energy_centers, mg_uncertainties, 'o', color=mg_color, markersize=4, zorder=4)

        # Plot ENDF uncertainties (solid step plot using true bin boundaries)
        endf_uncertainty_data = endf_mf34_covmat.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
        if endf_uncertainty_data is not None:
            endf_centers = np.asarray(endf_uncertainty_data['energies'], dtype=float)
            endf_unc = np.asarray(endf_uncertainty_data['uncertainties'], dtype=float)

            # Retrieve true bin boundaries
            boundaries = _find_endf_boundaries_for_order(order)
            if boundaries is not None and len(boundaries) == len(endf_unc) + 1:
                endf_boundaries = boundaries.astype(float)
            else:
                # Fallback: reconstruct boundaries approximately (maintain previous behaviour)
                if len(endf_centers) > 1:
                    log_e = np.log(endf_centers)
                    log_w = np.diff(log_e)
                    log_b = np.zeros(len(endf_centers) + 1)
                    log_b[1:-1] = (log_e[:-1] + log_e[1:]) / 2.0
                    log_b[0] = log_e[0] - (log_w[0] / 2.0 if len(log_w) else 0.1)
                    log_b[-1] = log_e[-1] + (log_w[-1] / 2.0 if len(log_w) else 0.1)
                    endf_boundaries = np.exp(log_b)
                else:
                    # Single point – create an arbitrary narrow boundary around it
                    e = endf_centers[0] if len(endf_centers) else 1.0
                    endf_boundaries = np.array([e * 0.9, e * 1.1])

            # Convert to absolute uncertainties if requested
            if uncertainty_type == "absolute":
                if endf_mf4_mt is None:
                    print(f"Warning: Cannot compute absolute ENDF uncertainties (missing MF4 for MT {mt}); showing relative instead.")
                else:
                    try:
                        native_energies = getattr(endf_mf4_mt, 'legendre_energies', [])
                        native_coeff_lists = getattr(endf_mf4_mt, 'legendre_coefficients', [])
                        coeff_vals = []
                        # Build coefficient values per uncertainty center
                        for center in endf_centers:
                            # Find closest native energy index
                            if not native_energies:
                                coeff_vals.append(1.0 if order == 0 else 0.0)
                                continue
                            idx_closest = int(np.argmin(np.abs(np.asarray(native_energies) - center)))
                            if order == 0:
                                coeff_vals.append(1.0)
                            else:
                                coeff_list = native_coeff_lists[idx_closest] if idx_closest < len(native_coeff_lists) else []
                                if (order - 1) < len(coeff_list):
                                    coeff_vals.append(coeff_list[order - 1])
                                else:
                                    coeff_vals.append(0.0)
                        coeff_vals = np.abs(np.asarray(coeff_vals, dtype=float))
                        endf_unc = endf_unc * coeff_vals
                    except Exception as _abs_err:
                        print(f"Warning: Failed converting ENDF uncertainties to absolute values for L={order}: {_abs_err}")
            else:
                # Relative percent conversion
                endf_unc = endf_unc * 100.0

            # Apply masking ONLY if user explicitly provided an energy range
            if user_supplied_energy_range:
                # Mask by bin centers
                mask = (endf_centers >= energy_range[0]) & (endf_centers <= energy_range[1])
                if not np.any(mask):
                    continue
                # Determine slice indices for boundaries
                idxs = np.where(mask)[0]
                i0, i1 = idxs[0], idxs[-1]
                plot_boundaries = endf_boundaries[i0:i1+2]  # +2 to include upper edge
                plot_unc = endf_unc[i0:i1+1]
            else:
                plot_boundaries = endf_boundaries
                plot_unc = endf_unc

            # Create step values
            endf_steps = np.append(plot_unc, plot_unc[-1])
            ax.step(
                plot_boundaries,
                endf_steps,
                where='post',
                linestyle='-',
                color=endf_color,
                linewidth=1.4,
                alpha=0.7,
                label=(f"ENDF {isotope_symbol} - L={order}" if single_order_mode else f"ENDF {isotope_symbol} - L={order}"),
                zorder=2
            )
    
    # Format the plot
    # Improved title with reaction name if available
    title_parts = []
    try:
        from kika._constants import MT_TO_REACTION
        if mt in MT_TO_REACTION:
            title_parts.append(f"MT={mt} {MT_TO_REACTION[mt]}")
        else:
            title_parts.append(f"MT={mt}")
    except ImportError:
        title_parts.append(f"MT={mt}")
    title_parts.append(isotope_symbol)
    title_parts.append("MG vs ENDF Uncertainties")
    title = " - ".join(title_parts)

    # Set y-axis label based on uncertainty type
    if uncertainty_type == "relative":
        y_label = "Relative uncertainty (%)"
    else:
        y_label = "Absolute uncertainty"

    ax = format_axes(
        ax,
        style=style,
        use_log_scale=True,
        is_energy_axis=True,
        x_label="Energy (eV)",
        y_label=y_label,
        title=title,
        legend_loc=legend_loc
    )
    ax.set_xlim(energy_range)
    
    # Manually compute y-limits based on visible data
    if energy_range is not None:
        e_min, e_max = energy_range
        y_values_in_range = []
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            # Filter to visible x-range
            mask = (xdata >= e_min) & (xdata <= e_max)
            if np.any(mask):
                y_values_in_range.extend(ydata[mask])
        
        if y_values_in_range:
            y_min = np.min(y_values_in_range)
            y_max = np.max(y_values_in_range)
            # Add 5% margin
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    return fig


def plot_mg_covariance_heatmap(
    mg_covmat: "MultigroupLegendreCovariance",
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    matrix_type: str = 'cov',
    covariance_type: str = 'rel',
    style: str = 'default',
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    colormap: Optional[str] = None,
    show_colorbar: bool = True,
    annotate: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_scale: bool = True,
    use_log_scale: Optional[bool] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a heatmap of the covariance or correlation matrix for specific isotope, MT, and Legendre orders.
    
    This function creates a heatmap visualization of either the covariance or correlation matrix
    for the specified isotope and MT reaction, optionally filtered by Legendre orders.
    
    Parameters
    ----------
    mg_covmat : MultigroupLegendreCovariance
        Multigroup covariance matrix object
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to include. If None, includes all available orders
    matrix_type : str, default 'cov'
        Type of matrix to plot: 'cov' for covariance, 'corr' for correlation
    covariance_type : str, default 'rel'
        Type of covariance matrix: 'rel' for relative, 'abs' for absolute
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    colormap : str, optional
        Matplotlib colormap name. If None, uses 'RdYlGn' for correlation and 'viridis' for covariance
    show_colorbar : bool
        Whether to show the colorbar
    annotate : bool
        Whether to annotate matrix values on the heatmap
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling  
    symmetric_scale : bool
        Whether to use symmetric color scale around zero (for correlation matrices)
    use_log_scale : bool, optional
        Whether to use logarithmic color scale. If None, uses linear for correlation and log for covariance
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap
    """
    # Validate inputs
    if matrix_type not in ['cov', 'corr']:
        raise ValueError("matrix_type must be 'cov' or 'corr'")
    if covariance_type not in ['rel', 'abs']:
        raise ValueError("covariance_type must be 'rel' or 'abs'")
    
    # Reset matplotlib to default state
    plt.rcdefaults()
    
    # Setup publication-quality settings (same as MF34 heatmap)
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.constrained_layout.use': False,
    })
    
    # Apply style customizations
    if style == 'dark':
        plt.rcParams.update({
            'axes.facecolor': 'black',
            'figure.facecolor': 'black',
            'savefig.facecolor': 'black',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'lines.linewidth': 3.0,
        })
    
    background_color = "#F0F0F0"
    
    # Filter the data for the specified isotope and MT
    filtered_mg = mg_covmat.filter_by_isotope_reaction(isotope, mt)
    
    if filtered_mg.num_matrices == 0:
        raise ValueError(f"No data found for isotope {isotope}, MT {mt}")
    
    # Determine which orders to include
    available_orders = sorted(set(filtered_mg.l_rows + filtered_mg.l_cols))
    
    if orders is None:
        orders = available_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_orders]
    
    if not orders:
        raise ValueError(f"Requested orders {orders} not available. Available orders: {available_orders}")
    
    # Build the matrix for the specified orders
    G = filtered_mg.num_groups
    N = len(orders)
    matrix_size = N * G
    
    # Create mapping from order to index
    order_to_idx = {order: i for i, order in enumerate(orders)}
    
    # Initialize the matrix
    if matrix_type == 'cov':
        full_matrix = np.zeros((matrix_size, matrix_size))
    else:  # correlation
        full_matrix = np.full((matrix_size, matrix_size), np.nan)
        
    # Fill in the matrix blocks
    for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, rel_matrix, abs_matrix) in enumerate(zip(
        filtered_mg.isotope_rows, filtered_mg.reaction_rows, filtered_mg.l_rows,
        filtered_mg.isotope_cols, filtered_mg.reaction_cols, filtered_mg.l_cols,
        filtered_mg.relative_matrices, filtered_mg.absolute_matrices
    )):
        # Skip if orders not in our selection
        if l_r not in orders or l_c not in orders:
            continue
            
        # Get the appropriate matrix
        if covariance_type == 'rel':
            matrix = rel_matrix
        else:
            matrix = abs_matrix
            
        # Calculate block positions
        row_idx = order_to_idx[l_r]
        col_idx = order_to_idx[l_c]
        r0, r1 = row_idx * G, (row_idx + 1) * G
        c0, c1 = col_idx * G, (col_idx + 1) * G
        
        # Fill the block
        full_matrix[r0:r1, c0:c1] = matrix
        
        # Fill transpose if not diagonal block
        if row_idx != col_idx:
            full_matrix[c0:c1, r0:r1] = matrix.T
    
    # Convert to correlation matrix if requested
    if matrix_type == 'corr':
        # Compute correlation matrix
        diag_sqrt = np.sqrt(np.diag(full_matrix))
        
        # Handle zero or negative diagonal elements
        valid_diag = diag_sqrt > 0
        corr_matrix = np.full_like(full_matrix, np.nan)
        
        if np.any(valid_diag):
            # Create outer product for normalization
            diag_outer = np.outer(diag_sqrt, diag_sqrt)
            
            # Only compute correlation for valid diagonal elements
            valid_mask = np.outer(valid_diag, valid_diag)
            corr_matrix[valid_mask] = (full_matrix[valid_mask] / diag_outer[valid_mask])
            
            # Set diagonal to 1.0 for valid elements
            np.fill_diagonal(corr_matrix, 1.0)
        
        full_matrix = corr_matrix
    
    # Set default colormap and log scale based on matrix type
    if colormap is None:
        if matrix_type == 'corr':
            colormap = 'RdYlGn'
        else:
            colormap = 'viridis'
    
    if use_log_scale is None:
        if matrix_type == 'corr':
            use_log_scale = False  # Linear scale for correlation
        else:
            use_log_scale = True   # Log scale for covariance
    
    # Setup plot style - create figure manually like MF34
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    ax.grid(False, which="both")
    for axis_obj in (ax.xaxis, ax.yaxis):
        axis_obj.grid(False)
    
    # Handle colormap selection - use user-specified or smart defaults
    if colormap is None:
        if matrix_type == 'corr':
            colormap_obj = plt.get_cmap("RdYlGn").copy()
        else:
            colormap_obj = plt.get_cmap("viridis").copy()
    else:
        # Use user-specified colormap
        if isinstance(colormap, str):
            try:
                colormap_obj = plt.get_cmap(colormap).copy()
            except ValueError:
                raise ValueError(f"Invalid colormap name: '{colormap}'. "
                               f"Please use a valid matplotlib colormap name.")
        elif hasattr(colormap, 'copy'):
            # Assume it's already a matplotlib colormap object
            colormap_obj = colormap.copy()
        else:
            raise ValueError("colormap must be a string name or matplotlib Colormap object")
    
    colormap_obj.set_bad(color=background_color)
    
    # Mask zeros and prepare matrix for display
    M = np.ma.masked_where(full_matrix == 0.0, full_matrix)
    
    # Determine color scale limits
    if matrix_type == 'corr':
        # For correlation matrices, clamp to [-1, 1] and use symmetric scale
        if symmetric_scale:
            if vmin is None and vmax is None:
                # Check for valid (non-NaN) correlation values
                valid_values = full_matrix[~np.isnan(full_matrix)]
                if len(valid_values) > 0:
                    max_abs = min(1.0, np.max(np.abs(valid_values)))
                    vmin, vmax = -max_abs, max_abs
                else:
                    vmin, vmax = -1.0, 1.0
            elif vmin is None:
                vmin = max(-1.0, -vmax)
            elif vmax is None:
                vmax = min(1.0, -vmin)
        else:
            if vmin is None:
                vmin = max(-1.0, np.nanmin(full_matrix[~np.isnan(full_matrix)]))
            if vmax is None:
                vmax = min(1.0, np.nanmax(full_matrix[~np.isnan(full_matrix)]))
                
        # Handle edge cases for normalization
        if np.isclose(vmin, vmax, atol=1e-10):
            if np.isclose(vmin, 0.0, atol=1e-10):
                vmin = -1e-6
                vmax = 1e-6
            else:
                padding = abs(vmax) * 0.01
                vmin = vmax - padding
                vmax = vmax + padding
        elif vmin >= vmax:
            vmin = vmax - 1e-6 if vmax != 0 else -1e-6
            
        vcenter = 0.0
        if vcenter <= vmin:
            vcenter = vmin + (vmax - vmin) * 0.1
        elif vcenter >= vmax:
            vcenter = vmax - (vmax - vmin) * 0.1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        
    else:
        # For covariance matrices, handle log scale
        valid_values = full_matrix[~np.isnan(full_matrix)]
        if use_log_scale and len(valid_values) > 0:
            # For log scale, ensure all values are positive
            min_positive = np.min(valid_values[valid_values > 0]) if np.any(valid_values > 0) else 1e-10
            if vmin is None:
                vmin = max(min_positive, np.nanmin(valid_values[valid_values > 0])) if np.any(valid_values > 0) else 1e-10
            if vmax is None:
                vmax = np.nanmax(valid_values)
            # Replace non-positive values with vmin for log scale
            full_matrix = np.where(full_matrix <= 0, vmin, full_matrix)
            M = np.ma.masked_where(full_matrix == 0.0, full_matrix)
        else:
            if vmin is None:
                vmin = np.nanmin(valid_values)
            if vmax is None:
                vmax = np.nanmax(valid_values)
        norm = None
    
    # Create the heatmap
    im = ax.imshow(M, cmap=colormap_obj, norm=norm,
                   origin="upper", interpolation="nearest", aspect="auto")
    
    # Add grid lines to separate Legendre order blocks
    for g_val in range(0, matrix_size + 1, G):
        ax.axhline(g_val - 0.5, color="white", lw=1)
        ax.axvline(g_val - 0.5, color="white", lw=1)
    
    # Set up tick labels - show Legendre orders at block centers
    order_centers = [i * G + (G - 1) / 2 for i in range(len(orders))]
    order_labels = [f'L={order}' for order in orders]
    
    ax.set_xticks(order_centers)
    ax.set_yticks(order_centers)
    ax.set_xticklabels(order_labels, rotation=0, ha="center")
    ax.set_yticklabels(order_labels)
    
    ax.set_xlabel("Legendre coefficient")
    ax.set_ylabel("Legendre coefficient")
    
    # Set tick parameters
    ax.tick_params(axis="both", which="major", length=0, pad=5)
    
    # Create title
    try:
        from kika._utils import zaid_to_symbol
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"
    
    if style not in {"paper", "publication"}:
        order_labels_str = ', '.join([str(order) for order in orders])
        matrix_name = "Covariance" if matrix_type == 'cov' else "Correlation"
        title_text = f"{isotope_symbol} MT:{mt}   L: {order_labels_str} {matrix_name}"
        ax.set_title(title_text)
    
    ax.tick_params(axis="both", which="major", length=0)
    
    # Add colorbar like MF34
    if show_colorbar:
        fig.canvas.draw()
        heatmap_pos = ax.get_position()
        
        cbar_horizontal_offset = 0.03
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,
            0.03,
            heatmap_pos.height
        ])
        
        cbar = fig.colorbar(im, cax=cbar_ax)
        if matrix_type == 'cov':
            matrix_name = "Covariance"
        else:
            matrix_name = "Correlation"
        cbar.set_label(matrix_name)
    
    # Layout configuration (like MF34)
    num_orders = len(orders)
    bottom_margin_config = {
        1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20,
    }
    bottom_margin = bottom_margin_config.get(num_orders, bottom_margin_config[7])
    
    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=bottom_margin)
    
    # Add annotations if requested
    if annotate and matrix_size <= 20:  # Only annotate for small matrices
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = full_matrix[i, j]
                if not np.isnan(value) and value != 0:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           fontsize=8, color='white' if abs(value) > 0.5 * max(abs(vmin or 0), abs(vmax or 1)) else 'black')
    
    return fig


