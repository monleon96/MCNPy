"""
DEPRECATED: Legacy plotting functions for ENDF MF4 data.

This module contains legacy plotting functions that use the old plotting style.
These functions are maintained for backward compatibility but are deprecated.

RECOMMENDED APPROACH:
--------------------
Use the `.to_plot_data()` methods on MF4 objects with PlotBuilder:

    from kika.plotting import PlotBuilder
    
    # Single coefficient
    data = endf.to_plot_data(mf=4, mt=2, order=1)
    fig = PlotBuilder().add_data(data).build()
    
    # Multiple coefficients with uncertainties
    builder = PlotBuilder(style='light')
    for order in [1, 2, 3]:
        data, unc_band = endf.to_plot_data(mf=4, mt=2, order=order, uncertainty=True)
        if unc_band is not None:
            builder.add_data(data, uncertainty=unc_band)
        else:
            builder.add_data(data)
    fig = builder.build()

The new approach provides:
- Consistent API with ACE plotting
- Better separation of data and presentation
- More flexibility for customization
- Proper integration with the plotting infrastructure

For 3D visualizations, use plotting3D module.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Optional, Union, Tuple, Dict, Any
from ....plotting._backend_utils import _is_notebook, _detect_interactive_backend, _configure_figure_interactivity
from ....plotting.styles import _apply_style_to_rcparams, _get_color_palette, format_energy_axis_ticks
from ....plotting.covariance import plot_legendre_uncertainty_bands


def _setup_legacy_plot(style='light', figsize=(8, 5), **kwargs):
    """
    Helper to setup matplotlib figure and axes for legacy plotting functions.
    
    Returns (fig, ax, colors) tuple.
    """
    # Detect notebook mode
    notebook_mode = _is_notebook()
    
    # Apply style
    _apply_style_to_rcparams(
        style=style,
        notebook_mode=notebook_mode,
        figsize=figsize,
        dpi=kwargs.get('dpi', 100),
        font_family=kwargs.get('font_family', 'serif'),
        projection=None
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    colors = _get_color_palette(style)
    
    return fig, ax, colors


def _format_legacy_axes(ax, style='light', x_label=None, y_label=None, title=None, 
                        legend_loc='best', use_log_x=False):
    """Helper to format axes for legacy plotting functions."""
    if use_log_x:
        ax.set_xscale('log')
        format_energy_axis_ticks(ax)
    
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    
    # Add legend if there are labeled items
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_kwargs = {'loc': legend_loc, 'framealpha': 0.9}
        if style == 'light':
            legend_kwargs.update({'frameon': True, 'fancybox': False, 'edgecolor': 'black'})
        else:
            legend_kwargs['fancybox'] = True
        ax.legend(**legend_kwargs)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    if style == 'light':
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.5)
    
    return ax


def _generate_unique_labels(endf_list, labels):
    """
    Generate unique labels for ENDF objects, handling duplicates properly.
    
    Parameters
    ----------
    endf_list : list
        List of ENDF objects
    labels : None, str, or list of str
        Label specification
        
    Returns
    -------
    list of str
        Unique labels for each ENDF object
    """
    if labels is None:
        # For multiple ENDF objects without custom labels, use simple indices
        if len(endf_list) > 1:
            endf_labels = [str(i+1) for i in range(len(endf_list))]
        else:
            # For single ENDF, no label needed (will just show isotope - coefficient)
            endf_labels = [None]
    elif isinstance(labels, str):
        if len(endf_list) == 1:
            endf_labels = [labels]
        else:
            # Use the string as a prefix with indices
            endf_labels = [f"{labels} {i+1}" for i in range(len(endf_list))]
    else:
        endf_labels = list(labels)
        # Pad with generic labels if not enough provided
        while len(endf_labels) < len(endf_list):
            endf_labels.append(f"ENDF {len(endf_labels)+1}")
    
    return endf_labels


#: The MF34 uncertainty band, kept under its old private name.
#:
#: The implementation moved to :mod:`kika.plotting.covariance` in phase 4's P4.
#: It draws a ``LegendreCovariance`` and touches no ENDF type, so it was
#: calculation-layer code living in the format package — and
#: ``kika/cov/multigroup/legacy_mg_plotting.py`` reached it here *through its
#: leading underscore*, which is the import the layering ratchet was counting.
#: The alias stays so this module's two call sites read unchanged.
_plot_uncertainty_bands = plot_legendre_uncertainty_bands


def plot_legendre_coefficients_from_endf(
    endf: Union[object, List[object]],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    legend_loc: str = 'best',
    marker: bool = False,
    include_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot specific Legendre coefficients as a function of energy from one or more ENDF objects.
    
    This function automatically accesses the MF4 file and specified MT from the ENDF object(s),
    and optionally includes uncertainties from MF34 if available.
    
    Parameters
    ----------
    endf : ENDF or list of ENDF
        ENDF data object(s) containing MF4 (and optionally MF34) files.
        If a list is provided, all ENDF objects will be plotted on the same figure.
    mt : int
        MT reaction number to plot (e.g., 2 for elastic scattering)
    orders : int or list of int, optional
        Polynomial orders to plot. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis. Values are automatically
        clamped to the available data range.
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    marker : bool
        Whether to include markers on the plot lines
    include_uncertainties : bool
        Whether to include uncertainty bands from MF34 data if available
    uncertainty_sigma : float
        Number of sigma levels for uncertainty bands (default: 1.0 for 1σ)
    labels : str or list of str, optional
        Labels for each ENDF object in the legend. If None, uses isotope symbols
        or generic labels. If a single string and multiple ENDF objects, uses
        the string as a prefix with indices.
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
        
    Raises
    ------
    ValueError
        If any ENDF object doesn't contain MF4 data or the specified MT
    
    Examples
    --------
    Plot elastic scattering Legendre coefficients from multiple ENDF files:
    
    >>> from kika.endf.read_endf import read_endf
    >>> endf1 = read_endf('path/to/endf_file1.txt')
    >>> endf2 = read_endf('path/to/endf_file2.txt')
    >>> fig = plot_legendre_coefficients_from_endf(
    ...     [endf1, endf2], mt=2, orders=[1, 2, 3], 
    ...     labels=['Library A', 'Library B'],
    ...     include_uncertainties=True
    ... )
    >>> fig.show()
    """
    # Convert single ENDF object to list for uniform processing
    if not isinstance(endf, list):
        endf_list = [endf]
    else:
        endf_list = endf
    
    # Handle labels
    endf_labels = _generate_unique_labels(endf_list, labels)
    
    # Validate all ENDF objects first
    for i, endf_obj in enumerate(endf_list):
        # Check if ENDF object has MF4 data
        if 4 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} ({endf_labels[i]}) does not contain MF4 (angular distribution) data")
        
        mf4 = endf_obj.mf[4]
        
        # Check if the specified MT exists in MF4
        if mt not in mf4.mt:
            available_mts = list(mf4.mt.keys())
            raise ValueError(f"MT{mt} not found in MF4 for ENDF object {i+1} ({endf_labels[i]}). Available MTs: {available_mts}")
    
    # Setup plot style
    fig, ax, colors = _setup_legacy_plot(style=style, figsize=figsize, **kwargs)
    
    # Get the first valid ENDF object to determine available orders
    first_endf = endf_list[0]
    first_mf4_data = first_endf.mf[4].mt[mt]
    
    # Check if the first object has Legendre coefficients
    if not hasattr(first_mf4_data, 'legendre_energies') or not hasattr(first_mf4_data, 'legendre_coefficients'):
        # Provide helpful error message for unsupported types
        obj_class_name = type(first_mf4_data).__name__
        if hasattr(first_mf4_data, 'distribution_type'):
            dist_type = first_mf4_data.distribution_type
            error_msg = f"Cannot plot Legendre coefficients for {obj_class_name} object.\n" \
                       f"This object contains {dist_type} angular distributions,\n" \
                       f"which do not have Legendre coefficients.\n\n" \
                       f"Supported object types: MF4MTMixed, MF4MTLegendre"
        else:
            error_msg = f"Cannot plot Legendre coefficients for {obj_class_name} object.\n" \
                       f"This object does not contain Legendre coefficients.\n\n" \
                       f"Supported object types: MF4MTMixed, MF4MTLegendre"
        
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
               transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title("Unsupported Data Type")
        return fig
    
    # Determine which orders to plot based on first ENDF object
    if orders is None:
        # Plot all available orders from first object
        first_coeffs_list = first_mf4_data.legendre_coefficients
        if first_coeffs_list:
            max_available = max(len(coeffs) for coeffs in first_coeffs_list)
            orders = list(range(max_available))
        else:
            ax.text(0.5, 0.5, 'No Legendre coefficient data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return fig
    elif isinstance(orders, int):
        orders = [orders]
    elif not isinstance(orders, list):
        orders = list(orders)  # Convert other iterables to list
    
    # Track all energies for setting x-axis limits later
    all_energies = []
    
    # Plot each ENDF object
    for endf_idx, (endf_obj, endf_label) in enumerate(zip(endf_list, endf_labels)):
        mf4_data = endf_obj.mf[4].mt[mt]
        
        # Skip if this object doesn't have Legendre coefficients
        if not hasattr(mf4_data, 'legendre_energies') or not hasattr(mf4_data, 'legendre_coefficients'):
            print(f"Warning: {endf_label} does not have Legendre coefficients, skipping")
            continue
            
        # Get data
        energies = mf4_data.legendre_energies
        coeffs_list = mf4_data.legendre_coefficients
        
        if not energies or not coeffs_list:
            print(f"Warning: {endf_label} has no Legendre coefficient data, skipping")
            continue
            
        all_energies.extend(energies)
        
        # Plot each order for this ENDF object
        for order_idx, order in enumerate(orders):
            # Calculate color index: different base colors for each ENDF, 
            # but cycle through orders within each ENDF
            color_idx = (endf_idx * len(orders) + order_idx) % len(colors)
            color = colors[color_idx]
            
            # Extract coefficient values for this order across all energies
            coeff_values = []
            energy_values = []
            
            for j, coeffs in enumerate(coeffs_list):
                # Note: coeffs[0] = a_1, coeffs[1] = a_2, etc.
                # So order 0 means a_0 (always 1), order 1 means a_1 (coeffs[0]), etc.
                if order == 0:
                    # a_0 is always 1 (implicit in ENDF format)
                    coeff_values.append(1.0)
                    energy_values.append(energies[j])
                elif order - 1 < len(coeffs):  # order 1 -> coeffs[0], order 2 -> coeffs[1], etc.
                    coeff_values.append(coeffs[order - 1])
                    energy_values.append(energies[j])
            
            if energy_values:
                # Create label combining ENDF label and coefficient order
                # Create base label with isotope and coefficient order
                isotope_symbol = endf_obj.isotope if hasattr(endf_obj, 'isotope') and endf_obj.isotope else 'Unknown'
                if include_uncertainties and 34 in endf_obj.mf and mt in endf_obj.mf[34].mt:
                    if uncertainty_sigma == 1.0:
                        base_label = f"{isotope_symbol} - $a_{{{order}}} \\pm 1\\sigma$"
                    else:
                        base_label = f"{isotope_symbol} - $a_{{{order}}} \\pm {uncertainty_sigma}\\sigma$"
                else:
                    base_label = f"{isotope_symbol} - $a_{{{order}}}$"
                
                # Add library identifier if multiple ENDFs
                if endf_label is not None:
                    label = f"{base_label} ({endf_label})"
                else:
                    label = base_label
                
                # Plot the main line
                line_style = '-'
                if marker:
                    ax.plot(energy_values, coeff_values, line_style, color=color,
                        label=label, linewidth=2, marker='o', markersize=2)
                else:
                    ax.plot(energy_values, coeff_values, line_style, color=color,
                            label=label, linewidth=2)
                
                # Add uncertainty bands if available and requested
                if include_uncertainties and 34 in endf_obj.mf and mt in endf_obj.mf[34].mt:
                    # Get the MF34 covariance matrix data
                    mf34_mt = endf_obj.mf[34].mt[mt]
                    mf34_covmat = mf34_mt.to_ang_covmat()
                    isotope_id = int(mf4_data.zaid) if hasattr(mf4_data, 'zaid') else 0
                    
                    # Use the helper function to plot uncertainty bands
                    _plot_uncertainty_bands(ax, energy_values, coeff_values, mf34_covmat, 
                                           isotope_id, mt, order, uncertainty_sigma, color)
    
    # Create an improved title with isotope and reaction information
    title_parts = []
    
    # Add reaction information
    from kika._constants import MT_TO_REACTION
    if mt in MT_TO_REACTION:
        reaction_name = MT_TO_REACTION[mt]
        title_parts.append(f"MT={mt} {reaction_name}")
    else:
        title_parts.append(f"MT={mt}")
    
    # Add "Legendre Coefficients" at the end
    title_parts.append("Legendre Coefficients")
    
    title = " - ".join(title_parts)
    
    # Format axes
    ax = _format_legacy_axes(
        ax, style=style,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=title,
        legend_loc=legend_loc,
        use_log_x=True
    )
    
    # Apply energy range limits if specified
    if energy_range is not None and all_energies:
        data_min, data_max = min(all_energies), max(all_energies)
        e_min = max(energy_range[0], data_min)
        e_max = min(energy_range[1], data_max)
        ax.set_xlim(e_min, e_max)

    return fig


def plot_legendre_coefficient_uncertainties_from_endf(
    endf: Union[object, List[object]],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    legend_loc: str = 'best',
    uncertainty_type: str = 'relative',
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot uncertainties of Legendre coefficients as a function of energy from one or more ENDF objects.
    
    This function specifically plots the uncertainty values (not the coefficients themselves)
    extracted from MF34 covariance data.
    
    Parameters
    ----------
    endf : ENDF or list of ENDF
        ENDF data object(s) containing MF4 and MF34 files.
        If a list is provided, all ENDF objects will be plotted on the same figure.
    mt : int
        MT reaction number to plot (e.g., 2 for elastic scattering)
    orders : int or list of int, optional
        Polynomial orders to plot uncertainties for. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis. Values are automatically
        clamped to the available data range.
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    uncertainty_type : str
        Type of uncertainty to plot: 'relative' (%) or 'absolute'
    labels : str or list of str, optional
        Labels for each ENDF object in the legend. If None, uses isotope symbols
        or generic labels. If a single string and multiple ENDF objects, uses
        the string as a prefix with indices.
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
        
    Raises
    ------
    ValueError
        If any ENDF object doesn't contain MF4 or MF34 data, or the specified MT
    
    Examples
    --------
    Plot uncertainties for elastic scattering Legendre coefficients from multiple libraries:
    
    >>> from kika.endf.read_endf import read_endf
    >>> endf1 = read_endf('path/to/endf_file1.txt')
    >>> endf2 = read_endf('path/to/endf_file2.txt')
    >>> fig = plot_legendre_coefficient_uncertainties_from_endf(
    ...     [endf1, endf2], mt=2, orders=[1, 2, 3], 
    ...     labels=['Library A', 'Library B'],
    ...     uncertainty_type='relative'
    ... )
    >>> fig.show()
    """
    # Convert single ENDF object to list for uniform processing
    if not isinstance(endf, list):
        endf_list = [endf]
    else:
        endf_list = endf
    
    # Handle labels
    endf_labels = _generate_unique_labels(endf_list, labels)
    
    # Validate all ENDF objects first
    for i, endf_obj in enumerate(endf_list):
        # Check if ENDF object has MF4 data
        if 4 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} ({endf_labels[i]}) does not contain MF4 (angular distribution) data")
        
        # Check if ENDF object has MF34 data
        if 34 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} ({endf_labels[i]}) does not contain MF34 (covariance) data")
        
        mf4 = endf_obj.mf[4]
        mf34 = endf_obj.mf[34]
        
        # Check if the specified MT exists in MF4
        if mt not in mf4.mt:
            available_mts = list(mf4.mt.keys())
            raise ValueError(f"MT{mt} not found in MF4 for ENDF object {i+1} ({endf_labels[i]}). Available MTs: {available_mts}")
        
        # Check if the specified MT exists in MF34
        if mt not in mf34.mt:
            available_mts = list(mf34.mt.keys())
            raise ValueError(f"MT{mt} not found in MF34 for ENDF object {i+1} ({endf_labels[i]}). Available MTs: {available_mts}")
    
    # Setup plot style
    fig, ax, colors = _setup_legacy_plot(style=style, figsize=figsize, **kwargs)
    
    # Get the first valid ENDF object to determine available orders
    first_endf = endf_list[0]
    first_mf4_data = first_endf.mf[4].mt[mt]
    
    # Check if the first object has Legendre coefficients
    if not hasattr(first_mf4_data, 'legendre_energies') or not hasattr(first_mf4_data, 'legendre_coefficients'):
        # Provide helpful error message for unsupported types
        obj_class_name = type(first_mf4_data).__name__
        if hasattr(first_mf4_data, 'distribution_type'):
            dist_type = first_mf4_data.distribution_type
            error_msg = f"Cannot plot Legendre coefficient uncertainties for {obj_class_name} object.\n" \
                       f"This object contains {dist_type} angular distributions,\n" \
                       f"which do not have Legendre coefficients.\n\n" \
                       f"Supported object types: MF4MTMixed, MF4MTLegendre"
        else:
            error_msg = f"Cannot plot Legendre coefficient uncertainties for {obj_class_name} object.\n" \
                       f"This object does not contain Legendre coefficients.\n\n" \
                       f"Supported object types: MF4MTMixed, MF4MTLegendre"
        
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
               transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title("Unsupported Data Type")
        return fig
    
    # Determine which orders to plot based on first ENDF object
    if orders is None:
        # Plot all available orders from first object
        first_coeffs_list = first_mf4_data.legendre_coefficients
        if first_coeffs_list:
            max_available = max(len(coeffs) for coeffs in first_coeffs_list)
            orders = list(range(max_available))
        else:
            ax.text(0.5, 0.5, 'No Legendre coefficient data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return fig
    elif isinstance(orders, int):
        orders = [orders]
    elif not isinstance(orders, list):
        orders = list(orders)  # Convert other iterables to list
    
    # Track all energies for setting x-axis limits later
    all_energies = []
    
    # Plot each ENDF object
    for endf_idx, (endf_obj, endf_label) in enumerate(zip(endf_list, endf_labels)):
        mf4_data = endf_obj.mf[4].mt[mt]
        
        # Skip if this object doesn't have Legendre coefficients
        if not hasattr(mf4_data, 'legendre_energies') or not hasattr(mf4_data, 'legendre_coefficients'):
            print(f"Warning: {endf_label} does not have Legendre coefficients, skipping")
            continue
            
        # Get data from MF4
        energies = mf4_data.legendre_energies
        coeffs_list = mf4_data.legendre_coefficients
        
        if not energies or not coeffs_list:
            print(f"Warning: {endf_label} has no Legendre coefficient data, skipping")
            continue
            
        all_energies.extend(energies)
        
        # Extract uncertainties from MF34 data
        uncertainties = {}
        try:
            # Use the existing to_ang_covmat method to convert MF34MT to LegendreCovariance
            mf34_mt = endf_obj.mf[34].mt[mt]
            mf34_covmat = mf34_mt.to_ang_covmat()
            
            # Get the isotope ID from mf4_data
            isotope_id = int(mf4_data.zaid) if hasattr(mf4_data, 'zaid') else 0
            
            # Extract uncertainties for each requested Legendre order
            uncertainties_data = mf34_covmat.get_uncertainties_for_legendre_coefficient(
                isotope=isotope_id, 
                mt=mt, 
                l_coefficient=orders
            )
            
            # Convert to the format expected by the plotting code
            if isinstance(uncertainties_data, dict):
                # Handle both single coefficient (dict) and multiple coefficients (dict of dicts)
                if 'energies' in uncertainties_data:  # Single coefficient result
                    # This shouldn't happen when passing a list, but handle it just in case
                    uncertainties[orders[0]] = uncertainties_data
                else:  # Multiple coefficients result
                    for l_order, unc_data in uncertainties_data.items():
                        if unc_data is not None:
                            uncertainties[l_order] = unc_data
                            
        except Exception as e:
            print(f"Warning: Could not extract uncertainties from MF34 for {endf_label}: {e}")
            continue
        
        if not uncertainties:
            print(f"Warning: No uncertainty data available for {endf_label}, skipping")
            continue
        
        # Plot uncertainties for each order for this ENDF object
        for order_idx, order in enumerate(orders):
            if order not in uncertainties:
                continue
                
            # Calculate color index: different base colors for each ENDF, 
            # but cycle through orders within each ENDF
            color_idx = (endf_idx * len(orders) + order_idx) % len(colors)
            color = colors[color_idx]
            
            unc_data = uncertainties[order]
            unc_energies = unc_data['energies']
            unc_values = unc_data['uncertainties']
            
            if len(unc_energies) > 0 and len(unc_values) > 0:
                # Determine what to plot based on uncertainty_type
                if uncertainty_type == 'relative':
                    # Plot relative uncertainties as percentages
                    plot_values = np.array(unc_values) * 100  # Convert to percentage
                    
                    # Create label with isotope and order, optionally adding custom label
                    isotope_symbol = endf_obj.isotope if hasattr(endf_obj, 'isotope') and endf_obj.isotope else 'Unknown'
                    label = f"{isotope_symbol} - (L={order})"
                    if endf_label is not None:
                        label = f"{label} ({endf_label})"
                elif uncertainty_type == 'absolute':
                    # For absolute uncertainties, we need the coefficient values
                    # Extract coefficient values for this order across all energies
                    coeff_values = []
                    energy_values = []
                    
                    for j, coeffs in enumerate(coeffs_list):
                        if order == 0:
                            # a_0 is always 1 (implicit in ENDF format)
                            coeff_values.append(1.0)
                            energy_values.append(energies[j])
                        elif order - 1 < len(coeffs):  # order 1 -> coeffs[0], order 2 -> coeffs[1], etc.
                            coeff_values.append(coeffs[order - 1])
                            energy_values.append(energies[j])
                    
                    # Interpolate coefficient values to uncertainty energy grid for calculation
                    if energy_values and coeff_values:
                        # Simple approach: use step-wise values as in the main function
                        interpolated_coeffs = []
                        for unc_energy in unc_energies:
                            # Find closest energy in coefficient data
                            closest_idx = min(range(len(energy_values)), 
                                            key=lambda k: abs(energy_values[k] - unc_energy))
                            interpolated_coeffs.append(abs(coeff_values[closest_idx]))
                        
                        # Convert relative to absolute uncertainties
                        plot_values = np.array(unc_values) * np.array(interpolated_coeffs)
                        isotope_symbol = endf_obj.isotope if hasattr(endf_obj, 'isotope') and endf_obj.isotope else 'Unknown'
                        label = f"{isotope_symbol} - (L={order})"
                        if endf_label is not None:
                            label = f"{label} ({endf_label})"
                    else:
                        print(f"Warning: Could not calculate absolute uncertainties for order {order} in {endf_label}")
                        continue
                else:
                    raise ValueError(f"uncertainty_type must be 'relative' or 'absolute', got '{uncertainty_type}'")
                
                # Plot the uncertainty values using step plot since uncertainties are constant within energy bins
                # Get the actual energy bin boundaries from the covariance matrix's energy_grids attribute
                bin_boundaries = None
                
                # Find the energy grid that corresponds to this (isotope, mt, order) combination
                for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
                    mf34_covmat.isotope_rows, mf34_covmat.reaction_rows, mf34_covmat.l_rows,
                    mf34_covmat.isotope_cols, mf34_covmat.reaction_cols, mf34_covmat.l_cols
                )):
                    # Look for diagonal variance matrix (L = L) for the specified parameters
                    if (iso_r == isotope_id and iso_c == isotope_id and 
                        mt_r == mt and mt_c == mt and 
                        l_r == order and l_c == order):
                        
                        bin_boundaries = np.array(mf34_covmat.energy_grids[i])
                        break
                
                if len(unc_energies) == 1:
                    # Single point - plot as horizontal line across entire range
                    ax.axhline(y=plot_values[0], color=color, label=label, linewidth=2)
                else:
                    if bin_boundaries is not None and len(bin_boundaries) == len(unc_energies) + 1:
                        # We have the actual bin boundaries - use them for proper step plot
                        step_energies = bin_boundaries
                        step_values = np.append(plot_values, plot_values[-1])  # Extend last value for step plot
                        
                        # Plot as step function
                        ax.step(step_energies[:-1], step_values[:-1], where='post', color=color, 
                               label=label, linewidth=2)
                        
                        # Extend the last step to the final boundary
                        ax.hlines(step_values[-1], step_energies[-2], step_energies[-1], 
                                 colors=color, linewidth=2)
                                 
                    else:
                        # Fallback: estimate bin boundaries from bin centers
                        # This approach may not be accurate for actual energy bin structure
                        step_energies = []
                        step_values = []
                        
                        # Add first boundary (extrapolated)
                        if len(unc_energies) > 1:
                            first_boundary = unc_energies[0] - (unc_energies[1] - unc_energies[0]) / 2
                            step_energies.append(max(first_boundary, 1e-5))  # Don't go below 1e-5 eV
                        else:
                            step_energies.append(unc_energies[0] / 2)
                        step_values.append(plot_values[0])
                        
                        # Add boundaries between consecutive points
                        for i in range(len(unc_energies) - 1):
                            boundary = (unc_energies[i] + unc_energies[i + 1]) / 2
                            step_energies.append(boundary)
                            step_values.append(plot_values[i])
                        
                        # Add the last bin
                        step_energies.append(unc_energies[-1] + (unc_energies[-1] - unc_energies[-2]) / 2)
                        step_values.append(plot_values[-1])
                        
                        # Plot as step function with 'post' positioning
                        ax.step(step_energies[:-1], step_values[:-1], where='post', color=color, 
                               label=label, linewidth=2)
                        
                        # Extend the last step to the final boundary
                        ax.hlines(step_values[-1], step_energies[-2], step_energies[-1], 
                                 colors=color, linewidth=2)
    
    # Create title with isotope and reaction information
    title_parts = []
    
    # Add reaction information
    from kika._constants import MT_TO_REACTION
    if mt in MT_TO_REACTION:
        reaction_name = MT_TO_REACTION[mt]
        title_parts.append(f"MT={mt} {reaction_name}")
    else:
        title_parts.append(f"MT={mt}")
    
    # Add "Legendre Coefficient Uncertainties" at the end
    title_parts.append("Legendre Coefficient Uncertainties")
    
    title = " - ".join(title_parts)
    
    # Set appropriate y-label based on uncertainty type
    if uncertainty_type == 'relative':
        y_label = "Relative uncertainty (%)"
    else:
        y_label = "Absolute uncertainty"
    
    # Format axes
    ax = _format_legacy_axes(ax, style=style, x_label="Energy (eV)", y_label=y_label, title=title, legend_loc=legend_loc
   , use_log_x=True)
    
    # Apply energy range limits if specified
    if energy_range is not None:
        # Use user-specified energy range directly, allowing extension beyond data range
        e_min, e_max = energy_range
        ax.set_xlim(e_min, e_max)
        
        # Manually compute y-limits based on visible data
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


# NOTE: Im not sure about the use of this function.
def plot_angular_distribution(
    mf4_mixed,
    energies: Optional[Union[float, Tuple[float, float], List[float]]] = None,
    energy_indices: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    data_type: str = 'legendre',
    cosine_range: Tuple[float, float] = (-1.0, 1.0),
    n_points: int = 181,
    style: str = 'default',
    figsize: Tuple[float, float] = (8,5),
    legend_loc: str = 'best',
    **kwargs
) -> plt.Figure:
    """
    Plot angular distributions from MF4MTMixed data.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energies : float, tuple of float, or list of float, optional
        - float: Find energy bin containing this value
        - tuple: Plot all energy bins in range [tuple[0], tuple[1]]
        - list: Specific energy values to plot (legacy support)
        If None, uses energy_indices
    energy_indices : int, tuple of int, or list of int, optional
        - int: Plot specific energy bin index
        - tuple: Plot energy bins from index tuple[0] to tuple[1] (inclusive)
        - list: Specific energy indices (legacy support)
        If None, plots first few energies
    data_type : str
        Type of data to plot: 'legendre', 'tabulated', or 'both'
    cosine_range : tuple
        Range of cosine values to evaluate (mu_min, mu_max)
    n_points : int
        Number of points for Legendre evaluation (tabulated uses all data points)
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments
    """
    # Setup plot style
    fig, ax, colors = _setup_legacy_plot(style=style, figsize=figsize, **kwargs)
    
    # Determine which energies to plot based on data_type
    plot_energies = _determine_plot_energies_by_type(mf4_mixed, energies, energy_indices, data_type)
    
    # Plot each selected energy
    for i, (energy, energy_type, energy_range) in enumerate(plot_energies):
        color = colors[i % len(colors)]
        
        if energy_type == 'legendre' and data_type in ('legendre', 'both'):
            # Create label with energy range for Legendre (no suffix)
            if energy_range:
                label = f"E=[{energy:.3e}, {energy_range:.3e}) eV"
            else:
                label = f"E={energy:.3e} eV"
            
            _plot_legendre_distribution(
                ax, mf4_mixed, energy, cosine_range, n_points, 
                color, label
            )
        
        if energy_type == 'tabulated' and data_type in ('tabulated', 'both'):
            _plot_tabulated_distribution(
                ax, mf4_mixed, energy, color, f"E={energy:.3e} eV (Tab)"
            )
    
    # Format axes
    ax = _format_legacy_axes(
        ax, style=style,
        x_label="Cosine of scattering angle (μ)",
        y_label="Probability density f(μ,E)",
        title=f"Angular Distribution - MT{mf4_mixed.number}",
        legend_loc=legend_loc,
        use_log_x=False
    )
    
    ax.set_xlim(cosine_range)




def _find_energy_bin(energy_grid: List[float], target_energy: float) -> Optional[float]:
    """
    Find the energy bin that contains the target energy.
    
    For energy bins [E_i, E_{i+1}), find E_i such that E_i <= target_energy < E_{i+1}.
    For the last bin, include the upper boundary.
    
    Parameters
    ----------
    energy_grid : List[float]
        Sorted list of energy bin boundaries
    target_energy : float
        Target energy to find bin for
        
    Returns
    -------
    float or None
        Energy bin boundary (E_i) that contains target_energy, or None if not found
    """
    if not energy_grid or target_energy < energy_grid[0]:
        return None
    
    # Check if target is beyond the last energy
    if target_energy > energy_grid[-1]:
        return None
    
    # Find the appropriate bin
    for i in range(len(energy_grid) - 1):
        if energy_grid[i] <= target_energy < energy_grid[i + 1]:
            return energy_grid[i]
    
    # Special case: if target equals the last energy, return the second-to-last bin
    if len(energy_grid) >= 2 and target_energy == energy_grid[-1]:
        return energy_grid[-2]
    
    return None


def _determine_plot_energies_by_type(mf4_mixed, energies, energy_indices, data_type) -> List[Tuple[float, str, Optional[float]]]:
    """
    Determine which energies to plot based on data type.
    
    Returns list of (energy, type, next_energy) tuples where type is 'legendre' or 'tabulated'
    and next_energy is the upper bound for Legendre energy bins (None for tabulated).
    """
    plot_energies = []
    
    if energies is not None:
        if isinstance(energies, tuple) and len(energies) == 2:
            # Plot all energies in range [energies[0], energies[1]]
            e_min, e_max = energies
            
            if data_type in ('legendre', 'both'):
                # Add Legendre energies in range with upper bounds
                for i, energy in enumerate(mf4_mixed.legendre_energies):
                    if e_min <= energy <= e_max:
                        next_energy = (mf4_mixed.legendre_energies[i + 1] 
                                     if i + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                # Add tabulated energies in range
                for energy in mf4_mixed.tabulated_energies:
                    if e_min <= energy <= e_max:
                        plot_energies.append((energy, 'tabulated', None))
                    
        elif isinstance(energies, (int, float)):
            # Find energy bins containing this value
            target_energy = float(energies)
            
            if data_type in ('legendre', 'both'):
                legendre_energy = _find_energy_bin(mf4_mixed.legendre_energies, target_energy)
                if legendre_energy is not None:
                    # Find the upper bound for this energy bin
                    idx = mf4_mixed.legendre_energies.index(legendre_energy)
                    next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                 if idx + 1 < len(mf4_mixed.legendre_energies) 
                                 else None)
                    plot_energies.append((legendre_energy, 'legendre', next_energy))
                
            if data_type in ('tabulated', 'both'):
                tabulated_energy = _find_energy_bin(mf4_mixed.tabulated_energies, target_energy)
                if tabulated_energy is not None:
                    plot_energies.append((tabulated_energy, 'tabulated', None))
                
        elif isinstance(energies, list):
            # Legacy support - exact energy matches
            if data_type in ('legendre', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.legendre_energies:
                        idx = mf4_mixed.legendre_energies.index(energy)
                        next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                     if idx + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.tabulated_energies:
                        plot_energies.append((energy, 'tabulated', None))
    
    elif energy_indices is not None:
        if isinstance(energy_indices, tuple) and len(energy_indices) == 2:
            # Plot energy bins from index tuple[0] to tuple[1] (inclusive)
            start, end = energy_indices
            
            if data_type in ('legendre', 'both'):
                # Add Legendre energies in index range
                for idx in range(start, min(end + 1, len(mf4_mixed.legendre_energies))):
                    energy = mf4_mixed.legendre_energies[idx]
                    next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                 if idx + 1 < len(mf4_mixed.legendre_energies) 
                                 else None)
                    plot_energies.append((energy, 'legendre', next_energy))
                
            if data_type in ('tabulated', 'both'):
                # Add tabulated energies in index range
                for idx in range(start, min(end + 1, len(mf4_mixed.tabulated_energies))):
                    energy = mf4_mixed.tabulated_energies[idx]
                    plot_energies.append((energy, 'tabulated', None))
                
        elif isinstance(energy_indices, int):
            # Plot specific energy bin indices
            if data_type in ('legendre', 'both') and energy_indices < len(mf4_mixed.legendre_energies):
                energy = mf4_mixed.legendre_energies[energy_indices]
                next_energy = (mf4_mixed.legendre_energies[energy_indices + 1] 
                             if energy_indices + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both') and energy_indices < len(mf4_mixed.tabulated_energies):
                energy = mf4_mixed.tabulated_energies[energy_indices]
                plot_energies.append((energy, 'tabulated', None))
                
        elif isinstance(energy_indices, list):
            # Legacy support - specific indices
            if data_type in ('legendre', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.legendre_energies):
                        energy = mf4_mixed.legendre_energies[idx]
                        next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                     if idx + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.tabulated_energies):
                        energy = mf4_mixed.tabulated_energies[idx]
                        plot_energies.append((energy, 'tabulated', None))
    
    else:
        # Default behavior based on data_type
        if data_type == 'legendre':
            # Plot first 3 Legendre energies with upper bounds
            for i, energy in enumerate(mf4_mixed.legendre_energies[:3]):
                next_energy = (mf4_mixed.legendre_energies[i + 1] 
                             if i + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
        elif data_type == 'tabulated':
            # Plot first 3 tabulated energies
            for energy in mf4_mixed.tabulated_energies[:3]:
                plot_energies.append((energy, 'tabulated', None))
        elif data_type == 'both':
            # Plot first few energies from both types
            for i, energy in enumerate(mf4_mixed.legendre_energies[:3]):
                next_energy = (mf4_mixed.legendre_energies[i + 1] 
                             if i + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
            for energy in mf4_mixed.tabulated_energies[:3]:
                plot_energies.append((energy, 'tabulated', None))
    
    return plot_energies


def _plot_legendre_distribution(ax, mf4_mixed, energy, cosine_range, n_points, color, label):
    """Plot angular distribution from Legendre coefficients."""
    coeffs = mf4_mixed.get_legendre_coefficients(energy)
    if not coeffs:
        return
    
    # Generate cosine values
    mu = np.linspace(cosine_range[0], cosine_range[1], n_points)
    
    # Evaluate Legendre expansion: f(μ) = Σ aₗ × (2l+1)/2 × Pₗ(μ)
    # Note: a_0 = 1 is implicit and not included in coeffs list
    # coeffs[0] = a_1, coeffs[1] = a_2, etc.
    
    # Pre-compute all Legendre polynomials up to required order
    # We need polynomials from P_0 to P_(len(coeffs))
    max_order = len(coeffs) + 1  # +1 to include P_0
    legendre_polys = np.zeros((max_order, n_points))
    
    # P₀(μ) = 1
    legendre_polys[0, :] = 1.0
    
    # P₁(μ) = μ
    if max_order > 1:
        legendre_polys[1, :] = mu
    
    # Use recurrence relation: (l+1)Pₗ₊₁(μ) = (2l+1)μPₗ(μ) - lPₗ₋₁(μ)
    for l in range(2, max_order):
        legendre_polys[l, :] = ((2*l - 1) * mu * legendre_polys[l-1, :] - 
                                (l - 1) * legendre_polys[l-2, :]) / l
    
    # Start with a_0 = 1, P_0(μ) term
    prob_density = (2*0 + 1) / 2 * legendre_polys[0, :]  # a_0 * (1/2) * P_0(μ) = 1/2
    
    # Add contributions from a_1, a_2, ... (coeffs[0], coeffs[1], ...)
    for i, coeff in enumerate(coeffs):
        l = i + 1  # coeffs[0] corresponds to a_1 (l=1), coeffs[1] to a_2 (l=2), etc.
        prob_density += coeff * (2*l + 1) / 2 * legendre_polys[l, :]
    
    ax.plot(mu, prob_density, '-', color=color, label=label, linewidth=2)


def _plot_tabulated_distribution(ax, mf4_mixed, energy, color, label):
    """Plot tabulated angular distribution using all data points with interpolation."""
    cosines, probabilities = mf4_mixed.get_tabulated_distribution(energy)
    if not cosines or not probabilities:
        return
    
    # Get the interpolation scheme for this energy
    try:
        energy_idx = mf4_mixed.tabulated_energies.index(energy)
        ang_interp = (mf4_mixed.angular_interpolation[energy_idx] 
                     if energy_idx < len(mf4_mixed.angular_interpolation) 
                     else [])
    except (ValueError, IndexError):
        ang_interp = []
    
    # Plot the original data points
    ax.plot(cosines, probabilities, 'o', color=color, markersize=6, 
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2,
            label=label)
    
    # If we have interpolation information, create interpolated curve
    if ang_interp and len(cosines) > 1:
        # Create a dense grid for smooth interpolation
        mu_interp = np.linspace(min(cosines), max(cosines), 200)
        prob_interp = _interpolate_angular_data(cosines, probabilities, mu_interp, ang_interp)
        
        # Plot the interpolated curve
        ax.plot(mu_interp, prob_interp, '-', color=color, linewidth=2, alpha=0.7)
    else:
        # Simple linear interpolation between points if no scheme specified
        ax.plot(cosines, probabilities, '-', color=color, linewidth=2, alpha=0.7)


def _interpolate_angular_data(cosines, probabilities, mu_interp, ang_interp):
    """
    Interpolate angular distribution data using ENDF interpolation schemes.
    
    Parameters
    ----------
    cosines : list
        Original cosine values
    probabilities : list
        Original probability values
    mu_interp : array
        Cosine values to interpolate to
    ang_interp : list
        List of (NBT, INT) interpolation scheme pairs
        
    Returns
    -------
    array
        Interpolated probability values
    """
    if not ang_interp:
        # Default to linear interpolation
        return np.interp(mu_interp, cosines, probabilities)
    
    prob_interp = np.zeros_like(mu_interp)
    
    # Process interpolation regions
    start_idx = 0
    for nbt, int_code in ang_interp:
        end_idx = min(nbt, len(cosines))
        
        # Extract data for this region
        region_cosines = cosines[start_idx:end_idx]
        region_probs = probabilities[start_idx:end_idx]
        
        if len(region_cosines) < 2:
            start_idx = end_idx
            continue
        
        # Find interpolation points in this region
        mask = (mu_interp >= region_cosines[0]) & (mu_interp <= region_cosines[-1])
        if not np.any(mask):
            start_idx = end_idx
            continue
            
        mu_region = mu_interp[mask]
        
        # Apply interpolation scheme based on INT code
        if int_code == 1:  # Histogram (constant in each bin)
            # For histogram, use previous value
            prob_region = np.zeros_like(mu_region)
            for i, mu in enumerate(mu_region):
                # Find the bin this mu falls into
                bin_idx = np.searchsorted(region_cosines, mu, side='right') - 1
                bin_idx = max(0, min(bin_idx, len(region_probs) - 1))
                prob_region[i] = region_probs[bin_idx]
                
        elif int_code == 2:  # Linear-linear interpolation
            prob_region = np.interp(mu_region, region_cosines, region_probs)
            
        elif int_code == 3:  # Linear-log interpolation
            # y is linear, x is log - but cosines can be negative, so use linear
            prob_region = np.interp(mu_region, region_cosines, region_probs)
            
        elif int_code == 4:  # Log-linear interpolation
            # x is linear, y is log
            region_probs_pos = np.maximum(region_probs, 1e-10)  # Avoid log(0)
            log_probs = np.log(region_probs_pos)
            log_prob_region = np.interp(mu_region, region_cosines, log_probs)
            prob_region = np.exp(log_prob_region)
            
        elif int_code == 5:  # Log-log interpolation
            # Both x and y are log - but cosines can be negative, so use linear-log
            region_probs_pos = np.maximum(region_probs, 1e-10)  # Avoid log(0)
            log_probs = np.log(region_probs_pos)
            log_prob_region = np.interp(mu_region, region_cosines, log_probs)
            prob_region = np.exp(log_prob_region)
            
        else:
            # Default to linear interpolation for unknown schemes
            prob_region = np.interp(mu_region, region_cosines, region_probs)
        
        prob_interp[mask] = prob_region
        start_idx = end_idx
    
    return prob_interp


def plot_legendre_coefficient_comparison(
    reference_endf,
    comparison_endfs: Union[object, List[object]],
    mt: int,
    order: int,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    legend_loc: str = 'best',
    include_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    reference_label: Optional[str] = None,
    comparison_labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a specific Legendre coefficient from multiple ENDF files for comparison.
    
    This function plots one Legendre coefficient order from a reference ENDF file 
    (solid line) and compares it with the same coefficient from one or more 
    comparison ENDF files (dashed lines), each with different colors.
    
    Parameters
    ----------
    reference_endf : ENDF
        Reference ENDF data object containing MF4 (and optionally MF34) files
    comparison_endfs : ENDF or list of ENDF
        Comparison ENDF data object(s) to plot against the reference
    mt : int
        MT reaction number to plot (e.g., 2 for elastic scattering)
    order : int
        Specific Legendre coefficient order to plot (e.g., 1 for a_1, 2 for a_2)
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis. Values are automatically
        clamped to the available data range.
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    include_uncertainties : bool
        Whether to include uncertainty bands from MF34 data for the reference file if available
    uncertainty_sigma : float
        Number of sigma levels for uncertainty bands (default: 1.0 for 1σ)
    reference_label : str, optional
        Label for the reference line. If None, uses "Reference"
    comparison_labels : str or list of str, optional
        Label(s) for comparison line(s). If None, uses "Comparison 1", "Comparison 2", etc.
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the comparison plot
        
    Raises
    ------
    ValueError
        If any ENDF object doesn't contain MF4 data or the specified MT
    
    Examples
    --------
    Compare elastic scattering first Legendre coefficient (a_1) between files:
    
    >>> from kika.endf.read_endf import read_endf
    >>> ref_endf = read_endf('reference_file.txt')
    >>> comp_endf = read_endf('comparison_file.txt')
    >>> fig = plot_legendre_coefficient_comparison(
    ...     ref_endf, comp_endf, mt=2, order=1,
    ...     reference_label="ENDF/B-VIII.0",
    ...     comparison_labels="JEFF-3.3"
    ... )
    >>> fig.show()
    
    Compare with uncertainties for the reference file:
    
    >>> fig = plot_legendre_coefficient_comparison(
    ...     ref_endf, comp_endf, mt=2, order=1,
    ...     include_uncertainties=True,
    ...     reference_label="ENDF/B-VIII.0",
    ...     comparison_labels="JEFF-3.3"
    ... )
    >>> fig.show()
    
    Compare with 2-sigma uncertainty bands:
    
    >>> fig = plot_legendre_coefficient_comparison(
    ...     ref_endf, comp_endf, mt=2, order=1,
    ...     include_uncertainties=True, uncertainty_sigma=2.0,
    ...     reference_label="ENDF/B-VIII.0",
    ...     comparison_labels="JEFF-3.3"
    ... )
    >>> fig.show()
    
    # Example with multiple files:
    # >>> comp_files = [read_endf('file1.txt'), read_endf('file2.txt')]
    # >>> fig = plot_legendre_coefficient_comparison(
    # ...     ref_endf, comp_files, mt=2, order=2,
    # ...     comparison_labels=["JEFF-3.3", "JENDL-5.0"]
    # ... )
    # >>> fig.show()
    """
    # Convert single comparison ENDF to list for uniform processing
    if not isinstance(comparison_endfs, list):
        comparison_endfs = [comparison_endfs]
    
    # Setup plot style
    fig, ax, colors = _setup_legacy_plot(style=style, figsize=figsize, **kwargs)
    
    # Helper function to extract coefficient data from an ENDF object
    def extract_coefficient_data(endf, mt, order):
        """Extract Legendre coefficient data from ENDF object."""
        # Check if ENDF object has MF4 data
        if 4 not in endf.mf:
            raise ValueError(f"ENDF object does not contain MF4 (angular distribution) data")
        
        mf4 = endf.mf[4]
        
        # Check if the specified MT exists in MF4
        if mt not in mf4.mt:
            available_mts = list(mf4.mt.keys())
            raise ValueError(f"MT{mt} not found in MF4. Available MTs: {available_mts}")
        
        mf4_data = mf4.mt[mt]
        
        # Check if the object has Legendre coefficients
        if not hasattr(mf4_data, 'legendre_energies') or not hasattr(mf4_data, 'legendre_coefficients'):
            obj_class_name = type(mf4_data).__name__
            if hasattr(mf4_data, 'distribution_type'):
                dist_type = mf4_data.distribution_type
                error_msg = f"Cannot plot Legendre coefficients for {obj_class_name} object with {dist_type} distributions"
            else:
                error_msg = f"Cannot plot Legendre coefficients for {obj_class_name} object"
            raise ValueError(error_msg + ". Supported object types: MF4MTMixed, MF4MTLegendre")
        
        # Get data
        energies = mf4_data.legendre_energies
        coeffs_list = mf4_data.legendre_coefficients
        
        if not energies or not coeffs_list:
            return None, None
        
        # Extract coefficient values for the specified order across all energies
        coeff_values = []
        energy_values = []
        for j, coeffs in enumerate(coeffs_list):
            if order == 0:
                # a_0 is always 1 (normalization)
                coeff_values.append(1.0)
                energy_values.append(energies[j])
            elif order - 1 < len(coeffs):
                # coeffs[0] = a_1, coeffs[1] = a_2, etc.
                coeff_values.append(coeffs[order - 1])
                energy_values.append(energies[j])
        return energy_values, coeff_values
    
    # Extract data from reference ENDF
    try:
        ref_energies, ref_coeffs = extract_coefficient_data(reference_endf, mt, order)
    except ValueError as e:
        ax.text(0.5, 0.5, f'Error in reference file:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        return fig
    if ref_energies is None:
        ax.text(0.5, 0.5, 'No Legendre coefficient data available in reference file', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot reference data (solid line)
    ref_color = colors[0]
    if include_uncertainties and 34 in reference_endf.mf and mt in reference_endf.mf[34].mt:
        if uncertainty_sigma == 1.0:
            ref_label = f"{reference_label or 'Reference'} ± 1σ"
        else:
            ref_label = f"{reference_label or 'Reference'} ± {uncertainty_sigma}σ"
    else:
        ref_label = reference_label or "Reference"
    ax.plot(ref_energies, ref_coeffs, '-', color=ref_color,
        label=ref_label, linewidth=2)
    
    # Add uncertainty bands for reference data if available and requested
    if include_uncertainties and 34 in reference_endf.mf and mt in reference_endf.mf[34].mt:
        # Get the MF34 covariance matrix data
        mf34_mt = reference_endf.mf[34].mt[mt]
        mf34_covmat = mf34_mt.to_ang_covmat()
        ref_mf4_data = reference_endf.mf[4].mt[mt]
        isotope_id = int(ref_mf4_data.zaid) if hasattr(ref_mf4_data, 'zaid') else 0
        
        # Use the helper function to plot uncertainty bands
        _plot_uncertainty_bands(ax, ref_energies, ref_coeffs, mf34_covmat, 
                               isotope_id, mt, order, uncertainty_sigma, ref_color)
    
    # Plot comparison data (dashed lines)
    for i, comp_endf in enumerate(comparison_endfs):
        color = colors[(i + 1) % len(colors)]
        try:
            comp_energies, comp_coeffs = extract_coefficient_data(comp_endf, mt, order)
            if comp_energies is None:
                print(f"Warning: No data available in comparison file {i+1}")
                continue
        except ValueError as e:
            print(f"Warning: Error in comparison file {i+1}: {e}")
            continue
        # Determine label
        if comparison_labels is None:
            comp_label = f"Comparison {i+1}"
        elif isinstance(comparison_labels, str) and len(comparison_endfs) == 1:
            comp_label = comparison_labels
        elif isinstance(comparison_labels, list) and i < len(comparison_labels):
            comp_label = comparison_labels[i]
        else:
            comp_label = f"Comparison {i+1}"
        # Plot comparison line (dashed)
        ax.plot(comp_energies, comp_coeffs, '--', color=color,
                label=comp_label, linewidth=2)
    
    # Create title with isotope and reaction information
    title_parts = []
    
    # Add isotope information from reference if available
    isotope_symbol = reference_endf.isotope
    if isotope_symbol:
        title_parts.append(f"{isotope_symbol}")
    
    # Add reaction information
    from kika._constants import MT_TO_REACTION
    if mt in MT_TO_REACTION:
        reaction_name = MT_TO_REACTION[mt]
        title_parts.append(f"MT={mt} {reaction_name}")
    else:
        title_parts.append(f"MT={mt}")
    
    # Add Legendre coefficient order
    if order == 0:
        title_parts.append(f"(L={order})")
    else:
        title_parts.append(f"(L={order})")
    
    title = " - ".join(title_parts)
    
    # Format axes
    ax = _format_legacy_axes(ax, style=style, x_label="Energy (eV)", y_label="Coefficient value", title=title, legend_loc=legend_loc
   , use_log_x=True)
    
    # Apply energy range limits if specified
    if energy_range is not None:
        # Get energy range from all data
        all_energies = ref_energies[:]
        for comp_endf in comparison_endfs:
            try:
                comp_energies, _ = extract_coefficient_data(comp_endf, mt, order)
                if comp_energies:
                    all_energies.extend(comp_energies)
            except:
                continue
        if all_energies:
            data_min, data_max = min(all_energies), max(all_energies)
            e_min = max(energy_range[0], data_min)
            e_max = min(energy_range[1], data_max)
            ax.set_xlim(e_min, e_max)
