"""
Multigroup covariance plotting functions using the modern PlotBuilder infrastructure.

This module provides convenience wrapper functions for plotting multigroup MF34
covariance data using the refactored plotting system with to_plot_data() and 
to_heatmap_data() methods along with PlotBuilder.

The functions maintain API compatibility while using the new, cleaner implementation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Tuple, Optional
from kika.cov.multigroup.mg_mf34_covmat import MGMF34CovMat
from kika.cov.mf34_covmat import MF34CovMat
from kika.plotting.plot_builder import PlotBuilder
from kika.plotting.heatmap_builder import HeatmapBuilder


def endf_uncertainty_band(
    endf_cov: MF34CovMat,
    mf4_mt,
    isotope: int,
    mt: int,
    order: int,
    sigma: float,
    nominal_data,
    color: Optional[str] = None,
) -> Optional["UncertaintyBand"]:
    """
    Build an ``UncertaintyBand`` from MF34 piecewise-constant relative uncertainties
    mapped onto a dense nominal curve.

    Parameters
    ----------
    endf_cov : MF34CovMat
        Continuous-energy covariance matrix (from ``MF34MT.to_ang_covmat()``).
    mf4_mt : MF4MT
        MF4 angular-distribution object (any concrete subclass).
    isotope : int
        ZAID of the isotope.
    mt : int
        Reaction MT number.
    order : int
        Legendre order.
    sigma : float
        Number of sigma levels for the band width.
    nominal_data : LegendreCoeffPlotData
        Dense nominal curve (x = energies, y = coefficient values).
    color : str, optional
        Band fill colour.

    Returns
    -------
    UncertaintyBand or None
    """
    from kika.plotting import UncertaintyBand

    if endf_cov is None or not hasattr(endf_cov, "get_uncertainties_for_legendre_coefficient"):
        return None
    unc_info = endf_cov.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
    if not unc_info:
        return None

    energies_unc = np.asarray(unc_info["energies"], dtype=float)
    rel_unc = np.asarray(unc_info["uncertainties"], dtype=float)
    is_relative = unc_info.get("is_relative", True)

    if not is_relative:
        if nominal_data is None:
            return None
        coeff_x = np.asarray(nominal_data.x, dtype=float)
        coeff_y = np.asarray(nominal_data.y, dtype=float)
        if len(coeff_y) == len(coeff_x) + 1:
            coeff_y = coeff_y[:-1]
        centers_unc = np.sqrt(energies_unc[:-1] * energies_unc[1:])
        coeff_interp = np.interp(centers_unc, coeff_x, coeff_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.where(coeff_interp != 0, rel_unc / np.abs(coeff_interp), np.nan)

    if rel_unc.size == 0 or nominal_data is None:
        return None

    nom_x = np.asarray(nominal_data.x, dtype=float)
    nom_y = np.asarray(nominal_data.y, dtype=float)
    if len(nom_y) == len(nom_x) + 1:
        nom_y = nom_y[:-1]

    mask = (nom_x >= energies_unc[0]) & (nom_x <= energies_unc[-1])
    if not np.any(mask):
        return None
    band_x = nom_x[mask]
    band_y = nom_y[mask]

    bin_idx = np.searchsorted(energies_unc, band_x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(rel_unc) - 1)
    rel_unc_dense = rel_unc[bin_idx]

    y_lower = band_y * (1.0 - sigma * rel_unc_dense)
    y_upper = band_y * (1.0 + sigma * rel_unc_dense)

    return UncertaintyBand(
        x=band_x,
        y_lower=y_lower,
        y_upper=y_upper,
        sigma=sigma,
        color=color,
    )


def endf_uncertainty_step_data(
    endf_cov: MF34CovMat,
    mf4_mt,
    isotope: int,
    mt: int,
    order: int,
    sigma: float = 1.0,
    label: Optional[str] = None,
    **styling_kwargs,
) -> Optional["LegendreUncertaintyPlotData"]:
    """
    Build a step-plot of ENDF relative uncertainties (%) for overlay comparison.

    Parameters
    ----------
    endf_cov : MF34CovMat
        Continuous-energy covariance matrix.
    mf4_mt : MF4MT or None
        MF4 angular-distribution object (needed when uncertainties are absolute).
    isotope : int
        ZAID of the isotope.
    mt : int
        Reaction MT number.
    order : int
        Legendre order.
    sigma : float, default 1.0
        Sigma multiplier.
    label : str, optional
        Plot label.  Defaults to ``"ENDF L={order}"``.
    **styling_kwargs
        Passed through to ``LegendreUncertaintyPlotData`` (color, linestyle, …).

    Returns
    -------
    LegendreUncertaintyPlotData or None
    """
    from kika._utils import zaid_to_symbol
    from kika.plotting import LegendreUncertaintyPlotData

    if endf_cov is None:
        return None
    unc_info = endf_cov.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
    if not unc_info:
        return None

    energies = np.asarray(unc_info["energies"], dtype=float)
    unc_values = np.asarray(unc_info["uncertainties"], dtype=float) * sigma
    is_relative = unc_info.get("is_relative", True)

    centers = np.sqrt(energies[:-1] * energies[1:])

    if is_relative:
        rel_unc_pct = unc_values * 100.0
    else:
        if mf4_mt is None or not hasattr(mf4_mt, "extract_legendre_coefficients"):
            return None
        coeffs = mf4_mt.extract_legendre_coefficients(
            energy=centers,
            max_legendre_order=order,
            out_of_range="zero",
        ).get(order)
        if coeffs is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc_pct = np.where(coeffs != 0, (unc_values / np.abs(coeffs)) * 100.0, np.nan)

    if label is None:
        label = f"ENDF L={order}"

    # Pop keys that LegendreUncertaintyPlotData does not accept as **kwargs
    # but that are part of its explicit fields — let them go through normally.
    return LegendreUncertaintyPlotData(
        x=energies,
        y=np.append(rel_unc_pct, rel_unc_pct[-1]),
        label=label,
        order=order,
        isotope=zaid_to_symbol(isotope) if isinstance(isotope, int) else isotope,
        mt=mt,
        uncertainty_type="relative",
        plot_type="step",
        **styling_kwargs,
    )


def plot_mg_legendre_coefficients(
    mg_covmat: MGMF34CovMat,
    nuclide: Union[int, str],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    *,
    style: str = "light",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    marker: bool = True,
    show_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    title: Optional[str] = "default",
    **styling_kwargs
) -> plt.Figure:
    """
    Plot multigroup Legendre coefficients with optional uncertainty bands.
    
    This modern implementation uses the PlotBuilder infrastructure with to_plot_data()
    for cleaner, more maintainable code.

    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        The multigroup MF34 covariance matrix object
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (10, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range (min, max) to limit the x-axis
    marker : bool, default True
        Whether to include markers on the plot lines
    show_uncertainties : bool, default False
        Whether to show uncertainty bands if available
    uncertainty_sigma : float, default 1.0
        Number of sigma levels for uncertainty bands
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    title : str or None, default "default"
        Plot title. If "default", generates a title from nuclide and MT.
        If None, no title is displayed. If a string, uses that as the title.
    **styling_kwargs
        Additional plotting arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot

    Examples
    --------
    Plot all available Legendre orders:
    
    >>> fig = plot_mg_legendre_coefficients(mg_covmat, nuclide=92235, mt=2)
    
    Plot specific orders with uncertainties:
    
    >>> fig = plot_mg_legendre_coefficients(
    ...     mg_covmat, nuclide='U235', mt=2,
    ...     orders=[1, 2, 3],
    ...     show_uncertainties=True
    ... )

    See Also
    --------
    plot_mg_covariance_heatmap : Plot covariance heatmaps for multigroup data
    """
    from kika._utils import symbol_to_zaid, zaid_to_symbol
    
    # Convert nuclide to isotope (ZAID) if string
    if isinstance(nuclide, str):
        isotope = symbol_to_zaid(nuclide)
    else:
        isotope = nuclide
    
    # Get available orders if not specified
    if orders is None:
        orders = sorted(set(l for (iso, m, l) in mg_covmat.legendre_coefficients.keys() 
                           if iso == isotope and m == mt))
    elif isinstance(orders, int):
        orders = [orders]
    
    if not orders:
        raise ValueError(f"No Legendre orders found for isotope={isotope}, MT={mt}")
    
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        use_log_x = True
    elif scale_normalized in ("lin", "linear"):
        use_log_x = False
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")
    
    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)
    color_cycle = getattr(builder, "_colors", None) or []
    
    # Add data for each order
    for idx, order in enumerate(orders):
        try:
            # Get Legendre coefficient data using to_plot_data
            legendre_data, unc_band = mg_covmat.to_plot_data(
                nuclide=nuclide,
                mt=mt,
                order=order,
                sigma=uncertainty_sigma,
                label=f"{zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide} - $a_{{{order}}}$",
                **styling_kwargs
            )
            
            if legendre_data is not None:
                color = getattr(legendre_data, "color", None)
                if color is None and color_cycle:
                    color = color_cycle[idx % len(color_cycle)]
                    legendre_data.color = color

                if unc_band is not None and getattr(unc_band, "color", None) is None:
                    unc_band.color = color

                builder.add_data(
                    legendre_data,
                    uncertainty=unc_band if show_uncertainties else None,
                )

                if marker:
                    try:
                        centers = np.sqrt(
                            np.asarray(mg_covmat.energy_grid[:-1], dtype=float)
                            * np.asarray(mg_covmat.energy_grid[1:], dtype=float)
                        )
                        coeffs = legendre_data.y[:-1]
                        from kika.plotting import PlotData

                        scatter = PlotData(
                            x=centers,
                            y=coeffs,
                            plot_type="scatter",
                            marker="o",
                            markersize=4,
                            color=color,
                            label=None,
                            alpha=getattr(legendre_data, "alpha", None),
                        )
                        builder.add_data(scatter)
                    except Exception:
                        pass
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not plot L={order} for isotope={isotope}, MT={mt}: {e}")
            continue
    
    # Configure scales and labels
    builder.set_scales(log_x=use_log_x, log_y=False)
    if energy_range is not None:
        builder.set_limits(x_lim=energy_range)
    
    # Set title based on parameter
    if title == "default":
        plot_title = f"{zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide} MT={mt} - Legendre Coefficients"
    else:
        plot_title = title
    
    builder.set_labels(
        x_label="Energy (eV)",
        y_label="Legendre Coefficient",
        title=plot_title
    )
    builder.set_legend(loc=legend_loc)
    
    # Build and return
    fig = builder.build()
    return fig


def plot_mg_vs_endf_comparison(
    mg_covmat: MGMF34CovMat,
    endf: object,
    nuclide: Union[int, str],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    *,
    style: str = "light",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    mg_marker: bool = True,
    show_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    endf_native: bool = False,
    title: Optional[str] = "default",
    mg_style: Optional[dict] = None,
    endf_style: Optional[dict] = None,
    **styling_kwargs
) -> plt.Figure:
    """
    Compare multigroup Legendre coefficients with continuous ENDF data.

    This modern implementation uses PlotBuilder to overlay multigroup step plots
    with continuous ENDF coefficient curves and optional ±σ bands.

    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        The multigroup MF34 covariance matrix object
    endf : ENDF object
        ENDF object containing MF4 data
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number to compare
    orders : int or list of int, optional
        Legendre orders to compare. If None, compares all common orders
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (10, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range (min, max) for the ENDF curve (used for dense grid evaluation)
    mg_marker : bool, default True
        Whether to include markers on multigroup steps
    show_uncertainties : bool, default False
        Whether to show ±σ bands when covariance data are available
    uncertainty_sigma : float, default 1.0
        Sigma level for uncertainty bands
    endf_native : bool, default False
        If True, sample ENDF coefficients on the native MF4 grid; otherwise use a dense
        log-spaced grid across the requested energy range.
    title : str or None, default "default"
        Plot title. If "default", generates a title from nuclide and MT.
        If None, no title is displayed. If a string, uses that as the title.
    mg_style : dict, optional
        Style overrides for the MG series. Keys are PlotData attributes:
        ``color``, ``linestyle``, ``linewidth``, ``label``, ``alpha``, etc.
    endf_style : dict, optional
        Style overrides for the ENDF series (same keys as *mg_style*).
    **styling_kwargs
        Additional styling arguments

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the comparison

    Examples
    --------
    >>> fig = plot_mg_vs_endf_comparison(
    ...     mg_covmat, endf,
    ...     nuclide=92235, mt=2,
    ...     orders=[1, 2, 3]
    ... )

    See Also
    --------
    plot_mg_vs_endf_uncertainties_comparison : Compare uncertainties
    """
    from kika._utils import symbol_to_zaid, zaid_to_symbol
    from kika.plotting import LegendreCoeffPlotData, PlotData, UncertaintyBand
    
    # Convert nuclide to isotope (ZAID) if string
    if isinstance(nuclide, str):
        isotope = symbol_to_zaid(nuclide)
    else:
        isotope = nuclide

    endf_obj = endf
    endf_cov = None
    mf4_mt = None
    if hasattr(endf_obj, "mf"):
        # Expect full ENDF object
        if 4 not in endf_obj.mf or mt not in endf_obj.mf[4].mt:
            raise ValueError("ENDF object does not contain MF4 data for the requested MT")
        mf4_mt = endf_obj.mf[4].mt[mt]
        if 34 in endf_obj.mf and mt in endf_obj.mf[34].mt:
            try:
                endf_cov = endf_obj.mf[34].mt[mt].to_ang_covmat()
            except Exception:
                endf_cov = None
    elif isinstance(endf_obj, MF34CovMat):
        # Covariance provided, but we still need MF4 for coefficients
        endf_cov = endf_obj
    else:
        raise ValueError("endf_covmat must be an ENDF object with MF4/34 data or an MF34CovMat paired with MF4 data")

    if mf4_mt is None:
        raise ValueError("ENDF MF4 data is required to plot the continuous coefficients")

    # Determine orders to plot
    if orders is None:
        mg_orders = set(l for (iso, m, l) in mg_covmat.legendre_coefficients.keys()
                        if iso == isotope and m == mt)
        available_endf_orders: set[int] = set()
        coeff_lists = getattr(mf4_mt, "legendre_coefficients", None)
        if coeff_lists:
            max_len = max(len(c) for c in coeff_lists)
            available_endf_orders = set(range(max_len + 1))
        orders = sorted(mg_orders & available_endf_orders) if available_endf_orders else sorted(mg_orders)
    elif isinstance(orders, int):
        orders = [orders]

    if not orders:
        raise ValueError(f"No Legendre orders found for isotope={isotope}, MT={mt}")

    # Normalize scale parameter
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        use_log_x = True
    elif scale_normalized in ("lin", "linear"):
        use_log_x = False
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")

    # Determine energy range for dense ENDF evaluation
    mg_edges = np.asarray(mg_covmat.energy_grid, dtype=float)
    if energy_range is None:
        energy_range = (mg_edges[0], mg_edges[-1]) if mg_edges.size else None

    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)
    color_cycle = getattr(builder, "_colors", None) or []

    def _build_endf_coeff_data(order: int, color: Optional[str]) -> Optional[LegendreCoeffPlotData]:
        """Create ENDF coefficient curve on native or dense grid."""
        if mf4_mt is None:
            return None
        try:
            label = f"ENDF L={order}"
            if not endf_native and hasattr(mf4_mt, "extract_legendre_coefficients") and energy_range is not None:
                e_min, e_max = energy_range
                dense_e = np.logspace(np.log10(e_min), np.log10(e_max), 1000)
                coeffs_dict = mf4_mt.extract_legendre_coefficients(
                    energy=dense_e,
                    max_legendre_order=order,
                    out_of_range="zero",
                )
                coeff_vals = coeffs_dict.get(order)
                data = LegendreCoeffPlotData(
                    x=dense_e,
                    y=coeff_vals,
                    order=order,
                    isotope=getattr(mf4_mt, "isotope", None),
                    mt=mt,
                    label=label,
                    plot_type="line",
                )
            else:
                data = mf4_mt.to_plot_data(order=order, label=label)
            data.linestyle = "--"
            if color is not None and data.color is None:
                data.color = color
            return data
        except Exception as exc:
            print(f"Warning: Could not build ENDF data for L={order}: {exc}")
            return None

    def _build_endf_uncertainty_band(
        order: int,
        color: Optional[str],
        nominal: Optional[LegendreCoeffPlotData],
    ) -> Optional[UncertaintyBand]:
        """Create UncertaintyBand from MF34 covariance if available."""
        if endf_cov is None or not hasattr(endf_cov, "get_uncertainties_for_legendre_coefficient"):
            return None
        unc_info = endf_cov.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
        if not unc_info:
            return None

        energies_unc = np.asarray(unc_info["energies"], dtype=float)
        rel_unc = np.asarray(unc_info["uncertainties"], dtype=float)
        is_relative = unc_info.get("is_relative", True)

        if not is_relative:
            # Convert absolute to relative using nominal coefficients when possible
            if nominal is None:
                return None
            coeff_x = np.asarray(nominal.x, dtype=float)
            coeff_y = np.asarray(nominal.y, dtype=float)
            if len(coeff_y) == len(coeff_x) + 1:
                coeff_y = coeff_y[:-1]
            centers_unc = np.sqrt(energies_unc[:-1] * energies_unc[1:])
            coeff_interp = np.interp(centers_unc, coeff_x, coeff_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_unc = np.where(coeff_interp != 0, rel_unc / np.abs(coeff_interp), np.nan)

        if rel_unc.size == 0:
            return None

        # Build the band on the dense nominal curve grid so it follows
        # the pointwise shape instead of looking blocky/stepped.
        if nominal is None:
            return None

        nom_x = np.asarray(nominal.x, dtype=float)
        nom_y = np.asarray(nominal.y, dtype=float)
        if len(nom_y) == len(nom_x) + 1:
            nom_y = nom_y[:-1]

        # Restrict to MF34 energy range
        mask = (nom_x >= energies_unc[0]) & (nom_x <= energies_unc[-1])
        if not np.any(mask):
            return None
        band_x = nom_x[mask]
        band_y = nom_y[mask]

        # Piecewise-constant lookup: map each dense point to its MF34 bin
        bin_idx = np.searchsorted(energies_unc, band_x, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(rel_unc) - 1)
        rel_unc_dense = rel_unc[bin_idx]

        y_lower = band_y * (1.0 - uncertainty_sigma * rel_unc_dense)
        y_upper = band_y * (1.0 + uncertainty_sigma * rel_unc_dense)

        band = UncertaintyBand(
            x=band_x,
            y_lower=y_lower,
            y_upper=y_upper,
            sigma=uncertainty_sigma,
            color=color,
        )
        return band

    # Add data for each order from both sources (ENDF first so MG draws on top)
    for idx, order in enumerate(orders):
        try:
            mg_label = f"MG L={order}"
            mg_data, mg_unc = mg_covmat.to_plot_data(
                nuclide=nuclide,
                mt=mt,
                order=order,
                sigma=uncertainty_sigma,
                label=mg_label,
            )
            base_color = getattr(mg_data, "color", None) if mg_data is not None else None
            if base_color is None and color_cycle:
                base_color = color_cycle[idx % len(color_cycle)]

            # --- ENDF (background layer) ---
            endf_data = _build_endf_coeff_data(order, base_color)
            if endf_data is not None and endf_style:
                for attr, val in endf_style.items():
                    if val is not None and hasattr(endf_data, attr):
                        setattr(endf_data, attr, val)
            endf_color = endf_data.color if endf_data is not None else base_color
            endf_band = _build_endf_uncertainty_band(order, endf_color, endf_data) if show_uncertainties else None
            if endf_data is not None:
                builder.add_data(
                    endf_data,
                    uncertainty=endf_band,
                )

            # --- MG (foreground layer) ---
            if mg_data is not None:
                mg_data.color = base_color
                if mg_style:
                    for attr, val in mg_style.items():
                        if val is not None and hasattr(mg_data, attr):
                            setattr(mg_data, attr, val)
                mg_color = mg_data.color
                if mg_unc is not None and getattr(mg_unc, "color", None) is None:
                    mg_unc.color = mg_color
                builder.add_data(
                    mg_data,
                    uncertainty=mg_unc if show_uncertainties else None,
                )
                if mg_marker:
                    try:
                        centers = np.sqrt(mg_edges[:-1] * mg_edges[1:])
                        coeffs = mg_data.y[:-1]
                        scatter = PlotData(
                            x=centers,
                            y=coeffs,
                            plot_type="scatter",
                            marker="o",
                            markersize=4,
                            color=mg_color,
                            label=None,
                            alpha=getattr(mg_data, "alpha", None),
                        )
                        builder.add_data(scatter)
                    except Exception:
                        pass

        except (ValueError, KeyError) as e:
            print(f"Warning: Could not compare L={order}: {e}")
            continue

    # Configure scales and labels
    builder.set_scales(log_x=use_log_x, log_y=False)
    if energy_range is not None:
        builder.set_limits(x_lim=energy_range)
    
    # Set title based on parameter
    if title == "default":
        plot_title = f"{zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide} MT={mt} - MG vs ENDF Comparison"
    else:
        plot_title = title
    
    builder.set_labels(
        x_label="Energy (eV)",
        y_label="Legendre Coefficient",
        title=plot_title
    )
    builder.set_legend(loc=legend_loc)
    
    # Build and return
    fig = builder.build()
    return fig


def plot_mg_vs_endf_uncertainties_comparison(
    mg_covmat: MGMF34CovMat,
    endf: Union[MF34CovMat, object],
    nuclide: Union[int, str],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    *,
    sigma: float = 1.0,
    style: str = "light",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    uncertainty_type: str = "relative",
    mg_marker: bool = True,
    title: Optional[str] = "default",
    mg_style: Optional[dict] = None,
    endf_style: Optional[dict] = None,
    **styling_kwargs
) -> plt.Figure:
    """
    Compare uncertainties between multigroup and continuous ENDF data.

    This modern implementation uses PlotBuilder to compare uncertainty profiles.

    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        The multigroup MF34 covariance matrix object
    endf : ENDF object or MF34CovMat
        ENDF object containing MF34/MF4 data, or MF34CovMat for covariance-only comparison
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number to compare
    orders : int or list of int, optional
        Legendre orders to compare. If None, compares all common orders
    sigma : float, default 1.0
        Sigma level for uncertainties
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (10, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range for plotting (limits x-axis)
    uncertainty_type : str, default "relative"
        Currently only 'relative' (%) is supported.
    mg_marker : bool, default True
        Whether to show markers on the MG uncertainty curve.
    title : str or None, default "default"
        Plot title. If "default", generates a title from nuclide and MT.
        If None, no title is displayed. If a string, uses that as the title.
    mg_style : dict, optional
        Style overrides for the MG series. Keys are PlotData attributes:
        ``color``, ``linestyle``, ``linewidth``, ``label``, ``alpha``, etc.
    endf_style : dict, optional
        Style overrides for the ENDF series (same keys as *mg_style*).
    **styling_kwargs
        Additional styling arguments

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the uncertainty comparison

    Examples
    --------
    >>> fig = plot_mg_vs_endf_uncertainties_comparison(
    ...     mg_covmat, endf,
    ...     nuclide=92235, mt=2,
    ...     orders=[1, 2, 3]
    ... )

    See Also
    --------
    plot_mg_vs_endf_comparison : Compare Legendre coefficients
    """
    from kika._utils import symbol_to_zaid, zaid_to_symbol
    from kika.plotting import LegendreUncertaintyPlotData
    
    if uncertainty_type.lower() != "relative":
        raise ValueError("Only relative uncertainties (percentage) are supported in this comparison.")

    # Convert nuclide to isotope (ZAID) if string
    if isinstance(nuclide, str):
        isotope = symbol_to_zaid(nuclide)
    else:
        isotope = nuclide

    endf_obj = endf
    endf_cov = None
    mf4_mt = None
    if hasattr(endf_obj, "mf"):
        if 34 in endf_obj.mf and mt in endf_obj.mf[34].mt:
            try:
                endf_cov = endf_obj.mf[34].mt[mt].to_ang_covmat()
            except Exception:
                endf_cov = None
        if 4 in endf_obj.mf and mt in endf_obj.mf[4].mt:
            mf4_mt = endf_obj.mf[4].mt[mt]
    elif isinstance(endf_obj, MF34CovMat):
        endf_cov = endf_obj
    else:
        endf_cov = None

    # Determine orders to plot
    if orders is None:
        mg_orders = set(l for (iso, m, l) in mg_covmat.legendre_coefficients.keys()
                        if iso == isotope and m == mt)
        endf_orders = set()
        if endf_cov is not None:
            for iso_r, mt_r, l_r in zip(endf_cov.isotope_rows, endf_cov.reaction_rows, endf_cov.l_rows):
                if iso_r == isotope and mt_r == mt:
                    endf_orders.add(l_r)
        orders = sorted(mg_orders & endf_orders) if endf_orders else sorted(mg_orders)
    elif isinstance(orders, int):
        orders = [orders]

    if not orders:
        raise ValueError(f"No common Legendre orders found for isotope={isotope}, MT={mt}")

    # Normalize scale parameter
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        use_log_x = True
    elif scale_normalized in ("lin", "linear"):
        use_log_x = False
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")

    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)
    color_cycle = getattr(builder, "_colors", None) or []

    def _build_endf_uncertainty(order: int, color: Optional[str]) -> Optional[LegendreUncertaintyPlotData]:
        if endf_cov is None:
            return None
        unc_info = endf_cov.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
        if not unc_info:
            return None

        energies = np.asarray(unc_info["energies"], dtype=float)
        unc_values = np.asarray(unc_info["uncertainties"], dtype=float) * sigma
        is_relative = unc_info.get("is_relative", True)

        centers = np.sqrt(energies[:-1] * energies[1:])

        if is_relative:
            rel_unc_pct = unc_values * 100.0
        else:
            if mf4_mt is None or not hasattr(mf4_mt, "extract_legendre_coefficients"):
                return None
            coeffs = mf4_mt.extract_legendre_coefficients(
                energy=centers,
                max_legendre_order=order,
                out_of_range="zero",
            ).get(order)
            if coeffs is None:
                return None
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_unc_pct = np.where(coeffs != 0, (unc_values / np.abs(coeffs)) * 100.0, np.nan)

        data = LegendreUncertaintyPlotData(
            x=energies,
            y=np.append(rel_unc_pct, rel_unc_pct[-1]),
            label=f"ENDF L={order}",
            order=order,
            isotope=zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide,
            mt=mt,
            uncertainty_type="relative",
            linestyle="--",
            color=color,
            plot_type="step",
        )
        return data

    for idx, order in enumerate(orders):
        try:
            mg_unc = mg_covmat._create_uncertainty_plotdata(
                isotope=isotope,
                mt=mt,
                order=order,
                sigma=sigma,
                use_centers=True,
                as_percentage=True,
                label=f"MG L={order}",
                **styling_kwargs,
            )
            base_color = getattr(mg_unc, "color", None) if mg_unc is not None else None
            if base_color is None and color_cycle:
                base_color = color_cycle[idx % len(color_cycle)]

            # --- ENDF (background layer) ---
            endf_unc = _build_endf_uncertainty(order, base_color)
            if endf_unc is not None and endf_style:
                for attr, val in endf_style.items():
                    if val is not None and hasattr(endf_unc, attr):
                        setattr(endf_unc, attr, val)
            if endf_unc is not None:
                builder.add_data(endf_unc)

            # --- MG (foreground layer) ---
            if mg_unc is not None:
                mg_unc.color = base_color
                mg_unc.plot_type = "step"
                if mg_marker:
                    mg_unc.marker = 'o'
                    mg_unc.markersize = 4
                if mg_style:
                    for attr, val in mg_style.items():
                        if val is not None and hasattr(mg_unc, attr):
                            setattr(mg_unc, attr, val)
                builder.add_data(mg_unc)

        except (ValueError, KeyError) as e:
            print(f"Warning: Could not compare uncertainties for L={order}: {e}")
            continue

    builder.set_scales(log_x=use_log_x, log_y=False)
    if energy_range is not None:
        builder.set_limits(x_lim=energy_range)
    
    # Set title based on parameter
    if title == "default":
        plot_title = f"{zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide} MT={mt} - Uncertainty Comparison"
    else:
        plot_title = title
    
    builder.set_labels(
        x_label="Energy (eV)",
        y_label="Relative Uncertainty (%)",
        title=plot_title
    )
    builder.set_legend(loc=legend_loc)
    
    fig = builder.build()
    return fig


def plot_mg_covariance_heatmap(
    mg_covmat: MGMF34CovMat,
    nuclide: Union[int, str],
    mt: int,
    legendre_coeffs: Union[int, List[int], Tuple[int, int]],
    *,
    matrix_type: str = "corr",
    covariance_type: str = "rel",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = False,
    show_energy_ticks: bool = True,
    cmap: Optional[str] = None,
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "default",
) -> plt.Figure:
    """
    Draw a covariance/correlation heatmap for multigroup MF34 data.
    
    This modern implementation uses the PlotBuilder infrastructure for cleaner code.

    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        The multigroup MF34 covariance matrix object
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number
    legendre_coeffs : int, list of int, or tuple of (L1, L2)
        Legendre coefficient(s) to plot
    matrix_type : str, default "corr"
        Matrix type: "corr"/"correlation" or "cov"/"covariance"
    covariance_type : str, default "rel"
        Covariance type: "rel" (relative) or "abs" (absolute) - only for matrix_type="cov"
    figsize : tuple, default (6, 6)
        Figure size
    dpi : int, default 300
        DPI for figure
    font_family : str, default "serif"
        Font family
    vmax, vmin : float, optional
        Color scale limits
    show_uncertainties : bool, default False
        Whether to show uncertainty plots above the heatmap
    show_energy_ticks : bool, default True
        Whether to show energy group ticks and labels on the heatmap axes
    cmap : str, optional
        Colormap name
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range for filtering
    title : str or None, default "default"
        Plot title. If "default", generates a title from nuclide and MT.
        If None, no title is displayed. If a string, uses that as the title.

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap

    Examples
    --------
    >>> fig = plot_mg_covariance_heatmap(
    ...     mg_covmat, nuclide=92235, mt=2,
    ...     legendre_coeffs=[1, 2, 3]
    ... )

    See Also
    --------
    plot_mg_legendre_coefficients : Plot Legendre coefficients
    """
    from kika._utils import symbol_to_zaid
    
    # Convert nuclide to isotope (ZAID) if string
    if isinstance(nuclide, str):
        isotope = symbol_to_zaid(nuclide)
    else:
        isotope = nuclide
    
    # Normalize matrix_type
    matrix_type_normalized = matrix_type.lower()
    if matrix_type_normalized in ("corr", "correlation"):
        matrix_type_normalized = "corr"
    elif matrix_type_normalized in ("cov", "covariance"):
        matrix_type_normalized = "cov"
    else:
        raise ValueError(f"matrix_type must be 'corr'/'correlation' or 'cov'/'covariance', got '{matrix_type}'")
    
    # Normalize scale
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        scale_normalized = "log"
    elif scale_normalized in ("lin", "linear"):
        scale_normalized = "linear"
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")
    
    # Prepare the heatmap data using to_heatmap_data
    heatmap_data = mg_covmat.to_heatmap_data(
        nuclide=nuclide,
        mt=mt,
        legendre_coeffs=legendre_coeffs,
        matrix_type=matrix_type_normalized,
        covariance_type=covariance_type,
        show_energy_ticks=show_energy_ticks,
        scale=scale_normalized,
        energy_range=energy_range,
    )
    
    # Override colormap if provided
    if cmap is not None:
        heatmap_data.cmap = cmap

    # Warn about deprecated vmin/vmax parameters
    import warnings
    if vmin is not None or vmax is not None:
        warnings.warn(
            "vmin and vmax parameters are deprecated and will be ignored. "
            "Auto-scaling is now used for colorbar normalization.",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Set title based on parameter
    from kika._utils import zaid_to_symbol
    if title == "default":
        matrix_label = "Correlation" if matrix_type_normalized == "corr" else "Covariance"
        nuclide_str = zaid_to_symbol(isotope) if isinstance(isotope, int) else nuclide
        heatmap_data.label = f"{nuclide_str} MT={mt} - MG {matrix_label}"
    elif title is not None:
        heatmap_data.label = title
    else:
        heatmap_data.label = None
    
    # Create the plot using HeatmapBuilder (always use light style for heatmaps)
    builder = HeatmapBuilder(style="light", figsize=figsize, dpi=dpi, font_family=font_family)
    fig = builder.add_heatmap(
        heatmap_data,
        show_uncertainties=show_uncertainties,
    ).build()

    # If uncertainties panels are shown, lower the title slightly for better layout
    if show_uncertainties and fig is not None and heatmap_data.label:
        try:
            fig.suptitle(heatmap_data.label, y=0.93)
        except Exception:
            pass
    
    return fig
