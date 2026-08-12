"""
Covariance matrix plotting functions using the modern PlotBuilder infrastructure.

This module provides convenience wrapper functions for plotting covariance heatmaps
using the refactored plotting system with to_heatmap_data() methods and PlotBuilder.

The functions maintain backward compatibility with the original API while using
the new, cleaner implementation.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Union, Sequence, Tuple, List, Optional
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.legendre_covariance import LegendreCovariance
from kika.plotting.heatmap_builder import HeatmapBuilder
from kika.plotting.plot_builder import PlotBuilder


def plot_covariance_heatmap(
    covmat: CrossSectionCovariance,
    nuclide: Union[int, str, List[Union[int, str]]],
    mt: Union[int, Sequence[int], Tuple[int, int]],
    *,
    matrix_type: str = "corr",
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 300,
    font_family: str = "serif",
    show_uncertainties: bool = True,
    show_energy_ticks: bool = True,
    show_block_labels: bool = True,
    show_colorbar: bool = True,
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    norm: Optional[Normalize] = None,
    colorbar_label: Optional[str] = None,
    title: Optional[str] = "default",
    title_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    energy_tick_fontsize: Optional[float] = None,
    block_label_fontsize: Optional[float] = None,
    colorbar_fontsize: Optional[float] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Draw a covariance or correlation matrix heatmap for one or more isotopes and
    one or more MT reactions, with optional uncertainty plots shown above the heatmap columns.

    This function uses the modern PlotBuilder infrastructure for cleaner, more
    maintainable code while maintaining the same API as the original implementation.

    Parameters
    ----------
    covmat : CrossSectionCovariance
        The covariance matrix object
    nuclide : int, str, or list of int/str
        Isotope identifier(s). Can be:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
        - List of ZAIDs or strings for multi-isotope heatmaps (e.g., ['Fe54', 'Fe56'])
    mt : int, sequence of int, or tuple of (row_mt, col_mt)
        MT reaction number(s). Can be:
        - Single int: diagonal block for that MT
        - Sequence of ints: diagonal blocks for those MTs
        - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
          (not supported for multi-isotope heatmaps)
    matrix_type : str, default "corr"
        Type of matrix to plot: "corr"/"correlation" for correlation matrix,
        or "cov"/"covariance" for covariance matrix
    figsize : tuple, default (6, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    show_uncertainties : bool, default True
        Whether to show uncertainty plots above the heatmap
    show_energy_ticks : bool, default True
        Whether to show energy tick marks on secondary axes (top/right)
    show_block_labels : bool, default True
        Whether to show block labels (MT numbers)
    show_colorbar : bool, default True
        Whether to show the colorbar
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear'
    energy_range : tuple of float, optional
        Energy range (min, max) for filtering. Values in eV.
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'RdYlGn')
    norm : matplotlib.colors.Normalize, optional
        Custom normalization for the colormap
    colorbar_label : str, optional
        Override colorbar label text
    title : str or None, default "default"
        Title for the plot. Use "default" for auto-generated title from
        heatmap_data.label, None to hide the title, or a custom string.
    title_fontsize : float, optional
        Font size for the title
    tick_fontsize : float, optional
        Font size for tick labels
    energy_tick_fontsize : float, optional
        Font size for energy tick labels on secondary axes
    block_label_fontsize : float, optional
        Font size for MT block labels
    colorbar_fontsize : float, optional
        Font size for colorbar label
    show : bool, default False
        Whether to display the figure immediately

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots

    Raises
    ------
    ValueError
        If no data found for specified isotope/MT combination

    Examples
    --------
    Plot a single MT diagonal block:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=92235, mt=2)
    >>> plt.savefig("u235_elastic.png")

    Plot with nuclide string:

    >>> fig = plot_covariance_heatmap(covmat, nuclide='U235', mt=[2, 18, 102])

    Plot covariance matrix instead of correlation:

    >>> fig = plot_covariance_heatmap(covmat, nuclide='Fe56', mt=2, matrix_type='cov')

    Plot an off-diagonal block between two MTs:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=92235, mt=(2, 18))

    Plot multi-isotope heatmap showing cross-isotope correlations:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=['Fe54', 'Fe56'], mt=[2, 18])

    See Also
    --------
    plot_covariance_difference_heatmap : Plot differences between two covariance matrices
    plot_mf34_covariance_heatmap : Plot MF34 angular distribution covariances
    """
    # Note: nuclide conversion and multi-isotope handling is done in to_heatmap_data()

    # Normalize matrix_type
    matrix_type_normalized = matrix_type.lower()
    if matrix_type_normalized in ("corr", "correlation"):
        matrix_type_normalized = "corr"
    elif matrix_type_normalized in ("cov", "covariance"):
        matrix_type_normalized = "cov"
    else:
        raise ValueError(f"matrix_type must be 'corr'/'correlation' or 'cov'/'covariance', got '{matrix_type}'")

    # Normalize scale parameter
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        scale_normalized = "log"
    elif scale_normalized in ("lin", "linear"):
        scale_normalized = "linear"
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")

    # Prepare the heatmap data using the new infrastructure
    heatmap_data = covmat.to_heatmap_data(
        nuclide=nuclide,
        mt=mt,
        matrix_type=matrix_type_normalized,
        scale=scale_normalized,
        energy_range=energy_range,
    )

    # Build styling overrides for add_heatmap()
    styling_overrides = {}
    if cmap is not None:
        styling_overrides["cmap"] = cmap
    if norm is not None:
        styling_overrides["norm"] = norm
    if colorbar_label is not None:
        styling_overrides["colorbar_label"] = colorbar_label

    # Create the plot using HeatmapBuilder (always use light style for heatmaps)
    builder = HeatmapBuilder(style="light", figsize=figsize, dpi=dpi, font_family=font_family)

    # Set title via set_labels() so the builder handles placement
    if title == "default":
        # Let the builder use heatmap_data.label as default
        pass
    elif title is None:
        builder.set_labels(title="")
    else:
        builder.set_labels(title=title)

    # Set font sizes if provided
    if title_fontsize is not None or tick_fontsize is not None:
        builder.set_font_sizes(title=title_fontsize, ticks=tick_fontsize)

    fig = builder.add_heatmap(
        heatmap_data,
        show_uncertainties=show_uncertainties,
        show_energy_ticks=show_energy_ticks,
        show_block_labels=show_block_labels,
        show_colorbar=show_colorbar,
        energy_tick_fontsize=energy_tick_fontsize,
        block_label_fontsize=block_label_fontsize,
        colorbar_fontsize=colorbar_fontsize,
        **styling_overrides,
    ).build(show=show)

    return fig


def plot_mf34_covariance_heatmap(
    mf34_covmat: LegendreCovariance,
    nuclide: Union[int, str],
    mt: int,
    legendre_coeffs: Union[int, List[int], Tuple[int, int]],
    *,
    matrix_type: str = "corr",
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 300,
    font_family: str = "serif",
    show_uncertainties: bool = False,
    show_energy_ticks: bool = True,
    show_block_labels: bool = True,
    show_colorbar: bool = True,
    cmap: Optional[str] = None,
    norm: Optional[Normalize] = None,
    colorbar_label: Optional[str] = None,
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "default",
    title_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    energy_tick_fontsize: Optional[float] = None,
    block_label_fontsize: Optional[float] = None,
    colorbar_fontsize: Optional[float] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Draw a covariance/correlation heatmap for MF34 angular distribution data
    with energy-proportional blocks and optional uncertainty panels.

    This function handles the more complex MF34 covariance structure where each
    Legendre coefficient can have a different energy grid.

    Parameters
    ----------
    mf34_covmat : LegendreCovariance
        The MF34 covariance matrix object
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number
    legendre_coeffs : int, list of int, or tuple of (L1, L2)
        Legendre coefficient(s) to plot. Can be:
        - Single int: diagonal block for that L
        - List of ints: diagonal blocks for those L values
        - Tuple of (L1, L2): off-diagonal block between L1 and L2
    matrix_type : str, default "corr"
        Matrix type to plot: "corr"/"correlation" for correlation matrix,
        or "cov"/"covariance" for covariance matrix
    figsize : tuple, default (6, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    show_uncertainties : bool, default False
        Whether to show uncertainty plots above the heatmap
    show_energy_ticks : bool, default True
        Whether to show energy tick marks on secondary axes (top/right)
    show_block_labels : bool, default True
        Whether to show block labels (Legendre orders)
    show_colorbar : bool, default True
        Whether to show the colorbar
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'RdYlGn')
    norm : matplotlib.colors.Normalize, optional
        Custom normalization for the colormap
    colorbar_label : str, optional
        Override colorbar label text
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range (min, max) for filtering. Values in eV.
    title : str or None, default "default"
        Title for the plot. Use "default" for auto-generated title from
        heatmap_data.label, None to hide the title, or a custom string.
    title_fontsize : float, optional
        Font size for the title
    tick_fontsize : float, optional
        Font size for tick labels
    energy_tick_fontsize : float, optional
        Font size for energy tick labels on secondary axes
    block_label_fontsize : float, optional
        Font size for Legendre order block labels
    colorbar_fontsize : float, optional
        Font size for colorbar label
    show : bool, default False
        Whether to display the figure immediately

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots

    Raises
    ------
    ValueError
        If no data found for specified isotope/MT/Legendre combination

    Examples
    --------
    Plot correlation matrix for Legendre coefficients L=1,2,3:

    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide=92235, mt=2,
    ...     legendre_coeffs=[1, 2, 3]
    ... )

    Plot with nuclide string:

    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide='U235', mt=2,
    ...     legendre_coeffs=[1, 2, 3]
    ... )

    Plot covariance for a single Legendre coefficient with uncertainties:

    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide='Fe56', mt=2,
    ...     legendre_coeffs=1, matrix_type="cov",
    ...     show_uncertainties=True
    ... )

    See Also
    --------
    plot_covariance_heatmap : Plot multigroup cross-section covariances
    """
    # Convert nuclide to ZAID if string
    from kika._utils import symbol_to_zaid

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

    # Prepare the heatmap data using the new infrastructure
    heatmap_data = mf34_covmat.to_heatmap_data(
        nuclide=nuclide,
        mt=mt,
        legendre_coeffs=legendre_coeffs,
        matrix_type=matrix_type_normalized,
        scale=scale_normalized,
        energy_range=energy_range,
    )

    # Build styling overrides for add_heatmap()
    styling_overrides = {}
    if cmap is not None:
        styling_overrides["cmap"] = cmap
    if norm is not None:
        styling_overrides["norm"] = norm
    if colorbar_label is not None:
        styling_overrides["colorbar_label"] = colorbar_label

    # Create the plot using HeatmapBuilder (always use light style for heatmaps)
    builder = HeatmapBuilder(style="light", figsize=figsize, dpi=dpi, font_family=font_family)

    # Set title via set_labels() so the builder handles placement
    if title == "default":
        # Let the builder use heatmap_data.label as default
        pass
    elif title is None:
        builder.set_labels(title="")
    else:
        builder.set_labels(title=title)

    # Set font sizes if provided
    if title_fontsize is not None or tick_fontsize is not None:
        builder.set_font_sizes(title=title_fontsize, ticks=tick_fontsize)

    fig = builder.add_heatmap(
        heatmap_data,
        show_uncertainties=show_uncertainties,
        show_energy_ticks=show_energy_ticks,
        show_block_labels=show_block_labels,
        show_colorbar=show_colorbar,
        energy_tick_fontsize=energy_tick_fontsize,
        block_label_fontsize=block_label_fontsize,
        colorbar_fontsize=colorbar_fontsize,
        **styling_overrides,
    ).build(show=show)

    return fig


def plot_mf34_uncertainties(
    mf34_covmat: LegendreCovariance,
    isotope: Union[int, str],
    mt: int,
    legendre_coeffs: Union[int, Sequence[int]],
    *,
    ax: Optional[plt.Axes] = None,
    uncertainty_type: str = "relative",
    style: str = "light",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 100,
    font_family: str = "serif",
    legend_loc: str = "best",
    energy_range: Optional[Tuple[float, float]] = None,
    sigma: float = 1.0,
    title: Optional[str] = "default",
    show: bool = False,
    **styling_kwargs,
) -> plt.Figure:
    """
    Plot MF34 Legendre-coefficient uncertainties for one isotope/MT.

    The MF34 counterpart of :func:`plot_uncertainties`, which serves
    ``CrossSectionCovariance``. One curve per requested Legendre order, drawn
    through ``LegendreCovariance.to_plot_data`` and ``PlotBuilder``.

    Parameters
    ----------
    mf34_covmat : LegendreCovariance
        The MF34 covariance object.
    isotope : int or str
        Isotope identifier — ZAID (``26056``) or symbol (``'Fe56'``).
    mt : int
        Reaction MT number.
    legendre_coeffs : int or sequence of int
        Legendre order(s) to plot. An empty sequence means every order
        available for this isotope/MT.
    ax : plt.Axes, optional
        Axes to draw into. If None, a new figure is created.
    uncertainty_type : {"relative", "absolute"}, default "relative"
        Relative uncertainties are plotted as percentages.
    style : str, default "light"
        ``'light'`` or ``'dark'``. ``'default'`` is accepted as a synonym for
        ``'light'`` because that is what the method wrapping this function has
        always declared as its default; the older style names ('paper',
        'publication', 'presentation') did not survive the move to
        ``PlotBuilder`` and raise.
    sigma : float, default 1.0
        Sigma level applied to the uncertainties.
    energy_range : tuple of float, optional
        (min, max) for the x-axis, in the covariance's own energy unit.
    title : str or None, default "default"
        ``"default"`` auto-generates, ``None`` omits, a string is used as given.
    **styling_kwargs
        Forwarded to ``to_plot_data`` (color, linestyle, linewidth, ...).

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If ``uncertainty_type`` is not recognised, if the isotope/MT pair holds
        no Legendre orders, or if a requested order is not among them.

    Examples
    --------
    >>> mf34 = endf.mf[34].mt[2].to_ang_covmat()
    >>> fig = plot_mf34_uncertainties(mf34, isotope=26056, mt=2,
    ...                               legendre_coeffs=[1, 2, 3])

    See Also
    --------
    plot_uncertainties : the same plot for cross-section covariances
    plot_mf34_covariance_heatmap : MF34 covariance/correlation heatmaps
    """
    from kika._utils import symbol_to_zaid, zaid_to_symbol

    if uncertainty_type not in ("relative", "absolute"):
        raise ValueError(
            f"uncertainty_type must be 'relative' or 'absolute', got {uncertainty_type!r}"
        )

    zaid = symbol_to_zaid(isotope) if isinstance(isotope, str) else int(isotope)

    available = sorted(
        {t[2] for t in mf34_covmat._get_param_triplets() if t[0] == zaid and t[1] == mt}
    )
    if not available:
        raise ValueError(f"No Legendre coefficients found for isotope={zaid}, MT={mt}")

    if isinstance(legendre_coeffs, int):
        requested = [legendre_coeffs]
    else:
        requested = list(legendre_coeffs) or list(available)

    missing = [l for l in requested if l not in available]
    if missing:
        raise ValueError(
            f"Legendre coefficient(s) {missing} not available for isotope={zaid}, "
            f"MT={mt}. Available: {available}"
        )

    builder = PlotBuilder(
        style="light" if style == "default" else style,
        figsize=figsize,
        dpi=dpi,
        font_family=font_family,
        ax=ax,
    )
    builder.set_scales(log_x=True, log_y=False)

    for order in requested:
        _, unc_data = mf34_covmat.to_plot_data(
            nuclide=zaid,
            mt=mt,
            order=order,
            sigma=sigma,
            uncertainty_type=uncertainty_type,
            **styling_kwargs,
        )
        builder.add_data(unc_data)

    if energy_range is not None:
        builder.set_limits(x_lim=(energy_range[0], energy_range[1]))

    if title == "default":
        orders = ",".join(str(l) for l in requested)
        builder.set_labels(title=f"{zaid_to_symbol(zaid)} MT={mt} L={orders} uncertainties")
    elif title is not None:
        builder.set_labels(title=title)

    fig = builder.build(show=show)

    if fig.axes:
        axis = fig.axes[0]
        axis.legend(loc=legend_loc)
        axis.set_xlabel(f"Energy ({mf34_covmat.energy_unit})")
        axis.set_ylabel(
            "Relative Uncertainty (%)"
            if uncertainty_type == "relative"
            else "Absolute Uncertainty"
        )

    return fig


def plot_uncertainties(
    covmat: CrossSectionCovariance,
    nuclide: Union[int, str, Sequence[Union[int, str]]],
    mt: Union[int, Sequence[int]],
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    sigma: float = 1.0,
    style: str = "light",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    xscale: str = "log",
    yscale: str = "linear",
    title: Optional[str] = "default",
    show: bool = False,
    **styling_kwargs
) -> plt.Figure:
    """
    Plot relative uncertainties for one or more (ZAID, MT) pairs from covariance data.

    This modern implementation uses the PlotBuilder infrastructure with to_plot_data()
    for cleaner, more maintainable code.

    Parameters
    ----------
    covmat : CrossSectionCovariance
        The covariance matrix object
    nuclide : int, str, or sequence of int/str
        Isotope ID(s) to plot (e.g., 92235 for U-235, 'U235')
    mt : int or sequence of int
        Reaction MT number(s) to plot
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis in MeV. If None, uses full range.
    sigma : float, default 1.0
        Number of sigma levels for uncertainty (e.g., 1.0 for 1-sigma, 2.0 for 2-sigma)
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    xscale : str, default "log"
        X-axis scale: "log"/"logarithmic" or "lin"/"linear"
    yscale : str, default "linear"
        Y-axis scale: "log"/"logarithmic" or "lin"/"linear"
    title : str or None, default "default"
        Title for the plot. Use "default" for auto-generated title,
        None to hide the title, or a custom string.
    show : bool, default False
        Whether to display the figure immediately
    **styling_kwargs
        Additional styling arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the uncertainty plots

    Examples
    --------
    Plot uncertainties for a single reaction:

    >>> fig = plot_uncertainties(covmat, nuclide=92235, mt=2)

    Plot multiple reactions:

    >>> fig = plot_uncertainties(covmat, nuclide=92235, mt=[2, 18, 102])

    Plot with custom styling:

    >>> fig = plot_uncertainties(
    ...     covmat, nuclide=92235, mt=2,
    ...     sigma=2.0, style='presentation'
    ... )

    See Also
    --------
    plot_multigroup_xs : Plot cross sections with optional uncertainties
    plot_covariance_heatmap : Plot covariance heatmaps
    """
    # Normalize inputs to lists and convert nuclide symbols to ZAID
    from kika._utils import symbol_to_zaid, zaid_to_symbol

    if isinstance(nuclide, (int, str)):
        nuclide_list = [nuclide]
    else:
        nuclide_list = list(nuclide)

    zaid_list: List[int] = []
    for n in nuclide_list:
        if isinstance(n, int):
            zaid_list.append(n)
        elif isinstance(n, str):
            zaid_list.append(symbol_to_zaid(n))
        else:
            raise ValueError(f"Invalid nuclide entry: {n!r}")
    mt_list = [mt] if isinstance(mt, int) else list(mt)

    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)

    # Apply axis scales
    lx = xscale.lower() if isinstance(xscale, str) else str(xscale)
    ly = yscale.lower() if isinstance(yscale, str) else str(yscale)
    if lx in ("log", "logarithmic"):
        log_x = True
    elif lx in ("lin", "linear"):
        log_x = False
    else:
        raise ValueError(f"Invalid xscale '{xscale}'; expected 'log' or 'linear'")

    if ly in ("log", "logarithmic"):
        log_y = True
    elif ly in ("lin", "linear"):
        log_y = False
    else:
        raise ValueError(f"Invalid yscale '{yscale}'; expected 'log' or 'linear'")

    builder.set_scales(log_x=log_x, log_y=log_y)

    # Add uncertainty data for each (zaid, mt) pair
    for z in zaid_list:
        for m in mt_list:
            try:
                # Get uncertainty data using to_plot_data
                _, unc_data = covmat.to_plot_data(nuclide=z, mt=m, sigma=sigma, **styling_kwargs)

                if unc_data is not None:
                    # Add to plot
                    builder.add_data(unc_data)
            except (ValueError, KeyError) as e:
                # Skip if data not available
                print(f"Warning: Could not plot uncertainties for ZAID={z}, MT={m}: {e}")
                continue

    # Set energy range if provided
    if energy_range is not None:
        builder.set_limits(x_lim=(energy_range[0], energy_range[1]))

    # Set title behavior:
    # - title == "default": construct a sensible default title
    # - title is None: explicitly omit title
    # - title is a string: use provided string
    if title == "default":
        # Construct default title from nuclide(s) and MT(s)
        try:
            names = [zaid_to_symbol(z) for z in zaid_list]
        except Exception:
            names = [str(z) for z in zaid_list]

        if len(names) == 1:
            nuclide_name = names[0]
        else:
            nuclide_name = ",".join(names)

        if len(mt_list) == 1:
            mt_title = str(mt_list[0])
        else:
            mt_title = ",".join(str(m) for m in mt_list)

        default_title = f"{nuclide_name} Uncertainties MT: {mt_title}"
        builder.set_labels(title=default_title)
    elif title is None:
        # Do not set title (explicitly omit)
        pass
    else:
        builder.set_labels(title=title)

    # Build and configure the plot
    fig = builder.build(show=show)

    # Add legend
    if fig.axes:
        ax = fig.axes[0]
        ax.legend(loc=legend_loc)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Relative Uncertainty (%)")

    return fig


def plot_multigroup_xs(
    covmat: CrossSectionCovariance,
    nuclide: Union[int, str, Sequence[Union[int, str]]],
    mt: Union[int, Sequence[int]],
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    show_uncertainties: bool = False,
    sigma: float = 1.0,
    style: str = "light",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    xscale: str = "log",
    yscale: str = "linear",
    title: Optional[str] = "default",
    show: bool = False,
    **styling_kwargs
) -> plt.Figure:
    """
    Plot multigroup cross sections with optional uncertainty bands.

    This modern implementation uses the PlotBuilder infrastructure with to_plot_data()
    for cleaner, more maintainable code.

    Parameters
    ----------
    covmat : CrossSectionCovariance
        The covariance matrix object
    nuclide : int, str, or sequence of int/str
        Isotope ID(s) to plot (e.g., 92235 for U-235, 'U235')
    mt : int or sequence of int
        Reaction MT number(s) to plot
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis in MeV. If None, uses full range.
    show_uncertainties : bool, default False
        Whether to show uncertainty bands around cross sections
    sigma : float, default 1.0
        Number of sigma levels for uncertainty bands (e.g., 1.0 for 1-sigma, 2.0 for 2-sigma)
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    xscale : str, default "log"
        X-axis scale: "log"/"logarithmic" or "lin"/"linear"
    yscale : str, default "linear"
        Y-axis scale: "log"/"logarithmic" or "lin"/"linear"
    title : str or None, default "default"
        Title for the plot. Use "default" for auto-generated title,
        None to hide the title, or a custom string.
    show : bool, default False
        Whether to display the figure immediately
    **styling_kwargs
        Additional styling arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the cross section plots

    Examples
    --------
    Plot cross sections for a single reaction:

    >>> fig = plot_multigroup_xs(covmat, nuclide=92235, mt=2)

    Plot multiple reactions with uncertainty bands:

    >>> fig = plot_multigroup_xs(
    ...     covmat, nuclide=92235, mt=[2, 18, 102],
    ...     show_uncertainties=True
    ... )

    See Also
    --------
    plot_uncertainties : Plot only uncertainties
    plot_covariance_heatmap : Plot covariance heatmaps
    """
    # Normalize inputs to lists and convert nuclide symbols to ZAID
    from kika._utils import symbol_to_zaid, zaid_to_symbol

    if isinstance(nuclide, (int, str)):
        nuclide_list = [nuclide]
    else:
        nuclide_list = list(nuclide)

    zaid_list: List[int] = []
    for n in nuclide_list:
        if isinstance(n, int):
            zaid_list.append(n)
        elif isinstance(n, str):
            zaid_list.append(symbol_to_zaid(n))
        else:
            raise ValueError(f"Invalid nuclide entry: {n!r}")

    mt_list = [mt] if isinstance(mt, int) else list(mt)

    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)

    # Apply axis scales
    lx = xscale.lower() if isinstance(xscale, str) else str(xscale)
    ly = yscale.lower() if isinstance(yscale, str) else str(yscale)
    if lx in ("log", "logarithmic"):
        log_x = True
    elif lx in ("lin", "linear"):
        log_x = False
    else:
        raise ValueError(f"Invalid xscale '{xscale}'; expected 'log' or 'linear'")

    if ly in ("log", "logarithmic"):
        log_y = True
    elif ly in ("lin", "linear"):
        log_y = False
    else:
        raise ValueError(f"Invalid yscale '{yscale}'; expected 'log' or 'linear'")

    builder.set_scales(log_x=log_x, log_y=log_y)

    # Add cross section data for each (zaid, mt) pair
    for z in zaid_list:
        for m in mt_list:
            try:
                # Get XS and uncertainty data using to_plot_data
                xs_data, unc_data = covmat.to_plot_data(nuclide=z, mt=m, sigma=sigma, **styling_kwargs)

                if xs_data is not None:
                    # Add to plot with optional uncertainty
                    if show_uncertainties and unc_data is not None:
                        builder.add_data(xs_data, uncertainty=unc_data)
                    else:
                        builder.add_data(xs_data)
            except (ValueError, KeyError) as e:
                # Skip if data not available
                print(f"Warning: Could not plot XS for ZAID={z}, MT={m}: {e}")
                continue

    # Set energy range if provided
    if energy_range is not None:
        builder.set_limits(x_lim=(energy_range[0], energy_range[1]))

    # Set title behavior (same semantics as uncertainties plot)
    if title == "default":
        try:
            names = [zaid_to_symbol(z) for z in zaid_list]
        except Exception:
            names = [str(z) for z in zaid_list]

        if len(names) == 1:
            nuclide_name = names[0]
        else:
            nuclide_name = ",".join(names)

        if len(mt_list) == 1:
            mt_title = str(mt_list[0])
        else:
            mt_title = ",".join(str(m) for m in mt_list)

        default_title = f"{nuclide_name} Cross Sections MT: {mt_title}"
        builder.set_labels(title=default_title)
    elif title is None:
        pass
    else:
        builder.set_labels(title=title)

    # Build and configure the plot
    fig = builder.build(show=show)

    # Add legend and labels
    if fig.axes:
        ax = fig.axes[0]
        ax.legend(loc=legend_loc)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (barns)")

    return fig
