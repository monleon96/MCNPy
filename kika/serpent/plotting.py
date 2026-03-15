from typing import Any, Mapping, Optional, Sequence, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from kika.serpent.sens import SensitivityFile
from kika.plotting.styles import format_axes, setup_plot_style, finalize_plot
from kika._utils import zaid_to_symbol


IndexLike = Union[int, str]
IndexSeq = Sequence[IndexLike]


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _material_indices(sf: SensitivityFile, materials: Optional[Union[IndexLike, IndexSeq]]) -> List[int]:
    if materials is None:
        return list(range(sf.n_materials))
    out = []
    for m in _as_list(materials):
        out.append(sf._material_index(m))
    return out


def _nuclide_indices(sf: SensitivityFile, nuclides: Optional[Union[IndexLike, IndexSeq]]) -> List[int]:
    if nuclides is None:
        return list(range(sf.n_nuclides))
    out = []
    for n in _as_list(nuclides):
        out.append(sf._nuclide_index(n))
    return out


def _collect_perturbations(
    sf: SensitivityFile,
    mt: Optional[Union[int, Sequence[int]]] = None,
    leg: Optional[Union[int, Sequence[int]]] = None,
    leg_channel: Optional[str] = None,
) -> List[int]:
    """
    Build perturbation index list from MT numbers and/or Legendre orders.
    - mt: int or list[int] of MT values (e.g., 1, 2, 102, ...). 
      Note: MT 400X are automatically recognized as Legendre moments.
      If empty list or None, returns ALL perturbations.
      Negative MT values exclude those reactions: mt=[-1, -2] excludes MT 1 and 2.
    - leg: int or list[int] of Legendre orders (L), e.g., 1, 2, 3
      These are converted to MT 400X format internally.
    - leg_channel: deprecated (Legendre moments are elastic only), kept for backward compatibility
    If both mt and leg are None or empty, returns ALL perturbations.
    """
    def _to_set_with_sign(x):
        """Convert to set, preserving sign information. Returns (positive_set, negative_set) or (None, None)."""
        if x is None:
            return None, None
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return None, None
            pos = set()
            neg = set()
            for v in x:
                v_int = int(v)
                if v_int < 0:
                    neg.add(-v_int)  # Store as positive value
                else:
                    pos.add(v_int)
            return (pos if pos else None), (neg if neg else None)
        else:
            v_int = int(x)
            if v_int < 0:
                return None, {-v_int}
            else:
                return {v_int}, None

    mt_pos, mt_neg = _to_set_with_sign(mt)
    leg_set = None
    if leg is not None:
        if isinstance(leg, (list, tuple, np.ndarray)):
            if len(leg) > 0:
                leg_set = set(int(v) for v in leg)
        else:
            leg_set = {int(leg)}

    # Convert Legendre orders to MT 400X format and merge with mt_pos
    if leg_set is not None:
        leg_as_mt = {4000 + L for L in leg_set}  # L=1 -> MT=4001, L=2 -> MT=4002, etc.
        if mt_pos is not None:
            mt_pos = mt_pos | leg_as_mt
        else:
            mt_pos = leg_as_mt

    # Case 1: No filters at all -> return all perturbations
    if mt_pos is None and mt_neg is None:
        return list(range(sf.n_perturbations))

    # Case 2: Only exclusions (negative MT values) -> start with all, then exclude
    if mt_pos is None and mt_neg is not None:
        idxs: List[int] = []
        for p in sf.perturbations:
            if p.mt is not None and p.mt not in mt_neg:
                idxs.append(p.index)
        return idxs

    # Case 3: Only inclusions (positive MT values) -> include only those
    if mt_pos is not None and mt_neg is None:
        idxs = []
        for p in sf.perturbations:
            if p.mt is not None and p.mt in mt_pos:
                idxs.append(p.index)
        # Remove duplicates while preserving order
        seen = set()
        uniq = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq

    # Case 4: Both inclusions and exclusions -> include from mt_pos, but exclude mt_neg
    idxs = []
    for p in sf.perturbations:
        if p.mt is not None and p.mt in mt_pos and p.mt not in mt_neg:
            idxs.append(p.index)
    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _default_label(
    sf: SensitivityFile, 
    response: str, 
    mi: int, 
    zi: int, 
    pi: int,
    show_response: bool = True,
    show_material: bool = True,
    show_nuclide: bool = True,
    show_pert: bool = True,
) -> str:
    """Generate legend label showing only varying parameters."""
    parts = []
    if show_response:
        parts.append(response)
    if show_material:
        parts.append(sf.materials[mi].name)
    if show_nuclide:
        znum = sf.nuclides[zi].zai
        parts.append(zaid_to_symbol(znum))
    if show_pert:
        plab = sf.perturbations[pi].short_label or sf.perturbations[pi].raw_label
        parts.append(plab)
    return " | ".join(parts) if parts else "Sensitivity"


def _generate_title(
    sf: SensitivityFile,
    resp_list: List[str],
    m_inds: List[int],
    z_inds: List[int],
    p_inds: List[int],
) -> str:
    """Generate an intelligent title based on what's being plotted."""
    parts = []
    
    # Add response if only one
    if len(resp_list) == 1:
        parts.append(f"Response: {resp_list[0]}")
    else:
        parts.append(f"{len(resp_list)} Responses")
    
    # Add material if only one
    if len(m_inds) == 1:
        parts.append(f"Material: {sf.materials[m_inds[0]].name}")
    
    # Add nuclide if only one
    if len(z_inds) == 1:
        znum = sf.nuclides[z_inds[0]].zai
        parts.append(f"Nuclide: {zaid_to_symbol(znum)}")
    
    return ", ".join(parts)


def _clip_range(
    edges: np.ndarray,
    vals: np.ndarray,
    rel: np.ndarray,
    widths: np.ndarray,
    energy_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip arrays to an energy range. Returns (edges_c, vals_c, rel_c, widths_c)."""
    if energy_range is None:
        return edges, vals, rel, widths

    emin, emax = energy_range
    if emin >= emax:
        raise ValueError("energy_range must satisfy emin < emax")

    low = edges[:-1]
    high = edges[1:]
    mask = (high > emin) & (low < emax)
    if not np.any(mask):
        return np.array([emin, emax], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - 1 - np.argmax(mask[::-1]))

    edges_c = edges[i0:i1 + 2].astype(float).copy()
    edges_c[0] = max(edges_c[0], float(emin))
    edges_c[-1] = min(edges_c[-1], float(emax))

    vals_c = vals[i0:i1 + 1]
    rel_c = rel[i0:i1 + 1]
    widths_c = widths[i0:i1 + 1]
    return edges_c, vals_c, rel_c, widths_c


def plot_energy_sensitivity(
    sf: SensitivityFile,
    responses: Union[str, Sequence[str]],
    *,
    materials: Optional[Union[IndexLike, IndexSeq]] = None,
    nuclides: Optional[Union[IndexLike, IndexSeq]] = None,
    mt: Optional[Union[int, Sequence[int]]] = None,
    leg: Optional[Union[int, Sequence[int]]] = None,
    leg_channel: Optional[str] = None,
    per_lethargy: bool = True,
    errorbars: bool = True,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    energy_range: Optional[Tuple[float, float]] = None,
    legend: bool = True,
    label_fmt: Optional[str] = None,
    title: Optional[str] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 6),
):
    """
    Step-plot energy-dependent sensitivities for one or multiple responses and selections.

    Parameters
    ----------
    sf : SensitivityFile
        Parsed sensitivity file object.
    responses : str | list[str]
        One response full-name (e.g., 'sens_ratio_BIN_0') or a list of them.
    materials, nuclides :
        Single index/name/ZAI or a list. If None -> plot all.
        - materials: int index or material name (str)
        - nuclides: int index, ZAI (int), or 'ZAI<num>' string
    mt : int | list[int] | None
        MT numbers to plot (e.g., 1 for total, 2, 102, ...). If None, plot all non-zero.
        Negative values exclude those MT numbers: mt=[-1, -2] excludes MT 1 and 2.
        Can mix positive and negative: mt=[1, 2, -18] includes MT 1, 2 but excludes MT 18.
    leg : int | list[int] | None
        Legendre orders to plot (e.g., 1, 2, 3). If None, not used.
    leg_channel : str | None
        Optional channel filter for legendre (e.g., 'ela'). If None, include all channels.
    per_lethargy : bool
        If True (default), values and errors are divided by lethargy widths.
    errorbars : bool
        Draw vertical error bars at bin midpoints (default True).
    ax : matplotlib.axes.Axes
        Optional axes to draw on; if None, a new figure/axes is created using setup_plot_style.
    logx : bool
        Use log scale on x-axis (default False).
    energy_range : (emin, emax) or None
        If provided, clip the plot range to [emin, emax].
    legend : bool
        Show legend (default True).
    label_fmt : str | None
        Optional legend label template. Fields: {response}, {material}, {nuclide}, {pert}.
    title : str | None
        Optional custom title for the plot. If None and style is not 'paper', 
        an intelligent title is generated showing response, material, and nuclide 
        (when only one of each is plotted).
    style : str
        Plot style key (integrates with your _plot_settings).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if isinstance(responses, str):
        resp_list = [responses]
    else:
        resp_list = list(responses)

    m_inds = _material_indices(sf, materials)
    z_inds = _nuclide_indices(sf, nuclides)
    p_inds = _collect_perturbations(sf, mt=mt, leg=leg, leg_channel=leg_channel)

    # Track all-zero sensitivities (MT numbers that have all zero values)
    all_zero_mts = set()

    # default y label up-front (so it exists even if no lines drawn)
    y_label = "Sensitivity per lethargy" if per_lethargy else "Sensitivity"
    
    # Determine which parameters vary (for intelligent legend generation)
    show_response_in_label = len(resp_list) > 1
    show_material_in_label = len(m_inds) > 1
    show_nuclide_in_label = len(z_inds) > 1
    # Always show perturbation (reaction) label in legend
    show_pert_in_label = True  # Changed from: len(p_inds) > 1

    # Setup plot style and get axes if not provided
    plot_settings = None
    if ax is None:
        plot_settings = setup_plot_style(style=style, figsize=figsize, ax=ax)
        ax = plot_settings['ax']
        fig = plot_settings['fig']
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key().get('color', ['C0'])
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key().get('color', ['C0'])
        fig = ax.get_figure()

    full_edges = sf.energy_grid
    full_widths = sf.lethargy_widths

    # Define line styles that will cycle for different responses
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    
    # Counter for perturbations (to assign consistent colors)
    perturbation_counter = 0

    for resp_idx, resp in enumerate(resp_list):
        # Get line style for this response
        linestyle = linestyles[resp_idx % len(linestyles)]
        
        # Reset perturbation counter for consistent colors across responses
        temp_counter = 0
        
        for mi in m_inds:
            for zi in z_inds:
                for pi in p_inds:
                    # Access the raw sensitivity data directly using the specific perturbation index
                    sset, b = sf._locate_set_and_bin(resp)
                    vals, rel = sset.get_ed_by_index(b, mi, zi, pi, None)
                    vals = np.asarray(vals, dtype=float).reshape(-1)
                    rel = np.asarray(rel, dtype=float).reshape(-1)
                    if vals.size != full_edges.size - 1:
                        raise RuntimeError("Unexpected shape from get_energy_dependent; expected length equals number of energy bins.")

                    # Check if all sensitivities are zero
                    if np.all(vals == 0.0):
                        # Track the MT number for this perturbation
                        pert_mt = sf.perturbations[pi].mt
                        if pert_mt is not None:
                            all_zero_mts.add(pert_mt)
                        continue

                    # Clip to energy_range (if any)
                    edges, v, r, w = _clip_range(full_edges, vals, rel, full_widths, energy_range)
                    if v.size == 0:
                        continue

                    if per_lethargy:
                        w_safe = np.where(w > 0.0, w, np.nan)
                        y = v / w_safe
                        yerr = np.abs(y) * r
                    else:
                        y = v
                        yerr = np.abs(v) * r

                    y_step = np.r_[y, y[-1]]

                    if label_fmt:
                        lbl = label_fmt.format(
                            response=resp,
                            material=sf.materials[mi].name,
                            nuclide=sf.nuclides[zi].zai,
                            pert=sf.perturbations[pi].short_label or sf.perturbations[pi].raw_label,
                        )
                    else:
                        lbl = _default_label(
                            sf, resp, mi, zi, pi,
                            show_response=show_response_in_label,
                            show_material=show_material_in_label,
                            show_nuclide=show_nuclide_in_label,
                            show_pert=show_pert_in_label,
                        )

                    # Use consistent color based on perturbation, but different line style per response
                    color = colors[temp_counter % len(colors)]
                    ax.step(edges, y_step, where="post", label=lbl, color=color, linestyle=linestyle)

                    if errorbars:
                        mids = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
                        ax.errorbar(mids, y, yerr=yerr, fmt="none", linewidth=1, capsize=2, zorder=3, color=color)

                    temp_counter += 1

    ax = format_axes(
        ax=ax,
        style=style,
        use_log_scale=logx,
        is_energy_axis=True,
        x_label="Energy (MeV)",
        y_label=y_label,
        legend_loc='best' if legend else None
    )

    if energy_range is not None:
        ax.set_xlim(energy_range)
    
    # Add title: always show if explicitly provided, otherwise only for non-paper styles
    if title is not None:
        # Use custom title if provided (even for 'paper' style)
        ax.set_title(title)
    elif style != 'paper':
        # Generate intelligent default title only for non-paper styles
        default_title = _generate_title(sf, resp_list, m_inds, z_inds, p_inds)
        ax.set_title(default_title)

    # Print warning if there are all-zero sensitivities
    if all_zero_mts:
        mt_list = sorted(all_zero_mts)
        print(f"The following sensitivities are all 0: MT {mt_list}")

    if plot_settings is not None:
        finalize_plot(fig, ax)

    return ax
