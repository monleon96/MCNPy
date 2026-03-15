"""
HeatmapBuilder class for composing and rendering heatmap plots.

This module provides the HeatmapBuilder class that specializes in creating
covariance and correlation matrix heatmaps with proper energy ticks,
block labels, and uncertainty panels.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings

from .plot_builder import PlotBuilder, _NOT_SET
from .plot_data import (
    HeatmapPlotData,
    CovarianceHeatmapData,
    LegendreHeatmapData
)
from ._backend_utils import _configure_figure_interactivity


class HeatmapBuilder(PlotBuilder):
    """
    Builder class specialized for creating heatmap plots.

    This class extends PlotBuilder with specialized heatmap rendering capabilities,
    including support for covariance/correlation matrices with energy ticks,
    block labels (MT numbers or Legendre orders), and uncertainty panels.

    Examples
    --------
    >>> # Create heatmap from covariance data (uses default title)
    >>> heatmap_data = covmat.to_heatmap_data(nuclide=92235, mt=[2, 18])
    >>> fig = HeatmapBuilder().add_heatmap(heatmap_data).build()

    >>> # Custom title
    >>> fig = (HeatmapBuilder()
    ...        .add_heatmap(heatmap_data)
    ...        .set_labels(title="My Custom Title")
    ...        .build())

    >>> # Hide title
    >>> fig = (HeatmapBuilder()
    ...        .add_heatmap(heatmap_data)
    ...        .set_labels(title="")  # or title=None
    ...        .build())

    >>> # Hide energy ticks and customize font sizes
    >>> fig = (HeatmapBuilder()
    ...        .add_heatmap(heatmap_data,
    ...                     show_energy_ticks=False,
    ...                     block_label_fontsize=14)
    ...        .build())

    >>> # Hide optional elements
    >>> fig = (HeatmapBuilder()
    ...        .add_heatmap(heatmap_data,
    ...                     show_uncertainties=False,
    ...                     show_colorbar=False)
    ...        .build())
    """

    def __init__(
        self,
        style: str = 'light',
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
        projection: Optional[str] = None,
        font_family: str = 'serif',
        notebook_mode: Optional[bool] = None,
        interactive: Optional[bool] = None,
    ):
        """
        Initialize the HeatmapBuilder.

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
        super().__init__(
            style=style,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
            projection=projection,
            font_family=font_family,
            notebook_mode=notebook_mode,
            interactive=interactive,
        )

        # Heatmap-specific configuration (defaults)
        self._heatmap_show_energy_ticks: bool = True
        self._heatmap_show_block_labels: bool = True
        self._heatmap_show_colorbar: bool = True
        self._heatmap_energy_tick_fontsize: Optional[float] = None
        self._heatmap_block_label_fontsize: Optional[float] = None
        self._heatmap_colorbar_fontsize: Optional[float] = None

    def _is_multi_block(self, heatmap_data: 'HeatmapPlotData') -> bool:
        """
        Check if heatmap data contains multiple blocks.

        Parameters
        ----------
        heatmap_data : HeatmapPlotData
            The heatmap data to check

        Returns
        -------
        bool
            True if the heatmap contains multiple MT or Legendre blocks
        """
        block_info = getattr(heatmap_data, 'block_info', None)
        if block_info is None:
            return False

        mts = block_info.get('mts', [])
        legendre = block_info.get('legendre_coeffs', [])
        n_blocks = len(mts) if mts else len(legendre)

        return n_blocks > 1

    def _crop_to_energy_range(
        self,
        heatmap_data: 'HeatmapPlotData',
        energy_lim: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray], Optional[Dict]]:
        """
        Unified method to crop heatmap data to energy range.
        Works for both single-block and multi-block heatmaps.

        Parameters
        ----------
        heatmap_data : HeatmapPlotData
            The heatmap data containing matrix, block info, and energy grids
        energy_lim : tuple
            (min_energy, max_energy) for filtering

        Returns
        -------
        tuple
            (cropped_matrix, new_x_edges, new_y_edges, new_block_info,
             cropped_energy_grid, cropped_uncertainty_data)
        """
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        # Get mask_value for reapplying after cropping
        mask_value = getattr(heatmap_data, 'mask_value', None)

        # Get the matrix data - use raw matrix_data for cropping to avoid
        # masked array issues with off-diagonal blocks
        M = heatmap_data.matrix_data

        lo, hi = energy_lim
        if lo > hi:
            lo, hi = hi, lo

        def _find_groups_in_range(energy_grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
            """Find indices of groups that overlap with energy range."""
            G = len(energy_grid) - 1
            in_range = []
            for i in range(G):
                e_lo_bin = energy_grid[i]
                e_hi_bin = energy_grid[i + 1]
                if (e_hi_bin >= lo) and (e_lo_bin <= hi):
                    in_range.append(i)
            if not in_range:
                return np.array([], dtype=int), 0, 0
            return np.array(in_range, dtype=int), in_range[0], in_range[-1] + 1

        scale = getattr(heatmap_data, 'scale', 'log')
        uncertainty_data = getattr(heatmap_data, 'uncertainty_data', None)

        # Handle CovarianceHeatmapData
        if isinstance(heatmap_data, CovarianceHeatmapData):
            energy_grid = heatmap_data.energy_grid
            if energy_grid is None:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, {}, None, uncertainty_data

            block_info = getattr(heatmap_data, 'block_info', None) or {}
            mts = block_info.get('mts', [])
            G = block_info.get('G', len(energy_grid) - 1)
            is_multi_block = len(mts) > 1 and heatmap_data.is_diagonal

            # Find which groups are in range
            groups_idx, start_g, end_g = _find_groups_in_range(energy_grid)
            if len(groups_idx) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data

            G_new = len(groups_idx)
            cropped_energy_grid = energy_grid[start_g:end_g + 1]

            if is_multi_block:
                # Multi-block: crop each block
                cropped_blocks = []
                for i, mt in enumerate(mts):
                    block_start = i * G
                    row_indices = block_start + groups_idx
                    block_rows = []
                    for j, mt2 in enumerate(mts):
                        block_start_col = j * G
                        col_indices = block_start_col + groups_idx
                        sub_block = M[np.ix_(row_indices, col_indices)]
                        block_rows.append(sub_block)
                    cropped_blocks.append(np.hstack(block_rows))
                cropped_M = np.vstack(cropped_blocks)

                # Calculate coordinate edges for multi-block
                if scale == 'log':
                    cropped_edges_local = np.log10(np.maximum(cropped_energy_grid, 1e-300))
                    cropped_edges_local = cropped_edges_local - cropped_edges_local[0]
                else:
                    cropped_edges_local = cropped_energy_grid - cropped_energy_grid[0]

                width = cropped_edges_local[-1] if len(cropped_edges_local) > 1 else 1.0

                new_x_edges_parts = []
                new_energy_ranges = {}
                for i, mt in enumerate(mts):
                    block_offset = i * width
                    if i == 0:
                        new_x_edges_parts.append(cropped_edges_local + block_offset)
                    else:
                        new_x_edges_parts.append((cropped_edges_local + block_offset)[1:])
                    new_energy_ranges[mt] = (block_offset, block_offset + width)

                new_x_edges = np.concatenate(new_x_edges_parts)
                new_y_edges = new_x_edges.copy()

                new_block_info = {
                    'mts': mts,
                    'G': G_new,
                    'energy_ranges': new_energy_ranges,
                    'cropped_energy_grids': {mt: cropped_energy_grid for mt in mts},
                }
            else:
                # Single-block: simple crop
                cropped_M = M[np.ix_(groups_idx, groups_idx)]

                if scale == 'log':
                    new_x_edges = np.log10(np.maximum(cropped_energy_grid, 1e-300))
                    new_x_edges = new_x_edges - new_x_edges[0]
                else:
                    new_x_edges = cropped_energy_grid - cropped_energy_grid[0]

                new_y_edges = new_x_edges.copy()

                # Build proper block_info for single-block with updated energy_ranges
                width = new_x_edges[-1] if len(new_x_edges) > 1 else 1.0
                new_block_info = {
                    'mts': mts if mts else list(uncertainty_data.keys()) if uncertainty_data else [],
                    'G': G_new,
                    'energy_ranges': {},
                    'cropped_energy_grids': {},
                }
                # Set energy_ranges for each MT key
                for mt_key in new_block_info['mts']:
                    new_block_info['energy_ranges'][mt_key] = (0.0, width)
                    new_block_info['cropped_energy_grids'][mt_key] = cropped_energy_grid

            # Crop uncertainty data
            cropped_uncertainty = None
            if uncertainty_data:
                cropped_uncertainty = {}
                for key, sigma in uncertainty_data.items():
                    if len(groups_idx) <= len(sigma):
                        cropped_uncertainty[key] = sigma[groups_idx]
                    else:
                        cropped_uncertainty[key] = sigma

            # Reapply mask to cropped matrix (NaN values + mask_value if set)
            nan_mask = ~np.isfinite(cropped_M)
            if mask_value is not None:
                nan_mask = nan_mask | (cropped_M == mask_value)
            cropped_M = np.ma.masked_where(nan_mask, cropped_M)

            return cropped_M, new_x_edges, new_y_edges, new_block_info, cropped_energy_grid, cropped_uncertainty

        # Handle LegendreHeatmapData
        elif isinstance(heatmap_data, LegendreHeatmapData):
            block_info = getattr(heatmap_data, 'block_info', None) or {}
            legendre_list = block_info.get('legendre_coeffs', heatmap_data.legendre_coeffs)
            energy_grids = heatmap_data.energy_grids
            is_multi_block = len(legendre_list) > 1

            if not energy_grids or len(legendre_list) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data

            # Get the first energy grid as reference
            first_L = legendre_list[0]
            first_grid = energy_grids.get(first_L)
            if first_grid is None:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data

            groups_idx, start_g, end_g = _find_groups_in_range(first_grid)
            if len(groups_idx) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data

            cropped_energy_grid = first_grid[start_g:end_g + 1]
            G_new = len(groups_idx)

            if is_multi_block and heatmap_data.is_diagonal:
                # Multi-block diagonal MF34
                ranges_dict = block_info.get('ranges', {})
                cropped_blocks_diag = []
                new_ranges = {}
                new_energy_ranges = {}
                new_G_per_L = {}
                new_x_edges_parts = []
                new_cropped_energy_grids = {}  # Store per-block cropped energy grids
                new_groups_idx_per_L = {}  # Store per-block group indices
                current_pos = 0.0

                for l_val in legendre_list:
                    energy_grid = energy_grids.get(l_val)
                    idx_range = ranges_dict.get(l_val)

                    if energy_grid is None:
                        continue

                    groups_idx_L, start_g_L, end_g_L = _find_groups_in_range(energy_grid)
                    if len(groups_idx_L) == 0:
                        continue

                    G_l_new = len(groups_idx_L)
                    new_G_per_L[l_val] = G_l_new

                    cropped_grid = energy_grid[start_g_L:end_g_L + 1]
                    new_cropped_energy_grids[l_val] = cropped_grid  # Store cropped grid
                    new_groups_idx_per_L[l_val] = groups_idx_L  # Store group indices

                    if idx_range is not None:
                        start_idx, end_idx = idx_range
                        block_indices = start_idx + groups_idx_L
                        sub_block = M[np.ix_(block_indices, block_indices)]
                    else:
                        sub_block = np.zeros((G_l_new, G_l_new))

                    cropped_blocks_diag.append(sub_block)

                    if scale == 'log':
                        edges_local = np.log10(np.maximum(cropped_grid, 1e-300))
                        edges_local = edges_local - edges_local[0]
                    else:
                        edges_local = cropped_grid - cropped_grid[0]

                    width = edges_local[-1] if len(edges_local) > 1 else 1.0
                    edges_global = edges_local + current_pos

                    if len(new_x_edges_parts) == 0:
                        new_x_edges_parts.append(edges_global)
                    else:
                        new_x_edges_parts.append(edges_global[1:])

                    new_ranges[l_val] = (len(cropped_blocks_diag) - 1, len(cropped_blocks_diag))
                    new_energy_ranges[l_val] = (current_pos, current_pos + width)
                    current_pos += width

                if not cropped_blocks_diag:
                    return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data

                total_size = sum(b.shape[0] for b in cropped_blocks_diag)
                cropped_M = np.zeros((total_size, total_size))

                # Calculate cumulative positions for placing blocks
                cumulative_positions = {}
                current_block_pos = 0
                valid_l_vals = [l for l in legendre_list if l in new_G_per_L]
                for l_val in valid_l_vals:
                    cumulative_positions[l_val] = current_block_pos
                    current_block_pos += new_G_per_L[l_val]

                # Place ALL blocks (diagonal and off-diagonal)
                for i, l_row in enumerate(valid_l_vals):
                    for j, l_col in enumerate(valid_l_vals):
                        row_groups_idx = new_groups_idx_per_L.get(l_row)
                        col_groups_idx = new_groups_idx_per_L.get(l_col)

                        if row_groups_idx is None or col_groups_idx is None:
                            continue

                        row_range = ranges_dict.get(l_row)
                        col_range = ranges_dict.get(l_col)

                        if row_range is None or col_range is None:
                            continue

                        # Calculate source indices in original matrix
                        row_start_orig = row_range[0] if isinstance(row_range, tuple) else 0
                        col_start_orig = col_range[0] if isinstance(col_range, tuple) else 0

                        row_indices_orig = row_start_orig + row_groups_idx
                        col_indices_orig = col_start_orig + col_groups_idx

                        # Extract the sub-block from original matrix
                        sub_block = M[np.ix_(row_indices_orig, col_indices_orig)]

                        # Calculate target position in cropped matrix
                        row_pos = cumulative_positions[l_row]
                        col_pos = cumulative_positions[l_col]

                        # Place the block
                        cropped_M[row_pos:row_pos+len(row_groups_idx),
                                  col_pos:col_pos+len(col_groups_idx)] = sub_block

                new_x_edges = np.concatenate(new_x_edges_parts) if new_x_edges_parts else heatmap_data.x_edges
                new_y_edges = new_x_edges.copy()

                new_block_info = {
                    'legendre_coeffs': legendre_list,
                    'ranges': new_ranges,
                    'energy_ranges': new_energy_ranges,
                    'G_per_L': new_G_per_L,
                    'cropped_energy_grids': new_cropped_energy_grids,
                    'groups_idx_per_L': new_groups_idx_per_L,
                }

                # Reapply mask to cropped matrix (multi-block diagonal) - NaN + mask_value
                nan_mask = ~np.isfinite(cropped_M)
                if mask_value is not None:
                    nan_mask = nan_mask | (cropped_M == mask_value)
                cropped_M = np.ma.masked_where(nan_mask, cropped_M)
            elif not heatmap_data.is_diagonal:
                # Off-diagonal block: L1 vs L2 with potentially different energy grids
                # Extract row and column Legendre coefficients
                row_l = legendre_list[0]
                col_l = legendre_list[1]
                
                # Get energy grids for each L
                row_energy_grid = energy_grids.get(row_l)
                col_energy_grid = energy_grids.get(col_l)
                
                if row_energy_grid is None or col_energy_grid is None:
                    return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data
                
                # Find groups in range separately for rows and columns
                row_groups_idx, row_start_g, row_end_g = _find_groups_in_range(row_energy_grid)
                col_groups_idx, col_start_g, col_end_g = _find_groups_in_range(col_energy_grid)
                
                if len(row_groups_idx) == 0 or len(col_groups_idx) == 0:
                    return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None, uncertainty_data
                
                # Crop matrix using separate row and column indices (rectangular)
                cropped_M = M[np.ix_(row_groups_idx, col_groups_idx)]
                
                # Build separate edges for rows (y-axis) and columns (x-axis)
                cropped_row_grid = row_energy_grid[row_start_g:row_end_g + 1]
                cropped_col_grid = col_energy_grid[col_start_g:col_end_g + 1]
                
                if scale == 'log':
                    new_y_edges = np.log10(np.maximum(cropped_row_grid, 1e-300))
                    new_y_edges = new_y_edges - new_y_edges[0]
                    new_x_edges = np.log10(np.maximum(cropped_col_grid, 1e-300))
                    new_x_edges = new_x_edges - new_x_edges[0]
                else:
                    new_y_edges = cropped_row_grid - cropped_row_grid[0]
                    new_x_edges = cropped_col_grid - cropped_col_grid[0]
                
                # Build block_info for off-diagonal with separate dimensions
                y_width = new_y_edges[-1] if len(new_y_edges) > 1 else 1.0
                x_width = new_x_edges[-1] if len(new_x_edges) > 1 else 1.0
                
                new_block_info = {
                    'legendre_coeffs': [row_l, col_l],
                    'cropped_energy_grids': {row_l: cropped_row_grid, col_l: cropped_col_grid},
                    'groups_idx_per_L': {row_l: row_groups_idx, col_l: col_groups_idx},
                    'energy_ranges': {row_l: (0.0, y_width), col_l: (0.0, x_width)},
                }
                
                # Use first available energy grid for return value (convention)
                cropped_energy_grid = cropped_row_grid
            else:
                # Single-block diagonal MF34
                cropped_M = M[np.ix_(groups_idx, groups_idx)]

                if scale == 'log':
                    new_x_edges = np.log10(np.maximum(cropped_energy_grid, 1e-300))
                    new_x_edges = new_x_edges - new_x_edges[0]
                else:
                    new_x_edges = cropped_energy_grid - cropped_energy_grid[0]

                new_y_edges = new_x_edges.copy()
                width = new_x_edges[-1] if len(new_x_edges) > 1 else 1.0

                # Build proper block_info for single-block with updated energy_ranges
                new_block_info = {}
                if legendre_list and len(legendre_list) > 0:
                    l_key = legendre_list[0]
                    new_block_info = {
                        'legendre_coeffs': [l_key],
                        'cropped_energy_grids': {l_key: cropped_energy_grid},
                        'groups_idx_per_L': {l_key: groups_idx},
                        'energy_ranges': {l_key: (0.0, width)},
                    }

            # Crop uncertainty data
            cropped_uncertainty = None
            if uncertainty_data:
                cropped_uncertainty = {}
                if is_multi_block and heatmap_data.is_diagonal:
                    # For multi-block, use per-block group indices
                    groups_idx_per_L = new_block_info.get('groups_idx_per_L', {})
                    for key, sigma in uncertainty_data.items():
                        key_groups_idx = groups_idx_per_L.get(key)
                        if key_groups_idx is not None and len(key_groups_idx) > 0 and len(key_groups_idx) <= len(sigma):
                            cropped_uncertainty[key] = sigma[key_groups_idx]
                        else:
                            cropped_uncertainty[key] = sigma
                else:
                    # For single-block MF34 or CovarianceHeatmapData, use per-key or common group indices
                    groups_idx_per_L = new_block_info.get('groups_idx_per_L', {})
                    for key, sigma in uncertainty_data.items():
                        # Try to get per-key group indices first (for MF34)
                        key_groups_idx = groups_idx_per_L.get(key) if groups_idx_per_L else None
                        if key_groups_idx is not None and len(key_groups_idx) > 0 and len(key_groups_idx) <= len(sigma):
                            cropped_uncertainty[key] = sigma[key_groups_idx]
                        elif len(groups_idx) <= len(sigma):
                            cropped_uncertainty[key] = sigma[groups_idx]
                        else:
                            cropped_uncertainty[key] = sigma

            # Reapply mask to cropped matrix (MF34 single-block or after multi-block diagonal) - NaN + mask_value
            nan_mask = ~np.isfinite(cropped_M)
            if mask_value is not None:
                nan_mask = nan_mask | (cropped_M == mask_value)
            cropped_M = np.ma.masked_where(nan_mask, cropped_M)

            return cropped_M, new_x_edges, new_y_edges, new_block_info, cropped_energy_grid, cropped_uncertainty

        # Fallback for unknown types
        return M, heatmap_data.x_edges, heatmap_data.y_edges, {}, None, uncertainty_data

    def _crop_blocks_to_energy_range(
        self,
        M: np.ndarray,
        heatmap_data: 'HeatmapPlotData',
        energy_lim: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        """
        Crop matrix data to energy range and recalculate coordinates.

        For multi-block heatmaps, this crops each block to only include
        energy groups within the specified range, then recalculates the
        coordinate edges so the cropped data fills the original block space.

        Parameters
        ----------
        M : np.ndarray
            The matrix data
        heatmap_data : HeatmapPlotData
            The heatmap data containing block information and energy grids
        energy_lim : tuple
            (min_energy, max_energy) for filtering

        Returns
        -------
        tuple
            (cropped_matrix, new_x_edges, new_y_edges, new_block_info, cropped_energy_grid)
        """
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        block_info = getattr(heatmap_data, 'block_info', None)
        if block_info is None:
            return M, heatmap_data.x_edges, heatmap_data.y_edges, {}, None

        lo, hi = energy_lim
        if lo > hi:
            lo, hi = hi, lo

        def _find_groups_in_range(energy_grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
            """Find indices of groups that overlap with energy range."""
            G = len(energy_grid) - 1
            in_range = []
            for i in range(G):
                e_lo_bin = energy_grid[i]
                e_hi_bin = energy_grid[i + 1]
                if (e_hi_bin >= lo) and (e_lo_bin <= hi):
                    in_range.append(i)
            if not in_range:
                return np.array([], dtype=int), 0, 0
            return np.array(in_range, dtype=int), in_range[0], in_range[-1] + 1

        if isinstance(heatmap_data, CovarianceHeatmapData):
            energy_grid = heatmap_data.energy_grid
            if energy_grid is None:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

            mts = block_info.get('mts', [])
            G = block_info.get('G', 0)
            original_energy_ranges = block_info.get('energy_ranges', {})

            if G == 0 or len(mts) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

            # Find which groups are in range
            groups_idx, start_g, end_g = _find_groups_in_range(energy_grid)
            if len(groups_idx) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

            G_new = len(groups_idx)

            # Crop energy grid
            cropped_energy_grid = energy_grid[start_g:end_g + 1]

            # Build cropped matrix by extracting relevant groups from each block
            cropped_blocks = []
            for i, mt in enumerate(mts):
                block_start = i * G
                # Extract rows for this block that are in range
                row_indices = block_start + groups_idx
                block_rows = []
                for j, mt2 in enumerate(mts):
                    block_start_col = j * G
                    col_indices = block_start_col + groups_idx
                    sub_block = M[np.ix_(row_indices, col_indices)]
                    block_rows.append(sub_block)
                cropped_blocks.append(np.hstack(block_rows))

            cropped_M = np.vstack(cropped_blocks)

            # Calculate new coordinate edges
            # Each block should still occupy the same visual range
            scale = getattr(heatmap_data, 'scale', 'log')
            if scale == 'log':
                cropped_edges_local = np.log10(np.maximum(cropped_energy_grid, 1e-300))
                cropped_edges_local = cropped_edges_local - cropped_edges_local[0]
            else:
                cropped_edges_local = cropped_energy_grid - cropped_energy_grid[0]

            width = cropped_edges_local[-1]

            # Build new x_edges for all blocks
            new_x_edges_parts = []
            new_energy_ranges = {}
            for i, mt in enumerate(mts):
                block_offset = i * width
                if i == 0:
                    new_x_edges_parts.append(cropped_edges_local + block_offset)
                else:
                    new_x_edges_parts.append((cropped_edges_local + block_offset)[1:])
                new_energy_ranges[mt] = (block_offset, block_offset + width)

            new_x_edges = np.concatenate(new_x_edges_parts)
            new_y_edges = new_x_edges.copy()

            new_block_info = {
                'mts': mts,
                'G': G_new,
                'energy_ranges': new_energy_ranges,
            }

            return cropped_M, new_x_edges, new_y_edges, new_block_info, cropped_energy_grid

        elif isinstance(heatmap_data, LegendreHeatmapData):
            legendre_list = block_info.get('legendre_coeffs', heatmap_data.legendre_coeffs)
            ranges_dict = block_info.get('ranges', {})
            energy_grids = heatmap_data.energy_grids

            if not energy_grids or len(legendre_list) == 0:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

            # For MF34, each Legendre order may have different energy grids
            # Crop each block independently
            scale = getattr(heatmap_data, 'scale', 'log')

            cropped_blocks_diag = []
            new_ranges = {}
            new_energy_ranges = {}
            new_G_per_L = {}
            new_energy_grids = {}
            new_x_edges_parts = []
            current_pos = 0.0

            for l_val in legendre_list:
                energy_grid = energy_grids.get(l_val)
                idx_range = ranges_dict.get(l_val)

                if energy_grid is None or idx_range is None:
                    continue

                start_idx, end_idx = idx_range

                # Find groups in range for this L
                groups_idx, start_g, end_g = _find_groups_in_range(energy_grid)
                if len(groups_idx) == 0:
                    continue

                G_l_new = len(groups_idx)
                new_G_per_L[l_val] = G_l_new

                # Crop energy grid for this L
                cropped_grid = energy_grid[start_g:end_g + 1]
                new_energy_grids[l_val] = cropped_grid

                # Extract sub-block for diagonal
                block_indices = start_idx + groups_idx
                sub_block = M[np.ix_(block_indices, block_indices)]
                cropped_blocks_diag.append(sub_block)

                # Calculate edges for this block
                if scale == 'log':
                    edges_local = np.log10(np.maximum(cropped_grid, 1e-300))
                    edges_local = edges_local - edges_local[0]
                else:
                    edges_local = cropped_grid - cropped_grid[0]

                width = edges_local[-1] if len(edges_local) > 1 else 1.0
                edges_global = edges_local + current_pos

                if len(new_x_edges_parts) == 0:
                    new_x_edges_parts.append(edges_global)
                else:
                    new_x_edges_parts.append(edges_global[1:])

                new_ranges[l_val] = (len(cropped_blocks_diag) - 1, len(cropped_blocks_diag))
                new_energy_ranges[l_val] = (current_pos, current_pos + width)
                current_pos += width

            if not cropped_blocks_diag:
                return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

            # Build block-diagonal matrix
            total_size = sum(b.shape[0] for b in cropped_blocks_diag)
            cropped_M = np.zeros((total_size, total_size))
            pos = 0
            for block in cropped_blocks_diag:
                size = block.shape[0]
                cropped_M[pos:pos+size, pos:pos+size] = block
                pos += size

            new_x_edges = np.concatenate(new_x_edges_parts) if new_x_edges_parts else heatmap_data.x_edges
            new_y_edges = new_x_edges.copy()

            new_block_info = {
                'legendre_coeffs': legendre_list,
                'ranges': new_ranges,
                'energy_ranges': new_energy_ranges,
                'G_per_L': new_G_per_L,
            }

            # Return first energy grid as representative
            first_grid = new_energy_grids.get(legendre_list[0]) if legendre_list else None

            return cropped_M, new_x_edges, new_y_edges, new_block_info, first_grid

        return M, heatmap_data.x_edges, heatmap_data.y_edges, block_info, None

    def add_heatmap(
        self,
        heatmap_data: 'HeatmapPlotData',
        show_uncertainties: bool = True,
        show_energy_ticks: bool = True,
        show_block_labels: bool = True,
        show_colorbar: bool = True,
        energy_tick_fontsize: Optional[float] = None,
        block_label_fontsize: Optional[float] = None,
        colorbar_fontsize: Optional[float] = None,
        **styling_overrides
    ) -> 'HeatmapBuilder':
        """
        Add heatmap data to be rendered when build() is called.

        This method stores heatmap data for rendering, following the builder pattern
        consistently with add_data(). Call build() to generate the figure.

        Parameters
        ----------
        heatmap_data : HeatmapPlotData
            Heatmap data object (CovarianceHeatmapData or LegendreHeatmapData)
        show_uncertainties : bool, default True
            Whether to show uncertainty panels above the heatmap
        show_energy_ticks : bool, default True
            Whether to show energy tick marks on secondary axes (top/right)
        show_block_labels : bool, default True
            Whether to show block labels (MT numbers or Legendre orders)
        show_colorbar : bool, default True
            Whether to show the colorbar
        energy_tick_fontsize : float, optional
            Font size for energy tick labels (default: 8)
        block_label_fontsize : float, optional
            Font size for MT/Legendre block labels (default: 11)
        colorbar_fontsize : float, optional
            Font size for colorbar label (default: 12)
        **styling_overrides
            Styling overrides for heatmap rendering. Common options:
            - cmap : str or Colormap - Override colormap (e.g., 'viridis', 'RdBu_r')
            - norm : Normalize - Custom normalization
            - colorbar_label : str - Override colorbar label

        Returns
        -------
        HeatmapBuilder
            Self for method chaining

        Notes
        -----
        Heatmaps and line plots are mutually exclusive. You cannot mix add_data() and
        add_heatmap() on the same builder instance.

        Title control: Use `set_labels(title=...)` to customize the title:
        - Don't call set_labels() -> uses default title from heatmap_data.label
        - `set_labels(title="Custom Title")` -> shows custom title
        - `set_labels(title="")` or `set_labels(title=None)` -> hides the title

        Examples
        --------
        >>> # Basic usage with all defaults (shows default title)
        >>> fig = HeatmapBuilder().add_heatmap(heatmap_data).build()

        >>> # Custom title
        >>> fig = (HeatmapBuilder()
        ...        .add_heatmap(heatmap_data)
        ...        .set_labels(title="My Custom Title")
        ...        .build())

        >>> # Hide title
        >>> fig = (HeatmapBuilder()
        ...        .add_heatmap(heatmap_data)
        ...        .set_labels(title="")
        ...        .build())

        >>> # Custom font sizes
        >>> fig = HeatmapBuilder().add_heatmap(
        ...     heatmap_data,
        ...     energy_tick_fontsize=10,
        ...     block_label_fontsize=14,
        ...     colorbar_fontsize=14
        ... ).build()
        """
        # Check for conflicts with line plot data
        if self._data_list:
            raise ValueError(
                "Cannot mix line plots and heatmaps. "
                "add_heatmap() cannot be called after add_data(). "
                "Create separate builder instances for heatmaps and line plots."
            )

        # Store heatmap data for rendering in build()
        self._heatmap_data = heatmap_data
        self._heatmap_show_uncertainties = show_uncertainties
        self._heatmap_styling_overrides = styling_overrides

        # Store new formatting options
        self._heatmap_show_energy_ticks = show_energy_ticks
        self._heatmap_show_block_labels = show_block_labels
        self._heatmap_show_colorbar = show_colorbar
        self._heatmap_energy_tick_fontsize = energy_tick_fontsize
        self._heatmap_block_label_fontsize = block_label_fontsize
        self._heatmap_colorbar_fontsize = colorbar_fontsize

        return self

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
        # Check if we have heatmap data
        if self._heatmap_data is not None:
            fig = self._build_heatmap()
            if show:
                plt.show()
            return fig

        # If no heatmap data, fall back to parent implementation
        return super().build(show=show)

    def _build_heatmap(self) -> plt.Figure:
        """
        Internal method to render heatmap. Called by build() when heatmap data is present.

        Returns
        -------
        matplotlib.figure.Figure
            The completed heatmap figure
        """
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData, HeatmapPlotData
        from .heatmap_utils import (
            setup_energy_group_ticks,
            setup_energy_group_ticks_single_block,
            format_uncertainty_ticks,
            add_mt_labels_to_heatmap
        )
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm

        # Get stored heatmap data
        heatmap_data = self._heatmap_data
        show_uncertainties = self._heatmap_show_uncertainties
        styling_overrides = self._heatmap_styling_overrides

        # For heatmaps, use manual formatting instead of style system
        plt.rcdefaults()
        figsize = getattr(self, "_figsize_user", None) or self.figsize
        dpi = getattr(self, "_dpi_user", None) or self.dpi

        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2.2,
            'lines.markersize': 7,
            'axes.grid': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'figure.constrained_layout.use': False,
        })

        # Close any pre-existing figure created during __init__
        if hasattr(self, "fig") and getattr(self, "ax", None) is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None
            self.ax = None

        # Determine if we have uncertainty data to show
        has_uncertainties = (
            show_uncertainties and
            hasattr(heatmap_data, 'uncertainty_data') and
            heatmap_data.uncertainty_data is not None and
            len(heatmap_data.uncertainty_data) > 0
        )

        # Check for symmetric matrix (covariance/correlation matrices)
        symmetric_matrix = isinstance(heatmap_data, CovarianceHeatmapData) or (
            isinstance(heatmap_data, LegendreHeatmapData) and getattr(heatmap_data, "is_diagonal", False))

        # Get user-specified energy limits
        user_x_lim = getattr(self, "_x_lim", None)
        user_y_lim = getattr(self, "_y_lim", None)

        # For symmetric matrices, enforce symmetry of limits
        if symmetric_matrix:
            if user_x_lim is None and user_y_lim is not None:
                user_x_lim = user_y_lim
            elif user_y_lim is None and user_x_lim is not None:
                user_y_lim = user_x_lim
            elif user_x_lim is not None and user_y_lim is not None and user_x_lim != user_y_lim:
                lo = min(user_x_lim[0], user_y_lim[0])
                hi = max(user_x_lim[1], user_y_lim[1])
                user_x_lim = (lo, hi)
                user_y_lim = (lo, hi)

        # Determine the effective energy limit for cropping
        energy_lim = user_x_lim if user_x_lim is not None else user_y_lim

        # Create figure and layout
        fig = plt.figure(figsize=figsize, dpi=dpi)

        if has_uncertainties:
            num_panels = len(heatmap_data.uncertainty_data)

            # For off-diagonal MF34 heatmaps, use single panel for all uncertainties
            is_off_diagonal = (
                hasattr(heatmap_data, 'is_diagonal') and
                not heatmap_data.is_diagonal
            )
            if is_off_diagonal:
                num_panels = 1

            fig.set_size_inches(figsize[0], figsize[1] * 1.2)

            gs = GridSpec(2, num_panels if num_panels > 1 else 1, figure=fig,
                         height_ratios=[0.2, 0.8], hspace=0.12, wspace=0.02)

            uncertainty_axes = []
            if num_panels == 1:
                ax_unc = fig.add_subplot(gs[0, :])
                uncertainty_axes.append(ax_unc)
            else:
                for i in range(num_panels):
                    ax_unc = fig.add_subplot(gs[0, i])
                    uncertainty_axes.append(ax_unc)

            ax_heatmap = fig.add_subplot(gs[1, :])
        else:
            ax_heatmap = fig.add_subplot(111)
            uncertainty_axes = None

        # Initialize cropped data variables
        use_cropped_data = False
        cropped_x_edges = None
        cropped_y_edges = None
        cropped_block_info = None
        cropped_energy_grid = None
        cropped_uncertainty_data = None

        # Get the matrix data (will be replaced if cropping)
        if hasattr(heatmap_data, 'get_masked_data'):
            M = heatmap_data.get_masked_data()
        else:
            M = heatmap_data.matrix_data

        # If energy limits are set, crop the data (works for both single and multi-block)
        if energy_lim is not None:
            (cropped_M, cropped_x_edges, cropped_y_edges,
             cropped_block_info, cropped_energy_grid, cropped_uncertainty_data) = self._crop_to_energy_range(
                heatmap_data, energy_lim
            )
            if cropped_M is not None and cropped_M.size > 0:
                M = cropped_M
                use_cropped_data = True

        # Set background color for masked regions
        ax_heatmap.set_facecolor("#F0F0F0")
        ax_heatmap.grid(False, which="both")

        # Apply styling overrides (priority: styling_overrides > heatmap_data attributes > defaults)
        effective_cmap = styling_overrides.get('cmap', heatmap_data.cmap)
        effective_norm = styling_overrides.get('norm', heatmap_data.norm)
        effective_colorbar_label = styling_overrides.get('colorbar_label', heatmap_data.colorbar_label)

        # Setup colormap
        if isinstance(effective_cmap, str):
            cmap = plt.get_cmap(effective_cmap).copy()
        else:
            cmap = effective_cmap

        if hasattr(cmap, 'set_bad'):
            cmap.set_bad(color="#F0F0F0")

        # Handle normalization (auto-detect if not provided)
        if effective_norm is not None:
            norm = effective_norm
        else:
            # Auto-determine normalization
            vmin = np.nanmin(M)
            vmax = np.nanmax(M)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Draw heatmap
        # Use cropped edges if available, otherwise original edges
        plot_x_edges = cropped_x_edges if use_cropped_data and cropped_x_edges is not None else getattr(heatmap_data, "x_edges", None)
        plot_y_edges = cropped_y_edges if use_cropped_data and cropped_y_edges is not None else getattr(heatmap_data, "y_edges", None)

        if plot_x_edges is not None and plot_y_edges is not None:
            X, Y = np.meshgrid(plot_x_edges, plot_y_edges)
            im = ax_heatmap.pcolormesh(X, Y, M, cmap=cmap, norm=norm, shading="flat")

            # Always use full extent of (cropped) data
            x_limits = (plot_x_edges[0], plot_x_edges[-1])
            y_limits = (plot_y_edges[0], plot_y_edges[-1])

            ax_heatmap.set_xlim(x_limits)
            ax_heatmap.set_ylim(y_limits[1], y_limits[0])
        elif getattr(heatmap_data, "extent", None) is not None:
            im = ax_heatmap.imshow(
                M,
                cmap=cmap,
                norm=norm,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
                extent=heatmap_data.extent,
            )

            x_limits = heatmap_data.extent[:2]
            y_limits = heatmap_data.extent[2:]

            ax_heatmap.set_xlim(x_limits)
            ax_heatmap.set_ylim(y_limits)
        else:
            im = ax_heatmap.imshow(M, cmap=cmap, norm=norm,
                                  origin="upper", interpolation="nearest", aspect="auto")

        # Full extents (used for figure-fixed MT/L labels)
        # Use cropped edges if available
        if use_cropped_data and cropped_x_edges is not None:
            full_xlim = (float(cropped_x_edges[0]), float(cropped_x_edges[-1]))
        elif getattr(heatmap_data, "x_edges", None) is not None:
            full_xlim = (float(heatmap_data.x_edges[0]), float(heatmap_data.x_edges[-1]))
        elif getattr(heatmap_data, "extent", None) is not None:
            full_xlim = (float(heatmap_data.extent[0]), float(heatmap_data.extent[1]))
        else:
            full_xlim = ax_heatmap.get_xlim()

        if use_cropped_data and cropped_y_edges is not None:
            full_ylim = (float(cropped_y_edges[0]), float(cropped_y_edges[-1]))
        elif getattr(heatmap_data, "y_edges", None) is not None:
            full_ylim = (float(heatmap_data.y_edges[0]), float(heatmap_data.y_edges[-1]))
        elif getattr(heatmap_data, "extent", None) is not None:
            full_ylim = (float(heatmap_data.extent[2]), float(heatmap_data.extent[3]))
        else:
            full_ylim = ax_heatmap.get_ylim()

        # Setup ticks and labels based on heatmap type
        # For cropped data, pass the cropped energy grid and block info
        # Note: Block labels are stored and drawn later after fig.subplots_adjust()
        pending_block_labels = None
        if use_cropped_data:
            pending_block_labels = self._setup_cropped_heatmap_ticks(
                fig, ax_heatmap, heatmap_data, cropped_block_info, cropped_energy_grid,
                full_xlim, full_ylim
            )
        elif isinstance(heatmap_data, CovarianceHeatmapData):
            # No energy filtering needed - data is already cropped (or not limited)
            pending_block_labels = self._setup_covariance_heatmap_ticks(fig, ax_heatmap, heatmap_data, full_xlim, full_ylim,
                                                  None, None)
        elif isinstance(heatmap_data, LegendreHeatmapData):
            # No energy filtering needed - data is already cropped (or not limited)
            pending_block_labels = self._setup_mf34_heatmap_ticks(fig, ax_heatmap, heatmap_data, full_xlim, full_ylim,
                                           None, None)

        # Draw grid lines separating MT/L blocks
        if use_cropped_data and cropped_block_info:
            self._draw_block_boundaries_from_info(ax_heatmap, cropped_block_info, heatmap_data)
        else:
            self._draw_block_boundaries(ax_heatmap, heatmap_data)

        # Set title based on _title value:
        # - _NOT_SET: use default from heatmap_data.label
        # - None or "": hide title
        # - non-empty string: use that title
        if self._title is _NOT_SET:
            # Use default title from heatmap data
            effective_title = getattr(heatmap_data, "label", None)
        elif self._title:
            # Use explicitly set title (non-empty string)
            effective_title = self._title
        else:
            # Title explicitly set to None or "" - hide title
            effective_title = None

        if self.style not in ('paper', 'publication') and effective_title:
            title_kwargs = {}
            if self._title_fontsize is not None:
                title_kwargs["fontsize"] = self._title_fontsize
            title_y = 0.95 if has_uncertainties else 1.05
            fig.suptitle(effective_title, y=title_y, **title_kwargs)

        if self._tick_labelsize is not None:
            ax_heatmap.tick_params(axis='both', which='both', labelsize=self._tick_labelsize)

        # Draw uncertainty panels if requested
        if has_uncertainties and uncertainty_axes is not None:
            if use_cropped_data:
                # For cropped data, pass cropped info and energy grid
                cropped_xlim = (cropped_x_edges[0], cropped_x_edges[-1]) if cropped_x_edges is not None else None
                self._draw_uncertainty_panels_cropped(
                    uncertainty_axes, heatmap_data, cropped_block_info,
                    cropped_energy_grid, energy_lim, cropped_xlim,
                    cropped_uncertainty_data=cropped_uncertainty_data
                )
            else:
                # No limits when not cropped
                self._draw_uncertainty_panels(uncertainty_axes, heatmap_data, ax_heatmap, None)

        # Calculate number of MTs/blocks for layout adjustment
        if isinstance(heatmap_data, CovarianceHeatmapData) and heatmap_data.block_info:
            num_blocks = len(heatmap_data.block_info.get('mts', [1]))
        elif isinstance(heatmap_data, LegendreHeatmapData) and heatmap_data.block_info:
            num_blocks = len(heatmap_data.legendre_coeffs)
        else:
            num_blocks = 1

        # Bottom margin configuration (extra space for block labels)
        extra_margin = max(0, num_blocks - 1) * 0.015
        bottom_margin = min(0.14 + extra_margin, 0.28)

        if has_uncertainties:
            fig.subplots_adjust(left=0.12, right=0.94, bottom=bottom_margin, top=0.90)
        else:
            fig.subplots_adjust(left=0.12, right=0.94, bottom=bottom_margin, top=0.93)

        # Draw pending block labels NOW (after layout is finalized)
        if pending_block_labels is not None and self._heatmap_show_block_labels:
            fig.canvas.draw()  # Ensure layout is updated
            self._add_block_labels_figure_fixed(
                fig, ax_heatmap,
                pending_block_labels.get('centers'),
                pending_block_labels.get('labels'),
                pending_block_labels.get('full_xlim', (0, 1)),
                pending_block_labels.get('full_ylim', (0, 1)),
                x_axis_label=pending_block_labels.get('x_axis_label'),
                y_axis_label=pending_block_labels.get('y_axis_label'),
                x_centers=pending_block_labels.get('x_centers'),
                x_labels=pending_block_labels.get('x_labels'),
                y_centers=pending_block_labels.get('y_centers'),
                y_labels=pending_block_labels.get('y_labels'),
            )

        # Add colorbar if enabled
        if self._heatmap_show_colorbar:
            fig.canvas.draw()
            heatmap_pos = ax_heatmap.get_position()
            cbar_ax = fig.add_axes([
                heatmap_pos.x1 + 0.10,
                heatmap_pos.y0,
                0.03,
                heatmap_pos.height
            ])
            cbar = fig.colorbar(im, cax=cbar_ax)
            if effective_colorbar_label:
                colorbar_fs = self._heatmap_colorbar_fontsize or 12
                cbar.set_label(effective_colorbar_label, fontsize=colorbar_fs)

        # Configure figure interactivity
        _configure_figure_interactivity(fig, self._interactive)
        return fig

    def _setup_covariance_heatmap_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        data: 'CovarianceHeatmapData',
        full_xlim: Tuple[float, float],
        full_ylim: Tuple[float, float],
        energy_x_lim: Optional[Tuple[float, float]] = None,
        energy_y_lim: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Setup ticks for CovarianceHeatmapData with MT labels and energy axes.

        Returns pending block labels info to be drawn after fig.subplots_adjust().
        """
        if not data.block_info:
            return None

        # Check for multi-isotope mode
        is_multi_isotope = data.block_info.get('is_multi_isotope', False)

        if is_multi_isotope:
            # Multi-isotope mode: blocks are (zaid, mt) tuples
            blocks = data.block_info.get('blocks', [])
            energy_ranges = data.block_info.get('energy_ranges', {}) or {}
            # Use provided mt_labels (should already be "Symbol-MT#" format)
            block_labels = data.mt_labels or [f"({z}, {m})" for (z, m) in blocks]

            pending_block_labels = None
            x_ranges = []
            for block in blocks:
                xr = energy_ranges.get(block)
                if xr is not None:
                    x_ranges.append(tuple(xr))

            if x_ranges and self._heatmap_show_block_labels:
                centers = [(a + b) * 0.5 for (a, b) in x_ranges]
                ax.set_xticks([])
                ax.set_yticks([])
                pending_block_labels = {
                    'centers': centers,
                    'labels': block_labels,
                    'full_xlim': full_xlim,
                    'full_ylim': full_ylim,
                    'x_axis_label': "Isotope-MT",
                    'y_axis_label': "Isotope-MT"
                }
            elif not self._heatmap_show_block_labels:
                ax.set_xticks([])
                ax.set_yticks([])

            # Energy ticks on secondary axes
            if self._heatmap_show_energy_ticks and data.energy_grid is not None and len(data.energy_grid) > 0:
                if len(blocks) > 1 and energy_ranges:
                    energy_grids_dict = {block: data.energy_grid for block in blocks}
                    block_ranges_dict = {block: tuple(energy_ranges[block]) for block in blocks if block in energy_ranges}
                    if block_ranges_dict:
                        self._add_multi_block_energy_ticks(ax, energy_grids_dict, block_ranges_dict, data.scale,
                                                            energy_x_lim, energy_y_lim)
                    else:
                        self._add_energy_ticks(ax, data.energy_grid, data.scale)
                else:
                    self._add_energy_ticks(ax, data.energy_grid, data.scale)

            return pending_block_labels

        # Single-isotope mode (existing logic)
        mts = data.block_info.get('mts', [])
        energy_ranges = data.block_info.get('energy_ranges', {}) or {}
        ranges_fallback = data.block_info.get('ranges', []) or []
        mt_labels = data.mt_labels or [str(m) for m in mts]

        pending_block_labels = None
        x_ranges = []
        if data.is_diagonal:
            for i, m in enumerate(mts):
                xr = energy_ranges.get(m)
                if xr is None and len(ranges_fallback) > i:
                    xr = ranges_fallback[i]
                if xr is not None:
                    x_ranges.append(tuple(xr))

            if x_ranges and self._heatmap_show_block_labels:
                centers = [(a + b) * 0.5 for (a, b) in x_ranges]
                ax.set_xticks([])
                ax.set_yticks([])
                # Return pending labels instead of drawing now
                pending_block_labels = {
                    'centers': centers,
                    'labels': mt_labels,
                    'full_xlim': full_xlim,
                    'full_ylim': full_ylim,
                    'x_axis_label': "MT Number",
                    'y_axis_label': "MT Number"
                }
            elif not self._heatmap_show_block_labels:
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Off-diagonal block: different MTs on each axis
            # For off-diagonal, use the full axis limits for label positioning
            # since both row and column span the same energy range
            ax.set_xticks([])
            ax.set_yticks([])
            
            if self._heatmap_show_block_labels and len(mts) >= 2:
                row_mt = mts[0]
                col_mt = mts[1]
                
                # Use full axis limits for label positioning
                x_center = (full_xlim[0] + full_xlim[1]) * 0.5
                y_center = (full_ylim[0] + full_ylim[1]) * 0.5
                
                pending_block_labels = {
                    'x_centers': [x_center],
                    'x_labels': [str(col_mt)],
                    'y_centers': [y_center],
                    'y_labels': [str(row_mt)],
                    'full_xlim': full_xlim,
                    'full_ylim': full_ylim,
                    'x_axis_label': "MT Number",
                    'y_axis_label': "MT Number"
                }

        # Energy ticks on secondary axes
        if self._heatmap_show_energy_ticks and data.energy_grid is not None and len(data.energy_grid) > 0:
            if len(mts) > 1 and energy_ranges:
                energy_grids_dict = {mt: data.energy_grid for mt in mts}
                block_ranges_dict = {}
                for i, mt in enumerate(mts):
                    rng = energy_ranges.get(mt)
                    if rng is None and len(ranges_fallback) > i:
                        rng = ranges_fallback[i]
                    if rng is not None:
                        block_ranges_dict[mt] = tuple(rng)
                if block_ranges_dict:
                    self._add_multi_block_energy_ticks(ax, energy_grids_dict, block_ranges_dict, data.scale,
                                                        energy_x_lim, energy_y_lim)
                else:
                    self._add_energy_ticks(ax, data.energy_grid, data.scale)
            else:
                self._add_energy_ticks(ax, data.energy_grid, data.scale)

        return pending_block_labels

    def _setup_mf34_heatmap_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        data: 'LegendreHeatmapData',
        full_xlim: Tuple[float, float],
        full_ylim: Tuple[float, float],
        energy_x_lim: Optional[Tuple[float, float]] = None,
        energy_y_lim: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Setup ticks for LegendreHeatmapData with Legendre labels and energy axes.

        Returns pending block labels info to be drawn after fig.subplots_adjust().
        """
        block_info = data.block_info or {}
        legendre_list = block_info.get('legendre_coeffs', data.legendre_coeffs)
        ranges_idx = block_info.get('ranges', {}) or {}
        energy_ranges = block_info.get('energy_ranges', {}) or {}
        legendre_labels = [str(l) for l in legendre_list]

        pending_block_labels = None

        # Legendre labels using figure-fixed positioning
        if self._heatmap_show_block_labels:
            if data.is_diagonal:
                centers = []
                labels = []
                for i, l_val in enumerate(legendre_list):
                    rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                    if rng:
                        centers.append((rng[0] + rng[1]) * 0.5)
                        labels.append(legendre_labels[i])
                if centers:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # Return pending labels instead of drawing now
                    pending_block_labels = {
                        'centers': centers,
                        'labels': labels,
                        'full_xlim': full_xlim,
                        'full_ylim': full_ylim,
                        'x_axis_label': "Legendre Order",
                        'y_axis_label': "Legendre Order"
                    }
            elif len(legendre_list) >= 2:
                row_l = legendre_list[0]
                col_l = legendre_list[1] if len(legendre_list) > 1 else legendre_list[0]
                x_rng = energy_ranges.get(col_l) or ranges_idx.get(col_l)
                y_rng = energy_ranges.get(row_l) or ranges_idx.get(row_l)

                ax.set_xticks([])
                ax.set_yticks([])

                # For off-diagonal, use separate x and y labels
                x_centers = []
                x_labels_list = []
                y_centers = []
                y_labels_list = []

                if x_rng:
                    x_centers = [(x_rng[0] + x_rng[1]) * 0.5]
                    x_labels_list = [str(col_l)]
                if y_rng:
                    y_centers = [(y_rng[0] + y_rng[1]) * 0.5]
                    y_labels_list = [str(row_l)]

                pending_block_labels = {
                    'x_centers': x_centers,
                    'x_labels': x_labels_list,
                    'y_centers': y_centers,
                    'y_labels': y_labels_list,
                    'full_xlim': full_xlim,
                    'full_ylim': full_ylim,
                    'x_axis_label': "Legendre Order",
                    'y_axis_label': "Legendre Order"
                }
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add energy ticks on secondary axes if enabled
        if self._heatmap_show_energy_ticks and data.energy_grids is not None and len(data.energy_grids) > 0:
            if len(legendre_list) > 1 and energy_ranges:
                block_ranges_dict = {}
                for l_val in legendre_list:
                    rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                    if rng is not None:
                        block_ranges_dict[l_val] = tuple(rng)
                if block_ranges_dict:
                    self._add_multi_block_energy_ticks(ax, data.energy_grids, block_ranges_dict, data.scale,
                                                        energy_x_lim, energy_y_lim)
                else:
                    first_grid = next(iter(data.energy_grids.values()))
                    self._add_energy_ticks(ax, first_grid, data.scale)
            else:
                first_grid = next(iter(data.energy_grids.values()))
                self._add_energy_ticks(ax, first_grid, data.scale)

        return pending_block_labels

    def _add_energy_ticks(self, ax: plt.Axes, energy_grid: np.ndarray, scale: str = 'log') -> None:
        """Add energy tick marks on secondary (top/right) axes."""
        from matplotlib.ticker import FuncFormatter, FixedLocator

        # Get fontsize (use configured or default)
        tick_fontsize = self._heatmap_energy_tick_fontsize or 8

        ax_top = ax.twiny()
        ax_right = ax.twinx()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if scale == 'log':
            e_min, e_max = energy_grid.min(), energy_grid.max()
            log_e_min = np.log10(np.maximum(e_min, 1e-300))
            log_e_max = np.log10(e_max)

            decade_min = int(np.floor(log_e_min))
            decade_max = int(np.ceil(log_e_max))

            major_tick_energies = [10**d for d in range(decade_min, decade_max + 1)]
            major_tick_positions = [np.log10(e) - log_e_min for e in major_tick_energies
                                   if e_min <= e <= e_max]

            minor_tick_positions = []
            for decade in range(decade_min, decade_max + 1):
                for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
                    e = multiplier * 10**decade
                    if e_min <= e <= e_max:
                        pos = np.log10(e) - log_e_min
                        minor_tick_positions.append(pos)

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

            ax_top.set_xlim(xlim)
            ax_top.xaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax_top.xaxis.set_major_formatter(formatter)
            ax_top.xaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax_top.tick_params(axis='x', which='major', labelsize=tick_fontsize, rotation=30, length=6, pad=2)
            for label in ax_top.get_xticklabels():
                label.set_ha('left')
            ax_top.tick_params(axis='x', which='minor', length=3)

            ax_right.set_ylim(ylim)
            ax_right.yaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax_right.yaxis.set_major_formatter(formatter)
            ax_right.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax_right.tick_params(axis='y', which='major', labelsize=tick_fontsize, length=6)
            ax_right.tick_params(axis='y', which='minor', length=3)

        else:
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

            # Use configured fontsize for linear scale too
            linear_fontsize = self._heatmap_energy_tick_fontsize or 10

            ax_top.set_xlim(xlim)
            ax_top.set_xticks(tick_positions)
            ax_top.xaxis.set_major_formatter(formatter)
            ax_top.tick_params(axis='x', labelsize=linear_fontsize, rotation=30, pad=2)
            for label in ax_top.get_xticklabels():
                label.set_ha('left')

            ax_right.set_ylim(ylim)
            ax_right.set_yticks(tick_positions)
            ax_right.yaxis.set_major_formatter(formatter)
            ax_right.tick_params(axis='y', labelsize=linear_fontsize)

    def _add_multi_block_energy_ticks(
        self,
        ax: plt.Axes,
        energy_grids: Dict[Any, np.ndarray],
        block_ranges: Dict[Any, Tuple[float, float]],
        scale: str = 'log',
        energy_x_lim: Optional[Tuple[float, float]] = None,
        energy_y_lim: Optional[Tuple[float, float]] = None
    ) -> None:
        """Add energy tick marks for multiple blocks on secondary axes.
        
        For multi-block heatmaps, each block represents the FULL energy range 
        mapped to that block's coordinate range. Ticks are filtered by the
        optional energy limits.
        
        Args:
            energy_x_lim: Optional tuple (min_energy, max_energy) to filter x-axis ticks
            energy_y_lim: Optional tuple (min_energy, max_energy) to filter y-axis ticks
        """
        from matplotlib.ticker import FuncFormatter, FixedLocator

        # Get fontsize
        tick_fontsize = self._heatmap_energy_tick_fontsize or 8

        ax_top = ax.twiny()
        ax_right = ax.twinx()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        all_major_x_positions = []
        all_minor_x_positions = []
        all_major_y_positions = []
        all_minor_y_positions = []
        tick_labels_x = {}
        tick_labels_y = {}

        sorted_keys = sorted(block_ranges.keys())

        for key in sorted_keys:
            if key not in energy_grids or key not in block_ranges:
                continue

            energy_grid = np.asarray(energy_grids[key], dtype=float)
            coord_start_full, coord_end_full = block_ranges[key]
            coord_span_full = coord_end_full - coord_start_full

            if len(energy_grid) < 2 or coord_span_full <= 0:
                continue

            # Determine visible portion of this block for x-axis (in coordinate space)
            coord_start_x = max(coord_start_full, xlim[0])
            coord_end_x = min(coord_end_full, xlim[1])
            
            # Determine visible portion of this block for y-axis (in coordinate space)
            coord_start_y = max(coord_start_full, min(ylim))
            coord_end_y = min(coord_end_full, max(ylim))
            
            # Skip blocks that are completely outside the visible range
            is_visible_x = coord_start_x < coord_end_x
            is_visible_y = coord_start_y < coord_end_y
            
            if not is_visible_x and not is_visible_y:
                continue

            # Each block represents the FULL energy range mapped to its coordinate range
            e_min, e_max = float(energy_grid.min()), float(energy_grid.max())
            
            # Determine which energy range is visible based on user limits
            # For x-axis filtering
            e_x_lo = energy_x_lim[0] if energy_x_lim is not None else e_min
            e_x_hi = energy_x_lim[1] if energy_x_lim is not None else e_max
            # For y-axis filtering
            e_y_lo = energy_y_lim[0] if energy_y_lim is not None else e_min
            e_y_hi = energy_y_lim[1] if energy_y_lim is not None else e_max

            if scale == 'log':
                log_e_min = np.log10(np.maximum(e_min, 1e-300))
                log_e_max = np.log10(e_max)
                log_span = log_e_max - log_e_min

                if log_span <= 0:
                    continue

                decade_min = int(np.floor(log_e_min))
                decade_max = int(np.ceil(log_e_max))

                for decade in range(decade_min, decade_max + 1):
                    e = 10**decade
                    if e_min <= e <= e_max:
                        frac = (np.log10(e) - log_e_min) / log_span
                        pos = coord_start_full + frac * coord_span_full

                        # Add to x-axis if in visible coord range AND within energy limits
                        if is_visible_x and coord_start_x <= pos <= coord_end_x and e_x_lo <= e <= e_x_hi:
                            all_major_x_positions.append(pos)
                            tick_labels_x[pos] = f'1e{decade:+03d}'
                        
                        # Add to y-axis if in visible coord range AND within energy limits
                        if is_visible_y and coord_start_y <= pos <= coord_end_y and e_y_lo <= e <= e_y_hi:
                            all_major_y_positions.append(pos)
                            tick_labels_y[pos] = f'1e{decade:+03d}'

                for decade in range(decade_min, decade_max + 1):
                    for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                        e = mult * 10**decade
                        if e_min <= e <= e_max:
                            frac = (np.log10(e) - log_e_min) / log_span
                            pos = coord_start_full + frac * coord_span_full
                            
                            # Add to x-axis if in visible coord range AND within energy limits
                            if is_visible_x and coord_start_x <= pos <= coord_end_x and e_x_lo <= e <= e_x_hi:
                                all_minor_x_positions.append(pos)
                            
                            # Add to y-axis if in visible coord range AND within energy limits
                            if is_visible_y and coord_start_y <= pos <= coord_end_y and e_y_lo <= e <= e_y_hi:
                                all_minor_y_positions.append(pos)

            else:
                from matplotlib.ticker import MaxNLocator

                locator = MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
                tick_energies = locator.tick_values(e_min, e_max)
                tick_energies = tick_energies[(tick_energies >= e_min) & (tick_energies <= e_max)]

                energy_span = e_max - e_min
                if energy_span <= 0:
                    continue

                for e in tick_energies:
                    frac = (e - e_min) / energy_span
                    pos = coord_start_full + frac * coord_span_full

                    if e == 0:
                        label = '0'
                    else:
                        exponent = int(np.floor(np.log10(abs(e))))
                        mantissa = e / (10**exponent)
                        if abs(mantissa - 1.0) < 0.1:
                            label = f'1e{exponent:+03d}'
                        else:
                            label = f'{int(np.round(mantissa))}e{exponent:+03d}'

                    # Add to x-axis if in visible coord range AND within energy limits
                    if is_visible_x and coord_start_x <= pos <= coord_end_x and e_x_lo <= e <= e_x_hi:
                        all_major_x_positions.append(pos)
                        tick_labels_x[pos] = label
                    
                    # Add to y-axis if in visible coord range AND within energy limits
                    if is_visible_y and coord_start_y <= pos <= coord_end_y and e_y_lo <= e <= e_y_hi:
                        all_major_y_positions.append(pos)
                        tick_labels_y[pos] = label

        # Set up top axis
        ax_top.set_xlim(xlim)
        if all_major_x_positions:
            ax_top.xaxis.set_major_locator(FixedLocator(all_major_x_positions))
            def format_x(val, pos=None):
                return tick_labels_x.get(val, '')
            ax_top.xaxis.set_major_formatter(FuncFormatter(format_x))
        if all_minor_x_positions:
            ax_top.xaxis.set_minor_locator(FixedLocator(all_minor_x_positions))
        ax_top.tick_params(axis='x', which='major', labelsize=tick_fontsize, rotation=30, length=6, pad=2)
        for label in ax_top.get_xticklabels():
            label.set_ha('left')
        ax_top.tick_params(axis='x', which='minor', length=3)

        # Set up right axis
        ax_right.set_ylim(ylim)
        if all_major_y_positions:
            ax_right.yaxis.set_major_locator(FixedLocator(all_major_y_positions))
            def format_y(val, pos=None):
                return tick_labels_y.get(val, '')
            ax_right.yaxis.set_major_formatter(FuncFormatter(format_y))
        if all_minor_y_positions:
            ax_right.yaxis.set_minor_locator(FixedLocator(all_minor_y_positions))
        ax_right.tick_params(axis='y', which='major', labelsize=tick_fontsize, length=6)
        ax_right.tick_params(axis='y', which='minor', length=3)

    def _add_uncertainty_energy_ticks(
        self,
        ax: plt.Axes,
        energy_grid: np.ndarray,
        scale: str = 'log',
        xlim: Optional[Tuple[float, float]] = None
    ) -> None:
        """Add energy tick marks on bottom of uncertainty panel."""
        from matplotlib.ticker import FuncFormatter, FixedLocator

        if xlim is None:
            xlim = ax.get_xlim()

        if scale == 'log':
            e_min, e_max = energy_grid.min(), energy_grid.max()
            log_e_min = np.log10(np.maximum(e_min, 1e-300))
            log_e_max = np.log10(e_max)

            decade_min = int(np.floor(log_e_min))
            decade_max = int(np.ceil(log_e_max))

            major_tick_energies = [10**d for d in range(decade_min, decade_max + 1)]
            major_tick_positions = [np.log10(e) - log_e_min for e in major_tick_energies
                                   if e_min <= e <= e_max]

            minor_tick_positions = []
            for decade in range(decade_min, decade_max + 1):
                for multiplier in [2, 3, 4, 5, 6, 7, 8, 9]:
                    e = multiplier * 10**decade
                    if e_min <= e <= e_max:
                        pos = np.log10(e) - log_e_min
                        minor_tick_positions.append(pos)

            ax.set_xlim(xlim)
            ax.xaxis.set_major_locator(FixedLocator(major_tick_positions))
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_minor_locator(FixedLocator(minor_tick_positions))
            ax.tick_params(axis='x', which='major', length=4,
                          bottom=True, top=False, labelbottom=False, labeltop=False, direction='in')
            ax.tick_params(axis='x', which='minor', length=2,
                          bottom=True, top=False, direction='in')
        else:
            from matplotlib.ticker import MaxNLocator

            locator = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
            tick_positions = locator.tick_values(energy_grid.min(), energy_grid.max())
            tick_positions = tick_positions[(tick_positions >= energy_grid.min()) &
                                           (tick_positions <= energy_grid.max())]

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
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        ranges_x = []
        ranges_y = []

        if isinstance(heatmap_data, CovarianceHeatmapData):
            info = heatmap_data.block_info or {}

            # Check for multi-isotope mode
            is_multi_isotope = info.get('is_multi_isotope', False)

            if is_multi_isotope:
                # Multi-isotope: blocks keyed by (zaid, mt) tuples
                blocks = info.get('blocks', [])
                energy_ranges = info.get('energy_ranges', {}) or {}

                for block in blocks:
                    rng = energy_ranges.get(block)
                    if rng is not None:
                        ranges_x.append(tuple(rng))
                        ranges_y.append(tuple(rng))
            else:
                # Single-isotope mode
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

        elif isinstance(heatmap_data, LegendreHeatmapData):
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

    def _draw_block_boundaries_from_info(
        self,
        ax: plt.Axes,
        block_info: Dict[str, Any],
        heatmap_data: 'HeatmapPlotData'
    ) -> None:
        """Draw grid lines from cropped block_info dictionary."""
        energy_ranges = block_info.get('energy_ranges', {})
        if not energy_ranges or len(energy_ranges) <= 1:
            return

        ranges_list = list(energy_ranges.values())
        sorted_ranges = sorted(ranges_list, key=lambda r: r[0])

        # Draw boundary at the start of each block (except the first)
        for rng in sorted_ranges[1:]:
            ax.axvline(rng[0], color="#404040", lw=1.0, alpha=0.8, zorder=5)
            ax.axhline(rng[0], color="#404040", lw=1.0, alpha=0.8, zorder=5)

    def _setup_cropped_heatmap_ticks(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        original_data: 'HeatmapPlotData',
        cropped_block_info: Dict[str, Any],
        cropped_energy_grid: Optional[np.ndarray],
        full_xlim: Tuple[float, float],
        full_ylim: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        """Setup ticks for cropped multi-block heatmaps.

        Returns pending block labels info to be drawn after fig.subplots_adjust().
        """
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        if not cropped_block_info:
            return None

        energy_ranges = cropped_block_info.get('energy_ranges', {})
        scale = getattr(original_data, 'scale', 'log')

        # Check for multi-isotope mode
        is_multi_isotope = cropped_block_info.get('is_multi_isotope', False)

        # Get block keys (MTs, Legendre coefficients, or (zaid, mt) tuples)
        if isinstance(original_data, CovarianceHeatmapData):
            if is_multi_isotope:
                # Multi-isotope: use (zaid, mt) tuples from blocks
                block_keys = cropped_block_info.get('blocks', [])
                # Use provided mt_labels if available
                if original_data.mt_labels and len(original_data.mt_labels) == len(block_keys):
                    block_labels = original_data.mt_labels
                else:
                    from kika._utils import zaid_to_symbol
                    block_labels = [f"{zaid_to_symbol(z)}-MT{m}" for (z, m) in block_keys]
            else:
                block_keys = cropped_block_info.get('mts', [])
                # Use original mt_labels if available (just numbers, no "MT" prefix)
                if original_data.mt_labels and len(original_data.mt_labels) == len(block_keys):
                    block_labels = original_data.mt_labels
                else:
                    block_labels = [str(m) for m in block_keys]
        else:
            block_keys = cropped_block_info.get('legendre_coeffs', [])
            block_labels = [f"L={l}" for l in block_keys]

        pending_block_labels = None

        # Check if this is an off-diagonal case (2 different values, not diagonal)
        is_off_diagonal_mf34 = (
            not isinstance(original_data, CovarianceHeatmapData) and
            not getattr(original_data, 'is_diagonal', True) and
            len(block_keys) == 2
        )
        is_off_diagonal_covmat = (
            isinstance(original_data, CovarianceHeatmapData) and
            not getattr(original_data, 'is_diagonal', True) and
            len(block_keys) == 2
        )
        is_off_diagonal = is_off_diagonal_mf34 or is_off_diagonal_covmat

        # Setup block labels
        if self._heatmap_show_block_labels and energy_ranges:
            if is_multi_isotope:
                x_axis_label = "Isotope-MT"
            elif isinstance(original_data, CovarianceHeatmapData):
                x_axis_label = "MT Number"
            else:
                x_axis_label = "Legendre Order"

            if is_off_diagonal:
                # Off-diagonal: separate x and y labels
                row_l = block_keys[0]
                col_l = block_keys[1]

                x_centers = []
                x_labels_list = []
                y_centers = []
                y_labels_list = []

                x_rng = energy_ranges.get(col_l)
                y_rng = energy_ranges.get(row_l)

                if x_rng:
                    x_centers = [(x_rng[0] + x_rng[1]) * 0.5]
                    x_labels_list = [str(col_l)]
                if y_rng:
                    y_centers = [(y_rng[0] + y_rng[1]) * 0.5]
                    y_labels_list = [str(row_l)]

                ax.set_xticks([])
                ax.set_yticks([])
                pending_block_labels = {
                    'x_centers': x_centers,
                    'x_labels': x_labels_list,
                    'y_centers': y_centers,
                    'y_labels': y_labels_list,
                    'full_xlim': full_xlim,
                    'full_ylim': full_ylim,
                    'x_axis_label': x_axis_label,
                    'y_axis_label': x_axis_label
                }
            else:
                # Diagonal or multiblock: same labels on both axes
                centers = []
                labels = []
                for i, key in enumerate(block_keys):
                    rng = energy_ranges.get(key)
                    if rng:
                        centers.append((rng[0] + rng[1]) * 0.5)
                        # Use pre-built label if available, otherwise convert key to string
                        if i < len(block_labels):
                            labels.append(block_labels[i])
                        else:
                            labels.append(str(key))

                if centers:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # Return pending labels instead of drawing now
                    pending_block_labels = {
                        'centers': centers,
                        'labels': labels,
                        'full_xlim': full_xlim,
                        'full_ylim': full_ylim,
                        'x_axis_label': x_axis_label,
                        'y_axis_label': x_axis_label
                    }
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Setup energy ticks on secondary axes
        if self._heatmap_show_energy_ticks and cropped_energy_grid is not None and len(cropped_energy_grid) > 0:
            # Create energy grids dict for each block
            energy_grids_dict = {key: cropped_energy_grid for key in block_keys}
            block_ranges_dict = {key: energy_ranges[key] for key in block_keys if key in energy_ranges}

            if block_ranges_dict:
                # No energy filtering for cropped data - show full cropped range
                self._add_multi_block_energy_ticks(
                    ax, energy_grids_dict, block_ranges_dict, scale,
                    energy_x_lim=None, energy_y_lim=None
                )

        return pending_block_labels

    def _draw_uncertainty_panels(
        self,
        uncertainty_axes: List[plt.Axes],
        heatmap_data: 'HeatmapPlotData',
        ax_heatmap: plt.Axes,
        x_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Draw uncertainty panels above heatmap."""
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        uncertainty_data = heatmap_data.uncertainty_data
        if not uncertainty_data:
            return

        background_color = "#F5F5F5"
        tick_grey = "#707070"

        def _nice_ylim_and_step(max_val: float) -> tuple:
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
            top, step = _nice_ylim_and_step(ymax)
            grid_vals = np.arange(0.0, top + 1e-9, step)
            for y in grid_vals:
                ax_u.axhline(y, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)
            x_label = xr[0] + 0.02 * (xr[1] - xr[0])
            for y in grid_vals:
                lbl = f"{y:g}" if step < 1 else f"{int(y)}"
                ax_u.text(x_label, y, lbl, ha="left", va="center",
                         color=tick_grey, fontsize=8, alpha=0.9, zorder=2)

        def _draw_xgrid_inside(ax_u, energy_grid, scale, xr):
            """Draw vertical grid lines at decade and sub-decade boundaries."""
            if energy_grid is None or len(energy_grid) < 2:
                return

            e_min, e_max = float(energy_grid.min()), float(energy_grid.max())

            if scale == 'log':
                log_e_min = np.log10(np.maximum(e_min, 1e-300))
                log_e_max = np.log10(e_max)
                log_span = log_e_max - log_e_min

                if log_span <= 0:
                    return

                decade_min = int(np.floor(log_e_min))
                decade_max = int(np.ceil(log_e_max))

                # Calculate axis span (transformed coordinates)
                axis_span = xr[1] - xr[0]

                # Draw major decade lines (1e5, 1e6, etc.) - darker
                for decade in range(decade_min, decade_max + 1):
                    e = 10**decade
                    if e_min <= e <= e_max:
                        frac = (np.log10(e) - log_e_min) / log_span
                        pos = xr[0] + frac * axis_span
                        ax_u.axvline(pos, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)

                # Draw sub-decade lines (2e5, 3e5, 5e5, etc.) - lighter
                for decade in range(decade_min, decade_max + 1):
                    for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                        e = mult * 10**decade
                        if e_min <= e <= e_max:
                            frac = (np.log10(e) - log_e_min) / log_span
                            pos = xr[0] + frac * axis_span
                            ax_u.axvline(pos, color=tick_grey, lw=0.4, alpha=0.2, zorder=0)
            else:
                # Linear scale: use nice tick values
                from matplotlib.ticker import MaxNLocator
                locator = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
                tick_energies = locator.tick_values(e_min, e_max)
                tick_energies = tick_energies[(tick_energies >= e_min) & (tick_energies <= e_max)]

                energy_span = e_max - e_min
                if energy_span <= 0:
                    return

                axis_span = xr[1] - xr[0]
                for e in tick_energies:
                    frac = (e - e_min) / energy_span
                    pos = xr[0] + frac * axis_span
                    ax_u.axvline(pos, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)

        keys = sorted(uncertainty_data.keys())

        if isinstance(heatmap_data, LegendreHeatmapData):
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
                for i, (ax_u, L) in enumerate(zip(uncertainty_axes, keys)):
                    sigma_pct = uncertainty_data[L]
                    xs_edges = _edges_for_L(L)
                    if xs_edges is None or xs_edges.size == 0:
                        continue

                    ax_u.set_facecolor(background_color)
                    ax_u.grid(False)

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

                    if x_limits is not None:
                        ax_u.set_xlim(x_limits)
                    else:
                        ax_u.set_xlim(0.0, xs_local[-1])

                    raw_grid = heatmap_data.energy_grids.get(L) if heatmap_data.energy_grids else None
                    # Remove ticks - grid lines are drawn by _draw_xgrid_inside below
                    ax_u.set_xticks([])
                    ax_u.set_yticks([])

                    _draw_ygrid_inside(ax_u, (0.0, xs_local[-1]), ax_u.get_ylim()[1])
                    if raw_grid is not None:
                        _draw_xgrid_inside(ax_u, raw_grid, heatmap_data.scale, (0.0, xs_local[-1]))

                    if i == 0:
                        ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

                    for side in ('left', 'right', 'top', 'bottom'):
                        ax_u.spines[side].set_visible(False)
            else:
                # Off-diagonal case: plot both L values on single panel with legend
                ax_u = uncertainty_axes[0]
                ax_u.set_facecolor(background_color)
                ax_u.grid(False)

                y_max_global = 1.0
                x_max_global = 0.0
                legend_handles = []
                legend_labels = []
                colors = ['C0', 'C1', 'C2', 'C3']

                for i, L in enumerate(keys):
                    if L not in uncertainty_data:
                        continue
                    sigma_pct = uncertainty_data[L]
                    xs_edges = _edges_for_L(L)
                    if xs_edges is None or xs_edges.size == 0:
                        continue

                    xs_local = xs_edges - xs_edges[0]
                    sigma_ext = np.append(sigma_pct, sigma_pct[-1]) if sigma_pct.size > 0 else sigma_pct

                    color = colors[i % len(colors)]
                    if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                        line, = ax_u.step(xs_local, sigma_ext, where='post',
                                  linewidth=1.4, color=color, zorder=3, label=f'L={L}')
                        legend_handles.append(line)
                        legend_labels.append(f'L={L}')
                        y_max_global = max(y_max_global, float(np.nanmax(sigma_ext)))

                    if len(xs_local) > 0:
                        x_max_global = max(x_max_global, xs_local[-1])

                # Set y limits based on all curves
                top, _ = _nice_ylim_and_step(y_max_global * 1.05)
                ax_u.set_ylim(0.0, top)

                if x_limits is not None:
                    ax_u.set_xlim(x_limits)
                elif x_max_global > 0:
                    ax_u.set_xlim(0.0, x_max_global)

                # Add legend if we have multiple curves
                if len(legend_handles) > 1:
                    ax_u.legend(legend_handles, legend_labels, loc='upper right',
                               fontsize=8, framealpha=0.7)

                # Use first L's grid for grid lines
                L_first = keys[0]
                raw_grid = heatmap_data.energy_grids.get(L_first) if heatmap_data.energy_grids else None
                ax_u.set_xticks([])
                ax_u.set_yticks([])

                if x_max_global > 0:
                    _draw_ygrid_inside(ax_u, (0.0, x_max_global), ax_u.get_ylim()[1])
                    if raw_grid is not None:
                        _draw_xgrid_inside(ax_u, raw_grid, heatmap_data.scale, (0.0, x_max_global))

                ax_u.set_ylabel('Unc. (%)', fontsize=10)

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_u.spines[side].set_visible(False)

        elif isinstance(heatmap_data, CovarianceHeatmapData):
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

                if x_limits is not None:
                    ax_unc.set_xlim(x_limits)
                else:
                    ax_unc.set_xlim(xs_plot[0], xs_plot[-1])

                # Remove ticks - grid lines are drawn by _draw_xgrid_inside below
                ax_unc.set_xticks([])
                ax_unc.set_yticks([])

                if xs_edges.size >= 2:
                    _draw_ygrid_inside(ax_unc, (xs_edges[0], xs_edges[-1]), ax_unc.get_ylim()[1])
                    if energy_grid is not None:
                        _draw_xgrid_inside(ax_unc, energy_grid, heatmap_data.scale, (xs_edges[0], xs_edges[-1]))

                if i == 0:
                    ax_unc.set_ylabel('Unc. (%)', fontsize=10, color='black')

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_unc.spines[side].set_visible(False)

    def _draw_uncertainty_panels_cropped(
        self,
        uncertainty_axes: List[plt.Axes],
        heatmap_data: 'HeatmapPlotData',
        cropped_block_info: Dict[str, Any],
        cropped_energy_grid: Optional[np.ndarray],
        energy_lim: Optional[Tuple[float, float]],
        x_limits: Optional[Tuple[float, float]] = None,
        cropped_uncertainty_data: Optional[Dict] = None
    ) -> None:
        """Draw uncertainty panels for cropped multi-block heatmaps."""
        from .plot_data import CovarianceHeatmapData, LegendreHeatmapData

        # Use cropped uncertainty data if provided, otherwise fall back to original
        if cropped_uncertainty_data is not None:
            uncertainty_data = cropped_uncertainty_data
        else:
            uncertainty_data = heatmap_data.uncertainty_data
        if not uncertainty_data or cropped_energy_grid is None:
            return

        background_color = "#F5F5F5"
        tick_grey = "#707070"

        def _nice_ylim_and_step(max_val: float) -> tuple:
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
            top, step = _nice_ylim_and_step(ymax)
            grid_vals = np.arange(0.0, top + 1e-9, step)
            for y in grid_vals:
                ax_u.axhline(y, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)
            x_label = xr[0] + 0.02 * (xr[1] - xr[0])
            for y in grid_vals:
                lbl = f"{y:g}" if step < 1 else f"{int(y)}"
                ax_u.text(x_label, y, lbl, ha="left", va="center",
                         color=tick_grey, fontsize=8, alpha=0.9, zorder=2)

        def _draw_xgrid_inside(ax_u, energy_grid, scale, xr):
            """Draw vertical grid lines at decade and sub-decade boundaries."""
            if energy_grid is None or len(energy_grid) < 2:
                return

            e_min, e_max = float(energy_grid.min()), float(energy_grid.max())

            if scale == 'log':
                log_e_min = np.log10(np.maximum(e_min, 1e-300))
                log_e_max = np.log10(e_max)
                log_span = log_e_max - log_e_min

                if log_span <= 0:
                    return

                decade_min = int(np.floor(log_e_min))
                decade_max = int(np.ceil(log_e_max))

                axis_span = xr[1] - xr[0]

                # Draw major decade lines (1e5, 1e6, etc.) - darker
                for decade in range(decade_min, decade_max + 1):
                    e = 10**decade
                    if e_min <= e <= e_max:
                        frac = (np.log10(e) - log_e_min) / log_span
                        pos = xr[0] + frac * axis_span
                        ax_u.axvline(pos, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)

                # Draw sub-decade lines (2e5, 3e5, 5e5, etc.) - lighter
                for decade in range(decade_min, decade_max + 1):
                    for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                        e = mult * 10**decade
                        if e_min <= e <= e_max:
                            frac = (np.log10(e) - log_e_min) / log_span
                            pos = xr[0] + frac * axis_span
                            ax_u.axvline(pos, color=tick_grey, lw=0.4, alpha=0.2, zorder=0)
            else:
                from matplotlib.ticker import MaxNLocator
                locator = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
                tick_energies = locator.tick_values(e_min, e_max)
                tick_energies = tick_energies[(tick_energies >= e_min) & (tick_energies <= e_max)]

                energy_span = e_max - e_min
                if energy_span <= 0:
                    return

                axis_span = xr[1] - xr[0]
                for e in tick_energies:
                    frac = (e - e_min) / energy_span
                    pos = xr[0] + frac * axis_span
                    ax_u.axvline(pos, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)

        # Find which groups are in the energy range
        def _find_groups_in_range(energy_grid: np.ndarray, elim: Tuple[float, float]) -> np.ndarray:
            lo, hi = elim
            if lo > hi:
                lo, hi = hi, lo
            G = len(energy_grid) - 1
            in_range = []
            for i in range(G):
                e_lo_bin = energy_grid[i]
                e_hi_bin = energy_grid[i + 1]
                if (e_hi_bin >= lo) and (e_lo_bin <= hi):
                    in_range.append(i)
            return np.array(in_range, dtype=int)

        # Get block keys and check if we have per-block energy grids
        if isinstance(heatmap_data, CovarianceHeatmapData):
            block_keys = cropped_block_info.get('mts', [])
        else:
            block_keys = cropped_block_info.get('legendre_coeffs', [])

        # Check if we have per-block cropped energy grids (multi-block MF34)
        cropped_energy_grids_per_block = cropped_block_info.get('cropped_energy_grids', {})
        has_per_block_grids = len(cropped_energy_grids_per_block) > 0

        scale = getattr(heatmap_data, 'scale', 'log')
        energy_ranges = cropped_block_info.get('energy_ranges', {})

        # Check if this is off-diagonal (single panel for multiple L values)
        is_off_diagonal = (
            hasattr(heatmap_data, 'is_diagonal') and
            not heatmap_data.is_diagonal and
            len(block_keys) >= 2
        )

        if is_off_diagonal:
            # Off-diagonal case: plot all curves on single panel with legend
            ax_u = uncertainty_axes[0]
            ax_u.set_facecolor(background_color)
            ax_u.grid(False)

            y_max_global = 1.0
            x_min_global = float('inf')
            x_max_global = 0.0
            legend_handles = []
            legend_labels = []
            colors = ['C0', 'C1', 'C2', 'C3']
            first_grid = None

            for i, key in enumerate(block_keys):
                sigma_pct = uncertainty_data.get(key)
                if sigma_pct is None:
                    continue

                # Get block-specific energy grid
                if has_per_block_grids:
                    block_energy_grid = cropped_energy_grids_per_block.get(key, cropped_energy_grid)
                else:
                    block_energy_grid = cropped_energy_grid

                if first_grid is None:
                    first_grid = block_energy_grid

                # Calculate transformed edges
                if scale == 'log':
                    block_edges = np.log10(np.maximum(block_energy_grid, 1e-300))
                    block_edges = block_edges - block_edges[0]
                else:
                    block_edges = block_energy_grid - block_energy_grid[0]

                xs_local = block_edges

                # Extend sigma for step plot
                if sigma_pct.size > 0:
                    sigma_ext = np.append(sigma_pct, sigma_pct[-1])
                else:
                    sigma_ext = sigma_pct

                color = colors[i % len(colors)]
                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    line, = ax_u.step(xs_local, sigma_ext, where='post',
                              linewidth=1.4, color=color, zorder=3, label=f'L={key}')
                    legend_handles.append(line)
                    legend_labels.append(f'L={key}')
                    y_max_global = max(y_max_global, float(np.nanmax(sigma_ext)))

                if len(xs_local) > 0:
                    x_min_global = min(x_min_global, xs_local[0])
                    x_max_global = max(x_max_global, xs_local[-1])

            # Set limits based on all curves
            top, _ = _nice_ylim_and_step(y_max_global * 1.05)
            ax_u.set_ylim(0.0, top)

            if x_max_global > x_min_global:
                ax_u.set_xlim(x_min_global, x_max_global)

            # Add legend
            if len(legend_handles) > 1:
                ax_u.legend(legend_handles, legend_labels, loc='upper right',
                           fontsize=8, framealpha=0.7)

            ax_u.set_xticks([])
            ax_u.set_yticks([])

            if x_max_global > x_min_global:
                _draw_ygrid_inside(ax_u, (x_min_global, x_max_global), ax_u.get_ylim()[1])
                if first_grid is not None:
                    _draw_xgrid_inside(ax_u, first_grid, scale, (x_min_global, x_max_global))

            ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

            for side in ('left', 'right', 'top', 'bottom'):
                ax_u.spines[side].set_visible(False)
        else:
            # Draw one uncertainty panel per block (diagonal or multi-block diagonal)
            for i, (ax_u, key) in enumerate(zip(uncertainty_axes, block_keys)):
                # Get uncertainty data for this block (already cropped)
                sigma_pct = uncertainty_data.get(key)
                if sigma_pct is None:
                    continue

                ax_u.set_facecolor(background_color)
                ax_u.grid(False)

                # Get block-specific energy grid if available, otherwise use common grid
                if has_per_block_grids:
                    block_energy_grid = cropped_energy_grids_per_block.get(key, cropped_energy_grid)
                else:
                    block_energy_grid = cropped_energy_grid

                # Calculate transformed edges for this block
                if scale == 'log':
                    block_edges = np.log10(np.maximum(block_energy_grid, 1e-300))
                    block_edges = block_edges - block_edges[0]
                else:
                    block_edges = block_energy_grid - block_energy_grid[0]

                # Get block range in coordinate space
                if has_per_block_grids:
                    block_range = energy_ranges.get(key, (0, block_edges[-1] if len(block_edges) > 0 else 1.0))
                else:
                    block_edges_ref = np.log10(np.maximum(cropped_energy_grid, 1e-300)) if scale == 'log' else cropped_energy_grid
                    block_edges_ref = block_edges_ref - block_edges_ref[0]
                    block_range = energy_ranges.get(key, (0, block_edges_ref[-1] if len(block_edges_ref) > 0 else 1.0))
                block_width = block_range[1] - block_range[0]

                # Scale edges to fit in block coordinate range
                if len(block_edges) > 0 and block_edges[-1] > 0:
                    xs_local = block_edges * (block_width / block_edges[-1]) + block_range[0]
                else:
                    xs_local = block_edges + block_range[0] if len(block_edges) > 0 else np.array([block_range[0]])

                # Extend sigma for step plot
                if sigma_pct.size > 0:
                    sigma_ext = np.append(sigma_pct, sigma_pct[-1])
                else:
                    sigma_ext = sigma_pct

                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    ax_u.step(xs_local, sigma_ext, where='post',
                              linewidth=1.4, color=f"C{i}", zorder=3)
                    y_max = float(np.nanmax(sigma_ext))
                else:
                    y_max = 5.0

                y_max = max(y_max, 1.0)
                top, _ = _nice_ylim_and_step(y_max * 1.05)
                ax_u.set_ylim(0.0, top)

                # For multi-block, each panel shows one block, so use the block's range
                # Don't use x_limits which spans all blocks
                is_multi_block = len(block_keys) > 1
                if x_limits is not None and not is_multi_block:
                    ax_u.set_xlim(x_limits)
                else:
                    ax_u.set_xlim(xs_local[0], xs_local[-1])

                # Remove ticks - grid lines are drawn by _draw_xgrid_inside below
                ax_u.set_xticks([])
                ax_u.set_yticks([])

                if len(xs_local) > 0:
                    _draw_ygrid_inside(ax_u, (xs_local[0], xs_local[-1]), ax_u.get_ylim()[1])
                    _draw_xgrid_inside(ax_u, block_energy_grid, scale, (xs_local[0], xs_local[-1]))

                if i == 0:
                    ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_u.spines[side].set_visible(False)

    def _add_block_labels_figure_fixed(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        centers: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        full_xlim: Tuple[float, float] = (0, 1),
        full_ylim: Tuple[float, float] = (0, 1),
        *,
        pad_frac_x: float = 0.020,
        pad_frac_y: float = 0.020,
        fontsize: Optional[float] = None,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        x_centers: Optional[List[float]] = None,
        x_labels: Optional[List[str]] = None,
        y_centers: Optional[List[float]] = None,
        y_labels: Optional[List[str]] = None,
    ) -> None:
        """Place block labels relative to the figure (not axis ticks).

        For off-diagonal blocks, use x_centers/x_labels for bottom (x-axis)
        and y_centers/y_labels for left (y-axis) to show different labels.
        For backward compatibility, centers/labels are used for both axes
        when x/y-specific parameters are not provided.
        """
        # Use explicit x/y if provided, else fall back to shared centers/labels
        actual_x_centers = x_centers if x_centers is not None else centers
        actual_x_labels = x_labels if x_labels is not None else labels
        actual_y_centers = y_centers if y_centers is not None else centers
        actual_y_labels = y_labels if y_labels is not None else labels

        ax_pos = ax.get_position()

        x0, x1 = float(full_xlim[0]), float(full_xlim[1])
        y0, y1 = float(full_ylim[0]), float(full_ylim[1])
        dx = (x1 - x0) if (x1 > x0) else 1.0
        dy = (y1 - y0) if (y1 > y0) else 1.0

        # Use configured fontsize or default
        fs = fontsize if fontsize is not None else (self._heatmap_block_label_fontsize or self._tick_labelsize or 11)
        outer_label_fs = fs

        # Bottom labels (x direction)
        if actual_x_centers and actual_x_labels:
            y_text = ax_pos.y0 - pad_frac_x
            for c, lab in zip(actual_x_centers, actual_x_labels):
                frac = (float(c) - x0) / dx
                frac = min(1.0, max(0.0, frac))
                x_text = ax_pos.x0 + frac * ax_pos.width
                fig.text(x_text, y_text, lab, ha="center", va="top", fontsize=fs)

        # Left labels (y direction) - inverted for heatmap
        if actual_y_centers and actual_y_labels:
            x_text = ax_pos.x0 - pad_frac_y
            for c, lab in zip(actual_y_centers, actual_y_labels):
                frac = (float(c) - y0) / dy
                frac = min(1.0, max(0.0, frac))
                frac = 1.0 - frac  # Invert for y-axis
                y_text = ax_pos.y0 + frac * ax_pos.height
                fig.text(x_text, y_text, lab, ha="right", va="center", fontsize=fs)

        # Outer axis labels
        outer_offset = 0.025

        if x_axis_label:
            x_center = ax_pos.x0 + ax_pos.width / 2
            y_outer = ax_pos.y0 - pad_frac_x - outer_offset
            fig.text(x_center, y_outer, x_axis_label, ha="center", va="top",
                    fontsize=outer_label_fs)

        if y_axis_label:
            x_outer = ax_pos.x0 - pad_frac_y - outer_offset
            y_center = ax_pos.y0 + ax_pos.height / 2
            fig.text(x_outer, y_center, y_axis_label, ha="right", va="center",
                    fontsize=outer_label_fs, rotation=90)
