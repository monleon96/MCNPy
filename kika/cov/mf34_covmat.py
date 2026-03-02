from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kika._utils import create_repr_section

if TYPE_CHECKING:
    from pathlib import Path
    from kika.plotting.plot_data import MF34HeatmapData, LegendreUncertaintyPlotData


@dataclass
class MF34CovMat: 
    """
    Class for storing angular distribution covariance matrix data (MF34).
    
    Attributes
    ----------
    isotope_rows : List[int]
        List of row isotope IDs
    reaction_rows : List[int]
        List of row reaction MT numbers
    l_rows : List[int]
        List of row Legendre coefficient indices
    isotope_cols : List[int]
        List of column isotope IDs
    reaction_cols : List[int]
        List of column reaction MT numbers
    l_cols : List[int]
        List of column Legendre coefficient indices
    energy_grids : List[List[float]]
        List of energy grids for each covariance matrix
    matrices : List[np.ndarray]
        List of covariance matrices
    is_relative : List[bool]
        List of flags indicating if matrix values are relative (True) or absolute (False).
        False only when LB=0 is present. For ENDF-normalized Legendre coefficients
        a_l = (c_l/c0)/(2l+1), the covariance values are inherently relative since
        the coefficients are already normalized by the total cross section.
    frame : List[str]
        List of reference frames for each matrix:
        - "same-as-MF4" when LCT=0
        - "LAB" when LCT=1  
        - "CM" when LCT=2
    energy_unit : str
        Energy unit for energy_grids: 'eV' (default) or 'MeV'
    """
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    l_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    l_cols: List[int] = field(default_factory=list)
    energy_grids: List[List[float]] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)

    # Metadata fields
    is_relative: List[bool] = field(default_factory=list)
    frame: List[str] = field(default_factory=list)
    energy_unit: str = 'eV'  # Energy unit: 'eV' or 'MeV'

    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

    def add_matrix(self, 
                  isotope_row: int, 
                  reaction_row: int,
                  l_row: int,
                  isotope_col: int, 
                  reaction_col: int,
                  l_col: int,
                  matrix: np.ndarray,
                  energy_grid: List[float],
                  is_relative: bool,
                  frame: str):
        """
        Add an angular covariance matrix to the collection.
        
        Parameters
        ----------
        isotope_row : int
            Row isotope ID
        reaction_row : int
            Row reaction MT number
        l_row : int
            Row Legendre coefficient index
        isotope_col : int
            Column isotope ID
        reaction_col : int
            Column reaction MT number
        l_col : int
            Column Legendre coefficient index
        matrix : np.ndarray
            Covariance matrix 
        energy_grid : List[float]
            Energy grid for this covariance matrix
        is_relative : bool
            True if matrix values are relative, False if absolute (LB=0 present)
        frame : str
            Reference frame: "same-as-MF4", "LAB", "CM", or "unknown LCT=X"
        """
        # No validation on matrix shape as each matrix can have a different size
        
        self.isotope_rows.append(isotope_row)
        self.reaction_rows.append(reaction_row)
        self.l_rows.append(l_row)
        self.isotope_cols.append(isotope_col)
        self.reaction_cols.append(reaction_col)
        self.l_cols.append(l_col)
        self.energy_grids.append(energy_grid)
        self.matrices.append(matrix)
        
        # Store metadata
        self.is_relative.append(is_relative)
        self.frame.append(frame)
        
    @classmethod
    def from_endf(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "MF34CovMat":
        """
        Create an MF34CovMat instance from an ENDF file containing MF34 data.
        
        This is a convenience class method that reads an ENDF file, extracts MF34
        (angular distribution covariance) data, and converts it to an MF34CovMat object.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the ENDF file containing MF34 data
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'
            
        Returns
        -------
        MF34CovMat
            MF34CovMat instance with angular distribution covariance data
            
        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file does not contain MF34 data
            
        Examples
        --------
        >>> mf34_covmat = MF34CovMat.from_endf('path/to/endf_file.txt')
        >>> print(f"Loaded {mf34_covmat.num_matrices} angular covariance matrices")
        >>> # Or specify MeV if needed
        >>> mf34_covmat_mev = MF34CovMat.from_endf('path/to/file.txt', energy_unit='MeV')
        
        Notes
        -----
        This method internally:
        1. Reads the ENDF file using read_endf()
        2. Extracts MF34 data
        3. Converts to MF34CovMat using the to_ang_covmat() method
        
        See Also
        --------
        read_endf : Function to read ENDF files
        """
        from pathlib import Path
        from kika.endf import read_endf
        
        # Convert to Path object for consistent handling
        file_path = Path(file_path)
        
        # Read the ENDF file, requesting MF34
        endf = read_endf(file_path, mf_numbers=34)
        
        # Get MF34 data
        mf34 = endf.get_file(34)
        
        if mf34 is None:
            raise ValueError(f"No MF34 (angular distribution covariance) data found in file: {file_path}")
        
        # Convert MF34 to MF34CovMat
        return mf34.to_ang_covmat(energy_unit=energy_unit)

    @classmethod
    def from_covfil(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "MF34CovMat":
        """
        Create an MF34CovMat instance from an NJOY-generated COVFIL/GENDF file.

        Parameters
        ----------
        file_path : str or Path
            Path to the COVFIL/GENDF covariance file
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        MF34CovMat
            MF34CovMat instance loaded from the file

        Raises
        ------
        TypeError
            If the file contains MF33 data instead of MF34
        """
        from pathlib import Path
        from kika.cov.parse_covmat import read_covfil

        file_path = Path(file_path)
        result = read_covfil(str(file_path), energy_unit=energy_unit)
        if not isinstance(result, cls):
            raise TypeError(
                "File contains MF33 data. Use CovMat.from_covfil() instead."
            )
        return result

    def to_covfil(self, file_path: Union[str, 'Path'], tape_label: str = '', temperature: float = 0.0) -> None:
        """
        Write this MF34CovMat to an NJOY COVFIL/GENDF text file.

        Parameters
        ----------
        file_path : str or Path
            Output file path
        tape_label : str, optional
            Label for the tape header line (max 66 chars)
        temperature : float, optional
            Temperature in K for MF1 MT451 CONT record
        """
        from kika.cov.parse_covmat import write_covfil
        write_covfil(self, str(file_path), tape_label=tape_label, temperature=temperature)

    # ------------------------------------------------------------------
    # User-friendly methods
    # ------------------------------------------------------------------

    def summary(self) -> 'pd.DataFrame':
        """
        Create a summary DataFrame with one row per matrix.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: isotope_row, reaction_row, L_row, isotope_col, 
            reaction_col, L_col, NE (len(energy_grid)), M (NE-1), is_relative, frame
        """
        
        data = {
            "isotope_row": self.isotope_rows,
            "MT_row": self.reaction_rows, 
            "L_row": self.l_rows,
            "isotope_col": self.isotope_cols,
            "MT_col": self.reaction_cols,
            "L_col": self.l_cols,
            "NE": [len(grid) for grid in self.energy_grids],
            "is_relative": self.is_relative,
            "frame": self.frame
        }
        
        return pd.DataFrame(data)

    def describe(self, i: int) -> str:
        """
        Pretty single-matrix summary in plain text.
        
        Parameters
        ----------
        i : int
            Index of the matrix to describe
            
        Returns
        -------
        str
            Human-readable description of the matrix
        """
        if i < 0 or i >= len(self.matrices):
            return f"Matrix index {i} out of range [0, {len(self.matrices)-1}]"
        
        matrix = self.matrices[i]
        energy_grid = self.energy_grids[i]
        
        desc = [
            f"Matrix {i}:",
            f"  Reaction: {self.isotope_rows[i]} MT{self.reaction_rows[i]} (L={self.l_rows[i]}) ↔ {self.isotope_cols[i]} MT{self.reaction_cols[i]} (L={self.l_cols[i]})",
            f"  Shape: {matrix.shape}, Energy grid: {len(energy_grid)} points ({len(energy_grid)-1} intervals)",
            f"  Type: {'Relative' if self.is_relative[i] else 'Absolute'}",
            f"  Reference frame: {self.frame[i]}",
        ]
        
        return '\n'.join(desc)



    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_matrices(self) -> int:
        """
        Get the number of covariance matrices stored.
        
        Returns
        -------
        int
            Number of matrices
        """
        return len(self.matrices)
    
    @property
    def isotopes(self) -> Set[int]:
        """
        Get the set of unique isotope IDs in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Sorted list of unique isotope IDs
        """
        return sorted(set(self.isotope_rows + self.isotope_cols))
    
    @property
    def reactions(self) -> Set[int]:
        """
        Get the set of unique reaction MT numbers in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Sorted list of unique reaction MT numbers
        """
        return sorted(set(self.reaction_rows + self.reaction_cols))
    
    @property
    def legendre_indices(self) -> Set[int]:
        """
        Get the set of unique Legendre coefficient indices in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Sorted list of unique Legendre coefficient indices
        """
        return sorted(set(self.l_rows + self.l_cols))
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full covariance matrix.
        
        Constructs the complete covariance matrix by assembling all sub-matrices
        with proper energy grid alignment using union grids.
        
        Returns
        -------
        np.ndarray
            Full covariance matrix of shape (N*G_max, N*G_max) where N is the
            number of unique (isotope, reaction, L) triplets and G_max is the
            maximum number of energy bins across all triplets
        """
        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}
        unions = getattr(self, "_union_grids", None) or self.compute_union_energy_grids()
        # number of bins (not points) per triplet on the union
        Gmap = {t: len(unions[t]) - 1 for t in param_triplets}
        max_G = max(Gmap.values()) if Gmap else 0
        N = len(param_triplets) * max_G
        full = np.full((N, N), np.nan, dtype=float)

        for ir, rr, lr, ic, rc, lc, matrix, grid in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.matrices, self.energy_grids
        ):
            tr = (ir, rr, lr); tc = (ic, rc, lc)
            i, j = idx_map[tr], idx_map[tc]
            # lift Σ to (union_r × union_c)
            Ar = self._lift_matrix(np.asarray(grid), unions[tr])
            Ac = self._lift_matrix(np.asarray(grid), unions[tc])
            Sigma = Ar @ matrix @ Ac.T

            Gr, Gc = Gmap[tr], Gmap[tc]
            r0, r1 = i*max_G, i*max_G + Gr
            c0, c1 = j*max_G, j*max_G + Gc
            full[r0:r1, c0:c1] = Sigma
            if i != j:
                full[c0:c1, r0:r1] = Sigma.T
        return full


    @property 
    def correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix computed from the covariance matrix.
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        """
        from kika.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=False, force_diagonal=True)

    @property
    def clipped_correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix clipped to [-1, 1] range.
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        """
        from kika.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=True, force_diagonal=True)

    @property
    def log_covariance_matrix(self) -> np.ndarray:
        """
        Return the log-space covariance matrix.
        Converts relative covariance to log-space using log1p transformation.
        """
        cov_rel = self.covariance_matrix
        Sigma_log = np.log1p(cov_rel)
        return Sigma_log

    def has_uniform_energy_grid(self) -> bool:
        """
        Check if all matrices have the same energy grid.
        
        Returns
        -------
        bool
            True if all energy grids are identical, False otherwise.
            Returns True for empty collections (vacuous truth).
        """
        if not self.energy_grids:
            return True
        
        # Compare all grids to the first one
        first_grid = self.energy_grids[0]
        
        for grid in self.energy_grids[1:]:
            # Check if lengths are different
            if len(grid) != len(first_grid):
                return False
            
            # Check if values are different (using numpy for numerical comparison)
            if not np.allclose(grid, first_grid, rtol=1e-15, atol=1e-15):
                return False
        
        return True

    def get_union_energy_grids(self):
        return getattr(self, "_union_grids", None) or self.compute_union_energy_grids()


    # ------------------------------------------------------------------
    # General methods
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the angular covariance matrix data to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, L_H, ISO_V, REAC_V, L_V, ENE, STD
        """
        # Convert matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.matrices]
        
        # Create DataFrame
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "L_H": self.l_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "L_V": self.l_cols,
            "ENE": self.energy_grids,
            "STD": matrix_lists
        }
        
        return pd.DataFrame(data)
    
    def to_heatmap_data(
        self,
        nuclide: Union[int, str],
        mt: int,
        legendre_coeffs: Union[int, List[int], Tuple[int, int]],
        *,
        matrix_type: str = 'corr',
        scale: str = 'log',
        energy_range: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> 'MF34HeatmapData':
        """
        Prepare MF34 covariance heatmap data for PlotBuilder rendering.
        
        This method handles the complex MF34 matrix structure including per-Legendre
        energy grids and prepares data for visualization.
        
        Parameters
        ----------
        nuclide : int or str
            Isotope identifier. Can be either:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
        mt : int
            MT reaction number
        legendre_coeffs : int, list of int, or tuple of (row_l, col_l)
            Legendre coefficient(s). Can be:
            - Single int: diagonal block for that L
            - List of ints: diagonal blocks for those L values
            - Tuple of (row_l, col_l): off-diagonal block between row and column L
        matrix_type : str, default 'corr'
            Type of matrix: 'corr'/'correlation' for correlation matrix,
            or 'cov'/'covariance' for covariance matrix
        scale : str, default 'log'
            Energy axis scale: 'log'/'logarithmic' or 'lin'/'linear'
        energy_range : tuple of float, optional
            Energy window (emin, emax). Only bins overlapping the window are kept.
        **kwargs
            Additional parameters (reserved for future use)
            
        Returns
        -------
        MF34HeatmapData
            Heatmap data object ready for PlotBuilder.add_heatmap()
            
        Examples
        --------
        >>> # Simple usage with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> heatmap_data = mf34_covmat.to_heatmap_data(
        ...     nuclide=92235, mt=2, legendre_coeffs=[1, 2, 3]
        ... )
        >>> fig = PlotBuilder(style='light').add_heatmap(heatmap_data)
        >>> fig.show()
        
        >>> # Can also use string symbols
        >>> heatmap_data = mf34_covmat.to_heatmap_data(
        ...     nuclide='U235', mt=2, legendre_coeffs=1, matrix_type='cov'
        ... )
        """
        from kika.plotting.plot_data import MF34HeatmapData
        from kika._utils import zaid_to_symbol, symbol_to_zaid
        
        # Convert nuclide to isotope (ZAID) if string
        if isinstance(nuclide, str):
            isotope = symbol_to_zaid(nuclide)
        else:
            isotope = nuclide
        
        # Normalize matrix_type parameter
        matrix_type_normalized = matrix_type.lower()
        if matrix_type_normalized in ("corr", "correlation"):
            matrix_type_normalized = "corr"
        elif matrix_type_normalized in ("cov", "covariance"):
            matrix_type_normalized = "cov"
        else:
            raise ValueError(
                f"matrix_type must be 'corr'/'correlation' or 'cov'/'covariance', got '{matrix_type}'"
            )
        
        # Normalize scale parameter
        scale_normalized = scale.lower()
        if scale_normalized in ("log", "logarithmic"):
            scale_normalized = "log"
        elif scale_normalized in ("lin", "linear"):
            scale_normalized = "linear"
        else:
            raise ValueError(
                f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'"
            )

        def _transform_edges(edges: np.ndarray) -> np.ndarray:
            if scale_normalized == "log":
                safe = np.maximum(edges, 1e-300)
                return np.log10(safe.astype(float))
            return edges.astype(float)

        def _crop_edges(edges: np.ndarray) -> np.ndarray:
            if energy_range is None:
                return edges
            emin, emax = energy_range
            if not (np.isfinite(emin) and np.isfinite(emax)) or emin >= emax:
                raise ValueError("energy_range must be a tuple (emin, emax) with emin < emax.")
            keep_mask = (edges[1:] > float(emin)) & (edges[:-1] < float(emax))
            if not np.any(keep_mask):
                raise ValueError("energy_range removed all groups; nothing to plot.")
            first, last = np.where(keep_mask)[0][[0, -1]]
            return edges[first:last + 2]

        # 1. Filter by isotope and MT
        filtered_mf34 = self.filter_by_isotope_reaction(isotope, mt)
        
        if filtered_mf34.num_matrices == 0:
            raise ValueError(f"No matrices found for isotope {isotope}, MT {mt}")
        
        # Get all available Legendre coefficients for this isotope/MT
        all_triplets = filtered_mf34._get_param_triplets()
        available_legendre = sorted(list(set(t[2] for t in all_triplets if t[0] == isotope and t[1] == mt)))
        
        if not available_legendre:
            raise ValueError(f"No Legendre coefficients found for isotope {isotope}, MT {mt}")
        
        # 2. Parse legendre_coeffs input
        if isinstance(legendre_coeffs, tuple) and len(legendre_coeffs) == 2:
            # Off-diagonal block
            is_diagonal = False
            row_l, col_l = legendre_coeffs
            legendre_list = [row_l, col_l]
        elif isinstance(legendre_coeffs, int):
            # Single L diagonal block
            is_diagonal = True
            legendre_list = [legendre_coeffs]
        else:
            # Multiple L diagonal blocks
            is_diagonal = True
            legendre_list = sorted(list(legendre_coeffs))
            # Fallback to all available if empty list (same behavior as old implementation)
            if not legendre_list:
                legendre_list = available_legendre
        
        # Validate requested Legendre coefficients
        for l_val in legendre_list:
            if l_val not in available_legendre:
                raise ValueError(f"Legendre coefficient L={l_val} not available for isotope {isotope}, MT {mt}. "
                               f"Available: {available_legendre}")
        
        # 3. Build (cropped) union grids so matrix geometry matches plotting geometry
        union_grids_full = filtered_mf34.compute_union_energy_grids()
        union_grids_cropped = {t: _crop_edges(np.asarray(g, dtype=float)) for t, g in union_grids_full.items()}
        filtered_mf34._union_grids = union_grids_cropped

        triplets = filtered_mf34._get_param_triplets()
        triplet_index = {t: i for i, t in enumerate(triplets)}
        G_map = {t: len(union_grids_cropped[t]) - 1 for t in triplets}
        max_G = max(G_map.values()) if G_map else 0

        # 4. Extract matrix with applied energy cropping
        # Always get correlation matrix to use its NaN mask for identifying "no-data" regions
        corr_matrix_all = filtered_mf34.clipped_correlation_matrix
        
        if matrix_type_normalized == 'corr':
            matrix_full_all = corr_matrix_all
            mask_value = 0.0
        else:  # 'cov'
            cov_matrix_all = filtered_mf34.covariance_matrix
            # Apply NaN mask from correlation matrix to covariance matrix
            # Where correlation is NaN (no data), covariance should also be NaN
            cov_matrix_all = cov_matrix_all.copy()
            cov_matrix_all[~np.isfinite(corr_matrix_all)] = np.nan
            matrix_full_all = cov_matrix_all
            # Use mask_value=0.0 for covariance too: zero covariance in off-diagonal
            # regions indicates no data from the lifting operation
            mask_value = 0.0

        # 5. Select rows/cols for requested Legendre coefficients
        energy_grids_dict: Dict[int, np.ndarray] = {}
        G_per_L: Dict[int, int] = {}
        ranges_dict: Dict[int, Tuple[int, int]] = {}
        energy_ranges: Dict[int, Tuple[float, float]] = {}
        edges_transformed_map: Dict[int, np.ndarray] = {}

        def _get_triplet_for_L(L: int) -> Tuple[int, int, int]:
            for t in triplets:
                if t[2] == L:
                    return t
            raise ValueError(f"Legendre coefficient L={L} not available after filtering for isotope {isotope}, MT {mt}.")

        if is_diagonal:
            selected_indices: List[int] = []
            for l_val in legendre_list:
                t = _get_triplet_for_L(l_val)
                g_len = G_map.get(t, 0)
                if g_len <= 0:
                    continue
                block_start = triplet_index[t] * max_G
                selected_indices.extend(range(block_start, block_start + g_len))
                G_per_L[l_val] = g_len
                energy_grids_dict[l_val] = union_grids_cropped[t]

            # Slice matrix to selected coefficients only (preserving requested order)
            matrix_full = matrix_full_all[np.ix_(selected_indices, selected_indices)]

            # Recompute contiguous ranges in the sliced matrix
            current_pos = 0
            x_edges_parts = []
            for i, l_val in enumerate(legendre_list):
                g_len = G_per_L.get(l_val, 0)
                ranges_dict[l_val] = (current_pos, current_pos + g_len)
                raw_edges = energy_grids_dict.get(l_val)
                if raw_edges is not None and raw_edges.size > 0:
                    transformed = _transform_edges(raw_edges)
                    # Start each block at the previous block's end to keep global coords
                    offset = x_edges_parts[-1][-1] if x_edges_parts else 0.0
                    edges_global = (transformed - transformed[0]) + offset
                    edges_transformed_map[l_val] = edges_global
                    energy_ranges[l_val] = (edges_global[0], edges_global[-1])
                    if i == 0:
                        x_edges_parts.append(edges_global)
                    else:
                        x_edges_parts.append(edges_global[1:])
                    current_pos += g_len
                else:
                    energy_ranges[l_val] = (current_pos, current_pos + g_len)
                    current_pos += g_len

            x_edges = np.concatenate(x_edges_parts) if x_edges_parts else None
            y_edges = x_edges.copy() if x_edges is not None else None
        else:
            row_l, col_l = legendre_list
            row_triplet = _get_triplet_for_L(row_l)
            col_triplet = _get_triplet_for_L(col_l)
            G_row = G_map.get(row_triplet, 0)
            G_col = G_map.get(col_triplet, 0)

            row_indices = list(range(triplet_index[row_triplet] * max_G, triplet_index[row_triplet] * max_G + G_row))
            col_indices = list(range(triplet_index[col_triplet] * max_G, triplet_index[col_triplet] * max_G + G_col))

            matrix_full = matrix_full_all[np.ix_(row_indices, col_indices)]

            energy_grids_dict[row_l] = union_grids_cropped[row_triplet]
            energy_grids_dict[col_l] = union_grids_cropped[col_triplet]
            G_per_L[row_l] = G_row
            G_per_L[col_l] = G_col
            ranges_dict[row_l] = (0, G_row)
            ranges_dict[col_l] = (0, G_col)

            y_edges = None
            if energy_grids_dict[row_l].size > 0:
                y_edges = _transform_edges(energy_grids_dict[row_l])
                y_edges = y_edges - y_edges[0]
                edges_transformed_map[row_l] = y_edges
                energy_ranges[row_l] = (y_edges[0], y_edges[-1])

            x_edges = None
            if energy_grids_dict[col_l].size > 0:
                x_edges = _transform_edges(energy_grids_dict[col_l])
                x_edges = x_edges - x_edges[0]
                edges_transformed_map[col_l] = x_edges
                energy_ranges[col_l] = (x_edges[0], x_edges[-1])

        extent = None
        if x_edges is not None and y_edges is not None:
            extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))

        block_info = {
            'legendre_coeffs': legendre_list,
            'G_per_L': G_per_L,
            'ranges': ranges_dict,
            'energy_ranges': energy_ranges,
            'edges_transformed': edges_transformed_map,
        }

        # 6. Compute uncertainties (rendering controlled at plot time)
        uncertainty_data = {}
        cov_full = filtered_mf34.covariance_matrix
        for l_val in legendre_list:
            t = _get_triplet_for_L(l_val)
            g_len = G_map.get(t, 0)
            if g_len <= 0:
                continue
            base = triplet_index[t] * max_G
            diag_variance = np.diag(cov_full[base: base + g_len, base: base + g_len])
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_percent = np.sqrt(np.abs(diag_variance)) * 100
                sigma_percent = np.nan_to_num(sigma_percent, nan=0.0, posinf=0.0, neginf=0.0)
            uncertainty_data[l_val] = sigma_percent
        if not uncertainty_data:
            uncertainty_data = None

        # 6. Generate label
        isotope_symbol = zaid_to_symbol(isotope)
        matrix_type_label = "Covariance" if matrix_type_normalized == "cov" else "Correlation"
        if is_diagonal:
            if len(legendre_list) == 1:
                label = f"{isotope_symbol} MT:{mt} L={legendre_list[0]} {matrix_type_label}"
            else:
                label = f"{isotope_symbol} MT:{mt} Angular Distribution {matrix_type_label}"
        else:
            label = f"{isotope_symbol} MT:{mt} L={legendre_list[0]} vs L={legendre_list[1]} {matrix_type_label}"
        
        # 7. Create and return MF34HeatmapData
        heatmap_data = MF34HeatmapData(
            matrix_data=matrix_full,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_list,
            matrix_type=matrix_type_normalized,
            scale=scale_normalized,
            extent=extent,
            x_edges=x_edges,
            y_edges=y_edges,
            block_info=block_info,
            uncertainty_data=uncertainty_data,
            energy_grids=energy_grids_dict,
            is_diagonal=is_diagonal,
            mask_value=mask_value,
            label=label
        )
        for key, val in kwargs.items():
            if key == "mask_color":
                continue  # mask color is fixed to lightgray
            if hasattr(heatmap_data, key):
                setattr(heatmap_data, key, val)
            else:
                heatmap_data.metadata[key] = val
        return heatmap_data
    
    def plot_covariance_heatmap(
        self,
        nuclide: Union[int, str],
        mt: int,
        legendre_coeffs: Union[int, List[int], Tuple[int, int]],
        ax: Optional['plt.Axes'] = None,
        *,
        matrix_type: str = "corr",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_family: str = "serif",
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        show_uncertainties: bool = False,
        cmap: Optional[any] = None,
        scale: str = "log",
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = "default",
        **imshow_kwargs,
    ) -> 'plt.Figure':
        """
        Draw a covariance or correlation matrix heatmap for MF34 angular distribution data.

        Parameters
        ----------
        nuclide : int or str
            Isotope identifier. Can be either:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
        mt : int
            Reaction MT number
        legendre_coeffs : int, list of int, or tuple of (row_l, col_l)
            Legendre coefficient(s). Can be:
            - Single int: diagonal block for that L
            - List of ints: diagonal blocks for those L values
            - Tuple of (row_l, col_l): off-diagonal block between row and column L
        ax : plt.Axes, optional
            Matplotlib axes to draw into (deprecated, only used when show_uncertainties=False)
        matrix_type : str, default "corr"
            Type of matrix to plot: "corr"/"correlation" for correlation matrix,
            or "cov"/"covariance" for covariance matrix
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Dots per inch for figure resolution
        font_family : str
            Font family for text elements
        vmax, vmin : float, optional
            Color scale limits
        show_uncertainties : bool
            Whether to show uncertainty plots above the heatmap
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for the heatmap. Can be a string name of any matplotlib 
            colormap (e.g., 'viridis', 'plasma', 'RdYlBu', 'coolwarm') or a matplotlib 
            Colormap object. If None, defaults to 'RdYlGn' for correlation matrices 
            and 'viridis' for covariance matrices.
        scale : str, default "log"
            Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
        energy_range : tuple of float, optional
            Energy range (min, max) for filtering. Values in eV.
        title : str or None, default "default"
            Plot title. If "default", auto-generates from nuclide, MT, and Legendre coefficient.
            If a string, uses that as the title. If None, suppresses the title.
        **imshow_kwargs
            Additional arguments passed to imshow (deprecated)

        Returns
        -------
        plt.Figure
            The matplotlib figure containing the heatmap and optional uncertainty plots
        """
        from kika.plotting.covariance import plot_mf34_covariance_heatmap as _plot_new

        return _plot_new(
            mf34_covmat=self,
            nuclide=nuclide,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            matrix_type=matrix_type,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            vmax=vmax,
            vmin=vmin,
            show_uncertainties=show_uncertainties,
            cmap=cmap,
            energy_range=energy_range,
            scale=scale,
            title=title,
        )

    def plot_uncertainties(
        self,
        isotope: int,
        mt: int,
        legendre_coeffs: Union[int, List[int]],
        ax: Optional['plt.Axes'] = None,
        *,
        uncertainty_type: str = "relative",
        style: str = "default",
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 100,
        font_family: str = "serif",
        legend_loc: str = "best",
        energy_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> 'plt.Figure':
        """
        Plot uncertainties for MF34 angular distribution data for specific Legendre coefficients.
        
        This method extracts and plots the diagonal uncertainties from the covariance matrix
        for the specified isotope, MT reaction, and Legendre coefficients.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        legendre_coeffs : int or list of int
            Legendre coefficient(s) to plot uncertainties for.
            Can be a single int or a list of ints.
        ax : plt.Axes, optional
            Matplotlib axes to draw into. If None, creates new figure.
        uncertainty_type : str, default "relative"
            Type of uncertainty to plot: "relative" (%) or "absolute"
        style : str, default "default"
            Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
        figsize : tuple, default (8, 5)
            Figure size in inches (width, height)
        dpi : int, default 100
            Dots per inch for figure resolution
        font_family : str, default "serif"
            Font family for text elements
        legend_loc : str, default "best"
            Legend location
        energy_range : tuple of float, optional
            Energy range (min, max) for x-axis. If None, uses the full data range.
            Values are used directly without clamping to data range.
        **kwargs
            Additional arguments passed to matplotlib plot functions
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the uncertainty plots
        
        Examples
        --------
        Plot relative uncertainties for Legendre coefficients L=1,2,3:
        
        >>> fig = mf34_covmat.plot_uncertainties(isotope=92235, mt=2, 
        ...                                     legendre_coeffs=[1, 2, 3])
        >>> fig.show()
        
        Plot absolute uncertainties for a single Legendre coefficient:
        
        >>> fig = mf34_covmat.plot_uncertainties(isotope=92235, mt=2,
        ...                                     legendre_coeffs=1, 
        ...                                     uncertainty_type="absolute")
        >>> fig.show()
        """
        from kika.cov.mf34cov_heatmap import plot_mf34_uncertainties as _plot_unc

        return _plot_unc(
            mf34_covmat=self,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            uncertainty_type=uncertainty_type,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            energy_range=energy_range,
            **kwargs
        )

    def to_plot_data(
        self,
        nuclide: Union[int, str],
        mt: int,
        order: int,
        sigma: float = 1.0,
        uncertainty_type: str = 'relative',
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for Legendre coefficient uncertainties.
        
        This is a convenience method to easily convert MF34 covariance data into
        a plottable format using the new plotting infrastructure.
        
        Parameters
        ----------
        nuclide : int or str
            Isotope identifier. Can be either:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
        mt : int
            Reaction MT number
        order : int
            Legendre polynomial order
        sigma : float, default 1.0
            Sigma level for uncertainty scaling (e.g., 1.0 for 1σ, 2.0 for 2σ)
        uncertainty_type : str, default 'relative'
            Type of uncertainty: 'relative' (%) or 'absolute'
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
            Note: Energy values are returned in eV (native ENDF-6 format) to ensure
            compatibility when combining with MF4 data.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        tuple of (None, LegendreUncertaintyPlotData)
            Tuple containing:
            - None: MF34 does not contain Legendre coefficient values (only uncertainties)
            - unc_data: Uncertainty data for the Legendre coefficients
            
        Raises
        ------
        ValueError
            If uncertainty data is not available for the specified parameters
            
        Examples
        --------
        >>> # Extract uncertainty data from MF34CovMat - both notation styles work
        >>> mf34_covmat = endf.mf[34].mt[2].to_ang_covmat()
        >>> coeff_data, unc_data = mf34_covmat.to_plot_data(nuclide=26056, mt=2, order=1)
        >>> # Note: coeff_data will be None (MF34 only has uncertainties, not values)
        >>> 
        >>> # Build a plot with just uncertainties
        >>> from kika.plotting import PlotBuilder
        >>> fig = PlotBuilder().add_data(unc_data).build()
        """
        from kika.plotting import LegendreUncertaintyPlotData
        from kika._utils import zaid_to_symbol, symbol_to_zaid
        
        # Convert nuclide to isotope (ZAID) if string
        if isinstance(nuclide, str):
            isotope = symbol_to_zaid(nuclide)
        else:
            isotope = nuclide
        
        # Get uncertainty data
        unc_data = self.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
        
        if unc_data is None:
            raise ValueError(
                f"No uncertainty data available for isotope={isotope}, MT={mt}, L={order}"
            )
        
        # Extract energies and uncertainties
        energies = unc_data['energies']
        uncertainties = unc_data['uncertainties']
        
        # Keep energies in original units (as stored in energy_grids)
        energies_arr = np.asarray(energies, dtype=float)
        
        # Get energy bin boundaries (also in eV)
        energy_bins = None
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols
        )):
            # Look for diagonal variance matrix (L = L) for the specified parameters
            if (iso_r == isotope and iso_c == isotope and 
                mt_r == mt and mt_c == mt and 
                l_r == order and l_c == order):
                energy_bins = np.array(self.energy_grids[i], dtype=float)  # Keep in original units
                break
        
        # Convert to percentage if relative and apply sigma multiplier
        if uncertainty_type.lower() == 'relative':
            uncertainties = uncertainties * 100.0 * sigma  # Convert to percentage with sigma
        else:
            uncertainties = uncertainties * sigma  # Apply sigma to absolute values
        
        # Generate label if not provided
        if label is None:
            isotope_symbol = zaid_to_symbol(isotope)
            sigma_str = f"{sigma}σ" if sigma != 1.0 else "σ"
            if uncertainty_type.lower() == 'relative':
                label = f"{isotope_symbol} MT={mt} L={order} ({sigma_str} %)"
            else:
                label = f"{isotope_symbol} MT={mt} L={order} ({sigma_str} abs)"
        
        # For step plots with histogram data:
        # - energies has N+1 bin boundaries
        # - uncertainties has N values (one per bin)
        # For proper step plotting with where='post', we need to duplicate the last
        # uncertainty value so that the last bin is drawn extending to the last boundary
        if len(energies_arr) == len(uncertainties) + 1:
            # Append the last uncertainty value to match the energy boundaries length
            uncertainties = np.append(uncertainties, uncertainties[-1])
        
        # Create PlotData object
        unc_data = LegendreUncertaintyPlotData(
            x=energies_arr,
            y=uncertainties,
            label=label,
            order=order,
            isotope=zaid_to_symbol(isotope),
            mt=mt,
            uncertainty_type=uncertainty_type,
            sigma=sigma,
            energy_bins=energy_bins,
            **styling_kwargs
        )
        
        # Return tuple for API consistency (None, unc_data)
        # MF34 does not contain Legendre coefficient values, only uncertainties
        return None, unc_data

    def filter_by_isotope_reaction(self, isotope: int, mt: int) -> "MF34CovMat":
        """
        Return a new MF34CovMat containing only matrices for the specified isotope and MT reaction.
        
        This method filters the covariance matrices to include only those where both
        row and column parameters match the specified isotope and MT reaction.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to filter by
        mt : int
            Reaction MT number to filter by
            
        Returns
        -------
        MF34CovMat
            New MF34CovMat object containing only the filtered matrices
        """
        # Find indices where both row and column match the specified isotope and MT
        matching_indices = []
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, 
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == isotope and mt_r == mt and iso_c == isotope and mt_c == mt:
                matching_indices.append(i)
        
        # Create new MF34CovMat with filtered data
        filtered_mf34 = MF34CovMat()
        
        for i in matching_indices:
            filtered_mf34.isotope_rows.append(self.isotope_rows[i])
            filtered_mf34.reaction_rows.append(self.reaction_rows[i])
            filtered_mf34.l_rows.append(self.l_rows[i])
            filtered_mf34.isotope_cols.append(self.isotope_cols[i])
            filtered_mf34.reaction_cols.append(self.reaction_cols[i])
            filtered_mf34.l_cols.append(self.l_cols[i])
            filtered_mf34.energy_grids.append(self.energy_grids[i])
            filtered_mf34.matrices.append(self.matrices[i])
            filtered_mf34.is_relative.append(self.is_relative[i])
            filtered_mf34.frame.append(self.frame[i])

        filtered_mf34.energy_unit = self.energy_unit

        return filtered_mf34

    def get_uncertainties_for_legendre_coefficient(
        self, 
        isotope: int, 
        mt: int, 
        l_coefficient: Union[int, List[int]],
    ) -> Union[Optional[Dict[str, np.ndarray]], Dict[int, Optional[Dict[str, np.ndarray]]]]:
        """
        Extract standard uncertainties (square root of diagonal variance) for Legendre coefficient(s).
        
        **IMPORTANT**: MF34 data is typically stored as RELATIVE covariances (fractional uncertainties δA_ℓ/A_ℓ).
        This method returns the uncertainties as stored in MF34, along with an 'is_relative' flag.
        
        To convert relative uncertainties to absolute: σ_abs = σ_rel × |A_ℓ|
        where A_ℓ are the Legendre coefficients from ENDF MF=4 data.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        l_coefficient : int or list of int
            Legendre coefficient index (L value) or list of L values
            
        Returns
        -------
        dict or dict of dicts
            For single int: Dictionary containing:
                - 'energies': np.ndarray - Energy bin boundaries (N+1 points for N bins) in eV or MeV
                - 'uncertainties': np.ndarray - Uncertainties (√diagonal of covariance) for each bin
                - 'is_relative': bool - True if relative (δA_ℓ/A_ℓ), False if absolute (δA_ℓ)
            For list of ints: Dictionary mapping L coefficient to uncertainty data (or None if not found).
            
        Notes
        -----
        - If is_relative=True, you must convert to absolute uncertainties by multiplying
          by the Legendre coefficients A_ℓ from ENDF MF=4 before using in propagation formulas.
        - The LB flag in ENDF-6 format determines if data is relative (LB=1,2,5) or absolute (LB=0).
        - Energies are returned as BIN BOUNDARIES, not bin centers. Each uncertainty value applies
          to the energy bin defined by consecutive boundary pairs [E[i], E[i+1]).
        """
        # Handle single coefficient case
        if isinstance(l_coefficient, int):
            # Find the matrix for this specific (isotope, mt, l_coefficient) combination
            matrix_is_relative = None
            for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, energy_grid, matrix) in enumerate(zip(
                self.isotope_rows, self.reaction_rows, self.l_rows,
                self.isotope_cols, self.reaction_cols, self.l_cols,
                self.energy_grids, self.matrices
            )):
                # Look for diagonal variance matrix (L = L) for the specified parameters
                if (iso_r == isotope and iso_c == isotope and 
                    mt_r == mt and mt_c == mt and 
                    l_r == l_coefficient and l_c == l_coefficient):
                    
                    # Store whether this matrix is relative
                    matrix_is_relative = self.is_relative[i]
                    
                    # Extract diagonal elements (variances) and take square root
                    diagonal_variances = np.diag(matrix)
                    
                    # Check for negative variances (which shouldn't happen for diagonal blocks)
                    if np.any(diagonal_variances < 0):
                        # Handle negative variances by setting them to zero
                        diagonal_variances = np.maximum(diagonal_variances, 0.0)
                    
                    uncertainties = np.sqrt(diagonal_variances)
                    
                    # Energy grid contains bin boundaries directly
                    energy_array = np.array(energy_grid)
                    
                    # Ensure we have the correct number of boundaries for uncertainties
                    if len(energy_array) == len(uncertainties) + 1:
                        # Perfect: N+1 boundaries for N uncertainties
                        pass
                    elif len(energy_array) > len(uncertainties) + 1:
                        # Too many energy points - truncate to N+1 boundaries
                        import warnings
                        warnings.warn(
                            f"MF34 data for isotope={isotope}, MT={mt}, L={l_coefficient}: "
                            f"Energy grid has {len(energy_array)} points but only {len(uncertainties)} uncertainties. "
                            f"Expected {len(uncertainties) + 1} energy points. Truncating energy grid.",
                            UserWarning
                        )
                        energy_array = energy_array[:len(uncertainties) + 1]
                    elif len(energy_array) < len(uncertainties) + 1:
                        # Too few energy points - truncate uncertainties to match
                        import warnings
                        warnings.warn(
                            f"MF34 data for isotope={isotope}, MT={mt}, L={l_coefficient}: "
                            f"Energy grid has {len(energy_array)} points but {len(uncertainties)} uncertainties. "
                            f"Expected {len(uncertainties) + 1} energy points. Truncating uncertainties.",
                            UserWarning
                        )
                        uncertainties = uncertainties[:len(energy_array) - 1]
                    
                    return {
                        'energies': energy_array,  # Bin boundaries (N+1 points for N bins)
                        'uncertainties': uncertainties,
                        'is_relative': matrix_is_relative
                    }
            
            return None
        
        # Handle list of coefficients case
        elif isinstance(l_coefficient, (list, tuple)):
            result = {}
            for l_coeff in l_coefficient:
                result[l_coeff] = self.get_uncertainties_for_legendre_coefficient(isotope, mt, l_coeff)
            return result
        
        else:
            raise TypeError(f"l_coefficient must be int or list of int, got {type(l_coefficient)}")

    def compute_union_energy_grids(self, atol: float = 1e-12):
        """
        Compute union energy grids for all parameter triplets.
        
        This method creates a unified energy grid for each (isotope, reaction, legendre) 
        triplet by merging all energy grids that involve that triplet, removing duplicates
        within tolerance.
        
        Parameters
        ----------
        atol : float, default 1e-12
            Absolute tolerance for merging energy points
            
        Returns
        -------
        Dict[Tuple[int, int, int], np.ndarray]
            Dictionary mapping (isotope, reaction, legendre) triplets to union energy grids
        """
        triplets = self._get_param_triplets()
        unions = {t: [] for t in triplets}
        for i, grid in enumerate(self.energy_grids):
            row = (self.isotope_rows[i], self.reaction_rows[i], self.l_rows[i])
            col = (self.isotope_cols[i], self.reaction_cols[i], self.l_cols[i])
            unions[row].extend(grid); unions[col].extend(grid)
        # deduplicate with tolerance
        for t, g in unions.items():
            g = np.unique(np.asarray(g, dtype=float))
            merged = [g[0]]
            for x in g[1:]:
                if not np.isclose(x, merged[-1], rtol=0.0, atol=atol):
                    merged.append(x)
            unions[t] = np.array(merged, dtype=float)
        self._union_grids = unions
        return unions

    def validate_union_grids(self, verbose: bool = True) -> bool:
        """
        Validate that union grids are properly constructed and aligned.
        
        Parameters
        ----------
        verbose : bool, default True
            Whether to print validation details
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        try:
            param_triplets = self._get_param_triplets()
            union_grids = self.get_union_energy_grids()
            
            if verbose:
                print(f"Validating union grids for {len(param_triplets)} parameter triplets")
            
            # Check that all triplets have union grids
            missing_grids = [t for t in param_triplets if t not in union_grids]
            if missing_grids:
                if verbose:
                    print(f"ERROR: Missing union grids for {len(missing_grids)} triplets")
                return False
            
            # Check grid properties
            max_G = 0
            for triplet, grid in union_grids.items():
                if len(grid) < 2:
                    if verbose:
                        print(f"WARNING: Triplet {triplet} has insufficient grid points: {len(grid)}")
                    continue
                
                num_bins = len(grid) - 1
                max_G = max(max_G, num_bins)
                
                # Check grid is sorted
                if not np.all(grid[1:] >= grid[:-1]):
                    if verbose:
                        print(f"ERROR: Grid for triplet {triplet} is not sorted")
                    return False
            
            # Check covariance matrix dimensions
            expected_dim = len(param_triplets) * max_G
            actual_shape = self.covariance_matrix.shape
            
            if actual_shape[0] != actual_shape[1]:
                if verbose:
                    print(f"ERROR: Covariance matrix is not square: {actual_shape}")
                return False
            
            if actual_shape[0] != expected_dim:
                if verbose:
                    print(f"ERROR: Covariance matrix dimension mismatch. "
                          f"Expected: {expected_dim}, Actual: {actual_shape[0]}")
                return False
            
            if verbose:
                print(f"Validation PASSED: {len(param_triplets)} triplets, max_G={max_G}, "
                      f"matrix shape={actual_shape}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"ERROR during union grids validation: {e}")
            return False


    # ------------------------------------------------------------------
    # Decomposition methods
    # ------------------------------------------------------------------

    def cholesky_decomposition(
        self,
        *,
        space: str = "log",        # "log" (default) or "linear"
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger = None,
    ) -> np.ndarray:
        """
        Robust Cholesky factor L such that M ≈ L L^T.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        jitter_scale : float
            Base jitter scale for PSD correction
        max_jitter_ratio : float
            Maximum jitter relative to matrix norm
        verbose : bool
            Whether to log progress
        logger : optional
            Logger instance for output
            
        Returns
        -------
        np.ndarray
            Lower triangular Cholesky factor L
        """
        from kika.cov.decomposition import cholesky_decomposition
        return cholesky_decomposition(
            self, 
            space=space, 
            jitter_scale=jitter_scale,
            max_jitter_ratio=max_jitter_ratio,
            verbose=verbose, 
            logger=logger
        )

    def eigen_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        verbose: bool = True,
        logger = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition with optional clipping instead of jitter.

        If ``clip_negatives`` is *True*, negative eigenvalues are set to
        zero and the user is informed of the number of clips and the minimum
        original value.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Whether to clip negative eigenvalues to zero
        verbose : bool
            Whether to log progress
        logger : optional
            Logger instance for output
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Eigenvalues and eigenvectors
        """
        from kika.cov.decomposition import eigen_decomposition
        return eigen_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            verbose=verbose,
            logger=logger
        )

    def svd_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        verbose: bool = True,
        full_matrices: bool = False,
        logger = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD with pre-clipping using eigendecomposition.

        For symmetric matrices, SVD and eigen are equivalent. If
        ``clip_negatives`` is activated, a preliminary eigendecomposition,
        clipping, and reconstruction step is performed before applying SVD,
        ensuring singular values consistent with a PSD matrix.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Whether to clip negative eigenvalues before SVD
        verbose : bool
            Whether to log progress
        full_matrices : bool
            Whether to return full-sized U and V matrices
        logger : optional
            Logger instance for output
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            U, singular values, V^T matrices
        """
        from kika.cov.decomposition import svd_decomposition
        return svd_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            verbose=verbose,
            full_matrices=full_matrices,
            logger=logger
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """String representation showing summary information."""
        unique_isos = len(self.isotopes)
        unique_mts = len(self.reactions)
        unique_ls = len(self.legendre_indices)
        
        return (f"MF34 Angular Covariance Matrix Data:\n" # Updated name
                f"- {self.num_matrices} matrices\n"
                f"- {unique_isos} unique isotopes\n"
                f"- {unique_mts} unique reaction types\n"
                f"- {unique_ls} unique Legendre indices")
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the MF34CovMat object.
        
        Returns
        -------
        str
            String representation with content summary
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'MF34 Angular Distribution Covariance Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of MF34 covariance matrix data
        description = (
            "This object contains covariance matrix data for angular distributions (MF34).\n"
            "Each matrix represents the covariance between Legendre coefficients for specific\n"
            "isotope-reaction pairs across energy groups.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "MF34 Covariance Data Summary:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add summary information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Covariance Matrices", self.num_matrices, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Isotopes", len(self.isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.reactions), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Legendre Indices", len(self.legendre_indices), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".num_matrices": "Get total number of covariance matrices",
            ".isotopes": "Get set of unique isotope IDs",
            ".reactions": "Get set of unique reaction MT numbers",
            ".legendre_indices": "Get set of unique Legendre indices (L values)"
        }
        
        data_access_section = create_repr_section(
            "How to Access Covariance Data:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add a blank line after the section
        data_access_section += "\n"
        
        # Create a section for available methods using create_repr_section
        methods = {
            ".to_dataframe()": "Convert all MF34 covariance data to DataFrame"
            # Add other methods here if they are implemented later
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + data_access_section + methods_section





    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_param_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Return a list of all (isotope, reaction, legendre) triplets present,
        sorted first by isotope, then by reaction, then by legendre coefficient.
        """
        triplets = set(zip(self.isotope_rows, self.reaction_rows, self.l_rows)) \
                 | set(zip(self.isotope_cols, self.reaction_cols, self.l_cols))
        return sorted(triplets, key=lambda t: (t[0], t[1], t[2]))

    def _lift_matrix(self, src_grid, dst_grid):
        """
        Create a lifting matrix to map covariance from source to destination energy grid.
        
        Constructs a mapping matrix A such that when applied, it transforms covariance
        matrices from the source energy grid to the destination (union) energy grid.
        Assumes destination grid is a subset or refinement of source grid.
        
        Parameters
        ----------
        src_grid : array-like
            Source energy grid boundaries (NE points)
        dst_grid : array-like
            Destination energy grid boundaries (NE points)
        
        Returns
        -------
        np.ndarray
            Lifting matrix of shape (Gd, Gs) where Gs = len(src_grid)-1
            and Gd = len(dst_grid)-1
        """
        # src_grid, dst_grid are boundary arrays (NE)
        Gs, Gd = len(src_grid)-1, len(dst_grid)-1
        A = np.zeros((Gd, Gs), dtype=float)
        j = 0
        for g in range(Gd):
            eL, eH = dst_grid[g], dst_grid[g+1]
            while j+1 < len(src_grid) and src_grid[j+1] <= eL + 1e-12:
                j += 1
            # assume dst is subset/refinement of src:
            A[g, j] = 1.0
        return A
