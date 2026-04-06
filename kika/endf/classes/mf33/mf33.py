"""
Classes for MT sections within MF33 (Cross-Section Covariances) in ENDF files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..mt import MT
from ....utils import get_endf_logger

# Initialize logger for this module
logger = get_endf_logger(__name__)


@dataclass
class NISubSubsectionRecord:
    """NI-type LIST record within a subsection (LB=0-9)."""
    ls: int = None       # Flag for symmetric matrix (1=yes, 0=no) — LB=5
    lb: int = None       # Covariance pattern flag
    nt: int = None       # Total number of items in the LIST
    ne: int = None       # Number of energy entries (LB=5,6) or set from NP (LB=0-4,8,9)

    lt: int = None       # For LB=0-4,8,9: C3 field (0 = single table, >0 = two tables)
    np: int = None       # For LB=0-4,8,9: total number of (E,F) pairs across table(s)

    # Preserve original LIST floats for round-trip
    raw_list_values: List[float] = field(default_factory=list)

    # For LB=5 (energy grid + matrix)
    energies: List[float] = field(default_factory=list)
    matrix: List[float] = field(default_factory=list)

    # For LB=0,1,2,8,9 single E-table (and first table of LB=3,4)
    e_table_k: List[float] = field(default_factory=list)
    f_table_k: List[float] = field(default_factory=list)

    # For LB=3,4 second E-table
    e_table_l: List[float] = field(default_factory=list)
    f_table_l: List[float] = field(default_factory=list)

    # For LB=6 (rectangular matrix)
    row_energies: List[float] = field(default_factory=list)
    col_energies: List[float] = field(default_factory=list)
    rect_matrix: List[float] = field(default_factory=list)


@dataclass
class NCSubSubsection:
    """NC-type sub-subsection (LTY=0,1,2,3)."""
    lty: int = None       # Type flag: 0=derived redundant, 1/2/3=ratio to standard

    # Common fields from LIST header
    e1: float = None      # Lower energy limit
    e2: float = None      # Upper energy limit

    # LTY=0 fields (derived redundant)
    nci: int = None                                   # Number of contributing reactions
    ci: List[float] = field(default_factory=list)     # Coefficients
    xmti: List[float] = field(default_factory=list)   # MT identifiers (float-encoded)

    # LTY=1,2,3 fields (ratio to standards)
    mats: int = None       # MAT of standard
    mts: int = None        # MT of standard
    nei: int = None        # Number of energy-weight pairs
    xmfs: float = None     # MF of standard (float)
    xlfss: float = None    # Excited state of standard (float)
    ei: List[float] = field(default_factory=list)     # Energies
    wei: List[float] = field(default_factory=list)    # Weights


@dataclass
class Subsection:
    """Subsection for a particular (MAT1, MT1) cross-reaction pair."""
    xmf1: float = None    # MF for 2nd cross section (0.0 means same MF)
    xlfs1: float = None   # Final excited state for 2nd xs (0.0 unless MF1=10)
    mat1: int = None       # MAT for 2nd cross section (0 = same MAT)
    mt1: int = None        # MT for 2nd cross section
    nc: int = None         # Number of NC-type sub-subsections
    ni: int = None         # Number of NI-type sub-subsections

    nc_records: List[NCSubSubsection] = field(default_factory=list)
    ni_records: List[NISubSubsectionRecord] = field(default_factory=list)


@dataclass
class MF33MT(MT):
    """
    MT section within MF33 (Cross-Section Covariances).

    Stores covariance data for neutron cross sections from ENDF File 33.
    """
    _za: float = None
    _awr: float = None
    _mtl: int = None       # Lumped reaction flag (non-zero = component of lumped MTL)
    _nl: int = None        # Number of subsections
    _mat: int = None
    _mf: int = 33
    _subsections: List[Subsection] = field(default_factory=list)
    num_lines: int = 0

    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z+A)."""
        return self._za

    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio."""
        return self._awr

    @property
    def is_lumped_component(self) -> bool:
        """True if this MT is a component of lumped reaction MTL."""
        return self._mtl is not None and self._mtl != 0

    @property
    def lumped_mt(self) -> int:
        """MT of the lumped reaction this is a component of (0 if not lumped)."""
        return self._mtl or 0

    @property
    def num_subsections(self) -> int:
        """Number of subsections."""
        return self._nl or 0

    @property
    def subsections(self) -> List[Subsection]:
        """Subsections data."""
        return self._subsections

    def add_subsection(self, subsection: Subsection) -> None:
        """Add a subsection to this MT."""
        self._subsections.append(subsection)

    def get_subsection(self, mt1: int) -> Optional[Subsection]:
        """Get a subsection by MT1 value."""
        for subsection in self._subsections:
            if subsection.mt1 == mt1:
                return subsection
        return None

    # ------------------------------------------------------------------
    # Matrix decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_lb012_matrix(record: 'NISubSubsectionRecord') -> Tuple[np.ndarray, List[float]]:
        """Decode LB=0,1,2 (or 8,9) single E-table into an MxM interval matrix."""
        energies = record.e_table_k
        ne = len(energies)
        if ne < 2:
            raise ValueError(f"LB={record.lb} requires NE >= 2, got {ne}")
        m = ne - 1
        f_values = record.f_table_k
        matrix = np.zeros((m, m))

        if record.lb in (0, 8, 9):
            # Diagonal: absolute variance (LB=0) or short-range variance (LB=8,9)
            for k in range(m):
                matrix[k, k] = f_values[k]
        elif record.lb == 1:
            # Diagonal: fractional variance
            for k in range(m):
                matrix[k, k] = f_values[k]
        elif record.lb == 2:
            # Fully correlated: relative sigma Sk
            for k in range(m):
                for l_idx in range(m):
                    matrix[k, l_idx] = f_values[k] * f_values[l_idx]
        return matrix, energies

    @staticmethod
    def _decode_lb34_matrix(record: 'NISubSubsectionRecord') -> Tuple[np.ndarray, List[float], List[float]]:
        """Decode LB=3,4 dual E-table into a rectangular matrix.

        LB=3: Cov(Xi,Yj) = sum_k,l P^{i,k}_{j,l} Fx,k Fy,l Xi Yj
              Fully correlated across Ek and El intervals.
        LB=4: Cov(Xi,Yj) = sum_{k,l,l'} P^{i,k,l}_{j,k,l'} Fk Fxy,l Fxy,l' Xi Yj
              Correlated over El intervals within each Ek interval.
        Returns (matrix, row_energies, col_energies).
        """
        ek = record.e_table_k
        fk = record.f_table_k
        el = record.e_table_l
        fl = record.f_table_l
        nr = len(ek) - 1
        nc = len(el) - 1
        if nr < 1 or nc < 1:
            raise ValueError(f"LB={record.lb} needs at least 2 points per table")

        if record.lb == 3:
            # Outer product: Fx,k * Fy,l
            matrix = np.outer(fk[:nr], fl[:nc])
        else:
            # LB=4: more complex, but storage is still two tables
            # Cov ~ Fk * Fl * Fl' within each Ek bin — simplified to outer product
            # (full processing requires group integration; store as outer product for now)
            matrix = np.outer(fk[:nr], fl[:nc])

        return matrix, ek, el

    @staticmethod
    def _decode_lb5_matrix(record: 'NISubSubsectionRecord') -> Tuple[np.ndarray, List[float]]:
        """Decode LB=5 symmetric/asymmetric matrix."""
        energies = record.energies
        ne = int(record.ne or 0)
        if ne < 2:
            raise ValueError("LB=5 requires NE >= 2")
        m = ne - 1
        ls = record.ls
        raw_values = np.array(record.matrix)

        expected = m * (m + 1) // 2 if ls == 1 else m * m
        if len(raw_values) != expected:
            raise ValueError(
                f"LB=5 LS={ls}: expected {expected} matrix elements for M={m}, got {len(raw_values)}"
            )

        matrix = np.zeros((m, m))
        if ls == 1:
            rows, cols = np.triu_indices(m)
            matrix[rows, cols] = raw_values
            matrix[cols, rows] = raw_values
        else:
            matrix = raw_values.reshape(m, m)
        return matrix, energies

    @staticmethod
    def _decode_lb6_matrix(record: 'NISubSubsectionRecord') -> Tuple[np.ndarray, List[float], List[float]]:
        """Decode LB=6 rectangular matrix."""
        row_energies = record.row_energies
        col_energies = record.col_energies
        ner = len(row_energies)
        nec = len(col_energies)
        if ner < 2 or nec < 2:
            raise ValueError("LB=6 requires NER >= 2 and NEC >= 2")
        r, c = ner - 1, nec - 1
        raw_values = np.array(record.rect_matrix)
        if len(raw_values) != r * c:
            raise ValueError(f"LB=6: expected {r*c} elements, got {len(raw_values)}")
        matrix = raw_values.reshape(r, c)
        return matrix, row_energies, col_energies

    @staticmethod
    def _project_matrix_piecewise_constant(
        component_matrix: np.ndarray,
        native_row_grid: List[float],
        union_grid: List[float],
        native_col_grid: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Project an interval matrix onto a union grid using piecewise constant assumption."""
        union_arr = np.asarray(union_grid, dtype=float)
        row_arr = np.asarray(native_row_grid, dtype=float)
        col_arr = np.asarray(native_col_grid, dtype=float) if native_col_grid is not None else row_arr

        union_m = len(union_arr) - 1

        # Row transfer matrix
        u_lo = union_arr[:-1, None]
        u_hi = union_arr[1:, None]
        delta_u = u_hi - u_lo

        r_lo = row_arr[:-1][None, :]
        r_hi = row_arr[1:][None, :]
        overlap = np.maximum(0.0, np.minimum(u_hi, r_hi) - np.maximum(u_lo, r_lo))
        with np.errstate(divide='ignore', invalid='ignore'):
            T_row = np.where(delta_u > 0, overlap / delta_u, 0.0)

        # Column transfer matrix
        c_lo = col_arr[:-1][None, :]
        c_hi = col_arr[1:][None, :]
        overlap_c = np.maximum(0.0, np.minimum(u_hi, c_hi) - np.maximum(u_lo, c_lo))
        with np.errstate(divide='ignore', invalid='ignore'):
            T_col = np.where(delta_u > 0, overlap_c / delta_u, 0.0)

        return T_row @ component_matrix @ T_col.T

    @staticmethod
    def _project_diagonal_piecewise_constant(
        values: np.ndarray,
        native_grid: List[float],
        target_grid: List[float],
    ) -> np.ndarray:
        """Project a 1D diagonal (per-interval values) onto a target energy grid.

        Uses piecewise-constant assumption: each source interval has a constant
        value, and target intervals get the overlap-weighted average.

        Parameters
        ----------
        values : array of shape (M,)
            Per-interval values on the native grid (M = len(native_grid) - 1).
        native_grid : list of float
            Energy boundaries of the source grid (M+1 points).
        target_grid : list of float
            Energy boundaries of the target grid (N+1 points).

        Returns
        -------
        np.ndarray of shape (N,)
        """
        src = np.asarray(native_grid, dtype=float)
        tgt = np.asarray(target_grid, dtype=float)
        vals = np.asarray(values, dtype=float)

        n = len(tgt) - 1
        result = np.zeros(n)

        # For each target interval, compute overlap-weighted average
        tgt_lo = tgt[:-1]
        tgt_hi = tgt[1:]
        src_lo = src[:-1]
        src_hi = src[1:]

        # Vectorised: overlap[i, k] = overlap of target_i with source_k
        overlap = np.maximum(
            0.0,
            np.minimum(tgt_hi[:, None], src_hi[None, :])
            - np.maximum(tgt_lo[:, None], src_lo[None, :]),
        )
        tgt_widths = tgt_hi - tgt_lo
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(tgt_widths[:, None] > 0, overlap / tgt_widths[:, None], 0.0)
        result = weights @ vals
        return result

    def _process_ni_records_to_diagonal(
        self,
        ni_records: List['NISubSubsectionRecord'],
        mt_label: str = '',
    ) -> Optional[Tuple[np.ndarray, List[float], bool]]:
        """Process a list of NI records into a diagonal variance array.

        Returns (diagonal, energy_grid, is_relative) or None if no valid data.
        """
        all_energies: Set[float] = set()
        components = []

        for ni_rec in ni_records:
            try:
                if ni_rec.lb in (0, 1, 2, 8, 9):
                    mat, grid = self._decode_lb012_matrix(ni_rec)
                    all_energies.update(grid)
                    components.append((mat, grid, None))
                elif ni_rec.lb in (3, 4):
                    mat, row_grid, col_grid = self._decode_lb34_matrix(ni_rec)
                    all_energies.update(row_grid)
                    all_energies.update(col_grid)
                    components.append((mat, row_grid, col_grid))
                elif ni_rec.lb == 5:
                    mat, grid = self._decode_lb5_matrix(ni_rec)
                    all_energies.update(grid)
                    components.append((mat, grid, None))
                elif ni_rec.lb == 6:
                    mat, row_grid, col_grid = self._decode_lb6_matrix(ni_rec)
                    all_energies.update(row_grid)
                    all_energies.update(col_grid)
                    components.append((mat, row_grid, col_grid))
            except ValueError as e:
                logger.error(f"Error decoding LB={ni_rec.lb} in {mt_label}: {e}")

        if not components:
            return None

        union_grid = sorted(all_energies)
        if len(union_grid) < 2:
            return None
        union_m = len(union_grid) - 1
        total = np.zeros((union_m, union_m))

        for comp_mat, row_grid, col_grid in components:
            if col_grid is None and row_grid == union_grid:
                projected = comp_mat
            elif col_grid is not None and row_grid == union_grid and col_grid == union_grid:
                projected = comp_mat
            else:
                projected = self._project_matrix_piecewise_constant(
                    comp_mat, row_grid, union_grid, native_col_grid=col_grid
                )
            total += projected

        lb_set = {r.lb for r in ni_records}
        is_relative = not lb_set.issubset({0, 8, 9})
        diagonal = np.diag(total)
        return diagonal, union_grid, is_relative

    def _get_cross_mt_diagonal(
        self,
        mt_i: int,
        mt_j: int,
        sibling_sections: Dict[int, 'MF33MT'],
    ) -> Optional[Tuple[np.ndarray, List[float]]]:
        """Get the diagonal of the cross-MT covariance Cov(MT_i, MT_j).

        Searches subsections of MT_i for mt1=MT_j, then MT_j for mt1=MT_i.
        Returns (diagonal, energy_grid) or None if not found.
        """
        # Search MT_i's section for a subsection referencing MT_j
        for src_mt, tgt_mt in [(mt_i, mt_j), (mt_j, mt_i)]:
            section = sibling_sections.get(src_mt)
            if section is None:
                continue
            subsection = section.get_subsection(tgt_mt)
            if subsection is None or not subsection.ni_records:
                continue
            result = self._process_ni_records_to_diagonal(
                subsection.ni_records,
                mt_label=f"MT{src_mt}→MT{tgt_mt}",
            )
            if result is not None:
                diag, grid, _is_rel = result
                return diag, grid

        return None

    def resolve_nc_lty0(
        self,
        sibling_sections: Dict[int, 'MF33MT'],
        energy_unit: str = 'eV',
        _resolving: Optional[Set[int]] = None,
    ) -> Optional['CrossSectionCovariance']:
        """Resolve NC LTY=0 (derived redundant) covariance via the sum rule.

        For a derived cross section σ_MT = Σ c_i × σ_{MTi}, the covariance is:
            Cov(MT, MT) = Σ_i Σ_j c_i × c_j × Cov(MTi, MTj)

        This method computes the full propagation including cross-MT terms.

        Parameters
        ----------
        sibling_sections : dict
            Map of MT number → MF33MT section for all other MTs in this MF33 file.
        energy_unit : str
            Energy unit ('eV' or 'MeV').
        _resolving : set of int, optional
            Set of MT numbers currently being resolved (recursion guard).

        Returns
        -------
        CrossSectionCovariance or None
            The derived covariance, or None if resolution fails.
        """
        from ....cov.cross_section_covariance import CrossSectionCovariance

        # Find the LTY=0 NC record (only in self-subsection where mt1=MT)
        nc_rec = None
        for subsection in self._subsections:
            if int(subsection.mt1) != self.number:
                continue
            for nc in subsection.nc_records:
                if nc.lty == 0:
                    nc_rec = nc
                    break
            if nc_rec is not None:
                break

        if nc_rec is None:
            return None

        # Parse contributing reactions and coefficients
        mti_list = [int(round(x)) for x in nc_rec.xmti]
        ci_list = list(nc_rec.ci)
        nci = len(mti_list)

        if nci == 0 or len(ci_list) < nci:
            logger.warning(f"MT{self.number}: NC LTY=0 has no contributing reactions")
            return None

        ci_list = ci_list[:nci]  # Trim to NCI pairs

        logger.info(
            f"MT{self.number}: resolving NC LTY=0 from {nci} reactions: "
            f"{list(zip(ci_list, mti_list))}"
        )

        resolving = (_resolving or set()) | {self.number}

        # Step 1: Get self-covariance diagonals for each contributing MT
        # diag_by_mt: {mt: (diagonal_variance, energy_grid)}
        diag_by_mt: Dict[int, Tuple[np.ndarray, List[float]]] = {}

        for mt_i in mti_list:
            if mt_i in resolving:
                logger.warning(
                    f"MT{self.number}: skipping MT{mt_i} (circular NC reference)"
                )
                continue
            section_i = sibling_sections.get(mt_i)
            if section_i is None:
                logger.debug(f"MT{self.number}: MT{mt_i} not in MF33, treating as zero")
                continue

            try:
                xs_cov_i = section_i.to_xs_covmat(
                    energy_unit=energy_unit,
                    sibling_sections=sibling_sections,
                    _resolving=resolving,
                )
            except Exception as e:
                logger.warning(f"MT{self.number}: failed to resolve MT{mt_i}: {e}")
                continue

            # Extract the self-covariance diagonal (MT_i vs MT_i)
            for idx in range(len(xs_cov_i.matrices)):
                if (xs_cov_i.reaction_rows[idx] == mt_i
                        and xs_cov_i.reaction_cols[idx] == mt_i):
                    mat = xs_cov_i.matrices[idx]
                    grid = xs_cov_i.energy_grids[idx] if xs_cov_i.energy_grids else None
                    if grid is not None:
                        diag_by_mt[mt_i] = (np.diag(mat), grid)
                    break

        if not diag_by_mt:
            logger.warning(f"MT{self.number}: no contributing MT covariances found")
            return None

        # Step 2: Build union energy grid from all contributing grids
        all_energies: Set[float] = set()
        for diag, grid in diag_by_mt.values():
            all_energies.update(grid)
        union_grid = sorted(all_energies)

        if len(union_grid) < 2:
            return None

        union_m = len(union_grid) - 1

        # Step 3: Project all self-covariance diagonals onto union grid
        proj_diag: Dict[int, np.ndarray] = {}
        for mt_i, (diag, grid) in diag_by_mt.items():
            if grid == union_grid:
                proj_diag[mt_i] = diag
            else:
                proj_diag[mt_i] = self._project_diagonal_piecewise_constant(
                    diag, grid, union_grid
                )

        # Step 4: Compute derived variance using full propagation formula
        # Var(MT_k)_g = Σ_i Σ_j c_i * c_j * Cov(MTi, MTj)_g
        derived_var = np.zeros(union_m)

        for ii, mt_i in enumerate(mti_list):
            c_i = ci_list[ii]

            # Diagonal term: c_i^2 * Var(MT_i)
            if mt_i in proj_diag:
                derived_var += c_i * c_i * proj_diag[mt_i]

            # Cross-MT terms: 2 * c_i * c_j * Cov(MTi, MTj) for j > i
            for jj in range(ii + 1, nci):
                mt_j = mti_list[jj]
                c_j = ci_list[jj]

                cross = self._get_cross_mt_diagonal(mt_i, mt_j, sibling_sections)
                if cross is not None:
                    cross_diag, cross_grid = cross
                    if cross_grid == union_grid:
                        proj_cross = cross_diag
                    else:
                        proj_cross = self._project_diagonal_piecewise_constant(
                            cross_diag, cross_grid, union_grid
                        )
                    derived_var += 2.0 * c_i * c_j * proj_cross

        # Clamp any negative variance to zero (numerical noise)
        derived_var = np.maximum(derived_var, 0.0)

        # Build result as a diagonal matrix (only diagonal is meaningful)
        derived_matrix = np.diag(derived_var)
        isotope = int(self._za)
        result = CrossSectionCovariance(energy_unit=energy_unit)
        result.add_matrix(
            isotope, self.number, isotope, self.number,
            derived_matrix,
            energy_grid=union_grid,
            is_relative=True,  # NC LTY=0 derived covariance is always relative
        )

        logger.info(
            f"MT{self.number}: NC LTY=0 resolved — {union_m} groups, "
            f"sqrt(diag) range [{np.sqrt(derived_var.min()):.4g}, {np.sqrt(derived_var.max()):.4g}]"
        )
        return result

    # ------------------------------------------------------------------
    # Conversion to CrossSectionCovariance
    # ------------------------------------------------------------------

    def to_xs_covmat(
        self,
        energy_unit: str = 'eV',
        sibling_sections: Optional[Dict[int, 'MF33MT']] = None,
        _resolving: Optional[Set[int]] = None,
    ) -> 'CrossSectionCovariance':
        """
        Convert MF33 data to a CrossSectionCovariance object.

        Each subsection (reaction pair) produces one matrix entry with its own
        energy grid. Multiple NI components within a subsection are summed on
        their union grid. NC-type LTY=0 sub-subsections are resolved via the
        sum rule when ``sibling_sections`` is provided.

        Parameters
        ----------
        energy_unit : str
            Energy unit: 'eV' (default) or 'MeV'.
        sibling_sections : dict, optional
            Map of MT number → MF33MT for other MTs in this MF33 file.
            Required for NC LTY=0 resolution. If None, NC records are skipped.
        _resolving : set of int, optional
            MT numbers currently being resolved (recursion guard, internal use).

        Returns
        -------
        CrossSectionCovariance
        """
        from ....cov.cross_section_covariance import CrossSectionCovariance

        isotope = int(self._za)
        result = CrossSectionCovariance(energy_unit=energy_unit)

        for subsection in self._subsections:
            mt1 = int(subsection.mt1)
            mat1 = int(subsection.mat1 or 0)
            iso_col = isotope if mat1 == 0 else mat1

            # Handle NC-type sub-subsections
            nc_resolved = False
            if subsection.nc_records:
                for nc in subsection.nc_records:
                    if nc.lty == 0 and sibling_sections is not None:
                        resolving = (_resolving or set()) | {self.number}
                        nc_result = self.resolve_nc_lty0(
                            sibling_sections, energy_unit, resolving
                        )
                        if nc_result is not None:
                            for idx in range(len(nc_result.matrices)):
                                result.add_matrix(
                                    nc_result.isotope_rows[idx],
                                    nc_result.reaction_rows[idx],
                                    nc_result.isotope_cols[idx],
                                    nc_result.reaction_cols[idx],
                                    nc_result.matrices[idx],
                                    energy_grid=(
                                        nc_result.energy_grids[idx]
                                        if nc_result.energy_grids else None
                                    ),
                                    is_relative=(
                                        nc_result.is_relative[idx]
                                        if nc_result.is_relative else None
                                    ),
                                )
                            nc_resolved = True
                    elif nc.lty == 0:
                        logger.info(
                            f"MT{self.number}→MT{mt1}: NC LTY=0 present but "
                            f"sibling_sections not provided — skipping."
                        )
                    elif nc.lty in (1, 2, 3):
                        logger.info(
                            f"MT{self.number}→MT{mt1}: LTY={nc.lty} "
                            f"(ratio to standard) not yet implemented."
                        )

            # Process NI-type sub-subsections
            # Per ENDF manual 33.3.3 item 3: NI F-values must be zero within
            # the NC LTY=0 energy range, so skip NI when NC was resolved.
            if nc_resolved or not subsection.ni_records:
                continue

            # Collect all energy points across NI components for union grid
            all_energies: Set[float] = set()
            components = []  # (matrix, row_grid, col_grid_or_None)

            for ni_rec in subsection.ni_records:
                try:
                    if ni_rec.lb in (0, 1, 2, 8, 9):
                        mat, grid = self._decode_lb012_matrix(ni_rec)
                        all_energies.update(grid)
                        components.append((mat, grid, None))
                    elif ni_rec.lb in (3, 4):
                        mat, row_grid, col_grid = self._decode_lb34_matrix(ni_rec)
                        all_energies.update(row_grid)
                        all_energies.update(col_grid)
                        components.append((mat, row_grid, col_grid))
                    elif ni_rec.lb == 5:
                        mat, grid = self._decode_lb5_matrix(ni_rec)
                        all_energies.update(grid)
                        components.append((mat, grid, None))
                    elif ni_rec.lb == 6:
                        mat, row_grid, col_grid = self._decode_lb6_matrix(ni_rec)
                        all_energies.update(row_grid)
                        all_energies.update(col_grid)
                        components.append((mat, row_grid, col_grid))
                except ValueError as e:
                    logger.error(f"Error decoding LB={ni_rec.lb} in MT{self.number}→MT{mt1}: {e}")

            if not components:
                continue

            union_grid = sorted(all_energies)
            if len(union_grid) < 2:
                continue
            union_m = len(union_grid) - 1
            total = np.zeros((union_m, union_m))

            for comp_mat, row_grid, col_grid in components:
                # Skip projection if grids already match
                if col_grid is None and row_grid == union_grid:
                    projected = comp_mat
                elif col_grid is not None and row_grid == union_grid and col_grid == union_grid:
                    projected = comp_mat
                else:
                    projected = self._project_matrix_piecewise_constant(
                        comp_mat, row_grid, union_grid, native_col_grid=col_grid
                    )
                total += projected

            # Determine if the combined matrix is relative or absolute
            # LB=0,8,9 are absolute; LB=1,2,3,4,5,6 are relative (fractional)
            lb_set = {r.lb for r in subsection.ni_records}
            is_relative = not lb_set.issubset({0, 8, 9})

            result.add_matrix(
                isotope, self.number, iso_col, mt1, total,
                energy_grid=union_grid, is_relative=is_relative,
            )

        return result

    def __str__(self) -> str:
        """Convert the MF33MT object back to ENDF format string."""
        from ...utils import (
            format_endf_data_line,
            ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        )

        mat = self._mat if self._mat is not None else 0
        mf = 33
        mt = self.number
        lines = []

        def blank_line_number(line: str) -> str:
            return line[:75] + "     "

        def _write_values_block(all_values):
            """Write a list of float values in blocks of 6 per ENDF line."""
            buf = []
            for val in all_values:
                buf.append(val)
                if len(buf) == 6:
                    ln = format_endf_data_line(buf, mat, mf, mt, 0)
                    lines.append(blank_line_number(ln))
                    buf = []
            if buf:
                while len(buf) < 6:
                    buf.append(None)
                ln = format_endf_data_line(
                    buf, mat, mf, mt, 0,
                    formats=[ENDF_FORMAT_FLOAT] * len(buf) + [ENDF_FORMAT_BLANK] * (6 - len(buf))
                )
                lines.append(blank_line_number(ln))

        # HEAD line
        head = format_endf_data_line(
            [self._za, self._awr, 0, self._mtl or 0, 0, self._nl or 0],
            mat, mf, mt, 0,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT]
        )
        lines.append(blank_line_number(head))

        # Lumped components have only HEAD + SEND
        if self.is_lumped_component:
            end_line = format_endf_data_line(
                [0, 0, 0, 0, 0, 0],
                mat, mf, 0, 99999,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(end_line)
            return "\n".join(lines)

        # Subsections
        for subsection in self._subsections:
            # Subsection CONT
            subsec_cont = format_endf_data_line(
                [subsection.xmf1 or 0.0, subsection.xlfs1 or 0.0,
                 subsection.mat1 or 0, subsection.mt1,
                 subsection.nc or 0, subsection.ni or 0],
                mat, mf, mt, 0,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(blank_line_number(subsec_cont))

            # NC-type sub-subsections
            for nc_rec in subsection.nc_records:
                # CONT line
                nc_cont = format_endf_data_line(
                    [0.0, 0.0, 0, nc_rec.lty, 0, 0],
                    mat, mf, mt, 0,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
                )
                lines.append(blank_line_number(nc_cont))

                if nc_rec.lty == 0:
                    # LIST: E1, E2, 0, 0, 2*NCI, NCI / {CI, XMTI}
                    nci = nc_rec.nci or 0
                    list_header = format_endf_data_line(
                        [nc_rec.e1, nc_rec.e2, 0, 0, 2 * nci, nci],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO,
                                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(list_header))
                    # Data: alternating CI, XMTI
                    all_values = []
                    for i in range(nci):
                        all_values.append(nc_rec.ci[i])
                        all_values.append(nc_rec.xmti[i])
                    _write_values_block(all_values)

                elif nc_rec.lty in (1, 2, 3):
                    # LIST: E1, E2, MATS, MTS, 2*NEI+2, NEI / (XMFS, XLFSS), {EI, WEI}
                    nei = nc_rec.nei or 0
                    list_header = format_endf_data_line(
                        [nc_rec.e1, nc_rec.e2, nc_rec.mats or 0, nc_rec.mts or 0,
                         2 * nei + 2, nei],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(list_header))
                    # Data: XMFS, XLFSS, then alternating EI, WEI
                    all_values = [nc_rec.xmfs or 0.0, nc_rec.xlfss or 0.0]
                    for i in range(nei):
                        all_values.append(nc_rec.ei[i])
                        all_values.append(nc_rec.wei[i])
                    _write_values_block(all_values)

            # NI-type sub-subsections
            for ni_rec in subsection.ni_records:
                if ni_rec.lb in (0, 1, 2, 8, 9):
                    # LIST header: 0.0, 0.0, LT, LB, NT, NP
                    rec_header = format_endf_data_line(
                        [0.0, 0.0, ni_rec.lt or 0, ni_rec.lb, ni_rec.nt, ni_rec.np],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(rec_header))
                    # Data: alternating Ek, Fk
                    all_values = []
                    for i in range(len(ni_rec.e_table_k)):
                        all_values.append(ni_rec.e_table_k[i])
                        all_values.append(ni_rec.f_table_k[i])
                    _write_values_block(all_values)

                elif ni_rec.lb in (3, 4):
                    # LIST header: 0.0, 0.0, LT, LB, NT, NP
                    rec_header = format_endf_data_line(
                        [0.0, 0.0, ni_rec.lt, ni_rec.lb, ni_rec.nt, ni_rec.np],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(rec_header))
                    # Data: first table {Ek, Fk}, then second table {El, Fl}
                    all_values = []
                    for i in range(len(ni_rec.e_table_k)):
                        all_values.append(ni_rec.e_table_k[i])
                        all_values.append(ni_rec.f_table_k[i])
                    for i in range(len(ni_rec.e_table_l)):
                        all_values.append(ni_rec.e_table_l[i])
                        all_values.append(ni_rec.f_table_l[i])
                    _write_values_block(all_values)

                elif ni_rec.lb == 5:
                    # LIST header: 0.0, 0.0, LS, LB=5, NT, NE
                    rec_header = format_endf_data_line(
                        [0.0, 0.0, ni_rec.ls, 5, ni_rec.nt, ni_rec.ne],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(rec_header))
                    _write_values_block(ni_rec.energies + ni_rec.matrix)

                elif ni_rec.lb == 6:
                    # LIST header: 0.0, 0.0, 0, LB=6, NT, NER
                    ner = len(ni_rec.row_energies)
                    rec_header = format_endf_data_line(
                        [0.0, 0.0, 0, 6, ni_rec.nt, ner],
                        mat, mf, mt, 0,
                        formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                 ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                                 ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                    )
                    lines.append(blank_line_number(rec_header))
                    _write_values_block(
                        ni_rec.row_energies + ni_rec.col_energies + ni_rec.rect_matrix
                    )

        # SEND marker
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)

        return "\n".join(lines)
