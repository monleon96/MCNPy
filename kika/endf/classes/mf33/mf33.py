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
    def _group_average_xs(
        mf3_section: object,
        energy_grid: List[float],
        n_samples: int = 64,
    ) -> np.ndarray:
        """Compute lethargy-weighted group-averaged cross sections.

        For each energy bin [E_i, E_{i+1}], evaluates:

            σ̄ = ∫ σ(E) / E dE  /  ∫ 1/E dE

        using Gauss-Legendre quadrature in log-E space for accuracy.

        Parameters
        ----------
        mf3_section : MF3MT
            Section with ``get_cross_section(energy)`` method.
        energy_grid : list of float
            Energy bin boundaries (N+1 points → N bins).
        n_samples : int
            Number of quadrature points per bin (default 64).

        Returns
        -------
        np.ndarray of shape (N,)
            Group-averaged cross sections (barns).
        """
        grid = np.asarray(energy_grid, dtype=float)
        n_bins = len(grid) - 1
        result = np.zeros(n_bins)

        # Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n_samples)

        for g in range(n_bins):
            # Log-space quadrature requires both edges > 0; bins touching 0 eV
            # leave result[g] = 0 (the slot was pre-zeroed). This corresponds
            # to "no σ contribution from a degenerate bin" — bin gets dropped
            # downstream wherever a positive σ is needed.
            if grid[g] <= 0 or grid[g + 1] <= 0:
                continue
            ln_lo = np.log(grid[g])
            ln_hi = np.log(grid[g + 1])
            if ln_hi <= ln_lo:
                continue
            half_width = 0.5 * (ln_hi - ln_lo)
            mid = 0.5 * (ln_lo + ln_hi)
            # Map nodes to [ln_lo, ln_hi]
            ln_e = mid + half_width * nodes
            e_pts = np.exp(ln_e)
            xs_pts = np.asarray(
                mf3_section.get_cross_section(e_pts), dtype=float
            )
            # ∫ σ(E) dln(E) / ∫ dln(E) = weighted average over quadrature
            result[g] = np.dot(weights, xs_pts) / np.sum(weights)
        return result

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

    def _process_ni_records_to_matrix(
        self,
        ni_records: List['NISubSubsectionRecord'],
        mt_label: str = '',
        target_grid: Optional[List[float]] = None,
    ) -> Optional[Tuple[np.ndarray, List[float], bool]]:
        """Process a list of NI records into a full covariance matrix.

        All NI components are decoded, projected onto a shared grid, and
        summed. The full matrix (with off-diagonal structure) is returned,
        not just the diagonal.

        Parameters
        ----------
        ni_records : list of NISubSubsectionRecord
            NI-type sub-subsections to sum.
        mt_label : str
            Diagnostic tag used in log messages.
        target_grid : list of float, optional
            Energy grid to project onto. If None, the union of every
            component's grid is used.

        Returns
        -------
        (matrix, energy_grid, is_relative) or None
            Full NxN matrix on the chosen grid, its boundaries, and a flag
            identifying whether the combined matrix is relative (any
            LB in {1,2,3,4,5,6}) or absolute (all LB in {0,8,9}).
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

        output_grid = list(target_grid) if target_grid is not None else sorted(all_energies)
        if len(output_grid) < 2:
            return None
        m = len(output_grid) - 1
        total = np.zeros((m, m))

        for comp_mat, row_grid, col_grid in components:
            if col_grid is None and row_grid == output_grid:
                projected = comp_mat
            elif col_grid is not None and row_grid == output_grid and col_grid == output_grid:
                projected = comp_mat
            else:
                projected = self._project_matrix_piecewise_constant(
                    comp_mat, row_grid, output_grid, native_col_grid=col_grid
                )
            total += projected

        lb_set = {r.lb for r in ni_records}
        is_relative = not lb_set.issubset({0, 8, 9})
        return total, output_grid, is_relative

    def _get_cross_mt_matrix(
        self,
        mt_i: int,
        mt_j: int,
        sibling_sections: Dict[int, 'MF33MT'],
    ) -> Optional[Tuple[np.ndarray, List[float], bool]]:
        """Return the full cross-MT covariance Cov(MT_i, MT_j) if stored.

        Searches subsections of MT_i for mt1=MT_j, then MT_j for mt1=MT_i.
        Returns (matrix, energy_grid, is_relative) or None if not found.
        """
        for src_mt, tgt_mt in [(mt_i, mt_j), (mt_j, mt_i)]:
            section = sibling_sections.get(src_mt)
            if section is None:
                continue
            subsection = section.get_subsection(tgt_mt)
            if subsection is None or not subsection.ni_records:
                continue
            result = section._process_ni_records_to_matrix(
                subsection.ni_records,
                mt_label=f"MT{src_mt}→MT{tgt_mt}",
            )
            if result is not None:
                return result
        return None

    def _bin_average_xs(
        self,
        xs_source: object,
        energy_grid: List[float],
        n_samples: int = 64,
    ) -> np.ndarray:
        """1/E-weighted bin average of σ(E) over each bin of ``energy_grid``.

        Accepts either an ENDF ``MF3MT`` (has ``get_cross_section``) or a
        canonical ``CrossSection`` (has ``energies``/``values`` arrays).
        Delegates to :meth:`_group_average_xs`, wrapping CrossSection-like
        inputs in a thin shim that interpolates linearly between tabulated
        points.
        """
        if hasattr(xs_source, 'get_cross_section'):
            return self._group_average_xs(xs_source, energy_grid, n_samples)
        energies = np.asarray(getattr(xs_source, 'energies'), dtype=float)
        values = np.asarray(getattr(xs_source, 'values'), dtype=float)

        class _InterpShim:
            def get_cross_section(self_, e):
                return np.interp(e, energies, values, left=0.0, right=0.0)

        return self._group_average_xs(_InterpShim(), energy_grid, n_samples)

    def _self_covariance_matrix(
        self,
        sibling_sections: Optional[Dict[int, 'MF33MT']] = None,
        _resolving: Optional[Set[int]] = None,
        mf3_sections: Optional[Dict[int, object]] = None,
    ) -> Optional[Tuple[np.ndarray, List[float], bool]]:
        """Return only the self-self covariance bundle of this MF33MT.

        Equivalent to calling :meth:`to_xs_covmat` and extracting the
        ``(reaction_rows[i] == self.number, reaction_cols[i] == self.number)``
        block, but without materializing every other subsection (cross-MT
        entries, MAT1≠0, etc.). The dominant cost in
        :meth:`resolve_nc_lty0` step 1 is doing this 54x for Fe-56 MT2;
        this helper avoids the per-contributor cross-MT decode work.

        Mirrors the per-subsection NC+NI logic in :meth:`to_xs_covmat`
        (lines processing one subsection: NC LTY=0 resolution, NI
        bundling, NC+NI combination on a union grid) restricted to the
        single subsection where ``mt1 == self.number``.

        Returns ``(matrix, energy_grid, is_relative)`` or ``None`` if no
        self-self subsection exists or all components fail to decode.
        """
        self_sub: Optional[Subsection] = None
        for sub in self._subsections:
            if int(sub.mt1) == self.number:
                self_sub = sub
                break
        if self_sub is None:
            return None

        nc_matrix: Optional[np.ndarray] = None
        nc_grid: Optional[List[float]] = None
        nc_is_rel: Optional[bool] = None
        if self_sub.nc_records and sibling_sections is not None:
            for nc in self_sub.nc_records:
                if nc.lty != 0:
                    continue
                resolving = (_resolving or set()) | {self.number}
                nc_result = self.resolve_nc_lty0(
                    sibling_sections, 'eV', resolving,
                    mf3_sections=mf3_sections,
                )
                if nc_result is not None and nc_result.matrices:
                    pick = 0
                    for i in range(len(nc_result.matrices)):
                        if (nc_result.reaction_rows[i] == self.number
                                and nc_result.reaction_cols[i] == self.number):
                            pick = i
                            break
                    nc_matrix = np.asarray(
                        nc_result.matrices[pick], dtype=float
                    )
                    nc_grid = (list(nc_result.energy_grids[pick])
                               if nc_result.energy_grids else None)
                    nc_is_rel = (nc_result.is_relative[pick]
                                 if (nc_result.is_relative
                                     and pick < len(nc_result.is_relative))
                                 else True)
                break  # one LTY=0 per subsection

        ni_bundle: Optional[Tuple[np.ndarray, List[float], bool]] = None
        if self_sub.ni_records:
            ni_bundle = self._process_ni_records_to_matrix(
                self_sub.ni_records,
                mt_label=f"MT{self.number}→MT{self.number}",
            )

        if nc_matrix is not None and ni_bundle is not None:
            ni_mat, ni_grid, ni_is_rel = ni_bundle
            ni_is_zero = not np.any(np.abs(ni_mat) > 0)
            if nc_is_rel != ni_is_rel and not ni_is_zero:
                logger.warning(
                    f"MT{self.number}→MT{self.number}: NC result is "
                    f"{'relative' if nc_is_rel else 'absolute'} but NI "
                    f"term is {'relative' if ni_is_rel else 'absolute'}; "
                    f"adding in NC's space — NI noise term may be mis-scaled."
                )
            combined_grid = sorted(set(nc_grid or []) | set(ni_grid))
            if len(combined_grid) < 2:
                return None
            nc_proj = (nc_matrix if list(nc_grid) == combined_grid
                       else self._project_matrix_piecewise_constant(
                           nc_matrix, nc_grid, combined_grid))
            ni_proj = (ni_mat if list(ni_grid) == combined_grid
                       else self._project_matrix_piecewise_constant(
                           ni_mat, ni_grid, combined_grid))
            return nc_proj + ni_proj, combined_grid, bool(nc_is_rel)
        if nc_matrix is not None:
            return (
                nc_matrix,
                list(nc_grid) if nc_grid else [],
                bool(nc_is_rel),
            )
        if ni_bundle is not None:
            return ni_bundle
        return None

    def resolve_nc_lty0(
        self,
        sibling_sections: Dict[int, 'MF33MT'],
        energy_unit: str = 'eV',
        _resolving: Optional[Set[int]] = None,
        mf3_sections: Optional[Dict[int, object]] = None,
    ) -> Optional['CrossSectionCovariance']:
        """Resolve NC LTY=0 (derived redundant) covariance via the bilinear sum rule.

        If the derived reaction is defined as

            σ_d(E) = Σ_i c_i · σ_i(E)    (ENDF MF33.3 item 3 sum rule)

        its full covariance matrix propagates via the bilinear contraction

            Cov_abs(σ_d, σ_d)[g, g'] = Σ_i Σ_j c_i c_j Cov_abs(σ_i, σ_j)[g, g']

        This implementation returns the full NxN covariance matrix (including
        off-diagonal terms) on the union of the contributing MTs' coarse MF33
        grids, matching the algebra NJOY ERRORR uses in its ``akxy``
        contraction — but on the data's native grid, not a multigroup one.

        Cross-MT covariance matrices (when stored in MF33 as subsections with
        MT1 ≠ MT) are picked up by :meth:`_get_cross_mt_matrix` and applied to
        the i ≠ j terms. Missing cross-MT entries are treated as zero.

        For relative→absolute conversion the bin-averaged cross section over
        each coarse bin is computed with 1/E weighting (see
        :meth:`_bin_average_xs`). In the resolved-resonance region, callers
        should pass resonance-reconstructed cross sections (from
        ``kika.processing.njoy_reconstruct``) rather than the raw MF3
        background, otherwise the rel↔abs conversion is dominated by the
        smooth component and underestimates variance.

        Parameters
        ----------
        sibling_sections : dict
            Map of MT → MF33MT for every MT in this MF33 file.
        energy_unit : str
            Energy unit of the output grid ('eV' or 'MeV').
        _resolving : set of int, optional
            MT numbers currently being resolved (recursion guard, internal).
        mf3_sections : dict, optional
            Map of MT → MF3MT or CrossSection (pointwise σ). Required for a
            correct rel↔abs conversion; if omitted, the method falls back to
            direct summation of relative covariances with a logged warning.

        Returns
        -------
        CrossSectionCovariance or None
            Self-covariance of the derived MT on the coarse union grid, or
            ``None`` if resolution fails.
        """
        from ....cov.cross_section_covariance import CrossSectionCovariance

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

        mti_list = [int(round(x)) for x in nc_rec.xmti]
        ci_list = list(nc_rec.ci)
        nci = len(mti_list)
        if nci == 0 or len(ci_list) < nci:
            logger.warning(f"MT{self.number}: NC LTY=0 has no contributing reactions")
            return None
        ci_list = ci_list[:nci]

        logger.info(
            f"MT{self.number}: resolving NC LTY=0 from {nci} reactions: "
            f"{list(zip(ci_list, mti_list))}"
        )

        resolving = (_resolving or set()) | {self.number}

        # Step 1: fetch each contributing MT's self-covariance as a full
        # matrix.  Uses the lite ``_self_covariance_matrix`` instead of
        # the full ``to_xs_covmat`` because the latter materializes every
        # subsection (cross-MT entries, MAT1≠0, ...) per contributor and
        # we discard everything but the self-self block — for Fe-56 MT2
        # with 54 contributors, that wasted work was the dominant cost.
        cov_by_mt: Dict[int, Tuple[np.ndarray, List[float], bool]] = {}
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
                bundle = section_i._self_covariance_matrix(
                    sibling_sections=sibling_sections,
                    _resolving=resolving,
                    mf3_sections=mf3_sections,
                )
            except Exception as e:
                logger.warning(f"MT{self.number}: failed to resolve MT{mt_i}: {e}")
                continue

            if bundle is None:
                continue
            mat, grid, is_rel = bundle
            mat = np.asarray(mat, dtype=float)
            if (grid is not None and mat.ndim == 2
                    and mat.shape[0] == mat.shape[1]
                    and mat.shape[0] == len(grid) - 1):
                cov_by_mt[mt_i] = (mat, list(grid), bool(is_rel))

        if not cov_by_mt:
            logger.warning(f"MT{self.number}: no contributing MT covariances found")
            return None

        # Step 2: union energy grid from every contributing MT's own grid
        all_energies: Set[float] = set()
        for _, grid, _ in cov_by_mt.values():
            all_energies.update(grid)
        union_grid = sorted(all_energies)
        if len(union_grid) < 2:
            return None
        m = len(union_grid) - 1

        # Step 3: project each contributing matrix onto the union grid
        proj_cov: Dict[int, np.ndarray] = {}
        is_rel_by_mt: Dict[int, bool] = {}
        for mt_i, (mat, grid, is_rel) in cov_by_mt.items():
            is_rel_by_mt[mt_i] = is_rel
            if grid == union_grid:
                proj_cov[mt_i] = mat
            else:
                proj_cov[mt_i] = self._project_matrix_piecewise_constant(
                    mat, grid, union_grid
                )

        have_xs = mf3_sections is not None
        any_relative = any(is_rel_by_mt.get(mt, True) for mt in mti_list
                           if mt in proj_cov)
        if any_relative and not have_xs:
            logger.warning(
                f"MT{self.number}: NC LTY=0 sum rule requires cross-section data "
                f"(mf3_sections) to convert relative covariances to absolute. "
                f"Falling back to direct summation of relative covariances — "
                f"result will be approximate."
            )

        # Step 4: bin-averaged σ per contributing MT (for rel↔abs conversion)
        xs_bin: Dict[int, np.ndarray] = {}
        xs_d: Optional[np.ndarray] = None
        if have_xs:
            for mt_i in mti_list:
                if mt_i in mf3_sections and mt_i in proj_cov:
                    try:
                        xs_bin[mt_i] = self._bin_average_xs(
                            mf3_sections[mt_i], union_grid
                        )
                    except Exception as e:
                        logger.warning(
                            f"MT{self.number}: bin-average σ failed for MT{mt_i}: {e}"
                        )
            # Derived σ: from MF3 if available, else from the sum rule
            if self.number in mf3_sections:
                try:
                    xs_d = self._bin_average_xs(
                        mf3_sections[self.number], union_grid
                    )
                except Exception as e:
                    logger.warning(
                        f"MT{self.number}: bin-average σ_derived failed: {e}"
                    )
                    xs_d = None
            if xs_d is None:
                xs_d = np.zeros(m)
                for ii, mt_i in enumerate(mti_list):
                    if mt_i in xs_bin:
                        xs_d = xs_d + ci_list[ii] * xs_bin[mt_i]

        # Step 5: cross-MT covariance matrices (projected to union grid).
        # Pre-index every cross-MT NI bundle stored under any contributing
        # MT's section so the (i,j) double loop becomes a dict lookup
        # instead of a per-pair linear scan + NI decode through
        # ``_get_cross_mt_matrix``.
        mti_set = set(mti_list)
        cross_index: Dict[
            Tuple[int, int], Tuple[np.ndarray, List[float], bool]
        ] = {}
        for src_mt in mti_set:
            sec = sibling_sections.get(src_mt)
            if sec is None:
                continue
            for sub in sec._subsections:
                tgt_mt = int(sub.mt1)
                if tgt_mt == src_mt or tgt_mt not in mti_set:
                    continue
                if not sub.ni_records:
                    continue
                bundle = sec._process_ni_records_to_matrix(
                    sub.ni_records,
                    mt_label=f"MT{src_mt}→MT{tgt_mt}",
                )
                if bundle is not None:
                    cross_index[(src_mt, tgt_mt)] = bundle

        cross_cov: Dict[Tuple[int, int], Tuple[np.ndarray, bool]] = {}
        for ii in range(nci):
            for jj in range(ii + 1, nci):
                mt_i = mti_list[ii]
                mt_j = mti_list[jj]
                r = cross_index.get((mt_i, mt_j))
                if r is None:
                    r = cross_index.get((mt_j, mt_i))
                if r is None:
                    continue
                cmat, cgrid, cis_rel = r
                if cgrid == union_grid:
                    proj_cross = cmat
                else:
                    proj_cross = self._project_matrix_piecewise_constant(
                        cmat, cgrid, union_grid
                    )
                cross_cov[(mt_i, mt_j)] = (proj_cross, bool(cis_rel))

        # Step 6: bilinear accumulation in ABSOLUTE covariance space
        #   Cov(d,d)[g,g'] = Σ_i Σ_j c_i c_j Cov(i,j)[g,g']
        cov_abs = np.zeros((m, m))

        def _scale_rel_to_abs(mat_rel: np.ndarray, s_row: np.ndarray,
                              s_col: np.ndarray) -> np.ndarray:
            return mat_rel * np.outer(s_row, s_col)

        for ii, mt_i in enumerate(mti_list):
            c_i = ci_list[ii]
            for jj, mt_j in enumerate(mti_list):
                c_j = ci_list[jj]

                if ii == jj:
                    pair = proj_cov.get(mt_i)
                    if pair is None:
                        continue
                    pair_is_rel = is_rel_by_mt.get(mt_i, True)
                    if pair_is_rel and have_xs and mt_i in xs_bin:
                        pair_abs = _scale_rel_to_abs(pair, xs_bin[mt_i], xs_bin[mt_i])
                    else:
                        pair_abs = pair  # already absolute or no σ available
                else:
                    entry = cross_cov.get((mt_i, mt_j))
                    pair_transposed = False
                    if entry is None:
                        entry = cross_cov.get((mt_j, mt_i))
                        pair_transposed = entry is not None
                    if entry is None:
                        continue  # unknown cross-MT cov → treat as zero
                    pair, pair_is_rel = entry
                    if pair_transposed:
                        pair = pair.T
                    if pair_is_rel and have_xs and mt_i in xs_bin and mt_j in xs_bin:
                        pair_abs = _scale_rel_to_abs(pair, xs_bin[mt_i], xs_bin[mt_j])
                    else:
                        pair_abs = pair

                cov_abs = cov_abs + c_i * c_j * pair_abs

        # Step 7: convert back to relative covariance when σ is available.
        # Apply an inert-bin floor on σ̄ to suppress 1/σ̄ blow-ups in
        # threshold-spanning bins and lumped-MT filler bins (the NC LTY=0
        # union grid frequently inherits foreign threshold edges from
        # contributing MTs, so xs_d can collapse to ~σ̄_max·1e-7 in a few
        # bins while the absolute variance there is just placeholder).
        if have_xs and xs_d is not None:
            xs_max = float(np.max(xs_d)) if xs_d.size else 0.0
            sigma_floor = max(1.0e-3 * xs_max, 1.0e-9)
            xs_safe = np.where(xs_d > sigma_floor, xs_d, 0.0)
            denom = np.outer(xs_safe, xs_safe)
            with np.errstate(divide='ignore', invalid='ignore'):
                cov_rel = np.where(denom > 0, cov_abs / np.where(denom > 0, denom, 1.0), 0.0)
            # Clip negative DIAGONAL only — variance must be ≥ 0 but
            # off-diagonal correlation sign is physically meaningful.
            diag = np.diag(cov_rel).copy()
            np.fill_diagonal(cov_rel, np.maximum(diag, 0.0))
            final = cov_rel
            is_relative_out = True
        else:
            # Fallback: no σ provided. cov_abs actually holds a direct sum
            # of the relative matrices (no scaling), so flag as relative.
            final = cov_abs
            is_relative_out = True

        # Symmetrize to kill any round-off asymmetry from the outer products
        final = 0.5 * (final + final.T)

        isotope = int(round(self._za))
        result = CrossSectionCovariance(energy_unit=energy_unit)
        result.add_matrix(
            isotope, self.number, isotope, self.number,
            final,
            energy_grid=list(union_grid),
            is_relative=is_relative_out,
        )

        diag = np.diag(final)
        diag_min = float(max(diag.min(), 0.0))
        diag_max = float(diag.max())
        logger.info(
            f"MT{self.number}: NC LTY=0 resolved — {m} bins, "
            f"sqrt(diag) range [{np.sqrt(diag_min):.4g}, {np.sqrt(diag_max):.4g}]"
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
        mf3_sections: Optional[Dict[int, object]] = None,
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
        mf3_sections : dict, optional
            Map of MT number → MF3MT (pointwise cross sections from File 3).
            Required for correct NC LTY=0 resolution when contributing MTs
            store relative covariances.

        Returns
        -------
        CrossSectionCovariance
        """
        from ....cov.cross_section_covariance import CrossSectionCovariance

        isotope = int(round(self._za))
        result = CrossSectionCovariance(energy_unit=energy_unit)

        own_mat = int(self._mat or 0)
        for subsection in self._subsections:
            mt1 = int(subsection.mt1)
            mat1 = int(subsection.mat1 or 0)
            # ENDF-6 §33.3.1 convention is MAT1=0 for intra-material
            # cross-MT blocks, but some evaluations (e.g. JEFF-4.0 H-1)
            # write the file's own MAT number instead. Treat that case as
            # same-isotope to avoid manufacturing a phantom isotope band
            # whose self-self diagonal is never populated (→ NaN bins).
            iso_col = isotope if (mat1 == 0 or mat1 == own_mat) else mat1

            # Step A: resolve NC-type sub-subsections (LTY=0 sum rule)
            nc_matrix: Optional[np.ndarray] = None
            nc_grid: Optional[List[float]] = None
            nc_is_rel: Optional[bool] = None
            if subsection.nc_records:
                for nc in subsection.nc_records:
                    if nc.lty == 0 and sibling_sections is not None:
                        resolving = (_resolving or set()) | {self.number}
                        nc_result = self.resolve_nc_lty0(
                            sibling_sections, energy_unit, resolving,
                            mf3_sections=mf3_sections,
                        )
                        if nc_result is not None and nc_result.matrices:
                            pick = 0
                            for i in range(len(nc_result.matrices)):
                                if (nc_result.reaction_rows[i] == self.number
                                        and nc_result.reaction_cols[i] == mt1):
                                    pick = i
                                    break
                            nc_matrix = np.asarray(
                                nc_result.matrices[pick], dtype=float
                            )
                            nc_grid = (list(nc_result.energy_grids[pick])
                                       if nc_result.energy_grids else None)
                            nc_is_rel = (nc_result.is_relative[pick]
                                         if (nc_result.is_relative
                                             and pick < len(nc_result.is_relative))
                                         else True)
                        break  # one LTY=0 per subsection
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

            # Step B: process NI-type sub-subsections (always — NI records
            # alongside an NC record are additive per ENDF 33.3.3 item 3)
            ni_bundle: Optional[Tuple[np.ndarray, List[float], bool]] = None
            if subsection.ni_records:
                ni_bundle = self._process_ni_records_to_matrix(
                    subsection.ni_records,
                    mt_label=f"MT{self.number}→MT{mt1}",
                )

            # Step C: combine NC and NI (on a shared union grid) and emit
            if nc_matrix is not None and ni_bundle is not None:
                ni_mat, ni_grid, ni_is_rel = ni_bundle
                # Many ENDF files carry an all-zero NI record alongside NC
                # purely to extend the energy range; skip the warning in
                # that case since the contribution is exactly zero.
                ni_is_zero = not np.any(np.abs(ni_mat) > 0)
                if nc_is_rel != ni_is_rel and not ni_is_zero:
                    logger.warning(
                        f"MT{self.number}→MT{mt1}: NC result is "
                        f"{'relative' if nc_is_rel else 'absolute'} but NI "
                        f"term is {'relative' if ni_is_rel else 'absolute'}; "
                        f"adding in NC's space — NI noise term may be "
                        f"mis-scaled."
                    )
                combined_grid = sorted(set(nc_grid or []) | set(ni_grid))
                if len(combined_grid) < 2:
                    continue
                nc_proj = (nc_matrix if list(nc_grid) == combined_grid
                           else self._project_matrix_piecewise_constant(
                               nc_matrix, nc_grid, combined_grid))
                ni_proj = (ni_mat if list(ni_grid) == combined_grid
                           else self._project_matrix_piecewise_constant(
                               ni_mat, ni_grid, combined_grid))
                result.add_matrix(
                    isotope, self.number, iso_col, mt1,
                    nc_proj + ni_proj,
                    energy_grid=combined_grid,
                    is_relative=bool(nc_is_rel),
                )
            elif nc_matrix is not None:
                result.add_matrix(
                    isotope, self.number, iso_col, mt1,
                    nc_matrix,
                    energy_grid=nc_grid,
                    is_relative=bool(nc_is_rel),
                )
            elif ni_bundle is not None:
                ni_mat, ni_grid, ni_is_rel = ni_bundle
                result.add_matrix(
                    isotope, self.number, iso_col, mt1,
                    ni_mat,
                    energy_grid=ni_grid,
                    is_relative=bool(ni_is_rel),
                )

        return result

    def __str__(self) -> str:
        """Convert the MF33MT object back to ENDF format string."""
        from ...utils import (
            format_endf_data_line,
            ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        )

        mat = self._mat if self._mat is not None else 0
        # **Not the literal 33.** MF31 reuses these records verbatim (§31.1:
        # the MT452/455/456 formats "are directly analogous to those of File
        # 33"), which is why `parse_mf31` builds `MF33MT` objects and tags them
        # `_mf = 31`. Writing the literal 33 here stamped every line of an MF31
        # section with MF33 in columns 71-72, while `_section_writer` placed it
        # correctly in the MF31 block -- a file that is wrong only in the
        # identifier columns, which nothing but a parser would notice.
        mf = int(self._mf) if self._mf is not None else 33
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
