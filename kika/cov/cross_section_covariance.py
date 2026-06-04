from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kika._constants import MT_TO_REACTION
from kika._utils import create_repr_section


def _safe_min_eigvalsh(M: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix, robust to LAPACK failures.

    Fallback cascade:
      1. numpy's eigvalsh (LAPACK dsyevd, divide-and-conquer; default, fast)
      2. scipy.linalg.eigh with driver='evr' (LAPACK dsyevr, RRR)
      3. scipy.linalg.eigh with driver='ev'  (LAPACK dsyev, QR iteration)
      4. ARPACK (scipy.sparse.linalg.eigsh) for smallest algebraic — totally
         different algorithm family, survives buggy LAPACK builds
      5. Drop all-zero rows/cols and retry numpy on the reduced matrix
      6. Last resort: warn and return -inf so callers proceed without
         claiming the matrix is PSD. The pipeline's downstream PSD repair
         (variance cap, threshold-bin rescale, Higham, eigen-clip) handles
         the rest; an eigenvalue check that we couldn't run shouldn't kill
         the whole job.
    """
    last_err: Exception = RuntimeError("no solver attempted")

    def _ok(v: float) -> bool:
        return np.isfinite(v)

    try:
        v = float(np.linalg.eigvalsh(M).min())
        if _ok(v):
            return v
        last_err = RuntimeError(f"numpy eigvalsh returned non-finite ({v})")
    except Exception as e:
        last_err = e

    try:
        from scipy.linalg import eigh as _sp_eigh
        for driver in ("evr", "ev"):
            try:
                v = float(_sp_eigh(M, eigvals_only=True, driver=driver).min())
                if _ok(v):
                    return v
                last_err = RuntimeError(f"scipy eigh ({driver}) returned non-finite ({v})")
            except Exception as e:
                last_err = e
    except ImportError as e:
        last_err = e

    try:
        from scipy.sparse.linalg import eigsh as _sp_eigsh
        v = float(_sp_eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
        if _ok(v):
            return v
        last_err = RuntimeError(f"ARPACK eigsh returned non-finite ({v})")
    except Exception as e:
        last_err = e

    try:
        nonzero = ~(np.all(M == 0, axis=0) & np.all(M == 0, axis=1))
        if 0 < int(nonzero.sum()) < M.shape[0]:
            M_red = M[np.ix_(nonzero, nonzero)]
            v_red = float(np.linalg.eigvalsh(M_red).min())
            if _ok(v_red):
                return min(v_red, 0.0)  # zero rows contribute zero eigenvalues
    except Exception as e:
        last_err = e

    # All solvers failed. Returning -inf would force downstream removal
    # escalation, but the removal loop also needs eigenvalues — so it falls
    # back to a diagonal-magnitude heuristic that hatchets through every MT.
    # Instead return 0.0 (== "acceptable at the boundary"): the caller skips
    # removal, and the pipeline's downstream PSD repair (variance cap,
    # threshold-bin rescale, inert-bin mask, Higham, eigen-clip) handles
    # actual indefiniteness on the masked submatrix where LAPACK usually
    # works fine. Worst case it doesn't fully repair — but that's strictly
    # better than deleting most of the perturbation parameters.
    import warnings
    warnings.warn(
        f"_safe_min_eigvalsh: all eigenvalue solvers failed ({type(last_err).__name__}: {last_err}); "
        "returning 0.0 (treat as boundary-PSD) so the caller skips removal escalation. "
        "Downstream Higham / eigen-clip will still repair indefiniteness on the masked submatrix.",
        RuntimeWarning,
        stacklevel=2,
    )
    return 0.0


if TYPE_CHECKING:
    from pathlib import Path
    from kika.plotting.plot_data import CovarianceHeatmapData, MultigroupCrossSectionPlotData, MultigroupUncertaintyPlotData


def _rescale_energy_grid(
    grid: Optional[Sequence[float]], from_unit: Optional[str], to_unit: Optional[str],
) -> Optional[List[float]]:
    """Convert an energy grid between 'eV' and 'MeV'. Returns a new list (or None).

    Covariance matrix *values* are unaffected by the energy-axis unit (relative
    covariances are dimensionless; absolute ones are in cross-section units), so
    only the grid boundaries are rescaled.
    """
    if grid is None:
        return None
    fu = (from_unit or 'eV').lower()
    tu = (to_unit or 'eV').lower()
    if fu == tu:
        return [float(x) for x in grid]
    if fu == 'ev' and tu == 'mev':
        factor = 1e-6
    elif fu == 'mev' and tu == 'ev':
        factor = 1e6
    else:
        return [float(x) for x in grid]  # unknown unit pair: leave magnitudes as-is
    return [float(x) * factor for x in grid]


@dataclass
class TransferResult:
    """Outcome of :meth:`CrossSectionCovariance.transfer_reactions`.

    Attributes
    ----------
    covariance : CrossSectionCovariance
        The merged covariance (a new object; inputs are not mutated).
    diagonal_transferred : list of (isotope, mt)
        Self-blocks copied from the source.
    cross_transferred : list of (iso_row, mt_row, iso_col, mt_col)
        Off-diagonal cross-correlation blocks copied from the source.
    cross_dropped : list of ((iso_row, mt_row, iso_col, mt_col), reason)
        Cross blocks that were *not* copied, with the reason.
    reactions_replaced : list of (isotope, mt)
        Destination reactions whose existing blocks were removed before re-adding.
    cross_sections_transferred : list of (isotope, mt)
        Reactions whose associated cross-section vector was copied.
    missing_in_source : list of (isotope, mt)
        Requested reactions not found in the source.
    """
    covariance: "CrossSectionCovariance"
    diagonal_transferred: List[Tuple[int, int]] = field(default_factory=list)
    cross_transferred: List[Tuple[int, int, int, int]] = field(default_factory=list)
    cross_dropped: List[Tuple[Tuple[int, int, int, int], str]] = field(default_factory=list)
    reactions_replaced: List[Tuple[int, int]] = field(default_factory=list)
    cross_sections_transferred: List[Tuple[int, int]] = field(default_factory=list)
    missing_in_source: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class CrossSectionCovariance:
    """
    Format-agnostic multigroup cross-section covariance.
    
    Attributes
    ----------
    num_groups : int
        Number of energy groups in the covariance matrices
    energy_grid : Optional[List[float]]
        List of energy grid boundaries (if available)
    isotope_rows : List[int]
        List of row isotope IDs
    reaction_rows : List[int]
        List of row reaction MT numbers
    isotope_cols : List[int]
        List of column isotope IDs
    reaction_cols : List[int]
        List of column reaction MT numbers
    matrices : List[np.ndarray]
        List of covariance matrices (one for each row-column combination)
    energy_unit : str
        Energy unit for energy_grid: 'eV' (default) or 'MeV'
    """
    num_groups: int = 0
    energy_grid: Optional[List[float]] = None
    energy_grids: List[List[float]] = field(default_factory=list)  # Per-matrix energy grids (pointwise ENDF)
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)
    is_relative: List[bool] = field(default_factory=list)  # Per-matrix: True=relative, False=absolute
    cross_sections: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    energy_unit: str = 'eV'  # Energy unit: 'eV' or 'MeV'
    # File-level scalar metadata propagated by I/O routines (COVFIL, COVERX, …).
    # Well-known keys: 'awr' (atomic weight ratio), 'temperature' (K).
    # New fields can be added without changing the class.
    metadata: Dict[str, Any] = field(default_factory=dict)


    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

    def copy(self) -> "CrossSectionCovariance":
        """
        Return a deep copy of this CrossSectionCovariance instance.
        
        Returns
        -------
        CrossSectionCovariance
            Deep copy of the current covariance matrix object
        """
        return copy.deepcopy(self)

    def add_matrix(
        self,
        isotope_row: int,
        reaction_row: int,
        isotope_col: int,
        reaction_col: int,
        matrix: np.ndarray,
        energy_grid: Optional[List[float]] = None,
        is_relative: Optional[bool] = None,
    ) -> None:
        """
        Add a covariance matrix to the collection.

        Parameters
        ----------
        isotope_row : int
            Row isotope ID
        reaction_row : int
            Row reaction MT number
        isotope_col : int
            Column isotope ID
        reaction_col : int
            Column reaction MT number
        matrix : np.ndarray
            Covariance matrix
        energy_grid : list of float, optional
            Energy grid for this matrix. If not provided, validates against
            ``self.num_groups`` (multigroup mode).
        is_relative : bool, optional
            Whether the matrix is relative (True) or absolute (False).
        """
        if energy_grid is not None:
            # Per-matrix grid mode (ENDF MF33 or explicit grid)
            expected_size = len(energy_grid) - 1
            if matrix.shape != (expected_size, expected_size) and matrix.shape[0] != expected_size:
                raise ValueError(
                    f"Matrix shape {matrix.shape} incompatible with energy_grid "
                    f"of {len(energy_grid)} points ({expected_size} intervals)"
                )
            self.energy_grids.append(list(energy_grid))
        else:
            # Multigroup mode: validate against num_groups
            if self.num_groups > 0 and matrix.shape != (self.num_groups, self.num_groups):
                raise ValueError(f"Matrix shape {matrix.shape} does not match expected shape ({self.num_groups}, {self.num_groups})")
            # Populate energy_grids from the shared energy_grid for consistency
            if self.energy_grid is not None:
                self.energy_grids.append(list(self.energy_grid))

        self.isotope_rows.append(isotope_row)
        self.reaction_rows.append(reaction_row)
        self.isotope_cols.append(isotope_col)
        self.reaction_cols.append(reaction_col)
        self.matrices.append(matrix)
        if is_relative is not None:
            self.is_relative.append(is_relative)

        # Invalidate cached adjacency graph (if any)
        self._invalidate_cov_graph_cache()
    
    @classmethod
    def from_gendf(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "CrossSectionCovariance":
        """
        Create a CrossSectionCovariance instance from an NJOY-generated GENDF covariance file.
        
        This is a convenience class method that wraps the `read_njoy_covmat` function
        to provide a more object-oriented API consistent with other data classes.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the GENDF covariance file
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'
            
        Returns
        -------
        CrossSectionCovariance
            CrossSectionCovariance instance loaded from the file

        Examples
        --------
        >>> covmat = CrossSectionCovariance.from_gendf('path/to/file.gendf')
        >>> print(f"Loaded {covmat.num_matrices} matrices")
        >>> # Or specify MeV if the file uses MeV
        >>> covmat_mev = CrossSectionCovariance.from_gendf('path/to/file.gendf', energy_unit='MeV')
        
        See Also
        --------
        read_covfil : Underlying function that performs the parsing
        """
        from pathlib import Path
        from kika.cov.parse_covmat import read_covfil

        file_path = Path(file_path)
        result = read_covfil(str(file_path), energy_unit=energy_unit)
        if not isinstance(result, cls):
            raise TypeError(
                "File contains MF34 data. Use LegendreCovariance.from_covfil() instead."
            )
        return result

    @classmethod
    def from_covfil(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "CrossSectionCovariance":
        """
        Create a CrossSectionCovariance instance from an NJOY-generated COVFIL/GENDF file.

        Parameters
        ----------
        file_path : str or Path
            Path to the COVFIL/GENDF covariance file
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        CrossSectionCovariance
            CrossSectionCovariance instance loaded from the file

        Raises
        ------
        TypeError
            If the file contains MF34 data instead of MF33

        See Also
        --------
        read_covfil : Underlying function that performs the parsing
        """
        from pathlib import Path
        from kika.cov.parse_covmat import read_covfil

        file_path = Path(file_path)
        result = read_covfil(str(file_path), energy_unit=energy_unit)
        if not isinstance(result, cls):
            raise TypeError(
                "File contains MF34 data. Use LegendreCovariance.from_covfil() instead."
            )
        return result

    def to_covfil(self, file_path: Union[str, 'Path'], tape_label: str = '', temperature: float = 0.0) -> None:
        """
        Write this CrossSectionCovariance to an NJOY COVFIL/GENDF text file.

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

    @classmethod
    def from_boxer(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "CrossSectionCovariance":
        """
        Create a CrossSectionCovariance instance from a BOXER card-image covariance file.

        Parameters
        ----------
        file_path : str or Path
            Path to the BOXER file
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        CrossSectionCovariance
            CrossSectionCovariance instance loaded from the file

        See Also
        --------
        read_boxer : Underlying function that performs the parsing
        """
        from pathlib import Path
        from kika.cov.parse_covmat import read_boxer

        return read_boxer(str(Path(file_path)), energy_unit=energy_unit)

    def to_boxer(
        self, file_path: Union[str, 'Path'],
        hlibid: str = '', hdescr: str = '', nvf: int = 10,
    ) -> None:
        """
        Write this CrossSectionCovariance to a BOXER card-image (ASCII) file.

        Parameters
        ----------
        file_path : str or Path
            Output file path
        hlibid : str, optional
            Library identifier (3 chars max)
        hdescr : str, optional
            Description (32 chars max)
        nvf : int, optional
            Value format code (7-14). Default 10 (1P8E10.3).
        """
        from kika.cov.parse_covmat import write_boxer
        write_boxer(self, str(file_path), hlibid=hlibid, hdescr=hdescr, nvf=nvf)

    def to_coverx(
        self, file_path: Union[str, 'Path'], fmt: str = 'binary', title: str = '',
    ) -> None:
        """
        Write this CrossSectionCovariance to a COVERX covariance file (text or binary).

        Parameters
        ----------
        file_path : str or Path
            Output file path
        fmt : str, optional
            ``'binary'`` (default) or ``'text'``
        title : str, optional
            File title / description
        """
        from kika.cov.parse_covmat import write_coverx
        write_coverx(self, str(file_path), fmt=fmt, title=title)

    @classmethod
    def from_coverx(cls, file_path: Union[str, 'Path'], ascending: bool = True, energy_unit: str = 'eV') -> "CrossSectionCovariance":
        """
        Create a CrossSectionCovariance instance from a COVERX covariance file (text or binary).

        Parameters
        ----------
        file_path : str or Path
            Path to the COVERX covariance file
        ascending : bool, optional
            If True, energies ordered ascending (default True)
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        CrossSectionCovariance
            CrossSectionCovariance instance loaded from the file

        See Also
        --------
        read_coverx : Underlying function that performs the parsing
        """
        from pathlib import Path
        from kika.cov.parse_covmat import read_coverx

        file_path = Path(file_path)

        return read_coverx(str(file_path), ascending=ascending, energy_unit=energy_unit)

    def remove_matrix(
        self,
        isotope: int,
        reaction_pairs: List[Tuple[int, int]],
        exceptions: Optional[List[Tuple[int, int]]] = None
    ) -> "CrossSectionCovariance":
        """
        Return a new CrossSectionCovariance without the specified matrices for a given isotope,
        but always keep any pairs listed in exceptions.
        Also removes cross section entries for diagonal or wildcard reactions that are removed.
        """
        if exceptions is None:
            exceptions = []
        exc_set = set()
        for e1, e2 in exceptions:
            exc_set.add((e1, e2))
            exc_set.add((e2, e1))

        new = CrossSectionCovariance(
            num_groups=self.num_groups,
            energy_grid=list(self.energy_grid) if self.energy_grid is not None else None,
        )
        # copy retained covariance matrices
        for idx, (ir, rr, ic, rc, M) in enumerate(zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        )):
            remove = False
            if ir == isotope and ic == isotope:
                for r1, r2 in reaction_pairs:
                    if r1 == r2:
                        if rr == r1 or rc == r1:
                            remove = True
                    elif r1 == 0 or r2 == 0:
                        target = r2 if r1 == 0 else r1
                        if rr == target or rc == target:
                            remove = True
                    else:
                        if (rr == r1 and rc == r2) or (rr == r2 and rc == r1):
                            remove = True
                    if remove and (rr, rc) in exc_set:
                        remove = False
                    if remove:
                        break
            if not remove:
                new.isotope_rows.append(ir)
                new.reaction_rows.append(rr)
                new.isotope_cols.append(ic)
                new.reaction_cols.append(rc)
                new.matrices.append(M.copy())
                if idx < len(self.energy_grids):
                    new.energy_grids.append(list(self.energy_grids[idx]))
                if idx < len(self.is_relative):
                    new.is_relative.append(self.is_relative[idx])
        # copy retained cross sections
        for key, xs in self.cross_sections.items():
            zaid, mt = key
            remove_cs = False
            if zaid == isotope:
                for r1, r2 in reaction_pairs:
                    # diagonal removal
                    if r1 == r2 and mt == r1:
                        remove_cs = True
                    # wildcard removal (mt,0)
                    elif (r1 == 0 and mt == r2) or (r2 == 0 and mt == r1):
                        remove_cs = True
                    if remove_cs and key in exc_set:
                        remove_cs = False
                    if remove_cs:
                        break
            if not remove_cs:
                new.cross_sections[key] = xs.copy()
        return new
    




    # ------------------------------------------------------------------
    # Merge / transfer
    # ------------------------------------------------------------------

    def _align_block_lists(self) -> None:
        """Pad per-block lists to ``len(matrices)`` so positional appends stay aligned.

        ``is_relative`` and ``energy_grids`` are permitted to be shorter than
        ``matrices`` (some readers leave them empty), with consumers defaulting
        the missing entries. Before merging we materialise those defaults so that
        blocks appended afterwards keep a correct 1:1 index with their flag/grid.
        ``is_relative`` defaults to ``True`` (the convention used by
        :meth:`get_uncertainty`); ``energy_grids`` defaults to the shared
        ``energy_grid`` when one is available.
        """
        n = len(self.matrices)
        if len(self.is_relative) < n:
            self.is_relative.extend([True] * (n - len(self.is_relative)))
        if self.energy_grid is not None and len(self.energy_grids) < n:
            self.energy_grids.extend(
                list(self.energy_grid) for _ in range(n - len(self.energy_grids))
            )

    def _drop_reaction(self, isotope: int, reaction: int) -> None:
        """Remove, in place, every block touching ``(isotope, reaction)`` on either
        axis, plus its cross-section entry. Assumes lists were aligned first."""
        n = len(self.matrices)
        keep = [
            i for i in range(n)
            if not (
                (self.isotope_rows[i] == isotope and self.reaction_rows[i] == reaction)
                or (self.isotope_cols[i] == isotope and self.reaction_cols[i] == reaction)
            )
        ]

        def _sel(lst):
            return [lst[i] for i in keep] if len(lst) == n else lst

        self.isotope_rows = _sel(self.isotope_rows)
        self.reaction_rows = _sel(self.reaction_rows)
        self.isotope_cols = _sel(self.isotope_cols)
        self.reaction_cols = _sel(self.reaction_cols)
        self.matrices = _sel(self.matrices)
        self.is_relative = _sel(self.is_relative)
        self.energy_grids = _sel(self.energy_grids)
        self.cross_sections.pop((isotope, reaction), None)

    def _validate_mergeable(
        self, source: "CrossSectionCovariance", *, grid_atol: float, grid_rtol: float,
    ) -> None:
        """Raise ``ValueError`` if ``source`` cannot be merged into ``self`` because
        of an incompatible group structure. Regridding is intentionally unsupported."""
        import warnings

        dest_g = self.num_groups or (self.matrices[0].shape[0] if self.matrices else 0)
        src_g = source.num_groups or (source.matrices[0].shape[0] if source.matrices else 0)
        if dest_g and src_g and dest_g != src_g:
            raise ValueError(
                f"Cannot merge covariances with different group counts "
                f"(destination has {dest_g} groups, source has {src_g}). "
                "Regridding is not supported; convert both files to a common "
                "group structure first."
            )

        if self.energy_grid is not None and source.energy_grid is not None:
            src_in_dest = _rescale_energy_grid(
                source.energy_grid, source.energy_unit, self.energy_unit,
            )
            a = np.asarray(self.energy_grid, dtype=float)
            b = np.asarray(src_in_dest, dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, rtol=grid_rtol, atol=grid_atol):
                raise ValueError(
                    "Cannot merge covariances on different energy grids "
                    f"(destination unit={self.energy_unit}, source unit={source.energy_unit}). "
                    "Regridding is not supported; convert both files to a common "
                    "group structure first."
                )
        else:
            warnings.warn(
                "One or both covariances have no explicit energy grid; "
                "merging on matching group count only.",
                RuntimeWarning, stacklevel=2,
            )

    def transfer_reactions(
        self,
        source: "CrossSectionCovariance",
        reactions: Sequence[Tuple[int, int]],
        *,
        cross_correlation: str = "both-present",
        grid_atol: float = 1e-6,
        grid_rtol: float = 1e-6,
        replace: bool = True,
    ) -> "TransferResult":
        """Copy selected reaction blocks from ``source`` into a copy of ``self``.

        ``self`` is the *destination* and is never mutated; a new
        :class:`CrossSectionCovariance` is returned inside the
        :class:`TransferResult`.

        Parameters
        ----------
        source : CrossSectionCovariance
            The covariance to take blocks from.
        reactions : sequence of (isotope_zaid, mt)
            Reactions to transfer. For each, the diagonal self-block and the
            associated cross-section vector are copied, plus off-diagonal
            cross-correlation blocks according to ``cross_correlation``.
        cross_correlation : {'both-present', 'diagonal-only', 'always'}
            How to handle off-diagonal blocks of a transferred reaction:

            - ``'both-present'`` (default): copy a cross block only when *both*
              partner reactions exist in the destination after the transfer;
              otherwise record it in ``cross_dropped``.
            - ``'diagonal-only'``: never copy cross blocks.
            - ``'always'``: copy every cross block the reaction participates in,
              even if the partner reaction is absent (may leave dangling refs).
        grid_atol, grid_rtol : float
            Tolerances for the energy-grid equality check (after eV/MeV
            normalisation).
        replace : bool
            If True, a transferred reaction that already exists in the
            destination has its existing blocks/cross-section removed first.

        Returns
        -------
        TransferResult
            The merged covariance plus a report of what was transferred/dropped.

        Raises
        ------
        ValueError
            If the group structures are incompatible, or ``cross_correlation``
            is not a recognised mode.
        """
        valid_modes = {"both-present", "diagonal-only", "always"}
        if cross_correlation not in valid_modes:
            raise ValueError(
                f"cross_correlation must be one of {sorted(valid_modes)}, "
                f"got {cross_correlation!r}"
            )

        self._validate_mergeable(source, grid_atol=grid_atol, grid_rtol=grid_rtol)

        new = self.copy()
        new._align_block_lists()

        # De-duplicate the request, preserving order.
        requested: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for z, m in reactions:
            key = (int(z), int(m))
            if key not in seen:
                seen.add(key)
                requested.append(key)

        source_pairs = set(source._get_param_pairs())
        present = [p for p in requested if p in source_pairs]
        present_set = set(present)

        result = TransferResult(covariance=new)
        result.missing_in_source = [p for p in requested if p not in source_pairs]

        dest_existing = set(new._get_param_pairs())
        dest_after = dest_existing | present_set

        if replace:
            for pair in present:
                if pair in dest_existing:
                    new._drop_reaction(pair[0], pair[1])
                    result.reactions_replaced.append(pair)

        use_grid = new.energy_grid is not None
        for i in range(len(source.matrices)):
            ir = int(source.isotope_rows[i])
            rr = int(source.reaction_rows[i])
            ic = int(source.isotope_cols[i])
            rc = int(source.reaction_cols[i])
            row_pair = (ir, rr)
            col_pair = (ic, rc)
            is_diagonal = row_pair == col_pair

            if is_diagonal:
                if row_pair not in present_set:
                    continue
            else:
                if row_pair not in present_set and col_pair not in present_set:
                    continue
                if cross_correlation == "diagonal-only":
                    result.cross_dropped.append(((ir, rr, ic, rc), "diagonal-only mode"))
                    continue
                if cross_correlation == "both-present" and not (
                    row_pair in dest_after and col_pair in dest_after
                ):
                    missing = col_pair if row_pair in dest_after else row_pair
                    result.cross_dropped.append((
                        (ir, rr, ic, rc),
                        f"partner reaction {missing} not present in destination",
                    ))
                    continue

            new.isotope_rows.append(ir)
            new.reaction_rows.append(rr)
            new.isotope_cols.append(ic)
            new.reaction_cols.append(rc)
            new.matrices.append(np.array(source.matrices[i], copy=True))
            is_rel = source.is_relative[i] if i < len(source.is_relative) else True
            new.is_relative.append(bool(is_rel))
            if use_grid:
                src_grid = (
                    source.energy_grids[i]
                    if i < len(source.energy_grids) and source.energy_grids[i]
                    else None
                )
                if src_grid is not None:
                    new.energy_grids.append(
                        _rescale_energy_grid(src_grid, source.energy_unit, new.energy_unit)
                    )
                else:
                    new.energy_grids.append(list(new.energy_grid))

            if is_diagonal:
                result.diagonal_transferred.append(row_pair)
            else:
                result.cross_transferred.append((ir, rr, ic, rc))

        for pair in present:
            if pair in source.cross_sections:
                new.cross_sections[pair] = np.array(source.cross_sections[pair], copy=True)
                result.cross_sections_transferred.append(pair)

        new._invalidate_cov_graph_cache()
        return result

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_to_grid(
        self,
        target_bin_edges_ev: np.ndarray,
        *,
        xs_source: Optional[Any] = None,
        target_mt: Optional[int] = None,
        target_isotope: Optional[int] = None,
    ) -> np.ndarray:
        """Project a self-self covariance entry onto a target bin-edge grid.

        Selects the matrix where ``reaction_row == reaction_col == target_mt``
        (defaulting to the most common MT in the covariance) and projects
        every matching self-self contribution onto the target grid via
        ``kika.processing.multigroup`` (piecewise-constant overlap).

        Parameters
        ----------
        target_bin_edges_ev : np.ndarray
            Target bin edges in eV (length ``N+1``, strictly increasing).
        xs_source : MF3MT or CrossSection, optional
            Pointwise σ(E) used to convert any relative-form matrices to
            absolute. Required when at least one matched entry has
            ``is_relative=True``; otherwise that entry is skipped with a
            warning.
        target_mt : int, optional
            MT to look up. Defaults to the first ``reaction_rows`` value.
        target_isotope : int, optional
            Isotope (ZA) to look up. Defaults to the first ``isotope_rows``
            value.

        Returns
        -------
        np.ndarray
            Absolute covariance (b²), shape ``(N, N)``, symmetric.

        Notes
        -----
        PSD enforcement is the caller's responsibility. Use
        ``kika.cov.decomposition.cholesky_decomposition`` downstream when
        a positive-definite factor is needed.
        """
        from kika.processing.multigroup import (
            collapse_covariance,
            compute_rebin_operator,
            relative_to_absolute,
        )
        import warnings

        edges = np.asarray(target_bin_edges_ev, dtype=float)
        n_target = edges.size - 1
        if n_target < 1:
            raise ValueError("target_bin_edges_ev must have at least 2 entries")

        if not self.matrices:
            warnings.warn("CrossSectionCovariance has no matrices to project")
            return np.zeros((n_target, n_target), dtype=float)

        if target_mt is None:
            target_mt = int(self.reaction_rows[0])
        if target_isotope is None:
            target_isotope = int(self.isotope_rows[0])

        cov_total = np.zeros((n_target, n_target), dtype=float)
        matched = 0
        n_skipped_relative = 0

        for i in range(len(self.matrices)):
            iso_row = int(self.isotope_rows[i])
            iso_col = int(self.isotope_cols[i])
            mt_row = int(self.reaction_rows[i])
            mt_col = int(self.reaction_cols[i])
            if (
                iso_row != target_isotope or iso_col != target_isotope
                or mt_row != target_mt or mt_col != target_mt
            ):
                continue
            matched += 1

            if i >= len(self.energy_grids) or not self.energy_grids[i]:
                warnings.warn(
                    f"Matrix {i} for MT={target_mt} has no energy grid; skipping"
                )
                continue
            native_grid = np.asarray(self.energy_grids[i], dtype=float)
            native_mat = np.asarray(self.matrices[i], dtype=float)
            is_rel = bool(self.is_relative[i]) if i < len(self.is_relative) else False

            if is_rel:
                if xs_source is None:
                    n_skipped_relative += 1
                    continue
                native_centers = 0.5 * (native_grid[:-1] + native_grid[1:])
                if hasattr(xs_source, "get_cross_section"):
                    xs_at_centers = np.asarray(
                        xs_source.get_cross_section(native_centers), dtype=float,
                    )
                else:
                    xs_at_centers = np.asarray(
                        np.interp(native_centers, xs_source.energies, xs_source.values),
                        dtype=float,
                    )
                native_mat = relative_to_absolute(
                    native_mat, xs_at_centers, xs_at_centers,
                )

            rebin_op = compute_rebin_operator(native_grid, edges)
            cov_total = cov_total + collapse_covariance(native_mat, rebin_op)

        if matched == 0:
            warnings.warn(
                f"No self-self covariance entry found for "
                f"isotope={target_isotope}, MT={target_mt}; returning zeros"
            )
        if n_skipped_relative > 0:
            warnings.warn(
                f"Skipped {n_skipped_relative} relative covariance entries "
                f"because xs_source was not provided — covariance is under-counted."
            )

        # Symmetrize defensively
        cov_total = 0.5 * (cov_total + cov_total.T)
        return cov_total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full covariance matrix of shape (N·G) × (N·G),
        where N = number of unique (iso,rxn) blocks and G = num_groups.

        If :meth:`ensure_psd` has been called, the cached PSD matrix is
        returned directly.
        """
        if getattr(self, "_psd_covariance_cache", None) is not None:
            return self._psd_covariance_cache
        param_pairs = self._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}

        G = self.num_groups
        N = len(param_pairs) * G
        full = np.full((N, N), np.nan, dtype=float)

        for ir, rr, ic, rc, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        ):
            i = idx_map[(ir, rr)]
            j = idx_map[(ic, rc)]
            r0, r1 = i*G, (i+1)*G
            c0, c1 = j*G, (j+1)*G

            full[r0:r1, c0:c1] = M
            if i != j:
                full[c0:c1, r0:r1] = M.T

        return full

    @property
    def log_covariance_matrix(self) -> np.ndarray:
        """
        Return the log-space covariance matrix.
        
        Converts relative covariance to log-space using log1p transformation.
        
        Returns
        -------
        np.ndarray
            Log-space covariance matrix
        """
        cov_rel = self.covariance_matrix
        Sigma_log = np.log1p(cov_rel)
        return Sigma_log

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
            Set of unique isotope IDs
        """
        return sorted(set(self.isotope_rows + self.isotope_cols))
    
    @property
    def reactions(self) -> Set[int]:
        """
        Get the set of unique reaction MT numbers in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Set of unique reaction MT numbers
        """
        return sorted(set(self.reaction_rows + self.reaction_cols))

    @property
    def correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix (unclipped).
        
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        No off-diagonal correction is applied.
        
        Returns
        -------
        np.ndarray
            Correlation matrix with no clipping applied
        """
        from kika.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=False, force_diagonal=True)

    @property
    def clipped_correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix clipped to [-1, 1] range.
        
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        
        Returns
        -------
        np.ndarray
            Correlation matrix with values clipped to valid range [-1, 1]
        """
        from kika.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=True, force_diagonal=True)




    # ------------------------------------------------------------------
    # General methods
    # ------------------------------------------------------------------

    def reactions_by_isotope(self, isotope: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        '''
        Get a mapping of isotopes to their available reactions, or list of reactions for a specific isotope.

        Parameters
        ----------
        isotope : Optional[int]
            If provided, return reactions only for this isotope.

        Returns
        -------
        Dict[int, List[int]] or List[int]
            Mapping from isotope IDs to sorted lists of MT numbers, or list of MT numbers for the specified isotope.
        '''
        result: Dict[int, set] = {}

        # Process all row combinations
        for i, iso in enumerate(self.isotope_rows):
            result.setdefault(iso, set()).add(self.reaction_rows[i])

        # Process all column combinations
        for i, iso in enumerate(self.isotope_cols):
            result.setdefault(iso, set()).add(self.reaction_cols[i])

        # Convert sets to sorted lists
        sorted_dict: Dict[int, List[int]] = {iso: sorted(reactions) for iso, reactions in result.items()}

        if isotope is not None:
            # Return the list for the specified isotope, or empty list if not found
            return sorted_dict.get(isotope, [])

        return sorted_dict

    def clean_cov(self, isotope: int) -> "CrossSectionCovariance":
        """
        Return a new CrossSectionCovariance containing only sub-matrices for *isotope*,
        always dropping reaction 1 and applying the mid/high-range rules:
            4 → 51-91, 103 → 600-649, 104 → 650-699,
            105 → 700-749, 106 → 750-799, 107 → 800
        """

        # ---- 1. indices with the requested isotope on both axes ----
        idxs = [
            i
            for i, (iso_r, iso_c) in enumerate(
                zip(self.isotope_rows, self.isotope_cols)
            )
            if iso_r == isotope and iso_c == isotope
        ]

        # ---- 2. reactions present in those sub-matrices + XS vectors ----
        reac_present = {
            *[self.reaction_rows[i] for i in idxs],
            *[self.reaction_cols[i] for i in idxs],
            *[
                mt
                for (iso, mt) in self.cross_sections.keys()
                if iso == isotope
            ],
        }

        # ---- 3. mid → high map ----
        mid_high = {
            4: set(range(51, 92)),
            103: set(range(600, 650)),
            104: set(range(650, 700)),
            105: set(range(700, 750)),
            106: set(range(750, 800)),
            107: {800},
        }

        # ---- 4. decide which mid codes to drop ----
        drop_mid = {
            mid: any(code in high_set for code in reac_present)
            for mid, high_set in mid_high.items()
        }

        # ---- 5. build the new object ----
        nuevo = CrossSectionCovariance(
            num_groups=self.num_groups,
            energy_grid=self.energy_grid.copy() if self.energy_grid is not None else None,
        )

        # copy matrices that survive the filters
        for i in idxs:
            r = self.reaction_rows[i]
            c = self.reaction_cols[i]

            if r == 1 or c == 1:
                continue
            if (r in drop_mid and drop_mid[r]) or (c in drop_mid and drop_mid[c]):
                continue

            nuevo.isotope_rows.append(isotope)
            nuevo.reaction_rows.append(r)
            nuevo.isotope_cols.append(isotope)
            nuevo.reaction_cols.append(c)
            nuevo.matrices.append(self.matrices[i])
            if i < len(self.energy_grids):
                nuevo.energy_grids.append(list(self.energy_grids[i]))
            if i < len(self.is_relative):
                nuevo.is_relative.append(self.is_relative[i])

        # copy XS vectors that survive the same filters
        for (iso, mt), xs in self.cross_sections.items():
            if iso != isotope:
                continue
            if mt == 1:
                continue
            if mt in drop_mid and drop_mid[mt]:
                continue
            nuevo.cross_sections[(iso, mt)] = xs.copy()

        return nuevo
    
    def filter_by_isotope(self, isotope: int) -> "CrossSectionCovariance":
        """
        Return a new CrossSectionCovariance containing only sub-matrices and cross-sections for the given isotope.
        All reactions for that isotope are retained.

        Similar to clean_cov, but without dropping any reactions.
        """
        # Indices where both row and column isotopes match the requested isotope
        idxs = [
            i for i, (iso_r, iso_c) in enumerate(
                zip(self.isotope_rows, self.isotope_cols)
            ) if iso_r == isotope and iso_c == isotope
        ]

        new_cov = CrossSectionCovariance(
            num_groups=self.num_groups,
            energy_grid=self.energy_grid.copy() if self.energy_grid is not None else None,
        )

        for i in idxs:
            new_cov.isotope_rows.append(isotope)
            new_cov.reaction_rows.append(self.reaction_rows[i])
            new_cov.isotope_cols.append(isotope)
            new_cov.reaction_cols.append(self.reaction_cols[i])
            new_cov.matrices.append(self.matrices[i])
            if i < len(self.energy_grids):
                new_cov.energy_grids.append(list(self.energy_grids[i]))
            if i < len(self.is_relative):
                new_cov.is_relative.append(self.is_relative[i])

        for (iso, mt), xs in self.cross_sections.items():
            if iso == isotope:
                new_cov.cross_sections[(iso, mt)] = xs.copy()

        return new_cov

    def filter_by_isotopes(self, isotopes: Sequence[int]) -> "CrossSectionCovariance":
        """
        Return a new CrossSectionCovariance containing only matrices where BOTH row and column
        isotopes are in the specified list. Preserves cross-isotope blocks.

        Parameters
        ----------
        isotopes : Sequence[int]
            List of isotope ZAIDs to include

        Returns
        -------
        CrossSectionCovariance
            Filtered CrossSectionCovariance containing only matrices involving the specified isotopes
        """
        isotope_set = set(isotopes)
        idxs = [
            i for i, (iso_r, iso_c) in enumerate(
                zip(self.isotope_rows, self.isotope_cols)
            ) if iso_r in isotope_set and iso_c in isotope_set
        ]

        new_cov = CrossSectionCovariance(
            num_groups=self.num_groups,
            energy_grid=self.energy_grid.copy() if self.energy_grid else None,
        )

        for i in idxs:
            new_cov.isotope_rows.append(self.isotope_rows[i])
            new_cov.reaction_rows.append(self.reaction_rows[i])
            new_cov.isotope_cols.append(self.isotope_cols[i])
            new_cov.reaction_cols.append(self.reaction_cols[i])
            new_cov.matrices.append(self.matrices[i])
            if i < len(self.energy_grids):
                new_cov.energy_grids.append(list(self.energy_grids[i]))
            if i < len(self.is_relative):
                new_cov.is_relative.append(self.is_relative[i])

        # Copy cross sections for selected isotopes
        for (iso, mt), xs in self.cross_sections.items():
            if iso in isotope_set:
                new_cov.cross_sections[(iso, mt)] = xs.copy()

        return new_cov

    def filter_by_reactions(
        self,
        mts: Sequence[int],
        isotope: Optional[int] = None,
    ) -> "CrossSectionCovariance":
        """
        Return a new CrossSectionCovariance containing only matrices whose
        row and column reactions are both in *mts*. When per-matrix
        ``energy_grids`` are available (e.g. ENDF MF33), the returned object
        has ``num_groups`` and ``energy_grid`` set from the diagonal block of
        the first requested MT so that all plotting methods work directly.

        Parameters
        ----------
        mts : sequence of int
            Reaction MT numbers to keep.
        isotope : int, optional
            If given, also filter by this isotope.
        """
        mt_set = set(mts)
        idxs = [
            i for i, (rr, rc) in enumerate(
                zip(self.reaction_rows, self.reaction_cols)
            )
            if rr in mt_set and rc in mt_set
            and (isotope is None or (self.isotope_rows[i] == isotope and self.isotope_cols[i] == isotope))
        ]

        # Determine num_groups / energy_grid from the first diagonal block
        ng = 0
        eg = None
        for i in idxs:
            if self.reaction_rows[i] == self.reaction_cols[i]:
                ng = self.matrices[i].shape[0]
                if i < len(self.energy_grids):
                    eg = list(self.energy_grids[i])
                elif self.energy_grid is not None:
                    eg = list(self.energy_grid)
                break
        if ng == 0:
            ng = self.num_groups
        if eg is None and self.energy_grid is not None:
            eg = list(self.energy_grid)

        new_cov = CrossSectionCovariance(
            num_groups=ng,
            energy_grid=eg,
            energy_unit=self.energy_unit,
        )

        for i in idxs:
            new_cov.isotope_rows.append(self.isotope_rows[i])
            new_cov.reaction_rows.append(self.reaction_rows[i])
            new_cov.isotope_cols.append(self.isotope_cols[i])
            new_cov.reaction_cols.append(self.reaction_cols[i])
            new_cov.matrices.append(self.matrices[i])
            if i < len(self.energy_grids):
                new_cov.energy_grids.append(list(self.energy_grids[i]))
            if i < len(self.is_relative):
                new_cov.is_relative.append(self.is_relative[i])

        for (iso, mt), xs in self.cross_sections.items():
            if mt in mt_set and (isotope is None or iso == isotope):
                new_cov.cross_sections[(iso, mt)] = xs.copy()

        return new_cov

    def to_dataframe(self) -> pd.DataFrame:
        '''
        Convert the covariance matrix data to a pandas DataFrame.

        Includes an extra row at the beginning to store the energy grid if available.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, ISO_V, REAC_V, STD
        '''
        # Convert matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.matrices]

        # Create DataFrame for covariance matrices
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "STD": matrix_lists
        }
        df = pd.DataFrame(data)

        # Add energy grid row if available
        if self.energy_grid is not None:
            energy_grid_row = pd.DataFrame({
                "ISO_H": [0],
                "REAC_H": [0],
                "ISO_V": [0],
                "REAC_V": [0],
                "STD": [self.energy_grid]  # Store the list directly
            })
            # Concatenate the energy grid row at the beginning
            df = pd.concat([energy_grid_row, df], ignore_index=True)

        # Sort by ISO_H, then REAC_H, then REAC_V
        df = df.sort_values(by=["ISO_H", "REAC_H", "REAC_V"]).reset_index(drop=True)

        return df

    def fix_covariance(
        self,
        *,
        level: str = "soft",            # 'soft' | 'medium' | 'hard'
        high_val_thresh: float = 5.0,
        clamp_target: float = 1.0,
        accept_tol: float = -1.0e-4,
        clamp_max_iter: int = 10,
        max_steps: int = 40,
        verbose: bool = True,
        clamp_detail: bool = False,
        logger = None,  # Optional logger for file output
    ) -> Tuple["CrossSectionCovariance", Dict[str, Any]]:
        """
        Clean the covariance matrix until it is positive-(semi)definite.

        Parameters
        ----------
        level
            'soft'   - clamp variances only
            'medium' - clamp then drop the worst *block pairs*
            'hard'   - clamp then drop the worst *reactions* (all blocks)
        high_val_thresh
            Diagonal variances above this threshold are clamped.
        clamp_target
            Target variance assigned to clamped diagonals (off-diagonals
            rescaled by sqrt(target/|old|) to preserve correlations).
        accept_tol
            Minimum eigenvalue tolerated for acceptance.
        clamp_max_iter
            Maximum clamping passes before switching strategy.
        max_steps
            Maximum block-removal iterations (if used).
        verbose
            Forwarded to the removal routine.
        clamp_detail
            Dump every off-diagonal before/after pair on each clamp event.
            Very noisy — opt in only to debug a specific clamp.
        logger
            Optional logger instance for file output. If None, uses print().
        """

        lvl = level.lower()
        if lvl not in ("soft", "medium", "hard"):
            raise ValueError("level must be 'soft', 'medium' or 'hard'")

        # ------------------------------------------------------------------
        # 1) Always start with variance clamping
        # ------------------------------------------------------------------
        cm_after_clamp, log = self._clamp_covariance(
            high_val_thresh=high_val_thresh,
            clamp_target=clamp_target,
            accept_tol=accept_tol,
            max_iter=clamp_max_iter,
            verbose=verbose,
            clamp_detail=clamp_detail,
            logger=logger,
        )

        # If clamping was enough, stop here
        if log.get("converged", False):
            log.update({
                "strategy": lvl,
                "used_removal": False,
                "soft_threshold_met": True,  # Add flag for soft level success
            })
            return cm_after_clamp, log

        # For soft level, we don't do removal but we mark that threshold wasn't met
        if lvl == "soft":
            log.update({
                "strategy": lvl,
                "used_removal": False,
                "soft_threshold_met": False,  # Add flag for soft level failure
            })
            return cm_after_clamp, log

        # ------------------------------------------------------------------
        # 2) Continue with block removal (medium / hard)
        # ------------------------------------------------------------------
        remove_whole_rxn = (lvl == "hard")
        cm_final, rem_log = cm_after_clamp._autofix_covariance(
            accept_tol=accept_tol,
            max_steps=max_steps,
            verbose=verbose,
            remove_all=remove_whole_rxn,
            high_val_thresh=high_val_thresh,
            logger=logger,
        )

        # Merge the two logs for convenience - ensure final eigenvalue is used
        out_log: Dict[str, Any] = {
            **log,
            **{k: v for k, v in rem_log.items() if k not in ["min_eigenvalue"]},  # Don't overwrite final eigenvalue
            "strategy": lvl,
            "used_removal": True,
            "removal_log": rem_log,
            "min_eigenvalue": rem_log.get("min_eigenvalue", log.get("min_eigenvalue")),  # Use final eigenvalue
        }
        return cm_final, out_log

    def ensure_psd(
        self,
        *,
        preserve_diagonal: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-10,
        eigval_floor: float = 0.0,
        verbose: bool = True,
        logger=None,
    ) -> Tuple["CrossSectionCovariance", Dict[str, Any]]:
        """
        Return a copy whose :attr:`covariance_matrix` is the nearest PSD matrix.

        Uses Higham's alternating-projection algorithm.  The original
        sub-matrices are **not** modified; instead, the assembled PSD matrix
        is cached and returned by the ``covariance_matrix`` /
        ``log_covariance_matrix`` properties.

        Parameters
        ----------
        preserve_diagonal : bool
            If True, preserve the original variances (diagonal elements).
        max_iter : int
            Maximum number of Higham iterations.
        tol : float
            Convergence tolerance on relative Frobenius change.
        eigval_floor : float
            Floor for eigenvalues in the PSD projection step.
        verbose : bool
            Whether to print diagnostic information.
        logger : optional
            Logger instance for file output.

        Returns
        -------
        Tuple[CrossSectionCovariance, dict]
            (psd_copy, info) where *psd_copy* has the PSD cache set and
            *info* is the diagnostic dict from ``nearest_psd_higham``.
        """
        from kika.cov.decomposition import nearest_psd_higham

        full = self.covariance_matrix
        psd_full, info = nearest_psd_higham(
            full,
            preserve_diagonal=preserve_diagonal,
            max_iter=max_iter,
            tol=tol,
            eigval_floor=eigval_floor,
            verbose=verbose,
            logger=logger,
        )
        out = self.copy()
        out._psd_covariance_cache = psd_full
        return out, info

    def eigen_block_contributions(
        self,
        idx: Optional[int] = None,
        which: str = "min",      # "max", "min", or ignored when idx is not None
        top_n: int = 10,
        tol: float = 1e-12,
        symmetric: bool = True,  # if True keep only a ≤ b blocks
        relative: bool = False   # if True add weight = |c| / |eigenvalue|
    ) -> Dict[str, Any]:
        """
        Measure how each (block_i, block_j) sub-matrix of the covariance matrix
        contributes to a chosen eigenvalue λ.

        Args
        ----
        idx        : Index of the eigenvalue to inspect (overrides *which*).
        which      : If *idx* is None, choose 'max' or 'min' eigenvalue.
        top_n      : Return only the *top_n* largest |contribution| entries.
        tol        : Skip blocks with |contribution| ≤ tol.
        symmetric  : If True, include only pairs with i ≤ j (avoids duplicates).
        relative   : If True, report each block's share of |λ|.

        Returns
        -------
        A dict with:
            'index'        : eigenvalue index used.
            'eigenvalue'   : eigenvalue λ.
            'contributions': list of dicts:
                { 'block': (param_pairs[i], param_pairs[j]),
                    'contribution': c_ij,
                    'weight': |c_ij| / |λ|   # only if relative=True }
        """
        M = self.covariance_matrix               # full (n × n) matrix
        G = self.num_groups                      # rows per single block
        param_pairs = self._get_param_pairs()    # list of block labels
        n_blocks = len(param_pairs)

        # --- eigen-decomposition -------------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(M)

        if idx is None:
            if which == "max":
                idx = int(np.argmax(eigvals))
            elif which == "min":
                idx = int(np.argmin(eigvals))
            else:
                raise ValueError("`which` must be 'max' or 'min' when idx is None")

        λ = float(eigvals[idx])
        v = eigvecs[:, idx]

        # Pre-slice the eigenvector into the same block structure
        v_blocks: List[np.ndarray] = [
            v[a * G:(a + 1) * G] for a in range(n_blocks)
        ]

        contribs: List[Dict[str, Any]] = []
        for i in range(n_blocks):
            vi = v_blocks[i]
            r0, r1 = i * G, (i + 1) * G
            for j in range(i if symmetric else 0, n_blocks):
                vj = v_blocks[j]
                c0, c1 = j * G, (j + 1) * G
                block = M[r0:r1, c0:c1]
                c_ij = float(vi @ block @ vj)
                if abs(c_ij) > tol:
                    entry: Dict[str, Any] = {
                        "block": (param_pairs[i], param_pairs[j]),
                        "contribution": c_ij,
                    }
                    if relative and abs(λ) > 0:
                        entry["weight"] = abs(c_ij) / abs(λ)
                    contribs.append(entry)

        # Sort and truncate
        contribs.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        contribs = contribs[:top_n]

        # --- Print nicely formatted summary ---
        print("=" * 80)
        print(f"Eigenvalue block contributions (idx={idx}, λ={λ:.4e})")
        print(f"Top {top_n} block contributions (tol={tol}, symmetric={symmetric}, relative={relative}):")
        print("-" * 80)
        header = (
            f"{'Block (iso,rxn)-(iso,rxn)':<35} {'Contribution':>15}"
            + (f" {'|c|/|λ|':>12}" if relative else "")
        )
        print(header)
        print("-" * 80)
        for entry in contribs:
            (b1, b2) = entry["block"]
            sblock = f"({b1[0]},{b1[1]})-({b2[0]},{b2[1]})"
            contrib_val = entry["contribution"]
            if relative and "weight" in entry:
                print(f"{sblock:<35} {contrib_val:15.6e} {entry['weight']:12.4e}")
            else:
                print(f"{sblock:<35} {contrib_val:15.6e}")
        print("=" * 80)

        return {
            "index": idx,
            "eigenvalue": λ,
            "contributions": contribs
        }
    
    def get_uncertainty(
        self,
        zaid: int,
        mt: int,
        energy_mev: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Get relative uncertainty (standard deviation / mean) for a specific isotope and reaction.
        
        The covariance matrices store relative covariances, so the square root of the
        diagonal elements directly gives relative uncertainties (as fractions).
        
        Parameters
        ----------
        zaid : int
            Isotope identifier (e.g., 26056 for Fe-56)
        mt : int
            Reaction MT number
        energy_mev : float, optional
            Specific energy in MeV. If provided, returns uncertainty at that energy.
            If None, returns array of uncertainties for all energy groups.
        
        Returns
        -------
        float or np.ndarray
            Relative uncertainty (as fraction, e.g., 0.05 for 5%).
            If energy_mev is provided, returns a single float.
            If energy_mev is None, returns array of uncertainties for all groups.
        
        Raises
        ------
        ValueError
            If the specified (zaid, mt) pair is not found in the covariance data
        """
        # Filter to get only this isotope's data
        try:
            iso_covmat = self.filter_by_isotope(zaid)
        except Exception as e:
            raise ValueError(f"Could not filter covariance data for ZAID={zaid}: {e}")
        
        # Get parameter pairs and find the index for this (zaid, mt) pair
        pairs = iso_covmat._get_param_pairs()
        if (zaid, mt) not in pairs:
            raise ValueError(
                f"No covariance data found for ZAID={zaid}, MT={mt}. "
                f"Available pairs: {pairs}"
            )
        
        # Get the diagonal of the covariance matrix
        G = iso_covmat.num_groups
        full_cov = iso_covmat.covariance_matrix
        diag = np.sqrt(np.maximum(np.diag(full_cov), 0.0))

        # Find the block index for this (zaid, mt) pair
        block_idx = pairs.index((zaid, mt))

        # Extract the uncertainties for this reaction
        uncertainties = diag[block_idx * G : (block_idx + 1) * G]

        # Determine if this block is relative or absolute
        mat_is_relative = True  # default: assume relative
        for mi, (ir, rr, ic, rc) in enumerate(zip(
            iso_covmat.isotope_rows, iso_covmat.reaction_rows,
            iso_covmat.isotope_cols, iso_covmat.reaction_cols)):
            if ir == zaid and rr == mt and ic == zaid and rc == mt:
                if mi < len(iso_covmat.is_relative):
                    mat_is_relative = iso_covmat.is_relative[mi]
                break

        if not mat_is_relative and (zaid, mt) in self.cross_sections:
            # Absolute covariance: divide by cross section to get relative
            xs = np.asarray(self.cross_sections[(zaid, mt)], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                uncertainties = uncertainties / np.abs(xs)
                uncertainties = np.nan_to_num(uncertainties, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If specific energy requested, find the corresponding group
        if energy_mev is not None:
            if self.energy_grid is None:
                raise ValueError("No energy grid available in covariance data")
            
            energy_grid = np.array(self.energy_grid)
            
            # Detect units: if first value > 1000, assume eV, otherwise MeV
            if energy_grid[0] > 1000:
                # Grid is in eV, convert input energy to eV
                energy_to_match = energy_mev * 1e6
            else:
                # Grid is already in MeV
                energy_to_match = energy_mev
            
            # Find the bin index (energy grid has G+1 boundaries for G groups)
            group_idx = None
            for i in range(len(energy_grid) - 1):
                if energy_grid[i] <= energy_to_match < energy_grid[i+1]:
                    group_idx = i
                    break
            
            # Check upper boundary edge case
            if group_idx is None:
                tolerance = 1e-3 if energy_grid[0] <= 1000 else 1e3
                if abs(energy_to_match - energy_grid[-1]) < tolerance:
                    group_idx = len(energy_grid) - 2
            
            if group_idx is None:
                if energy_grid[0] > 1000:
                    raise ValueError(
                        f"Energy {energy_mev} MeV ({energy_to_match:.4e} eV) is outside "
                        f"covariance energy range: {energy_grid[0]/1e6:.4f} - "
                        f"{energy_grid[-1]/1e6:.4f} MeV"
                    )
                else:
                    raise ValueError(
                        f"Energy {energy_mev} MeV is outside covariance energy range: "
                        f"{energy_grid[0]:.4f} - {energy_grid[-1]:.4f} MeV"
                    )
            
            return float(uncertainties[group_idx])
        
        # Return all uncertainties
        return uncertainties
    

    
    def plot_uncertainties(
        self,
        nuclide: Union[int, str, Sequence[Union[int, str]]],
        mt:   Union[int, Sequence[int]],
        *,
        energy_range: Optional[Tuple[float, float]] = None,
        style: str = 'default',
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 300,
        font_family: str = 'serif',
        legend_loc: str = 'best',
        xscale: str = 'log',
        yscale: str = 'linear',
        title: Optional[str] = 'default',
        **step_kwargs
    ) -> plt.Figure:
        """
        Plot relative uncertainties for one or more (ZAID, MT) pairs.
        
        This method now uses the modern PlotBuilder-based implementation from
        kika.plotting.covariance for cleaner, more maintainable code.
        """
        from kika.plotting.covariance import plot_uncertainties as _plot_uncertainties

        return _plot_uncertainties(
            covmat=self,
            nuclide=nuclide,
            mt=mt,
            energy_range=energy_range,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            xscale=xscale,
            yscale=yscale,
            title=title,
            **step_kwargs,
        )
    
    def to_plot_data(
        self,
        nuclide: Union[int, str],
        mt: int,
        sigma: float = 1.0,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create PlotData objects for multigroup cross sections with uncertainties.
        
        This unified method extracts both nominal cross section data and uncertainty data.
        Both are returned as PlotData objects that can be plotted independently or combined.
        
        Parameters
        ----------
        nuclide : int or str
            Isotope identifier. Can be either:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
        mt : int
            Reaction MT number
        sigma : float, optional
            Number of sigma levels for uncertainty bands (default: 1.0 for 1σ).
        label : str, optional
            Custom label for the plot. If None, auto-generates from ZAID and MT.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        tuple of (MultigroupCrossSectionPlotData, MultigroupUncertaintyPlotData)
            - xs_data: Cross section data for plotting (or None if not available)
            - unc_data: Uncertainty data as percentages (or None if not available)
            
        Raises
        ------
        ValueError
            If the specified (nuclide, mt) pair is not found in either cross sections or covariance data
            
        Examples
        --------
        >>> # Extract data - both integer ZAID and string notation work
        >>> covmat = read_njoy_covmat('file.gendf')
        >>> xs_data, unc_data = covmat.to_plot_data(nuclide=26056, mt=2)
        >>> xs_data, unc_data = covmat.to_plot_data(nuclide='Fe56', mt=2)
        >>> 
        >>> # Use with PlotBuilder:
        >>> from kika.plotting import PlotBuilder
        >>> 
        >>> # Option 1: Plot just cross sections
        >>> fig1 = PlotBuilder().add_data(xs_data).build()
        >>> 
        >>> # Option 2: Plot cross sections with uncertainty shading
        >>> fig2 = PlotBuilder().add_data(xs_data, uncertainty=unc_data).build()
        >>> 
        >>> # Option 3: Plot just uncertainties as a line
        >>> fig3 = PlotBuilder().add_data(unc_data).build()
        """
        from kika.plotting import MultigroupUncertaintyPlotData
        from kika._utils import zaid_to_symbol, symbol_to_zaid
        
        # Convert nuclide to ZAID if string
        if isinstance(nuclide, str):
            zaid = symbol_to_zaid(nuclide)
        else:
            zaid = nuclide
        
        # Extract cross section data (will add sigma notation to label if uncertainties exist)
        xs_data = self._extract_xs_data(zaid, mt, label, sigma, **styling_kwargs)
        
        # Extract uncertainty data as MultigroupUncertaintyPlotData
        unc_data = None
        key = (zaid, mt)
        
        # Find the diagonal covariance matrix
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows,
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == zaid and mt_r == mt and iso_c == zaid and mt_c == mt:
                # Found diagonal block - extract uncertainties
                cov_matrix = self.matrices[i]
                diag = np.diag(cov_matrix)
                
                # Get cross sections for this reaction (needed for creating MultigroupUncertaintyPlotData)
                if key in self.cross_sections:
                    xs = np.asarray(self.cross_sections[key], dtype=float)
                    
                    # IMPORTANT: The covariance matrix from GENDF is already relative!
                    # sqrt(diag) gives us the relative standard deviation (fractional)
                    # We should NOT divide by xs again!
                    rel_unc = np.sqrt(diag)
                    
                    # Ensure finite values
                    rel_unc = np.where(np.isfinite(rel_unc), rel_unc, 0.0)
                    
                    # Convert to percentage and apply sigma multiplier
                    rel_unc_pct = rel_unc * 100.0 * sigma
                    
                    # For step plots with 'post', we need G+1 points (bin edges)
                    # with the last y-value repeated to show all G bins properly
                    if self.energy_grid is not None:
                        energy_edges = np.asarray(self.energy_grid, dtype=float)
                        # Use energy edges (G+1 points) for x-axis
                        x_values = energy_edges
                        # Extend y-values to G+1 by repeating the last value
                        y_values = np.r_[rel_unc_pct, rel_unc_pct[-1]]
                    else:
                        # Fallback: use indices as edges
                        x_values = np.arange(len(rel_unc_pct) + 1, dtype=float)
                        y_values = np.r_[rel_unc_pct, rel_unc_pct[-1]]
                    
                    # Generate label (for uncertainty data line plot)
                    if label is None:
                        try:
                            isotope_symbol = zaid_to_symbol(zaid)
                        except Exception:
                            isotope_symbol = f"ZAID {zaid}"
                        
                        reaction_name = MT_TO_REACTION.get(mt, f"MT={mt}")
                        
                        sigma_str = f"{sigma}σ" if sigma != 1.0 else "1σ"
                        label = f"{isotope_symbol} {reaction_name} Uncertainty ({sigma_str})"
                    
                    # Create MultigroupUncertaintyPlotData
                    unc_data = MultigroupUncertaintyPlotData(
                        x=x_values,
                        y=y_values,
                        label=label,
                        zaid=zaid,
                        mt=mt,
                        uncertainty_type='relative',
                        energy_bins=energy_edges if self.energy_grid is not None else None,
                        step_where='post',
                        **styling_kwargs
                    )
                break
        
        # Check if at least one of them is available
        if xs_data is None and unc_data is None:
            raise ValueError(
                f"No data found for ZAID={zaid}, MT={mt}. "
                "Neither cross sections nor covariance data available."
            )
        
        return xs_data, unc_data
    
    def plot_multigroup_xs(
        self,
        nuclide: Union[int, str, Sequence[Union[int, str]]],
        mt: Union[int, Sequence[int]],
        ax: plt.Axes = None,
        *,
        energy_range: Optional[Tuple[float, float]] = None,
        show_uncertainties: bool = False,
        sigma: float = 1.0,
        style: str = 'default',
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 300,
        font_family: str = 'serif',
        legend_loc: str = 'best',
        xscale: str = 'log',
        yscale: str = 'linear',
        title: Optional[str] = 'default',
        **step_kwargs
    ) -> plt.Figure:
        """
        Plot multigroup cross sections with optional uncertainty bands.
        
        This method now uses the modern PlotBuilder-based implementation from
        kika.plotting.covariance for cleaner, more maintainable code.
        """
        from kika.plotting.covariance import plot_multigroup_xs as _plot_multigroup_xs

        return _plot_multigroup_xs(
            covmat=self,
            nuclide=nuclide,
            mt=mt,
            energy_range=energy_range,
            show_uncertainties=show_uncertainties,
            sigma=sigma,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            xscale=xscale,
            yscale=yscale,
            title=title,
            **step_kwargs,
        )
    
    def to_heatmap_data(
        self,
        nuclide: Union[int, str, Sequence[Union[int, str]]],
        mt: Union[int, Sequence[int], Tuple[int, int]],
        *,
        matrix_type: str = 'corr',
        scale: str = 'log',
        energy_range: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> 'CovarianceHeatmapData':
        """
        Prepare covariance heatmap data for PlotBuilder rendering.

        This method extracts the relevant matrix data, computes uncertainties,
        and packages everything into a CovarianceHeatmapData object that can
        be rendered by PlotBuilder.add_heatmap().

        Parameters
        ----------
        nuclide : int, str, or sequence of int/str
            Isotope identifier(s). Can be:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
            - List of ZAIDs or strings for multi-isotope heatmaps (e.g., ['Fe54', 'Fe56'])
        mt : int, sequence of int, or tuple of (row_mt, col_mt)
            MT reaction number(s). Can be:
            - Single int: diagonal block for that MT
            - Sequence of ints: diagonal blocks for those MTs
            - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
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
        CovarianceHeatmapData
            Heatmap data object ready for PlotBuilder.add_heatmap()

        Examples
        --------
        >>> # Simple usage with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> heatmap_data = covmat.to_heatmap_data(nuclide=92235, mt=[2, 18, 102])
        >>> fig = PlotBuilder(style='light').add_heatmap(heatmap_data)
        >>> fig.show()

        >>> # Can also use string symbols
        >>> heatmap_data = covmat.to_heatmap_data(nuclide='U235', mt=2, matrix_type='cov')

        >>> # Multi-isotope heatmap
        >>> heatmap_data = covmat.to_heatmap_data(nuclide=['Fe54', 'Fe56'], mt=[2, 18])
        """
        from kika.plotting.plot_data import CovarianceHeatmapData
        from kika._utils import symbol_to_zaid

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

        # Normalize nuclide(s) to ZAID(s) - detect multi-isotope case
        if isinstance(nuclide, (list, tuple)) and not isinstance(nuclide, str):
            zaids = [symbol_to_zaid(n) if isinstance(n, str) else n for n in nuclide]
            if len(zaids) > 1:
                # Multi-isotope case - delegate to specialized method
                return self._to_heatmap_data_multi_isotope(
                    zaids=zaids, mt=mt, matrix_type=matrix_type_normalized,
                    scale=scale_normalized, energy_range=energy_range, **kwargs
                )
            zaid = zaids[0]  # Single element list - use existing logic
        else:
            # Single nuclide (int or str)
            if isinstance(nuclide, str):
                zaid = symbol_to_zaid(nuclide)
            else:
                zaid = nuclide

        def _transform_edges(edges: np.ndarray) -> np.ndarray:
            if scale_normalized == "log":
                safe = np.maximum(edges, 1e-300)
                return np.log10(safe.astype(float))
            return edges.astype(float)

        def _crop_edges(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Return cropped edges and keep_idx mask (bin indices kept)."""
            if energy_range is None:
                keep_mask = np.ones(len(edges) - 1, dtype=bool)
                return edges, keep_mask
            emin, emax = energy_range
            if not (np.isfinite(emin) and np.isfinite(emax)) or emin >= emax:
                raise ValueError("energy_range must be a tuple (emin, emax) with emin < emax.")
            keep_mask = (edges[1:] > float(emin)) & (edges[:-1] < float(emax))
            if not np.any(keep_mask):
                raise ValueError("energy_range removed all groups; nothing to plot.")
            first, last = np.where(keep_mask)[0][[0, -1]]
            cropped = edges[first:last + 2]
            new_mask = np.zeros(len(edges) - 1, dtype=bool)
            new_mask[first:last + 1] = True
            return cropped, new_mask

        # 1. Filter by isotope
        iso_cov = self.filter_by_isotope(zaid)
        pairs = iso_cov._get_param_pairs()
        if iso_cov.num_groups == 0:
            raise ValueError(f"No data found for isotope {zaid}")

        # Energy grid and optional cropping
        if iso_cov.energy_grid is None:
            raise ValueError("Covariance matrix missing energy grid information for plotting.")
        edges_raw_full = np.asarray(iso_cov.energy_grid, dtype=float)

        # Convert to eV if energy is in different units (for consistency with MF34)
        energy_unit = getattr(self, 'energy_unit', 'eV')
        if energy_unit.lower() == 'mev':
            edges_raw_full = edges_raw_full * 1e6  # MeV to eV

        edges_cropped, keep_mask = _crop_edges(edges_raw_full)
        G = len(edges_cropped) - 1

        # 2. Parse MT input and determine diagonal vs off-diagonal
        if isinstance(mt, tuple) and len(mt) == 2:
            # Off-diagonal block: (row_mt, col_mt)
            is_diagonal = False
            row_mt, col_mt = mt
            mts = [row_mt, col_mt]
            
            # Find row indices for row_mt
            row_pairs = [(z, m) for z, m in pairs if z == zaid and m == row_mt]
            if not row_pairs:
                raise ValueError(f"MT {row_mt} not found for isotope {zaid}")
            
            # Find col indices for col_mt
            col_pairs = [(z, m) for z, m in pairs if z == zaid and m == col_mt]
            if not col_pairs:
                raise ValueError(f"MT {col_mt} not found for isotope {zaid}")
            
            # Calculate index ranges
            row_idx = pairs.index(row_pairs[0])
            col_idx = pairs.index(col_pairs[0])
            rows_full = list(range(row_idx * iso_cov.num_groups, (row_idx + 1) * iso_cov.num_groups))
            cols_full = list(range(col_idx * iso_cov.num_groups, (col_idx + 1) * iso_cov.num_groups))
            rows = [rows_full[i] for i, keep in enumerate(keep_mask) if keep]
            cols = [cols_full[i] for i, keep in enumerate(keep_mask) if keep]
            
        else:
            # Diagonal blocks
            is_diagonal = True
            
            # Normalize to list
            if isinstance(mt, int):
                mts = [mt]
            else:
                mts = sorted(list(mt))
            
            # Find indices for each MT
            rows = []
            cols = []
            for m in mts:
                mt_pairs = [(z, mt_val) for z, mt_val in pairs if z == zaid and mt_val == m]
                if not mt_pairs:
                    raise ValueError(f"MT {m} not found for isotope {zaid}")
                
                idx = pairs.index(mt_pairs[0])
                base_indices = list(range(idx * iso_cov.num_groups, (idx + 1) * iso_cov.num_groups))
                kept = [base_indices[i] for i, keep in enumerate(keep_mask) if keep]
                rows.extend(kept)
                cols.extend(kept)
        
        # 3. Extract matrix
        if matrix_type_normalized == 'corr':
            M_full = iso_cov.clipped_correlation_matrix[np.ix_(rows, cols)]
            mask_value = 0.0
        else:  # 'cov'
            M_full = iso_cov.covariance_matrix[np.ix_(rows, cols)]
            mask_value = None

        # For covariance matrices, mark zero-variance regions as NaN
        # so they render as grey (same as correlation matrices)
        if matrix_type_normalized == 'cov':
            if is_diagonal:
                diag_var = np.diag(M_full)
                std = np.sqrt(np.abs(diag_var))
                # Use threshold to catch near-zero variances (floating-point tolerance)
                max_std = np.nanmax(std) if np.any(np.isfinite(std)) else 1.0
                threshold = max_std * 1e-12 if max_std > 0 else 1e-30
                invalid_mask = ~np.isfinite(std) | (std < threshold)
                if np.any(invalid_mask):
                    M_full = M_full.copy()
                    M_full[invalid_mask, :] = np.nan
                    M_full[:, invalid_mask] = np.nan
            else:
                # Off-diagonal: get variances from the full covariance matrix
                full_cov = iso_cov.covariance_matrix
                row_var = np.diag(full_cov)[rows]
                col_var = np.diag(full_cov)[cols]
                row_std = np.sqrt(np.abs(row_var))
                col_std = np.sqrt(np.abs(col_var))
                # Use threshold to catch near-zero variances (floating-point tolerance)
                all_std = np.concatenate([row_std[np.isfinite(row_std)], col_std[np.isfinite(col_std)]])
                max_std = np.nanmax(all_std) if len(all_std) > 0 else 1.0
                threshold = max_std * 1e-12 if max_std > 0 else 1e-30
                row_invalid = ~np.isfinite(row_std) | (row_std < threshold)
                col_invalid = ~np.isfinite(col_std) | (col_std < threshold)
                if np.any(row_invalid) or np.any(col_invalid):
                    M_full = M_full.copy()
                    M_full[row_invalid, :] = np.nan
                    M_full[:, col_invalid] = np.nan

        # 4. Prepare geometry and block_info
        transformed_edges = _transform_edges(edges_cropped)
        width = transformed_edges[-1] - transformed_edges[0]
        energy_ranges = {}
        ranges_energy = []

        if is_diagonal:
            x_parts = []
            for i, _ in enumerate(mts):
                start = i * width
                block_edges = (transformed_edges - transformed_edges[0]) + start
                x_parts.append(block_edges if i == 0 else block_edges[1:])
                energy_ranges[mts[i]] = (block_edges[0], block_edges[-1])
                ranges_energy.append((block_edges[0], block_edges[-1]))
            x_edges = np.concatenate(x_parts) if x_parts else None
            y_edges = x_edges.copy() if x_edges is not None else None
            extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])) if x_edges is not None else None
            ranges = [(r[0], r[1]) for r in ranges_energy]
        else:
            x_edges = transformed_edges - transformed_edges[0]
            y_edges = transformed_edges - transformed_edges[0]
            extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
            energy_ranges[mts[0]] = (y_edges[0], y_edges[-1])
            energy_ranges[mts[1]] = (x_edges[0], x_edges[-1])
            ranges = [(0.0, width), (0.0, width)]

        block_info = {
            'mts': mts,
            'G': G,
            'ranges': ranges,
            'energy_ranges': energy_ranges,
        }
        
        # 5. Compute uncertainties (always computed; rendering controlled by PlotBuilder.add_heatmap)
        uncertainty_data = {}

        if True:  # Always compute uncertainties
            cov_matrix = iso_cov.covariance_matrix
            for m in mts:
                mt_pairs = [(z, mt_val) for z, mt_val in pairs if z == zaid and mt_val == m]
                if not mt_pairs:
                    continue

                idx = pairs.index(mt_pairs[0])
                base_indices = list(range(idx * iso_cov.num_groups, (idx + 1) * iso_cov.num_groups))
                mt_rows = [base_indices[i] for i, keep in enumerate(keep_mask) if keep]

                diag_variance = np.diag(cov_matrix)[mt_rows]

                # Determine if the diagonal block for this (zaid, m) is relative
                mat_is_relative = True  # default: assume relative
                for mi, (ir, rr, ic, rc) in enumerate(zip(
                    iso_cov.isotope_rows, iso_cov.reaction_rows,
                    iso_cov.isotope_cols, iso_cov.reaction_cols)):
                    if ir == zaid and rr == m and ic == zaid and rc == m:
                        if mi < len(iso_cov.is_relative):
                            mat_is_relative = iso_cov.is_relative[mi]
                        break

                if mat_is_relative:
                    # Data is already relative variance -> sqrt * 100
                    sigma_percent = np.sqrt(np.abs(diag_variance)) * 100.0
                elif (zaid, m) in self.cross_sections:
                    # Absolute variance -> divide by cross section to get relative
                    nominal_xs_full = np.asarray(self.cross_sections[(zaid, m)], dtype=float)
                    nominal_xs = nominal_xs_full[keep_mask] if nominal_xs_full.size == iso_cov.num_groups else nominal_xs_full

                    with np.errstate(divide='ignore', invalid='ignore'):
                        sigma_percent = np.sqrt(np.abs(diag_variance)) / np.abs(nominal_xs) * 100.0
                        sigma_percent = np.nan_to_num(sigma_percent, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # Absolute variance but no cross sections -- fall back to sqrt * 100
                    sigma_percent = np.sqrt(np.abs(diag_variance)) * 100.0

                uncertainty_data[m] = sigma_percent

            if not uncertainty_data:
                uncertainty_data = None
        
        # 6. Get energy grid (cropped)
        energy_grid = edges_cropped
        
        # 7. Generate label
        from kika._utils import zaid_to_symbol
        isotope_symbol = zaid_to_symbol(zaid)
        matrix_type_label = "Covariance" if matrix_type_normalized == 'cov' else "Correlation"
        if is_diagonal:
            if len(mts) == 1:
                label = f"{isotope_symbol} MT:{mts[0]} {matrix_type_label}"
            else:
                label = f"{isotope_symbol} {matrix_type_label} Matrix"
        else:
            label = f"{isotope_symbol} MT:{mts[0]} vs MT:{mts[1]} {matrix_type_label}"
        
        # 8. Create and return CovarianceHeatmapData
        heatmap_data = CovarianceHeatmapData(
            matrix_data=M_full,
            matrix_type=matrix_type_normalized,
            zaid=zaid,
            block_info=block_info,
            uncertainty_data=uncertainty_data,
            energy_grid=energy_grid,
            mt_labels=[str(m) for m in mts],
            is_diagonal=is_diagonal,
            mask_value=mask_value,
            scale=scale_normalized,
            x_edges=x_edges,
            y_edges=y_edges,
            extent=extent,
            label=label
        )
        # Apply any extra heatmap kwargs onto the dataclass or metadata
        for key, val in kwargs.items():
            if key == "mask_color":
                continue  # mask color is fixed to lightgray
            if hasattr(heatmap_data, key):
                setattr(heatmap_data, key, val)
            else:
                heatmap_data.metadata[key] = val
        return heatmap_data

    def _to_heatmap_data_multi_isotope(
        self,
        zaids: List[int],
        mt: Union[int, Sequence[int], Tuple[int, int]],
        matrix_type: str,
        scale: str,
        energy_range: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> 'CovarianceHeatmapData':
        """
        Internal method to prepare multi-isotope covariance heatmap data.

        This method handles the case where multiple isotopes are provided,
        creating a flattened multiblock heatmap showing cross-isotope correlations.

        Parameters
        ----------
        zaids : List[int]
            List of isotope ZAIDs
        mt : int, sequence of int, or tuple
            MT reaction number(s)
        matrix_type : str
            Normalized matrix type ('corr' or 'cov')
        scale : str
            Normalized scale ('log' or 'linear')
        energy_range : tuple of float, optional
            Energy window (emin, emax)
        **kwargs
            Additional parameters

        Returns
        -------
        CovarianceHeatmapData
            Heatmap data object for multi-isotope visualization
        """
        from kika.plotting.plot_data import CovarianceHeatmapData
        from kika._utils import zaid_to_symbol

        # Filter to include only the specified isotopes
        iso_cov = self.filter_by_isotopes(zaids)
        if iso_cov.num_groups == 0:
            raise ValueError(f"No data found for isotopes {zaids}")

        if iso_cov.energy_grid is None:
            raise ValueError("Covariance matrix missing energy grid information for plotting.")

        # Parse MT input
        if isinstance(mt, tuple) and len(mt) == 2:
            # Off-diagonal specification - not supported for multi-isotope
            raise ValueError(
                "Off-diagonal MT specification (row_mt, col_mt) is not supported for multi-isotope heatmaps. "
                "Use a list of MTs instead."
            )

        # Normalize to list of MTs
        if isinstance(mt, int):
            mts = [mt]
        else:
            mts = sorted(list(mt))

        # Build ordered list of (zaid, mt) blocks
        # Sort by (zaid, mt) to ensure consistent ordering
        blocks = []
        for z in sorted(zaids):
            for m in mts:
                # Check if this (zaid, mt) pair exists in the data
                pairs = iso_cov._get_param_pairs()
                if (z, m) in pairs:
                    blocks.append((z, m))

        if not blocks:
            raise ValueError(f"No data found for the specified isotopes and MTs")

        # Energy grid handling - convert to eV for consistency with MF34
        G = iso_cov.num_groups
        edges_raw_full = np.asarray(iso_cov.energy_grid, dtype=float)

        # Convert to eV if energy is in different units
        energy_unit = getattr(self, 'energy_unit', 'eV')
        if energy_unit.lower() == 'mev':
            edges_raw_full = edges_raw_full * 1e6  # MeV to eV

        # Handle energy range cropping
        if energy_range is not None:
            emin, emax = energy_range
            if not (np.isfinite(emin) and np.isfinite(emax)) or emin >= emax:
                raise ValueError("energy_range must be a tuple (emin, emax) with emin < emax.")
            keep_mask = (edges_raw_full[1:] > float(emin)) & (edges_raw_full[:-1] < float(emax))
            if not np.any(keep_mask):
                raise ValueError("energy_range removed all groups; nothing to plot.")
            first, last = np.where(keep_mask)[0][[0, -1]]
            edges_cropped = edges_raw_full[first:last + 2]
        else:
            keep_mask = np.ones(G, dtype=bool)
            edges_cropped = edges_raw_full

        G_cropped = len(edges_cropped) - 1

        # Transform edges based on scale
        if scale == "log":
            transformed_edges = np.log10(np.maximum(edges_cropped, 1e-300).astype(float))
        else:
            transformed_edges = edges_cropped.astype(float)

        width = transformed_edges[-1] - transformed_edges[0]

        # Build the multiblock matrix
        n_blocks = len(blocks)
        matrix_size = n_blocks * G_cropped
        M_full = np.zeros((matrix_size, matrix_size))

        # Get full matrices
        pairs = iso_cov._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(pairs)}

        if matrix_type == 'corr':
            full_matrix = iso_cov.clipped_correlation_matrix
            mask_value = 0.0
        else:
            full_matrix = iso_cov.covariance_matrix
            mask_value = None

        # Fill in the multiblock matrix
        for i, (z_row, mt_row) in enumerate(blocks):
            for j, (z_col, mt_col) in enumerate(blocks):
                # Get indices in the full correlation/covariance matrix
                row_pair_idx = idx_map.get((z_row, mt_row))
                col_pair_idx = idx_map.get((z_col, mt_col))

                if row_pair_idx is None or col_pair_idx is None:
                    continue

                # Extract subblock (with energy cropping)
                row_start_full = row_pair_idx * iso_cov.num_groups
                col_start_full = col_pair_idx * iso_cov.num_groups
                row_indices_full = list(range(row_start_full, row_start_full + iso_cov.num_groups))
                col_indices_full = list(range(col_start_full, col_start_full + iso_cov.num_groups))
                row_indices = [row_indices_full[k] for k, keep in enumerate(keep_mask) if keep]
                col_indices = [col_indices_full[k] for k, keep in enumerate(keep_mask) if keep]

                subblock = full_matrix[np.ix_(row_indices, col_indices)]

                # Place in multiblock matrix
                row_start = i * G_cropped
                col_start = j * G_cropped
                M_full[row_start:row_start + G_cropped, col_start:col_start + G_cropped] = subblock

        # Build coordinate edges
        x_parts = []
        energy_ranges = {}
        for i, block in enumerate(blocks):
            start = i * width
            block_edges = (transformed_edges - transformed_edges[0]) + start
            x_parts.append(block_edges if i == 0 else block_edges[1:])
            energy_ranges[block] = (block_edges[0], block_edges[-1])

        x_edges = np.concatenate(x_parts) if x_parts else None
        y_edges = x_edges.copy() if x_edges is not None else None
        extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])) if x_edges is not None else None

        # Build block_info with multi-isotope structure
        block_info = {
            'blocks': blocks,  # List of (zaid, mt) tuples
            'zaids': sorted(zaids),
            'mts': mts,
            'G': G_cropped,
            'ranges': {block: (i * G_cropped, (i + 1) * G_cropped) for i, block in enumerate(blocks)},
            'energy_ranges': energy_ranges,
            'is_multi_isotope': True,
        }

        # Compute uncertainties keyed by (zaid, mt) tuples
        uncertainty_data = {}
        cov_matrix = iso_cov.covariance_matrix

        for block in blocks:
            z, m = block
            pair_idx = idx_map.get((z, m))
            if pair_idx is None:
                continue

            base_indices_full = list(range(pair_idx * iso_cov.num_groups, (pair_idx + 1) * iso_cov.num_groups))
            mt_rows = [base_indices_full[k] for k, keep in enumerate(keep_mask) if keep]

            diag_variance = np.diag(cov_matrix)[mt_rows]

            # Use cross sections if available
            if (z, m) in self.cross_sections:
                nominal_xs_full = np.asarray(self.cross_sections[(z, m)], dtype=float)
                nominal_xs = nominal_xs_full[keep_mask] if nominal_xs_full.size == iso_cov.num_groups else nominal_xs_full

                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma_percent = np.sqrt(np.abs(diag_variance)) / np.abs(nominal_xs) * 100.0
                    sigma_percent = np.nan_to_num(sigma_percent, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                sigma_percent = np.sqrt(np.abs(diag_variance)) * 100.0

            uncertainty_data[block] = sigma_percent

        if not uncertainty_data:
            uncertainty_data = None

        # Generate labels in "Symbol-MT#" format
        mt_labels = [f"{zaid_to_symbol(z)}-MT{m}" for (z, m) in blocks]

        # No default title - user can set via set_labels() on HeatmapBuilder
        label = None

        # Create and return CovarianceHeatmapData
        heatmap_data = CovarianceHeatmapData(
            matrix_data=M_full,
            matrix_type=matrix_type,
            zaid=zaids,  # Now stores list of ZAIDs
            block_info=block_info,
            uncertainty_data=uncertainty_data,
            energy_grid=edges_cropped,
            mt_labels=mt_labels,
            is_diagonal=True,
            mask_value=mask_value,
            scale=scale,
            x_edges=x_edges,
            y_edges=y_edges,
            extent=extent,
            label=label
        )

        # Apply any extra heatmap kwargs
        for key, val in kwargs.items():
            if key == "mask_color":
                continue
            if hasattr(heatmap_data, key):
                setattr(heatmap_data, key, val)
            else:
                heatmap_data.metadata[key] = val

        return heatmap_data

    def plot_covariance_heatmap(
        self,
        nuclide: Union[int, str],
        mt: Union[int, Sequence[int], Tuple[int, int]],
        ax: plt.Axes = None,
        *,
        matrix_type: str = "corr",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_family: str = "serif",
        vmax: float = None,
        vmin: float = None,
        show_uncertainties: bool = True,
        scale: str = "log",
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = "default",
        **imshow_kwargs
    ) -> plt.Figure:
        """
        Draw a covariance or correlation matrix heatmap for a specified isotope and MT reaction(s).
        
        This method now uses the modern PlotBuilder-based implementation from
        kika.plotting.covariance for cleaner, more maintainable code.

        Parameters
        ----------
        nuclide : int or str
            Isotope identifier. Can be either:
            - Integer ZAID (e.g., 92235 for U-235)
            - Element-mass string (e.g., 'U235', 'Fe56')
        mt : int, sequence of int, or tuple of (row_mt, col_mt)
            MT reaction number(s). Can be:
            - Single int: diagonal block for that MT
            - Sequence of ints: diagonal blocks for those MTs  
            - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
        ax : plt.Axes, optional
            Matplotlib axes to draw into (deprecated, kept for compatibility)
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
        scale : str, default "log"
            Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
        energy_range : tuple of float, optional
            Energy range (min, max) for filtering. Values in eV.
        title : str or None, default "default"
            Plot title. If "default", auto-generates from nuclide and MT.
            If a string, uses that as the title. If None, suppresses the title.
        **imshow_kwargs
            Additional arguments passed to imshow (deprecated)

        Returns
        -------
        plt.Figure
            The matplotlib figure containing the heatmap and optional uncertainty plots
        """
        from kika.plotting.covariance import plot_covariance_heatmap as _plot_covariance_heatmap
        
        return _plot_covariance_heatmap(
            covmat=self,
            nuclide=nuclide,
            mt=mt,
            matrix_type=matrix_type,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            vmax=vmax,
            vmin=vmin,
            show_uncertainties=show_uncertainties,
            scale=scale,
            energy_range=energy_range,
            title=title,
        )
    
    
    # ------------------------------------------------------------------
    # Decomposition methods
    # ------------------------------------------------------------------

    def cholesky_decomposition(
        self,
        *,
        space: str = "log",
        psd_method: str = "auto",
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger=None,
    ) -> np.ndarray:
        """Robust Cholesky factor L such that M ≈ L Lᵀ."""
        from kika.cov.decomposition import cholesky_decomposition
        return cholesky_decomposition(
            self,
            space=space,
            psd_method=psd_method,
            jitter_scale=jitter_scale,
            max_jitter_ratio=max_jitter_ratio,
            verbose=verbose,
            logger=logger,
        )

    def eigen_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        psd_method: Optional[str] = None,
        verbose: bool = True,
        logger=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition with PSD correction."""
        from kika.cov.decomposition import eigen_decomposition
        return eigen_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            psd_method=psd_method,
            verbose=verbose,
            logger=logger,
        )

    def svd_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        psd_method: Optional[str] = None,
        verbose: bool = True,
        full_matrices: bool = False,
        logger=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD with PSD pre-processing."""
        from kika.cov.decomposition import svd_decomposition
        return svd_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            psd_method=psd_method,
            verbose=verbose,
            full_matrices=full_matrices,
            logger=logger,
        )

    

    
    

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the CrossSectionCovariance object.
        
        Returns
        -------
        str
            String representation with content summary
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Covariance Matrix Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of covariance matrix data
        description = (
            "This object contains covariance matrix data from nuclear data files (SCALE, NJOY, etc).\n"
            "Each matrix represents the covariance between cross sections for specific\n"
            "isotope-reaction pairs across energy groups.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3 # -3 for spacing and formatting
        
        info_table = "Covariance Data Summary:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add summary information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Groups", self.num_groups, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Covariance Matrices", self.num_matrices, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Isotopes", len(self.isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.reactions), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".unique_isotopes": "Get set of unique isotope IDs",
            ".unique_reactions": "Get set of unique reaction MT numbers",
            ".num_matrices": "Get total number of covariance matrices",
            ".num_groups": "Get number of energy groups"
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
            ".get_matrix(...)": "Get specific covariance matrix",
            ".get_isotope_reactions()": "Get mapping of isotopes to their reactions",
            ".get_reactions_summary()": "Get DataFrame of isotopes with their reactions",
            ".get_isotope_covariance_matrix(...)": "Build combined covariance matrix for an isotope",
            ".to_dataframe()": "Convert all covariance data to DataFrame",
            ".save_excel()": "Save covariance data to Excel file"
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

    def _get_param_pairs(self) -> List[Tuple[int,int]]:
        """
        Return a list of all (isotope, reaction) pairs present,
        sorted first by isotope number (ascending), then by reaction number (ascending).
        """
        pairs = set(zip(self.isotope_rows, self.reaction_rows)) \
              | set(zip(self.isotope_cols, self.reaction_cols))
        # explicit sort by isotope then reaction
        return sorted(pairs, key=lambda p: (p[0], p[1]))

    def _autofix_covariance(
        self,
        *,
        accept_tol: float = -1.0e-4,
        max_steps: int = 40,
        verbose: bool = True,
        remove_all: bool = False,
        high_val_thresh: float = 5.0,
        logger = None, 
    ) -> Tuple["CrossSectionCovariance", Dict[str, Any]]:
        """
        Iteratively remove block pairs or reactions until the
        matrix is positive-(semi)definite.

        New rules
        ─────────
        1. **Do not drop blocks before the eigen-analysis.**
        2. Before each eigen-analysis, *count* how many entries
        in every block exceed `high_val_thresh`, and accumulate
        that count **per reaction MT**.
        3. After eigen-analysis, quantify each negative-mode
        contribution exactly as before (`pair_scores`).
        • Boost any (ra, rb) by the *combined* high-value count
            of `ra` and `rb`.  
        • Reaction MT 2 is protected: if MT 2 ties with any
            other reaction for "worst", keep MT 2 and drop the
            other one.
        """
        if len(self.isotopes) != 1:
            raise ValueError("auto_fix_covariance works only for single-isotope matrices.")

        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)

        current: "CrossSectionCovariance" = self
        removed: List[Tuple[int, int]] = []
        removed_mts: List[int] = []  # Track individual MTs removed in "hard" mode
        removed_correlations: List[Tuple[int, int]] = []  # Track off-diagonal block removals
        iso = self.isotope_rows[0] if self.isotope_rows else None
        separator = "-" * 60

        if verbose:
            _log_message(f"\n[COV] [AUTOFIX]")
            _log_message(f"  Checking covariance matrix for isotope {iso}")
            _log_message(f"{separator}")

        for step in range(1, max_steps + 1):
            M = current.covariance_matrix
            G = self.num_groups
            param_pairs = current._get_param_pairs()     # (iso, MT)

            # ──────────────────────────────────────────────────────
            # 1 Count large-magnitude entries by reaction
            # ──────────────────────────────────────────────────────
            high_cnt_per_rxn: Dict[int, int] = defaultdict(int)
            high_cnt_per_pair: Dict[Tuple[int, int], int] = {}

            for a, (_, ra) in enumerate(param_pairs):
                for b, (_, rb) in enumerate(param_pairs):
                    block = M[a*G:(a+1)*G, b*G:(b+1)*G]
                    cnt = int(np.sum(np.abs(block) > high_val_thresh))
                    if cnt:
                        high_cnt_per_pair[(ra, rb)] = cnt
                        high_cnt_per_rxn[ra] += cnt
                        high_cnt_per_rxn[rb] += cnt

            # ──────────────────────────────────────────────────────
            # 2 Eigen-analysis
            # ──────────────────────────────────────────────────────
            # numpy's default driver (dsyevd) sometimes silently returns NaN
            # eigenvalues on rank-deficient block-covariance matrices instead
            # of raising. Treat NaN as a failure too, and fall back to scipy's
            # 'evr' (dsyevr, RRR) driver, which is more numerically robust.
            eigvals, eigvecs = None, None
            _eigh_err: Exception = RuntimeError("no solver attempted")
            for _backend in ("numpy", "scipy_evr", "scipy_ev"):
                try:
                    if _backend == "numpy":
                        _ev, _vc = np.linalg.eigh(M)
                    else:
                        from scipy.linalg import eigh as _sp_eigh
                        _ev, _vc = _sp_eigh(M, driver=_backend.split("_")[1])
                    if np.all(np.isfinite(_ev)) and np.all(np.isfinite(_vc)):
                        eigvals, eigvecs = _ev, _vc
                        break
                    _eigh_err = RuntimeError(f"{_backend} eigh returned non-finite values")
                except Exception as e:
                    _eigh_err = e
            if eigvals is None:
                # Cannot compute eigenvectors → cannot pick worst block to
                # remove. Bail out cleanly; pipeline proceeds with downstream
                # PSD repair (variance cap, Higham, eigen-clip).
                if verbose:
                    _log_message(
                        f"[COV] [AUTOFIX] [WARNING] eigendecomposition failed at step {step:02d} "
                        f"({type(_eigh_err).__name__}: {_eigh_err}); aborting removal loop"
                    )
                return current, {
                    "iterations": step,
                    "min_eigenvalue": _safe_min_eigvalsh(M),
                    "converged": False,
                    "removed_pairs": removed,
                    "removed_mts": removed_mts,
                    "removed_correlations": removed_correlations,
                    "lapack_failed": True,
                }
            min_ev = float(eigvals.min())
            if verbose:
                _log_message(f"[COV] [AUTOFIX] [STEP {step:02d}] Smallest eigenvalue: {min_ev:.4e}")

            if min_ev >= accept_tol:
                _log_message(f"[COV] [AUTOFIX] [SUCCESS] Matrix accepted (λ_min={min_ev:.4e} >= {accept_tol:.4e})")
                _log_message(separator)
                return current, {
                    "iterations": step,
                    "min_eigenvalue": min_ev,
                    "converged": True,  # Fix: This should be True when eigenvalue is acceptable
                    "removed_pairs": removed,
                    "removed_mts": removed_mts,
                    "removed_correlations": removed_correlations,
                }

            # ──────────────────────────────────────────────────────
            # 3 Score negative-mode contributions (unchanged core)
            # ──────────────────────────────────────────────────────
            pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
            neg_idxs = np.where(eigvals < accept_tol)[0]

            for idx in neg_idxs:
                v = eigvecs[:, idx]
                for a, (_, ra) in enumerate(param_pairs):
                    v_a = v[a*G:(a+1)*G]
                    for b, (_, rb) in enumerate(param_pairs):
                        block = M[a*G:(a+1)*G, b*G:(b+1)*G]
                        contrib = float(v_a @ block @ v[b*G:(b+1)*G])
                        if contrib < accept_tol:
                            pair_scores[(ra, rb)] += abs(contrib)

            # ──────────────────────────────────────────────────────
            # 4 Combine scores with high-value information
            #    (higher counts ⇒ stronger penalty)
            # ──────────────────────────────────────────────────────
            boosted_scores: Dict[Tuple[int, int], float] = {}
            for (ra, rb), base in pair_scores.items():
                boost = high_cnt_per_rxn.get(ra, 0) + high_cnt_per_rxn.get(rb, 0)
                boosted_scores[(ra, rb)] = base * (1.0 + boost)

            if not boosted_scores:
                if verbose:
                    _log_message(f"[COV] [AUTOFIX] [WARNING] No negative contributions found below threshold {accept_tol:.4e}")
                    _log_message(f"[COV] [AUTOFIX] [ACTION] Using fallback: remove block with largest absolute eigenvalue contribution")
                
                # Fallback: find the block with the largest absolute contribution to ANY negative eigenvalue
                fallback_scores: Dict[Tuple[int, int], float] = defaultdict(float)
                for idx in neg_idxs:
                    v = eigvecs[:, idx]
                    for a, (_, ra) in enumerate(param_pairs):
                        v_a = v[a*G:(a+1)*G]
                        for b, (_, rb) in enumerate(param_pairs):
                            block = M[a*G:(a+1)*G, b*G:(b+1)*G]
                            contrib = float(v_a @ block @ v[b*G:(b+1)*G])
                            fallback_scores[(ra, rb)] += abs(contrib)  # Use absolute value
                
                if fallback_scores:
                    boosted_scores = fallback_scores
                else:
                    # Ultimate fallback: remove largest diagonal variance
                    if verbose:
                        _log_message(f"[COV] [AUTOFIX] [WARNING] No eigenvalue contributions found - using diagonal variance fallback")
                    diag_scores = {}
                    for a, (_, ra) in enumerate(param_pairs):
                        block = M[a*G:(a+1)*G, a*G:(a+1)*G]
                        diag_scores[(ra, ra)] = float(np.sum(np.abs(np.diag(block))))
                    boosted_scores = diag_scores if diag_scores else {(param_pairs[0][1], param_pairs[0][1]): 1.0}

            worst_pair = max(boosted_scores, key=boosted_scores.get)
            ra, rb = worst_pair

            # tie-break in favour of MT 2
            if 2 in worst_pair:
                ra, rb = (rb, ra) if ra == 2 else (ra, rb)

            if remove_all:
                # choose which single reaction to drop
                r_drop = ra if boosted_scores.get((ra, ra), 0) >= boosted_scores.get((rb, rb), 0) else rb
                if r_drop == 2 and r_drop != (ra if ra != 2 else rb):
                    r_drop = rb if r_drop == ra else ra        # keep MT 2
                if verbose:
                    _log_message(f"  [STEP {step:02d}] [ACTION] Removing all blocks for reaction MT = {r_drop}")
                current = current.remove_matrix(isotope=iso,
                                                reaction_pairs=[(r_drop, 0)],
                                                exceptions=[])
                removed.append((r_drop, r_drop))
                removed_mts.append(r_drop)
            else:
                if verbose:
                    _log_message(f"  [STEP {step:02d}] [ACTION] Removing block pair = {(ra, rb)}")
                current = current.remove_matrix(isotope=iso,
                                                reaction_pairs=[(ra, rb), (rb, ra)],
                                                exceptions=[])
                removed.append((ra, rb))
                # For medium level, track when diagonal blocks are removed vs off-diagonal
                if ra == rb:
                    removed_mts.append(ra)
                else:
                    # This is an off-diagonal block removal (correlation removal)
                    removed_correlations.append((ra, rb))

        # ----------------------------------------------------------------------
        # Not converged within max_steps
        # ----------------------------------------------------------------------
        if verbose:
            _log_message(f"[COV] [AUTOFIX] [ERROR] Reached limit of {max_steps} steps without convergence")
        # Directly compute min eigenvalue instead of using analyze_covariance
        min_eigenvalue = _safe_min_eigvalsh(current.covariance_matrix)

        logg = {
            "steps": max_steps,
            "min_eigenvalue": min_eigenvalue,
            "removed_pairs": removed,
            "removed_mts": removed_mts,
            "removed_correlations": removed_correlations,
            "converged": False,
        }

        if verbose:
            _log_message(f"  [SUMMARY]")
            _log_message(f"    Pairs removed: {logg['removed_pairs']}")
            _log_message(f"    MTs removed:   {logg['removed_mts']}")
            _log_message(f"    Correlations removed: {logg['removed_correlations']}")
            _log_message(f"    Final smallest eigenvalue: {logg['min_eigenvalue']:.4e}")
            _log_message(f"{separator}")

        return current, logg

    def _clamp_covariance(
        self,
        *,
        high_val_thresh: float = 5.0,
        clamp_target: float = 1.0,
        accept_tol: float = -1.0e-4,
        max_iter: int = 5,
        verbose: bool = True,
        clamp_detail: bool = False,
        logger = None,  # Optional logger for file output
    ) -> Tuple["CrossSectionCovariance", Dict[str, Any]]:
        """
        Cap diagonal variances larger than `high_val_thresh` to `clamp_target`
        and rescale the connected row/column by sqrt(target/|old|) so that
        correlations remain untouched.

        verbose=True prints the [CLAMP #N] header and per-clamp adjusted-count
        summary. clamp_detail=True additionally dumps every off-diagonal
        before/after pair (very noisy on large matrices — opt in only when
        debugging a specific clamp event).
        """
        if self.num_groups == 0:
            raise ValueError("num_groups is zero.")

        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)

        current = self.copy()
        G = self.num_groups
        param_pairs = current._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}
        separator = "-" * 60

        for iter_num in range(1, max_iter + 1):
            M = current.covariance_matrix
            changed_any = False
            clamped_count = 0

            # ──────────────────────────────────────────────────────────
            # 1 Clamp any |variance| > high_val_thresh → +1.0
            # ──────────────────────────────────────────────────────────
            
            _log_message(f"  [ITERATION {iter_num:02d}] Scanning variance values")
            
            for a, (iso, mt) in enumerate(param_pairs):
                r0 = a * G
                for g in range(G):
                    idx = r0 + g
                    old_var = M[idx, idx]
                    if abs(old_var) <= high_val_thresh:
                        continue

                    new_var = clamp_target
                    M[idx, idx] = new_var
                    changed_any = True
                    clamped_count += 1

                    _log_message(f"    [CLAMP #{clamped_count}] MT={mt:>3} G={g:>2} "
                        f"variance {old_var:.6g} → {new_var:.6g}")

                    # keep correlations
                    scale = np.sqrt(abs(new_var / old_var))
                    row_before = M[idx, :].copy()
                    col_before = M[:, idx].copy()

                    M[idx, :] *= scale
                    M[:, idx] *= scale
                    M[idx, idx] = new_var

                    diff_row = np.abs(row_before - M[idx, :]) > 0
                    diff_col = np.abs(col_before - M[:, idx]) > 0
                    diff_total = (np.count_nonzero(diff_row) +
                                np.count_nonzero(diff_col) - 1)

                    if clamp_detail and diff_total:
                        for j in np.where(diff_row)[0]:
                            if j == idx:
                                continue
                            pp_j = param_pairs[j // G]
                            mt_j = pp_j[1]
                            g_j = j % G
                            _log_message(
                                f"      cov(MT={mt:>3}, G={g:>2}; "
                                f"MT={mt_j:>3}, G={g_j+1:>2}) "
                                f"{row_before[j]:12.4e} → {M[idx, j]:12.4e}"
                            )
                        for i in np.where(diff_col)[0]:
                            if i == idx:
                                continue
                            pp_i = param_pairs[i // G]
                            mt_i = pp_i[1]
                            g_i = i % G
                            _log_message(
                                f"      cov(MT={mt_i:>3}, G={g_i:>2}; "
                                f"MT={mt:>3}, G={g:>2}) "
                                f"{col_before[i]:12.4e} → {M[i, idx]:12.4e}"
                            )

                    if verbose:
                        _log_message(f"      {diff_total} covariances adjusted")

            if not changed_any:
                _log_message(f"  [ITERATION {iter_num:02d}] No variances above threshold; stopping clamping")
                break
            
            _log_message(f"  [ITERATION {iter_num:02d}] Clamped {clamped_count} variance values")

            # push edits back in the block structure  (unchanged)
            for ir, rr, ic, rc, mat_ref in zip(
                current.isotope_rows,
                current.reaction_rows,
                current.isotope_cols,
                current.reaction_cols,
                current.matrices,
            ):
                i = idx_map[(ir, rr)]
                j = idx_map[(ic, rc)]
                mat_ref[:, :] = M[i*G:(i+1)*G, j*G:(j+1)*G]

            min_ev = _safe_min_eigvalsh(M)
            _log_message(f"  [ITERATION {iter_num:02d}] Smallest eigenvalue: {min_ev:.4e}")

            if min_ev >= accept_tol:
                _log_message(f"  [SUCCESS] Matrix accepted (λ_min = {min_ev:.4e} >= {accept_tol:.4e})")
                _log_message(separator)
                return current, {
                    "iterations": iter_num,
                    "min_eigenvalue": min_ev,
                    "converged": True,  # Fix: This should be True when eigenvalue is acceptable
                    "clamped_values": clamped_count
                }
        
            # clamping not enough – fall back to removal strategy if autofix is True
            _log_message(f"[COV] [CLAMP] [WARNING] After clamping, smallest eigenvalue ({min_ev:.4e}) still below threshold ({accept_tol:.4e})")
        
        # Return the clamped matrix and log after clamping - Fix: Check final eigenvalue here too
        final_M = current.covariance_matrix
        min_ev_final = _safe_min_eigvalsh(final_M)
        
        # Check if final result is actually acceptable
        converged = min_ev_final >= accept_tol
        
        log = {
            "iterations": iter_num,
            "min_eigenvalue": min_ev_final,
            "converged": converged,  # Fix: Set based on actual final eigenvalue check
            "used_fallback": False,
            "clamp_iter": iter_num,
        }
        return current, log

    def _extract_xs_data(
        self,
        zaid: int,
        mt: int,
        label: str = None,
        sigma: float = 1.0,
        **styling_kwargs
    ):
        """Extract cross section data."""
        from kika.plotting import MultigroupCrossSectionPlotData
        from kika._utils import zaid_to_symbol
        
        # Check if cross section data exists
        if (zaid, mt) not in self.cross_sections:
            # Cross sections not available, return None
            return None
        
        # Get cross section data
        xs_data = self.cross_sections[(zaid, mt)]
        G = int(self.num_groups)
        
        if len(xs_data) != G:
            raise ValueError(
                f"Cross section data length ({len(xs_data)}) does not match num_groups ({G})"
            )
        
        # Get energy boundaries
        energy_bins = None
        if hasattr(self, "energy_grid") and self.energy_grid is not None:
            eg = np.asarray(self.energy_grid, dtype=float)
            if eg.size == G + 1:
                energy_bins = eg
            elif eg.size == G:
                # Infer boundaries from centers
                diffs = np.diff(eg)
                left0 = eg[0] - diffs[0] / 2.0
                rightN = eg[-1] + diffs[-1] / 2.0
                mid = (eg[:-1] + eg[1:]) / 2.0
                energy_bins = np.concatenate([[max(left0, 1e-5)], mid, [rightN]])
        
        # For step plots, we need G+1 points (repeat the last value)
        if energy_bins is not None and len(energy_bins) == G + 1:
            x = energy_bins
            y = np.r_[xs_data, xs_data[-1]]
        else:
            # Fallback to group indices
            x = np.arange(G + 1, dtype=float)
            y = np.r_[xs_data, xs_data[-1]]
            energy_bins = None
        
        # Check if uncertainty data exists for this (zaid, mt) pair
        has_uncertainty = False
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows,
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == zaid and mt_r == mt and iso_c == zaid and mt_c == mt:
                has_uncertainty = True
                break
        
        # Generate label if not provided
        if label is None:
            isotope_symbol = zaid_to_symbol(zaid)
            reaction_name = MT_TO_REACTION.get(mt, "")
            if reaction_name:
                label = f"{isotope_symbol} MT={mt} {reaction_name}"
            else:
                label = f"{isotope_symbol} MT={mt}"
            
            # Add sigma notation if uncertainties are available
            if has_uncertainty:
                sigma_suffix = f" (±{sigma}σ)" if sigma != 1.0 else " (±1σ)"
                label += sigma_suffix
        
        # Create and return the plot data
        return MultigroupCrossSectionPlotData(
            x=x,
            y=y,
            label=label,
            zaid=zaid,
            mt=mt,
            energy_bins=energy_bins,
            **styling_kwargs
        )






    # ------------------------------------------------------------------
    # Graph / UI helper methods (covariance connectivity)
    # ------------------------------------------------------------------

    def _invalidate_cov_graph_cache(self) -> None:
        """Invalidate any cached covariance connectivity graphs."""
        if hasattr(self, "_cov_graph_cache"):
            self._cov_graph_cache = {}

    @staticmethod
    def _node_id(zaid: int, mt: int) -> str:
        """Stable string id for frontend graphs."""
        from kika._utils import zaid_to_symbol
        nuclide = zaid_to_symbol(int(zaid))
        return f"{nuclide}:{int(mt)}"

    def _normalize_nuclide(self, nuclide: Union[int, str]) -> int:
        """Accept ZAID int or symbol string like 'Fe56'."""
        if isinstance(nuclide, str):
            from kika._utils import symbol_to_zaid
            return int(symbol_to_zaid(nuclide))
        return int(nuclide)

    def _node_info(self, zaid: int, mt: int, *, degree: Optional[int] = None) -> Dict[str, Any]:
        """JSON-friendly node payload."""
        # Reaction label is stable from constants; isotope label is best-effort.
        from kika._utils import zaid_to_symbol
        
        reaction_label = MT_TO_REACTION.get(int(mt), f"MT={int(mt)}")
        nuclide = zaid_to_symbol(int(zaid))

        out = {
            "id": self._node_id(zaid, mt),
            "nuclide": nuclide,  # Use nuclide (e.g., "Fe56") instead of ZAID
            "mt": int(mt),
            "reaction_label": reaction_label,
        }
        if degree is not None:
            out["degree"] = int(degree)
        return out

    def _cov_adjacency(
        self,
        *,
        require_nonzero: bool = True,
        tol: float = 0.0,
    ) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """
        Build (or fetch cached) undirected adjacency mapping:
            (zaid, mt) -> set of connected (zaid, mt)

        Connectivity rule:
        - If a covariance block exists between the two nodes AND
          (require_nonzero => the block has any |value| > tol),
          then we add an edge.
        """
        key = (bool(require_nonzero), float(tol))

        cache: Dict[Tuple[bool, float], Dict[Tuple[int, int], Set[Tuple[int, int]]]] = getattr(self, "_cov_graph_cache", {})
        if key in cache:
            return cache[key]

        adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)

        for ir, rr, ic, rc, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices,
        ):
            a = (int(ir), int(rr))
            b = (int(ic), int(rc))

            if require_nonzero:
                # "At least one covariance element" -> any non-zero entry in the block
                # tol lets you ignore numerical noise.
                if not np.any(np.abs(M) > tol):
                    continue

            if a == b:
                # ensure node exists even if only diagonal blocks exist
                _ = adj[a]
                continue

            adj[a].add(b)
            adj[b].add(a)

        # Save cache
        cache[key] = adj
        self._cov_graph_cache = cache
        return adj

    def list_cov_nodes(
        self,
        *,
        require_nonzero: bool = False,
        tol: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return all nodes (zaid, mt) known to the covariance store.
        If require_nonzero=True, nodes are restricted to those that participate
        in at least one non-zero block (or have a non-zero diagonal block).
        """
        adj = self._cov_adjacency(require_nonzero=require_nonzero, tol=tol)

        # If require_nonzero=False, adjacency might not include isolated nodes.
        # In that case, fall back to the param-pair registry.
        if not require_nonzero:
            pairs = self._get_param_pairs()
            nodes = sorted({(int(z), int(m)) for (z, m) in pairs}, key=lambda x: (x[0], x[1]))
            return [self._node_info(z, m, degree=len(adj.get((z, m), set()))) for z, m in nodes]

        nodes = sorted(adj.keys(), key=lambda x: (x[0], x[1]))
        return [self._node_info(z, m, degree=len(adj.get((z, m), set()))) for z, m in nodes]

    def get_cov_connections(
        self,
        nuclide: Union[int, str],
        mt: int,
        *,
        depth: int = 1,
        require_nonzero: bool = True,
        tol: float = 0.0,
        max_neighbors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main method for your UI graph.

        Returns a JSON-friendly payload with:
        - center node
        - nodes + edges within BFS depth
        - (if depth >= 2) a grouped mapping of second-level nodes per first-level neighbor,
          ideal for your "mini nodes next to each neighbor".

        Parameters
        ----------
        depth : int
            1 -> show center + its neighbors
            2 -> also include neighbors-of-neighbors
        require_nonzero : bool
            True: only create edges if the covariance block has any |value| > tol
            False: edge exists if the block exists (even if all zeros)
        tol : float
            Threshold for nonzero detection when require_nonzero=True
        max_neighbors : Optional[int]
            If set, cap the number of neighbors expanded per node (safety for huge graphs)
        """
        zaid = self._normalize_nuclide(nuclide)
        mt = int(mt)
        start = (zaid, mt)

        adj = self._cov_adjacency(require_nonzero=require_nonzero, tol=tol)

        # Validate node existence (prefer using registry of pairs)
        pairs = set((int(z), int(m)) for (z, m) in self._get_param_pairs())
        if start not in pairs and start not in adj:
            # helpful error for UI
            available = sorted({m for (z, m) in pairs if z == zaid})
            raise ValueError(f"(ZAID={zaid}, MT={mt}) not found. Available MTs for this ZAID: {available}")

        # BFS up to depth
        visited: Set[Tuple[int, int]] = {start}
        edges: Set[Tuple[str, str]] = set()

        layer1: List[Tuple[int, int]] = []
        layer2_by_neighbor: Dict[str, List[Tuple[int, int]]] = {}

        frontier = [start]
        for d in range(depth):
            next_frontier: List[Tuple[int, int]] = []
            for u in frontier:
                nbrs = list(adj.get(u, set()))
                # deterministic ordering: same result each call
                nbrs.sort(key=lambda x: (x[0], x[1]))
                if max_neighbors is not None:
                    nbrs = nbrs[: int(max_neighbors)]

                for v in nbrs:
                    su = self._node_id(*u)
                    sv = self._node_id(*v)
                    a, b = (su, sv) if su < sv else (sv, su)
                    edges.add((a, b))

                    if v not in visited:
                        visited.add(v)
                        next_frontier.append(v)

                # Capture layer1 explicitly
                if d == 0:
                    layer1 = nbrs

            frontier = next_frontier
            if not frontier:
                break

        # Build the grouped second-level map (for your “mini nodes next to each neighbor”)
        if depth >= 2:
            for n in layer1:
                n_id = self._node_id(*n)
                second = sorted(adj.get(n, set()) - {start}, key=lambda x: (x[0], x[1]))
                if max_neighbors is not None:
                    second = second[: int(max_neighbors)]
                layer2_by_neighbor[n_id] = second

        # Serialize nodes + edges
        all_nodes = sorted(visited, key=lambda x: (x[0], x[1]))
        nodes_payload = [
            self._node_info(z, m, degree=len(adj.get((z, m), set())))
            for (z, m) in all_nodes
        ]
        edges_payload = [{"source": a, "target": b} for (a, b) in sorted(edges)]

        payload: Dict[str, Any] = {
            "center": self._node_info(*start, degree=len(adj.get(start, set()))),
            "nodes": nodes_payload,
            "edges": edges_payload,
            "depth": int(depth),
            "require_nonzero": bool(require_nonzero),
            "tol": float(tol),
            "layer1": [self._node_id(*p) for p in layer1],
        }

        if depth >= 2:
            payload["layer2_by_neighbor"] = {
                nid: [self._node_id(*p) for p in plist]
                for nid, plist in layer2_by_neighbor.items()
            }

        return payload





    #------------------------------------------------------------------
    # Methods and properties related to correlation matrix (Unused)
    #------------------------------------------------------------------

    def report_large_values(
        self,
        threshold: float = 1.0,
        top_n: int = 30,
        return_text: bool = False
        ) -> Optional[Tuple[str, dict]]:
        """
        Scan each block of the relative covariance matrices and generate a detailed report
        of entries that exceed the specified threshold.

        If return_text is True, returns a tuple:
        (report_text, summary_dict)
        where summary_dict contains:
        - zaid: the main isotope ZAID
        - name: human-readable symbol
        - count: number of entries > threshold
        - max_value: largest flagged value
        """
        G = self.num_groups
        if G == 0:
            if return_text:
                return None
            print("No energy groups available. Cannot generate report.")
            return None

        total_checked = 0
        total_flagged = 0
        max_value = 0.0
        large_values = []

        # map MT to reaction names
        reaction_names = {mt: MT_TO_REACTION.get(mt, f"MT={mt}")
                        for mt in self.reactions}

        if self.isotope_rows:
            raw_zaid = self.isotope_rows[0]
            try:
                zaid_int = int(raw_zaid)
            except (TypeError, ValueError):
                # if it’s not parseable, just pass it through
                zaid_int = raw_zaid
        else:
            zaid_int = 0

        from kika._utils import zaid_to_symbol
        isotope_name = zaid_to_symbol(zaid_int)
        
        # scan each block
        for iso_r, rxn_r, iso_c, rxn_c, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        ):
            for ig in range(G):
                for jg in range(G):
                    total_checked += 1
                    val = M[ig, jg]
                    if val > threshold:
                        total_flagged += 1
                        if val > max_value:
                            max_value = val

                        entry = {
                            'value': val,
                            'iso_row': iso_r,
                            'rxn_row': rxn_r,
                            'rxn_name_row': reaction_names[rxn_r],
                            'iso_col': iso_c,
                            'rxn_col': rxn_c,
                            'rxn_name_col': reaction_names[rxn_c],
                            'row_idx': ig,
                            'col_idx': jg
                        }
                        # add energy ranges if available
                        if (self.energy_grid is not None
                            and len(self.energy_grid) >= G+1):
                            entry.update({
                                'e_row_low':  self.energy_grid[ig],
                                'e_row_high': self.energy_grid[ig+1],
                                'e_col_low':  self.energy_grid[jg],
                                'e_col_high': self.energy_grid[jg+1]
                            })
                        large_values.append(entry)

        # nothing found?
        if total_flagged == 0:
            if return_text:
                return None
            print(f"No entries > {threshold:.4e} for {isotope_name} (ZAID:{zaid_int}).")
            return None

        # sort & truncate
        large_values.sort(key=lambda x: x['value'], reverse=True)
        truncated = False
        if top_n is not None and len(large_values) > top_n:
            large_values = large_values[:top_n]
            truncated = True

        # build detailed report
        lines = []
        lines.append("\n" + "="*100)
        lines.append(f"LARGE VALUES REPORT FOR {isotope_name} (ZAID:{zaid_int}) > {threshold:.4e}")
        lines.append("="*100)
        lines.append("\nSUMMARY:")
        lines.append(f"  Total elements checked:     {total_checked:,}")
        lines.append(f"  Elements exceeding threshold:{total_flagged:,} "
                    f"({total_flagged/total_checked*100:.2f}%)")
        lines.append(f"  Maximum value found:        {max_value:.4e}")
        lines.append("\nDETAILED REPORT:")
        if truncated:
            lines.append(f"  (Showing top {top_n} entries)")

        # table header
        header = (
            f"{'#':>4} | {'Value':>10} | "
            f"{'Row Block':^20} | {'Col Block':^20} | "
            f"{'R#':>3} | {'C#':>3} | "
            f"{'Energy Row':^17} | {'Energy Col':^17}"
        )
        lines.append("\n" + "-"*len(header))
        lines.append(header)
        lines.append("-"*len(header))

        for i, e in enumerate(large_values, start=1):
            row_blk = f"{e['iso_row']},{e['rxn_row']}({e['rxn_name_row'][:6]})"
            col_blk = f"{e['iso_col']},{e['rxn_col']}({e['rxn_name_col'][:6]})"
            val_str = f"{e['value']:.4e}"
            energy_row = energy_col = ""
            if 'e_row_low' in e:
                energy_row = f"{e['e_row_low']:.4e}-{e['e_row_high']:.4e}"
                energy_col = f"{e['e_col_low']:.4e}-{e['e_col_high']:.4e}"
            line = (
                f"{i:>4} | {val_str:>10} | "
                f"{row_blk:<20} | {col_blk:<20} | "
                f"{e['row_idx']:>3} | {e['col_idx']:>3} | "
                f"{energy_row:<17} | {energy_col:<17}"
            )
            lines.append(line)

        lines.append("-"*len(header))
        if truncated:
            extra = total_flagged - top_n
            lines.append(f"Note: {extra:,} more entries not shown.")
        lines.append("\n" + "="*100)

        report_text = "\n".join(lines)

        if return_text:
            summary = {
                'zaid': zaid_int,
                'name': isotope_name,
                'count': total_flagged,
                'max_value': max_value
            }
            return report_text, summary
        else:
            print(report_text)
            return None

    def verify_correlation(self, atol: float = 1e-12, rtol: float = 1e-4) -> None:
        """Run basic checks on the correlation matrix and print any problems.

        Checks:
        1. Symmetry: ρ_ij == ρ_ji within *atol*.
        2. Diagonal consistency: 1 if variance > 0, else 0.
        3. Range: off‑diagonal in [‑1, 1].
        """
        R = self.correlation_matrix
        G = self.num_groups
        param_pairs = self._get_param_pairs()
        N = len(param_pairs)

        # Build helper lookup to turn a flat index into (ZAID, MT, group)
        def flat_to_components(idx: int) -> Tuple[int, int, int]:
            p_idx, g = divmod(idx, G)
            zaid, mt = param_pairs[p_idx]
            return zaid, mt, g + 1  # +1 for 1‑based group number

        # ------------------------------------------------------------------
        # 1. Symmetry
        # ------------------------------------------------------------------
        diff = np.abs(R - R.T)
        bad = np.argwhere(diff > atol)
        for i, j in bad:
            val_ij = R[i, j]
            val_ji = R[j, i]
            zr, mr, gr = flat_to_components(i)
            zc, mc, gc = flat_to_components(j)
            print(
                f"Asymmetry: ρ_ij={val_ij:.4e} but ρ_ji={val_ji:.4e} for "
                f"({zr},{mr}) ({zc},{mc}) groups {gr} {gc}."
            )

        # ------------------------------------------------------------------
        # 2 & 3. Diagonal and range
        # ------------------------------------------------------------------
        for i in range(R.shape[0]):
            zr, mr, gr = flat_to_components(i)
            diag_val = R[i, i]

            # Diagonal rule
            if np.isclose(diag_val, 0.0, atol=atol):
                # Variance must be zero → entire row/col should be zero
                row_nonzero = np.argwhere(~np.isclose(R[i, :], 0.0, atol=atol))
                col_nonzero = np.argwhere(~np.isclose(R[:, i], 0.0, atol=atol))
                offenders = set(row_nonzero.flatten()) | set(col_nonzero.flatten())
                offenders.discard(i)
                for j in offenders:
                    zc, mc, gc = flat_to_components(j)
                    val = R[i, j]
                    print(
                        f"Found value {val:.4e} for ({zr},{mr}) ({zc},{mc}) energy group {gr}. "
                        "Value should be 0 for correlation from a 0 variance component."
                    )
            else:
                if not np.isclose(diag_val, 1.0, atol=atol, rtol=rtol):
                    print(
                        f"Found value {diag_val:.4e} for ({zr},{mr}) ({zr},{mr}) energy group {gr}. "
                        "Value should be 1 for a diagonal element."
                    )

        # Off‑diagonal range check
        off_diag_indices = np.argwhere(~np.eye(R.shape[0], dtype=bool))
        for i, j in off_diag_indices:
            val = R[i, j]
            if val < -1 - rtol or val > 1 + rtol:
                zr, mr, gr = flat_to_components(i)
                zc, mc, gc = flat_to_components(j)
                print(
                    f"Found value {val:.4e} for ({zr},{mr}) ({zc},{mc}) energy group {gr}. "
                    "Value should be [-1,1] for an off-diagonal element."
                )

    def sanitize_by_correlation(
        self,
        *,
        max_abs_corr: float = 1.0,
        zero_threshold: float = 1.5,      # any |ρᵢⱼ| > zero_threshold → 0
        report_tol: float = 1e-6,
        eigen_floor: float = 1e-12,
        project_psd: bool = True,
        psd_method: str = "auto",
        verbose: bool = True,
    ):
        """
        Clip out-of-range correlations, zero out huge outliers, inspect eigenvalues,
        and (optionally) project to PSD.

        Parameters
        ----------
        max_abs_corr
            Any correlation with |ρ| > max_abs_corr (but ≤ zero_threshold)
            is clipped to ±max_abs_corr.
        zero_threshold
            Any |ρ| > zero_threshold is set to 0.
        report_tol
            Minimum |old–new| change to actually log.
        eigen_floor
            Floor for eigenvalues when projecting to PSD.
        project_psd
            Whether to do the PSD projection here.
        verbose
            Print detailed diagnostics.
        """

        # --- build and symmetrize covariance C0 ---
        C0 = (self.covariance_matrix + self.covariance_matrix.T) * 0.5
        var = np.diag(C0)
        std = np.sqrt(var)
        D = np.outer(std, std)

        # --- form correlation R ---
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.divide(C0, D, out=np.zeros_like(C0), where=D>0)

        # --- prepare for logging ---
        p = R.shape[0]
        G = self.num_groups
        pairs = self._get_param_pairs()
        eye = np.eye(p, dtype=bool)

        # --- find zero-out entries ---
        too_big   = np.abs(R) > zero_threshold
        off_diag  = too_big & ~eye
        # only unique pairs in upper triangle
        tri_mask  = np.triu(np.ones_like(off_diag, dtype=bool), k=1)
        off_pairs = np.argwhere(off_diag & tri_mask)
        total_hits= int(too_big.sum())

        if verbose:
            print("[COV] [CORRELATION] Starting correlation-based sanitization")
            if off_pairs.size:
                print(f"[COV] [CORRELATION] Hard-zeroed {len(off_pairs)} off-diagonal pairs (total {total_hits} entries including symmetry):")
                for i, j in off_pairs:
                    bi, gi = divmod(i, G)
                    bj, gj = divmod(j, G)
                    iso_i, rxn_i = pairs[bi]
                    iso_j, rxn_j = pairs[bj]
                    old = R[i, j]
                    print(f"  ({iso_i},{rxn_i})-g{gi+1:02d} ↔ ({iso_j},{rxn_j})-g{gj+1:02d}   {old:+.3e} → +0.000e+00")
            else:
                print("[COV] [CORRELATION] No entries exceeded zero_threshold")

        # --- apply the zeroing ---
        R_clipped = R.copy()
        R_clipped[too_big] = 0.0

        # --- now clip to ±max_abs_corr ---
        to_clip    = (np.abs(R_clipped) > max_abs_corr) & ~too_big
        clip_pairs = np.argwhere(to_clip & tri_mask)

        if verbose and clip_pairs.size:
            print(f"[COV] [CORRELATION] Clipped {len(clip_pairs)} off-diagonal pairs to ±{max_abs_corr}:")
            for i, j in clip_pairs:
                bi, gi = divmod(i, G)
                bj, gj = divmod(j, G)
                iso_i, rxn_i = pairs[bi]
                iso_j, rxn_j = pairs[bj]
                old = R_clipped[i, j]
                new = np.sign(old) * max_abs_corr
                if abs(old - new) > report_tol:
                    print(f"  ({iso_i},{rxn_i})-g{gi+1:02d} ↔ ({iso_j},{rxn_j})-g{gj+1:02d}   {old:+.3e} → {new:+.3e}")
        R_clipped[to_clip] = np.sign(R_clipped[to_clip]) * max_abs_corr

        # --- enforce exact 1’s on diagonal (or 0 if var==0) ---
        R_clipped[eye & (var != 0.0)] = 1.0
        R_clipped[eye & (var == 0.0)] = 0.0

        # --- rebuild covariance and symmetrize ---
        C1 = R_clipped * D
        C1[D == 0.0] = 0.0
        C1 = (C1 + C1.T) * 0.5

        # --- eigen before PSD ---
        w1, V1 = np.linalg.eigh(C1)
        if verbose:
            print("[COV] [CORRELATION] Smallest 5 eigenvalues AFTER clip:", " ".join(f"{x:+.3e}" for x in w1[:5]))

        # --- optional PSD projection ---
        if project_psd:
            from kika.cov.decomposition import _validate_psd_method, _PSD_AUTO_THRESHOLD
            _validate_psd_method(psd_method)
            neg_count = int((w1 < 0).sum())

            # Resolve "auto" using the already-computed eigendecomposition
            if psd_method == "auto":
                lam_max = max(float(w1.max()), 1e-300)
                lam_min_neg = max(-float(w1.min()), 0.0)
                ratio = lam_min_neg / lam_max
                psd_method = "clip" if ratio < _PSD_AUTO_THRESHOLD else "higham"
                if verbose:
                    print(
                        f"[COV] [CORRELATION] PSD auto: ratio={ratio:.3e} "
                        f"(thresh={_PSD_AUTO_THRESHOLD:.0e}) -> {psd_method}"
                    )

            if psd_method == "higham" and neg_count > 0:
                from kika.cov.decomposition import nearest_psd_higham
                if verbose:
                    print(f"[COV] [CORRELATION] PSD-proj (Higham): {neg_count} eigenvalues < 0")
                C2, _psd_info = nearest_psd_higham(
                    C1, preserve_diagonal=True, eigval_floor=eigen_floor,
                    verbose=verbose,
                )
            else:
                # "clip", "none", or "higham" with no negatives → eigenvalue floor
                if neg_count and verbose:
                    print(f"[COV] [CORRELATION] PSD-proj: {neg_count} eigenvalues < 0, floored to {eigen_floor:.1e}")
                w2 = np.maximum(w1, eigen_floor)
                C2 = V1 @ np.diag(w2) @ V1.T
                C2 = (C2 + C2.T) * 0.5
                if verbose:
                    print("[COV] [CORRELATION] Smallest 5 eigenvalues AFTER PSD:", " ".join(f"{x:+.3e}" for x in w2[:5]))
        else:
            C2 = C1

        # --- scatter back into blocks and return ---
        out = self.copy()
        out._scatter_full_into_blocks(C2)
        return out
    
    def _scatter_full_into_blocks(self, full_matrix: np.ndarray) -> None:
        """
        Scatter a full covariance matrix back into the individual block matrices.
        
        Parameters
        ----------
        full_matrix : np.ndarray
            Full covariance matrix of shape (N·G, N·G) where N is the number of 
            parameter pairs and G is num_groups.
        """
        param_pairs = self._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}
        G = self.num_groups

        # Update each matrix block with the corresponding section from the full matrix
        for i, (ir, rr, ic, rc, mat_ref) in enumerate(zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices,
        )):
            row_idx = idx_map[(ir, rr)]
            col_idx = idx_map[(ic, rc)]
            r0, r1 = row_idx * G, (row_idx + 1) * G
            c0, c1 = col_idx * G, (col_idx + 1) * G
            
            # Update the matrix in place
            mat_ref[:, :] = full_matrix[r0:r1, c0:c1]


CovMat = CrossSectionCovariance  # backward compat
