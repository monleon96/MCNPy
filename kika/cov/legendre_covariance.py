import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kika._utils import create_repr_section


def _endf_mfmt_of(data) -> Optional[Tuple[int, int]]:
    """``(MF, MT)`` from a ``DataLink``'s ``ENDF_MFMT``, either separator.

    ``"34/2"`` and ``"34,2"`` both give ``(34, 2)``. GNDS §25.2.3 defines the
    attribute with a comma and every distributed covariance file writes one;
    kika's ENDF adapter writes a slash. Parsing only the slash made this
    module's MF34 filter return an empty result — **without raising** — for any
    suite decoded from GNDS. See ``docs/gnds_endf_conflicts.md`` §3.1 and §7.1.

    Duck-typed, like everything else this module reads off a suite, so it works
    on a real ``DataLink`` and on anything that quacks like one.
    """
    raw = getattr(data, 'ENDF_MFMT', None) if data is not None else None
    if not raw:
        return None
    try:
        mf, mt = (int(part) for part in str(raw).replace('/', ',').split(',', 1))
    except ValueError:
        return None
    return mf, mt


def _is_endf_mf(data, mf: int) -> bool:
    """Does this link belong to *mf*? The separator-agnostic ``startswith``."""
    parts = _endf_mfmt_of(data)
    return parts is not None and parts[0] == mf


def _endf_mt_of(data) -> int:
    """The MT half of a ``DataLink``'s ``ENDF_MFMT``, e.g. ``"34/2"`` -> 2."""
    parts = _endf_mfmt_of(data)
    if parts is None:
        raise ValueError(
            f"a covariance link whose ENDF_MFMT is "
            f"{getattr(data, 'ENDF_MFMT', None)!r} names no MT"
        )
    return parts[1]

if TYPE_CHECKING:
    from pathlib import Path
    from kika.plotting.plot_data import LegendreCoeffPlotData, LegendreHeatmapData, LegendreUncertaintyPlotData
    from kika.endf.classes.mf34.mf34 import MF34MT


@dataclass
class LegendreCovariance:
    """
    Format-agnostic covariance of Legendre expansion coefficients.

    The canonical fields (matrices, energy grids, isotope/reaction/L tags,
    frame, is_relative) hold the data in a representation that is shared
    across source formats.  Format-specific extras travel in two metadata
    dictionaries so that round-trips back to the same format produce
    a compatible (and where possible identical) section.

    Canonical fields
    ----------------
    isotope_rows, reaction_rows, l_rows : List[int]
        Row tags (ZA, MT, Legendre order) per matrix.
    isotope_cols, reaction_cols, l_cols : List[int]
        Column tags per matrix.
    energy_grids : List[List[float]]
        Energy boundaries for each matrix (NE points → matrix is (NE-1)x(NE-1)
        or rectangular for cross-L blocks).
    matrices : List[np.ndarray]
        Covariance matrix per (isotope, reaction, L) row/col tuple.
    is_relative : List[bool]
        ``True`` when the matrix stores relative covariance, ``False`` for
        absolute (LB=0).
    frame : List[str]
        Reference frame: ``"same-as-MF4"`` (LCT=0), ``"LAB"`` (LCT=1),
        ``"CM"`` (LCT=2), or ``"unknown LCT=…"``.
    energy_unit : str
        ``"eV"`` (default) or ``"MeV"``.
    legendre_coefficients : Dict[(isotope, mt, l), np.ndarray]
        Optional cell-averaged nominal coefficients on the matrix grids.

    Format-specific metadata
    ------------------------
    metadata : Dict[str, Any]
        Whole-object extras.  Conventional keys:

        - ``"source_format"``: ``"endf"``, ``"covfil"``, ``"gendf"``, …
        - ``"source_path"``: original file path (string)
        - any other format-specific defaults the user wants to round-trip.

    mt_metadata : Dict[(isotope, mt), Dict[str, Any]]
        Per-section header data.  ENDF MF34 keys:

        - ``"za"``: ZA = 1000*Z + A (float)
        - ``"awr"``: atomic-weight ratio
        - ``"mat"``: ENDF MAT number
        - ``"ltt"``: Legendre representation flag (1 = a_1…, 2 = a_0…).

        Populated by :meth:`from_endf` / :meth:`MF34MT.to_ang_covmat` and
        consulted by :meth:`to_mf34` to fill the MF34 header.  Caller
        overrides on :meth:`to_mf34` always win.
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
    # Whole-object metadata: file-level scalars propagated by the I/O routines
    # (COVFIL, COVERX, …) alongside format-specific extras. Well-known keys:
    # 'awr' (atomic weight ratio), 'temperature' (K), 'source_format',
    # 'source_path'. See the class docstring for the conventional set. New
    # keys can be added without changing the class.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Nominal Legendre coefficients keyed by (isotope, reaction_mt, l_order)
    legendre_coefficients: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    # Per-(isotope, reaction_mt) MF34 header data so a LegendreCovariance
    # loaded from ENDF can be written back via :meth:`to_mf34`.
    mt_metadata: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)

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
    def from_covariance_suite(cls, suite, nuclide: int = 0,
                              energy_unit: str = 'eV') -> "LegendreCovariance":
        """Every MF34 section of a GNDS ``covarianceSuite``, as one carrier.

        The covariance half of what phase 4 set out to do: the multigroup
        collapse can be driven from a ``reactionSuite`` on both sides now — a_l
        through :func:`~kika.cov.multigroup.collapse.legendre_source_from_model`,
        and the covariance through here.

        The model is treated as one more source format, on the same footing as
        GENDF, COVFIL, BOXER and COVERX, exactly as
        :meth:`~kika.cov.cross_section_covariance.CrossSectionCovariance.from_covariance_section`
        already does. **Duck-typed rather than imported**, and for the same
        reason: ``kika.cov`` does not depend on the model, and phase 4's P4 had
        just finished removing the last import-time reason it might.

        A *suite* rather than a section, unlike the cross-section twin, because
        the two are keyed differently. There, several MF35 bands share one MT and
        would be indistinguishable inside one object. Here the key is
        ``(isotope, MT, l)`` and every MF34 section carries a distinct one, so a
        whole file assembles without collision — which is what the collapse
        needs, since it walks the blocks looking for ``(l_row, l_col)`` pairs.

        Sections whose ``rowData.ENDF_MFMT`` is not ``34/…`` are skipped, so an
        MF31 or MF33 section living in the same suite is left alone rather than
        swept in as an angular block.

        Parameters
        ----------
        suite
            A ``CovarianceSuite``: anything iterable of sections, or exposing
            ``covarianceSections``. Each section needs ``form.matrix``,
            ``form.rowGrid``, ``form.isRelative`` and ``rowData.ENDF_MFMT``,
            with ``rowData.legendreOrder`` giving l.
        nuclide : int, optional
            ZAID to file the matrices under, used when
            ``section.provenance.za`` is absent. The model identifies a
            covariance by href rather than by a material header.
        energy_unit : str, optional
            Unit of ``form.rowGrid``. The model keeps ENDF's native eV.

        Notes
        -----
        ⚠ **The model states two grids and this carrier stores one.** Where a
        section's row and column grids differ, the model is the one telling the
        truth and this conversion loses the distinction — the same asymmetry
        ``kika.sampling.model_blocks._mf34_entries`` documents from the sampling
        side. A warning is raised rather than the difference being absorbed
        silently. No MF34 seen so far states two.
        """
        result = cls()
        result.energy_unit = energy_unit

        for section in getattr(suite, 'covarianceSections', suite):
            rowData = getattr(section, 'rowData', None)
            if not _is_endf_mf(rowData, 34):
                continue
            colData = getattr(section, 'columnData', None) or rowData

            form = getattr(section, 'form', None)
            if form is None or getattr(form, 'matrix', None) is None:
                continue

            row_grid = np.asarray(form.rowGrid, dtype=float)
            col_grid = getattr(form, 'columnGrid', None)
            if col_grid is not None and not np.array_equal(
                row_grid, np.asarray(col_grid, dtype=float)
            ):
                warnings.warn(
                    f"MF34 section {getattr(section, 'label', '?')!r} states "
                    f"different row and column energy grids; LegendreCovariance "
                    f"stores one per matrix, so the column grid is dropped here",
                    stacklevel=2,
                )

            za = int(getattr(getattr(section, 'provenance', None), 'za', 0) or nuclide)
            result.add_matrix(
                isotope_row=za,
                reaction_row=_endf_mt_of(rowData),
                l_row=int(rowData.legendreOrder),
                isotope_col=za,
                reaction_col=_endf_mt_of(colData),
                l_col=int(colData.legendreOrder),
                matrix=np.asarray(form.matrix, dtype=float),
                energy_grid=list(row_grid),
                is_relative=bool(form.isRelative),
                frame=getattr(form, 'productFrame', None) or 'same-as-MF4',
            )
        return result

    @classmethod
    def from_endf(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "LegendreCovariance":
        """
        Create a LegendreCovariance instance from an ENDF file containing MF34 data.

        This is a convenience class method that reads an ENDF file, extracts MF34
        (angular distribution covariance) data, and converts it to a LegendreCovariance object.

        Parameters
        ----------
        file_path : str or Path
            Path to the ENDF file containing MF34 data
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        LegendreCovariance
            LegendreCovariance instance with angular distribution covariance data

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file does not contain MF34 data

        Examples
        --------
        >>> mf34_covmat = LegendreCovariance.from_endf('path/to/endf_file.txt')
        >>> print(f"Loaded {mf34_covmat.num_matrices} angular covariance matrices")
        >>> # Or specify MeV if needed
        >>> mf34_covmat_mev = LegendreCovariance.from_endf('path/to/file.txt', energy_unit='MeV')

        Notes
        -----
        This method internally:
        1. Reads the ENDF file using read_endf()
        2. Extracts MF34 data
        3. Converts to LegendreCovariance using the to_ang_covmat() method
        
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

        # Convert MF34 to LegendreCovariance
        lc = mf34.to_ang_covmat(energy_unit=energy_unit)
        lc.metadata.setdefault("source_format", "endf")
        lc.metadata.setdefault("source_path", str(file_path))
        return lc

    @classmethod
    def from_covfil(cls, file_path: Union[str, 'Path'], energy_unit: str = 'eV') -> "LegendreCovariance":
        """
        Create a LegendreCovariance instance from an NJOY-generated COVFIL/GENDF file.

        Parameters
        ----------
        file_path : str or Path
            Path to the COVFIL/GENDF covariance file
        energy_unit : str, optional
            Energy unit for the energy grid: 'eV' (default) or 'MeV'

        Returns
        -------
        LegendreCovariance
            LegendreCovariance instance loaded from the file

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
                "File contains MF33 data. Use CrossSectionCovariance.from_covfil() instead."
            )
        result.metadata.setdefault("source_format", "covfil")
        result.metadata.setdefault("source_path", str(file_path))
        return result

    def to_covfil(self, file_path: Union[str, 'Path'], tape_label: str = '', temperature: float = 0.0) -> None:
        """
        Write this LegendreCovariance to an NJOY COVFIL/GENDF text file.

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

    def to_mf34(
        self,
        isotope: int,
        mt: int,
        *,
        za: Optional[float] = None,
        awr: Optional[float] = None,
        mat: Optional[int] = None,
        ltt: Optional[int] = None,
        frame: Optional[str] = None,
    ) -> "MF34MT":
        """Build an MF34MT from the entries tagged ``(isotope, mt)``.

        Matrices with ``isotope_row == isotope`` and ``reaction_row == mt`` are
        grouped by ``reaction_col`` (MT1) into one subsection each, and within
        a subsection by ``(L, L1)`` into one sub-subsection per pair.  For
        symmetric self-correlation (MT1 == MT) only the upper triangle
        ``L <= L1`` is emitted.  Diagonal blocks (L == L1) are written as LB=5
        and off-diagonal blocks as LB=6.

        Parameters
        ----------
        isotope, mt : int
            Selector keys; matrices with matching row tags are included.
        za, awr, mat, ltt : optional
            Override the corresponding MF34 header fields.  When omitted,
            values are taken from ``self.mt_metadata[(isotope, mt)]`` if
            available, then from sensible defaults (``za=isotope``,
            ``awr=1.0``, ``mat=0``, ``ltt`` inferred — 2 if any L=0 pair is
            present, else 1).
        frame : optional
            Override the per-matrix frame (LCT) for every emitted
            sub-subsection.  Accepts ``"same-as-MF4"``, ``"LAB"``, or ``"CM"``.

        Returns
        -------
        MF34MT

        Notes
        -----
        Multiple LIST records (NI > 1) per sub-subsection in the original
        file are collapsed to a single LB=5/LB=6 record on the matrix's
        stored grid; the numeric content is preserved but the original
        per-record split is not.
        """
        from kika.endf.classes.mf34.mf34 import MF34MT, Subsection, SubSubsection
        from kika.endf.writers.mf34_writer import _make_lb5_record, _make_lb6_record

        indices = [
            i for i in range(self.num_matrices)
            if self.isotope_rows[i] == isotope and self.reaction_rows[i] == mt
        ]
        if not indices:
            raise ValueError(
                f"No matrices found for isotope={isotope}, MT={mt}"
            )

        md = self.mt_metadata.get((isotope, mt), {})

        resolved_za = float(za if za is not None else md.get('za', float(isotope)))
        resolved_awr = float(awr if awr is not None else md.get('awr', 1.0))
        resolved_mat = int(mat if mat is not None else md.get('mat', 0) or 0)

        has_l0 = any(self.l_rows[i] == 0 or self.l_cols[i] == 0 for i in indices)
        if ltt is not None:
            resolved_ltt = int(ltt)
        elif md.get('ltt') is not None:
            resolved_ltt = int(md['ltt'])
        else:
            resolved_ltt = 2 if has_l0 else 1

        mf34 = MF34MT(number=mt)
        mf34._za = resolved_za
        mf34._awr = resolved_awr
        mf34._mat = resolved_mat
        mf34._ltt = resolved_ltt
        mf34._mf = 34

        # Group by MT1 (reaction_col)
        by_mt1: Dict[int, List[int]] = {}
        for i in indices:
            by_mt1.setdefault(int(self.reaction_cols[i]), []).append(i)

        frame_to_lct = {"same-as-MF4": 0, "LAB": 1, "CM": 2}

        for mt1 in sorted(by_mt1.keys()):
            mt1_indices = by_mt1[mt1]
            is_self = (mt1 == mt)

            ls = [self.l_rows[i] for i in mt1_indices]
            l1s = [self.l_cols[i] for i in mt1_indices]
            nl = max(ls) if ls else 1
            nl1 = max(l1s) if l1s else 1

            if is_self:
                mt1_indices = [
                    i for i in mt1_indices
                    if self.l_rows[i] <= self.l_cols[i]
                ]

            subsection = Subsection(mt1=mt1, nl=nl, nl1=nl1, mat1=0.0)

            sorted_indices = sorted(
                mt1_indices, key=lambda i: (self.l_rows[i], self.l_cols[i])
            )
            for i in sorted_indices:
                l = int(self.l_rows[i])
                l1 = int(self.l_cols[i])
                matrix = np.asarray(self.matrices[i], dtype=float)
                grid = [float(e) for e in self.energy_grids[i]]

                f = frame if frame is not None else self.frame[i]
                lct = frame_to_lct.get(f, 0)

                if l == l1:
                    matrix = 0.5 * (matrix + matrix.T)
                    rec = _make_lb5_record(matrix, grid)
                else:
                    rec = _make_lb6_record(matrix, grid, grid)

                ss = SubSubsection(l=l, l1=l1, lct=lct, ni=1, records=[rec])
                subsection.sub_subsections.append(ss)

            mf34.add_subsection(subsection)

        mf34._nmt1 = len(mf34._subsections)
        return mf34

    def to_endf(
        self,
        source_endf: Union[str, 'Path'],
        output_path: Union[str, 'Path'],
        isotope: int,
        mt: int,
        *,
        replace_existing: bool = True,
        update_directory: bool = True,
        **mf34_overrides,
    ) -> str:
        """Write a single (isotope, MT) MF34 section into an ENDF file.

        Convenience wrapper that calls :meth:`to_mf34` to build the section
        and :func:`kika.endf.writers.write_mf34_to_file` to splice it into
        a template file.  Extra keyword arguments are forwarded to
        :meth:`to_mf34` (``za``, ``awr``, ``mat``, ``ltt``, ``frame``).
        """
        from kika.endf.writers.mf34_writer import write_mf34_to_file

        mf34 = self.to_mf34(isotope, mt, **mf34_overrides)
        return write_mf34_to_file(
            source_endf=str(source_endf),
            mf34=mf34,
            output_path=str(output_path),
            replace_existing=replace_existing,
            update_directory=update_directory,
        )

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
    
    def copy(self) -> "LegendreCovariance":
        """Return a deep copy of this object."""
        import copy as _copy
        return _copy.deepcopy(self)

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full covariance matrix.

        If :meth:`ensure_psd` has been called, the cached PSD matrix is
        returned directly instead of re-assembling from sub-matrices.

        Returns
        -------
        np.ndarray
            Full covariance matrix of shape (N*G_max, N*G_max)
        """
        if getattr(self, "_psd_covariance_cache", None) is not None:
            return self._psd_covariance_cache
        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}
        unions = getattr(self, "_union_grids", None) or self.compute_union_energy_grids()
        # number of bins (not points) per triplet on the union
        Gmap = {t: len(unions[t]) - 1 for t in param_triplets}
        max_G = max(Gmap.values()) if Gmap else 0
        N = len(param_triplets) * max_G
        full = np.zeros((N, N), dtype=float)

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
        cov_rel = self.covariance_matrix  # uses PSD cache if available
        Sigma_log = np.log1p(cov_rel)
        return Sigma_log

    def ensure_psd(
        self,
        *,
        preserve_diagonal: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-10,
        eigval_floor: float = 0.0,
        verbose: bool = True,
        logger=None,
    ) -> Tuple["LegendreCovariance", dict]:
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
        Tuple[LegendreCovariance, dict]
            (psd_copy, info) where *psd_copy* has the PSD cache set and
            *info* is the diagnostic dict from ``nearest_psd_higham``.
        """
        from kika.cov.decomposition import nearest_psd_higham

        full = self.covariance_matrix  # assembled from sub-matrices
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
    ) -> 'LegendreHeatmapData':
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
        LegendreHeatmapData
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
        from kika.plotting.plot_data import LegendreHeatmapData
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
        
        # 7. Create and return LegendreHeatmapData
        heatmap_data = LegendreHeatmapData(
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
        from kika.plotting.covariance import plot_mf34_uncertainties as _plot_unc

        return _plot_unc(
            mf34_covmat=self,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            ax=ax,
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
        tuple of (LegendreCoeffPlotData or None, LegendreUncertaintyPlotData)
            Tuple containing:
            - coeff_data: Legendre coefficient data if available in ``legendre_coefficients``, else None
            - unc_data: Uncertainty data for the Legendre coefficients
            
        Raises
        ------
        ValueError
            If uncertainty data is not available for the specified parameters
            
        Examples
        --------
        >>> # Extract uncertainty data from LegendreCovariance - both notation styles work
        >>> mf34_covmat = endf.mf[34].mt[2].to_ang_covmat()
        >>> coeff_data, unc_data = mf34_covmat.to_plot_data(nuclide=26056, mt=2, order=1)
        >>> # Note: coeff_data will be None (MF34 only has uncertainties, not values)
        >>> 
        >>> # Build a plot with just uncertainties
        >>> from kika.plotting import PlotBuilder
        >>> fig = PlotBuilder().add_data(unc_data).build()
        """
        from kika.plotting import LegendreCoeffPlotData, LegendreUncertaintyPlotData
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
            plot_type='step',
            **styling_kwargs
        )
        
        # Build LegendreCoeffPlotData if nominal coefficients are available
        coeff_data = None
        key = (isotope, mt, order)
        if key in self.legendre_coefficients:
            coeffs = np.asarray(self.legendre_coefficients[key], dtype=float)
            if energy_bins is not None and coeffs.size == energy_bins.size - 1:
                coeffs_extended = np.append(coeffs, coeffs[-1])
                coeff_label = f"{zaid_to_symbol(isotope)} - $a_{{{order}}}$"
                coeff_data = LegendreCoeffPlotData(
                    x=energy_bins,
                    y=coeffs_extended,
                    label=coeff_label,
                    order=order,
                    isotope=zaid_to_symbol(isotope),
                    mt=mt,
                    plot_type='step',
                    **styling_kwargs
                )
                coeff_data.step_where = 'post'

        return coeff_data, unc_data

    def filter_by_isotope_reaction(self, isotope: int, mt: int) -> "LegendreCovariance":
        """
        Return a new LegendreCovariance containing only matrices for the specified isotope and MT reaction.

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
        LegendreCovariance
            New LegendreCovariance object containing only the filtered matrices
        """
        # Find indices where both row and column match the specified isotope and MT
        matching_indices = []
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, 
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == isotope and mt_r == mt and iso_c == isotope and mt_c == mt:
                matching_indices.append(i)
        
        # Create new LegendreCovariance with filtered data
        filtered_mf34 = LegendreCovariance()
        
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

        # Propagate matching legendre_coefficients
        filtered_mf34.legendre_coefficients = {
            k: v for k, v in self.legendre_coefficients.items()
            if k[0] == isotope and k[1] == mt
        }

        return filtered_mf34

    def remap_mt(self, old_mt: int, new_mt: int) -> "LegendreCovariance":
        """
        Return a copy with all occurrences of *old_mt* replaced by *new_mt*.

        This is useful when NJOY GENDF files use different MT numbering
        (e.g. MT=251 for elastic scattering) that needs to be mapped back
        to the standard ENDF MT numbers (e.g. MT=2).

        Parameters
        ----------
        old_mt : int
            MT number to replace.
        new_mt : int
            MT number to use instead.

        Returns
        -------
        LegendreCovariance
            New instance with remapped MT numbers.
        """
        import copy

        new = copy.copy(self)
        new.reaction_rows = [new_mt if m == old_mt else m for m in self.reaction_rows]
        new.reaction_cols = [new_mt if m == old_mt else m for m in self.reaction_cols]

        # Remap keys in legendre_coefficients
        new.legendre_coefficients = {
            (iso, (new_mt if mt == old_mt else mt), l): vals
            for (iso, mt, l), vals in self.legendre_coefficients.items()
        }

        return new

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

    def get_ll_prime_correlations(
        self,
        isotope: int,
        mt: int,
    ) -> Dict[str, Any]:
        """
        Extract per-energy l-l' correlation blocks from ENDF MF34 data.

        For each energy bin, builds the L×L covariance sub-block from all
        stored (l_row, l_col) sub-matrices that share the same energy grid,
        then converts to correlation.

        Parameters
        ----------
        isotope : int
            Isotope ID (e.g. 2631 for Fe-56).
        mt : int
            Reaction MT number (e.g. 2 for elastic).

        Returns
        -------
        dict
            'energy_grid': np.ndarray — bin boundaries (N+1 points)
            'l_values': list of int — sorted Legendre orders present
            'correlations': list of np.ndarray — one L×L correlation matrix per energy bin
            'mean_abs_offdiag': float — mean |off-diagonal| across all bins
        """
        # Collect all sub-matrices matching (isotope, mt)
        blocks = {}  # (l_row, l_col) -> (energy_grid, matrix)
        for i, (ir, rr, lr, ic, rc, lc) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
        )):
            if ir == isotope and ic == isotope and rr == mt and rc == mt:
                blocks[(lr, lc)] = (self.energy_grids[i], self.matrices[i])

        if not blocks:
            return {'energy_grid': np.array([]), 'l_values': [],
                    'correlations': [], 'mean_abs_offdiag': 0.0}

        # Determine L values and energy grid from a diagonal block
        l_set = set()
        for (lr, lc) in blocks:
            l_set.add(lr)
            l_set.add(lc)
        l_values = sorted(l_set)
        l_idx = {l: i for i, l in enumerate(l_values)}
        n_l = len(l_values)

        # Use energy grid from first block (assumed uniform for same isotope/mt)
        ref_grid = list(blocks.values())[0][0]
        energy_grid = np.array(ref_grid)
        n_bins = len(energy_grid) - 1

        # Build per-bin L×L covariance and correlation
        correlations = []
        off_diag_all = []
        for g in range(n_bins):
            cov_block = np.zeros((n_l, n_l))
            for (lr, lc), (_, mat) in blocks.items():
                ir_l, ic_l = l_idx[lr], l_idx[lc]
                if g < mat.shape[0] and g < mat.shape[1]:
                    cov_block[ir_l, ic_l] = mat[g, g]
                    if ir_l != ic_l:
                        cov_block[ic_l, ir_l] = mat[g, g]

            # Convert to correlation
            std = np.sqrt(np.maximum(np.diag(cov_block), 0.0))
            std[std == 0] = 1.0
            corr = cov_block / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)
            correlations.append(corr)

            if n_l > 1:
                mask = ~np.eye(n_l, dtype=bool)
                off_diag_all.extend(np.abs(corr[mask]).tolist())

        mean_abs = float(np.mean(off_diag_all)) if off_diag_all else 0.0

        return {
            'energy_grid': energy_grid,
            'l_values': l_values,
            'correlations': correlations,
            'mean_abs_offdiag': mean_abs,
        }

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
        space: str = "log",
        psd_method: str = "auto",
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger=None,
    ) -> np.ndarray:
        """
        Robust Cholesky factor L such that M ≈ L L^T.

        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        psd_method : str
            PSD correction: "auto" (default; clip when negatives are tiny,
            else Higham), "higham", "clip", or "none" (jitter on Cholesky failure).
        jitter_scale : float
            Base jitter scale (only used when *psd_method="jitter"*).
        max_jitter_ratio : float
            Maximum jitter relative to matrix norm.
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
        """
        Eigendecomposition with PSD correction.

        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Deprecated. Use *psd_method* instead.
        psd_method : str or None
            "auto" (default), "higham", "clip", or "none".
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
        """
        SVD with PSD pre-processing.

        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Deprecated. Use *psd_method* instead.
        psd_method : str or None
            "auto" (default), "higham", "clip", or "none".
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
            psd_method=psd_method,
            verbose=verbose,
            full_matrices=full_matrices,
            logger=logger,
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
        Get a detailed string representation of the LegendreCovariance object.
        
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

        Constructs a mapping matrix ``A`` such that ``A @ Sigma @ A.T`` restates
        ``Sigma`` on the destination (union) grid. Row ``g`` selects the source
        bin containing destination bin ``g``; a source bin split by another
        section's boundaries becomes several destination bins carrying that
        bin's variance and perfectly correlated, which is what the file says.

        **A destination bin outside the source's stated range gets an all-zero
        row.** The destination is the union over every section mentioning a
        Legendre order, so it is a refinement of any one section's grid only
        when all of them share a range -- and real files do not. This method
        assumed otherwise until 2026-08-11 and was wrong at both ends
        (``docs/library-gaps.md`` D9 and D10):

        * **above** the source's last boundary the cursor ran past the end and
          raised ``IndexError``. The shipped Fe-56 ``_a0cross`` tape does this:
          its a0 blocks carry the MF33 cross term and so sit on the magnitude
          grid out to 150 MeV while the L>=1 blocks stop at 20 MeV, and 21 of
          its 28 sections are short of their own union.
        * **below** the source's first boundary the cursor sat at 0 and this
          wrote ``A[g, 0] = 1``, replicating the section's first bin downwards.
          Silent, and therefore the one that changed numbers: on the run-86
          multigroup tape ``L2xL6`` is stated from 846.8 keV while every other
          section starts at 1e-5 eV, so its covariance was being repeated across
          every bin beneath -- 10 590 entries, worst 2.07e-2 against diagonals
          of order 1e-1.

        Zero is right rather than merely safe. A section states a covariance
        over its own bins and says *nothing* about a bin outside them, and
        nothing is zero covariance -- not the value of the nearest bin it does
        cover, which is what clamping the cursor would assert.

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
        src_grid = np.asarray(src_grid, dtype=float)
        dst_grid = np.asarray(dst_grid, dtype=float)
        Gs, Gd = len(src_grid) - 1, len(dst_grid) - 1
        A = np.zeros((Gd, Gs), dtype=float)
        j = 0
        for g in range(Gd):
            eL, eH = dst_grid[g], dst_grid[g + 1]
            while j + 1 < len(src_grid) and src_grid[j + 1] <= eL + 1e-12:
                j += 1
            if j >= Gs or eH <= src_grid[0] + 1e-12 or eL >= src_grid[-1] - 1e-12:
                continue    # the section says nothing here; see D9/D10 above
            A[g, j] = 1.0
        return A


MF34CovMat = LegendreCovariance  # backward compat
