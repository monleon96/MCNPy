"""
Classes for MT sections within MF34 (Angular Distribution Covariances) in ENDF files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import time # Import time for performance checks

from ..mt import MT
from ....cov.legendre_covariance import LegendreCovariance 
from ....utils.logging_utils import get_endf_logger
import numpy as np # Make sure numpy is imported

# Initialize logger for this module
logger = get_endf_logger(__name__)

@dataclass
class SubSubsectionRecord:
    """Record representing a LIST within a sub-subsection"""
    ls: int = None       # Flag for symmetric matrix (1=yes, 0=no)
    lb: int = None       # Flag for covariance pattern
    nt: int = None       # Total number of items in the list
    ne: int = None       # Number of energy entries
    
    lt: int = None                    # for LB=0..2: LT lives in C3
    np: int = None                    # for LB=0..2: number of (E,F) pairs across table(s)

    # Preserve original LIST floats for round-trip in rare cases (e.g., LT>0)
    raw_list_values: List[float] = field(default_factory=list)

    # For LB=5 (original format)
    energies: List[float] = field(default_factory=list)   # Energy grid
    matrix: List[float] = field(default_factory=list)     # Fk,k' covariance matrix values
    
    # For LB=0-4
    e_table_k: List[float] = field(default_factory=list)  # Energies in first table
    f_table_k: List[float] = field(default_factory=list)  # F-values in first table
    e_table_l: List[float] = field(default_factory=list)  # Energies in second table (when LT > 0)
    f_table_l: List[float] = field(default_factory=list)  # F-values in second table (when LT > 0)
    
    # For LB=6 (rectangular matrix)
    row_energies: List[float] = field(default_factory=list)   # NER row energies
    col_energies: List[float] = field(default_factory=list)   # NEC column energies
    rect_matrix:   List[float] = field(default_factory=list)   # NER×NEC matrix values

@dataclass
class SubSubsection:
    """Sub-subsection for a particular (L,L1) pair"""
    l: int = None        # Index of Legendre coefficient for reaction MT
    l1: int = None       # Index of Legendre coefficient for reaction MT1
    lct: int = None      # Frame of reference flag
    ni: int = None       # Number of LIST records
    records: List[SubSubsectionRecord] = field(default_factory=list)  # LIST records in this sub-subsection

@dataclass
class Subsection:
    """Subsection for a particular MT1 reaction"""
    mt1: int = None      # 'Other' reaction type 
    nl: int = None       # Number of Legendre coefficients for MT
    nl1: int = None      # Number of Legendre coefficients for MT1
    mat1: float = None   # MAT1 value (normally 0.0)
    sub_subsections: List[SubSubsection] = field(default_factory=list)  # Sub-subsections for this subsection


@dataclass
class MF34MT(MT):
    """
    Class for MT sections within MF34 (Angular Distribution Covariances).
    
    This class stores covariance data for angular distributions.
    """
    _za: float = None
    _awr: float = None
    _ltt: int = None
    _nmt1: int = None
    _mat: int = None
    _mf: int = 34
    _subsections: List[Subsection] = field(default_factory=list)
    num_lines: int = 0

    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z+A)"""
        return self._za
    
    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio"""
        return self._awr
    
    @property
    def representation_flag(self) -> str:
        """Legendre representation used in MF4/MF34."""
        ltt = None if self._ltt is None else int(self._ltt)
        if ltt in (None, 0):
            # MF4 isotropic / unspecified case can propagate here; don't raise
            return "isotropic or unspecified (MF4 conventions)"
        if ltt == 1:
            return "Legendre coefficients starting with a1"
        if ltt == 2:
            return "Legendre coefficients starting with a0"
        if ltt == 3:
            return "either L or L1 may be 0"
        # Be tolerant: some files use extensions—surface as text instead of raising
        return f"unknown (LTT={ltt})"
    
    @property
    def num_subsections(self) -> int:
        """Number of subsections"""
        return self._nmt1
    
    @property
    def subsections(self) -> List[Subsection]:
        """Subsections data"""
        return self._subsections
    
    def add_subsection(self, subsection: Subsection) -> None:
        """Add a subsection to this MT"""
        self._subsections.append(subsection)
        
    def get_subsection(self, mt1: int) -> Optional[Subsection]:
        """Get a subsection by MT1 value"""
        for subsection in self._subsections:
            if subsection.mt1 == mt1:
                return subsection
        return None
    
    def __str__(self) -> str:
        """
        Convert the MF34MT object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 34
        mt = self.number
        lines = []
        
        # Helper function to remove line number field and replace with spaces
        def blank_line_number(line: str) -> str:
            # Replace line number field (positions 76-80) with spaces
            return line[:75] + "     "
        
        # Format first line - header - ZA, AWR as float, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, self._nmt1],
            mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, 
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT]
        )
        lines.append(blank_line_number(line1))
        
        # Format subsections
        for subsection in self._subsections:
            # Format subsection header line
            subsec_header = format_endf_data_line(
                [0.0, 0.0, subsection.mat1, subsection.mt1, subsection.nl, subsection.nl1],
                mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(blank_line_number(subsec_header))
            
            # Format sub-subsections
            for sub_subsec in subsection.sub_subsections:
                # Format sub-subsection header
                format_lct = ENDF_FORMAT_INT_ZERO if sub_subsec.lct == 0 else ENDF_FORMAT_INT
                sub_header = format_endf_data_line(
                    [0.0, 0.0, sub_subsec.l, sub_subsec.l1, sub_subsec.lct, sub_subsec.ni],
                    mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                             ENDF_FORMAT_INT, format_lct, ENDF_FORMAT_INT]
                )
                lines.append(blank_line_number(sub_header))
                
                # Format LIST records
                for record in sub_subsec.records:
                    # Different formatting based on LB value
                    if record.lb in (0, 1, 2):
                        # Format LIST record header for LB=0-2 only
                        record_header = format_endf_data_line(
                            [0.0, 0.0, record.lt, record.lb, record.nt, record.np],
                            mat, mf, mt, 0,
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(record_header))
                        

                        # data lines
                        if record.lt in (None, 0):

                            # one (E_k,F_k) table only
                            all_values = []
                            for i in range(len(record.e_table_k)):
                                all_values.append(record.e_table_k[i])
                                all_values.append(record.f_table_k[i])
                            
                            # Format in blocks of 6
                            current_values = []
                            for val in all_values:
                                current_values.append(val)
                                if len(current_values) == 6:
                                    value_line = format_endf_data_line(current_values, mat, mf, mt, 0)
                                    lines.append(blank_line_number(value_line))
                                    current_values = []
                            
                            # Add any remaining values
                            if current_values:
                                # Pad with None for blanks
                                while len(current_values) < 6:
                                    current_values.append(None)
                                value_line = format_endf_data_line(
                                    current_values, 
                                    mat, mf, mt, 0,
                                    formats=[ENDF_FORMAT_FLOAT] * len(current_values) + [ENDF_FORMAT_BLANK] * (6 - len(current_values))
                                )
                                lines.append(blank_line_number(value_line))

                        else:
                            # LT>0: write back the raw floats exactly as read (6 per ENDF line)
                            buf = []
                            for val in record.raw_list_values:
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
                                    formats=[ENDF_FORMAT_FLOAT]*len(buf) + [ENDF_FORMAT_BLANK]*(6-len(buf))
                                )
                                lines.append(blank_line_number(ln))

                    elif record.lb == 5:
                        # Format LIST record header for LB=5 (original format)
                        record_header = format_endf_data_line(
                            [0.0, 0.0, record.ls, record.lb, record.nt, record.ne],
                            mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(record_header))
                        
                        # Format energy grid and matrix values in blocks of 6
                        all_values = record.energies + record.matrix
                        current_values = []
                        
                        for val in all_values:
                            current_values.append(val)
                            if len(current_values) == 6:
                                value_line = format_endf_data_line(current_values, mat, mf, mt, 0)
                                lines.append(blank_line_number(value_line))
                                current_values = []
                        
                        # Add any remaining values
                        if current_values:
                            # Pad with None for blanks
                            while len(current_values) < 6:
                                current_values.append(None)
                            value_line = format_endf_data_line(
                                current_values, 
                                mat, mf, mt, 0,
                                formats=[ENDF_FORMAT_FLOAT] * len(current_values) + [ENDF_FORMAT_BLANK] * (6 - len(current_values))
                            )
                            lines.append(blank_line_number(value_line))
                    
                    elif record.lb == 6:
                        # LIST header: LS=0, LB=6, NT=C5, NE=C6 (NER)
                        ner = len(record.row_energies)
                        nt  = record.nt
                        rl_header = format_endf_data_line(
                            [0.0, 0.0, 0, 6, nt, ner],
                            mat, mf, mt, 0,
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(rl_header))

                        # stack row-energies, col-energies, then matrix values
                        all_vals = record.row_energies + record.col_energies + record.rect_matrix
                        buf = []
                        for v in all_vals:
                            buf.append(v)
                            if len(buf) == 6:
                                ln = format_endf_data_line(buf, mat, mf, mt, 0)
                                lines.append(blank_line_number(ln))
                                buf = []
                        if buf:
                            # pad to 6
                            while len(buf) < 6: buf.append(None)
                            ln = format_endf_data_line(
                                buf,
                                mat, mf, mt, 0,
                                formats=[ENDF_FORMAT_FLOAT]*len(buf) + [ENDF_FORMAT_BLANK]*(6-len(buf))
                            )
                            lines.append(blank_line_number(ln))
        
        # End of section marker - all integers - this one keeps the line number (99999)
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,  # Note MT=0 for end of section, use 99999 for the standard end marker
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, 
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
    
    def _decode_lb012_matrix(self, record: SubSubsectionRecord) -> Tuple[np.ndarray, List[float]]:
        """
        Decodes LB=0, 1, 2 LIST record into an MxM interval matrix and its NE point energy grid.
        M = NE - 1.
        """
        logger.debug(f"Decoding LB={record.lb}...")
        energies = record.e_table_k # Point grid Ek (length NE)
        ne = len(energies)
        if ne < 2:
            raise ValueError(f"LB={record.lb} requires at least 2 energy points (NE >= 2) to define intervals.")
        m = ne - 1 # Number of intervals
        f_values = record.f_table_k # Values corresponding to Ek
        matrix = np.zeros((m, m))
        logger.debug(f"Native grid NE={ne}, Matrix size M={m}x{m}")

        if record.lb == 0: # Absolute variance, diagonal on intervals
            # Needs division by <Xk><Yk> from MF4 for relative - DEFERRED
            # Fk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                # Use Fk associated with the start of the interval Ek
                matrix[k, k] = f_values[k] # Store absolute variance Fk on diagonal C(k,k)
        elif record.lb == 1: # Fractional variance, diagonal on intervals
            # Fk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                 # Use Fk associated with the start of the interval Ek
                matrix[k, k] = f_values[k] # Fk is fractional variance C(k,k)
        elif record.lb == 2: # Relative sigma Sk, fully correlated across intervals
            # Sk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                for l in range(m):
                    # Use Sk and Sl associated with the start of intervals k and l
                    matrix[k, l] = f_values[k] * f_values[l] # Sk * Sl

        logger.debug(f"Finished decoding LB={record.lb}.")
        return matrix, energies # Return MxM matrix and NE point grid

    def _decode_lb5_matrix(self, record: SubSubsectionRecord) -> Tuple[np.ndarray, List[float]]:
        """
        Decodes LB=5 LIST record into an MxM interval matrix and its NE point energy grid.
        M = NE - 1.
        """
        logger.debug(f"Decoding LB=5 LS={record.ls}...")
        energies = record.energies # Point grid Ek (length NE)
        ne = record.ne
        if ne < 2:
            raise ValueError("LB=5 requires at least 2 energy points (NE >= 2) to define intervals.")
        m = ne - 1 # Number of intervals
        ls = record.ls
        raw_values = np.array(record.matrix) # Covariance values Fk,l for intervals
        matrix = np.zeros((m, m))
        logger.debug(f"Native grid NE={ne}, Matrix size M={m}x{m}, Raw values count={len(raw_values)}")


        # Check data size consistency with strict validation
        expected_size = 0
        if ls == 1: # Symmetric upper triangle M(M+1)/2
            expected_size = m * (m + 1) // 2
        else: # Asymmetric M*M (based on ENDF manual, differs from recipe's (M-1)^2)
             expected_size = m * m
        
        if len(raw_values) != expected_size:
             # Log clear error pointing to the issue
             logger.error(f"LB=5 LS={ls} matrix size mismatch for M={m}:")
             logger.error(f"  Expected: {expected_size} elements")
             logger.error(f"  Got: {len(raw_values)} elements")
             logger.error(f"  Formula: LS={ls} → {'M*(M+1)/2' if ls == 1 else 'M*M'} = {expected_size}")
             raise ValueError(f"LB=5 LS={ls}: Incorrect number of matrix values. Expected {expected_size} for M={m}, got {len(raw_values)}.")

        raw_arr = np.asarray(raw_values, dtype=float)
        if ls == 1:  # Symmetric upper triangle, row-wise
            rows, cols = np.triu_indices(m)
            matrix[rows, cols] = raw_arr
            matrix[cols, rows] = raw_arr  # mirror for symmetry
        else:  # Asymmetric full matrix M*M, row-wise
            matrix = raw_arr.reshape(m, m)

        logger.debug(f"Finished decoding LB=5.")
        return matrix, energies # Return MxM matrix and NE point grid

    def _decode_lb6_matrix(self, record: SubSubsectionRecord) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        Decodes LB=6 LIST record into an RxC interval matrix and its NER, NEC point energy grids.
        R = NER - 1, C = NEC - 1.
        """
        logger.debug(f"Decoding LB=6...")
        row_energies = record.row_energies # Point grid ERk (length NER)
        col_energies = record.col_energies # Point grid ECl (length NEC)
        ner = len(row_energies)
        nec = len(col_energies)
        if ner < 2 or nec < 2:
             raise ValueError("LB=6 requires NER >= 2 and NEC >= 2 to define intervals.")
        r = ner - 1 # Number of row intervals
        c = nec - 1 # Number of column intervals
        raw_values = np.array(record.rect_matrix) # Covariance values Fr,c for intervals
        matrix = np.zeros((r, c))
        logger.debug(f"Row grid NER={ner}, Col grid NEC={nec}, Matrix size R={r}x C={c}, Raw values count={len(raw_values)}")


        # Strict validation: exactly R*C elements expected
        expected_size = r * c
        if len(raw_values) != expected_size:
             logger.error(f"LB=6 matrix size mismatch for R={r}, C={c}:")
             logger.error(f"  Expected: {expected_size} elements (R*C)")
             logger.error(f"  Got: {len(raw_values)} elements")
             logger.error(f"  Row intervals (NER-1): {r}")
             logger.error(f"  Column intervals (NEC-1): {c}")
             raise ValueError(f"LB=6: Incorrect number of matrix values. Expected {expected_size} for R={r}*C={c}, got {len(raw_values)}.")

        idx = 0
        start_fill = time.time()
        for k in range(r): # Row interval index k
            for l in range(c): # Column interval index l
                if idx < len(raw_values):
                    matrix[k, l] = raw_values[idx]
                    idx += 1
        end_fill = time.time()
        logger.debug(f"Matrix fill time: {end_fill - start_fill:.4f}s")

        # Final validation that we used all data
        if idx != len(raw_values):
             logger.error(f"LB=6: Used {idx} elements but had {len(raw_values)} available. Internal indexing error.")
             raise ValueError(f"LB=6 internal error: indexing mismatch")

        logger.debug(f"Finished decoding LB=6.")
        return matrix, row_energies, col_energies # Return RxC matrix and NER, NEC point grids

    def _project_matrix_piecewise_constant(self,
                                           component_matrix: np.ndarray,
                                           native_point_grid: List[float],
                                           union_point_grid: List[float],
                                           is_lb6: bool = False,
                                           native_col_point_grid: Optional[List[float]] = None
                                           ) -> np.ndarray:
        """
        Projects an interval matrix onto the union grid using piecewise constant assumption (bin sharing).
        Handles square (MxM -> M_union x M_union) and rectangular (RxC -> M_union x M_union) projection.
        """
        logger.debug(f"Projecting matrix...")
        start_proj = time.time()

        # --- Optimization: Check for identical grids ---
        grids_are_identical = False
        if not is_lb6:
            if native_point_grid == union_point_grid:
                grids_are_identical = True
        else:
            # For LB=6, both row and column grids must match the union grid
            if native_point_grid == union_point_grid and native_col_point_grid == union_point_grid:
                grids_are_identical = True

        if grids_are_identical:
            logger.debug("Native and Union grids are identical. Skipping projection calculation.")
            # Ensure the returned matrix has the correct shape (union_m x union_m)
            # This should already be the case if grids are identical, but double-check
            union_ne = len(union_point_grid)
            union_m = union_ne - 1 if union_ne >= 2 else 0
            if component_matrix.shape == (union_m, union_m):
                 end_proj = time.time()
                 logger.debug(f"Projection time (skipped): {end_proj - start_proj:.4f}s")
                 return component_matrix.copy() # Return a copy
            else:
                 # This case should ideally not happen if grids are identical and decoding was correct
                 logger.warning(f"Grids identical but matrix shape {component_matrix.shape} mismatch union shape ({union_m}x{union_m}). Proceeding with full projection.")
        # --- End Optimization ---


        native_ne = len(native_point_grid)
        union_ne = len(union_point_grid)
        if native_ne < 2 or union_ne < 2:
            logger.debug("Cannot project, NE < 2.")
            return np.zeros((union_ne - 1, union_ne - 1)) # Cannot project without intervals

        native_m = native_ne - 1
        union_m = union_ne - 1

        native_col_ne = len(native_col_point_grid) if is_lb6 and native_col_point_grid else native_ne
        native_col_m = native_col_ne - 1

        logger.debug(f"Native shape ({native_m}x{native_col_m if is_lb6 else native_m}), Union shape ({union_m}x{union_m})")

        if is_lb6 and component_matrix.shape != (native_m, native_col_m):
             raise ValueError(f"LB=6 Projection: Component matrix shape {component_matrix.shape} doesn't match expected ({native_m}, {native_col_m})")
        elif not is_lb6 and component_matrix.shape != (native_m, native_m):
             raise ValueError(f"Projection: Component matrix shape {component_matrix.shape} doesn't match expected ({native_m}, {native_m})")

        # Build row transfer matrix T_row[i,k] = overlap fraction of union bin i with native row bin k
        union_arr = np.asarray(union_point_grid, dtype=float)
        native_row_arr = np.asarray(native_point_grid, dtype=float)

        u_low = union_arr[:-1, None]   # (union_m, 1)
        u_high = union_arr[1:, None]   # (union_m, 1)
        delta_u = u_high - u_low       # (union_m, 1)

        nr_low = native_row_arr[:-1][None, :]   # (1, native_m)
        nr_high = native_row_arr[1:][None, :]   # (1, native_m)

        overlap_low = np.maximum(u_low, nr_low)
        overlap_high = np.minimum(u_high, nr_high)
        overlap_width = np.maximum(0.0, overlap_high - overlap_low)
        # Avoid division by zero for zero-width union bins
        with np.errstate(divide='ignore', invalid='ignore'):
            T_row = np.where(delta_u > 0, overlap_width / delta_u, 0.0)

        # Build column transfer matrix (same as row for non-LB6, different grid for LB6)
        if is_lb6 and native_col_point_grid is not None:
            native_col_arr = np.asarray(native_col_point_grid, dtype=float)
        else:
            native_col_arr = native_row_arr

        nc_low = native_col_arr[:-1][None, :]
        nc_high = native_col_arr[1:][None, :]

        overlap_low_c = np.maximum(u_low, nc_low)
        overlap_high_c = np.minimum(u_high, nc_high)
        overlap_width_c = np.maximum(0.0, overlap_high_c - overlap_low_c)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_col = np.where(delta_u > 0, overlap_width_c / delta_u, 0.0)

        # Projected = T_row @ component_matrix @ T_col.T
        projected_matrix = T_row @ component_matrix @ T_col.T

        end_proj = time.time()
        logger.debug(f"Projection time (calculated): {end_proj - start_proj:.4f}s")
        return projected_matrix


    def to_ang_covmat(self, energy_unit: str = 'eV', mf4_data=None) -> 'LegendreCovariance':
        """
        Convert the MF34MT data to an LegendreCovariance object, aggregating LIST records
        per sub-subsection (L, L1 pair) according to ENDF rules for relative covariance.

        Parameters
        ----------
        energy_unit : str, optional
            Energy unit for the energy grids: 'eV' (default) or 'MeV'
        mf4_data : MF4-like object, optional
            MF4 file object whose ``.mt[reaction]`` entries provide
            ``extract_legendre_coefficients()``.  When supplied, grouped
            Legendre coefficients are computed on the covariance energy grid
            and stored in ``ang_covmat.legendre_coefficients``.

        Returns
        -------
        LegendreCovariance
            Angular distribution covariance matrix object containing MxM relative matrices.
        """
        # Return cached result if available (avoids expensive recomputation)
        if not hasattr(self, '_ang_covmat_cache'):
            self._ang_covmat_cache = {}
        cache_key = (energy_unit, id(mf4_data))
        if cache_key in self._ang_covmat_cache:
            return self._ang_covmat_cache[cache_key]

        _debug = logger.isEnabledFor(logging.DEBUG)
        if _debug:
            logger.debug(f"Starting to_ang_covmat for MF34 MT={self.number}")
        from kika.cov.legendre_covariance import LegendreCovariance # Local import

        isotope = int(self._za)
        reaction = self.number
        ang_covmat = LegendreCovariance(energy_unit=energy_unit)

        # Preserve MF34 provenance so the section can be reconstructed.
        ang_covmat.mt_metadata[(isotope, reaction)] = {
            'za': self._za,
            'awr': self._awr,
            'mat': self._mat,
            'ltt': self._ltt,
        }

        # Process each subsection (MT1)
        logger.debug(f"Found {len(self._subsections)} subsections (MT1).")
        for subsec_idx, subsection in enumerate(self._subsections):
            mt1 = subsection.mt1
            logger.debug(f"Processing subsection {subsec_idx+1}/{len(self._subsections)}: MT1={mt1}")

            # Process each sub-subsection (L, L1 pair)
            logger.debug(f"Found {len(subsection.sub_subsections)} sub-subsections (L,L1).")
            for subsub_idx, sub_subsec in enumerate(subsection.sub_subsections):
                encountered_lbs: Set[int] = set()
                l = sub_subsec.l
                l1 = sub_subsec.l1
                is_variance_matrix = (l == l1)  # Only same L coefficients represent variances
                logger.debug(f"Processing sub-subsection {subsub_idx+1}/{len(subsection.sub_subsections)}: L={l}, L1={l1}, NI={sub_subsec.ni}")
                if is_variance_matrix:
                    logger.debug(f"This is a VARIANCE matrix (L=L1={l}) - diagonal should be non-negative")
                else:
                    logger.debug(f"This is a COVARIANCE matrix (L≠L1) - diagonal can be negative")

                if not sub_subsec.records:
                    logger.debug("Skipping sub-subsection - No LIST records.")
                    continue

                # 1. Determine Union Point Grid for this sub-subsection
                if _debug:
                    start_grid = time.time()
                all_energies_set: Set[float] = set()
                native_grids_map = {} # Store native grids for projection
                component_matrices = [] # Store decoded matrices before projection

                logger.debug(f"Found {len(sub_subsec.records)} LIST records. Determining union grid...")
                for idx, record in enumerate(sub_subsec.records):
                    encountered_lbs.add(record.lb)
                    native_grid = None
                    native_col_grid = None
                    component_matrix = None
                    is_lb6 = False

                    try:
                        # Track this LB type
                        encountered_lbs.add(record.lb)
                        
                        if record.lb in (0, 1, 2):
                            component_matrix, native_grid = self._decode_lb012_matrix(record)
                            if record.lb == 0:
                                logger.warning("LB=0 matrix generated is absolute variance. Needs MF4 data for relative conversion (Deferred). Treating as relative for summation.")
                                # TODO: Implement MF4 lookup and conversion here when available
                        elif record.lb == 5:
                            component_matrix, native_grid = self._decode_lb5_matrix(record)
                        elif record.lb == 6:
                            is_lb6 = True
                            component_matrix, native_grid, native_col_grid = self._decode_lb6_matrix(record)
                            all_energies_set.update(native_col_grid) # Add col grid points to union
                        else:
                             logger.warning(f"Skipping unsupported LB={record.lb} in record {idx+1}")
                             continue

                        if component_matrix is not None and native_grid is not None:
                             all_energies_set.update(native_grid)
                             native_grids_map[idx] = (native_grid, native_col_grid) # Store grids
                             component_matrices.append((idx, component_matrix, is_lb6)) # Store matrix and type
                             if _debug:
                                 logger.debug(f"Decoded component {idx+1} (LB={record.lb}), Native NE={len(native_grid)}")
                                 # Check for problematic diagonals in decoded component
                                 diag_vals = np.diag(component_matrix)
                                 if is_variance_matrix:
                                     neg_diag_indices = np.where(diag_vals < 0)[0]
                                     if len(neg_diag_indices) > 0:
                                         logger.debug(f"*** COMPONENT {idx+1} (LB={record.lb}) HAS NEGATIVE VARIANCE DIAGONALS! ***")
                                         logger.debug(f"This is PROBLEMATIC for variance matrix (L=L1={l})")
                                         logger.debug(f"Negative diagonal indices: {neg_diag_indices}")
                                         logger.debug(f"Negative diagonal values: {diag_vals[neg_diag_indices]}")
                                         logger.debug(f"Min diagonal value: {np.min(diag_vals)}")
                                         logger.debug(f"Max diagonal value: {np.max(diag_vals)}")
                                     else:
                                         logger.debug(f"Component {idx+1} variance diagonal OK - min: {np.min(diag_vals):.2e}, max: {np.max(diag_vals):.2e}")
                                 else:
                                     neg_diag_indices = np.where(diag_vals < 0)[0]
                                     if len(neg_diag_indices) > 0:
                                         logger.debug(f"Component {idx+1} (LB={record.lb}) has negative covariance diagonals (EXPECTED for L≠L1)")
                                         logger.debug(f"Negative diagonal count: {len(neg_diag_indices)}")
                                         logger.debug(f"Diagonal range: [{np.min(diag_vals):.2e}, {np.max(diag_vals):.2e}]")
                                     else:
                                         logger.debug(f"Component {idx+1} covariance diagonal - min: {np.min(diag_vals):.2e}, max: {np.max(diag_vals):.2e}")

                    except ValueError as e: # Catch decoding errors (e.g., NE<2, size mismatch)
                        logger.error(f"Error decoding record {idx+1} (LB={record.lb}) in sub-subsection (L={l}, L1={l1}): {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing record {idx+1} (LB={record.lb}) in sub-subsection (L={l}, L1={l1}): {e}")

                if _debug:
                    end_grid = time.time()
                    logger.debug(f"Grid determination and decoding time: {end_grid - start_grid:.4f}s")

                if not all_energies_set:
                    if _debug:
                        logger.debug("Skipping sub-subsection - No valid energies/components found after decoding.")
                    continue

                union_point_grid = sorted(list(all_energies_set))
                union_ne = len(union_point_grid)
                if union_ne < 2:
                    if _debug:
                        logger.debug("Skipping sub-subsection - Union grid NE < 2.")
                    continue
                union_m = union_ne - 1
                total_cov_matrix = np.zeros((union_m, union_m))
                if _debug:
                    logger.debug(f"Union grid NE={union_ne}, Final matrix size M={union_m}x{union_m}")


                # 4. Project each component onto the Union Grid and Sum
                if _debug:
                    start_sum = time.time()
                    logger.debug(f"Projecting and summing {len(component_matrices)} components...")
                for comp_idx, (idx, component_matrix, is_lb6) in enumerate(component_matrices):
                    try:
                        if _debug:
                            logger.debug(f"Projecting component {idx+1} (#{comp_idx+1}/{len(component_matrices)})...")
                        native_grid, native_col_grid = native_grids_map[idx]
                        projected_component = self._project_matrix_piecewise_constant(
                            component_matrix,
                            native_grid,
                            union_point_grid,
                            is_lb6=is_lb6,
                            native_col_point_grid=native_col_grid
                        )

                        if _debug:
                            # Check for problematic diagonals after projection
                            proj_diag_vals = np.diag(projected_component)
                            if is_variance_matrix:
                                neg_proj_indices = np.where(proj_diag_vals < 0)[0]
                                if len(neg_proj_indices) > 0:
                                    logger.debug(f"*** PROJECTED COMPONENT {idx+1} HAS NEGATIVE VARIANCE DIAGONALS! ***")
                                    logger.debug(f"This is PROBLEMATIC for variance matrix (L=L1={l})")
                                    logger.debug(f"Negative diagonal indices: {neg_proj_indices}")
                                    logger.debug(f"Negative diagonal values: {proj_diag_vals[neg_proj_indices]}")
                                    logger.debug(f"Min diagonal value: {np.min(proj_diag_vals)}")
                                    logger.debug(f"Max diagonal value: {np.max(proj_diag_vals)}")
                                else:
                                    logger.debug(f"Projected component {idx+1} variance diagonal OK - min: {np.min(proj_diag_vals):.2e}, max: {np.max(proj_diag_vals):.2e}")
                            else:
                                neg_proj_indices = np.where(proj_diag_vals < 0)[0]
                                if len(neg_proj_indices) > 0:
                                    logger.debug(f"Projected component {idx+1} has negative covariance diagonals (EXPECTED for L≠L1)")
                                    logger.debug(f"Negative diagonal count: {len(neg_proj_indices)}")
                                    logger.debug(f"Diagonal range: [{np.min(proj_diag_vals):.2e}, {np.max(proj_diag_vals):.2e}]")
                                else:
                                    logger.debug(f"Projected component {idx+1} covariance diagonal - min: {np.min(proj_diag_vals):.2e}, max: {np.max(proj_diag_vals):.2e}")

                            # Check total matrix diagonal before adding this component
                            total_diag_before = np.diag(total_cov_matrix).copy()
                            min_before = np.min(total_diag_before)
                            max_before = np.max(total_diag_before)

                        # 5. Add to total covariance matrix
                        total_cov_matrix += projected_component

                        if _debug:
                            # Check total matrix diagonal after adding this component
                            total_diag_after = np.diag(total_cov_matrix)
                            min_after = np.min(total_diag_after)
                            max_after = np.max(total_diag_after)

                            logger.debug(f"Total matrix diagonal after adding component {idx+1}:")
                            logger.debug(f"Before: min={min_before:.2e}, max={max_before:.2e}")
                            logger.debug(f"After:  min={min_after:.2e}, max={max_after:.2e}")

                            if is_variance_matrix:
                                neg_total_indices = np.where(total_diag_after < 0)[0]
                                if len(neg_total_indices) > 0:
                                    logger.debug(f"*** TOTAL VARIANCE MATRIX NOW HAS {len(neg_total_indices)} NEGATIVE DIAGONALS! ***")
                                    logger.debug(f"This is PROBLEMATIC for variance matrix (L=L1={l})")
                                    logger.debug(f"First few negative indices: {neg_total_indices[:10]}")
                                    logger.debug(f"First few negative values: {total_diag_after[neg_total_indices[:10]]}")

                                    newly_negative = np.where((total_diag_before >= 0) & (total_diag_after < 0))[0]
                                    if len(newly_negative) > 0:
                                        logger.debug(f"*** COMPONENT {idx+1} CAUSED {len(newly_negative)} NEW NEGATIVE VARIANCE DIAGONALS! ***")
                                        logger.debug(f"Newly negative indices: {newly_negative[:10]}")
                                        logger.debug(f"Values before: {total_diag_before[newly_negative[:10]]}")
                                        logger.debug(f"Values after: {total_diag_after[newly_negative[:10]]}")
                                        logger.debug(f"Component contribution: {proj_diag_vals[newly_negative[:10]]}")
                                else:
                                    logger.debug(f"Total variance matrix diagonal still OK after component {idx+1}")
                            else:
                                neg_total_indices = np.where(total_diag_after < 0)[0]
                                if len(neg_total_indices) > 0:
                                    logger.debug(f"Total covariance matrix has {len(neg_total_indices)} negative diagonals (EXPECTED for L≠L1)")
                                else:
                                    logger.debug(f"Total covariance matrix has no negative diagonals")

                    except Exception as e:
                         logger.error(f"Error projecting component {idx} in sub-subsection (L={l}, L1={l1}): {e}")

                if _debug:
                    end_sum = time.time()
                    logger.debug(f"Projection and summation time: {end_sum - start_sum:.4f}s")

                # Sanity Checks
                # 1. Size
                if total_cov_matrix.shape != (union_m, union_m):
                     logger.error(f"Final matrix shape {total_cov_matrix.shape} != expected ({union_m}, {union_m}) for L={l}, L1={l1}")
                     continue # Skip adding this matrix

                # 2. Diagonal Positivity - only enforce for variance matrices (L=L1)
                if is_variance_matrix and not np.all(np.diag(total_cov_matrix) >= -1e-9):
                     neg_diags = np.where(np.diag(total_cov_matrix) < -1e-9)[0]
                     logger.warning(f"Negative diagonal elements found in VARIANCE matrix for L=L1={l} at indices {neg_diags}. Clamping to zero.")

                     if _debug:
                         final_diag = np.diag(total_cov_matrix)
                         logger.debug(f"*** FINAL VARIANCE MATRIX NEGATIVE DIAGONAL SUMMARY ***")
                         logger.debug(f"This is PROBLEMATIC - variance matrix should have non-negative diagonal")
                         logger.debug(f"Total negative diagonals: {len(neg_diags)}")
                         logger.debug(f"Most negative value: {np.min(final_diag)}")
                         logger.debug(f"Negative values range: [{np.min(final_diag[neg_diags]):.2e}, {np.max(final_diag[neg_diags]):.2e}]")
                         logger.debug(f"Matrix size: {total_cov_matrix.shape}")
                         logger.debug(f"Sub-subsection: L=L1={l}")
                         logger.debug(f"Number of components summed: {len(component_matrices)}")

                     total_cov_matrix[np.diag_indices_from(total_cov_matrix)] = np.maximum(np.diag(total_cov_matrix), 0)
                elif not is_variance_matrix and _debug:
                    # For covariance matrices, just report final state without clamping
                    final_diag = np.diag(total_cov_matrix)
                    neg_diags = np.where(final_diag < 0)[0]
                    if len(neg_diags) > 0:
                        logger.debug(f"Final covariance matrix (L={l}, L1={l1}) has {len(neg_diags)} negative diagonals (EXPECTED)")
                        logger.debug(f"Diagonal range: [{np.min(final_diag):.2e}, {np.max(final_diag):.2e}]")
                    else:
                        logger.debug(f"Final covariance matrix (L={l}, L1={l1}) has no negative diagonals")
                        logger.debug(f"Diagonal range: [{np.min(final_diag):.2e}, {np.max(final_diag):.2e}]")


                # Determine metadata for this sub-subsection
                # is_relative: False only when LB=0 is present
                is_relative = 0 not in encountered_lbs
                
                # frame: based on LCT value
                raw_lct = sub_subsec.lct
                lct = int(raw_lct) if raw_lct is not None else 0  # normalize to int
                if lct == 0:
                    frame = "same-as-MF4"
                elif lct == 1:
                    frame = "LAB"
                elif lct == 2:
                    frame = "CM"
                else:
                    frame = f"unknown LCT={lct}"

                # Add the final aggregated MxM matrix for this sub-subsection (L, L1)
                if _debug:
                    logger.debug(f"Adding final matrix for L={l}, L1={l1} to LegendreCovariance.")
                    logger.debug(f"Matrix metadata: is_relative={is_relative}, frame={frame}")
                ang_covmat.add_matrix(
                    isotope, reaction, l,
                    isotope, mt1, l1,
                    total_cov_matrix,
                    union_point_grid,
                    is_relative=is_relative,
                    frame=frame
                )


        # Populate Legendre coefficients from MF4 data when available
        if mf4_data is not None:
            self._populate_legendre_from_mf4(ang_covmat, mf4_data, isotope)

        self._ang_covmat_cache[cache_key] = ang_covmat
        return ang_covmat

    @staticmethod
    def _populate_legendre_from_mf4(
        ang_covmat: 'LegendreCovariance',
        mf4_data,
        isotope: int,
    ) -> None:
        """Extract cell-averaged Legendre coefficients from MF4 onto MF34 energy grids.

        For each unique (reaction, l_order) found in the covariance matrices,
        compute ā_l per energy cell via trapezoidal integration of the
        pointwise a_l(E) returned by ``mf4_data.mt[reaction].extract_legendre_coefficients()``.
        Results are stored in ``ang_covmat.legendre_coefficients[(isotope, reaction, l)]``.
        """
        if not hasattr(mf4_data, 'mt'):
            return

        # Collect unique (reaction, l, grid_index) triples from the covariance data
        seen: dict = {}  # (reaction, l) → energy_grid index
        for i, (mt_r, l_r) in enumerate(zip(ang_covmat.reaction_rows, ang_covmat.l_rows)):
            key = (mt_r, l_r)
            if key not in seen:
                seen[key] = i

        for (reaction, l_order), grid_idx in seen.items():
            if reaction not in mf4_data.mt:
                continue
            mf4_mt = mf4_data.mt[reaction]
            if not hasattr(mf4_mt, 'extract_legendre_coefficients'):
                continue

            grid = np.asarray(ang_covmat.energy_grids[grid_idx], dtype=float)
            n_cells = len(grid) - 1
            if n_cells < 1:
                continue

            # Build all sub-grid energies at once (5 points per cell)
            n_sub = 5
            sub_e_2d = np.column_stack(
                [np.linspace(grid[c], grid[c + 1], n_sub) for c in range(n_cells)]
            ).T  # shape (n_cells, n_sub)

            # ONE call with all energies flattened
            coeffs_dict = mf4_mt.extract_legendre_coefficients(
                sub_e_2d.ravel(), max_legendre_order=l_order, out_of_range='zero',
            )

            if l_order in coeffs_dict:
                vals_2d = np.asarray(coeffs_dict[l_order], dtype=float).reshape(n_cells, n_sub)
                cell_avg = np.trapezoid(vals_2d, sub_e_2d, axis=1) / (grid[1:] - grid[:-1])
            else:
                cell_avg = np.zeros(n_cells, dtype=float)

            ang_covmat.legendre_coefficients[(isotope, reaction, l_order)] = cell_avg

    def to_plot_data(
        self,
        order: int,
        sigma: float = 1.0,
        uncertainty_type: str = 'relative',
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for Legendre coefficient uncertainties.
        
        This is a convenience method to easily convert MF34 uncertainty data into
        a plottable format using the new plotting infrastructure. It automatically
        converts the MF34MT data to an LegendreCovariance and extracts the uncertainty
        for the specified Legendre order.
        
        Parameters
        ----------
        order : int
            Legendre polynomial order
        sigma : float, default 1.0
            Number of sigma levels for uncertainty (1.0 = 1σ, 2.0 = 2σ)
        uncertainty_type : str, default 'relative'
            Type of uncertainty: 'relative' (%) or 'absolute'
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        tuple of (None, LegendreUncertaintyPlotData)
            Returns a tuple where the first element is None (for consistency with
            other to_plot_data methods) and the second is the uncertainty plot data
            object ready to be added to a PlotBuilder
            
        Raises
        ------
        ValueError
            If uncertainty data is not available for the specified order
            
        Examples
        --------
        >>> # Extract 1-sigma uncertainty data from MF34MT
        >>> mf34_mt2 = endf.mf[34].mt[2]
        >>> _, unc_data = mf34_mt2.to_plot_data(order=1)
        >>> 
        >>> # Extract 2-sigma uncertainty data
        >>> unc_2sigma = mf34_mt2.to_plot_data(order=1, sigma=2.0)
        >>> 
        >>> # Build a plot
        >>> from kika.plotting import PlotBuilder
        >>> fig = PlotBuilder().add_data(unc_data).build()
        """
        # Convert to LegendreCovariance
        mf34_covmat = self.to_ang_covmat()
        
        # Extract isotope and MT from this object
        isotope = int(self._za)
        mt = self.number
        
        # Delegate to LegendreCovariance's to_plot_data method (returns tuple)
        return mf34_covmat.to_plot_data(
            nuclide=isotope,
            mt=mt,
            order=order,
            sigma=sigma,
            uncertainty_type=uncertainty_type,
            label=label,
            **styling_kwargs
        )
