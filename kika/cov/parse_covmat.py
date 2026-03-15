import struct
import warnings
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import pandas as pd

from kika._constants import MT_TO_REACTION, ENDF_MAT_TO_ZAID, ZAID_TO_ENDF_MAT
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.legendre_covariance import LegendreCovariance
from kika.endf.utils import (
    parse_number, parse_line, parse_endf_id,
    format_endf_number, format_endf_data_line,
    ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
)
from kika.energy_grids.grids import SCALE44, SCALE56, SCALE238, SCALE252


class EmptyParsingError(Exception):
    """Raised when no data was extracted during parsing."""
    pass

class InvalidDataFormatError(Exception):
    """Raised when the data format is invalid or corrupted."""
    pass


# ---------------------------------------------------------------------------
# COVERX format helpers
# ---------------------------------------------------------------------------

def _read_fortran_record(f, endian: str) -> bytes:
    """
    Read a single Fortran unformatted sequential record.

    Parameters
    ----------
    f : file object
        Open binary file handle
    endian : str
        Struct endian prefix: '>' for big-endian, '<' for little-endian

    Returns
    -------
    bytes
        The record payload (without markers), or None at EOF.

    Raises
    ------
    InvalidDataFormatError
        If record markers do not match.
    """
    marker_fmt = endian + 'i'
    head = f.read(4)
    if len(head) < 4:
        return None
    rec_len = struct.unpack(marker_fmt, head)[0]
    data = f.read(rec_len)
    tail = struct.unpack(marker_fmt, f.read(4))[0]
    if rec_len != tail:
        raise InvalidDataFormatError(
            f"Fortran record marker mismatch: header={rec_len}, trailer={tail}"
        )
    return data


def _write_fortran_record(f, data: bytes, endian: str) -> None:
    """
    Write a single Fortran unformatted sequential record.

    Parameters
    ----------
    f : file object
        Open binary file handle
    data : bytes
        Record payload
    endian : str
        Struct endian prefix: '>' for big-endian, '<' for little-endian
    """
    marker = struct.pack(endian + 'i', len(data))
    f.write(marker + data + marker)


def _dense_to_banded(matrix: np.ndarray):
    """
    Convert a dense matrix to COVERX banded storage.

    For each row, find the first and last non-zero column indices
    and pack the contiguous band (including interior zeros).

    Parameters
    ----------
    matrix : np.ndarray
        Dense square matrix of shape (ngroup, ngroup)

    Returns
    -------
    jband : list of int
        Band width for each row
    ijj : list of int
        Diagonal offset for each row (1-based: ``row - first_col + 1``)
    values : list of float
        Packed band values (concatenated across rows)
    """
    ngroup = matrix.shape[0]
    jband, ijj, values = [], [], []
    for j in range(ngroup):
        nz = np.nonzero(matrix[j, :])[0]
        if len(nz) == 0:
            jband.append(0)
            ijj.append(j + 1)
        else:
            c1, c2 = int(nz[0]), int(nz[-1])
            jband.append(c2 - c1 + 1)
            ijj.append(j - c1 + 1)
            values.extend(matrix[j, c1:c2 + 1].tolist())
    return jband, ijj, values


def _detect_coverx_format(file_path: str) -> str:
    """
    Detect whether a COVERX file is text or binary.

    Returns
    -------
    str
        ``'text'`` or ``'binary'``
    """
    with open(file_path, 'rb') as f:
        header = f.read(4)
    if len(header) < 4:
        raise InvalidDataFormatError(f"File too small: {file_path}")
    if all(b == 0x09 or b == 0x0A or b == 0x0D or (0x20 <= b <= 0x7E)
           for b in header):
        return 'text'
    return 'binary'


def _detect_endianness(file_path: str) -> str:
    """
    Detect endianness of a binary COVERX file.

    The first Fortran record marker encodes the byte-length of Record 1
    (typically 22 bytes).

    Returns
    -------
    str
        ``'>'`` for big-endian, ``'<'`` for little-endian.
    """
    with open(file_path, 'rb') as f:
        raw = f.read(4)
    big = struct.unpack('>i', raw)[0]
    little = struct.unpack('<i', raw)[0]
    if 4 <= big <= 1000:
        return '>'
    elif 4 <= little <= 1000:
        return '<'
    raise InvalidDataFormatError(
        f"Cannot determine endianness from first 4 bytes: {raw.hex()}"
    )


# ---------------------------------------------------------------------------
# Public API — COVERX (text or binary, auto-detected)
# ---------------------------------------------------------------------------

def read_coverx(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CrossSectionCovariance:
    """
    Read a COVERX covariance file (text or binary) and return a CrossSectionCovariance object.

    The format is auto-detected: if the first bytes are printable ASCII the
    file is treated as text; otherwise it is parsed as a Fortran unformatted
    binary COVERX file.

    Parameters
    ----------
    file_path : str
        Path to the COVERX covariance file
    ascending : bool, optional
        If True, energies are reordered in ascending order (default True)
    energy_unit : str, optional
        Energy unit for the energy grid: ``'eV'`` (default) or ``'MeV'``

    Returns
    -------
    CrossSectionCovariance
        Parsed covariance data

    Raises
    ------
    EmptyParsingError
        If no valid covariance matrices were found in the file
    InvalidDataFormatError
        If the binary structure is corrupted
    """
    fmt = _detect_coverx_format(file_path)
    if fmt == 'binary':
        return _read_coverx_binary(file_path, ascending, energy_unit)
    return _read_coverx_text(file_path, ascending, energy_unit)


# ---------------------------------------------------------------------------
# COVERX text parser (formerly read_scale_covmat)
# ---------------------------------------------------------------------------

def _read_coverx_text(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CrossSectionCovariance:
    """
    Read a SCALE covariance matrix text file and convert it to a CrossSectionCovariance object.

    Parameters
    ----------
    file_path : str
        Path to the SCALE covariance matrix text file
    ascending : bool, optional
        If True, the energies will be ordered in ascending order (default is True)
    energy_unit : str, optional
        Energy unit for the energy grid: 'eV' (default) or 'MeV'

    Returns
    -------
    CrossSectionCovariance
        CrossSectionCovariance object containing the parsed covariance data

    Raises
    ------
    EmptyParsingError
        If no data was extracted from the file
    FileNotFoundError
        If the input file does not exist
    """
    # Read the file
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    # Parse the group number from the second line
    num_groups = int(file_lines[1].split()[0])

    # Create CrossSectionCovariance object
    covmat = CrossSectionCovariance(num_groups, energy_unit=energy_unit)

    # Determine the energy grid based on the number of groups
    potential_grids = {
        len(SCALE44) - 1: SCALE44,
        len(SCALE56) - 1: SCALE56,
        len(SCALE238) - 1: SCALE238,
        len(SCALE252) - 1: SCALE252,
    }

    if num_groups in potential_grids:
        covmat.energy_grid = potential_grids[num_groups]

    # Parse the file
    for i, line in enumerate(file_lines):
        if i > 2 and len(line.split()) == 5:
            try:
                # Parse isotope and reaction numbers
                reaction_row = int(line.split()[1])
                reaction_col = int(line.split()[3])

                if (reaction_row in MT_TO_REACTION and reaction_col in MT_TO_REACTION):

                    isotope_row = int(line.split()[0])
                    isotope_col = int(line.split()[2])

                    # Read matrix values
                    matrix_values = []
                    values_read = 0
                    j = 0
                    while values_read < num_groups * num_groups:
                        for val in file_lines[i + 1 + j].split():
                            matrix_values.append(float(val))
                        values_read += len(file_lines[i + 1 + j].split())
                        j += 1

                    # Convert to numpy array and reshape
                    matrix = np.array(matrix_values).reshape(num_groups, num_groups)

                    if ascending:
                        matrix = np.flipud(np.fliplr(matrix))

                    # Add to CrossSectionCovariance object
                    covmat.add_matrix(isotope_row, reaction_row, isotope_col, reaction_col, matrix)
            except (ValueError, IndexError):
                # Skip lines with invalid data
                continue

    # Verify we found at least some valid data
    if covmat.num_matrices == 0:
        raise EmptyParsingError(f"No valid data was extracted from the covariance matrix file: {file_path}")

    return covmat


# ---------------------------------------------------------------------------
# COVERX binary parser
# ---------------------------------------------------------------------------

def _read_coverx_binary(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CrossSectionCovariance:
    """
    Read a binary COVERX covariance file and return a CrossSectionCovariance object.

    Supports both big-endian and little-endian Fortran unformatted files
    (endianness is auto-detected).

    Parameters
    ----------
    file_path : str
        Path to the binary COVERX file
    ascending : bool, optional
        If True, reorder energies and matrices to ascending order (default True)
    energy_unit : str, optional
        Energy unit: ``'eV'`` (default) or ``'MeV'``

    Returns
    -------
    CrossSectionCovariance
        Parsed covariance data

    Raises
    ------
    EmptyParsingError
        If no valid covariance matrices were found
    InvalidDataFormatError
        If record markers are inconsistent
    """
    endian = _detect_endianness(file_path)

    with open(file_path, 'rb') as f:
        # Record 1: File identification (hname + huse + ivers)
        _read_fortran_record(f, endian)

        # Record 2: File control — 7 integers
        rec2 = _read_fortran_record(f, endian)
        ngroup, nngrup, _nggrup, _ntype, nmmp, nmtrix, nholl = struct.unpack(
            endian + '7i', rec2
        )

        # Record 3: File description
        _read_fortran_record(f, endian)

        # Record 4: Energy group boundaries — (nngrup + 1) floats
        rec4 = _read_fortran_record(f, endian)
        n_energies = nngrup + 1
        energy_grid = list(struct.unpack(endian + f'{n_energies}f',
                                         rec4[:n_energies * 4]))

        # Record 5: Material-reaction pair metadata — nmmp × (matid, mtid, mwgt)
        rec5 = _read_fortran_record(f, endian)
        triplets = struct.unpack(endian + f'{nmmp * 3}i', rec5)
        pairs = [(triplets[i * 3], triplets[i * 3 + 1]) for i in range(nmmp)]

        # Records 6 … 6+nmmp-1: cross-section data per (matid, mtid) pair
        xs_data: Dict[Tuple[int, int], np.ndarray] = {}
        for matid, mtid in pairs:
            rec = _read_fortran_record(f, endian)
            xs = np.array(struct.unpack(endian + f'{ngroup}f', rec[:ngroup * 4]))
            if not np.allclose(xs, 0.0):
                xs_data[(matid, mtid)] = xs

        # --- Covariance matrix blocks (nmtrix total) ----------------------
        covmat = CrossSectionCovariance(num_groups=ngroup, energy_unit=energy_unit)

        # Use predefined SCALE grids when available (consistent with text parser)
        potential_grids = {
            len(SCALE44) - 1: SCALE44,
            len(SCALE56) - 1: SCALE56,
            len(SCALE238) - 1: SCALE238,
            len(SCALE252) - 1: SCALE252,
        }
        if ngroup in potential_grids:
            covmat.energy_grid = potential_grids[ngroup]
        else:
            covmat.energy_grid = energy_grid

        # Store cross-sections (flip to ascending order if requested)
        for key, xs in xs_data.items():
            covmat.cross_sections[key] = xs[::-1] if ascending else xs

        for _ in range(nmtrix):
            # Control record: mat1, mt1, mat2, mt2, nblock
            rec_ctrl = _read_fortran_record(f, endian)
            mat1, mt1, mat2, mt2, nblock = struct.unpack(endian + '5i', rec_ctrl)

            # Band + Legendre record: ngroup×(jband,ijj) + nblock×lgpr
            rec_band = _read_fortran_record(f, endian)
            n_ints = 2 * ngroup + nblock
            ints = struct.unpack(endian + f'{n_ints}i', rec_band[:n_ints * 4])

            jband = [ints[i * 2] for i in range(ngroup)]
            ijj = [ints[i * 2 + 1] for i in range(ngroup)]

            # Data record(s) — one per Legendre block; use only P0 (first)
            total_vals = sum(jband)
            rec_data = _read_fortran_record(f, endian)
            vals = struct.unpack(endian + f'{total_vals}f', rec_data[:total_vals * 4])

            # Skip remaining Legendre blocks if nblock > 1
            for _ in range(1, nblock):
                _read_fortran_record(f, endian)

            # Unpack banded sparse into dense matrix
            matrix = np.zeros((ngroup, ngroup))
            offset = 0
            for j in range(ngroup):
                if jband[j] > 0:
                    start_col = j - ijj[j] + 1
                    matrix[j, start_col:start_col + jband[j]] = vals[offset:offset + jband[j]]
                    offset += jband[j]

            # Skip zero-sum matrices
            if np.isclose(matrix.sum(), 0.0):
                continue

            # Skip unknown reaction types
            if mt1 not in MT_TO_REACTION or mt2 not in MT_TO_REACTION:
                continue

            if ascending:
                matrix = np.flipud(np.fliplr(matrix))

            covmat.add_matrix(mat1, mt1, mat2, mt2, matrix)

        # Flip energy grid if ascending (only needed for file-read grids;
        # predefined SCALE grids are already in ascending order)
        if ascending and ngroup not in potential_grids:
            covmat.energy_grid = list(reversed(covmat.energy_grid))

    if covmat.num_matrices == 0:
        raise EmptyParsingError(
            f"No valid data was extracted from the binary COVERX file: {file_path}"
        )

    return covmat


# ---------------------------------------------------------------------------
# COVERX writers
# ---------------------------------------------------------------------------

def _write_coverx_binary(
    data: CrossSectionCovariance, file_path: str, title: str = '', endian: str = '>'
) -> None:
    """
    Write a :class:`CrossSectionCovariance` to a binary COVERX file (Fortran unformatted).

    Parameters
    ----------
    data : CrossSectionCovariance
        Covariance data to write.
    file_path : str
        Output file path.
    title : str, optional
        File title (max 18 characters). Default ``''``.
    endian : str, optional
        Endianness: ``'>'`` big-endian (default), ``'<'`` little-endian.
    """
    warnings.warn(
        "COVERX binary uses float32: covariance values will be truncated "
        "to ~7 significant figures.",
        stacklevel=3,
    )

    ngroup = data.num_groups
    energy_grid = list(data.energy_grid) if data.energy_grid else []

    # Collect unique (isotope, mt) pairs from matrices and cross_sections
    pairs_set = set()
    for idx in range(data.num_matrices):
        pairs_set.add((data.isotope_rows[idx], data.reaction_rows[idx]))
        pairs_set.add((data.isotope_cols[idx], data.reaction_cols[idx]))
    for key in data.cross_sections:
        pairs_set.add(key)
    pairs = sorted(pairs_set)

    nmmp = len(pairs)
    nmtrix = data.num_matrices
    nholl = max(1, (len(title) + 3) // 4)  # Hollerith words

    # Determine if energy grid needs flipping (must be descending for COVERX)
    ascending_input = len(energy_grid) >= 2 and energy_grid[0] < energy_grid[-1]

    # Convert energy to eV if in MeV
    eg = list(energy_grid)
    if data.energy_unit == 'MeV':
        eg = [e * 1e6 for e in eg]

    # Ensure descending order
    if ascending_input:
        eg = list(reversed(eg))

    with open(file_path, 'wb') as f:
        # Record 1: File identification — title (18 bytes) + ivers (4 bytes)
        title_bytes = title[:18].ljust(18).encode('ascii')
        rec1 = title_bytes + struct.pack(endian + 'i', 1)
        _write_fortran_record(f, rec1, endian)

        # Record 2: File control — 7 integers
        rec2 = struct.pack(
            endian + '7i',
            ngroup, ngroup, 0, 2, nmmp, nmtrix, nholl,
        )
        _write_fortran_record(f, rec2, endian)

        # Record 3: File description — Hollerith words
        desc = title[:nholl * 4].ljust(nholl * 4).encode('ascii')
        _write_fortran_record(f, desc, endian)

        # Record 4: Energy group boundaries — (ngroup+1) float32, descending
        n_energies = ngroup + 1
        rec4 = struct.pack(endian + f'{n_energies}f', *eg[:n_energies])
        _write_fortran_record(f, rec4, endian)

        # Record 5: Pair metadata — nmmp × (matid, mtid, mwgt=4)
        triplets = []
        for iso, mt in pairs:
            triplets.extend([iso, mt, 4])
        rec5 = struct.pack(endian + f'{nmmp * 3}i', *triplets)
        _write_fortran_record(f, rec5, endian)

        # Records 6..6+nmmp-1: Cross-section data per pair
        pair_index = {p: i for i, p in enumerate(pairs)}
        for iso, mt in pairs:
            key = (iso, mt)
            if key in data.cross_sections:
                xs = np.array(data.cross_sections[key], dtype=np.float32)
                if ascending_input:
                    xs = xs[::-1]
            else:
                xs = np.zeros(ngroup, dtype=np.float32)
            rec = struct.pack(endian + f'{ngroup}f', *xs)
            _write_fortran_record(f, rec, endian)

        # Covariance blocks — one per matrix
        for idx in range(data.num_matrices):
            mat1 = data.isotope_rows[idx]
            mt1 = data.reaction_rows[idx]
            mat2 = data.isotope_cols[idx]
            mt2 = data.reaction_cols[idx]
            matrix = data.matrices[idx].copy()

            # Flip to descending order if input is ascending
            if ascending_input:
                matrix = np.flipud(np.fliplr(matrix))

            # Control record: mat1, mt1, mat2, mt2, nblock=1
            ctrl = struct.pack(endian + '5i', mat1, mt1, mat2, mt2, 1)
            _write_fortran_record(f, ctrl, endian)

            # Compute banded storage
            jband, ijj, vals = _dense_to_banded(matrix)

            # Band record: interleaved jband[j], ijj[j] + lgpr=[ngroup]
            band_ints = []
            for j in range(ngroup):
                band_ints.append(jband[j])
                band_ints.append(ijj[j])
            band_ints.append(ngroup)  # lgpr for single block
            rec_band = struct.pack(endian + f'{len(band_ints)}i', *band_ints)
            _write_fortran_record(f, rec_band, endian)

            # Data record: packed band values as float32
            rec_data = struct.pack(endian + f'{len(vals)}f', *[float(v) for v in vals])
            _write_fortran_record(f, rec_data, endian)


def _write_e15_values(f, values: list) -> None:
    """Write floats 5 per line in Fortran E15.7 format."""
    buf = []
    for v in values:
        buf.append(f"{v:15.7E}")
        if len(buf) == 5:
            f.write(''.join(buf) + '\n')
            buf = []
    if buf:
        f.write(''.join(buf) + '\n')


def _write_coverx_text(data: CrossSectionCovariance, file_path: str, title: str = '') -> None:
    """
    Write a :class:`CrossSectionCovariance` to a text COVERX file.

    The output mirrors the format of SCALE text covariance files
    (e.g. ``scale.rev05.44groupcov.txt``).

    Parameters
    ----------
    data : CrossSectionCovariance
        Covariance data to write.
    file_path : str
        Output file path.
    title : str, optional
        File title. Default ``''``.
    """
    if data.cross_sections:
        warnings.warn(
            f"COVERX text format does not support cross-section storage. "
            f"{len(data.cross_sections)} cross-section vector(s) will be lost.",
            stacklevel=3,
        )

    ngroup = data.num_groups
    energy_grid = list(data.energy_grid) if data.energy_grid else []
    nmtrix = data.num_matrices

    # Collect unique (isotope, mt) pairs
    pairs_set = set()
    for idx in range(data.num_matrices):
        pairs_set.add((data.isotope_rows[idx], data.reaction_rows[idx]))
        pairs_set.add((data.isotope_cols[idx], data.reaction_cols[idx]))
    nmmp = len(pairs_set)

    # Determine if energy grid needs flipping
    ascending_input = len(energy_grid) >= 2 and energy_grid[0] < energy_grid[-1]

    # Convert to eV if needed
    eg = list(energy_grid)
    if data.energy_unit == 'MeV':
        eg = [e * 1e6 for e in eg]

    # Ensure descending order
    if ascending_input:
        eg = list(reversed(eg))

    with open(file_path, 'w') as f:
        # Line 1: Title
        f.write(title + '\n')

        # Line 2: ngroup, 2, nmmp, nmtrix in I12 format
        f.write(f"{ngroup:12d}{2:12d}{nmmp:12d}{nmtrix:12d}\n")

        # Energy grid: ngroup+1 values, E15.7, descending
        _write_e15_values(f, eg)

        # Covariance matrices
        for idx in range(data.num_matrices):
            iso_r = data.isotope_rows[idx]
            mt_r = data.reaction_rows[idx]
            iso_c = data.isotope_cols[idx]
            mt_c = data.reaction_cols[idx]
            matrix = data.matrices[idx].copy()

            # Flip to descending if input ascending
            if ascending_input:
                matrix = np.flipud(np.fliplr(matrix))

            # Header: iso_row, mt_row, iso_col, mt_col, 1
            f.write(
                f"{iso_r:12d}{mt_r:12d}{iso_c:12d}{mt_c:12d}{1:12d}\n"
            )

            # Dense values: ngroup² values, 5 per line, E15.7
            flat = matrix.flatten().tolist()
            _write_e15_values(f, flat)


def write_coverx(
    data: CrossSectionCovariance, file_path: str, fmt: str = 'binary',
    title: str = '', endian: str = '>',
) -> None:
    """
    Write a :class:`CrossSectionCovariance` to a COVERX covariance file (text or binary).

    Parameters
    ----------
    data : CrossSectionCovariance
        Covariance data to write.
    file_path : str
        Output file path.
    fmt : str, optional
        ``'binary'`` (default) or ``'text'``.
    title : str, optional
        File title / description.
    endian : str, optional
        Endianness for binary format: ``'>'`` big-endian (default).
    """
    if isinstance(data, LegendreCovariance):
        raise TypeError(
            "COVERX format does not support MF34 (angular distribution) covariances. "
            "Use write_covfil() instead."
        )
    if fmt == 'text':
        _write_coverx_text(data, file_path, title=title)
    else:
        _write_coverx_binary(data, file_path, title=title, endian=endian)


# ---------------------------------------------------------------------------
# BOXER format constants
# ---------------------------------------------------------------------------

# Value field formats: nvf -> (values_per_line, field_width)
# Fortran formats from NJOY BOXR subroutine
_BOXER_VALUE_FORMATS = {
    7:  (11, 7),   # (11F7.4)
    8:  (10, 8),   # (10F8.5)
    9:  (8,  9),   # (1P8E9.2)
    10: (8,  10),  # (1P8E10.3) — NJOY default
    11: (7,  11),  # (1P7E11.4)
    12: (6,  12),  # (1P6E12.5)
    13: (6,  13),  # (1P6E13.6)
    14: (5,  14),  # (1P5E14.7)
}

# Control integer field formats: ncf -> (values_per_line, field_width)
_BOXER_CONTROL_FORMATS = {
    1: (80, 1), 2: (40, 2), 3: (26, 3),
    4: (20, 4), 5: (16, 5), 6: (13, 6),
}


# ---------------------------------------------------------------------------
# BOXER header parsing / formatting
# ---------------------------------------------------------------------------

def _parse_boxer_header(line: str) -> dict:
    """
    Parse a BOXER 80-character header card.

    Fortran format: ``(I1, A3, 8A4, 2(I5,I4), 2(I4,I3), 3I4)``

    Returns
    -------
    dict
        Keys: itype, hlib, hdescr, mat, mt, mat1, mt1,
              nval, nvf, ncon, ncf, nrowm, nrowh, ncolh
    """
    # Pad line to at least 80 chars
    line = line.rstrip('\n').ljust(80)
    return {
        'itype':  int(line[0:1].strip() or 0),
        'hlib':   line[1:4],
        'hdescr': line[4:36],
        'mat':    int(line[36:41].strip() or 0),
        'mt':     int(line[41:45].strip() or 0),
        'mat1':   int(line[45:50].strip() or 0),
        'mt1':    int(line[50:54].strip() or 0),
        'nval':   int(line[54:58].strip() or 0),
        'nvf':    int(line[58:61].strip() or 0),
        'ncon':   int(line[61:65].strip() or 0),
        'ncf':    int(line[65:68].strip() or 0),
        'nrowm':  int(line[68:72].strip() or 0),
        'nrowh':  int(line[72:76].strip() or 0),
        'ncolh':  int(line[76:80].strip() or 0),
    }


def _format_boxer_header(
    itype: int, mat: int, mt: int, mat1: int, mt1: int,
    nval: int, nvf: int, ncon: int, ncf: int,
    nrowm: int, nrowh: int, ncolh: int,
    hlib: str = '', hdescr: str = '',
) -> str:
    """
    Format a BOXER 80-character header card.

    Fortran format: ``(I1, A3, 8A4, 2(I5,I4), 2(I4,I3), 3I4)``
    """
    text = f"{itype:1d}"
    text += f"{hlib:3s}"
    text += f"{hdescr:32s}"
    text += f"{mat:5d}{mt:4d}{mat1:5d}{mt1:4d}"
    text += f"{nval:4d}{nvf:3d}{ncon:4d}{ncf:3d}"
    text += f"{nrowm:4d}{nrowh:4d}{ncolh:4d}"
    return text


# ---------------------------------------------------------------------------
# BOXER value / control integer I/O
# ---------------------------------------------------------------------------

def _read_boxer_values(lines: list, cursor: int, nval: int, nvf: int):
    """
    Read *nval* floating-point values from BOXER value lines.

    Returns ``(values, new_cursor)``.
    """
    if nval == 0:
        return [], cursor
    vpl, width = _BOXER_VALUE_FORMATS[nvf]
    values = []
    while len(values) < nval:
        line = lines[cursor].rstrip('\n')
        for i in range(vpl):
            if len(values) >= nval:
                break
            field = line[i * width:(i + 1) * width]
            if not field.strip():
                values.append(0.0)
            else:
                values.append(float(field))
        cursor += 1
    return values, cursor


def _read_boxer_controls(lines: list, cursor: int, ncon: int, ncf: int):
    """
    Read *ncon* control integers from BOXER control lines.

    Returns ``(controls, new_cursor)``.
    """
    if ncon == 0:
        return [], cursor
    vpl, width = _BOXER_CONTROL_FORMATS[ncf]
    controls = []
    while len(controls) < ncon:
        line = lines[cursor].rstrip('\n')
        for i in range(vpl):
            if len(controls) >= ncon:
                break
            field = line[i * width:(i + 1) * width]
            if not field.strip():
                controls.append(0)
            else:
                controls.append(int(field))
        cursor += 1
    return controls, cursor


def _write_boxer_values(f, values: list, nvf: int) -> None:
    """Write floating-point values using BOXER value format *nvf*."""
    vpl, width = _BOXER_VALUE_FORMATS[nvf]
    # Determine Fortran-style format
    if nvf <= 8:
        # Fw.d format (fixed-point)
        decimals = width - 3  # sign + digit + dot
    else:
        # 1PEw.d format (scientific): sign + digit + dot + d + E + sign + 2-digit exp = 7 + d
        decimals = width - 7
    buf = []
    for i, v in enumerate(values):
        if nvf <= 8:
            s = f"{v:{width}.{decimals}f}"
        else:
            s = f"{v:>{width}.{decimals}E}"
        buf.append(s)
        if len(buf) == vpl:
            f.write(''.join(buf) + '\n')
            buf = []
    if buf:
        f.write(''.join(buf) + '\n')


def _write_boxer_controls(f, controls: list, ncf: int) -> None:
    """Write control integers using BOXER control format *ncf*."""
    vpl, width = _BOXER_CONTROL_FORMATS[ncf]
    buf = []
    for c in controls:
        buf.append(f"{c:{width}d}")
        if len(buf) == vpl:
            f.write(''.join(buf) + '\n')
            buf = []
    if buf:
        f.write(''.join(buf) + '\n')


# ---------------------------------------------------------------------------
# BOXER decompression (from NJOY BOXR algorithm)
# ---------------------------------------------------------------------------

def _decompress_boxer(
    xval: list, icons: list,
    nrowh: int, ncolh: int,
    istart: int = 0, matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Decompress BOXER compressed data into a dense matrix.

    Parameters
    ----------
    xval : list of float
        Compressed values.
    icons : list of int
        Control integers (positive = carry-down, negative = new value).
    nrowh : int
        Total number of rows in the matrix.
    ncolh : int
        Number of columns (0 → symmetric, matrix is nrowh×nrowh).
    istart : int
        Starting row for continuation blocks.
    matrix : np.ndarray or None
        Existing matrix for continuation; created if None.

    Returns
    -------
    np.ndarray
        Decompressed matrix.
    """
    symmetric = (ncolh == 0)
    ncol = nrowh if symmetric else ncolh
    if matrix is None:
        matrix = np.zeros((nrowh, ncol))

    i = istart
    j = -1
    if i > 0 and symmetric:
        j = i - 1  # symmetric: first column of row i is column i

    iv = 0  # index into xval

    for ic in icons:
        if ic < 0:
            nload = -ic
            cload = xval[iv]
            iv += 1
            is_new = True
        else:
            nload = ic
            is_new = False

        for _ in range(nload):
            j += 1
            if j >= ncol:
                i += 1
                j = i if symmetric else 0

            if not is_new:
                # carry-down: copy from row above
                if i > istart:
                    cload = matrix[i - 1, j]
                else:
                    cload = 0.0

            matrix[i, j] = cload
            if symmetric and i != j:
                matrix[j, i] = cload

    return matrix


# ---------------------------------------------------------------------------
# BOXER compression
# ---------------------------------------------------------------------------

def _compress_boxer(matrix: np.ndarray, symmetric: bool = True):
    """
    Compress a dense matrix into BOXER format.

    Parameters
    ----------
    matrix : np.ndarray
        Dense matrix to compress.
    symmetric : bool
        If True, only traverse lower triangle.

    Returns
    -------
    xval : list of float
        Compressed values.
    icons : list of int
        Control integers.
    """
    nrow, ncol = matrix.shape
    xval = []
    icons = []

    # Build traversal order
    elements = []
    for i in range(nrow):
        jstart = i if symmetric else 0
        for j in range(jstart, ncol):
            above = matrix[i - 1, j] if i > 0 else 0.0
            elements.append((i, j, matrix[i, j], above))

    idx = 0
    while idx < len(elements):
        i, j, val, above = elements[idx]
        if np.isclose(val, above, rtol=0, atol=1e-30):
            # Carry-down run
            count = 0
            while idx < len(elements):
                ei, ej, ev, ea = elements[idx]
                if np.isclose(ev, ea, rtol=0, atol=1e-30):
                    count += 1
                    idx += 1
                else:
                    break
            icons.append(count)
        else:
            # New-value run: count consecutive elements with the same value
            run_val = val
            count = 0
            scan = idx
            while scan < len(elements):
                si, sj, sv, sa = elements[scan]
                if np.isclose(sv, run_val, rtol=0, atol=1e-30) and not np.isclose(sv, sa, rtol=0, atol=1e-30):
                    count += 1
                    scan += 1
                else:
                    break
            icons.append(-count)
            xval.append(run_val)
            idx += count

    return xval, icons


def _choose_boxer_ncf(controls: list) -> int:
    """Pick the smallest NCF that fits the maximum |control value|."""
    if not controls:
        return 4
    max_pos = max((c for c in controls if c > 0), default=0)
    max_neg_abs = max((abs(c) for c in controls if c < 0), default=0)
    has_neg = max_neg_abs > 0
    for ncf in sorted(_BOXER_CONTROL_FORMATS.keys()):
        _, width = _BOXER_CONTROL_FORMATS[ncf]
        # Positive: up to 10^width - 1
        max_pos_repr = 10 ** width - 1
        # Negative: sign takes 1 char, so up to -(10^(width-1) - 1)
        max_neg_repr = 10 ** (width - 1) - 1 if width > 1 else 0
        if max_pos <= max_pos_repr and (not has_neg or max_neg_abs <= max_neg_repr):
            return ncf
    return 6  # largest available


# ---------------------------------------------------------------------------
# Public API — BOXER format
# ---------------------------------------------------------------------------

def read_boxer(file_path: str, energy_unit: str = 'eV') -> CrossSectionCovariance:
    """
    Read a BOXER card-image (ASCII) covariance file and return a CrossSectionCovariance.

    BOXER is produced by NJOY's COVR module. It stores energy boundaries,
    cross-section vectors, standard-deviation vectors, and covariance or
    correlation matrices in a compressed card-image format.

    Parameters
    ----------
    file_path : str
        Path to the BOXER file.
    energy_unit : str, optional
        Energy unit for the energy grid: ``'eV'`` (default) or ``'MeV'``.

    Returns
    -------
    CrossSectionCovariance
        Parsed covariance data.

    Raises
    ------
    EmptyParsingError
        If no valid covariance matrices were found.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    covmat = CrossSectionCovariance(energy_unit=energy_unit)
    stddevs: Dict[Tuple[int, int], np.ndarray] = {}  # for correlation→covariance
    cursor = 0

    # Current matrix being assembled (for continuation blocks)
    cur_matrix = None
    cur_mat = cur_mt = cur_mat1 = cur_mt1 = 0
    cur_nrowh = cur_ncolh = 0
    cur_itype = 0
    last_row = -1

    while cursor < len(lines):
        line = lines[cursor]
        if len(line.strip()) == 0:
            cursor += 1
            continue

        hdr = _parse_boxer_header(line)
        cursor += 1

        itype = hdr['itype']
        mat = hdr['mat']
        mt = hdr['mt']
        mat1 = hdr['mat1']
        mt1 = hdr['mt1']
        nval = hdr['nval']
        nvf = hdr['nvf']
        ncon = hdr['ncon']
        ncf = hdr['ncf']
        nrowm = hdr['nrowm']
        nrowh = hdr['nrowh']
        ncolh = hdr['ncolh']

        # Read values and controls
        xval, cursor = _read_boxer_values(lines, cursor, nval, nvf)
        icons, cursor = _read_boxer_controls(lines, cursor, ncon, ncf)

        if itype == 0:
            # Energy boundaries
            # Decompress the vector
            vec = _decompress_boxer(xval, icons, nrowh, ncolh if ncolh > 0 else 1)
            energy_grid = vec[:, 0].tolist() if vec.shape[1] == 1 else vec[0, :].tolist()
            covmat.energy_grid = energy_grid
            covmat.num_groups = len(energy_grid) - 1

        elif itype == 1:
            # Cross-section vector
            vec = _decompress_boxer(xval, icons, nrowh, ncolh if ncolh > 0 else 1)
            xs = vec[:, 0] if vec.shape[1] == 1 else vec[0, :]
            zaid = int(_map_mat(str(mat)))
            covmat.cross_sections[(zaid, mt)] = xs

        elif itype == 2:
            # Standard-deviation vector
            vec = _decompress_boxer(xval, icons, nrowh, ncolh if ncolh > 0 else 1)
            sd = vec[:, 0] if vec.shape[1] == 1 else vec[0, :]
            zaid = int(_map_mat(str(mat)))
            stddevs[(zaid, mt)] = sd

        elif itype in (3, 4):
            # Covariance (3) or correlation (4) matrix
            if nrowm == 0:
                # First block (or single block)
                cur_matrix = _decompress_boxer(xval, icons, nrowh, ncolh)
                cur_mat, cur_mt, cur_mat1, cur_mt1 = mat, mt, mat1, mt1
                cur_nrowh, cur_ncolh = nrowh, ncolh
                cur_itype = itype
                # Determine last row filled
                symmetric = (ncolh == 0)
                ncol_eff = nrowh if symmetric else ncolh
                total_elements = sum(abs(c) for c in icons)
                # Calculate last row from element count
                if symmetric:
                    last_row = _boxer_last_row_sym(total_elements, nrowh)
                else:
                    last_row = (total_elements // ncol_eff) - 1 + 0  # 0-based
            else:
                # Continuation block — nrowm is the starting row (1-based)
                istart = nrowm
                cur_matrix = _decompress_boxer(
                    xval, icons, cur_nrowh, cur_ncolh,
                    istart=istart, matrix=cur_matrix,
                )

            # Check if matrix is complete: last continuation has data through the end
            # A block is the final one if there are no more continuation blocks
            # Peek at next header to see if it continues
            is_final = True
            if cursor < len(lines) and len(lines[cursor].strip()) > 0:
                next_hdr = _parse_boxer_header(lines[cursor])
                if next_hdr['nrowm'] > 0 and next_hdr['itype'] == itype:
                    if (next_hdr['mat'] == mat and next_hdr['mt'] == mt and
                            next_hdr['mat1'] == mat1 and next_hdr['mt1'] == mt1):
                        is_final = False

            if is_final and cur_matrix is not None:
                zaid_row = int(_map_mat(str(cur_mat)))
                zaid_col = int(_map_mat(str(cur_mat1)))

                if cur_itype == 4:
                    # Correlation → covariance
                    sd_row = stddevs.get((zaid_row, cur_mt))
                    sd_col = stddevs.get((zaid_col, cur_mt1))
                    if sd_row is not None and sd_col is not None:
                        cur_matrix = cur_matrix * np.outer(sd_row, sd_col)

                if not np.isclose(cur_matrix.sum(), 0.0):
                    covmat.add_matrix(zaid_row, cur_mt, zaid_col, cur_mt1, cur_matrix)

                cur_matrix = None

    if covmat.num_matrices == 0:
        raise EmptyParsingError(
            f"No valid data was extracted from the BOXER file: {file_path}"
        )
    return covmat


def _boxer_last_row_sym(total_elements: int, nrowh: int) -> int:
    """Determine the last row filled in a symmetric traversal given *total_elements*."""
    count = 0
    for i in range(nrowh):
        row_elems = nrowh - i  # elements in row i of lower triangle
        if count + row_elems > total_elements:
            return i
        count += row_elems
        if count >= total_elements:
            return i
    return nrowh - 1


def write_boxer(
    data: CrossSectionCovariance, file_path: str,
    hlibid: str = '', hdescr: str = '',
    nvf: int = 10,
) -> None:
    """
    Write a :class:`CrossSectionCovariance` to a BOXER card-image (ASCII) file.

    Parameters
    ----------
    data : CrossSectionCovariance
        Covariance data to write.
    file_path : str
        Output file path.
    hlibid : str, optional
        Library identifier (3 chars max). Default ``''``.
    hdescr : str, optional
        Description (32 chars max). Default ``''``.
    nvf : int, optional
        Value format code (7–14). Default 10 (``1P8E10.3``).
    """
    if isinstance(data, LegendreCovariance):
        raise TypeError(
            "BOXER format does not support MF34 (angular distribution) covariances. "
            "Use write_covfil() instead."
        )

    # Check for MAT overflow (BOXER I5 field: max 99999)
    bad_zaids = set()
    for idx in range(data.num_matrices):
        for zaid in [data.isotope_rows[idx], data.isotope_cols[idx]]:
            mat_val = _zaid_to_mat(zaid)
            if mat_val > 99999:
                bad_zaids.add(zaid)
    if bad_zaids:
        warnings.warn(
            f"BOXER MAT field is 5 characters (max 99999). "
            f"The following isotope IDs exceed this limit and will produce "
            f"corrupted output: {sorted(bad_zaids)}. "
            f"Use write_coverx() for these isotopes.",
            stacklevel=2,
        )

    if nvf <= 10:
        warnings.warn(
            f"BOXER NVF={nvf} provides ~{nvf - 7} significant figures. "
            f"Consider nvf=14 for higher precision.",
            stacklevel=2,
        )

    ngrp = data.num_groups
    energy_grid = list(data.energy_grid) if data.energy_grid else []

    # Convert MeV to eV if needed
    if data.energy_unit == 'MeV':
        energy_grid = [e * 1e6 for e in energy_grid]
        warnings.warn(
            "Energy grid converted from MeV to eV for BOXER output.",
            stacklevel=2,
        )

    with open(file_path, 'w') as f:
        # --- ITYPE=0: Energy boundaries ---
        if energy_grid:
            n_energies = len(energy_grid)
            # Store as a column vector: nrowh=n_energies, ncolh=1
            egrid_matrix = np.array(energy_grid).reshape(n_energies, 1)
            xval, icons = _compress_boxer(egrid_matrix, symmetric=False)
            ncf = _choose_boxer_ncf(icons)
            header = _format_boxer_header(
                itype=0, mat=0, mt=0, mat1=0, mt1=0,
                nval=len(xval), nvf=nvf, ncon=len(icons), ncf=ncf,
                nrowm=0, nrowh=n_energies, ncolh=1,
                hlib=hlibid[:3], hdescr=hdescr[:32],
            )
            f.write(header + '\n')
            _write_boxer_values(f, xval, nvf)
            _write_boxer_controls(f, icons, ncf)

        # --- ITYPE=1/2: Cross-sections and std-devs for diagonal blocks ---
        written_xs = set()
        for idx in range(data.num_matrices):
            iso_r = data.isotope_rows[idx]
            mt_r = data.reaction_rows[idx]
            iso_c = data.isotope_cols[idx]
            mt_c = data.reaction_cols[idx]
            mat_r = _zaid_to_mat(iso_r)
            mat_c = _zaid_to_mat(iso_c)

            # Write cross-section for row reaction (if available and not yet written)
            for zaid, mt_val, mat_val in [(iso_r, mt_r, mat_r), (iso_c, mt_c, mat_c)]:
                key = (zaid, mt_val)
                if key not in written_xs and key in data.cross_sections:
                    xs = data.cross_sections[key]
                    xs_matrix = np.array(xs).reshape(len(xs), 1)
                    xval, icons = _compress_boxer(xs_matrix, symmetric=False)
                    ncf = _choose_boxer_ncf(icons)
                    header = _format_boxer_header(
                        itype=1, mat=mat_val, mt=mt_val, mat1=0, mt1=0,
                        nval=len(xval), nvf=nvf, ncon=len(icons), ncf=ncf,
                        nrowm=0, nrowh=len(xs), ncolh=1,
                        hlib=hlibid[:3], hdescr=hdescr[:32],
                    )
                    f.write(header + '\n')
                    _write_boxer_values(f, xval, nvf)
                    _write_boxer_controls(f, icons, ncf)
                    written_xs.add(key)

            # Write std-dev for diagonal blocks
            if iso_r == iso_c and mt_r == mt_c:
                matrix = data.matrices[idx]
                diag = np.sqrt(np.maximum(np.diag(matrix), 0.0))
                sd_matrix = diag.reshape(len(diag), 1)
                xval, icons = _compress_boxer(sd_matrix, symmetric=False)
                ncf = _choose_boxer_ncf(icons)
                header = _format_boxer_header(
                    itype=2, mat=mat_r, mt=mt_r, mat1=0, mt1=0,
                    nval=len(xval), nvf=nvf, ncon=len(icons), ncf=ncf,
                    nrowm=0, nrowh=len(diag), ncolh=1,
                    hlib=hlibid[:3], hdescr=hdescr[:32],
                )
                f.write(header + '\n')
                _write_boxer_values(f, xval, nvf)
                _write_boxer_controls(f, icons, ncf)

        # --- ITYPE=3: Covariance matrices ---
        for idx in range(data.num_matrices):
            iso_r = data.isotope_rows[idx]
            mt_r = data.reaction_rows[idx]
            iso_c = data.isotope_cols[idx]
            mt_c = data.reaction_cols[idx]
            mat_r = _zaid_to_mat(iso_r)
            mat_c = _zaid_to_mat(iso_c)
            matrix = data.matrices[idx]

            # Determine symmetry
            symmetric = (iso_r == iso_c and mt_r == mt_c)
            ncolh_val = 0 if symmetric else ngrp

            xval, icons = _compress_boxer(matrix, symmetric=symmetric)
            ncf = _choose_boxer_ncf(icons)
            header = _format_boxer_header(
                itype=3, mat=mat_r, mt=mt_r, mat1=mat_c, mt1=mt_c,
                nval=len(xval), nvf=nvf, ncon=len(icons), ncf=ncf,
                nrowm=0, nrowh=ngrp, ncolh=ncolh_val,
                hlib=hlibid[:3], hdescr=hdescr[:32],
            )
            f.write(header + '\n')
            _write_boxer_values(f, xval, nvf)
            _write_boxer_controls(f, icons, ncf)


# ---------------------------------------------------------------------------
# Public API — COVFIL / GENDF (NJOY format)
# ---------------------------------------------------------------------------

def _read_n_values(lines: List[str], cursor: int, n: int) -> Tuple[List[float], int]:
    """
    Read exactly *n* ENDF float values from consecutive lines starting at *cursor*.

    Each 80-column ENDF line carries up to 6 values in 11-char fields (columns 1-66).
    Uses :func:`parse_number` for robust parsing of ENDF scientific notation.

    Returns
    -------
    values : list of float
        Parsed values (length *n*).
    new_cursor : int
        Line index immediately after the last consumed line.
    """
    values: List[float] = []
    while len(values) < n:
        line = lines[cursor]
        for i in range(6):
            if len(values) >= n:
                break
            field = line[i * 11:(i + 1) * 11]
            val = parse_number(field)
            if val is None:
                val = 0.0
            values.append(float(val))
        cursor += 1
    return values, cursor


def _parse_covfil_mf1(lines: List[str], cursor: int) -> Tuple[float, float, int, List[float], int]:
    """
    Parse MF1 MT451 energy-grid section of a COVFIL file.

    Returns
    -------
    za : float
        ZA identifier (Z*1000 + A).
    awr : float
        Atomic weight ratio.
    ngrp : int
        Number of energy groups.
    energy_grid : list of float
        Energy-group boundaries (length ngrp+1).
    new_cursor : int
        Line index after the MF1 block (past SEND/FEND).
    """
    # HEAD record
    head = parse_line(lines[cursor])
    za = float(head['C1'] or 0)
    awr = float(head['C2'] or 0)
    cursor += 1

    # CONT record — contains NGRP in C3, NGRP+1 in C5
    cont = parse_line(lines[cursor])
    ngrp = int(cont['C3'] or 0)
    n_energies = int(cont['C5'] or 0)
    cursor += 1

    # Read energy boundaries
    energy_grid, cursor = _read_n_values(lines, cursor, n_energies)

    # Skip SEND (MT=0) and FEND (MF=0) lines
    while cursor < len(lines):
        _, mf, mt = parse_endf_id(lines[cursor])
        cursor += 1
        if mf == 0:  # FEND
            break
    return za, awr, ngrp, energy_grid, cursor


def _parse_covfil_mf3(
    lines: List[str], cursor: int, ngrp: int
) -> Tuple[Dict[Tuple[int, int], np.ndarray], int]:
    """
    Parse all MF3 cross-section sections until FEND (MF=0).

    Returns
    -------
    xs_dict : dict
        ``{(zaid, mt): np.ndarray}`` with group-averaged cross sections.
    new_cursor : int
        Line index after the MF3 FEND line.
    """
    xs_dict: Dict[Tuple[int, int], np.ndarray] = {}
    while cursor < len(lines):
        mat, mf, mt = parse_endf_id(lines[cursor])
        # FEND — end of MF3
        if mf == 0:
            cursor += 1
            break
        # SEND — skip
        if mt == 0:
            cursor += 1
            continue

        # HEAD for this MT section: (ZA, 0, 0, 0, NGRP, 0)
        head = parse_line(lines[cursor])
        cursor += 1
        n_vals = int(head['C5'] or ngrp)

        # Read cross-section values
        values, cursor = _read_n_values(lines, cursor, n_vals)

        zaid = int(_map_mat(str(mat)))
        xs_dict[(zaid, mt)] = np.array(values)
    return xs_dict, cursor


def _parse_covfil_mf33(
    lines: List[str], cursor: int, ngrp: int, mat: int
) -> Tuple[List[Tuple[int, int, int, int, np.ndarray]], int]:
    """
    Parse all MF33 covariance sections until FEND.

    Returns
    -------
    blocks : list of (iso_row, mt_row, iso_col, mt_col, matrix)
    new_cursor : int
    """
    zaid = int(_map_mat(str(mat)))
    blocks: List[Tuple[int, int, int, int, np.ndarray]] = []

    while cursor < len(lines):
        rmat, mf, mt = parse_endf_id(lines[cursor])
        # FEND
        if mf == 0:
            cursor += 1
            break
        # SEND
        if mt == 0:
            cursor += 1
            continue

        # HEAD record for this MT section
        head = parse_line(lines[cursor])
        nmt1 = int(head['C6'] or 0)
        mt_row = mt
        cursor += 1

        # Loop over NMT1 reaction pairs
        for _ in range(nmt1):
            # CONT: (0, 0, MAT1, MT1, 0, NG)
            cont = parse_line(lines[cursor])
            mat1 = int(cont['C3'] or 0)
            mt1 = int(cont['C4'] or 0)
            ng = int(cont['C6'] or ngrp)
            cursor += 1

            iso_col = int(_map_mat(str(mat1))) if mat1 != 0 else zaid

            sub_mat = np.zeros((ng, ng))

            # Read row-by-row LIST records until IROW == NG or next CONT/SEND
            while cursor < len(lines):
                # Peek: is it a row header or the next CONT/SEND?
                peek_mat, peek_mf, peek_mt = parse_endf_id(lines[cursor])
                if peek_mt == 0 or peek_mf == 0:
                    break  # SEND or FEND — done with this pair

                row_head = parse_line(lines[cursor])
                nw = int(row_head['C3'] or 0)
                istart = int(row_head['C4'] or 1)
                irow = int(row_head['C6'] or 1)
                cursor += 1

                if nw > 0:
                    vals, cursor = _read_n_values(lines, cursor, nw)
                    sub_mat[irow - 1, istart - 1: istart - 1 + nw] = vals

                # Check if this was a CONT for the next pair
                # (next line could be another row header or a new CONT)
                if cursor < len(lines):
                    next_head = parse_line(lines[cursor])
                    next_mat, next_mf, next_mt = parse_endf_id(lines[cursor])
                    # If next line is a CONT (C3≠0 means MAT1, C5=0, C4=MT1 pattern)
                    # or a SEND/FEND, we're done with this pair
                    if next_mt == 0 or next_mf == 0:
                        break
                    # Detect if next line is a new CONT (not a row header)
                    # Row headers have C3=NW>0 or C3=0 and C4>0(ISTART) and C6>0(IROW)
                    # CONTs for pairs have C4=MT1>0 and C5=0 and C6=NG
                    # Key difference: row headers have C3=NW, C4=ISTART, C5=NW(dup), C6=IROW
                    # CONTs have C3=MAT1, C4=MT1, C5=0, C6=NG
                    # If C5==0 and C3 is small and C4 is large (MT number), it's a CONT
                    # But the safest check: if irow == ng, we're done with this pair
                    if irow == ng:
                        break

            if not np.isclose(sub_mat.sum(), 0.0):
                blocks.append((zaid, mt_row, iso_col, mt1, sub_mat))

    return blocks, cursor


def _parse_covfil_mf34(
    lines: List[str], cursor: int, ngrp: int, mat: int, energy_grid: List[float]
) -> Tuple[List[Tuple[int, int, int, int, int, int, np.ndarray]], int]:
    """
    Parse all MF34 covariance sections until FEND.

    Returns
    -------
    blocks : list of (iso_row, mt_row, l_row, iso_col, mt_col, l_col, matrix)
    new_cursor : int
    """
    zaid = int(_map_mat(str(mat)))
    blocks: List[Tuple[int, int, int, int, int, int, np.ndarray]] = []

    while cursor < len(lines):
        rmat, mf, mt = parse_endf_id(lines[cursor])
        # FEND
        if mf == 0:
            cursor += 1
            break
        # SEND
        if mt == 0:
            cursor += 1
            continue

        # HEAD: (ZA, AWR, LTT, NMT1, NL, NL1)
        head = parse_line(lines[cursor])
        nmt1 = int(head['C4'] or 0)
        mt_row = mt
        cursor += 1

        # Loop over reaction pairs (including L/L1 combinations)
        for _ in range(nmt1):
            # CONT: (0, 0, MT1, L, L1, NG)
            cont = parse_line(lines[cursor])
            mt1 = int(cont['C3'] or 0)
            l_row = int(cont['C4'] or 0)
            l_col = int(cont['C5'] or 0)
            ng = int(cont['C6'] or ngrp)
            cursor += 1

            sub_mat = np.zeros((ng, ng))

            # Read row-by-row LIST records
            while cursor < len(lines):
                peek_mat, peek_mf, peek_mt = parse_endf_id(lines[cursor])
                if peek_mt == 0 or peek_mf == 0:
                    break

                row_head = parse_line(lines[cursor])
                nw = int(row_head['C3'] or 0)
                istart = int(row_head['C4'] or 1)
                irow = int(row_head['C6'] or 1)
                cursor += 1

                if nw > 0:
                    vals, cursor = _read_n_values(lines, cursor, nw)
                    sub_mat[irow - 1, istart - 1: istart - 1 + nw] = vals

                if cursor < len(lines):
                    next_mat, next_mf, next_mt = parse_endf_id(lines[cursor])
                    if next_mt == 0 or next_mf == 0:
                        break
                    if irow == ng:
                        break

            if not np.isclose(sub_mat.sum(), 0.0):
                blocks.append((zaid, mt_row, l_row, zaid, mt1, l_col, sub_mat))

    return blocks, cursor


def read_covfil(file_path: str, energy_unit: str = 'eV') -> Union[CrossSectionCovariance, LegendreCovariance]:
    """
    Parse an NJOY-generated COVFIL/GENDF covariance file.

    Returns a :class:`CrossSectionCovariance` for MF33 files (cross-section covariances) or an
    :class:`LegendreCovariance` for MF34 files (angular-distribution covariances).

    MF3 cross sections are stored in ``CrossSectionCovariance.cross_sections`` (MF33 case only).

    Parameters
    ----------
    file_path : str
        Path to the NJOY-generated covariance file.
    energy_unit : str, optional
        Energy unit for the energy grid: ``'eV'`` (default) or ``'MeV'``.

    Returns
    -------
    CrossSectionCovariance or LegendreCovariance
        Parsed covariance data.

    Raises
    ------
    EmptyParsingError
        If no valid covariance matrices were found.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Line 0 is tape header — skip it
    cursor = 1

    # --- MF1 MT451: energy grid ---
    za, awr, ngrp, energy_grid, cursor = _parse_covfil_mf1(lines, cursor)
    mat, _, _ = parse_endf_id(lines[1])  # MAT from first data line

    # --- MF3: cross sections (optional) ---
    xs_dict: Dict[Tuple[int, int], np.ndarray] = {}
    if cursor < len(lines):
        peek_mat, peek_mf, peek_mt = parse_endf_id(lines[cursor])
        if peek_mf == 3:
            xs_dict, cursor = _parse_covfil_mf3(lines, cursor, ngrp)

    # --- Determine covariance type: MF33 or MF34 ---
    if cursor < len(lines):
        peek_mat, peek_mf, peek_mt = parse_endf_id(lines[cursor])
    else:
        raise EmptyParsingError(
            f"No covariance data found in file: {file_path}"
        )

    if peek_mf == 33:
        blocks, cursor = _parse_covfil_mf33(lines, cursor, ngrp, mat)
        covmat = CrossSectionCovariance(num_groups=ngrp, energy_unit=energy_unit)
        covmat.energy_grid = energy_grid
        for iso_row, mt_row, iso_col, mt_col, matrix in blocks:
            covmat.add_matrix(iso_row, mt_row, iso_col, mt_col, matrix)
        covmat.cross_sections.update(xs_dict)
        if covmat.num_matrices == 0:
            raise EmptyParsingError(
                f"No valid data was extracted from the NJOY covariance matrix file: {file_path}"
            )
        return covmat

    elif peek_mf == 34:
        blocks34, cursor = _parse_covfil_mf34(lines, cursor, ngrp, mat, energy_grid)
        mf34 = LegendreCovariance(energy_unit=energy_unit)
        for iso_row, mt_row, l_row, iso_col, mt_col, l_col, matrix in blocks34:
            mf34.add_matrix(
                isotope_row=iso_row, reaction_row=mt_row, l_row=l_row,
                isotope_col=iso_col, reaction_col=mt_col, l_col=l_col,
                matrix=matrix, energy_grid=energy_grid,
                is_relative=True, frame="unknown",
            )
        # Store grouped Legendre coefficients from MF3 sections
        # NJOY convention: MT = 250 + L (MT251→a_1, MT252→a_2, …)
        for (zaid, mt_xs), values in xs_dict.items():
            l_order = mt_xs - 250
            if l_order >= 1:
                mf34.legendre_coefficients[(zaid, mt_xs, l_order)] = values
        if mf34.num_matrices == 0:
            raise EmptyParsingError(
                f"No valid data was extracted from the NJOY covariance matrix file: {file_path}"
            )
        return mf34

    else:
        raise EmptyParsingError(
            f"Expected MF33 or MF34 covariance data, found MF={peek_mf} in: {file_path}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _map_mat(mat_str: str) -> str:
    """
    Returns the ZAID code corresponding to a MAT.
    If there is no entry in the dictionary, it is left as is.

    Parameters
    ----------
    mat_str : str
        The MAT field as read (it may contain spaces).

    Returns
    -------
    str
        MAT translated to ZAID, as a string, so that everything
        continues to flow exactly as before.
    """
    try:
        mat_int = int(mat_str.strip())
    except ValueError:
        return mat_str
    return str(ENDF_MAT_TO_ZAID.get(mat_int, mat_int))


def _zaid_to_mat(zaid: int) -> int:
    """Convert a ZAID code to an ENDF MAT number (reverse of ``_map_mat``)."""
    return ZAID_TO_ENDF_MAT.get(zaid, zaid)


# ---------------------------------------------------------------------------
# Write helpers — COVFIL / GENDF format
# ---------------------------------------------------------------------------

def _write_list_values(
    f, values: List[float], mat: int, mf: int, mt: int, seq: int
) -> int:
    """
    Write a sequence of floats in ENDF format (6 per line).

    Returns the next sequence number after writing.
    """
    for i in range(0, len(values), 6):
        chunk = values[i:i + 6]
        line = format_endf_data_line(chunk, mat, mf, mt, seq)
        f.write(line + '\n')
        seq += 1
    return seq


def _write_send(f, mat: int, mf: int, seq: int = 99999) -> None:
    """Write a SEND record (MT=0)."""
    f.write(format_endf_data_line([], mat, mf, 0, seq) + '\n')


def _write_fend(f, mat: int) -> None:
    """Write a FEND record (MF=0, MT=0)."""
    f.write(format_endf_data_line([], mat, 0, 0, 0) + '\n')


def _write_mend(f) -> None:
    """Write a MEND record (MAT=0)."""
    f.write(format_endf_data_line([], 0, 0, 0, 0) + '\n')


def _write_tend(f) -> None:
    """Write a TEND record (MAT=-1)."""
    f.write(format_endf_data_line([], -1, 0, 0, 0) + '\n')


def _write_covfil_mf33(covmat: CrossSectionCovariance, f, mat: int, za: float, awr: float) -> None:
    """Write all MF33 covariance sections for *covmat*."""
    ngrp = covmat.num_groups

    # Group matrices by MT (row reaction) to form sections
    from collections import defaultdict
    mt_groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(covmat.num_matrices):
        mt_groups[covmat.reaction_rows[idx]].append(idx)

    for mt_row, indices in mt_groups.items():
        nmt1 = len(indices)
        seq = 1

        # HEAD: (ZA, AWR, 0, 0, 0, NMT1)
        head_line = format_endf_data_line(
            [za, awr, 0, 0, 0, nmt1], mat, 33, mt_row, seq,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        f.write(head_line + '\n')
        seq += 1

        for idx in indices:
            iso_col = covmat.isotope_cols[idx]
            mt_col = covmat.reaction_cols[idx]
            matrix = covmat.matrices[idx]
            mat1 = _zaid_to_mat(iso_col) if iso_col != int(_map_mat(str(mat))) else 0

            # CONT: (0, 0, MAT1, MT1, 0, NG)
            cont_line = format_endf_data_line(
                [0, 0, mat1, mt_col, 0, ngrp], mat, 33, mt_row, seq,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
            )
            f.write(cont_line + '\n')
            seq += 1

            # Write rows
            for irow in range(ngrp):
                row = matrix[irow, :]
                # Find contiguous non-zero span
                nonzero_idx = np.nonzero(row)[0]
                if len(nonzero_idx) > 0:
                    istart = int(nonzero_idx[0]) + 1  # 1-based
                    iend = int(nonzero_idx[-1]) + 1
                    nw = iend - istart + 1
                    vals = row[istart - 1: istart - 1 + nw].tolist()
                elif irow == ngrp - 1:
                    # Last row always written as terminator
                    istart = ngrp
                    nw = 1
                    vals = [0.0]
                else:
                    continue  # skip all-zero rows (except last)

                # Row LIST header: (0, 0, NW, ISTART, NW, IROW)
                row_head = format_endf_data_line(
                    [0, 0, nw, istart, nw, irow + 1], mat, 33, mt_row, seq,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                )
                f.write(row_head + '\n')
                seq += 1
                seq = _write_list_values(f, vals, mat, 33, mt_row, seq)

        _write_send(f, mat, 33)


def _write_covfil_mf34(mf34: LegendreCovariance, f, mat: int, za: float, awr: float) -> None:
    """Write all MF34 covariance sections for *mf34*."""
    # Group matrices by MT (row reaction) to form sections
    from collections import defaultdict
    mt_groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(mf34.num_matrices):
        mt_groups[mf34.reaction_rows[idx]].append(idx)

    for mt_row, indices in mt_groups.items():
        nmt1 = len(indices)

        # Determine NL, NL1 from unique L/L1 values
        nl_set = set()
        nl1_set = set()
        for idx in indices:
            nl_set.add(mf34.l_rows[idx])
            nl1_set.add(mf34.l_cols[idx])
        nl = len(nl_set)
        nl1 = len(nl1_set)

        seq = 1
        # HEAD: (ZA, AWR, LTT=0, NMT1, NL, NL1)
        head_line = format_endf_data_line(
            [za, awr, 0, nmt1, nl, nl1], mat, 34, mt_row, seq,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        f.write(head_line + '\n')
        seq += 1

        for idx in indices:
            mt_col = mf34.reaction_cols[idx]
            l_row = mf34.l_rows[idx]
            l_col = mf34.l_cols[idx]
            matrix = mf34.matrices[idx]
            ngrp = matrix.shape[0]

            # CONT: (0, 0, MT1, L, L1, NG)
            cont_line = format_endf_data_line(
                [0, 0, mt_col, l_row, l_col, ngrp], mat, 34, mt_row, seq,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT],
            )
            f.write(cont_line + '\n')
            seq += 1

            # Write rows
            for irow in range(ngrp):
                row = matrix[irow, :]
                nonzero_idx = np.nonzero(row)[0]
                if len(nonzero_idx) > 0:
                    istart = int(nonzero_idx[0]) + 1
                    iend = int(nonzero_idx[-1]) + 1
                    nw = iend - istart + 1
                    vals = row[istart - 1: istart - 1 + nw].tolist()
                elif irow == ngrp - 1:
                    istart = ngrp
                    nw = 1
                    vals = [0.0]
                else:
                    continue

                row_head = format_endf_data_line(
                    [0, 0, nw, istart, nw, irow + 1], mat, 34, mt_row, seq,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                )
                f.write(row_head + '\n')
                seq += 1
                seq = _write_list_values(f, vals, mat, 34, mt_row, seq)

        _write_send(f, mat, 34)


def write_covfil(
    data: Union[CrossSectionCovariance, LegendreCovariance],
    file_path: str,
    tape_label: str = '',
    temperature: float = 0.0,
) -> None:
    """
    Write a :class:`CrossSectionCovariance` or :class:`LegendreCovariance` to an NJOY COVFIL/GENDF text file.

    Parameters
    ----------
    data : CrossSectionCovariance or LegendreCovariance
        Covariance data to write.
    file_path : str
        Output file path.
    tape_label : str, optional
        Label for the tape header line (max 66 chars). Default ``''``.
    temperature : float, optional
        Temperature in K written into MF1 MT451 CONT record. Default ``0.0``.
    """
    is_mf33 = isinstance(data, CrossSectionCovariance)

    # Check for MAT overflow (COVFIL 4-char field: max 9999)
    bad_zaids = set()
    for idx in range(data.num_matrices):
        for zaid in [data.isotope_rows[idx], data.isotope_cols[idx]]:
            mat_val = _zaid_to_mat(zaid)
            if mat_val > 9999:
                bad_zaids.add(zaid)
    if bad_zaids:
        warnings.warn(
            f"COVFIL MAT field is 4 characters (max 9999). "
            f"The following isotope IDs exceed this limit and will produce "
            f"corrupted output: {sorted(bad_zaids)}. "
            f"Use write_coverx() for these isotopes.",
            stacklevel=2,
        )

    # Determine MAT, ZA, AWR, NGRP
    if is_mf33:
        ngrp = data.num_groups
        energy_grid = list(data.energy_grid) if data.energy_grid else []
        # Infer ZA from the first isotope
        first_zaid = data.isotope_rows[0] if data.isotope_rows else 0
        za = float(first_zaid)
        mat = _zaid_to_mat(first_zaid)
        awr = 0.0  # will be approximated from ZA
    else:
        energy_grid = list(data.energy_grids[0]) if data.energy_grids else []
        ngrp = len(energy_grid) - 1 if energy_grid else 0
        first_zaid = data.isotope_rows[0] if data.isotope_rows else 0
        za = float(first_zaid)
        mat = _zaid_to_mat(first_zaid)
        awr = 0.0

    # Convert MeV to eV if needed
    if data.energy_unit == 'MeV':
        energy_grid = [e * 1e6 for e in energy_grid]
        warnings.warn(
            "Energy grid converted from MeV to eV for COVFIL output.",
            stacklevel=2,
        )

    with open(file_path, 'w') as f:
        # --- Tape header ---
        header = tape_label[:66].ljust(66) + f"{0:4d}{0:2d}{0:3d}{0:5d}"
        f.write(header + '\n')

        # --- MF1 MT451: energy grid ---
        seq = 1
        # HEAD: (ZA, AWR, ?, 0, -NGRP(?), 0)  — match NJOY convention
        head = format_endf_data_line(
            [za, awr, 6, 0, -ngrp, 0], mat, 1, 451, seq,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        f.write(head + '\n')
        seq += 1

        # CONT: (temperature, 0, NGRP, 0, NGRP+1, 0)
        n_energies = len(energy_grid)
        cont = format_endf_data_line(
            [temperature, 0.0, ngrp, 0, n_energies, 0], mat, 1, 451, seq,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )
        f.write(cont + '\n')
        seq += 1

        # Energy values
        seq = _write_list_values(f, energy_grid, mat, 1, 451, seq)

        _write_send(f, mat, 1)
        _write_fend(f, mat)

        # --- MF3: cross sections (CrossSectionCovariance only) ---
        if is_mf33 and data.cross_sections:
            for (xs_zaid, xs_mt), xs_vals in sorted(data.cross_sections.items()):
                seq = 1
                n = len(xs_vals)
                head3 = format_endf_data_line(
                    [za, 0.0, 0, 0, n, 0], mat, 3, xs_mt, seq,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                             ENDF_FORMAT_INT, ENDF_FORMAT_INT],
                )
                f.write(head3 + '\n')
                seq += 1
                seq = _write_list_values(f, xs_vals.tolist(), mat, 3, xs_mt, seq)
                _write_send(f, mat, 3)
            _write_fend(f, mat)

        # --- MF33 or MF34 covariance data ---
        if is_mf33:
            _write_covfil_mf33(data, f, mat, za, awr)
        else:
            _write_covfil_mf34(data, f, mat, za, awr)

        _write_fend(f, mat)
        _write_mend(f)
        _write_tend(f)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
read_scale_covmat = read_coverx
read_njoy_covmat = read_covfil
