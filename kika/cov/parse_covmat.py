import struct

import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple, List, Optional

from kika._constants import MT_TO_REACTION, ENDF_MAT_TO_ZAID
from kika.cov.covmat import CovMat
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

def read_coverx(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CovMat:
    """
    Read a COVERX covariance file (text or binary) and return a CovMat object.

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
    CovMat
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

def _read_coverx_text(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CovMat:
    """
    Read a SCALE covariance matrix text file and convert it to a CovMat object.

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
    CovMat
        CovMat object containing the parsed covariance data

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

    # Create CovMat object
    covmat = CovMat(num_groups, energy_unit=energy_unit)

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

                if (reaction_row != 1 and reaction_col != 1 and
                    reaction_row in MT_TO_REACTION and reaction_col in MT_TO_REACTION):

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

                    # Add to CovMat object
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

def _read_coverx_binary(file_path: str, ascending: bool = True, energy_unit: str = 'eV') -> CovMat:
    """
    Read a binary COVERX covariance file and return a CovMat object.

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
    CovMat
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

        # Records 6 … 6+nmmp-1: cross-section / error data per pair (skip)
        for _ in range(nmmp):
            _read_fortran_record(f, endian)

        # --- Covariance matrix blocks (nmtrix total) ----------------------
        covmat = CovMat(num_groups=ngroup, energy_unit=energy_unit)

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

            # Apply same MT filters as the text parser
            if (mt1 == 1 or mt2 == 1 or
                    mt1 not in MT_TO_REACTION or mt2 not in MT_TO_REACTION):
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
# Public API — COVFIL / GENDF (NJOY format)
# ---------------------------------------------------------------------------

def read_covfil(file_path: str, energy_unit: str = 'eV') -> CovMat:
    """
    Parse an NJOY-generated COVFIL/GENDF covariance file and return a CovMat instance.

    The routine stores MF3 cross sections in ``CovMat.cross_sections``
    instead of treating them as pseudo-matrices with REAC_V = 0.

    Parameters
    ----------
    file_path : str
        Path to the NJOY-generated covariance file
    energy_unit : str, optional
        Energy unit for the energy grid: ``'eV'`` (default) or ``'MeV'``

    Returns
    -------
    CovMat
        CovMat instance with parsed covariance data
    """

    dikt_cov = {'ISO_H': [], 'REAC_H': [],
                'ISO_V': [], 'REAC_V': [],
                'STD':    []}

    # Temporary store for cross-section vectors
    xs_dict: Dict[Tuple[int, int], np.ndarray] = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    iMAT1 = lines[2][66:70]
    group_nb = int(lines[2].split()[2])

    grep_data = False
    val_tot_nb = start_x_idx = start_y_idx = None
    vals: List[float] = []
    energymesh: Optional[List[float]] = None

    i_line = 0
    while i_line < (len(lines) - 4):
        i_line += 1
        line = lines[i_line]

        splited_part = [line[i * 11:(i + 1) * 11].replace(' ', '')
                        for i in range(6)]
        splited_part = [x for x in splited_part if x]

        infos_part = line[66:]
        iMAT = infos_part[:4]
        iMF = str(int(infos_part[4:6]))
        iMT = str(int(infos_part[6:9]))

        # SEND ------------------------------------------------------------
        if (iMAT, iMF) != ('0', '0') and iMT == '0':
            continue
        # FEND ------------------------------------------------------------
        if iMAT != '0' and (iMF, iMT) == ('0', '0'):
            continue

        # MF1 MT451 – energy grid ----------------------------------------
        if iMAT != '0' and iMF == '1' and iMT == '451':
            i_line += 1
            line = lines[i_line]
            LIST_MF1451 = [line[i * 11:(i + 1) * 11].replace(' ', '')
                           for i in range(6)]
            LIST_MF1451 = [x for x in LIST_MF1451 if x]
            energymesh = []
            while len(energymesh) < int(LIST_MF1451[4]):
                i_line += 1
                line = lines[i_line]
                energylist = [line[i * 11:(i + 1) * 11].replace(' ', '')
                              for i in range(6)]
                energylist = [x for x in energylist if x]
                for energy in energylist:
                    if re.search('-', energy[-3:]):
                        valE = energy[:-3] + energy[-3:].split('-')[0] + 'E-' + energy[-3:].split('-')[1]
                        valE = float(valE)
                    elif re.search(r'\+', energy):
                        valE = energy[:-3] + energy[-3:].split('+')[0] + 'E+' + energy[-3:].split('+')[1]
                        valE = float(valE)
                    else:
                        valE = float(energy)
                    energymesh.append(valE)
            # keep grid in place for later test
            dikt_cov['ISO_H'].append('0')
            dikt_cov['REAC_H'].append('0')
            dikt_cov['ISO_V'].append('0')
            dikt_cov['REAC_V'].append('0')
            dikt_cov['STD'].append(energymesh)
            continue

        # MF3 – point cross sections -------------------------------------
        if iMAT != '0' and iMF == '3' and iMT != '0':
            crossSectionLine: List[float] = []
            while len(crossSectionLine) < int(splited_part[4]):
                i_line += 1
                line = lines[i_line]
                LIST_MF3 = [line[i * 11:(i + 1) * 11].replace(' ', '')
                            for i in range(6)]
                LIST_MF3 = [x for x in LIST_MF3 if x]
                for xs_str in LIST_MF3:
                    if re.search('-', xs_str[-3:]):
                        valXS = xs_str[:-3] + xs_str[-3:].split('-')[0] + 'E-' + xs_str[-3:].split('-')[1]
                    elif re.search(r'\+', xs_str):
                        valXS = xs_str[:-3] + xs_str[-3:].split('+')[0] + 'E+' + xs_str[-3:].split('+')[1]
                    else:
                        valXS = xs_str
                    crossSectionLine.append(float(valXS))
            # store – NO entry in dikt_cov
            iso_num = _map_mat(iMAT)
            xs_dict[(int(iso_num), int(iMT))] = np.array(crossSectionLine)
            continue

        # MF33/34 – covariance blocks ------------------------------------
        if len(splited_part) > 4 and splited_part[2] == '0' and splited_part[4] == '0':
            reac_2_id = splited_part[3]
            grep_data = True
            sub_mat = np.zeros((group_nb, group_nb))
            continue

        elif grep_data:
            if (val_tot_nb, start_x_idx, start_y_idx) == (None, None, None):
                val_tot_nb = int(splited_part[2])
                start_x_idx = int(splited_part[3]) - 1
                start_y_idx = int(splited_part[5]) - 1
                continue

            for val_str in splited_part:
                if re.search('-', val_str[-3:]):
                    val = val_str[:-3] + val_str[-3:].split('-')[0] + 'E-' + val_str[-3:].split('-')[1]
                elif re.search(r'\+', val_str):
                    val = val_str[:-3] + val_str[-3:].split('+')[0] + 'E+' + val_str[-3:].split('+')[1]
                else:
                    val = val_str
                vals.append(float(val))

            if len(vals) == val_tot_nb:
                sub_mat[start_y_idx][start_x_idx:start_x_idx + val_tot_nb] = vals
                val_tot_nb = start_x_idx = start_y_idx = None
                vals = []
                if (len(lines[i_line + 1].split()) < 3
                        or lines[i_line + 1].split()[2] == '0'):
                    if not np.isclose(sub_mat.sum(), 0.0):
                        dikt_cov['ISO_H'].append(_map_mat(iMAT))
                        dikt_cov['REAC_H'].append(iMT)
                        dikt_cov['ISO_V'].append(_map_mat(iMAT1))
                        dikt_cov['REAC_V'].append(reac_2_id)
                        dikt_cov['STD'].append(sub_mat.tolist())
                    grep_data = False
                continue

    # ------------------------------------------------------------------
    # Build CovMat
    # ------------------------------------------------------------------
    covmat = CovMat(num_groups=group_nb, energy_unit=energy_unit)

    # energy grid --------------------------------------------------------
    if (dikt_cov['STD']
            and isinstance(dikt_cov['STD'][0], list)
            and len(dikt_cov['STD'][0]) == group_nb + 1):
        covmat.energy_grid = dikt_cov['STD'][0]
        start_idx = 1
    elif energymesh is not None:
        covmat.energy_grid = energymesh
        start_idx = 0
    else:
        start_idx = 0

    # covariance matrices ------------------------------------------------
    for idx in range(start_idx, len(dikt_cov['STD'])):
        try:
            iso_h = int(str(dikt_cov['ISO_H'][idx]).strip())
            reac_h = int(str(dikt_cov['REAC_H'][idx]).strip())
            iso_v = int(str(dikt_cov['ISO_V'][idx]).strip())
            reac_v = int(str(dikt_cov['REAC_V'][idx]).strip())
            matrix = np.array(dikt_cov['STD'][idx])
            if matrix.shape == (group_nb, group_nb):
                covmat.add_matrix(iso_h, reac_h, iso_v, reac_v, matrix)
        except Exception:
            continue

    # attach cross sections ---------------------------------------------
    covmat.cross_sections.update(xs_dict)

    if covmat.num_matrices == 0:
        raise EmptyParsingError(
            f"No valid data was extracted from the NJOY covariance matrix file: {file_path}"
        )

    return covmat


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


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
read_scale_covmat = read_coverx
read_njoy_covmat = read_covfil
