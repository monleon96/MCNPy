"""WWINP file writer with optimised bulk formatting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kika.wwinp.classes.wwinp import WWINP
from kika.wwinp.classes.geometry import GeometryAxis

# Constants
VALS_PER_LINE = 6
WRITE_BUFFER = 8 * 1024 * 1024  # 8 MB
CHUNK_LINES = 10_000


def write_wwinp(wwinp: WWINP, filepath: str | Path, overwrite: bool = False) -> None:
    """Write a :class:`WWINP` object to disk in FORTRAN-compatible format."""
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to replace."
        )

    h = wwinp.header

    with open(filepath, "w", buffering=WRITE_BUFFER) as f:
        # ── Block 1: Header ─────────────────────────────────────────
        # First line: if iv ni nr  (20-char gap)  probid
        f.write(
            f"{h.if_:10d}{h.iv:10d}{h.ni:10d}{h.nr:10d}"
            + " " * 20
            + f"{h.probid:19s}\n"
        )

        # Time bin counts (if time-dependent)  — format: 7i10
        if h.has_time_dependency:
            _write_int_array_7i10(f, h.nt)

        # Energy bin counts — format: 7i10
        _write_int_array_7i10(f, h.ne)

        # Mesh dimensions and origin  (6g13.5)
        f.write(
            f"{float(h.nfx):13.5e}{float(h.nfy):13.5e}{float(h.nfz):13.5e}"
            f"{h.x0:13.5e}{h.y0:13.5e}{h.z0:13.5e}\n"
        )

        # Geometry-specific parameters
        if h.nr == 10:
            f.write(
                f"{float(h.ncx):13.5e}{float(h.ncy):13.5e}"
                f"{float(h.ncz):13.5e}{float(h.nwg):13.5e}\n"
            )
        elif h.nr == 16:
            rv = h.ref_vec if h.ref_vec is not None else np.zeros(3)
            rd = h.ref_dir if h.ref_dir is not None else np.zeros(3)
            f.write(
                f"{float(h.ncx):13.5e}{float(h.ncy):13.5e}"
                f"{float(h.ncz):13.5e}"
                f"{rv[0]:13.5e}{rv[1]:13.5e}{rv[2]:13.5e}\n"
            )
            f.write(
                f"{rd[0]:13.5e}{rd[1]:13.5e}{rd[2]:13.5e}"
                f"{float(h.nwg):13.5e}\n"
            )

        # ── Block 2: Geometry ───────────────────────────────────────
        for axis in wwinp.geometry.axes:
            _write_axis(f, axis)

        # ── Block 3: Values ─────────────────────────────────────────
        for i in range(h.ni):
            # Time bins — spec says "if nt(i)>1" (Table A.1)
            if h.has_time_dependency and h.nt[i] > 1 and i in wwinp.time_bins:
                _write_float_array(f, wwinp.time_bins[i])
            # Energy bins
            if i in wwinp.energy_bins:
                _write_float_array(f, wwinp.energy_bins[i])

            # Weight window values — flatten 5-D to 1-D, then write
            arr = wwinp.values.ww_values[i]  # (nt, ne, nfz, nfy, nfx)
            nt, ne = arr.shape[0], arr.shape[1]
            for t in range(nt):
                for e in range(ne):
                    _write_float_array(f, arr[t, e].ravel())


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _write_int_array_7i10(f, values: list[int]) -> None:
    """Write integers in FORTRAN 7i10 format (7 per line, 10 chars each)."""
    for i in range(0, len(values), 7):
        chunk = values[i : i + 7]
        f.write("".join(f"{v:10d}" for v in chunk) + "\n")


def _write_axis(f, axis: GeometryAxis) -> None:
    """Write one geometry axis in 6g13.5 format."""
    vals = [axis.origin]
    for q, p, s in zip(axis.q, axis.p, axis.s):
        vals.extend([float(q), float(p), float(s)])
    _write_float_list(f, vals)


def _write_float_list(f, values: list[float]) -> None:
    """Write a Python list of floats in 6g13.5 format."""
    for i in range(0, len(values), VALS_PER_LINE):
        chunk = values[i : i + VALS_PER_LINE]
        f.write("".join(f"{v:13.5e}" for v in chunk) + "\n")


def _write_float_array(f, arr: np.ndarray) -> None:
    """Write a flat numpy array in 6g13.5 format, chunked for performance."""
    values = arr.ravel()
    n = len(values)

    # Process in chunks of CHUNK_LINES * VALS_PER_LINE values
    chunk_vals = CHUNK_LINES * VALS_PER_LINE
    for start in range(0, n, chunk_vals):
        end = min(start + chunk_vals, n)
        block = values[start:end]

        lines: list[str] = []
        for i in range(0, len(block), VALS_PER_LINE):
            row = block[i : i + VALS_PER_LINE]
            lines.append("".join(f"{v:13.5e}" for v in row))

        f.write("\n".join(lines) + "\n")
