"""WWINP file parser with buffered token reader.

The TokenReader reads the file line-by-line and serves tokens on demand.
For large value blocks it provides ``read_float_array`` which collects
tokens in bulk and converts them in a single ``np.array`` call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kika.wwinp._exceptions import WWINPFormatError, WWINPParsingError
from kika.wwinp.classes.header import Header
from kika.wwinp.classes.geometry import GeometryAxis, GeometryData
from kika.wwinp.classes.weight_windows import WeightWindowValues
from kika.wwinp.classes.wwinp import WWINP
from kika.wwinp.utils import verify_and_correct


class TokenReader:
    """Buffered whitespace-token reader with line tracking."""

    def __init__(self, filepath: str | Path) -> None:
        self._file = open(filepath, "r")  # noqa: SIM115
        self._buffer: list[str] = []
        self._pos: int = 0
        self.line_no: int = 0

    def close(self) -> None:
        self._file.close()

    # -- low-level --------------------------------------------------

    def _refill(self) -> bool:
        """Read the next non-blank line into the buffer.  Return False on EOF."""
        while True:
            raw = self._file.readline()
            if not raw:
                return False
            self.line_no += 1
            tokens = raw.split()
            if tokens:
                self._buffer = tokens
                self._pos = 0
                return True
            # skip blank lines silently (more forgiving than WWINPy)

    def _ensure(self) -> None:
        if self._pos >= len(self._buffer):
            if not self._refill():
                raise WWINPParsingError("Unexpected end of file", line=self.line_no)

    # -- public API -------------------------------------------------

    def next_token(self) -> str:
        self._ensure()
        tok = self._buffer[self._pos]
        self._pos += 1
        return tok

    def next_int(self) -> int:
        tok = self.next_token()
        try:
            return int(tok)
        except ValueError:
            raise WWINPParsingError(
                f"Expected integer, got '{tok}'", line=self.line_no
            )

    def next_float(self) -> float:
        tok = self.next_token()
        try:
            return float(tok)
        except ValueError:
            raise WWINPParsingError(
                f"Expected float, got '{tok}'", line=self.line_no
            )

    def read_float_array(self, count: int) -> np.ndarray:
        """Read *count* floats into a numpy array in bulk.

        Collects raw string tokens across as many lines as needed, then
        does a single ``np.array(tokens, dtype=np.float64)`` conversion.
        """
        if count <= 0:
            return np.empty(0, dtype=np.float64)

        tokens: list[str] = []

        # Drain any leftover tokens in the current buffer first
        remaining = len(self._buffer) - self._pos
        if remaining > 0:
            take = min(remaining, count)
            tokens.extend(self._buffer[self._pos : self._pos + take])
            self._pos += take

        # Read full lines until we have enough tokens
        while len(tokens) < count:
            raw = self._file.readline()
            if not raw:
                raise WWINPParsingError(
                    f"Unexpected EOF: needed {count} values, got {len(tokens)}",
                    line=self.line_no,
                )
            self.line_no += 1
            line_tokens = raw.split()
            if not line_tokens:
                continue
            tokens.extend(line_tokens)

        # Push overflow tokens back into the buffer
        if len(tokens) > count:
            self._buffer = tokens[count:]
            self._pos = 0
            tokens = tokens[:count]
        else:
            self._buffer = []
            self._pos = 0

        try:
            return np.array(tokens, dtype=np.float64)
        except ValueError as exc:
            raise WWINPParsingError(
                f"Could not convert tokens to float: {exc}", line=self.line_no
            )

    def peek_token(self) -> str | None:
        """Peek at the next token without consuming it.  Returns None on EOF."""
        try:
            self._ensure()
        except WWINPParsingError:
            return None
        return self._buffer[self._pos]


# -----------------------------------------------------------------------
# Block parsers
# -----------------------------------------------------------------------

def _parse_header(reader: TokenReader) -> Header:
    """Parse Block 1 (header)."""
    if_ = reader.next_int()
    iv = reader.next_int()
    ni = reader.next_int()
    nr = reader.next_int()

    # probid: next token might be a string id or already numeric
    peek = reader.peek_token()
    if peek is not None and not _is_numeric(peek):
        probid = reader.next_token()
    else:
        probid = ""

    # nt (time bins per particle) if iv == 2
    nt: list[int] = []
    if iv == 2:
        for _ in range(ni):
            nt.append(reader.next_int())

    # ne (energy bins per particle)
    ne: list[int] = []
    for _ in range(ni):
        ne.append(reader.next_int())

    # Verify and correct
    ni, nt_corrected, ne = verify_and_correct(ni, nt if iv == 2 else None, ne, iv)
    if iv == 2 and nt_corrected is not None:
        nt = nt_corrected

    # Geometry dimensions: nfx nfy nfz x0 y0 z0
    nfx = int(reader.next_float())
    nfy = int(reader.next_float())
    nfz = int(reader.next_float())
    x0 = reader.next_float()
    y0 = reader.next_float()
    z0 = reader.next_float()

    # nr-dependent parameters
    ref_vec = None
    ref_dir = None

    if nr == 10:
        ncx = int(reader.next_float())
        ncy = int(reader.next_float())
        ncz = int(reader.next_float())
        nwg = int(reader.next_float())
    elif nr == 16:
        ncx = int(reader.next_float())
        ncy = int(reader.next_float())
        ncz = int(reader.next_float())
        x1 = reader.next_float()
        y1 = reader.next_float()
        z1 = reader.next_float()
        x2 = reader.next_float()
        y2 = reader.next_float()
        z2 = reader.next_float()
        nwg = int(reader.next_float())
        ref_vec = np.array([x1, y1, z1])
        ref_dir = np.array([x2, y2, z2])
    else:
        raise WWINPFormatError(f"Unsupported nr value: {nr}", line=reader.line_no)

    return Header(
        if_=if_,
        iv=iv,
        ni=ni,
        nr=nr,
        probid=probid,
        nt=nt,
        ne=ne,
        nfx=nfx,
        nfy=nfy,
        nfz=nfz,
        x0=x0,
        y0=y0,
        z0=z0,
        ncx=ncx,
        ncy=ncy,
        ncz=ncz,
        nwg=nwg,
        ref_vec=ref_vec,
        ref_dir=ref_dir,
    )


def _parse_geometry(reader: TokenReader, header: Header) -> GeometryData:
    """Parse Block 2 (geometry axes)."""
    axes: list[GeometryAxis] = []
    for nc in (header.ncx, header.ncy, header.ncz):
        origin = reader.next_float()
        q_list, p_list, s_list = [], [], []
        for _ in range(nc):
            q_list.append(reader.next_float())
            p_list.append(reader.next_float())
            s_list.append(int(reader.next_float()))
        axes.append(
            GeometryAxis(
                origin=origin,
                q=np.array(q_list, dtype=np.float64),
                p=np.array(p_list, dtype=np.float64),
                s=np.array(s_list, dtype=np.int32),
            )
        )

    return GeometryData(
        axes=(axes[0], axes[1], axes[2]),
        geometry_type=header.nwg,
    )


def _parse_values(
    reader: TokenReader, header: Header
) -> tuple[WeightWindowValues, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Parse Block 3 (time bins, energy bins, weight window values).

    Returns ``(values, time_bins, energy_bins)``.
    """
    time_bins: dict[int, np.ndarray] = {}
    energy_bins: dict[int, np.ndarray] = {}
    ww_dict: dict[int, np.ndarray] = {}

    n_spatial = header.nfx * header.nfy * header.nfz

    for i in range(header.ni):
        # Time bins — spec says "if nt(i)>1" (Table A.1)
        if header.has_time_dependency and header.nt[i] > 1:
            t = reader.read_float_array(header.nt[i])
            time_bins[i] = t
            nt = header.nt[i]
        else:
            nt = max(header.nt[i], 1) if header.has_time_dependency else 1

        # Energy bins
        e = reader.read_float_array(header.ne[i])
        energy_bins[i] = e
        ne = len(e)

        # Weight window values — single bulk read, then reshape to 5-D
        total = nt * ne * n_spatial
        raw = reader.read_float_array(total)
        ww_dict[i] = raw.reshape(nt, ne, header.nfz, header.nfy, header.nfx)

    return WeightWindowValues(ww_values=ww_dict), time_bins, energy_bins


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def parse_wwinp_file(filepath: str | Path) -> WWINP:
    """Parse a WWINP file into a :class:`WWINP` object."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"WWINP file not found: {filepath}")

    reader = TokenReader(filepath)
    try:
        header = _parse_header(reader)
        geometry = _parse_geometry(reader, header)
        values, time_bins, energy_bins = _parse_values(reader, header)
    finally:
        reader.close()

    return WWINP(
        filename=str(filepath),
        header=header,
        geometry=geometry,
        values=values,
        time_bins=time_bins,
        energy_bins=energy_bins,
    )


def _is_numeric(s: str) -> bool:
    """Check whether *s* looks like a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False
