"""
Compact storage helpers for per-group vectors.

Sensitivity / energy vectors are stored in SQLite as zstd-compressed float32
BLOBs. This keeps the built database ~0.3-0.6 GB (versus multiple GB uncompressed
for ~4,000 benchmarks x tens of reactions x 238 groups) while preserving the full
per-group profile needed for plotting and later sandwich UQ.

zstd is provided by ``pyarrow`` (already a kika dependency); no extra package is
required.
"""

from typing import Sequence

import numpy as np
import pyarrow as pa

_CODEC = "zstd"


def pack_f32(vec: Sequence[float]) -> bytes:
    """Compress a numeric vector to a zstd float32 BLOB."""
    raw = np.asarray(vec, dtype=np.float32).tobytes()
    return bytes(pa.compress(raw, codec=_CODEC))


def unpack_f32(blob: bytes, n: int) -> np.ndarray:
    """
    Decompress a zstd float32 BLOB back to an ``n``-element float32 array.

    Parameters
    ----------
    blob : bytes
        The compressed payload as stored in the database.
    n : int
        Number of float32 elements expected (needed to size the output buffer).
    """
    raw = pa.decompress(blob, decompressed_size=n * 4, codec=_CODEC)
    return np.frombuffer(raw, dtype=np.float32)
