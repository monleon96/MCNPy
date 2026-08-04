"""Tier-1 golden: the ACE read-write spine.

The ENDF half of this gate lives in ``kika/endf/tests/test_roundtrip_golden.py``.
This is the ACE half, and it splits cleanly in two.

**The numbers survive exactly.** ACE keeps everything in one flat XSS array.
Reading a real 16 MB Fe-56 file and writing it straight back returns all
821 670 entries bitwise identical — not ``allclose``, identical. That is the
gate, and it is what the perturbation pipeline depends on: it edits a handful
of XSS entries and writes the rest back untouched.

**The text does not.** 6 903 of 205 430 lines differ, and every one of them for
the same reason: an XSS entry whose value is a whole number is written in
floating-point notation (``5`` becomes ``5.00000000000E+00``). ACE has no types
— MT numbers, block lengths and pointers live in the same REAL array as the
cross sections — so kika cannot tell which entries were written as integers,
and renders them all as floats. MCNP's list-directed reads accept both, which
is why this has never been noticed.

Note the symmetry with the ENDF side: there kika writes floats as integers
where the value is zero, here it writes integers as floats. Both are the same
missing piece — the format layer does not carry how a number was spelled. That
is a thing the GNDS canonical model can record properly, and a reason to keep
the byte-level pin visible until it does.

No fixture is committed: a real ACE file is 16 MB at best, and truncating one
means recomputing the internal pointer blocks, which would make the fixture
kika's arithmetic rather than the library's. These tests run against the shared
tree, under ``tape``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.writers.write_ace import write_ace


def xss_array(ace) -> np.ndarray:
    """The XSS block as plain floats.

    Index 0 is the bare ``0`` ``read_xss`` seeds for FORTRAN 1-based indexing;
    every entry after it is an ``XssEntry``.
    """
    return np.array(
        [getattr(entry, "value", entry) for entry in ace.xss_data], dtype=float
    )


@pytest.fixture(scope="module")
def roundtripped(fe56_ace, tmp_path_factory):
    """Read a real ACE file and write it straight back, unmodified."""
    out = tmp_path_factory.mktemp("ace") / "roundtrip.02c"
    write_ace(read_ace(str(fe56_ace)), str(out), overwrite=True)
    return Path(fe56_ace), out


@pytest.mark.slow
def test_xss_survives_the_roundtrip_bitwise(roundtripped, fe56_ace):
    """Every XSS entry comes back with the same bits. The gate.

    Both sides are parsed from files, which keeps this honest about what the
    writer put on disk rather than about what it left in memory.
    """
    _, out = roundtripped
    before = xss_array(read_ace(str(fe56_ace)))
    after = xss_array(read_ace(str(out)))

    assert after.size == before.size, f"XSS length {before.size} -> {after.size}"
    n_diff = int(np.count_nonzero(before != after))
    assert n_diff == 0, (
        f"{n_diff} of {before.size} XSS entries changed; "
        f"first at {int(np.flatnonzero(before != after)[0])}"
    )


@pytest.mark.slow
def test_roundtrip_preserves_the_line_count(roundtripped):
    """The layout is unchanged even though individual fields are respelled.

    A differing line count would mean entries per line changed, which would
    break MCNP's reader. Only the rendering of each field moves.
    """
    src, out = roundtripped
    assert len(out.read_text().splitlines()) == len(src.read_text().splitlines())


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Deferred to phase 3 by decision, 2026-08-04, after the phase-1 defect "
        "burn-down cleared the other eight pins. ACE stores MT numbers, block "
        "lengths, pointers and cross sections in one untyped REAL array, so "
        "kika cannot tell which entries the source spelled as integers and "
        "writes them all as floats: '                  11' comes back as "
        "'   1.10000000000E+01'. 6 903 of 205 430 lines differ this way on the "
        "TENDL Fe-56 file. Deferred rather than fixed because the cost is "
        "real and the consequence is not: the values are unaffected (the "
        "companion test proves the XSS is bitwise identical) and MCNP's "
        "list-directed reads accept both spellings, while a fix needs a "
        "per-entry spelling flag on XssEntry and gives up the np.savetxt fast "
        "path over 820 000 entries, in a pipeline that writes one ACE per "
        "sample per temperature. Recording how a number was written is a job "
        "for the canonical model in phase 3. The mirror-image defect on the "
        "ENDF side — float fields rendered as integers — was fixed in phase 1, "
        "because there it cost nothing."
    ),
)
def test_ace_roundtrip_is_byte_identical(roundtripped):
    src, out = roundtripped
    assert (
        hashlib.sha256(out.read_bytes()).hexdigest()
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )
