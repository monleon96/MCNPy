"""``write_ace`` must not rewrite the object it was handed.

It used to. The guard read element 0 to decide whether the XSS still held raw
numbers::

    if not hasattr(ace.xss_data[0], 'index'):
        ace.xss_data = [XssEntry(index=i, value=val) for i, val in enumerate(...)]

but ``read_xss`` seeds index 0 with a bare ``0`` as the FORTRAN 1-based
placeholder, and ``hasattr(0, 'index')`` is False. So the branch fired on every
parsed file and re-wrapped each already-wrapped entry into
``XssEntry(index=i, value=XssEntry(...))`` — on the caller's object, not a copy.

Two private helpers existed only to paper over it: ``unwrap_value`` inside the
writer and a copy of it in the round-trip test. Both are gone with the fix.

These tests are deliberately free of any ACE tape, so they run in CI where the
rest of ``kika/ace`` cannot.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace.classes.ace import Ace
from kika.ace.classes.header import Header
from kika.ace.classes.xss import XssEntry
from kika.ace.writers.write_ace import write_ace


def _tiny_ace() -> Ace:
    """An Ace shaped like a parsed one: bare 0 placeholder, then XssEntry."""
    values = [11.0, 2.5, 3.0, 4.25, 5.0, 6.5, 7.0]
    xss = [0] + [XssEntry(index=i, value=v) for i, v in enumerate(values, start=1)]
    header = Header(
        format_version="legacy",
        zaid=26056,
        extension=".02c",
        atomic_weight_ratio=55.454,
        temperature=2.5301e-08,
        date="01/01/26",
        comment="synthetic fixture",
        matid=2631,
        izaw_array=[(0, 0.0)] * 16,
        nxs_array=[len(xss) - 1] + [0] * 15,
        jxs_array=[1] + [0] * 31,
    )
    return Ace(filename=None, header=header, xss_data=xss)


def _values(ace) -> list:
    return [getattr(entry, "value", entry) for entry in ace.xss_data]


def test_write_ace_leaves_its_input_alone(tmp_path):
    ace = _tiny_ace()
    before_types = [type(entry) for entry in ace.xss_data]
    before_values = _values(ace)
    before_identity = [id(entry) for entry in ace.xss_data]

    write_ace(ace, str(tmp_path / "out.02c"), overwrite=True)

    assert [type(e) for e in ace.xss_data] == before_types, (
        "write_ace changed the type of the caller's XSS entries"
    )
    assert _values(ace) == before_values, "write_ace changed the caller's XSS values"
    assert [id(e) for e in ace.xss_data] == before_identity, (
        "write_ace replaced the caller's XSS entry objects"
    )


def test_no_entry_ends_up_wrapping_another_entry(tmp_path):
    """The specific corruption: XssEntry(value=XssEntry(...))."""
    ace = _tiny_ace()
    write_ace(ace, str(tmp_path / "out.02c"), overwrite=True)

    nested = [
        entry for entry in ace.xss_data
        if isinstance(getattr(entry, "value", None), XssEntry)
    ]
    assert not nested, f"{len(nested)} XSS entries wrap another entry"


def test_writing_twice_gives_the_same_file(tmp_path):
    """A mutating writer is not idempotent; this is what that looked like."""
    ace = _tiny_ace()
    first = tmp_path / "first.02c"
    second = tmp_path / "second.02c"

    write_ace(ace, str(first), overwrite=True)
    write_ace(ace, str(second), overwrite=True)

    assert first.read_text() == second.read_text()


def test_raw_float_input_is_still_accepted(tmp_path):
    """The wrapping branch existed for a reason: a caller may pass raw numbers."""
    ace = _tiny_ace()
    ace.xss_data = [0.0, 1.0, 2.0, 3.0]

    write_ace(ace, str(tmp_path / "out.02c"), overwrite=True)

    assert ace.xss_data == [0.0, 1.0, 2.0, 3.0], "raw input was rewritten in place"


def test_values_reach_the_file(tmp_path):
    """Guard against the mutation fix quietly dropping the payload.

    The XSS block is the tail of the file, after the header's IZAW/NXS/JXS
    arrays — which are numeric too, hence the tail slice rather than a scan.
    Index 0 is the placeholder and is not written.
    """
    ace = _tiny_ace()
    out = tmp_path / "out.02c"
    write_ace(ace, str(out), overwrite=True)

    numeric = []
    for line in reversed(out.read_text().splitlines()):
        try:
            row = [float(tok) for tok in line.split()]
        except ValueError:
            break
        numeric = row + numeric

    expected = [entry.value for entry in _tiny_ace().xss_data[1:]]
    np.testing.assert_allclose(numeric[-len(expected):], expected)
