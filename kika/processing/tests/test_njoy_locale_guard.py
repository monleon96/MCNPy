"""Unit tests for :mod:`kika.njoy.locale_guard`.

These run without NJOY: the listing text is the interface being tested, and
the samples below are copied from real runs of NJOY 2016.78 -- one that read
its input and one that misread it.

They live under ``processing`` rather than beside the module they test because
``conftest`` holds back anything whose node path carries the ``njoy`` keyword,
which would keep this file out of CI.  The guard is what stops a misread run
from reaching :mod:`kika.processing.njoy_reconstruct`, so it has to run there.
"""
from __future__ import annotations

from kika.njoy.locale_guard import (
    PROBE_ERR,
    PROBE_ERRINT,
    PROBE_ERRMAX,
    ProbeReport,
    build_probe_deck,
    build_probe_tape,
    check_reconr_listing,
)

CLEAN = """
 material to be processed .............       9228
 reconstruction tolerance .............      0.005
 reconstruction temperature ...........       0.00k
 resonance-integral-check tolerance ...      0.050
 max resonance-integral error .........  2.500E-07
"""

# The same deck, read by a binary that had switched to a ',' decimal
# separator: every number stops at the decimal point.
CORRUPT = """
 material to be processed .............       9228
 reconstruction tolerance .............      5.000
 reconstruction temperature ...........       0.00k
 resonance-integral-check tolerance ...      5.000
 max resonance-integral error .........  2.000E+00
"""


def test_clean_listing_passes() -> None:
    check = check_reconr_listing(CLEAN, err=5e-3, errmax=5e-2, errint=2.5e-7)
    assert check.checked and check.ok
    assert not check.corrupted


def test_corrupt_listing_is_caught_and_says_why() -> None:
    check = check_reconr_listing(CORRUPT, err=5e-3, errmax=5e-2, errint=2.5e-7)
    assert check.corrupted
    assert "sent 0.005" in check.detail and "read 5" in check.detail


def test_missing_listing_never_blocks_a_run() -> None:
    """An unreadable listing must not be treated as a failure."""
    for listing in (None, "", "njoy 2016.78 ran and said nothing we parse"):
        check = check_reconr_listing(listing, err=5e-3, errmax=5e-2, errint=2.5e-7)
        assert not check.checked
        assert check.ok and not check.corrupted


def test_tolerance_below_the_printed_precision_is_not_flagged() -> None:
    """err is printed with f10.3, so 1e-4 legitimately echoes as 0.000."""
    listing = """
 reconstruction tolerance .............      0.000
 resonance-integral-check tolerance ...      0.001
 max resonance-integral error .........  5.000E-09
"""
    check = check_reconr_listing(listing, err=1e-4, errmax=1e-3, errint=5e-9)
    assert check.checked and check.ok


def test_a_truncated_tiny_tolerance_still_stands_out() -> None:
    """The same 1e-4 misread lands on the mantissa, far from 0.000."""
    listing = """
 reconstruction tolerance .............      1.000
 max resonance-integral error .........  5.000E+00
"""
    check = check_reconr_listing(listing, err=1e-4, errmax=1e-3, errint=5e-9)
    assert check.corrupted


def test_probe_tape_is_a_valid_80_column_endf_tape() -> None:
    lines = build_probe_tape().splitlines()
    assert {len(line) for line in lines} == {80}
    # TEND is the last record, MEND the one before it.
    assert lines[-1][66:70].strip() == "-1"
    assert lines[-2][66:70].strip() == "0"


def test_probe_deck_writes_a_binary_tape_before_reading_a_number() -> None:
    """moder has to come first: that write is what triggers the bug."""
    deck = build_probe_deck().splitlines()
    assert deck[0] == "moder"
    assert deck.index("reconr") > deck.index("moder")
    assert f"{PROBE_ERR:.4e}" in deck[6]
    assert f"{PROBE_ERRMAX:.4e}" in deck[6]
    assert f"{PROBE_ERRINT:.4e}" in deck[6]


def test_probe_report_reads_as_a_rate() -> None:
    clean = ProbeReport(executable="njoy", runs=3, corrupted=0, inconclusive=0)
    assert clean.ok and clean.conclusive
    assert "3 runs" in clean.summary

    flaky = ProbeReport(executable="njoy", runs=10, corrupted=6, inconclusive=0)
    assert not flaky.ok
    assert "6 of 10" in flaky.summary

    blind = ProbeReport(executable="njoy", runs=2, corrupted=0, inconclusive=2)
    assert not blind.conclusive
    assert "did not run" in blind.summary
