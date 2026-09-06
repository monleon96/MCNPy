"""Catch the NJOY runs that misread their own decimal point.

On Windows a gfortran/MinGW NJOY binary can switch its C runtime over to the
system locale the first time it writes an unformatted (binary) tape.  From
that moment every formatted read stops at the ``.`` that the locale does not
recognise as a decimal separator, so ``2.330248+2`` is read as ``2``,
``5.0000e-03`` as ``5``, and the run either dies inside ``lunion`` blaming the
evaluation -- or, far worse, finishes and writes a PENDF built from numbers
nobody asked for.

Measured on a Windows ``es-ES`` machine (decimal separator ``,``) with
NJOY 2016.78, ten runs of the same deck and tape per binary:

===============================  ================
``build/njoy.exe``               6/10 corrupted
``build_static/njoy.exe``        10/10 corrupted
``build_static/njoy_fixed.exe``  3/10 corrupted
===============================  ================

The rate is per *run*, not per binary: the same executable reads its input
correctly on one run and misreads it on the next.  Nothing outside the binary
changes that, and both obvious escapes were measured and ruled out:

* ``LC_ALL``/``LC_NUMERIC``/``LANG`` are ignored -- the Windows CRT takes its
  locale from the user's regional settings, not from the environment.  We
  still forward them because they do matter on Linux and macOS.
* No deck avoids the trigger.  Dropping the leading ``moder`` makes ``reconr``
  read its input card correctly, but the run still fails: ``reconr`` opens its
  own unformatted scratch tapes whatever tape formats the deck asks for
  (``reconr.f90:205``), so the ENDF reads that follow are exposed anyway.

A binary built with a ``setlocale(LC_ALL, "C")`` constructor lowers the rate a
lot (0/20 and 1/20 for a minimal reproducer, against 20/20 without it) but does
not remove it.  So the only defence available to every user, whatever binary
they configured, is to check after each run that NJOY read back the numbers we
handed it, and to run it again when it did not.

``reconr`` echoes its input card into the listing it writes to ``output``
(``reconr.f90:462``)::

    material to be processed .............       9228
    reconstruction tolerance .............      0.005     (f10.3)
    resonance-integral-check tolerance ...      0.050     (f10.3)
    max resonance-integral error .........  2.500E-07     (1p,e10.3)

``errint`` is the discriminating one: printed in scientific notation, it
separates a clean read from a truncated one at any magnitude.  NJOY only
substitutes defaults for these three when they arrive non-positive
(``reconr.f90:428-430``), and we always send positive values, so a mismatch
means a misread and never a substitution.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)

LISTING_FILENAME = "output"
"""NJOY writes its full listing to a file named ``output`` in its cwd."""


class NjoyDecimalLocaleError(RuntimeError):
    """NJOY misread the decimal point in the input KIKA handed it.

    Carries the numbers that proved it, so callers can show them.
    """

    def __init__(self, message: str, *, detail: str = ""):
        super().__init__(message)
        self.detail = detail


# The listing labels are padded with dots to a fixed width, so label and value
# are separated by a run of '.' of unknown length.
_ECHO_PATTERNS = {
    "err": re.compile(r"reconstruction tolerance\s*\.{2,}\s*([-+0-9.eEdD]+)"),
    "errmax": re.compile(
        r"resonance-integral-check tolerance\s*\.{2,}\s*([-+0-9.eEdD]+)"
    ),
    "errint": re.compile(r"max resonance-integral error\s*\.{2,}\s*([-+0-9.eEdD]+)"),
}


@dataclass(frozen=True)
class ListingCheck:
    """What a listing said about the numbers NJOY read.

    ``checked`` is False when the listing was missing or held none of the
    echoes we know how to read -- a different NJOY version, say.  That is not
    a failure: an unrecognised listing must never block a run that would
    otherwise work.
    """

    checked: bool
    ok: bool
    detail: str = ""

    @property
    def corrupted(self) -> bool:
        return self.checked and not self.ok


def read_listing(workdir: str | os.PathLike) -> Optional[str]:
    """Return the text of NJOY's ``output`` listing, or None if absent."""
    path = Path(workdir) / LISTING_FILENAME
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _parse_fortran_float(text: str) -> Optional[float]:
    """Parse a number as Fortran prints it (``2.500E-07``, ``0.005``, ``1.0D0``)."""
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _echoed(listing: str, key: str) -> Optional[float]:
    match = _ECHO_PATTERNS[key].search(listing)
    if match is None:
        return None
    return _parse_fortran_float(match.group(1))


def check_reconr_listing(
    listing: Optional[str], *, err: float, errmax: float, errint: float
) -> ListingCheck:
    """Compare the values ``reconr`` echoed against the ones we sent it.

    ``err`` and ``errmax`` are printed with ``f10.3``, so they are compared at
    that precision: a tolerance below 0.001 legitimately prints as ``0.000``,
    while every truncated read lands on the mantissa (``5.000`` for
    ``5.0000e-03``) and stays far outside the rounding window.  ``errint`` is
    printed with four significant figures and is compared relatively.
    """
    if not listing:
        return ListingCheck(checked=False, ok=True)

    expected = {"err": err, "errmax": errmax, "errint": errint}
    mismatches: List[str] = []
    checked_any = False

    for key, want in expected.items():
        got = _echoed(listing, key)
        if got is None:
            continue
        checked_any = True
        if key == "errint":
            ok = abs(got - want) <= abs(want) * 1e-3
        else:
            # Printed to three decimals; allow half a unit in the last place.
            ok = abs(got - want) <= 5.05e-4
        if not ok:
            mismatches.append(f"{key}: sent {want:g}, NJOY read {got:g}")

    if not checked_any:
        return ListingCheck(checked=False, ok=True)
    if mismatches:
        return ListingCheck(checked=True, ok=False, detail="; ".join(mismatches))
    return ListingCheck(checked=True, ok=True)


_RECONR_LINE = re.compile(r"^\s*reconr\s*$", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")


def parse_reconr_card(deck: str) -> Optional[tuple]:
    """Pull ``(err, errmax, errint)`` out of a deck's ``reconr`` card 4.

    Card 4 is the fourth card of the module: units, label, ``mat ncards
    ngrid``, then the tolerances.  Returns None when the deck has no
    ``reconr`` module or its card cannot be read, which is the signal to skip
    the check rather than to fail a run.

    Reading the deck rather than trusting a template constant means the check
    follows whatever deck was actually sent -- including one the user wrote.
    """
    lines = [line for line in deck.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if not _RECONR_LINE.match(line):
            continue
        # units, label, mat/ncards/ngrid, tolerances
        if i + 4 >= len(lines):
            return None
        numbers = _NUMBER.findall(lines[i + 4])
        if len(numbers) < 4:
            return None
        try:
            err = float(numbers[0].replace("D", "E").replace("d", "e"))
            errmax = float(numbers[2].replace("D", "E").replace("d", "e"))
            errint = float(numbers[3].replace("D", "E").replace("d", "e"))
        except ValueError:
            return None
        # NJOY fills these in itself when they arrive non-positive
        # (reconr.f90:428-430), so we would be comparing against a value we
        # never sent.  Leave those runs unchecked.
        if err <= 0 or errmax <= 0 or errint <= 0:
            return None
        return err, errmax, errint
    return None


def check_run(deck: str, listing: Optional[str]) -> ListingCheck:
    """Check any NJOY run for which we have both the deck and the listing."""
    card = parse_reconr_card(deck)
    if card is None:
        return ListingCheck(checked=False, ok=True)
    err, errmax, errint = card
    return check_reconr_listing(listing, err=err, errmax=errmax, errint=errint)


def locale_error_message(detail: str, *, attempts: int) -> str:
    """The message a user gets once the retries are exhausted."""
    how_often = (
        f"on {attempts} consecutive runs" if attempts > 1 else "on this run"
    )
    return (
        f"NJOY misread its own input {how_often} ({detail}). "
        "This is the MinGW/gfortran decimal-separator bug: on a Windows "
        "regional format whose decimal symbol is not '.', an NJOY binary can "
        "switch to the system locale after writing a binary tape and then read "
        "every number only as far as the decimal point. The evaluation file is "
        "fine. Fixes, best first: rebuild NJOY with a setlocale(LC_ALL, \"C\") "
        "constructor, run NJOY under WSL (Settings accepts a \\\\wsl.localhost\\ "
        "path), or set the Windows decimal symbol to '.'."
    )


# ---------------------------------------------------------------------------
# Standalone probe: is this binary affected, and how often?
# ---------------------------------------------------------------------------

PROBE_MAT = 125
PROBE_ERR = 5.0e-3
PROBE_ERRMAX = 5.0e-2
PROBE_ERRINT = 2.5e-7


def _record(c1, c2, l1, l2, n1, n2, mat, mf, mt, ns) -> str:
    return (
        f"{c1:>11}{c2:>11}{l1:>11}{l2:>11}{n1:>11}{n2:>11}"
        f"{mat:>4}{mf:>2}{mt:>3}{ns:>5}"
    )


def _text_record(body, mat, mf, mt, ns) -> str:
    return f"{body:<66}{mat:>4}{mf:>2}{mt:>3}{ns:>5}"


def build_probe_tape() -> str:
    """A synthetic one-material ENDF tape, small enough to process instantly.

    Built rather than shipped as a data file: an 80-column format is one
    trailing-whitespace strip away from being invalid, and a data file would
    also have to be re-declared in the app's PyInstaller spec.
    """
    mat = PROBE_MAT
    lines = [
        _text_record(" KIKA NJOY decimal-locale probe tape", 1, 0, 0, 0),
        # MF1/MT451 -- H-1 identity, no resonance parameters (LRP=0).
        _record("1.001000+3", "9.991673-1", 0, 0, 0, 0, mat, 1, 451, 1),
        _record("0.000000+0", "0.000000+0", 0, 0, 0, 6, mat, 1, 451, 2),
        _record("1.000000+0", "2.000000+7", 0, 0, 10, 8, mat, 1, 451, 3),
        _record("0.000000+0", "0.000000+0", 0, 0, 3, 2, mat, 1, 451, 4),
        _text_record(" 1-H -  1 KIKA       PROBE", mat, 1, 451, 5),
        _text_record(" synthetic tape, checks decimal parsing only", mat, 1, 451, 6),
        _text_record(" no physics here", mat, 1, 451, 7),
        _record("0.000000+0", "0.000000+0", 1, 451, 7, 0, mat, 1, 451, 8),
        _record("0.000000+0", "0.000000+0", 3, 1, 4, 0, mat, 1, 451, 9),
        _record(0, 0, 0, 0, 0, 0, mat, 1, 0, 99999),
        _record(0, 0, 0, 0, 0, 0, mat, 0, 0, 0),
        # MF3/MT1 -- a two-point flat cross section.
        _record("1.001000+3", "9.991673-1", 0, 0, 0, 0, mat, 3, 1, 1),
        _record("0.000000+0", "0.000000+0", 0, 0, 1, 2, mat, 3, 1, 2),
        _record(2, 2, 0, 0, 0, 0, mat, 3, 1, 3),
        _record(
            "1.000000-5", "1.000000+0", "2.000000+7", "1.000000+0", "", "",
            mat, 3, 1, 4,
        ),
        _record(0, 0, 0, 0, 0, 0, mat, 3, 0, 99999),
        _record(0, 0, 0, 0, 0, 0, mat, 0, 0, 0),
        _record(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        _record(0, 0, 0, 0, 0, 0, -1, 0, 0, 0),
    ]
    return "\n".join(lines) + "\n"


def build_probe_deck() -> str:
    """``moder`` first, so the binary write lands before reconr reads a number."""
    return "\n".join(
        [
            "moder",
            " 20 -25",
            "reconr",
            " -25 -21",
            "'kika decimal-locale probe'/",
            f" {PROBE_MAT} 0 0",
            f" {PROBE_ERR:.4e} 0 {PROBE_ERRMAX:.4e} {PROBE_ERRINT:.4e}",
            " 0 /",
            "stop",
            "",
        ]
    )


@dataclass(frozen=True)
class ProbeReport:
    """How a given NJOY executable behaved over ``runs`` probe runs."""

    executable: str
    runs: int
    corrupted: int
    inconclusive: int

    @property
    def ok(self) -> bool:
        return self.corrupted == 0

    @property
    def conclusive(self) -> bool:
        return self.inconclusive < self.runs

    @property
    def summary(self) -> str:
        if not self.conclusive:
            return (
                "Could not read NJOY's listing, so the decimal-separator check "
                "did not run. The executable may still be fine."
            )
        if self.ok:
            return (
                f"NJOY read every number correctly on {self.runs} runs. "
                "KIKA still checks each real run."
            )
        return (
            f"NJOY misread its input on {self.corrupted} of {self.runs} runs "
            "(MinGW decimal-separator bug). KIKA detects this and retries on "
            "every reconstruction it drives, so those results stay correct, but "
            "decks you run yourself in NJOY Process cannot be checked. "
            "Rebuilding NJOY with a setlocale(LC_ALL, \"C\") constructor, or "
            "running it under WSL, removes it at the source."
        )


def probe_executable(
    njoy_executable: str | os.PathLike,
    *,
    runs: int = 3,
    timeout_s: float = 120.0,
) -> ProbeReport:
    """Run a tiny deck ``runs`` times and count how often NJOY misreads it.

    Each run takes well under a second.  Because the bug is per-run, one clean
    probe proves nothing; repeating is what estimates the rate.
    """
    from kika.njoy.launcher import build_njoy_command

    forward_env = {"LC_ALL": "C", "LC_NUMERIC": "C", "LANG": "C"}
    env = os.environ.copy()
    env.update(forward_env)
    cmd = build_njoy_command(njoy_executable, forward_env=forward_env)
    deck = build_probe_deck()
    tape = build_probe_tape()

    corrupted = 0
    inconclusive = 0
    for _ in range(max(1, runs)):
        with tempfile.TemporaryDirectory(prefix="kika_njoy_probe_") as td:
            workdir = Path(td)
            (workdir / "tape20").write_text(tape, encoding="ascii", newline="\n")
            try:
                subprocess.run(
                    cmd,
                    cwd=str(workdir),
                    input=deck,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                _log.warning("NJOY locale probe could not run: %s", exc)
                inconclusive += 1
                continue
            check = check_reconr_listing(
                read_listing(workdir),
                err=PROBE_ERR,
                errmax=PROBE_ERRMAX,
                errint=PROBE_ERRINT,
            )
            if not check.checked:
                inconclusive += 1
            elif check.corrupted:
                corrupted += 1

    return ProbeReport(
        executable=str(njoy_executable),
        runs=max(1, runs),
        corrupted=corrupted,
        inconclusive=inconclusive,
    )


__all__ = [
    "LISTING_FILENAME",
    "ListingCheck",
    "NjoyDecimalLocaleError",
    "ProbeReport",
    "build_probe_deck",
    "build_probe_tape",
    "check_reconr_listing",
    "check_run",
    "parse_reconr_card",
    "locale_error_message",
    "probe_executable",
    "read_listing",
]
