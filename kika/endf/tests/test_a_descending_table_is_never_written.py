"""No TAB1 leaves kika with its abscissae going backwards.

**Written because of D31**, and the shape of that defect is why the gate sits
where it does. ``_augment_with_step_duplicates`` inserted step points at MF33
bin edges in the wrong order whenever two edges fell inside one interval of the
pointwise grid. The result was a descending energy column in every perturbed MT
of every replica -- and *nothing said so*: NJOY read the tape, returned ``0``,
and wrote an ACE whose elastic cross section was up to 145% wrong between 10
and 30 MeV.

The lesson is not "check the applier". It is that a malformed record has to be
refused at the point where records are written, because that is the only place
that also covers the appliers nobody has written yet. Every TAB1 in the library
-- MF1, MF2, MF3, MF5, MF6, MF7 -- goes through ``format_tab1``.

Measured before the check was made a hard error: of the 27 tapes available on
this machine, including the whole JEFF-4.0 Fe-56, zero sections violate it. It
refuses nothing a real evaluation writes.
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf3.mf3mt import MF3MT
from kika.endf.utils import NonMonotonicTable, format_tab1

DATA = Path(__file__).resolve().parent / "data"
MAT, MF, MT = 2631, 3, 2


def _tab1(xs, ys=None):
    ys = [1.0] * len(xs) if ys is None else ys
    return format_tab1(0.0, 0.0, 0, 0, [(len(xs), 2)], xs, ys, MAT, MF, MT, 1)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_an_ascending_table_is_written():
    lines, _ = _tab1([1.0, 2.0, 3.0])
    assert lines


def test_a_repeated_abscissa_is_a_step_and_is_allowed():
    """The whole perturbation machinery depends on this being legal."""
    lines, _ = _tab1([1.0, 2.0, 2.0, 3.0])
    assert lines


def test_a_descending_abscissa_is_refused():
    with pytest.raises(NonMonotonicTable) as excinfo:
        _tab1([1.0, 3.0, 2.0, 4.0])

    error = excinfo.value
    assert error.mf == MF and error.mt == MT
    assert error.index == 1
    assert error.previous == 3.0 and error.current == 2.0


def test_the_message_says_where_and_how_many():
    with pytest.raises(NonMonotonicTable, match=r"point 2 of 5"):
        _tab1([1.0, 3.0, 2.0, 9.0, 8.0])
    with pytest.raises(NonMonotonicTable, match=r"2 pair\(s\) go backwards"):
        _tab1([1.0, 3.0, 2.0, 9.0, 8.0])


@pytest.mark.parametrize("xs", [[], [1.0]])
def test_a_table_too_short_to_descend_is_left_alone(xs):
    lines, _ = _tab1(xs)
    assert isinstance(lines, list)


# ---------------------------------------------------------------------------
# Through the writers that had the defect
# ---------------------------------------------------------------------------

def _section(energies, values):
    section = MF3MT(number=MT)
    section._za, section._awr, section._mat = 26056.0, 55.454, MAT
    section._qm, section._qi, section._lr = 0.0, 0.0, 0
    section._energies = list(energies)
    section._cross_sections = list(values)
    section._nr, section._np = 1, len(energies)
    section._interpolation = [(len(energies), 2)]
    return section


def test_serialising_a_section_refuses_it():
    """``str(section)`` is what every writer ultimately calls."""
    with pytest.raises(NonMonotonicTable):
        str(_section([1.0, 3.0, 2.0], [1.0, 1.0, 1.0]))


def test_the_writer_does_not_flatten_it_into_a_false(tmp_path):
    """``replace_mt_section`` answers ``bool`` and swallows exceptions.

    Not this one. "I could not write the file" and "the data you handed me is
    not a function" are different answers, and flattening the second into the
    first is how D31 ran for months: a caller sees a failure with no cause,
    logs it and carries on.
    """
    from kika.endf.writers.endf_writer import ENDFWriter

    source = tmp_path / "host.endf"
    source.write_text((DATA / "micro_fe56_structural.endf").read_text())

    writer = ENDFWriter(str(source))
    with pytest.raises(NonMonotonicTable):
        writer.replace_mt_section(
            _section([1.0, 3.0, 2.0], [1.0, 1.0, 1.0]),
            mf_number=3, update_directory=False,
        )


def test_the_applier_that_had_the_defect_now_cannot_write_one():
    """D31's own reproduction, shrunk: two bin edges inside one interval.

    With the fix in ``_augment_with_step_duplicates`` this produces an ascending
    table, so the gate never fires -- and that is the assertion. The gate is
    there for the next applier, not this one.
    """
    from kika.sampling.mf33_sampling import (
        _augment_with_step_duplicates,
        perturb_pointwise_xs,
    )

    energies = np.array([1.0e-5, 2.53e-2, 1.0, 10.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bins = np.array([1.7e-3, 1.6e-2, 5.0])

    xs, ys, interp = _augment_with_step_duplicates(
        energies, values, [(4, 2)], bins)
    scaled, _, _ = perturb_pointwise_xs(xs, ys, np.array([1.5, 0.5]), bins)

    assert np.all(np.diff(xs) >= 0.0)
    lines, _ = format_tab1(0.0, 0.0, 0, 0, interp, list(xs), list(scaled),
                           MAT, MF, MT, 1)
    assert lines


# ---------------------------------------------------------------------------
# It refuses nothing real
# ---------------------------------------------------------------------------

def test_no_committed_tape_violates_it():
    """The measurement that made a hard error safe, kept as a test.

    If a future fixture is added that does violate it, this is where that gets
    noticed -- and the answer will be to look at the fixture, not to loosen the
    gate.
    """
    checked = 0
    for path in sorted(glob.glob(str(DATA / "*.endf"))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                endf = read_endf(path)
            except Exception:
                continue
            for mf in sorted(endf.files):
                for _mt, section in sorted(endf.files[mf].sections.items()):
                    try:
                        str(section)
                    except NonMonotonicTable:
                        pytest.fail(f"{Path(path).name} MF{mf}: descending TAB1")
                    except Exception:
                        continue
                    checked += 1
    assert checked > 100, f"only {checked} sections re-serialised; too few to mean anything"
