"""``suite.cross_section()`` returns barns and eV, and says so out loud.

**What this is guarding.** A unit lives on an ``axis`` (§5.1.2). The shortcut
:meth:`~kika.nuclear_data.model.suite.ReactionSuite.cross_section` returns two
plain numpy arrays, so the unit is dropped at exactly that line, and every
consumer — the reconstructor, the chi2 pipeline, the plotting layer — reads what
comes back as ``(b, eV)`` without asking.

That assumption is **correct for every file anyone has**, and it was correct by
luck rather than by construction. What being wrong looks like is on the record:
``ScatteringRadius.constant`` **used to hold** 5.444 down the GNDS path and
0.5444 down the ENDF path, for the same Fe-56 radius, because ENDF's AP is in
units of 10⁻¹² cm and states so nowhere. Nothing raised. That one was closed on
2026-08-20 by giving the model a canonical radius unit — fm, GNDS's
(``MODEL_RADIUS_UNIT``, ``docs/library/gnds_endf_conflicts.md`` §4.1) — which is the
same move this file argues for and a reason to keep the example rather than
drop it: the cross section's ``(b, eV)`` is still correct by luck, and the
radius is the one that was checked.

So the assumption is now checked at the one line that drops it, and this file
holds both halves: that a stated wrong unit raises, and that an unstated one
does not — silence and disagreement are different things and §2.3.3 keeps them
apart.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.nuclear_data.model import (CROSS_SECTION_UNITS, Axes, CrossSection,
                                     Reaction, ReactionId, ReactionSuite,
                                     XYs1d)
from kika.nuclear_data.model.cross_section_forms import EVAL_LABEL

#: The census behind the assumption, run 2026-08-13 over every neutron
#: evaluation of ENDF/B-VIII.1-GNDS. Recorded here because the number is the
#: reason the check can be strict without breaking anything — and because a
#: measurement in a docstring is a measurement that gets re-run for nothing.
CENSUS = {
    "files": 558,
    "crossSection axes": (35_259, "b"),
    "energy_in axes": (475_003, "eV"),
    "multiplicity axes": (212_813, ""),
}


def _suite(unit: str = "b", energyUnit: str = "eV", axes: bool = True):
    """A one-reaction suite whose MT2 cross section states *unit*."""
    form = XYs1d(
        xs=np.array([1e3, 2e7]), ys=np.array([1.0, 2.0]), label=EVAL_LABEL,
        axes=Axes.forFunction1d("crossSection", unit, "energy_in", energyUnit)
        if axes else None,
    )
    reaction = Reaction(id=ReactionId(label="MT2", ENDF_MT=2),
                        crossSection=CrossSection(forms={EVAL_LABEL: form}))
    suite = ReactionSuite(evaluation="test", projectile="n", target="Fe56")
    suite.reactions.append(reaction)
    return suite


def test_the_units_the_shortcut_promises_are_the_ones_every_file_states():
    assert CROSS_SECTION_UNITS == (CENSUS["crossSection axes"][1],
                                   CENSUS["energy_in axes"][1]) == ("b", "eV")


def test_barns_and_eV_pass_through():
    E, xs = _suite().cross_section(2)
    assert xs.tolist() == [1.0, 2.0]
    assert E.tolist() == [1e3, 2e7]


@pytest.mark.parametrize("unit,energyUnit", [
    ("mb", "eV"),        # the classic: three orders of magnitude, no warning
    ("b", "MeV"),        # six orders on the abscissa
    ("1/b", "eV"),       # not even the right dimension
])
def test_a_stated_wrong_unit_raises_instead_of_being_rescaled(unit, energyUnit):
    with pytest.raises(ValueError, match="bare arrays"):
        _suite(unit, energyUnit).cross_section(2)


def test_the_message_names_both_what_was_found_and_what_was_expected():
    """A unit error the reader cannot act on is a unit error twice."""
    with pytest.raises(ValueError) as caught:
        _suite("mb").cross_section(2)
    message = str(caught.value)
    assert "mb" in message and "b" in message and "MT2" in message
    assert "does not rescale" in message


@pytest.mark.parametrize("kwargs", [
    {"axes": False},               # no axes at all
    {"unit": ""},                  # axes present, dependent unit unstated
    {"energyUnit": ""},            # axes present, abscissa unit unstated
])
def test_an_unstated_unit_is_silence_and_not_an_error(kwargs):
    """§2.3.3 separates "says nothing" from "says something else".

    Every hand-built form in this repository's tests is in the first case, and
    so is anything a reader could not resolve an ``axes href`` for. Complaining
    about silence would make the check fire everywhere except where it matters.
    """
    E, xs = _suite(**kwargs).cross_section(2)
    assert xs.tolist() == [1.0, 2.0]


def test_the_form_itself_is_still_reachable_and_still_states_its_unit():
    """Raising must not be a dead end — the data and its unit are both there."""
    suite = _suite("mb")
    form = suite.reactionByENDF_MT(2).crossSection[EVAL_LABEL]
    assert form.axes.dependent.unit == "mb"
    assert form.ys.tolist() == [1.0, 2.0]
