"""The model produces the same a_l the ENDF section does.

``compute_base_cell_means`` stopped taking an ``MF4MT`` and started taking a
:class:`~kika.cov.multigroup.collapse.LegendreSource`, so the multigroup
collapse can now be fed from a ``reactionSuite``. This file is the evidence that
"can be fed" means the same numbers rather than merely the same shape.

**The comparison is against the ENDF producer, not against a stored golden**,
and deliberately: a golden would freeze whatever the model path happened to
return on the day it was written. What matters is that the two producers agree,
today and after either of them changes.

The model side arrives through ``kika.read()`` — the one door — rather than by
calling ``decodeMF4MT``. That is not politeness: it means this test exercises
the path a user actually has, and it keeps ``kika/cov`` free of the adapter
import, which is what P4 had just finished removing.
"""
from __future__ import annotations

import numpy as np
import pytest

import kika
from kika.cov.multigroup.collapse import (
    legendre_source_from_endf,
    legendre_source_from_model,
)
from kika.endf import read_endf

#: Fe-56 elastic. High enough to reach orders the file leaves implicit, so the
#: zero-fill convention is exercised rather than assumed.
MAX_ORDER = 8
MT = 2


@pytest.fixture(scope="module")
def endf_source(micro_tape):
    """a_l off the ENDF MF4/MT2 section, as the collapse samples it."""
    endf = read_endf(str(micro_tape))
    return legendre_source_from_endf(
        endf.mf[4].mt[MT], fallback_grid=np.array([1e-5, 2e7]), max_order=MAX_ORDER
    )


@pytest.fixture(scope="module")
def model_source(micro_tape):
    """a_l off the same section, reached through the model.

    ``on_tabulated="skip"`` because this tape's MT2 is **LTT=3**: 3960 energies
    of Legendre coefficients and 19 of tabulated ``P(mu)``. The claim under test
    is about the first 3960 — where the file states coefficients, the model
    returns exactly those — and the tabulated region is compared nowhere,
    because the model path deliberately does not project it.
    """
    suite = kika.read(str(micro_tape))
    reaction = suite.reactionByENDF_MT(MT)
    angular = reaction.outputChannel.products[0].distribution["eval"]
    return legendre_source_from_model(
        angular, max_order=MAX_ORDER, on_tabulated="skip"
    )


@pytest.fixture(scope="module")
def alignment(endf_source, model_source):
    """Where each model energy sits in the ENDF producer's mesh.

    **Matched by energy, not by position**, and the difference is the point.
    The ENDF producer builds its mesh with ``np.unique`` over the Legendre and
    tabulated breakpoints, so it comes back sorted, 3977 long, and with the two
    energies the regions share collapsed into one. The model keeps its 3960 in
    file order **including the one repeated incident energy** — a discontinuity
    the evaluator wrote down. Neither is a subsequence of the other, so a
    positional comparison would report a difference where there is none.
    """
    index = np.searchsorted(endf_source.native_energies, model_source.native_energies)
    assert np.array_equal(
        endf_source.native_energies[index], model_source.native_energies
    ), "a model energy is absent from the ENDF producer's mesh"
    return index


def test_the_tape_really_is_the_mixed_case(endf_source, model_source):
    """Otherwise the restriction to the Legendre region would be vacuous."""
    assert len(model_source.native_energies) < len(endf_source.native_energies)


def test_the_model_keeps_the_repeated_energy_the_endf_mesh_drops(model_source):
    """The np.unique on the ENDF side is lossy, and this is where it shows.

    A repeated incident energy is two distinct angular distributions sharing an
    abscissa — a discontinuity. ``np.unique`` collapses it. Recorded here rather
    than fixed: the collapse interpolates a_l linearly between breakpoints, so
    it has no way to represent the jump either, and changing the mesh would move
    every number the golden pins.
    """
    energies = model_source.native_energies
    assert len(energies) - len(np.unique(energies)) == 1


def test_the_two_producers_agree_on_every_order(endf_source, model_source, alignment):
    """Bit-for-bit, at rtol=0 — these are reads of the same numbers, not a fit."""
    assert set(model_source.coefficients) == set(endf_source.coefficients)
    for order in sorted(endf_source.coefficients):
        np.testing.assert_array_equal(
            model_source.coefficients[order],
            endf_source.coefficients[order][alignment],
            err_msg=f"a_{order} differs between the ENDF and model producers",
        )


def test_a_0_is_one_and_is_stated_rather_than_implied(model_source):
    """ENDF leaves a_0 implicit; the model writes it down. Both must say 1.

    Exactly 1, not 1 to within a ULP: the model reads the coefficient the
    decoder stored, so no arithmetic happens between the file and here.
    """
    np.testing.assert_array_equal(
        model_source.coefficients[0], np.ones_like(model_source.native_energies)
    )


def test_orders_the_file_does_not_carry_come_back_zero(model_source):
    """The ENDF convention, applied on the model path too rather than assumed."""
    assert model_source.max_order == MAX_ORDER
    top = model_source.coefficients[MAX_ORDER]
    assert top.shape == model_source.native_energies.shape


def test_the_frame_survives_the_trip(endf_source, model_source):
    """A CM distribution collapsed as though it were lab is a wrong answer that
    looks right, so the frame is part of the source rather than read separately."""
    assert model_source.frame == endf_source.frame


def test_a_tabulated_child_is_refused_rather_than_projected():
    """The model path declines what it cannot do without a second quadrature."""
    from kika.nuclear_data.model.distributions import AngularTwoBody
    from kika.nuclear_data.model.functions.higher import XYs2d
    from kika.nuclear_data.model.functions.xys1d import XYs1d

    tabulated = XYs1d(xs=np.array([-1.0, 1.0]), ys=np.array([0.5, 0.5]))
    tabulated.outerDomainValue = 1.0e6
    angular = AngularTwoBody(angular=XYs2d(function1ds=[tabulated]))

    with pytest.raises(NotImplementedError, match="not Legendre"):
        legendre_source_from_model(angular, max_order=4)
