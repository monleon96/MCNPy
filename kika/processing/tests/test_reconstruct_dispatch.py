"""Which formalism goes to which formula — and what happens to the ones that go nowhere.

The goldens next door pin the arithmetic for the formalisms that *are*
implemented. This pins the edge of that set, which is where a reconstructor
does its real damage: a range it cannot compute must leave, loudly, rather than
reach a formula that will produce numbers from the wrong shape.
"""
from __future__ import annotations

import pytest

from kika.nuclear_data.model import Nuclide, PhysicalQuantity, PoPs
from kika.nuclear_data.model.resonances import (Channel, RMatrix,
                                                RMatrixSpinGroup,
                                                ResolvedRegion, Resonances,
                                                ScatteringRadius)
from kika.processing import reconstruct


def _target() -> PoPs:
    pops = PoPs(name="resolved resonances")
    pops.add(Nuclide(id="Fe57", Z=26, A=57, spin=PhysicalQuantity(0.5, "hbar")))
    return pops


def _rmlRegion() -> Resonances:
    """An R-Matrix-Limited range: one J per spin group, five channels.

    Hand-built rather than read from Fe-57, so the shape is checked in the fast
    lane and not only under ``--deep``. What matters is exactly what an RML
    group has that an ENDF LRF=3 block has not — a group-level ``spin`` with an
    empty ``spins``, and channels beyond the four Reich-Moore ones.
    """
    labels = ["neutron", "capture", "inelastic1", "inelastic2", "inelastic3"]
    group = RMatrixSpinGroup(
        label="1/2+", spin=0.5,
        channels=[
            Channel(label=label, resonanceReaction=label, L=0, columnIndex=index)
            for index, label in enumerate(labels)
        ],
        energies=[1.2e3, 4.5e3],
        widths=[[1.0, 0.2, 0.0, 0.0, 0.0], [1.4, 0.3, 0.0, 0.0, 0.0]],
        atomicWeightRatio=56.44,
    )
    return Resonances(
        scatteringRadius=ScatteringRadius(constant=0.51),
        resolved=[ResolvedRegion(
            domainMin=1.0e-5, domainMax=1.9e5,
            formalism=RMatrix(approximation="ReichMoore", spinGroups=[group],
                              scatteringRadius=0.51, PoPs=_target()),
        )],
    )


def test_an_rml_range_is_declined_rather_than_fed_to_the_lrf3_formula():
    """The approximation name is the same; the parameterisation is not.

    ENDF's LRF=7 with KRM=3 **is** Reich-Moore, and the decoder labels it so,
    correctly. But ``reich_moore_cross_sections`` implements LRF=3's
    parameterisation of it — blocked by l, a J on every resonance record, four
    fixed channels — and an RML group has one J for the whole group and as many
    channels as the evaluator declared. Dispatching on the name alone handed it
    the second shape and it raised an ``IndexError`` reaching for a
    per-resonance spin that does not exist. Found on Fe-57 JEFF-4.0.
    """
    with pytest.warns(UserWarning, match="Unsupported formalism"):
        assert reconstruct(_rmlRegion(), atomicWeightRatio=56.44) == {}


def test_the_warning_says_which_reich_moore_it_declined_and_what_to_use():
    """"RMatrix/ReichMoore" names both parameterisations, so it names neither.

    The channel count is what differs and what makes the range unsupported, so
    it goes in the message: a user reading it can tell an unimplemented LRF=7
    from a decode that lost something.

    It also has to name the alternative. This module implements three
    formalisms and — decided 2026-08-12 — is not going to grow a fourth, so a
    declined range is not a "coming soon" but a redirection to NJOY's RECONR,
    which kika already calls through ``njoy_reconstruct``. A warning that says
    only "unsupported" leaves the reader to conclude the evaluation cannot be
    reconstructed at all, which is false.
    """
    with pytest.warns(UserWarning) as caught:
        reconstruct(_rmlRegion(), atomicWeightRatio=56.44)

    message = str(caught[0].message)
    assert "ReichMoore" in message and "5" in message, message
    assert "njoy_reconstruct" in message, message


@pytest.mark.parametrize("tape", ["fe57_host_tape"])
def test_the_real_lrf7_evaluation_returns_nothing_and_says_so(request, tape):
    """Under ``--deep``. Fe-57 JEFF-4.0, the evaluation this is really about.

    It has one resolved range and that range is LRF=7, so this reconstructor
    has nothing left to compute and returns an empty result. **That is the
    settled state, not a gap** — decided 2026-08-12: kika is not writing an
    R-Matrix-Limited reconstructor, because agreement with the evaluation is
    what is wanted and RECONR is what delivers it. So what this pins is that
    the empty result is *loud* and names the road out, which is the whole
    difference from the flat path's silent ``[]``.
    """
    from kika.endf.processing.reconstruct import reconstruct as endf_reconstruct
    from kika.endf.read_endf import read_endf

    endf = read_endf(str(request.getfixturevalue(tape)), mf_numbers=[2, 3])
    with pytest.warns(UserWarning, match="Unsupported formalism"):
        produced = endf_reconstruct(endf.mf[2].mt[151], endf.files.get(3))

    assert produced == {}, (
        "Fe-57 now reconstructs something. If an RML formalism was added after "
        "all, this test is the one to rewrite into a golden -- and the decision "
        "in the GNDS roadmap's 'what will not be built' has to be revisited "
        "with it, because the reason it was declined was not difficulty."
    )
