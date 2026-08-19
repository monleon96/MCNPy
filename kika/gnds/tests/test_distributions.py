"""§18 distributions, read and written back — phase 7b's ``uncorrelated``.

Beside ``test_resonances.py`` and for the same reason: the chapter has its own
reader module now, and the assertions that belong to it are the shapes §18.3
takes rather than anything about a ``reactionSuite``.

**What the three committed fixtures cover between them**, counted from the XML
rather than assumed:

===============  ==============  ==========================================
fixture          ``<angular>``   ``<energy>``
===============  ==============  ==========================================
``h2``           isotropic2d ×3  NBodyPhaseSpace ×2, primaryGamma ×1
``micro_fe56``   isotropic2d ×1  XYs2d ×1 (``interpolationQualifier``)
``micro_ta182``  isotropic2d ×21 discreteGamma ×20, XYs2d ×1
===============  ==============  ==========================================

So four of ``uncorrelated/energy``'s eleven choices have a witness here and
seven do not — the six analytic spectra and ``regions2d``. Those are declared
in :data:`kika.gnds.nodes.NODES` and reported, and the tests at the bottom are
about the reporting rather than about the forms.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (DiscreteGamma, Isotropic2d,
                                     NBodyPhaseSpace, PrimaryGamma,
                                     Uncorrelated, XYs2d)


def _uncorrelated(suite):
    """Every ``Uncorrelated`` in a suite, in document order."""
    forms = []
    for container in (suite.reactions, suite.orphanProducts, suite.sums,
                      suite.productions, suite.fissionComponents):
        for reaction in container:
            channel = getattr(reaction, "outputChannel", None)
            if channel is None:
                continue
            forms.extend(_fromChannel(channel))
    return forms


def _fromChannel(channel):
    for product in channel.products:
        if product.distribution is not None:
            forms = product.distribution.forms.values()
            yield from (f for f in forms if isinstance(f, Uncorrelated))
        if product.outputChannel is not None:
            yield from _fromChannel(product.outputChannel)


@pytest.fixture(scope="module")
def h2(h2_gnds):
    return readReactionSuite(Document.parse(h2_gnds))[0]


@pytest.fixture(scope="module")
def ta182(micro_ta182_gnds):
    return readReactionSuite(Document.parse(micro_ta182_gnds))[0]


@pytest.fixture(scope="module")
def fe56(micro_fe56_gnds):
    return readReactionSuite(Document.parse(micro_fe56_gnds))[0]


# ---------------------------------------------------------------------------
# 1. the four energy forms that have a witness
# ---------------------------------------------------------------------------

def test_a_discrete_gamma_keeps_its_line_and_its_range(ta182):
    gammas = [f.energy for f in _uncorrelated(ta182)
              if isinstance(f.energy, DiscreteGamma)]
    assert len(gammas) == 20
    assert all(not isinstance(g, PrimaryGamma) for g in gammas), (
        "a primaryGamma must not answer isinstance(DiscreteGamma) — the "
        "writer dispatches on it and would emit one as the other"
    )
    assert all(g.domainMin == pytest.approx(1e-5) for g in gammas)
    assert all(g.axes is not None for g in gammas), (
        "gnds.xsd:1777 makes <axes> a required child, so losing it here "
        "writes an invalid node back out"
    )


def test_a_primary_gamma_is_not_a_discrete_one(h2):
    gammas = [f.energy for f in _uncorrelated(h2)
              if isinstance(f.energy, PrimaryGamma)]
    assert len(gammas) == 1
    assert gammas[0].value == pytest.approx(6251002.0)
    assert gammas[0].domainMax == pytest.approx(1.5e8)
    assert gammas[0].axes is not None


def test_an_n_body_phase_space_carries_its_mass_with_its_unit(h2):
    """§4.1's lesson, applied where the file *does* state the unit.

    ``<mass value= unit=>`` is a PhysicalQuantity, and it is kept as one rather
    than as a float plus a convention — which is the shape that cost a factor
    of ten on the scattering radius.
    """
    spaces = [f.energy for f in _uncorrelated(h2)
              if isinstance(f.energy, NBodyPhaseSpace)]
    assert len(spaces) == 2
    assert all(s.numberOfProducts == 3 for s in spaces)
    assert spaces[0].mass.value == pytest.approx(3.02460278964)
    assert spaces[0].mass.unit == "amu"


def test_a_tabulated_energy_keeps_its_interpolation_qualifier(fe56):
    """The one ``XYs2d`` energy in the committed set, and it is qualified.

    ``interpolationQualifier="unitbase"`` is not decoration: §3.4.5 makes it
    the rule for interpolating between two outgoing spectra, and dropping it
    would leave a file that interpolates differently and says nothing about it.
    """
    energies = [f.energy for f in _uncorrelated(fe56)]
    assert len(energies) == 1
    assert isinstance(energies[0], XYs2d)
    assert str(energies[0].interpolationQualifier) == "unitBase"


def test_the_angular_half_takes_the_frame_from_its_parent(ta182):
    """``<isotropic2d/>`` inside ``<angular>`` has no attributes at all
    (``gnds.xsd:1693``), so a child left on the class default would claim
    ``centerOfMass`` on a lab-frame distribution."""
    forms = _uncorrelated(ta182)
    assert forms
    for form in forms:
        assert isinstance(form.angular, Isotropic2d)
        assert form.angular.productFrame == form.productFrame == "lab"


# ---------------------------------------------------------------------------
# 2. the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["h2_gnds", "micro_fe56_gnds",
                                  "micro_ta182_gnds"])
def test_every_uncorrelated_survives_a_round_trip(name, request, tmp_path):
    """Read → write → read, comparing the energy halves member by member.

    The fixed-point test in ``test_encode.py`` compares two *written* files,
    which is blind to anything both writes drop the same way. This compares the
    model against itself across a write, so a member the writer forgets shows
    up as a difference and not as a matching absence.
    """
    source = request.getfixturevalue(name)
    first = readReactionSuite(Document.parse(source))[0]

    path = tmp_path / "out.gnds.xml"
    kika.write(first, path)
    second = readReactionSuite(Document.parse(path))[0]

    before, after = _uncorrelated(first), _uncorrelated(second)
    assert len(before) == len(after) > 0
    for one, two in zip(before, after):
        assert type(one.energy) is type(two.energy)
        assert type(one.angular) is type(two.angular)
        assert one.productFrame == two.productFrame
        assert one.label == two.label
        if isinstance(one.energy, (DiscreteGamma, PrimaryGamma)):
            assert one.energy.value == two.energy.value
            assert one.energy.domainMin == two.energy.domainMin
            assert one.energy.domainMax == two.energy.domainMax
            assert getattr(one.energy, "finalState", None) == \
                getattr(two.energy, "finalState", None)
            assert len(list(one.energy.axes)) == len(list(two.energy.axes))
        if isinstance(one.energy, NBodyPhaseSpace):
            assert one.energy.numberOfProducts == two.energy.numberOfProducts
            assert one.energy.mass.value == two.energy.mass.value
            assert one.energy.mass.unit == two.energy.mass.unit
        if isinstance(one.energy, XYs2d):
            assert len(one.energy) == len(two.energy)
            assert one.energy.interpolationQualifier == \
                two.energy.interpolationQualifier


# ---------------------------------------------------------------------------
# 3. the seven forms with no witness, and the half node the schema forbids
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["evaporation", "generalEvaporation",
                                 "simpleMaxwellianFission", "Watt",
                                 "MadlandNix", "weightedFunctionals"])
def test_an_analytic_spectrum_is_reported_and_not_guessed(tag, h2_gnds,
                                                          tmp_path):
    """Six of the eleven energy choices are formulae with named parameters.

    Tabulating one would put numbers in the file the evaluator never wrote, so
    each is reported with its xPath instead — and the reason comes from
    ``nodes.NODES``, so the report and the table cannot say different things.
    """
    tree = ET.parse(h2_gnds)
    energy = tree.getroot().find(".//uncorrelated/energy")
    for child in list(energy):
        energy.remove(child)
    ET.SubElement(energy, tag)
    path = tmp_path / f"{tag}.gnds.xml"
    tree.write(path)

    suite, report = readReactionSuite(Document.parse(path))
    named = [e for e in report.unsupported if f"/{tag}:" in e]
    assert len(named) == 1
    assert "analytic §18.3 spectrum" in named[0]
    assert named[0].startswith("/reactionSuite/reactions/reaction[@label=")

    halved = [f for f in _uncorrelated(suite) if not f.isComplete]
    assert len(halved) == 1
    assert isinstance(halved[0].angular, Isotropic2d), (
        "the half that could be read is kept; dropping it as well would lose "
        "something the file did state"
    )


def test_a_forward_angular_form_is_reported(h2_gnds, tmp_path):
    """``<forward/>`` (``gnds.xsd:1694``) exists at no other choice point in
    the schema and in no committed fixture. Declared, reported, not modelled."""
    tree = ET.parse(h2_gnds)
    angular = tree.getroot().find(".//uncorrelated/angular")
    for child in list(angular):
        angular.remove(child)
    ET.SubElement(angular, "forward")
    path = tmp_path / "forward.gnds.xml"
    tree.write(path)

    _suite, report = readReactionSuite(Document.parse(path))
    assert [e for e in report.unsupported if "/forward:" in e]


def test_a_half_read_uncorrelated_is_not_written_at_all(h2_gnds, tmp_path):
    """``gnds.xsd:1677-1680`` is an ``xs:sequence``: both children or none.

    A node with only ``<angular>`` would validate against nothing and read as a
    complete statement, which is worse than the empty ``<distribution/>`` the
    writer emits instead. The refusal lives in one place and reports why.
    """
    tree = ET.parse(h2_gnds)
    energy = tree.getroot().find(".//uncorrelated/energy")
    for child in list(energy):
        energy.remove(child)
    ET.SubElement(energy, "Watt")
    source = tmp_path / "watt.gnds.xml"
    tree.write(source)

    suite, _ = readReactionSuite(Document.parse(source))
    written = tmp_path / "out.gnds.xml"
    report = kika.write(suite, written)

    root = ET.parse(written).getroot()
    assert [d for d in root.iter("distribution") if len(d) == 0]
    assert all(len(u) == 2 for u in root.iter("uncorrelated")), (
        "every uncorrelated that *is* written carries both halves"
    )
    assert any("requires both children" in entry
               for entry in report.unsupported)
