"""§18 distributions, read and written back — phase 7b's ``uncorrelated``
and ``energyAngular``.

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

**``energyAngular`` has no witness among the three**, so its tests graft one in.
The only registered tape that carries a real one is ``fe56_gnds`` — 18.8 MB, on
the share, marked ``tape`` — and the gate that uses it is
``test_cross_section_oracle.py``'s report set. A grafted node still walks the
whole pipeline (``kika.read`` → ``kika.write`` → ``kika.read``) rather than the
writer alone, so what it does not cover is *the shapes a real evaluation uses*,
not the wiring.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (DiscreteGamma, EnergyAngular,
                                     Isotropic2d, NBodyPhaseSpace,
                                     PrimaryGamma, Uncorrelated, XYs2d, XYs3d)


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


# ---------------------------------------------------------------------------
# §18.4 energyAngular — grafted, because no committed fixture carries one
# ---------------------------------------------------------------------------

#: One ``<energyAngular>``: P(E′,mu|E) over two incident energies, the inner
#: ``XYs2d`` giving P(mu) at two outgoing energies each. Small on purpose — the
#: shape is what is under test, not the numbers.
ENERGY_ANGULAR_XML = """
<energyAngular label="graft" productFrame="lab">
  <XYs3d interpolationQualifier="unitbase">
    <axes>
      <axis index="3" label="energy_in" unit="eV"/>
      <axis index="2" label="energy_out" unit="eV"/>
      <axis index="1" label="mu" unit=""/>
      <axis index="0" label="P(mu,energy_out|energy_in)" unit=""/>
    </axes>
    <function2ds>
      <XYs2d outerDomainValue="1e6">
        <function1ds>
          <XYs1d outerDomainValue="1e4"><values>-1 0.4 1 0.6</values></XYs1d>
          <XYs1d outerDomainValue="2e4"><values>-1 0.3 1 0.7</values></XYs1d>
        </function1ds>
      </XYs2d>
      <XYs2d outerDomainValue="2e6">
        <function1ds>
          <XYs1d outerDomainValue="1e4"><values>-1 0.2 1 0.8</values></XYs1d>
          <XYs1d outerDomainValue="3e4"><values>-1 0.1 1 0.9</values></XYs1d>
        </function1ds>
      </XYs2d>
    </function2ds>
  </XYs3d>
</energyAngular>
"""


def graftDistributionForm(source, destination, xml: str):
    """Append one distribution form to the first ``<distribution>`` of a file.

    ``DistributionType`` (``gnds.xsd:1647``) is an unbounded ``xs:choice``, so a
    second form beside the one already there is what the schema expects of a
    file that states its distribution two ways, and the labels key them apart.
    """
    tree = ET.parse(source)
    distribution = tree.getroot().find(".//distribution")
    assert distribution is not None, source
    distribution.append(ET.fromstring(xml))
    tree.write(destination)
    return destination


def _grafted(source, tmp_path, xml: str = ENERGY_ANGULAR_XML):
    path = graftDistributionForm(source, tmp_path / "grafted.gnds.xml", xml)
    return readReactionSuite(Document.parse(path))


def _energyAngulars(suite):
    forms = []
    for container in (suite.reactions, suite.orphanProducts, suite.sums,
                      suite.productions, suite.fissionComponents):
        for reaction in container:
            channel = getattr(reaction, "outputChannel", None)
            if channel is None:
                continue
            for product in _products(channel):
                if product.distribution is None:
                    continue
                forms.extend(f for f in product.distribution.forms.values()
                             if isinstance(f, EnergyAngular))
    return forms


def _products(channel):
    for product in channel.products:
        yield product
        if product.outputChannel is not None:
            yield from _products(product.outputChannel)


def test_an_energy_angular_reads_into_one_xys3d(h2_gnds, tmp_path):
    """§18.4 is an ``xs:sequence`` of one ``XYs3d`` and the model says so.

    The label and ``productFrame`` are required attributes of the node, not of
    the function, so they live on :class:`EnergyAngular` — and the outermost
    grid belongs to the ``XYs3d``, which is what makes it a 3-d form rather
    than a dictionary of 2-d ones.
    """
    suite, report = _grafted(h2_gnds, tmp_path)
    forms = _energyAngulars(suite)
    assert len(forms) == 1
    form = forms[0]
    assert form.label == "graft"
    assert str(form.productFrame) == "lab"
    assert form.isComplete
    assert isinstance(form.xys3d, XYs3d)
    assert form.xys3d.outerDomainValues == [1e6, 2e6]
    assert [c.outerDomainValue for c in form.xys3d[1]] == [1e4, 3e4]
    assert not [e for e in report.unsupported if "energyAngular" in e]


def test_an_energy_angular_survives_a_round_trip(h2_gnds, tmp_path):
    """Through the whole pipeline, not through the writer alone.

    The ``axes`` identity is the part that a writer-only test cannot see: the
    reader shares one object from the ``XYs3d`` down to its 1-d grandchildren
    and the writer decides inheritance by ``is``, so a second read of what was
    written is what proves the file did not repeat them.
    """
    suite, _ = _grafted(h2_gnds, tmp_path)
    written = tmp_path / "out.gnds.xml"
    report = kika.write(suite, written)
    again, _ = readReactionSuite(Document.parse(written))

    before, after = _energyAngulars(suite)[0], _energyAngulars(again)[0]
    assert after.label == before.label
    assert after.productFrame == before.productFrame
    assert after.xys3d.outerDomainValues == before.xys3d.outerDomainValues
    assert after.xys3d.interpolationQualifier == \
        before.xys3d.interpolationQualifier
    for child in after.xys3d:
        assert child.axes is after.xys3d.axes
        for grandchild in child:
            assert grandchild.axes is after.xys3d.axes
    assert not [e for e in report.losses if "axes of its own" in e], \
        report.losses


def test_an_energy_angular_with_no_function_is_not_written_at_all(
        h2_gnds, tmp_path):
    """``gnds.xsd:1798-1800`` is an ``xs:sequence``: the child or nothing.

    :meth:`uncorrelated`'s judgement about its two halves, one node over. A
    ``<energyAngular>`` with no ``XYs3d`` is not a partial statement, it is an
    invalid one, so the refusal is the writer's and it reports why.
    """
    empty = ENERGY_ANGULAR_XML[:ENERGY_ANGULAR_XML.index("<XYs3d")] + \
        "<gridded3d/>" + \
        ENERGY_ANGULAR_XML[ENERGY_ANGULAR_XML.index("</energyAngular>"):]
    suite, readReport = _grafted(h2_gnds, tmp_path, empty)

    assert _energyAngulars(suite)[0].isComplete is False
    assert [e for e in readReport.unsupported if "gridded3d" in e], \
        readReport.unsupported

    written = tmp_path / "out.gnds.xml"
    report = kika.write(suite, written)
    root = ET.parse(written).getroot()
    assert not list(root.iter("energyAngular"))
    assert any("requires the child" in entry for entry in report.unsupported)


def test_angular_energy_is_reported_rather_than_read_as_its_mirror(
        h2_gnds, tmp_path):
    """The two share a complexType and differ only in which variable is outer.

    So the same bytes are a *valid* ``angularEnergy`` and a *valid*
    ``energyAngular``, and nothing in the schema distinguishes them — which is
    exactly why kika does not reuse the reader. Decoding one into
    :class:`EnergyAngular` would produce a model, and a file written back from
    it, that states the wrong physics with no gate anywhere able to see it.
    """
    mirrored = ENERGY_ANGULAR_XML.replace("energyAngular", "angularEnergy")
    suite, report = _grafted(h2_gnds, tmp_path, mirrored)

    assert _energyAngulars(suite) == [], (
        "an angularEnergy must not arrive as an EnergyAngular"
    )
    assert [e for e in report.unsupported if "/angularEnergy:" in e], \
        report.unsupported

