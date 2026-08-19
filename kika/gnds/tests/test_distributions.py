"""§18 distributions, read and written back — phase 7b's ``uncorrelated``
and ``energyAngular``.

Beside ``test_resonances.py`` and for the same reason: the chapter has its own
reader module now, and the assertions that belong to it are the shapes §18.3
takes rather than anything about a ``reactionSuite``.

**What the four committed fixtures cover between them**, counted from the XML
rather than assumed:

===============  ==============  ==========================================
fixture          ``<angular>``   ``<energy>``
===============  ==============  ==========================================
``h2``           isotropic2d ×3  NBodyPhaseSpace ×2, primaryGamma ×1
``micro_fe56``   isotropic2d ×1  XYs2d ×1 (``interpolationQualifier``)
``micro_ta182``  isotropic2d ×21 discreteGamma ×20, XYs2d ×1
``h3``           **XYs2d ×1**    XYs2d ×1
===============  ==============  ==========================================

So four of ``uncorrelated/energy``'s eleven choices have a witness here and
seven do not — the six analytic spectra and ``regions2d``. Those are declared
in :data:`kika.gnds.nodes.NODES` and reported, and the tests at the bottom are
about the reporting rather than about the forms.

**``h3`` is the fourth fixture and it was added for one cell of that table.**
``uncorrelated/angular`` has three members (``gnds.xsd:1686``) and the first
three fixtures are ``isotropic2d`` twenty-five times out of twenty-five, so the
``XYs2d`` branch shipped with ``uncorrelated`` unwitnessed. The census puts it
at 406 occurrences in 144 of the 558 distributed evaluations — not a rare
shape, just one the committed set happened to miss.

**Neither ordering of ``DistributionAEType`` has a witness among the four**, so
their tests graft one in. For ``energyAngular`` the registered tape that carries
real ones is ``fe56_gnds`` — 18.8 MB, on the share, marked ``tape`` — and the
gate that uses it is ``test_cross_section_oracle.py``'s report set. For
``angularEnergy`` there is no tape at all: the census found **two occurrences in
the whole distribution**, both in ``n-004_Be_009``, which is what
``micro_be9_gnds`` is trimmed from. A grafted node still walks the whole
pipeline (``kika.read`` → ``kika.write`` → ``kika.read``) rather than the writer
alone, so what it does not cover is *the shapes a real evaluation uses*, not the
wiring.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (AngularEnergy, DiscreteGamma,
                                     EnergyAngular, Isotropic2d,
                                     NBodyPhaseSpace, PrimaryGamma,
                                     Uncorrelated, XYs2d, XYs3d)


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


@pytest.fixture(scope="module")
def h3(h3_gnds):
    return readReactionSuite(Document.parse(h3_gnds))


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


def test_a_tabulated_angular_half_is_read_from_a_real_evaluation(h3):
    """The ``XYs2d`` branch of ``uncorrelated/angular``, on a file that has one.

    It shipped with ``uncorrelated`` and had **no witness**: the other three
    fixtures are 25 ``isotropic2d`` out of 25, and the branch was reached only
    by the isotropic sibling beside it. The census counted 406 real ones across
    144 of the 558 distributed evaluations, so this is a branch the library
    walks and not a defensive one — ``h3_gnds`` is the smallest file carrying
    it.

    The assertions are the three things an ``isotropic2d`` cannot check:
    that the sub-functions arrive (an empty ``XYs2d`` reads as a present node
    and writes an invalid one), that the node kept **its own** ``axes``
    (``gnds.xsd:2195`` makes them required, and the angular ones are
    ``mu``/``P(mu|energy_in)`` rather than the energy axes of the sibling), and
    that the frame still comes from the parent, since ``xData_XYs2d_primary``
    has no ``productFrame`` of its own.
    """
    suite, report = h3
    forms = _uncorrelated(suite)
    assert len(forms) == 1
    angular = forms[0].angular
    assert isinstance(angular, XYs2d)
    assert len(angular.function1ds) == 15
    assert angular.function1ds[0].outerDomainValue == pytest.approx(8.35e6)
    assert angular.axes is not None
    assert [axis.label for axis in angular.axes.axes] == [
        "energy_in", "mu", "P(mu|energy_in)"]
    assert forms[0].productFrame == "lab"
    assert report.unsupported == [], (
        "every law in this file is one kika reads, and that is half of why it "
        "is the fixture: an entry here is a regression and not a known gap"
    )


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
                                  "micro_ta182_gnds", "h3_gnds"])
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
        if isinstance(one.angular, XYs2d):
            # ``h3_gnds`` is the only fixture that reaches this branch, and it
            # is the reason it was committed: comparing the *type* alone would
            # pass on an angular half written empty.
            assert len(one.angular) == len(two.angular)
            assert [f.outerDomainValue for f in one.angular.function1ds] == \
                [f.outerDomainValue for f in two.angular.function1ds]
            assert [axis.label for axis in one.angular.axes.axes] == \
                [axis.label for axis in two.angular.axes.axes]


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


def _formsOfType(suite, cls):
    """Every distribution form of exactly one class, in document order.

    ``isinstance`` and not ``type() is``: it is the same check the writer
    dispatches on, so a subclass relationship between the two orderings would
    show up here as it would there. There is none, and that is the point.
    """
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
                             if isinstance(f, cls))
    return forms


def _energyAngulars(suite):
    return _formsOfType(suite, EnergyAngular)


def _angularEnergies(suite):
    return _formsOfType(suite, AngularEnergy)


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


def test_an_angular_energy_reads_into_its_own_class_and_never_the_mirror(
        h2_gnds, tmp_path):
    """The two share a complexType and differ only in which variable is outer.

    So the same bytes are a *valid* ``angularEnergy`` and a *valid*
    ``energyAngular``, and **nothing in the schema distinguishes them**. That is
    what makes this pair worth a test of its own rather than a copy of §18.4's:
    the element name is the entire signal, so the only failure mode that matters
    is the two crossing over, and no validator anywhere downstream could catch
    it. Read one, assert it is not the other; then the mirror image below.

    This test used to assert the opposite — that an ``angularEnergy`` was
    *reported* rather than read — because §18.5 was undecided rather than
    queued. The census settled it at two occurrences, so it is now implemented
    and this is the stronger statement in the same place.
    """
    mirrored = ENERGY_ANGULAR_XML.replace("energyAngular", "angularEnergy")
    suite, report = _grafted(h2_gnds, tmp_path, mirrored)

    forms = _angularEnergies(suite)
    assert len(forms) == 1
    assert isinstance(forms[0].xys3d, XYs3d)
    assert [f.outerDomainValue for f in forms[0].xys3d.function2ds] == [1e6, 2e6]
    assert _energyAngulars(suite) == [], (
        "an angularEnergy must not arrive as an EnergyAngular"
    )
    assert not [e for e in report.unsupported if "angularEnergy" in e], \
        report.unsupported


def test_an_energy_angular_never_arrives_as_an_angular_energy(h2_gnds,
                                                              tmp_path):
    """The other half of the pair, and it is not redundant.

    A dispatch that collapsed both tags onto one class would pass the test
    above as long as that class happened to be :class:`AngularEnergy`. Reading
    the unmirrored XML and asserting the *complement* is what makes the pair
    say "these two are distinguished", rather than "one of them is recognised".
    """
    suite, report = _grafted(h2_gnds, tmp_path)

    assert len(_energyAngulars(suite)) == 1
    assert _angularEnergies(suite) == []
    assert report.unsupported == []


def test_the_real_angular_energy_puts_mu_outside_the_outgoing_energy(
        micro_be9_gnds):
    """The two §18.5 nodes that exist, and the axes are what make them §18.5.

    ``micro_be9`` is the trim of the only file in the distribution that carries
    an ``angularEnergy``, and both of its occurrences are here — one per product
    of ``2n + 2He4``.

    **The assertion that matters is the axis order.** ``DistributionAEType``
    (``gnds.xsd:1797``) is shared with ``energyAngular``, so nothing structural
    tells the two apart; what does is which variable the file nests outermost,
    and the axes say it in words. Here it reads ``energy_in`` → ``mu`` →
    ``energy_out``, and the grafted ``energyAngular`` beside it reads
    ``energy_in`` → ``energy_out`` → ``mu``. If kika ever decoded one into the
    other's class this test is where the evaluation itself contradicts it.
    """
    suite, report = readReactionSuite(Document.parse(micro_be9_gnds))

    forms = _angularEnergies(suite)
    assert len(forms) == 2
    assert _energyAngulars(suite) == []
    assert report.unsupported == []
    for form in forms:
        assert form.isComplete
        assert form.productFrame == "lab"
        assert [axis.label for axis in form.xys3d.axes.axes] == [
            "energy_in", "mu", "energy_out", "P(mu,energy_out|energy_in)"]
        assert form.xys3d.function2ds, "an XYs3d with no children is invalid"


def test_a_real_angular_energy_survives_a_round_trip(micro_be9_gnds, tmp_path):
    """Read → write → read on the witness, not on a graft.

    The graft tests below cover the wiring; this covers the shapes an actual
    evaluation uses, which is the whole reason the file was committed. The tag
    has to come back as ``angularEnergy``, and the axis order with it — writing
    the axes in the other order would be the same error as writing the other
    tag, and just as invisible to a validator.
    """
    first, _ = readReactionSuite(Document.parse(micro_be9_gnds))
    path = tmp_path / "be9-out.gnds.xml"
    kika.write(first, path)
    second, report = readReactionSuite(Document.parse(path))

    before, after = _angularEnergies(first), _angularEnergies(second)
    assert len(before) == len(after) == 2
    assert _energyAngulars(second) == []
    assert list(ET.parse(path).getroot().iter("angularEnergy"))
    for one, two in zip(before, after):
        assert one.label == two.label
        assert one.productFrame == two.productFrame
        assert [a.label for a in one.xys3d.axes.axes] == \
            [a.label for a in two.xys3d.axes.axes]
        assert [f.outerDomainValue for f in one.xys3d.function2ds] == \
            [f.outerDomainValue for f in two.xys3d.function2ds]


def test_the_two_orderings_write_back_the_tag_they_were_read_from(h2_gnds,
                                                                  tmp_path):
    """Read → write, once per ordering, checking the element name survives.

    The model round trip above proves the class is right; this proves the
    *writer* does not collapse them either. Both directions matter because the
    reader and the writer dispatch through different mechanisms — a tag
    comparison one way, ``isinstance`` the other — so a defect in one is
    invisible to a test of the other.
    """
    for tag in ("energyAngular", "angularEnergy"):
        xml = ENERGY_ANGULAR_XML.replace("energyAngular", tag)
        source = graftDistributionForm(h2_gnds, tmp_path / f"{tag}.gnds.xml",
                                       xml)
        suite, _ = readReactionSuite(Document.parse(source))

        written = tmp_path / f"{tag}-out.gnds.xml"
        kika.write(suite, written)
        root = ET.parse(written).getroot()

        assert len(list(root.iter(tag))) == 1, tag
        other = "angularEnergy" if tag == "energyAngular" else "energyAngular"
        assert not list(root.iter(other)), (
            f"a <{tag}> was written back as a <{other}>, which validates and "
            f"states the wrong physics"
        )

