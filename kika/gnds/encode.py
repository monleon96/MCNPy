"""The model → GNDS XML. The other direction, and the harder rule.

Reading a file kika only half-understands is fine: the parts it does understand
are right, and the report says what the rest was. **Writing one is not.** A file
that comes out of here is indistinguishable from an evaluation, and anything
missing from it is not "unread" — it is *asserted absent*. The roadmap's rule
for this phase is written for exactly that asymmetry: *"the writer emits only
what kika owns and declares the gaps in the returned ConversionReport. A
structurally valid, physically incomplete reactionSuite is worse than none,
because it carries authority it has not earned."*

Three consequences, all of them visible in the code below:

**Nothing is invented to fill a hole.** A product whose §18 law kika does not
read comes out with an *empty* ``<distribution/>``, not with an
``<unspecified/>``. The two would look identical to a schema and mean opposite
things — ``unspecified`` is the evaluator stating that no distribution is given,
and writing it where kika merely failed to read one would forge a statement
nobody made. The empty element does not validate, which is the point: the file
announces its own incompleteness to any validator, and
:func:`writeReactionSuite`'s report names every product it happened to.

**Nothing is recomputed.** A ``crossSectionSum``'s values are written as the
evaluation stated them, not as the sum of its summands; a
``resonancesWithBackground``'s background regions are written as three curves,
not reconstructed and flattened into one.

**Byte identity with FUDGE is not a goal and is not achievable.** FUDGE writes
``1e-5`` where Python's shortest round-tripping repr is ``1e-05``, and its
indentation and attribute order are its own. What *is* guaranteed is that every
number written here reads back as the identical double — :func:`_number` uses
``repr``, which is the shortest string that round-trips — so
``read → write → read`` is a fixed point on the numbers, and
``write → read → write`` is a fixed point on the tree. Those are the gates.

**The covarianceSuite is a separate document**, per §25.1.1, and
:func:`writeCovarianceSuite` produces it. ``kika.write`` emits both and links
them, because a ``reactionSuite`` whose ``externalFiles`` names a file that was
never written is worse than one with no ``externalFiles`` at all.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (AngularTwoBody,
                                     AverageParameterCovariance, Background,
                                     BreitWigner, ConversionReport,
                                     Constant1d, CovarianceMatrix,
                                     CrossSectionSum, DiscreteGamma,
                                     EnergyAngular, Evaluated, Isotropic2d,
                                     Legendre, Mixed, NBodyPhaseSpace,
                                     Nuclide, ParameterCovariance,
                                     Polynomial1d, PrimaryGamma,
                                     GndsProvenance, Reference,
                                     Regions1d, Regions2d,
                                     ResonancesWithBackground, RMatrix,
                                     ShortRangeSelfScalingVariance, Sum,
                                     Uncorrelated, Unspecified, XYs1d,
                                     XYs2d, XYs3d)

from .nodes import writes
from .primitives import formatFraction
from .styles import writeStyles
from .version import ACCEPTED, DEFAULT_WRITE_FORMAT, UnsupportedGndsVersion

__all__ = ["writeReactionSuite", "writeCovarianceSuite", "chooseFormat",
           "serialise"]

#: §14.1.1's children, in the order ``ReactionSuiteType`` declares them. XML
#: Schema sequences are ordered, so a writer that emits ``sums`` before
#: ``reactions`` produces a file no validator accepts and every reader still
#: reads — the worst kind of defect, invisible until somebody else's tool.
SUITE_ORDER = ("externalFiles", "styles", "PoPs", "resonances", "reactions",
               "orphanProducts", "sums", "fissionComponents", "productions",
               "incompleteReactions", "applicationData")

#: model container → (wrapper node, child node). The five §14.1.1 reaction lists.
REACTION_LISTS = (
    ("reactions", "reactions", "reaction"),
    ("orphanProducts", "orphanProducts", "orphanProduct"),
    ("fissionComponents", "fissionComponents", "fissionComponent"),
    ("productions", "productions", "production"),
    ("incompleteReactions", "incompleteReactions", "reaction"),
)


def _number(value) -> str:
    """One float, as the shortest string that reads back identically.

    ``repr`` on a Python float is guaranteed to round-trip, which is the
    property the gates rest on. It is *not* FUDGE's spelling — FUDGE writes
    ``1e-5`` where this writes ``1e-05`` — and that difference is why byte
    identity with the distribution is explicitly not a goal.
    """
    return repr(float(value))


def _numbers(values) -> str:
    return " ".join(_number(v) for v in np.asarray(values).ravel())


def _set(element: ET.Element, **attributes) -> ET.Element:
    """Set the attributes that are not ``None``, in the order given."""
    for name, value in attributes.items():
        if value is not None:
            element.attrib[name] = value
    return element


def _boolean(value: bool) -> Optional[str]:
    """``true`` or nothing. GNDS omits a false flag rather than writing it."""
    return "true" if value else None


def chooseFormat(suite, requested: Optional[str] = None) -> str:
    """Which ``format`` this file declares.

    The decision, in order:

    1. ``requested`` wins, and is refused if it is not one this library reads —
       writing a version kika could not read back is a trap for its own user.
    2. A suite that **came from GNDS** keeps the version it came from. A
       round trip that silently upgraded 2.0 to 2.1 would make every diff
       against the source file start with a changed root attribute.
    3. Anything else — an ENDF- or ACE-sourced suite, which carries a
       ``provenance`` from that format and no GNDS origin — gets
       :data:`~kika.gnds.version.DEFAULT_WRITE_FORMAT`, today **2.0**, because
       that is what every tool in the field currently emits and reads. Flip it
       when the libraries move; the model has been 2.1 all along and this is a
       statement about what to hand somebody else, not about what kika is.
    """
    if requested is not None:
        if requested not in ACCEPTED:
            raise UnsupportedGndsVersion(
                f"kika can write GNDS {' and '.join(ACCEPTED)}, not {requested!r}. "
                f"Writing a version this library could not read back would be a "
                f"trap for whoever opens the file next."
            )
        return requested
    provenance = getattr(suite, "provenance", None)
    if isinstance(provenance, GndsProvenance) and provenance.formatVersion in ACCEPTED:
        return provenance.formatVersion
    return DEFAULT_WRITE_FORMAT


# ---------------------------------------------------------------------------
# functionals
# ---------------------------------------------------------------------------

def _axes(parent: ET.Element, axes) -> None:
    if axes is None:
        return
    element = ET.SubElement(parent, "axes")
    if getattr(axes, "href", None):
        element.attrib["href"] = axes.href
        return
    for axis in axes:
        _set(ET.SubElement(element, "axis"), index=str(axis.index),
             label=axis.label or "", unit=axis.unit or "")


def _values(parent: ET.Element, numbers, label: Optional[str] = None) -> ET.Element:
    element = ET.SubElement(parent, "values")
    if label is not None:
        element.attrib["label"] = label
    element.text = _numbers(numbers)
    return element


def _interpolation(element: ET.Element, form) -> None:
    """Write ``interpolation`` only when it is not the default.

    §6.1.1's default is lin-lin and most of the library omits the attribute, so
    writing it everywhere would put a changed line on almost every functional in
    a diff against the source without changing what the file says.
    """
    interpolation = getattr(form, "interpolation", None)
    if interpolation is not None and str(interpolation) != "lin-lin":
        element.attrib["interpolation"] = str(interpolation)


@writes("function1d", "XYs1d", "regions1d", "constant1d", "Legendre",
        "polynomial1d")
@writes("function2d", "XYs2d", "regions2d")
@writes("function3d", "XYs3d")
def _function(parent: ET.Element, form, report: ConversionReport,
              where: str, nested: bool = False,
              parentAxes=None, index: Optional[int] = None) -> Optional[ET.Element]:
    """One functional → its node, or ``None`` plus a report entry.

    ``nested`` is what makes the difference between the schema's
    ``xData_XYs1d_primary`` — the top of a container, which **must** carry
    ``axes`` — and ``xData_XYs1d``, a child inside ``function1ds``, which must
    **not**: §5.1.1 has a child inherit its parent's axes and the schema
    forbids it from repeating them. The reader shares one ``Axes`` object
    between a container and its children, so writing them again would produce a
    file that is invalid and that says nothing new.

    ``index`` is the second half of the same idea, for attributes rather than
    for ``axes``: it is the ordinal this form has **inside a regions
    container**, and ``None`` everywhere else. See :func:`_writeCommon` for why
    the two positions cannot both write the same attributes.
    """
    if isinstance(form, XYs1d):
        element = ET.SubElement(parent, "XYs1d")
        _writeCommon(element, form, nested, index)
        _interpolation(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.interleaved())
        return element
    if isinstance(form, Regions1d):
        # A region set of one is not a region set: `function1ds_inRegions`
        # (`gnds.xsd:2143-2147`) puts `minOccurs="2"` on its children, so the
        # node the schema wants here is the single `XYs1d` itself. kika's ENDF
        # cross sections are `Regions1d` whatever the tape's NR says --
        # `decodeMF3MT` keeps one shape so there is one inverse -- and MT1, MT2
        # and MT102 of the Fe-56 micro-tape are all NR=1.
        #
        # Done here and not in the model on purpose: changing what
        # `CrossSection[...]` *is* would reach the flat-path parity tests and
        # the processing code, for a question that is only about how GNDS
        # spells it. The model keeps one shape; the writer spells it the way
        # the schema reads it. Inert on the three committed GNDS fixtures --
        # none holds a single-region container, because none could and validate.
        if len(form.function1ds) == 1:
            only = form.function1ds[0]
            collapsed = _function(parent, only, report, where, nested, parentAxes, index)
            if collapsed is not None and not nested:
                # The primary's `label` and `axes` describe the curve, not the
                # wrapper, so they survive the collapse.
                _set(collapsed, label=getattr(form, "label", None))
                if collapsed.find("axes") is None:
                    _axes(collapsed, form.axes)
            return collapsed
        element = ET.SubElement(parent, "regions1d")
        _writeCommon(element, form, nested, index)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function1ds")
        for position, child in enumerate(form.function1ds):
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes, index=position)
        return element
    if isinstance(form, Constant1d):
        element = ET.SubElement(parent, "constant1d")
        _writeCommon(element, form, nested, index)
        # `form.constant`, not `form.value`: the model spells the number
        # `constant` (functions/simple.py:27), and this line asked for a `value`
        # no Constant1d has ever had -- so writing a constant1d as a *functional*
        # raised AttributeError. Unreachable through the committed fixtures,
        # none of which carries a constant1d cross section, and found by
        # test_nodes' behavioural writer check: it builds a minimal instance of
        # every class the registry keys on and asserts the tag that comes out.
        element.attrib["value"] = _number(form.constant)
        _set(element,
             domainMin=None if form.domainMin is None else _number(form.domainMin),
             domainMax=None if form.domainMax is None else _number(form.domainMax))
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        return element
    if isinstance(form, Legendre):
        element = ET.SubElement(parent, "Legendre")
        _writeCommon(element, form, nested, index)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.coefficients)
        return element
    if isinstance(form, Polynomial1d):
        element = ET.SubElement(parent, "polynomial1d")
        _writeCommon(element, form, nested, index)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.coefficients)
        return element
    if isinstance(form, XYs2d):
        element = ET.SubElement(parent, "XYs2d")
        _writeCommon(element, form, nested, index)
        _interpolation(element, form)
        qualifier = getattr(form, "interpolationQualifier", None)
        if qualifier is not None:
            # §3.4.5 spells it `unitBase`; FUDGE and all 558 distributed files
            # write `unitbase`. The reader accepts both; the writer emits the
            # one the field actually uses, so a file kika writes opens in FUDGE.
            element.attrib["interpolationQualifier"] = (
                "unitbase" if str(qualifier) == "unitBase" else str(qualifier)
            )
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function1ds")
        for child in form.function1ds:
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes)
        return element
    if isinstance(form, XYs3d):
        element = ET.SubElement(parent, "XYs3d")
        _writeCommon(element, form, nested, index)
        _interpolation(element, form)
        qualifier = getattr(form, "interpolationQualifier", None)
        if qualifier is not None:
            element.attrib["interpolationQualifier"] = (
                "unitbase" if str(qualifier) == "unitBase" else str(qualifier)
            )
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function2ds")
        for child in form.function2ds:
            # No `index`: `function2ds` (`gnds.xsd:2253`) is a choice of
            # `xData_XYs2d`/`xData_regions_2d`, and both put `use="required"` on
            # `outerDomainValue` and have no `index` attribute at all. That is
            # the opposite of `function2ds_inRegions`, whose children are
            # indexed and carry no outer value -- the same split `_writeCommon`
            # already handles one floor down.
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes)
        return element
    if isinstance(form, Regions2d):
        element = ET.SubElement(parent, "regions2d")
        _writeCommon(element, form, nested, index)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function2ds")
        for position, child in enumerate(form.function2ds):
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes, index=position)
        return element

    report.unsupportedNode(
        f"{where}: kika's writer has no serialisation for a "
        f"{type(form).__name__}; the node is absent from the file"
    )
    return None


def _axesUnlessNested(element: ET.Element, form, report: ConversionReport,
                      where: str, nested: bool, parentAxes) -> None:
    """Write ``axes`` at the top of a container and never on a child.

    The reader hands a container's ``Axes`` **object** to each of its children,
    so ``child.axes is parent.axes`` is how inheritance is spelled in the model
    and is silent here. A child holding a *different* axes object is something
    no distributed file does and the schema has no slot for, so it is reported
    rather than written into a position that would make the file invalid.
    """
    if not nested:
        _axes(element, form.axes)
        return
    own = getattr(form, "axes", None)
    if own is not None and own is not parentAxes:
        report.lost(
            f"{where}: a nested {type(form).__name__} carries axes of its own, "
            f"which §5.1.1 has no slot for on a child; they are not written"
        )


def _writeCommon(element: ET.Element, form, nested: bool = False,
                 index: Optional[int] = None) -> None:
    """``label``, ``index`` and ``outerDomainValue`` — **one** of the three, and
    which one is decided by where the node sits, not by what the model holds.

    This used to write all three whenever the model carried them, which is a
    statement the schema does not allow anyone to make. ``gnds.xsd:2109-2245``
    gives a functional three distinct types by position, and they are mutually
    exclusive:

    ============================  ====================  ==================
    position                      type                  attribute
    ============================  ====================  ==================
    top of a container            ``xData_*_primary``   ``label``
    child of a regions container  ``xData_*_inRegions`` ``index`` (required)
    child of an ``XYs2d``         ``xData_XYs1d``, …    ``outerDomainValue``
    ============================  ====================  ==================

    Writing an ``index`` on the Legendre children of an ``XYs2d`` is what made
    3 960 of the 3 999 errors that the missing MF4 ``axes`` was hiding — the
    schema rejects the attribute outright there, and the whole subtree was
    unreachable so nothing said so.

    The ordinal is passed in rather than read off ``form.index``: the two
    ``XYs2d`` a LTT=3 ``Regions2d`` holds have no ``index`` in the model at all,
    and ``Regions1d.fromEndfRegions`` skips empty regions while its counter
    keeps going, so a model index can have a hole in it. §6.4's ``index`` *is*
    the position, and taking it from :func:`enumerate` is the only spelling that
    cannot disagree with the file.
    """
    if index is not None:
        element.attrib["index"] = str(index)
        return
    if nested:
        outer = getattr(form, "outerDomainValue", None)
        _set(element,
             outerDomainValue=None if outer is None else _number(outer))
        return
    _set(element, label=getattr(form, "label", None))


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------

class _SuiteWriter:
    """One write of one ``reactionSuite``."""

    def __init__(self, suite, report: ConversionReport) -> None:
        self.suite = suite
        self.report = report

    def write(self, format: str) -> ET.Element:
        root = ET.Element("reactionSuite")
        _set(root,
             projectile=self.suite.projectile,
             target=self.suite.target,
             evaluation=self.suite.evaluation,
             format=format,
             projectileFrame=str(self.suite.projectileFrame),
             interaction=self.suite.interaction)

        # Emitted in SUITE_ORDER, not in whatever order is convenient here:
        # `ReactionSuiteType` is an xs:sequence, so `sums` after `productions`
        # is a file every reader still reads and no validator accepts.
        lists = {attribute: (wrapper, child)
                 for attribute, wrapper, child in REACTION_LISTS}
        for name in SUITE_ORDER:
            if name == "externalFiles" and len(self.suite.externalFiles):
                self.externalFiles(root)
            elif name == "styles":
                self.styles(root)
            elif name == "PoPs":
                self.pops(root)
            elif name == "resonances" and self.suite.resonances is not None:
                self.resonances(root)
            elif name == "sums":
                self.sums(root)
            elif name in lists:
                container = getattr(self.suite, name)
                if not len(container):
                    continue
                wrapper, child = lists[name]
                element = ET.SubElement(root, wrapper)
                for reaction in container:
                    self.reaction(element, reaction, child)
        self.declareWhatIsMissing()
        return root

    @property
    def domain(self) -> Tuple[str, str]:
        """The evaluation's energy domain, for the ``constant1d`` nodes that
        must declare one.

        ``xData_constant1d`` makes ``domainMin`` and ``domainMax`` **required**,
        and kika's ``Q`` and ``Multiplicity`` hold a number without one. The
        domain written is the ``evaluated`` style's ``projectileEnergyDomain``,
        which is where FUDGE takes it from and what it means — the constant
        holds over the evaluation. It is read off the model, never guessed: a
        suite with no evaluated style falls back to the widest range ENDF admits
        and the report says so once.
        """
        for style in self.suite.styles:
            if isinstance(style, Evaluated) and style.projectileEnergyDomain:
                return (_number(style.projectileEnergyDomain.min),
                        _number(style.projectileEnergyDomain.max))
        if not self._warnedAboutDomain:
            self._warnedAboutDomain = True
            self.report.warn(
                "no evaluated style carries a projectileEnergyDomain, so the "
                "constant1d nodes for Q and multiplicity declare 1e-5 to 20 MeV "
                "— the range ENDF assumes — rather than this evaluation's own"
            )
        return "1e-05", "20000000.0"

    # -- externalFiles / styles / PoPs -------------------------------------

    def externalFiles(self, root: ET.Element) -> None:
        container = ET.SubElement(root, "externalFiles")
        for entry in self.suite.externalFiles:
            _set(ET.SubElement(container, "externalFile"),
                 label=entry.label, path=entry.path,
                 checksum=entry.checksum, algorithm=entry.algorithm)

    def styles(self, root: ET.Element) -> None:
        writeStyles(root, self.suite.styles, _number, documentation=True)

    def pops(self, root: ET.Element) -> None:
        """§12, written back as minimally as it was read — and it says so.

        kika's PoPs holds ids, masses, spins, parities and charges. A file
        written from it has a ``PoPs`` with those and nothing else: no decay
        data, no halflives, no aliases, no level energies. That is a real loss
        against the file it was read from and it is reported, once, with the
        count — because a reader of the *written* file has no way to tell a
        minimal PoPs from a particle database that genuinely says this much.
        """
        container = ET.SubElement(root, "PoPs")
        _set(container, name=self.suite.PoPs.name or "protare_internal",
             version=self.suite.PoPs.version or "1.0", format="2.0")
        nuclides = [p for p in self.suite.PoPs.particles.values()
                    if isinstance(p, Nuclide)]
        others = [p for p in self.suite.PoPs.particles.values()
                  if not isinstance(p, Nuclide)]

        for particle in others:
            wrapper = ("gaugeBosons" if particle.id == "photon" else "baryons")
            group = container.find(wrapper)
            if group is None:
                group = ET.SubElement(container, wrapper)
            node = ET.SubElement(group, wrapper[:-1])
            node.attrib["id"] = particle.id
            self.particleProperties(node, particle)

        if nuclides:
            elements = ET.SubElement(container, "chemicalElements")
            byZ: Dict[int, List[Nuclide]] = {}
            for nuclide in nuclides:
                byZ.setdefault(nuclide.Z, []).append(nuclide)
            for Z in sorted(k for k in byZ if k is not None):
                chemical = ET.SubElement(elements, "chemicalElement")
                _set(chemical, symbol=byZ[Z][0].id.rstrip("0123456789"),
                     Z=str(Z), name=byZ[Z][0].id.rstrip("0123456789"))
                isotopes = ET.SubElement(chemical, "isotopes")
                byA: Dict[int, List[Nuclide]] = {}
                for nuclide in byZ[Z]:
                    byA.setdefault(nuclide.A, []).append(nuclide)
                for A in sorted(k for k in byA if k is not None):
                    isotope = ET.SubElement(isotopes, "isotope")
                    _set(isotope, symbol=byA[A][0].id, A=str(A))
                    holder = ET.SubElement(isotope, "nuclides")
                    for nuclide in byA[A]:
                        node = ET.SubElement(holder, "nuclide")
                        node.attrib["id"] = nuclide.id
                        self.nuclideProperties(node, nuclide)

        if len(self.suite.PoPs):
            self.report.lost(
                f"PoPs was written from kika's minimal §12 model: "
                f"{len(self.suite.PoPs)} particles with their masses, spins, "
                f"parities, charges and halflives and nothing else. Decay data, "
                f"aliases and nuclear level energies are not in the model and so "
                f"are not in this file; a reader of it cannot tell that from a "
                f"database that genuinely says only this much"
            )

    def particleProperties(self, node: ET.Element, particle) -> None:
        if particle.mass is not None:
            _set(ET.SubElement(ET.SubElement(node, "mass"), "double"),
                 label="eval", value=_number(particle.mass.value),
                 unit=particle.mass.unit)
        if particle.spin is not None:
            _set(ET.SubElement(ET.SubElement(node, "spin"), "fraction"),
                 label="eval", value=formatFraction(particle.spin.value),
                 unit=particle.spin.unit)
        if particle.parity is not None:
            _set(ET.SubElement(ET.SubElement(node, "parity"), "integer"),
                 label="eval", value=str(particle.parity))
        if particle.charge is not None:
            _set(ET.SubElement(ET.SubElement(node, "charge"), "integer"),
                 label="eval", value=str(particle.charge), unit="e")
        self.halflife(node, particle.halflife)

    def halflife(self, node: ET.Element, halflife) -> None:
        """§12's ``halflife``, in whichever of its two spellings the model holds.

        Mandatory on a ``baryon`` and a ``gaugeBoson``, so a particle with none
        gets ``<string value="unknown">`` — which is a real §12 value and the
        only honest thing to write: kika does not know, and the alternatives are
        omitting a required element or asserting a number.
        """
        element = ET.SubElement(node, "halflife")
        if halflife is None:
            _set(ET.SubElement(element, "string"), label="eval",
                 value="unknown", unit="s")
        elif isinstance(halflife, str):
            _set(ET.SubElement(element, "string"), label="eval",
                 value=halflife, unit="s")
        else:
            _set(ET.SubElement(element, "double"), label="eval",
                 value=_number(halflife.value), unit=halflife.unit or "s")

    def nuclideProperties(self, node: ET.Element, nuclide: Nuclide) -> None:
        """The atom's mass and charge on the ``nuclide``, the nucleus's spin and
        parity on the ``nucleus`` — which is where each was read from."""
        if nuclide.mass is not None:
            _set(ET.SubElement(ET.SubElement(node, "mass"), "double"),
                 label="eval", value=_number(nuclide.mass.value),
                 unit=nuclide.mass.unit)
        if nuclide.charge is not None:
            _set(ET.SubElement(ET.SubElement(node, "charge"), "integer"),
                 label="eval", value=str(nuclide.charge), unit="e")
        nucleus = ET.SubElement(node, "nucleus")
        _set(nucleus, id=nuclide.id.lower(), index=str(nuclide.nuclearLevel))
        if nuclide.spin is not None:
            _set(ET.SubElement(ET.SubElement(nucleus, "spin"), "fraction"),
                 label="eval", value=formatFraction(nuclide.spin.value),
                 unit=nuclide.spin.unit)
        if nuclide.parity is not None:
            _set(ET.SubElement(ET.SubElement(nucleus, "parity"), "integer"),
                 label="eval", value=str(nuclide.parity))
        if nuclide.Z is not None:
            _set(ET.SubElement(ET.SubElement(nucleus, "charge"), "integer"),
                 label="eval", value=str(nuclide.Z), unit="e")
        if nuclide.halflife is not None:
            self.halflife(nucleus, nuclide.halflife)

    # -- reactions ---------------------------------------------------------

    def reaction(self, parent: ET.Element, reaction, tag: str) -> None:
        element = ET.SubElement(parent, tag)
        _set(element, label=reaction.label,
             ENDF_MT=None if reaction.ENDF_MT is None else str(reaction.ENDF_MT),
             fissionGenre=reaction.id.fissionGenre)
        self.crossSection(element, reaction.crossSection,
                          f"{reaction.label!r}")
        self.outputChannel(element, reaction.outputChannel,
                           f"{reaction.label!r}")

    @writes("crossSectionForm", "resonancesWithBackground", "reference")
    def crossSection(self, parent: ET.Element, crossSection, where: str) -> None:
        element = ET.SubElement(parent, "crossSection")
        for label, form in crossSection.forms.items():
            if isinstance(form, ResonancesWithBackground):
                self.resonancesWithBackground(element, form, label, where)
            elif isinstance(form, Reference):
                _set(ET.SubElement(element, "reference"),
                     label=label, href=form.href)
            else:
                _function(element, form, self.report,
                          f"{where} crossSection[{label!r}]")

    def resonancesWithBackground(self, parent: ET.Element, form, label: str,
                                 where: str) -> None:
        element = ET.SubElement(parent, "resonancesWithBackground")
        element.attrib["label"] = label
        _set(ET.SubElement(element, "resonances"),
             href=form.resonanceRegionHref)
        background = ET.SubElement(element, "background")
        regions = form.background.regions if form.background is not None else {}
        for name, region in regions.items():
            if region is None:
                continue
            _function(ET.SubElement(background, name), region, self.report,
                      f"{where} background/{name}")

    def outputChannel(self, parent: ET.Element, channel, where: str) -> None:
        element = ET.SubElement(parent, "outputChannel")
        _set(element, genre=channel.genre, process=channel.process)
        self.Q(element, channel.Q, where)
        products = ET.SubElement(element, "products")
        for product in channel.products:
            self.product(products, product, where)

    def Q(self, parent: ET.Element, q, where: str) -> None:
        element = ET.SubElement(parent, "Q")
        if not q.isKnown:
            self.report.lost(
                f"{where}: the output channel's Q is not known, so the <Q> node "
                f"is empty; §17.1.1 requires a value in it"
            )
            return
        constant = ET.SubElement(element, "constant1d")
        domainMin, domainMax = self.domain
        _set(constant, label=q.label or "eval", value=_number(q.value),
             domainMin=domainMin, domainMax=domainMax)
        axes = ET.SubElement(constant, "axes")
        _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
        _set(ET.SubElement(axes, "axis"), index="0", label="Q", unit=q.unit or "eV")

    def product(self, parent: ET.Element, product, where: str) -> None:
        element = ET.SubElement(parent, "product")
        _set(element, pid=product.pid, label=product.label or product.pid)
        self.multiplicity(element, product.multiplicity,
                          f"{where} product {product.pid!r}")
        self.distribution(element, product.distribution,
                          f"{where} product {product.pid!r}")
        if product.outputChannel is not None:
            self.outputChannel(element, product.outputChannel,
                               f"{where} product {product.pid!r}")

    @writes("multiplicityForm", "constant1d")
    def multiplicity(self, parent: ET.Element, multiplicity, where: str) -> None:
        element = ET.SubElement(parent, "multiplicity")
        if multiplicity is None:
            self.report.lost(f"{where}: no multiplicity, so <multiplicity> is empty")
            return
        if multiplicity.function is not None:
            _function(element, multiplicity.function, self.report, where)
            return
        if multiplicity.constant is None:
            self.report.lost(
                f"{where}: the multiplicity was read from a node kika does not "
                f"model — a reference, a branching1d or an unspecified — so "
                f"<multiplicity> is empty rather than filled with a 1"
            )
            return
        constant = ET.SubElement(element, "constant1d")
        domainMin, domainMax = self.domain
        _set(constant, label=multiplicity.label or "eval",
             value=_number(multiplicity.constant),
             domainMin=domainMin, domainMax=domainMax)
        axes = ET.SubElement(constant, "axes")
        _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
        _set(ET.SubElement(axes, "axis"), index="0", label="multiplicity", unit="")

    @writes("distributionForm", "angularTwoBody", "unspecified")
    def distribution(self, parent: ET.Element, distribution, where: str) -> None:
        """§18. **An empty ``<distribution/>`` is deliberate and is reported.**

        The alternative — writing ``<unspecified/>`` — would produce a file that
        validates and lies: ``unspecified`` is the *evaluator* saying no
        distribution is given, and kika would be saying it on their behalf about
        a law it simply cannot read yet. The empty element fails validation,
        which is the file telling the next tool the truth.
        """
        element = ET.SubElement(parent, "distribution")
        forms = {} if distribution is None else distribution.forms
        for label, form in forms.items():
            if isinstance(form, Uncorrelated):
                self.uncorrelated(element, form, label, where)
            elif isinstance(form, EnergyAngular):
                self.energyAngular(element, form, label, where)
            elif isinstance(form, AngularTwoBody):
                self.angularTwoBody(element, form, label, where)
            elif isinstance(form, Isotropic2d):
                # A bare Isotropic2d is what `kika/endf/model_adapter/
                # angular.py:146` returns for every MF4 with LTT=0, so this is
                # the majority shape of an ENDF-sourced angular distribution and
                # not an edge case. It used to be written as <isotropic2d>
                # directly under <distribution>, which is wrong three ways:
                # `DistributionType` (gnds.xsd:1647-1662) has no such child,
                # `DistributionIsotropic2dType` (:1693) has no attributes at all
                # so `label` and `productFrame` were invalid on it, and kika's
                # own reader (decode.py's readDistribution) drops the node into
                # the report. An isotropic MF4 lost its distribution between
                # kika's two halves.
                #
                # The fix is here and not in the reader: adding a reader would
                # encode a node the schema does not have. What an LTT=0 MF4 *is*
                # in GNDS is a two-body angular distribution that happens to be
                # isotropic, and `readAngularTwoBody` already reads that back.
                twoBody = ET.SubElement(element, "angularTwoBody")
                _set(twoBody, label=label, productFrame=str(form.productFrame))
                ET.SubElement(twoBody, "isotropic2d")
            elif isinstance(form, Unspecified):
                _set(ET.SubElement(element, "unspecified"), label=label,
                     productFrame=str(form.productFrame))
            else:
                self.report.unsupportedNode(
                    f"{where}: kika's writer has no serialisation for a "
                    f"{type(form).__name__} distribution"
                )
        if len(element) == 0:
            # Counted on the **element**, not on the model dict. A product
            # whose only form is one this writer cannot serialise leaves the
            # element childless too, and counting the dict missed it — so the
            # file came out invalid without the "does not validate" sentence
            # that `declareWhatIsMissing` is there to put in the report.
            self.incompleteProducts.append(where)

    @writes("distributionForm", "energyAngular")
    def energyAngular(self, parent: ET.Element, form, label: str,
                      where: str) -> None:
        """§18.4, and **the one child or none**.

        ``DistributionAEType`` (``gnds.xsd:1797-1803``) is an ``xs:sequence`` of
        one ``XYs3d``, so an ``energyAngular`` whose function kika could not
        read is not a partial node but an invalid one — the same judgement
        :meth:`uncorrelated` makes about its two halves, made in the same place
        and for the same reason.

        ``nested=False`` on the child: it is the *primary* of its container
        (``xData_XYs3d_primary``), so it carries its own ``axes``, which
        ``:2260`` makes a required child rather than an optional one.

        **A known way this writes an invalid file, and it is not a defect
        here.** ``xData_XYs3d_primary`` declares no ``interpolation`` attribute
        where every 2-d type does (``library-gaps.md`` D24). If a real
        ``energyAngular`` states a non-lin-lin law on its outermost axis, the
        round trip writes it back and the file fails validation — which is
        correct: dropping the attribute would validate by discarding what the
        evaluation said.
        """
        if not form.isComplete:
            self.report.unsupportedNode(
                f"{where}: an <energyAngular> whose XYs3d kika could not read; "
                f"gnds.xsd:1798-1800 requires the child, so the node is not "
                f"written at all"
            )
            return
        element = ET.SubElement(parent, "energyAngular")
        _set(element, label=label, productFrame=str(form.productFrame))
        _function(element, form.xys3d, self.report, where)

    @writes("distributionForm", "uncorrelated")
    @writes("uncorrelatedAngularForm", "isotropic2d")
    def uncorrelated(self, parent: ET.Element, form, label: str,
                     where: str) -> None:
        """§18.3, and **both halves or neither**.

        ``DistributionUncorrelatedType`` (``gnds.xsd:1676-1682``) is an
        ``xs:sequence`` of ``<angular>`` and ``<energy>``, so a node with one
        of them is not a partial statement — it is an invalid one. When kika
        read only one half, the honest output is the same empty
        ``<distribution/>`` an unread law gets: it fails validation and says so,
        instead of shipping a node whose missing child a reader would have to
        guess at.
        """
        if not form.isComplete:
            missing = "angular" if form.angular is None else "energy"
            self.report.unsupportedNode(
                f"{where}: an <uncorrelated> whose <{missing}> form kika could "
                f"not read; gnds.xsd:1677-1680 requires both children, so the "
                f"half that was read is not written either"
            )
            return
        element = ET.SubElement(parent, "uncorrelated")
        _set(element, label=label, productFrame=str(form.productFrame))
        angular = ET.SubElement(element, "angular")
        if isinstance(form.angular, Isotropic2d):
            ET.SubElement(angular, "isotropic2d")
        else:
            _function(angular, form.angular, self.report, where)
        self._uncorrelatedEnergy(element, form.energy, where)

    @writes("uncorrelatedEnergyForm", "discreteGamma", "primaryGamma",
            "NBodyPhaseSpace")
    def _uncorrelatedEnergy(self, parent: ET.Element, form,
                            where: str) -> None:
        """``uncorrelated/energy``: the four non-functional forms, by hand.

        None of the three gamma/phase-space nodes is a functional — no
        ``function1ds``, no interpolation — so they do not go through
        :func:`_function`, and their ``<axes>`` is a *required* child rather
        than the inheritable one §5.1.1 lets a nested functional drop.
        """
        element = ET.SubElement(parent, "energy")
        if isinstance(form, PrimaryGamma):
            gamma = ET.SubElement(element, "primaryGamma")
            _set(gamma, value=_number(form.value),
                 domainMin=_number(form.domainMin),
                 domainMax=_number(form.domainMax),
                 finalState=form.finalState)
            self._gammaAxes(gamma, form.axes, where, "primaryGamma")
            return
        if isinstance(form, DiscreteGamma):
            gamma = ET.SubElement(element, "discreteGamma")
            _set(gamma, value=_number(form.value),
                 domainMin=_number(form.domainMin),
                 domainMax=_number(form.domainMax))
            self._gammaAxes(gamma, form.axes, where, "discreteGamma")
            return
        if isinstance(form, NBodyPhaseSpace):
            phaseSpace = ET.SubElement(element, "NBodyPhaseSpace")
            _set(phaseSpace, numberOfProducts=str(form.numberOfProducts))
            if form.mass is not None:
                _set(ET.SubElement(phaseSpace, "mass"),
                     value=_number(form.mass.value), unit=form.mass.unit)
            return
        _function(element, form, self.report, where)

    def _gammaAxes(self, parent: ET.Element, axes, where: str,
                   tag: str) -> None:
        """``<axes>`` is a *required* child of both gamma nodes.

        Unlike a nested functional's, which §5.1.1 lets it inherit and
        the schema then forbids it from repeating, this one has nothing
        to inherit from: the gamma sits inside ``<energy>``, which is a
        wrapper and not a container. Absent, it is a validation error,
        so it is reported rather than quietly skipped.
        """
        if axes is None:
            self.report.lost(
                f"{where}: a <{tag}> with no axes; gnds.xsd:1777/1786 "
                f"require the child, so the node written is invalid"
            )
            return
        _axes(parent, axes)

    @writes("angularTwoBodyForm", "isotropic2d", "recoil")
    def angularTwoBody(self, parent: ET.Element, form, label: str,
                       where: str) -> None:
        element = ET.SubElement(parent, "angularTwoBody")
        _set(element, label=label, productFrame=str(form.productFrame))
        if form.isRecoil:
            _set(ET.SubElement(element, "recoil"), href=form.recoilHref)
            return
        if isinstance(form.angular, Isotropic2d):
            ET.SubElement(element, "isotropic2d")
            return
        if form.angular is not None:
            _function(element, form.angular, self.report, where)

    # -- sums --------------------------------------------------------------

    def sums(self, root: ET.Element) -> None:
        crossSectionSums = [s for s in self.suite.sums]
        multiplicitySums = list(self.suite.sums.multiplicitySums)
        if not crossSectionSums and not multiplicitySums:
            return
        container = ET.SubElement(root, "sums")
        if crossSectionSums:
            element = ET.SubElement(container, "crossSectionSums")
            for entry in crossSectionSums:
                self.crossSectionSum(element, entry)
        if multiplicitySums:
            element = ET.SubElement(container, "multiplicitySums")
            for entry in multiplicitySums:
                self.multiplicitySum(element, entry)

    def crossSectionSum(self, parent: ET.Element, entry) -> None:
        element = ET.SubElement(parent, "crossSectionSum")
        _set(element, label=entry.label,
             ENDF_MT=None if entry.ENDF_MT is None else str(entry.ENDF_MT))
        summands = ET.SubElement(element, "summands")
        if isinstance(entry, CrossSectionSum):
            for add in entry.summands:
                _set(ET.SubElement(summands, "add"), href=add.href)
        else:
            # A `Sums` filled by the ENDF adapter holds plain `Reaction`s: ENDF
            # states MT1 and MT4 as ordinary sections and says nowhere what they
            # sum. An empty `<summands/>` is what that is, and it is reported
            # rather than filled with a guess at the partials.
            self.report.lost(
                f"crossSectionSum {entry.label!r} has no summands: it came from "
                f"a format that states the total without saying what it is the "
                f"total of, and kika does not infer the list"
            )
        self.Q(element, entry.outputChannel.Q, f"sum {entry.label!r}")
        self.crossSection(element, entry.crossSection, f"sum {entry.label!r}")

    def multiplicitySum(self, parent: ET.Element, entry) -> None:
        element = ET.SubElement(parent, "multiplicitySum")
        _set(element, label=entry.label,
             ENDF_MT=None if entry.ENDF_MT is None else str(entry.ENDF_MT))
        summands = ET.SubElement(element, "summands")
        for add in entry.summands:
            _set(ET.SubElement(summands, "add"), href=add.href)
        self.multiplicity(element, entry.multiplicity,
                          f"multiplicitySum {entry.label!r}")

    # -- resonances --------------------------------------------------------

    def resonances(self, root: ET.Element) -> None:
        from .encode_resonances import writeResonances

        writeResonances(root, self.suite.resonances, self.report, self.domain)

    # -- the declaration ---------------------------------------------------

    incompleteProducts: List[str]
    _warnedAboutDomain: bool

    def declareWhatIsMissing(self) -> None:
        if not self.incompleteProducts:
            return
        self.report.unsupportedNode(
            f"{len(self.incompleteProducts)} products were written with an "
            f"empty <distribution/> because kika does not read their §18 law. "
            f"**This file does not validate against the GNDS schema, and that is "
            f"deliberate**: writing an <unspecified/> in their place would make "
            f"it valid by asserting, on the evaluator's behalf, that no "
            f"distribution was given. The first few are "
            f"{self.incompleteProducts[:3]}"
        )


def writeReactionSuite(suite, format: Optional[str] = None,
                       report: Optional[ConversionReport] = None
                       ) -> Tuple[ET.ElementTree, ConversionReport]:
    """§14.1.1 ``reactionSuite`` → an ``ElementTree``, plus what was not written.

    Read the report. It is the only thing that distinguishes a file kika wrote
    completely from one it wrote with holes, and the holes are invisible in the
    XML — an empty ``<distribution/>`` looks like a formatting accident.
    """
    report = report if report is not None else ConversionReport()
    writer = _SuiteWriter(suite, report)
    writer.incompleteProducts = []
    writer._warnedAboutDomain = False
    root = writer.write(chooseFormat(suite, format))
    return ET.ElementTree(root), report


# ---------------------------------------------------------------------------
# covarianceSuite
# ---------------------------------------------------------------------------

def writeCovarianceSuite(covarianceSuite, format: str,
                         report: Optional[ConversionReport] = None
                         ) -> Tuple[ET.ElementTree, ConversionReport]:
    """§25.1.1 ``covarianceSuite`` → its own document.

    Matrices are written **uncompressed and lower-triangular** where they are
    symmetric, which is the storage FUDGE uses for the majority of them and the
    one whose semantics need no reconstruction. ``compression="flattened"`` and
    ``"diagonal"`` are read (33 and 1 875 arrays in the library) and not written:
    a writer that chose a compression would have to decide when a matrix is
    "sparse enough", and getting that wrong costs file size and nothing else,
    while getting the *encoding* wrong costs correctness.

    §25.3's ``parameterCovariances`` are written too, and the container is a
    **sequence**: all the ``parameterCovariance`` nodes, then all the
    ``averageParameterCovariance`` ones, whatever order the model list had.
    """
    report = report if report is not None else ConversionReport()
    root = ET.Element("covarianceSuite")
    _set(root,
         projectile=covarianceSuite.projectile,
         target=covarianceSuite.target,
         evaluation=covarianceSuite.evaluation,
         interaction=covarianceSuite.interaction or "nuclear",
         format=format)

    if covarianceSuite.externalFiles:
        container = ET.SubElement(root, "externalFiles")
        for entry in covarianceSuite.externalFiles:
            _set(ET.SubElement(container, "externalFile"),
                 label=entry.label, path=entry.path,
                 checksum=entry.checksum, algorithm=entry.algorithm)

    # §25.1.1's sequence: externalFiles, then styles, then the sections. The
    # styles element is mandatory, and a suite decoded from ENDF has none — so
    # one is synthesised from the evaluation name rather than the element being
    # omitted, and the report says which happened.
    if covarianceSuite.styles is not None and len(covarianceSuite.styles):
        writeStyles(root, covarianceSuite.styles, _number,
                    documentation=False)
    else:
        from kika.nuclear_data.model import (Evaluated, PhysicalQuantity,
                                             RangeQuantity, Styles)

        synthetic = Styles()
        synthetic.add(Evaluated(
            label="eval", library=covarianceSuite.evaluation or "unknown",
            version="", date="1970-01-01",
            temperature=PhysicalQuantity(value=0.0, unit="K"),
            projectileEnergyDomain=RangeQuantity(min=1e-5, max=2e7, unit="eV"),
        ))
        writeStyles(root, synthetic, _number, documentation=False)
        report.approximated(
            "the covarianceSuite carried no styles and §25.1.1 requires one, so "
            "an `evaluated` style was synthesised: label 'eval', temperature "
            "0 K, domain 1e-5 to 20 MeV and a placeholder date. **Nothing in "
            "the source said any of that** — it is the minimum the schema "
            "accepts, and a reader of the written file cannot tell it from a "
            "style the evaluator wrote"
        )

    if covarianceSuite.covarianceSections:
        container = ET.SubElement(root, "covarianceSections")
        for section in covarianceSuite.covarianceSections:
            _covarianceSection(container, section, report)
    if covarianceSuite.parameterCovariances:
        _parameterCovariances(root, covarianceSuite.parameterCovariances,
                              report)
    return ET.ElementTree(root), report


def _covarianceSection(parent: ET.Element, section, report) -> None:
    element = ET.SubElement(parent, "covarianceSection")
    _set(element, label=section.label, crossTerm=_boolean(section.crossTerm))
    _dataLink(element, "rowData", section.rowData)
    _dataLink(element, "columnData", section.columnData)
    _covarianceForm(element, section.form, report, f"section {section.label!r}")


def _mfmt(link) -> Optional[str]:
    """``ENDF_MFMT`` in **the spelling §25.2.3 defines**, whatever the model holds.

    p. 363 defines it as a comma-separated pair, and all 270 covariance files in
    ENDF/B-VIII.1-GNDS agree — ``"33,2"``. kika's *ENDF* adapter writes
    ``"33/2"``, and nine parse sites across ``kika/cov`` and ``kika/sampling``
    split on that slash, two of them in the deployed thesis pipeline. So the
    divergence is not fixed at the source, where it would move a number the
    thesis depends on; it is fixed **here**, at the one place that hands the
    string to somebody else's tool. A GNDS file kika writes says ``33,2``
    whichever reader filled the model.

    ``DataLink.ENDF_MF``/``ENDF_MT`` already read both spellings, which is why
    this is a re-render and not a string substitution: a value that parses as
    neither is passed through untouched rather than mangled.
    """
    if link.ENDF_MF is not None and link.ENDF_MT is not None:
        return f"{link.ENDF_MF},{link.ENDF_MT}"
    return link.ENDF_MFMT


def _dataLink(parent: ET.Element, tag: str, link) -> None:
    if link is None:
        return
    element = ET.SubElement(parent, tag)
    _set(element, href=link.href, ENDF_MFMT=_mfmt(link),
         dimension=None if link.dimension is None else str(link.dimension))
    if len(link.slices):
        container = ET.SubElement(element, "slices")
        for entry in link.slices:
            _set(ET.SubElement(container, "slice"),
                 dimension=str(entry.dimension),
                 domainValue=(None if entry.domainValue is None
                              else _number(entry.domainValue)),
                 domainMin=(None if entry.domainMin is None
                            else _number(entry.domainMin)),
                 domainMax=(None if entry.domainMax is None
                            else _number(entry.domainMax)),
                 domainUnit=entry.domainUnit or None)


#: The style label ``writeCovarianceSuite`` synthesises for a suite that has
#: none. It is also what :func:`_formLabel` falls back to, and the two are the
#: same constant on purpose rather than by coincidence: the suites that arrive
#: without styles are exactly the suites whose forms arrive without labels —
#: both are things ENDF has no concept of — so the label the fallback invents is
#: always the label of a style the same call has just written.
SYNTHESISED_STYLE_LABEL = "eval"


def _formLabel(form, report, where: str) -> str:
    """``label`` for a §25 form, and **all four of them require one.**

    ``covariances.xsd`` marks it ``use="required"`` on ``covarianceMatrix``
    (:118), ``shortRangeSelfScalingVariance`` (:126), ``mixed`` (:135) and
    ``sum`` (:143), and the ENDF adapter sets it on none of them —
    ``covariances.py:113`` builds every gridded matrix without one, because in
    ENDF a covariance belongs to a file and a section, not to a *style*. So a
    covariance sibling written from a tape was invalid once per matrix, which is
    D25 and is what ``test_the_endf_decoded_covariance_sibling_is_pinned``
    measures.

    The value is not invented out of nothing: in GNDS a form's label names the
    style it belongs to, and the only suites reaching this branch are the ones
    ``writeCovarianceSuite`` has just given a synthesised ``evaluated`` style
    called :data:`SYNTHESISED_STYLE_LABEL`. It is still reported — the source
    said nothing about which style any of this belongs to, and a reader of the
    written file cannot tell the synthesis from an evaluator's choice.
    """
    label = getattr(form, "label", None)
    if label is not None:
        return label
    report.approximated(
        f"{where}: the form carried no label and §25 requires one on all four "
        f"covariance forms, so it is written as "
        f"{SYNTHESISED_STYLE_LABEL!r} -- the style label the suite writer "
        f"synthesises for a suite that has no styles, which is the same case. "
        f"The source said nothing about which style this form belongs to"
    )
    return SYNTHESISED_STYLE_LABEL


@writes("covarianceForm", "covarianceMatrix", "mixed", "sum",
        "shortRangeSelfScalingVariance")
def _covarianceForm(parent: ET.Element, form, report, where: str) -> None:
    if form is None:
        report.lost(f"{where} has no covariance form and is written empty")
        return
    if isinstance(form, CovarianceMatrix):
        _covarianceMatrix(parent, "covarianceMatrix", form, report, where)
    elif isinstance(form, ShortRangeSelfScalingVariance):
        # The one form that is not itself a matrix: the model nests the gridded
        # data as `.matrix`, deliberately, so that nothing can add a short-range
        # term to its siblings by taking it for a CovarianceMatrix. So the
        # writer has to unwrap it again — passing the outer object here asked it
        # for an `isRelative` it has never had.
        if form.matrix is None:
            report.lost(
                f"{where} holds a shortRangeSelfScalingVariance with no matrix, "
                f"and it is written as nothing at all"
            )
            return
        element = _covarianceMatrix(parent, "shortRangeSelfScalingVariance",
                                    form.matrix, report, where)
        if element is not None:
            # Its own label overwrites the wrapped matrix's, which is the right
            # way round: `covariances.xsd:126` requires the attribute on *this*
            # element, and the matrix inside it is not a node of its own here.
            _set(element, label=_formLabel(form, report, where),
                 dependenceOnProcessedGroupWidth=
                 form.dependenceOnProcessedGroupWidth)
    elif isinstance(form, Mixed):
        element = ET.SubElement(parent, "mixed")
        _set(element, label=_formLabel(form, report, where))
        for component in form.components:
            _covarianceForm(element, component, report, where)
    elif isinstance(form, Sum):
        element = ET.SubElement(parent, "sum")
        _set(element, label=_formLabel(form, report, where),
             domainMin=None if form.domainMin is None else _number(form.domainMin),
             domainMax=None if form.domainMax is None else _number(form.domainMax),
             domainUnit=form.domainUnit or None)
        for summand in form.summands:
            _set(ET.SubElement(element, "summand"), href=summand.href,
                 ENDF_MFMT=summand.ENDF_MFMT,
                 coefficient=_number(summand.coefficient))
    else:
        report.unsupportedNode(
            f"{where}: kika's writer has no serialisation for a "
            f"{type(form).__name__} covariance form"
        )


def _covarianceMatrix(parent: ET.Element, tag: str, form, report,
                      where: str) -> Optional[ET.Element]:
    element = ET.SubElement(parent, tag)
    _set(element, label=_formLabel(form, report, where),
         type="relative" if form.isRelative else "absolute")
    gridded = ET.SubElement(element, "gridded2d")
    axes = ET.SubElement(gridded, "axes")
    # §25.2.2's axis order: 2 is the row grid, 1 the column grid, 0 the matrix.
    _set(ET.SubElement(axes, "axis"), index="0", label="matrix_elements", unit="")
    for index, values, label in ((1, form.columnGrid, "column_energy_bounds"),
                                 (2, form.rowGrid, "row_energy_bounds")):
        if values is None:
            _set(ET.SubElement(axes, "axis"), index=str(index), label=label,
                 unit="eV")
            continue
        grid = ET.SubElement(axes, "grid")
        _set(grid, index=str(index), label=label, unit="eV", style="boundaries")
        _values(grid, values)

    _array(gridded, form.matrix)
    return element


def _array(parent: ET.Element, matrix) -> ET.Element:
    """§25's ``array``, on its own — uncompressed, lower-triangular if symmetric.

    Its own function because §25.3.2's ``parameterCovarianceMatrix`` holds an
    ``array`` **directly** (``covariances.xsd:170-186``), where §25.2.2's
    ``covarianceMatrix`` wraps one in a ``gridded2d``. Writing the parameter
    matrix by calling :func:`_covarianceMatrix` and ignoring the grids would
    emit that ``gridded2d`` and its ``axes`` as well, and the schema admits
    neither there — the same shape of defect as D19, a node written into a file
    that no reader of the standard takes back. A parameter covariance has no
    grid to put in an ``axes`` either: row 47 is *a neutron width*, and what
    says so is the ``parameterLink`` list, not an energy boundary.
    """
    matrix = np.asarray(matrix)
    array = ET.SubElement(parent, "array")
    array.attrib["shape"] = f"{matrix.shape[0]},{matrix.shape[1]}"
    symmetric = (matrix.shape[0] == matrix.shape[1]
                 and np.array_equal(matrix, matrix.T))
    if symmetric:
        array.attrib["symmetry"] = "lower"
        _values(array, np.concatenate(
            [matrix[i, :i + 1] for i in range(matrix.shape[0])]
        ))
    else:
        _values(array, matrix)
    return array


# ---------------------------------------------------------------------------
# §25.3 parameter covariances
# ---------------------------------------------------------------------------

@writes("parameterCovarianceForm", "parameterCovariance",
        "averageParameterCovariance")
def _parameterCovariances(parent: ET.Element, covariances, report) -> None:
    """§25.3's container — and it is an ``xs:sequence``, not a bag.

    ``covariances.xsd:27-34`` gives ``<parameterCovariances>`` a sequence of two
    unbounded refs: **every** ``parameterCovariance`` first, then every
    ``averageParameterCovariance``. The model keeps both kinds in one list in
    the order the file had them, which for Tm-171 happens to be that order
    already — so writing them as they come would validate on the one fixture
    that has both and fail on a file that interleaved them. The two passes below
    are what makes that not depend on the input.

    The container is dropped again when nothing could be written into it. An
    empty ``<parameterCovariances/>`` is schema-valid, and that is the problem:
    it says the evaluation has no parameter covariances, which is a different
    statement from "kika could not write the ones it read".
    """
    container = ET.SubElement(parent, "parameterCovariances")
    for covariance in covariances:
        if isinstance(covariance, ParameterCovariance):
            _parameterCovariance(container, covariance, report)
    for covariance in covariances:
        if isinstance(covariance, AverageParameterCovariance):
            _averageParameterCovariance(container, covariance, report)
    for covariance in covariances:
        if not isinstance(covariance, (ParameterCovariance,
                                       AverageParameterCovariance)):
            report.unsupportedNode(
                f"parameterCovariances holds a "
                f"{type(covariance).__name__}, which is neither of §25.3's two "
                f"nodes and is not written"
            )
    if len(container) == 0:
        parent.remove(container)


def _parameterCovariance(parent: ET.Element, covariance, report) -> None:
    """§25.3.1. ``rowData`` then the matrix — and **no ``columnData``**.

    The model carries a ``columnData`` and an ``isCrossTerm`` built off it, the
    reader fills it if a file has one, and ``covariances.xsd:160-168`` does not
    admit it: a ``parameterCovariance`` is ``rowData`` plus
    ``parameterCovarianceMatrix``, full stop. So a cross-term parameter
    covariance is a thing the model can hold and this format cannot say, and the
    honest write is to report the link rather than emit an attribute that makes
    the file invalid. ``averageParameterCovariance``, three functions down, is
    the one of the two that *does* take a ``columnData``.
    """
    where = f"parameterCovariance {covariance.label!r}"
    form = covariance.form
    if form is None:
        report.lost(
            f"{where} holds no parameterCovarianceMatrix, which §25.3.1 makes "
            f"mandatory, so the whole covariance is left out of the file"
        )
        return
    if not getattr(form, "parameters", None):
        # `_readParameterCovarianceMatrix` drops the links, deliberately, when
        # they do not account for every row -- and `<parameters>` needs at least
        # one `parameterLink` to be a valid element. A matrix written with its
        # rows unnamed would be a block of numbers with no statement anywhere of
        # what row 47 is, which is the very thing the reader refused to assert.
        report.lost(
            f"{where} has a matrix and no parameterLinks, so nothing in the "
            f"file could say what its rows are; §25.3.2 requires at least one "
            f"link and the covariance is left out rather than written unnamed"
        )
        return

    element = ET.SubElement(parent, "parameterCovariance")
    _set(element, label=covariance.label)
    if covariance.rowData is None:
        report.lost(
            f"{where} has no rowData and §25.3.1 requires one; the covariance "
            f"is written without it and **the file does not validate**"
        )
    _dataLink(element, "rowData", covariance.rowData)
    if covariance.columnData is not None:
        report.lost(
            f"{where} is a cross term -- its columnData points at "
            f"{covariance.columnData.href!r} -- and §25.3.1 has nowhere to put "
            f"that. The matrix is written; the file no longer says the two "
            f"axes are about different parameters"
        )
    _parameterCovarianceMatrix(element, form, report, where)


def _parameterCovarianceMatrix(parent: ET.Element, form, report,
                               where: str) -> ET.Element:
    """§25.3.2. The links, then the bare array.

    **``matrixStartIndex`` is written exactly as the model holds it.**
    ``covariances/covariances.py:_readParameterCovarianceMatrix`` documents why
    it is already zero-based — Si-32's ``scatteringRadius`` at 0 and its 18
    ``resonanceParameters`` at 1 fill a 19x19 matrix only if row 1 is the
    second row — and the reader records the number unchanged. Adding one here
    to "convert back to one-based" would move every link by a row and no test
    of counts would notice, because the counts would still sum to the order.
    """
    element = ET.SubElement(parent, "parameterCovarianceMatrix")
    # §25.3.2 requires `label` exactly as §25.2.2 does, and
    # `kika/endf/model_adapter/parameter_covariances.py` sets it exactly as
    # rarely -- never. Same hole, same fallback, one function.
    _set(element, label=_formLabel(form, report, where),
         type="relative" if form.isRelative else "absolute")

    container = ET.SubElement(element, "parameters")
    named = 0
    for link in form.parameters:
        _set(ET.SubElement(container, "parameterLink"),
             label=link.label, href=link.href,
             nParameters=str(link.nParameters),
             matrixStartIndex=str(link.matrixStartIndex))
        named += len(link.parameterNames)
    if named:
        # `ParameterLink.parameterNames` is what makes `rowLabels()` able to say
        # "the neutron width of resonance 12" rather than "row 47", and it comes
        # from ENDF's MF32 slot order. §25.3.2's `parameterLink` has label, href,
        # nParameters and matrixStartIndex and nothing else, so the names go
        # nowhere: a GNDS reader recovers them by following the href into the
        # reactionSuite's resonance table, which is where GNDS keeps them.
        report.lost(
            f"{where}: {named} parameter names are dropped -- §25.3.2's "
            f"parameterLink has no attribute for them, and a reader of the "
            f"written file gets them by following the href into the "
            f"reactionSuite instead"
        )

    _array(element, form.matrix)
    return element


def _averageParameterCovariance(parent: ET.Element, covariance, report) -> None:
    """§25.3. A URR average parameter — an ordinary gridded matrix about it.

    The one of §25.3's two nodes whose form is a §25.2.2 ``covarianceMatrix``,
    ``gridded2d`` and all: its rows are energy bins of one unresolved-region
    average, not individual resonance parameters. So this reuses
    :func:`_covarianceMatrix` where :func:`_parameterCovariance` must not.

    Only the GNDS reader ever builds one — ``parameter_covariances.py:449-454``
    turns ENDF's LRU=2 into a relative :class:`ParameterCovariance` instead — so
    Tm-171's ten are the only witnesses there are.
    """
    where = f"averageParameterCovariance {covariance.label!r}"
    form = covariance.form
    if form is None:
        report.lost(
            f"{where} holds no covarianceMatrix, which §25.3 makes mandatory, "
            f"so the whole covariance is left out of the file"
        )
        return
    if not isinstance(form, CovarianceMatrix):
        report.unsupportedNode(
            f"{where} holds a {type(form).__name__}; §25.3 admits only a "
            f"covarianceMatrix there, and the covariance is not written"
        )
        return

    element = ET.SubElement(parent, "averageParameterCovariance")
    _set(element, label=covariance.label,
         crossTerm=_boolean(covariance.crossTerm))
    if covariance.rowData is None:
        report.lost(
            f"{where} has no rowData and §25.3 requires one; the covariance is "
            f"written without it and **the file does not validate**"
        )
    _dataLink(element, "rowData", covariance.rowData)
    _dataLink(element, "columnData", covariance.columnData)
    _covarianceMatrix(element, "covarianceMatrix", form, report, where)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def serialise(tree: ET.ElementTree) -> bytes:
    """One tree → the bytes of a file, indented the way the library indents."""
    ET.indent(tree, space="  ")
    from io import BytesIO

    buffer = BytesIO()
    tree.write(buffer, encoding="UTF-8", xml_declaration=True)
    return buffer.getvalue()


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()
