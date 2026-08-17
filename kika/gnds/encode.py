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

from kika.nuclear_data.model import (AngularTwoBody, Background,
                                     BreitWigner, ConversionReport,
                                     Constant1d, CovarianceMatrix,
                                     CrossSectionSum, Evaluated, Isotropic2d,
                                     Legendre, Mixed, Nuclide, Polynomial1d,
                                     GndsProvenance, Reference,
                                     Regions1d, Regions2d,
                                     ResonancesWithBackground, RMatrix,
                                     ShortRangeSelfScalingVariance, Sum,
                                     Unspecified, XYs1d, XYs2d)

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
def _function(parent: ET.Element, form, report: ConversionReport,
              where: str, nested: bool = False,
              parentAxes=None) -> Optional[ET.Element]:
    """One functional → its node, or ``None`` plus a report entry.

    ``nested`` is what makes the difference between the schema's
    ``xData_XYs1d_primary`` — the top of a container, which **must** carry
    ``axes`` — and ``xData_XYs1d``, a child inside ``function1ds``, which must
    **not**: §5.1.1 has a child inherit its parent's axes and the schema
    forbids it from repeating them. The reader shares one ``Axes`` object
    between a container and its children, so writing them again would produce a
    file that is invalid and that says nothing new.
    """
    if isinstance(form, XYs1d):
        element = ET.SubElement(parent, "XYs1d")
        _writeCommon(element, form)
        _interpolation(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.interleaved())
        return element
    if isinstance(form, Regions1d):
        element = ET.SubElement(parent, "regions1d")
        _writeCommon(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function1ds")
        for child in form.function1ds:
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes)
        return element
    if isinstance(form, Constant1d):
        element = ET.SubElement(parent, "constant1d")
        _writeCommon(element, form)
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
        _writeCommon(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.coefficients)
        return element
    if isinstance(form, Polynomial1d):
        element = ET.SubElement(parent, "polynomial1d")
        _writeCommon(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        _values(element, form.coefficients)
        return element
    if isinstance(form, XYs2d):
        element = ET.SubElement(parent, "XYs2d")
        _writeCommon(element, form)
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
    if isinstance(form, Regions2d):
        element = ET.SubElement(parent, "regions2d")
        _writeCommon(element, form)
        _axesUnlessNested(element, form, report, where, nested, parentAxes)
        container = ET.SubElement(element, "function2ds")
        for child in form.function2ds:
            _function(container, child, report, where, nested=True,
                      parentAxes=form.axes)
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


def _writeCommon(element: ET.Element, form) -> None:
    """``label``, ``index`` and ``outerDomainValue`` — the three every form may
    carry, and each of which changes what the file means if it is dropped."""
    outer = getattr(form, "outerDomainValue", None)
    index = getattr(form, "index", None)
    _set(element,
         label=getattr(form, "label", None),
         index=None if index is None else str(index),
         outerDomainValue=None if outer is None else _number(outer))


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

    @writes("distributionForm", "angularTwoBody", "unspecified",
            "isotropic2d")
    def distribution(self, parent: ET.Element, distribution, where: str) -> None:
        """§18. **An empty ``<distribution/>`` is deliberate and is reported.**

        The alternative — writing ``<unspecified/>`` — would produce a file that
        validates and lies: ``unspecified`` is the *evaluator* saying no
        distribution is given, and kika would be saying it on their behalf about
        a law it simply cannot read yet. The empty element fails validation,
        which is the file telling the next tool the truth.
        """
        element = ET.SubElement(parent, "distribution")
        if distribution is None or len(distribution) == 0:
            self.incompleteProducts.append(where)
            return
        for label, form in distribution.forms.items():
            if isinstance(form, AngularTwoBody):
                self.angularTwoBody(element, form, label, where)
            elif isinstance(form, Isotropic2d):
                _set(ET.SubElement(element, "isotropic2d"), label=label,
                     productFrame=str(form.productFrame))
            elif isinstance(form, Unspecified):
                _set(ET.SubElement(element, "unspecified"), label=label,
                     productFrame=str(form.productFrame))
            else:
                self.report.unsupportedNode(
                    f"{where}: kika's writer has no serialisation for a "
                    f"{type(form).__name__} distribution"
                )

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
        report.unsupportedNode(
            f"{len(covarianceSuite.parameterCovariances)} parameterCovariances "
            f"were read and are not written: §25.3's resonance-parameter "
            f"covariances have no writer yet, and an empty "
            f"<parameterCovariances/> would assert the file has none"
        )
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
            _set(element, label=form.label,
                 dependenceOnProcessedGroupWidth=
                 form.dependenceOnProcessedGroupWidth)
    elif isinstance(form, Mixed):
        element = ET.SubElement(parent, "mixed")
        _set(element, label=form.label)
        for component in form.components:
            _covarianceForm(element, component, report, where)
    elif isinstance(form, Sum):
        element = ET.SubElement(parent, "sum")
        _set(element, label=form.label,
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
    _set(element, label=form.label,
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

    matrix = np.asarray(form.matrix)
    array = ET.SubElement(gridded, "array")
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
    return element


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
