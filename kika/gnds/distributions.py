"""GNDS-2.1 §18 ``distribution`` → the model.

Split out of :mod:`kika.gnds.decode` for the reason §19 and §25 were split
before it: ``_SuiteReader`` was one class spanning five specification chapters,
and §18 is the one about to grow. The reader for a chapter that has its own
choice points, its own sub-forms and its own report vocabulary reads better
beside them than interleaved with §14's reaction lists.

**What a `<distribution>` is.** §18.1.1 gives a product's distribution as a
choice of thirteen forms, keyed by style label — the same shape the cross
section has, and the reason :class:`~kika.nuclear_data.model.distributions.Distribution`
is a dict rather than a single object. kika reads the forms it has a model node
for and reports the rest by xPath; a product that loses a form keeps its
multiplicity and its channel.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable, Optional

from kika.nuclear_data.model import (AngularEnergy, AngularTwoBody,
                                     Branching3d,
                                     ConversionReport, DiscreteGamma,
                                     EnergyAngular, Frame, Isotropic2d,
                                     KalbachMann, NBodyPhaseSpace,
                                     PhysicalQuantity, PrimaryGamma,
                                     Uncorrelated, Unspecified)
from kika.nuclear_data.model.distributions import Distribution

from .nodes import NODES, Status, reads
from .primitives import (UnsupportedNode, readAxes, readFunction2d,
                         readFunction3d)

__all__ = ["readDistribution"]

#: Children of a functional or container that carry no data kika reads. The
#: same tuple :mod:`kika.gnds.decode` uses, kept here so this module does not
#: import back into the one that imports it.
IGNORED = ("documentation", "axes")


def _declaredAndUnread(family: str) -> dict:
    """``{tag: reason}`` for the members of one choice kika does not model.

    Derived from :data:`kika.gnds.nodes.NODES` rather than listed again here,
    so the report a reader emits and the table a test checks cannot drift
    apart. Without it, an unmodelled §18.3 spectrum falls through to
    ``readFunction2d`` and is reported as "not a two-dimensional node" — true,
    and not what is wrong with it: an ``<evaporation>`` is a declared choice at
    that point, and saying so is the difference between a gap and a surprise.
    """
    return {tag: spec.reason for (thisFamily, tag), spec in NODES.items()
            if thisFamily == family and spec.status is Status.NEITHER}


#: Built at import, because ``NODES`` is a static table and not filled in by
#: the decorators — those register *against* it.
UNREAD_ANGULAR_FORMS = _declaredAndUnread("uncorrelatedAngularForm")
UNREAD_ENERGY_FORMS = _declaredAndUnread("uncorrelatedEnergyForm")
#: §18.1.1's own choice, same derivation. It used to be one hard-coded sentence
#: saying "phase 7b" about every law, which is false for ``angularEnergy``:
#: nothing is queued for it, its two occurrences in the library are why it is a
#: decision instead. The registry already carries each reason; this reads them.
UNREAD_DISTRIBUTION_FORMS = _declaredAndUnread("distributionForm")


def _quoted(label: str) -> str:
    """One xPath step, with the label predicate GNDS uses to key siblings."""
    return f"[@label='{label}']" if label else ""


class _DistributionReader:
    """The §18 half of one ``reactionSuite`` read.

    Carries the two things every form reader needs — the href resolver, for the
    ``axes`` a functional may point at instead of spelling, and the report,
    which is where every node kika cannot model goes.
    """

    def __init__(self, resolve: Optional[Callable],
                 report: ConversionReport) -> None:
        self.resolve = resolve
        self.report = report

    def unsupported(self, tag: str, path: str, reason: str) -> None:
        self.report.unsupportedNode(f"{path}/{tag}: {reason}")

    @reads("distributionForm", "angularTwoBody", "unspecified", "branching3d")
    def read(self, element: ET.Element, path: str) -> Distribution:
        here = f"{path}/distribution"
        distribution = Distribution()
        for child in element:
            if child.tag in IGNORED:
                continue
            label = child.attrib.get("label", "")
            if child.tag == "angularTwoBody":
                form = self.angularTwoBody(child, here)
            elif child.tag == "uncorrelated":
                form = self.uncorrelated(child, here)
            elif child.tag == "energyAngular":
                form = self.energyAngular(child, here)
            elif child.tag == "angularEnergy":
                form = self.angularEnergy(child, here)
            elif child.tag == "KalbachMann":
                form = self.kalbachMann(child, here)
            elif child.tag == "unspecified":
                form = Unspecified(
                    label=label,
                    productFrame=Frame(child.attrib.get("productFrame", "lab")),
                )
            elif child.tag == "branching3d":
                # Two attributes and no content (gnds.xsd:1816-1819). Read so
                # the node survives a round trip; **not** resolved against the
                # nuclide's PoPs decayData, which is what evaluating an isomeric
                # branching would mean and is a §12 walk the model does not do.
                form = Branching3d(
                    label=label,
                    productFrame=Frame(child.attrib.get("productFrame", "lab")),
                )
            else:
                # Anything with no entry falls to the generic sentence. The five
                # §18.1.1 members kika does not name occur zero times across the
                # 558 evaluations the census walked, so nothing distributed
                # reaches this branch.
                self.unsupported(
                    child.tag, here,
                    UNREAD_DISTRIBUTION_FORMS.get(
                        child.tag,
                        "a §18 law kika declares and does not implement; the "
                        "product keeps its multiplicity and loses only this "
                        "form",
                    )
                )
                continue
            distribution[label] = form
        return distribution

    @reads("angularTwoBodyForm", "isotropic2d", "recoil")
    def angularTwoBody(self, element: ET.Element,
                       path: str) -> AngularTwoBody:
        """§18's ``angularTwoBody``, in all four of the shapes it takes."""
        label = element.attrib.get("label", "")
        here = f"{path}/angularTwoBody{_quoted(label)}"
        twoBody = AngularTwoBody(
            label=label,
            productFrame=Frame(element.attrib.get("productFrame",
                                                  "centerOfMass")),
        )
        for child in element:
            if child.tag in IGNORED:
                continue
            if child.tag == "recoil":
                twoBody.recoilHref = child.attrib.get("href", "")
            elif child.tag == "isotropic2d":
                twoBody.angular = Isotropic2d(
                    label=label, productFrame=twoBody.productFrame
                )
            else:
                try:
                    twoBody.angular = readFunction2d(
                        child, readAxes(child, self.resolve)
                    )
                except UnsupportedNode as exc:
                    self.unsupported(child.tag, here, exc.args[0])
        return twoBody

    # -- §18.3, uncorrelated ------------------------------------------------

    @reads("distributionForm", "uncorrelated")
    def uncorrelated(self, element: ET.Element, path: str) -> Uncorrelated:
        """§18.3's ``uncorrelated``, the library's commonest distribution.

        The two halves are read independently and either can come back
        ``None``: a form kika does not model is reported with its xPath and the
        other half is kept, because losing the angular distribution as well
        would throw away something that *was* read. What the schema forbids —
        a node with one child — is refused by the **writer**, in one place.
        """
        label = element.attrib.get("label", "")
        here = f"{path}/uncorrelated{_quoted(label)}"
        form = Uncorrelated(
            label=label,
            productFrame=Frame(element.attrib.get("productFrame", "lab")),
        )
        angular = element.find("angular")
        if angular is not None:
            form.angular = self.uncorrelatedAngular(
                angular, here, label, form.productFrame)
        energy = element.find("energy")
        if energy is not None:
            form.energy = self.uncorrelatedEnergy(energy, here)
        return form

    @reads("uncorrelatedAngularForm", "isotropic2d")
    def uncorrelatedAngular(self, element: ET.Element, path: str,
                            label: str, productFrame: Frame):
        """``uncorrelated/angular``: ``XYs2d``, ``isotropic2d`` or ``forward``.

        A different choice from ``angularTwoBody``'s (``gnds.xsd:1686`` against
        ``:1665``) — no ``regions2d``, no ``recoil``, and a ``forward`` that
        appears nowhere else — which is why ``isotropic2d`` is declared twice
        in the registry, once per family.
        """
        here = f"{path}/angular"
        for child in element:
            if child.tag in IGNORED:
                continue
            if child.tag == "isotropic2d":
                # The node itself has no attributes (gnds.xsd:1693);
                # the frame is the parent's, and copying it here keeps
                # the object from defaulting to centerOfMass and saying
                # something the file did not.
                return Isotropic2d(label=label,
                                   productFrame=productFrame)
            if child.tag in UNREAD_ANGULAR_FORMS:
                self.unsupported(child.tag, here,
                                 UNREAD_ANGULAR_FORMS[child.tag])
                return None
            try:
                return readFunction2d(child, readAxes(child, self.resolve))
            except UnsupportedNode as exc:
                self.unsupported(child.tag, here, exc.args[0])
        return None

    @reads("uncorrelatedEnergyForm", "discreteGamma", "primaryGamma",
           "NBodyPhaseSpace")
    def uncorrelatedEnergy(self, element: ET.Element, path: str):
        """``uncorrelated/energy``: eleven choices, four of which kika models.

        The seven it does not are the analytic spectra (``evaporation``,
        ``Watt``, ``MadlandNix``, …). They are reported rather than guessed:
        each is a formula with named parameters, and inventing a tabulation for
        one would put numbers in the file that the evaluator never wrote.
        """
        here = f"{path}/energy"
        for child in element:
            if child.tag in IGNORED:
                continue
            if child.tag == "discreteGamma":
                return DiscreteGamma(
                    value=float(child.attrib["value"]),
                    domainMin=float(child.attrib["domainMin"]),
                    domainMax=float(child.attrib["domainMax"]),
                    axes=readAxes(child, self.resolve),
                )
            if child.tag == "primaryGamma":
                return PrimaryGamma(
                    value=float(child.attrib["value"]),
                    domainMin=float(child.attrib["domainMin"]),
                    domainMax=float(child.attrib["domainMax"]),
                    axes=readAxes(child, self.resolve),
                    finalState=child.attrib.get("finalState"),
                )
            if child.tag in UNREAD_ENERGY_FORMS:
                self.unsupported(child.tag, here,
                                 UNREAD_ENERGY_FORMS[child.tag])
                return None
            if child.tag == "NBodyPhaseSpace":
                mass = child.find("mass")
                return NBodyPhaseSpace(
                    numberOfProducts=int(child.attrib["numberOfProducts"]),
                    mass=None if mass is None else PhysicalQuantity(
                        value=float(mass.attrib["value"]),
                        unit=mass.attrib.get("unit", ""),
                    ),
                )
            try:
                return readFunction2d(child, readAxes(child, self.resolve))
            except UnsupportedNode as exc:
                self.unsupported(child.tag, here, exc.args[0])
        return None


    # -- §18.4 and §18.5, the two orderings of DistributionAEType -----------

    def _distributionAE(self, element: ET.Element, path: str, tag: str,
                        cls: type):
        """The body both §18.4 and §18.5 have, because the schema gives them
        one complexType (``gnds.xsd:1797-1803``): an ``xs:sequence`` of a
        single ``XYs3d``, ``label`` and ``productFrame``.

        There is no choice to walk — but the child is still read through
        :func:`readFunction3d` rather than assumed, because a file that puts
        something else there should be *reported*, not decoded as if it were
        the expected node.

        **Both the tag and the class come from the caller, and neither is read
        off the element.** The two methods below are what say which ordering
        this is, each keyed by the tag it registered against, so the one thing
        the schema cannot express is decided by dispatch rather than by a
        string comparison in here. Sharing the body is safe precisely because
        the naming is not part of it.
        """
        label = element.attrib.get("label", "")
        here = f"{path}/{tag}{_quoted(label)}"
        form = cls(
            label=label,
            productFrame=Frame(element.attrib.get("productFrame", "lab")),
        )
        for child in element:
            if child.tag in IGNORED:
                continue
            try:
                form.xys3d = readFunction3d(child, readAxes(child, self.resolve))
            except UnsupportedNode as exc:
                self.unsupported(child.tag, here, exc.args[0])
        return form

    @reads("distributionForm", "energyAngular")
    def energyAngular(self, element: ET.Element, path: str) -> EnergyAngular:
        """§18.4: P(E′,mu|E), the outgoing energy outermost."""
        return self._distributionAE(element, path, "energyAngular",
                                    EnergyAngular)

    @reads("distributionForm", "angularEnergy")
    def angularEnergy(self, element: ET.Element, path: str) -> AngularEnergy:
        """§18.5: P(mu,E′|E), the *angle* outermost.

        A separate method rather than a flag on the one above, for the reason
        :class:`~kika.nuclear_data.model.distributions.AngularEnergy` gives:
        the element name is the only thing in the file that distinguishes the
        two, so decoding one into the other's class is a mistake nothing
        downstream could detect.
        """
        return self._distributionAE(element, path, "angularEnergy",
                                    AngularEnergy)

    # -- §18.6, KalbachMann -------------------------------------------------

    #: The three children of ``DistributionKalbachMannType`` in schema order.
    #: ``a`` is last and optional (``gnds.xsd:1810``); the other two are not.
    _KALBACH_CHILDREN = ("f", "r", "a")

    @reads("distributionForm", "KalbachMann")
    def kalbachMann(self, element: ET.Element, path: str) -> KalbachMann:
        """§18.6's ``KalbachMann``: the systematics' two (or three) parameters.

        ``gnds.xsd:1805-1814`` is an ``xs:sequence`` of ``f``, ``r`` and an
        optional ``a``, each an ``XYs2dWrapperType`` (``:2204-2208``) — a
        wrapper with no attributes holding exactly one primary ``XYs2d``. So
        the read is: unwrap, then hand the ``XYs2d`` to the functional reader,
        which is where ``axes`` and the interpolation attributes are already
        handled.

        **The wrapper is unwrapped rather than assumed away.** A ``<f>`` that
        holds something other than an ``XYs2d``, or nothing at all, is reported
        with its xPath instead of silently leaving the slot empty — the slot
        being empty is what :attr:`KalbachMann.isComplete` reads as "do not
        write this node", and the two must not be the same signal.

        An unknown child is reported and skipped: the sequence is closed, so
        anything else is a file kika should say something about rather than
        drop.
        """
        label = element.attrib.get("label", "")
        here = f"{path}/KalbachMann{_quoted(label)}"
        form = KalbachMann(
            label=label,
            productFrame=Frame(element.attrib.get("productFrame",
                                                  "centerOfMass")),
        )
        for child in element:
            if child.tag in IGNORED:
                continue
            if child.tag not in self._KALBACH_CHILDREN:
                self.unsupported(
                    child.tag, here,
                    f"gnds.xsd:1806-1811 gives <KalbachMann> the children "
                    f"{list(self._KALBACH_CHILDREN)} and nothing else"
                )
                continue
            for wrapped in child:
                if wrapped.tag in IGNORED:
                    continue
                try:
                    setattr(form, child.tag, readFunction2d(
                        wrapped, readAxes(wrapped, self.resolve)))
                except UnsupportedNode as exc:
                    self.unsupported(wrapped.tag, f"{here}/{child.tag}",
                                     exc.args[0])
        return form


def readDistribution(element: ET.Element, path: str,
                     resolve: Optional[Callable],
                     report: ConversionReport) -> Distribution:
    """§18's ``distribution`` node → the model. Never raises; reports instead."""
    return _DistributionReader(resolve, report).read(element, path)
