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

from kika.nuclear_data.model import (AngularTwoBody, ConversionReport,
                                     DiscreteGamma, Frame, Isotropic2d,
                                     NBodyPhaseSpace, PhysicalQuantity,
                                     PrimaryGamma, Uncorrelated, Unspecified)
from kika.nuclear_data.model.distributions import Distribution

from .nodes import NODES, Status, reads
from .primitives import UnsupportedNode, readAxes, readFunction2d

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

    @reads("distributionForm", "angularTwoBody", "unspecified")
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
            elif child.tag == "unspecified":
                form = Unspecified(
                    label=label,
                    productFrame=Frame(child.attrib.get("productFrame", "lab")),
                )
            else:
                self.unsupported(
                    child.tag, here,
                    "a §18 law kika declares and does not implement (phase 7b); "
                    "the product keeps its multiplicity and loses only this form"
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


def readDistribution(element: ET.Element, path: str,
                     resolve: Optional[Callable],
                     report: ConversionReport) -> Distribution:
    """§18's ``distribution`` node → the model. Never raises; reports instead."""
    return _DistributionReader(resolve, report).read(element, path)
