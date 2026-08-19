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

from kika.nuclear_data.model import (AngularTwoBody, ConversionReport, Frame,
                                     Isotropic2d, Unspecified)
from kika.nuclear_data.model.distributions import Distribution

from .nodes import reads
from .primitives import UnsupportedNode, readAxes, readFunction2d

__all__ = ["readDistribution"]

#: Children of a functional or container that carry no data kika reads. The
#: same tuple :mod:`kika.gnds.decode` uses, kept here so this module does not
#: import back into the one that imports it.
IGNORED = ("documentation", "axes")


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


def readDistribution(element: ET.Element, path: str,
                     resolve: Optional[Callable],
                     report: ConversionReport) -> Distribution:
    """§18's ``distribution`` node → the model. Never raises; reports instead."""
    return _DistributionReader(resolve, report).read(element, path)
