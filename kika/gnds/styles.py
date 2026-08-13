"""§9 ``styles``, read and written — shared by both roots.

A ``reactionSuite`` has one and a ``covarianceSuite`` has one, and the schema
makes it **mandatory** on each. They are the same node, so the code for it lives
here rather than in either reader: the first version put it on the reactionSuite
reader alone, and the covariance file kika wrote came out without a ``styles``
and failed validation on the first line after ``externalFiles``.

Only the two style nodes the library uses are built — ``evaluated`` (558 files)
and ``crossSectionReconstructed`` (482). The rest of §9 is modelled and occurs
in no distributed neutron evaluation, so meeting one is reported rather than
guessed at.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from kika.nuclear_data.model import (ConversionReport, CrossSectionReconstructed,
                                     Evaluated, PhysicalQuantity, RangeQuantity,
                                     Style, Styles)

__all__ = ["STYLES", "readStyles", "writeStyles", "readPhysicalQuantity",
           "readRange"]

#: §9 node name → the class it becomes.
STYLES = {
    "evaluated": Evaluated,
    "crossSectionReconstructed": CrossSectionReconstructed,
}


def readPhysicalQuantity(element: Optional[ET.Element]) -> Optional[PhysicalQuantity]:
    if element is None or "value" not in element.attrib:
        return None
    return PhysicalQuantity(value=float(element.attrib["value"]),
                            unit=element.attrib.get("unit", ""))


def readRange(element: Optional[ET.Element]) -> Optional[RangeQuantity]:
    if element is None or "min" not in element.attrib:
        return None
    return RangeQuantity(min=float(element.attrib["min"]),
                         max=float(element.attrib["max"]),
                         unit=element.attrib.get("unit", ""))


def readStyles(element: ET.Element, path: str, report: ConversionReport,
               tally=None) -> Styles:
    """§9.1 ``styles`` → the model.

    ``tally`` is the caller's counter for the losses that would otherwise be
    reported once per style — ``documentation``, which every ``evaluated``
    carries and no model node holds.
    """
    styles = Styles()
    for child in element:
        cls = STYLES.get(child.tag)
        if cls is None:
            report.unsupportedNode(
                f"{path}/{child.tag}: kika models this style but no distributed "
                f"neutron evaluation carries one, so reading it would be untested"
            )
            continue
        style: Style = cls(
            label=child.attrib.get("label", ""),
            derivedFrom=child.attrib.get("derivedFrom"),
            date=child.attrib.get("date"),
        )
        if isinstance(style, Evaluated):
            style.library = child.attrib.get("library")
            style.version = child.attrib.get("version")
            style.temperature = readPhysicalQuantity(child.find("temperature"))
            style.projectileEnergyDomain = readRange(
                child.find("projectileEnergyDomain")
            )
        if child.find("documentation") is not None and tally is not None:
            tally(
                "style <documentation>: free-text provenance, including the "
                "verbatim ENDF-6 header in <endfCompatible>, for which the "
                "model has no node"
            )
        styles.add(style)
    return styles


def writeStyles(root: ET.Element, styles: Styles, number,
                documentation: bool = True) -> ET.Element:
    """The model → ``<styles>``. ``number`` formats a float, from the encoder.

    **The two schemas disagree about ``documentation``, and that is what the
    flag is for.** ``gnds.xsd``'s ``RS_EvaluatedType`` makes the element
    *mandatory* on a reactionSuite's ``evaluated``; ``covariances.xsd``'s own
    ``evaluated`` does not admit it at all. Writing the same style node into
    both roots therefore cannot be right, and the first version that tried
    produced a covariance file rejected on its fifth line.

    Where it is written it is empty, which is both valid — every child of
    ``DocumentationType`` is optional — and honest: kika read no documentation
    and invents none.
    """
    container = ET.SubElement(root, "styles")
    for style in styles:
        element = ET.SubElement(container, style.gndsNodeName)
        for name, value in (("label", style.label),
                            ("derivedFrom", style.derivedFrom),
                            ("date", style.date)):
            if value is not None:
                element.attrib[name] = value
        if not isinstance(style, Evaluated):
            continue
        for name, value in (("library", style.library),
                            ("version", style.version)):
            if value is not None:
                element.attrib[name] = value
        if style.temperature is not None:
            node = ET.SubElement(element, "temperature")
            node.attrib["value"] = number(style.temperature.value)
            node.attrib["unit"] = style.temperature.unit
        if style.projectileEnergyDomain is not None:
            domain = style.projectileEnergyDomain
            node = ET.SubElement(element, "projectileEnergyDomain")
            node.attrib["min"] = number(domain.min)
            node.attrib["max"] = number(domain.max)
            node.attrib["unit"] = domain.unit
        if documentation:
            ET.SubElement(element, "documentation")
    return container
