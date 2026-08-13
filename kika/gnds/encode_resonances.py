"""§19 ``resonances``, model → XML. The other half of :mod:`kika.gnds.resonances`.

Separate from :mod:`kika.gnds.encode` for the reason the reader is separate: §19
is the one part of a ``reactionSuite`` that is not shaped like the rest, and
putting three formalisms' serialisation in the middle of the reaction writer
would bury both.

**The two things this has to get right, because nothing downstream would catch
them.** The R-Matrix table's ``columnHeaders`` must name the columns in the
order the widths were stored *and* each channel's ``columnIndex`` must point at
its own column — get either wrong and the file holds every number the model held,
in the wrong places, which reads back as a different evaluation. And a
``BreitWigner``'s l-blocks have to be flattened back into one table with ``L``
as a column, in the order the groups are in, so the round trip does not reorder
the resonances.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from kika.nuclear_data.model import (BreitWigner, ConversionReport, RMatrix,
                                     ScatteringRadius)

from .primitives import formatFraction

__all__ = ["writeResonances"]

#: §19.3.6's table columns, in the order FUDGE writes them and in the order the
#: two column sets in ENDF/B-VIII.1-GNDS use. ``fissionWidth`` is written only
#: when some resonance in the range has a non-zero one — 37 of the library's 386
#: ``BreitWigner`` tables carry the column and the rest do not, and adding it
#: everywhere would put a column of zeros into every non-fissile evaluation.
BREIT_WIGNER_COLUMNS = (
    ("energy", "eV", "energy"),
    ("L", "", None),
    ("J", "", "spin"),
    ("totalWidth", "eV", "totalWidth"),
    ("neutronWidth", "eV", "neutronWidth"),
    ("captureWidth", "eV", "captureWidth"),
    ("fissionWidth", "eV", "fissionWidth"),
)


def _number(value) -> str:
    from .encode import _number as formatNumber

    return formatNumber(value)


def _set(element: ET.Element, **attributes) -> ET.Element:
    for name, value in attributes.items():
        if value is not None:
            element.attrib[name] = value
    return element


def _true(value: bool) -> Optional[str]:
    return "true" if value else None


def writeResonances(root: ET.Element, resonances, report: ConversionReport,
                    domain) -> None:
    """``domain`` is ``(min, max)`` as strings — every ``constant1d`` in §19 has
    to declare one and the model's scalars do not carry it, so the evaluation's
    own ``projectileEnergyDomain`` is used, exactly as in the reaction writer."""
    element = ET.SubElement(root, "resonances")
    _scatteringRadius(element, resonances.scatteringRadius, report,
                      "resonances", domain=domain)
    for region in resonances.resolved:
        _resolved(element, region, report, domain)
    if resonances.unresolved is not None:
        _unresolved(element, resonances.unresolved, report, domain)


def _constant(parent: ET.Element, tag: Optional[str], value: float, domain,
              unit: str = "eV", label: str = "radius") -> ET.Element:
    """One ``<x><constant1d value= domainMin= domainMax=><axes/></constant1d>``.

    ``xData_constant1d`` makes both domain attributes and the ``axes`` child
    mandatory; a scalar written without them is a file no validator accepts.
    """
    holder = parent if tag is None else ET.SubElement(parent, tag)
    node = ET.SubElement(holder, "constant1d")
    _set(node, label="eval", value=_number(value),
         domainMin=domain[0], domainMax=domain[1])
    axes = ET.SubElement(node, "axes")
    _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
    _set(ET.SubElement(axes, "axis"), index="0", label=label, unit=unit)
    return node


def _scatteringRadius(parent: ET.Element, radius: Optional[ScatteringRadius],
                      report: ConversionReport, where: str, domain,
                      tag: str = "scatteringRadius") -> None:
    """§19's radius. **The unit is written as the model holds it, or not at all.**

    An ENDF-sourced suite carries the radius in ENDF's units — a tenth of a
    femtometre count — and ``unit`` is ``None`` for it, because the ENDF adapter
    states none. Writing ``fm`` there would be a factor-of-ten error dressed as
    a label, so the axis goes out with an empty unit and the report says why.
    """
    if radius is None:
        return
    element = ET.SubElement(parent, tag)
    if radius.unit is None:
        report.warn(
            f"{where}: the scattering radius carries no unit — it came from a "
            f"reader that states none, and ENDF's AP is in units of 10^-12 cm "
            f"while GNDS's is in fm. The axis is written with an empty unit "
            f"rather than labelled fm, because the number may not be in fm"
        )
    if radius.isEnergyDependent:
        node = ET.SubElement(element, "XYs1d")
        node.attrib["label"] = "eval"
        _radiusAxes(node, radius.unit)
        values = ET.SubElement(node, "values")
        values.text = " ".join(
            _number(v) for pair in zip(radius.energies, radius.values) for v in pair
        )
        return
    node = ET.SubElement(element, "constant1d")
    _set(node, label="eval", value=_number(radius.constant),
         domainMin=domain[0], domainMax=domain[1])
    _radiusAxes(node, radius.unit)


def _radiusAxes(parent: ET.Element, unit: Optional[str]) -> None:
    axes = ET.SubElement(parent, "axes")
    _set(ET.SubElement(axes, "axis"), index="1", label="energy_in", unit="eV")
    _set(ET.SubElement(axes, "axis"), index="0", label="radius", unit=unit or "")


def _resolved(parent: ET.Element, region, report: ConversionReport,
              domain) -> None:
    element = ET.SubElement(parent, "resolved")
    _set(element, domainMin=_number(region.domainMin),
         domainMax=_number(region.domainMax),
         domainUnit=region.domainUnit or "eV")
    formalism = region.formalism
    if isinstance(formalism, RMatrix):
        _rMatrix(element, formalism, report, domain)
    elif isinstance(formalism, BreitWigner):
        _breitWigner(element, formalism, report, domain)
    elif formalism is not None:
        report.unsupportedNode(
            f"resolved region: kika's writer has no serialisation for a "
            f"{type(formalism).__name__} formalism; the region is written empty"
        )


def _rMatrix(parent: ET.Element, formalism: RMatrix,
             report: ConversionReport, domain) -> None:
    element = ET.SubElement(parent, "RMatrix")
    _set(element, label=formalism.label or "eval",
         approximation=formalism.approximation,
         boundaryCondition=formalism.boundaryCondition,
         calculateChannelRadius=_true(formalism.calculateChannelRadius),
         relativisticKinematics=_true(formalism.relativisticKinematics),
         reducedWidthAmplitudes=_true(formalism.reducedWidthAmplitudes))
    if formalism.scatteringRadius is not None:
        _scatteringRadius(element, ScatteringRadius(
            constant=formalism.scatteringRadius), report, "RMatrix", domain)

    reactions = ET.SubElement(element, "resonanceReactions")
    for reaction in formalism.resonanceReactions:
        _resonanceReaction(reactions, reaction, report, domain)

    groups = ET.SubElement(element, "spinGroups")
    for group in formalism.spinGroups:
        _spinGroup(groups, group, report, domain)


def _resonanceReaction(parent: ET.Element, reaction, report: ConversionReport,
                       domain) -> None:
    element = ET.SubElement(parent, "resonanceReaction")
    _set(element, label=reaction.label, ejectile=reaction.ejectile,
         eliminated=_true(reaction.eliminated))
    if reaction.href:
        _set(ET.SubElement(element, "link"), href=reaction.href)
    else:
        # §19.3.3 makes the link mandatory and it is the only formal tie between
        # a resonance channel and a reaction. An ENDF-decoded evaluation has
        # none to give — ENDF states the channels by position — so the file
        # comes out without it and this says so rather than inventing an xPath.
        report.lost(
            f"resonanceReaction {reaction.label!r} has no link to a reaction, "
            f"which §19.3.3 requires; it came from a format that identifies "
            f"resonance channels by position rather than by reference"
        )
    if reaction.Q is not None:
        _constant(element, "Q", reaction.Q, domain, unit="eV", label="Q")
    if reaction.scatteringRadius is not None:
        _constant(element, "scatteringRadius", reaction.scatteringRadius,
                  domain, unit="fm")


def _spinGroup(parent: ET.Element, group, report: ConversionReport,
               domain) -> None:
    element = ET.SubElement(parent, "spinGroup")
    _set(element, label=group.label,
         spin=None if group.spin is None else formatFraction(group.spin),
         parity=None if group.parity is None else str(group.parity))

    channels = ET.SubElement(element, "channels")
    for channel in group.channels:
        node = ET.SubElement(channels, "channel")
        _set(node, label=channel.label,
             resonanceReaction=channel.resonanceReaction,
             L=None if channel.L is None else str(channel.L),
             channelSpin=(None if channel.channelSpin is None
                          else formatFraction(channel.channelSpin)),
             columnIndex=(None if channel.columnIndex is None
                          else str(channel.columnIndex)))
        for tag, value in (("scatteringRadius", channel.scatteringRadius),
                           ("hardSphereRadius", channel.hardSphereRadius)):
            if value is not None:
                _constant(node, tag, value, domain, unit="fm")

    _rMatrixTable(element, group, report)


def _rMatrixTable(parent: ET.Element, group, report: ConversionReport) -> None:
    """§19.3.5's ``table``, with each channel's width back in its own column.

    The table is laid out by ``columnIndex``, not by channel order: the two are
    the same in every distributed file and need not be, and a writer that
    assumed they were would silently transpose the widths of any file where they
    differ.
    """
    parameters = ET.SubElement(parent, "resonanceParameters")
    columns = 1 + len(group.channels)
    byIndex = {}
    for position, channel in enumerate(group.channels):
        if channel.columnIndex is None:
            report.lost(
                f"spinGroup {group.label!r}: channel {channel.label!r} has no "
                f"columnIndex, so its widths have no column to go in and are "
                f"not written"
            )
            continue
        columns = max(columns, channel.columnIndex + 1)
        byIndex[channel.columnIndex] = (position, channel)

    table = ET.SubElement(parameters, "table")
    _set(table, rows=str(len(group.energies)), columns=str(columns))
    headers = ET.SubElement(table, "columnHeaders")
    _set(ET.SubElement(headers, "column"), index="0", name="energy", unit="eV")
    for index in range(1, columns):
        entry = byIndex.get(index)
        _set(ET.SubElement(headers, "column"), index=str(index),
             name=f"{entry[1].resonanceReaction} width" if entry else f"column{index}",
             unit="eV")

    rows = []
    for position, energy in enumerate(group.energies):
        row = [energy] + [0.0] * (columns - 1)
        for index, (channelPosition, _) in byIndex.items():
            row[index] = group.widths[position][channelPosition]
        rows.append(row)
    data = ET.SubElement(table, "data")
    data.text = " ".join(_number(v) for row in rows for v in row)


def _breitWigner(parent: ET.Element, formalism: BreitWigner,
                 report: ConversionReport, domain) -> None:
    element = ET.SubElement(parent, "BreitWigner")
    _set(element, label=formalism.label or "eval",
         approximation=str(formalism.approximation),
         calculateChannelRadius=_true(formalism.calculateChannelRadius))
    if formalism.scatteringRadius is not None:
        _scatteringRadius(element, ScatteringRadius(
            constant=formalism.scatteringRadius), report, "BreitWigner", domain)

    resonances = [(group.L, resonance)
                  for group in formalism.resonanceParameters.spinGroups
                  for resonance in group.resonances]
    withFission = any(r.fissionWidth for _, r in resonances)
    columns = [c for c in BREIT_WIGNER_COLUMNS
               if c[0] != "fissionWidth" or withFission]

    parameters = ET.SubElement(element, "resonanceParameters")
    table = ET.SubElement(parameters, "table")
    _set(table, rows=str(len(resonances)), columns=str(len(columns)))
    headers = ET.SubElement(table, "columnHeaders")
    for index, (name, unit, _) in enumerate(columns):
        _set(ET.SubElement(headers, "column"), index=str(index), name=name,
             unit=unit)

    numbers = []
    for L, resonance in resonances:
        for name, _, field in columns:
            numbers.append(L if field is None else getattr(resonance, field))
    data = ET.SubElement(table, "data")
    data.text = " ".join(_number(v) for v in numbers)


def _unresolved(parent: ET.Element, region, report: ConversionReport,
                domain) -> None:
    element = ET.SubElement(parent, "unresolved")
    _set(element, domainMin=_number(region.domainMin),
         domainMax=_number(region.domainMax),
         domainUnit=region.domainUnit or "eV")
    widths = region.tabulatedWidths
    if widths is None:
        report.lost("the unresolved region has no tabulatedWidths and is empty")
        return

    node = ET.SubElement(element, "tabulatedWidths")
    _set(node, label=widths.label or "eval",
         approximation="SingleLevelBreitWigner",
         useForSelfShieldingOnly=_true(widths.selfShieldingOnly))
    if widths.scatteringRadius is not None:
        _scatteringRadius(node, ScatteringRadius(
            constant=widths.scatteringRadius), report, "tabulatedWidths", domain)

    reactions = ET.SubElement(node, "resonanceReactions")
    for reaction in widths.resonanceReactions:
        _resonanceReaction(reactions, reaction, report, domain)
    if not widths.resonanceReactions:
        report.lost(
            "the unresolved region names no resonanceReactions, which §19.4.1 "
            "requires; the channels are identified only by the labels on their "
            "own widths, and reconstructing the list from those would produce "
            "entries with no link in them"
        )

    Ls = ET.SubElement(node, "Ls")
    byL = {}
    for group in widths.spinGroups:
        byL.setdefault(group.L, []).append(group)
    for position, (L, groups) in enumerate(byL.items()):
        lNode = ET.SubElement(Ls, "L")
        _set(lNode, label=str(position), value=str(L))
        Js = ET.SubElement(lNode, "Js")
        for index, group in enumerate(groups):
            _unresolvedSpinGroup(Js, group, index, widths.energyGrid, report,
                                 domain)


def _channelLabels(widths) -> list:
    seen = []
    for group in widths.spinGroups:
        for channel in group.channels:
            if channel.label not in seen:
                seen.append(channel.label)
    return seen


def _unresolvedSpinGroup(parent: ET.Element, group, index: int, blockGrid,
                         report: ConversionReport, domain) -> None:
    element = ET.SubElement(parent, "J")
    _set(element, label=str(index), value=formatFraction(group.J))

    grid = group.levelSpacingEnergies if group.levelSpacingEnergies is not None \
        else blockGrid
    _average(element, "levelSpacing", grid, group.levelSpacing, None,
             f"spinGroup L={group.L} J={group.J} levelSpacing", report, domain,
             unit="eV", label="levelSpacing")

    widths = ET.SubElement(element, "widths")
    for position, channel in enumerate(group.channels):
        node = ET.SubElement(widths, "width")
        _set(node, label=str(position), resonanceReaction=channel.label,
             degreesOfFreedom=_number(channel.degreesOfFreedom))
        grid = channel.energies if channel.energies is not None else blockGrid
        _average(node, None, grid, channel.widths, channel.constantWidth,
                 f"spinGroup L={group.L} J={group.J} width {channel.label!r}",
                 report, domain, unit="eV", label="width")


def _average(parent: ET.Element, tag: Optional[str], grid, values,
             constant: Optional[float], where: str,
             report: ConversionReport, domain, unit: str = "eV",
             label: str = "width") -> None:
    """One §19.4.1 average: an ``XYs1d`` over its grid, or a ``constant1d``."""
    if constant is not None:
        _constant(parent, tag, constant, domain, unit=unit, label=label)
        return
    holder = parent if tag is None else ET.SubElement(parent, tag)
    if values is None:
        report.lost(f"{where}: no values, so the node is written empty")
        return
    if grid is None or len(grid) != len(values):
        report.lost(
            f"{where}: {len(values)} values and "
            f"{'no' if grid is None else len(grid)} energies to put them on, so "
            f"the average is not written rather than written against a grid it "
            f"does not belong to"
        )
        return
    node = ET.SubElement(holder, "XYs1d")
    node.attrib["label"] = "eval"
    _radiusAxes(node, unit)
    text = ET.SubElement(node, "values")
    text.text = " ".join(_number(v)
                         for pair in zip(np.asarray(grid), np.asarray(values))
                         for v in pair)
