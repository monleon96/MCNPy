"""GNDS §19 ``resonances`` → the model.

The block :mod:`kika.gnds.decode` leaves alone, split out because it is the one
part of a ``reactionSuite`` that is *not* shaped like the rest: three
formalisms, each with its own tree, none of them a functional.

**Where the widths are, and why the shapes differ.** §19.3 stores an R-Matrix
spin group's parameters as a ``table`` — a flat block of numbers with a
``columnHeaders`` block naming each column — and the channels index into it by
``columnIndex``. §19.3.6's ``BreitWigner`` uses the same table with the columns
named for the widths themselves (``energy``, ``L``, ``J``, ``totalWidth``,
``neutronWidth``, ``captureWidth``, and ``fissionWidth`` in 37 of 386 files).
§19.4.1's ``tabulatedWidths`` uses neither: it nests ``Ls/L/Js/J`` and hangs an
``XYs1d`` off each average. All three are read here, onto the model's three
formalism nodes, which were built for exactly this split.

**The two things measured across all 558 evaluations before this was written:**

*The R-Matrix tables carry two to six width columns* (639 / 105 / 12 / 2 blocks
at 2, 3, 4 and 5), so nothing here may assume Reich-Moore's two. The column is
found through ``channel/@columnIndex``, which is what §19.3.4 provides it for.

*66 of the 351 unresolved blocks tabulate their averages on more than one energy
grid* — up to seven. ENDF's URR has one grid per range, so kika's model had one
``energyGrid``; reading a GNDS file through that would attach one grid's
energies to another grid's widths, which is a wrong average rather than a
missing one. :attr:`~kika.nuclear_data.model.resonances.UnresolvedChannel.energies`
was added for it, and the block-level grid is filled only when every curve
agrees.

**What is not read, and is reported instead:** ``externalRMatrix`` (7 nodes in
the library), a ``hardSphereRadius`` on a ``resonanceReaction`` (4), and the
per-region interpolation of a ``regions1d`` average width — that last one is an
*approximation*, not a loss, because the flattened numbers look exactly like
data and only the rule connecting them is gone.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable, List, Optional

import numpy as np

from kika.nuclear_data.model import (BreitWigner, BreitWignerApproximation,
                                     Channel, ConversionReport, Regions1d,
                                     Resonance, ResonanceParameters,
                                     ResonanceReaction, Resonances,
                                     ResolvedRegion, RMatrix, RMatrixSpinGroup,
                                     ScatteringRadius, SpinGroup,
                                     MODEL_RADIUS_UNIT, radiusFromStatedUnit,
                                     TabulatedWidths, UnresolvedChannel,
                                     UnresolvedRegion, UnresolvedSpinGroup,
                                     XYs1d)

from .primitives import UnsupportedNode, readAxes, readForm, readFraction

__all__ = ["readResonances"]

#: §19.3.6's ``approximation`` attribute, onto the model's enumeration. The
#: spec's spellings, and the only two the library uses.
BREIT_WIGNER_APPROXIMATIONS = {
    "SingleLevel": BreitWignerApproximation.singleLevel,
    "MultiLevel": BreitWignerApproximation.multiLevel,
}

#: §19.3.6's table columns that are *not* widths. Everything else in the header
#: is one, and is matched to a model field by name.
BREIT_WIGNER_INDEX_COLUMNS = ("energy", "L", "J")

#: ``columnHeaders`` name → the :class:`Resonance` field it fills. A column not
#: here is reported rather than dropped, because the two column sets in the
#: library differ only by ``fissionWidth`` and a third would be a new formalism.
BREIT_WIGNER_WIDTHS = {
    "totalWidth": "totalWidth",
    "neutronWidth": "neutronWidth",
    "captureWidth": "captureWidth",
    "fissionWidth": "fissionWidth",
}


def _optionalFloat(element: ET.Element, name: str) -> Optional[float]:
    value = element.attrib.get(name)
    return None if value is None else float(value)


def _isTrue(element: ET.Element, name: str) -> bool:
    return element.attrib.get(name) == "true"


def _constant(element: Optional[ET.Element]) -> Optional[float]:
    """The scalar out of a ``<x><constant1d value=.../></x>`` wrapper.

    Returns ``None`` for a wrapper holding a *table* rather than a constant;
    the caller that can hold a table asks for it separately, and the one that
    cannot gets ``None`` instead of a number picked out of a curve.
    """
    if element is None:
        return None
    constant = element.find("constant1d")
    return None if constant is None else float(constant.attrib["value"])


class _ResonanceReader:
    """One read of one ``resonances`` block, carrying the report and resolver."""

    def __init__(self, resolve: Optional[Callable], report: ConversionReport,
                 readPoPs: Callable) -> None:
        self.resolve = resolve
        self.report = report
        #: :mod:`kika.gnds.decode`'s minimal §12 reader, passed in rather than
        #: imported: §19.3.1's local ``PoPs`` is the same node as the suite's,
        #: and importing it the other way would make the two modules circular.
        self.readPoPs = readPoPs

    # -- reporting ---------------------------------------------------------

    def unsupported(self, tag: str, path: str, reason: str) -> None:
        self.report.unsupportedNode(f"{path}/{tag}: {reason}")

    def function(self, element: ET.Element, path: str):
        try:
            return readForm(element, readAxes(element, self.resolve))
        except UnsupportedNode as exc:
            self.unsupported(element.tag, path, exc.args[0])
            return None

    # -- the block ---------------------------------------------------------

    def read(self, element: ET.Element, path: str) -> Resonances:
        here = f"{path}/resonances"
        resonances = Resonances(
            scatteringRadius=self.readScatteringRadius(
                element.find("scatteringRadius"), here
            )
        )
        for child in element.findall("resolved"):
            resonances.resolved.append(self.readResolved(child, here))
        unresolved = element.find("unresolved")
        if unresolved is not None:
            resonances.unresolved = self.readUnresolved(unresolved, here)

        known = {"scatteringRadius", "resolved", "unresolved"}
        for child in element:
            if child.tag not in known:
                self.unsupported(child.tag, here, "not a §19 resonance node kika reads")
        return resonances

    def readScatteringRadius(self, element: Optional[ET.Element],
                             path: str) -> Optional[ScatteringRadius]:
        """§19. A constant in 834 files, an ``XYs1d`` in 53.

        The energy-dependent form keeps its interpolation, because the radius
        sets the hard-sphere phase shift and reading it with the wrong rule
        between nodes is a wrong phase shift rather than a missing one — the
        reason the model grew the field in phase 4.
        """
        if element is None:
            return None
        here = f"{path}/scatteringRadius"
        constant = element.find("constant1d")
        if constant is not None:
            value, unit = self.toModelUnits(
                float(constant.attrib["value"]), self.radiusUnit(constant), here)
            return ScatteringRadius(constant=value, unit=unit)
        table = element.find("XYs1d")
        if table is None:
            self.unsupported(
                "scatteringRadius", path,
                f"holds {[c.tag for c in element]}, neither a constant1d nor "
                f"an XYs1d"
            )
            return None
        form = self.function(table, here)
        if form is None:
            return None
        values, unit = self.toModelUnits(form.ys, self.radiusUnit(table), here)
        return ScatteringRadius(
            energies=form.xs, values=values, interpolation=form.interpolation,
            unit=unit,
        )

    def toModelUnits(self, value, stated: Optional[str], where: str):
        """A radius as the file states it → ``(fm, unit)`` for the model.

        The model states every radius in fm (``MODEL_RADIUS_UNIT``), so this is
        where a GNDS file's own unit is honoured rather than assumed. Every
        radius axis in ENDF/B-VIII.1-GNDS says ``fm``, which makes this a no-op
        on the whole library — and that is exactly why it is worth having: the
        one file that says something else would otherwise be read as fm without
        a word, which is the shape of the error §7.2 is about.

        A unit the converter does not know is **kept as read and reported**,
        with the unit it came with, so a consumer asking for ``radiusUnit`` can
        still tell. Refusing outright would lose a whole evaluation over a
        field that most consumers never touch.
        """
        converted, problem = radiusFromStatedUnit(value, stated)
        if problem is not None:
            self.report.warn(f"{where}: {problem}")
            return converted, stated
        return converted, MODEL_RADIUS_UNIT

    def modelRadius(self, wrapper: Optional[ET.Element], where: str):
        """``(radius in fm, unit)`` out of a ``constant1d`` wrapper, or ``(None, None)``."""
        value = _constant(wrapper)
        if value is None:
            return None, None
        return self.toModelUnits(value, self.constantUnit(wrapper), where)

    def constantUnit(self, wrapper: Optional[ET.Element]) -> Optional[str]:
        """The radius unit out of the same wrapper :func:`_constant` reads.

        The bare-float radii — a channel's, a resonance reaction's, an
        l-block's — are numbers in the model rather than
        :class:`~kika.nuclear_data.model.resonances.ScatteringRadius` objects,
        because a reconstruction reads them as numbers. Their unit still has to
        survive the read, or the writer has nothing to write and either drops
        it or, as it did until now, asserts ``fm`` over a number that may be in
        ENDF's tenths of a femtometre.
        """
        if wrapper is None:
            return None
        constant = wrapper.find("constant1d")
        return None if constant is None else self.radiusUnit(constant)

    def radiusUnit(self, element: ET.Element) -> Optional[str]:
        """The radius axis's unit — ``fm`` in every file of the library.

        Recorded because the ENDF path stores the *same* radius as a tenth of
        this number: ENDF writes AP in units of 10^-12 cm. See
        :attr:`~kika.nuclear_data.model.resonances.ScatteringRadius.unit`.
        """
        axes = readAxes(element, self.resolve)
        if axes is None:
            return None
        for axis in axes:
            if axis.index == 0:
                return axis.unit or None
        return None

    # -- resolved ----------------------------------------------------------

    def readResolved(self, element: ET.Element, path: str) -> ResolvedRegion:
        here = f"{path}/resolved"
        region = ResolvedRegion(
            domainMin=float(element.attrib["domainMin"]),
            domainMax=float(element.attrib["domainMax"]),
            domainUnit=element.attrib.get("domainUnit", "eV"),
        )
        for child in element:
            if child.tag == "RMatrix":
                region.formalism = self.readRMatrix(child, here)
            elif child.tag == "BreitWigner":
                region.formalism = self.readBreitWigner(child, here)
            else:
                self.unsupported(
                    child.tag, here,
                    "§19.2 admits it and no distributed neutron evaluation uses "
                    "one; kika models BreitWigner and RMatrix"
                )
        return region

    def readRMatrix(self, element: ET.Element, path: str) -> RMatrix:
        label = element.attrib.get("label", "")
        here = f"{path}/RMatrix"
        pops = element.find("PoPs")
        rMatrixRadius, rMatrixRadiusUnit = self.modelRadius(
            element.find("scatteringRadius"), here)
        if element.attrib.get("supportsAngularReconstruction") == "true":
            # A capability hint FUDGE writes for its own reconstructor, not a
            # property of the evaluation. Recorded so the writer does not have
            # to guess whether its absence was meaningful.
            self.report.lost(
                f"{here}: supportsAngularReconstruction=true, a hint about what "
                f"a reconstructor can do with these parameters rather than a "
                f"property of them; kika has no node for it"
            )
        return RMatrix(
            label=label,
            approximation=element.attrib.get("approximation"),
            boundaryCondition=element.attrib.get("boundaryCondition"),
            calculateChannelRadius=_isTrue(element, "calculateChannelRadius"),
            # §19.3.1's two flags. Both were **written** from the model
            # (encode_resonances.py:161-162) and never read back, so a file that
            # declared either came out of a kika round trip with it False.
            # `reducedWidthAmplitudes` is ENDF's IFG: it says whether `widths`
            # are widths in eV or reduced-width amplitudes in eV^1/2, and the two
            # are not interchangeable, so losing it silently reinterprets every
            # width in the table.
            reducedWidthAmplitudes=_isTrue(element, "reducedWidthAmplitudes"),
            relativisticKinematics=_isTrue(element, "relativisticKinematics"),
            scatteringRadius=rMatrixRadius,
            radiusUnit=rMatrixRadiusUnit,
            PoPs=None if pops is None else self.readPoPs(pops),
            resonanceReactions=self.readResonanceReactions(element, here),
            spinGroups=[
                self.readSpinGroup(group, f"{here}/spinGroups")
                for group in element.findall("spinGroups/spinGroup")
            ],
        )

    def readResonanceReactions(self, element: ET.Element,
                               path: str) -> List[ResonanceReaction]:
        out = []
        for child in element.findall("resonanceReactions/resonanceReaction"):
            here = f"{path}/resonanceReactions/resonanceReaction" \
                   f"[@label='{child.attrib.get('label', '')}']"
            link = child.find("link")
            reactionRadius, reactionRadiusUnit = self.modelRadius(
                child.find("scatteringRadius"), here)
            if child.find("hardSphereRadius") is not None:
                self.unsupported(
                    "hardSphereRadius", here,
                    "§19.3.3 allows one here and §19.3.4 allows one per channel; "
                    "kika's model carries the channel's, which is the one the "
                    "phase shift uses, and 4 nodes in the whole library set this"
                )
            out.append(ResonanceReaction(
                label=child.attrib.get("label", ""),
                ejectile=child.attrib.get("ejectile"),
                eliminated=_isTrue(child, "eliminated"),
                Q=_constant(child.find("Q")),
                scatteringRadius=reactionRadius,
                radiusUnit=reactionRadiusUnit,
                href=None if link is None else link.attrib.get("href"),
            ))
        return out

    def readSpinGroup(self, element: ET.Element, path: str) -> RMatrixSpinGroup:
        label = element.attrib.get("label", "")
        here = f"{path}/spinGroup[@label='{label}']"
        spin = element.attrib.get("spin")
        channels = [
            self.readChannel(child, here)
            for child in element.findall("channels/channel")
        ]
        energies, widths = self.readParameterTable(element, channels, here)
        return RMatrixSpinGroup(
            label=label,
            spin=None if spin is None else readFraction(spin),
            parity=(int(element.attrib["parity"])
                    if "parity" in element.attrib else None),
            channels=channels,
            energies=energies,
            widths=widths,
        )

    def readChannel(self, element: ET.Element, path: str) -> Channel:
        label = element.attrib.get("label", "")
        channelSpin = element.attrib.get("channelSpin")
        if element.find("externalRMatrix") is not None:
            self.unsupported(
                "externalRMatrix", f"{path}/channels/channel[@label='{label}']",
                "§19.3.4's parametrisation of the R-matrix outside the fitted "
                "range; 7 nodes in the whole library carry one and kika's model "
                "has no node for it"
            )
        where = f"{path}/channels/channel[@label='{label}']"
        channelRadius, channelRadiusUnit = self.modelRadius(
            element.find("scatteringRadius"), where)
        hardSphere, hardSphereUnit = self.modelRadius(
            element.find("hardSphereRadius"), where)
        return Channel(
            label=label,
            resonanceReaction=element.attrib.get("resonanceReaction", ""),
            L=int(element.attrib["L"]) if "L" in element.attrib else None,
            channelSpin=None if channelSpin is None else readFraction(channelSpin),
            columnIndex=(int(element.attrib["columnIndex"])
                         if "columnIndex" in element.attrib else None),
            scatteringRadius=channelRadius,
            hardSphereRadius=hardSphere,
            # One unit for both, as the field's docstring says: they come
            # off the same node and no file states them differently.
            radiusUnit=channelRadiusUnit or hardSphereUnit,
        )

    def readParameterTable(self, element: ET.Element, channels: List[Channel],
                           path: str):
        """§19.3.5's ``table`` → ``(energies, [[width per channel] per resonance])``.

        The width columns are located through each channel's ``columnIndex``,
        not by position, because a spin group may have two to six of them and
        because §19.3.4 provides the index for exactly this. Column 0 is the
        energy in every table in the library, and that is checked rather than
        assumed: a header that named it otherwise would silently make every
        resonance energy a width.
        """
        table = element.find("resonanceParameters/table")
        if table is None:
            return [], []
        data = self.readTable(table, path)
        if data is None or data.size == 0:
            return [], []

        headers = [c.attrib.get("name", "") for c in table.findall("columnHeaders/column")]
        if headers and headers[0] != "energy":
            self.unsupported(
                "table", path,
                f"column 0 is {headers[0]!r} and every resonance table in "
                f"ENDF/B-VIII.1-GNDS names it 'energy'; the table is not read "
                f"rather than read with the columns transposed"
            )
            return [], []

        energies = data[:, 0].tolist()
        widths = []
        for row in data:
            widths.append([
                float(row[channel.columnIndex])
                if channel.columnIndex is not None
                and channel.columnIndex < row.size else 0.0
                for channel in channels
            ])
        missing = [c.label for c in channels if c.columnIndex is None]
        if missing:
            self.report.lost(
                f"{path}: channels {missing} carry no columnIndex, so their "
                f"widths could not be located in the table and are zero"
            )
        return energies, widths

    def readTable(self, element: ET.Element, path: str) -> Optional[np.ndarray]:
        """The ``data`` block, reshaped to the ``rows`` x ``columns`` declared.

        The declared shape is checked against the count actually present. A
        table whose ``data`` is short reshapes to something with the wrong
        number of resonances in it, and every energy after the first row would
        be off by the shortfall — visible nowhere except in the cross section.
        """
        rows = int(element.attrib["rows"])
        columns = int(element.attrib["columns"])
        data = element.find("data")
        text = "" if data is None else (data.text or "")
        values = np.fromstring(text, dtype=float, sep=" ") if text.strip() else np.empty(0)
        if values.size != rows * columns:
            self.unsupported(
                "table", path,
                f"declares {rows} rows x {columns} columns = {rows * columns} "
                f"numbers and its data holds {values.size}; the table is not "
                f"read rather than reshaped into a plausible wrong answer"
            )
            return None
        return values.reshape(rows, columns)

    # -- Breit-Wigner ------------------------------------------------------

    def readBreitWigner(self, element: ET.Element, path: str) -> BreitWigner:
        """§19.3.6. One table for the whole range, grouped into l-blocks here.

        GNDS states the resonances as one flat table with ``L`` as a column;
        kika's model groups them by L, which is ENDF's l-block and what a
        reconstructor iterates. The grouping keeps first-appearance order rather
        than sorting, so a round trip writes the rows back where they were.
        """
        label = element.attrib.get("label", "")
        here = f"{path}/BreitWigner"
        bwRadius, bwRadiusUnit = self.modelRadius(
            element.find("scatteringRadius"), here)
        pops = element.find("PoPs")
        approximation = element.attrib.get("approximation")
        if approximation not in BREIT_WIGNER_APPROXIMATIONS:
            self.unsupported(
                "BreitWigner", here,
                f"approximation={approximation!r}; §19.3.6 defines "
                f"{sorted(BREIT_WIGNER_APPROXIMATIONS)} and the model enumerates "
                f"those two"
            )

        return BreitWigner(
            label=label,
            approximation=BREIT_WIGNER_APPROXIMATIONS.get(
                approximation, BreitWignerApproximation.multiLevel
            ),
            calculateChannelRadius=_isTrue(element, "calculateChannelRadius"),
            scatteringRadius=bwRadius,
            radiusUnit=bwRadiusUnit,
            PoPs=None if pops is None else self.readPoPs(pops),
            resonanceParameters=self.readBreitWignerTable(element, here),
        )

    def readBreitWignerTable(self, element: ET.Element,
                             path: str) -> ResonanceParameters:
        parameters = ResonanceParameters()
        table = element.find("resonanceParameters/table")
        if table is None:
            return parameters
        data = self.readTable(table, path)
        if data is None or data.size == 0:
            return parameters

        headers = [c.attrib.get("name", "") for c in table.findall("columnHeaders/column")]
        index = {name: position for position, name in enumerate(headers)}
        for name in ("energy", "L", "J"):
            if name not in index:
                self.unsupported(
                    "table", path,
                    f"has no {name!r} column; its headers are {headers}, and a "
                    f"Breit-Wigner table without one of energy/L/J cannot be "
                    f"grouped into l-blocks"
                )
                return parameters
        unknown = [name for name in headers
                   if name not in BREIT_WIGNER_INDEX_COLUMNS
                   and name not in BREIT_WIGNER_WIDTHS]
        if unknown:
            self.unsupported(
                "table", path,
                f"columns {unknown} are neither an index column nor a width the "
                f"model names; their numbers are dropped"
            )

        groups = {}
        for row in data:
            L = int(row[index["L"]])
            if L not in groups:
                groups[L] = SpinGroup(L=L)
                parameters.spinGroups.append(groups[L])
            widths = {
                field: float(row[index[name]])
                for name, field in BREIT_WIGNER_WIDTHS.items() if name in index
            }
            groups[L].resonances.append(Resonance(
                energy=float(row[index["energy"]]),
                spin=float(row[index["J"]]),
                **widths,
            ))
        return parameters

    # -- unresolved --------------------------------------------------------

    def readUnresolved(self, element: ET.Element, path: str) -> UnresolvedRegion:
        here = f"{path}/unresolved"
        region = UnresolvedRegion(
            domainMin=float(element.attrib["domainMin"]),
            domainMax=float(element.attrib["domainMax"]),
            domainUnit=element.attrib.get("domainUnit", "eV"),
        )
        widths = element.find("tabulatedWidths")
        if widths is None:
            self.unsupported(
                "unresolved", path,
                f"holds {[c.tag for c in element]}; §19.4 gives the region a "
                f"tabulatedWidths and kika reads no other parametrisation"
            )
            return region
        region.tabulatedWidths = self.readTabulatedWidths(widths, here)
        return region

    def readTabulatedWidths(self, element: ET.Element,
                            path: str) -> TabulatedWidths:
        here = f"{path}/tabulatedWidths"
        urrRadius, urrRadiusUnit = self.modelRadius(
            element.find("scatteringRadius"), here)
        pops = element.find("PoPs")
        widths = TabulatedWidths(
            label=element.attrib.get("label", ""),
            selfShieldingOnly=_isTrue(element, "useForSelfShieldingOnly"),
            scatteringRadius=urrRadius,
            radiusUnit=urrRadiusUnit,
            PoPs=None if pops is None else self.readPoPs(pops),
            resonanceReactions=self.readResonanceReactions(element, here),
        )
        for L in element.findall("Ls/L"):
            for J in L.findall("Js/J"):
                widths.spinGroups.append(self.readUnresolvedSpinGroup(L, J, here))

        self.setBlockGrid(widths, here)
        return widths

    def readUnresolvedSpinGroup(self, L: ET.Element, J: ET.Element,
                                path: str) -> UnresolvedSpinGroup:
        here = (f"{path}/Ls/L[@label='{L.attrib.get('label', '')}']"
                f"/Js/J[@label='{J.attrib.get('label', '')}']")
        spacing = self.readAverage(J.find("levelSpacing"), f"{here}/levelSpacing")
        group = UnresolvedSpinGroup(
            L=int(L.attrib["value"]),
            J=readFraction(J.attrib["value"]),
            levelSpacing=None if spacing is None else spacing[1],
            levelSpacingEnergies=None if spacing is None else spacing[0],
        )
        for width in J.findall("widths/width"):
            label = width.attrib.get("label", "")
            average = self.readAverage(width, f"{here}/widths/width[@label='{label}']")
            group.channels.append(UnresolvedChannel(
                label=width.attrib.get("resonanceReaction", label),
                degreesOfFreedom=float(width.attrib.get("degreesOfFreedom", 1.0)),
                widths=None if average is None else average[1],
                constantWidth=None if average is not None else _constant(width),
                energies=None if average is None else average[0],
            ))
        return group

    def readAverage(self, element: Optional[ET.Element], path: str):
        """``(energies, values)`` for one average, or ``None`` for a constant.

        A ``regions1d`` — 414 of the library's 5 937 widths — is flattened onto
        one grid, which drops the differing interpolation rule *between* its
        regions. That is an **approximation** and not a loss: the numbers that
        come back are the evaluator's at every node and something else between
        two nodes that belonged to different regions, and nothing in the result
        says so. It is reported for that reason.
        """
        if element is None:
            return None
        for child in element:
            if child.tag == "constant1d":
                return None
            if child.tag in ("XYs1d", "regions1d"):
                form = self.function(child, path)
                if form is None:
                    return None
                if isinstance(form, Regions1d):
                    self.report.approximated(
                        f"{path}: a regions1d average width was flattened onto "
                        f"one grid; the per-region interpolation rules are gone "
                        f"and the values between two nodes of different regions "
                        f"are no longer the evaluator's"
                    )
                    xs, ys, _ = form.toEndfRegions()
                    return xs, ys
                return form.xs, form.ys
            if child.tag != "axes":
                self.unsupported(child.tag, path, "not an average kika reads")
        return None

    def setBlockGrid(self, widths: TabulatedWidths, path: str) -> None:
        """Fill ``energyGrid`` when every curve in the block shares one grid.

        285 of the library's 351 unresolved blocks do, and for those the
        per-channel grids are redundant and are cleared, so the model comes out
        the way an ENDF-decoded evaluation does. For the other 66 the block
        grid stays ``None`` and each curve keeps its own — the alternative,
        picking one and attaching it to all of them, produces average widths
        that are wrong at every energy without anything saying so.
        """
        grids = []
        for group in widths.spinGroups:
            if group.levelSpacingEnergies is not None:
                grids.append(group.levelSpacingEnergies)
            grids.extend(c.energies for c in group.channels if c.energies is not None)
        if not grids:
            return
        first = grids[0]
        if not all(g.shape == first.shape and np.array_equal(g, first) for g in grids):
            self.report.warn(
                f"{path}: its averages are tabulated on "
                f"{len({g.tobytes() for g in grids})} different energy grids, so "
                f"each keeps its own and the block-level energyGrid stays unset"
            )
            return
        widths.energyGrid = first
        for group in widths.spinGroups:
            group.levelSpacingEnergies = None
            for channel in group.channels:
                channel.energies = None


def readResonances(element: ET.Element, path: str, resolve: Optional[Callable],
                   report: ConversionReport, readPoPs: Callable) -> Resonances:
    """§19's ``resonances`` node → the model. Never raises; reports instead."""
    return _ResonanceReader(resolve, report, readPoPs).read(element, path)
