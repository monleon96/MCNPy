"""GNDS §14 ``reactionSuite`` → the model.

The other root of a GNDS evaluation. :mod:`kika.gnds.covariances` reads the one
that carries matrices; this one reads the one that carries cross sections,
products and distributions, and it is the file an ``externalFile`` in the
covariance suite points back at.

**Scope, and how it was chosen.** Every node name and every attribute below was
counted across all 558 neutron evaluations of ENDF/B-VIII.1-GNDS before a line
of it was written, so "the reader handles what occurs" is a claim about the
data. What the census settled:

=========================  ====================================================
``crossSection`` forms     ``XYs1d`` 27 929, ``reference`` 3 714,
                           ``regions1d`` 3 620,
                           ``resonancesWithBackground`` 1 613 — **and nothing
                           else.** The four forms kika models are the four the
                           library uses; ``gridded1d``, ``Ys1d``,
                           ``CoulombPlusNuclearElastic`` and
                           ``thermalNeutronScatteringLaw1d`` occur zero times.
``Q``                      ``constant1d`` 56 846, ``XYs1d`` zero. Every Q in
                           the library is a number, which is what the model's
                           ``Q.value`` can hold.
``distribution``           ``uncorrelated`` 126 501, ``angularTwoBody`` 45 080,
                           ``unspecified`` 31 623, ``branching3d`` 14 032,
                           ``KalbachMann`` 3 730, ``energyAngular`` 2 948,
                           ``angularEnergy`` 2. Two of the seven are read; the
                           rest are §18 laws phase 7b fills, and each one met
                           is named in the report with its xPath.
``angularTwoBody``         ``recoil`` 22 540, ``XYs2d`` 21 649,
                           ``isotropic2d`` 780, ``regions2d`` 111.
``styles``                 ``evaluated`` 558, ``crossSectionReconstructed``
                           482. No ``heated``, no ``averageProductData``.
=========================  ====================================================

So this reader is complete for cross sections, Q values, multiplicities and
two-body angular distributions, and deliberately partial for the energy and
energy-angle laws. **Partial is stated, never implied**: an unread node becomes
a ``report.unsupportedNode`` entry carrying the node name *and its xPath*, so
"what did this decode skip" is answerable from the returned object rather than
from reading this module.

**Why a class where :mod:`kika.gnds.covariances` uses free functions.** A
``covarianceSection`` is three levels deep and threading ``(resolve, report)``
through three frames is honest. A ``reactionSuite`` is seven — suite, reaction,
outputChannel, product, distribution, form, function — and the same threading
would put two parameters that never change on every signature, plus the xPath
that does. The resolver, the report and the path builder are the reader's state,
so they live on it.

**PoPs is read minimally and says so.** §12 is its own database and its own
project; ``ReactionSuite`` requires the node, and what phase 3 needs from it is
ids, masses, spins, charges and halflives. Aliases, decay data and nuclear level
energies are dropped, and the report carries one aggregated entry saying how
many of each — one entry rather than 34 375, because a report nobody can read
declares nothing.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model import (Background,
                                     ConversionReport, CrossSection,
                                     CrossSectionSum,
                                     ExternalFile, ExternalFiles,
                                     FissionComponents, Frame, GndsProvenance,
                                     Multiplicity, Nuclide, OrphanProducts,
                                     OutputChannel, Particle, PhysicalQuantity,
                                     PoPs, Product, Productions, Q,
                                     RangeQuantity, Reaction, ReactionId,
                                     ReactionSuite, Reactions, Reference,
                                     ResonancesWithBackground, Style, Styles,
                                     Sums)
from kika.nuclear_data.model.distributions import Distribution
from kika.nuclear_data.model.reactions import IncompleteReactions
from kika.nuclear_data.model.sums import Add, MultiplicitySum, Summands

from .distributions import readDistribution
from .primitives import (UnsupportedNode, readAxes, readForm,
                         readFraction)
from .resonances import readResonances
from .styles import readStyles
from .nodes import reads
from .version import checkFormat
from .xpath import Document, Resolver, readExternalFiles, verifyChecksum

__all__ = ["readReactionSuite"]

#: §14.1.1. The reaction-list children, and the container class for each.
REACTION_LISTS = (
    ("reactions", "reaction", Reactions),
    ("orphanProducts", "orphanProduct", OrphanProducts),
    ("fissionComponents", "fissionComponent", FissionComponents),
    ("productions", "production", Productions),
    ("incompleteReactions", "reaction", IncompleteReactions),
)

#: Children of a functional or container that carry no data kika reads. Skipped
#: without comment; everything else that is skipped is reported.
IGNORED = ("documentation", "axes")


def _quoted(label: str) -> str:
    """One xPath step, with the label predicate GNDS uses to key siblings.

    Labels contain spaces, brackets and plus signs (``Fe57 + photon
    [inclusive]``), which is why :mod:`kika.gnds.xpath`'s scanner is
    quote-aware; this only has to produce what that scanner reads back.
    """
    return f"[@label='{label}']" if label else ""


class _SuiteReader:
    """One read of one ``reactionSuite``. Not reusable, and not meant to be."""

    def __init__(self, document: Document,
                 documents: Optional[Dict[str, Document]],
                 report: ConversionReport) -> None:
        self.document = document
        self.resolver = Resolver(document, documents)
        self.report = report
        #: Counters for the losses that would otherwise be reported once per
        #: node. Emitted as one aggregated entry each at the end of the read.
        self.tallies: Dict[str, int] = {}

    # -- reporting ---------------------------------------------------------

    def resolve(self, href: str, node: ET.Element) -> Optional[ET.Element]:
        return self.resolver.resolve(href, context=node).element

    def unsupported(self, tag: str, path: str, reason: str) -> None:
        self.report.unsupportedNode(f"{path}/{tag}: {reason}")

    def tally(self, key: str) -> None:
        self.tallies[key] = self.tallies.get(key, 0) + 1

    def flushTallies(self) -> None:
        """Turn the counters into one report line each, in a stable order."""
        for key in sorted(self.tallies):
            self.report.lost(f"{self.tallies[key]} x {key}")

    def form(self, element: ET.Element, path: str, owner: str):
        """A functional child → its model object, or ``None`` plus a report line."""
        try:
            return readForm(element, readAxes(element, self.resolve))
        except UnsupportedNode as exc:
            self.unsupported(element.tag, path, f"{exc.args[0]} (in {owner})")
            return None

    # -- the suite ---------------------------------------------------------

    def read(self) -> ReactionSuite:
        root = self.document.root
        suite = ReactionSuite(
            evaluation=root.attrib.get("evaluation", ""),
            projectile=root.attrib.get("projectile", ""),
            target=root.attrib.get("target", ""),
            projectileFrame=Frame(root.attrib.get("projectileFrame", "lab")),
            interaction=root.attrib.get("interaction", "nuclear"),
            format=root.attrib["format"],
        )
        path = "/reactionSuite"

        suite.externalFiles = self.readExternalFiles(root)
        styles = root.find("styles")
        if styles is not None:
            suite.styles = readStyles(styles, f"{path}/styles",
                                      self.report, self.tally)
        pops = root.find("PoPs")
        if pops is not None:
            suite.PoPs = self.readPoPs(pops)

        resonances = root.find("resonances")
        if resonances is not None:
            suite.resonances = readResonances(
                resonances, path, self.resolve, self.report, self.readPoPs
            )

        for container, childTag, cls in REACTION_LISTS:
            element = root.find(container)
            if element is None:
                continue
            setattr(suite, container, cls(
                self.readReactions(element, childTag, f"{path}/{container}")
            ))

        sums = root.find("sums")
        if sums is not None:
            suite.sums = self.readSums(sums, f"{path}/sums")

        applicationData = root.find("applicationData")
        if applicationData is not None:
            self.report.lost(
                f"{path}/applicationData holds "
                f"{[c.attrib.get('label', c.tag) for c in applicationData]}; "
                f"kika has no typed home for application-specific data and does "
                f"not keep raw XML in the model, so it is dropped rather than "
                f"carried as an opaque blob"
            )

        self.flushTallies()
        return suite

    # -- externalFiles -----------------------------------------------------

    def readExternalFiles(self, root: ET.Element) -> ExternalFiles:
        """§14.1.1, with each declared digest checked where the file is at hand.

        A wrong checksum is a **warning**, not a refusal: the two files still
        parse and their contents are still what they are. What it says is that
        the pair was not written together, which is exactly the state a trimmed
        fixture or a hand-edited covariance file is in, and the user is the one
        who knows whether that is intended.
        """
        out = ExternalFiles()
        for entry in readExternalFiles(root):
            out.files.append(ExternalFile(
                label=entry.label, path=entry.path,
                checksum=entry.checksum, algorithm=entry.algorithm,
            ))
            resolved = entry.resolvedAgainst(self.document.path)
            if resolved is not None and resolved.exists():
                problem = verifyChecksum(entry, resolved)
                if problem:
                    self.report.warn(problem)
        return out

    # -- styles ------------------------------------------------------------

    @staticmethod
    def readPhysicalQuantity(element: Optional[ET.Element]
                             ) -> Optional[PhysicalQuantity]:
        if element is None or "value" not in element.attrib:
            return None
        return PhysicalQuantity(
            value=float(element.attrib["value"]),
            unit=element.attrib.get("unit", ""),
        )

    @staticmethod
    def readRange(element: Optional[ET.Element]) -> Optional[RangeQuantity]:
        if element is None or "min" not in element.attrib:
            return None
        return RangeQuantity(
            min=float(element.attrib["min"]),
            max=float(element.attrib["max"]),
            unit=element.attrib.get("unit", ""),
        )

    # -- PoPs --------------------------------------------------------------

    def readPoPs(self, element: ET.Element) -> PoPs:
        """§12, minimally: the particles, their masses, spins and charges.

        Walked rather than descended, because the path to a nuclide is six
        containers deep (``chemicalElements/chemicalElement/isotopes/isotope/
        nuclides/nuclide``) and every one of those levels exists only to group.
        ``Z`` and ``A`` are the exception — they live on the grouping nodes and
        nowhere else — so those two are carried down.
        """
        pops = PoPs(
            name=element.attrib.get("name"),
            version=element.attrib.get("version"),
        )
        for particle in element.iter("gaugeBoson"):
            pops.add(self.readParticle(particle))
        for particle in element.iter("baryon"):
            pops.add(self.readParticle(particle))
        for chemicalElement in element.iter("chemicalElement"):
            Z = int(chemicalElement.attrib["Z"])
            for isotope in chemicalElement.iter("isotope"):
                A = int(isotope.attrib["A"])
                for nuclide in isotope.iter("nuclide"):
                    pops.add(self.readNuclide(nuclide, Z, A))

        for dropped, name in (
            (len(element.findall("aliases/alias")), "PoPs <alias>"),
            (len(element.findall("aliases/metaStable")), "PoPs <metaStable>"),
            (len(list(element.iter("decayData"))), "PoPs <decayData>"),
        ):
            for _ in range(dropped):
                self.tally(f"{name}: outside kika's minimal §12 particle model")
        return pops

    def readParticle(self, element: ET.Element) -> Particle:
        return Particle(
            id=element.attrib["id"],
            mass=self.readPhysicalQuantity(element.find("mass/double")),
            spin=self.readSpin(element.find("spin/fraction")),
            parity=self.readInteger(element.find("parity/integer")),
            charge=self.readInteger(element.find("charge/integer")),
            halflife=self.readHalflife(element.find("halflife")),
        )

    def readHalflife(self, element: Optional[ET.Element]):
        """§12's two spellings: a ``<double>`` in seconds, or ``<string>`` "stable"."""
        if element is None:
            return None
        text = element.find("string")
        if text is not None:
            return text.attrib.get("value")
        return self.readPhysicalQuantity(element.find("double"))

    def readNuclide(self, element: ET.Element, Z: int, A: int) -> Nuclide:
        """One ``nuclide``, with its ``nucleus``'s spin and parity folded in.

        GNDS separates the atom from its nucleus; kika's minimal PoPs has one
        particle. The two carry different things — the mass and the net charge
        are the atom's, the spin, the parity and the level index are the
        nucleus's — so folding loses nothing here, and the level *energy*, which
        has no slot, is counted in the tallies.
        """
        nucleus = element.find("nucleus")
        nucleus = nucleus if nucleus is not None else ET.Element("nucleus")
        energy = nucleus.find("energy/double")
        # A ground state's energy is 0 and `nuclearLevel=0` reproduces it
        # exactly, so only an excited level is a loss. Counting the ground
        # states too would put 171 entries in Fe-56's report for 3 real ones.
        if energy is not None and float(energy.attrib.get("value", 0)) != 0.0:
            self.tally(
                "PoPs <nucleus><energy>: the excitation energy of an excited "
                "level, for which the model carries only the level index"
            )
        return Nuclide(
            id=element.attrib["id"],
            mass=self.readPhysicalQuantity(element.find("mass/double")),
            spin=self.readSpin(nucleus.find("spin/fraction")),
            parity=self.readInteger(nucleus.find("parity/integer")),
            charge=self.readInteger(element.find("charge/integer")),
            halflife=self.readHalflife(nucleus.find("halflife")),
            Z=Z, A=A,
            nuclearLevel=int(nucleus.attrib.get("index", 0)),
        )

    @staticmethod
    def readSpin(element: Optional[ET.Element]) -> Optional[PhysicalQuantity]:
        if element is None or "value" not in element.attrib:
            return None
        return PhysicalQuantity(
            value=readFraction(element.attrib["value"]),
            unit=element.attrib.get("unit", ""),
        )

    @staticmethod
    def readInteger(element: Optional[ET.Element]) -> Optional[int]:
        if element is None or "value" not in element.attrib:
            return None
        return int(element.attrib["value"])

    # -- reactions ---------------------------------------------------------

    def readReactions(self, element: ET.Element, childTag: str,
                      path: str) -> List[Reaction]:
        out = []
        for child in element:
            if child.tag != childTag:
                self.unsupported(
                    child.tag, path,
                    f"a <{element.tag}> holds <{childTag}> children"
                )
                continue
            out.append(self.readReaction(child, path))
        return out

    def readReaction(self, element: ET.Element, path: str) -> Reaction:
        label = element.attrib.get("label", "")
        here = f"{path}/{element.tag}{_quoted(label)}"
        channel = element.find("outputChannel")
        outputChannel = (
            OutputChannel() if channel is None
            else self.readOutputChannel(channel, here)
        )
        crossSection = element.find("crossSection")

        if element.find("doubleDifferentialCrossSection") is not None:
            self.unsupported(
                "doubleDifferentialCrossSection", here,
                "the model declares the slot and phase 7b fills it"
            )
        return Reaction(
            id=ReactionId(
                label=label,
                products=tuple(p.pid for p in outputChannel.products),
                ENDF_MT=(int(element.attrib["ENDF_MT"])
                         if "ENDF_MT" in element.attrib else None),
                fissionGenre=element.attrib.get("fissionGenre"),
            ),
            crossSection=(CrossSection() if crossSection is None
                          else self.readCrossSection(crossSection, here)),
            outputChannel=outputChannel,
        )

    # -- crossSection ------------------------------------------------------

    @reads("crossSectionForm", "resonancesWithBackground", "reference")
    def readCrossSection(self, element: ET.Element, path: str) -> CrossSection:
        """§16.1.1: a mapping from style label to form, which is what it is."""
        here = f"{path}/crossSection"
        crossSection = CrossSection()
        for child in element:
            if child.tag in IGNORED:
                continue
            label = child.attrib.get("label", "")
            if child.tag == "resonancesWithBackground":
                form = self.readResonancesWithBackground(child, here)
            elif child.tag == "reference":
                form = Reference(href=child.attrib.get("href", ""), label=label)
            else:
                form = self.form(child, here, "crossSection")
                # §7 hangs the uncertainty on the *form*, not on the container,
                # so this is asked of each child rather than once of the parent.
                self.noteUncertainty(child, here, child.tag)
            if form is not None:
                crossSection[label] = form
        return crossSection

    def readResonancesWithBackground(self, element: ET.Element,
                                     path: str) -> ResonancesWithBackground:
        label = element.attrib.get("label", "")
        here = f"{path}/resonancesWithBackground{_quoted(label)}"
        resonances = element.find("resonances")
        background = element.find("background")
        self.noteUncertainty(element, here, "resonancesWithBackground")
        return ResonancesWithBackground(
            background=(None if background is None
                        else self.readBackground(background, here)),
            resonanceRegionHref=(None if resonances is None
                                 else resonances.attrib.get("href")),
            label=label,
        )

    def readBackground(self, element: ET.Element, path: str) -> Background:
        """§16.1.1's three regions, each holding one ``XYs1d`` or ``regions1d``.

        A region the file does not carry stays ``None``. That is not the same
        statement as a region carrying a zero background — 1 541 of the
        library's 1 613 backgrounds have a ``resolvedRegion`` and all 1 613 have
        a ``fastRegion``, so absence here is information.
        """
        here = f"{path}/background"
        background = Background()
        for child in element:
            if child.tag not in background.regions:
                if child.tag != "uncertainty":
                    self.unsupported(child.tag, here, "not a background region")
                continue
            functions = [c for c in child if c.tag not in IGNORED]
            if not functions:
                self.unsupported(child.tag, here, "the region holds no function")
                continue
            form = self.form(functions[0], f"{here}/{child.tag}", "background")
            setattr(background, child.tag, form)
        return background

    def noteUncertainty(self, element: ET.Element, path: str,
                        owner: str) -> None:
        """§7's ``uncertainty`` on a form: counted, and explained once.

        It is a **back-link**, not data: ``<uncertainty><covariance href=...>``
        points at the covarianceSuite section that is about this form. Nothing
        in the model holds a pointer in that direction, and it is not needed —
        §25.2.3's ``rowData/href`` states the same relation from the other end,
        and that is the end :mod:`kika.gnds.covariances` reads.
        """
        if element.find("uncertainty") is None:
            return
        self.tally(
            f"<{owner}><uncertainty>: the back-link from a form to its "
            f"covariance, which the covarianceSuite states in the other "
            f"direction through rowData/href"
        )

    # -- outputChannel -----------------------------------------------------

    def readOutputChannel(self, element: ET.Element, path: str) -> OutputChannel:
        here = f"{path}/outputChannel"
        channel = OutputChannel(
            genre=element.attrib.get("genre"),
            process=element.attrib.get("process"),
        )
        q = element.find("Q")
        if q is not None:
            channel.Q = self.readQ(q, here)
        products = element.find("products")
        for child in (products if products is not None else []):
            if child.tag != "product":
                self.unsupported(child.tag, f"{here}/products", "not a product")
                continue
            channel.products.products.append(self.readProduct(child, here))
        if element.find("fissionFragmentData") is not None:
            self.unsupported(
                "fissionFragmentData", here,
                "delayed neutrons and fission energy release are read from ENDF "
                "MF1/455 and MF1/458 today; the GNDS side is phase 7b"
            )
        return channel

    def readQ(self, element: ET.Element, path: str) -> Q:
        """§17.1.1. Every Q in the library is a ``constant1d``, so Q is a number.

        Its unit is on the ``axes``, not on the node — §2.3.3's rule that a unit
        lives with an axis or with a physicalQuantity and never on an array —
        so it is read from axis 0 rather than assumed to be eV.
        """
        here = f"{path}/Q"
        forms = [c for c in element if c.tag not in IGNORED]
        chosen = self.pickOneForm(forms, here, "Q")
        if chosen is None or chosen.tag != "constant1d":
            if chosen is not None:
                self.unsupported(
                    chosen.tag, here,
                    "kika's Q holds a number; an energy-dependent Q occurs "
                    "nowhere in ENDF/B-VIII.1-GNDS and has no model slot"
                )
            return Q()
        axes = readAxes(chosen, self.resolve)
        unit = ""
        if axes is not None:
            for axis in axes:
                if axis.index == 0:
                    unit = axis.unit
        return Q(
            value=float(chosen.attrib["value"]),
            unit=unit,
            label=chosen.attrib.get("label"),
        )

    def pickOneForm(self, forms: List[ET.Element], path: str,
                    owner: str) -> Optional[ET.Element]:
        """The ``eval`` form of a container the model keeps unlabelled.

        ``crossSection`` and ``distribution`` are dictionaries keyed by style
        label, because §9.1 says a container holds one form per style. ``Q`` and
        ``multiplicity`` are not, in kika's model — they hold one value — so
        when a file gives two, one has to be chosen and the other said out loud.
        """
        if not forms:
            return None
        if len(forms) == 1:
            return forms[0]
        labelled = {c.attrib.get("label", ""): c for c in forms}
        chosen = labelled.get("eval", forms[0])
        self.report.lost(
            f"{path}: this <{owner}> carries "
            f"{sorted(labelled)} and kika's model holds one; "
            f"{chosen.attrib.get('label', '')!r} was kept"
        )
        return chosen

    # -- products ----------------------------------------------------------

    def readProduct(self, element: ET.Element, path: str) -> Product:
        label = element.attrib.get("label", "")
        here = f"{path}/products/product{_quoted(label)}"
        product = Product(pid=element.attrib["pid"], label=label)

        multiplicity = element.find("multiplicity")
        if multiplicity is not None:
            product.multiplicity = self.readMultiplicity(multiplicity, here)
        distribution = element.find("distribution")
        if distribution is not None:
            product.distribution = readDistribution(
                distribution, here, self.resolve, self.report)
        channel = element.find("outputChannel")
        if channel is not None:
            # §17.2.1's recursion: a product that breaks up or decays carries
            # its own channel. The model has the slot; nothing else is special.
            product.outputChannel = self.readOutputChannel(channel, here)
        if element.find("averageProductEnergy") is not None:
            self.unsupported(
                "averageProductEnergy", here,
                "a processed quantity; kika reads evaluated data"
            )
        return product

    @reads("multiplicityForm", "constant1d", "reference", "unspecified", "branching1d")
    def readMultiplicity(self, element: ET.Element, path: str) -> Multiplicity:
        here = f"{path}/multiplicity"
        forms = [c for c in element if c.tag not in IGNORED]
        chosen = self.pickOneForm(forms, here, "multiplicity")
        if chosen is None:
            return Multiplicity()
        label = chosen.attrib.get("label")

        if chosen.tag == "constant1d":
            return Multiplicity(constant=float(chosen.attrib["value"]),
                                label=label)
        if chosen.tag in ("reference", "unspecified", "branching1d"):
            self.unsupported(
                chosen.tag, here,
                "kika's multiplicity is a constant or a function of E; a link, "
                "an absence and an isomeric branching each need a node the "
                "model does not have (phase 7b)"
            )
            return Multiplicity(label=label)
        return Multiplicity(function=self.form(chosen, here, "multiplicity"),
                            label=label)

    # -- sums --------------------------------------------------------------

    def readSums(self, element: ET.Element, path: str) -> Sums:
        """§21.1: ``crossSectionSums`` become the reaction list, as they always
        have; ``multiplicitySums`` are the second child and stay separate."""
        sums = Sums()
        container = element.find("crossSectionSums")
        for child in (container if container is not None else []):
            if child.tag != "crossSectionSum":
                self.unsupported(child.tag, f"{path}/crossSectionSums",
                                 "not a crossSectionSum")
                continue
            sums.append(self.readCrossSectionSum(
                child, f"{path}/crossSectionSums"
            ))
        container = element.find("multiplicitySums")
        for child in (container if container is not None else []):
            if child.tag != "multiplicitySum":
                self.unsupported(child.tag, f"{path}/multiplicitySums",
                                 "not a multiplicitySum")
                continue
            sums.multiplicitySums.append(self.readMultiplicitySum(
                child, f"{path}/multiplicitySums"
            ))
        return sums

    def readCrossSectionSum(self, element: ET.Element,
                            path: str) -> CrossSectionSum:
        label = element.attrib.get("label", "")
        here = f"{path}/crossSectionSum{_quoted(label)}"
        crossSection = element.find("crossSection")
        q = element.find("Q")
        return CrossSectionSum(
            id=ReactionId(
                label=label,
                ENDF_MT=(int(element.attrib["ENDF_MT"])
                         if "ENDF_MT" in element.attrib else None),
            ),
            crossSection=(CrossSection() if crossSection is None
                          else self.readCrossSection(crossSection, here)),
            outputChannel=OutputChannel(
                Q=Q() if q is None else self.readQ(q, here)
            ),
            summands=self.readSummands(element, here),
        )

    def readMultiplicitySum(self, element: ET.Element,
                            path: str) -> MultiplicitySum:
        label = element.attrib.get("label", "")
        here = f"{path}/multiplicitySum{_quoted(label)}"
        multiplicity = element.find("multiplicity")
        return MultiplicitySum(
            label=label,
            multiplicity=(None if multiplicity is None
                          else self.readMultiplicity(multiplicity, here)),
            summands=self.readSummands(element, here),
            ENDF_MT=(int(element.attrib["ENDF_MT"])
                     if "ENDF_MT" in element.attrib else None),
        )

    def readSummands(self, element: ET.Element, path: str) -> Summands:
        """§21.3. ``add`` is modelled; ``subtract`` is admitted by the schema and
        occurs nowhere in the library, so meeting one is reported."""
        summands = Summands()
        container = element.find("summands")
        for child in (container if container is not None else []):
            if child.tag == "add":
                summands.append(Add(href=child.attrib.get("href", "")))
            else:
                self.unsupported(
                    child.tag, f"{path}/summands",
                    "§21.3 admits <subtract> and no distributed neutron "
                    "evaluation uses one; kika's Summands holds additions"
                )
        return summands


def readReactionSuite(
    document: Document,
    documents: Optional[Dict[str, Document]] = None,
    report: Optional[ConversionReport] = None,
) -> Tuple[ReactionSuite, ConversionReport]:
    """§14.1.1 ``reactionSuite`` → the model, plus what the read could not do.

    Parameters
    ----------
    document
        The parsed evaluation.
    documents
        ``externalFile`` label → parsed :class:`~kika.gnds.xpath.Document`, for
        following ``$covariances#`` links. Omitting it is normal — a
        ``reactionSuite`` is self-contained apart from its covariances, and the
        covariance link is read from the covariance file's end anyway.

    Returns
    -------
    (ReactionSuite, ConversionReport)
        The report is **not** optional reading. This decode is partial by
        design: the §18 energy and energy-angle laws are phase 7b, and a file
        whose products carry ``uncorrelated`` distributions comes back with
        those products present, their multiplicities read, and no distribution.
        The report names each one with its xPath.
    """
    report = report if report is not None else ConversionReport()
    root = document.root
    if root.tag != "reactionSuite":
        raise ValueError(
            f"{document.name} is a <{root.tag}>, not a <reactionSuite>"
        )
    checkFormat(root.attrib.get("format"), document.name)

    suite = _SuiteReader(document, documents, report).read()
    # The positive statement that this came from GNDS, which `kika.write` needs
    # to decide the version to declare. See `GndsProvenance`.
    suite.provenance = GndsProvenance(
        formatVersion=root.attrib.get("format"),
        path=None if document.path is None else str(document.path),
    )
    suite.report = report
    return suite, report
