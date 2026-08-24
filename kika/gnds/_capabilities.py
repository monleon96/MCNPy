"""What of GNDS kika reads, what it writes, and what it does not touch.

**The question this answers, which nothing else could.** ``kika/gnds/nodes.py``
compares the reader against the writer and is an invariant of the test suite;
``ConversionReport`` says what *one file* lost. Neither answers "what does this
library support", and until this module there was no way for a user to ask.
Saying "we read the ENDF/B-VIII.1 evaluations we care about" and saying "we
implement GNDS 2.1" are different claims, and only the first was ever true.

**Where the left-hand column comes from, and why not from `NODES`.** From the
*schema*. ``kika/gnds/nodes.py`` declares 58 forms at 12 choice points, by a
criterion its own docstring defends, and it **deliberately omits** the members
a census counted zero times — five of §18.1.1's twelve among them. A profile
built on it would not say ``thermalNeutronScatteringLaw`` is unsupported; it
would say *nothing at all* about it, which is the one failure a support profile
may not have. So the column is every ``xs:element`` FUDGE's ``gnds.xsd`` and
``covariances.xsd`` declare — **300 of them** — extracted by
``tests/data/build_schema_census.py`` and pinned in
``tests/data/gnds_schema_nodes.json``. The schema says what *exists*; the
census of the 558 distributed evaluations says what is *exercised*. They are
different columns and this one is the first.

**Twelve more rows that the schema does not declare.** Cross the 300 against
``NODES`` and twelve of the names kika itself uses are not ``xs:element``\\ s of
that schema at all — eight styles, and four functional forms. The schema is
therefore *not* a superset of what kika writes, and a table keyed only on it
would drop, in silence, the nodes where kika is most likely to produce a file
somebody else's reader rejects. They are here, marked
:data:`NOT_IN_SCHEMA` and rendered apart, because that is the single most
useful thing this module can tell a user who is about to hand a kika-written
file to FUDGE.

**Which schema.** FUDGE 6.10.0 ships the **2.0** schema — its own header's
newest entry is *"version 2.0, November 2021"* — while
:mod:`kika.gnds.version` accepts 2.0 and 2.1 through one path, because for
everything kika models they are the same format. A node absent from the column
is absent from *that* file; where a name is a 2.1 addition the reason says so
rather than implying the format never had it.

**The three words, and the one rule that makes all 300 decidable.**

``full``
    kika reads the node into the model and writes it back from the model. A
    round trip reproduces it.
``partial``
    it survives in one direction only, or at some of the places the schema
    admits it and not others, or the node survives while its content does not.
``unsupported``
    no part of it reaches the model.

Whether kika *says so* is :attr:`Capability.reportedVia`, not the status. That
is deliberate: folding "but it is reported" into ``partial`` would put a node
kika merely names in the same word as ``reference``, which genuinely round
trips at two of its three choice points. Reporting is a second axis and is
carried as one.

    A node is judged at each place the schema admits it **whose parent this
    table marks full or partial**. A context under an unsupported parent does
    not count.

That rule is what keeps ``XYs1d`` ``full`` although §26's coherent-photon
``formFactor`` holds one kika never reads, and it is what still makes
``styles`` ``partial`` — because ``PoPs`` *is* read, and ``PoPs/styles``
(``gnds.xsd:482``) is skipped. Without it every leaf in the schema would be
dragged down by the worst place it appears and the table would say nothing.

**Why this module imports nothing.** ``nodes.py`` reaches ~50 model classes at
module scope, so importing it here would make ``import kika.gnds`` wake
``kika.nuclear_data.model`` for the cluster pipeline, the desktop app and every
notebook —  the thing
``kika/nuclear_data/model/tests/test_dormancy.py`` exists to prevent.
Measured: ``import kika.gnds`` loads zero model modules, ``import
kika.gnds.nodes`` loads thirty. So the table is **static**, the statuses are
**not derived** from ``NODES``, and the join between the two lives in
``tests/test_capabilities.py``, which may import both.

That is also the argument ``nodes.py:37-45`` already makes for itself: a table
derived from the code makes "the table agrees with the code" true by
construction and asserts nothing. Here the derivation would additionally be
one-way — ``NODES`` **bounds** a capability and does not fix it. ``reference``
is ``PAIRED`` twice and is ``partial``, because §18.1.1 admits it a third time
and kika has no entry there.

**And the leading underscore is load-bearing.** ``importlib`` binds a submodule
onto its package as a side effect of importing it, so a module named
``kika/gnds/capabilities.py`` would overwrite the exported function with itself
on first access — first call returns the function, every later one returns the
module. ``kika/__init__.py:40-43`` documents the same trap and solves it the
same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Optional, Tuple

__all__ = ["Coverage", "Capability", "Capabilities", "capabilities",
           "CAPABILITIES", "NOT_IN_SCHEMA", "SCHEMA_NODE_COUNT"]


class Coverage(str, Enum):
    """What happens to the data in a node. See the module docstring."""

    #: Read into the model and written back from it.
    FULL = "full"
    #: One direction only, or some contexts only, or the node without its
    #: content.
    PARTIAL = "partial"
    #: No part of it reaches the model. Say so in ``why``; whether kika says so
    #: at run time is ``reportedVia``.
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Capability:
    """One GNDS node, and what kika does with it."""

    #: The ``xs:element`` name, verbatim. Unique across the table, and the key.
    node: str
    #: Which schema declares it: ``"gnds.xsd"``, ``"covariances.xsd"``,
    #: ``"gnds.xsd + covariances.xsd"`` for the thirteen both do, or
    #: ``"not in gnds.xsd"`` for the twelve names kika uses and FUDGE 6.10.0's
    #: schema does not have. The *line* is in the committed census under
    #: ``tests/data/`` rather than here, so three hundred magic numbers cannot
    #: rot in this file; the ``.xsd:line`` a reader wants is in ``why``, where
    #: the rule already forces a citation.
    where: str
    coverage: Coverage
    #: Mandatory, including on ``FULL``. Must cite a ``§``, a ``.py:line`` or a
    #: ``.xsd:line`` — a reason that cites nothing is how a row becomes an
    #: excuse, which is the rule ``test_nodes.py:64`` already enforces next
    #: door.
    why: str
    #: The group the row was written with, for filtering and for rendering. One
    #: honest sentence often covers a whole subtree; the group is what says
    #: which sentence, and it is never a substitute for the node being listed.
    group: str
    #: The node whose ``ConversionReport`` line names this loss at run time, or
    #: ``None`` where nothing does. ``None`` on a non-``FULL`` row is a **silent
    #: drop**, and the count of those is pinned by a test so each repair has to
    #: come through it.
    reportedVia: Optional[str] = None
    #: The ``nodes.py`` families that also declare this tag, when any do. The
    #: bridge test joins on this; the module never reads it.
    families: Tuple[str, ...] = ()
    #: Why the coverage sits *below* what ``families`` alone would allow. Only
    #: a row with one may be worse than its ``NODES`` ceiling, and it must cite
    #: something like ``why`` does.
    caveat: Optional[str] = None


#: The eighteen nodes only ``covariances.xsd`` declares, and the thirteen both
#: files do. Kept as data because a row has to say *which* schema declares it
#: and this module may not read the census file — that lives under ``tests/``
#: and does not ship in the wheel. §25.1.1 is why there are two schemas at all:
#: ``covarianceSuite`` has no global declaration in ``gnds.xsd``, it is a root
#: in its own right.
_COVARIANCES_ONLY = frozenset("""
    averageParameterCovariance columnData covarianceMatrix covarianceSection
    covarianceSections covarianceSuite mixed parameterCovariance
    parameterCovarianceMatrix parameterCovariances parameterLink parameters
    rowData shortRangeSelfScalingVariance slice slices sum summand
""".split())

_IN_BOTH = frozenset("""
    array axes axis evaluated externalFile externalFiles grid gridded2d link
    projectileEnergyDomain styles temperature values
""".split())

#: ``tag -> the nodes.py families that also declare it``, hand-copied on
#: purpose. Deriving it would need ``import kika.gnds.nodes``, which wakes the
#: model; and a derived copy would make the bridge test true by construction,
#: which is the failure ``nodes.py:37-45`` argues against for its own table.
#: ``test_capabilities.py`` asserts this is set-equal to what ``NODES``
#: declares, in both directions, so the duplication is guarded rather than
#: trusted.
_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "CoulombPlusNuclearElastic": ("crossSectionForm",),
    "KalbachMann": ("distributionForm",),
    "Legendre": ("function1d",),
    "MadlandNix": ("uncorrelatedEnergyForm",),
    "NBodyPhaseSpace": ("uncorrelatedEnergyForm",),
    "Watt": ("uncorrelatedEnergyForm",),
    "XYs1d": ("function1d",),
    "XYs2d": ("function2d",),
    "XYs3d": ("function3d",),
    "angularEnergy": ("distributionForm",),
    "angularTwoBody": ("distributionForm",),
    "averageParameterCovariance": ("parameterCovarianceForm",),
    "averageProductData": ("style",),
    "branching1d": ("multiplicityForm",),
    "branching3d": ("distributionForm",),
    "constant1d": ("function1d", "multiplicityForm"),
    "covarianceMatrix": ("covarianceForm",),
    "crossSectionReconstructed": ("style",),
    "discreteGamma": ("uncorrelatedEnergyForm",),
    "energyAngular": ("distributionForm",),
    "evaluated": ("style",),
    "evaporation": ("uncorrelatedEnergyForm",),
    "forward": ("uncorrelatedAngularForm",),
    "generalEvaporation": ("uncorrelatedEnergyForm",),
    "heated": ("style",),
    "isotropic2d": ("angularTwoBodyForm", "uncorrelatedAngularForm"),
    "mixed": ("covarianceForm",),
    "parameterCovariance": ("parameterCovarianceForm",),
    "polynomial1d": ("function1d",),
    "primaryGamma": ("uncorrelatedEnergyForm",),
    "recoil": ("angularTwoBodyForm",),
    "reference": ("crossSectionForm", "multiplicityForm"),
    "regions1d": ("function1d",),
    "regions2d": ("function2d",),
    "resonancesWithBackground": ("crossSectionForm",),
    "shortRangeSelfScalingVariance": ("covarianceForm",),
    "simpleMaxwellianFission": ("uncorrelatedEnergyForm",),
    "sum": ("covarianceForm",),
    "thermalNeutronScatteringLaw1d": ("crossSectionForm",),
    "uncorrelated": ("distributionForm",),
    "unspecified": ("distributionForm", "multiplicityForm"),
    "weightedFunctionals": ("uncorrelatedEnergyForm",),
}


def _where(node: str) -> str:
    if node in _COVARIANCES_ONLY:
        return "covariances.xsd"
    if node in _IN_BOTH:
        return "gnds.xsd + covariances.xsd"
    return "gnds.xsd"


_TABLE: Dict[str, Capability] = {}


def _one(node, coverage, reportedVia, group, why, caveat=None, where=None):
    """One row. Duplicate keys raise rather than overwriting — the table is
    written as groups and a node named twice is a partition bug, not a later
    entry winning."""
    if node in _TABLE:
        raise AssertionError(
            f"<{node}> is declared twice, as {_TABLE[node].group} and as "
            f"{group}. The groups partition the schema; overlapping them "
            f"would let one honest sentence quietly replace another"
        )
    _TABLE[node] = Capability(
        node=node, where=where or _where(node), coverage=coverage, why=why,
        group=group, reportedVia=reportedVia,
        families=_FAMILIES.get(node, ()), caveat=caveat,
    )


def _group(group, coverage, reportedVia, why, nodes):
    """A whole subtree that one sentence covers honestly. Every member still
    gets its own row — the group is what says *which* sentence, never a reason
    to leave a node off the table."""
    for node in nodes.split():
        _one(node, coverage, reportedVia, group, why)
# -- full: the nodes a round trip reproduces ------------------------------

_group("suiteSkeleton", Coverage.FULL, None,
       "§14.1.1's fixed grammar. The five reaction lists are one table "
       "on each side -- decode.py:96-104 and the writer's mirror of it "
       "-- so a container cannot be read without being written; "
       "externalFile round trips with its checksum verified rather than "
       "copied",
       """
       reactionSuite externalFiles externalFile reactions reaction
       orphanProducts orphanProduct fissionComponents fissionComponent
       productions production incompleteReactions
       """)

_group("styles", Coverage.FULL, None,
       "§9.2.1 and §9.3.1, the two styles the distributed libraries use: "
       "evaluated in all 558 and crossSectionReconstructed in 482. "
       "kika/gnds/styles.py:29-32 is the tag table, and temperature and "
       "projectileEnergyDomain are the two children RS_EvaluatedType "
       "(gnds.xsd:96-108) makes mandatory",
       """
       evaluated crossSectionReconstructed temperature
       projectileEnergyDomain
       """)

_group("pops", Coverage.FULL, None,
       "§12, in kika's minimal particle model: decode.py reads "
       "chemicalElement -> isotope -> nuclide and folds the atom's mass "
       "and charge and the nucleus's spin and parity into one particle. "
       "gnds.xsd:479 is the root",
       """
       PoPs gaugeBosons gaugeBoson baryons baryon chemicalElements
       chemicalElement isotopes isotope nuclides nuclide nucleus mass
       spin parity charge halflife double integer fraction string
       """)

_group("resonances", Coverage.FULL, None,
       "§19, and its grammar is xs:sequence throughout -- a missing node "
       "moves every resonance and test_resonance_oracle sees it, which "
       "is why nodes.py:20 keeps §19 out of the choice-point table "
       "entirely. Read by kika/gnds/resonances.py, written by "
       "kika/gnds/encode_resonances.py",
       """
       resonances resolved unresolved scatteringRadius hardSphereRadius
       BreitWigner RMatrix resonanceReactions resonanceReaction
       spinGroups spinGroup channels channel externalRMatrix
       resonanceParameters table columnHeaders column data link
       tabulatedWidths Ls L Js J levelSpacing widths width
       """)

_group("crossSection", Coverage.FULL, None,
       "§16.1.1. The three background regions carry no string literal on "
       "either side because both key off one dict -- Background.regions "
       "in kika/nuclear_data/model/cross_section_forms.py:81-89 -- which "
       "the reader tests membership against and the writer uses as the "
       "tag",
       """
       crossSection resonancesWithBackground background resolvedRegion
       unresolvedRegion fastRegion
       """)

_group("sums", Coverage.FULL, None,
       "§21.1 and §21.3's summed cross sections and multiplicities. "
       "`subtract` is the one member of §21.3's choice kika does not "
       "model and has its own row",
       """
       sums crossSectionSums crossSectionSum multiplicitySums
       multiplicitySum summands add
       """)

_group("outputChannel", Coverage.FULL, None,
       "§17.1-17.2, including §17.2.1's recursion where a product "
       "carries its own outputChannel. The identically named nodes under "
       "§12's decayData sit under an unsupported parent and are not "
       "counted -- see the parent rule above",
       """
       outputChannel products product multiplicity
       """)

_group("distributionForms", Coverage.FULL, None,
       "§18.1-18.6. Every law the 558-evaluation census found: all seven "
       "of §18.1.1's occurring members and the sub-forms of §18.2, §18.3 "
       "and §18.6. These are nodes.py's PAIRED entries and "
       "kika/gnds/distributions.py is the reader half",
       """
       distribution angularTwoBody uncorrelated angular isotropic2d
       recoil energyAngular angularEnergy KalbachMann f r a branching3d
       branching1d discreteGamma primaryGamma NBodyPhaseSpace
       unspecified
       """)

_group("functionals", Coverage.FULL, None,
       "§5-6, the containers every other chapter's data ends up in. Both "
       "halves dispatch off nodes.readersOf rather than a literal list "
       "(kika/gnds/primitives.py:513-521), so a form that exists is in "
       "the registry by construction",
       """
       XYs1d regions1d constant1d Legendre polynomial1d XYs2d regions2d
       XYs3d function1ds function2ds values axes axis grid array
       gridded2d
       """)

_group("covariances", Coverage.FULL, None,
       "§25.1-25.3 -- **every node covariances.xsd declares**, all 31 of "
       "them. It is the one chapter of GNDS kika covers completely, and "
       "covariances.xsd:29 onwards is the whole of it",
       """
       covarianceSuite covarianceSections covarianceSection rowData
       columnData covarianceMatrix mixed sum summand
       shortRangeSelfScalingVariance slices slice parameterCovariances
       parameterCovariance parameterCovarianceMatrix parameters
       parameterLink averageParameterCovariance
       """)

# -- partial: one direction, some contexts, or the node without
#    its content ------------------------------------------------------

_one("styles", Coverage.PARTIAL, None, "stylesContext",
     "read and written under reactionSuite and under covarianceSuite; "
     "§12's PoPs/styles (gnds.xsd:482) is neither. It is minOccurs=0, "
     "so the file kika writes stays valid -- this is a read loss and "
     "not an invalid write")

_one("heated", Coverage.PARTIAL, None, "aceOnlyStyles",
     "write only: **built live** by "
     "kika/ace/model_adapter/decode.py:141 and emitted, while "
     "kika/gnds/styles.py reports it on read rather than guessing at "
     "what it means. And what is emitted is invalid -- RS_HeatedType "
     "(gnds.xsd:120-124) makes <temperature> mandatory and the writer "
     "attaches one only to an Evaluated")

_one("averageProductData", Coverage.PARTIAL, None, "aceOnlyStyles",
     # Self-standing on purpose. The render groups by (group, why), so this
     # entry and `heated` come out as two blocks in an order neither of them
     # chooses -- and "as `heated`" alone reads as a dangling reference when
     # this block lands first, which it does.
     "write only, and on the same terms as the `heated` style it accompanies: "
     "RS_AverageProductDataType (gnds.xsd:130-134) makes <temperature> "
     "mandatory too, so the same write is invalid")

_one("documentation", Coverage.PARTIAL, 'documentation', "documentation",
     "the node survives and its text does not. Skipped on read "
     "(decode.py:106 lists it in IGNORED), counted once per style, and "
     "written back **empty** because RS_EvaluatedType "
     "(gnds.xsd:96-108) requires the element to be there")

_one("reference", Coverage.PARTIAL, None, "referenceForm",
     "round trips as a §16.1.1 crossSection form and as a §17.3 "
     "multiplicity form. §18.1.1 (gnds.xsd:1647-1662) admits it a "
     "third time as a distribution form and kika has no entry there -- "
     "zero occurrences across the 558 distributed neutron evaluations, "
     "the reason nodes.py:149-156 gives for naming seven of the twelve",
     caveat="both nodes.py families are PAIRED, so the registry alone would "
            "put this row at full. The third choice point — gnds.xsd:1654, "
            "inside §18.1.1's xs:choice — has no entry, and a node is judged "
            "at every place the schema admits it")

_one("Q", Coverage.PARTIAL, 'Q', "qValue",
     "the node round trips. §17.1.1's choice also admits an "
     "energy-dependent XYs1d Q, which the model's scalar Q.value "
     "cannot hold and which the reader reports instead: constant1d 56 "
     "846 occurrences against XYs1d zero")

_one("energy", Coverage.PARTIAL, 'energy', "levelEnergy",
     "full as §18.3's uncorrelated/energy. As a nucleus's level energy "
     "(gnds.xsd:668) the model carries a level index and not an "
     "energy, so only a non-zero excitation is counted as a loss")

# -- unsupported ----------------------------------------------------------

_group("popsAliases", Coverage.UNSUPPORTED, 'aliases',
       "outside kika's minimal §12 particle model (gnds.xsd:513-518). "
       "The reader counts them into one aggregated report line each "
       "rather than into 34 375",
       """
       aliases alias metaStable
       """)

_group("popsDecay", Coverage.UNSUPPORTED, 'decayData',
       "§12's decay database (gnds.xsd:700-801) is its own project and "
       "kika does not model any of it; every subtree is counted once by "
       "the reader rather than walked",
       """
       decayData decayModes decayMode decayPath decay probability
       spectra spectrum discrete continuum intensity
       internalConversionCoefficients photonEmissionProbabilities
       positronEmissionIntensity internalPairFormationCoefficient shell
       averageEnergies averageEnergy
       """)

_group("documentationContent", Coverage.UNSUPPORTED, 'documentation',
       "free-text provenance (gnds.xsd:191-470) for which the model has "
       "no node. It is counted where it hangs off a style and skipped "
       "elsewhere (decode.py:106). `endfCompatible` is the verbatim "
       "ENDF-6 header and is the one members of this group whose loss is "
       "numerical rather than editorial",
       """
       authors author contributors contributor collaborations
       collaboration affiliations affiliation dates date copyright
       acknowledgements acknowledgement keywords keyword relatedItems
       relatedItem title abstract body computerCodes computerCode
       codeRepo executionArguments inputDecks inputDeck outputDecks
       outputDeck experimentalDataSets exforDataSets exforDataSet
       covarianceScript correctionScript bibliography bibitem
       endfCompatible note
       """)

_group("applicationData", Coverage.UNSUPPORTED, 'applicationData',
       "kika has no typed home for application-specific data and does "
       "not keep raw XML in a format-neutral model, so the reader lists "
       "the children by label and drops them. gnds.xsd:1194-1200, which "
       "leaves the content unspecified on purpose",
       """
       applicationData institution
       """)

_group("energyIntervals", Coverage.UNSUPPORTED, 'energyIntervals',
       "multiple resolved regions (gnds.xsd:835) are deprecated and one "
       "ENDF-VII evaluation uses them. Reported by "
       "kika/gnds/resonances.py's else-branch, which names any child of "
       "<resolved> that is not RMatrix or BreitWigner -- so the name "
       "appears nowhere in kika and the loss is still not silent",
       """
       energyIntervals
       """)

_group("subtract", Coverage.UNSUPPORTED, 'subtract',
       "§21.3 admits it (gnds.xsd:1136) and no distributed neutron "
       "evaluation carries one; the model's Summands holds additions",
       """
       subtract
       """)

_group("analyticSpectra", Coverage.UNSUPPORTED, 'energy',
       "§18.3's analytic spectra (gnds.xsd:1701-1762) and their "
       "parameter children. Each is a formula with named parameters "
       "rather than a table, so kika reports one instead of tabulating "
       "it -- a tabulation would put numbers in the file the evaluator "
       "never wrote. nodes.py:158-193 carries the census count of each, "
       "and four of the six have a witness while Watt and MadlandNix "
       "have none",
       """
       evaporation generalEvaporation simpleMaxwellianFission Watt
       MadlandNix weightedFunctionals weighted U theta g b EFL EFH T_M
       """)

_group("census0Forms", Coverage.UNSUPPORTED, 'distribution',
       "**zero** occurrences across the 558 distributed neutron "
       "evaluations the census walked, so reading one would be untested "
       "code. This is the group nodes.py:149-156 names and deliberately "
       "keeps out of its own table, and it is the reason that table "
       "cannot back this one: on these six it would say nothing rather "
       "than no. coherentPhotonScattering is the import-time ratchet's "
       "guinea pig in test_nodes.py",
       """
       forward CoulombPlusNuclearElastic coherentPhotonScattering
       incoherentPhotonScattering thermalNeutronScatteringLaw
       thermalNeutronScatteringLaw1d
       """)

_group("doubleDifferential", Coverage.UNSUPPORTED, 'doubleDifferentialCrossSection',
       "gnds.xsd:1249 onwards. The model declares the slot and nothing "
       "fills it; these are not §18 laws and were never in phase 7b's "
       "scope, so no phase is scheduled and the reader says so rather "
       "than naming one",
       """
       doubleDifferentialCrossSection RutherfordScattering
       nuclearAmplitudeExpansion nuclearPlusInterference nuclearTerm
       realInterferenceTerm imaginaryInterferenceTerm formFactor
       realAnomalousFactor imaginaryAnomalousFactor scatteringFactor
       """)

_group("thermalScattering", Coverage.UNSUPPORTED, 'doubleDifferentialCrossSection',
       "the thermal neutron scattering law (gnds.xsd:1346 onwards), "
       "reachable only under doubleDifferentialCrossSection. TSL "
       "evaluations reach kika through kika/endf/classes/mf7 and not "
       "through here, which is what nodes.py:243-246 says of the §16.1.1 "
       "form. Worth stating plainly: GNDS 2.1's own stated focus over "
       "2.0 is this chapter, and it is the chapter kika does not read",
       """
       thermalNeutronScatteringLaw_coherentElastic
       thermalNeutronScatteringLaw_incoherentElastic
       thermalNeutronScatteringLaw_incoherentInelastic S_table
       BraggEdges BraggEdge BraggEnergy structureFactor
       boundAtomCrossSection boundAtomCrossSectionByNuclide
       DebyeWallerIntegral scatteringAtoms scatteringAtom e_critical
       e_max coherentAtomCrossSection distinctScatteringKernel
       selfScatteringKernel T_effective gridded3d GaussianApproximation
       SCTApproximation freeGasApproximation phononSpectrum
       """)

_group("fissionFragmentData", Coverage.UNSUPPORTED, 'fissionFragmentData',
       "delayed neutrons and fission energy release "
       "(gnds.xsd:1457-1565). kika reads both from ENDF MF1/455 and "
       "MF1/458 today and the GNDS side has no phase scheduled, so the "
       "reader names the container and stops",
       """
       fissionFragmentData delayedNeutrons delayedNeutron rate
       fissionEnergyReleases fissionEnergyRelease promptProductKE
       promptNeutronKE delayedNeutronKE promptGammaEnergy
       delayedGammaEnergy delayedBetaEnergy neutrinoEnergy
       nonNeutrinoEnergy totalEnergy productYields productYield
       elapsedTimes elapsedTime time yields incidentEnergies
       incidentEnergy
       """)

_group("averageProductEnergy", Coverage.UNSUPPORTED, 'averageProductEnergy',
       "§17.4 (gnds.xsd:1606) is a processed quantity and kika reads "
       "evaluated data",
       """
       averageProductEnergy
       """)

_group("function3ds", Coverage.UNSUPPORTED, None,
       "**unreachable, not unscheduled.** Its only container is the "
       "complexType xData_regions_3d_primary (gnds.xsd:2286) and no "
       "xs:element anywhere in the schema is of it, so no valid GNDS-2.0 "
       "file can contain one and nothing can report meeting it. This row "
       "is permanent -- the same standing nodes.py:227 gives regions3d",
       """
       function3ds
       """)

_group("uncertaintyBackLink", Coverage.UNSUPPORTED, 'uncertainty',
       "§7's <uncertainty> (gnds.xsd:2295-2301) is a **back-link** and "
       "not data: it names the covariance that describes a function, and "
       "§25.2.3's rowData/href states the same relation from the end "
       "kika reads. The parent is dropped with a report line and its two "
       "children are never reached",
       """
       uncertainty
       """)

_group("popsAtomic", Coverage.UNSUPPORTED, None,
       "atomic-shell structure (gnds.xsd:599-622), outside kika's "
       "minimal §12 particle model. The reader descends chemicalElement "
       "-> isotope -> nuclide and never sees this branch, so nothing "
       "reports it",
       """
       atomic configurations configuration bindingEnergy
       """)

_group("popsOther", Coverage.UNSUPPORTED, None,
       "the reader iterates gaugeBoson and baryon only "
       "(gnds.xsd:486-489) and no distributed neutron evaluation carries "
       "a lepton or an unorthodox particle, so the branch is never "
       "entered and nothing reports it",
       """
       leptons lepton unorthodoxes unorthodox
       """)

_group("physicalQuantityUncertainty", Coverage.UNSUPPORTED, None,
       "the uncertainty structure of a §3.4.3 physical quantity "
       "(gnds.xsd:1956-1995). readPhysicalQuantity takes value and unit "
       "and nothing else, so these are passed over without a line",
       """
       confidenceIntervals interval standard
       """)

_group("targetInfo", Coverage.UNSUPPORTED, None,
       "**the one silent drop under a node kika reads.** "
       "RS_EvaluatedType (gnds.xsd:96-108) hangs targetInfo off "
       "<evaluated>; kika/gnds/styles.py takes temperature, "
       "projectileEnergyDomain and documentation and never looks at the "
       "rest. It is minOccurs=0, so what kika writes stays valid -- this "
       "is a read loss, and it is the entry in this group that ought to "
       "shrink",
       """
       targetInfo isotopicAbundances
       """)

_group("energyInterval", Coverage.UNSUPPORTED, None,
       "the child of a container the §19 reader already reports "
       "(gnds.xsd:970); nothing descends into it, so it carries no line "
       "of its own",
       """
       energyInterval
       """)

_group("covarianceBackLink", Coverage.UNSUPPORTED, None,
       "the two children of §7's <uncertainty> back-link "
       "(gnds.xsd:2299-2300). The parent is dropped with a report line "
       "before they are reached, which is why they have none",
       """
       covariance listOfCovariances
       """)

# -- the twelve kika names that this schema does not declare ---------------
#
# Not a footnote. Eight of them are styles `RS_StylesType` (gnds.xsd:86-93)
# does not admit -- it admits four and kika can write twelve -- so a suite
# decoded from ACE and written with `kika.write` produces a file FUDGE's own
# schema rejects. nodes.py:383-387 already says this of `griddedCrossSection`;
# what was missing is anywhere a *user* could find it out. The other four are
# forms the model declares and no valid GNDS-2.0 document can carry.
#
# They are rendered apart from the 300 and never counted with them: a row
# called `realization` inside a list of GNDS nodes would mis-state what the
# format has. `where` says so on every one.

_NOT_IN_SCHEMA_WHY = (
    "kika can write this style and FUDGE 6.10.0's gnds.xsd declares no such "
    "element -- RS_StylesType (gnds.xsd:86-93) admits four styles and kika "
    "models twelve. An ACE-decoded suite written with kika.write therefore "
    "carries a node the reference implementation's schema rejects, and "
    "kika's own reader will not take it back either"
)

for _style in ("realization", "angularDistributionReconstructed",
               "CoulombPlusNuclearElasticMuCutoff", "MonteCarlo_cdf",
               "griddedCrossSection", "URR_probabilityTables",
               "heatedMultiGroup", "SnElasticUpScatter"):
    _one(_style, Coverage.PARTIAL, None, "notInSchemaStyles",
         _NOT_IN_SCHEMA_WHY,
         where="not in gnds.xsd")

_one("Ys1d", Coverage.UNSUPPORTED, None, "notInSchemaForms",
     "modelled, and gnds.xsd declares no <Ys1d> at all -- §6.2.5 is a 2.1 "
     "addition this 2.0 schema predates, and no distributed neutron "
     "evaluation carries one, which is the reason nodes.py:217-219 gives for "
     "leaving it unread",
     where="not in gnds.xsd")
_one("gridded1d", Coverage.UNSUPPORTED, None, "notInSchemaForms",
     "modelled; §6.2.6 is a processed representation, kika reads evaluated "
     "data, and this schema declares no such element (nodes.py:220-222)",
     where="not in gnds.xsd")
_one("regions3d", Coverage.UNSUPPORTED, None, "notInSchemaForms",
     "**unreachable, and permanently so.** gnds.xsd:2286 defines the type "
     "xData_regions_3d_primary and no xs:element anywhere is of it, so no "
     "valid GNDS-2.0 file can contain a regions3d — which is exactly why "
     "test_nodes.py:232 builds its planted-failure ratchet on this entry",
     where="not in gnds.xsd")
_one("URR_probabilityTables1d", Coverage.UNSUPPORTED, None, "notInSchemaForms",
     "modelled, and **produced live** by kika/ace/model_adapter/decode.py:284 "
     "— so an ACE-decoded suite already holds one that neither half of "
     "kika/gnds can move, and this schema has no element to write it as. "
     "§16.1.1 admits it in GNDS 2.1; this is the 2.0 file",
     where="not in gnds.xsd")

#: Every row, keyed by node name.
CAPABILITIES: Dict[str, Capability] = dict(sorted(_TABLE.items()))

#: The twelve rows whose node the schema does not declare. Never counted with
#: the 300; see the comment above.
NOT_IN_SCHEMA: Tuple[str, ...] = tuple(
    node for node, entry in CAPABILITIES.items()
    if entry.where == "not in gnds.xsd")

#: How many nodes ``gnds.xsd`` and ``covariances.xsd`` declare between them.
#: The census under ``tests/data/`` is the authority; this is the number the
#: gate compares against, stated here so a reader of this module sees it.
SCHEMA_NODE_COUNT = len(CAPABILITIES) - len(NOT_IN_SCHEMA)


class Capabilities:
    """A view of the table: the whole thing, or whatever a filter left.

    Filters return another :class:`Capabilities`, so they compose, and an
    unknown filter word **raises** rather than returning an empty view — "I do
    not know that word" and "there is nothing of that kind" are different
    answers and a support profile is the last place to conflate them.
    """

    def __init__(self, entries: Tuple[Capability, ...]) -> None:
        self._entries = entries

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # Always true, for ConversionReport's reason (conversion.py:67-70): a
        # view that exists must not read as "nothing to report".
        return True

    def __getitem__(self, node: str) -> Capability:
        for entry in self._entries:
            if entry.node == node:
                return entry
        raise KeyError(
            f"<{node}> is not a node gnds.xsd or covariances.xsd declares, "
            f"and kika does not name it either. If a real GNDS file contains "
            f"one, the schema this table was built from is not the schema "
            f"that file was written to"
        )

    def _filter(self, keep) -> "Capabilities":
        return Capabilities(tuple(e for e in self._entries if keep(e)))

    @property
    def full(self) -> "Capabilities":
        return self._filter(lambda e: e.coverage is Coverage.FULL)

    @property
    def partial(self) -> "Capabilities":
        return self._filter(lambda e: e.coverage is Coverage.PARTIAL)

    @property
    def unsupported(self) -> "Capabilities":
        return self._filter(lambda e: e.coverage is Coverage.UNSUPPORTED)

    @property
    def silent(self) -> "Capabilities":
        """The nodes kika drops **without saying so**. The list a maintainer
        should read first, and the one that may only get shorter.

        ``UNSUPPORTED`` and not ``not FULL``, deliberately: a ``PARTIAL`` row
        is a node that survives, and calling ``heated`` — write-only and never
        read — a silent loss would put it in a list whose whole point is data
        that vanishes with nobody told.

        It spans **whatever view it is asked of**, the twelve names the schema
        does not declare included, so on the whole table it is four longer than
        the number :meth:`summary` quotes — that one counts GNDS's own nodes
        and nothing else.
        """
        return self._filter(lambda e: e.coverage is Coverage.UNSUPPORTED
                            and e.reportedVia is None)

    def summary(self) -> str:
        """One line, in ``ConversionReport.summary()``'s idiom."""
        counted = [e for e in self._entries if e.where != "not in gnds.xsd"]
        beyond = len(self._entries) - len(counted)
        parts = []
        for coverage in Coverage:
            n = sum(1 for e in counted if e.coverage is coverage)
            if n:
                parts.append(f"{n} {coverage.value}")
        silent = sum(1 for e in counted
                     if e.coverage is Coverage.UNSUPPORTED
                     and e.reportedVia is None)
        line = (f"{len(counted)} of GNDS's nodes: " + ", ".join(parts)
                if counted else "no node of GNDS itself")
        if silent:
            line += f" ({silent} lost without a report line)"
        if beyond:
            line += (f"; and {beyond} node{'s' if beyond > 1 else ''} kika "
                     f"names that gnds.xsd does not declare")
        return line

    def text(self, width: int = 78) -> str:
        """The three columns, grouped, as text a docstring can hold.

        Grouped rather than flat because three hundred alphabetical rows are
        not readable and the group is the unit somebody thinks in — "does it
        do thermal scattering", not "does it do BraggEdge".
        """
        lines = [self.summary(), ""]
        # Keyed on the sentence and not on the group alone. A group whose
        # members share one honest reason renders as one block; a member that
        # earned its own sentence gets its own, and no reason is dropped
        # because a neighbour happened to be listed first.
        blocks = {}
        for entry in self._entries:
            blocks.setdefault((entry.group, entry.why), []).append(entry)
        for (group, why), members in blocks.items():
            head = members[0]
            lines.append(f"{group}  [{head.coverage.value}]  {head.where}")
            for chunk in _wrap(why, width - 4):
                lines.append(f"    {chunk}")
            if head.caveat:
                for chunk in _wrap("caveat: " + head.caveat, width - 4):
                    lines.append(f"    {chunk}")
            if head.reportedVia:
                lines.append(f"    reported via <{head.reportedVia}>")
            elif head.coverage is Coverage.UNSUPPORTED:
                lines.append("    nothing reports this at run time")
            for chunk in _wrap(" ".join(e.node for e in members), width - 6):
                lines.append(f"      {chunk}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def __repr__(self) -> str:
        return f"Capabilities({self.summary()})"

    __str__ = text


def _wrap(text: str, width: int):
    out, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def capabilities(*, coverage: Optional[str] = None,
                 group: Optional[str] = None,
                 node: Optional[str] = None) -> Capabilities:
    """What kika does with each node of GNDS. See the module docstring.

    With no argument, the whole table::

        >>> import kika.gnds
        >>> print(kika.gnds.capabilities().summary())

    ``coverage`` is one of ``"full"``, ``"partial"``, ``"unsupported"``;
    ``group`` one of the keys the table is written in; ``node`` an exact node
    name. Each raises on a word the table does not have, rather than answering
    "nothing" — see :class:`Capabilities`.

    **This does not open a file and does not know about yours.** It says what
    the library can lose; ``suite.report`` says what your file lost. The two
    are the static and the dynamic answer to the same question and neither
    stands in for the other.
    """
    view = Capabilities(tuple(CAPABILITIES.values()))
    if coverage is not None:
        try:
            wanted = Coverage(coverage)
        except ValueError:
            raise ValueError(
                f"{coverage!r} is not a coverage; the three are "
                f"{', '.join(c.value for c in Coverage)}"
            ) from None
        view = view._filter(lambda e: e.coverage is wanted)
    if group is not None:
        known = {e.group for e in CAPABILITIES.values()}
        if group not in known:
            raise ValueError(
                f"{group!r} is not a group; the table is written in "
                f"{', '.join(sorted(known))}"
            )
        view = view._filter(lambda e: e.group == group)
    if node is not None:
        view = Capabilities((view[node],))
    return view
