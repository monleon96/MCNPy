"""ENDF MF1/452, /455, /456 ↔ the model's fission multiplicities.

**Why this module exists at all.** It was written for MF31. A covariance
section is identified by an ``href`` into a ``reactionSuite`` (§25.2.3), so a
nu-bar covariance can only be decoded once there is a nu-bar node to point at —
and until this module, kika parsed MF1/452, /455 and /456 into
:class:`~kika.endf.classes.mf1.mf1mt452.MF1MT452` and friends and then dropped
them on the floor: ``decodeMF1MT451`` read MT451 and nothing read the other
three. That is the whole reason ``decodeCovarianceSuite`` used to answer "MF31
is present and parsed by kika, but this adapter covers MF33 and MF34 only".

**The three MTs do not go to the same place, and this is the part worth
reading.** ENDF writes nu_total (452), nu_delayed (455) and nu_prompt (456)
side by side as three sections of one file. GNDS does not, because only one of
them is a *primitive*:

``MT456`` §17.3 — the multiplicity of the neutron product of the fission
    output channel. This is the number that says how many neutrons come out,
    and it is stored on the product.
``MT455`` §21.3 — a ``multiplicitySum``, because the delayed nu-bar is the sum
    over the precursor families. Its decay constants become §18.4
    ``delayedNeutron`` nodes on the channel's ``fissionFragmentData``.
``MT452`` §21.3 — a ``multiplicitySum`` too: total = prompt + delayed.

with one exception that the ENDF files force: an evaluation may give **only**
MT452, and then the total *is* the primitive and goes on the product. That is
the ``separatePrompt`` argument threaded through :func:`nubarHref`, and getting
it wrong would put a covariance on the wrong node without anything failing.

The placement is not kika's invention — FUDGE's ENDF reader makes the same
three (``ENDF_ITYPE_0.py``: ``MultiplicitySum(label="total fission neutron
multiplicity", ENDF_MT=452)``, summands from the fission channel's neutron
products plus the delayed sum).

**What ENDF has that the model does not get filled with.** MF1/455 gives the
*aggregate* delayed nu-bar and NNF decay constants. The per-family
multiplicities — what the aggregate is the sum of — are the MF5/455 subsection
weights, and MF5 is not decoded into the model (phase 7b). So the families are
created with their rates and an empty multiplicity, the aggregate goes on the
``multiplicitySum``, and the ``summands`` list stays empty rather than being
filled with links to nodes that hold nothing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    EVAL_LABEL,
    Add,
    ConversionReport,
    DelayedNeutron,
    EndfProvenance,
    FissionFragmentData,
    Multiplicity,
    MultiplicitySum,
    PhysicalQuantity,
    Polynomial1d,
    Product,
    Regions1d,
    multiplicityAxes,
)

__all__ = [
    "NUBAR_MT", "FISSION_MT", "TOTAL_NUBAR_LABEL", "DELAYED_NUBAR_LABEL",
    "decodeMF1Nubar", "attachNubar", "nubarHref", "multiplicitySumHref",
    "fissionProductMultiplicityHref", "delayedNeutronMultiplicityHref",
    "nubarNode", "encodeMF1MT452", "encodeMF1MT455", "encodeMF1MT456",
]

#: The three MF1 sections this module reads, in ENDF's own order.
NUBAR_MT = (452, 455, 456)

#: The reaction the nu-bars belong to. ENDF puts them in File 1 with no MT18
#: attached, but they are the multiplicity of *fission*, and GNDS says so
#: structurally.
FISSION_MT = 18

#: §21.3 labels. Spelled as FUDGE spells them, so a kika-written GNDS file and a
#: FUDGE-written one name the same node the same way; an href resolved by string
#: match across the two would otherwise miss.
TOTAL_NUBAR_LABEL = "total fission neutron multiplicity"
DELAYED_NUBAR_LABEL = "delayed fission neutron multiplicity"

#: ENDF's conventional lower bound on incident energy. Used only to give an
#: LNU=1 polynomial a domain, and always reported when it is.
DEFAULT_EMIN_EV = 1.0e-5
DEFAULT_EMAX_EV = 2.0e7


# ---------------------------------------------------------------------------
# xPaths
# ---------------------------------------------------------------------------

def fissionProductMultiplicityHref() -> str:
    """The multiplicity of the neutron coming out of fission (§17.3).

    ``@label`` rather than ``@pid`` to match :func:`~kika.endf.model_adapter.
    covariances.angularDistributionHref`; kika labels its neutron products
    ``'n'``, so the two agree.
    """
    return (
        f"/reactionSuite/reactions/reaction[@label='MT{FISSION_MT}']"
        f"/outputChannel/products/product[@label='n']/multiplicity"
    )


def multiplicitySumHref(label: str) -> str:
    """The multiplicity held by one §21.3 ``multiplicitySum``."""
    return (
        f"/reactionSuite/sums/multiplicitySums"
        f"/multiplicitySum[@label='{label}']/multiplicity"
    )


def delayedNeutronMultiplicityHref(familyLabel: str) -> str:
    """One precursor family's own multiplicity (§18.4).

    Nothing links here yet — see the module docstring on why ``summands`` is
    empty — but the path is written down once, here, so that the MF5 work does
    not invent a second spelling of it.
    """
    return (
        f"/reactionSuite/reactions/reaction[@label='MT{FISSION_MT}']"
        f"/outputChannel/fissionFragmentData/delayedNeutrons"
        f"/delayedNeutron[@label='{familyLabel}']/product/multiplicity"
    )


def nubarHref(mt: int, separatePrompt: bool = True) -> str:
    """Where the nu-bar of ENDF ``mt`` lives in a ``reactionSuite``.

    ``separatePrompt`` is whether the evaluation carries MT456. When it does
    not, MT452 is the only multiplicity there is and it sits on the product;
    when it does, MT452 is a derived sum. Both files exist, so the caller has
    to say which one it read — defaulting the answer would silently mis-place
    every covariance on a total-only evaluation.
    """
    if mt == 456 or (mt == 452 and not separatePrompt):
        return fissionProductMultiplicityHref()
    if mt == 452:
        return multiplicitySumHref(TOTAL_NUBAR_LABEL)
    if mt == 455:
        return multiplicitySumHref(DELAYED_NUBAR_LABEL)
    raise ValueError(
        f"MT{mt} is not a fission multiplicity; MF1 carries nu-bar in "
        f"MT452, MT455 and MT456 only"
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _multiplicityFunction(section, report: ConversionReport,
                          domain: Optional[Tuple[float, float]]):
    """The ``Multiplicity`` one MF1 nu-bar section states, in its own terms."""
    lnu = int(getattr(section, "lnu", 0) or 0)
    mt = int(getattr(section, "number", 0))
    axes = multiplicityAxes()

    if lnu == 2:
        energies = np.asarray(section.energies, dtype=float)
        values = np.asarray(section.nubar_values, dtype=float)
        regions = [(int(nbt), int(code)) for nbt, code in section.interpolation]
        if not regions and energies.size:
            regions = [(int(energies.size), 2)]
            report.approximated(
                f"MF1/MT{mt}: no interpolation regions in the TAB1; assumed a "
                f"single lin-lin region over all {energies.size} points"
            )
        function = Regions1d.fromEndfRegions(
            energies, values, regions, axes=axes, label=EVAL_LABEL
        )
        return Multiplicity(function=function, label=EVAL_LABEL), regions

    if lnu == 1:
        coefficients = np.asarray(section.coefficients, dtype=float)
        if domain is None:
            domain = (DEFAULT_EMIN_EV, DEFAULT_EMAX_EV)
            report.approximated(
                f"MF1/MT{mt}: LNU=1 gives polynomial coefficients and no energy "
                f"range, and no EMAX was available from MF1/451, so the "
                f"polynomial was given the conventional "
                f"{DEFAULT_EMIN_EV:g}-{DEFAULT_EMAX_EV:g} eV domain. The "
                f"coefficients are the file's; the domain is not"
            )
        else:
            report.approximated(
                f"MF1/MT{mt}: LNU=1 states no energy range of its own, so the "
                f"polynomial's domain was taken from MF1/451's EMAX "
                f"({domain[1]:g} eV)"
            )
        function = Polynomial1d(
            coefficients=coefficients,
            domainMin_=float(domain[0]), domainMax_=float(domain[1]),
            axes=axes, label=EVAL_LABEL,
        )
        return Multiplicity(function=function, label=EVAL_LABEL), []

    report.lost(
        f"MF1/MT{mt}: LNU={lnu} is neither 1 (polynomial) nor 2 (tabulated), "
        f"so the multiplicity was not decoded"
    )
    return None, []


def decodeMF1Nubar(section, report: Optional[ConversionReport] = None,
                   domain: Optional[Tuple[float, float]] = None):
    """One MF1/452, /455 or /456 section → ``(Multiplicity, EndfProvenance, report)``.

    ``domain`` is ``(EMIN, EMAX)`` in eV, used only by LNU=1. Pass MF1/451's
    EMAX when there is one; the fallback is reported, never silent.
    """
    report = report if report is not None else ConversionReport()
    mt = int(getattr(section, "number", 0))

    multiplicity, regions = _multiplicityFunction(section, report, domain)

    headerFields: Dict[str, object] = {"lnu": getattr(section, "lnu", None)}
    if mt == 455:
        ldg = int(getattr(section, "ldg", 0) or 0)
        headerFields["ldg"] = ldg
        if ldg == 1:
            # §1.4's energy-dependent decay constants. The model's `rate` is a
            # scalar (§18.4 gives a `rate` suite of physicalQuantity, not a
            # function of E), so these have nowhere to go -- but they are the
            # section's own records and the encoder must write them back, so
            # they travel verbatim rather than being dropped.
            headerFields["decayInterpolation"] = [
                (int(a), int(b)) for a, b in getattr(section, "_decay_interp", [])
            ]
            headerFields["decayEnergies"] = [
                float(e) for e in getattr(section, "_decay_energies", [])
            ]
            headerFields["decayData"] = [
                [float(v) for v in row] for row in getattr(section, "_decay_data", [])
            ]
            report.unsupportedNode(
                "MF1/455 has LDG=1 (energy-dependent precursor decay "
                "constants). §18.4's `rate` is a scalar physicalQuantity, so "
                "the energy dependence is not in the model; the records are "
                "kept in provenance and are written back unchanged"
            )

    provenance = EndfProvenance(
        mat=getattr(section, "_mat", None),
        awr=getattr(section, "atomic_weight_ratio", None),
        za=_za(section),
        interpolationRegions=regions,
        headerFields=headerFields,
    )
    if multiplicity is not None:
        multiplicity.provenance = provenance
    return multiplicity, provenance, report


def _za(section) -> Optional[int]:
    """ZA rounded, not truncated — see :func:`kika.endf.model_adapter.decode._za`."""
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return None if value is None else int(round(float(value)))


def _delayedNeutrons(section, report: ConversionReport):
    """The §18.4 precursor families MF1/455 states, one per decay constant."""
    families = FissionFragmentData()
    # LDG=1 puts lambda(E) in the TAB2 block rather than in this list, so a
    # file with energy-dependent rates yields no families here -- the rates are
    # in provenance and `decodeMF1Nubar` has already declared them unsupported.
    constants = list(getattr(section, "decay_constants", []) or [])
    for index, rate in enumerate(constants, start=1):
        families.delayedNeutrons.append(DelayedNeutron(
            label=str(index),
            rate=PhysicalQuantity(value=float(rate), unit="1/s"),
            product=Product(pid="n", label="n"),
        ))
    if families.delayedNeutrons:
        report.lost(
            f"MF1/455: the {len(families.delayedNeutrons)} precursor families "
            f"have their decay rates and no multiplicity of their own — the "
            f"per-family split is MF5/455's subsection weights, which this "
            f"adapter does not decode. The aggregate delayed nu-bar is on the "
            f"'{DELAYED_NUBAR_LABEL}' multiplicitySum"
        )
    return families


def attachNubar(suite, mf1, report: Optional[ConversionReport] = None):
    """Hang MF1's nu-bars on *suite*, each where §17.3/§18.4/§21.3 put it.

    Returns the report. Does nothing at all when the file carries none of the
    three MTs, which is every non-fissile evaluation — including Fe-56, so this
    costs the thesis pipeline nothing.
    """
    report = report if report is not None else ConversionReport()
    sections = {
        mt: mf1.mt[mt] for mt in NUBAR_MT if mt in getattr(mf1, "mt", {})
    } if mf1 is not None else {}
    if not sections:
        return report

    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    if reaction is None:
        report.lost(
            f"MF1 carries nu-bar (MT{', MT'.join(str(mt) for mt in sections)}) "
            f"but the evaluation has no MF3/MT{FISSION_MT}, so there is no "
            f"fission reaction to hang it on and it is absent from this "
            f"reactionSuite"
        )
        return report

    emax = (suite.provenance.headerFields or {}).get("emax") if suite.provenance else None
    domain = (DEFAULT_EMIN_EV, float(emax)) if emax else None

    decoded = {}
    for mt, section in sections.items():
        multiplicity, _, report = decodeMF1Nubar(section, report, domain)
        if multiplicity is not None:
            decoded[mt] = multiplicity

    channel = reaction.outputChannel

    # §17.3: the primitive. Prompt when the file separates it, total when the
    # file has only a total -- see `nubarHref`. The product is created only when
    # there is a multiplicity to put on it: an MF1 carrying MT455 alone would
    # otherwise leave an empty neutron on the channel, which reads as "kika
    # looked and found no multiplicity" rather than "there is none to find".
    separatePrompt = 456 in decoded
    primitiveMT = 456 if separatePrompt else (452 if 452 in decoded else None)
    if primitiveMT is not None:
        channel.ensureProduct("n").multiplicity = decoded[primitiveMT]

    # §18.4 + §21.3: the delayed families and their aggregate.
    if 455 in decoded:
        channel.fissionFragmentData = _delayedNeutrons(sections[455], report)
        suite.sums.multiplicitySums.append(MultiplicitySum(
            label=DELAYED_NUBAR_LABEL,
            multiplicity=decoded[455],
            ENDF_MT=455,
        ))

    # §21.3: the total, but only when it is genuinely derived.
    if 452 in decoded and separatePrompt:
        total = MultiplicitySum(
            label=TOTAL_NUBAR_LABEL,
            multiplicity=decoded[452],
            ENDF_MT=452,
        )
        total.summands.append(Add(href=fissionProductMultiplicityHref()))
        if 455 in decoded:
            total.summands.append(Add(href=multiplicitySumHref(DELAYED_NUBAR_LABEL)))
        suite.sums.multiplicitySums.append(total)
    elif 452 in decoded and 455 in decoded and not separatePrompt:
        report.warn(
            "MF1 carries MT452 and MT455 but no MT456: the total is on the "
            "fission product and the prompt nu-bar is not in the file, so "
            "nothing links the two"
        )

    return report


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def nubarNode(suite, mt: int):
    """The :class:`Multiplicity` of ENDF ``mt``, wherever it was put, or ``None``.

    The inverse of the placement :func:`attachNubar` performs, and written as
    one function so the encoders cannot disagree with the decoder about where
    to look.
    """
    if mt not in NUBAR_MT:
        raise ValueError(f"MT{mt} is not a fission multiplicity")

    summed = suite.sums.multiplicitySums.byENDF_MT(mt)
    if summed is not None:
        return summed.multiplicity

    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    if reaction is not None and mt in (452, 456):
        for product in reaction.outputChannel.products:
            if product.pid == "n" and product.multiplicity is not None:
                return product.multiplicity
    return None


def _tab1FromMultiplicity(multiplicity, mt: int):
    """``(interpolation, energies, values)`` out of a tabulated multiplicity."""
    function = multiplicity.function
    if isinstance(function, Regions1d):
        xs, ys, pairs = function.toEndfRegions()
        return [(int(nbt), int(code)) for nbt, code in pairs], list(xs), list(ys)
    raise ValueError(
        f"MF1/MT{mt}: the multiplicity is a {type(function).__name__}, which is "
        f"not a tabulated form; only regions1d (LNU=2) and polynomial1d (LNU=1) "
        f"can be written to MF1"
    )


def _fillNubarSection(section, multiplicity, mt: int, mat):
    """Populate an ``MF1MT452``/``455``/``456`` from the model. Shared by all three."""
    provenance = multiplicity.provenance
    header = (getattr(provenance, "headerFields", None)) or {}
    lnu = int(header.get("lnu") or 0)
    if not lnu:
        lnu = 1 if isinstance(multiplicity.function, Polynomial1d) else 2

    section._za = float(provenance.za) if provenance and provenance.za else None
    section._awr = float(provenance.awr) if provenance and provenance.awr else None
    section._mat = int(mat) if mat is not None else getattr(provenance, "mat", None)
    section._lnu = lnu

    if lnu == 1:
        coefficients = list(np.asarray(multiplicity.function.coefficients, dtype=float))
        section._nc = len(coefficients)
        section._coefficients = coefficients
    else:
        interpolation, energies, values = _tab1FromMultiplicity(multiplicity, mt)
        # The file's own (NBT, INT) pairs when they were kept: the same argument
        # `encodeMF3MT` makes -- a round trip must not depend on the
        # reconstruction from regions1d staying byte-for-byte faithful.
        kept = getattr(provenance, "interpolationRegions", None)
        section._interpolation = (
            [(int(a), int(b)) for a, b in kept] if kept else interpolation
        )
        section._nr = len(section._interpolation)
        section._np = len(energies)
        section._energies = energies
        section._nubar = values
    return section


def _encodeOne(cls, suite, mt: int, mat, report):
    report = report if report is not None else ConversionReport()
    multiplicity = nubarNode(suite, mt)
    if multiplicity is None:
        raise ValueError(
            f"this reactionSuite carries no MT{mt} nu-bar, so MF1/{mt} cannot "
            f"be written from it"
        )
    section = cls()
    _fillNubarSection(section, multiplicity, mt, mat)
    return section, report


def encodeMF1MT452(suite, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None):
    """A ``ReactionSuite`` → its ``MF1MT452`` (total nu-bar)."""
    from kika.endf.classes.mf1.mf1mt452 import MF1MT452

    return _encodeOne(MF1MT452, suite, 452, mat, report)


def encodeMF1MT456(suite, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None):
    """A ``ReactionSuite`` → its ``MF1MT456`` (prompt nu-bar)."""
    from kika.endf.classes.mf1.mf1mt456 import MF1MT456

    return _encodeOne(MF1MT456, suite, 456, mat, report)


def encodeMF1MT455(suite, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None):
    """A ``ReactionSuite`` → its ``MF1MT455`` (delayed nu-bar and decay rates).

    The rates come back off the §18.4 ``delayedNeutron`` nodes, so a caller who
    edited one edits the file. LDG=1's energy-dependent block is written from
    the provenance the decoder kept, since the model has no home for it.
    """
    from kika.endf.classes.mf1.mf1mt455 import MF1MT455

    report = report if report is not None else ConversionReport()
    multiplicity = nubarNode(suite, 455)
    if multiplicity is None:
        raise ValueError(
            "this reactionSuite carries no MT455 nu-bar, so MF1/455 cannot be "
            "written from it"
        )

    section = MF1MT455()
    _fillNubarSection(section, multiplicity, 455, mat)

    header = getattr(multiplicity.provenance, "headerFields", None) or {}
    ldg = int(header.get("ldg") or 0)
    section._ldg = ldg

    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    families = []
    if reaction is not None and reaction.outputChannel.fissionFragmentData is not None:
        families = list(reaction.outputChannel.fissionFragmentData.delayedNeutrons)

    if ldg == 0:
        rates = [
            float(family.rate.convertedTo("1/s").value)
            for family in families if family.rate is not None
        ]
        section._nnf = len(rates)
        section._decay_constants = rates
    else:
        section._nnf = len(families)
        section._decay_interp = [
            (int(a), int(b)) for a, b in header.get("decayInterpolation", [])
        ]
        section._decay_energies = list(header.get("decayEnergies", []))
        section._decay_ne = len(section._decay_energies)
        section._decay_data = [list(row) for row in header.get("decayData", [])]

    return section, report
