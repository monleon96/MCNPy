"""ACE → :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

**ACE is processed data, and the model has to say so.** An ENDF evaluation and
an ACE file are not two formats holding the same thing: ACE has already been
reconstructed from resonance parameters, Doppler-broadened to a temperature and
put on a unionised grid. GNDS §9 exists to record exactly that, through a chain
of styles linked by ``derivedFrom``. So this decoder does not produce an
``eval``-labelled cross section the way the ENDF decoder does — it produces a
``gridded`` one, hanging off a ``heated`` style, hanging off the ``evaluated``
style the file *refers to and does not contain*.

**The finding that changed what this module does.** The roadmap recorded, from
phase 1, that "ACE lacks ``qm``/``qi``", and phase 3's plan said this decoder
should report that ACE carries no Q values. **That is wrong.** ACE's LQR block
carries one Q per reaction, in MeV, aligned with MTR, and it is correct: Fe-56
capture reads +7.646 MeV, (n,2n) −11.197 MeV, (n,p) −2.913 MeV. What ACE lacks
is **QM** — the mass-difference Q — and **LR**. So the Q = 0 defect on the ACE
side was never "the format has no Q"; it was ``CrossSection.from_ace`` not
reading a block that is there. This decoder reads it.

**Not gated in CI.** Phase 0 decided against committing a 16 MB ACE fixture, so
every test here is ``@pytest.mark.tape`` and runs only where an ACE file is
reachable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from kika.nuclear_data.model import (
    AceProvenance,
    ConversionReport,
    CrossSection,
    Evaluated,
    Frame,
    GriddedCrossSection,
    Heated,
    Interpolation,
    Nuclide,
    OutputChannel,
    PhysicalQuantity,
    PoPs,
    Q,
    Reaction,
    ReactionId,
    ReactionSuite,
    URR_probabilityTables,
    URR_probabilityTables1d,
    XYs1d,
    crossSectionAxes,
    pidFromZA,
)

__all__ = ["decodeAce", "aceStyles", "qValuesByMT",
           "EVALUATED_LABEL", "HEATED_LABEL", "GRIDDED_LABEL", "URR_LABEL"]

#: Style labels this decoder assigns. They are kika's, not GNDS-mandated: §9
#: fixes the node *names*, and leaves the labels to the writer.
EVALUATED_LABEL = "eval"
HEATED_LABEL = "heated"
GRIDDED_LABEL = "gridded"
URR_LABEL = "URR"

#: ACE stores energies and Q values in MeV; the model works in eV, as ENDF does.
MEV_TO_EV = 1.0e6


def _temperatureK(ace) -> Optional[float]:
    """ACE's header temperature is **kT in MeV**, not a temperature in kelvin.

    ``Header.temperature``'s inline comment says "Temperature in K" and its
    class docstring says MeV. The docstring is right — Fe-56 at ``2.53e-08``
    is 293.6 K — and ``CrossSection.from_ace`` already divides by k_B. Recorded
    here because the two disagree in the source and only one of them can be
    followed.
    """
    from kika._constants import BOLTZMANN_CONSTANT

    kT = getattr(ace.header, "temperature", None)
    return float(kT) / BOLTZMANN_CONSTANT if kT else None


def qValuesByMT(ace, report: Optional[ConversionReport] = None) -> Dict[int, float]:
    """``{MT: Q in eV}`` from the LQR block, aligned with MTR.

    LQR is parallel to MTR — one Q per *non-elastic* reaction, in the same order
    — so the mapping is positional and there is no MT written next to a Q. MT 2
    is absent from MTR because elastic scattering has Q = 0 by definition, which
    is knowledge and not a default, so it is filled in here.
    """
    report = report if report is not None else ConversionReport()

    qBlock = getattr(ace, "q_values", None)
    reactions = getattr(ace, "reaction_mt_data", None)
    if qBlock is None or not getattr(qBlock, "has_q_values", False) or reactions is None:
        report.lost("this ACE file carries no LQR block, so no reaction has a Q value")
        return {}

    mtNumbers = list(reactions.get_mt_values("neutron"))
    qValues = qBlock.get_all_q_values()
    if len(mtNumbers) != len(qValues):
        report.warn(
            f"MTR has {len(mtNumbers)} entries and LQR has {len(qValues)}; they are "
            f"positionally aligned by definition, so the shorter one is used and "
            f"the rest have no Q"
        )

    byMT = {int(mt): float(q) * MEV_TO_EV for mt, q in zip(mtNumbers, qValues)}
    byMT.setdefault(2, 0.0)
    return byMT


def aceStyles(ace, report: Optional[ConversionReport] = None):
    """The ``derivedFrom`` chain an ACE file implies, and nothing more.

    ``evaluated`` → ``heated`` → ``griddedCrossSection`` [→ ``URR_probabilityTables``].

    **``crossSectionReconstructed`` is deliberately not in it.** A continuous-energy
    ACE has almost certainly been through RECONR, but the file does not say so,
    and a style chain is a claim about provenance. Inventing a step because it is
    probable is how a processed file starts asserting things nobody checked.

    The ``evaluated`` style *is* declared, empty: §9 makes it the root every
    ``derivedFrom`` chain resolves to, and ACE refers to an evaluation it does
    not contain. That absence is reported rather than hidden.
    """
    report = report if report is not None else ConversionReport()
    styles = []

    styles.append(Evaluated(label=EVALUATED_LABEL,
                            library=(getattr(ace.header, "comment", None) or "").strip() or None))
    report.lost(
        "ACE refers to an evaluation it does not contain, so the 'eval' style is "
        "declared with no data behind it; nothing in this reactionSuite carries an "
        "eval-labelled form"
    )

    temperature = _temperatureK(ace)
    if temperature is None:
        report.warn("this ACE file has no header temperature; the heated style has none")
    styles.append(Heated(
        label=HEATED_LABEL, derivedFrom=EVALUATED_LABEL,
        temperature=(PhysicalQuantity(value=temperature, unit="K")
                     if temperature is not None else None),
    ))
    styles.append(GriddedCrossSection(label=GRIDDED_LABEL, derivedFrom=HEATED_LABEL))

    urr = getattr(ace, "unresolved_resonance", None)
    if urr is not None and getattr(urr, "has_data", False):
        styles.append(URR_probabilityTables(label=URR_LABEL, derivedFrom=GRIDDED_LABEL))

    return styles, report


def decodeAce(ace, report: Optional[ConversionReport] = None):
    """A parsed :class:`~kika.ace.classes.ace.Ace` → a :class:`ReactionSuite`.

    The cross sections land under the ``gridded`` style, because that is what
    they are: values on ACE's unionised grid at the file's temperature. Nothing
    is labelled ``eval``.
    """
    report = report if report is not None else ConversionReport()

    styles, report = aceStyles(ace, report)
    provenance = AceProvenance(
        zaid=getattr(ace.header, "zaid", None),
        extension=getattr(ace.header, "extension", None),
        awr=getattr(ace.header, "atomic_weight_ratio", None),
        comment=getattr(ace.header, "comment", None),
        date=getattr(ace.header, "date", None),
        matid=getattr(ace.header, "matid", None),
        formatVersion=getattr(ace.header, "format_version", None),
        temperatureMeV=getattr(ace.header, "temperature", None),
    )

    za = int(provenance.zaid) if provenance.zaid is not None else 0
    pops = PoPs()
    if za:
        pops.add(Nuclide(id=pidFromZA(za), Z=za // 1000, A=za % 1000))

    suite = ReactionSuite(
        evaluation=(provenance.comment or "").strip() or "",
        projectile="n",
        target=next(iter(pops.particles), "unknown"),
        projectileFrame=Frame.lab,
    )
    suite.PoPs = pops
    suite.provenance = provenance
    for style in styles:
        suite.styles.add(style)

    qByMT = qValuesByMT(ace, report)
    for mt in _mtNumbers(ace):
        reaction, report = decodeAceReaction(ace, mt, qByMT, provenance, report)
        if reaction is not None:
            suite.reactions.append(reaction)

    _attachURR(ace, suite, report)

    # What ACE genuinely does not have. Stated per item, because "ACE carries no
    # Q values" was the previous formulation and it is false.
    report.unsupportedNode(
        "ACE has no QM (the mass-difference Q) and no LR (the complex-breakup "
        "flag): both are ENDF MF3 fields with no ACE counterpart. QI *is* present, "
        "in the LQR block, and is decoded."
    )
    report.unsupportedNode(
        "ACE carries no covariances at all, so there is no covarianceSuite to "
        "build from one"
    )
    # Both ways to the same object: the tuple is unchanged, and the
    # attribute is the one that survives `suite, _ = ...`. §11.4.
    suite.report = report
    return suite, report


def _mtNumbers(ace) -> List[int]:
    """Every MT the file exposes a cross section for, composites included.

    Composites (MT 4, 18, 101, ...) are summed on demand by
    ``CrossSectionData._get_or_compute_reaction``, so they are real cross
    sections with no Q of their own — which is correct: a sum of reactions with
    different Q values has none.
    """
    from kika._constants import MT_COMPOSITE_ORDER

    direct = list(getattr(ace.cross_section, "mt_numbers", []) or [])
    composites = [mt for mt in MT_COMPOSITE_ORDER if mt not in direct]
    return sorted(set(direct)) + [mt for mt in composites]


def decodeAceReaction(ace, mt: int, qByMT: Dict[int, float],
                      provenance: AceProvenance, report: ConversionReport):
    """One MT → a :class:`Reaction` with a ``gridded``-labelled cross section.

    An ``XYs1d`` rather than the ``regions1d`` the ENDF decoder builds: ACE has
    no ``(NBT, INT)`` regions, it is lin-lin on one union grid, and a
    single-region ``regions1d`` would assert a structure the file does not have.
    """
    reactionData = ace.cross_section._get_or_compute_reaction(mt)
    if reactionData is None:
        return None, report

    energies = np.asarray(reactionData.energies, dtype=float) * MEV_TO_EV
    values = np.asarray(reactionData.xs_values, dtype=float)

    crossSection = CrossSection()
    crossSection[GRIDDED_LABEL] = XYs1d(
        xs=energies, ys=values, interpolation=Interpolation.linlin,
        axes=crossSectionAxes(), label=GRIDDED_LABEL,
    )

    channel = OutputChannel(Q=Q(value=qByMT.get(mt), unit="eV"))
    if mt not in qByMT:
        report.lost(
            f"MT{mt}: no entry in ACE's LQR block, so this reaction's Q is unknown "
            f"(it is a composite or a summed reaction if it is one of "
            f"4, 18, 101, 27, 3)"
        )

    return Reaction(
        id=ReactionId(label=f"MT{mt}", ENDF_MT=mt),
        crossSection=crossSection,
        outputChannel=channel,
        provenance=provenance,
    ), report


def _attachURR(ace, suite: ReactionSuite, report: ConversionReport) -> None:
    """Declare the URR probability tables on the reactions they apply to.

    §16.1.1's ``URR_probabilityTables1d`` is a **reference**, not a container of
    the tables themselves — it carries an ``href``. The tables are a processed
    sampling aid, and re-expressing ACE's (cumulative probability, σ) bands as a
    GNDS node is its own piece of work. Declaring the form with an href says
    "these exist and here is where they live" rather than silently omitting a
    block that changes self-shielded reaction rates.
    """
    urr = getattr(ace, "unresolved_resonance", None)
    if urr is None or not getattr(urr, "has_data", False):
        return

    for mt in (1, 2, 18, 102):
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is None:
            continue
        reaction.crossSection[URR_LABEL] = URR_probabilityTables1d(
            href=f"/reactionSuite/reactions/reaction[@label='MT{mt}']/crossSection",
            label=URR_LABEL,
        )

    report.unsupportedNode(
        f"ACE's URR probability tables ({getattr(urr, 'num_energies', 0)} energies "
        f"x {getattr(urr, 'table_length', 0)} bands) are declared as "
        f"URR_probabilityTables1d references only; their contents are not "
        f"converted into GNDS nodes"
    )
