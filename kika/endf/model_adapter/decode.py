"""ENDF → :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

**Why this lives under ``kika/endf/``.** It imports both the format classes and
the model, so it has to live on one side or the other, and the arrow points from
format to calculation: ``kika.endf`` may import ``kika.nuclear_data``, never the
reverse. Putting the adapter in the model would recreate exactly the inverted
dependency phase 2 spent an increment straightening.

**Nothing outside this package may import it.** Not ``kika/endf/__init__.py``,
not ``read_endf``. A module-level import from either would put the model on the
``read_endf`` critical path and change what ``import kika`` costs for the
cluster and the app — and ``kika.endf.model_adapter`` is deliberately absent
from kika-app's PyInstaller ``hiddenimports``, so a lazy import from an
app-reachable path would break the frozen build.
:mod:`kika.endf.model_adapter.tests.test_nothing_imports_the_adapter` enforces it.

The decode is deliberately *lossless in the format's own terms*: every field the
ENDF section carries is either mapped to a model node or kept in
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`, so
:mod:`~kika.endf.model_adapter.encode` can rebuild the section without
recomputing anything.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    EVAL_LABEL,
    AngularTwoBody,
    Axes,
    ConversionReport,
    CrossSection,
    Distribution,
    EndfProvenance,
    Evaluated,
    Frame,
    Nuclide,
    OutputChannel,
    PhysicalQuantity,
    PoPs,
    Q,
    RangeQuantity,
    Isotropic2d,
    Reaction,
    ReactionId,
    ReactionSuite,
    Regions1d,
    Uncorrelated,
    XYs2d,
    crossSectionAxes,
    pidFromZA,
)

from .resonances import decodeMF2MT151

__all__ = ["decodeMF3MT", "decodeMF1MT451", "decodeReactionSuite"]

#: MF numbers kika's parser registry covers. Everything else is declared
#: unsupported by the report rather than silently skipped.
SUPPORTED_MF = (1, 2, 3, 4, 5, 6, 31, 32, 33, 34, 35)

#: Of those, the ones whose content belongs to the ``covarianceSuite``
#: (§25.1.1) rather than to the ``reactionSuite``. Kept as a set of its own
#: because MF5 broke the rule the redirect loop used to assume — that
#: "supported and not one of 1, 2, 3, 4" meant "covariance". MF5 is neither:
#: it is reactionSuite content, and since the MF5 adapter landed it is decoded
#: like the four. MF6 is the second of the same kind and is decoded the same
#: way, one product at a time rather than one section.
COVARIANCE_MF = (31, 32, 33, 34, 35)


def _za(section) -> int:
    """ZA, **rounded** rather than truncated.

    ENDF's fixed-format floats do not round-trip exactly: Th-232's
    ``9.023200+4`` reads back as ``90231.99999999999``, and ``int()`` on that
    names Ac-231. See :func:`kika.nuclear_data.model.provenance._asEndfInt`.
    """
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return int(round(float(value))) if value is not None else 0


def decodeMF3MT(mf3mt, report: Optional[ConversionReport] = None) -> Tuple[Reaction, ConversionReport]:
    """One MF3/MT section → a :class:`Reaction` with an ``eval``-labelled form.

    The cross section becomes a ``regions1d`` built from the file's own
    ``(NBT, INT)`` pairs, which is the honest representation: a table with three
    interpolation laws has three regions, not one "dominant" law with the rest
    hidden in a metadata dict.

    ``QI`` becomes the output channel's ``Q``; ``QM`` and ``LR`` go to
    provenance. See :class:`EndfProvenance` for why.
    """
    report = report if report is not None else ConversionReport()

    energies = np.asarray(mf3mt.energies, dtype=float)
    values = np.asarray(mf3mt.cross_sections, dtype=float)
    regions = [(int(nbt), int(code)) for nbt, code in mf3mt.energy_interpolation]

    if not regions and energies.size:
        regions = [(int(energies.size), 2)]
        report.approximated(
            f"MT{mf3mt.number}: MF3 carried no interpolation regions; assumed a "
            f"single lin-lin region over all {energies.size} points"
        )

    axes = crossSectionAxes()
    crossSection = CrossSection()
    crossSection[EVAL_LABEL] = Regions1d.fromEndfRegions(
        energies, values, regions, axes=axes, label=EVAL_LABEL
    )

    provenance = EndfProvenance(
        mat=getattr(mf3mt, "_mat", None),
        awr=mf3mt.atomic_weight_ratio,
        za=_za(mf3mt),
        qm=mf3mt.q_mass_difference,
        lr=mf3mt.breakup_flag,
        interpolationRegions=regions,
    )

    channel = OutputChannel(Q=Q(value=_qi(mf3mt, report), unit="eV"))

    reaction = Reaction(
        id=ReactionId(label=f"MT{mf3mt.number}", ENDF_MT=int(mf3mt.number)),
        crossSection=crossSection,
        outputChannel=channel,
        provenance=provenance,
    )
    return reaction, report


def _qi(mf3mt, report: ConversionReport) -> Optional[float]:
    value = mf3mt.q_reaction
    if value is None:
        report.lost(f"MT{mf3mt.number}: MF3 carried no QI, so the reaction Q is unknown")
        return None
    return float(value)


def decodeMF1MT451(mt451, report: Optional[ConversionReport] = None):
    """MF1/451 → ``(PoPs, Evaluated style, EndfProvenance)``.

    The whole 451 header goes to provenance verbatim. Most of it is ENDF
    bookkeeping with no GNDS counterpart (``NLIB``, ``NMOD``, ``LDRV``, ...) and
    the encoder has to write it back unchanged.
    """
    report = report if report is not None else ConversionReport()

    fields = {
        name: getattr(mt451, attr, None)
        for name, attr in (
            ("lrp", "_lrp"), ("lfi", "_lfi"), ("nlib", "_nlib"), ("nmod", "_nmod"),
            ("elis", "_elis"), ("sta", "_sta"), ("lis", "_lis"), ("liso", "_liso"),
            ("nfor", "_nfor"), ("awi", "_awi"), ("emax", "_emax"), ("lrel", "_lrel"),
            ("nsub", "_nsub"), ("nver", "_nver"), ("ldrv", "_ldrv"),
            # TEMP is physics rather than bookkeeping, but MF1/451 is the only
            # place ENDF states the target temperature and the encoder has to
            # write it back, so it travels with the rest of the header.
            ("temp", "_temp"),
        )
    }
    # Read through the **public properties**, not through guessed private names.
    # The first version of this guessed `_laboratory`, `_authors`, `_eval_date`
    # and four more from the property names; the real fields are `_alab`,
    # `_auth`, `_edate`, `_ref`, `_ddate`, `_rdate`, `_zsymam`. `getattr(..., None)`
    # meant every one silently returned "", so every reactionSuite decoded since
    # P6 has had an empty evaluationInfo and an empty `evaluation`. Nothing
    # caught it until P9 compared the façade against the body it replaced.
    evaluationInfo = {
        name: getattr(mt451, name, None) or ""
        for name in ("laboratory", "authors", "eval_date", "reference",
                     "dist_date", "revision_date", "material_id")
    }

    # The NWD descriptive records and the NXC directory. Neither is physics and
    # neither is parsed into anything else -- `evaluationInfo` covers the seven
    # fields of the first two text records and nothing covers the remaining
    # ~600 lines of an evaluator's comment block -- so an encoder that does not
    # keep them can only approximate the section. `_text_lines` is the whole
    # section's raw lines, header included, so the text starts at index 4; the
    # ID columns are dropped because they are regenerated on the way out.
    nwd = int(getattr(mt451, "_nwd", 0) or 0)
    rawLines = getattr(mt451, "_text_lines", None) or []
    descriptiveText = [line[:66] for line in rawLines[4:4 + nwd]]
    if nwd and len(descriptiveText) < nwd:
        report.warn(
            f"MF1/451 declares NWD={nwd} descriptive records but only "
            f"{len(descriptiveText)} were kept, so it cannot be written back "
            f"as it was read"
        )

    provenance = EndfProvenance(
        mat=getattr(mt451, "_mat", None),
        awr=getattr(mt451, "atomic_weight_ratio", None),
        za=_za(mt451),
        headerFields=fields,
        evaluationInfo=evaluationInfo,
        descriptiveText=descriptiveText,
        directory=[tuple(entry) for entry in getattr(mt451, "_directory", None) or []],
    )

    za = _za(mt451)
    pops = PoPs()
    if za:
        pops.add(Nuclide(id=pidFromZA(za), Z=za // 1000, A=za % 1000))

    style = Evaluated(
        label=EVAL_LABEL,
        library=str(fields.get("nlib") or ""),
        version=str(fields.get("nver") or ""),
        date=evaluationInfo.get("eval_date") or None,
    )
    return pops, style, provenance, report


def decodeReactionSuite(endf, report: Optional[ConversionReport] = None):
    """A parsed ``ENDF`` object → a :class:`ReactionSuite`, plus the report.

    MF1 (451 and the nu-bars), MF2, MF3 and MF4 are read. The covariances are
    **not** part of a ``reactionSuite``: §25.1.1 makes ``covarianceSuite`` a
    root node in its own right, linked through ``externalFiles``, so
    MF31/33/34/35 go through
    :func:`~kika.endf.model_adapter.covariances.decodeCovarianceSuite` instead.
    Every MF present in the file that neither decoder reads is **declared**
    unsupported, which is the whole reason the report exists.
    """
    report = report if report is not None else ConversionReport()

    pops, style, headerProvenance = PoPs(), None, None
    mf1 = endf.mf.get(1) if hasattr(endf, "mf") else None
    if mf1 is not None and 451 in getattr(mf1, "mt", {}):
        pops, style, headerProvenance, report = decodeMF1MT451(mf1.mt[451], report)
    else:
        report.lost("no MF1/451: the evaluation has no header, so no PoPs and no style")

    target = next(iter(pops.particles), "unknown")
    suite = ReactionSuite(
        evaluation=(headerProvenance.evaluationInfo.get("material_id", "") if headerProvenance else ""),
        projectile="n",
        target=target,
        projectileFrame=Frame.lab,
    )
    suite.PoPs = pops
    if style is not None:
        suite.styles.add(style)
    suite.provenance = headerProvenance

    mf3 = endf.mf.get(3) if hasattr(endf, "mf") else None
    if mf3 is not None:
        for mt in sorted(getattr(mf3, "mt", {})):
            reaction, report = decodeMF3MT(mf3.mt[mt], report)
            suite.reactions.append(reaction)
    else:
        report.lost("no MF3: the evaluation carries no cross sections")

    # After MF3, and it has to be: the nu-bars hang off the fission reaction,
    # which does not exist until MF3/MT18 has been decoded.
    if mf1 is not None:
        from .multiplicity import attachNubar

        report = attachNubar(suite, mf1, report)

    mf2 = endf.mf.get(2) if hasattr(endf, "mf") else None
    if mf2 is not None and 151 in getattr(mf2, "mt", {}):
        # The provenance is **kept**, not discarded. `encodeMF2MT151` needs it —
        # QX, LRX, LAD and the particle-pair columns have no model node — and
        # dropping it here left a suite whose resonances could be read and not
        # written. See `Resonances.provenance`.
        resonances, resonanceProvenance, report = decodeMF2MT151(mf2.mt[151], report)
        resonances.provenance = resonanceProvenance
        suite.resonances = resonances

    mf4 = endf.mf.get(4) if hasattr(endf, "mf") else None
    if mf4 is not None:
        for mt in sorted(getattr(mf4, "mt", {})):
            report = _attachAngularDistribution(suite, mf4.mt[mt], mt, report)

    # After MF4, and it has to be: an MT that states both is **one**
    # `uncorrelated` in GNDS, not two forms, and this pass rewrites what that
    # one left rather than appending beside it.
    mf5 = endf.mf.get(5) if hasattr(endf, "mf") else None
    if mf5 is not None:
        for mt in sorted(getattr(mf5, "mt", {})):
            report = _attachEnergyDistribution(suite, mf5.mt[mt], mt, report)

    # After MF5, and it has to be: an MT that states MF6 with a negative LAW
    # also states MF4 or MF5, and the deferring product must not overwrite the
    # `uncorrelated` those two passes built.
    mf6 = endf.mf.get(6) if hasattr(endf, "mf") else None
    if mf6 is not None:
        for mt in sorted(getattr(mf6, "mt", {})):
            report = _attachEnergyAngleDistributions(suite, mf6.mt[mt], mt, report)

    present = set(getattr(endf, "mf", {}))
    for mf in sorted(present - set(SUPPORTED_MF)):
        report.unsupportedNode(
            f"MF{mf} is present in the file and kika's parser registry does not "
            f"cover it; it is absent from this reactionSuite"
        )
    for mf in sorted((present & set(SUPPORTED_MF)) & set(COVARIANCE_MF)):
        report.unsupportedNode(
            f"MF{mf} is present and parsed; it is a covariance file, so it "
            f"belongs to the covarianceSuite (§25.1.1) and not to this "
            f"reactionSuite. Call decodeCovarianceSuite for it."
        )

    # MF5 is decoded above, one section at a time. What stays here is the part
    # that is *not* about this reactionSuite: `report_gaps` names the laws the
    # *reader* kept as bytes -- LF=12 and anything a future revision adds --
    # and declaring MF5 supported without saying which laws only pass through
    # would turn a silent skip into a silent false claim of coverage.
    #
    # This list is narrower than it used to be, and `decodeMF5MT` covers the
    # difference. Since the reader learned LF=5/7/9/11 those laws are decoded
    # and so are *not* gaps here, while the model still has no node for them --
    # so the section-level decode states that loss itself rather than leaning
    # on this pass, which would now stay quiet about it.
    #
    # Note what is deliberately absent: no message sends the user to
    # `decodeCovarianceSuite` for MF5, which is what the old `- {1,2,3,4}` set
    # did and was simply false.
    if mf5 is not None:
        for mt in sorted(getattr(mf5, "mt", {})):
            for gap in getattr(mf5.mt[mt], "report_gaps", list)():
                report.unsupportedNode(gap)

    # MF6 needs the same `report_gaps` pass as MF5, and it is worth reading
    # even though every MF6 law *is* parsed: a product whose LAW is -14 or -15
    # defers its distribution to MF14 or MF15, and those have no parser — so
    # the section can be read in full and still not have the distribution in
    # hand. `_attachEnergyAngleDistributions` says the same thing from the
    # model's side, per product; this says it from the file's.
    if mf6 is not None:
        for mt in sorted(getattr(mf6, "mt", {})):
            for gap in getattr(mf6.mt[mt], "report_gaps", list)():
                report.unsupportedNode(gap)

    if style is not None:
        report = _attachEvaluatedDomain(suite, style, headerProvenance, report)

    # Both ways to the same object: the tuple is unchanged, and the
    # attribute is the one that survives `suite, _ = ...`. §11.4.
    suite.report = report
    return suite, report


def _attachEvaluatedDomain(suite: ReactionSuite, style: Evaluated,
                           provenance: Optional[EndfProvenance],
                           report: ConversionReport) -> ConversionReport:
    """The ``evaluated`` style's ``temperature`` and ``projectileEnergyDomain``.

    ``RS_EvaluatedType`` makes both mandatory and the ENDF path was giving
    neither, so every GNDS file kika wrote from a tape failed validation on its
    fourth line. Both numbers are already read -- MF1/451 states the target
    temperature in TEMP and the upper limit in EMAX -- they were simply never
    put on the style.

    **This is not only about validity.** ``kika/gnds/encode.py``'s
    ``_SuiteWriter.domain`` reads the domain off this style to fill the
    ``domainMin``/``domainMax`` that ``xData_constant1d`` requires on every Q
    and multiplicity, and with no style to read it fell back to ENDF's
    assumed 1e-5 eV to 20 MeV. For a 150 MeV evaluation that is a *false*
    statement in the file, not a missing one.

    Called after the reactions because the lower bound is not in the header:
    ENDF has no EMIN, so it comes off the MF3 grid the file actually carries.
    Nothing is invented -- a suite with no cross sections keeps ``None`` and the
    writer's existing warning stays the truth.
    """
    header = provenance.headerFields if provenance is not None else {}

    temperature = header.get("temp")
    if temperature is not None:
        style.temperature = PhysicalQuantity(value=float(temperature), unit="K")

    maximum = header.get("emax")
    minima = [getattr(reaction.crossSection[label], "domainMin", None)
              for reaction in suite.reactions
              for label in reaction.crossSection]
    minima = [value for value in minima if value is not None and value == value]
    if maximum is None or not minima:
        report.warn(
            "MF1/451 gave no EMAX or the evaluation carries no cross section "
            "grid, so the evaluated style has no projectileEnergyDomain and "
            "the constant1d nodes fall back to the range ENDF assumes"
        )
        return report
    style.projectileEnergyDomain = RangeQuantity(
        min=float(min(minima)), max=float(maximum), unit="eV"
    )
    return report


def _attachAngularDistribution(suite: ReactionSuite, mf4mt, mt: int,
                               report: ConversionReport) -> ConversionReport:
    """Hang one MF4 section on the neutron product of its reaction.

    GNDS puts a distribution on a *product* of an output channel, not on the
    reaction — which is why an MF4 section with no MF3 counterpart has nowhere
    to go, and is reported rather than dropped into an invented reaction.
    """
    from .angular import decodeMF4MT

    # `findReaction...`, not `reactionByENDF_MT`: the strict one raises, so the
    # branch below was unreachable when this was written. It never fired because
    # every MF4/MT on the tapes tested has an MF3/MT, which is exactly the sort
    # of thing that stays hidden until the one tape where it is not true.
    reaction = suite.findReactionByENDF_MT(mt)
    if reaction is None:
        report.lost(
            f"MF4/MT{mt} has no MF3/MT{mt} to hang from; GNDS attaches a "
            f"distribution to a product of a reaction, and there is no reaction"
        )
        return report

    distribution, provenance, report = decodeMF4MT(mf4mt, report)
    if distribution is None:
        return report

    channel = reaction.outputChannel
    channel.genre = "twoBody"
    # `ensureProduct`, not a fresh `Product`: MF1's nu-bar may already have put
    # a neutron on this channel, and §17.2.1 gives one product one multiplicity
    # *and* one distribution. Appending here produced two neutrons on the
    # fission channel of every fissile tape -- see `OutputChannel.ensureProduct`.
    product = channel.ensureProduct("n")
    product.provenance = provenance
    if product.distribution is None:
        product.distribution = Distribution()
    product.distribution[EVAL_LABEL] = distribution
    return report


def _attachEnergyDistribution(suite: ReactionSuite, mf5mt, mt: int,
                              report: ConversionReport) -> ConversionReport:
    """Hang one MF5 section on the neutron product of its reaction.

    **MF4 and MF5 are one node, not two.** §18.3's ``uncorrelated`` is the
    evaluation stating P(mu|E) and P(E'|E) separately, which is exactly what a
    tape carrying both files does, so this pass *rewrites* what
    :func:`_attachAngularDistribution` left rather than appending beside it.
    Three shapes come out of that, and the third is the commonest in the
    library by two orders of magnitude:

    ==============  ==============  ====================================
    MF4             MF5             result
    ==============  ==============  ====================================
    yes             no              ``angularTwoBody`` — untouched here
    yes             yes             ``uncorrelated``, both halves stated
    no              yes             ``uncorrelated``, angular **inferred**
    ==============  ==============  ====================================

    **The third row is an inference, it is reported as one, and it fires on cut
    tapes only.** Measured 2026-08-24 over ENDF/B-VIII.1's 595 MF5 sections:
    **zero** of the 487 modellable ones lack an MF4 sibling. So the row is a
    rule for trimmed and partial tapes — the committed ``micro_pfns_tape`` is
    one, its cut having dropped Cf-252's real MF4/MT18 — and not a description
    of the library. Where it does fire, ENDF says nothing about angle: §5
    leaves the emission isotropic in the lab by convention, and the GNDS
    distribution agrees (126 095 ``isotropic2d`` against 406 ``XYs2d`` in the
    ``uncorrelated/angular`` position). Agreeing with a convention is still not
    reading a number, so it goes to ``report.approximations``: a loss is
    visible, an approximation looks like data.
    """
    from .energy import decodeMF5MT

    # Decode **before** looking for a reaction. The two questions are
    # independent — "can kika model this law?" and "is there a product to hang
    # it on?" — and asking them the other way round made an unmodellable
    # MT455 come back as "there is no MF3/MT455" with no word about the law,
    # which is a true sentence that leaves the reader with the wrong idea.
    energy, provenance, report = decodeMF5MT(mf5mt, report)

    reaction = suite.findReactionByENDF_MT(mt)
    if reaction is None:
        # MT455 is this branch's whole population: the delayed spectrum has no
        # cross section, so it has no MF3 and no reaction. Its GNDS home is
        # §18.4's `fissionFragmentData/delayedNeutrons`, which `attachNubar`
        # already builds from MF1/455 and which no decoder fills distributions
        # into yet. Declared rather than dropped, and the section is not
        # written back either.
        report.lost(
            f"MF5/MT{mt} has no MF3/MT{mt} to hang from; GNDS attaches a "
            f"distribution to a product of a reaction, and there is no "
            f"reaction. For MT455 the home is §18.4's delayedNeutrons and not "
            f"a reaction at all — a separate increment. The section is absent "
            f"from this reactionSuite and from anything written back from it"
        )
        return report

    channel = reaction.outputChannel
    standing = channel.products.byPid("n")
    if energy is None and not standing:
        # A law with no model node, on a channel that has no neutron product
        # either — nothing read MF4 and no nu-bar built one. The provenance
        # would have nowhere to live but a **hollow** product: a `<product>`
        # with no multiplicity and no distribution, which §17.2.1 does not
        # admit and which would trade one declared loss for an invalid node.
        # So the section is declared instead, and it is not written back.
        report.lost(
            f"MF5/MT{mt} states only laws kika does not model and its reaction "
            f"has no neutron product to carry the section's provenance; it is "
            f"absent from this reactionSuite and from anything written back "
            f"from it. A product holding neither a multiplicity nor a "
            f"distribution is not a place to put it"
        )
        return report

    product = channel.ensureProduct("n")

    # One product, one provenance. MF4's fields are flat in `headerFields` and
    # MF5's live under its own key, so neither file overwrites the other and
    # `encodeMF4MT` keeps reading exactly what it wrote.
    existing = getattr(product, "provenance", None)
    if existing is not None and getattr(existing, "sourceFormat", None) == "endf":
        existing.headerFields["mf5"] = provenance.headerFields["mf5"]
    else:
        product.provenance = provenance

    if energy is None:
        # NK>1, or a law with no model node, on a product that exists anyway.
        # The provenance carries the whole section, so the tape still comes
        # back; the reactionSuite gains no distribution, and `decodeMF5MT` has
        # already said why.
        return report

    if product.distribution is None:
        product.distribution = Distribution()
    existingForm = product.distribution.get(EVAL_LABEL)

    angular, frame = _angularHalf(existingForm, mt, report)
    if angular is None:
        return report

    product.distribution[EVAL_LABEL] = Uncorrelated(
        angular=angular, energy=energy, productFrame=frame
    )
    # Not two-body: `twoBody` says the kinematics fix E' from mu, and a section
    # that tabulates P(E'|E) independently is the statement that they do not.
    channel.genre = "NBody"
    return report



def _attachEnergyAngleDistributions(suite: ReactionSuite, mf6mt, mt: int,
                                    report: ConversionReport) -> ConversionReport:
    """Hang one MF6 section on the products of its reaction, creating them.

    **This is the pass that makes an output channel more than a neutron.** MF4
    and MF5 describe the one secondary their files are about, so both passes
    reach for ``ensureProduct('n')``; MF6 names every product the reaction has
    — C-12's MT5 lists twenty-one, and a Fe-56 section a hundred and two — and
    each carries its own yield and its own law.

    The provenance goes on the **reaction**, under an ``"mf6"`` key beside
    MF3's flat fields, because MF6 is one section per MT the way MF3 is. Each
    of its per-product records names the pid it produced, and that is what
    :func:`~kika.endf.model_adapter.energy_angle.encodeMF6MT` looks the form up
    by on the way out.
    """
    from .energy_angle import decodeMF6MT

    # Decode before looking for a reaction, for the reason
    # `_attachEnergyDistribution` gives: "can kika model these laws?" and "is
    # there a channel to hang them on?" are independent questions, and asking
    # them the other way round hides the first behind the second.
    entries, provenance, report = decodeMF6MT(mf6mt, report)

    reaction = suite.findReactionByENDF_MT(mt)
    if reaction is None:
        report.lost(
            f"MF6/MT{mt} has no MF3/MT{mt} to hang from; GNDS attaches a "
            f"distribution to a product of a reaction, and there is no "
            f"reaction. The section is absent from this reactionSuite and "
            f"from anything written back from it"
        )
        return report

    # One reaction, one provenance. MF3's fields are flat in `headerFields`
    # and MF6's live under their own key, so neither file decides the other's
    # -- the same rule MF4 and MF5 follow on a product.
    existing = getattr(reaction, "provenance", None)
    if existing is not None and getattr(existing, "sourceFormat", None) == "endf":
        existing.headerFields["mf6"] = provenance.headerFields["mf6"]
    else:
        reaction.provenance = provenance

    channel = reaction.outputChannel
    channel.genre = _channelGenre(mf6mt, mt)

    for label, pid, multiplicity, distribution in entries:
        # `ensureProduct` matches on the label, which is what lets two
        # subsections of one section name the same particle and still get two
        # nodes -- and what lets MF6's neutron on a fission channel be the
        # product MF1's nu-bar already put there, since both are labelled 'n'.
        product = channel.ensureProduct(pid, label)
        _attachMultiplicity(product, multiplicity, label, mt, report)
        if distribution is None:
            continue
        if product.distribution is None:
            product.distribution = Distribution()
        product.distribution[EVAL_LABEL] = distribution

    return report


def _attachMultiplicity(product, multiplicity, pid: str, mt: int,
                        report: ConversionReport) -> None:
    """MF6's ``y(E)`` on a product, unless another file already answered.

    **The only collision is MT18's neutron, and MF1 wins it.** ``attachNubar``
    puts the prompt nu-bar there from MF1/456, and MF6's ``y(E)`` for the same
    product is a different quantity: ENDF/B-VIII.1's U-235 states 0.0158 at
    thermal for the ``LAW=0`` subsection's share against a nu-bar of 2.414, and
    the distributed GNDS file carries the 2.414. Overwriting would put the
    share where §17.3 promises the multiplicity, which no schema can catch.
    """
    if getattr(product, "multiplicity", None) is None:
        product.multiplicity = multiplicity
        return
    report.warn(
        f"MT{mt}: product {pid!r} already carries a multiplicity from another "
        f"file — MF1's nu-bar is the only case — so MF6's y(E) for it is not "
        f"written to the model. The two are different quantities and the "
        f"nu-bar is the one §17.3 asks for. MF6's yield is kept in the "
        f"provenance and comes back on the tape unchanged"
    )


def _channelGenre(mf6mt, mt: int) -> str:
    """§17.1.1's ``genre``, from the laws the section actually states.

    ``twoBody`` says the kinematics fix E' from mu, which is true exactly when
    every product is a two-body one — LAW=2 (angle given), LAW=3 (isotropic) or
    LAW=4 (the recoil of one of those). Anything that tabulates an outgoing
    energy is ``NBody``.

    MT5 is neither, and GNDS has a third word for it: it is the sum of every
    channel the evaluation did not give an MT of its own, so its products are
    not the products of one reaction. That is what the distributed files write.
    """
    if mt == 5:
        return "sumOfRemainingOutputChannels"
    laws = {product.law for product in mf6mt.products}
    return "twoBody" if laws and laws <= {2, 3, 4} else "NBody"


def _angularHalf(existingForm, mt: int, report: ConversionReport):
    """The ``angular`` child of the ``uncorrelated``, and the product frame."""
    if existingForm is None:
        report.approximated(
            f"MT{mt}: the tape states MF5 and no MF4, so the angular half of "
            f"§18.3's uncorrelated is not read but inferred — ENDF-6 §5 leaves "
            f"the emission isotropic in the lab when MF4 is absent, and "
            f"gnds.xsd makes the angular child mandatory. Written as "
            f"isotropic2d, which is what 126 095 of the library's 126 501 "
            f"uncorrelated nodes carry; it is still a convention and not a "
            f"number the evaluator wrote. Nothing is written back to ENDF for "
            f"it — no MF4/MT{mt} section is created — so only a GNDS file "
            f"carries it. Zero of ENDF/B-VIII.1's 487 modellable MF5 sections "
            f"lack an MF4 sibling (measured 2026-08-24), so this is a trimmed "
            f"or partial tape"
        )
        return Isotropic2d(productFrame=Frame.lab), Frame.lab

    if isinstance(existingForm, Isotropic2d):
        return existingForm, existingForm.productFrame

    if isinstance(existingForm, AngularTwoBody):
        angular = existingForm.angular
        if isinstance(angular, (XYs2d, Isotropic2d)):
            return angular, existingForm.productFrame
        report.unsupportedNode(
            f"MT{mt}: MF4 gave a {type(angular).__name__} and §18.3's "
            f"uncorrelated/angular admits only isotropic2d, XYs2d, forward or "
            f"recoil — no regions2d. That is an LTT=3 section on an MT that "
            f"also states MF5. The angularTwoBody is kept as it was and the "
            f"MF5 energy distribution is absent from this reactionSuite"
        )
        return None, None

    report.unsupportedNode(
        f"MT{mt}: MF5 arrived on a product already carrying a "
        f"{type(existingForm).__name__}, which is not a shape MF4 produces; "
        f"the energy distribution is absent from this reactionSuite"
    )
    return None, None
