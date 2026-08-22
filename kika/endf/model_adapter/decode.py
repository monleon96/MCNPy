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
    Reaction,
    ReactionId,
    ReactionSuite,
    Regions1d,
    crossSectionAxes,
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
#: it is reactionSuite content that this adapter does not decode yet. MF6 is the
#: second of the same kind, added when its parser landed.
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
        pops.add(Nuclide(id=f"ZA{za}", Z=za // 1000, A=za % 1000))

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

    # MF5 is the case that broke the loop above. It is parsed, it is *not* a
    # covariance file, and it has no decode here yet — so it needs its own
    # notice. Sending the user to `decodeCovarianceSuite` for it, which is what
    # the old `- {1,2,3,4}` set did, was simply false.
    mf5 = endf.mf.get(5) if hasattr(endf, "mf") else None
    if mf5 is not None:
        report.unsupportedNode(
            "MF5 (energy distributions) is present and parsed by kika, but "
            "nothing decodes it into this reactionSuite; the distributions "
            "are absent from the products below. **The model slots exist** "
            "since GNDS phase 7b — what is missing is this adapter, the "
            "ENDF -> model direction, which that phase did not cover and no "
            "phase is scheduled for. The parsed sections are reachable as "
            "endf.mf[5] and are what the PFNS perturbation pipeline reads."
        )
        for mt in sorted(getattr(mf5, "mt", {})):
            for gap in getattr(mf5.mt[mt], "report_gaps", list)():
                report.unsupportedNode(gap)

    # MF6 is the same case as MF5 and needs the same notice. Its `report_gaps`
    # is worth reading even though every MF6 law *is* decoded: a product whose
    # LAW is -14 or -15 defers its distribution to MF14 or MF15, and those have
    # no parser — so the section can be read in full and still not have the
    # distribution in hand.
    mf6 = endf.mf.get(6) if hasattr(endf, "mf") else None
    if mf6 is not None:
        report.unsupportedNode(
            "MF6 (product energy-angle distributions) is present and parsed by "
            "kika, but nothing decodes it into this reactionSuite; the "
            "distributions are absent from the products below. **The model "
            "slots exist** since GNDS phase 7b — energyAngular, angularEnergy "
            "and KalbachMann are exactly MF6's LAW=1/LANG=1, LAW=7 and "
            "LAW=1/LANG=2 — so what is missing is this adapter and not a "
            "model node. The parsed sections are reachable as endf.mf[6]."
        )
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
