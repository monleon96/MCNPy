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
    PoPs,
    Product,
    Q,
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
SUPPORTED_MF = (1, 2, 3, 4, 31, 33, 34)


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
        )
    }
    evaluationInfo = {
        name: getattr(mt451, attr, None) or ""
        for name, attr in (
            ("laboratory", "_laboratory"), ("authors", "_authors"),
            ("eval_date", "_eval_date"), ("reference", "_reference"),
            ("dist_date", "_dist_date"), ("revision_date", "_revision_date"),
            ("material_id", "_material_id"),
        )
    }

    provenance = EndfProvenance(
        mat=getattr(mt451, "_mat", None),
        awr=getattr(mt451, "atomic_weight_ratio", None),
        za=_za(mt451),
        headerFields=fields,
        evaluationInfo=evaluationInfo,
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

    MF1, MF2, MF3 and MF4 are read. The covariances are **not** part of a
    ``reactionSuite``: §25.1.1 makes ``covarianceSuite`` a root node in its own
    right, linked through ``externalFiles``, so MF31/33/34 go through
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

    mf2 = endf.mf.get(2) if hasattr(endf, "mf") else None
    if mf2 is not None and 151 in getattr(mf2, "mt", {}):
        suite.resonances, report = decodeMF2MT151(mf2.mt[151], report)

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
    for mf in sorted((present & set(SUPPORTED_MF)) - {1, 2, 3, 4}):
        report.unsupportedNode(
            f"MF{mf} is present and parsed; it is a covariance file, so it "
            f"belongs to the covarianceSuite (§25.1.1) and not to this "
            f"reactionSuite. Call decodeCovarianceSuite for it."
        )

    return suite, report


def _attachAngularDistribution(suite: ReactionSuite, mf4mt, mt: int,
                               report: ConversionReport) -> ConversionReport:
    """Hang one MF4 section on the neutron product of its reaction.

    GNDS puts a distribution on a *product* of an output channel, not on the
    reaction — which is why an MF4 section with no MF3 counterpart has nowhere
    to go, and is reported rather than dropped into an invented reaction.
    """
    from .angular import decodeMF4MT

    reaction = suite.reactionByENDF_MT(mt)
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
    product = Product(pid="n", label="n", provenance=provenance,
                      distribution=Distribution())
    product.distribution[EVAL_LABEL] = distribution
    channel.products.products.append(product)
    return report
