"""MF6 ↔ the §18 distribution forms that state P(E′,mu|E) for one *product*.

MF6 is where a modern evaluation puts everything MF4 and MF5 split in two. A
section is a HEAD and ``NK`` product subsections, each carrying its own yield
``y(E)`` and its own distribution law, and GNDS says the same thing one level
down: a distribution belongs to a ``product`` of an ``outputChannel``, not to
the reaction. So this adapter builds *products* where
:mod:`~kika.endf.model_adapter.angular` and :mod:`~kika.endf.model_adapter.energy`
decorate the one neutron their files describe.

**The mapping is measured, not chosen.** Every row below was checked product by
product against the FUDGE translation of the same library
(``ENDF-B-VIII.1-GNDS/``), and the two counts agree:

=========================  ==================================================
MF6                        §18 form
=========================  ==================================================
LAW=1 LANG=1, all ``NA=0``  ``uncorrelated(isotropic2d, XYs2d)``
LAW=1 LANG=1, any ``NA>0``  ``energyAngular(XYs3d)``
LAW=1 LANG=2                ``KalbachMann(f, r[, a])``
LAW=1 LANG=11-15            not modelled — no witness in five libraries
LAW=2                       ``angularTwoBody(XYs2d)``
LAW=3                       ``angularTwoBody(isotropic2d)``
LAW=4                       ``angularTwoBody(recoil href)``
LAW=5                       not modelled — 0 of 558 GNDS evaluations carry one
LAW=6                       ``uncorrelated(isotropic2d, NBodyPhaseSpace)``
LAW=7                       ``angularEnergy(XYs3d)``
LAW=0                       ``unspecified``
LAW<0                       **no product at all** — see below
=========================  ==================================================

**LANG=1 is not ``energyAngular``, and the counts are how we know.** The ENDF
census gives 18 738 LANG=1 products and the GNDS census only 2 948
``energyAngular``. The difference is ``NA=0``: a LAW=1 record with no angular
coefficients is a pure outgoing spectrum, and the evaluation says nothing about
angle. Fe-56 splits 104 / 10 on the ENDF side and 104 ``uncorrelated`` / 10
``energyAngular`` on the GNDS side; C-12 gives 17 / 0. Writing an
``energyAngular`` whose inner Legendre series is a single ``a_0`` would state a
correlation the file does not.

**A product whose LAW is negative is not a product of this model.** The
subsection carries no records of its own — it is a pointer saying "the
distribution is in File |LAW|" — and the evaluations that use it use it in
bulk: ENDF/B-VIII.1's U-235 MT18 declares NK=56, of which 40 are ``ZAP=0,
LIP=1..40, LAW=-15`` and 14 are ``ZAP=1, LIP=1..14, LAW=-5``. FUDGE emits two
products for that section, the two with ``LIP=0``. Giving one model ``product``
to each subsection would put 56 ``<product>`` nodes in a file no other library
writes, and ``LIP`` there is a *line index* rather than an excited state, so
even their names would be invented. They are declared in the report and kept in
the provenance, and that is all.

Once they are out, ``(ZAP, LIP)`` is unique within a section: measured over 296
sections and 808 products of eight tapes (Pu-239, U-235, U-238, Al-27, O-16,
C-12, Li-6, Be-9), **zero** repeats. That is what lets
:meth:`~kika.nuclear_data.model.output_channel.OutputChannel.ensureProduct` be
the primitive here, and it is what the encoder's lookup by pid rests on.

**What is kept and where.** The tables are the model's. ``JP``, ``LCT``, ``NK``,
and per subsection ``ZAP``, ``AWP``, ``LIP``, ``LAW``, the yield TAB1 and each
law's own scalars are ENDF bookkeeping with no GNDS counterpart, and they go to
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`'s ``headerFields``
under an ``"mf6"`` key — on the **reaction**, because MF6 is one section per MT
the way MF3 is, and its per-product fields belong in a list inside that block
rather than scattered over products that may not exist. The verbatim records of
a law kika does not model go there too, **as text**, for the reason
:mod:`~kika.endf.model_adapter.energy` gives: a list of strings is plain data,
and putting a parsed ``MF6Law`` in the model would be the accidental format
surface the model exists in order not to have.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ENDF_INT_TO_INTERPOLATION,
    EVAL_LABEL,
    AngularEnergy,
    AngularTwoBody,
    ConversionReport,
    EndfProvenance,
    EnergyAngular,
    Frame,
    Isotropic2d,
    KalbachMann,
    Legendre,
    Multiplicity,
    NBodyPhaseSpace,
    Regions1d,
    Uncorrelated,
    Unspecified,
    XYs1d,
    XYs2d,
    angularAxes,
    energyAngularAxes,
    energyAxes,
    fromEndfTab2,
    fromEndfTab3,
    kalbachMannAxes,
    multiplicityAxes,
    pidFromZA,
    toEndfTab2,
    toEndfTab3,
)

__all__ = ["decodeMF6MT", "encodeMF6MT", "recoilAngularHref", "frameForProduct"]


# ---------------------------------------------------------------------------
# xPaths
# ---------------------------------------------------------------------------

def recoilAngularHref(label: str) -> str:
    """Where a LAW=4 recoil's angular distribution actually lives.

    In two-body kinematics the residual's P(mu|E) is the ejectile's mirrored, so
    the evaluation states it once and the recoil points at it. **Relative**,
    unlike the absolute paths
    :mod:`~kika.endf.model_adapter.multiplicity` writes, and that is the
    distributed convention rather than a preference: every ``<recoil>`` in
    ENDF-B-VIII.1-GNDS is spelled exactly this way, and it has to be — the
    target is a sibling in the same output channel, and an absolute path would
    have to name the reaction, which the section does not know.
    """
    return (f"../../../../product[@label='{label}']"
            f"/distribution/angularTwoBody[@label='{EVAL_LABEL}']")


# ---------------------------------------------------------------------------
# The frame, which LCT=3 makes a per-product question
# ---------------------------------------------------------------------------

def frameForProduct(lct: int, zap: int) -> Frame:
    """§6.1's ``LCT`` → the frame *this* product's distribution is stated in.

    LCT=1 and 2 are the whole section's answer. **LCT=3 is not**: ENDF-6 added
    it for evaluations that give light particles in the centre of mass and heavy
    recoils in the laboratory, and the split is on the product's mass number,
    ``A <= 4``. It is not exotic — 979 of ENDF/B-VIII.1's 12 388 sections use it,
    C-12's MT5 among them, where the neutron, H1, H2 and He4 come out in the
    centre of mass and Li6 upwards in the laboratory.

    A photon has ``A = 0`` and therefore falls on the centre-of-mass side of a
    literal reading. That is odd physics and it is what the rule says and what
    FUDGE writes (C-12 MT5's photon product is ``productFrame="centerOfMass"``),
    so it is reproduced rather than special-cased: disagreeing with the
    distributed file here would be a silent, unexplainable difference.
    """
    if lct == 1:
        return Frame.lab
    if lct == 2:
        return Frame.centerOfMass
    if lct == 3:
        return Frame.centerOfMass if int(zap) % 1000 <= 4 else Frame.lab
    return Frame.centerOfMass


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _za(section) -> Optional[int]:
    """ZA, **rounded**. See :func:`kika.nuclear_data.model.provenance._asEndfInt`."""
    value = getattr(section, "zaid", None)
    if value is None:
        value = getattr(section, "_za", None)
    return int(round(float(value))) if value is not None else None


def _verbatimBody(product, mat, mf, mt, pad) -> List[str]:
    """Columns 1-66 of every record of *product* after its yield TAB1.

    Emitted and sliced rather than read off the law object, exactly as
    :func:`kika.endf.model_adapter.energy._verbatimBody` does it, so that one
    function serves a law kika models and a law it does not. The line numbers
    are throwaway — every emitter re-stamps MAT/MF/MT and the running sequence.
    """
    header, _ = product.emit_header(mat, mf, mt, 1, pad)
    whole, _ = product.emit(mat, mf, mt, 1, pad)
    return [line[:66] for line in whole[len(header):]]


def _productRecord(product, index: int, pid: Optional[str],
                   label: Optional[str], modelled: bool, mat, mt, pad) -> dict:
    """One subsection's fields, as primitives.

    ``label`` is the model product this subsection became, or ``None`` when it
    became none — a negative LAW. It is what the encoder looks the form up by
    (``label`` and not ``pid``, because two subsections can share a pid), and
    it is stored rather than recomputed so that a change to
    :func:`~kika.nuclear_data.model.pops.pidFromZA` cannot silently re-point an
    existing provenance at a different product.
    """
    record = {
        "index": index,
        "zap": float(product.zap),
        "awp": float(product.awp),
        "lip": int(product.lip),
        "law": int(product.law),
        "y_interp": [(int(nbt), int(code)) for nbt, code in product.y_interp],
        "y_energies": [float(v) for v in product.y_energies],
        "y_values": [float(v) for v in product.y_values],
        "pid": pid,
        "label": label,
        "modelled": bool(modelled),
    }
    if modelled:
        record["law_fields"] = _lawFields(product.law_data)
    else:
        record["raw_lines"] = _verbatimBody(product, mat, 6, mt, pad)
    return record


def _lawFields(body) -> dict:
    """The law's own scalars, for a body that *did* reach the model.

    Only what the §18 form cannot state. The tables are not here — the encoder
    rebuilds them from the model, which is the whole point of the round trip —
    and neither is anything derivable from them: the incident grid and the
    TAB2's ``(NBT, INT)`` pairs come back out of the form through
    :func:`~kika.nuclear_data.model.functions.higher.toEndfTab2`, exactly as
    MF5's encoder gets them.

    ``ND`` is the one that has to be here rather than derived. "The first ND
    outgoing points of this node are discrete lines" is a statement the model
    has no node for — 45 017 of the library's 329 630 LAW=1 records make it —
    and without it a re-emitted section turns a line spectrum into a continuum.
    """
    if body.law == 1:
        return {"lang": int(body.lang), "lep": int(body.lep),
                "nd": [int(v) for v in body.nd], "na": [int(v) for v in body.na]}
    if body.law == 2:
        # LANG is per node here. It is derivable from the child's class — a
        # Legendre means 0 and an XYs1d 10+INT — and it is kept anyway, because
        # deriving it would make the file's own statement a consequence of how
        # this adapter chose to model it.
        return {"lang": [int(v) for v in body.lang]}
    if body.law == 6:
        # APSX is the total mass of the N products in units of the neutron
        # mass, and the model deliberately does not carry it: see `_phaseSpace`.
        return {"apsx": float(body.apsx), "npsx": int(body.npsx)}
    if body.law == 7:
        return {}
    return {}


def _headerProvenance(mf6mt, records: List[dict]) -> EndfProvenance:
    pad = mf6mt.pad
    return EndfProvenance(
        mat=getattr(mf6mt, "_mat", None),
        awr=getattr(mf6mt, "_awr", None),
        za=_za(mf6mt),
        headerFields={
            "mf6": {
                # **MF6's own MAT, ZA and AWR.** One reaction can carry more
                # than one file's provenance and the three fields are not
                # shared: two ENDF/B-VIII.1 tapes state a different AWR in MF5
                # than in MF4, and a byte-exact gate fails on the sixth digit.
                "mat": getattr(mf6mt, "_mat", None),
                "za": _za(mf6mt),
                "awr": getattr(mf6mt, "_awr", None),
                # JP is zero on 12 387 of 12 388 sections and U-235's MT18
                # writes 11. kika stores and re-emits it and does not interpret
                # it; hard-coding the zero loses what MT18 claims its products
                # are.
                "jp": int(getattr(mf6mt, "_jp", 0) or 0),
                "lct": int(getattr(mf6mt, "_lct", 2)),
                "nk": mf6mt.num_products,
                # The padding convention this section's writer used, probed at
                # parse time. Without it the re-emitted section differs from the
                # source on every short record.
                "pad": {"pairs": pad.pairs, "values": pad.values,
                        "interp": pad.interp},
                "products": records,
            },
        },
    )


# ---------------------------------------------------------------------------
# The yield
# ---------------------------------------------------------------------------

def _multiplicityOf(product, axes) -> Multiplicity:
    """A product's ``y(E)`` TAB1 → §17.3's multiplicity.

    A :class:`Regions1d` when the record really has more than one interpolation
    region, and the single :class:`XYs1d` inside it when it does not — the rule
    :func:`kika.endf.model_adapter.energy._spectrumAt` states and ``gnds.xsd``
    imposes, since ``minOccurs="2"`` makes a one-region ``regions1d`` invalid.
    """
    pairs = [(int(nbt), int(code)) for nbt, code in product.y_interp]
    x = np.asarray(product.y_energies, dtype=float)
    if not pairs and x.size:
        pairs = [(int(x.size), 2)]
    function = Regions1d.fromEndfRegions(
        x, np.asarray(product.y_values, dtype=float), pairs,
        axes=axes, label=EVAL_LABEL,
    )
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
        function.axes = axes
        function.label = EVAL_LABEL
    return Multiplicity(form=function)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decodeMF6MT(mf6mt, report: Optional[ConversionReport] = None):
    """One MF6/MT section → ``(entries, provenance, report)``.

    ``entries`` is a list of ``(label, pid, multiplicity, distribution)``, in
    file order, one per subsection that becomes a model product.
    ``distribution`` is ``None`` where the law has no §18 form kika builds; the
    product still exists, because its *yield* is data whatever its law.

    ``label`` and ``pid`` differ only where one section names a particle twice —
    see :func:`_repeatedProductMessage` — and it is the **label** the encoder
    looks a form up by, because §17.2.1 is what distinguishes the two nodes.

    The provenance is returned in every case, and it is the whole section: that
    is what lets :func:`encodeMF6MT` write back a thing the model never held.
    """
    report = report if report is not None else ConversionReport()
    mt = mf6mt.number
    lct = int(getattr(mf6mt, "_lct", 2))
    mat = getattr(mf6mt, "_mat", 0) or 0
    pad = mf6mt.pad

    if lct not in (1, 2, 3):
        report.warn(
            f"MF6/MT{mt}: LCT={lct} is none of 1 (lab), 2 (centre of mass) or "
            f"3 (centre of mass for A<=4); every product frame is left as "
            f"centre of mass and LCT is kept verbatim, so the tape is unaffected"
        )

    yieldAxes = multiplicityAxes()

    entries: List[Tuple[str, str, Multiplicity, Optional[object]]] = []
    records: List[dict] = []
    seen: Dict[str, int] = {}
    ejectile: Optional[str] = None
    for index, product in enumerate(mf6mt.products):
        if product.law < 0:
            report.unsupportedNode(_deferralMessage(mt, index, product))
            records.append(_productRecord(product, index, None, None, False,
                                          mat, mt, pad))
            continue

        pid = pidFromZA(product.za, product.lip)
        label = pid if pid not in seen else f"{pid}__{index}"
        if pid in seen:
            report.approximated(_repeatedProductMessage(mt, index, product,
                                                        pid, label, seen[pid]))
        else:
            seen[pid] = index

        frame = frameForProduct(lct, product.za)
        distribution, modelled, report = _distributionOf(
            product, index, mt, frame, ejectile, report
        )
        if product.law in (2, 3, 5) and distribution is not None:
            # ENDF-6 §6.2.4: a LAW=4 subsection is "the recoil of the two-body
            # product given above", so the target is the most recent product
            # that stated one, and it is read here rather than guessed later.
            # **Only when that product reached the model**, so a recoil never
            # points at a node this suite does not contain -- which is what
            # keeps every commit of this adapter's slicing internally valid.
            ejectile = label
        entries.append((label, pid, _multiplicityOf(product, yieldAxes),
                        distribution))
        records.append(_productRecord(product, index, pid, label, modelled,
                                      mat, mt, pad))

    return entries, _headerProvenance(mf6mt, records), report


def _deferralMessage(mt: int, index: int, product) -> str:
    """Why a negative-LAW subsection produces no product, in its own terms."""
    where = {4: "MF4", 5: "MF5", 14: "MF14", 15: "MF15"}.get(
        abs(product.law), f"MF{abs(product.law)}")
    reachable = abs(product.law) in (4, 5)
    return (
        f"MF6/MT{mt} product {index} (ZAP={product.za}, LIP={product.lip}) has "
        f"LAW={product.law}: it states no distribution of its own and defers to "
        f"{where}, which kika {'reads' if reachable else 'does not read'}. It "
        f"is not given a product of its own — its LIP is a line index and not "
        f"an excited state, and ENDF/B-VIII.1's U-235 MT18 states 54 of these "
        f"beside two real products. The subsection is kept in the provenance, "
        f"so the section still comes back"
    )


def _repeatedProductMessage(mt: int, index: int, product, pid: str,
                            label: str, first: int) -> str:
    """Why two subsections with one pid get two products with two labels.

    It happens, and the neutron sublibrary is why it looked as though it did
    not: over 296 sections and 808 non-deferring products of eight neutron
    tapes there is not one repeat. The charged-particle sublibrary has three in
    nine committed fixtures — ``a+He4`` and ``d+H2`` MT2, where elastic
    scattering off an identical particle makes the ejectile (LAW=5) and its
    recoil (LAW=4) the same nuclide, and ``t+Li7`` MT24, which emits two He4.

    §17.2.1 admits it: ``pid`` names the particle and ``label`` distinguishes
    the nodes, which is why
    :class:`~kika.nuclear_data.model.output_channel.Products` answers
    ``byPid`` with a list. **The label is this adapter's own**: no GNDS
    translation of a charged-particle evaluation was available to copy a
    convention from, so the subsection's own ordinal is used, which is stable,
    unique and traceable back to the record it came from.
    """
    return (
        f"MF6/MT{mt} products {first} and {index} are both ZAP={product.za} "
        f"LIP={product.lip}, so they are two nodes for one particle: the "
        f"second is labelled {label!r} rather than {pid!r}. §17.2.1 allows it "
        f"— identical-particle elastic scattering and multi-alpha emission "
        f"both produce it — but the label is kika's ordinal and not a spelling "
        f"read from the file"
    )


def _distributionOf(product, index: int, mt: int, frame: Frame,
                    ejectile: Optional[str], report: ConversionReport):
    """One subsection's law → ``(form, modelled, report)``.

    ``modelled`` is whether the *body* came through the model. It is not the
    same question as "is there a form": LAW=0, 3 and 4 have a §18 form and no
    body at all, so they are modelled with nothing to keep verbatim, while a
    LAW=5 has a body kika does not interpret and keeps as bytes.
    """
    law = product.law

    if law == 0:
        return Unspecified(productFrame=frame), True, report

    if law == 3:
        # Isotropic in the centre of mass, stated by the LAW number alone. The
        # same node MF4's LTT=0 produces, and it goes inside an angularTwoBody
        # rather than bare: §18.1.1 admits `isotropic2d` only under
        # `angularTwoBody`, which is the defect `gnds` commit 2 of phase 7b-0
        # fixed on the MF4 side.
        return (AngularTwoBody(angular=Isotropic2d(productFrame=frame),
                               productFrame=frame),
                True, report)

    if law == 4:
        # The recoil of a two-body break-up: its P(mu|E) is the ejectile's,
        # mirrored, so the evaluation states it once and points at it.
        if ejectile is None:
            report.lost(
                f"MF6/MT{mt} product {index} is LAW=4, the recoil of a two-body "
                f"product, and no LAW=2 or LAW=3 subsection precedes it in this "
                f"section — so there is nothing for the recoil to point at. The "
                f"product and its multiplicity are here and its distribution is "
                f"not; a <recoil> with no href is a node the schema rejects, and "
                f"inventing a target would be worse than saying so"
            )
            return None, True, report
        return (AngularTwoBody(recoilHref=recoilAngularHref(ejectile),
                               productFrame=frame),
                True, report)

    if law == 1 and product.law_data.lang == 2:
        return _kalbachMann(product.law_data, index, mt, frame, report)

    if law == 1 and product.law_data.lang == 1:
        return _continuumLegendre(product.law_data, index, mt, frame, report)

    if law == 2:
        return _twoBody(product.law_data, index, mt, frame, report)

    if law == 6:
        return _phaseSpace(product.law_data, frame), True, report

    if law == 7:
        return _labAngleEnergy(product.law_data, index, mt, frame, report)

    report.unsupportedNode(
        f"MF6/MT{mt} product {index} is {product.law_data.describe()}, and "
        f"nothing maps it to a §18 form yet; the product and its multiplicity "
        f"are in this reactionSuite and its distribution is not. The "
        f"subsection's records are kept verbatim in the provenance, so the "
        f"section still comes back byte for byte"
    )
    return None, False, report




# ---------------------------------------------------------------------------
# LAW=2, 6, 7 — the two-body angle, the phase space and the lab surface
# ---------------------------------------------------------------------------

def _twoBody(body, index: int, mt: int, frame: Frame, report: ConversionReport):
    """LAW=2 → §18.2's ``angularTwoBody``. The same node MF4 produces.

    ``LANG`` is **per node** here, where LAW=1 states it once for the whole
    product, so one ``XYs2d`` can hold Legendre children and tabulated ones side
    by side. Nothing in GNDS objects — ``xData_XYs2d`` takes a list of 1-d
    functions and does not require them to be the same kind — and no measured
    section mixes them, but the loop is written per node rather than per product
    because the file is.
    """
    functions = []
    for k, energy in enumerate(body.incident_energies):
        lang = body.lang[k]
        if lang == 0:
            functions.append(_twoBodyLegendreAt(body, k, energy))
        elif lang in (12, 14):
            functions.append(_twoBodyTabulatedAt(body, k, energy, lang))
        else:
            report.unsupportedNode(
                f"MF6/MT{mt} product {index} node {k} is LAW=2 LANG={lang}, "
                f"which ENDF-6 §6.2.2 does not define — it gives 0 for Legendre "
                f"coefficients and 12/14 for a tabulated (mu, f). The whole "
                f"distribution is absent from this reactionSuite and the "
                f"subsection is kept verbatim"
            )
            return None, False, report

    axes = angularAxes()
    for function in functions:
        function.axes = axes
    angular = fromEndfTab2(functions, body.tab2_interp, axes=axes)
    return AngularTwoBody(angular=angular, productFrame=frame), True, report


def _twoBodyLegendreAt(body, k: int, energy: float) -> Legendre:
    """``A_1 .. A_NL`` → a Legendre series. ``A_0 = 1`` is implicit on the tape.

    The same rule :func:`kika.endf.model_adapter.angular._legendreAt` follows
    for MF4, and for the same reason: the two-body angular distribution is
    normalised, so ENDF does not write the leading coefficient and GNDS does.
    """
    row = np.asarray(body.legendre(k), dtype=float)
    coefficients = np.empty(row.size + 1, dtype=float)
    coefficients[0] = 1.0
    coefficients[1:] = row
    return Legendre(coefficients=coefficients, outerDomainValue=float(energy),
                    index=k)


def _twoBodyTabulatedAt(body, k: int, energy: float, lang: int) -> XYs1d:
    """``(mu, f)`` → a tabulated P(mu). ``LANG`` carries the interpolation.

    ENDF-6 spells it as ``LANG = 10 + INT``, so 12 is lin-lin and 14 is
    log-lin. 12 is the only one with a witness on this machine — 503 nodes in
    the charged-particle sublibrary, nine of them in ``micro_t_li7_mf6.endf``
    MT50 — and 14 has 490 in ENDF/B-VIII.1's neutron tapes.
    """
    mu, f = body.tabulated(k)
    return XYs1d(xs=mu, ys=f,
                 interpolation=ENDF_INT_TO_INTERPOLATION.get(lang - 10, "lin-lin"),
                 outerDomainValue=float(energy), index=k)


def _phaseSpace(body, frame: Frame) -> Uncorrelated:
    """LAW=6 → ``uncorrelated`` with §18.3's ``NBodyPhaseSpace`` as its energy.

    Five occurrences in ENDF/B-VIII.1, three of them Li-6's MT41, and the
    distributed GNDS translation of that section writes exactly this: an
    ``isotropic2d`` angular half and an ``NBodyPhaseSpace`` energy half with
    ``numberOfProducts`` and no mass.

    **``APSX`` stays in the provenance and ``mass`` stays ``None``.** ENDF gives
    the total mass of the N products in units of the neutron mass and
    :class:`~kika.nuclear_data.model.distributions.NBodyPhaseSpace` states it as
    a :class:`~kika.nuclear_data.model.quantities.PhysicalQuantity`, so filling
    it means choosing a neutron mass and writing a number the evaluator did not.
    The §4.1 scattering radius made that mistake expensive once already.
    """
    return Uncorrelated(
        angular=Isotropic2d(productFrame=frame),
        energy=NBodyPhaseSpace(numberOfProducts=int(body.npsx)),
        productFrame=frame,
    )


def _labAngleEnergy(body, index: int, mt: int, frame: Frame,
                    report: ConversionReport):
    """LAW=7 → §18.5's ``angularEnergy``. Two occurrences in the whole library.

    Both are Be-9's MT16, and both are in ``micro_be9_mf6.endf``. It is the one
    law already nested three deep on the ENDF side — a TAB2 over incident
    energy whose nodes are TAB2s over cosine whose nodes are TAB1s over outgoing
    energy — so the mapping is a relabelling rather than a reshape.

    **``angularEnergy`` and not ``energyAngular``**, and the two share a
    complexType exactly: the element name is the only thing in the file that
    says which variable is outermost, and LAW=7 puts mu outside E'. Writing the
    mirror would produce a document that validates and states the wrong physics.
    """
    axes = energyAngularAxes("mu")
    nodes = []
    for k, block in enumerate(body.blocks):
        functions = [
            _labTableAt(angle, m)
            for m, angle in enumerate(block.angles)
        ]
        node = fromEndfTab2(functions, block.mu_interp)
        node.outerDomainValue = float(block.energy)
        node.index = k
        nodes.append(node)

    return (AngularEnergy(xys3d=fromEndfTab3(nodes, body.tab2_interp, axes=axes),
                          productFrame=frame),
            True, report)


def _labTableAt(angle, m: int):
    """One LAW=7 cosine: the TAB1 of ``f(E')`` at fixed mu.

    Collapsed to its single :class:`XYs1d` when the record has one region, for
    the ``minOccurs="2"`` reason that governs every other TAB1 in this adapter.
    """
    pairs = [(int(nbt), int(code)) for nbt, code in angle.interp]
    x = np.asarray(angle.e_out, dtype=float)
    if not pairs and x.size:
        pairs = [(int(x.size), 2)]
    function = Regions1d.fromEndfRegions(x, np.asarray(angle.f, dtype=float), pairs)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
    function.outerDomainValue = float(angle.mu)
    function.index = m
    return function


# ---------------------------------------------------------------------------
# LAW=1 LANG=1 — §18.3 uncorrelated when NA=0, §18.4 energyAngular when not
# ---------------------------------------------------------------------------

def _continuumLegendre(body, index: int, mt: int, frame: Frame,
                       report: ConversionReport):
    """LAW=1 LANG=1 → ``uncorrelated`` or ``energyAngular``, and ``NA`` decides.

    **The split is measured rather than chosen**, and it is why "LANG=1 is
    energyAngular" is wrong: the ENDF census counts 18 738 LANG=1 products and
    the GNDS census only 2 948 ``energyAngular``. A record with ``NA=0`` carries
    ``(E', f0)`` and nothing else — an outgoing spectrum with no statement about
    angle at all — and the distributed translation writes it as §18.3's
    ``uncorrelated`` with an isotropic angular half. Fe-56 splits 104 NA=0 / 10
    NA>0 on the ENDF side and 104 ``uncorrelated`` / 10 ``energyAngular`` on the
    GNDS side; C-12's seventeen NA=0 products give seventeen ``uncorrelated``
    and no ``energyAngular``.

    An ``energyAngular`` whose inner Legendre series is a lone ``a_0`` would say
    the evaluation gave a correlated distribution that happens to be isotropic.
    It did not: it gave no angular information.

    **``NA`` is per node**, so a product may state angle at some incident
    energies and not others. That is still one ``energyAngular``: the nodes with
    ``NA=0`` get a one-coefficient Legendre, which *there* is a real statement —
    the file wrote a value for ``a_0`` and none for the rest at that energy —
    rather than the absence the all-zero case is.
    """
    interpolation = ENDF_INT_TO_INTERPOLATION.get(body.lep, "lin-lin")

    if any(nd for nd in body.nd):
        report.lost(_discreteLinesMessage(mt, index, body))

    if not any(body.na):
        axes = energyAxes()
        functions = [
            _spectrumAt(body, k, energy, interpolation, axes)
            for k, energy in enumerate(body.incident_energies)
        ]
        energy = fromEndfTab2(functions, body.tab2_interp, axes=axes)
        return (Uncorrelated(angular=Isotropic2d(productFrame=frame),
                             energy=energy, productFrame=frame),
                True, report)

    axes = energyAngularAxes("energy_out")
    nodes = [
        _legendreSurfaceAt(body, k, energy, interpolation, axes)
        for k, energy in enumerate(body.incident_energies)
    ]
    return (EnergyAngular(xys3d=fromEndfTab3(nodes, body.tab2_interp, axes=axes),
                          productFrame=frame),
            True, report)


def _spectrumAt(body, k: int, energy: float, interpolation, axes) -> XYs1d:
    """One ``NA=0`` node → P(E'|E) at one incident energy."""
    table = body.block(k)
    return XYs1d(xs=table[:, 0], ys=table[:, 1], interpolation=interpolation,
                 axes=axes, outerDomainValue=float(energy), index=k)


def _legendreSurfaceAt(body, k: int, energy: float, interpolation, axes) -> XYs2d:
    """One ``NA>0`` node → P(mu|E,E') over the whole outgoing grid.

    The inner functions are :class:`Legendre` and the outer ``XYs2d`` carries
    ``LEP`` — the interpolation along the *outgoing* axis, which is the one this
    container walks. The coefficients are the record's ``f_0 .. f_NA`` verbatim:
    unlike MF4, where ``a_0 = 1`` is implicit because the distribution is
    normalised, LAW=1 writes ``f_0`` because it is the spectrum's magnitude at
    that outgoing energy.
    """
    table = body.block(k)
    return XYs2d(
        function1ds=[
            Legendre(coefficients=table[j, 1:], outerDomainValue=float(table[j, 0]),
                     index=j)
            for j in range(table.shape[0])
        ],
        interpolation=interpolation,
        outerDomainValue=float(energy),
        index=k,
    )


def _discreteLinesMessage(mt: int, index: int, body) -> str:
    """``ND>0``: the first ND outgoing points are lines, and the model cannot say so."""
    return (
        f"MF6/MT{mt} product {index} is LAW=1 with ND>0 at "
        f"{sum(1 for nd in body.nd if nd)} of {len(body.nd)} incident energies: "
        f"those outgoing points are discrete lines and the model carries them "
        f"as ordinary points of the spectrum. ND is kept in the provenance, so "
        f"the ENDF section comes back; a GNDS file written from this suite "
        f"would not say which points are lines. 45 017 of the library's 329 630 "
        f"LAW=1 records make the statement"
    )


# ---------------------------------------------------------------------------
# LAW=1 LANG=2 — §18.6 KalbachMann
# ---------------------------------------------------------------------------

def _kalbachMann(body, index: int, mt: int, frame: Frame,
                 report: ConversionReport):
    """LAW=1 LANG=2 → :class:`KalbachMann`. 3 730 in ENDF/B-VIII.1.

    The record is ``(E', f0, r)`` per outgoing point, and ``(E', f0, r, a)``
    where the evaluator gave the slope too. GNDS states the same three as three
    ``XYs2d`` in schema order, each a bare wrapper holding one primary function
    with its own axes — so this is a transposition of the columns and nothing
    more.

    **``a`` is written only where ``NA >= 2``.** ENDF-6 lets an evaluator give
    only the pre-compound fraction and leave the slope to Kalbach's systematics,
    which is what all 3 730 of the library's nodes do — the census counted zero
    ``<a>``. ``MF6LawContinuum.kalbach`` returns NaN rather than 0.0 for exactly
    this case, because a zero slope is a physically meaningful isotropic value,
    and putting those NaNs in the model would be writing the systematics'
    answer as though the file had stated it.
    """
    if len(set(body.na)) > 1:
        # Not seen, and it would make `a` exist at some incident energies and
        # not others — a shape the schema has no way to say.
        report.unsupportedNode(
            f"MF6/MT{mt} product {index} is LAW=1 LANG=2 whose NA varies along "
            f"the incident grid ({sorted(set(body.na))}); a KalbachMann states "
            f"one <a> for the whole node or none, so the distribution is absent "
            f"from this reactionSuite and the subsection is kept verbatim"
        )
        return None, False, report

    if any(nd for nd in body.nd):
        report.lost(_discreteLinesMessage(mt, index, body))

    columns = {"f": 1, "r": 2}
    if body.na and body.na[0] >= 2:
        columns["a"] = 3

    halves = {}
    for component, column in columns.items():
        axes = kalbachMannAxes(component)
        functions = [
            _columnAt(body, k, column, energy, axes)
            for k, energy in enumerate(body.incident_energies)
        ]
        halves[component] = fromEndfTab2(functions, body.tab2_interp, axes=axes)

    return (KalbachMann(productFrame=frame, **halves), True, report)


def _columnAt(body, k: int, column: int, energy: float, axes) -> XYs1d:
    """One column of one LAW=1 node → a function over E'.

    An :class:`XYs1d` and never a ``regions1d``: the LIST body of a LAW=1 node
    has **one** interpolation rule for the whole outgoing grid, ``LEP``, and no
    region structure to lose. C-12's MT5 writes ``LEP=1``, a histogram, which is
    what the distributed GNDS file spells ``interpolation="flat"``.
    """
    table = body.block(k)
    return XYs1d(
        xs=table[:, 0], ys=table[:, column],
        interpolation=ENDF_INT_TO_INTERPOLATION.get(body.lep, "lin-lin"),
        axes=axes, outerDomainValue=float(energy), index=k,
    )


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _restoreProduct(record: dict, forms, mt: int):
    """One provenance record → the ``MF6Product`` it was read from."""
    from kika.endf.classes.mf6.laws import MF6LawNoBody, MF6LawElsewhere, MF6LawRaw
    from kika.endf.classes.mf6.products import MF6Product

    law = int(record["law"])
    if law < 0:
        body = MF6LawElsewhere(law=law)
    elif law in (0, 3, 4):
        body = MF6LawNoBody(law=law)
    elif record["modelled"]:
        body = _restoreLaw(record, forms, mt)
    else:
        body = MF6LawRaw(law=law, raw_lines=list(record.get("raw_lines", [])))

    return MF6Product(
        zap=record["zap"],
        awp=record["awp"],
        lip=int(record["lip"]),
        y_interp=[(int(a), int(b)) for a, b in record["y_interp"]],
        y_energies=list(record["y_energies"]),
        y_values=list(record["y_values"]),
        law_data=body,
    )


def _restoreLaw(record: dict, form, mt: int):
    """One modelled law body, rebuilt from its §18 form and its provenance."""
    law = int(record["law"])
    fields = record.get("law_fields") or {}
    index = record["index"]

    if law == 1 and fields.get("lang") == 2:
        if not isinstance(form, KalbachMann):
            raise ValueError(
                f"MF6/MT{mt} product {index}: the provenance says LAW=1 LANG=2, "
                f"which decodes to a KalbachMann, and it was handed "
                f"{'nothing' if form is None else type(form).__name__}"
            )
        return _restoreKalbachMann(form, fields, index, mt)

    if law == 1 and fields.get("lang") == 1:
        if isinstance(form, Uncorrelated):
            nodes, pairs = toEndfTab2(form.energy)
            rows = [_interleave(node.xs, node.ys) for node in nodes]
        elif isinstance(form, EnergyAngular):
            nodes, pairs = toEndfTab3(form.xys3d)
            rows = [_legendreRows(node, index, mt) for node in nodes]
        else:
            raise ValueError(
                f"MF6/MT{mt} product {index}: the provenance says LAW=1 LANG=1, "
                f"which decodes to an uncorrelated when every NA is zero and an "
                f"energyAngular otherwise, and it was handed "
                f"{'nothing' if form is None else type(form).__name__}"
            )
        return _continuumBody(fields, nodes, pairs, rows, index, mt)

    if law == 2:
        if not isinstance(form, AngularTwoBody) or form.angular is None:
            raise ValueError(
                f"MF6/MT{mt} product {index}: the provenance says LAW=2, which "
                f"decodes to an angularTwoBody carrying an angular function, "
                f"and it was handed "
                f"{'nothing' if form is None else type(form).__name__}"
            )
        return _restoreTwoBody(form, fields, index, mt)

    if law == 6:
        return _restorePhaseSpace(form, fields, index, mt)

    if law == 7:
        if not isinstance(form, AngularEnergy) or form.xys3d is None:
            raise ValueError(
                f"MF6/MT{mt} product {index}: the provenance says LAW=7, which "
                f"decodes to an angularEnergy, and it was handed "
                f"{'nothing' if form is None else type(form).__name__}"
            )
        return _restoreLabAngleEnergy(form, index, mt)

    raise NotImplementedError(
        f"MF6/MT{mt} product {index}: the provenance says LAW={law} came "
        f"through the model, and this encoder has no branch to put it back. "
        f"decode and encode land in the same commit precisely so this cannot "
        f"be reached"
    )


def _restoreTwoBody(form: AngularTwoBody, fields: dict, index: int, mt: int):
    """An ``angularTwoBody`` → the LAW=2 body it was read from."""
    from kika.endf.classes.mf6.laws import MF6LawTwoBody

    functions, pairs = toEndfTab2(form.angular)
    langs = [int(v) for v in (fields.get("lang") or [])]
    if len(langs) != len(functions):
        raise ValueError(
            f"MF6/MT{mt} product {index}: the angularTwoBody has "
            f"{len(functions)} incident energies and the provenance kept LANG "
            f"for {len(langs)}"
        )

    values = []
    for k, (function, lang) in enumerate(zip(functions, langs)):
        if lang == 0:
            if not isinstance(function, Legendre):
                raise ValueError(
                    f"MF6/MT{mt} product {index} node {k}: the provenance says "
                    f"LANG=0, Legendre coefficients, and the model carries a "
                    f"{type(function).__name__}"
                )
            # A_0 is 1 by normalisation and is not on the tape.
            values.append([float(c) for c in
                           np.asarray(function.coefficients, dtype=float)[1:]])
        else:
            if isinstance(function, Legendre):
                raise ValueError(
                    f"MF6/MT{mt} product {index} node {k}: the provenance says "
                    f"LANG={lang}, a tabulated (mu, f), and the model carries a "
                    f"Legendre series"
                )
            mu, f, _regions = function.toEndfRegions()
            values.append(_interleave(mu, f))

    return MF6LawTwoBody(
        law=2, tab2_interp=pairs,
        incident_energies=[float(f.outerDomainValue) for f in functions],
        lang=langs, values=values,
    )


def _restorePhaseSpace(form, fields: dict, index: int, mt: int):
    """An ``uncorrelated`` holding an ``NBodyPhaseSpace`` → the LAW=6 body."""
    from kika.endf.classes.mf6.laws import MF6LawPhaseSpace

    energy = getattr(form, "energy", None)
    if not isinstance(energy, NBodyPhaseSpace):
        raise ValueError(
            f"MF6/MT{mt} product {index}: the provenance says LAW=6, which "
            f"decodes to an uncorrelated whose energy half is an "
            f"NBodyPhaseSpace, and it was handed "
            f"{'nothing' if form is None else type(form).__name__}"
        )
    # NPSX from the model and APSX from the provenance, because the model
    # carries the one and deliberately not the other.
    return MF6LawPhaseSpace(law=6, apsx=float(fields.get("apsx", 0.0)),
                            npsx=int(energy.numberOfProducts))


def _restoreLabAngleEnergy(form: AngularEnergy, index: int, mt: int):
    """An ``angularEnergy`` → the LAW=7 body, three levels back down."""
    from kika.endf.classes.mf6.laws import (LawSevenAngle, LawSevenEnergy,
                                            MF6LawLabAngleEnergy)

    nodes, pairs = toEndfTab3(form.xys3d)
    blocks = []
    for node in nodes:
        functions, muPairs = toEndfTab2(node)
        angles = []
        for function in functions:
            e_out, f, regions = function.toEndfRegions()
            angles.append(LawSevenAngle(
                mu=float(function.outerDomainValue),
                interp=[(int(a), int(b)) for a, b in regions],
                e_out=[float(v) for v in e_out], f=[float(v) for v in f],
            ))
        blocks.append(LawSevenEnergy(energy=float(node.outerDomainValue),
                                     mu_interp=muPairs, angles=angles))

    return MF6LawLabAngleEnergy(law=7, tab2_interp=pairs, blocks=blocks)



def _interleave(xs, ys) -> List[float]:
    """``[x0, y0, x1, y1, ...]`` — a LAW=1 ``NA=0`` node's LIST body."""
    table = np.empty(len(xs) * 2, dtype=float)
    table[0::2] = np.asarray(xs, dtype=float)
    table[1::2] = np.asarray(ys, dtype=float)
    return [float(v) for v in table]


def _legendreRows(node, index: int, mt: int) -> List[float]:
    """One ``XYs2d`` of an ``energyAngular`` → its LIST body, ``E'`` first."""
    rows: List[float] = []
    for function in node.function1ds:
        if not isinstance(function, Legendre):
            raise ValueError(
                f"MF6/MT{mt} product {index}: an energyAngular decoded from "
                f"LAW=1 LANG=1 has Legendre children, and this node carries a "
                f"{type(function).__name__}. ENDF writes the coefficients and "
                f"has no record shape for a tabulated inner function here"
            )
        rows.append(float(function.outerDomainValue))
        rows.extend(float(c) for c in function.coefficients)
    return rows


def _continuumBody(fields: dict, nodes, pairs, rows, index: int, mt: int):
    """The shared tail of both LANG=1 shapes: check ND/NA, build the body."""
    from kika.endf.classes.mf6.laws import MF6LawContinuum

    nd = list(fields.get("nd") or [])
    na = list(fields.get("na") or [])
    if len(nd) != len(nodes) or len(na) != len(nodes):
        raise ValueError(
            f"MF6/MT{mt} product {index}: the model has {len(nodes)} incident "
            f"energies and the provenance kept ND/NA for {len(nd)}/{len(na)}. "
            f"ND says which outgoing points are discrete lines and the model "
            f"does not carry it, so a node added or removed in the model "
            f"cannot be written back"
        )
    for k, row in enumerate(rows):
        width = na[k] + 2
        if len(row) % width:
            raise ValueError(
                f"MF6/MT{mt} product {index} node {k}: the provenance says "
                f"NA={na[k]}, so the LIST is groups of {width}, and the model "
                f"gives {len(row)} values"
            )

    return MF6LawContinuum(
        law=1, lang=int(fields["lang"]), lep=int(fields["lep"]),
        tab2_interp=pairs,
        incident_energies=[float(node.outerDomainValue) for node in nodes],
        nd=nd, na=na, values=rows,
    )


def _restoreKalbachMann(form: KalbachMann, fields: dict, index: int, mt: int):
    """A :class:`KalbachMann` → the LAW=1 LANG=2 body it was read from."""
    from kika.endf.classes.mf6.laws import MF6LawContinuum

    functions, pairs = toEndfTab2(form.f)
    columns = [functions]
    for half in (form.r, form.a):
        if half is not None:
            columns.append(toEndfTab2(half)[0])

    nd = list(fields.get("nd") or [])
    na = list(fields.get("na") or [])
    if len(nd) != len(functions) or len(na) != len(functions):
        raise ValueError(
            f"MF6/MT{mt} product {index}: the KalbachMann has "
            f"{len(functions)} incident energies and the provenance kept ND/NA "
            f"for {len(nd)}/{len(na)}. ND says which outgoing points are "
            f"discrete lines and the model does not carry it, so a node added "
            f"or removed in the model cannot be written back"
        )
    if na and na[0] + 2 != len(columns) + 1:
        raise ValueError(
            f"MF6/MT{mt} product {index}: the provenance says NA={na[0]}, so "
            f"each outgoing point is E' plus {na[0] + 1} values, and the "
            f"KalbachMann carries {len(columns)}"
        )

    values = []
    for k in range(len(functions)):
        table = np.empty((len(functions[k].xs), len(columns) + 1), dtype=float)
        table[:, 0] = functions[k].xs
        for column, half in enumerate(columns, start=1):
            table[:, column] = half[k].ys
        values.append([float(v) for v in table.reshape(-1)])

    return MF6LawContinuum(
        law=1, lang=int(fields["lang"]), lep=int(fields["lep"]),
        tab2_interp=pairs,
        incident_energies=[float(f.outerDomainValue) for f in functions],
        nd=nd, na=na, values=values,
    )


def encodeMF6MT(forms: Optional[Dict[str, object]],
                provenance: Optional[EndfProvenance], mt: int,
                report: Optional[ConversionReport] = None):
    """The inverse of :func:`decodeMF6MT`, byte-identical to the source section.

    ``forms`` maps a product's **label** to its §18 distribution form — the mapping
    :func:`decodeMF6MT`'s ``entries`` becomes once the products are on a
    channel. Requires the ENDF provenance: ``JP``, ``LCT``, ``NK``, every
    ``ZAP``/``AWP``/``LIP``, the yields and the laws kika does not model live
    there and nowhere else, so without it this is not a lossy encode, it is an
    impossible one.
    """
    from kika.endf.classes.mf6.base import MF6MT
    from kika.endf.utils import PadStyle

    report = report if report is not None else ConversionReport()
    fields = (provenance.headerFields.get("mf6")
              if provenance is not None and provenance.sourceFormat == "endf"
              else None)
    if fields is None:
        raise ValueError(
            "encodeMF6MT needs the EndfProvenance decodeMF6MT produced: JP, "
            "LCT, NK, the per-product ZAP/AWP/LIP and the yields are not "
            "recoverable from the model alone, and neither are the laws it "
            "does not model"
        )

    records = fields["products"]
    expected = {record["label"] for record in records
                if record["label"] is not None}
    given = set(forms or {})
    if given - expected:
        raise ValueError(
            f"MT{mt}: encodeMF6MT was handed forms for {sorted(given - expected)}, "
            f"which the provenance does not list as products of this section; "
            f"it lists {sorted(expected)}"
        )

    section = MF6MT(number=mt)
    # From the MF6 block and not from the provenance's top level, which is
    # MF3's. Older provenances have no such keys, so the top level is the
    # fallback rather than the source.
    za = fields.get("za", provenance.za)
    section._za = float(za) if za is not None else None
    section._awr = fields.get("awr", provenance.awr)
    section._mat = fields.get("mat", provenance.mat)
    section._jp = int(fields.get("jp", 0) or 0)
    section._lct = int(fields.get("lct", 2))
    section._nk = fields["nk"]
    pad = fields.get("pad") or {}
    section.pad = PadStyle(**pad) if pad else PadStyle()
    section.products = [
        _restoreProduct(record, (forms or {}).get(record["label"]), mt)
        for record in records
    ]
    return section, report
