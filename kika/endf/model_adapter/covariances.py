"""MF31, MF33, MF34, MF35 → :class:`~kika.nuclear_data.model.covariances.CovarianceSuite`.

**The mapping that matters, and the one this module exists to get right.**
§25.2.5-6 give a ``rowData``/``columnData`` an optional ``slices``, and a slice
takes a ``dimension`` plus a ``domainValue``. An MF34 covariance is between two
*Legendre coefficients* ``a_l`` of an angular distribution — and in GNDS the
angular distribution is one function of order and energy, so a covariance about
order 1 is that function **sliced at order 1**, not a covariance of some separate
quantity called "order 1".

Modelling it the other way — a `reaction` per Legendre order — would produce a
structurally valid file that says something false about what the data is, and
nothing downstream would notice until someone tried to read it back into an
angular distribution. So the row and column links are built through
:meth:`DataLink.forLegendreOrder`, and a test asserts the ``domainValue`` is the
order.

MF33 is the simpler case: a covariance between two cross sections, which are
already whole quantities, so the links carry no slices. **MF31 is MF33's records
pointed somewhere else**: §31.1 makes the MT452/455/456 formats "directly
analogous to those of File 33", so the matrices arrive by the same route, but a
nu-bar covariance is about a multiplicity and the three MTs live on three
different nodes — see :mod:`kika.endf.model_adapter.multiplicity`.

**MF35 is the same idea one step further**, and is why this module is worth
having rather than three bespoke containers. An MF35 band is a covariance about
an *energy* distribution restricted to a range of incident energy — the same
object sliced a different way, so it is built through
:meth:`DataLink.forIncidentEnergyBand` and needs no new container. The bands of
one section have different orders (84, 641, 641, 641, 641 on ENDF/B-VIII.1
U-235), which is exactly what a suite of independent sections expresses and
what any single-matrix container gets wrong silently.

**What is not converted here.** kika's ``kika/cov`` package holds 14 000 lines of
COVERX/COVFIL/BOXER/GENDF I/O. None of it is rewritten or replaced: it stays as
a set of format encoders on the same footing as ENDF and ACE. This module reads
the *decoded* ``CrossSectionCovariance`` and ``LegendreCovariance`` objects that
package already produces and re-expresses them in the model's vocabulary.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport,
    CovarianceMatrix,
    CovarianceSection,
    CovarianceSuite,
    DataLink,
    EndfProvenance,
)

__all__ = ["decodeMF31MT", "decodeMF33MT", "decodeMF34MT", "decodeMF35MT",
           "decodeCovarianceSuite",
           "encodeMF31MT", "encodeMF33MT", "encodeMF34MT",
           "reactionHref", "angularDistributionHref",
           "energyDistributionHref", "nubarShims"]


def reactionHref(mt: int) -> str:
    """xPath into a ``reactionSuite`` for the cross section of one MT.

    §25.2.3's ``href``. The path follows the labels this library's decoder
    assigns (``MT2``), which is a kika convention rather than a GNDS-mandated
    one — GNDS only requires that the xPath resolve inside the companion
    ``reactionSuite``.
    """
    return f"/reactionSuite/reactions/reaction[@label='MT{mt}']/crossSection"


def angularDistributionHref(mt: int) -> str:
    """xPath to the angular distribution an MF34 covariance is about."""
    return (
        f"/reactionSuite/reactions/reaction[@label='MT{mt}']"
        f"/outputChannel/products/product[@label='n']/distribution"
    )


def energyDistributionHref(mt: int) -> str:
    """xPath to the energy distribution an MF35 covariance is about.

    The same node as :func:`angularDistributionHref`, and deliberately so: in
    GNDS one ``distribution`` holds the outgoing product's full dependence on
    angle *and* energy. MF34 and MF35 are covariances about the same object,
    distinguished by which of its dimensions they slice — order for MF34,
    incident energy for MF35 — not by pointing at different objects.
    """
    return (
        f"/reactionSuite/reactions/reaction[@label='MT{mt}']"
        f"/outputChannel/products/product[@label='n']/distribution"
    )


#: Which dimension of an angular distribution the Legendre order indexes.
#: The distribution is P(mu|E): dimension 2 is incident energy, dimension 1 is
#: the angular variable the Legendre expansion represents.
LEGENDRE_DIMENSION = 1

#: Which dimension of an energy distribution the MF35 band restricts.
#: The distribution is chi(E'|E): dimension 2 is incident energy, the same
#: convention as above, and dimension 1 is the outgoing energy the LB=7 matrix
#: is gridded on.
INCIDENT_ENERGY_DIMENSION = 2


def _matrixForm(matrix, grid, isRelative: bool,
                productFrame: Optional[str] = None) -> CovarianceMatrix:
    grid = np.asarray(grid, dtype=float) if grid is not None else None
    return CovarianceMatrix(
        matrix=np.asarray(matrix, dtype=float),
        rowGrid=grid,
        columnGrid=grid,
        isRelative=bool(isRelative),
        productFrame=productFrame,
    )


def _sectionProvenance(section, ltt: Optional[int] = None) -> EndfProvenance:
    """ZA, AWR, MAT — and MF34's LTT — off an MF33 or MF34 section header.

    None of the four has a GNDS counterpart: §25.2.2 identifies a covariance by
    its ``href``, not by a material header. They are kept so the encoder writes
    the header the file had rather than one it defaulted.
    """
    return EndfProvenance(
        mat=getattr(section, "_mat", None),
        awr=getattr(section, "_awr", None),
        za=getattr(section, "_za", None),
        headerFields={} if ltt is None else {"ltt": int(ltt)},
    )


def decodeMF33MT(mf33mt, report: Optional[ConversionReport] = None):
    """One MF33/MT section → a list of :class:`CovarianceSection`.

    One section per (row MT, column MT) block the file carries, including the
    cross-MT blocks — those are what make a covariance *suite* rather than a
    list of variances, and dropping them is the classic way to lose half the
    information while everything still looks fine.
    """
    report = report if report is not None else ConversionReport()
    covmat = mf33mt.to_xs_covmat()

    # `CrossSectionCovariance` has no `mt_metadata`, so unlike MF34 the section
    # header does not survive the trip through `kika/cov` at all: ZA reaches the
    # isotope tags and AWR and MAT reach nothing. Read them off the section.
    provenance = _sectionProvenance(mf33mt)

    sections = []
    for index, matrix in enumerate(covmat.matrices):
        rowMT = int(covmat.reaction_rows[index])
        colMT = int(covmat.reaction_cols[index])
        grid = covmat.energy_grids[index] if index < len(covmat.energy_grids) else None

        sections.append(CovarianceSection(
            label=f"MF33-MT{rowMT}" + (f"-MT{colMT}" if colMT != rowMT else ""),
            rowData=DataLink(href=reactionHref(rowMT), ENDF_MFMT=f"33/{rowMT}"),
            columnData=(
                DataLink(href=reactionHref(colMT), ENDF_MFMT=f"33/{colMT}")
                if colMT != rowMT else None
            ),
            form=_matrixForm(matrix, grid, covmat.is_relative[index]),
            provenance=provenance,
        ))

    if not sections:
        report.lost(f"MF33/MT{mf33mt.number}: no covariance blocks decoded")
    return sections, report


class _NubarShim:
    """``(energies, values)`` of one nu-bar, in the shape ``resolve_nc_lty0`` wants.

    MF31/452 is an **NC-type** sub-subsection on every evaluation checked: the
    total nu-bar covariance is not stored, it is declared to be the sum of MT455
    and MT456 (ENDF/B-VIII.1 U-235 writes exactly ``LTY=0`` with ``c=1`` for
    both). Resolving that sum needs more than the two child matrices — the
    children are *relative* covariances, so combining them requires the values
    they are relative to.

    ``MF33MT.resolve_nc_lty0`` already does the algebra and already accepts
    either an ``MF3MT`` or anything exposing ``energies``/``values``. For MF33
    the values are cross sections; for MF31 they are nu-bars. Passing the
    nu-bars is what makes the total's variance right instead of merely
    plausible — a straight sum of the two relative matrices would weight prompt
    and delayed equally, when delayed is under one per cent of the total.
    """

    __slots__ = ("energies", "values")

    def __init__(self, energies, values):
        self.energies = np.asarray(energies, dtype=float)
        self.values = np.asarray(values, dtype=float)

    @classmethod
    def fromMF1(cls, section) -> Optional["_NubarShim"]:
        """``None`` for an LNU=1 polynomial, which has no tabulated values."""
        energies = getattr(section, "energies", None)
        values = getattr(section, "nubar_values", None)
        if not energies or not values:
            return None
        return cls(energies, values)


def nubarShims(mf1) -> dict:
    """``{MT: _NubarShim}`` for whichever of MT452/455/456 a parsed MF1 carries."""
    from .multiplicity import NUBAR_MT

    shims = {}
    for mt in NUBAR_MT:
        section = getattr(mf1, "mt", {}).get(mt) if mf1 is not None else None
        if section is None:
            continue
        shim = _NubarShim.fromMF1(section)
        if shim is not None:
            shims[mt] = shim
    return shims


def decodeMF31MT(mf31mt, report: Optional[ConversionReport] = None,
                 separatePrompt: bool = True, siblingSections=None,
                 nubarValues=None):
    """One MF31/MT section → a list of :class:`CovarianceSection`.

    Structurally this is :func:`decodeMF33MT`: §31.1 says the MT452/455/456
    formats *"are directly analogous to those of File 33"* and kika's parser
    already reads MF31 with the MF33 record machinery, so the matrices arrive by
    the same route. What differs is the **href**, and only the href — a nu-bar
    covariance is about a multiplicity, not about a cross section, and the three
    MTs land on three different nodes. See
    :func:`~kika.endf.model_adapter.multiplicity.nubarHref`.

    ``siblingSections`` and ``nubarValues`` are what the NC-type total needs;
    without them an MF31/452 that is declared as a sum decodes to nothing and
    says so in the report rather than quietly returning an empty list.
    """
    from .multiplicity import nubarHref

    report = report if report is not None else ConversionReport()
    mt = int(getattr(mf31mt, "number", 0))
    covmat = mf31mt.to_xs_covmat(
        sibling_sections=siblingSections, mf3_sections=nubarValues
    )
    provenance = _sectionProvenance(mf31mt)
    derived = any(
        nc.lty == 0
        for subsection in getattr(mf31mt, "_subsections", [])
        for nc in getattr(subsection, "nc_records", [])
    )
    if derived:
        # Not physics, and not recoverable from a matrix: the file said "this
        # covariance IS the sum of MT455 and MT456" rather than storing it, and
        # the encoder has to be able to say that it is writing back something
        # the file never wrote explicitly.
        provenance.headerFields["wasDerived"] = True

    sections = []
    for index, matrix in enumerate(covmat.matrices):
        rowMT = int(covmat.reaction_rows[index])
        colMT = int(covmat.reaction_cols[index])
        grid = covmat.energy_grids[index] if index < len(covmat.energy_grids) else None

        sections.append(CovarianceSection(
            label=f"MF31-MT{rowMT}" + (f"-MT{colMT}" if colMT != rowMT else ""),
            rowData=DataLink(href=nubarHref(rowMT, separatePrompt),
                             ENDF_MFMT=f"31/{rowMT}"),
            columnData=(
                DataLink(href=nubarHref(colMT, separatePrompt),
                         ENDF_MFMT=f"31/{colMT}")
                if colMT != rowMT else None
            ),
            form=_matrixForm(matrix, grid, covmat.is_relative[index]),
            provenance=provenance,
        ))

    if not sections:
        if derived and siblingSections is None:
            report.lost(
                f"MF31/MT{mt}: the covariance is NC-type (LTY=0) — the file "
                f"declares it as a sum of other MTs rather than storing it — "
                f"and no sibling sections were passed, so it could not be "
                f"resolved"
            )
        else:
            report.lost(f"MF31/MT{mt}: no covariance blocks decoded")
    return sections, report


def decodeMF34MT(mf34mt, report: Optional[ConversionReport] = None, mf4Data=None):
    """One MF34/MT section → :class:`CovarianceSection` per Legendre-order block.

    Each block becomes a section whose row and column links are **sliced at the
    Legendre order** they are about (§25.2.5-6). See the module docstring for
    why that is the correct shape and what the plausible wrong one would cost.
    """
    report = report if report is not None else ConversionReport()
    covmat = mf34mt.to_ang_covmat(mf4_data=mf4Data)

    # LTT as well as ZA/AWR/MAT: §34.1 makes it the difference between a section
    # whose blocks start at a_0 and one that starts at a_1, and `to_mf34` would
    # otherwise infer it from whether an L=0 pair is present. The inference is
    # right, and preferring the file's own value means a round trip does not
    # depend on it staying right — the same argument `encodeMF3MT` makes for the
    # (NBT, INT) pairs.
    provenance = _sectionProvenance(mf34mt, ltt=getattr(mf34mt, "_ltt", None))

    sections = []
    for index, matrix in enumerate(covmat.matrices):
        rowMT = int(covmat.reaction_rows[index])
        colMT = int(covmat.reaction_cols[index])
        rowOrder = int(covmat.l_rows[index])
        colOrder = int(covmat.l_cols[index])
        grid = covmat.energy_grids[index] if index < len(covmat.energy_grids) else None

        sections.append(CovarianceSection(
            label=f"MF34-MT{rowMT}-L{rowOrder}"
                  + (f"-MT{colMT}-L{colOrder}" if (colMT, colOrder) != (rowMT, rowOrder) else ""),
            rowData=DataLink.forLegendreOrder(
                angularDistributionHref(rowMT), rowOrder,
                ENDF_MFMT=f"34/{rowMT}", dimension=LEGENDRE_DIMENSION,
            ),
            columnData=DataLink.forLegendreOrder(
                angularDistributionHref(colMT), colOrder,
                ENDF_MFMT=f"34/{colMT}", dimension=LEGENDRE_DIMENSION,
            ),
            form=_matrixForm(matrix, grid, covmat.is_relative[index],
                             productFrame=covmat.frame[index]),
            provenance=provenance,
        ))

    if not sections:
        report.lost(f"MF34/MT{mf34mt.number}: no Legendre covariance blocks decoded")
    return sections, report


def decodeMF35MT(mf35mt, report: Optional[ConversionReport] = None):
    """One MF35/MT section → one :class:`CovarianceSection` per band.

    **No cross-band blocks exist**, so the result is block-diagonal by
    construction and there is deliberately nothing here that concatenates the
    bands into a single matrix. Their orders differ — 84, 641, 641, 641, 641 on
    ENDF/B-VIII.1 U-235 — so a container that assembled every block at one
    dimension would produce a malformed matrix silently. That is also why MF35
    does not go through ``kika/cov``'s ``CrossSectionCovariance`` the way MF33
    does; the decode is direct.

    **The matrix is absolute and stays absolute.** LB=7 holds the covariance of
    group-integrated probabilities, which are already dimensionless and already
    sum to one. There is no absolute/relative conversion here and no MF5 needed
    to do one — unlike MF34, which needs its MF4 to convert. The measured
    evidence for that reading, and the row-sum test that pins it, are in
    :mod:`kika.endf.classes.mf35.mf35`.
    """
    report = report if report is not None else ConversionReport()
    mt = int(getattr(mf35mt, "number", 0))
    provenance = _sectionProvenance(mf35mt)

    sections = []
    for index, band in enumerate(getattr(mf35mt, "subsections", [])):
        link = DataLink.forIncidentEnergyBand(
            energyDistributionHref(mt), band.e1, band.e2,
            ENDF_MFMT=f"35/{mt}", dimension=INCIDENT_ENERGY_DIMENSION,
        )
        sections.append(CovarianceSection(
            label=f"MF35-MT{mt}-band{index}",
            rowData=link,
            columnData=None,
            form=_matrixForm(band.matrix(), band.energy_grid(), isRelative=False),
            provenance=provenance,
        ))

    if not sections:
        report.lost(f"MF35/MT{mt}: no covariance bands decoded")
    return sections, report


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
#
# **These are projections back onto `kika/cov`, not record writers**, and the
# reason is structural rather than a shortcut. The decoders above read through
# `to_xs_covmat()` / `to_ang_covmat()`, which have already collapsed the file's
# NC/NI subsection structure -- LB type, derived-versus-explicit -- into dense
# matrices on a grid. `CovarianceMatrix` keeps the matrix, the grids, the frame
# and whether it is relative, and that is all there is to keep. **The LB
# structure is not recoverable from the model**, so the honest thing is to hand
# the policy back to the code that already owns it: `to_mf34` chooses LB=5 on
# the diagonal and LB=6 off it, `create_mf33_from_covariance` its equivalent.
# Reimplementing that choice here would make it two policies that agree until
# they don't -- which is exactly how "ACE stores no reaction Q values" came to
# be written in three places while the values sat parsed (`docs/library-gaps.md`
# D4).
#
# The consequence for testing is stated where it matters, in
# `tests/test_covariance_round_trip.py`: the gate is a numerical **fixed point**,
# not the byte identity MF3, MF4 and MF1 are held to.

def _endfMT(link) -> int:
    """The MT an ``ENDF_MFMT`` names — ``"34/2"`` → 2."""
    if link is None or not link.ENDF_MFMT:
        raise ValueError("a covariance link with no ENDF_MFMT cannot be written to ENDF")
    return int(str(link.ENDF_MFMT).split("/")[1])


def _legendreOrder(link) -> int:
    """The Legendre order a link is sliced at (§25.2.5-6)."""
    for entry in link.slices.slices:
        if entry.domainValue is not None:
            return int(entry.domainValue)
    raise ValueError(
        f"the link to {link.href!r} carries no slice, so the Legendre order it "
        f"is about is unknown; MF34 cannot be written from it"
    )


def _covarianceSections(source, prefix: str, mt: int):
    """The sections of *source* that belong to one MF and row MT."""
    sections = getattr(source, "covarianceSections", source)
    selected = [
        section for section in sections
        if section.rowData is not None
        and str(section.rowData.ENDF_MFMT or "").startswith(prefix)
        and _endfMT(section.rowData) == mt
    ]
    if not selected:
        raise ValueError(f"no MF{prefix.rstrip('/')} covariance sections for MT{mt}")
    return selected


def encodeMF34MT(source, mt: int, mat: Optional[int] = None,
                 report: Optional[ConversionReport] = None):
    """A :class:`CovarianceSuite` → an ``MF34MT`` for one MT.

    Rebuilds the :class:`~kika.cov.legendre_covariance.LegendreCovariance` the
    decoder read through and lets :meth:`LegendreCovariance.to_mf34` write the
    records. Everything comes from the model — the Legendre orders from the row
    and column **slices**, which is where §25.2.5-6 put them, and the header
    from the provenance the decoder kept.
    """
    from kika.cov.legendre_covariance import LegendreCovariance

    report = report if report is not None else ConversionReport()
    sections = _covarianceSections(source, "34/", mt)

    covmat = LegendreCovariance()
    isotope = None
    for section in sections:
        provenance = section.provenance
        za = int(getattr(provenance, "za", None) or 0)
        isotope = isotope if isotope is not None else za
        form = section.form

        covmat.isotope_rows.append(za)
        covmat.reaction_rows.append(_endfMT(section.rowData))
        covmat.l_rows.append(_legendreOrder(section.rowData))
        covmat.isotope_cols.append(za)
        covmat.reaction_cols.append(_endfMT(section.columnData))
        covmat.l_cols.append(_legendreOrder(section.columnData))
        covmat.matrices.append(np.asarray(form.matrix, dtype=float))
        covmat.energy_grids.append([float(e) for e in form.rowGrid])
        covmat.is_relative.append(bool(form.isRelative))
        covmat.frame.append(form.productFrame)

    provenance = sections[0].provenance
    covmat.mt_metadata[(isotope, mt)] = {
        "za": getattr(provenance, "za", None),
        "awr": getattr(provenance, "awr", None),
        "mat": getattr(provenance, "mat", None),
        "ltt": (getattr(provenance, "headerFields", None) or {}).get("ltt"),
    }

    section = covmat.to_mf34(isotope, mt, mat=mat)
    report.approximated(
        f"MF34/MT{mt}: written through kika/cov, so NI>1 sub-subsections are "
        f"collapsed to one LB=5/LB=6 record on the stored grid. The numbers are "
        f"preserved; the file's original per-record split is not."
    )
    return section, report


def encodeMF33MT(source, mt: int, mat: Optional[int] = None,
                 report: Optional[ConversionReport] = None):
    """A :class:`CovarianceSuite` → an ``MF33MT`` for one MT.

    One subsection per (row MT, column MT) block, each built by
    :func:`~kika.endf.writers.mf33_writer.create_mf33_from_covariance` so the
    LB=5/LB=6 record layout has exactly one implementation.
    """
    from kika.endf.writers.mf33_writer import create_mf33_from_covariance

    report = report if report is not None else ConversionReport()
    sections = _covarianceSections(source, "33/", mt)
    provenance = sections[0].provenance

    za = getattr(provenance, "za", None)
    awr = getattr(provenance, "awr", None)
    if za is None or awr is None:
        raise ValueError(
            f"MF33/MT{mt} carries no ZA/AWR, so the section header would be "
            f"invented. Decode from ENDF, where the header comes from the file."
        )
    resolvedMat = mat if mat is not None else getattr(provenance, "mat", None)

    built = None
    for section in sections:
        form = section.form
        colMT = _endfMT(section.columnData) if section.columnData is not None else mt
        if not form.isRelative:
            report.approximated(
                f"MF33/MT{mt}-MT{colMT}: the block is absolute and is written as "
                f"LB=5/LB=6, which ENDF-6 reads as relative"
            )
        one = create_mf33_from_covariance(
            cov_matrix=np.asarray(form.matrix, dtype=float),
            energy_grid_ev=np.asarray(form.rowGrid, dtype=float),
            za=float(za), awr=float(awr), mat=int(resolvedMat or 0),
            mt=mt, mt1=colMT,
            lb=5 if colMT == mt else 6,
            col_energy_grid_ev=(
                None if colMT == mt else np.asarray(form.columnGrid, dtype=float)
            ),
        )
        if built is None:
            built = one
        else:
            built.add_subsection(one.subsections[0])

    built._nl = len(built.subsections)
    return built, report


def encodeMF31MT(source, mt: int, mat: Optional[int] = None,
                 report: Optional[ConversionReport] = None):
    """A :class:`CovarianceSuite` → an ``MF31MT`` for one nu-bar MT.

    The records are MF33's — §31.1 again — so this goes through the same writer
    and then **stamps the section as MF31**. That stamp is load-bearing in two
    places: ``kika.endf.writers._section_writer`` places the section by
    ``section._mf``, and ``MF33MT.__str__`` reads it for columns 71-72. The
    second of those only started reading it when this function was written; it
    used to write the literal ``33``, which would have put MF33 identifiers on
    every line of a section sitting in the MF31 block.
    """
    from kika.endf.writers.mf33_writer import create_mf33_from_covariance

    report = report if report is not None else ConversionReport()
    sections = _covarianceSections(source, "31/", mt)
    provenance = sections[0].provenance

    za = getattr(provenance, "za", None)
    awr = getattr(provenance, "awr", None)
    if za is None or awr is None:
        raise ValueError(
            f"MF31/MT{mt} carries no ZA/AWR, so the section header would be "
            f"invented. Decode from ENDF, where the header comes from the file."
        )
    resolvedMat = mat if mat is not None else getattr(provenance, "mat", None)

    built = None
    for section in sections:
        form = section.form
        colMT = _endfMT(section.columnData) if section.columnData is not None else mt
        if not form.isRelative:
            report.approximated(
                f"MF31/MT{mt}-MT{colMT}: the block is absolute and is written "
                f"as LB=5/LB=6, which ENDF-6 reads as relative"
            )
        one = create_mf33_from_covariance(
            cov_matrix=np.asarray(form.matrix, dtype=float),
            energy_grid_ev=np.asarray(form.rowGrid, dtype=float),
            za=float(za), awr=float(awr), mat=int(resolvedMat or 0),
            mt=mt, mt1=colMT,
            lb=5 if colMT == mt else 6,
            col_energy_grid_ev=(
                None if colMT == mt else np.asarray(form.columnGrid, dtype=float)
            ),
        )
        if built is None:
            built = one
        else:
            built.add_subsection(one.subsections[0])

    built._nl = len(built.subsections)
    built._mf = 31
    if any(
        (getattr(s.provenance, "headerFields", None) or {}).get("wasDerived")
        for s in sections
    ):
        report.approximated(
            f"MF31/MT{mt}: the file stated this covariance as an NC-type sum "
            f"(LTY=0) and it is written back as an explicit LB=5 matrix. The "
            f"numbers are the resolved sum; the declaration that it *is* a sum "
            f"is not recoverable from the model"
        )
    return built, report


def decodeCovarianceSuite(endf, report: Optional[ConversionReport] = None,
                          evaluation: Optional[str] = None):
    """Every MF33 and MF34 section in a parsed ENDF → one :class:`CovarianceSuite`.

    §25.1.1 makes the suite a **root node in its own right**, linked to the
    ``reactionSuite`` through ``externalFiles`` rather than nested inside it.
    Building it separately here keeps that separation honest from the start,
    even though kika hangs the result off ``ReactionSuite.covarianceSuite`` for
    convenience.
    """
    report = report if report is not None else ConversionReport()
    suite = CovarianceSuite(evaluation=evaluation, projectile="n", interaction="nuclear")

    mf1 = endf.mf.get(1) if hasattr(endf, "mf") else None

    mf31 = endf.mf.get(31) if hasattr(endf, "mf") else None
    if mf31 is not None:
        # The whole MF31 file is the sibling set: §31.1's NC-type MT452 is
        # declared as the sum of MT455 and MT456, both of which are sections of
        # this same file. `to_xs_covmat` resolves that only when handed them.
        siblings = {mt: mf31.mt[mt] for mt in getattr(mf31, "mt", {})}
        values = nubarShims(mf1)
        separatePrompt = 456 in getattr(mf1, "mt", {}) if mf1 is not None else False
        if not values and mf1 is None:
            report.warn(
                "MF31 is present but MF1 is not, so the nu-bar values a "
                "relative covariance is relative to are unavailable and an "
                "NC-type total cannot be weighted correctly"
            )
        for mt in sorted(siblings):
            sections, report = decodeMF31MT(
                mf31.mt[mt], report, separatePrompt=separatePrompt,
                siblingSections=siblings, nubarValues=values,
            )
            suite.covarianceSections.extend(sections)

    mf33 = endf.mf.get(33) if hasattr(endf, "mf") else None
    if mf33 is not None:
        for mt in sorted(getattr(mf33, "mt", {})):
            sections, report = decodeMF33MT(mf33.mt[mt], report)
            suite.covarianceSections.extend(sections)

    mf34 = endf.mf.get(34) if hasattr(endf, "mf") else None
    if mf34 is not None:
        mf4 = endf.mf.get(4) if hasattr(endf, "mf") else None
        for mt in sorted(getattr(mf34, "mt", {})):
            mf4Section = getattr(mf4, "mt", {}).get(mt) if mf4 is not None else None
            if mf4Section is None:
                report.warn(
                    f"MF34/MT{mt} has no matching MF4 section, so any relative "
                    f"covariance cannot be converted to absolute"
                )
            sections, report = decodeMF34MT(mf34.mt[mt], report, mf4Data=mf4Section)
            suite.covarianceSections.extend(sections)

    mf35 = endf.mf.get(35) if hasattr(endf, "mf") else None
    if mf35 is not None:
        for mt in sorted(getattr(mf35, "mt", {})):
            sections, report = decodeMF35MT(mf35.mt[mt], report)
            suite.covarianceSections.extend(sections)

    if mf31 is None and mf33 is None and mf34 is None and mf35 is None:
        # MF32 is a covariance this adapter does not convert, so an evaluation
        # carrying only it still yields an empty suite — but saying it "carries
        # no covariances" would be false, and the two cases want different
        # follow-up from the reader.
        if endf.mf.get(32) is not None:
            report.lost(
                "no MF31, MF33, MF34 or MF35: the only covariance this "
                "evaluation carries is MF32, which this adapter does not convert"
            )
        else:
            report.lost(
                "no MF31, MF33, MF34 or MF35: this evaluation carries no "
                "covariances"
            )

    if endf.mf.get(32) is not None:
        report.unsupportedNode(
            "MF32 (resonance parameter covariances) is present and parsed by "
            "kika, but the model has nowhere to put it: §25.3 parameter "
            "covariances are phase 7b, so covarianceSuite.parameterCovariances "
            "stays empty. Read it through endf.mf[32].mt[151]"
        )

    return suite, report
