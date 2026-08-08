"""MF33 and MF34 → :class:`~kika.nuclear_data.model.covariances.CovarianceSuite`.

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
already whole quantities, so the links carry no slices.

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

__all__ = ["decodeMF33MT", "decodeMF34MT", "decodeCovarianceSuite",
           "encodeMF33MT", "encodeMF34MT",
           "reactionHref", "angularDistributionHref"]


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


#: Which dimension of an angular distribution the Legendre order indexes.
#: The distribution is P(mu|E): dimension 2 is incident energy, dimension 1 is
#: the angular variable the Legendre expansion represents.
LEGENDRE_DIMENSION = 1


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

    if mf33 is None and mf34 is None:
        report.lost("no MF33 and no MF34: this evaluation carries no covariances")

    if endf.mf.get(31) is not None:
        report.unsupportedNode(
            "MF31 (nubar covariances) is present and parsed by kika, but this "
            "adapter covers MF33 and MF34 only"
        )
    if endf.mf.get(32) is not None:
        report.unsupportedNode(
            "MF32 (resonance parameter covariances) is present; kika's parser "
            "registry does not cover it and §25.3 parameter covariances are phase 7b"
        )

    return suite, report
