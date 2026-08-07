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
)

__all__ = ["decodeMF33MT", "decodeMF34MT", "decodeCovarianceSuite",
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


def _matrixForm(matrix, grid, isRelative: bool) -> CovarianceMatrix:
    grid = np.asarray(grid, dtype=float) if grid is not None else None
    return CovarianceMatrix(
        matrix=np.asarray(matrix, dtype=float),
        rowGrid=grid,
        columnGrid=grid,
        isRelative=bool(isRelative),
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
            form=_matrixForm(matrix, grid, covmat.is_relative[index]),
        ))

    if not sections:
        report.lost(f"MF34/MT{mf34mt.number}: no Legendre covariance blocks decoded")
    return sections, report


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
