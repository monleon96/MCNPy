"""One evaluation, two encodings: GNDS Fe-56 against ENDF Fe-56.

Phase 5's acceptance test for the covariance reader, and the only one that can
say the matrices are *right* rather than merely well-formed. Everything else in
``kika/gnds/tests`` checks that a file was read the way it was written; this
checks that what was read is the same physics kika already gets from the same
evaluation in ENDF-6.

**What it proves and what it does not.** ``n-026_Fe_056.endf.gnds.xml`` is
ENDF/B-VIII.1 *translated to GNDS by FUDGE* — its own ``title`` says so. So
agreement here is agreement with FUDGE's translation, not with the standard,
and a disagreement could in principle be FUDGE's rather than kika's. It is still
the strongest available check: the two paths share no code, no parser and no
array unpacking, and they agree or they do not.

Marked ``tape`` and ``gnds``: it needs the 18.8 MB GNDS file and the 40 MB ENDF
tape from the shared tree, and skips honestly without them.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.gnds.covariances import readCovarianceSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import CovarianceMatrix, Mixed

#: The MTs Fe-56's covariance file carries, from its own ``rowData``. MT102 is
#: the one that matters most here: it is a ``mixed`` of three components on
#: three different grids, the largest 628x628, so it exercises the grid reading
#: and the triangle reflection at a size where an error cannot hide.
EXPECTED_MTS = (1, 2, 4, 5, 16, 102, 103)


def _endfSections(path):
    from kika.endf.model_adapter.covariances import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf

    endf = read_endf(str(path))
    suite, _ = decodeCovarianceSuite(endf)
    return suite


def _byMT(suite, mf=33):
    """``{mt: [matrix, …]}`` for the diagonal blocks of one MF, either encoding.

    Keyed through ``DataLink.ENDF_MT``, which reads both the comma the standard
    specifies and the slash kika's ENDF adapter writes — see that property for
    why the divergence is not fixed here.
    """
    out = {}
    for section in suite.covarianceSections:
        link = section.rowData
        if link is None or link.ENDF_MF != mf or section.isCrossTerm:
            continue
        matrices = list(_walk(section.form))
        if matrices:
            out.setdefault(link.ENDF_MT, []).extend(matrices)
    return out


def _walk(form):
    if isinstance(form, CovarianceMatrix):
        yield form
    elif isinstance(form, Mixed):
        for component in form.components:
            yield from _walk(component)


@pytest.fixture(scope="module")
def gndsCovariances(fe56_gnds_cov_tape, fe56_gnds_tape):
    reactions = Document.parse(fe56_gnds_tape)
    suite, report = readCovarianceSuite(
        Document.parse(fe56_gnds_cov_tape), {"reactions": reactions}
    )
    return suite, report


def test_the_full_fe56_covariance_file_reads_with_every_link_followed(gndsCovariances):
    """The whole published file, not the trim, with its reactionSuite in hand."""
    suite, report = gndsCovariances
    assert len(suite.covarianceSections) == len(EXPECTED_MTS)
    assert {s.rowData.ENDF_MT for s in suite.covarianceSections} == set(EXPECTED_MTS)
    assert all(s.rowData.ENDF_MF == 33 for s in suite.covarianceSections)
    assert not report.losses, f"unfollowable links: {report.losses}"


#: MTs whose Fe-56 covariance is a single component in **both** encodings, so
#: the two matrices are the same object and can be compared element by element.
#: The rest (MT1, MT2, MT102) are ``mixed`` in GNDS and merged in ENDF — see
#: :func:`test_the_encodings_group_a_mixed_covariance_differently`.
DIRECTLY_COMPARABLE = (4, 5, 16, 103)


def test_fe56_covariances_agree_with_the_endf_path(gndsCovariances, fe56_b81_tape):
    """The oracle. Same evaluation, two encodings, two readers sharing no code.

    Compared as **relative** covariances on their own grids: both paths report
    ``isRelative`` and ENDF/B-VIII.1's Fe-56 MF33 is relative throughout, so no
    conversion is involved and any difference is a difference in what was read.

    Restricted to :data:`DIRECTLY_COMPARABLE`. That is not a weakening to make
    the test pass — the other three MTs are genuinely not the same object on the
    two sides, and the next test is what covers them.
    """
    gnds, _ = gndsCovariances
    fromGnds = _byMT(gnds)
    fromEndf = _byMT(_endfSections(fe56_b81_tape))

    compared = 0
    for mt in DIRECTLY_COMPARABLE:
        assert mt in fromGnds and mt in fromEndf, f"MT{mt} missing from an encoding"
        gndsMatrices, endfMatrices = fromGnds[mt], fromEndf[mt]
        assert len(gndsMatrices) == len(endfMatrices) == 1, (
            f"MT{mt} is no longer a single component on both sides: "
            f"{len(gndsMatrices)} against {len(endfMatrices)}"
        )
        left, right = gndsMatrices[0], endfMatrices[0]
        assert left.matrix.shape == right.matrix.shape
        assert left.isRelative == right.isRelative
        np.testing.assert_allclose(
            left.matrix, right.matrix, rtol=1e-9, atol=0,
            err_msg=f"MT{mt} differs between the encodings",
        )
        np.testing.assert_allclose(
            left.rowGrid, right.rowGrid, rtol=1e-9, atol=0,
            err_msg=f"MT{mt} row grid differs between the encodings",
        )
        compared += 1
    assert compared == len(DIRECTLY_COMPARABLE)


def test_the_encodings_group_a_mixed_covariance_differently(
    gndsCovariances, fe56_b81_tape
):
    """MT1, MT2 and MT102 are one matrix in ENDF and several in GNDS.

    **Neither side is wrong.** ENDF's MF33 states a covariance as a set of NI
    sub-subsections that add; kika's ENDF adapter sums them onto the union of
    their grids, so MT102 arrives as one 631x631. GNDS keeps them apart in a
    ``mixed`` — 3x3 + 3x3 + 628x628 — and this reader deliberately does not add
    them, because a ``mixed`` may hold a ``shortRangeSelfScalingVariance`` whose
    magnitude depends on the processing group width and which therefore cannot
    be added to a fixed grid at all.

    What *can* be checked without inventing the arithmetic is the grid: the
    union of the GNDS components' boundaries must be exactly the grid the ENDF
    adapter merged onto. It is, to the last bit, for all seven MTs — which says
    the components were read on the right grids even where their sum was not
    formed.

    Combining a ``mixed`` onto one grid is a real operation kika will need, and
    it is not phase 5's: it belongs wherever the two paths are made to produce
    one answer, with its own test that the sum reproduces the ENDF matrix.
    """
    gnds, _ = gndsCovariances
    fromGnds = _byMT(gnds)
    fromEndf = _byMT(_endfSections(fe56_b81_tape))

    multiComponent = [mt for mt, matrices in fromGnds.items() if len(matrices) > 1]
    assert sorted(multiComponent) == [1, 2, 102], sorted(multiComponent)

    for mt in sorted(fromGnds):
        components = fromGnds[mt]
        assert all(c.rowGrid is not None for c in components), f"MT{mt}"
        union = np.unique(np.concatenate([c.rowGrid for c in components]))
        merged = fromEndf[mt][0].rowGrid
        np.testing.assert_allclose(
            union, merged, rtol=1e-12, atol=0,
            err_msg=(
                f"MT{mt}: the union of the GNDS components' grids is not the "
                f"grid kika's ENDF adapter merged onto"
            ),
        )


def test_the_sum_of_a_mixed_reproduces_the_matrix_endf_merged(
    gndsCovariances, fe56_b81_tape
):
    """The test the docstring above says belongs "wherever the two paths are
    made to produce one answer". This is that place.

    **The arithmetic is not invented here.** ``MF33MT._process_ni_records_to_matrix``
    already performs it on the ENDF side and has since long before GNDS: every
    NI sub-subsection projected piecewise-constant onto the union of the grids
    and added, LB=8/9 decoded as LB=0 — i.e. the short-range self-scaling term
    is summed in like any other. That is the carrier the thesis runs on, so the
    question is not "what should combining a ``mixed`` do" but "does the ENDF
    answer come back out of the GNDS components", and the answer decides
    §2.2 rather than opening it.

    The previous test pins the grids; this one pins the values, which is the
    half that was never measured.
    """
    from kika.endf.classes.mf33.mf33 import MF33MT

    gnds, _ = gndsCovariances
    fromGnds = _byMT(gnds)
    fromEndf = _byMT(_endfSections(fe56_b81_tape))

    compared = 0
    for mt in sorted(fromGnds):
        components = fromGnds[mt]
        if len(components) == 1:
            continue
        merged = fromEndf[mt][0]
        union = np.unique(np.concatenate([c.rowGrid for c in components]))

        assert {c.isRelative for c in components} == {merged.isRelative}, (
            f"MT{mt}: the components and the merged matrix disagree about "
            f"relative-ness, so adding them is not the same operation"
        )

        total = np.zeros((len(union) - 1, len(union) - 1), dtype=float)
        for component in components:
            total += MF33MT._project_matrix_piecewise_constant(
                component.matrix, list(component.rowGrid), list(union),
                list(component.columnGrid) if component.columnGrid is not None
                else None,
            )

        np.testing.assert_allclose(
            total, merged.matrix, rtol=1e-9, atol=0,
            err_msg=f"MT{mt}: the summed GNDS components are not ENDF's matrix",
        )
        compared += 1

    assert compared == 3, f"expected MT1, MT2 and MT102 to be mixed; {compared}"
