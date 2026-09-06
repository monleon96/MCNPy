"""Phase 5 P3: §25 ``covarianceSuite``.

Two kinds of test, and both are needed. The hand-built ``array`` cases pin the
three storage modes against matrices small enough to write out by hand — that is
where a transcription error from FUDGE's ``xDataArray`` would hide, because a
wrong unpacking produces a plausible matrix rather than an exception. The
fixture sweeps then assert that the modes as they occur *in published files*
come out symmetric, correctly sized and correctly linked.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kika.gnds.covariances import readArray, readCovarianceSuite
from kika.gnds.version import UnsupportedGndsVersion
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (AverageParameterCovariance,
                                     CovarianceMatrix, Mixed,
                                     ParameterCovariance,
                                     ShortRangeSelfScalingVariance, Sum)


def _array(text: str) -> np.ndarray:
    return readArray(ET.fromstring(text))


# ---------------------------------------------------------------------------
# array storage, by hand
# ---------------------------------------------------------------------------

def test_a_full_array_is_row_major():
    built = _array('<array shape="2,3"><values>1 2 3 4 5 6</values></array>')
    np.testing.assert_array_equal(built, [[1, 2, 3], [4, 5, 6]])


def test_a_lower_triangular_array_is_mirrored():
    """``Full.constructArray``: ``tril_indices``, then reflect."""
    built = _array(
        '<array shape="3,3" symmetry="lower"><values>1 2 3 4 5 6</values></array>'
    )
    np.testing.assert_array_equal(built, [[1, 2, 4], [2, 3, 5], [4, 5, 6]])
    np.testing.assert_array_equal(built, built.T)


def test_a_diagonal_array_fills_only_the_diagonal():
    built = _array('<array shape="3,3" compression="diagonal"><values>7 8 9</values></array>')
    np.testing.assert_array_equal(built, np.diag([7, 8, 9]))


def test_a_flattened_arrays_starts_index_the_full_matrix():
    """The one mode that cannot be inferred from the data alone.

    ``Flattened.constructArray`` fills a buffer of ``n*m``, reshapes, and only
    *then* mirrors. So a start of 4 in a 3x3 array is element (1, 1) — not the
    fifth entry of the lower triangle, which would be (2, 1). Reading it the
    other way puts variance on the wrong element and raises nothing.
    """
    built = _array(
        '<array shape="3,3" compression="flattened">'
        '<values valueType="Integer32" label="starts">0 4</values>'
        '<values valueType="Integer32" label="lengths">1 2</values>'
        "<values>5 6 7</values></array>"
    )
    # No `symmetry`, so nothing is reflected: flat indices 0, 4, 5 are (0,0),
    # (1,1) and (1,2).
    np.testing.assert_array_equal(built, [[5, 0, 0], [0, 6, 7], [0, 0, 0]])


def test_a_flattened_lower_array_mirrors_after_the_reshape():
    built = _array(
        '<array shape="3,3" symmetry="lower" compression="flattened">'
        '<values valueType="Integer32" label="starts">3</values>'
        '<values valueType="Integer32" label="lengths">2</values>'
        "<values>1 2</values></array>"
    )
    # Flat indices 3 and 4 are (1, 0) and (1, 1); (1, 0) mirrors to (0, 1).
    np.testing.assert_array_equal(built, [[0, 1, 0], [1, 2, 0], [0, 0, 0]])


@pytest.mark.parametrize("xml,message", [
    ('<array shape="3,3"><values>1 2 3</values></array>', "needs 9 values"),
    ('<array shape="3,3" symmetry="lower"><values>1 2</values></array>', "needs 6 values"),
    ('<array shape="3,3" compression="diagonal"><values>1 2</values></array>', "is 3 values"),
    ('<array shape="2,2,2"><values>1</values></array>', "two-dimensional"),
    ('<array shape="2,2" compression="sparse"><values>1</values></array>', "not one of GNDS"),
])
def test_a_miscounted_array_is_refused(xml, message):
    """Silence is the failure mode; every count is checked against the shape."""
    with pytest.raises(ValueError, match=message):
        _array(xml)


def test_a_multi_diagonal_array_is_refused_rather_than_truncated():
    """FUDGE can store several diagonals; no published file does.

    Reading one as the main diagonal would move off-diagonal variance onto the
    diagonal with nothing to show for it.
    """
    with pytest.raises(ValueError, match="startingIndices"):
        _array(
            '<array shape="3,3" compression="diagonal">'
            '<values label="startingIndices">0 0 0 1</values>'
            "<values>1 2 3 4 5</values></array>"
        )


def test_a_flattened_array_with_inconsistent_runs_is_refused():
    with pytest.raises(ValueError, match="run lengths account for"):
        _array(
            '<array shape="3,3" compression="flattened">'
            '<values valueType="Integer32" label="starts">0</values>'
            '<values valueType="Integer32" label="lengths">2</values>'
            "<values>5</values></array>"
        )


# ---------------------------------------------------------------------------
# the fixture sweep
# ---------------------------------------------------------------------------

def test_every_covariance_fixture_reads_without_its_reaction_suite(
    gnds_covariance_fixture,
):
    """The standalone case: read, report the links, raise nothing."""
    suite, report = readCovarianceSuite(Document.parse(gnds_covariance_fixture))
    assert suite.target
    assert suite.covarianceSections or suite.parameterCovariances
    # Every section's row link points into a file that was not supplied, so
    # each is reported. The matrices are read regardless.
    assert report.losses, "no unfollowable link was reported"
    assert all("is read" in entry or "was not supplied" in entry
               for entry in report.losses)


def _walk(form):
    """Every :class:`CovarianceMatrix` inside one form, ``mixed`` included."""
    if isinstance(form, CovarianceMatrix):
        yield form
    elif isinstance(form, ShortRangeSelfScalingVariance):
        if form.matrix is not None:
            yield form.matrix
    elif isinstance(form, Mixed):
        for component in form.components:
            yield from _walk(component)


def _griddedMatrices(suite):
    """``(matrix, isDiagonalBlock)`` for every gridded covariance in a suite.

    ``isDiagonalBlock`` matters more than it looks: a **cross-term** section
    holds C(A, B) for two *different* quantities, which is square whenever the
    two grids happen to have the same length and is **not** symmetric — its
    transpose is C(B, A). Asserting symmetry on it would be asserting something
    false about the physics.
    """
    for section in suite.covarianceSections:
        for matrix in _walk(section.form):
            yield matrix, not section.isCrossTerm
    for covariance in suite.parameterCovariances:
        if isinstance(covariance, AverageParameterCovariance):
            for matrix in _walk(covariance.form):
                yield matrix, not covariance.isCrossTerm


def test_every_matrix_read_from_a_real_file_is_sized_against_its_grid(
    gnds_covariance_fixture,
):
    """Shape against grid, on every gridded matrix in every committed file.

    §5.1.3's grids are bin *boundaries*, so a grid has one value more than the
    matrix has rows. ``CovarianceMatrix.__post_init__`` enforces that for the
    row grid, which makes this sweep also the test that the row and column axes
    were not read the wrong way round — §25 puts the row axis at index 2 and the
    column axis at index 1, and swapping them survives every shape check on a
    square matrix.
    """
    suite, _ = readCovarianceSuite(Document.parse(gnds_covariance_fixture))
    checked = 0
    for matrix, _diagonal in _griddedMatrices(suite):
        assert matrix.matrix.ndim == 2
        if matrix.rowGrid is not None:
            assert matrix.rowGrid.size == matrix.matrix.shape[0] + 1
        if matrix.columnGrid is not None:
            assert matrix.columnGrid.size == matrix.matrix.shape[1] + 1
        checked += 1
    if not checked:
        # Si-32 is committed precisely for this shape: a covarianceSuite whose
        # only content is parameterCovariances. Asserting a gridded matrix here
        # would make the fixture set's own edge case a failure.
        assert not suite.covarianceSections


def test_a_diagonal_block_from_a_real_file_really_is_symmetric(
    gnds_covariance_fixture,
):
    """The triangle reflection, checked on published data rather than on a 3x3.

    Diagonal blocks only — a cross term is not symmetric and must not be
    asserted to be. If the reflection were wrong the result would still be a
    matrix of the right shape, so nothing but this would notice.
    """
    suite, _ = readCovarianceSuite(Document.parse(gnds_covariance_fixture))
    diagonal = [m for m, isDiagonal in _griddedMatrices(suite)
                if isDiagonal and m.isSquare]
    if not diagonal:
        assert not suite.covarianceSections      # Si-32 again
        return
    for matrix in diagonal:
        np.testing.assert_array_equal(matrix.matrix, matrix.matrix.T)


def test_a_cross_term_block_is_not_forced_to_be_symmetric(gnds_data_dir):
    """C(A, B) is not C(B, A), and F-19 has six sections that say so.

    This is the positive statement behind the exclusion above: the reader must
    not symmetrise a cross term, and a test that only skipped them would not
    notice if it did.
    """
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml")
    )
    crossTerms = [m for m, isDiagonal in _griddedMatrices(suite) if not isDiagonal]
    assert crossTerms, "F-19 lost its cross-term matrices"
    assert any(
        m.isSquare and not np.array_equal(m.matrix, m.matrix.T) for m in crossTerms
    ), "every cross term came back symmetric; the reader is symmetrising them"


def test_the_parameter_covariance_matrices_are_square(gnds_covariance_fixture):
    """§25.3.2's own invariant, which the model enforces at construction."""
    suite, _ = readCovarianceSuite(Document.parse(gnds_covariance_fixture))
    for covariance in suite.parameterCovariances:
        if isinstance(covariance, ParameterCovariance) and covariance.form is not None:
            matrix = covariance.form.matrix
            assert matrix.shape[0] == matrix.shape[1]


# ---------------------------------------------------------------------------
# the constructs each fixture was committed for
# ---------------------------------------------------------------------------

def test_la139_carries_the_mf34_legendre_slice(gnds_data_dir):
    """``slices/slice[@domainValue]`` — MF34, and the reason this fixture exists.

    ``DataLink.forLegendreOrder`` was written in phase 3b against the
    specification; this is the first time it meets a published file. A
    covariance about Legendre order *l* is the angular distribution **sliced**
    at that order, not a separate quantity.
    """
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-057_La_139.endf.gnds-covar.xml")
    )
    sliced = [
        section for section in suite.covarianceSections
        if section.rowData is not None and len(section.rowData.slices)
    ]
    assert sliced, "La-139 lost its sliced sections"
    orders = {section.rowData.legendreOrder for section in sliced}
    assert orders <= {1, 2, 3, 4}, orders
    assert all(
        (section.rowData.ENDF_MFMT or "").startswith("34,") for section in sliced
    ), "a slice appeared on something that is not an MF34 quantity"


def test_f19_carries_cross_terms_and_a_stated_sum(gnds_data_dir):
    """``crossTerm``, ``columnData``, and ``sum``/``summand`` with coefficients.

    ENDF's NC-type relation — MT4's covariance *is* MT1's minus the others —
    written as a statement instead of a matrix. The signs are the content: a
    reader that dropped the coefficients would turn a difference into a sum.
    """
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml")
    )
    crossTerms = [s for s in suite.covarianceSections if s.isCrossTerm]
    assert crossTerms, "F-19 lost its cross terms"
    assert all(s.crossTerm for s in crossTerms), "the file's own attribute was dropped"
    assert any(s.columnData is not None for s in crossTerms)

    sums = [s.form for s in suite.covarianceSections if isinstance(s.form, Sum)]
    assert sums, "F-19 lost its sum"
    summand = sums[0]
    assert len(summand) > 1
    assert {c for c in summand.coefficients} == {1.0, -1.0}
    assert all(term.ENDF_MFMT for term in summand)
    assert summand.domainMax == 2e7


def test_f19_keeps_short_range_variance_out_of_the_ordinary_components(gnds_data_dir):
    """A ``mixed`` holding one is not a set of matrices to add together."""
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml")
    )
    shortRange = [
        component
        for section in suite.covarianceSections
        if isinstance(section.form, Mixed)
        for component in section.form.components
        if isinstance(component, ShortRangeSelfScalingVariance)
    ]
    assert shortRange, "F-19 lost its shortRangeSelfScalingVariance"
    assert all(s.dependenceOnProcessedGroupWidth == "inverse" for s in shortRange)
    assert all(not isinstance(s, CovarianceMatrix) for s in shortRange)


def test_a_mixed_keeps_its_label(gnds_data_dir):
    """``covariances.xsd:135-142`` makes it ``use="required"``, like every other
    §25.2 form. It was read for the error messages and thrown away, and the
    writer then asked the model for it — see
    ``test_encode.test_a_suite_holding_a_mixed_can_be_written_at_all``."""
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml")
    )
    mixed = [s.form for s in suite.covarianceSections
             if isinstance(s.form, Mixed)]
    assert mixed, "F-19 lost its mixed sections"
    assert all(form.label for form in mixed)


def test_si32_is_a_suite_of_parameter_covariances_only(gnds_data_dir):
    """No ``covarianceSections`` node at all — the shape a reader may assume away."""
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-014_Si_032.endf.gnds-covar.xml")
    )
    assert suite.covarianceSections == []
    assert len(suite.parameterCovariances) == 1
    covariance = suite.parameterCovariances[0]
    assert isinstance(covariance, ParameterCovariance)
    matrix = covariance.form
    assert matrix.order == matrix.matrix.shape[0]
    assert matrix.parameters, "the parameter links were dropped"
    # The links must account for every row, or naming a row is impossible.
    assert sum(link.nParameters for link in matrix.parameters) == matrix.order
    assert len(matrix.rowLabels()) == matrix.order


def test_matrix_start_index_is_read_as_zero_based(gnds_data_dir):
    """``matrixStartIndex`` counts rows from 0, and the arithmetic proves it.

    Si-32's matrix is 19x19 with two links: ``scatteringRadius`` carrying no
    attributes at all, and ``resonanceParameters`` with ``nParameters="18"
    matrixStartIndex="1"``. Those add to 19 only if row 1 is the second row.
    Subtracting one to "convert from one-based" — which this reader did first —
    puts both runs at row 0, and the model's only check is that the counts sum
    to the matrix order, so nothing else would catch it.
    """
    suite, _ = readCovarianceSuite(
        Document.parse(gnds_data_dir / "Covariances/n-014_Si_032.endf.gnds-covar.xml")
    )
    matrix = suite.parameterCovariances[0].form
    links = matrix.parameters
    assert [(link.label, link.nParameters, link.matrixStartIndex) for link in links] == [
        ("scatteringRadius", 1, 0),
        ("resonanceParameters", 18, 1),
    ]
    # The runs must tile the matrix without overlapping, which is the property
    # a one-based misreading destroys.
    for previous, current in zip(links, links[1:]):
        assert current.matrixStartIndex == previous.matrixStartIndex + previous.nParameters
    assert links[-1].matrixStartIndex + links[-1].nParameters == matrix.order


def test_tm171_carries_urr_averages_and_a_flattened_array(gnds_data_dir):
    """``averageParameterCovariance`` — the third kind of covariance row.

    Not bins of a cross-section grid and not individual resonance parameters:
    energy bins of one unresolved-region average, linked into
    ``tabulatedWidths``. Tm-171 is also the smallest file storing an array with
    ``compression="flattened"``.
    """
    path = gnds_data_dir / "Covariances/n-069_Tm_171.endf.gnds-covar.xml"
    suite, _ = readCovarianceSuite(Document.parse(path))
    averages = [c for c in suite.parameterCovariances
                if isinstance(c, AverageParameterCovariance)]
    assert averages, "Tm-171 lost its averageParameterCovariance"
    assert all(a.form is not None for a in averages)
    assert any("levelSpacing" in (a.rowData.href or "") for a in averages)

    root = ET.parse(path).getroot()
    assert any(a.attrib.get("compression") == "flattened" for a in root.iter("array")), \
        "the flattened array this fixture is for is gone"


# ---------------------------------------------------------------------------
# with the reactionSuite in hand
# ---------------------------------------------------------------------------

def test_with_the_pair_supplied_every_link_is_followed(h2_gnds, h2_gnds_cov):
    """The other half of the standalone case: nothing left unresolved."""
    covariances = Document.parse(h2_gnds_cov)
    reactions = Document.parse(h2_gnds)
    suite, report = readCovarianceSuite(covariances, {"reactions": reactions})

    assert suite.covarianceSections
    assert not report.losses, f"links reported as unfollowable: {report.losses}"
    assert all(section.rowData is not None for section in suite.covarianceSections)


def test_the_row_data_says_which_endf_quantity_each_section_is_about(h2_gnds_cov):
    suite, _ = readCovarianceSuite(Document.parse(h2_gnds_cov))
    mfmt = {section.rowData.ENDF_MFMT for section in suite.covarianceSections}
    assert mfmt == {"33,1", "33,2", "33,16", "33,102"}


def test_a_reaction_suite_is_refused_by_this_reader(h2_gnds):
    with pytest.raises(ValueError, match="not a <covarianceSuite>"):
        readCovarianceSuite(Document.parse(h2_gnds))


def test_the_version_gate_applies_to_covariances_too(h2_gnds_cov, tmp_path):
    text = h2_gnds_cov.read_text().replace('format="2.0"', 'format="1.10"', 1)
    path = tmp_path / "old.xml"
    path.write_text(text)
    with pytest.raises(UnsupportedGndsVersion, match="1.10"):
        readCovarianceSuite(Document.parse(path))
