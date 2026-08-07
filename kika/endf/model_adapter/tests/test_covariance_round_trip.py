"""MF33 and MF34 → ``CovarianceSuite``, element for element.

The gate is ``assert_array_equal``: every matrix in the suite is the same array
``kika/cov`` already produces, not a rounded or re-projected version of it. The
adapter re-expresses; it must not recompute.

The sharper half is the **MF34 shape**. §25.2.5-6 make a Legendre-order
covariance a *slice* of the angular distribution at that order. The plausible
wrong model — one quantity per order — produces a structurally valid file that
misstates what the data is, and nothing downstream notices until someone reads
it back. So the order is asserted to be a slice ``domainValue``, and the
cross-order block (L1 × L2) is asserted to exist, because that block is the
thing an order-per-quantity model cannot express at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF33MT, decodeMF34MT
from kika.endf.read_endf import read_endf


@pytest.fixture(scope="module")
def covEndf(micro_cov_tape):
    return read_endf(str(micro_cov_tape))


@pytest.fixture(scope="module")
def suite(covEndf):
    covarianceSuite, report = decodeCovarianceSuite(covEndf, evaluation="micro")
    return covarianceSuite, report


def test_the_fixture_carries_both_mf33_and_mf34(covEndf):
    """A tape with only one of them would make half the file vacuous."""
    assert 33 in covEndf.mf and 34 in covEndf.mf


def test_every_mf33_matrix_is_the_array_kika_cov_already_produced(covEndf):
    for mt in sorted(covEndf.mf[33].mt):
        section = covEndf.mf[33].mt[mt]
        expected = section.to_xs_covmat()
        decoded, _ = decodeMF33MT(section)

        assert len(decoded) == len(expected.matrices)
        for entry, matrix in zip(decoded, expected.matrices):
            np.testing.assert_array_equal(entry.form.matrix, np.asarray(matrix, dtype=float))


def test_every_mf34_matrix_is_the_array_kika_cov_already_produced(covEndf):
    for mt in sorted(covEndf.mf[34].mt):
        section = covEndf.mf[34].mt[mt]
        expected = section.to_ang_covmat()
        decoded, _ = decodeMF34MT(section)

        assert len(decoded) == len(expected.matrices)
        for entry, matrix in zip(decoded, expected.matrices):
            np.testing.assert_array_equal(entry.form.matrix, np.asarray(matrix, dtype=float))


def test_the_energy_grids_survive_unchanged(covEndf):
    for mt in sorted(covEndf.mf[34].mt):
        expected = covEndf.mf[34].mt[mt].to_ang_covmat()
        decoded, _ = decodeMF34MT(covEndf.mf[34].mt[mt])
        for entry, grid in zip(decoded, expected.energy_grids):
            np.testing.assert_array_equal(
                entry.form.rowGrid, np.asarray(grid, dtype=float)
            )


# ---------------------------------------------------------------------------
# The shape, which is the part with a real chance of being wrong
# ---------------------------------------------------------------------------

def test_a_legendre_order_is_a_slice_domain_value_not_a_separate_quantity(suite):
    covarianceSuite, _ = suite
    mf34 = [s for s in covarianceSuite if s.label.startswith("MF34")]
    assert mf34, "no MF34 sections decoded"

    for section in mf34:
        assert section.rowData.legendreOrder is not None, (
            f"{section.label}: the row link carries no Legendre-order slice. "
            f"§25.2.5-6 make the order a slice of the angular distribution, not "
            f"a covariance of a separate quantity."
        )
        assert len(section.rowData.slices) == 1
        assert section.rowData.slices.slices[0].domainValue is not None
        # A Legendre order is an index, not a physical quantity, so it has no unit.
        assert section.rowData.slices.slices[0].domainUnit == ""


def test_the_cross_order_block_survives(suite):
    """L1 × L2 is what an order-per-quantity model could not express at all."""
    covarianceSuite, _ = suite
    crossOrder = [
        s for s in covarianceSuite
        if s.rowData is not None and s.columnData is not None
        and s.rowData.legendreOrder is not None
        and s.rowData.legendreOrder != s.columnData.legendreOrder
    ]
    assert crossOrder, (
        "the fixture's L1 x L2 block did not survive the conversion; a covariance "
        "suite without its cross-order blocks is a list of variances"
    )


def test_mf33_links_carry_no_legendre_slice(suite):
    """A cross section is a whole quantity, so there is nothing to slice."""
    covarianceSuite, _ = suite
    for section in (s for s in covarianceSuite if s.label.startswith("MF33")):
        assert section.rowData.legendreOrder is None
        assert len(section.rowData.slices) == 0


def test_the_links_say_which_endf_section_they_came_from(suite):
    """``ENDF_MFMT`` (§25.2.3) is what lets a reader get back to the source."""
    covarianceSuite, _ = suite
    for section in covarianceSuite:
        assert section.rowData.ENDF_MFMT, f"{section.label} has no ENDF_MFMT"
        assert section.rowData.href.startswith("/reactionSuite/")


def test_the_report_declares_the_covariance_files_it_does_not_read(covEndf):
    """MF31 and MF32 are covariances this adapter does not convert."""
    from kika.nuclear_data.model import ConversionReport

    class _Endf:
        mf = {31: object(), 33: covEndf.mf[33]}

    _, report = decodeCovarianceSuite(_Endf(), ConversionReport())
    assert any("MF31" in entry for entry in report.unsupported)


def test_a_file_with_no_covariances_says_so():
    from kika.nuclear_data.model import ConversionReport

    class _Empty:
        mf: dict = {}

    suiteOut, report = decodeCovarianceSuite(_Empty(), ConversionReport())
    assert len(suiteOut) == 0
    assert any("no MF33 and no MF34" in entry for entry in report.losses)


def test_an_a0_section_reaches_the_model_with_its_order_zero_blocks():
    """The gate the branch merge needed, and the one nothing else covers.

    ``develop`` fixed two MF34 defects while this adapter was being written on a
    branch (``98d7d23``): NL is the *number* of Legendre coefficients, not the
    highest index, and a section carrying a_0 is LTT=3 rather than LTT=2. Both
    live in ``parse_mf34``/``mf34_writer``, and this decoder reads through
    ``to_ang_covmat`` rather than counting sub-subsections itself — so the fix
    propagates for free. "For free" is a claim, and this is the measurement.

    Every other MF34 test here runs on evaluations whose sections start at a_1,
    where the count convention and the highest index coincide and the two
    readings are indistinguishable. The sections that carry a_0 are the ones
    *kika itself writes* — the sigma↔a_l cross blocks — so this builds one, puts
    it through the writer and the parser, and asserts the model gets every
    block including the L=0 row. Under the pre-fix reading NSS was
    under-declared, ``parse_mf34_mt`` looped NSS times, and the tail of the
    section went missing with no error anywhere.
    """
    from kika.endf.parsers.parse_mf34 import parse_mf34_mt
    from kika.endf.writers.mf34_writer import create_mf34_from_covariance

    maxOrder, nShape, nCross = 2, 3, 2
    shapeGrid = np.array([0.85e6, 1.5e6, 2.5e6, 4.0e6])
    crossGrid = np.array([0.85e6, 2.5e6, 4.0e6])
    base = np.arange(1, nShape * maxOrder + 1, dtype=float)
    shapeCov = 0.0005 * np.outer(base, base) + np.diag(0.01 * base)
    shapeCov = 0.5 * (shapeCov + shapeCov.T)

    written = create_mf34_from_covariance(
        shapeCov, shapeGrid, maxOrder, 26056.0, 55.454, 2631, 2,
        ltt=1,
        cross_cov={1: 0.01 * np.ones((nCross, nShape)),
                   2: -0.005 * np.ones((nCross, nShape))},
        cross_energy_grid_ev=crossGrid,
    )
    assert written._ltt == 3, "the premise is gone: this is no longer an a_0 section"

    reparsed = parse_mf34_mt(str(written).splitlines(), 2)
    sections, _ = decodeMF34MT(reparsed)

    orders = {(s.rowData.legendreOrder, s.columnData.legendreOrder)
              for s in sections}
    assert orders == {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}, (
        f"the model did not receive the full a_0 upper triangle: {sorted(orders)}"
    )
    nl = maxOrder + 1
    assert len(sections) == nl * (nl + 1) // 2


@pytest.mark.parametrize("tape", ["fe56_host_tape", "fe56_jendl_tape"])
def test_real_tapes_convert_element_for_element(request, tape):
    """Under ``--deep``, on the real MF33/MF34 rather than the synthetic fixture."""
    endf = read_endf(str(request.getfixturevalue(tape)))
    if 33 not in endf.mf:
        pytest.skip(f"{tape} carries no MF33")

    for mt in sorted(endf.mf[33].mt):
        expected = endf.mf[33].mt[mt].to_xs_covmat()
        decoded, _ = decodeMF33MT(endf.mf[33].mt[mt])
        for entry, matrix in zip(decoded, expected.matrices):
            np.testing.assert_array_equal(entry.form.matrix, np.asarray(matrix, dtype=float))
