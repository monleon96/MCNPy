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
