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

from kika.endf.model_adapter import (decodeCovarianceSuite, decodeMF33MT,
                                     decodeMF34MT, encodeMF33MT, encodeMF34MT)
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
    """MF31 and MF32 are covariances this adapter does not convert.

    The MF32 half of this was unverifiable until MF32 got a parser: the branch
    that raises the notice keys off ``endf.mf[32]``, which the registry could
    never populate, so the docstring claimed a behaviour no assertion reached.
    """
    from kika.nuclear_data.model import ConversionReport

    class _Endf:
        mf = {31: object(), 32: object(), 33: covEndf.mf[33]}

    _, report = decodeCovarianceSuite(_Endf(), ConversionReport())
    assert any("MF31" in entry for entry in report.unsupported)
    assert any("MF32" in entry for entry in report.unsupported)


def test_a_file_carrying_only_unconvertible_covariances_does_not_claim_it_has_none(covEndf):
    """An MF32-only evaluation has covariances; it just has none this reads.

    ``read()`` calls ``decodeCovarianceSuite`` for any MF in ``COVARIANCE_MF``,
    which MF32 joined when it got a parser. Without this distinction such a tape
    came back declaring it "carries no covariances", which is the opposite of
    why it was routed here.
    """
    from kika.nuclear_data.model import ConversionReport

    class _Endf:
        mf = {32: object()}

    suiteOut, report = decodeCovarianceSuite(_Endf(), ConversionReport())
    assert len(suiteOut) == 0
    assert any("the only covariances" in entry and "MF32" in entry
               for entry in report.losses)
    assert not any(entry.endswith("carries no covariances")
                   for entry in report.losses)


def test_a_file_with_no_covariances_says_so():
    from kika.nuclear_data.model import ConversionReport

    class _Empty:
        mf: dict = {}

    suiteOut, report = decodeCovarianceSuite(_Empty(), ConversionReport())
    assert len(suiteOut) == 0
    # The message names all three covariance files the adapter now reads. MF35
    # joined MF33 and MF34 with the PFNS work; a tape carrying only MF35 must
    # not be reported as carrying no covariances.
    assert any("no MF33, MF34 or MF35" in entry for entry in report.losses)


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


# ---------------------------------------------------------------------------
# The encode direction
# ---------------------------------------------------------------------------
#
# **The gate here is a fixed point, and deliberately not the byte identity MF1,
# MF3 and MF4 are held to.** Those three decode into a model that keeps the
# file's own record structure -- the (NBT, INT) pairs, the NWD block -- so
# re-encoding can reproduce the bytes. Covariances do not: the decoders read
# through `to_xs_covmat()` / `to_ang_covmat()`, which have already collapsed
# NC/NI subsections and LB types into dense matrices, and `CovarianceMatrix`
# keeps only the matrix, the grids, the frame and `isRelative`.
#
# Claiming byte identity would therefore be claiming something the model cannot
# do, and the honest statement is weaker and still strong: **encoding then
# decoding returns the same arrays**, so nothing is lost or altered in the half
# of the trip the model is responsible for. What the round trip does *not*
# preserve -- an evaluator's per-record NI split -- is declared by the encoder's
# own `ConversionReport` rather than left for a reader to discover.

def _sectionsFor(suite, mf, mt):
    return [s for s in suite.covarianceSections
            if str(s.rowData.ENDF_MFMT or "").startswith(f"{mf}/")
            and int(str(s.rowData.ENDF_MFMT).split("/")[1]) == mt]


def _assertFixedPoint(original, rebuilt, what):
    assert len(rebuilt) == len(original), (
        f"{what}: {len(original)} sections in, {len(rebuilt)} back out"
    )
    for before, after in zip(original, rebuilt):
        assert after.label == before.label, f"{what}: label moved"
        np.testing.assert_array_equal(after.form.matrix, before.form.matrix)
        np.testing.assert_array_equal(after.form.rowGrid, before.form.rowGrid)
        np.testing.assert_array_equal(after.form.columnGrid, before.form.columnGrid)


def test_mf33_survives_a_round_trip_through_the_model(covEndf, suite):
    covarianceSuite, _ = suite
    for mt in sorted(covEndf.mf[33].mt):
        built, _ = encodeMF33MT(covarianceSuite, mt)
        rebuilt, _ = decodeMF33MT(built)
        _assertFixedPoint(_sectionsFor(covarianceSuite, 33, mt), rebuilt, f"MF33/MT{mt}")


def test_mf34_survives_a_round_trip_through_the_model(covEndf, suite):
    covarianceSuite, _ = suite
    for mt in sorted(covEndf.mf[34].mt):
        built, _ = encodeMF34MT(covarianceSuite, mt)
        rebuilt, _ = decodeMF34MT(built)
        _assertFixedPoint(_sectionsFor(covarianceSuite, 34, mt), rebuilt, f"MF34/MT{mt}")


def test_the_encoded_header_is_the_file_s_and_not_a_default(covEndf, suite):
    """`to_mf34` defaults AWR to 1.0 and MAT to 0 when it is told nothing.

    That default is why the header has to be carried: a section written with
    ``awr=1.0`` is structurally valid, silently wrong, and indistinguishable
    from a correct one without the source tape in hand.
    """
    covarianceSuite, _ = suite
    for mf, encode in ((33, encodeMF33MT), (34, encodeMF34MT)):
        for mt in sorted(covEndf.mf[mf].mt):
            source = covEndf.mf[mf].mt[mt]
            built, _ = encode(covarianceSuite, mt)
            assert built._za == pytest.approx(source._za), f"MF{mf}/MT{mt} ZA"
            assert built._awr == pytest.approx(source._awr), f"MF{mf}/MT{mt} AWR"
            assert int(built._mat or 0) == int(source._mat or 0), f"MF{mf}/MT{mt} MAT"


def test_mf34_keeps_the_file_s_ltt_rather_than_inferring_it(covEndf, suite):
    """§34.1: LTT says whether the blocks start at a_0 or a_1.

    `to_mf34` infers it from whether an L=0 pair is present, which is right —
    and preferring the file's own value means the round trip does not depend on
    the inference staying right, the same argument `encodeMF3MT` makes for the
    (NBT, INT) pairs.
    """
    covarianceSuite, _ = suite
    for mt in sorted(covEndf.mf[34].mt):
        built, _ = encodeMF34MT(covarianceSuite, mt)
        assert built._ltt == covEndf.mf[34].mt[mt]._ltt


def test_the_encoder_declares_what_the_round_trip_cannot_preserve(covEndf, suite):
    """Nothing converts silently — the NI collapse is stated, not discovered."""
    covarianceSuite, _ = suite
    mt = sorted(covEndf.mf[34].mt)[0]
    _, report = encodeMF34MT(covarianceSuite, mt)
    assert any("NI" in message for message in report.approximations)


def test_encoding_an_mt_the_suite_does_not_carry_raises(suite):
    covarianceSuite, _ = suite
    with pytest.raises(ValueError, match="MT999"):
        encodeMF34MT(covarianceSuite, 999)


@pytest.mark.parametrize("tape", ["fe56_host_tape", "fe56_jendl_tape"])
def test_the_fixed_point_holds_on_real_tapes(request, tape):
    """Under ``--deep``, on real MF33/MF34 rather than the synthetic fixture."""
    endf = read_endf(str(request.getfixturevalue(tape)))
    if 33 not in endf.mf:
        pytest.skip(f"{tape} carries no MF33")
    covarianceSuite, _ = decodeCovarianceSuite(endf)

    for mt in sorted(endf.mf[33].mt):
        built, _ = encodeMF33MT(covarianceSuite, mt)
        rebuilt, _ = decodeMF33MT(built)
        _assertFixedPoint(_sectionsFor(covarianceSuite, 33, mt), rebuilt,
                          f"{tape} MF33/MT{mt}")

    for mt in sorted(getattr(endf.mf.get(34), "mt", {})):
        built, _ = encodeMF34MT(covarianceSuite, mt)
        rebuilt, _ = decodeMF34MT(built)
        _assertFixedPoint(_sectionsFor(covarianceSuite, 34, mt), rebuilt,
                          f"{tape} MF34/MT{mt}")
