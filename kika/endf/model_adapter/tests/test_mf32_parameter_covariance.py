"""MF32 → §25.3 ``parameterCovariances``.

The gate here cannot be the byte identity ``test_mf32_roundtrip.py`` holds the
flat classes to, and it cannot be the numerical fixed point MF33/MF34 get
either: this decode goes one way, from a file that stores correlations as
packed integers beside separately-stored standard deviations, to a covariance.
So the gate is **the file's own arithmetic**. Every body type states, in a field
of its own, how many rows its matrix has — NNN for LCOMP=2, ``MPAR*NRB`` for
LCOMP=1, ``sum (NCH+1)*NRSA`` for LRF=7 — and a decoder that mis-assigned rows
would disagree with it. The tests below assert against those counts and against
the *values* the flat classes read, never against numbers written out by hand.

The sharpest of them is :func:`test_the_vector_is_resonance_major`. Read
parameter-major, every matrix here still has the right order, is still
symmetric, and still passes every structural check — it just describes the
wrong physical quantities. Only comparing an individual row's standard
deviation against the flat record's own DGN can tell the two apart, which is
why that test reaches into the flat section rather than trusting the model.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kika.endf.classes.mf32.mf32mt151 import LCOMP0Body
from kika.endf.classes.mf32.records import PackedList, Record
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF32MT
from kika.endf.parsers.parse_endf import parse_endf_file
from kika.sampling.core import draw_samples
from kika.sampling.model_blocks import (parameter_covariance_blocks,
                                        parameter_covariance_index)


@pytest.fixture(scope="module")
def mf32Endf(micro_mf32_tape):
    return parse_endf_file(str(micro_mf32_tape))


@pytest.fixture(scope="module")
def decoded(mf32Endf):
    return decodeMF32MT(mf32Endf.mf[32].mt[151])


def _byName(endf):
    """The tape's nuclide, so a test can key on which micro-tape it got."""
    return int(round(float(endf.mf[32].mt[151]._za)))


# ---------------------------------------------------------------------------
# Structure — true of every body type
# ---------------------------------------------------------------------------

def test_every_micro_tape_decodes_to_at_least_one_node(decoded):
    covariances, report = decoded
    assert covariances, f"nothing decoded; report says {report.summary()}"


def test_the_links_account_for_exactly_the_rows_they_index(decoded):
    """The invariant that makes a row interpretable at all.

    ``ParameterCovarianceMatrix`` raises on construction if these disagree, so
    this asserts the decoder built links deliberately rather than the class
    tolerating a mismatch.
    """
    covariances, _ = decoded
    for covariance in covariances:
        form = covariance.form
        assert sum(link.nParameters for link in form.parameters) == form.order
        assert len(form.rowLabels()) == form.order
        assert form.parameterValues.size == form.order


def test_every_matrix_is_symmetric_and_finite(decoded):
    covariances, _ = decoded
    for covariance in covariances:
        matrix = covariance.form.matrix
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)


def test_the_diagonal_is_never_negative(decoded):
    """A negative variance is a decode error, not an evaluation's choice.

    Off-diagonal defects are a different matter: the INTG packing rounds every
    correlation to NDIGIT digits, which costs these matrices positive
    definiteness outright — see :func:`test_the_matrices_are_not_assumed_psd`.
    """
    covariances, _ = decoded
    for covariance in covariances:
        assert (np.diag(covariance.form.matrix) >= 0).all()


def test_the_matrices_are_not_assumed_psd(decoded):
    """Pinned, because it is a property of MF32 that sampling has to handle.

    The decoder must not quietly repair this: the file says what it says, and
    which PSD projection to apply is the sampler's decision (``psd_method``),
    not the reader's. If a future change starts returning exactly-PSD matrices
    here, that is a silent numerical edit and this test is where it surfaces.
    """
    covariances, _ = decoded
    smallest = min(
        float(np.linalg.eigvalsh(c.form.matrix).min()) for c in covariances
    )
    largest = max(
        float(np.linalg.eigvalsh(c.form.matrix).max()) for c in covariances
    )
    # Negative eigenvalues are admissible; being *large* relative to the
    # spectrum would mean a structural error rather than INTG rounding.
    assert smallest > -1e-3 * largest


# ---------------------------------------------------------------------------
# The measured facts, one test each
# ---------------------------------------------------------------------------

def test_the_vector_is_resonance_major(mf32Endf):
    """Every parameter of resonance k, then every parameter of resonance k+1.

    Th-232 is the tape that can show it: 927 resonances by three retained
    parameters, so a parameter-major reading shifts row 3k+1 from resonance k's
    GN to resonance ``(k*3+1) // 927``'s ER and nothing else changes shape.
    """
    if _byName(mf32Endf) != 90232:
        pytest.skip("resonance-major is asserted on Th-232")

    section = mf32Endf.mf[32].mt[151]
    body = section.isotopes[0].energy_ranges[0].body
    flat = np.asarray(body.parameters.values, dtype=float).reshape(-1, 12)

    covariances, _ = decodeMF32MT(section)
    form = covariances[0].form
    sigma = form.uncertainties()

    # LRF=3 slots are (ER, AJ, GN, GG, GFA, GFB); the retained three are
    # ER, GN, GG, so resonance k owns rows 3k, 3k+1, 3k+2.
    for k in (0, 1, 42, 500, 926):
        np.testing.assert_allclose(sigma[3 * k + 0], flat[k, 6 + 0], rtol=1e-12)
        np.testing.assert_allclose(sigma[3 * k + 1], flat[k, 6 + 2], rtol=1e-12)
        np.testing.assert_allclose(sigma[3 * k + 2], flat[k, 6 + 3], rtol=1e-12)
        np.testing.assert_allclose(form.parameterValues[3 * k + 1], flat[k, 2],
                                   rtol=1e-12)


def test_lcomp2_covers_columns_not_entries(mf32Endf):
    """Na-23: 69 rows against 65 non-zero uncertainties.

    The four rows in between are the point. A decoder that retained parameters
    *entry* by entry would build a 65-row matrix, disagree with NNN=69, and —
    if it did not check NNN — shift every row after the first zero onto the
    wrong resonance.
    """
    if _byName(mf32Endf) != 11023:
        pytest.skip("the column/entry distinction is visible on Na-23")

    section = mf32Endf.mf[32].mt[151]
    body = section.isotopes[0].energy_ranges[0].body
    flat = np.asarray(body.parameters.values, dtype=float).reshape(-1, 12)
    nonZeroEntries = int(np.count_nonzero(flat[:, 6:]))

    covariances, _ = decodeMF32MT(section)
    form = covariances[0].form

    assert form.order == int(body.correlations.nnn) == 69
    assert nonZeroEntries == 65
    assert int((form.uncertainties() == 0).sum()) == 4
    assert form.parameters[0].parameterNames == ["ER", "GN", "GG"]


def test_lrf7_retains_every_parameter_including_the_zero_ones(mf32Endf):
    """Cl-35: NNN is ``sum (NCH+1)*NRSA``, zero uncertainties included.

    The opposite rule to the one above, and the reason the two RML bodies have
    their own decoder rather than sharing the LCOMP=2 one.
    """
    if _byName(mf32Endf) != 17035:
        pytest.skip("LRF=7 retention is asserted on Cl-35")

    section = mf32Endf.mf[32].mt[151]
    body = section.isotopes[0].energy_ranges[0].body
    expected = sum((g.nch + 1) * g.nrsa for g in body.spin_groups)

    covariances, _ = decodeMF32MT(section)
    form = covariances[0].form

    assert form.order == expected == int(body.correlations.nnn) == 1088
    assert int((form.uncertainties() == 0).sum()) > 0, (
        "Cl-35 has unassigned channels; if none survive, the zero rows were "
        "dropped and the retention rule silently became the LCOMP=2 one"
    )
    assert form.parameters[0].parameterNames == ["ER", "GAM1", "GAM2", "GAM3"]
    assert form.rowLabels()[0] == "spinGroup0/resonance0/ER"


def test_lcomp0_is_block_diagonal_by_resonance(mf32Endf):
    """Cm-244: §32.2.1 correlates nothing between two resonances.

    Block-diagonality here is the format's content, not an approximation, so
    an off-diagonal entry between resonances would mean the 18-number stride
    had slipped.
    """
    if _byName(mf32Endf) != 96244:
        pytest.skip("LCOMP=0 is committed as Cm-244")

    covariances, _ = decodeMF32MT(mf32Endf.mf[32].mt[151])
    form = covariances[0].form
    assert form.parameters[0].parameterNames == ["ER", "GN", "GG", "GF"]

    matrix = form.matrix.copy()
    for start in range(0, form.order, 4):
        matrix[start:start + 4, start:start + 4] = 0.0
    assert not matrix.any(), "an LCOMP=0 matrix correlated two resonances"


def test_the_unresolved_range_is_relative_and_the_resolved_one_is_not(mf32Endf):
    """Th-232 carries both, which is why it is the committed two-range tape.

    §32.2.4's matrix is a *relative* covariance and §32.2.3's is absolute.
    Losing that distinction is invisible structurally and wrong by the square
    of a level spacing.
    """
    if _byName(mf32Endf) != 90232:
        pytest.skip("only Th-232 has both a resolved and an unresolved range")

    covariances, _ = decodeMF32MT(mf32Endf.mf[32].mt[151])
    assert len(covariances) == 2
    resolved, unresolved = covariances

    assert resolved.form.isRelative is False
    assert unresolved.form.isRelative is True
    assert unresolved.form.order == 15    # MPAR=3 over five (L, J) states
    assert unresolved.form.parameters[0].parameterNames == ["D", "GNO", "GG"]
    assert unresolved.rowData.href.endswith("tabulatedWidths")
    assert resolved.rowData.href.endswith("resonanceParameters/table")


def test_the_row_link_carries_the_energy_range(decoded):
    """Which range a parameter set belongs to is not recoverable otherwise."""
    covariances, _ = decoded
    for covariance in covariances:
        band = covariance.rowData.incidentEnergyBand
        assert band is not None and band[1] > band[0]


def test_reich_moore_points_at_an_r_matrix_node(mf32Endf):
    """LRF=3 is an R-matrix approximation, and kika decodes it onto ``RMatrix``.

    A href naming ``BreitWigner`` would resolve to nothing for exactly the
    evaluations MF32 is most often written for.
    """
    section = mf32Endf.mf[32].mt[151]
    covariances, _ = decodeMF32MT(section)
    for covariance, energyRange in zip(
            covariances, section.isotopes[0].energy_ranges):
        if energyRange.lru != 1:
            continue
        expected = "BreitWigner" if energyRange.lrf in (1, 2) else "RMatrix"
        assert f"/{expected}/" in covariance.rowData.href


# ---------------------------------------------------------------------------
# The guard against the failure mode that has no other symptom
# ---------------------------------------------------------------------------

def test_a_disagreeing_nnn_is_reported_and_not_reshaped(mf32Endf):
    """The one check that stands between a wrong row map and a plausible matrix.

    Every structural assertion above still passes if the retained-column rule
    is wrong — the matrix is square, symmetric and full of real numbers. NNN is
    the file's own statement of the order, so a decoder that disagrees with it
    must refuse rather than produce something that looks fine.
    """
    section = mf32Endf.mf[32].mt[151]
    body = section.isotopes[0].energy_ranges[0].body
    if getattr(body, "correlations", None) is None:
        pytest.skip("this tape's first range has no INTG record to doctor")

    original = body.correlations.nnn
    try:
        body.correlations.nnn = original + 1
        covariances, report = decodeMF32MT(section)
    finally:
        body.correlations.nnn = original

    assert any("NNN" in line for line in report.losses)
    assert all("range0" not in c.label for c in covariances)


def test_lcomp0_puts_the_width_cross_terms_where_they_belong():
    """Synthetic, and it has to be: no real tape exercises this.

    Both LCOMP=0 evaluations on this machine (Cm-244, Am-241) write zero for
    DNDG, DNDF and DGDF, so the correlation part of §32.2.1's 4x4 is reachable
    only by construction. The alternative is leaving three of the seven decoded
    terms untested, which is how a transposed pair survives for years.
    """
    er, aj, gt, gn, gg, gf = 10.0, 0.5, 3.0, 1.0, 2.0, 0.5
    de2, dn2, dndg, dg2, dndf, dgdf, df2 = 4.0, 9.0, 1.5, 16.0, 2.5, 3.5, 25.0
    values = [er, aj, gt, gn, gg, gf,
              de2, dn2, dndg, dg2, dndf, dgdf, df2,
              0.0, 0.0, 0.0, 0.0, 0.0]

    block = Record(raw="", l2=0, n1=18, n2=1, body=PackedList())
    block.body.set_values(values)
    body = LCOMP0Body(control=Record(raw="", n1=1), l_blocks=[block])

    from kika.endf.model_adapter.parameter_covariances import _decodeLCOMP0
    from kika.nuclear_data.model import ConversionReport

    form = _decodeLCOMP0(body, lrf=2, href="/x", report=ConversionReport())

    expected = np.array([
        [de2, 0.0,  0.0,  0.0],
        [0.0, dn2,  dndg, dndf],
        [0.0, dndg, dg2,  dgdf],
        [0.0, dndf, dgdf, df2],
    ])
    np.testing.assert_array_equal(form.matrix, expected)
    np.testing.assert_array_equal(form.parameterValues, [er, gn, gg, gf])


# ---------------------------------------------------------------------------
# LCOMP=1 — no micro-tape, so this is the only coverage it has
# ---------------------------------------------------------------------------

def _mf32Section(path):
    """The MF32/MT151 section of a real tape, without parsing the rest of it.

    Mn-55's MF32 is 26 467 lines and its tape carries eight other files; going
    through ``parse_endf_file`` would spend most of a minute on data no test
    here looks at.
    """
    from kika.endf.parsers.parse_mf32 import parse_mf32_mt151
    from kika.endf.utils import parse_endf_id
    from pathlib import Path

    lines = []
    for line in Path(path).read_text().splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf == 32 and mt == 151:
            lines.append(line.rstrip())
    return parse_mf32_mt151(lines)


def test_lcomp1_decodes_and_is_resonance_major(mn55_b81_tape):
    """The only sub-format with no committed fixture, on the only tape that has it.

    ``MPAR*NRB`` is the file's own statement of the order, and the standard
    deviations are recoverable from the stored triangle independently of the
    decoder — so this compares the decode against the block's own diagonal
    rather than against anything written here.
    """
    section = _mf32Section(mn55_b81_tape)
    block = section.isotopes[0].energy_ranges[0].body.short_range[0]
    mpar, count = int(block.l1), int(block.n2)
    order = mpar * count

    covariances, report = decodeMF32MT(section)
    assert len(covariances) == 1
    form = covariances[0].form

    assert form.order == order == 561
    assert form.parameters[0].parameterNames == ["ER", "GN", "GG"]
    assert form.isRelative is False

    # The stored upper triangle, walked independently of the decoder.
    raw = np.asarray(block.values, dtype=float)
    table = raw[:6 * count].reshape(count, 6)
    triangle = raw[6 * count:]
    diagonal = np.empty(order)
    cursor = 0
    for row in range(order):
        diagonal[row] = triangle[cursor]
        cursor += order - row
    np.testing.assert_allclose(form.matrix.diagonal(), diagonal, rtol=1e-12)

    # LRF=3 slots are (ER, AJ, GN, GG, GFA, GFB) and MPAR=3 keeps ER, GN, GG.
    for k in (0, 1, 100, 186):
        np.testing.assert_allclose(form.parameterValues[3 * k + 0], table[k, 0])
        np.testing.assert_allclose(form.parameterValues[3 * k + 1], table[k, 2])
        np.testing.assert_allclose(form.parameterValues[3 * k + 2], table[k, 3])
    assert not report.losses


def test_lcomp1_short_range_blocks_become_separate_nodes(mn55_b81_tape):
    """One node per block, not one padded block-diagonal matrix.

    Mn-55 has a single short-range block so the count is 1 here; what the test
    pins is the *rule*, because the alternative — zero-padding disjoint blocks
    into a common matrix — would state a zero covariance between resonances the
    file says nothing about.
    """
    section = _mf32Section(mn55_b81_tape)
    body = section.isotopes[0].energy_ranges[0].body
    covariances, _ = decodeMF32MT(section)
    assert len(covariances) == len(body.short_range)


# ---------------------------------------------------------------------------
# The suite, and the sampling seam
# ---------------------------------------------------------------------------

def test_the_suite_files_mf32_under_parameter_covariances(mf32Endf):
    """§25.3 and §25.2 are different lists, and MF32 belongs to the first.

    Also pins that the old "the model has nowhere to put it" notice is gone:
    it stayed accurate only until this decoder existed.
    """
    suite, report = decodeCovarianceSuite(mf32Endf, evaluation="micro")
    assert suite.parameterCovariances
    assert not suite.covarianceSections   # these micro-tapes carry no MF31/33/34/35
    assert not any("nowhere to put it" in line for line in report.unsupported)


def test_the_blocks_reach_draw_samples_unchanged(mf32Endf):
    """The test of whether the seam is in the right place.

    ``draw_samples`` is the same function the PFNS path calls and was not
    touched for MF32; if a parameter covariance needed anything of it, the
    split between the model and the sampler would not be real.
    """
    suite, _ = decodeCovarianceSuite(mf32Endf)
    blocks = parameter_covariance_blocks(suite, isotope="micro", relative=False)
    assert blocks

    # One block, and the smallest, so the test stays cheap on Th-232's 2781.
    key, matrix = min(blocks, key=lambda item: item[1].shape[0])
    draws, diagnostics = draw_samples(
        [(key, matrix)], n_samples=8, seed=1, returns="deltas", verbose=False,
    )
    assert draws[key].shape == (8, matrix.shape[0])
    assert np.isfinite(draws[key]).all()
    assert key in diagnostics


def test_the_index_says_what_each_row_is(mf32Endf):
    """A drawn parameter vector is uninterpretable without this."""
    suite, _ = decodeCovarianceSuite(mf32Endf)
    blocks = parameter_covariance_blocks(suite, isotope="micro")
    index = parameter_covariance_index(suite, isotope="micro")

    assert set(index) == {key for key, _ in blocks}
    for key, matrix in blocks:
        entry = index[key]
        assert len(entry["labels"]) == matrix.shape[0]
        assert entry["values"].size == matrix.shape[0]
        assert entry["href"]


def test_relative_and_absolute_blocks_are_separable(mf32Endf):
    """Th-232 mixes them, and one ``draw_samples`` call cannot serve both.

    An absolute block wants ``returns="deltas"`` and a relative one
    ``returns="factors"``; the filter is what keeps a caller from applying one
    convention to the other by accident.
    """
    if _byName(mf32Endf) != 90232:
        pytest.skip("only Th-232 carries both kinds")

    suite, _ = decodeCovarianceSuite(mf32Endf)
    absolute = parameter_covariance_blocks(suite, relative=False)
    relative = parameter_covariance_blocks(suite, relative=True)

    assert len(absolute) == 1 and len(relative) == 1
    assert len(parameter_covariance_blocks(suite)) == 2
