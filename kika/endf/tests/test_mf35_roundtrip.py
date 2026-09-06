"""MF35: the read-write gate, and the reading the whole sampler rests on.

The gate is the same contract as MF5's — parse and re-emit gives the bytes
back, at 75 columns on ENDF/B-VIII.1 and all 80 on JEFF-4.0.

The rest of this module is about **what the LB=7 matrix is the covariance of**.
It is the covariance of the group-integrated probabilities
``P_i = ∫_{g_i}^{g_i+1} χ(E→E') dE'``, not of the spectrum density χ(E'). Every
design decision downstream follows from that: linear sampling space, the
normalisation constraint, the P⁰-weighted projection. So the reading is not
asserted once and trusted — it is tested from both sides. ``C·1 ≈ 0`` is what a
covariance of quantities summing to one must satisfy, and the dE-weighted
version *not* vanishing is what makes the first test evidence rather than a
coincidence that any smooth matrix would produce.

If a future tape fails ``test_mf35_row_sums_vanish``, the reading is wrong for
that tape and the sampler must refuse it rather than perturb something else.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf35.mf35 import MF35SubSection
from kika.endf.parsers.parse_mf35 import parse_mf35_mt
from kika.endf.utils import parse_endf_id

# The MF5 module owns these two; the tests directory has no ``__init__.py``, so
# pytest's prepend import mode puts it on ``sys.path`` and a plain import works.
# Shared rather than copied a third time: ``first_difference`` exists so that a
# failure on a 137 777-line section reports one line instead of building a diff
# of the whole thing, and three drifting copies of that would defeat the point.
from test_mf5_roundtrip import first_difference, roundtrip  # noqa: E402

#: The threshold for ``max|Σ_j C_ij| / max|C|``. Set from the measurement, not
#: from taste: JEFF-4.0 U-235 band 0 comes in at 3.08e-3, three orders of
#: magnitude above every other band on every tape read. 1e-4 would fail on a
#: real evaluation; 1e-2 passes all three tapes and still fails by orders of
#: magnitude on a density-covariance reading, whose ratio is ~6e+4.
ROW_SUM_TOLERANCE = 1e-2


def measured_row_sums(section) -> list[float]:
    return [band.row_sum_residual() for band in section.subsections]


def de_weighted_row_sum(band) -> float:
    """The same ratio with each column weighted by its group width.

    What the residual *would* have to look like if the matrix were the
    covariance of the spectrum density: then ``Σ_j C_ij ΔE_j`` would be the
    quantity constrained to zero, not the bare sum.
    """
    matrix = band.matrix()
    widths = np.diff(band.energy_grid())
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs((matrix * widths).sum(axis=1))) / scale)


# ---------------------------------------------------------------------------
# The densifier, on data small enough to check by eye
# ---------------------------------------------------------------------------

def test_lb7_densify_upper_triangle():
    """A hand-built 3×3, row-major upper triangle including the diagonal.

    Fifteen lines of code, and the one place a transposition or an off-by-one
    in the triangle ordering hides. The matrix is deliberately asymmetric in
    its *values* so that a transposed reading gives a different answer.
    """
    band = MF35SubSection(
        e1=0.0, e2=1.0, ls=1, lb=7, ne=4,
        nt=MF35SubSection.expected_nt(4),
        boundaries=[0.0, 1.0, 2.0, 3.0],
        upper_triangle=[1.0, 2.0, 3.0,
                        4.0, 5.0,
                        6.0],
    )
    np.testing.assert_array_equal(band.matrix(), np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0],
    ]))
    assert band.order == 3
    assert band.nt == 4 + 6


def test_expected_nt_is_the_identity_the_parser_checks():
    for ne, expected in [(2, 3), (4, 10), (123, 7626), (303, 46056), (641, 205761)]:
        assert MF35SubSection.expected_nt(ne) == expected


def lb7_lines(ls: int = 1, lb: int = 7, ne: int = 3,
              nt: int | None = None) -> list[str]:
    """A minimal two-band-free MF35/MT18 section, for the rejection tests."""
    nt = MF35SubSection.expected_nt(ne) if nt is None else nt
    return [
        " 9.223500+4 2.330248+2          0          0          1          0922835 18",
        f" 1.000000-5 1.000000+3{ls:11d}{lb:11d}{nt:11d}{ne:11d}922835 18",
        " 0.000000+0 1.000000+6 2.000000+7 1.000000-4 2.000000-5 3.000000-4922835 18",
    ]


def test_lb7_rejects_ls0_and_other_lb():
    """Anything but LS=1/LB=7 stops by name rather than being mis-split.

    A body read with the wrong record shape produces a plausible-looking matrix
    of the wrong quantity, which nothing downstream could detect.
    """
    with pytest.raises(ValueError, match=r"is LS=0, LB=7"):
        parse_mf35_mt(lb7_lines(ls=0), 18)
    with pytest.raises(ValueError, match=r"is LS=1, LB=5"):
        parse_mf35_mt(lb7_lines(lb=5), 18)


def test_lb7_rejects_an_nt_that_contradicts_ne():
    with pytest.raises(ValueError, match=r"declares NT=7 for NE=3.*must be 6"):
        parse_mf35_mt(lb7_lines(nt=7), 18)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_mf35_roundtrip_cols_1_75(micro_pfns_tape):
    """Four real LB=7 bands, 5 089 lines, byte for byte."""
    assert not roundtrip(micro_pfns_tape, 35, 18, 75)


def test_jeff40_u235_mf35_roundtrip_is_byte_identical(u235_tape):
    """Eight bands at the full 80 columns, sequence numbers included."""
    assert not roundtrip(u235_tape, 35, 18, 80)


@pytest.mark.slow
def test_endfb81_u235_mf35_roundtrip(u235_b81_tape):
    """Five bands, 137 777 lines, and **no padding xfail**.

    Worth stating rather than leaving as a silent pass: MF5 needs an xfail on
    this tape because interpolation records are zero-filled there and kika
    blank-pads them. An MF35 LIST record has no interpolation record, so the
    defect has nothing to attach to and the gate holds outright.
    """
    assert not roundtrip(u235_b81_tape, 35, 18, 75)


# ---------------------------------------------------------------------------
# What the matrix is the covariance of
# ---------------------------------------------------------------------------

def test_mf35_row_sums_vanish(micro_pfns_tape):
    """``C·1 ≈ 0`` — forced by ``Σ_i P_i = 1``, and measured on every band.

    **This is the test that pins the group-integrated-probability reading.**
    Measured on Cf-252: 4.2e-6, 2.2e-6, 4.2e-6, 7.1e-6.
    """
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    residuals = measured_row_sums(section)
    assert len(residuals) == 4
    assert max(residuals) < ROW_SUM_TOLERANCE
    assert max(residuals) < 1e-5, f"Cf-252 measured 7.1e-6, got {max(residuals):.2e}"


def test_mf35_de_weighted_row_sums_do_not_vanish(micro_pfns_tape):
    """The converse, and the reason the first test is evidence.

    If MF35 held the covariance of the spectrum *density*, the constrained
    combination would be ``Σ_j C_ij ΔE_j`` and that is what would vanish. It
    does not — measured ~6e+4 on Cf-252, ten orders of magnitude away from the
    unweighted sum. Both tests passing at once would mean the matrix was simply
    small, and neither reading would be supported.
    """
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    for index, band in enumerate(section.subsections):
        weighted = de_weighted_row_sum(band)
        assert weighted > 1e3, (
            f"band {index}: dE-weighted row sums came out at {weighted:.2e}, "
            f"which would support reading MF35 as a density covariance"
        )


def test_expected_rank_deficiency(micro_pfns_tape):
    """Assert the null space, do not repair it.

    Roughly half to two thirds of every band is null — the evaluators built
    these from a handful of model parameters. It is a property of the data, so
    the sampler records ``n_null`` and moves on. What it must never do is
    "fix" the matrix to full rank, which would invent uncertainty the
    evaluation does not claim.

    Measured on Cf-252: 63, 61, 61 and 62 null directions out of 122.
    """
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    for index, band in enumerate(section.subsections):
        eigenvalues = np.linalg.eigvalsh(band.matrix())
        n_null = int((eigenvalues <= 1e-10 * eigenvalues.max()).sum())
        assert n_null >= 1, f"band {index} is full rank; that is not this data"
        assert 40 <= n_null <= 90, f"band {index}: n_null drifted to {n_null}"


def test_tiny_negative_eigenvalues_are_present_and_small(micro_pfns_tape):
    """Why ``psd_method='auto'`` routes to ``clip`` and Cholesky is rejected.

    ``|λmin|/λmax`` is ~1e-4 here: far too large for Cholesky, far too small to
    be anything but the evaluator's own arithmetic. ``clip`` rebuilds
    ``V·clip(Λ,0)·Vᵀ``, which preserves the eigenvectors and so preserves the
    near-null directions that carry the normalisation constraint.
    """
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    for band in section.subsections:
        eigenvalues = np.linalg.eigvalsh(band.matrix())
        ratio = abs(eigenvalues.min()) / eigenvalues.max()
        assert 1e-6 < ratio < 1e-2, f"|λmin|/λmax = {ratio:.2e}"


def test_normalisation_drift_budget_is_measured_not_zero(micro_pfns_tape):
    """``sqrt(1ᵀC1)`` — small, and deliberately not reported as exactly zero.

    ``1ᵀC1`` is a near-cancelling sum of ~15 000 terms whose rounding noise
    straddles zero, so about half the bands give a slightly negative value. The
    magnitude is what a draw's normalisation actually drifts by, and it is
    ~1e-5 — negligible physically, ~75× the input tape's own residual, and
    therefore something the sampler closes explicitly.
    """
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    drifts = [band.normalisation_drift() for band in section.subsections]
    assert all(0.0 < d < 1e-4 for d in drifts), drifts


def test_band_lookup_covers_the_incident_axis_without_overlap(micro_pfns_tape):
    """Half-open bands, with the top one closed — every energy lands once."""
    section = read_endf(str(micro_pfns_tape), mf_numbers=[35]).files[35].sections[18]
    edges = [band.e1 for band in section.subsections] + [section.subsections[-1].e2]

    for index, band in enumerate(section.subsections):
        assert section.band_for_incident(band.e1) == index
    assert section.band_for_incident(edges[-1]) == len(section.subsections) - 1
    assert section.band_for_incident(edges[0] / 10.0) is None
    assert section.band_for_incident(edges[-1] * 10.0) is None


# ---------------------------------------------------------------------------
# The same, on the real tapes
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize(
    "tape_fixture,n_bands,worst_row_sum",
    [("u235_tape", 8, 3.08e-3), ("u235_b81_tape", 5, 7.11e-7)],
)
def test_real_u235_tapes_support_the_same_reading(
    request, tape_fixture, n_bands, worst_row_sum,
):
    """Both U-235 evaluations, where the band geometry is real.

    JEFF-4.0's band 0 is the weakest zero-sum on any tape read — 3.08e-3, three
    orders of magnitude above the rest — and it is the band that sets
    ``ROW_SUM_TOLERANCE``. Named here so that a future tolerance change has to
    argue with a measurement.
    """
    path = request.getfixturevalue(tape_fixture)
    section = read_endf(str(path), mf_numbers=[35]).files[35].sections[18]

    assert section.num_bands == n_bands
    residuals = measured_row_sums(section)
    assert max(residuals) < ROW_SUM_TOLERANCE
    assert max(residuals) == pytest.approx(worst_row_sum, rel=0.1)

    for band in section.subsections:
        assert de_weighted_row_sum(band) > 1e1
        eigenvalues = np.linalg.eigvalsh(band.matrix())
        assert int((eigenvalues <= 1e-10 * eigenvalues.max()).sum()) >= 1
