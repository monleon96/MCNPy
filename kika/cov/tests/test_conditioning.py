"""``kika.cov.conditioning`` — the pre-flight, and what it says about real files.

Two kinds of test here and they are worth telling apart. Most pin the checks
against matrices built to trigger exactly one of them. The last few pin
*measurements on committed evaluations* — that ``clip`` moves the stated
variances of a Cf-252 PFNS band by a few percent while ``clip_rescale`` leaves
them exact and destroys the sum rule instead. Those are the numbers a human
reads to choose a repair, so if one of them moves, the choice may have to move
with it and a green suite would be the wrong answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.conditioning import (
    BLOCKS,
    DISTORTS,
    NOTE,
    as_blocks,
    inspect_blocks,
    inspect_matrix,
    predict_psd_repairs,
)


def _checks(report):
    return {f.check for f in report.findings}


def _finding(report, check):
    matches = [f for f in report.findings if f.check == check]
    assert matches, f"expected a {check!r} finding, got {sorted(_checks(report))}"
    return matches[0]


def _indefinite(n=40, seed=0, negativity=1e-6, spread=1.0, check=True):
    """A low-rank PSD matrix with a wide diagonal, nudged below zero.

    The way the real ones arrive: INTG rounding and group collapsing both leave
    a covariance whose smallest eigenvalues sit slightly negative.

    ``spread`` is not arbitrary. At the 7-decade diagonal this started with,
    subtracting ``negativity × λ_max`` from the spectrum drove the *smallest
    diagonal entry itself* negative — an unphysical matrix, which the checker
    correctly reported as blocking and which is not what "slightly non-PSD"
    means. ``check`` pins that, so the fixture cannot drift back into testing
    the wrong thing.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, 6))
    scale = np.exp(rng.normal(scale=spread, size=n))
    matrix = (factor * scale[:, None]) @ (factor * scale[:, None]).T
    values, vectors = np.linalg.eigh(matrix)
    values[:5] = -abs(values[-1]) * negativity
    matrix = (vectors * values[None, :]) @ vectors.T
    matrix = (matrix + matrix.T) / 2
    if check:
        assert np.all(np.diag(matrix) > 0), "fixture is unphysical, not merely non-PSD"
    return matrix


# ---------------------------------------------------------------------------
# The blocking checks — a draw that would fail or mean nothing
# ---------------------------------------------------------------------------

def test_a_well_conditioned_matrix_reports_nothing():
    report = inspect_matrix(np.eye(8))
    assert report.findings == ()
    assert report.samplable and report.faithful


def test_non_finite_entries_block():
    matrix = np.eye(4)
    matrix[1, 2] = np.nan
    finding = _finding(inspect_matrix(matrix), "finiteness")
    assert finding.severity == BLOCKS
    assert finding.evidence["n_non_finite"] == 1


def test_a_negative_stated_variance_blocks_and_offers_nothing():
    """The one finding with no remedy, and the reason is not an omission.

    σ² < 0 is not a conditioning problem — no projection makes the evaluation
    have said something else. Offering a repair here would invite someone to
    apply one.
    """
    matrix = np.eye(4)
    matrix[2, 2] = -0.5
    finding = _finding(inspect_matrix(matrix), "negative_variance")
    assert finding.severity == BLOCKS
    assert finding.remedies == ()


def test_a_correlation_above_one_blocks():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    finding = _finding(inspect_matrix(matrix), "correlation_bound")
    assert finding.severity == BLOCKS
    assert finding.evidence["max_abs_correlation"] == pytest.approx(2.0)


def test_roundoff_asymmetry_is_not_reported_and_real_asymmetry_blocks():
    """One threshold, both sides of it, because the difference is the point.

    Every reconstructed covariance is asymmetric at float64 round-off and
    symmetrising it is free. A matrix asymmetric at 1e-1 has had half of it
    thrown away by whoever symmetrises next, and which half was right is not
    a question this module can answer.
    """
    matrix = np.eye(6) + 1e-16
    matrix[0, 1] += 1e-17
    assert "symmetry" not in _checks(inspect_matrix(matrix))

    matrix = np.eye(6)
    matrix[0, 1] = 0.5
    finding = _finding(inspect_matrix(matrix), "symmetry")
    assert finding.severity == BLOCKS


def test_a_wholly_inert_block_blocks_but_a_partly_inert_one_does_not():
    inert = np.zeros((5, 5))
    assert _finding(inspect_matrix(inert), "inert_rows").severity == BLOCKS

    partial = np.eye(5)
    partial[3, 3] = 0.0
    finding = _finding(inspect_matrix(partial), "inert_rows")
    assert finding.severity == NOTE
    assert finding.evidence["indices"] == [3]


# ---------------------------------------------------------------------------
# The distorting checks — a draw that succeeds, of something else
# ---------------------------------------------------------------------------

def test_non_psd_is_distorting_not_blocking_and_names_what_auto_would_do():
    finding = _finding(inspect_matrix(_indefinite()), "definiteness")
    assert finding.severity == DISTORTS
    assert finding.evidence["auto_would_choose"] == "clip"
    # A ratio past the auto threshold flips the answer, which is the fact a
    # caller needs in order to know a silent repair is about to be a big one.
    coarse = _finding(
        inspect_matrix(_indefinite(negativity=0.5, check=False)), "definiteness"
    )
    assert coarse.evidence["auto_would_choose"] == "higham"


def test_outliers_are_judged_within_their_family():
    """Without families the question is the wrong one, and the test shows it.

    Two reactions whose variances differ by four decades: nothing is an outlier
    for its own reaction, but the smaller reaction's whole block looks like one
    against a pooled median — and vice versa.
    """
    diagonal = np.concatenate([np.full(10, 1.0), np.full(10, 1e-6)])
    matrix = np.diag(diagonal)
    families = ["a"] * 10 + ["b"] * 10

    assert "variance_outliers" not in _checks(
        inspect_matrix(matrix, families=families)
    )

    matrix[0, 0] = 1e5          # 1e5x its own family median
    finding = _finding(inspect_matrix(matrix, families=families), "variance_outliers")
    assert finding.severity == DISTORTS
    assert [d["index"] for d in finding.evidence["detail"]] == [0]
    assert finding.evidence["detail"][0]["family"] == "a"


def test_rank_deficiency_is_a_note_not_a_problem():
    """Low rank is the normal state of an evaluated covariance, not a defect."""
    finding = _finding(inspect_matrix(_indefinite()), "rank")
    assert finding.severity == NOTE
    assert finding.evidence["rank"] < finding.evidence["order"]


# ---------------------------------------------------------------------------
# Remedy prediction
# ---------------------------------------------------------------------------

def test_prediction_measures_rather_than_describes():
    matrix = _indefinite()
    remedies = {r.name: r for r in predict_psd_repairs(matrix)}
    assert set(remedies) == {"clip", "clip_rescale", "higham"}

    # The defining difference between the two, measured rather than asserted
    # from the algorithm: the congruence puts the stated diagonal back exactly
    # and the projection does not.
    # Stated as a comparison rather than an absolute bar: how far clip moves
    # the diagonal is a property of how non-PSD this particular fixture is,
    # while "clip_rescale restores it and clip does not" is the claim.
    assert remedies["clip_rescale"].effect["stated_diagonal_max_relative_change"] < 1e-12
    assert (
        remedies["clip"].effect["stated_diagonal_max_relative_change"]
        > 1e6 * remedies["clip_rescale"].effect["stated_diagonal_max_relative_change"]
    )

    for remedy in remedies.values():
        assert remedy.effect["min_eigenvalue_ratio"] > -1e-12   # all leave it PSD
        assert remedy.effect["predicted"] is True


def test_prediction_leaves_the_input_untouched():
    """The module's central promise: it measures repairs, it does not apply them."""
    matrix = _indefinite()
    before = matrix.copy()
    predict_psd_repairs(matrix, candidates=("clip", "clip_rescale", "higham"))
    np.testing.assert_array_equal(matrix, before)


def test_higham_is_measured_up_to_its_own_order_ceiling():
    """It scales as iterations x n³ where the others are one n³, so it has one.

    Measured, quiet: ~0.03 s at 50 iterations and ~0.57 s at 200 on a 122x122,
    against ~1.5 ms for clip. A single ceiling for all three would either drop
    Higham far too early or let it run for minutes on a block clip handles in
    a second.
    """
    finding = _finding(inspect_matrix(_indefinite(n=40)), "definiteness")
    higham = next(r for r in finding.remedies if r.name == "higham")
    assert higham.effect["predicted"] is True
    assert higham.effect["converged"] in (True, False)

    finding = _finding(
        inspect_matrix(_indefinite(n=40), predict_max_order=1000), "definiteness"
    )
    by_name = {r.name: r for r in finding.remedies}
    assert by_name["clip"].effect["predicted"] is True

    # Past the Higham ceiling only Higham drops out, and it says which ceiling.
    import kika.cov.conditioning as conditioning
    ceiling = conditioning.PREDICT_HIGHAM_MAX_ORDER
    try:
        conditioning.PREDICT_HIGHAM_MAX_ORDER = 10
        finding = _finding(inspect_matrix(_indefinite(n=40)), "definiteness")
        by_name = {r.name: r for r in finding.remedies}
        assert by_name["clip"].effect["predicted"] is True
        assert by_name["higham"].effect["predicted"] is False
        assert "PREDICT_HIGHAM_MAX_ORDER" in by_name["higham"].effect["reason"]
    finally:
        conditioning.PREDICT_HIGHAM_MAX_ORDER = ceiling


def test_a_block_too_large_to_predict_says_so_instead_of_predicting():
    finding = _finding(
        inspect_matrix(_indefinite(n=30), predict_max_order=10), "definiteness"
    )
    assert all(r.effect["predicted"] is False for r in finding.remedies)
    assert "PREDICT_MAX_ORDER" in finding.remedies[0].effect["reason"]


def test_an_unknown_repair_is_refused():
    with pytest.raises(ValueError, match="unknown PSD repair"):
        predict_psd_repairs(_indefinite(), candidates=("jitter",))


# ---------------------------------------------------------------------------
# Cross blocks
# ---------------------------------------------------------------------------

def test_a_rectangular_block_must_declare_itself_a_cross_block():
    with pytest.raises(ValueError, match="square_form=False"):
        inspect_matrix(np.ones((3, 4)))


def test_a_cross_block_runs_only_the_checks_that_mean_anything():
    """Found by running this module over Fe-56: ``MF34-MT2-L1-MT2-L2``.

    An L=1-against-L=2 partition is asymmetric and indefinite *by construction*
    — it is an off-diagonal corner of a larger covariance, so its diagonal is
    not a variance and its spectrum is not a spectrum of anything. Reporting it
    as unsamplable would be the checker's bug, not the file's.
    """
    matrix = np.array([[1.0, 0.5, 0.0], [0.3, 1.0, 0.2], [0.0, 0.1, 1.0]])
    report = inspect_matrix(matrix, square_form=False)
    assert _checks(report) == {"cross_block"}
    assert report.samplable

    finite = np.array([[1.0, np.nan], [0.0, 1.0]])
    assert "finiteness" in _checks(inspect_matrix(finite, square_form=False))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_samplable_and_faithful_are_different_questions():
    report = inspect_blocks({"clean": np.eye(4), "bent": _indefinite()})
    assert report.samplable          # both can be drawn from
    assert not report.faithful       # one of them not as stated
    assert [b.key for b in report.distorting()] == ["bent"]
    assert report.failing() == ()


def test_the_three_block_shapes_agree():
    matrices = [np.eye(3), np.eye(4)]
    from_bare = inspect_blocks(matrices)
    from_pairs = inspect_blocks([(0, matrices[0]), (1, matrices[1])])
    from_mapping = inspect_blocks({0: matrices[0], 1: matrices[1]})
    assert [b.key for b in from_bare.blocks] == [0, 1]
    assert [b.order for b in from_pairs.blocks] == [3, 4]
    assert [b.order for b in from_mapping.blocks] == [3, 4]


def test_per_block_arguments_may_be_a_mapping():
    blocks = {"square": np.eye(3), "cross": np.ones((3, 4))}
    report = inspect_blocks(blocks, square_form={"cross": False})
    cross = next(b for b in report.blocks if b.key == "cross")
    assert _checks(cross) == {"cross_block"}


def test_the_markdown_table_survives_a_pipe_in_a_summary():
    """``|ρ|`` in a cell splits the row unless escaped, and renders wrong quietly."""
    table = inspect_blocks({"bad": np.array([[1.0, 2.0], [2.0, 1.0]])}).to_markdown()
    row = next(line for line in table.split("\n") if "correlation_bound" in line)
    assert r"\|ρ\|" in row
    assert row.count("|") - row.count(r"\|") == 6   # five columns, six delimiters


def test_the_block_normaliser_is_shared_with_the_sampler():
    """One definition of "a set of blocks", so a pre-flight inspects what runs."""
    from kika.sampling.core import _as_blocks
    assert _as_blocks is as_blocks


# ---------------------------------------------------------------------------
# Measured on committed evaluations
# ---------------------------------------------------------------------------

def _pfns_blocks(micro_pfns_tape):
    from kika.endf.model_adapter import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf
    from kika.sampling.model_blocks import covariance_suite_blocks

    suite, _ = decodeCovarianceSuite(read_endf(str(micro_pfns_tape)))
    return covariance_suite_blocks(suite)


def test_the_pfns_bands_are_recognised_as_carrying_a_sum_rule(micro_pfns_tape):
    """And at 1e-6, not at machine epsilon — which is why the tolerance is 1e-3.

    MF35 covaries group-integrated probabilities that sum to one, so ``C·1``
    vanishes; it vanishes only to the six significant figures ENDF writes, and
    a tolerance set from first principles rather than from a file would have
    missed every real case.
    """
    report = inspect_blocks(_pfns_blocks(micro_pfns_tape))
    assert len(report.blocks) == 4
    for block in report.blocks:
        finding = _finding(block, "sum_rule")
        assert 1e-9 < finding.evidence["residual"] < 1e-4


def test_the_pfns_repair_trade_off_reproduces_the_hand_measurement(micro_pfns_tape):
    """The L1 investigation, as an assertion instead of a session.

    Established by hand on these four bands: ``clip_rescale`` restores the
    stated diagonal exactly and destroys ``C·1 ≈ 0`` by some hundreds of times,
    which is why the 5σ clamp in ``generate_pfns_samples`` stays and
    ``psd_method`` on the PFNS path stays ``clip``. If this assertion fails
    because the degradation went away, that is a real improvement and the clamp
    needs revisiting — a green suite would hide it.
    """
    report = inspect_blocks(_pfns_blocks(micro_pfns_tape))
    for block in report.blocks:
        remedies = {r.name: r for r in _finding(block, "definiteness").remedies}
        clip, rescale = remedies["clip"], remedies["clip_rescale"]

        assert clip.effect["stated_diagonal_max_relative_change"] > 1e-2
        assert rescale.effect["stated_diagonal_max_relative_change"] < 1e-12

        before = clip.effect["sum_rule_residual_before"]
        assert clip.effect["sum_rule_residual_after"] <= 2 * before
        assert rescale.effect["sum_rule_residual_after"] > 100 * before


def test_every_mf32_sub_format_decodes_to_something_drawable(micro_mf32_tape):
    """All four committed sub-formats, and the ratchet is on *samplable* only.

    Th-232's resolved range is 2781 parameters, not PSD and 540 directions
    short of full rank — none of which blocks a draw, all of which mean the
    draw is of a projected matrix. Keeping the two questions apart is what the
    report is for, so the assertion is the weaker one deliberately: a decoder
    change that starts emitting NaNs, negative variances or |ρ| > 1 fails here,
    and ordinary ill-conditioning does not.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf
    from kika.sampling.model_blocks import parameter_covariance_blocks

    suite, _ = decodeCovarianceSuite(read_endf(str(micro_mf32_tape)))
    blocks = parameter_covariance_blocks(suite)
    assert blocks, f"{micro_mf32_tape.name} decoded no parameter covariance"

    report = inspect_blocks(blocks)
    assert report.samplable, report.summary()

    # Too large to predict at the default ceiling: say so rather than spend
    # three eigendecompositions of a 2781x2781 inside a pre-flight check.
    for block in report.blocks:
        if block.order <= 1500:
            continue
        for remedy in _finding(block, "definiteness").remedies:
            assert remedy.effect["predicted"] is False
