"""``psd_method="clip_rescale"`` — the congruence rescale, and its one cost.

Written for the loose end ``docs/pfns/pfns_mf5_mf35_roadmap.md`` records as L1: the
``clip`` projection preserves eigenvectors but not the diagonal, so groups the
file gives almost no variance come out of it with some, and
``generate_pfns_samples`` clamps at 5 stated σ to bound the damage. The
candidate was a congruence rescale.

**It does the arithmetic it promised and fails the purpose anyway**, and both
halves are pinned here. The diagonal comes back exactly and the matrix stays
PSD; the sum rule ``C·1 ≈ 0`` degrades by some 500x, because the all-ones
vector is a near-null eigenvector and a congruence maps it to ``D·1``. So the
clamp stays and this method is for covariances with no sum rule — which is why
the test that would matter most is the one asserting it is *not* the default.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.decomposition import clip_and_rescale, eigen_decomposition
from kika.sampling.core import draw_samples


def _indefinite(n=40, seed=0):
    """A PSD matrix with a non-uniform diagonal, nudged slightly indefinite.

    Built the way the real ones arrive: a low-rank covariance whose smallest
    eigenvalues have been perturbed below zero, as INTG rounding and group
    collapsing both do.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, 6))
    scale = np.exp(rng.normal(scale=2.0, size=n))     # decades of spread
    matrix = (factor * scale[:, None]) @ (factor * scale[:, None]).T
    values, vectors = np.linalg.eigh(matrix)
    values[:5] = -np.abs(values[-1]) * 1e-9
    matrix = (vectors * values[None, :]) @ vectors.T
    return (matrix + matrix.T) / 2


def test_the_diagonal_comes_back_exactly():
    matrix = _indefinite()
    rescaled, info = clip_and_rescale(matrix)
    np.testing.assert_allclose(np.diag(rescaled), np.diag(matrix), rtol=1e-12)
    assert info["max_diagonal_error_after"] < info["max_diagonal_error_before"]


def test_the_result_is_psd():
    """A congruence of a PSD matrix is PSD; this pins that the code is one."""
    rescaled, _ = clip_and_rescale(_indefinite())
    smallest = float(np.linalg.eigvalsh(rescaled).min())
    largest = float(np.linalg.eigvalsh(rescaled).max())
    assert smallest > -1e-12 * largest


def test_a_component_with_no_stated_variance_gets_an_empty_row():
    """Stronger than what ``clip`` gives it, and the point of the exercise.

    ``clip`` spreads each clipped eigenvalue over the components in proportion
    to its eigenvector, so a zero-variance component acquires some. The rescale
    multiplies that row by zero.
    """
    matrix = _indefinite()
    matrix[7, :] = 0.0
    matrix[:, 7] = 0.0

    rescaled, _ = clip_and_rescale(matrix)
    np.testing.assert_allclose(rescaled[7, :], 0.0, atol=0)
    np.testing.assert_allclose(rescaled[:, 7], 0.0, atol=0)


def test_the_symmetric_result_is_exactly_symmetric():
    rescaled, _ = clip_and_rescale(_indefinite())
    np.testing.assert_array_equal(rescaled, rescaled.T)


def test_it_is_not_reachable_from_auto():
    """The frozen pipelines run on ``clip`` and must not move under them.

    ``auto`` resolves to ``clip`` or ``higham`` and to nothing else; adding an
    option must not have changed an existing numerical answer.
    """
    matrix = _indefinite()
    fromAuto, _ = eigen_decomposition(matrix=matrix, psd_method="auto",
                                      space="linear", verbose=False)
    fromClip, _ = eigen_decomposition(matrix=matrix, psd_method="clip",
                                      space="linear", verbose=False)
    np.testing.assert_array_equal(np.sort(fromAuto), np.sort(fromClip))


def test_an_unknown_method_is_still_refused():
    with pytest.raises(ValueError, match="psd_method must be one of"):
        eigen_decomposition(matrix=_indefinite(), psd_method="rescale",
                            space="linear", verbose=False)


def test_draw_samples_accepts_it_end_to_end():
    matrix = _indefinite()
    key = ("block", 0)
    drawn, info = draw_samples([(key, matrix)], n_samples=32, seed=7,
                               psd_method="clip_rescale", returns="deltas",
                               verbose=False)
    assert drawn[key].shape == (32, matrix.shape[0])
    assert np.isfinite(drawn[key]).all()
    assert info[key]["psd_method"] == "clip_rescale"


# ---------------------------------------------------------------------------
# The cost, measured on the real thing
# ---------------------------------------------------------------------------

def test_the_sum_rule_does_not_survive_the_rescale(micro_pfns_tape):
    """The finding that keeps this off the PFNS path, pinned so it stays known.

    If a future change makes this pass by preserving the sum rule, that is a
    real improvement and the docstring, the roadmap's L1 entry and the 5σ clamp
    all need revisiting — which is exactly what a failure here should prompt.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf

    endf = read_endf(str(micro_pfns_tape))
    suite, _ = decodeCovarianceSuite(endf)

    def residual(matrix):
        scale = float(np.max(np.abs(matrix)))
        return float(np.max(np.abs(matrix.sum(axis=1))) / scale)

    worsened = 0
    for section in suite.covarianceSections:
        original = np.asarray(section.form.matrix, dtype=float)
        rescaled, info = clip_and_rescale(original)

        # The diagonal restoration works on real data too, not only synthetic.
        assert info["max_diagonal_error_after"] < 1e-15

        if residual(rescaled) > 100 * residual(original):
            worsened += 1

    assert worsened == len(suite.covarianceSections), (
        "the congruence rescale used to break C·1≈0 on every band; if it no "
        "longer does, the reason the 5σ clamp stays has changed"
    )
