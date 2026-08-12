"""The model assembles the same MF34 blocks the ENDF path does.

``LegendreCovariance.from_covariance_suite`` is the covariance half of phase 4's
collapse work: with it and ``legendre_source_from_model``, the multigroup
collapse can be driven from a ``reactionSuite`` on both sides.

**The gate is agreement with ``MF34MT.to_ang_covmat``, block for block, at
rtol=0.** Not a stored golden — the claim is that two readings of one file agree,
and a golden would only freeze whichever was written down first.

Both sides come off the same committed tape, and the model side arrives through
``decodeCovarianceSuite`` because that is what a covariance file goes through;
the reactionSuite door (``kika.read``) carries the reaction data, not the
covariance suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.legendre_covariance import LegendreCovariance
from kika.endf import read_endf


@pytest.fixture(scope="module")
def endf(micro_cov_tape):
    return read_endf(str(micro_cov_tape))


@pytest.fixture(scope="module")
def from_endf(endf):
    """The MF34 blocks as the ENDF section assembles them."""
    return endf.mf[34].mt[2].to_ang_covmat()


@pytest.fixture(scope="module")
def from_model(endf):
    """The same blocks, read off the decoded covarianceSuite."""
    from kika.endf.model_adapter import decodeCovarianceSuite

    decoded = decodeCovarianceSuite(endf)
    suite = decoded[0] if isinstance(decoded, tuple) else decoded
    return LegendreCovariance.from_covariance_suite(suite, nuclide=26056)


def _blocks(covariance):
    """Each block keyed by what identifies it, so order cannot flatter the test."""
    return {
        (covariance.reaction_rows[i], covariance.l_rows[i],
         covariance.reaction_cols[i], covariance.l_cols[i]): i
        for i in range(covariance.num_matrices)
    }


def test_the_same_blocks_are_present(from_endf, from_model):
    """Same (MT, l) pairs, and the MF33 section in the same suite is not swept in."""
    assert from_endf.num_matrices > 0, "fixture states no MF34 blocks"
    assert set(_blocks(from_model)) == set(_blocks(from_endf))


def test_every_matrix_is_identical(from_endf, from_model):
    """rtol=0. These are two readings of one file, not two calculations."""
    endf_index, model_index = _blocks(from_endf), _blocks(from_model)
    for key, i in endf_index.items():
        np.testing.assert_array_equal(
            from_model.matrices[model_index[key]],
            from_endf.matrices[i],
            err_msg=f"block {key} differs between the ENDF and model readings",
        )


def test_every_grid_is_identical(from_endf, from_model):
    endf_index, model_index = _blocks(from_endf), _blocks(from_model)
    for key, i in endf_index.items():
        np.testing.assert_array_equal(
            np.asarray(from_model.energy_grids[model_index[key]], dtype=float),
            np.asarray(from_endf.energy_grids[i], dtype=float),
            err_msg=f"block {key} sits on a different energy grid",
        )


def test_the_metadata_that_changes_the_answer_survives(from_endf, from_model):
    """``is_relative`` and ``frame`` are not labels.

    A relative matrix read as absolute is off by the square of the cross
    section, and a CM matrix collapsed as lab is a different physical claim. The
    collapse branches on both, so they are part of the comparison rather than a
    nicety.
    """
    endf_index, model_index = _blocks(from_endf), _blocks(from_model)
    for key, i in endf_index.items():
        j = model_index[key]
        assert from_model.is_relative[j] == from_endf.is_relative[i], key
        assert from_model.frame[j] == from_endf.frame[i], key


def test_a_non_mf34_section_is_skipped_rather_than_assembled():
    """An MF33 section in the same suite must not arrive as an angular block.

    The committed covariance tape carries one, which is why the count test above
    is a real constraint and not an accident of the fixture.
    """

    class _Link:
        ENDF_MFMT = "33/2"
        legendreOrder = 0

    class _Form:
        matrix = np.eye(3)
        rowGrid = np.array([1.0, 2.0, 3.0, 4.0])
        columnGrid = None
        isRelative = True
        productFrame = None

    class _Section:
        label = "MF33-MT2"
        rowData = _Link()
        columnData = None
        form = _Form()
        provenance = None

    assert LegendreCovariance.from_covariance_suite([_Section()]).num_matrices == 0
