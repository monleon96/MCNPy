"""The MF34 draw, before and after the migration, on a file with real structure.

`test_legendre_blocks.py` gates the *assembly* — that the joint matrix the model
builds is the one `LegendreCovariance` built. This gates the *draw*: that feeding
it to `draw_samples` produces the same perturbation factors `generate_endf_samples`
produced, so nothing moved because the covariance arrived in a different object.

`micro_fe56_structural.endf` is the right file for it and the cov micro-tape is
not. It carries six Legendre orders over a 252x252 joint of which **87 directions
are null**, so it exercises the one place the two draws can disagree; the cov
tape's 6x6 is full rank and would agree for the wrong reason.

**This file has a scheduled death.** It compares against `generate_endf_samples`,
which the migration deletes, and it holds `null_tol=None`, which the truncation
commit removes. Both of those are the point: the equivalence is a migration
gate, not a property MF34 sampling is supposed to have forever. When truncation
is turned on this test *should* fail, and it goes then — replaced by the measured
before/after recorded in `sampling_migration_roadmap.md`. Do not "fix" it by
loosening `assert_array_equal` to `allclose`; that would hide exactly the drift
it exists to catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.sampling.core import draw_samples
from kika.sampling.endf_perturbation import (
    _filter_mf34_covariance,
    _create_parameter_mapping,
    _mf34_available,
    _parameter_mapping_from_index,
    load_mf34_covariance,
    load_mf34_suite,
)
from kika.sampling.generators import generate_endf_samples
from kika.sampling.model_blocks import (
    legendre_covariance_blocks,
    legendre_covariance_index,
)
from kika.sampling.utils import resolve_signed_request

STRUCTURAL = "kika/endf/tests/data/micro_fe56_structural.endf"

#: What `exfor_to_endf_research.py` ships: multigroup, linear, svd, random, and
#: `psd_method` left at `perturb_ENDF_files`' own default of "none".
MT_REQUEST = [2]
L_REQUEST = list(range(1, 7))
SEED = 12345


@pytest.fixture(scope="module")
def both():
    carrier = load_mf34_covariance(STRUCTURAL)
    suite = load_mf34_suite(STRUCTURAL)
    mt_resolved, _ = resolve_signed_request(MT_REQUEST, sorted(carrier.reactions))
    l_resolved, _ = resolve_signed_request(L_REQUEST, sorted(carrier.legendre_indices))
    return carrier, suite, mt_resolved, l_resolved


def test_the_suite_and_the_carrier_offer_the_same_mts_and_orders(both):
    """What `resolve_signed_request` is handed must not change under the swap."""
    carrier, suite, _mt, _l = both
    assert _mf34_available(suite) == (
        sorted(carrier.reactions), sorted(carrier.legendre_indices),
    )


def test_the_filtered_joint_is_the_carriers_filtered_joint(both):
    """Filtered, because filtered is what the pipeline actually samples."""
    carrier, suite, mt_resolved, l_resolved = both
    filtered = _filter_mf34_covariance(carrier, mt_resolved, l_resolved)
    (_key, joint), = legendre_covariance_blocks(
        suite, mt=mt_resolved, orders=l_resolved, relative=True,
    )

    assert joint.shape == (252, 252)
    np.testing.assert_array_equal(joint, filtered.covariance_matrix)


def test_the_flat_layout_is_unchanged(both):
    """A drawn sample is a flat vector; the appliers read it through this.

    If the mapping moved, every written coefficient would land on the wrong
    energy bin -- and the tapes would still be written, and still be valid ENDF.
    """
    carrier, suite, mt_resolved, l_resolved = both
    filtered = _filter_mf34_covariance(carrier, mt_resolved, l_resolved)
    old_mapping, old_grids = _create_parameter_mapping(filtered)

    index = legendre_covariance_index(
        suite, mt=mt_resolved, orders=l_resolved, relative=True)
    new_mapping, new_grids = _parameter_mapping_from_index(
        next(iter(index.values()))
    )

    assert new_mapping == old_mapping
    assert new_grids == old_grids


def test_the_draw_itself_is_bit_for_bit(both):
    """The migration gate. 87 null directions, and not one drawn column moves.

    `null_tol=None` is what makes this provable: truncation would change the QMC
    dimension from 252 to 165 and move every column, leaving no way to tell a
    plumbing mistake from the truncation.
    """
    carrier, suite, mt_resolved, l_resolved = both
    filtered = _filter_mf34_covariance(carrier, mt_resolved, l_resolved)

    old, _mts, _diag = generate_endf_samples(
        filtered, 8, space="linear", decomposition_method="svd",
        sampling_method="random", seed=SEED, psd_method="none", verbose=False,
    )

    blocks = legendre_covariance_blocks(
        suite, mt=mt_resolved, orders=l_resolved, relative=True)
    (key, _joint), = blocks
    samples, diagnostics = draw_samples(
        blocks, 8, space="linear", returns="factors",
        decomposition_method="svd", sampling_method="random", seed=SEED,
        psd_method="none", null_tol=None, dtype=np.float32, verbose=False,
    )

    assert diagnostics[key]["n"] == 252
    assert diagnostics[key]["n_null"] == 87
    np.testing.assert_array_equal(samples[key], old)


def test_truncation_would_have_moved_every_column(both):
    """Why the two changes could not share a commit.

    Not a defect in either draw -- it is what truncating the QMC dimension does.
    Recorded as a test so the reason the migration was split is checkable rather
    than remembered.
    """
    _carrier, suite, mt_resolved, l_resolved = both
    blocks = legendre_covariance_blocks(
        suite, mt=mt_resolved, orders=l_resolved, relative=True)
    (key, _joint), = blocks

    common = dict(space="linear", returns="factors", decomposition_method="svd",
                  sampling_method="random", seed=SEED, psd_method="none",
                  dtype=np.float32, verbose=False)
    kept, _ = draw_samples(blocks, 8, null_tol=None, **common)
    truncated, _ = draw_samples(blocks, 8, **common)

    moved = np.abs(kept[key] - truncated[key]).max(axis=0)
    assert np.all(moved > 0.0)
