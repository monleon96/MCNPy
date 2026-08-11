"""MF34 sections → the one covariance they are partitions of.

The gate on the MF34 sampling migration. The property being held is not "the new
code runs" but **"the new code assembles what the old carrier assembled"**, on
every file where the old carrier assembles anything at all — because the samples
already drawn through `generate_endf_samples` were drawn in that layout, and a
migration that quietly changes it changes results without failing.

The two halves are deliberate:

* where `LegendreCovariance.covariance_matrix` produces a matrix, we must equal
  it under `assert_array_equal` -- not `allclose`;
* where it raises, we must not. That is not a nicety: it raises on the tape the
  Fe-56 track actually ships (`docs/library-gaps.md` D9).
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite
from kika.endf.read_endf import read_endf
from kika.sampling.model_blocks import (
    assemble_joint,
    covariance_suite_blocks,
    legendre_covariance_blocks,
    legendre_covariance_index,
)

MICRO_COV = "kika/endf/tests/data/micro_fe56_cov.endf"


@pytest.fixture(scope="module")
def micro():
    endf = read_endf(MICRO_COV)
    suite, _ = decodeCovarianceSuite(endf)
    return endf, suite


def test_the_assembly_reproduces_the_carrier_bit_for_bit(micro):
    """The migration gate. `allclose` would not do -- the samples must not move."""
    endf, suite = micro
    carrier = endf.mf[34].to_ang_covmat().covariance_matrix
    (_key, joint), = legendre_covariance_blocks(suite)

    assert joint.shape == carrier.shape
    np.testing.assert_array_equal(joint, carrier)


def test_one_joint_block_not_one_per_section(micro):
    """Four sections, three of them MF34, one matrix.

    The shape of the answer is the point. `covariance_suite_blocks` returns four
    separate blocks for this file, of which one is an MF33 section and one is an
    L=1-against-L=2 corner that is not a covariance at all.
    """
    _endf, suite = micro
    blocks = legendre_covariance_blocks(suite)
    assert len(blocks) == 1

    naive = covariance_suite_blocks(suite)
    assert len(naive) == 4
    assert sum(m.shape[0] for _, m in naive) == 12
    assert blocks[0][1].shape == (6, 6)


def test_an_mf33_section_in_the_same_suite_is_not_swept_in(micro):
    """`covariance_suite_blocks` emits MF33-MT2 among the MF34 blocks; we do not."""
    _endf, suite = micro
    (key, _joint), = legendre_covariance_blocks(suite)
    _isotope, label, triplets = key
    assert label == "MF34"
    assert triplets == ((26056, 2, 1), (26056, 2, 2))


def test_the_cross_order_block_is_placed_and_is_not_negligible(micro):
    """D8: the cross corner is 19 % of the largest entry, not a rounding term."""
    _endf, suite = micro
    (key, joint), = legendre_covariance_blocks(suite)
    stride = legendre_covariance_index(suite)[key]["stride"]

    off = joint[:stride, stride:]
    assert np.abs(off).max() > 0.0
    np.testing.assert_array_equal(off, joint[stride:, :stride].T)
    assert np.abs(off).max() / np.abs(joint).max() > 0.15


def test_the_index_says_what_the_rows_are(micro):
    _endf, suite = micro
    (key, joint), = legendre_covariance_blocks(suite)
    index = legendre_covariance_index(suite)[key]

    assert index["triplets"] == [(26056, 2, 1), (26056, 2, 2)]
    assert index["dimension"] == joint.shape[0]
    assert index["stride"] * len(index["triplets"]) == joint.shape[0]
    for triplet, grid in index["grids"].items():
        assert grid.size - 1 == index["widths"][triplet]


def test_a_union_bin_outside_a_section_gets_zero_not_the_nearest_bin():
    """D9, in miniature, and the reason the carrier raises on the shipped tape.

    Two components: one stated out to 150, one only out to 20, correlated. The
    union runs to 150, so the short section is short of it. The carrier walks a
    cursor off the end of its own grid; here the uncovered rows must come back
    identically zero -- the section says nothing above 20, and nothing is zero,
    not the value of its top bin smeared across the rest of the axis.
    """
    long_grid = np.array([0.0, 10.0, 20.0, 150.0])
    short_grid = np.array([0.0, 10.0, 20.0])
    entries = [
        (("a",), ("a",), np.eye(3) * 4.0, long_grid, long_grid),
        (("b",), ("b",), np.eye(2) * 9.0, short_grid, short_grid),
        (("a",), ("b",), np.ones((3, 2)), long_grid, short_grid),
    ]
    keys, joint, stride = assemble_joint(entries)

    assert keys == [("a",), ("b",)]
    assert stride == 3
    assert joint.shape == (6, 6)

    # "b" covers only the first two of the three union bins.
    b = joint[stride:, stride:]
    np.testing.assert_array_equal(np.diag(b), [9.0, 9.0, 0.0])
    assert np.all(joint[stride + 2, :] == 0.0)
    assert np.all(joint[:, stride + 2] == 0.0)


def test_the_cross_block_is_transposed_in_rather_than_stated_twice():
    """A file states one corner; the assembly owes the other."""
    grid = np.array([0.0, 1.0, 2.0])
    entries = [
        (("a",), ("a",), np.eye(2), grid, grid),
        (("b",), ("b",), np.eye(2), grid, grid),
        (("a",), ("b",), np.array([[0.5, 0.25], [0.125, 0.0625]]), grid, grid),
    ]
    _keys, joint, stride = assemble_joint(entries)
    np.testing.assert_array_equal(joint, joint.T)
    np.testing.assert_array_equal(
        joint[:stride, stride:], np.array([[0.5, 0.25], [0.125, 0.0625]])
    )


def test_an_empty_suite_returns_no_blocks_rather_than_an_empty_matrix(micro):
    """A suite with no MF34 must not produce a 0x0 block for a sampler to draw."""
    _endf, suite = micro
    assert legendre_covariance_blocks(suite, mt=102) == []
    assert legendre_covariance_index(suite, mt=102) == {}
