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

from types import SimpleNamespace

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite
from kika.endf.read_endf import read_endf
from kika.sampling.endf_perturbation import _filter_mf34_covariance
from kika.sampling.model_blocks import (
    _mf34_entries,
    assemble_joint,
    covariance_suite_blocks,
    legendre_covariance_blocks,
    legendre_covariance_index,
)

MICRO_COV = "kika/endf/tests/data/micro_fe56_cov.endf"


def _link(mfmt: str, order: int) -> SimpleNamespace:
    """The two fields `_mf34_entries` reads off a covariance link."""
    return SimpleNamespace(
        ENDF_MFMT=mfmt,
        href=f"#{mfmt}/L{order}",
        slices=SimpleNamespace(slices=(SimpleNamespace(domainValue=order),)),
    )


def _section(rowData, columnData, grid=(0.0, 1.0, 2.0)) -> SimpleNamespace:
    grid = np.asarray(grid, dtype=float)
    return SimpleNamespace(
        label=f"{rowData.href}x{columnData.href}",
        rowData=rowData,
        columnData=columnData,
        provenance=SimpleNamespace(za=26056),
        form=SimpleNamespace(
            matrix=np.eye(grid.size - 1),
            rowGrid=grid,
            columnGrid=grid,
            isRelative=True,
        ),
    )


def _suite(sections) -> SimpleNamespace:
    return SimpleNamespace(covarianceSections=list(sections))


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


def test_the_order_filter_agrees_with_the_carrier_it_replaces(micro):
    """The migration gate again, this time on a *filtered* selection.

    `perturb_ENDF_files` never assembles the whole file: it filters by MT and by
    Legendre order first, and `_filter_mf34_covariance` keeps a section only when
    **both** its sides pass. Byte-identity of the unfiltered joint says nothing
    about whether the filtered ones agree, and the filtered one is what the
    pipeline actually samples.
    """
    endf, suite = micro
    carrier = _filter_mf34_covariance(endf.mf[34].to_ang_covmat(), [2], [1])
    (_key, joint), = legendre_covariance_blocks(suite, mt=[2], orders=[1])

    np.testing.assert_array_equal(joint, carrier.covariance_matrix)


def test_an_excluded_order_takes_its_cross_blocks_with_it(micro):
    """A cross block half-placed is worse than one dropped.

    Keeping L=1 and dropping L=2 must drop the L1×L2 block too. Placing it would
    state a correlation against a component that is not in the matrix, and its
    rows would have nowhere to go.
    """
    _endf, suite = micro
    (key, joint), = legendre_covariance_blocks(suite, orders=[1])
    index = legendre_covariance_index(suite, orders=[1])[key]

    assert index["triplets"] == [(26056, 2, 1)]
    assert joint.shape == (3, 3)
    assert key[2] == ((26056, 2, 1),)


def test_all_orders_is_spelled_several_ways_and_they_agree(micro):
    """`None`, `[]` and `[-1]` all mean everything.

    `[-1]` is `resolve_signed_request`'s own spelling of "all of them" and it
    reaches here through `legendre_coeffs`. Read literally it would select no
    order at all and return an empty joint -- a pipeline drawing nothing, without
    failing.
    """
    _endf, suite = micro
    (_k, everything), = legendre_covariance_blocks(suite)
    for spelling in ([], [-1], None):
        (_key, joint), = legendre_covariance_blocks(suite, orders=spelling)
        np.testing.assert_array_equal(joint, everything)


def test_an_absolute_section_is_dropped_only_when_asked_for_relative():
    """`load_mf34_covariance` skipped absolute sections; the sampling path must too.

    A drawn MF34 sample is a multiplicative factor -- the appliers multiply a_L
    by it -- and an absolute covariance does not describe one. Dropping them is
    not the default here, though: assembling a file to look at it should show
    what the file states, and the byte-identity gate against `to_ang_covmat`
    (which drops nothing) has to keep comparing like with like.
    """
    relative = _section(_link("34/2", order=1), _link("34/2", order=1))
    absolute = _section(_link("34/2", order=2), _link("34/2", order=2))
    absolute.form.isRelative = False
    suite = _suite([relative, absolute])

    assert len(_mf34_entries(suite)) == 2
    assert len(_mf34_entries(suite, relative=True)) == 1
    assert len(_mf34_entries(suite, relative=False)) == 1

    (_key, joint), = legendre_covariance_blocks(suite, relative=True)
    assert joint.shape == (2, 2)


def test_a_section_whose_column_names_an_excluded_mt_is_dropped():
    """The MT test applies to both sides, as the order test does.

    Before this, only the row's MT was checked, so a cross-*reaction* section
    survived a filter that excluded the reaction on its column.
    """
    row = _link("34/2", order=1)
    kept = _section(row, row)
    crossed = _section(row, _link("34/102", order=1))
    suite = _suite([kept, crossed])

    assert len(_mf34_entries(suite)) == 2
    assert len(_mf34_entries(suite, mt=[2])) == 1


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


def test_a_union_bin_below_a_section_gets_zero_not_its_first_bin():
    """D10, the live half, and the one that did not raise.

    A cursor starting at 0 and only moving forward gave a union bin *below* a
    section's first boundary that section's **first bin**, replicating a
    covariance stated from 846.8 keV down to 1e-5 eV on the shipped Fe-56
    multigroup tape -- 10 590 entries, worst 2.07e-2 against diagonals of order
    1e-1.

    **Both implementations are asserted**, and they must now agree: the carrier
    was fixed rather than routed around, so the duplication here is a drift
    detector. If this starts failing, one of the two lifts moved without the
    other, and `refactor-backlog.md` names them as the pair to collapse.
    """
    from kika.cov.legendre_covariance import LegendreCovariance
    from kika.sampling.model_blocks import _lift_matrix

    src = np.array([100.0, 200.0, 300.0])              # stated only from 100
    dst = np.array([0.0, 50.0, 100.0, 200.0, 300.0])   # union runs from 0

    expected = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    carrier = LegendreCovariance()._lift_matrix(src, dst)
    ours = _lift_matrix(src, dst)

    np.testing.assert_array_equal(carrier, expected)
    np.testing.assert_array_equal(ours, expected)


def test_the_carrier_no_longer_walks_off_the_top_either():
    """D9 on the carrier: this raised IndexError before 2026-08-11."""
    from kika.cov.legendre_covariance import LegendreCovariance

    src = np.array([0.0, 10.0, 20.0])                  # stated only to 20
    dst = np.array([0.0, 10.0, 20.0, 150.0])           # union runs to 150

    lift = LegendreCovariance()._lift_matrix(src, dst)
    np.testing.assert_array_equal(
        lift, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    )


def test_a_repeated_block_is_refused_rather_than_silently_overwritten():
    """Summing and overwriting are both defensible and they disagree."""
    grid = np.array([0.0, 1.0, 2.0])
    entries = [
        (("a",), ("a",), np.eye(2), grid, grid),
        (("a",), ("a",), np.eye(2) * 3.0, grid, grid),
    ]
    with pytest.raises(ValueError, match="no defined way to combine"):
        assemble_joint(entries)


def test_an_empty_suite_returns_no_blocks_rather_than_an_empty_matrix(micro):
    """A suite with no MF34 must not produce a 0x0 block for a sampler to draw."""
    _endf, suite = micro
    assert legendre_covariance_blocks(suite, mt=102) == []
    assert legendre_covariance_index(suite, mt=102) == {}
