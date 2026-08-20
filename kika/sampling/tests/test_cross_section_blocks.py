"""MF33 sections → the one covariance they are partitions of.

The cross-*reaction* half of what `test_legendre_blocks.py` does for Legendre
orders, and P3 of `docs/library/sampling_migration_roadmap.md`. The property held is the
same one, for the same reason: the samples the MF33 pipelines have already drawn
were drawn in `CrossSectionCovariance`'s layout, so an assembly that quietly
changes it changes results without failing.

**The one asymmetry with the MF34 gate is the fill.** The carrier allocates
``np.full((N, N), np.nan)`` and writes only the blocks the file carries, so a
pair of MTs the evaluation does not correlate comes back NaN;
``generate_samples`` replaces every non-finite entry with 0 immediately before
decomposing. The matrix actually decomposed today is therefore the zero-filled
one, and that — not the NaN one — is what equivalence is owed to. The tests
below assert both halves of that, so the claim rests on a measurement rather
than on the reading of one comment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite
from kika.endf.read_endf import read_endf
from kika.sampling.model_blocks import (
    _is_endf_mf,
    _cross_section_entries,
    cross_section_covariance_blocks,
    cross_section_covariance_index,
    legendre_covariance_blocks,
)

MICRO_MF33 = str(Path(__file__).resolve().parents[2]
                 / "endf" / "tests" / "data" / "micro_fe56_mf33.endf")


@pytest.fixture(scope="module")
def micro():
    endf = read_endf(MICRO_MF33)
    suite, _ = decodeCovarianceSuite(endf)
    return endf, suite


@pytest.fixture(scope="module")
def carrier(micro):
    """The `CrossSectionCovariance` the shipped MF33 path assembles."""
    from kika.sampling.mf33_sampling import load_mf33_covariance

    cov, _mf3, grid, present = load_mf33_covariance(
        MICRO_MF33, None, [4, 16], energy_unit="eV",
    )
    return cov, grid, present


def test_the_mf33_assembly_reproduces_the_carrier(micro, carrier):
    """The migration gate. `allclose` would not do — the samples must not move."""
    _endf, suite = micro
    cov, _grid, _present = carrier

    reference = cov.covariance_matrix
    reference = np.where(np.isfinite(reference), reference, 0.0)

    blocks = cross_section_covariance_blocks(suite)
    assert len(blocks) == 1, "MF33 assembles to one joint block, not one per section"
    _key, joint = blocks[0]

    assert joint.shape == reference.shape
    np.testing.assert_array_equal(joint, reference)


def test_the_carrier_states_half_of_it_as_nan(carrier):
    """D11, pinned where the assembly can see it.

    Not incidental colour: it is the reason the gate above zero-fills its
    reference, and if it ever stopped being true that gate would be comparing
    against something else without saying so.
    """
    cov, _grid, _present = carrier
    reference = cov.covariance_matrix
    n_nan = int(np.isnan(reference).sum())

    assert reference.shape == (248, 248)
    assert n_nan == 30752, "the unstated MT4xMT16 block is what makes this half NaN"
    assert n_nan * 2 == reference.size


def test_the_ordering_convention_is_the_carrier_s(micro, carrier):
    """`sorted((za, mt))` and `_get_param_pairs()` are the same order.

    They arrive at it independently — one sorts tuples, the other sorts by an
    explicit ``(isotope, reaction)`` key — so this is the assertion that keeps
    them from drifting apart. Every downstream slice into the flat factors
    vector depends on it.
    """
    _endf, suite = micro
    cov, _grid, _present = carrier

    index = cross_section_covariance_index(suite)
    entry = next(iter(index.values()))
    assert entry["pairs"] == [tuple(p) for p in cov._get_param_pairs()]
    assert entry["dimension"] == cov.covariance_matrix.shape[0]
    assert entry["stride"] == cov.num_groups


def test_a_component_maps_to_the_rows_the_carrier_gives_it(micro, carrier):
    """The index is what a caller writes a drawn perturbation back with.

    Asserted against `extract_mt_param_blocks`, which is the slicing the shipped
    MF33 path uses today — that function multiplies an index by ``num_groups``,
    which is only right while every component is the same width, and ``widths``
    is what will say when it is not.
    """
    from kika.sampling.mf33_sampling import extract_mt_param_blocks

    _endf, suite = micro
    cov, _grid, _present = carrier

    entry = next(iter(cross_section_covariance_index(suite).values()))
    stride = entry["stride"]
    shipped = extract_mt_param_blocks(cov)

    for position, (_za, mt) in enumerate(entry["pairs"]):
        assert shipped[mt] == slice(position * stride, (position + 1) * stride)
        assert entry["widths"][(_za, mt)] == stride


def test_mf34_sections_are_not_swept_into_an_mf33_assembly(micro):
    """The filter is by `ENDF_MFMT`, both sides, as it is for MF34.

    `covariance_suite_blocks` filters nothing and emits MF33 sections among the
    MF34 ones on the committed micro-tape (`library-gaps.md` D8). The two
    typed enumerators are what fixed that, and each has to stay blind to the
    other's format for it to hold.
    """
    _endf, suite = micro
    assert cross_section_covariance_blocks(suite), "the fixture does carry MF33"
    assert legendre_covariance_blocks(suite) == [], "the fixture carries no MF34"


def test_selecting_one_mt_leaves_a_covariance_of_that_mt_alone(micro, carrier):
    """`mt=` narrows the joint, and the surviving block is unchanged.

    A filter that renormalised, re-lifted or reordered anything would show up
    here as a difference from the corresponding diagonal sub-block of the full
    assembly.
    """
    _endf, suite = micro
    _cov, _grid, _present = carrier

    full = cross_section_covariance_blocks(suite)[0][1]
    entry = next(iter(cross_section_covariance_index(suite).values()))
    stride = entry["stride"]
    position = [mt for _za, mt in entry["pairs"]].index(16)

    only16 = cross_section_covariance_blocks(suite, mt=16)
    assert len(only16) == 1
    _key, block = only16[0]
    assert block.shape == (stride, stride)
    np.testing.assert_array_equal(
        block,
        full[position * stride:(position + 1) * stride,
             position * stride:(position + 1) * stride],
    )


def test_an_absolute_section_is_dropped_when_a_caller_asks_to_sample(micro):
    """`relative=True` is what a sampling caller wants, for MF33's own reason.

    The appliers multiply by what comes back, so an absolute covariance does not
    describe a factor. The fixture's sections are relative, so this asserts the
    filter is wired rather than that it fires — `_cross_section_entries` is where the
    rule lives and `legendre_covariance_blocks` carries the identical one.
    """
    _endf, suite = micro
    assert all(s.form.isRelative for s in suite.covarianceSections
               if _is_endf_mf(s.rowData, 33))
    assert len(_cross_section_entries(suite, relative=True)) == 2
    assert _cross_section_entries(suite, relative=False) == []


# ----------------------------------------------------------------------
# The grid, which is what deferred the source migration
# ----------------------------------------------------------------------
#
# `sampling_migration_roadmap.md` P4 left MF33's *source* on the carrier because
# the two `UNION_MODES` give different dimensions on a real evaluation, and read
# that as an evaluation decision that "moves every drawn column either way".
#
# It is a decision about the improvement, not about the migration: one of the
# two modes reproduces the carrier exactly. These are the tests of that, and
# they are deliberately split — the fixture cannot see the difference, so the
# claim needs the tape.


def test_the_two_union_modes_agree_when_every_component_shares_a_grid(micro):
    """Why the committed fixture could not have found this.

    `micro_fe56_mf33.endf` states MT4 and MT16 on the same 124-bin grid, so the
    pooled union and the per-key unions are the same grid and both modes give
    the same 248x248. The fixture is not wrong — it simply cannot distinguish
    the two, which is exactly why P4's measurement had to be taken on a tape.
    """
    _endf, suite = micro

    perComponent = cross_section_covariance_blocks(suite, union="per-component")
    globalGrid = cross_section_covariance_blocks(suite, union="global")

    assert globalGrid[0][1].shape == (248, 248)
    np.testing.assert_array_equal(globalGrid[0][1], perComponent[0][1])


def test_an_unknown_union_mode_is_refused_by_name(micro):
    """The two modes are two different covariances, so a typo must not pick one."""
    _endf, suite = micro
    with pytest.raises(ValueError, match="union must be one of"):
        cross_section_covariance_blocks(suite, union="union")


def test_the_index_and_the_block_are_built_on_the_same_grid(micro):
    """An index on one bin structure and a block on the other is wrong
    everywhere without being wrong anywhere visible."""
    _endf, suite = micro
    for union in ("global", "per-component"):
        block = cross_section_covariance_blocks(suite, union=union)[0][1]
        entry = next(iter(cross_section_covariance_index(
            suite, union=union).values()))
        assert entry["dimension"] == block.shape[0], union


def test_the_global_mode_reproduces_the_carrier_on_a_real_evaluation(
    fe56_host_tape,
):
    """**The migration gate the fixture cannot give.** `allclose` would not do.

    Fe-56 JEFF-4.0 states MF33 for MT 1/2/4/5/16/102/103 on *four different*
    native grids — 630, 630, 124, 124, 124, 631, 124 bins — which is the case
    `micro_fe56_mf33.endf` does not have and P4 was deferred on. Measured
    2026-08-13:

    | mode | dimension | padded rows |
    |---|---|---|
    | `per-component` | 7 x 631 = 4417 | 2236 |
    | `global` | 7 x 730 = **5110** | 0 |

    and the 5110 is bit-identical to the carrier. So the source half of the MF33
    migration moves no number, which is what `equivalent first, then improve`
    asks for before anything else is considered.
    """
    from kika.sampling.mf33_sampling import load_mf33_covariance

    mts = [1, 2, 4, 5, 16, 102, 103]
    cov, _mf3, grid, present = load_mf33_covariance(
        str(fe56_host_tape), None, mts, energy_unit="eV",
    )
    assert present == mts, "the tape's MF33 MT set is what this gate is about"

    reference = cov.covariance_matrix
    reference = np.where(np.isfinite(reference), reference, 0.0)
    assert reference.shape == (5110, 5110)

    suite, _ = decodeCovarianceSuite(read_endf(str(fe56_host_tape)))

    blocks = cross_section_covariance_blocks(suite, mt=mts, union="global")
    assert len(blocks) == 1
    _key, joint = blocks[0]

    assert joint.shape == reference.shape
    np.testing.assert_array_equal(joint, reference)

    # The pooled grid is `_build_union_grid`'s, not merely the same size as it.
    entry = next(iter(cross_section_covariance_index(
        suite, mt=mts, union="global").values()))
    np.testing.assert_array_equal(
        entry["grids"][entry["pairs"][0]], np.asarray(grid, dtype=float)
    )
    assert entry["pairs"] == [tuple(p) for p in cov._get_param_pairs()]
    assert set(entry["widths"].values()) == {entry["stride"]}, (
        "under the global union every component is stated at full width; a "
        "padded row here would mean the two modes had been mixed"
    )


def test_the_per_component_mode_is_a_different_covariance_on_that_tape(
    fe56_host_tape,
):
    """The other half of the same claim, and the reason the default matters.

    Recorded as its own test so that "the two modes agree" can never be
    concluded from the fixture alone. 2236 of 4417 rows are padding — a
    *decomposition* returns nothing for them, so this is not a relabelling of
    the 5110 but a smaller covariance.
    """
    suite, _ = decodeCovarianceSuite(read_endf(str(fe56_host_tape)))
    mts = [1, 2, 4, 5, 16, 102, 103]

    joint = cross_section_covariance_blocks(
        suite, mt=mts, union="per-component")[0][1]

    assert joint.shape == (4417, 4417)
    assert int((~joint.any(axis=1)).sum()) == 2236


# ----------------------------------------------------------------------
# MF31 — the same records pointed at a multiplicity
# ----------------------------------------------------------------------
#
# §31.1 makes the MT452/455/456 formats "directly analogous to those of File
# 33", and `mf31_sampling.py` acts on it: it imports MF33's `_build_union_grid`
# and packs into the same carrier. So the assembly is one function with an `mf`
# parameter, not two, and these are the tests that say the sharing is real
# rather than assumed.


@pytest.fixture(scope="module")
def nubar(micro_nubar_tape):
    endf = read_endf(str(micro_nubar_tape))
    suite, _ = decodeCovarianceSuite(endf)
    return endf, suite


@pytest.fixture(scope="module")
def nubar_carrier(micro_nubar_tape):
    from kika.sampling.mf31_sampling import load_mf31_covariance

    cov, _sections, grid, present = load_mf31_covariance(
        str(micro_nubar_tape), energy_unit="eV",
    )
    return cov, grid, present


def test_the_mf31_assembly_reproduces_its_carrier(nubar, nubar_carrier):
    """The MF31 half of the migration gate, on the committed U-235 slice.

    That fixture is the one that carries all three nu-bar MTs *and* the NC-type
    (LTY=0) total — the covariance the file declares as a sum rather than
    storing — so this is not the easy subset of MF31.
    """
    _endf, suite = nubar
    cov, _grid, present = nubar_carrier

    reference = cov.covariance_matrix
    reference = np.where(np.isfinite(reference), reference, 0.0)

    blocks = cross_section_covariance_blocks(suite, mf=31, mt=present)
    assert len(blocks) == 1
    key, joint = blocks[0]

    assert key[1] == "MF31", "the MF is in the key, so MF31 and MF33 cannot collide"
    assert joint.shape == reference.shape
    np.testing.assert_array_equal(joint, reference)


def test_the_two_files_do_not_leak_into_each_other(nubar):
    """`mf=` is a filter, not a hint, and each file's assembly excludes the other.

    The fixture carries MF31 and no MF33, so the MF33 call must come back empty
    rather than sweeping the nu-bar sections in — the same property
    `test_mf34_sections_are_not_swept_into_an_mf33_assembly` holds one file over.
    """
    _endf, suite = nubar
    assert cross_section_covariance_blocks(suite, mf=31), "the fixture carries MF31"
    assert cross_section_covariance_blocks(suite, mf=33) == []
    assert cross_section_covariance_index(suite, mf=33) == {}


def test_an_mf_whose_components_are_not_reactions_is_refused_by_name(nubar):
    """MF34's key carries a Legendre order and MF35's a band.

    Neither is a `(ZA, MT)` pair, so this must refuse rather than silently
    assemble a covariance whose rows mean something other than what the index
    says they mean.
    """
    _endf, suite = nubar
    for mf in (34, 35, 32):
        with pytest.raises(ValueError, match="mf must be one of"):
            cross_section_covariance_blocks(suite, mf=mf)
