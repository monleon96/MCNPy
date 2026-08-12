"""MF33 sections → the one covariance they are partitions of.

The cross-*reaction* half of what `test_legendre_blocks.py` does for Legendre
orders, and P3 of `docs/sampling_migration_roadmap.md`. The property held is the
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
    _mf33_entries,
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
    filter is wired rather than that it fires — `_mf33_entries` is where the
    rule lives and `legendre_covariance_blocks` carries the identical one.
    """
    _endf, suite = micro
    assert all(s.form.isRelative for s in suite.covarianceSections
               if str(s.rowData.ENDF_MFMT or "").startswith("33/"))
    assert len(_mf33_entries(suite, relative=True)) == 2
    assert _mf33_entries(suite, relative=False) == []
