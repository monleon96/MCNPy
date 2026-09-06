"""M1's gate: the covariance *source* moves onto the model and no number moves.

The draw migrated in P4 (``sampling_migration_roadmap.md``) and the source did
not: ``mf33_sampling`` and ``mf31_sampling`` still assemble a
:class:`~kika.cov.cross_section_covariance.CrossSectionCovariance` out of format
objects, hand it to ``carrier_blocks``, and draw. This is the other half —
:func:`~kika.sampling.mf33_sampling.loadCrossSectionBlocks` reads the
``CovarianceSuite`` and assembles the same joint matrix — and the property held
is the one P4 held: **bit identity, on the drawn factors**, not ``allclose`` and
not on the matrix alone.

*The matrix alone is not enough, and that is not pedantry.* Two matrices that
compare equal can still be drawn from differently if the key, the pair order or
the stride differ, because those are what the flat factors vector is sliced by.
So the gate here runs the draw the pipelines run, with their seed and their
repair plan, and compares the factors.

**Why the absolute→relative conversion is tested separately and at all.**
``load_mf33_covariance`` converts a block stated in absolute units by dividing
by a bin-averaged σ̄ *after* projecting it onto the union grid. The covariance
suite has no σ̄ — it is on the PENDF — so the model source cannot inherit that
step and has to be handed the same input. It is the one piece of
``load_mf33_covariance`` that is not assembly, it is the piece the committed
fixtures do not exercise (every MF33 section on both micro-tapes is relative,
which the first test asserts so that a fixture change cannot quietly void the
rest), and so it is measured on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._covariance_forms import require_single_matrix
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.endf.read_endf import read_endf
from kika.sampling.carrier_blocks import (cross_section_carrier_blocks,
                                          cross_section_carrier_index)
from kika.sampling.mf33_sampling import (_absolute_to_relative, _bin_average_xs,
                                         _project_matrix, extractParamBlocks,
                                         extract_mt_param_blocks,
                                         loadCrossSectionBlocks,
                                         load_mf33_covariance,
                                         relativiseAbsoluteSections)
from kika.sampling.model_blocks import (_cross_section_entries, _endf_mt,
                                        _is_endf_mf, _lift_matrix)
from kika.sampling.multigroup_draw import draw_relative_factors

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
MICRO_MF33 = str(DATA / "micro_fe56_mf33.endf")
MICRO_NUBAR = str(DATA / "micro_u235_nubar.endf")

#: What `exfor_to_endf_research.py` ships, so the gate holds the shipped draw.
SEED = 12345
N_SAMPLES = 8
MTS = [4, 16]


@pytest.fixture(scope="module")
def mf33():
    """Both sources of the same covariance, built once."""
    cov, mf3, grid, present = load_mf33_covariance(
        MICRO_MF33, None, MTS, energy_unit="eV")
    endf = read_endf(MICRO_MF33)
    blocks, index, unionGrid, mts = loadCrossSectionBlocks(endf, MTS, mf3)
    return dict(cov=cov, grid=grid, present=present, endf=endf,
                blocks=blocks, index=index, unionGrid=unionGrid, mts=mts)


# ---------------------------------------------------------------------------
# The fixture's own shape, so the rest means what it says
# ---------------------------------------------------------------------------

def test_every_section_of_the_fixture_is_relative(mf33):
    """The premise of the equivalence tests below, asserted rather than assumed.

    If a future fixture carried an absolute block, the tests below would exercise
    the conversion path silently and stop measuring what they claim to.
    """
    suite, _ = decodeCovarianceSuite(mf33["endf"])
    flags = [require_single_matrix(section.form, "").isRelative
             for section in suite.covarianceSections
             if _is_endf_mf(section.rowData, 33)]
    assert flags and all(flags)


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------

def test_the_grid_and_the_mt_set_are_the_carrier_s(mf33):
    np.testing.assert_array_equal(np.asarray(mf33["unionGrid"]),
                                  np.asarray(mf33["grid"]))
    assert mf33["mts"] == mf33["present"]


def test_the_joint_matrix_is_bit_identical_to_the_carrier_s(mf33):
    (_key, joint), = mf33["blocks"]
    (_ckey, carrier), = cross_section_carrier_blocks(mf33["cov"])
    reference = np.where(np.isfinite(carrier), carrier, 0.0)

    assert joint.shape == reference.shape
    np.testing.assert_array_equal(joint, reference)


def test_the_block_key_and_the_index_are_the_carrier_s(mf33):
    """The layout, not just the numbers. A different key or pair order slices
    the flat factors vector differently and nothing about the matrix says so."""
    (key, _joint), = mf33["blocks"]
    (carrierKey, _), = cross_section_carrier_blocks(mf33["cov"])
    assert key == carrierKey

    (meta,) = mf33["index"].values()
    (carrierMeta,) = cross_section_carrier_index(mf33["cov"]).values()
    assert meta["pairs"] == [tuple(p) for p in carrierMeta["pairs"]]
    assert meta["stride"] == carrierMeta["stride"]
    assert meta["dimension"] == carrierMeta["dimension"]
    assert extractParamBlocks(mf33["index"]) == extract_mt_param_blocks(mf33["cov"])


# ---------------------------------------------------------------------------
# The draw — the gate P4 was held to
# ---------------------------------------------------------------------------

def _draw(key, matrix, meta, grid, *, space, thresholds):
    factors, _info = draw_relative_factors(
        matrix, N_SAMPLES, key=key, pairs=meta["pairs"],
        stride=meta["stride"], bins=grid, space=space,
        decomposition_method="svd", sampling_method="sobol", seed=SEED,
        mt_thresholds=thresholds, verbose=False,
    )
    return factors


@pytest.mark.parametrize("space", ["log", "linear"])
@pytest.mark.parametrize("withThresholds", [False, True])
def test_the_drawn_factors_are_bit_identical(mf33, space, withThresholds):
    grid = mf33["grid"]
    thresholds = ({4: float(grid[len(grid) // 3]),
                   16: float(grid[len(grid) // 2])} if withThresholds else None)

    (carrierKey, carrierMatrix), = cross_section_carrier_blocks(mf33["cov"])
    (carrierMeta,), = (cross_section_carrier_index(mf33["cov"]).values(),)
    fromCarrier = _draw(carrierKey, carrierMatrix, carrierMeta, grid,
                        space=space, thresholds=thresholds)

    (modelKey, modelMatrix), = mf33["blocks"]
    (modelMeta,) = mf33["index"].values()
    fromModel = _draw(modelKey, modelMatrix, modelMeta, mf33["unionGrid"],
                      space=space, thresholds=thresholds)

    assert fromModel.shape == fromCarrier.shape
    np.testing.assert_array_equal(fromModel, fromCarrier)


# ---------------------------------------------------------------------------
# MF31, the same function with one integer changed
# ---------------------------------------------------------------------------

def test_mf31_reproduces_its_carrier_too():
    from kika.sampling.mf31_sampling import (build_mf31_blocks,
                                              load_mf31_covariance)

    cov, nubar, grid, present = load_mf31_covariance(
        MICRO_NUBAR, None, energy_unit="eV")
    endf = read_endf(MICRO_NUBAR)
    blocks, index, modelNubar, unionGrid, mts = build_mf31_blocks(endf, None)

    # The wrapper hands back the same sections the appliers rewrite, which is a
    # different object from the central-value shim the conversion divides by.
    assert set(modelNubar) == set(nubar)

    (_key, joint), = blocks
    (_ckey, carrier), = cross_section_carrier_blocks(cov)
    reference = np.where(np.isfinite(carrier), carrier, 0.0)

    assert mts == present
    np.testing.assert_array_equal(np.asarray(unionGrid), np.asarray(grid))
    np.testing.assert_array_equal(joint, reference)

    (meta,) = index.values()
    (carrierMeta,) = cross_section_carrier_index(cov).values()
    assert meta["pairs"] == [tuple(p) for p in carrierMeta["pairs"]]
    assert extractParamBlocks(index) == extract_mt_param_blocks(cov)

    factors = _draw(("mf31", "MF31", tuple(meta["pairs"])), joint, meta,
                    unionGrid, space="log", thresholds=None)
    reference_factors = _draw(("mf31", "MF31", tuple(meta["pairs"])), reference,
                              carrierMeta, grid, space="log", thresholds=None)
    np.testing.assert_array_equal(factors, reference_factors)


# ---------------------------------------------------------------------------
# The one step the model cannot inherit
# ---------------------------------------------------------------------------

def test_lifting_then_dividing_is_what_the_carrier_does(mf33):
    """``_lift_matrix`` and ``_project_matrix`` agree, so the order carries over.

    The carrier projects and then divides. The model source lifts and then
    divides. That is the same matrix only while the two mappings agree, which is
    a fact about these grids rather than about the two functions — so it is
    measured here, on the real blocks, rather than reasoned about.
    """
    suite, _ = decodeCovarianceSuite(mf33["endf"])
    union = np.asarray(mf33["unionGrid"], dtype=float)
    entries = _cross_section_entries(suite, mf=33, mt=MTS)
    assert entries

    for _rowKey, _colKey, matrix, rowGrid, colGrid in entries:
        projected = _project_matrix(matrix, list(rowGrid), list(union))
        lifted = (_lift_matrix(rowGrid, union) @ matrix
                  @ _lift_matrix(colGrid, union).T)
        np.testing.assert_array_equal(lifted, projected)


def test_an_absolute_section_is_converted_the_way_the_carrier_converts_it(mf33):
    """The conversion, on a block made absolute on purpose.

    No committed fixture carries an absolute MF33 block (``is_relative`` is
    ``not lb_set.issubset({0, 8, 9})``, and neither micro-tape uses those LBs),
    so the only way to exercise the path is to state one. The reference is the
    arithmetic ``load_mf33_covariance`` runs on the same inputs, not a
    hand-computed expectation.
    """
    import dataclasses

    suite, _ = decodeCovarianceSuite(mf33["endf"])
    union = np.asarray(mf33["unionGrid"], dtype=float)

    # A plausible σ̄, on the union grid, so the division is a real one.
    sigma = {mt: np.linspace(1.0, 3.0, union.size - 1) * (1 + mt / 100.0)
             for mt in MTS}

    made = []
    for section in suite.covarianceSections:
        if not _is_endf_mf(section.rowData, 33):
            continue
        form = require_single_matrix(section.form, "")
        made.append(dataclasses.replace(
            section, form=dataclasses.replace(form, isRelative=False)))
    assert made, "the fixture has no MF33 section to make absolute"

    converted = relativiseAbsoluteSections(made, sigma, union, mf=33)
    assert len(converted) == len(made)

    for original, result in zip(made, converted):
        form = require_single_matrix(original.form, "")
        mt = _endf_mt(original.rowData)
        rowGrid = np.asarray(form.rowGrid, dtype=float)
        expected = _absolute_to_relative(
            _project_matrix(np.asarray(form.matrix, dtype=float),
                            list(rowGrid), list(union)),
            sigma[mt], sigma[mt])

        got = require_single_matrix(result.form, "")
        assert got.isRelative is True
        np.testing.assert_array_equal(np.asarray(got.matrix), expected)
        np.testing.assert_array_equal(np.asarray(got.rowGrid), union)


def test_an_absolute_block_with_no_central_value_is_dropped_and_said(mf33, caplog):
    """The carrier skips it with a warning; so does this, and for the same reason."""
    import dataclasses
    import logging

    suite, _ = decodeCovarianceSuite(mf33["endf"])
    union = np.asarray(mf33["unionGrid"], dtype=float)

    made = [dataclasses.replace(
                section,
                form=dataclasses.replace(
                    require_single_matrix(section.form, ""), isRelative=False))
            for section in suite.covarianceSections
            if _is_endf_mf(section.rowData, 33)]

    logger = logging.getLogger("m1-gate")
    with caplog.at_level(logging.WARNING, logger="m1-gate"):
        kept = relativiseAbsoluteSections(made, {}, union, mf=33, logger=logger)

    assert kept == []
    assert "no central values" in caplog.text


def test_a_suite_handed_in_is_not_modified(mf33):
    """The conversion returns new sections. A caller holding the carrier as well
    would otherwise find its covariance changed underneath it."""
    import dataclasses

    suite, _ = decodeCovarianceSuite(mf33["endf"])
    union = np.asarray(mf33["unionGrid"], dtype=float)
    sigma = {mt: np.full(union.size - 1, 2.0) for mt in MTS}

    made = [dataclasses.replace(
                section,
                form=dataclasses.replace(
                    require_single_matrix(section.form, ""), isRelative=False))
            for section in suite.covarianceSections
            if _is_endf_mf(section.rowData, 33)]
    before = [np.asarray(require_single_matrix(s.form, "").matrix).copy()
              for s in made]

    relativiseAbsoluteSections(made, sigma, union, mf=33)

    for section, original in zip(made, before):
        form = require_single_matrix(section.form, "")
        assert form.isRelative is False
        np.testing.assert_array_equal(np.asarray(form.matrix), original)


# ---------------------------------------------------------------------------
# Written output — the gate P1 had to learn
# ---------------------------------------------------------------------------

def test_the_written_pendf_is_byte_identical(tmp_path):
    """The migration is measured on a file, not on an array.

    P1 first ticked its acceptance on an inference from the draw and had to be
    corrected the same day: comparing matrices in memory and assuming the tapes
    follow is how a wrong migration gets through. So this runs the applier both
    pipelines run and diffs the bytes.

    The micro-tape stands in for the PENDF -- it carries MF3 for both MTs, which
    is all ``apply_factors_to_pendf_mf3`` reads -- so the whole gate runs off
    committed data.
    """
    from kika.sampling.mf33_sampling import (apply_factors_to_pendf_mf3,
                                             load_mf33_blocks)

    cov, mf3, grid, present = load_mf33_covariance(
        MICRO_MF33, MICRO_MF33, MTS, energy_unit="eV")
    (carrierKey, carrierMatrix), = cross_section_carrier_blocks(cov)
    (carrierMeta,) = cross_section_carrier_index(cov).values()
    carrierFactors = _draw(carrierKey, carrierMatrix, carrierMeta, grid,
                           space="log", thresholds=None)

    blocks, index, modelMf3, unionGrid, mts = load_mf33_blocks(
        MICRO_MF33, MICRO_MF33, MTS)
    (modelKey, modelMatrix), = blocks
    (modelMeta,) = index.values()
    modelFactors = _draw(modelKey, modelMatrix, modelMeta, unionGrid,
                         space="log", thresholds=None)

    np.testing.assert_array_equal(modelFactors, carrierFactors)
    assert set(modelMf3) == set(mf3)

    fromCarrier = tmp_path / "carrier.pendf"
    fromModel = tmp_path / "model.pendf"
    apply_factors_to_pendf_mf3(
        MICRO_MF33, str(fromCarrier), mf3, carrierFactors[0],
        extract_mt_param_blocks(cov), grid)
    apply_factors_to_pendf_mf3(
        MICRO_MF33, str(fromModel), modelMf3, modelFactors[0],
        extractParamBlocks(index), unionGrid)

    assert fromModel.read_bytes() == fromCarrier.read_bytes()
    # And the perturbation actually happened -- two identical *unperturbed*
    # files would pass the line above and mean nothing.
    assert fromModel.read_bytes() != Path(MICRO_MF33).read_bytes()


def test_per_component_with_an_absolute_block_is_refused(mf33, monkeypatch):
    """The unsound combination raises instead of guessing.

    ``per-component`` gives each key its own bins, so there is no single grid to
    state a converted absolute block on. Converting it on one key's grid while
    the assembly expects another would be wrong everywhere and visible nowhere —
    exactly the class of defect this whole roadmap is about — so the mode is
    refused while an absolute block is present, and works when none is.
    """
    from kika.sampling import mf33_sampling

    # No absolute block: the mode is fine and assembles.
    blocks, index, grid, mts = loadCrossSectionBlocks(
        mf33["endf"], MTS, None, union="per-component")
    assert blocks and mts == MTS

    # One absolute block: refused, and the message says which and why.
    monkeypatch.setattr(mf33_sampling, "_absoluteSections",
                        lambda suite, mf: [(4, 4)])
    with pytest.raises(NotImplementedError, match=r"per-component gives each"):
        loadCrossSectionBlocks(mf33["endf"], MTS, None, union="per-component")

    # And `global`, the shipped layout, is unaffected by the same block.
    blocks, _index, _grid, _mts = loadCrossSectionBlocks(
        mf33["endf"], MTS, None, union="global")
    assert blocks


# ---------------------------------------------------------------------------
# Written output — the gate P1 had to learn
# ---------------------------------------------------------------------------

def test_the_written_pendf_is_byte_identical(tmp_path):
    """The migration is measured on a file, not on an array.

    P1 first ticked its acceptance on an inference from the draw and had to be
    corrected the same day: comparing matrices in memory and assuming the tapes
    follow is how a wrong migration gets through. So this runs the applier both
    pipelines run and diffs the bytes.

    The micro-tape stands in for the PENDF -- it carries MF3 for both MTs, which
    is all ``apply_factors_to_pendf_mf3`` reads -- so the whole gate runs off
    committed data.
    """
    from kika.sampling.mf33_sampling import (apply_factors_to_pendf_mf3,
                                             load_mf33_blocks)

    cov, mf3, grid, present = load_mf33_covariance(
        MICRO_MF33, MICRO_MF33, MTS, energy_unit="eV")
    (carrierKey, carrierMatrix), = cross_section_carrier_blocks(cov)
    (carrierMeta,) = cross_section_carrier_index(cov).values()
    carrierFactors = _draw(carrierKey, carrierMatrix, carrierMeta, grid,
                           space="log", thresholds=None)

    blocks, index, modelMf3, unionGrid, mts = load_mf33_blocks(
        MICRO_MF33, MICRO_MF33, MTS)
    (modelKey, modelMatrix), = blocks
    (modelMeta,) = index.values()
    modelFactors = _draw(modelKey, modelMatrix, modelMeta, unionGrid,
                         space="log", thresholds=None)

    np.testing.assert_array_equal(modelFactors, carrierFactors)
    assert set(modelMf3) == set(mf3)

    fromCarrier = tmp_path / "carrier.pendf"
    fromModel = tmp_path / "model.pendf"
    apply_factors_to_pendf_mf3(
        MICRO_MF33, str(fromCarrier), mf3, carrierFactors[0],
        extract_mt_param_blocks(cov), grid)
    apply_factors_to_pendf_mf3(
        MICRO_MF33, str(fromModel), modelMf3, modelFactors[0],
        extractParamBlocks(index), unionGrid)

    assert fromModel.read_bytes() == fromCarrier.read_bytes()
    # And the perturbation actually happened -- two identical *unperturbed*
    # files would pass the line above and mean nothing.
    assert fromModel.read_bytes() != Path(MICRO_MF33).read_bytes()


def test_per_component_with_an_absolute_block_is_refused(mf33, monkeypatch):
    """The unsound combination raises instead of guessing.

    ``per-component`` gives each key its own bins, so there is no single grid to
    state a converted absolute block on. Converting it on one key's grid while
    the assembly expects another would be wrong everywhere and visible nowhere —
    exactly the class of defect this whole roadmap is about — so the mode is
    refused while an absolute block is present, and works when none is.
    """
    from kika.sampling import mf33_sampling

    # No absolute block: the mode is fine and assembles.
    blocks, index, grid, mts = loadCrossSectionBlocks(
        mf33["endf"], MTS, None, union="per-component")
    assert blocks and mts == MTS

    # One absolute block: refused, and the message says which and why.
    monkeypatch.setattr(mf33_sampling, "_absoluteSections",
                        lambda suite, mf: [(4, 4)])
    with pytest.raises(NotImplementedError, match=r"per-component gives each"):
        loadCrossSectionBlocks(mf33["endf"], MTS, None, union="per-component")

    # And `global`, the shipped layout, is unaffected by the same block.
    blocks, _index, _grid, _mts = loadCrossSectionBlocks(
        mf33["endf"], MTS, None, union="global")
    assert blocks
