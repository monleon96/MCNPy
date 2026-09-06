"""The perturbation-matrix manifest must outlive the run that builds it.

Every pipeline (nu-bar, PENDF, ACE) writes its per-isotope parquet parts into a
temporary directory, records what it did in ``metadata.json`` there, and then
calls :func:`_finalize_master_perturbation_matrix`, which concatenates the parts
into one master parquet and deletes the directory. The manifest is the only
place on disk that says which of the master's columns were actually applied —
the caller's summary dict never leaves memory — so it has to be copied out
before the delete.

These are unit tests on the helpers themselves: no tape, no NJOY, no sampling.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kika.sampling.utils import (
    _initialize_master_perturbation_matrix,
    _write_isotope_parquet,
    _merge_isotope_metadata,
    _finalize_master_perturbation_matrix,
)

GRID = [1.0e-5, 1.0e3, 1.0e6, 2.0e7]


def _build_parts(tmp_path, extra_metadata=None, zaid=94241):
    matrix_dir = _initialize_master_perturbation_matrix(
        str(tmp_path), "20260807_100000", num_samples=2
    )
    factors = np.ones((2, 2 * (len(GRID) - 1)))
    frag = _write_isotope_parquet(
        matrix_dir, zaid, factors, [455, 456], GRID,
        verbose=False, extra_metadata=extra_metadata,
    )
    _merge_isotope_metadata(matrix_dir, [frag])
    return matrix_dir


def _manifest(tmp_path):
    import glob
    hits = glob.glob(os.path.join(str(tmp_path), "*_metadata.json"))
    assert hits, f"no manifest beside the master parquet in {os.listdir(tmp_path)}"
    with open(hits[0]) as f:
        return json.load(f)


def test_manifest_survives_the_parts_cleanup(tmp_path):
    matrix_dir = _build_parts(tmp_path)
    master = _finalize_master_perturbation_matrix(matrix_dir, verbose=False)

    assert os.path.exists(master)
    assert not os.path.exists(matrix_dir), "the parts directory should be gone"
    meta = _manifest(tmp_path)
    assert meta["num_samples"] == 2
    assert meta["isotopes_processed"] == [94241]


def test_manifest_carries_the_per_isotope_details(tmp_path):
    # What the nu-bar pipeline puts there: MF31 samples the redundant total but
    # the sum rule derives it, so its columns are in the parquet without being
    # free parameters. Nothing downstream can see that from the columns alone.
    matrix_dir = _build_parts(tmp_path, extra_metadata={
        "derived_mt": 452, "mts_sampled": [452, 455, 456],
        "mts_applied": [455, 456],
    })
    _finalize_master_perturbation_matrix(matrix_dir, verbose=False)

    details = _manifest(tmp_path)["isotope_details"]["94241"]
    assert details["derived_mt"] == 452
    assert details["mts_applied"] == [455, 456]


def test_pipelines_without_details_get_an_unchanged_manifest(tmp_path):
    # PENDF and ACE pass no extra metadata; their manifest must not sprout an
    # empty ``isotope_details`` key.
    matrix_dir = _build_parts(tmp_path)
    _finalize_master_perturbation_matrix(matrix_dir, verbose=False)

    assert "isotope_details" not in _manifest(tmp_path)


def test_finalize_without_parts_writes_no_master(tmp_path):
    # Nothing was produced, so there is no artefact for a manifest to annotate.
    matrix_dir = _initialize_master_perturbation_matrix(
        str(tmp_path), "20260807_100000", num_samples=2
    )
    assert _finalize_master_perturbation_matrix(matrix_dir, verbose=False) == ""
