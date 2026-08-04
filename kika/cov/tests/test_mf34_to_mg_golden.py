"""The sigma weighting of the MF34 multigroup collapse, frozen before it moves.

``MF34_to_MG`` weights every group by sigma(E)*phi(E), and gets sigma(E) by
calling ``endf_object.reconstruct_xs()`` — with no try/except and, until this
file, not one test. That reconstructor is documented as *"not working
correctly and should not be used"*.

So today's collapsed MF34 matrices are weighted by a sigma nobody trusts, and
nothing would notice if it changed. ``MF34_to_MG`` (aliased
``collapse_to_multigroup``) is what the thesis notebooks call — Sandwich_leg,
ASPIS88/sandwich_leg, PHYSOR26/check_mg, the chapter 3 figures — so "changed"
means chapter numbers move.

This golden is deliberately written *before* the sigma source changes, so the
next commit can say by how much rather than hoping it was small. It is a
characterization test: it asserts today's numbers, not correct ones.

Regenerate after an intentional change with::

    REGEN_NUMERIC_GOLDENS=1 pytest kika/cov/tests/test_mf34_to_mg_golden.py

and commit the diff in the same commit as the change that moved it.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from kika.cov.multigroup import MF34_to_MG
from kika.endf import read_endf

DATA = Path(__file__).resolve().parent / "data"
REGEN = bool(os.environ.get("REGEN_NUMERIC_GOLDENS"))
RTOL = 1e-12

#: Coarse enough to stay small, wide enough to span the MF34 range where the
#: Fe-56 elastic cross section actually varies.
GRID_EV = [1.0e5, 5.0e5, 1.0e6, 5.0e6, 2.0e7]


def _check_golden(name: str, produced: dict) -> None:
    path = DATA / f"{name}.npz"

    if REGEN:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **produced)
        return

    if not path.is_file():
        pytest.fail(
            f"golden {path.name} is missing — generate it with "
            f"REGEN_NUMERIC_GOLDENS=1 and commit it"
        )

    with np.load(path) as golden:
        assert sorted(golden.files) == sorted(produced), (
            f"{name}: array set changed: {sorted(golden.files)} -> {sorted(produced)}"
        )
        for key in sorted(produced):
            want, have = golden[key], np.asarray(produced[key])
            assert have.shape == want.shape, (
                f"{name}[{key}]: shape {want.shape} -> {have.shape}"
            )
            if want.dtype.kind in "US":  # digests compare exactly
                assert have == want, f"{name}[{key}] changed: {want} -> {have}"
                continue
            np.testing.assert_allclose(
                have, want, rtol=RTOL, atol=0.0, err_msg=f"{name}[{key}] moved"
            )


@pytest.fixture(scope="module")
def collapsed(micro_tape):
    """MF34/MT2 collapsed onto GRID_EV, with whatever sigma the code uses."""
    endf = read_endf(str(micro_tape))
    with warnings.catch_warnings():
        # The reconstructor this depends on warns that it is broken. That is
        # the fact this file exists to pin, not a reason to fail collection.
        warnings.simplefilter("ignore", DeprecationWarning)
        result = MF34_to_MG(endf, energy_grid=GRID_EV, mt=2)
    return endf, result


def _arrays(result) -> dict:
    out = {"energy_grid": np.asarray(result.energy_grid, dtype=float)}
    for i, matrix in enumerate(result.relative_matrices):
        out[f"relative_{i}"] = np.asarray(matrix, dtype=float)
    for i, matrix in enumerate(result.absolute_matrices):
        out[f"absolute_{i}"] = np.asarray(matrix, dtype=float)
    return out


def test_collapsed_matrices_golden(collapsed):
    """The matrices the thesis notebooks consume."""
    _, result = collapsed
    _check_golden("mf34_to_mg_mt2", _arrays(result))


def test_the_sigma_weight_golden(collapsed):
    """The weight itself, separately — this is what the next commit moves.

    Pinning the collapse alone would say *that* something changed; pinning
    sigma says what, and lets the two be compared independently.

    Summarised rather than stored whole: the reconstructed grid runs to ~10^5
    points per MT, which is megabytes of fixture for a number that only has to
    be comparable. Six statistics plus a digest catch any real change, and the
    digest catches a change the statistics would miss.
    """
    endf, _ = collapsed
    assert endf.pendf, "nothing populated pendf, so no sigma was weighted with"

    arrays = {}
    for mt in sorted(endf.pendf):
        section = endf.pendf[mt]
        energies = np.asarray(section.energies, dtype=float)
        values = np.asarray(section.cross_sections, dtype=float)
        arrays[f"mt{mt}_summary"] = np.array([
            energies.size,
            energies[0], energies[-1],
            values.min(), values.max(), values.sum(),
        ], dtype=float)
        arrays[f"mt{mt}_digest"] = np.array(
            hashlib.sha256(
                np.ascontiguousarray(energies).tobytes()
                + np.ascontiguousarray(values).tobytes()
            ).hexdigest()
        )
    _check_golden("mf34_to_mg_sigma", arrays)


def test_the_collapse_is_symmetric(collapsed):
    """Structural, so it survives the sigma source changing under it."""
    _, result = collapsed
    for i, matrix in enumerate(result.relative_matrices):
        matrix = np.asarray(matrix, dtype=float)
        np.testing.assert_allclose(
            matrix, matrix.T, rtol=1e-10, atol=0.0,
            err_msg=f"relative matrix {i} is not symmetric",
        )


def test_the_grid_is_what_was_asked_for(collapsed):
    _, result = collapsed
    np.testing.assert_allclose(
        np.asarray(result.energy_grid, dtype=float), np.asarray(GRID_EV, dtype=float)
    )
