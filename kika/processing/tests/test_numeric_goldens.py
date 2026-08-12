"""Tier-2 goldens: the numerics, frozen.

The tier-1 golden proves kika gives back the bytes it was given. This one
proves it gives back the same *numbers* — resonance reconstruction, the
adaptive linearization grid, and the multigroup collapse. Phases 2 and 4 of the
GNDS roadmap move all of this code onto a new model. Without a frozen answer,
"the refactor changed nothing" is an opinion.

Each case writes one ``.npz`` under ``data/``. Regenerate after an intentional
change with::

    REGEN_NUMERIC_GOLDENS=1 pytest kika/processing/tests/test_numeric_goldens.py

and commit the diff **in the same commit as the change that caused it**, with
one line saying why it moved. A golden updated in a commit of its own is a
golden nobody reviewed.

**Tolerance.** Array *shapes* are compared exactly — a linearization that emits
a different number of points has changed, full stop, and that is the signal
this file exists to catch. Values are compared at ``rtol=1e-12``, which is
tight enough to catch any change of algorithm and loose enough to survive a
different CPU running the CI job. Bit-exactness is the acceptance criterion for
phase 2, where the code moves without being touched; here the code is merely
frozen where it stands.

**Deliberately included: a reconstructor whose output is not trustworthy.**
Phase 0 froze it through ``ENDF.reconstruct_xs()``, a wrapper documented as
"not working correctly and should not be used" that four live sites called
anyway. Phase 1 deleted the wrapper and made every caller name its own sigma
source; the reconstructor underneath survives, so the pin moved one level down
to ``kika.endf.processing.reconstruct`` and the golden did not change. Phase 3
restructures the resonance code per formalism, and this is what will show what
that moves.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.nuclear_data.resonance_parameters import (
    LGroup,
    ResonanceParameters,
    ResonanceRecord,
)
from kika.processing import (
    collapse_covariance,
    compute_rebin_operator,
    reconstruct,
)
from kika.processing.linearization import linearize

DATA = Path(__file__).resolve().parent / "data"
REGEN = bool(os.environ.get("REGEN_NUMERIC_GOLDENS"))

#: Values match to this relative tolerance; shapes must match exactly.
RTOL = 1e-12


# ---------------------------------------------------------------------------
# Golden I/O
# ---------------------------------------------------------------------------

def check_golden(name: str, produced: dict[str, np.ndarray]) -> None:
    """Compare *produced* against the committed golden, or rewrite it."""
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
                have, want, rtol=RTOL, atol=0.0,
                err_msg=f"{name}[{key}] moved",
            )


# ---------------------------------------------------------------------------
# Synthetic resonance sets — one per formalism
# ---------------------------------------------------------------------------
# Three levels on a light-ish nuclide, chosen so every formalism has something
# to do: an s-wave and a p-wave group, a fission width that is zero for SLBW
# and MLBW and split across the two Reich-Moore channels. The numbers are
# arbitrary but fixed; what matters is that they never change.

_AWRI = 55.454
_ZA = 26056
_SPIN = 0.0
_AP = 0.5444
_RANGE = (1.0e-5, 1.0e5)


def _synthetic(formalism: str, fission: bool = False) -> list[ResonanceParameters]:
    if formalism in ("SLBW", "MLBW"):
        # c3=GT, c4=GN, c5=GG, c6=GF
        s_wave = [
            ResonanceRecord(energy=1.15e3, spin=0.5, c3=1.40, c4=1.02, c5=0.38, c6=0.0),
            ResonanceRecord(energy=2.28e4, spin=0.5, c3=2.10, c4=1.55, c5=0.55, c6=0.0),
        ]
        p_wave = [
            ResonanceRecord(energy=7.40e4, spin=1.5, c3=0.92, c4=0.61, c5=0.31, c6=0.0),
        ]
    else:  # Reich-Moore: c3=GN, c4=GG, c5=GFA, c6=GFB
        # The two fission widths are split unevenly and given opposite signs on
        # one level: GFA and GFB enter as reduced-width *amplitudes*, so a sign
        # is physics and not bookkeeping, and a lookup that swapped the two
        # channels would be invisible if they were equal.
        gfa = (0.21, 0.34, 0.12) if fission else (0.0, 0.0, 0.0)
        gfb = (-0.09, 0.17, 0.05) if fission else (0.0, 0.0, 0.0)
        s_wave = [
            ResonanceRecord(energy=1.15e3, spin=0.5, c3=1.02, c4=0.38,
                            c5=gfa[0], c6=gfb[0]),
            ResonanceRecord(energy=2.28e4, spin=0.5, c3=1.55, c4=0.55,
                            c5=gfa[1], c6=gfb[1]),
        ]
        p_wave = [
            ResonanceRecord(energy=7.40e4, spin=1.5, c3=0.61, c4=0.31,
                            c5=gfa[2], c6=gfb[2]),
        ]

    return [
        ResonanceParameters(
            nuclide_id=_ZA,
            spin=_SPIN,
            scattering_radius=_AP,
            formalism=formalism,
            energy_range=_RANGE,
            l_groups=[
                LGroup(awri=_AWRI, l=0, resonances=s_wave),
                LGroup(awri=_AWRI, l=1, resonances=p_wave),
            ],
            metadata={"awr": _AWRI, "mat": 2631},
        )
    ]


def _as_arrays(reconstructed: dict) -> dict[str, np.ndarray]:
    """Flatten ``{MT: CrossSection}`` into a saveable dict of arrays."""
    out: dict[str, np.ndarray] = {}
    first = next(iter(sorted(reconstructed)))
    out["energies"] = np.asarray(reconstructed[first].energies, dtype=float)
    for mt in sorted(reconstructed):
        xs = reconstructed[mt]
        np.testing.assert_array_equal(
            np.asarray(xs.energies), out["energies"],
            err_msg=f"MT{mt} is on a different grid from MT{first}",
        )
        out[f"mt{mt}"] = np.asarray(xs.values, dtype=float)
    return out


@pytest.mark.parametrize("formalism", ["SLBW", "MLBW", "RM"])
def test_reconstruction_golden(formalism):
    """Pointwise sigma(E) from three resonances, per formalism."""
    produced = reconstruct(_synthetic(formalism), tolerance=1e-3)
    assert produced, f"{formalism}: reconstruct returned nothing"
    check_golden(f"reconstruct_{formalism.lower()}", _as_arrays(produced))


def test_reconstruction_golden_reich_moore_with_fission():
    """The 3×3 collision-matrix branch, which no other golden reaches.

    ``reich_moore_cross_sections`` builds a 1×1 R-matrix when every GFA and GFB
    is zero and a 3×3 one otherwise, and the 3×3 arm is a different piece of
    code: an explicit inverse per energy point, two more reduced-width
    amplitudes, and the only place ``sigma_fission`` is non-zero. Every case
    above and the Fe-56 one below are structural materials with no fission, so
    the branch was frozen nowhere at all — and it is the branch where a
    per-formalism rewrite has to get *which width is which channel* right,
    because GFA and GFB are the two widths that ``c3..c6`` numbers and a named
    model has to look up.
    """
    produced = reconstruct(_synthetic("RM", fission=True), tolerance=1e-3)
    assert 18 in produced, "the fission case produced no MT18"
    check_golden("reconstruct_rm_fission", _as_arrays(produced))


#: ``(name, energy grid or None, level spacing, widths)`` for the two URR
#: shapes the corpus actually has. Case A is energy-independent — every average
#: is a scalar and ``_interpolate_urr_params`` returns its input untouched;
#: case C tabulates every average against an energy grid and goes through
#: ``np.interp``. Case B exists in ENDF and in no tape on this machine.
_URR_GRID = np.array([1.0e5, 3.0e5, 8.5e5], dtype=float)


def _synthetic_urr(case: str):
    from kika.nuclear_data.resonance_parameters import (
        URR_JGroup, URR_LGroup, UnresolvedResonanceParameters,
    )

    if case == "A":
        grid = None
        def j(spin, d, gn0, gg, gf, gx):
            return URR_JGroup(j=spin, amun=1.0, d=d, gn0=gn0, gg=gg, gf=gf, gx=gx)
        s_wave = [j(0.5, 7.2e3, 1.9e-2, 0.62, 0.0, 0.0)]
        p_wave = [j(0.5, 4.1e3, 8.0e-3, 0.55, 0.0, 0.0),
                  j(1.5, 2.6e3, 1.1e-2, 0.58, 0.0, 0.0)]
    else:
        grid = _URR_GRID
        def j(spin, d, gn0, gg, gf, gx):
            return URR_JGroup(
                j=spin, amun=1.0,
                d=np.asarray(d, dtype=float), gn0=np.asarray(gn0, dtype=float),
                gg=np.asarray(gg, dtype=float), gf=np.asarray(gf, dtype=float),
                gx=np.asarray(gx, dtype=float),
            )
        s_wave = [j(0.5, [7.2e3, 6.9e3, 6.4e3], [1.9e-2, 1.8e-2, 1.7e-2],
                    [0.62, 0.63, 0.65], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
        p_wave = [j(0.5, [4.1e3, 3.9e3, 3.6e3], [8.0e-3, 7.7e-3, 7.1e-3],
                    [0.55, 0.56, 0.58], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                  j(1.5, [2.6e3, 2.5e3, 2.3e3], [1.1e-2, 1.0e-2, 9.4e-3],
                    [0.58, 0.59, 0.61], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]

    return [UnresolvedResonanceParameters(
        nuclide_id=_ZA,
        spin=_SPIN,
        scattering_radius=_AP,
        energy_range=(1.0e5, 8.5e5),
        lssf=0,
        l_groups=[
            URR_LGroup(awri=_AWRI, l=0, j_groups=s_wave),
            URR_LGroup(awri=_AWRI, l=1, j_groups=p_wave),
        ],
        energy_grid=grid,
        metadata={"awr": _AWRI, "mat": 2631},
    )]


@pytest.mark.parametrize("case", ["A", "C"])
def test_urr_reconstruction_golden(case):
    """The unresolved region, which had no frozen answer of any kind.

    ``urr_formulas`` is a third of the reconstruction path and no golden
    touched it: the three synthetic cases are resolved-only and both Fe-56
    cases run on the committed micro-tape, which carries one resolved range and
    nothing else. So a rewrite of the URR code could have moved every
    unresolved cross section in the library and every test would still have
    passed.

    Fed with no resolved range on purpose, so this is the URR arithmetic and
    the geometric grid it builds for itself — nothing merged, nothing
    interpolated onto someone else's grid.
    """
    produced = reconstruct(_synthetic_urr(case), tolerance=1e-3)
    assert produced, f"URR case {case}: reconstruct returned nothing"
    check_golden(f"reconstruct_urr_case_{case.lower()}", _as_arrays(produced))


@pytest.mark.slow
def test_reconstruction_golden_real_fe56(micro_tape):
    """The real thing: 320 Reich-Moore resonances from the committed slice.

    The synthetic cases above exercise the formulas; this one exercises the
    whole path, including the adaptive grid seeding around 320 real levels.
    """
    endf = read_endf(str(micro_tape))
    params = ResonanceParameters.from_endf(endf.files[2].sections[151])
    assert len(params) == 1 and params[0].formalism == "RM"

    produced = reconstruct(params, tolerance=1e-2)
    check_golden("reconstruct_fe56_rm", _as_arrays(produced))


# ---------------------------------------------------------------------------
# Linearization — the grid is the artifact
# ---------------------------------------------------------------------------

def test_linearization_grid_golden():
    """The adaptive grid for a fixed analytic sigma(E).

    What is frozen is the **grid**, not the values on it: any two
    implementations that agree on the grid agree everywhere, and a change in
    the refinement rule shows up here as a different number of points long
    before it shows up as a visible error in a cross section.
    """
    def sigma(E):
        E = np.asarray(E, dtype=float)
        # Two Lorentzians plus a 1/v tail — smooth, sharp and steep at once.
        return (
            120.0 / (1.0 + ((E - 1.15e3) / 1.4) ** 2)
            + 45.0 / (1.0 + ((E - 2.28e4) / 2.1) ** 2)
            + 13.0 / np.sqrt(np.maximum(E, 1e-30))
        )

    grid = linearize(sigma, 1.0e-5, 1.0e5, tol=1e-3)
    check_golden(
        "linearize_grid",
        {"grid": np.asarray(grid, dtype=float), "sigma": sigma(grid)},
    )


# ---------------------------------------------------------------------------
# Multigroup collapse
# ---------------------------------------------------------------------------

def test_multigroup_collapse_golden():
    """Rebin operator and the congruence transform it drives.

    ``collapse_covariance`` is one line (``M @ C @ M.T``); the content is in
    ``compute_rebin_operator``, which integrates the weighting spectrum over
    every coarse/fine bin overlap. Both are frozen together because it is
    their composition that the pipeline uses.
    """
    coarse = np.array([1.0e-5, 1.0e2, 1.0e4, 1.0e6, 2.0e7])
    fine = np.array([1.0e-5, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 5.0e6, 2.0e7])

    n = len(coarse) - 1
    base = np.arange(1, n + 1, dtype=float)
    cov = 0.01 * np.outer(base, base) + np.diag(0.05 * base)
    cov = 0.5 * (cov + cov.T)

    operator = compute_rebin_operator(coarse, fine)
    collapsed = collapse_covariance(cov, operator)

    # Structural invariants, checked as well as frozen: the operator is
    # row-stochastic and the transform preserves symmetry.
    np.testing.assert_allclose(operator.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(collapsed, collapsed.T, rtol=1e-12)

    check_golden(
        "multigroup_collapse",
        {"operator": operator, "collapsed": collapsed, "coarse_cov": cov},
    )


# ---------------------------------------------------------------------------
# The ENDF-typed reconstruction adapter, frozen as it stands
# ---------------------------------------------------------------------------

def test_endf_reconstruct_adapter_golden(micro_tape):
    """``kika.endf.processing.reconstruct`` on a real MF2/MF3 pair.

    This is the ENDF-typed adapter: MF2/151 + MF3 in, ``{MT: MF3MT}`` out,
    with ``kika.processing.reconstruct`` doing the physics in between and
    ``CrossSection`` as the canonical form in the middle.

    It used to be reached through ``ENDF.reconstruct_xs()``, a convenience
    wrapper that also memoised onto ``endf._pendf``. That wrapper is gone — it
    was documented as producing incorrect cross sections and four live call
    sites used it anyway — but the reconstructor underneath is not, and phases
    2 and 4 still move it. So the pin stays, pointed one level down at the code
    that survived. Same numbers, same golden file: nothing about the arithmetic
    changed, only who calls it — the file is renamed, its contents are not.

    The output is still not trustworthy physics. It is frozen so that when the
    resonance work in phase 3 restructures it per formalism, the diff is
    visible.
    """
    from kika.endf.processing.reconstruct import reconstruct as endf_reconstruct

    endf = read_endf(str(micro_tape))
    produced = endf_reconstruct(endf.mf[2].mt[151], endf.files.get(3))

    assert produced, "the adapter returned nothing — its shape changed"

    # Fingerprinted rather than stored in full. The output runs to hundreds of
    # thousands of points per MT, and committing 3 MB to pin it is not a good
    # trade. A digest cannot say *how* it moved, but for a wholesale change
    # "it moved" is the message, and the summary statistics next to it give
    # the shape of the change.
    arrays: dict[str, np.ndarray] = {}
    for mt in sorted(produced):
        section = produced[mt]
        energies = np.asarray(section._energies, dtype=float)
        values = np.asarray(section._cross_sections, dtype=float)
        arrays[f"mt{mt}_summary"] = np.array([
            energies.size,
            energies[0], energies[-1],
            values.min(), values.max(), values.sum(),
        ])
        arrays[f"mt{mt}_digest"] = np.array(
            hashlib.sha256(energies.tobytes() + values.tobytes()).hexdigest()
        )
    check_golden("endf_reconstruct_adapter", arrays)
