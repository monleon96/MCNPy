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
this file exists to catch. Values are compared at ``rtol`` below, tight enough
to catch any change of algorithm. Bit-exactness is the acceptance criterion for
phase 2, where the code moves without being touched; here the code is merely
frozen where it stands.

*This paragraph used to say ``rtol=1e-12`` was "loose enough to survive a
different CPU running the CI job", and to describe a sha256 digest stored
beside each array. Both were wrong, and nothing could tell us so, because CI
had not run a test since 2026-08-07 — the install step was dying on a stale
``poetry.lock``. The first run after that was fixed failed here: Fe-56 MT102
came out different on a GitHub runner, on 4 of 20 459 points, by 3.66e-12
relative. The digests are gone and the tolerance is 1e-9; see ``RTOL``. The
lesson worth keeping is not about tolerances — it is that a claim about CI in a
docstring is worth exactly as much as the last green CI run.*

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

import os
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.nuclear_data.model import (
    Nuclide,
    PhysicalQuantity,
    PoPs,
)
from kika.nuclear_data.model.resonances import (
    BreitWigner,
    BreitWignerApproximation,
    Channel,
    RMatrix,
    RMatrixSpinGroup,
    Resonance,
    ResonanceParameters,
    ResolvedRegion,
    Resonances,
    ScatteringRadius,
    SpinGroup,
    TabulatedWidths,
    UnresolvedChannel,
    UnresolvedRegion,
    UnresolvedSpinGroup,
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
#:
#: **1e-12 until 2026-08-17, and it was a tolerance this code cannot honour off
#: this machine.** Measured on a GitHub runner: the Fe-56 MT102 reconstruction
#: differs from the workstation's on 4 of 20 459 points, by 1.33e-15 absolute
#: and 3.66e-12 relative. That is libm, not arithmetic anyone wrote -- capture
#: comes out of a cancellation, so it is where the last ULP surfaces first.
#: 1e-9 keeps roughly three orders of headroom over the observed spread while
#: staying far tighter than any real change: a moved formula, grid or Q value
#: shifts these numbers by parts in 10^3, not parts in 10^9.
#:
#: The digests that used to sit beside these arrays are gone for the same
#: reason -- see the module docstring.
RTOL = 1e-9


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


def _target() -> PoPs:
    """The local ``PoPs`` a resonance formalism carries: the target and its spin."""
    pops = PoPs(name="resolved resonances")
    pops.add(Nuclide(id="Fe56", Z=26, A=56,
                     spin=PhysicalQuantity(_SPIN, "hbar")))
    return pops


def _resonances(formalism) -> Resonances:
    return Resonances(
        scatteringRadius=ScatteringRadius(constant=_AP),
        resolved=[ResolvedRegion(domainMin=_RANGE[0], domainMax=_RANGE[1],
                                 formalism=formalism)],
    )


def _synthetic(formalism: str, fission: bool = False) -> Resonances:
    if formalism in ("SLBW", "MLBW"):
        # Breit-Wigner names its widths GT, GN, GG, GF.
        s_wave = [
            Resonance(energy=1.15e3, spin=0.5, totalWidth=1.40,
                      neutronWidth=1.02, captureWidth=0.38, fissionWidth=0.0),
            Resonance(energy=2.28e4, spin=0.5, totalWidth=2.10,
                      neutronWidth=1.55, captureWidth=0.55, fissionWidth=0.0),
        ]
        p_wave = [
            Resonance(energy=7.40e4, spin=1.5, totalWidth=0.92,
                      neutronWidth=0.61, captureWidth=0.31, fissionWidth=0.0),
        ]
        approximation = (BreitWignerApproximation.singleLevel if formalism == "SLBW"
                         else BreitWignerApproximation.multiLevel)
        return _resonances(BreitWigner(
            approximation=approximation,
            resonanceParameters=ResonanceParameters(spinGroups=[
                SpinGroup(L=0, resonances=s_wave, atomicWeightRatio=_AWRI),
                SpinGroup(L=1, resonances=p_wave, atomicWeightRatio=_AWRI),
            ]),
            scatteringRadius=_AP,
            PoPs=_target(),
        ))

    # Reich-Moore: widths belong to channels, and there are two fission ones.
    # They are split unevenly and given opposite signs on one level: GFA and
    # GFB enter as reduced-width *amplitudes*, so a sign is physics and not
    # bookkeeping, and a lookup that swapped the two channels would be
    # invisible if they were equal.
    gfa = (0.21, 0.34, 0.12) if fission else (0.0, 0.0, 0.0)
    gfb = (-0.09, 0.17, 0.05) if fission else (0.0, 0.0, 0.0)

    def _group(L, energies, spins, widths):
        return RMatrixSpinGroup(
            label=f"L{L}",
            channels=[
                Channel(label=label, resonanceReaction=reaction, L=L, columnIndex=index)
                for index, (label, reaction) in enumerate([
                    ("neutron", "elastic"), ("capture", "capture"),
                    ("fissionA", "fission"), ("fissionB", "fission"),
                ])
            ],
            energies=list(energies), spins=list(spins),
            widths=[list(row) for row in widths],
            atomicWeightRatio=_AWRI,
        )

    return _resonances(RMatrix(
        approximation="ReichMoore",
        spinGroups=[
            _group(0, [1.15e3, 2.28e4], [0.5, 0.5],
                   [[1.02, 0.38, gfa[0], gfb[0]], [1.55, 0.55, gfa[1], gfb[1]]]),
            _group(1, [7.40e4], [1.5], [[0.61, 0.31, gfa[2], gfb[2]]]),
        ],
        scatteringRadius=_AP,
        PoPs=_target(),
    ))


def _as_arrays(reconstructed: dict) -> dict[str, np.ndarray]:
    """Flatten ``{MT: XYs1d}`` into a saveable dict of arrays."""
    out: dict[str, np.ndarray] = {}
    first = next(iter(sorted(reconstructed)))
    out["energies"] = np.asarray(reconstructed[first].xs, dtype=float)
    for mt in sorted(reconstructed):
        form = reconstructed[mt]
        np.testing.assert_array_equal(
            np.asarray(form.xs), out["energies"],
            err_msg=f"MT{mt} is on a different grid from MT{first}",
        )
        out[f"mt{mt}"] = np.asarray(form.ys, dtype=float)
    return out


@pytest.mark.parametrize("formalism", ["SLBW", "MLBW", "RM"])
def test_reconstruction_golden(formalism):
    """Pointwise sigma(E) from three resonances, per formalism."""
    produced = reconstruct(_synthetic(formalism), tolerance=1e-3,
                           atomicWeightRatio=_AWRI)
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
    produced = reconstruct(_synthetic("RM", fission=True), tolerance=1e-3,
                           atomicWeightRatio=_AWRI)
    assert 18 in produced, "the fission case produced no MT18"
    check_golden("reconstruct_rm_fission", _as_arrays(produced))


#: ``(name, energy grid or None, level spacing, widths)`` for the two URR
#: shapes the corpus actually has. Case A is energy-independent — every average
#: is a scalar and ``_interpolate_urr_params`` returns its input untouched;
#: case C tabulates every average against an energy grid and goes through
#: ``np.interp``. Case B exists in ENDF and in no tape on this machine.
_URR_GRID = np.array([1.0e5, 3.0e5, 8.5e5], dtype=float)


def _synthetic_urr(case: str) -> Resonances:
    def _group(L, J, spacing, gn0, gg):
        if case == "A":
            # Case A is energy-independent, and the decoder builds exactly two
            # channels for it — no fission, no competitive. Their widths read
            # as zero, which is what the scalar 0.0 meant before.
            channels = [
                UnresolvedChannel(label="neutron", degreesOfFreedom=1.0,
                                  constantWidth=gn0),
                UnresolvedChannel(label="capture", degreesOfFreedom=1.0,
                                  constantWidth=gg),
            ]
            levelSpacing = np.asarray([spacing], dtype=float)
        else:
            channels = [
                UnresolvedChannel(label="neutron", degreesOfFreedom=1.0,
                                  widths=np.asarray(gn0, dtype=float)),
                UnresolvedChannel(label="capture", degreesOfFreedom=1.0,
                                  widths=np.asarray(gg, dtype=float)),
                UnresolvedChannel(label="fission", degreesOfFreedom=1.0,
                                  widths=np.zeros(len(_URR_GRID))),
                UnresolvedChannel(label="competitive", degreesOfFreedom=1.0,
                                  widths=np.zeros(len(_URR_GRID))),
            ]
            levelSpacing = np.asarray(spacing, dtype=float)
        return UnresolvedSpinGroup(L=L, J=J, levelSpacing=levelSpacing,
                                   channels=channels, atomicWeightRatio=_AWRI)

    if case == "A":
        groups = [
            _group(0, 0.5, 7.2e3, 1.9e-2, 0.62),
            _group(1, 0.5, 4.1e3, 8.0e-3, 0.55),
            _group(1, 1.5, 2.6e3, 1.1e-2, 0.58),
        ]
        grid = None
    else:
        groups = [
            _group(0, 0.5, [7.2e3, 6.9e3, 6.4e3], [1.9e-2, 1.8e-2, 1.7e-2],
                   [0.62, 0.63, 0.65]),
            _group(1, 0.5, [4.1e3, 3.9e3, 3.6e3], [8.0e-3, 7.7e-3, 7.1e-3],
                   [0.55, 0.56, 0.58]),
            _group(1, 1.5, [2.6e3, 2.5e3, 2.3e3], [1.1e-2, 1.0e-2, 9.4e-3],
                   [0.58, 0.59, 0.61]),
        ]
        grid = _URR_GRID

    return Resonances(
        scatteringRadius=ScatteringRadius(constant=_AP),
        unresolved=UnresolvedRegion(
            domainMin=1.0e5, domainMax=8.5e5,
            tabulatedWidths=TabulatedWidths(
                spinGroups=groups, energyGrid=grid, scatteringRadius=_AP,
                selfShieldingOnly=False, PoPs=_target(),
            ),
        ),
    )


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
    produced = reconstruct(_synthetic_urr(case), tolerance=1e-3,
                           atomicWeightRatio=_AWRI)
    assert produced, f"URR case {case}: reconstruct returned nothing"
    check_golden(f"reconstruct_urr_case_{case.lower()}", _as_arrays(produced))


@pytest.mark.slow
def test_reconstruction_golden_real_fe56(micro_tape):
    """The real thing: 320 Reich-Moore resonances from the committed slice.

    The synthetic cases above exercise the formulas; this one exercises the
    whole path, including the adaptive grid seeding around 320 real levels.
    """
    from kika.endf.model_adapter import decodeMF2MT151

    endf = read_endf(str(micro_tape))
    resonances, provenance, _ = decodeMF2MT151(endf.files[2].sections[151])
    assert len(resonances.resolved) == 1
    assert isinstance(resonances.resolved[0].formalism, RMatrix)

    produced = reconstruct(resonances, tolerance=1e-2,
                           atomicWeightRatio=provenance.awr)
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

    # Summarised rather than stored in full. The output runs to hundreds of
    # thousands of points per MT, and committing 3 MB to pin it is not a good
    # trade. Six statistics per MT say that it moved and give the shape of the
    # move; see the module docstring for the digest that used to sit beside
    # them and why it is gone.
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
    check_golden("endf_reconstruct_adapter", arrays)
