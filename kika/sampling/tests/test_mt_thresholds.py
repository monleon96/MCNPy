"""The threshold map, frozen before the σ-source shape changes.

``derive_mt_thresholds`` is the one place where the *shape* of a PENDF MF3
section reaches **sampled numbers**. It reads ``.energies`` and
``.cross_sections`` — ENDF's spelling — off whatever
``read_pendf_mf3_sections`` returned, and hands the answer to
``draw_relative_factors`` as ``mt_thresholds``, which pins the bin containing
each reaction's threshold. A wrong map perturbs a cross section where it is
still zero.

The GNDS roadmap's phase 4 **P2** changes what that function returns, and the
canonical ``CrossSection`` spells the second array ``.values``. So this file
exists to make P2 a measurement rather than a hope: it freezes the map on a
committed tape, and the values below are read off that tape rather than
reasoned about.

**No NJOY here, and no PENDF.** A PENDF's MF3 is parsed by the same
``read_endf`` as any other tape, so a committed ENDF fixture exercises the same
code with none of the cost. What that does *not* cover is a threshold whose
first positive point only exists after reconstruction; that is the end-to-end
test's business, not this one's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.pendf_perturbation import derive_mt_thresholds

#: Committed slice carrying MF3 and MF33 for MT4 and MT16 — the same tape
#: ``test_mf33_migration_equivalence.py`` gates the draw on, located the same way.
TAPE = (Path(__file__).resolve().parents[2]
        / "endf" / "tests" / "data" / "micro_fe56_mf33.endf")

#: Measured on that tape, 2026-08-12. Not a design target — a record of what the
#: file says, to six figures, so a shift of one grid point fails rather than
#: rounds away.
#:
#: Both sit just above their reaction's kinematic threshold
#: ``|QI| (A+1)/A``, which is the check that these are physics and not an
#: artefact of where the slice was cut: MT4 (inelastic, QI = -846 778 eV) gives
#: 862 047 against 862 892 tabulated, and MT16 ((n,2n), QI = -11 197 000 eV)
#: gives 11 398 900 against 11 500 000.
EXPECTED = {4: 862891.9, 16: 11500000.0}


@pytest.fixture(scope="module")
def mf3_sections():
    endf = read_endf(str(TAPE))
    return {int(mt): section for mt, section in endf.mf[3].mt.items()}


def test_the_threshold_map_is_what_it_was(mf3_sections):
    """The frozen values. This is the assertion P2 has to keep green."""
    thresholds = derive_mt_thresholds(mf3_sections, [4, 16])
    assert set(thresholds) == set(EXPECTED)
    for mt, expected in EXPECTED.items():
        assert thresholds[mt] == pytest.approx(expected, rel=1e-6), (
            f"MT{mt} threshold moved: {thresholds[mt]!r}"
        )


def test_it_is_the_first_positive_point_and_not_the_first_point(mf3_sections):
    """Otherwise the map would be the section's domain minimum, which is free.

    MT16's section starts below its threshold with zeros, so "first energy" and
    "first energy with sigma > 0" are different numbers — which is the whole
    reason the function looks at the cross sections at all.
    """
    section = mf3_sections[16]
    energies = np.asarray(section.energies, dtype=float)
    values = np.asarray(section.cross_sections, dtype=float)

    assert values[0] == 0.0, "fixture no longer exercises the leading-zero case"
    assert derive_mt_thresholds(mf3_sections, [16])[16] > float(energies[0])


def test_an_absent_mt_is_absent_rather_than_zero(mf3_sections):
    """A missing threshold must mean "do not pin", not "pin at 0 eV"."""
    assert 102 not in derive_mt_thresholds(mf3_sections, [4, 16, 102])


def test_an_all_zero_section_yields_no_threshold():
    """A reaction that is nowhere open has no threshold to pin."""

    class _Flat:
        energies = np.array([1.0, 2.0, 3.0])
        cross_sections = np.zeros(3)

    assert derive_mt_thresholds({7: _Flat()}, [7]) == {}


def test_the_map_reads_the_endf_spelling_and_would_notice_the_other_one():
    """Why P2 has to run this file.

    ``.cross_sections`` is ``MF3MT``'s name for the array. The canonical
    ``CrossSection`` calls it ``.values`` and carries no ``.cross_sections`` at
    all, so a return-type change that is invisible to every type checker here
    raises ``AttributeError`` at the point where the draw is configured.
    Asserting the failure is what makes the change *have* to be deliberate.
    """

    class _CanonicalShaped:
        energies = np.array([1.0, 2.0, 3.0])
        values = np.array([0.0, 1.0, 2.0])

    with pytest.raises(AttributeError):
        derive_mt_thresholds({7: _CanonicalShaped()}, [7])
