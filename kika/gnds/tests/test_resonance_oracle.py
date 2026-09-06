"""Fe-56's 312 resonances, out of GNDS and out of ENDF-6, compared.

Phase 5's acceptance test for the §19 reader, and the third of the three
oracles. It is the most demanding of them, because the two encodings do not even
*group* the resonances the same way:

===============  ==========================================================
GNDS §19.3       **5 spin groups**, one per (J, parity), with **2 channels**
                 each — the eliminated capture channel and elastic — and the
                 widths located through ``channel/@columnIndex``.
ENDF LRF=3       **3 l-blocks**, one per L, with **4 channels** each, because
                 the record positions ``c3..c6`` mean ``GN, GG, GFA, GFB``
                 whether or not the nuclide fissions. Fe-56 does not, so two
                 of the four are zero throughout.
===============  ==========================================================

So nothing can be compared position by position. What is compared is the
resonances themselves — (L, energy, J, elastic width, capture width) — which is
the physics, and the two groupings are then checked to partition each other.
This is what the model's ``c3..c6`` replacement is *for*: naming a width by its
channel is what lets two files that number their columns differently be
compared at all.

**This is the second, independent check on the LRF=7 finding.** kika's ENDF path
reads Fe-56's Reich-Moore parameters through its own MF2/151 parser; FUDGE read
the same section with a completely different one. 312 resonances agreeing to
1e-12 says both parsers are right about this file.

Marked ``tape`` and ``gnds``.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document

#: Elastic and capture, spelled as each encoding spells them. kika's ENDF
#: adapter names the channels after the reaction; GNDS names them after the
#: ``resonanceReaction``, which for Fe-56 is the reaction's own GNDS label.
CHANNELS = {
    "gnds": ("n + Fe56", "Fe57 + photon [inclusive]"),
    "endf": ("elastic", "capture"),
}

#: Both paths read the same decimal text into the same doubles by different
#: routes, so the target is the last bit. Measured maximum: exact.
RTOL = 1e-12


@pytest.fixture(scope="module")
def fromGnds(fe56_gnds_tape):
    suite, report = readReactionSuite(Document.parse(fe56_gnds_tape))
    return suite, report


@pytest.fixture(scope="module")
def fromEndf(fe56_b81_tape):
    from kika.endf.model_adapter.decode import decodeReactionSuite
    from kika.endf.read_endf import read_endf

    suite, _ = decodeReactionSuite(read_endf(str(fe56_b81_tape)))
    return suite


def _resonances(formalism, elastic, capture):
    """``[(L, energy, J, elasticWidth, captureWidth), ...]``, sorted.

    Sorted by (L, energy) rather than left in file order, because the two
    encodings emit the same resonances in different orders — GNDS walks the
    spin groups, ENDF walks the l-blocks — and the *set* is what agrees.
    """
    out = []
    for group in formalism.spinGroups:
        column = {c.resonanceReaction: i for i, c in enumerate(group.channels)}
        L = group.channels[0].L
        assert all(c.L == L for c in group.channels), (
            "a spin group whose channels disagree on L cannot be keyed by it"
        )
        # An RMatrix spin group carries one J for all its resonances; an
        # ENDF l-block carries AJ per resonance, which is `spins`.
        spins = group.spins if group.spins else [group.spin] * len(group.energies)
        for energy, widths, J in zip(group.energies, group.widths, spins):
            out.append((L, energy, J,
                        widths[column[elastic]], widths[column[capture]]))
    return sorted(out, key=lambda row: (row[0], row[1]))


def test_both_encodings_hold_the_same_312_resonances(fromGnds, fromEndf):
    """The oracle. Same parameters, two parsers, two groupings, no shared code."""
    gnds, _ = fromGnds
    left = _resonances(gnds.resonances.resolved[0].formalism, *CHANNELS["gnds"])
    right = _resonances(fromEndf.resonances.resolved[0].formalism,
                        *CHANNELS["endf"])

    assert len(left) == len(right) == 312
    for a, b in zip(left, right):
        assert a[0] == b[0], f"L differs: {a} against {b}"
        np.testing.assert_allclose(a[1:], b[1:], rtol=RTOL, atol=0,
                                   err_msg=f"resonance at {a[1]} eV differs")


def test_the_two_groupings_partition_each_other(fromGnds, fromEndf):
    """5 (J, parity) groups against 3 l-blocks, and neither straddles the other.

    40 / 63 / 78 / 75 / 56 on one side and 40 / 141 / 131 on the other, with
    63 + 78 = 141 and 75 + 56 = 131. A GNDS spin group spanning two ENDF
    l-blocks would mean one of the two readers had assigned an L wrongly, and
    the resonance-by-resonance comparison above would not catch it — both sides
    would still hold the same 312 rows.
    """
    gnds, _ = fromGnds
    gndsGroups = gnds.resonances.resolved[0].formalism.spinGroups
    endfGroups = fromEndf.resonances.resolved[0].formalism.spinGroups

    assert [len(g) for g in gndsGroups] == [40, 63, 78, 75, 56]
    assert [(g.spin, g.parity) for g in gndsGroups] == [
        (0.5, 1), (0.5, -1), (1.5, -1), (1.5, 1), (2.5, 1)
    ]
    # L lives on the channels in both encodings — an `RMatrixSpinGroup` is
    # keyed by (J, parity), and the ENDF adapter's l-blocks put their L on
    # every channel of the block rather than on the group.
    def sizesByL(groups):
        sizes = {}
        for group in groups:
            L = group.channels[0].L
            sizes[L] = sizes.get(L, 0) + len(group)
        return sizes

    assert sizesByL(endfGroups) == {0: 40, 1: 141, 2: 131}
    assert sizesByL(gndsGroups) == sizesByL(endfGroups)


def test_the_two_encodings_describe_the_same_region_and_formalism(fromGnds,
                                                                  fromEndf):
    gnds, _ = fromGnds
    for suite in (gnds, fromEndf):
        region = suite.resonances.resolved[0]
        assert (region.domainMin, region.domainMax) == (1e-5, 8.5e5)
        assert region.formalism.approximation == "ReichMoore"
        assert region.formalism.reducedWidthAmplitudes is False
    # Fe-56's RRR ends at 850 keV and there is no unresolved region above it.
    assert gnds.resonances.unresolved is None
    assert fromEndf.resonances.unresolved is None


def test_the_two_paths_agree_on_the_scattering_radius(fromGnds, fromEndf):
    """**5.444 fm both ways. This test is the record of that change.**

    Its previous version asserted the opposite — ``fromGnds.constant ==
    10 * fromEndf.constant``, with the ENDF side stating no unit — and closed
    with "if a later phase makes the two agree on one canonical unit, this test
    fails and is the place to record the change". That phase is 2026-08-20 and
    this is the record.

    ENDF still states AP in units of 10^-12 cm and GNDS still states fm; what
    changed is that the **model** now states fm (``MODEL_RADIUS_UNIT``) and the
    ENDF adapter converts at the boundary. So the disagreement is gone from the
    place it mattered — a consumer reading ``constant`` — and stays where it
    belongs, in each format's own file.

    The reconstruction did not move with it: the conversion back to ENDF units
    happens at the edge of :mod:`kika.processing.resonance_formulas` rather than
    inside it, and ``test_numeric_goldens`` is the gate on that.
    """
    gnds, _ = fromGnds
    fromGndsRadius = gnds.resonances.scatteringRadius
    fromEndfRadius = fromEndf.resonances.scatteringRadius

    assert fromGndsRadius.unit == "fm"
    assert fromEndfRadius.unit == "fm", "the ENDF adapter converts and says so"
    assert fromEndfRadius.constant == pytest.approx(
        fromGndsRadius.constant, rel=1e-12
    )


def test_the_resonance_reactions_name_the_same_two_channels(fromGnds, fromEndf):
    """Different labels, same physics: one eliminated capture channel, one
    elastic. The labels are each format's own and are not made to match."""
    gnds, _ = fromGnds
    fromGndsReactions = {
        r.label: (r.ejectile, r.eliminated)
        for r in gnds.resonances.resolved[0].formalism.resonanceReactions
    }
    assert fromGndsReactions == {
        "Fe57 + photon [inclusive]": ("photon", True),
        "n + Fe56": ("n", False),
    }
    fromEndfReactions = {
        r.label: (r.ejectile, r.eliminated)
        for r in fromEndf.resonances.resolved[0].formalism.resonanceReactions
    }
    assert fromEndfReactions["capture"] == ("photon", True)
    assert fromEndfReactions["elastic"] == ("n", False)
    # Only the GNDS side carries the link back to the reaction node.
    assert all(r.href is not None
               for r in gnds.resonances.resolved[0].formalism.resonanceReactions)
    assert all(r.href is None
               for r in fromEndf.resonances.resolved[0].formalism.resonanceReactions)
