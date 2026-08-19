"""Fe-56's cross sections, out of GNDS and out of ENDF-6, compared.

Phase 5's acceptance test for the ``reactionSuite`` reader, and the counterpart
of :mod:`kika.gnds.tests.test_covariance_oracle`: the same evaluation in two
encodings, read by two paths that share no parser and no unpacking, and they
agree or they do not.

The same caveat applies. ``n-026_Fe_056.endf.gnds.xml`` is ENDF/B-VIII.1
*translated to GNDS by FUDGE*, so agreement here is agreement with FUDGE's
translation rather than with the standard, and a disagreement could in principle
be FUDGE's. It is still the strongest check available, and it is a demanding
one: 77 shared MTs, the three resonance-region ones arriving from GNDS in two
background pieces and from ENDF as one table.

**The one systematic difference, and why it is not a defect.** ENDF's convention
for a discontinuity is a repeated abscissa, and it writes some of them
redundantly — MT4 carries the energy 862 047.8 eV four times with sigma = 0 at
each. FUDGE's translation drops a point that repeats both its energy *and* its
value, because such a point changes no interpolation anywhere. Collapsing those
runs on the ENDF side is therefore part of the comparison, not a fudge of it,
and :func:`_collapseRepeatedPoints` is deliberately strict: it removes a point
only when the pair before it is identical in both coordinates.

Marked ``tape`` and ``gnds``: it needs the 18.8 MB GNDS file and the 40 MB ENDF
tape from the shared tree, and skips honestly without them.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import (Reference, Regions1d,
                                     ResonancesWithBackground, XYs1d)

#: How closely the two paths must agree. Both read the same decimal text into
#: the same IEEE doubles by different routes, so the honest target is the last
#: bit, not a physics tolerance. Measured maximum across all 77 MTs: 2.8e-16.
RTOL = 1e-14


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


def _pointwise(form):
    """``(E, sigma)`` for a form that tabulates one, else ``None``.

    A ``Regions1d`` is flattened by ``toEndfRegions``, which is what the ENDF
    side stores it as anyway, so both encodings are compared on the same layout.
    """
    if isinstance(form, XYs1d):
        return form.xs, form.ys
    if isinstance(form, Regions1d):
        xs, ys, _ = form.toEndfRegions()
        return xs, ys
    return None


def _gndsPointwise(form):
    """The GNDS side, with a ``resonancesWithBackground`` joined up.

    §16.1.1 states the background over the resolved region and above it as two
    functions. ENDF's MF3 is one table covering both, and the ENDF file's
    resonance-region entries *are* the background — an evaluation with LRU=1
    adds MF3 to the reconstructed resonances. So concatenating the regions in
    domain order is what makes the two encodings comparable, and it is exactly
    what phase 5's model was reshaped to allow: with the old single-function
    background one of the two pieces would simply not be here.
    """
    if isinstance(form, ResonancesWithBackground):
        pieces = [_pointwise(region)
                  for region in form.background.regions.values()
                  if region is not None]
        assert all(piece is not None for piece in pieces)
        return (np.concatenate([piece[0] for piece in pieces]),
                np.concatenate([piece[1] for piece in pieces]))
    return _pointwise(form)


def _collapseRepeatedPoints(xs, ys):
    """Drop each point that repeats the previous one in *both* coordinates.

    Strict on purpose. A repeated energy with a *different* sigma is ENDF's
    discontinuity, which carries information and must survive; only the fully
    redundant repeat goes.
    """
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = ~((xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1]))
    return xs[keep], ys[keep]


def _byMT(suite):
    out = {}
    for container in (suite.reactions, suite.sums, suite.productions,
                      suite.fissionComponents, suite.incompleteReactions):
        for reaction in container:
            if reaction.ENDF_MT is not None:
                out.setdefault(reaction.ENDF_MT, reaction)
    return out


def test_the_two_encodings_hold_the_same_reactions(fromGnds, fromEndf):
    """Same MTs, one addition, and the addition is a sum rather than a reaction."""
    gnds, _ = fromGnds
    left, right = _byMT(gnds), _byMT(fromEndf)
    assert set(right) <= set(left)
    # MT103 is (n,p) stated as the sum of MT600-649. ENDF/B-VIII.1's Fe-56 has
    # no MF3/MT103 section, so only the GNDS side carries it — and it is in
    # `sums`, not in `reactions`, which is the distinction ENDF cannot make.
    assert sorted(set(left) - set(right)) == [103]
    assert gnds.sums.ENDF_MTs == [1, 4, 103]
    assert 103 not in gnds.reactions


def test_every_shared_cross_section_agrees_to_the_last_bit(fromGnds, fromEndf):
    """The oracle. 77 MTs, two readers, no shared code."""
    gnds, _ = fromGnds
    left, right = _byMT(gnds), _byMT(fromEndf)

    compared = 0
    for mt in sorted(set(left) & set(right)):
        gndsForm = left[mt].crossSection.forms.get("eval")
        endfForm = right[mt].crossSection.forms.get("eval")
        if isinstance(gndsForm, Reference):
            continue           # a link, not a table; nothing to compare here
        gndsPair = _gndsPointwise(gndsForm)
        endfPair = _pointwise(endfForm)
        assert gndsPair is not None, f"MT{mt}: GNDS gave a {type(gndsForm).__name__}"
        assert endfPair is not None, f"MT{mt}: ENDF gave a {type(endfForm).__name__}"

        endfEnergies, endfValues = _collapseRepeatedPoints(*endfPair)
        assert gndsPair[0].shape == endfEnergies.shape, (
            f"MT{mt}: {gndsPair[0].size} points from GNDS against "
            f"{endfEnergies.size} from ENDF once redundant repeats are dropped"
        )
        np.testing.assert_allclose(
            gndsPair[0], endfEnergies, rtol=RTOL, atol=0,
            err_msg=f"MT{mt}: the energy grids differ",
        )
        np.testing.assert_allclose(
            gndsPair[1], endfValues, rtol=RTOL, atol=0,
            err_msg=f"MT{mt}: the cross sections differ",
        )
        compared += 1
    assert compared == 77


def test_the_redundant_repeats_are_real_and_only_in_the_endf_encoding(fromGnds,
                                                                     fromEndf):
    """Name the difference rather than leave it inside a helper.

    Four of Fe-56's MTs carry fully redundant repeated points in ENDF and none
    do in GNDS. If FUDGE ever stopped dropping them — or if kika's ENDF reader
    started to — this test fails and says so, instead of the comparison above
    quietly succeeding for a different reason.
    """
    gnds, _ = fromGnds
    left, right = _byMT(gnds), _byMT(fromEndf)

    withRepeats = {}
    for mt in sorted(set(left) & set(right)):
        endfPair = _pointwise(right[mt].crossSection.forms.get("eval"))
        if endfPair is None:
            continue
        dropped = endfPair[0].size - _collapseRepeatedPoints(*endfPair)[0].size
        if dropped:
            withRepeats[mt] = dropped
    assert withRepeats == {1: 1, 2: 1, 4: 3, 102: 1}

    for mt in withRepeats:
        gndsPair = _gndsPointwise(left[mt].crossSection.forms.get("eval"))
        assert gndsPair[0].size == _collapseRepeatedPoints(*gndsPair)[0].size, (
            f"MT{mt}: the GNDS side now carries redundant repeats too"
        )


def test_the_resonance_backgrounds_join_at_the_energy_endf_switches_at(fromGnds):
    """The two background regions meet at 850 keV — the top of Fe-56's RRR.

    A reader that kept only one region would still produce a curve, and it would
    be a cross section missing everything above (or below) this energy. The
    joint is asserted rather than assumed because that failure looks like data.
    """
    gnds, _ = fromGnds
    for mt in (1, 2, 102):
        background = gnds.reactionByENDF_MT(mt).crossSection["eval"].background
        assert background.unresolvedRegion is None      # Fe-56 has no URR
        assert background.resolvedRegion.xs[-1] == 850000.0
        assert background.fastRegion.xs[0] == 850000.0


def test_the_full_evaluation_reads_with_a_report_that_names_every_gap(fromGnds):
    """The whole 18.8 MB file, not the trim, and what it could not read."""
    gnds, report = fromGnds
    assert len(gnds.reactions) == 75
    assert len(gnds.PoPs) == 174
    assert not report.approximations, report.approximations
    # Every unsupported entry is an xPath into this document, so "what was
    # skipped" is answerable without reading the reader.
    assert all(entry.startswith("/reactionSuite") for entry in report.unsupported)
    laws = {entry.rsplit("/", 1)[-1].split(":")[0] for entry in report.unsupported}
    # Two names left this set in phase 7b and the set is the record of it:
    # `uncorrelated` with increment 1, `energyAngular` with the XYs3d one. What
    # remains is the two branching nodes, which are §14 isomeric transitions
    # rather than §18 laws. This assert is the gate those increments are
    # measured by — the 18.8 MB evaluation, not a trim of it — so it is edited
    # when a law lands and never loosened to make a run pass.
    assert laws == {"branching1d", "branching3d"}
    # §19 is read, so it is not in that list; the resonance oracle checks it.
    assert gnds.hasResonances
