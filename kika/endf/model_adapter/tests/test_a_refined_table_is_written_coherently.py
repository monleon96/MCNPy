"""The encoders survive a model that was edited after it was decoded.

**What this is for.** Every other gate in this directory reads a tape, builds
the model and writes it straight back, so the model handed to an encoder always
has exactly the shape the file stated. That is the round trip, and it is not
the case the perturbation work creates: an applier *refines the grid* — the MF33
factor application inserts a duplicate abscissa at every covariance bin edge, by
construction, because that is what makes the realised factor equal the drawn one
— and then asks for a section.

``NBT`` is cumulative and one-based, so the last ``(NBT, INT)`` pair is a
statement about how many points the table has. ``encodeMF3MT`` and the nu-bar
encoder both *prefer the file's own pairs* over the ones rebuilt from the
``regions1d``, for a good reason (``encode.py`` says it: a round trip should not
depend on the reconstruction staying byte-faithful for every tape ever written)
that stops holding the moment the table changes length. Before this gate, MT2 of
the committed Fe-56 micro-tape came back as ``NP=15049`` beside
``NBT=[(15048, 2)]`` — a TAB1 whose final interpolation region does not reach the
end of its own table, written without an exception, a warning or a schema
complaint.

**MF4 and MF5 were checked and do not have it**: ``encodeMF4MT`` and
``encodeMF5MT`` rebuild their pairs from the object through ``toEndfTab2`` /
``toEndfRegions``. The trap was MF3 and MF1/nu-bar, and only those two.

``docs/library/library-gaps.md`` D28, ``perturbation_model_roadmap.md`` M0.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import (decodeReactionSuite, encodeMF1MT452,
                                     encodeMF3MT)
from kika.endf.model_adapter.encode import usableInterpolationRegions
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import EVAL_LABEL, ConversionReport


# ---------------------------------------------------------------------------
# The predicate, on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kept, npoints, expected", [
    ([(10, 2)], 10, [(10, 2)]),                       # the ordinary case
    ([(4, 1), (10, 2)], 10, [(4, 1), (10, 2)]),       # several regions
    ([(10, 2)], 11, None),                            # a point was inserted
    ([(10, 2)], 9, None),                             # a point was dropped
    ([], 10, None),                                   # nothing was kept
    (None, 10, None),                                 # no provenance at all
    ([(10, 2), (4, 1)], 10, None),                    # NBT must ascend
    ([(0, 2), (10, 2)], 10, None),                    # and start above zero
])
def test_the_kept_pairs_are_used_only_while_they_describe_the_table(
        kept, npoints, expected):
    assert usableInterpolationRegions(kept, npoints) == expected


# ---------------------------------------------------------------------------
# MF3
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoded(micro_tape):
    endf = read_endf(str(micro_tape))
    suite, _ = decodeReactionSuite(endf)
    return suite


def _refine(form, at: int = 5):
    """Insert one duplicate abscissa, the way a factor application does."""
    region = form.function1ds[0]
    region.xs = np.insert(region.xs, at, region.xs[at])
    region.ys = np.insert(region.ys, at, region.ys[at] * 1.2)


def test_an_unedited_reaction_still_writes_the_file_s_own_pairs(decoded):
    """The preference is kept, not removed. This is the half that must not move.

    If the fix had been "always rebuild", every MF3 section on every tape would
    now depend on the reconstruction being byte-faithful — which is the thing
    the preference exists to avoid. So the ordinary path is asserted first.
    """
    for mt in (2, 102):
        reaction = decoded.reactionByENDF_MT(mt)
        section, report = encodeMF3MT(reaction)
        assert section._interpolation == [
            (int(a), int(b)) for a, b in reaction.provenance.interpolationRegions
        ]
        assert report.isEmpty, report.summary()


def test_a_refined_cross_section_gets_pairs_that_cover_it(decoded):
    """The defect, and the whole point of the module."""
    reaction = decoded.reactionByENDF_MT(2)
    form = reaction.crossSection[EVAL_LABEL]
    before = form.toEndfRegions()[0].size

    _refine(form)

    report = ConversionReport()
    section, report = encodeMF3MT(reaction, report=report)

    assert section._np == before + 1
    assert section._interpolation[-1][0] == section._np, (
        "the last NBT does not reach the end of the table: the section was "
        "written with the pairs of a table that no longer exists"
    )
    assert section._nr == len(section._interpolation)
    assert len(section._energies) == section._np
    assert len(section._cross_sections) == section._np

    # And it says so. A silent rebuild would be a second thing to discover.
    # `isClean` deliberately ignores warnings -- nothing was *lost* here, the
    # rebuild is exact -- so the assertion is on the warning itself.
    assert report.isClean and not report.isEmpty, report.summary()
    assert any("interpolation regions" in message
               for message in report.warnings), report.summary()


def test_the_written_section_is_a_valid_tab1_after_refinement(decoded):
    """Rendering it is the check the field widths cannot fake.

    ``_np`` and the pair list agreeing in memory is necessary; what a consumer
    reads is the text, so the section is rendered and read back.
    """
    from kika.endf.parsers.parse_mf3 import MF3MT as _  # noqa: F401  (import guard)

    reaction = decoded.reactionByENDF_MT(102)
    form = reaction.crossSection[EVAL_LABEL]
    _refine(form, at=3)

    section, _ = encodeMF3MT(reaction)
    text = str(section)
    assert text, "the refined section rendered as nothing"
    # The TAB1 CONT record carries NR then NP in fields 5 and 6.
    header = text.splitlines()[1]
    nr = int(header[44:55])
    np_ = int(header[55:66])
    assert (nr, np_) == (section._nr, section._np)
    assert section._interpolation[-1][0] == np_


# ---------------------------------------------------------------------------
# MF1 nu-bar — the same trap, the other encoder
# ---------------------------------------------------------------------------

def test_a_refined_nubar_gets_pairs_that_cover_it(micro_nubar_tape):
    endf = read_endf(str(micro_nubar_tape))
    suite, _ = decodeReactionSuite(endf)

    from kika.endf.model_adapter.multiplicity import nubarNode

    multiplicity = nubarNode(suite, 452)
    assert multiplicity is not None, "the nu-bar fixture carries no MT452"

    section, report = encodeMF1MT452(suite)
    assert section._interpolation[-1][0] == section._np, "unedited, and already wrong"
    before = section._np

    region = multiplicity.form.function1ds[0]
    region.xs = np.insert(region.xs, 2, region.xs[2])
    region.ys = np.insert(region.ys, 2, region.ys[2] * 1.01)

    report = ConversionReport()
    section, report = encodeMF1MT452(suite, report=report)

    assert section._np == before + 1
    assert section._interpolation[-1][0] == section._np
    assert section._nr == len(section._interpolation)
    assert any("interpolation regions" in message
               for message in report.warnings), report.summary()
