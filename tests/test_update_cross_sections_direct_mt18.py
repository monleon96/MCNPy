"""Regression tests for ACE.update_cross_sections() redundant-reaction rebuild.

Guards the fix for the reported bug: when an ACE file stores fission directly as
MT=18 (without the partial sub-reactions MT 19/20/21/38), the redundant-total
rebuild must NOT overwrite the (already-perturbed) MT=18 with zeros. The same
guard protects every composite (MT=3/4/101/103-107 and MT=1).

The repo ships no ``.ace`` fixtures, so these tests build lightweight stubs that
mirror the exact attribute interface ``update_cross_sections`` touches:
``ace.cross_section.reaction`` is a dict of reaction objects, each exposing
``energies``, ``xs_values`` (live view of the entries), and ``_xs_entries``
whose elements carry a mutable ``value``.
"""

import types

from kika.ace.classes.ace import Ace


class _Entry:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = float(value)


class _Reaction:
    """Minimal stand-in for ReactionCrossSection."""

    def __init__(self, energies, values):
        self.energies = list(energies)
        self._xs_entries = [_Entry(v) for v in values]

    @property
    def xs_values(self):
        # Live view: reflects whatever update_cross_sections wrote into entries.
        return [e.value for e in self._xs_entries]


def _make_ace(reactions):
    """Build a bare ACE-like object exposing only ``cross_section.reaction``."""
    ace = Ace.__new__(Ace)  # skip the dataclass __init__; we only need cross_section
    cross_section = types.SimpleNamespace(reaction=reactions)
    object.__setattr__(ace, "cross_section", cross_section)
    return ace


# Shared energy grid for all cases.
E = [1.0, 2.0, 3.0]


def test_direct_mt18_not_zeroed():
    """MT=18 stored directly (no partials) must survive update unchanged."""
    reactions = {
        2: _Reaction(E, [10.0, 9.0, 8.0]),   # elastic
        18: _Reaction(E, [4.0, 5.0, 6.0]),   # fission stored directly (perturbed)
        1: _Reaction(E, [0.0, 0.0, 0.0]),    # total, to be rebuilt
    }
    ace = _make_ace(reactions)
    ace.update_cross_sections()

    # MT=18 preserved, not overwritten with zeros.
    assert reactions[18].xs_values == [4.0, 5.0, 6.0]
    # MT=1 rebuilt and includes the fission term: 2 + 18.
    assert reactions[1].xs_values == [14.0, 14.0, 14.0]


def test_split_fission_reconstructed():
    """When MT 19/20/21/38 are present, MT=18 is the sum of the partials."""
    reactions = {
        2: _Reaction(E, [10.0, 9.0, 8.0]),
        19: _Reaction(E, [1.0, 1.0, 1.0]),
        20: _Reaction(E, [2.0, 2.0, 2.0]),
        21: _Reaction(E, [0.5, 0.5, 0.5]),
        38: _Reaction(E, [0.5, 0.5, 0.5]),
        18: _Reaction(E, [0.0, 0.0, 0.0]),   # stale; must be rebuilt from partials
        1: _Reaction(E, [0.0, 0.0, 0.0]),
    }
    ace = _make_ace(reactions)
    ace.update_cross_sections()

    # MT=18 == sum of the four chance-fission partials.
    assert reactions[18].xs_values == [4.0, 4.0, 4.0]
    # MT=1 == elastic + fission.
    assert reactions[1].xs_values == [14.0, 13.0, 12.0]


def test_mt3_cascade_includes_direct_mt18():
    """MT=3 (non-elastic) must still include MT=18 when fission is stored direct."""
    reactions = {
        2: _Reaction(E, [10.0, 9.0, 8.0]),
        18: _Reaction(E, [4.0, 5.0, 6.0]),   # direct fission, no partials
        3: _Reaction(E, [0.0, 0.0, 0.0]),    # non-elastic, rebuilt; feeds on 18
        1: _Reaction(E, [0.0, 0.0, 0.0]),
    }
    ace = _make_ace(reactions)
    ace.update_cross_sections()

    # MT=18 untouched.
    assert reactions[18].xs_values == [4.0, 5.0, 6.0]
    # MT=3 picks up the (non-zeroed) fission contribution.
    assert reactions[3].xs_values == [4.0, 5.0, 6.0]
    # MT=1 = elastic + non-elastic.
    assert reactions[1].xs_values == [14.0, 14.0, 14.0]
