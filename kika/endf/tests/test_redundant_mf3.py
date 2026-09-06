"""Tests for rebuilding the MF3 summation cross sections.

Two levels, as elsewhere in this package:

* **Synthetic tapes** built from :class:`MF3MT` sections whose grids and values
  are chosen for the property under test -- a partial on a coarser grid than
  the total, a step discontinuity written as a repeated energy, a HEAD record
  carrying a field the dataclass does not model.

* **Real evaluations**, where the assertion that matters is a negative one:
  resumming a tape nobody edited must give the tape back byte for byte.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika._constants import MF3_SUM_ORDER, MF3_SUM_RULES
from kika.endf.classes.mf3.mf3mt import MF3MT
from kika.endf.utils import (
    format_endf_fend_record,
    format_endf_mend_record,
    parse_endf_id,
)
from kika.endf.writers.endf_writer import ENDFWriter
from kika.endf.writers.redundant import (
    _parse_baseline,
    recompute_redundant_mf3,
    resolve_sum_components,
)

MAT = 2631


# ---------------------------------------------------------------------------
# Synthetic tape construction
# ---------------------------------------------------------------------------

def _section(mt, energies, xs, *, qi=0.0, head_l2=0):
    """One MF3 section as tape lines (HEAD + TAB1 + SEND)."""
    sec = MF3MT(number=mt)
    sec._za, sec._awr, sec._mat = 26056.0, 55.454, MAT
    sec._qm, sec._qi, sec._lr = qi, qi, 0
    sec._energies = [float(e) for e in energies]
    sec._cross_sections = [float(v) for v in xs]
    sec._nr, sec._np = 1, len(energies)
    sec._interpolation = [(len(energies), 2)]
    lines = str(sec).split("\n")
    if head_l2:
        # Columns 34-44 of the HEAD record, which MF3MT does not model.
        lines[0] = lines[0][:33] + f"{head_l2:>11d}" + lines[0][44:]
    return lines


def _tape(*sections):
    lines = [line for section in sections for line in section]
    lines += [format_endf_fend_record(MAT), format_endf_mend_record()]
    return "\n".join(lines) + "\n"


def _values(content, mt):
    """(energies, cross sections) of one MF3 section of a tape."""
    section = _parse_baseline(content)[mt]
    return (np.asarray(section.energies, dtype=float),
            np.asarray(section.cross_sections, dtype=float))


# ---------------------------------------------------------------------------
# The sum rules
# ---------------------------------------------------------------------------

def test_every_at_reference_points_at_a_rule_that_comes_earlier():
    """The order MF3_SUM_ORDER declares has to be a real topological one."""
    for position, mt in enumerate(MF3_SUM_ORDER):
        for entry in MF3_SUM_RULES[mt][0]:
            if isinstance(entry, str):
                ref = int(entry.lstrip("@"))
                assert MF3_SUM_ORDER.index(ref) < position, (
                    f"MT{mt} references MT{ref}, which is summed after it")


def test_every_rule_key_is_in_the_order():
    assert set(MF3_SUM_RULES) == set(MF3_SUM_ORDER)


def test_a_redundant_the_file_gives_is_used_instead_of_its_own_partials():
    """MT4 present means MT1 takes MT4, not MT51+MT52 -- or it double-counts."""
    assert resolve_sum_components(1, {1, 2, 4, 51, 52}) == (2, 4)


def test_a_missing_redundant_falls_back_to_its_partials():
    """A file giving MT600-649 but no MT103 still owes (n,p) to MT101."""
    assert resolve_sum_components(101, {101, 102, 600, 601}) == (102, 600, 601)
    assert resolve_sum_components(101, {101, 102, 103, 600}) == (102, 103)


def test_mt50_is_not_summed_into_the_inelastic_total():
    """(n,n0) is elastic under another name; ENDF-6 sums MT51-91."""
    assert resolve_sum_components(4, {4, 50, 51, 52}) == (51, 52)


def test_a_redundant_with_no_partials_resolves_to_nothing():
    assert resolve_sum_components(1, {1}) == ()
    assert resolve_sum_components(1, {1, 3}) == (3,)


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------

def test_changing_a_level_rebuilds_the_inelastic_total_and_then_the_total():
    """MT52 moves, so MT4 moves, and MT1 moves because MT4 did."""
    grid = [1.0e5, 1.0e6, 2.0e7]
    before = _tape(
        _section(1, grid, [10.0, 8.0, 6.0]),
        _section(2, grid, [7.0, 5.0, 3.0]),
        _section(4, grid, [3.0, 3.0, 3.0]),
        _section(51, grid, [2.0, 2.0, 2.0]),
        _section(52, grid, [1.0, 1.0, 1.0]),
    )
    after = _tape(
        _section(1, grid, [10.0, 8.0, 6.0]),
        _section(2, grid, [7.0, 5.0, 3.0]),
        _section(4, grid, [3.0, 3.0, 3.0]),
        _section(51, grid, [2.0, 2.0, 2.0]),
        _section(52, grid, [4.0, 4.0, 4.0]),      # the edit
    )

    out, updates = recompute_redundant_mf3(
        after, changed_mts=[52], baseline_content=before)
    status = {u.mt: u for u in updates}

    assert status[4].status == "updated"
    assert status[1].status == "updated"
    assert status[4].components == (51, 52)
    assert status[1].components == (2, 4)

    np.testing.assert_allclose(_values(out, 4)[1], [6.0, 6.0, 6.0])
    np.testing.assert_allclose(_values(out, 1)[1], [13.0, 11.0, 9.0])


def test_a_redundant_the_change_cannot_reach_is_left_alone():
    """Editing capture says nothing about the inelastic total."""
    grid = [1.0e5, 2.0e7]
    tape = _tape(
        _section(1, grid, [9.0, 9.0]),
        _section(2, grid, [7.0, 7.0]),
        _section(4, grid, [2.0, 2.0]),
        _section(51, grid, [2.0, 2.0]),
        _section(102, grid, [1.0, 1.0]),
    )
    out, updates = recompute_redundant_mf3(tape, changed_mts=[102])
    touched = {u.mt for u in updates}

    assert 4 not in touched
    assert {u.mt for u in updates if u.status == "updated"} == {1}


def test_the_sum_is_taken_on_the_union_of_the_partial_grids():
    """A partial with structure the total does not sample carries it in."""
    tape = _tape(
        _section(1, [1.0, 3.0], [2.0, 4.0]),
        _section(2, [1.0, 3.0], [1.0, 1.0]),
        _section(102, [1.0, 2.0, 3.0], [1.0, 5.0, 3.0]),
    )
    out, _ = recompute_redundant_mf3(tape, changed_mts=[102])
    energies, values = _values(out, 1)

    np.testing.assert_allclose(energies, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 6.0, 4.0])


def test_the_total_keeps_energies_its_partials_do_not_have():
    """Its own grid joins the union, so a rewrite can never drop a point."""
    tape = _tape(
        _section(1, [1.0, 1.5, 2.0], [2.0, 3.0, 4.0]),
        _section(2, [1.0, 2.0], [1.0, 1.0]),
        _section(102, [1.0, 2.0], [1.0, 3.0]),
    )
    out, _ = recompute_redundant_mf3(tape, changed_mts=[102])
    energies, values = _values(out, 1)

    np.testing.assert_allclose(energies, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(values, [2.0, 3.0, 4.0])


def test_a_repeated_energy_survives_the_resummation():
    """A repeated energy is a step in sigma, and both limits have to come out.

    Collapsing the pair would turn the step into a ramp across the whole
    preceding interval, which is a real change to the cross section.
    """
    tape = _tape(
        _section(1, [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        _section(2, [1.0, 3.0], [1.0, 1.0]),
        _section(102, [1.0, 2.0, 2.0, 3.0], [1.0, 1.0, 7.0, 7.0]),
    )
    out, _ = recompute_redundant_mf3(tape, changed_mts=[102])
    energies, values = _values(out, 1)

    np.testing.assert_allclose(energies, [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 2.0, 8.0, 8.0])


def test_a_partial_below_its_threshold_contributes_zero():
    tape = _tape(
        _section(1, [1.0, 10.0], [1.0, 1.0]),
        _section(2, [1.0, 10.0], [1.0, 1.0]),
        _section(51, [5.0, 10.0], [0.0, 4.0], qi=-5.0),
    )
    out, _ = recompute_redundant_mf3(tape, changed_mts=[51])
    energies, values = _values(out, 1)

    np.testing.assert_allclose(energies, [1.0, 5.0, 10.0])
    np.testing.assert_allclose(values, [1.0, 1.0, 5.0])


def test_the_rebuilt_section_says_lin_lin_over_one_region():
    tape = _tape(
        _section(1, [1.0, 2.0], [2.0, 2.0]),
        _section(2, [1.0, 2.0], [1.0, 1.0]),
        _section(102, [1.0, 1.5, 2.0], [1.0, 4.0, 1.0]),
    )
    out, _ = recompute_redundant_mf3(tape, changed_mts=[102])
    section = _parse_baseline(out)[1]

    assert section.num_interpolation_regions == 1
    assert section.energy_interpolation == [(3, 2)]
    assert section.num_energy_points == 3


# ---------------------------------------------------------------------------
# What it refuses to touch
# ---------------------------------------------------------------------------

def test_a_head_field_the_dataclass_does_not_model_is_preserved():
    """ENDF/B-VIII.1 B-10 writes L2=2 on an MF3 HEAD. Rewriting must keep it."""
    tape = _tape(
        _section(1, [1.0, 2.0], [2.0, 2.0], head_l2=2),
        _section(2, [1.0, 2.0], [1.0, 1.0]),
        _section(102, [1.0, 1.5, 2.0], [1.0, 4.0, 1.0]),
    )
    out, updates = recompute_redundant_mf3(tape, changed_mts=[102])

    assert [u.status for u in updates if u.mt == 1] == ["updated"]
    head = next(line for line in out.splitlines()
                if parse_endf_id(line)[1:] == (3, 1) and line[75:80].strip() == "1")
    assert head[33:44].strip() == "2"


def test_an_explicitly_transferred_redundant_is_left_as_the_user_placed_it():
    grid = [1.0, 2.0]
    tape = _tape(
        _section(1, grid, [99.0, 99.0]),
        _section(2, grid, [1.0, 1.0]),
        _section(102, grid, [1.0, 1.0]),
    )
    out, updates = recompute_redundant_mf3(
        tape, changed_mts=[1, 102], protected_mts=[1])

    assert [u.status for u in updates if u.mt == 1] == ["skipped"]
    np.testing.assert_allclose(_values(out, 1)[1], [99.0, 99.0])


def test_a_tape_whose_partials_never_added_up_is_reported_not_rebuilt(micro_tape):
    """A cut-down tape keeps MT1 but not the partials that make it.

    Summing what survived would replace a real total with a fragment of it, so
    the baseline gate has to catch this and say why.
    """
    content = micro_tape.open(encoding="utf-8", newline="").read()
    out, updates = recompute_redundant_mf3(
        content, changed_mts=[2], baseline_content=content)
    total = next(u for u in updates if u.mt == 1)

    assert total.status == "skipped"
    assert total.baseline_deviation > 0.5
    assert "before the edit" in total.reason
    assert out == content


def test_without_a_baseline_that_same_tape_is_rebuilt():
    """The gate is the baseline's doing, not a rule of its own."""
    path = Path(__file__).parent / "data" / "micro_fe56_structural.endf"
    content = path.open(encoding="utf-8", newline="").read()
    _, updates = recompute_redundant_mf3(content, changed_mts=[2])

    assert [u.status for u in updates if u.mt == 1] == ["updated"]


def test_a_tape_with_no_mf3_is_returned_unchanged():
    content = "\n".join([format_endf_fend_record(MAT),
                         format_endf_mend_record()]) + "\n"
    out, updates = recompute_redundant_mf3(content)

    assert out == content
    assert updates == []


# ---------------------------------------------------------------------------
# Real evaluations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mt", [1, 4])
def test_a_real_evaluation_already_satisfies_the_sum_rules(fe56_host_tape, mt):
    """JEFF-4.0 Fe-56, as distributed, to the round-off of a six-digit float."""
    sections = _parse_baseline(fe56_host_tape.open(encoding="utf-8", newline="").read())
    components = resolve_sum_components(mt, sections)
    assert components

    from kika.endf.writers.redundant import (_evaluate, _relative_deviation,
                                             _sum_partials)
    own = np.asarray(sections[mt].energies, dtype=float)
    grid, occurrence, total = _sum_partials(sections, components, extra_grids=[own])
    deviation = _relative_deviation(_evaluate(sections[mt], grid, occurrence), total)

    assert deviation < 1e-4


def test_resumming_an_untouched_evaluation_gives_the_tape_back(fe56_host_tape):
    """The strongest statement available: no edit, no rewrite, byte for byte."""
    content = fe56_host_tape.open(encoding="utf-8", newline="").read()
    out, updates = recompute_redundant_mf3(content)

    assert out == content
    assert not [u for u in updates if u.status == "updated"]


# ---------------------------------------------------------------------------
# Reached from the operation that motivates it
# ---------------------------------------------------------------------------

def _transfer_tape(tmp_path, name, *sections):
    path = tmp_path / name
    path.write_text(_tape(*sections))
    return path


def _replacement(mt, energies, xs):
    """An MF3MT ready to hand to ``replace_mt_section``."""
    section = MF3MT(number=mt)
    section._za, section._awr, section._mat = 26056.0, 55.454, MAT
    section._qm, section._qi, section._lr = 0.0, 0.0, 0
    section._energies = [float(e) for e in energies]
    section._cross_sections = [float(v) for v in xs]
    section._nr, section._np = 1, len(energies)
    section._interpolation = [(len(energies), 2)]
    return section


def test_replacing_a_level_can_carry_the_totals_with_it(tmp_path):
    """The scenario the module exists for, through the API that performs it.

    ``replace_mt_section`` is the operation that transfers a section in from
    another evaluation; before this it left MT4, MT3 and MT1 stating the old
    sum, and the file was internally inconsistent while every section parsed.
    """
    grid = [1.0e5, 1.0e6, 2.0e7]
    path = _transfer_tape(
        tmp_path, "host.endf",
        _section(1, grid, [10.0, 8.0, 6.0]),
        _section(2, grid, [7.0, 5.0, 3.0]),
        _section(4, grid, [3.0, 3.0, 3.0]),
        _section(51, grid, [2.0, 2.0, 2.0]),
        _section(52, grid, [1.0, 1.0, 1.0]),
    )

    writer = ENDFWriter(str(path))
    assert writer.replace_mt_section(
        _replacement(52, grid, [4.0, 4.0, 4.0]), mf_number=3,
        update_directory=False, resum_redundant=True,
    )

    content = path.read_text()
    np.testing.assert_allclose(_values(content, 52)[1], [4.0, 4.0, 4.0])
    np.testing.assert_allclose(_values(content, 4)[1], [6.0, 6.0, 6.0])
    np.testing.assert_allclose(_values(content, 1)[1], [13.0, 11.0, 9.0])

    status = {u.mt: u.status for u in writer.redundant_updates}
    assert status[4] == "updated" and status[1] == "updated"


def test_the_option_is_off_by_default(tmp_path):
    """A replacement stays a byte operation on one section unless asked."""
    grid = [1.0e5, 1.0e6, 2.0e7]
    path = _transfer_tape(
        tmp_path, "host.endf",
        _section(1, grid, [10.0, 8.0, 6.0]),
        _section(2, grid, [7.0, 5.0, 3.0]),
        _section(4, grid, [3.0, 3.0, 3.0]),
        _section(51, grid, [2.0, 2.0, 2.0]),
        _section(52, grid, [1.0, 1.0, 1.0]),
    )

    writer = ENDFWriter(str(path))
    assert writer.replace_mt_section(
        _replacement(52, grid, [4.0, 4.0, 4.0]), mf_number=3,
        update_directory=False,
    )

    content = path.read_text()
    np.testing.assert_allclose(_values(content, 52)[1], [4.0, 4.0, 4.0])
    np.testing.assert_allclose(_values(content, 4)[1], [3.0, 3.0, 3.0])
    assert writer.redundant_updates == []


def test_transferring_a_redundant_keeps_it_and_carries_the_ones_above(tmp_path):
    """Replacing MT4 keeps the MT4 that was placed, and moves MT1 to match.

    Note which guard does the work here, because it is not the obvious one.
    MT4 survives because the change cannot *reach* it -- MT4 is rebuilt from
    MT51 and MT52, and neither moved -- not because ``protected_mts`` caught
    it. The protection matters only when the same call both edits a partial and
    has to leave a redundant alone, which is the scenario
    ``test_an_explicitly_transferred_redundant_is_left_as_the_user_placed_it``
    covers at the function's own level.
    """
    grid = [1.0e5, 2.0e7]
    path = _transfer_tape(
        tmp_path, "host.endf",
        _section(1, grid, [10.0, 10.0]),
        _section(2, grid, [7.0, 7.0]),
        _section(4, grid, [3.0, 3.0]),
        _section(51, grid, [2.0, 2.0]),
        _section(52, grid, [1.0, 1.0]),
    )

    writer = ENDFWriter(str(path))
    assert writer.replace_mt_section(
        _replacement(4, grid, [9.0, 9.0]), mf_number=3,
        update_directory=False, resum_redundant=True,
    )

    content = path.read_text()
    np.testing.assert_allclose(_values(content, 4)[1], [9.0, 9.0])
    np.testing.assert_allclose(_values(content, 1)[1], [16.0, 16.0])
    assert {u.mt for u in writer.redundant_updates if u.status == "updated"} == {1}


def test_a_two_section_transfer_survives_one_call_at_a_time(tmp_path):
    """And it is the baseline gate that saves it, not ``protected_mts``.

    ``protected_mts`` only ever holds the MT of the call it belongs to, so a
    transfer of several sections has no way to say "and MT4 is mine too". That
    looks like a hole: move MT4 in, then edit MT52 with the option on, and the
    second call should rebuild MT4 from the partials and discard the MT4 just
    placed.

    It does not, because the baseline of the second call is the file as it was
    when that call started -- which already carries the transferred MT4, whose
    stated 9 is 67% away from the 3 that MT51+MT52 make. The rule "only restore the invariant where it held
    beforehand" then declines to touch it. Worth a test of its own precisely
    because the two guards look interchangeable and are not: one is about what
    the caller named, the other about what the file already claimed.

    The case where the gate lets the rebuild through is the one where it does no
    harm -- a transferred total that already agrees with the partials is
    rewritten to the value it already had.
    """
    grid = [1.0e5, 2.0e7]
    path = _transfer_tape(
        tmp_path, "host.endf",
        _section(1, grid, [10.0, 10.0]),
        _section(2, grid, [7.0, 7.0]),
        _section(4, grid, [3.0, 3.0]),
        _section(51, grid, [2.0, 2.0]),
        _section(52, grid, [1.0, 1.0]),
    )

    ENDFWriter(str(path)).replace_mt_section(
        _replacement(4, grid, [9.0, 9.0]), mf_number=3, update_directory=False)
    writer = ENDFWriter(str(path))
    writer.replace_mt_section(
        _replacement(52, grid, [1.0, 1.0]), mf_number=3,
        update_directory=False, resum_redundant=True)

    np.testing.assert_allclose(_values(path.read_text(), 4)[1], [9.0, 9.0])
    skipped = {u.mt: u for u in writer.redundant_updates if u.status == "skipped"}
    assert 4 in skipped
    assert "from the sum of its partials before the edit" in skipped[4].reason
    assert skipped[4].baseline_deviation == pytest.approx(6.0 / 9.0)
