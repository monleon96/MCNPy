"""Every field the MF2/151 parser holds survives ``decodeMF2MT151``.

**Why this test is the deliverable of B1a and not a by-product of it.** The
plan for an MF2 encoder assumed the work was emitting five formalisms' record
layouts. It was not: ``MF2MT151.__str__`` already writes the section byte for
byte (``test_mf2_writes_back_what_it_read.py``), so the encoder is a
model→flat-tree rebuild — and the obstacle is that **the model did not retain
enough to drive it**. Measured before B1a, the decoder was dropping the
l-dependent radius on half the corpus, ten of the twelve columns of every LRF=7
particle pair, BND, APE, APT, IFG, KRL, LAD, QX/LRX, URR case C's INT and its
per-J energy grid, case B's MUF, and every LRU=0 range in its entirety.

Discovering those one at a time from a failing byte-identity gate is the slow
way round: each failure names a *line*, not a field, and the next one only
appears once the previous is fixed. So the losses are asserted directly, field
by field, against the parsed section — and this test is what makes the encoder's
gate a formality rather than an investigation.

**It compares against the file's own parse, never against the flat
``ResonanceParameters``.** That is the P7b lesson: a gate that compares the
model against the code it replaces cannot see a defect the model *inherits*,
because both sides agree and the agreement reads as proof. ``from_endf`` drops
LRF=7 outright, so half of what is checked here has no flat counterpart at all.

**Where each field is allowed to live.** Physics on the model, ENDF bookkeeping
in ``provenance.headerFields`` — decision 1(a) of ``docs/library/mf2-encoder-notes.md``.
The test asserts the *value* survives and says which side it read it from, so a
future move between the two is a one-line edit here rather than a silent
regression.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kika.endf.model_adapter import decodeMF2MT151
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import radiusToEndf

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]

#: The leaves that cross a **unit** boundary and therefore cannot be compared
#: bit for bit, and only those.
#:
#: Every other field in this file is copied, so the comparison below is exact on
#: purpose. These three are multiplied by ten on the way in and divided by ten
#: on the way back out — the model states radii in fm (``MODEL_RADIUS_UNIT``) —
#: and ``x * 10 / 10`` is not the identity in binary floating point. JENDL's
#: Fe-56 is the tape that shows it: APL is ``0.48960000000000004`` in the file
#: and ``0.4896000000000001`` after the round trip, one ULP apart.
#:
#: **The tolerance hides nothing, because a tighter gate runs next door.**
#: ENDF writes six significant figures, so a one-ULP difference cannot reach the
#: file — and ``test_mf2_writes_back_what_it_read`` compares the *written bytes*
#: on all six of these tapes. If a radius were genuinely rebuilt rather than
#: kept, 1e-12 would not absorb it and the byte gate would fail as well.
#:
#: ``apl_or_qx`` is in the set although under LRF=1/2 the same field is QX, a Q
#: value that is copied and not converted. Naming the field twice by LRF would
#: be more precise and less readable; 1e-12 on a Q value in eV is far below any
#: difference a rebuild could produce.
RESCALED_LEAVES = frozenset({"apl_or_qx", "apt", "ape"})


def _load(request, tape):
    path = request.getfixturevalue(tape)
    section = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151]
    return section, decodeMF2MT151(section)


# ---------------------------------------------------------------------------
# The inventories: the same nested shape, built from the two sides
# ---------------------------------------------------------------------------

def _flatInventory(section) -> dict:
    """Every field the parser holds, as a plain nested structure."""
    return {
        "za": int(round(float(section.zaid))),
        "awr": section.atomic_weight_ratio,
        "mat": section._mat,
        "nis": section._nis,
        "isotopes": [
            {
                "za": isotope.za,
                "abn": isotope.abn,
                "lfw": isotope.lfw,
                "ner": len(isotope.energy_ranges),
                "ranges": [_flatRange(r) for r in isotope.energy_ranges],
            }
            for isotope in section.isotopes
        ],
    }


def _flatRange(energyRange) -> dict:
    parameters = energyRange.parameters
    entry = {
        "el": energyRange.el, "eh": energyRange.eh,
        "lru": energyRange.lru, "lrf": energyRange.lrf,
        "nro": energyRange.nro, "naps": energyRange.naps,
        "spi": getattr(parameters, "spi", None),
        "ap": getattr(parameters, "ap", None),
    }
    if energyRange.ap_e is not None:
        entry["radius_table"] = {
            "interpolation": [tuple(pair) for pair in energyRange.ap_e.interpolation],
            "energies": np.asarray(energyRange.ap_e.energies, dtype=float),
            "values": np.asarray(energyRange.ap_e.ap_values, dtype=float),
        }

    if energyRange.lru == 0:
        return entry

    if energyRange.lru == 2:
        entry.update({"lssf": parameters.lssf, "nls": parameters.nls,
                      "urr": _flatUnresolved(parameters)})
        return entry

    if energyRange.lrf == 7:
        entry.update({
            "ifg": parameters.ifg, "krm": parameters.krm, "krl": parameters.krl,
            "pairs": [
                {name: getattr(pair, name) for name in
                 ("ma", "mb", "za", "zb", "ia", "ib", "q", "pnt", "shf", "mt", "pa", "pb")}
                for pair in parameters.particle_pairs
            ],
            "groups": [
                {
                    "aj": group.aj, "pj": group.pj,
                    "kbk": group.kbk, "kps": group.kps,
                    "channels": [
                        {"ipp": c.ipp, "l": c.l, "sch": c.sch,
                         "bnd": c.bnd, "ape": c.ape, "apt": c.apt}
                        for c in group.channels
                    ],
                    "energies": [r.er for r in group.resonances],
                    "widths": [list(r.widths) for r in group.resonances],
                }
                for group in parameters.spin_groups
            ],
        })
        return entry

    entry.update({
        "nls": parameters.nls, "nlsc": parameters.nlsc, "lad": parameters.lad,
        "blocks": [
            {
                "l": block.l, "awri": block.awri,
                "num_resonances": block.num_resonances,
                "apl_or_qx": block.apl_or_qx, "lrx": block.lrx,
                "resonances": [(r.energy, r.spin, r.c3, r.c4, r.c5, r.c6)
                               for r in block.resonances],
            }
            for block in parameters.l_values
        ],
    })
    return entry


def _flatUnresolved(parameters) -> dict:
    from kika.endf.classes.mf2.mf2mt151 import (UnresolvedCaseA, UnresolvedCaseB,
                                                UnresolvedCaseC)
    if isinstance(parameters, UnresolvedCaseA):
        return {
            "case": "A",
            "blocks": [
                {"l": b.l, "awri": b.awri,
                 "states": [{"aj": s.aj, "d": s.d, "amun": s.amun,
                             "gn0": s.gn0, "gg": s.gg} for s in b.j_states]}
                for b in parameters.l_values
            ],
        }
    if isinstance(parameters, UnresolvedCaseB):
        return {
            "case": "B",
            "energies": np.asarray(parameters.energies, dtype=float),
            "blocks": [
                {"l": b.l, "awri": b.awri,
                 "states": [{"aj": s.aj, "d": s.d, "amun": s.amun, "gn0": s.gn0,
                             "gg": s.gg, "muf": s.muf,
                             "gf": np.asarray(s.gf, dtype=float)} for s in b.j_states]}
                for b in parameters.l_values
            ],
        }
    if isinstance(parameters, UnresolvedCaseC):
        return {
            "case": "C",
            "blocks": [
                {"l": b.l, "awri": b.awri,
                 "states": [{
                     "aj": s.aj, "int_code": s.int_code,
                     "amux": s.amux, "amun": s.amun,
                     "amug": s.amug, "amuf": s.amuf,
                     "es": np.array([p.es for p in s.energy_points]),
                     "d": np.array([p.d for p in s.energy_points]),
                     "gx": np.array([p.gx for p in s.energy_points]),
                     "gn0": np.array([p.gn0 for p in s.energy_points]),
                     "gg": np.array([p.gg for p in s.energy_points]),
                     "gf": np.array([p.gf for p in s.energy_points]),
                 } for s in b.j_states]}
                for b in parameters.l_values
            ],
        }
    raise AssertionError(f"unhandled URR case {type(parameters).__name__}")


def _decodedInventory(resonances, provenance) -> dict:
    """The same shape, rebuilt from what the decoder kept."""
    header = provenance.headerFields
    regions = header["regions"]
    resolved = iter(resonances.resolved)

    byIsotope: dict = {}
    for fields in regions:
        entry = _decodedRange(fields, resolved, resonances)
        byIsotope.setdefault(fields["isotope_index"], []).append(entry)

    return {
        "za": provenance.za,
        "awr": provenance.awr,
        "mat": provenance.mat,
        "nis": header["nis"],
        "isotopes": [
            {**isotope, "ranges": byIsotope.get(index, [])}
            for index, isotope in enumerate(header["isotopes"])
        ],
    }


def _decodedRange(fields, resolved, resonances) -> dict:
    entry = {key: fields[key] for key in
             ("el", "eh", "lru", "lrf", "nro", "naps", "spi", "ap")}
    if "radius_table" in fields:
        energies, values, interpolation = fields["radius_table"]
        entry["radius_table"] = {"interpolation": [tuple(p) for p in interpolation],
                                 "energies": energies, "values": values}

    if fields["kind"] == "radiusOnly":
        return entry

    if fields["kind"] == "unresolved":
        entry.update({"lssf": fields["lssf"], "nls": fields["nls"],
                      "urr": _decodedUnresolved(fields, resonances.unresolved)})
        return entry

    formalism = next(resolved).formalism

    if fields["lrf"] == 7:
        entry.update({
            "ifg": int(formalism.reducedWidthAmplitudes),
            "krm": fields["krm"],
            "krl": int(formalism.relativisticKinematics),
            "pairs": fields["particle_pairs"],
            "groups": [
                {
                    "aj": group.spin, "pj": perGroup["pj"],
                    "kbk": perGroup["kbk"], "kps": perGroup["kps"],
                    "channels": [
                        {"ipp": index + 1, "l": c.L, "sch": c.channelSpin,
                         "bnd": c.boundaryConditionValue,
                         "ape": radiusToEndf(c.hardSphereRadius),
                         "apt": radiusToEndf(c.scatteringRadius)}
                        for index, c in _pairIndices(group, formalism)
                    ],
                    "energies": list(group.energies),
                    "widths": [list(row) for row in group.widths],
                }
                for group, perGroup in zip(formalism.spinGroups, fields["spin_groups"])
            ],
        })
        return entry

    entry.update({
        "nls": fields["nls"], "nlsc": fields["nlsc"], "lad": fields["lad"],
        "blocks": [
            {
                "l": _blockL(group),
                "awri": group.atomicWeightRatio,
                "num_resonances": block["num_resonances"],
                # LRF=3's APL is on the model; LRF=1/2's QX is bookkeeping and
                # is not. ``None`` on the model means the file wrote 0.
                "apl_or_qx": (_blockRadius(group) or 0.0) if fields["lrf"] == 3
                             else block["qx"],
                "lrx": block["lrx"],
                "resonances": _blockResonances(group),
            }
            for group, block in zip(_spinGroups(formalism), fields["l_blocks"])
        ],
    })
    return entry


def _pairIndices(group, formalism):
    """Channel → its one-based index into the particle-pair list.

    The model stores the *label* of the resonance reaction, so IPP comes back by
    looking the label up rather than being carried twice.
    """
    order = {reaction.label: index + 1
             for index, reaction in enumerate(formalism.resonanceReactions)}
    return [(order.get(channel.resonanceReaction, 0) - 1, channel)
            for channel in group.channels]


def _spinGroups(formalism):
    from kika.nuclear_data.model import BreitWigner
    if isinstance(formalism, BreitWigner):
        return formalism.resonanceParameters.spinGroups
    return formalism.spinGroups


def _blockL(group):
    return group.L if getattr(group, "L", None) is not None else group.channels[0].L


def _blockRadius(group):
    """The block's APL **back in the file's units**, which is what is compared.

    The model states radii in fm (``MODEL_RADIUS_UNIT``) and the file states
    them in ENDF's 10^-12 cm, so a bare comparison reports a loss where there is
    a *rescale*. Converting here rather than relaxing the assertion keeps what
    this file is for: that the value survives, exactly, and that the one thing
    the decoder is allowed to change is the scale it is stated on.
    """
    if getattr(group, "channels", None):
        return radiusToEndf(group.channels[0].scatteringRadius)
    return radiusToEndf(group.scatteringRadius)


def _blockResonances(group):
    from kika.nuclear_data.model.resonances import SpinGroup
    if isinstance(group, SpinGroup):
        return [r.toFlat() for r in group.resonances]
    return [(energy, spin, *widths) for energy, spin, widths
            in zip(group.energies, group.spins, group.widths)]


def _decodedUnresolved(fields, unresolved) -> dict:
    widths = unresolved.tabulatedWidths
    case = fields["urr_case"]
    states = fields["j_states"]

    byL: dict = {}
    for group, state in zip(widths.spinGroups, states):
        entry = byL.setdefault(group.L, {"l": group.L,
                                         "awri": group.atomicWeightRatio,
                                         "states": []})
        channels = {c.label: c for c in group.channels}
        if case == "A":
            entry["states"].append({
                "aj": group.J, "d": float(np.asarray(group.levelSpacing)[0]),
                "amun": channels["neutron"].degreesOfFreedom,
                "gn0": channels["neutron"].constantWidth,
                "gg": channels["capture"].constantWidth,
            })
        elif case == "B":
            entry["states"].append({
                "aj": group.J, "d": float(np.asarray(group.levelSpacing)[0]),
                "amun": channels["neutron"].degreesOfFreedom,
                "gn0": channels["neutron"].constantWidth,
                "gg": channels["capture"].constantWidth,
                "muf": state["muf"],
                "gf": np.asarray(state["gf"], dtype=float),
            })
        else:
            entry["states"].append({
                "aj": group.J, "int_code": state["int_code"],
                "amux": channels["competitive"].degreesOfFreedom,
                "amun": channels["neutron"].degreesOfFreedom,
                "amug": channels["capture"].degreesOfFreedom,
                "amuf": channels["fission"].degreesOfFreedom,
                "es": np.asarray(state["energy_grid"], dtype=float),
                "d": np.asarray(group.levelSpacing, dtype=float),
                "gx": np.asarray(channels["competitive"].widths, dtype=float),
                "gn0": np.asarray(channels["neutron"].widths, dtype=float),
                "gg": np.asarray(channels["capture"].widths, dtype=float),
                "gf": np.asarray(channels["fission"].widths, dtype=float),
            })

    out = {"case": case, "blocks": [byL[key] for key in byL]}
    if case == "B":
        out["energies"] = np.asarray(widths.energyGrid, dtype=float)
    return out


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def _diff(expected, actual, path="") -> list:
    """Every leaf that does not survive, named by its path. Not just the first."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a mapping, decoded {type(actual).__name__}"]
        problems = []
        for key in expected:
            if key not in actual:
                problems.append(f"{path}.{key}: absent from the decoded side")
            else:
                problems += _diff(expected[key], actual[key], f"{path}.{key}")
        return problems

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return [f"{path}: expected a sequence, decoded {type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} entries in the file, {len(actual)} decoded"]
        problems = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            problems += _diff(left, right, f"{path}[{index}]")
        return problems

    if isinstance(expected, np.ndarray):
        actualArray = np.asarray(actual, dtype=float)
        if expected.shape != actualArray.shape:
            return [f"{path}: shape {expected.shape} in the file, {actualArray.shape} decoded"]
        if not np.array_equal(expected, actualArray):
            worst = int(np.argmax(np.abs(expected - actualArray)))
            return [f"{path}[{worst}]: {expected[worst]!r} in the file, "
                    f"{actualArray[worst]!r} decoded"]
        return []

    if expected is None or actual is None:
        return [] if expected == actual else [
            f"{path}: {expected!r} in the file, {actual!r} decoded"]

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # Exact, because nothing here is computed — every value is copied, so a
        # tolerance would only hide a field being rebuilt instead of kept.
        # The three exceptions are the radii, which are *converted*: see
        # RESCALED_LEAVES.
        if path.rsplit(".", 1)[-1] in RESCALED_LEAVES:
            return [] if math.isclose(float(expected), float(actual),
                                      rel_tol=1e-12, abs_tol=0.0) else [
                f"{path}: {expected!r} in the file, {actual!r} decoded"]
        return [] if float(expected) == float(actual) else [
            f"{path}: {expected!r} in the file, {actual!r} decoded"]

    return [] if expected == actual else [
        f"{path}: {expected!r} in the file, {actual!r} decoded"]


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_decoder_keeps_every_field_the_parser_holds(request, tape):
    section, (resonances, provenance, _) = _load(request, tape)
    problems = _diff(_flatInventory(section), _decodedInventory(resonances, provenance))
    assert not problems, (
        f"{tape}: {len(problems)} MF2/151 field(s) do not survive the decode:\n  "
        + "\n  ".join(problems[:40])
    )


def test_the_inventory_would_notice_a_dropped_field(fe56_host_tape):
    """The comparison bites. Without this the test above can pass by being blind.

    Plants the exact defect B1a fixed — a per-l radius collapsed away — and
    requires it to be reported.
    """
    section = read_endf(str(fe56_host_tape), mf_numbers=[2]).mf[2].mt[151]
    resonances, provenance, _ = decodeMF2MT151(section)

    formalism = resonances.resolved[0].formalism
    for group in formalism.spinGroups:
        for channel in group.channels:
            channel.scatteringRadius = None

    problems = _diff(_flatInventory(section), _decodedInventory(resonances, provenance))
    assert any("apl_or_qx" in problem for problem in problems), (
        f"a blanked per-l scattering radius went unreported: {problems}"
    )


#: One planted defect per field group B1a closed, ``(name, mutation)``. The
#: first run of this module passed all thirteen tape assertions immediately,
#: which is not evidence of anything on its own — a comparison that reaches
#: nothing also passes. Each entry below was checked to be reported, and three
#: candidate mutations were discarded on the way because they mutated a field
#: the file already holds as zero and so changed nothing: Fe-57's first channel
#: has ``BND = APE = APT = 0``. A no-op mutation looks exactly like a blind
#: test, which is the trap this list exists to have already fallen into.
_PLANTED = [
    ("boundary condition",
     lambda f, p: setattr(f.spinGroups[0].channels[1], "boundaryConditionValue", 9.9)),
    ("effective channel radius APE",
     lambda f, p: setattr(f.spinGroups[0].channels[1], "hardSphereRadius", 0.0)),
    ("true channel radius APT",
     lambda f, p: setattr(f.spinGroups[0].channels[1], "scatteringRadius", 0.0)),
    ("the channel's particle pair",
     lambda f, p: setattr(f.spinGroups[0].channels[1], "resonanceReaction", "MT51")),
    ("a whole channel",
     lambda f, p: f.spinGroups[0].channels.pop()),
    ("reduced-width amplitudes IFG",
     lambda f, p: setattr(f, "reducedWidthAmplitudes", True)),
    ("relativistic kinematics KRL",
     lambda f, p: setattr(f, "relativisticKinematics", True)),
    ("a particle-pair column",
     lambda f, p: p["particle_pairs"][0].__setitem__("ma", 9.9)),
    ("a whole particle pair",
     lambda f, p: p.__setitem__("particle_pairs", p["particle_pairs"][:-1])),
    ("the background R-matrix count KBK",
     lambda f, p: p["spin_groups"][0].__setitem__("kbk", 3)),
]


@pytest.mark.parametrize("what,plant", _PLANTED, ids=[name for name, _ in _PLANTED])
def test_a_planted_lrf7_defect_is_reported(fe57_host_tape, what, plant):
    """Fe-57 JEFF-4.0, the only LRF=7 evaluation to hand and where B1a's losses were."""
    section = read_endf(str(fe57_host_tape), mf_numbers=[2]).mf[2].mt[151]
    resonances, provenance, _ = decodeMF2MT151(section)

    clean = _diff(_flatInventory(section), _decodedInventory(resonances, provenance))
    assert not clean, f"the fixture does not decode cleanly to begin with: {clean[:5]}"

    plant(resonances.resolved[0].formalism, provenance.headerFields["regions"][0])
    problems = _diff(_flatInventory(section), _decodedInventory(resonances, provenance))
    assert problems, f"a corrupted {what} went unreported"


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_every_range_in_the_file_appears_in_the_provenance(request, tape):
    """Counted separately, because a *missing range* cannot fail the diff above.

    The inventory pairs ranges by position, so a range dropped before the append
    — which is what LRU=0 used to be — shifts every later range up one and is
    reported as a value mismatch somewhere else entirely, if at all.
    """
    section, (_, provenance, _) = _load(request, tape)
    inFile = sum(len(isotope.energy_ranges) for isotope in section.isotopes)
    kept = len(provenance.headerFields["regions"])
    assert kept == inFile, (
        f"{tape}: the file has {inFile} energy range(s) and provenance kept {kept}"
    )
