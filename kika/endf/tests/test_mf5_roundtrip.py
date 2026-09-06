"""MF5: the read-write gate, and the exactness the sampler is built on.

Two jobs.

**The gate.** Nothing may perturb an MF5 table until parsing and re-emitting one
gives the bytes back. That is asserted on the committed Cf-252 slice, and — at
the full 80 columns — on the real JEFF-4.0 U-235 tape, where kika's ENDF dialect
and the evaluator's agree completely. ENDF/B-VIII.1 U-235 is pinned as a strict
xfail against the ``format_interp_pairs`` padding defect it shares with MF3.

**The exactness.** The PFNS sampler preserves normalisation because it inserts
outgoing nodes and incident nodes that *refine* the ENDF interpolant rather than
approximating it. If a refinement is not exact, the group integrals move, the
sum rule stops closing, and the renormalisation step silently absorbs the
difference — a perturbation that looks fine and is not the one that was
computed. So the refinement identities are tested against 10 000 random points,
not against a tolerance chosen to pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf5.partials import MF5PartialTabulated
from kika.endf.parsers.parse_mf5 import parse_mf5_mt
from kika.endf.utils import parse_endf_id


def data_lines(text: str, mf: int, mt: int) -> list[str]:
    return [
        line for line in text.splitlines()
        if len(line) >= 75 and parse_endf_id(line)[1:] == (mf, mt)
    ]


def first_difference(expected: list[str], got: list[str]) -> str:
    """Same contract as the golden module's: never build a whole-file diff."""
    if expected == got:
        return ""
    if len(expected) != len(got):
        return f"line count {len(expected)} -> {len(got)}"
    for i, (want, have) in enumerate(zip(expected, got)):
        if want != have:
            n_diff = sum(1 for a, b in zip(expected, got) if a != b)
            return (
                f"{n_diff} of {len(expected)} lines differ; first at index {i}\n"
                f"  source: {want!r}\n"
                f"  kika  : {have!r}"
            )
    return "lists differ but no differing line was found"


def roundtrip(path, mf: int, mt: int, columns: int) -> str:
    """Parse one section out of *path* and report how its bytes differ."""
    text = Path(path).read_text()
    endf = read_endf(str(path), mf_numbers=[mf])
    section = endf.files[mf].sections[mt]
    source = [line[:columns] for line in data_lines(text, mf, mt)]
    got = [line[:columns] for line in data_lines(str(section), mf, mt)]
    return first_difference(source, got)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_mf5_mt18_roundtrip_cols_1_75(micro_pfns_tape):
    """The LF=1 path: TAB2 over incident energy, NE outgoing TAB1 records."""
    assert not roundtrip(micro_pfns_tape, 5, 18, 75)


def test_mf5_mt455_lf5_roundtrip_cols_1_75(micro_pfns_tape):
    """The verbatim path: six LF=5 partials nothing in kika interprets.

    Cheap to get right and worth having — it is the evidence that an NK>1
    section with one tabulated partial and the rest passed through will write
    back correctly, which is the case that decides whether the pipeline is safe
    on a tape nobody has read.
    """
    assert not roundtrip(micro_pfns_tape, 5, 455, 75)


def test_jeff40_u235_mf5_roundtrip_is_byte_identical(u235_tape):
    """The full 80-column gate, on the real 7.3 MB tape.

    JEFF-4.0 writes sequence numbers and pads unused CONT fields with blanks,
    which is exactly what kika's emitters do — so here byte-identity is
    available and is demanded, all 9 894 lines of MT18 and 2 353 of MT455.
    """
    assert not roundtrip(u235_tape, 5, 18, 80)
    assert not roundtrip(u235_tape, 5, 455, 80)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Shared with MF3, not specific to MF5: ENDF/B-VIII.1 U-235 zero-fills "
        "the unused fields of an interpolation record, where "
        "`format_interp_pairs` (kika/endf/utils.py:610) writes blanks. 25 "
        "lines in MF5/MT18 and 6 in MF5/MT455 here; 347 in MF3/MT1 alone on "
        "the same tape. Fixing it is one switch in the shared formatter and "
        "must be decided for MF3 and MF5 together, so it is pinned rather "
        "than worked around inside the MF5 classes. Cf-252 blank-pads and so "
        "round-trips clean, which is why the committed fixture cannot see this."
    ),
)
def test_endfb81_u235_mf5_roundtrip(u235_b81_tape):
    assert not roundtrip(u235_b81_tape, 5, 18, 75)


@pytest.mark.slow
def test_endfb81_pu239_mf5_head_roundtrip(pu239_b81_tape):
    """ZA and AWR written in plain decimal, not ENDF exponential.

    ENDF/B-VIII.1 Pu-239 writes ``94239.0000`` in the MF5/MT18 HEAD.
    ``parse_number`` tries ``float()`` first and reads it without complaint, so
    this is a write-back question only: kika re-emits the canonical
    ``9.423900+4``. The values are equal; the bytes are not.

    Asserted as a *measured* divergence rather than an xfail, because unlike
    the padding defect there is nothing to fix — re-emitting a non-canonical
    field verbatim would mean storing the source text of every number.
    """
    text = Path(pu239_b81_tape).read_text()
    endf = read_endf(str(pu239_b81_tape), mf_numbers=[5])
    section = endf.files[5].sections[18]

    source_head = data_lines(text, 5, 18)[0]
    kika_head = data_lines(str(section), 5, 18)[0]

    assert source_head[:11].strip() == "94239.0000"
    assert kika_head[:11].strip() == "9.423900+4"
    assert section._za == pytest.approx(94239.0)

    body_source = [line[:75] for line in data_lines(text, 5, 18)[1:]]
    body_kika = [line[:75] for line in data_lines(str(section), 5, 18)[1:]]
    assert len(body_source) == len(body_kika)


# ---------------------------------------------------------------------------
# Parsing behaviour
# ---------------------------------------------------------------------------

def test_unknown_lf_raises_naming_the_partial():
    """An LF with no known record layout must stop, not guess.

    Guessing the record count would swallow the *next* subsection whole and
    produce a section that parses, emits and is wrong.
    """
    lines = [
        " 9.825200+4 2.499160+2          0          0          1          09861 5 18",
        " 0.000000+0 0.000000+0          0         99          1          29861 5 18",
        "          2          2                                            9861 5 18",
        " 1.000000-5 1.000000+0 2.000000+7 1.000000+0                      9861 5 18",
    ]
    with pytest.raises(ValueError, match=r"MF5/MT18 subsection 0 uses LF=99"):
        parse_mf5_mt(lines, 18)


def test_a_bad_partial_loses_only_its_own_mt(micro_pfns_tape, caplog):
    """The file-level loop logs and continues, as MF4's does."""
    from kika.endf.parsers.parse_mf5 import parse_mf5

    text = Path(micro_pfns_tape).read_text().splitlines()
    mf5_lines = [line for line in text
                 if len(line) >= 75 and parse_endf_id(line)[1] == 5]
    broken = [line.replace("         5 ", "        99 ", 1)
              if parse_endf_id(line)[2] == 455 and line[33:44].strip() == "5"
              else line
              for line in mf5_lines]

    mf = parse_mf5(broken)
    assert 18 in mf.sections, "a bad MT455 took MT18 down with it"


# ---------------------------------------------------------------------------
# The exactness the sampler rests on
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_partial() -> MF5PartialTabulated:
    """Two incident nodes with deliberately different outgoing grids."""
    grid_lo = np.array([0.0, 1.0e5, 4.0e5, 1.0e6, 3.0e6, 1.0e7, 2.0e7])
    grid_hi = np.array([0.0, 2.0e5, 5.0e5, 1.5e6, 4.0e6, 2.0e7])
    chi_lo = np.sqrt(grid_lo) * np.exp(-grid_lo / 1.3e6)
    chi_hi = np.sqrt(grid_hi) * np.exp(-grid_hi / 1.6e6)
    return MF5PartialTabulated(
        u=0.0, lf=1, p_interp=[(2, 2)],
        p_energies=[1.0e-5, 2.0e7], p_values=[1.0, 1.0],
        tab2_interp=[(2, 2)],
        incident_energies=[1.0e-5, 2.0e7],
        outgoing_grids=[list(grid_lo), list(grid_hi)],
        chi=[list(chi_lo), list(chi_hi)],
        outgoing_interp=[[(len(grid_lo), 2)], [(len(grid_hi), 2)]],
    )


def chi_at(partial: MF5PartialTabulated, energy: float,
           points: np.ndarray) -> np.ndarray:
    """The ENDF 2-D interpolant, evaluated the long way round."""
    grid, values = partial.evaluate_at_incident(energy)
    return np.interp(points, grid, values, left=0.0, right=0.0)


def test_evaluate_at_incident_is_exact_on_the_nodes(synthetic_partial):
    for k, energy in enumerate(synthetic_partial.incident_energies):
        grid, values = synthetic_partial.evaluate_at_incident(energy)
        np.testing.assert_array_equal(grid, synthetic_partial.outgoing_grids[k])
        np.testing.assert_array_equal(values, synthetic_partial.chi[k])


def test_lf1_node_insertion_is_an_exact_refinement(synthetic_partial):
    """Inserting an incident node must not move chi anywhere.

    10 000 random (E, E′). This is the identity the band-edge shoulder of P3.5
    depends on: the new node exists only so a piecewise-constant factor has
    somewhere to land, and it must be invisible until a factor is applied.

    **The tolerance is set from what double arithmetic can deliver, not from
    what would be nice.** Measured worst case 4.1e-15 relative — about 18 ULP,
    which is what two chained linear interpolations cost and nothing more. The
    bound is 5e-14, twelve times that. The error this test exists to catch is
    copying the neighbouring table instead of blending, which is ~1e-3: eleven
    orders of magnitude clear of the noise, so the headroom buys nothing away.
    """
    rng = np.random.default_rng(20260810)
    probe_e = rng.uniform(1.0e-5, 2.0e7, 200)
    probe_ep = rng.uniform(0.0, 2.0e7, 50)

    before = np.array([chi_at(synthetic_partial, e, probe_ep) for e in probe_e])

    index = synthetic_partial.insert_incident_node(6.0e6)
    assert synthetic_partial.incident_energies[index] == 6.0e6
    assert len(synthetic_partial.incident_energies) == 3
    assert synthetic_partial.tab2_interp == [(3, 2)]

    after = np.array([chi_at(synthetic_partial, e, probe_ep) for e in probe_e])
    np.testing.assert_allclose(after, before, rtol=5e-14, atol=1e-300)


def test_inserting_an_existing_incident_node_is_a_no_op(synthetic_partial):
    before = list(synthetic_partial.incident_energies)
    assert synthetic_partial.insert_incident_node(2.0e7) == 1
    assert synthetic_partial.incident_energies == before


def test_group_integrals_sum_to_the_table_normalisation(synthetic_partial):
    """Σ_j P_j over boundaries spanning the table equals ∫χ, exactly."""
    boundaries = [0.0, 1.0e5, 7.0e5, 2.5e6, 9.0e6, 2.0e7]
    for k in range(len(synthetic_partial.incident_energies)):
        integrals = synthetic_partial.group_integrals(k, boundaries)
        assert integrals.sum() == pytest.approx(
            synthetic_partial.normalisation(k), rel=1e-14
        )


def test_group_integrals_are_exact_against_a_refined_table(synthetic_partial):
    """Clipping the limits must give what refining the grid would give.

    ``group_integrals`` integrates on the table's own panels rather than
    inserting the boundaries first. The two have to agree to machine precision,
    or the P⁰ the sampler perturbs is not the P⁰ MF35 is the covariance of.
    """
    boundaries = np.array([5.0e4, 3.3e5, 1.234e6, 6.0e6, 1.9e7])
    for k in range(len(synthetic_partial.incident_energies)):
        got = synthetic_partial.group_integrals(k, boundaries)

        grid = np.asarray(synthetic_partial.outgoing_grids[k])
        values = np.asarray(synthetic_partial.chi[k])
        refined = np.union1d(grid, boundaries)
        refined_values = np.interp(refined, grid, values, left=0.0, right=0.0)
        expected = np.array([
            np.trapezoid(
                refined_values[(refined >= lo) & (refined <= hi)],
                refined[(refined >= lo) & (refined <= hi)],
            )
            for lo, hi in zip(boundaries[:-1], boundaries[1:])
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_a_non_exact_interpolation_code_is_refused(synthetic_partial):
    """log-log tables have no exact group integral here, so they raise."""
    synthetic_partial.outgoing_interp[0] = [(7, 5)]
    with pytest.raises(NotImplementedError, match=r"interpolation code\(s\) \[5\]"):
        synthetic_partial.group_integrals(0, [0.0, 1.0e6, 2.0e7])


def test_real_cf252_spectra_are_already_normalised(micro_pfns_tape):
    """The input's own sum-rule residual, measured before anything perturbs it.

    ``sum_rule_residual`` does this for nu-bar and it is how you find out the
    file was off before blaming your own code.

    Measured, 2026-08-10: every incident node of Cf-252 MT18 integrates to 1
    within **4.31e-7**, worst node 5. That is an order of magnitude looser than
    ENDF/B-VIII.1 U-235's 4.8e-8, and it is the tape's own residual, not
    kika's — so the sampler's normalisation target is this number, not 1.
    Recorded here so the sampler's diagnostics have something to be compared
    against rather than only asserted.
    """
    endf = read_endf(str(micro_pfns_tape), mf_numbers=[5])
    partial = endf.files[5].sections[18].partials[0]
    residuals = np.array([
        partial.normalisation(k) - 1.0
        for k in range(len(partial.incident_energies))
    ])
    assert np.max(np.abs(residuals)) < 1e-6
    assert np.max(np.abs(residuals)) == pytest.approx(4.306e-7, rel=1e-3)
