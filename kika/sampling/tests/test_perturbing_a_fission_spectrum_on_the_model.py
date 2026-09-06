"""M6: the fission spectrum perturbed on the model, and the gate that says so.

Three of kika's four covariance files could already be perturbed through the
model — ``applyFactors`` for MF33 and MF31, ``applyLegendreFactors`` for MF34 —
and MF5 could not, for a reason that was measured rather than assumed: the
applier needs to *integrate* the spectrum over the covariance's groups, and
nothing in ``kika.nuclear_data.model`` could integrate anything. The four
operations it needed lived on ``MF5PartialTabulated``, which is an ENDF class,
so perturbing a PFNS was format work by construction.

**What this file gates.**

1. The four capabilities the model grew — ``table``, ``normalisation``,
   ``groupIntegrals``, ``replaceTable``, plus the exact outer-axis refinement —
   give the same numbers as the ENDF class on a real evaluation. Not "close":
   the same, because both call
   :mod:`kika.processing.panel_integrals` after the arithmetic moved down there.
2. :func:`~kika.nuclear_data.model.perturbation.applySpectrumFactors` reproduces
   :func:`~kika.sampling.mf35_sampling.perturb_pfns_partial` **point for point**
   on real drawn deltas, on the synthetic tape and on Cf-252. That is the
   *equivalent first* rule this repository applies to every migrated applier,
   and it is what makes the shipped pipeline and the model path two spellings of
   one operation rather than two operations.
3. A zero perturbation reproduces chi exactly. It is the strongest test in the
   PFNS suite and it is what catches an inexact refinement — a central-value
   shift that looks like physics.
4. The plumbing states what it did: one block per band and never merged, the
   band recorded on the realisation, the absolute semantics recorded per block,
   and a delta tape whose other files survive byte for byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter.energy import decodeMF5MT
from kika.nuclear_data.model.perturbation import (SPECTRUM_ENERGY_RTOL,
                                                  SPECTRUM_OUTER_SHOULDER,
                                                  SPECTRUM_STEP_SHOULDER,
                                                  applySpectrumFactors)
from kika.sampling.joint_blocks import (PER_SECTION_MF, SUPPORTED_MF,
                                        ComponentKey, assembleRequest,
                                        collectEntries, componentDomains,
                                        samplingGroups)
from kika.sampling.mf35_sampling import (ENDF_ENERGY_RTOL,
                                         INCIDENT_STEP_SHOULDER,
                                         OUTGOING_STEP_SHOULDER, band_grids,
                                         build_pfns_covariance,
                                         generate_pfns_samples,
                                         perturb_pfns_partial, pfns_ratio_rule)
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.perturbation_set import (SEMANTICS, SEMANTICS_OF_MF,
                                            PerturbationSet)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"

#: Both committed PFNS tapes. The synthetic one is eight groups over two bands
#: and small enough to reason about by hand; Cf-252 is a real evaluation with
#: four LB=7 bands of 122 groups, outgoing grids that differ from the MF5
#: table's own, and spectrum mass outside MF35's coverage at both ends.
TAPES = ("micro_pfns_cov.endf", "micro_cf252_pfns.endf")


def _read(name):
    """``(endf, covariance suite, mf5 section, bands, band grids)``."""
    endf = read_endf(str(DATA / name), mf_numbers=[5, 35])
    suite, mf5, bands = build_pfns_covariance(endf, mt=18)
    return endf, suite, mf5, bands, band_grids(suite)


def _deltas(suite, seed=7):
    """One sample's absolute group-probability deltas, keyed by band index."""
    samples, _diagnostics = generate_pfns_samples(suite, 1, seed=seed,
                                                  verbose=False)
    keys = sorted(samples, key=lambda key: key[-1])
    return {index: samples[key][0] for index, key in enumerate(keys)}


# ======================================================================
# 1. The capabilities the model grew
# ======================================================================

@pytest.mark.parametrize("name", TAPES)
def test_the_model_node_integrates_exactly_what_the_endf_class_does(name):
    """Identical, not close — the two call the same panel arithmetic.

    ``exact_segment_codes``, ``cumulative_integral``, ``integral_to`` and
    ``evaluate_table`` moved from ``kika/endf/classes/mf5/partials.py`` down to
    ``kika.processing.panel_integrals`` so that the model could reach them
    without importing a format package. The move was verbatim, so ``0.0`` here
    is the expected answer and any nonzero difference means one of the two grew
    an implementation of its own.
    """
    endf, _suite, mf5, _bands, grids = _read(name)
    partial = mf5.partials[0]
    form, _provenance, _report = decodeMF5MT(mf5)

    assert list(form.outerDomainValues) == list(partial.incident_energies)

    for k in range(len(partial.incident_energies)):
        xEndf, yEndf = partial.table(k)
        xModel, yModel = form.table(k)
        assert np.array_equal(xEndf, xModel)
        assert np.array_equal(yEndf, yModel)
        assert form.normalisation(k) == partial.normalisation(k)
        for grid in grids:
            assert np.array_equal(form.groupIntegrals(k, grid),
                                  partial.group_integrals(k, grid))


@pytest.mark.parametrize("name", TAPES)
def test_refining_the_incident_axis_is_the_same_exact_blend(name):
    """The blend between two nodes, which is where a 0.1 % error would hide."""
    _endf, _suite, mf5, _bands, _grids = _read(name)
    partial = mf5.partials[0]
    form, _provenance, _report = decodeMF5MT(mf5)

    energies = list(partial.incident_energies)
    midpoints = [0.5 * (energies[i] + energies[i + 1])
                 for i in range(len(energies) - 1)]
    for energy in midpoints:
        xEndf, yEndf = partial.evaluate_at_incident(energy)
        xModel, yModel = form.evaluateAtOuter(energy)
        assert np.array_equal(xEndf, xModel)
        assert np.array_equal(yEndf, yModel)


def test_a_node_that_is_not_a_table_refuses_to_be_integrated():
    """A Legendre child has no grid, and saying so beats an AttributeError."""
    from kika.nuclear_data.model.functions import Legendre

    with pytest.raises(TypeError, match="not a tabulated function"):
        Legendre(coefficients=np.array([1.0, 0.2])).integrate()


def test_a_multi_region_child_will_not_be_flattened_silently():
    """``replaceTable`` refuses what the ENDF twin does by default.

    ``MF5PartialTabulated.replace_table`` declares one lin-lin region whatever
    it was handed, which relabels every panel of a table whose later regions
    were histogram. No MF5 tape read so far has one; if one turns up, the caller
    has to say what it wants.
    """
    from kika.nuclear_data.model.enums import Interpolation
    from kika.nuclear_data.model.functions import Regions1d, XYs1d, XYs2d

    flat = XYs1d(xs=np.array([0.0, 1.0]), ys=np.array([1.0, 1.0]),
                 interpolation=Interpolation.flat)
    linear = XYs1d(xs=np.array([1.0, 2.0]), ys=np.array([1.0, 0.0]))
    child = Regions1d(function1ds=[flat, linear])
    child.outerDomainValue = 1.0
    form = XYs2d(function1ds=[child])

    with pytest.raises(ValueError, match="not one rule"):
        form.replaceTable(0, [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])

    # ...and it is accepted the moment the caller states the regions.
    form.replaceTable(0, [0.0, 1.0, 2.0], [2.0, 2.0, 0.0],
                      regions=[(2, 1), (3, 2)])
    assert form.normalisation(0) == pytest.approx(2.0 + 1.0)


# ======================================================================
# 2. The gate: the model applier against the shipped one
# ======================================================================

@pytest.mark.parametrize("name", TAPES)
def test_the_model_applier_reproduces_the_format_applier(name):
    """Point for point on real drawn deltas, tables and diagnostics alike.

    This is the acceptance of M6. ``perturb_pfns_partial`` is what the shipped
    PFNS driver runs and it is untouched; the model applier is written beside
    it, and the two are handed the same sample of the same covariance. Equality
    here is what makes them one operation with two spellings — and it covers the
    load-bearing step, because a refinement that put the group boundary in the
    wrong place would move the tables and the renormalisation scalar would hide
    it from every other measurement.
    """
    endf, suite, mf5, bands, grids = _read(name)
    deltas = _deltas(suite)

    fresh = read_endf(str(DATA / name), mf_numbers=[5, 35])
    partial, formatDiagnostics = perturb_pfns_partial(
        fresh.mf[5].mt[18].partials[0], deltas, bands, grids)

    form, _provenance, _report = decodeMF5MT(mf5)
    perturbed, modelDiagnostics = applySpectrumFactors(
        form, dict(enumerate(bands)), dict(enumerate(grids)),
        pfns_ratio_rule(deltas))

    # The equality below is worth nothing if the sample barely moved anything,
    # so the size of what was applied is asserted first. Measured on these two
    # tapes at seed 7: max|r-1| of 0.48 and 1.93, and every band inserts nodes.
    assert modelDiagnostics["total_outgoing_inserted"] > 0
    assert modelDiagnostics["max_group_ratio_error"] > 0.0

    assert list(perturbed.outerDomainValues) == list(partial.incident_energies)
    for k in range(len(partial.incident_energies)):
        xEndf, yEndf = partial.table(k)
        xModel, yModel = perturbed.table(k)
        assert np.array_equal(xEndf, xModel), f"outgoing grid moved at node {k}"
        assert np.array_equal(yEndf, yModel), f"chi moved at node {k}"

    for field in ("max_renormalisation_error", "max_group_ratio_error",
                  "max_group_mass_error", "total_steps_dropped"):
        assert modelDiagnostics[field] == formatDiagnostics[field]
    assert modelDiagnostics["n_outer_inserted"] == \
        formatDiagnostics["n_incident_inserted"]


def test_the_two_appliers_write_a_step_the_same_way():
    """The constants are one number each, in two places, and are gated as such.

    ``NUBAR_STEP_SHOULDER`` set the precedent: an applier pair that must agree
    keeps its own name for the value and a test holds them equal. The shoulder
    is the one that was measurably wrong when copied from nu-bar — an
    energy-relative shoulder spans a tenth of Cf-252's top group.
    """
    assert SPECTRUM_STEP_SHOULDER == OUTGOING_STEP_SHOULDER
    assert SPECTRUM_OUTER_SHOULDER == INCIDENT_STEP_SHOULDER
    assert SPECTRUM_ENERGY_RTOL == ENDF_ENERGY_RTOL


@pytest.mark.parametrize("name", TAPES)
def test_a_zero_perturbation_reproduces_chi_pointwise(name):
    """δ ≡ 0 must give the evaluation back, node inserted or not.

    Every node the applier inserts is an exact refinement, so a run with no
    perturbation in it changes nothing anywhere — including at the shoulders it
    inserts below each band edge, which is precisely where an approximate
    refinement would show up as a central-value shift of a few parts in a
    thousand.
    """
    _endf, suite, mf5, bands, grids = _read(name)
    form, _provenance, _report = decodeMF5MT(mf5)
    zeros = {index: np.zeros(len(grid) - 1) for index, grid in enumerate(grids)}

    perturbed, diagnostics = applySpectrumFactors(
        form, dict(enumerate(bands)), dict(enumerate(grids)),
        pfns_ratio_rule(zeros))

    assert diagnostics["max_renormalisation_error"] < 1e-12
    for k, energy in enumerate(perturbed.outerDomainValues):
        xs, ys = perturbed.table(k)
        assert np.allclose(ys, form.evaluateAtOuter(energy)[1], rtol=0,
                           atol=1e-14 * max(np.max(np.abs(ys)), 1e-300))


@pytest.mark.parametrize("name", TAPES)
def test_the_evaluated_node_is_left_where_it_was(name):
    """The applier returns a new node; the one it was given still holds chi."""
    _endf, suite, mf5, bands, grids = _read(name)
    form, _provenance, _report = decodeMF5MT(mf5)
    before = [form.table(k) for k in range(len(form.outerDomainValues))]

    applySpectrumFactors(form, dict(enumerate(bands)), dict(enumerate(grids)),
                         pfns_ratio_rule(_deltas(suite)))

    assert len(form.outerDomainValues) == len(before)
    for k, (xs, ys) in enumerate(before):
        assert np.array_equal(form.table(k)[0], xs)
        assert np.array_equal(form.table(k)[1], ys)


@pytest.mark.parametrize("name", TAPES)
def test_a_partial_band_request_steps_where_the_perturbation_starts(name):
    """Nodes outside the requested bands are untouched, and the step is stated.

    The factor is piecewise constant in incident energy, so it steps at the
    lower edge of the lowest requested band just as it does at every edge
    between two requested ones -- below it the spectrum is the evaluation. The
    shipped applier skips ``bands[1:]`` there, which is right only because it is
    always handed the whole band list; this one is not, so it asks the domain
    rather than the position.
    """
    _endf, suite, mf5, bands, grids = _read(name)
    form, _provenance, _report = decodeMF5MT(mf5)
    chosen = len(bands) - 1                       # the topmost band, alone
    deltas = {chosen: _deltas(suite)[chosen]}

    perturbed, diagnostics = applySpectrumFactors(
        form, {chosen: bands[chosen]}, {chosen: grids[chosen]},
        pfns_ratio_rule(deltas))

    assert diagnostics["n_nodes"] > 0, "the chosen band has incident nodes"
    assert {entry["band"] for entry in diagnostics["per_node"]} == {chosen}
    assert diagnostics["n_outer_inserted"] == 1, (
        "one shoulder, just below the band the perturbation starts at")

    lower, _upper = bands[chosen]
    for k, energy in enumerate(perturbed.outerDomainValues):
        if energy >= lower:
            continue
        assert np.array_equal(perturbed.table(k)[1],
                              form.evaluateAtOuter(energy)[1]), (
            f"the spectrum moved at {energy:g} eV, below every requested band")


# ======================================================================
# 3. Assembly: one block per band, and it says which band
# ======================================================================

def test_a_band_is_its_own_matrix_and_is_never_merged():
    """MF35's sections are not components of one covariance, unlike MF34's.

    The bands are disjoint in incident energy, ENDF-6 §35 has no record for a
    cross-band block, and their orders differ on a real library. So the default
    ``"mf"`` grouping — which merges everything of one file — has to make an
    exception, and this is it.
    """
    assert 35 in SUPPORTED_MF and 35 in PER_SECTION_MF

    _endf, suite, _mf5, _bands, grids = _read("micro_cf252_pfns.endf")
    entries = collectEntries(suite, {35: None})
    groups = samplingGroups(entries, grouping="mf")

    assert len(groups) == len(grids), "one independent draw per band"
    assert all(len(group) == 1 for group in groups)
    blocks, index = assembleRequest(entries,
                                    domains=componentDomains(suite, {35: None}))
    assert [matrix.shape[0] for _key, matrix in blocks] == \
        [len(grid) - 1 for grid in grids]
    for meta in index.values():
        component, = meta["components"]
        assert meta["domains"][component][0] < meta["domains"][component][1]


def test_asking_for_some_bands_gets_those_bands():
    _endf, suite, _mf5, _bands, _grids = _read("micro_cf252_pfns.endf")
    entries = collectEntries(suite, {35: {"index": [1, 3]}})
    assert sorted(row.index for row, _col, *_ in entries) == [1, 3]


def test_the_band_index_is_the_file_order_not_the_request_order():
    """A partial request must not renumber the bands it did not ask for.

    ``applySpectrumFactors`` looks each band up by key, and the key is the band
    the file states. Renumbering by position would apply band 3's deltas at
    band 1's incident energies, which is a wrong answer that normalises
    perfectly.
    """
    _endf, suite, _mf5, bands, _grids = _read("micro_cf252_pfns.endf")
    domains = componentDomains(suite, {35: {"index": [2]}})
    component, = domains
    assert component.index == 2
    assert domains[component] == pytest.approx(bands[2])


# ======================================================================
# 4. The realisation says what it is
# ======================================================================

def test_the_set_records_the_band_and_the_absolute_semantics(tmp_path):
    """A block that multiplies and a block that is added are not the same thing.

    ``SEMANTICS`` was a one-element tuple until MF35 arrived, on the grounds
    that "the file says relative and the code assumed absolute" is a failure
    that produces plausible numbers. MF35 is the case that makes the statement
    load-bearing rather than decorative.
    """
    _endf, suite, _mf5, bands, _grids = _read("micro_cf252_pfns.endf")
    entries = collectEntries(suite, {35: None})
    domains = componentDomains(suite, {35: None})
    blocks, index = assembleRequest(entries, domains=domains)

    drawn = {key: np.zeros(meta["dimension"]) for key, meta in index.items()}
    pset = PerturbationSet.fromDraw(drawn, index, label="realization-0000")

    assert SEMANTICS_OF_MF[35] == "additive-absolute" != SEMANTICS[0]
    for component in pset.components():
        assert pset.semanticsOf(component) == "additive-absolute"
        assert pset.outerDomains[component] == pytest.approx(
            bands[component.index])

    path = pset.write(tmp_path / "perturbation.json")
    again = PerturbationSet.read(path)
    assert again.outerDomains == pset.outerDomains
    assert again.componentSemantics == pset.componentSemantics
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["blocks"][0]["semantics"] == "additive-absolute"
    assert payload["blocks"][0]["outerDomain"]


def test_a_spectrum_block_without_its_band_is_refused():
    """The band is not optional: without it nothing says where the block acts."""
    component = ComponentKey(98252, 35, 18, 0)
    with pytest.raises(ValueError, match="no incident-energy band"):
        PerturbationSet(label="realization-0000",
                        factors={component: np.zeros(3)},
                        binEdges={component: np.array([1.0, 2.0, 3.0, 4.0])})


def test_a_version_2_set_still_reads():
    """Every block in one is relative with no outer domain — the defaults."""
    component = ComponentKey(26056, 33, 2, 0)
    payload = {
        "format": 2, "label": "realization-0000",
        "semantics": SEMANTICS[0], "edgeRule": "endf-step-duplicate",
        "provenance": {}, "groups": [[list(component)]],
        "blocks": [{"component": list(component), "quantity": "crossSection",
                    "factors": [1.1, 0.9], "binEdges": [1.0, 2.0, 3.0]}],
    }
    pset = PerturbationSet.from_dict(payload)
    assert pset.semanticsOf(component) == SEMANTICS[0]
    assert pset.outerDomains == {}


# ======================================================================
# 5. End to end, through the pipeline that emits
# ======================================================================

def _sections(path):
    """``{(MF, MT): [columns 1-66 of each record]}`` of a tape."""
    out = {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if len(line) < 75:
            continue
        try:
            key = (int(line[70:72]), int(line[72:75]))
        except ValueError:
            continue
        out.setdefault(key, []).append(line[:66])
    return out


def test_a_pfns_run_writes_a_delta_tape_that_only_moved_mf5(tmp_path):
    """The fidelity claim, checked record by record.

    ``endf-delta`` re-encodes the sections a realisation touched and patches
    them into the tape it read, so everything else survives byte for byte. For a
    PFNS run that means MF3 and — the one worth stating — **MF35 itself**: a
    realisation of a covariance does not restate the covariance it came from.

    MF1/451 changes by design and by exactly one record: the directory entry
    counting MF5/MT18's lines, which grew because the perturbation inserted
    nodes.
    """
    source = DATA / "micro_cf252_pfns.endf"
    run = perturbFromModel(str(source), {35: None}, nSamples=2, seed=11,
                           outputDir=tmp_path, formats=("endf-delta",))
    first, second = run.paths("endf-delta")

    original, written = _sections(source), _sections(first)
    assert set(original) == set(written)
    moved = sorted(key for key in original if original[key] != written[key])
    assert (5, 18) in moved
    assert (3, 18) not in moved and (35, 18) not in moved

    directory = [(before, after)
                 for before, after in zip(original[(1, 451)], written[(1, 451)])
                 if before != after]
    assert len(directory) == 1, "only one directory entry may move"
    before, after = (line.split() for line in directory[0])
    assert before[0] == after[0] == "5" and before[1] == after[1] == "18"
    assert int(after[2]) > int(before[2]), "MF5/MT18 grew and the index says so"

    assert first.read_bytes() != second.read_bytes(), (
        "two samples of the same covariance are two different tapes")


def test_the_whole_file_emitters_carry_both_forms(tmp_path):
    """A realisation goes *beside* the evaluation, which is what a GNDS reader sees.

    §9.3's ``realization`` style says "this is a sample of that evaluation", and
    it cannot say it about a form that has displaced the evaluation it came
    from. The ENDF path never sees the difference -- one section per file, so
    the label picks one form and writes it -- which is why the check is on the
    GNDS document.
    """
    run = perturbFromModel(str(DATA / "micro_cf252_pfns.endf"), {35: None},
                           nSamples=1, seed=11, outputDir=tmp_path,
                           formats=("endf-tape", "gnds"))
    document = run.paths("gnds")[0].read_text(encoding="utf-8")
    assert 'label="eval"' in document
    assert 'label="realization-0000"' in document

    written = read_endf(str(run.paths("endf-tape")[0]), mf_numbers=[5])
    partial = written.mf[5].mt[18].partials[0]
    assert len(partial.incident_energies) > 0


def test_the_written_tape_still_normalises(tmp_path):
    """ENDF-6 §5 wants ``int chi dE' = 1``, and the file has to say so.

    Measured on the input evaluation at 4.3e-7, so the bar is the file's own
    residual and not machine precision: the applier preserves whatever the
    integral was, it does not repair it.
    """
    source = DATA / "micro_cf252_pfns.endf"
    run = perturbFromModel(str(source), {35: None}, nSamples=1, seed=11,
                           outputDir=tmp_path, formats=("endf-delta",))

    partial = read_endf(str(run.paths("endf-delta")[0]),
                        mf_numbers=[5]).mf[5].mt[18].partials[0]
    worst = max(abs(partial.normalisation(k) - 1.0)
                for k in range(len(partial.incident_energies)))
    assert worst < 1e-5


def test_the_run_records_the_band_of_every_block(tmp_path):
    """``run_metadata.json`` and the per-sample set both say where each block acts."""
    run = perturbFromModel(str(DATA / "micro_cf252_pfns.endf"), {35: None},
                           nSamples=1, seed=11, outputDir=tmp_path)
    payload = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert len(payload["blocks"]) == 4, "one block per band"

    pset = PerturbationSet.read(run.samples[0]["files"]["perturbation-set"])
    assert len(pset.outerDomains) == 4
    assert {c.index for c in pset.components()} == {0, 1, 2, 3}


def test_the_spectrum_blocks_are_drawn_as_deltas_and_not_as_factors():
    """A relative draw of an absolute covariance would be silently plausible.

    MF35's rows sum to zero because the probabilities sum to one, so a linear
    draw of it is centred on **0** and a factor draw would be centred on 1.
    Nothing downstream would raise either way — the projection would absorb it —
    so the distinction is checked where it is made.
    """
    from kika.sampling.model_perturbation import _drawEverything, _splitBySemantics

    _endf, suite, _mf5, _bands, _grids = _read("micro_cf252_pfns.endf")
    entries = collectEntries(suite, {35: None})
    blocks, index = assembleRequest(
        entries, domains=componentDomains(suite, {35: None}))

    factorBlocks, deltaBlocks = _splitBySemantics(blocks, index)
    assert not factorBlocks and len(deltaBlocks) == len(blocks)

    samples, _diagnostics = _drawEverything(
        blocks, index, 8, seed=3, space="log", decompositionMethod="svd",
        samplingMethod="sobol", psdMethod="auto", nullTol=None, logger=None)

    from kika.sampling.core import draw_samples
    asFactors, _info = draw_samples(blocks, 8, space="linear",
                                    returns="factors", seed=3,
                                    psd_method="auto", null_tol=None,
                                    verbose=False)
    for key, drawn in samples.items():
        # The discrimination, stated rather than assumed: the same block drawn
        # as factors is the same numbers plus one, so "centred on zero" is a
        # test only because "centred on one" is what the other convention gives.
        assert abs(float(np.mean(drawn))) < 1e-3
        assert float(np.mean(asFactors[key])) == pytest.approx(
            1.0 + float(np.mean(drawn)))


def test_two_realisations_of_one_suite_both_start_from_the_evaluation():
    """A second sample must not perturb the first sample's spectrum.

    The applier reads ``distribution[EVAL_LABEL]`` and writes beside it, so
    compounding cannot happen — and that is worth a test rather than a comment,
    because a labelled form left on the node between samples is exactly the
    defect ``_forget`` exists to prevent and the symptom would be an ensemble
    that drifts in one direction.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite
    from kika.nuclear_data.model import EVAL_LABEL

    # The whole tape, not just MF5 and MF35: a distribution hangs on a product
    # of a *reaction*, and without MF3 there is no MT18 to hang it on -- which
    # `decodeReactionSuite` reports rather than inventing.
    endf = read_endf(str(DATA / "micro_cf252_pfns.endf"))
    covariances, _r1 = decodeCovarianceSuite(endf)
    suite, _r2 = decodeReactionSuite(endf)
    entries = collectEntries(covariances, {35: None})
    domains = componentDomains(covariances, {35: None})
    _blocks, index = assembleRequest(entries, domains=domains)

    drawn = {key: np.zeros(meta["dimension"]) for key, meta in index.items()}
    evaluated = suite.reactionByENDF_MT(18).outputChannel.products[0]         .distribution[EVAL_LABEL].energy
    before = [evaluated.table(k) for k in range(len(evaluated.outerDomainValues))]

    for label in ("realization-0000", "realization-0001"):
        PerturbationSet.fromDraw(drawn, index, label=label).applyToSuite(suite)

    for k, (xs, ys) in enumerate(before):
        assert np.array_equal(evaluated.table(k)[0], xs)
        assert np.array_equal(evaluated.table(k)[1], ys)

    product = suite.reactionByENDF_MT(18).outputChannel.products[0]
    first = product.distribution["realization-0000"].energy
    second = product.distribution["realization-0001"].energy
    assert len(first.outerDomainValues) == len(second.outerDomainValues)
    for k in range(len(first.outerDomainValues)):
        assert np.array_equal(first.table(k)[1], second.table(k)[1]), (
            "two zero-perturbation realisations of one evaluation differ, so "
            "one of them was built on the other")


def test_a_run_without_a_spectrum_draws_exactly_what_it_drew_before():
    """The seed ladder is continued, not restarted, and an unmixed run is unmoved.

    Splitting the draw in two is what lets one request carry blocks with
    different semantics. It would be a poor trade if it moved every existing
    ensemble, so the factor blocks keep the seeds they had — which is what this
    asserts against ``draw_samples`` called directly.
    """
    from kika.sampling.core import draw_samples
    from kika.sampling.model_perturbation import _drawEverything

    endf = read_endf(str(DATA / "micro_fe56_cov.endf"))
    from kika.endf.model_adapter import decodeCovarianceSuite

    covariances, _report = decodeCovarianceSuite(endf)
    entries = collectEntries(covariances, {33: None})
    blocks, index = assembleRequest(entries)

    split, _diagnostics = _drawEverything(
        blocks, index, 4, seed=5, space="log", decompositionMethod="svd",
        samplingMethod="sobol", psdMethod="none", nullTol=None, logger=None)
    direct, _info = draw_samples(blocks, 4, space="log", returns="factors",
                                 decomposition_method="svd",
                                 sampling_method="sobol", seed=5,
                                 psd_method="none", null_tol=None,
                                 verbose=False)
    assert set(split) == set(direct)
    for key in direct:
        assert np.array_equal(split[key], direct[key])
