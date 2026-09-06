"""Perturb on the model, write to whichever format is asked for.

This is the pipeline the four single-quantity ones do not add up to. It reads a
tape once, assembles whatever covariance the request names -- cross sections,
angular distributions, multiplicities, in any combination -- decides from the
file which of them have to be drawn together, draws, puts each realisation on
the model nodes it belongs to, and emits.

It is a **fork, not a flag.** ``perturb_PENDF_files`` and ``perturb_ENDF_files``
are untouched, and the reason is risk to a finished result rather than tidiness:
they are what the thesis ensembles were drawn through, so a mode switch inside
them would put this code on a path that must not move. Nothing here imports
them except to compare against in tests.

Three things it does differently, each of them deliberate:

* **No ``autofix``.** Conditioning is a pre-flight a human runs and approves --
  :mod:`kika.cov.conditioning` -- and the plan it produces goes in the run
  directory next to the realisations. A new pipeline starts without autofix so
  that nobody has to take it away later; ``psd_method`` therefore defaults to
  ``"none"`` here, because by the time the draw happens the repairs have already
  been made and named.
* **One suite, labelled forms, no ``deepcopy``.** §9.1's multi-form container
  holds the realisation beside the evaluation, which is what it is for; the
  label is removed again once the sample is written, so memory does not grow
  with the ensemble. The measurement that decided this is D6 in
  ``docs/library/perturbation_model_roadmap.md``.
* **The emitters declare what they cannot carry.** ``endf-delta`` re-encodes
  only the files a realisation touched and patches them into the tape that was
  read, so everything else survives verbatim and ``cmp`` between two samples
  still means something. ``endf-tape`` and ``gnds`` write the whole model, which
  is less than the tape holds -- MF7, MF12-15 and MF32 have no model to come
  from -- and the report says so rather than the file being quietly shorter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kika.sampling.joint_blocks import (assembleRequest, collectEntries,
                                        describeRequest)
from kika.sampling.perturbation_set import PerturbationSet

__all__ = ["perturbFromModel", "RunResult", "EMITTERS"]

#: The formats a realisation can be written as, and what each one is for.
#:
#: ``"endf-delta"``
#:     Re-encode only the sections the realisation touched and patch them into
#:     the source tape. The fidelity option: MF7, MF12-15, MF32 and anything
#:     else the model does not carry survive byte for byte, because they are
#:     never re-rendered. This is what a propagation run wants.
#: ``"endf-tape"``
#:     ``kika.write(..., format='endf')``: the whole tape assembled from the
#:     model, under the realisation's label with a fall back to ``eval`` for
#:     everything the draw did not cover. Says in its report what the model
#:     could not carry.
#: ``"gnds"``
#:     ``kika.write(..., format='gnds')``: a ``reactionSuite`` carrying **both**
#:     forms -- the evaluated one and the realisation, each under its own label,
#:     which is what §9.3's ``realization`` style is for and what the GNDS writer
#:     already does with a multi-form container. The covariance is not rewritten:
#:     a realisation does not restate the matrix it came from.
EMITTERS = ("endf-delta", "endf-tape", "gnds")


@dataclass
class RunResult:
    """What a run produced, and what it decided on the way."""

    label: str
    outputDir: Optional[Path]
    samples: List[Dict[str, Any]] = field(default_factory=list)
    index: Dict[Hashable, Dict[str, Any]] = field(default_factory=dict)
    grouping: str = "mf"
    description: str = ""
    diagnostics: Dict[Hashable, Dict[str, Any]] = field(default_factory=dict)
    #: Things true of every sample that no output file states. Read them: a run
    #: that perturbs a summed cross section and its partials writes a tape whose
    #: total is not the sum of its parts, and only this says so.
    notes: List[str] = field(default_factory=list)

    @property
    def nSamples(self) -> int:
        return len(self.samples)

    def paths(self, emitter: str) -> List[Path]:
        """Every file one emitter wrote, in sample order."""
        return [sample["files"][emitter] for sample in self.samples
                if emitter in sample["files"]]


def _redundancyNote(suite, pset) -> Optional[str]:
    """Whether this realisation perturbs a summed cross section and its parts.

    **This does not repair anything, and saying so is the point.** ENDF states
    MT1 and MT4 as ordinary sections, so a request for "every MT the file states"
    routinely names a sum *and* its partials, and each is perturbed from its own
    covariance block. The tape that comes out therefore has MT1 that is no longer
    the sum of what it sums -- which is what ``apply_factors_to_pendf_mf3`` half
    solves for the PENDF case, by expanding a composite over its partials and
    excluding the partials that are perturbed in their own right.

    Re-deriving here is decision 3 of the roadmap (Juan, 2026-08-29: the sum is
    re-derived when the partials are perturbed) and it **moves numbers**, so it
    is not something to slip into a pipeline unannounced. What it needs first is
    for the applier to know which MT sums which, and the ENDF-decoded model does
    not carry that: ``f982268`` puts a summed MT in ``suite.sums``, but §25's
    ``<summands/>`` comes out empty because ENDF says nowhere what MT1 is made
    of, and ``kika._constants.MT_COMPOSITES`` only approximates it.

    So the run records the fact instead of quietly leaving it out of the file.
    """
    reactions = getattr(getattr(suite, "sums", None), "reactions", None)
    if not reactions:
        return None
    # `Sums.reactions` is a plain list of `Reaction`s -- the ENDF adapter fills
    # it that way because ENDF says nowhere what MT1 sums, so there is no
    # `CrossSectionSum` to build. Asking it for `byENDF_MT` raised
    # AttributeError, which an over-broad `except` then swallowed: the note
    # could never fire and the run said nothing. Read the list.
    summedMTs = {int(reaction.ENDF_MT) for reaction in reactions
                 if getattr(reaction, "ENDF_MT", None) is not None}
    summed = [mt for mt in pset.reactions() if mt in summedMTs]
    if not summed or len(pset.reactions()) < 2:
        return None
    return (
        f"MT{summed} is a summed cross section and this realisation also "
        f"perturbs {[mt for mt in pset.reactions() if mt not in summed]}: each "
        f"is scaled by its own factor block, so the sum is NOT re-derived and "
        f"the tape states a total that is not the sum of its parts. Re-deriving "
        f"is decision 3 of docs/library/perturbation_model_roadmap.md and moves "
        f"numbers; it is not done here"
    )


def _sumRuleNote(applied) -> Optional[str]:
    """Whether this realisation rebuilt a nu-bar from the sum rule, and what it cost.

    Rebuilding the redundant member repairs whatever the *input* evaluation was
    off by, and that moves the central value by something the perturbation never
    asked for. The rule is right and the repair is unavoidable; what is not
    acceptable is it being invisible, so the size of it goes in the run's own
    account of itself.
    """
    for component, info in applied.items():
        if not info.get("derived_from_sum_rule"):
            continue
        residual = info.get("baseline_residual") or {}
        size = residual.get("max_bin_rel")
        return (
            f"MT{component.mt} was rebuilt from the sum rule on "
            f"{info.get('n_points')} points, from MT{info.get('contributors')}: "
            f"its own factor block is discarded, which is the convention "
            f"perturb_nubar_family states. Rebuilding also repairs the input "
            f"file's own residual"
            + (f", which was {size:.2e} at worst per bin" if size is not None
               else " (not measurable on this family)")
        )
    return None


def _readTape(source):
    """*source* as ``(ENDF object, path or None)``."""
    from kika.endf import read_endf

    if isinstance(source, (str, Path)):
        return read_endf(str(source)), Path(source)
    return source, None


def _touchedFiles(pset: PerturbationSet, alsoChanged=()) -> Dict[int, List[int]]:
    """Which ENDF files a realisation changed, and which MTs of each.

    The mapping is from the *model node* back to ENDF, and it is not the MF the
    covariance came from: MF34's Legendre order 0 is the magnitude, it lands on
    the cross section, and therefore it changes **MF3**. Getting this wrong
    writes a tape whose MF3 is perturbed and whose MF1 directory says it is not.

    *alsoChanged* is the components the **applier** moved that the request did
    not name -- today exactly one thing: the nu-bar the sum rule derives. It has
    to be here rather than inferred from the request, because a realisation that
    perturbs MT455 and MT456 also rewrites MT452, and a delta that wrote only
    what was asked for would leave the tape stating a total that is not the sum
    of the parts it just changed. That was the first thing this emitter got
    wrong, and it looked right: two files written, both perturbed, and the third
    silently stale.
    """
    touched: Dict[int, set] = {}
    for component in list(pset.components()) + list(alsoChanged):
        if component.mf == 33 or (component.mf == 34 and component.index == 0):
            touched.setdefault(3, set()).add(component.mt)
        elif component.mf == 34:
            touched.setdefault(4, set()).add(component.mt)
        elif component.mf == 31:
            touched.setdefault(1, set()).add(component.mt)
    return {mf: sorted(mts) for mf, mts in sorted(touched.items())}


def _emitEndfDelta(suite, endfObj, sourcePath, pset, outPath, report,
                   alsoChanged=()) -> Path:
    """Re-encode the touched sections and patch them into the source tape.

    Everything not re-encoded is copied through as bytes, which is what keeps a
    perturbed tape comparable to the one it came from -- and what makes ``cmp``
    between two samples show exactly the perturbation and nothing else.
    """
    from kika.endf.model_adapter import encodeMF3MT, encodeMF4MT
    from kika.endf.model_adapter.multiplicity import (encodeMF1MT452,
                                                      encodeMF1MT455,
                                                      encodeMF1MT456)
    from kika.endf.writers.endf_writer import ENDFWriter
    from kika.nuclear_data.model import EVAL_LABEL

    _MF1_ENCODERS = {452: encodeMF1MT452, 455: encodeMF1MT455,
                     456: encodeMF1MT456}

    if sourcePath is None:
        raise ValueError(
            "endf-delta patches the tape it read, so it needs the path of that "
            "tape. Pass a path rather than a parsed object, or ask for "
            "endf-tape, which builds the file from the model"
        )

    touched = _touchedFiles(pset, alsoChanged)
    if not touched:
        raise ValueError("this realisation touches nothing; there is no delta")

    current = Path(sourcePath)
    outPath = Path(outPath)
    for mf, mts in touched.items():
        mfFile = endfObj.get_file(mf)
        if mfFile is None:
            raise ValueError(
                f"the realisation perturbs MF{mf} and the tape has no MF{mf}")
        for mt in mts:
            if mf == 3:
                encoded, _report = encodeMF3MT(suite.reactionByENDF_MT(mt),
                                               label=pset.label, report=report)
            elif mf == 4:
                reaction = suite.reactionByENDF_MT(mt)
                product, _angular = PerturbationSet._angularOf(reaction, mt)
                form = product.distribution.get(pset.label) or \
                    product.distribution[EVAL_LABEL]
                encoded, _report = encodeMF4MT(form, product.provenance, mt,
                                               report)
            elif mf == 1:
                # The MF1 encoders take the *suite* and find the node with
                # `nubarNode`, the same lookup the applier perturbed through, so
                # what they write is what was perturbed. That is why a nu-bar
                # realisation has to replace the form rather than sit beside it:
                # there is no label for these encoders to be told about.
                encoded, _report = _MF1_ENCODERS[mt](suite, None, report)
            else:
                raise NotImplementedError(
                    f"no delta encoder for MF{mf}")
            mfFile.sections[mt] = encoded
        writer = ENDFWriter(str(current))
        if not writer.replace_mf_section(mfFile, str(outPath)):
            raise RuntimeError(f"writing MF{mf} of {outPath} failed")
        current = outPath
    return outPath


def _emitWholeFile(suite, pset, outPath, fmt, mat=None) -> Path:
    from kika import write as kikaWrite

    outPath = Path(outPath)
    if fmt == "endf-tape":
        kikaWrite(suite, str(outPath), format="endf", mat=mat, label=pset.label)
    else:
        kikaWrite(suite, str(outPath), format="gnds")
    return outPath


def perturbFromModel(source, request, nSamples: int = 1, *, seed: int = 0,
                     outputDir=None, formats: Sequence[str] = ("endf-delta",),
                     grouping: str = "mf", space: str = "log",
                     labelPrefix: str = "realization",
                     decompositionMethod: str = "svd",
                     samplingMethod: str = "sobol",
                     psdMethod: str = "none",
                     conditioningPlan=None, mat: Optional[int] = None,
                     nullTol: Optional[float] = None,
                     writeSets: bool = True, logger=None) -> RunResult:
    """Draw *nSamples* realisations of *request* and write each one out.

    Parameters
    ----------
    source
        A path to an ENDF tape, or an already parsed ENDF object. The path is
        required by ``endf-delta``, which patches it.
    request
        What to perturb, in any of the spellings
        :func:`~kika.sampling.joint_blocks.collectEntries` accepts --
        ``{33: None, 34: [2]}`` for "every cross section the file states, and
        MT2's angular distribution".
    nSamples, seed, space, decompositionMethod, samplingMethod
        Passed to :func:`~kika.sampling.core.draw_samples`. ``space="log"`` is
        the cross-section default and keeps factors positive.
    psdMethod
        ``"none"`` by default, and that is the design rather than a shortcut:
        the repairs a matrix needs are decided before a run by
        :mod:`kika.cov.conditioning` and applied through *conditioningPlan*, so
        that a run can say which projection it used instead of a log line
        recording that one happened.
    nullTol
        ``None`` -- every direction retained -- because that is what the shipped
        pipelines draw with today, so a comparison against them measures the
        path and not the truncation. See ``draw_samples``.
    formats
        Any of :data:`EMITTERS`.
    writeSets
        Write each realisation's :class:`~kika.sampling.perturbation_set.PerturbationSet`
        as JSON beside its files. On by default: a run that cannot say what it
        drew is not reproducible.

    Returns
    -------
    RunResult
        With one entry per sample carrying its label, its
        :class:`PerturbationSet`, the files written, and the applier's
        diagnostics.
    """
    from kika.cov.conditioning import apply_plan
    from kika.endf.model_adapter import (decodeCovarianceSuite,
                                         decodeReactionSuite)
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.nuclear_data.model.conversion import ConversionReport
    from kika.sampling.core import draw_samples

    unknown = [fmt for fmt in formats if fmt not in EMITTERS]
    if unknown:
        raise ValueError(f"unknown emitter(s) {unknown}; known: {list(EMITTERS)}")

    endfObj, sourcePath = _readTape(source)
    covariances, covReport = decodeCovarianceSuite(endfObj)
    suite, suiteReport = decodeReactionSuite(endfObj)

    entries = collectEntries(covariances, request)
    blocks, index = assembleRequest(entries, grouping=grouping)
    description = describeRequest(entries, grouping=grouping)
    if logger is not None:
        logger.info(description)

    if conditioningPlan is not None:
        blocks = apply_plan(blocks, conditioningPlan)

    samples, drawDiagnostics = draw_samples(
        blocks, nSamples, space=space, returns="factors",
        decomposition_method=decompositionMethod, sampling_method=samplingMethod,
        seed=seed, psd_method=psdMethod, null_tol=nullTol, verbose=False,
        logger=logger)

    outputDir = Path(outputDir) if outputDir is not None else None
    if outputDir is not None:
        outputDir.mkdir(parents=True, exist_ok=True)

    result = RunResult(label=labelPrefix, outputDir=outputDir, index=index,
                       grouping=grouping, description=description,
                       diagnostics=drawDiagnostics)

    stem = sourcePath.stem if sourcePath is not None else "perturbed"
    for number in range(nSamples):
        label = f"{labelPrefix}-{number:04d}"
        pset = PerturbationSet.fromDraw(
            {key: samples[key][number] for key in samples}, index, label=label,
            provenance={"seed": seed, "sample": number, "space": space,
                        "grouping": grouping, "source": str(sourcePath or "")})
        displaced: Dict[Any, Any] = {}
        applied = pset.applyToSuite(suite, multiplicityResolver=nubarNode,
                                    displaced=displaced)

        files: Dict[str, Path] = {}
        report = ConversionReport()
        if outputDir is not None:
            sampleDir = outputDir / f"{number:04d}"
            sampleDir.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                if fmt == "endf-delta":
                    files[fmt] = _emitEndfDelta(
                        suite, endfObj, sourcePath, pset,
                        sampleDir / f"{stem}_{number:04d}.endf", report,
                        alsoChanged=tuple(displaced))
                elif fmt == "endf-tape":
                    files[fmt] = _emitWholeFile(
                        suite, pset, sampleDir / f"{stem}_{number:04d}.tape.endf",
                        fmt, mat=mat)
                else:
                    files[fmt] = _emitWholeFile(
                        suite, pset, sampleDir / f"{stem}_{number:04d}.gnds.xml",
                        fmt)
            if writeSets:
                files["perturbation-set"] = pset.write(
                    sampleDir / "perturbation.json")

        for note in (_redundancyNote(suite, pset), _sumRuleNote(applied)):
            if note is not None and note not in result.notes:
                result.notes.append(note)
                if logger is not None:
                    logger.warning(note)
        result.samples.append({"label": label, "set": pset, "files": files,
                               "applied": applied})
        _forget(suite, pset, displaced)

    if outputDir is not None:
        _writeRunMetadata(result, outputDir, covReport, suiteReport,
                          sourcePath, request, seed, space)
    return result


def _forget(suite, pset: PerturbationSet, displaced=None) -> None:
    """Drop a realisation's forms once it has been written, and put back what it took.

    The labelled form is what replaces the per-sample ``deepcopy``; leaving it in
    place would put the whole ensemble in memory one form at a time, which is the
    cost the copy was rejected for. Removing it is not a cleanup detail -- it is
    the other half of that decision.

    *displaced* is the second half of a different one. A nu-bar realisation has
    no labelled slot to sit in, so it **replaces** the evaluated form; this puts
    the evaluation back, and it is what makes the replacement transient rather
    than a suite that quietly stops carrying its own evaluation after sample 0.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    for component, form in (displaced or {}).items():
        from kika.endf.model_adapter.multiplicity import nubarNode

        node = nubarNode(suite, component.mt)
        if node is not None:
            node.form = form

    for mt in pset.reactions():
        if mt in (452, 455, 456):
            continue
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is None:
            continue
        reaction.crossSection.forms.pop(pset.label, None)
        channel = getattr(reaction, "outputChannel", None)
        for product in (getattr(channel, "products", None) or ()):
            distribution = getattr(product, "distribution", None)
            if distribution is not None and hasattr(distribution, "forms"):
                distribution.forms.pop(pset.label, None)
        assert EVAL_LABEL in reaction.crossSection, (
            "the evaluated form was removed with the realisation")


def _writeRunMetadata(result: RunResult, outputDir: Path, covReport, suiteReport,
                      sourcePath, request, seed: int, space: str) -> Path:
    """The run's own account of itself, beside the samples.

    Deliberately includes the grouping description in full: "these quantities
    were drawn independently" is a claim about the evaluation, and it is the one
    thing about a mixed run that cannot be recovered from the output files.
    """
    path = outputDir / "run_metadata.json"
    payload = {
        "source": str(sourcePath or ""),
        "request": {str(key): value for key, value in request.items()} if isinstance(
            request, Mapping) else str(request),
        "seed": seed,
        "space": space,
        "grouping": result.grouping,
        "nSamples": result.nSamples,
        "groups": result.description,
        "blocks": [
            {"key": str(key), "dimension": meta["dimension"],
             "union": meta["union"], "quantities": meta["quantities"],
             "components": [list(component) for component in meta["components"]]}
            for key, meta in result.index.items()
        ],
        "notes": list(result.notes),
        "covarianceDecode": covReport.summary() if covReport is not None else "",
        "suiteDecode": suiteReport.summary() if suiteReport is not None else "",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
