"""Perturb on the model, write to whichever format is asked for.

This is the pipeline the four single-quantity ones do not add up to. It reads a
tape once, assembles whatever covariance the request names -- cross sections,
angular distributions, multiplicities, fission spectra, in any combination --
decides from the file which of them have to be drawn together, draws, puts each
realisation on the model nodes it belongs to, and emits.

**Not everything it draws is a factor.** MF35's bands are the absolute
covariance of group-integrated probabilities, so they are drawn linearly as
deltas while the other three files are drawn in log space as factors. One
request may carry both; the draw is split by what each covariance *states*, not
by what the caller asked for. See :func:`_splitBySemantics`.

It is a **fork, not a flag.** ``perturb_PENDF_files`` and ``perturb_ENDF_files``
are untouched, and the reason is risk to a finished result rather than tidiness:
they are what the thesis ensembles were drawn through, so a mode switch inside
them would put this code on a path that must not move. Nothing here imports
them except to compare against in tests.

Three things it does differently, each of them deliberate:

* **No ``autofix``.** Conditioning is a pre-flight -- :mod:`kika.cov.conditioning`
  -- and the plan it produces goes in the run directory next to the
  realisations. By default the run inspects its own blocks and applies the
  pre-flight's recommendation, which repairs definiteness and nothing else;
  a caller can hand in an edited plan, or ``False`` to draw the blocks exactly
  as stated. A new pipeline starts without autofix so that nobody has to take
  it away later; ``psd_method`` therefore defaults to ``"none"`` here, because
  by the time the draw happens the repairs have already been made and named.
  The whole argument is in ``docs/library/autofix_in_the_model.md``.

* **It keeps a log of events, and it can rehearse.** Every stage is a timed
  event with its numbers on a :class:`~kika.sampling.run_log.RunLog`, written
  as ``run.log.jsonl`` and rendered as ``run.log``; ``dryRun=True`` does all of
  it but the tapes, so a run can be checked on a laptop before it is launched.

* **One factors table per run.** Every realisation goes to ``factors.parquet``
  with one ``factors_index.json``, rather than a JSON per sample.
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

from kika.cov.conditioning import block_key_text
from kika.sampling.joint_blocks import (assembleRequest, collectEntries,
                                        componentDomains, describeRequest)
from kika.sampling.perturbation_set import PerturbationSet

__all__ = ["perturbFromModel", "RunResult", "EMITTERS", "TAPE_EMITTERS", "AceOptions"]

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
#: ``"ace"``
#:     NJOY on the realisation's ENDF tape, one ACE (and its xsdir line) per
#:     temperature, under ``NNNN/ace/``; NJOY's own input, listing and output
#:     under ``NNNN/njoy/``. Needs an :class:`AceOptions` -- which NJOY, which
#:     temperatures, which library -- and an ENDF tape to feed it: the
#:     ``endf-delta`` of the sample when the source is a tape, otherwise the
#:     ``endf-tape`` written from the model, produced on demand if neither was
#:     asked for. A dry run validates the options (the executable exists, the
#:     temperatures are given) and runs nothing.
EMITTERS = ("endf-delta", "endf-tape", "gnds", "ace")

#: The emitters that need nothing beyond the run itself -- every one but
#: ``"ace"``, which needs NJOY and an :class:`AceOptions`.
TAPE_EMITTERS = ("endf-delta", "endf-tape", "gnds")


@dataclass(frozen=True)
class AceOptions:
    """What producing ACE needs beyond the perturbation itself.

    Parameters
    ----------
    temperatures
        Kelvin, one ACE per value. A value below 1.0 is read as MeV, as the
        NJOY runner has always done.
    njoyExe
        Path to the NJOY executable. Defaults to the ``NJOY_EXE`` environment
        variable; a run that cannot find NJOY is refused before anything is
        drawn, and a dry run says so too.
    libraryName
        For the ACE title and the library digit of the suffix -- ``'jeff40'``,
        ``'endfb81'``, see ``kika._constants.NDLIBRARY_TO_SUFFIX``.
    extensions
        One ACE extension per temperature (``'02c'``, ``'06c'`` ...). When
        given, the ACE is ``<ZAID>.<ext>`` and the xsdir line beside it; when
        not, the runner's legacy naming from the temperature applies.
    njoyVersion
        The version string NJOY's title carries.
    keepNjoyFiles
        Keep NJOY's input, output and listing beside the ACE. On by default:
        an ACE that came out wrong is diagnosed from the listing, and the run
        that produced it is the only chance to keep one.
    """

    temperatures: Tuple[float, ...] = ()
    njoyExe: Optional[str] = None
    libraryName: str = "endfb81"
    extensions: Optional[Tuple[str, ...]] = None
    njoyVersion: str = "NJOY 2016.78"
    keepNjoyFiles: bool = True

    def __post_init__(self) -> None:
        import os

        temperatures = tuple(float(t) for t in (
            (self.temperatures,) if isinstance(self.temperatures, (int, float))
            else self.temperatures))
        object.__setattr__(self, "temperatures", temperatures)
        if self.extensions is not None:
            object.__setattr__(self, "extensions", tuple(str(e) for e in self.extensions))
        if self.njoyExe is None:
            object.__setattr__(self, "njoyExe", os.environ.get("NJOY_EXE"))

    def validate(self) -> None:
        """Refuse now what NJOY would refuse later, with the reason."""
        if not self.temperatures:
            raise ValueError("AceOptions needs at least one temperature")
        if self.extensions is not None and len(self.extensions) != len(self.temperatures):
            raise ValueError(
                f"{len(self.extensions)} extension(s) for {len(self.temperatures)} "
                f"temperature(s): give one per temperature, or none")
        if not self.njoyExe:
            raise ValueError(
                "no NJOY executable: pass AceOptions(njoyExe=...) or set the "
                "NJOY_EXE environment variable")
        if not Path(self.njoyExe).is_file():
            raise FileNotFoundError(f"NJOY executable not found: {self.njoyExe}")

    def to_dict(self) -> Dict[str, Any]:
        return {"temperatures": list(self.temperatures), "njoyExe": self.njoyExe,
                "libraryName": self.libraryName,
                "extensions": list(self.extensions) if self.extensions else None,
                "njoyVersion": self.njoyVersion, "keepNjoyFiles": self.keepNjoyFiles}


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
    #: The conditioning plan the run was handed, and one record per step
    #: :func:`~kika.cov.conditioning.apply_plan` actually ran on the assembled
    #: blocks. Empty when no plan was given -- which means the blocks were
    #: drawn exactly as the file states them, under ``psdMethod``.
    conditioningPlan: Any = None
    conditioning: Tuple[Dict[str, Any], ...] = ()
    #: How the plan came to be: ``"auto"`` (the pre-flight's recommendation,
    #: the default), ``"explicit"`` (handed in) or ``"none"`` (refused).
    conditioningMode: str = "none"
    #: The :class:`~kika.cov.conditioning.ConditioningReport` an ``auto`` run
    #: made, so a caller can read the findings without re-inspecting.
    report: Any = None
    #: The run's :class:`~kika.sampling.run_log.RunLog`: every stage, timed,
    #: with its numbers. Written beside the samples as ``run.log.jsonl`` and
    #: ``run.log``.
    log: Any = None
    dryRun: bool = False
    #: The request in its MF/MT spelling, whatever spelling it was given in.
    request: Any = None
    sourceFormat: str = "endf"
    aceOptions: Any = None
    #: Run-level files: the factors table and its index, the plan, the
    #: metadata, the log. Per-sample files are on each entry of ``samples``.
    files: Dict[str, Path] = field(default_factory=dict)

    @property
    def nSamples(self) -> int:
        return len(self.samples)

    def paths(self, emitter: str) -> List[Path]:
        """Every file one emitter wrote, in sample order.

        For ``"ace"`` that is every ACE of every sample, temperatures in order.
        """
        out: List[Path] = []
        for sample in self.samples:
            value = sample["files"].get(emitter)
            if value is None:
                continue
            out.extend(value if isinstance(value, list) else [value])
        return out

    def aceFailures(self) -> List[Tuple[int, float, Optional[int]]]:
        """``(sample, temperature, return code)`` for every NJOY run that failed."""
        return [(number, record["temperature"], record["returncode"])
                for number, sample in enumerate(self.samples)
                for record in sample.get("ace", ())
                if record["returncode"] != 0 or not record["ace"]]


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


def _spectrumNote(applied) -> Optional[str]:
    """How much of the spectrum an MF35 realisation did *not* perturb.

    An MF5 table and its MF35 grid do not have to cover the same range, and on
    real evaluations they do not: ENDF/B-VIII.1's tables start at ``E'=0`` where
    MF35 starts at 1e-5, and Cf-252's top group runs past the end of some
    tables. The mass outside the bands' groups is left where it is and then
    carried by the renormalisation scalar, so it moves *with* the spectrum and
    is never perturbed in its own right.

    That is the right thing to do -- there is no covariance for it -- and it is
    invisible in the output tape, which is exactly what this list is for. A run
    whose bands cover 99.99 % of the spectrum and one whose bands cover 80 % of
    it write files that look alike and mean different things.
    """
    worst = 0.0
    bands = []
    for component, info in applied.items():
        if component.mf != 35:
            continue
        bands.append(component.index)
        worst = max(worst, float(info.get("max_frac_mass_outside_bands", 0.0)))
    if not bands:
        return None
    return (
        f"MF35/MT{sorted({c.mt for c in applied if c.mf == 35})}: band(s) "
        f"{sorted(bands)} perturb the spectrum only where their groups reach "
        f"it. At worst {worst:.2e} of an incident node's integral falls outside "
        f"them; that mass is not perturbed, it is only carried by the "
        f"renormalisation that puts the integral back where it was"
    )


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
    seen = set()
    for component in list(pset.components()) + list(alsoChanged):
        if component in seen:
            continue
        seen.add(component)
        if component.mf == 33 or (component.mf == 34 and component.index == 0):
            touched.setdefault(3, set()).add(component.mt)
        elif component.mf == 34:
            touched.setdefault(4, set()).add(component.mt)
        elif component.mf == 31:
            touched.setdefault(1, set()).add(component.mt)
        elif component.mf == 35:
            # Every band of one MT rewrites the same MF5 section, so the set
            # collapses them to one entry -- which is what a delta emitter
            # needs, because re-encoding that section four times would write
            # the same file four times and call it a fidelity guarantee.
            touched.setdefault(5, set()).add(component.mt)
    return {mf: sorted(mts) for mf, mts in sorted(touched.items())}


def _emitEndfDelta(suite, endfObj, sourcePath, pset, outPath, report,
                   alsoChanged=()) -> Path:
    """Re-encode the touched sections and patch them into the source tape.

    Everything not re-encoded is copied through as bytes, which is what keeps a
    perturbed tape comparable to the one it came from -- and what makes ``cmp``
    between two samples show exactly the perturbation and nothing else.
    """
    from kika.endf.model_adapter import encodeMF3MT, encodeMF4MT, encodeMF5MT
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
            elif mf == 5:
                # The MF5 encoder needs the provenance decodeMF5MT produced --
                # NK, LF, U, p(E) and the verbatim records of every law kika
                # does not model are not recoverable from the node -- and that
                # provenance lives on the product, beside MF4's. So this asks
                # the same product the applier perturbed rather than looking
                # the section up again: what gets written is what was moved.
                reaction = suite.reactionByENDF_MT(mt)
                product, _energy = PerturbationSet._energyOf(reaction, mt)
                form = product.distribution.get(pset.label) or                     product.distribution[EVAL_LABEL]
                encoded, _report = encodeMF5MT(form.energy, product.provenance,
                                               mt, report)
            elif mf == 1:
                # The MF1 encoders take the *suite* and find the node with
                # `nubarNode` -- the same lookup the applier perturbed through,
                # so what they write is what was perturbed -- and `label=` picks
                # the realisation's form from the container, falling back to
                # `eval` for a member this draw did not touch. Before
                # `Multiplicity` was a `Component` there was no label to pass and
                # the realisation had to displace the evaluated form; passing it
                # is what replaces that.
                encoded, _report = _MF1_ENCODERS[mt](suite, None, report,
                                                     label=pset.label)
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


def _emitAce(endfPath: Path, sampleDir: Path, options: AceOptions, log,
             sample: int) -> List[Dict[str, Any]]:
    """NJOY on one sample's tape, once per temperature. Records, never raises.

    A failed NJOY run is an ``error`` event naming the sample and the
    temperature, with NJOY's listing kept for the reader; the run goes on to
    the next sample, because one bad ACE out of a thousand is something to
    report, not something to lose the other 999 over.
    """
    import shutil
    import tempfile

    from kika.njoy.run_njoy import run_njoy

    aceDir = sampleDir / "ace"
    njoyDir = sampleDir / "njoy"
    aceDir.mkdir(parents=True, exist_ok=True)
    njoyDir.mkdir(parents=True, exist_ok=True)
    produced: List[Dict[str, Any]] = []
    for at, temperature in enumerate(options.temperatures):
        extension = options.extensions[at] if options.extensions else None
        record: Dict[str, Any] = {"temperature": temperature, "extension": extension,
                                  "returncode": None, "ace": None, "xsdir": None,
                                  "listing": None}
        with log.timed("emitted", f"ace at {temperature:g} K", subject="ace",
                       sample=sample, temperature=temperature) as info:
            with tempfile.TemporaryDirectory(prefix="njoy_", dir=sampleDir) as scratch:
                result = run_njoy(
                    njoy_exe=options.njoyExe, endf_path=endfPath,
                    temperature=temperature, library_name=options.libraryName,
                    output_dir=scratch, njoy_version=options.njoyVersion,
                    additional_suffix=f"{sample:04d}" if extension is None else None,
                    extension=extension, ace_dir=aceDir, xsdir_dir=aceDir,
                    njoy_files_dir=njoyDir if options.keepNjoyFiles else Path(scratch),
                )
            record["returncode"] = int(result.get("returncode", -1))
            record["ace"] = result.get("ace_file")
            record["xsdir"] = result.get("xsdir_file")
            record["listing"] = result.get("njoy_listing") or result.get("njoy_output")
            info["returncode"] = record["returncode"]
            if record["ace"]:
                info["path"] = str(record["ace"])
                info["bytes"] = Path(record["ace"]).stat().st_size
        if record["returncode"] != 0 or not record["ace"]:
            log.error(f"NJOY failed at {temperature:g} K (return code "
                      f"{record['returncode']}); listing: {record['listing']}",
                      subject="ace", sample=sample, temperature=temperature,
                      returncode=record["returncode"], listing=record["listing"])
        produced.append(record)
    return produced


def _matOf(suite) -> Optional[int]:
    """The ENDF MAT of the suite's target, from the library's own table.

    An ENDF-decoded suite carries its MAT in the provenance and the writer
    reads it there. A GNDS-decoded one has none -- MAT is not a GNDS concept
    -- so the tape has to be stamped with the standard assignment for the
    target: ``kika._constants.ZAID_TO_ENDF_MAT``, the ground state, or the
    isomer table when the PoPs id carries an ``_e<n>`` suffix. Refusing and
    asking the user for a number the library already knows would be a poor
    default.
    """
    from kika._constants import ZAID_TO_ENDF_MAT, ZAID_TO_ENDF_MAT_ISOMER
    from kika.sampling.model_blocks import _za_of

    za = _za_of(suite, None)
    if not za:
        return None
    target = str(getattr(suite, "target", "") or "")
    if "_e" in target and target.split("_e", 1)[1].isdigit() \
            and int(target.split("_e", 1)[1]) > 0:
        return ZAID_TO_ENDF_MAT_ISOMER.get(za) or ZAID_TO_ENDF_MAT.get(za)
    return ZAID_TO_ENDF_MAT.get(za)


def _blockLabel(key, meta) -> str:
    """``MF34: MT2 L=1, L=2, L=3 (ZA 26056)`` -- what a log line calls a block.

    The block key is a tuple of :class:`ComponentKey` tuples and reads as
    exactly that; a subject a person scans down a column needs to be short,
    and the same in the report and the timeline.
    """
    components = list(meta["components"])
    mfs = sorted({c.mf for c in components})
    zas = sorted({c.za for c in components})
    parts = []
    for c in components:
        tail = ""
        if c.mf == 34:
            tail = f" L={c.index}"
        elif c.mf == 35:
            tail = f" band={c.index}"
        parts.append(f"MT{c.mt}{tail}")
    head = "+".join(f"MF{mf}" for mf in mfs)
    body = ", ".join(parts[:6]) + (f", ... ({len(parts)} components)" if len(parts) > 6 else "")
    za = f" (ZA {', '.join(str(z) for z in zas)})" if zas else ""
    return f"{head}: {body}{za}"


def _describeRequestForPeople(request) -> str:
    from kika.sampling.joint_blocks import QUANTITY_OF_MF, Selection

    bits = []
    for mf, value in (request.items() if isinstance(request, Mapping) else ()):
        name = QUANTITY_OF_MF.get(int(mf), f"MF{mf}")
        if value is None or value is True:
            bits.append(f"{name}: every reaction stated")
            continue
        sel = value if isinstance(value, Selection) else None
        mt = sel.mt if sel else (value.get("mt") if isinstance(value, dict) else value)
        index = sel.index if sel else (value.get("index") if isinstance(value, dict) else None)
        what = "every reaction" if mt is None else \
            ", ".join(f"MT{m}" for m in (mt if isinstance(mt, (list, tuple)) else [mt]))
        if index is not None:
            coordinate = "orders" if int(mf) == 34 else "bands" if int(mf) == 35 else "index"
            values = index if isinstance(index, (list, tuple)) else [index]
            what += f" {coordinate} {', '.join(str(v) for v in values)}"
        bits.append(f"{name}: {what}")
    return "; ".join(bits) if bits else str(request)


def _splitBySemantics(blocks, index):
    """``(factorBlocks, deltaBlocks)`` -- what multiplies, and what is added.

    **Two draws and not one, because ``space`` is a property of the covariance
    and not of the run.** MF31, MF33 and MF34 state relative covariances, so
    their blocks are drawn in log space and come back as factors that stay
    positive. MF35's bands are the absolute covariance of group probabilities
    that sum to one, where ``C.1 ~ 0`` -- so a *linear* draw satisfies the
    normalisation constraint to within the drift the file itself carries, and a
    log draw of it would be arithmetic on the wrong object.
    ``mf35_sampling.SAMPLING_SPACE`` states that as a constant so it cannot be
    "fixed" back to log by analogy with nu-bar.

    A block is homogeneous by construction -- MF35 is never merged with another
    file, see ``joint_blocks.PER_SECTION_MF`` -- and that is checked rather than
    assumed, because a mixed block would be drawn under one of the two
    conventions and be wrong under the other with nothing to show for it.
    """
    from kika.sampling.perturbation_set import SEMANTICS, SEMANTICS_OF_MF

    factorBlocks, deltaBlocks = [], []
    for key, matrix in blocks:
        semantics = {SEMANTICS_OF_MF.get(component.mf, SEMANTICS[0])
                     for component in index[key]["components"]}
        if len(semantics) > 1:
            raise ValueError(
                f"block {key} holds components of {sorted(semantics)}: one "
                f"matrix cannot be drawn as factors and as absolute deltas at "
                f"once. MF35 is assembled per section for this reason, so a "
                f"mixed block means the grouping stopped honouring it"
            )
        if semantics == {SEMANTICS[1]}:
            deltaBlocks.append((key, matrix))
        else:
            factorBlocks.append((key, matrix))
    return factorBlocks, deltaBlocks


def _drawEverything(blocks, index, nSamples, *, seed, space, decompositionMethod,
                    samplingMethod, psdMethod, nullTol, logger):
    """Draw every block, each under the convention its covariance states.

    Returns the merged ``(samples, diagnostics)``, keyed exactly as a single
    call would have keyed them.

    **The seed ladder is continued rather than restarted.**
    :func:`~kika.sampling.core.draw_samples` gives block *i* of a call the seed
    ``seed + i * BLOCK_SEED_STRIDE``, counting from that call's own first block.
    Two calls that both started at *seed* would hand two different blocks the
    same stream, which is a correlation nobody asked for and nothing would
    report. So the second call starts where the first left off, and the result
    is the ladder one call would have produced. A run with no MF35 in it draws
    exactly what it drew before this function existed: the delta list is empty,
    the factor list is the whole thing in the same order, and the offset is
    zero.
    """
    from kika.sampling.core import BLOCK_SEED_STRIDE, draw_samples

    factorBlocks, deltaBlocks = _splitBySemantics(blocks, index)
    samples: Dict[Hashable, np.ndarray] = {}
    diagnostics: Dict[Hashable, Dict[str, Any]] = {}

    if factorBlocks:
        drawn, info = draw_samples(
            factorBlocks, nSamples, space=space, returns="factors",
            decomposition_method=decompositionMethod,
            sampling_method=samplingMethod, seed=seed, psd_method=psdMethod,
            null_tol=nullTol, verbose=False, logger=logger)
        samples.update(drawn)
        diagnostics.update(info)

    if deltaBlocks:
        from kika.sampling.mf35_sampling import SAMPLING_SPACE

        offset = len(factorBlocks) * BLOCK_SEED_STRIDE
        drawn, info = draw_samples(
            deltaBlocks, nSamples, space=SAMPLING_SPACE, returns="deltas",
            decomposition_method=decompositionMethod,
            sampling_method=samplingMethod, seed=seed + offset,
            psd_method=psdMethod, null_tol=nullTol, verbose=False, logger=logger)
        samples.update(drawn)
        diagnostics.update(info)

    return samples, diagnostics


def _readSource(source, log):
    """*source* as ``(suite, covariances, endfObj, path, format, report)``.

    An ENDF tape is parsed once and decoded twice -- the model and the
    covariance suite -- and the parsed object is kept because ``endf-delta``
    patches it. A GNDS file goes through :func:`kika.read`, which follows the
    ``externalFile`` link to the covariance sibling; there is no tape to patch,
    so ``endf-delta`` is refused for it with the reason.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite

    if not isinstance(source, (str, Path)):
        # An already parsed ENDF object, as before.
        endfObj, path = source, None
        with log.timed("read", "decoded a parsed ENDF object") as info:
            covariances, covReport = decodeCovarianceSuite(endfObj)
            suite, suiteReport = decodeReactionSuite(endfObj)
            info["covarianceDecode"] = covReport.summary()
            info["suiteDecode"] = suiteReport.summary()
        return suite, covariances, endfObj, None, "endf", (covReport, suiteReport)

    from kika._read import sniff_format

    path = Path(source)
    fmt = sniff_format(path)
    if fmt == "gnds":
        import kika

        with log.timed("read", "read as gnds", subject=path.name) as info:
            suite = kika.read(str(path), format="gnds")
            covariances = getattr(suite, "covarianceSuite", None)
            info["decode"] = suite.report.summary()
            info["format"] = "gnds"
        if covariances is None or not getattr(covariances, "covarianceSections", None):
            raise ValueError(
                f"{path.name} brought no covarianceSuite: a GNDS evaluation "
                f"states its covariances in a sibling file named by "
                f"<externalFiles>, and either the link is absent or the file is "
                f"not beside it. The decode report says which: "
                f"{suite.report.summary()}"
            )
        return suite, covariances, None, path, "gnds", (suite.report, suite.report)
    if fmt != "endf":
        raise ValueError(
            f"{path.name} is {fmt!r}; a perturbation source is an ENDF tape or "
            f"a GNDS reactionSuite with its covariance sibling")

    from kika.endf import read_endf

    with log.timed("read", "read as endf", subject=path.name) as info:
        endfObj = read_endf(str(path))
        covariances, covReport = decodeCovarianceSuite(endfObj)
        suite, suiteReport = decodeReactionSuite(endfObj)
        info["covarianceDecode"] = covReport.summary()
        info["suiteDecode"] = suiteReport.summary()
        info["format"] = "endf"
    return suite, covariances, endfObj, path, "endf", (covReport, suiteReport)


def _condition(blocks, index, conditioningPlan, log, labels=None):
    """Stage 2 and 3 of the pre-flight, as the pipeline runs them.

    ``None`` (the default) inspects the assembled blocks and applies
    :meth:`~kika.cov.conditioning.ConditioningReport.recommended_plan`, which
    projects an indefinite block onto the PSD cone and does nothing else -- the
    ``clip`` the draw would otherwise make silently, now with a name and a
    reason in the run directory. ``False`` touches nothing: the blocks are
    drawn as the file states them, which is what an equivalence gate wants. A
    :class:`~kika.cov.conditioning.ConditioningPlan` is applied as given.

    Returns ``(blocks, plan, report, applied, mode)``.
    """
    from kika.cov.conditioning import ConditioningPlan, apply_plan, inspect_blocks
    from kika.sampling.joint_blocks import rowFamilies

    if conditioningPlan is False:
        log.event("inspected", "conditioning skipped by request "
                  "(conditioningPlan=False): blocks drawn as stated", mode="none")
        return blocks, None, None, (), "none"

    report = None
    mode = "explicit"
    if conditioningPlan is None:
        mode = "auto"
        families = rowFamilies(index)
        with log.timed("inspected", f"pre-flight on {len(blocks)} block(s)") as info:
            # `predict=False`: the recommendation only needs the spectrum, and
            # measuring three remedies on every block is the interactive
            # pre-flight's job, not the pipeline's.
            report = inspect_blocks(blocks, families=families, predict=False)
            info["summary"] = report.summary()
            info["samplable"] = report.samplable
            info["faithful"] = report.faithful
        labels = labels or {}
        for block in report.blocks:
            for finding in block.findings:
                text = block_key_text(block.key)
                log.event("inspected", f"{finding.check}: {finding.summary}",
                          subject=labels.get(text, text), block=text,
                          level="warning" if finding.severity != "note" else "info",
                          check=finding.check, severity=finding.severity,
                          **{k: v for k, v in finding.evidence.items()
                             if isinstance(v, (int, float, str, bool))})
        if not report.samplable:
            raise ValueError(
                "the pre-flight says these blocks cannot be sampled as they "
                f"stand and no automatic repair applies: {report.summary()}. "
                "Run kika.cov.conditioning.inspect_blocks on them, decide, and "
                "pass the plan")
        conditioningPlan = report.recommended_plan()
    elif not isinstance(conditioningPlan, ConditioningPlan):
        raise TypeError(
            f"conditioningPlan is None (inspect and apply the recommendation), "
            f"False (touch nothing) or a ConditioningPlan, got "
            f"{type(conditioningPlan).__name__}")

    with log.timed("conditioned", f"applied {len(conditioningPlan.steps)} step(s)",
                   mode=mode) as info:
        blocks, applied = apply_plan(blocks, conditioningPlan)
        info["changed"] = sum(1 for record in applied if record["changed"])
    labels = labels or {}
    for record in applied:
        log.event("conditioned",
                  f"{record['remedy']}" + (" (no change)" if not record["changed"] else ""),
                  subject=labels.get(record["block"], record["block"]),
                  block=record["block"], remedy=record["remedy"],
                  changed=record["changed"], reason=record["reason"],
                  stated_diagonal_max_relative_change=record.get(
                      "stated_diagonal_max_relative_change"))
    return blocks, conditioningPlan, report, tuple(applied), mode


def _checkRealisation(pset: PerturbationSet, log, sample: int) -> None:
    """What a dry run is for: say whether the realisation is one anybody can use.

    Factors under a multiplicative block must be finite and positive -- a
    log-space draw guarantees it and a linear one does not -- and every block
    must be finite. The sum rule and the spectrum normalisation are checked
    where they are applied; their notes reach the log through ``notes``.
    """
    from kika.sampling.perturbation_set import SEMANTICS

    for component in pset.components():
        values = np.asarray(pset.factors[component], dtype=float)
        finite = bool(np.all(np.isfinite(values)))
        multiplicative = pset.semanticsOf(component) == SEMANTICS[0]
        positive = bool(np.all(values > 0.0)) if multiplicative else True
        payload = dict(n=int(values.size),
                       min=float(values.min()) if values.size else float("nan"),
                       max=float(values.max()) if values.size else float("nan"),
                       mean=float(values.mean()) if values.size else float("nan"),
                       semantics=pset.semanticsOf(component))
        if not finite:
            log.error("non-finite value(s) in the realisation",
                      subject=component.describe(), sample=sample, **payload)
        elif not positive:
            log.warning("a multiplicative factor <= 0: the linear draw crossed "
                        "zero; draw in log space or condition the block",
                        subject=component.describe(), sample=sample, **payload)
        else:
            log.event("checked", "finite" + (", positive" if multiplicative else ""),
                      subject=component.describe(), sample=sample, **payload)


def perturbFromModel(source, request, nSamples: int = 1, *, seed: int = 0,
                     outputDir=None, formats: Sequence[str] = ("endf-delta",),
                     grouping: str = "mf", space: str = "log",
                     labelPrefix: str = "realization",
                     decompositionMethod: str = "svd",
                     samplingMethod: str = "sobol",
                     psdMethod: str = "none",
                     conditioningPlan=None, mat: Optional[int] = None,
                     nullTol: Optional[float] = None,
                     dryRun: bool = False,
                     writeFactors: bool = True,
                     writeSets: bool = False,
                     ace: Optional[AceOptions] = None,
                     runLog=None, logger=None) -> RunResult:
    """Draw *nSamples* realisations of *request* and write each one out.

    Parameters
    ----------
    source
        A path to an ENDF tape, a path to a GNDS ``reactionSuite`` whose
        covariance sibling sits beside it, or an already parsed ENDF object.
        ``endf-delta`` needs the ENDF path, because it patches that file.
    request
        What to perturb, in either spelling
        :func:`~kika.sampling.joint_blocks.normaliseRequest` accepts. By
        covariance file: ``{33: None, 34: [2]}`` for "every cross section the
        file states, and MT2's angular distribution", ``{35: None}`` for every
        band of the fission spectrum, ``{35: {"index": [0, 1]}}`` for the two
        lowest. By quantity, naming reactions by MT or by label:
        ``{"crossSection": None, "angularDistribution": {"reaction": "MT2",
        "order": [1, 2, 3]}}``.
    nSamples, seed, space, decompositionMethod, samplingMethod
        Passed to :func:`~kika.sampling.core.draw_samples`. ``space="log"`` is
        the cross-section default and keeps factors positive. **It does not
        reach MF35**: a band's covariance is absolute and its rows sum to zero,
        so those blocks are drawn linearly as deltas whatever this says --
        see :func:`_splitBySemantics`.
    conditioningPlan
        ``None`` (default): inspect the assembled blocks and apply the
        pre-flight's recommendation, which repairs definiteness and nothing
        else; the plan it produced is written to the run directory as if a
        human had handed it in. ``False``: touch nothing and draw the blocks
        as stated, for equivalence gates. A
        :class:`~kika.cov.conditioning.ConditioningPlan`: apply exactly that.
    psdMethod
        ``"none"`` by default, and that is the design rather than a shortcut:
        the repairs a matrix needs are decided before the draw and recorded,
        so that a run can say which projection it used instead of a log line
        recording that one happened.
    nullTol
        ``None`` -- every direction retained -- because that is what the shipped
        pipelines draw with today, so a comparison against them measures the
        path and not the truncation. See ``draw_samples``.
    formats
        Any of :data:`EMITTERS`. ``"ace"`` needs *ace*.
    ace
        An :class:`AceOptions` when ``"ace"`` is among *formats*: NJOY, the
        temperatures, the library. Validated up front, dry run included.
    mat
        ENDF only, and only for a source that carries no MAT (a GNDS file):
        the material number to stamp the tape with. Defaults to the standard
        assignment for the target, from the library's own MAT table.
    dryRun
        Do everything but write tapes: read, assemble, inspect, condition,
        draw, put every realisation on the model and check it, and record it
        all in the log. With an *outputDir* the run-level files still go out
        -- the log, the metadata, the plan and the factors table -- so a dry
        run on a laptop is what a full run on the cluster will do, minus the
        files that take the time.
    writeFactors
        Write every realisation's factors to **one** ``factors.parquet`` with
        its ``factors_index.json`` -- see
        :func:`~kika.sampling.perturbation_set.writeFactorsTable`. On by
        default.
    writeSets
        Also write each realisation's
        :class:`~kika.sampling.perturbation_set.PerturbationSet` as JSON beside
        its files. Off by default since the table exists: a thousand samples
        were a thousand files saying the same bin edges a thousand times.
    runLog, logger
        A :class:`~kika.sampling.run_log.RunLog` to record into, or none to
        make one; a ``logging.Logger`` to forward every event to as a line.

    Returns
    -------
    RunResult
        With one entry per sample carrying its label, its
        :class:`PerturbationSet`, the files written, and the applier's
        diagnostics; the run's :class:`~kika.sampling.run_log.RunLog`; and the
        pre-flight report when the run made one.
    """
    from kika.endf.model_adapter.multiplicity import nubarNode
    from kika.nuclear_data.model.conversion import ConversionReport
    from kika.sampling.joint_blocks import normaliseRequest
    from kika.sampling.perturbation_set import writeFactorsTable
    from kika.sampling.run_log import RunLog

    unknown = [fmt for fmt in formats if fmt not in EMITTERS]
    if unknown:
        raise ValueError(f"unknown emitter(s) {unknown}; known: {list(EMITTERS)}")
    if "ace" in formats:
        if ace is None:
            raise ValueError(
                "formats includes 'ace' and no AceOptions were given: NJOY, "
                "the temperatures and the library have to be named")
        ace.validate()

    log = runLog if runLog is not None else RunLog(logger=logger, label=labelPrefix)
    log.event("started", f"perturbFromModel: {nSamples} sample(s), seed {seed}",
              nSamples=nSamples, seed=seed, space=space, grouping=grouping,
              decompositionMethod=decompositionMethod,
              samplingMethod=samplingMethod, psdMethod=psdMethod,
              dryRun=dryRun, formats=list(formats),
              source=str(source) if isinstance(source, (str, Path)) else "<parsed>",
              outputDir=str(outputDir) if outputDir is not None else None,
              ace=ace.to_dict() if ace is not None else None)

    suite, covariances, endfObj, sourcePath, sourceFormat, reports = \
        _readSource(source, log)
    covReport, suiteReport = reports

    if (sourceFormat == "gnds" and "endf-delta" in formats
            and outputDir is not None and not dryRun):
        raise ValueError(
            "endf-delta patches the ENDF tape it read, and a GNDS source has "
            "none. Ask for endf-tape, gnds or ace"
        )
    if sourceFormat == "gnds":
        needsTape = [f for f in formats if f in ("endf-tape", "ace")]
        hasHeader = bool(getattr(getattr(suite, "provenance", None), "headerFields", None))
        if needsTape and not hasHeader and not dryRun and outputDir is not None:
            raise ValueError(
                f"{needsTape} need an ENDF tape written from the model, and a "
                f"suite read from GNDS cannot be written as one yet: it carries "
                f"no MF1/451 header, no AWR and no Q/LR for its summed reactions, "
                f"and the ENDF encoders refuse to invent them. That is the "
                f"GNDS->ENDF increment deferred as §6.3 of "
                f"docs/library/gnds_endf_conflicts.md; until it lands a GNDS "
                f"source can be perturbed to 'gnds' only. (MAT would be "
                f"{_matOf(suite)} from the library's table when it does.)"
            )
        if mat is None:
            mat = _matOf(suite)
            if mat is not None and needsTape:
                log.note(f"MAT {mat} taken from the library's table for "
                         f"{getattr(suite, 'target', '?')}; the GNDS source states none",
                         mat=mat)

    original = request
    request = normaliseRequest(request, suite)
    log.event("request", _describeRequestForPeople(request),
              request={str(k): _jsonableRequest(v) for k, v in request.items()}
              if isinstance(request, Mapping) else str(request),
              asGiven=_jsonableRequest(original))

    with log.timed("assembled", "assembled the covariance blocks") as info:
        entries = collectEntries(covariances, request)
        domains = componentDomains(covariances, request)
        blocks, index = assembleRequest(entries, grouping=grouping,
                                        domains=domains)
        description = describeRequest(entries, grouping=grouping)
        info["groups"] = len(blocks)
        info["description"] = description
    labels = {block_key_text(key): _blockLabel(key, meta)
              for key, meta in index.items()}
    for key, meta in index.items():
        log.event("assembled", f"{meta['dimension']}x{meta['dimension']}, "
                  f"{len(meta['components'])} component(s), union={meta['union']}",
                  subject=labels[block_key_text(key)], block=block_key_text(key),
                  dimension=meta["dimension"],
                  stride=meta["stride"], union=meta["union"],
                  quantities=list(meta["quantities"]),
                  components=[c.describe() for c in meta["components"]])

    blocks, plan, report, conditioningApplied, conditioningMode = _condition(
        blocks, index, conditioningPlan, log, labels)

    with log.timed("drawn", f"drew {nSamples} sample(s) of {len(blocks)} block(s)"):
        samples, drawDiagnostics = _drawEverything(
            blocks, index, nSamples, seed=seed, space=space,
            decompositionMethod=decompositionMethod,
            samplingMethod=samplingMethod, psdMethod=psdMethod, nullTol=nullTol,
            logger=None)
    for key, diag in drawDiagnostics.items():
        log.event("drawn", f"rank {diag['rank']} of {diag['n']}, "
                  f"{diag['n_null']} null direction(s)",
                  subject=labels[block_key_text(key)], block=block_key_text(key),
                  **{k: v for k, v in diag.items()
                     if isinstance(v, (int, float, str, bool))})

    outputDir = Path(outputDir) if outputDir is not None else None
    if outputDir is not None:
        outputDir.mkdir(parents=True, exist_ok=True)

    result = RunResult(label=labelPrefix, outputDir=outputDir, index=index,
                       grouping=grouping, description=description,
                       diagnostics=drawDiagnostics,
                       conditioningPlan=plan,
                       conditioning=tuple(conditioningApplied),
                       conditioningMode=conditioningMode,
                       report=report, log=log, dryRun=dryRun,
                       request=request, sourceFormat=sourceFormat,
                       aceOptions=ace if "ace" in formats else None)

    stem = sourcePath.stem if sourcePath is not None else "perturbed"
    if stem.endswith(".gnds"):
        stem = stem[:-5]
    emitTapes = outputDir is not None and not dryRun
    for number in range(nSamples):
        label = f"{labelPrefix}-{number:04d}"
        pset = PerturbationSet.fromDraw(
            {key: samples[key][number] for key in samples}, index, label=label,
            provenance={"seed": seed, "sample": number, "space": space,
                        "grouping": grouping, "source": str(sourcePath or ""),
                        "sourceFormat": sourceFormat})
        with log.timed("applied", f"{label} on the model", sample=number) as info:
            applied = pset.applyToSuite(suite, multiplicityResolver=nubarNode)
            info["components"] = [c.describe() for c in applied]
        _checkRealisation(pset, log, number)

        files: Dict[str, Path] = {}
        aceProduced: List[Dict[str, Any]] = []
        conversion = ConversionReport()
        if emitTapes:
            sampleDir = outputDir / f"{number:04d}"
            sampleDir.mkdir(parents=True, exist_ok=True)
            # ACE needs an ENDF tape on disk. The delta is the faithful one
            # when there is a tape to patch; otherwise the whole tape from the
            # model. Either is written on demand if it was not asked for.
            tapeFormats = [fmt for fmt in formats if fmt != "ace"]
            if "ace" in formats and not any(
                    fmt in tapeFormats for fmt in ("endf-delta", "endf-tape")):
                tapeFormats.append("endf-delta" if sourceFormat == "endf"
                                   else "endf-tape")
            for fmt in tapeFormats:
                with log.timed("emitted", f"{fmt}", subject=fmt, sample=number) as info:
                    if fmt == "endf-delta":
                        files[fmt] = _emitEndfDelta(
                            suite, endfObj, sourcePath, pset,
                            sampleDir / f"{stem}_{number:04d}.endf", conversion,
                            alsoChanged=tuple(applied))
                    elif fmt == "endf-tape":
                        files[fmt] = _emitWholeFile(
                            suite, pset, sampleDir / f"{stem}_{number:04d}.tape.endf",
                            fmt, mat=mat)
                    else:
                        files[fmt] = _emitWholeFile(
                            suite, pset, sampleDir / f"{stem}_{number:04d}.gnds.xml",
                            fmt)
                    info["path"] = str(files[fmt])
                    info["bytes"] = files[fmt].stat().st_size
            if "ace" in formats:
                tape = files.get("endf-delta") or files["endf-tape"]
                aceProduced = _emitAce(tape, sampleDir, ace, log, number)
                good = [r["ace"] for r in aceProduced if r["ace"] and r["returncode"] == 0]
                if good:
                    files["ace"] = [Path(p) for p in good]
            if writeSets:
                files["perturbation-set"] = pset.write(
                    sampleDir / "perturbation.json")

        for note in (_redundancyNote(suite, pset), _sumRuleNote(applied),
                     _spectrumNote(applied)):
            if note is not None and note not in result.notes:
                result.notes.append(note)
                log.warning(note, sample=number)
        result.samples.append({"label": label, "set": pset, "files": files,
                               "applied": applied, "ace": aceProduced})
        _forget(suite, pset, applied)

    if outputDir is not None:
        if writeFactors and result.samples:
            with log.timed("written", "factors table", subject="factors.parquet") as info:
                table, _same = writeFactorsTable(
                    [sample["set"] for sample in result.samples], outputDir)
                result.files["factors"] = table
                info["bytes"] = table.stat().st_size
        if plan is not None:
            planPath = outputDir / "conditioning_plan.json"
            planPath.write_text(json.dumps(plan.to_dict(), indent=2),
                                encoding="utf-8")
            result.files["conditioning-plan"] = planPath
            log.event("written", "conditioning plan", subject=planPath.name,
                      mode=conditioningMode)
    log.event("finished",
              f"{'dry run' if dryRun else 'run'} complete: {nSamples} sample(s)"
              + (f", {len(log.problems())} warning(s)" if log.problems() else ""),
              seconds=log.elapsed, dryRun=dryRun,
              nProblems=len(log.problems()))
    if outputDir is not None:
        result.files["log"] = outputDir / "run.log.jsonl"
        result.files["log-text"] = outputDir / "run.log"
        result.files["metadata"] = outputDir / "run_metadata.json"
        _writeRunMetadata(result, outputDir, covReport, suiteReport, sourcePath,
                          original, seed, space, psdMethod)
        log.write(outputDir)
    return result


def _jsonableRequest(value):
    from kika.sampling.joint_blocks import Selection

    if isinstance(value, Selection):
        return {"mf": value.mf, "mt": _jsonableRequest(value.mt),
                "index": _jsonableRequest(value.index), "relative": value.relative}
    if isinstance(value, Mapping):
        return {str(k): _jsonableRequest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonableRequest(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _forget(suite, pset: PerturbationSet, applied=()) -> None:
    """Drop a realisation's forms once it has been written.

    The labelled form is what replaces the per-sample ``deepcopy``; leaving it in
    place would put the whole ensemble in memory one form at a time, which is the
    cost the copy was rejected for. Removing it is not a cleanup detail -- it is
    the other half of that decision.

    *applied* is what the applier reported, which is a superset of what the
    request named: the nu-bar the sum rule **derives** is rewritten without ever
    having been asked for, so forgetting only the requested components would
    leave it on the suite and every later sample would derive its total from the
    previous sample's parts.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from kika.endf.model_adapter.multiplicity import NUBAR_MT, nubarNode

    for component in applied:
        if component.mf != 31:
            continue
        node = nubarNode(suite, component.mt)
        if node is not None:
            node.forms.pop(pset.label, None)

    for mt in pset.reactions():
        if mt in NUBAR_MT:
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
                      sourcePath, request, seed: int, space: str,
                      psdMethod: str = "none") -> Path:
    """The run's own account of itself, beside the samples.

    Deliberately includes the grouping description in full: "these quantities
    were drawn independently" is a claim about the evaluation, and it is the one
    thing about a mixed run that cannot be recovered from the output files.

    The conditioning record is here for the same reason. ``psd_method`` says
    what the *draw* was allowed to do to a matrix on its own; ``conditioning``
    says what was done to it beforehand, step by step, with the reason each
    step was chosen. Under the design in ``kika/cov/conditioning.py`` the
    first should be ``"none"`` and the second should be the whole story.
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
        "dryRun": result.dryRun,
        "sourceFormat": result.sourceFormat,
        "requestNormalised": {str(key): _jsonableRequest(value)
                              for key, value in result.request.items()}
        if isinstance(result.request, Mapping) else str(result.request),
        "psd_method": psdMethod,
        "conditioning": {
            "mode": result.conditioningMode,
            "plan": ("conditioning_plan.json" if result.conditioningPlan is not None
                     else None),
            "applied": [dict(record) for record in result.conditioning],
            "preflight": result.report.summary() if result.report is not None else None,
        },
        "files": {name: path.name for name, path in result.files.items()},
        "ace": ({"options": result.aceOptions.to_dict(),
                 "failures": [{"sample": s, "temperature": t, "returncode": rc}
                              for s, t, rc in result.aceFailures()]}
                if result.aceOptions is not None else None),
        "log": result.log.summary() if result.log is not None else None,
        "covarianceDecode": covReport.summary() if covReport is not None else "",
        "suiteDecode": suiteReport.summary() if suiteReport is not None else "",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
