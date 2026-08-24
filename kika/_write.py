"""``kika.write()`` — one door out, symmetric with :mod:`kika._read`.

    >>> report = kika.write(evaluation, "Fe56.gnds.xml")
    >>> report.summary()

The same shape as the door in: one function, the format chosen at the threshold,
one object type going through. What comes back is not the file — it is the
:class:`~kika.nuclear_data.model.conversion.ConversionReport`, because the file
cannot tell you what is missing from it and the report can.

**GNDS and ENDF-6.** ``format="endf"`` assembles a whole tape from the model
(``kika/endf/writers/assemble.py``, ``gnds_endf_conflicts.md`` §2.8) and is a
different thing from ``kika/endf/write_endf.py``, which is a *patch-in-place*
editor: that one rewrites the sections it is given inside a tape it already
read, and cannot produce a tape it was not handed. This door can, which is what
makes a GNDS file convertible to ENDF at all.

**What it cannot carry, it says.** MF7, MF12-15 and MF32 have no
ENDF → model adapter or no encoder; MF5 has one only for its tabulated LF=1,
and MF6 for every law but LAW=5 — so a model that never held them writes a tape
without them, and the returned report names each one. The gate the writer
was built against is a fixed point *inside the model* — read, write, read again,
compare — and not byte identity against the tape it came from; §2.8 says why,
and says what that gate cannot see.

**Two files, not one.** §25.1.1 makes ``covarianceSuite`` a root node in its own
right. A suite with covariances is therefore written as a pair — the evaluation,
and a sibling under ``Covariances/`` — with the ``externalFile`` entry and its
SHA-1 written to match what was actually put on disk. A ``reactionSuite`` naming
a file that does not exist is worse than one naming none, because the first is a
broken link and the second is an honest absence.

**Where this is not for.** The thesis pipeline perturbs ENDF tapes and scores
ENDF tapes; nothing here is on that path. This exists so that a model built or
modified in kika can leave in a format somebody else's code reads — phase 8's
``realization`` ensemble is the case it was built for.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["write"]

#: Formats the door can write.
WRITE_FORMATS = ("gnds", "endf")

#: Where a covariance sibling goes, relative to the evaluation. The layout the
#: ENDF/B-VIII.1 GNDS distribution uses, so a pair kika writes sits where a
#: reader of that distribution already looks.
COVARIANCE_SUBDIRECTORY = "Covariances"


def write(suite, path, format: str = "gnds", gnds: Optional[str] = None,
          mat: Optional[int] = None, tapeId: Optional[str] = None):
    """Write a :class:`ReactionSuite` out, and say what did not go with it.

    Parameters
    ----------
    suite
        The evaluation. Its ``covarianceSuite``, if it has one, is written
        beside it as the separate document §25.1.1 requires.
    path
        Where the ``reactionSuite`` goes. The covariance sibling is derived from
        it — same stem, under ``Covariances/``.
    format
        ``'gnds'`` or ``'endf'``. ENDF writes **one** file: §25.1.1's split into
        two documents is GNDS's, and an ENDF tape states the covariances in the
        same material.
    gnds
        ``'2.0'`` or ``'2.1'``, forcing the declared version. The default
        mirrors what the suite was read from, and is ``'2.0'`` for a suite with
        no GNDS origin — see :func:`kika.gnds.encode.chooseFormat`. Ignored by
        the ENDF path.
    mat
        ENDF only. The material number every record is stamped with. The
        default is the one the suite's provenance recorded; a suite that never
        came from an ENDF tape has none, and is **refused** rather than stamped
        with a guess.
    tapeId
        ENDF only. The 66 text columns of the tape identification record.
        ``read_endf`` does not keep a tape's first line, so a round trip cannot
        reproduce the original and the default label says as much in the report.

    Returns
    -------
    ConversionReport
        **Read it.** A partial model writes a partial file, and the file itself
        cannot say so: a product whose §18 law kika does not read comes out with
        an empty ``<distribution/>``, which looks like nothing at all. The report
        names each one, and says that such a file does not validate — which is
        deliberate, because the alternative is asserting on the evaluator's
        behalf that no distribution was given.
    """
    format = format.lower()
    if format not in WRITE_FORMATS:
        raise ValueError(
            f"format must be one of {WRITE_FORMATS}, got {format!r}"
        )
    if format == "endf":
        # Imported here, not at module scope, for the reason `_writeGnds` gives
        # and one more: the assembler reaches `kika.endf.model_adapter`, and
        # `test_nothing_imports_the_adapter` is the rule that keeps the model
        # off `read_endf`'s critical path.
        from kika.endf.writers.assemble import writeEndfTape

        return writeEndfTape(suite, Path(os.fspath(path)), mat=mat,
                             tapeId=tapeId)
    return _writeGnds(suite, Path(os.fspath(path)), gnds)


def _writeGnds(suite, path: Path, gnds: Optional[str]):
    # Imported here, not at module scope: importing the encoder imports the
    # model, and `import kika` must not wake it. Same reason as `_read`.
    from kika.gnds.encode import (chooseFormat, serialise, sha1,
                                  writeCovarianceSuite, writeReactionSuite)
    from kika.nuclear_data.model import ConversionReport, ExternalFile

    report = ConversionReport()
    format = chooseFormat(suite, gnds)

    # Both links below are rewritten on the caller's own objects, so they are
    # put back before returning. **Writing a file must not edit the model you
    # were handed**: without this, `kika.write(suite, a)` followed by
    # `kika.write(suite, b)` leaves the suite naming b, and a user who wrote a
    # copy out for inspection finds their evaluation quietly changed.
    originalExternalFiles = list(suite.externalFiles.files)
    originalBackLinks = (list(suite.covarianceSuite.externalFiles)
                         if suite.covarianceSuite is not None else None)
    try:
        return _writeLinkedPair(suite, path, format, report,
                                writeReactionSuite, writeCovarianceSuite,
                                serialise, sha1, ExternalFile)
    finally:
        suite.externalFiles.files = originalExternalFiles
        if originalBackLinks is not None:
            suite.covarianceSuite.externalFiles = originalBackLinks


def _writeLinkedPair(suite, path, format, report, writeReactionSuite,
                     writeCovarianceSuite, serialise, sha1, ExternalFile):
    covariancePath = None
    if suite.covarianceSuite is not None:
        covariancePath = (
            path.parent / COVARIANCE_SUBDIRECTORY
            / f"{path.name.replace('.xml', '')}-covar.xml"
        )
        # **The back-link is rewritten too.** Every `rowData` href in a
        # covariance file is `$reactions#/…`, and `$reactions` resolves through
        # that file's *own* `externalFiles`. Carrying over the entry the suite
        # was read with names the file kika read, not the one it is about to
        # write, and the pair comes out with all its cross-section links dead —
        # found by reading the written pair back, not by inspection.
        #
        # No checksum: the reactionSuite names the covariance file's digest, so
        # a digest in the other direction cannot exist. The distribution's own
        # covariance files carry none for the same reason.
        suite.covarianceSuite.externalFiles = [
            entry for entry in suite.covarianceSuite.externalFiles
            if entry.label != "reactions"
        ] + [ExternalFile(label="reactions", path=f"../{path.name}")]
        tree, report = writeCovarianceSuite(suite.covarianceSuite, format, report)
        covariancePath.parent.mkdir(parents=True, exist_ok=True)
        payload = serialise(tree)
        covariancePath.write_bytes(payload)

        # The externalFile is rewritten to match what was *actually written*,
        # digest included. Carrying the digest the suite was read with would
        # name a file whose contents kika has just changed, and the next reader
        # would report a checksum failure kika caused.
        relative = f"{COVARIANCE_SUBDIRECTORY}/{covariancePath.name}"
        suite.externalFiles.files = [
            entry for entry in suite.externalFiles.files
            if entry.label != "covariances"
        ] + [ExternalFile(label="covariances", path=relative,
                          checksum=sha1(payload), algorithm="sha1")]

    tree, report = writeReactionSuite(suite, format, report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise(tree))

    if covariancePath is not None:
        report.warn(
            f"the covariances were written as a separate document, "
            f"{covariancePath.name}, which §25.1.1 requires; the evaluation's "
            f"externalFile entry and its SHA-1 were rewritten to match it"
        )
    return report
