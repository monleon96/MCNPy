"""``kika.read()`` — the front door, and the invariants that keep it honest.

This lives in ``kika/tests/`` rather than under a package because, like
``test_layering.py`` beside it, what it asserts belongs to the repository as a
whole: the door spans ``kika.endf``, ``kika.ace`` and ``kika.nuclear_data.model``
and is owned by none of them.

Three things are worth more than the rest and are called out where they appear:

1. **Dormancy survives the door.** The whole safety argument for phases 3a-3d is
   "the model is built beside the old code and nothing imports it". Adding a
   public entry point is exactly the change that would end that quietly.
2. **The door transforms nothing.** It must be a *route* to the adapters, not a
   second decoder. The moment it computes something the manual path does not,
   two answers to the same question exist.
3. **The report is not silently emptied.** Its whole value is that a partial
   decode announces itself.

Everything runs on the committed micro-tapes (~1 s). The full Fe-56 tape takes
minutes to parse and belongs in no test that runs by default.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import kika
from kika import sniff_format
from kika._read import UnknownFormatError

#: **Removed 2026-08-12.** This module used to hold
#: ``ACE_FILE = "kika/ace/files/260560_80.02c"`` — a repo-*relative* path to a
#: 113 MB file that ``.gitignore:26`` excludes. It resolves only in a clone that
#: happens to have that file beside it and only when pytest is run from the
#: repository root, so the two tests below **failed rather than skipped** in
#: every git worktree, which is exactly the false red that ``conftest.py`` was
#: written to abolish ("four hardcoded absolute paths ... now go through the
#: fixtures"). They take the ``fe56_ace`` fixture now, which resolves through
#: the shared tree and skips honestly when it is not there.


# ----------------------------------------------------------------------
# Sniffing
# ----------------------------------------------------------------------

def test_endf_tapes_are_recognised(micro_tape, micro_cov_tape):
    assert sniff_format(micro_tape) == "endf"
    assert sniff_format(micro_cov_tape) == "endf"


def test_an_ace_file_is_recognised(fe56_ace):
    assert sniff_format(str(fe56_ace)) == "ace"


def test_gnds_xml_is_recognised(tmp_path):
    """Sniffing is content-based, so a ``.xml`` that is not GNDS is not GNDS."""
    path = tmp_path / "Fe56.xml"
    path.write_text('<?xml version="1.0"?>\n<reactionSuite projectile="n"/>\n')
    assert sniff_format(path) == "gnds"


def test_a_gnds_version_the_reader_refuses_says_which_and_why(tmp_path):
    """1.9 is where the real break is — 149 change requests to 2.0 — and the
    door must say so rather than fail somewhere inside the decode."""
    from kika.gnds.version import UnsupportedGndsVersion

    path = tmp_path / "old.gnds.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<reactionSuite projectile="n" target="Fe56" evaluation="x" '
        'format="1.9" projectileFrame="lab" interaction="nuclear"/>\n'
    )
    with pytest.raises(UnsupportedGndsVersion, match="1.9"):
        kika.read(path)


def test_an_unknown_file_names_what_was_tried(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a nuclear data file\n")
    with pytest.raises(UnknownFormatError) as excinfo:
        sniff_format(path)
    message = str(excinfo.value)
    for expected in ("ENDF", "ACE", "GNDS", "format="):
        assert expected in message


def test_a_binary_file_is_rejected_rather_than_raising_from_the_decoder(tmp_path):
    """``errors='replace'`` in the sniffer exists for this; without it the user
    gets a ``UnicodeDecodeError`` from inside kika instead of an answer."""
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 8)
    with pytest.raises(UnknownFormatError):
        sniff_format(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        sniff_format(tmp_path / "absent.endf")


def test_format_can_be_forced_past_the_sniffer(micro_tape):
    """For the malformed header, and for the ACE 2.0 file the sniffer cannot place."""
    assert kika.read(micro_tape, format="endf").reactions.ENDF_MTs
    with pytest.raises(ValueError, match="format must be one of"):
        kika.read(micro_tape, format="gibberish")


# ----------------------------------------------------------------------
# The door decodes, and decodes the same thing the manual path does
# ----------------------------------------------------------------------

def test_reading_an_endf_tape_gives_a_populated_suite(micro_tape):
    suite = kika.read(micro_tape)
    assert suite.reactions.ENDF_MTs, "no reactions decoded"
    assert suite.report is not None, "the report was not attached"
    assert suite.styles.labels, "no style decoded from MF1/451"


def test_reading_an_ace_file_gives_a_populated_suite(fe56_ace):
    suite = kika.read(str(fe56_ace))
    assert 102 in suite.reactions
    assert suite.report is not None


def test_the_door_is_a_route_not_a_second_decoder(micro_tape):
    """Array-identical to ``read_endf`` -> ``decodeReactionSuite``, by hand.

    If this ever fails, the door has started computing something of its own and
    there are two answers to "what is the Fe-56 capture cross section".
    """
    from kika.endf.model_adapter.decode import decodeReactionSuite
    from kika.endf.read_endf import read_endf

    byHand, _ = decodeReactionSuite(read_endf(str(micro_tape)))
    throughDoor = kika.read(micro_tape)

    assert throughDoor.reactions.ENDF_MTs == byHand.reactions.ENDF_MTs
    for mt in byHand.reactions.ENDF_MTs:
        expectedE, expectedXs = byHand.cross_section(mt)
        gotE, gotXs = throughDoor.cross_section(mt)
        np.testing.assert_array_equal(gotE, expectedE, err_msg=f"MT{mt} energies")
        np.testing.assert_array_equal(gotXs, expectedXs, err_msg=f"MT{mt} sigma")


def test_covariances_are_decoded_by_default(micro_cov_tape):
    suite = kika.read(micro_cov_tape)
    assert suite.covarianceSuite is not None
    assert len(suite.covarianceSuite) > 0


def test_covariances_can_be_declined(micro_cov_tape):
    suite = kika.read(micro_cov_tape, covariances=False)
    assert suite.covarianceSuite is None


# ----------------------------------------------------------------------
# GNDS through the same door
# ----------------------------------------------------------------------

def test_reading_gnds_gives_the_same_kind_of_suite_as_endf(h2_gnds):
    """One object type comes out, whatever went in. That is the door's premise."""
    suite = kika.read(h2_gnds)
    assert suite.reactions.ENDF_MTs == [2, 16, 102]
    assert suite.styles.labels == ["eval"]
    assert suite.report is not None
    E, sigma = suite.cross_section(102)
    assert E.size == sigma.size > 0


def test_the_gnds_door_is_a_route_not_a_second_decoder(h2_gnds):
    """Array-identical to ``readReactionSuite`` called by hand, same as ENDF's."""
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document

    byHand, _ = readReactionSuite(Document.parse(h2_gnds))
    throughDoor = kika.read(h2_gnds, covariances=False)

    assert throughDoor.reactions.ENDF_MTs == byHand.reactions.ENDF_MTs
    for mt in byHand.reactions.ENDF_MTs:
        expectedE, expectedXs = byHand.cross_section(mt)
        gotE, gotXs = throughDoor.cross_section(mt)
        np.testing.assert_array_equal(gotE, expectedE, err_msg=f"MT{mt} energies")
        np.testing.assert_array_equal(gotXs, expectedXs, err_msg=f"MT{mt} sigma")


#: The signature of :func:`kika.gnds.covariances._noteUnfollowableLink`'s entry:
#: a covariance section whose ``rowData`` href reached nothing.
UNFOLLOWABLE = "The covariance itself is read"


def test_the_gnds_door_follows_the_external_file_to_the_covariances(h2_gnds,
                                                                   h2_gnds_cov):
    """§25.1.1's covarianceSuite is a **second file**, and the door fetches it.

    It also hands *this* file back to the covariance reader under the label that
    file uses for it, so every ``$reactions#…`` href on a ``rowData`` resolves.
    That is checked by contrast rather than by assertion: reading the covariance
    file on its own leaves one unfollowable link per section, and going through
    the door leaves none.
    """
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.xpath import Document

    _, alone = readCovarianceSuite(Document.parse(h2_gnds_cov))
    assert len([loss for loss in alone.losses if UNFOLLOWABLE in loss]) == 4

    suite = kika.read(h2_gnds)
    assert suite.covarianceSuite is not None
    assert len(suite.covarianceSuite) == 4
    assert not [loss for loss in suite.report.losses if UNFOLLOWABLE in loss]


def test_gnds_covariances_can_be_declined_and_then_no_second_file_is_read(h2_gnds):
    suite = kika.read(h2_gnds, covariances=False)
    assert suite.covarianceSuite is None
    assert suite.externalFiles.byLabel("covariances") is not None


def test_a_missing_covariance_sibling_is_reported_and_not_an_error(tmp_path,
                                                                  h2_gnds):
    """A user handed one half of a pair is in a normal situation, and their
    cross sections are perfectly readable."""
    copy = tmp_path / h2_gnds.name
    copy.write_bytes(h2_gnds.read_bytes())

    suite = kika.read(copy)
    assert suite.covarianceSuite is None
    assert suite.reactions.ENDF_MTs == [2, 16, 102]
    assert any("is not beside" in loss for loss in suite.report.losses), \
        suite.report.losses


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------

def test_decoding_the_covariances_clears_the_redirect_that_told_us_to(micro_cov_tape):
    """The notice says "call decodeCovarianceSuite for it". The door did.

    Leaving it would make ``isClean`` false for every ordinary covariance tape,
    which is how a report teaches its reader to ignore it. The corresponding
    half — that declining covariances *keeps* the notice — is the next test, and
    the pair is what stops this from becoming a blanket whitewash.
    """
    suite = kika.read(micro_cov_tape)
    redirects = [m for m in suite.report.unsupported if "belongs to the covarianceSuite" in m]
    assert not redirects, f"stale redirects survived: {redirects}"


def test_declining_the_covariances_keeps_the_redirect(micro_cov_tape):
    suite = kika.read(micro_cov_tape, covariances=False)
    redirects = [m for m in suite.report.unsupported if "belongs to the covarianceSuite" in m]
    assert redirects, "the notice was dropped even though nothing decoded the covariances"


def test_the_low_road_attaches_the_report_too(micro_cov_tape, gnds_data_dir):
    """§11.4. Two ways to the same object, from *both* doors.

    ``suite.report`` exists because a tuple element is what a caller in a hurry
    drops: ``suite, _ = decodeReactionSuite(endf)`` is the idiom, and the
    underscore is exactly the thing that says which MFs went unread. Only
    ``kika.read()`` filled it, so the answer to "does this object know what its
    decode lost?" depended on which door was used — and the low road is the
    door the library's own modules use.

    All four decoders are checked here, in one place, because the property is
    the same one four times and splitting it across four packages is how three
    of them would come to disagree. ``CovarianceSuite`` had no ``report`` field
    at all until this went in, which is why the covariance halves are not just
    a repeat of the reaction ones.
    """
    from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document

    endf = read_endf(str(micro_cov_tape))
    suite, report = decodeReactionSuite(endf)
    assert suite.report is report

    covariances, covarianceReport = decodeCovarianceSuite(endf)
    assert covariances.report is covarianceReport

    document = Document.parse(gnds_data_dir / "n-001_H_002.endf.gnds.xml")
    fromGnds, gndsReport = readReactionSuite(document)
    assert fromGnds.report is gndsReport

    sibling = Document.parse(
        gnds_data_dir / "Covariances/n-001_H_002.endf.gnds-covar.xml")
    gndsCovariances, siblingReport = readCovarianceSuite(sibling)
    assert gndsCovariances.report is siblingReport


def test_the_front_door_and_the_low_road_agree_on_the_report(micro_cov_tape):
    """The same object, not a copy of it. ``kika.read`` adds entries *after* the
    adapter returns — the unparsed-MF rescan, the redirect cleanup — and they
    have to land on the report the suite is carrying, not beside it."""
    suite = kika.read(micro_cov_tape)
    assert suite.covarianceSuite is not None
    assert suite.covarianceSuite.report is suite.report, (
        "the pair came out of one read and carries two different reports"
    )


def test_an_mf_with_no_parser_is_reported_rather_than_passed_over(tmp_path, micro_tape):
    """``endf.mf`` holds only the MFs that *had* a parser, so the parsed object
    cannot answer "did this tape carry an MF6?". The door rescans the tape.

    Built by appending a minimal MF6 section to a real tape, so the rest of the
    decode is unchanged and the only difference is the section under test.
    """
    lines = micro_tape.read_text().splitlines(keepends=True)
    mat = int(lines[1][66:70])
    injected = (
        f"{' 0.000000+0 0.000000+0          0          0          0          0':<66}"
        f"{mat:>4}{6:>2}{5:>3}{1:>5}\n"
    )
    doctored = tmp_path / "with_mf6.endf"
    doctored.write_text("".join(lines[:-1] + [injected] + lines[-1:]))

    report = kika.read(doctored).report
    assert any("MF6" in m for m in report.unsupported), (
        f"MF6 was on the tape and no parser read it, yet the report is silent: "
        f"{report.unsupported}"
    )


# ----------------------------------------------------------------------
# Dormancy — the invariant the door is most likely to break
# ----------------------------------------------------------------------

def _inSubprocess(body: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    return completed.stdout.strip()


def test_the_door_is_visible_without_waking_the_model():
    """Both halves at once, and they pull in opposite directions.

    Discoverable (``dir``, tab-completion, ``from kika import *``) usually means
    imported. Here it must not: resolving ``kika.read`` reaches the adapters and
    the adapters reach the model, and the model must stay dormant for the
    cluster, the app and every notebook that merely does ``import kika``.
    """
    assert _inSubprocess("""
        import sys, kika
        assert 'read' in dir(kika), 'read is not discoverable'
        leaked = sorted(m for m in sys.modules if m.startswith('kika.nuclear_data.model'))
        assert not leaked, 'import kika woke the model: ' + ', '.join(leaked)
        print('dormant')
    """) == "dormant"


def test_repeated_access_keeps_returning_the_function():
    """The submodule-shadows-function trap, which this failed on first.

    ``importlib`` binds a submodule onto its package as a side effect, so a
    module named ``kika/read.py`` would replace ``kika.read`` with itself: first
    access a function, every later one a module. The module is ``_read`` for this
    reason, and a rename back would be silent without this test.
    """
    assert _inSubprocess("""
        import kika
        kinds = [type(kika.read).__name__ for _ in range(3)]
        assert kinds == ['function'] * 3, kinds
        print('function')
    """) == "function"
