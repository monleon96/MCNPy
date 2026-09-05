"""
Cross-section reconstruction via NJOY ``reconr``.

Delegates the full resonance reconstruction (SLBW, MLBW, Reich-Moore,
R-Matrix Limited, URR) to a locally installed NJOY binary.  The caller
supplies the path to the NJOY executable — typically read from the app's
Settings (``localStorage`` entry ``kika_njoy_executable_path``).

The workflow is:

1. Copy the input ENDF tape into a temporary directory as ``tape20``.
2. Run NJOY with a minimal ``reconr``-only input deck.
3. Parse the resulting ``tape21`` PENDF with :func:`kika.endf.read_endf`.
4. Extract MF3 sections as :class:`CrossSection` objects, keyed by MT.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Deferred with the read_endf import below, and for the same reason:
# kika.njoy imports kika.endf at module scope, so importing it here
# would put kika.endf on kika.processing's import path and re-create
# the cycle that blocks moving interpolate_1d down.
from kika.nuclear_data.cross_section import CrossSection

_log = logging.getLogger(__name__)

def _read_endf(path: str, mf_numbers: Optional[List[int]] = None):
    """Parse an ENDF/PENDF tape, importing the parser only when actually used.

    ``mf_numbers`` is forwarded to :func:`kika.endf.read_endf.read_endf`; a
    caller that wants one file asks for it, because a bare read parses every
    MF with a registered parser and the registry keeps growing.

    ``kika/processing`` must not pull ``kika.endf`` in at *import* time: a
    module-level import here puts ``kika.endf`` on ``kika.processing``'s import
    path, and then nothing in ``kika.endf`` can import from ``kika.processing``
    at module scope -- which is precisely what blocked moving ``interpolate_1d``
    down in phase 2 of the GNDS roadmap.

    Reading a PENDF tape genuinely needs the parser, so the dependency stays and
    only its timing changes. One function rather than an import at each call
    site, so the layering ratchet keeps counting one violation and not two.
    """
    from kika.endf.read_endf import read_endf

    return read_endf(path, mf_numbers=mf_numbers)


def _mat_from_tape_header(path: Path) -> Optional[int]:
    """The MAT of the first material on an ENDF tape, read off the header.

    reconr only needs the number; parsing the whole evaluation to get it cost
    as much as reconr itself on a heavy tape (MF6 and MF32 are the largest
    files of an actinide evaluation) and duplicated a parse the calling
    application had already done. The ENDF-6 record layout puts MAT in columns
    67-70, MF in 71-72 and MT in 73-75; the tape identification line is
    skipped by requiring the record to be MF1/MT451, the first of any material.
    Returns ``None`` when no such record is found among the first lines, and
    the caller falls back to the parser.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(8):
                line = fh.readline()
                if not line:
                    break
                if len(line) < 75:
                    continue
                try:
                    mat = int(line[66:70])
                    mf = int(line[70:72])
                    mt = int(line[72:75])
                except ValueError:
                    continue
                if mat > 0 and mf == 1 and mt == 451:
                    return mat
    except OSError:
        return None
    return None




class NjoyReconstructError(RuntimeError):
    """Raised when the NJOY ``reconr`` subprocess fails or PENDF is malformed."""

    def __init__(
        self,
        message: str,
        *,
        returncode: Optional[int] = None,
        stderr_tail: Optional[str] = None,
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


_MAX_TOLERANCE = 0.1


def _tolerance_fallback_sequence(initial: float) -> List[float]:
    """Return a sequence of tolerances to try if the initial value fails.

    Some ENDF files (e.g. Fe-56 JEFF-4.0) have threshold MT3 sections that
    reconr's ``lunion`` cannot linearize tightly without raising an
    "ill-behaved threshold" error.  We fall back to progressively looser
    tolerances (doubling each step) so the user is not stuck on files
    that need only slightly more slack than the previous cap allowed.

    Capped at :data:`_MAX_TOLERANCE` (10 %) — beyond that reconr output
    is coarse enough that the reconstruction is no longer scientifically
    useful, and we surface the original NJOY error to the caller instead
    of silently producing a misleading PENDF.
    """
    if initial >= _MAX_TOLERANCE:
        return [initial]
    seq = [initial]
    val = initial
    while val < _MAX_TOLERANCE:
        val = min(val * 2.0, _MAX_TOLERANCE)
        if val > seq[-1]:
            seq.append(val)
        if val >= _MAX_TOLERANCE:
            break
    return seq


def _build_reconr_input(mat: int, tolerance: float) -> str:
    """Return a minimal NJOY deck: moder (ASCII→bin) → reconr → moder (bin→ASCII).

    Tapes:
        20  ASCII ENDF input    (user file)
       -25  binary ENDF         (moder output of 20)
       -21  binary PENDF        (reconr output)
        22  ASCII PENDF         (final output for our reader)
    """
    err = f"{tolerance:.4e}"
    errmax = f"{tolerance * 10.0:.4e}"
    errint = f"{tolerance * 5.0e-5:.4e}"  # ~err/20000 per NJOY default
    return "\n".join(
        [
            "moder",
            " 20 -25",
            "reconr",
            " -25 -21",
            "'kika reconr reconstruction'/",
            f" {mat} 0 0",
            f" {err} 0 {errmax} {errint}",
            " 0 /",
            "moder",
            " -21 22",
            "stop",
            "",
        ]
    )


def njoy_reconstruct(
    endf_path: str | Path,
    njoy_executable: str | Path,
    *,
    tolerance: float = 0.001,
    timeout_s: float = 600.0,
    keep_workdir: bool = False,
) -> Dict[int, CrossSection]:
    """Reconstruct pointwise cross sections via NJOY reconr.

    Blocking wrapper around :func:`njoy_reconstruct_stream` that discards
    the live log output and returns only the final MT→CrossSection map.

    On an "ill-behaved threshold" error (common on some JEFF-4.0 files),
    automatically retries with progressively looser tolerance up to
    :data:`_MAX_TOLERANCE`.
    """
    result: Optional[Dict[int, CrossSection]] = None
    for event in njoy_reconstruct_stream(
        endf_path,
        njoy_executable,
        tolerance=tolerance,
        timeout_s=timeout_s,
        keep_workdir=keep_workdir,
    ):
        kind = event[0]
        if kind == "result":
            result = event[1]  # type: ignore[assignment]
        elif kind == "warning":
            _log.warning("%s", event[1])
        # "log" and "pendf" events are silently dropped in blocking mode
    if result is None:
        raise NjoyReconstructError("NJOY reconstruction produced no result")
    return result


def njoy_reconstruct_stream(
    endf_path: str | Path,
    njoy_executable: str | Path,
    *,
    tolerance: float = 0.001,
    timeout_s: float = 600.0,
    keep_workdir: bool = False,
) -> Iterator[Tuple[str, Any]]:
    """Streaming variant of :func:`njoy_reconstruct`.

    Yields 2-tuples ``(kind, payload)`` where ``kind`` is one of:

    - ``"log"``     : a single NJOY stdout line (as text, with trailing ``\\n``).
    - ``"warning"`` : a human-readable notice (e.g. tolerance relaxation).
    - ``"result"`` : a ``Dict[int, CrossSection]`` — terminal success event.

    On failure, raises :class:`NjoyReconstructError`.  Consumers can abort
    early by closing the generator (``gen.close()``); any running
    subprocess is killed in that case.
    """
    endf_path = Path(endf_path).expanduser().resolve()
    if not endf_path.is_file():
        raise NjoyReconstructError(f"ENDF file not found: {endf_path}")

    njoy_executable = Path(str(njoy_executable)).expanduser()
    # WSL UNC paths (\\wsl.localhost\<distro>\...) are reachable from Windows
    # via SMB; is_file() still works there.  We only skip the check entirely
    # for those cases where Windows can't see the share (e.g. WSL stopped) —
    # in that case wsl.exe will fail with a clearer error than os.stat.
    from kika.njoy.launcher import build_njoy_command, is_wsl_unc_path
    if not is_wsl_unc_path(njoy_executable) and not njoy_executable.is_file():
        raise NjoyReconstructError(
            f"NJOY executable not found: {njoy_executable}. "
            "Set the NJOY path in app Settings."
        )

    # Deferred on purpose: kika/processing must not pull kika.endf in at
    # *import* time. A module-level import here makes kika.processing's
    # __init__ depend on kika.endf, and then nothing in kika.endf can import
    # from kika.processing at module scope -- which is what blocked moving
    # interpolate_1d down in phase 2. Reading a PENDF tape genuinely needs
    # the parser, so the dependency stays; only its timing changes.
    mat = _mat_from_tape_header(endf_path)
    if mat is None:
        # Not a tape the header scan recognises: let the parser decide, but
        # only over MF1, which is all the MAT needs.
        endf = _read_endf(str(endf_path), mf_numbers=[1])
        if endf.mat is None:
            raise NjoyReconstructError(
                f"Could not determine MAT number from {endf_path.name}"
            )
        mat = int(endf.mat)

    tolerances = _tolerance_fallback_sequence(tolerance)
    last_err: Optional[NjoyReconstructError] = None

    for idx, tol in enumerate(tolerances):
        log_tail = ""
        try:
            for event in _run_reconr_once_stream(
                endf_path=endf_path,
                njoy_executable=njoy_executable,
                mat=mat,
                tolerance=tol,
                timeout_s=timeout_s,
                keep_workdir=keep_workdir,
            ):
                if event[0] == "log":
                    log_tail = (log_tail + event[1])[-4000:]
                    yield event
                elif event[0] == "result":
                    if tol != tolerance:
                        yield (
                            "warning",
                            f"NJOY reconr required loosened tolerance "
                            f"{tol:g} (requested {tolerance:g}).",
                        )
                    yield event
                    return
                else:
                    # Forward any other event kinds (e.g. "pendf_path",
                    # "warning") so downstream consumers can react.  The
                    # "pendf_path" event, in particular, must reach the app
                    # *before* the inner generator returns — its value is
                    # only valid while the TemporaryDirectory is still open.
                    yield event
        except NjoyReconstructError as e:
            last_err = e
            tail = (e.stderr_tail or log_tail or "").lower()
            if "ill-behaved threshold" in tail and idx < len(tolerances) - 1:
                next_tol = tolerances[idx + 1]
                yield (
                    "warning",
                    f"NJOY reconr ill-behaved threshold at tol={tol:g}; "
                    f"retrying at tol={next_tol:g}.",
                )
                continue
            raise

    assert last_err is not None
    raise last_err


def _run_reconr_once_stream(
    *,
    endf_path: Path,
    njoy_executable: Path,
    mat: int,
    tolerance: float,
    timeout_s: float,
    keep_workdir: bool,
) -> Iterator[Tuple[str, Any]]:
    """Run NJOY moder→reconr→moder once, streaming stdout.

    Yields ``("log", line)`` for every line NJOY prints, then finally
    ``("result", Dict[int, CrossSection])`` on success.
    """
    workdir_cm = (
        _no_cleanup_tempdir() if keep_workdir
        else tempfile.TemporaryDirectory(prefix="kika_reconr_")
    )

    with workdir_cm as td:
        workdir = Path(td)
        shutil.copy2(endf_path, workdir / "tape20")

        input_text = _build_reconr_input(mat, tolerance)
        (workdir / "njoy.inp").write_text(input_text, encoding="utf-8")

        env = os.environ.copy()
        forward_env = {
            "LC_ALL": "C",
            "LANG": "C",
            # Force gfortran runtime to flush stdout on every write.  Without
            # this NJOY (a gfortran/MinGW binary) block-buffers stdout whenever
            # it is not a TTY, so the parent only sees output after ~4-64 KB
            # accumulate — which for reconr often means nothing appears until
            # the process exits.  GFORTRAN_UNBUFFERED_PRECONNECTED covers the
            # preconnected units only (stdin, stdout, stderr), which is the
            # one we read live.  The _ALL variant also unbuffered every tape
            # NJOY writes, and the PENDF of an actinide is a hundred
            # megabytes of 80-character records: one syscall per record on
            # Windows is a cost the live log never asked for.
            "GFORTRAN_UNBUFFERED_PRECONNECTED": "y",
            "PYTHONUNBUFFERED": "1",
        }
        env.update(forward_env)

        from kika.njoy.launcher import build_njoy_command, is_wsl_unc_path
        cmd = build_njoy_command(njoy_executable, forward_env=forward_env)

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                text=True,
                errors="replace",
            )
        except OSError as e:
            raise NjoyReconstructError(
                f"Failed to launch NJOY ({njoy_executable}): {e}"
            ) from e

        collected: List[str] = []
        start = time.monotonic()
        timed_out = False
        try:
            # Push the deck to stdin first.  NJOY decks are <4 KiB so the
            # pipe buffer absorbs them without needing a concurrent writer.
            if process.stdin is not None:
                try:
                    process.stdin.write(input_text)
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            assert process.stdout is not None
            # Use readline() instead of ``for line in process.stdout`` —
            # the iterator protocol buffers internally and can withhold
            # lines even when the child has already flushed them.
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                collected.append(line)
                yield ("log", line)
                if time.monotonic() - start > timeout_s:
                    timed_out = True
                    process.kill()
                    break

            remaining = max(1.0, timeout_s - (time.monotonic() - start))
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=5.0)
        except GeneratorExit:
            try:
                process.kill()
                process.wait(timeout=5.0)
            except Exception:
                pass
            raise
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass

        log_text = "".join(collected)
        (workdir / "njoy.out").write_text(log_text, encoding="utf-8")

        if timed_out:
            raise NjoyReconstructError(
                f"NJOY reconr timed out after {timeout_s:.0f} s",
                returncode=process.returncode,
                stderr_tail=log_text[-2000:],
            )

        # NJOY on Windows with mingw sometimes exits 0 despite STOP N.
        # Detect failure via the log text as well.
        log_indicates_failure = (
            "***error" in log_text.lower()
            or "stop 77" in log_text.lower()
        )

        pendf_tape = workdir / "tape22"
        pendf_ok = pendf_tape.is_file() and pendf_tape.stat().st_size > 0

        if process.returncode != 0 or (log_indicates_failure and not pendf_ok):
            raise NjoyReconstructError(
                f"NJOY reconr failed (exit {process.returncode})",
                returncode=process.returncode,
                stderr_tail=log_text[-2000:],
            )

        if not pendf_ok:
            raise NjoyReconstructError(
                "NJOY reconr produced no PENDF output (tape22 missing or empty)",
                returncode=process.returncode,
                stderr_tail=log_text[-2000:],
            )

        try:
            # Deferred on purpose: kika/processing must not pull kika.endf in at
            # *import* time. A module-level import here makes kika.processing's
            # __init__ depend on kika.endf, and then nothing in kika.endf can import
            # from kika.processing at module scope -- which is what blocked moving
            # interpolate_1d down in phase 2. Reading a PENDF tape genuinely needs
            # the parser, so the dependency stays; only its timing changes.
            # MF3 is the only file the result is built from; a PENDF's MF2
            # is a stub after reconr and MF1 is a directory.
            pendf = _read_endf(str(pendf_tape), mf_numbers=[3])
        except Exception as e:
            raise NjoyReconstructError(
                f"Failed to parse PENDF output: {e}",
                returncode=process.returncode,
                stderr_tail=log_text[-2000:],
            ) from e

        # Emit the path to the raw tape22 so callers that care about it
        # (the app's "Save PENDF to workspace" action) can copy it out of
        # the generator's TemporaryDirectory before it is cleaned up.
        # Callers that only want cross sections can ignore this event.
        #
        # The path points *inside* this generator's TemporaryDirectory and
        # is only valid until the generator returns (and the ``with``
        # block below exits).  Consumers must copy bytes out synchronously
        # during event iteration, not defer.
        yield ("pendf_path", str(pendf_tape))

        yield ("result", _extract_mf3_cross_sections(pendf))


def _extract_mf3_cross_sections(pendf) -> Dict[int, CrossSection]:
    """Convert parsed PENDF MF3 sections into CrossSection objects by MT."""
    mf3 = pendf.mf.get(3)
    if mf3 is None or not getattr(mf3, "mt", None):
        raise NjoyReconstructError("PENDF output has no MF3 cross sections")

    result: Dict[int, CrossSection] = {}
    for mt_num, mt_section in mf3.mt.items():
        try:
            xs = CrossSection.from_endf(mt_section)
        except Exception:
            continue
        result[int(mt_num)] = xs

    if not result:
        raise NjoyReconstructError(
            "PENDF MF3 contained no parseable cross sections"
        )
    return result


class _no_cleanup_tempdir:
    """tempdir context manager that does not clean up (for debugging)."""

    def __enter__(self):
        self._dir = tempfile.mkdtemp(prefix="kika_reconr_")
        return self._dir

    def __exit__(self, exc_type, exc, tb):
        return False
