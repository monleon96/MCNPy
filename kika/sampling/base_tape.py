"""Build the tape the replicas are written onto: same physics, no covariance.

WHY. `endf_perturbation._process_sample` writes a **whole copy of the source
tape** per replica, with MF4 spliced in. Pointed at the Fe-56 deliverable that
is 570 MB x 512 = 292 GB for one ensemble, and 96 % of those bytes are MF33 and
MF34 — MF34/MT2 alone is 84 % of the file. **ACER never reads any of it.**
NJOY's covariance modules (ERRORR, COVR) do, and they are not in this chain.

So the two tapes are separated on purpose:

* the tape that is **read**, once, for its covariance: the deliverable;
* the tape that is **written**, 512 times, and handed to NJOY: this one, ~27 MB.

They must agree on the physics, and :func:`build_base_tape` returns the report
that says so rather than asserting it silently — the MF4 section the sampler
perturbs has to be the deliverable's own, or the ensemble is centred on a
different evaluation than the covariance describes.

⚠ Not an optimisation with a fallback. A run that "just uses the big tape"
fills the share; `/share_snc` was at 93 % with 589 GB free when this was
written.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["COVARIANCE_MF", "strip_covariance_files", "build_base_tape",
           "mf_section_digest"]

#: The covariance files. ACER reads none of them; ERRORR/COVR do, and they are
#: not in the ENDF -> PENDF -> ACE chain this package drives.
#:
#: MF32 is in the list even though the deliverable has none — the window is
#: above Fe-56's resolved resonance range, which ends at 850 keV — because the
#: rule is "covariance does not travel with a replica", not "whatever this one
#: tape happens to contain".
COVARIANCE_MF: Tuple[int, ...] = (31, 32, 33, 34, 35)


def mf_section_digest(path: str, mf: int, mt: Optional[int] = None) -> Tuple[str, int]:
    """``(sha256, n_records)`` of one MF (or MF/MT) section's lines.

    Streams the file, so it is safe on a 570 MB tape. Columns 71-72 carry MF
    and 73-75 MT, which is how every section in this project is located
    (fixed-column ENDF-6, §0.6.3).
    """
    h = hashlib.sha256()
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 75:
                continue
            try:
                line_mf = int(line[70:72])
                line_mt = int(line[72:75])
            except ValueError:
                continue
            if line_mf != mf or line_mt == 0:
                continue
            if mt is not None and line_mt != int(mt):
                continue
            h.update(line[:66].encode("ascii", "replace"))
            n += 1
    return h.hexdigest(), n


def strip_covariance_files(
    source: str,
    output: str,
    mf_numbers: Sequence[int] = COVARIANCE_MF,
) -> Dict[str, object]:
    """Copy *source* to *output* without the given whole MF files.

    Delegates to :func:`kika.endf.writers.remove_sections` — the tested line
    filter that also drops the section's SEND/FEND bookkeeping — and then
    rebuilds MF1/451's directory, because NXC and the per-section record counts
    are only true of the file that was actually written.

    ⚠ Reads the whole tape into memory. That is fine where this runs (a cluster
    node) and deliberate: the alternative is a second line-scanning
    implementation of something already gated by tests.
    """
    from kika.endf.writers import remove_sections, update_mf1_directory

    content = Path(source).read_text(encoding="utf-8", errors="replace")
    stripped, removed = remove_sections(
        content, [(int(mf), None) for mf in mf_numbers])
    Path(output).write_text(stripped, encoding="utf-8")
    update_mf1_directory(str(output))
    return {
        "source": str(source),
        "output": str(output),
        "mf_removed": [int(mf) for mf in mf_numbers],
        "sections_removed": int(removed),
        "bytes_before": len(content),
        "bytes_after": Path(output).stat().st_size,
    }


def build_base_tape(
    source: str,
    output: str,
    *,
    mf_numbers: Sequence[int] = COVARIANCE_MF,
    verify_mf: Iterable[int] = (3, 4),
    logger=None,
) -> Dict[str, object]:
    """:func:`strip_covariance_files` plus the check that the physics survived.

    ``verify_mf`` names the files whose content must come through untouched —
    MF3 and MF4 by default, the two the ACE chain is built from and the two the
    perturbation writes into. Their digests are taken **before and after** and
    compared; a mismatch raises, because a base tape whose MF4 is not the
    deliverable's centres the whole ensemble somewhere the covariance does not
    describe.

    Returns the report. Write it next to the tape: it is the evidence for the
    gate, and a stripped tape carries no record of what it was stripped from.
    """
    before = {int(mf): mf_section_digest(str(source), int(mf))
              for mf in verify_mf}
    report = strip_covariance_files(source, output, mf_numbers)
    after = {int(mf): mf_section_digest(str(output), int(mf))
             for mf in verify_mf}

    moved = [mf for mf in before if before[mf] != after[mf]]
    if moved:
        detail = "; ".join(
            f"MF{mf}: {before[mf][1]} records / {before[mf][0][:12]} -> "
            f"{after[mf][1]} / {after[mf][0][:12]}" for mf in moved)
        raise ValueError(
            f"stripping the covariance moved MF{moved} ({detail}). The base "
            f"tape must differ from the source ONLY in the covariance files; "
            f"anything else means the replicas would be centred on a different "
            f"evaluation than the covariance that generated them."
        )
    report["verified_mf"] = {int(mf): {"sha256": before[mf][0],
                                       "records": before[mf][1]}
                             for mf in before}
    report["shrink_factor"] = (report["bytes_before"] / report["bytes_after"]
                               if report["bytes_after"] else None)
    if logger is not None:
        logger.info(
            f"[BASE] {report['bytes_before']/1e6:.1f} MB -> "
            f"{report['bytes_after']/1e6:.1f} MB "
            f"(x{report['shrink_factor']:.1f}), MF{list(mf_numbers)} removed, "
            f"MF{sorted(before)} byte-identical")
    return report
