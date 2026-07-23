"""Parser for legacy SDF (Sensitivity Data File) format.

This module reads two dialects of the SDF format into the same ``SDFData``
object:

* **KIKA's own writer output** (``sdf.SDFData.write_file``) – for which parsing
  and re-writing yields a byte‑for‑byte identical file (modulo line endings,
  which we normalise to ``\n`` on read).
* **Standard SCALE / TSUNAMI SDF files** produced by other tools (e.g. the OECD
  MCNP→SCALE conversion script). These use a free‑form title line, lowercase
  ``e`` scientific notation, a ``k-eff`` line with plain decimals and trailing
  comment text, and energies expressed in **eV**.

To keep the rest of the library consistent, foreign files are *normalised* to
KIKA's canonical form on read: the title is stored verbatim, and energy
boundaries are converted to **MeV** (KIKA's internal/plotting/grid convention).
As a consequence, re-writing a foreign file will NOT reproduce it byte-for-byte
(the writer emits KIKA's header and MeV grid) – only KIKA-generated files
round-trip exactly.

We therefore:

1. Preserve ordering of reactions exactly as they appear in the file.
2. Do not attempt to 'normalise' floating point formatting – we store floats as
   Python floats but reproduce writer formatting which matches the original
   writer (scientific notation with 6 decimals, width 14, right aligned, 5 per line).
3. Reconstruct the perturbation energy boundaries in ascending order (writer
   reverses them when outputting).  The file lists the boundaries in descending
   order immediately under the line ``energy boundaries:`` with 5 values per line.

The legacy block structure for each reaction (as produced by
``SDFData._format_reaction_data``) is:

<nuclideSymbol(ljust13)><reactionName(ljust17)><ZAID(>5)><MT(>7)><\n>
"      0      0"\n
"  0.000000E+00  0.000000E+00      0      0"\n
<5 derived scalar values (ignored on parse, recomputed on write)>\n
<group sensitivities reversed (5 per line)>\n
<group errors reversed (5 per line)>\n
We intentionally IGNORE the 5 scalar values while parsing because the writer
recomputes them; keeping the parsed numbers could conceal inconsistencies.

API
---
parse_sdf(path: str) -> SDFData
    Parse file and return ``SDFData`` instance.

roundtrip_equal(path: str, tmp_dir: Optional[str] = None) -> bool
    Convenience helper that parses, writes to a temporary directory and
    performs textual comparison with the original file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging
import re

from .sdf import SDFData, SDFReactionData
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, MT_TO_REACTION

logger = logging.getLogger(__name__)

# Magic header produced by KIKA's own writer; the title precedes it. When this
# matches we recover the *bare* title so KIKA files round-trip byte-identically.
# Foreign files (free-form first line) don't match and the whole line is the title.
HEADER_RE = re.compile(r"^(?P<title>.+) MCNP to SCALE sdf (?P<ngroups>\d+)gr\s*$")
NGROUP_LINE_RE = re.compile(r"^\s*(?P<ngroups>\d+) number of neutron groups\s*$")
NPROF_LINE_RE = re.compile(r"^\s*(?P<nprofiles>\d+)\s+ number of sensitivity profiles\s+(?P<nprofiles2>\d+) are region integrated\s*$")
# The second and third fixed lines inside a reaction block are literal
FIXED_LINE_1 = "      0      0"
FIXED_LINE_2 = "  0.000000E+00  0.000000E+00      0      0"

# Permissive float matcher: accepts lowercase/uppercase ``e``, an optional
# exponent (plain decimals such as the k-eff value ``1.003930``), and multi-digit
# mantissas. Covers both KIKA's writer (uppercase ``E``) and standard SCALE files.
FLOAT_PATTERN = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"

# Metadata line 1 inside a reaction block: two leading integer columns
# (unit, region). Standard SCALE files may append a free-text label directly
# after the integers (e.g. "      1      1zpr-9/31 loading 22"), so match only
# the two leading integers and ignore any trailing text.
_META1_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)")

# Above ~1 GeV in "MeV" is unphysical for the reactor/criticality domain this
# tool targets (neutron grids top out near 20 MeV), so a grid whose maximum
# boundary exceeds this threshold is interpreted as eV and converted to MeV.
EV_DETECTION_THRESHOLD_MEV = 1.0e3


def _parse_scientific_numbers(line: str) -> List[float]:
    return [float(x) for x in re.findall(FLOAT_PATTERN, line)]


def read_sdf(path: str) -> SDFData:
    """Parse an SDF file returning an ``SDFData`` instance.

    Parameters
    ----------
    path : str
        Path to SDF file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    with p.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    idx = 0
    if idx >= len(lines):
        raise ValueError("Empty SDF file")

    # Line 1: title. KIKA's writer appends a magic " MCNP to SCALE sdf <N>gr"
    # suffix; recover the bare title from it so KIKA files round-trip exactly.
    # Standard SCALE files use a free-form first line, which we keep verbatim.
    header_match = HEADER_RE.match(lines[idx])
    title = header_match.group("title") if header_match else lines[idx]
    idx += 1

    # Line 2: authoritative neutron-group count (KIKA's line-1 suffix may be
    # absent, so we always read ngroups from here).
    m = NGROUP_LINE_RE.match(lines[idx]) if idx < len(lines) else None
    if not m:
        raise ValueError("Missing/invalid neutron groups line")
    ngroups_declared = int(m.group("ngroups"))
    idx += 1

    m = NPROF_LINE_RE.match(lines[idx]) if idx < len(lines) else None
    if not m:
        raise ValueError("Missing/invalid sensitivity profiles line")
    nprofiles_declared = int(m.group("nprofiles"))
    # Cross check second number. In standard SCALE/TSUNAMI files line 3 carries
    # both the total profile count and the smaller energy-integrated count, so a
    # mismatch is the common case (only KIKA's own writer sets them equal). Keep
    # the larger (total) and note it at debug level rather than reject the file.
    if int(m.group("nprofiles2")) != nprofiles_declared:
        logger.debug(
            "SDF profile counts differ on line 3 (%s vs %s); using %s",
            nprofiles_declared, m.group("nprofiles2"), nprofiles_declared,
        )
    idx += 1

    # k-eff line: "<r0> +/- <e0>" with a trailing comment (e.g.
    # "1.003930 +/- 0.000290  k-eff from the forward case"). Some standard SCALE
    # files omit the uncertainty entirely ("1.000717    k-eff from the forward
    # case"); in that case e0 is taken as 0.
    if idx >= len(lines):
        raise ValueError("Missing r0/e0 line")
    kline = lines[idx]
    if "+/-" in kline:
        before, _, after = kline.partition("+/-")
        r0_nums = _parse_scientific_numbers(before)
        e0_nums = _parse_scientific_numbers(after)
        if not r0_nums or not e0_nums:
            raise ValueError(f"Could not parse r0/e0 from line: '{kline}'")
        r0, e0 = r0_nums[0], e0_nums[0]
    else:
        r0_nums = _parse_scientific_numbers(kline)
        if not r0_nums:
            raise ValueError(f"Could not parse r0 from line: '{kline}'")
        r0, e0 = r0_nums[0], 0.0
    idx += 1

    if idx >= len(lines) or lines[idx].strip() != "energy boundaries:":
        raise ValueError("Expected 'energy boundaries:' line")
    idx += 1

    # Collect energy boundary numbers until we have ngroups+1 values.
    energy_values: List[float] = []
    while idx < len(lines) and len(energy_values) < (ngroups_declared + 1):
        nums = _parse_scientific_numbers(lines[idx])
        energy_values.extend(nums)
        idx += 1
    if len(energy_values) != (ngroups_declared + 1):
        raise ValueError("Energy boundaries count mismatch")

    # File stores descending order; convert to ascending for internal object.
    pert_energies = list(reversed(energy_values))

    # Normalise units to MeV (KIKA's internal convention for plotting, grid
    # identification and covariance). Standard SCALE files express the grid in
    # eV; detect that from the maximum boundary and convert. The threshold is
    # intentionally conservative (see EV_DETECTION_THRESHOLD_MEV).
    if pert_energies and max(pert_energies) > EV_DETECTION_THRESHOLD_MEV:
        pert_energies = [e / 1.0e6 for e in pert_energies]

    reactions: List[SDFReactionData] = []

    # Parse reaction blocks until we have the declared number of profiles. We stop
    # at nprofiles_declared rather than EOF so that any trailing footer (standard
    # SCALE files append a "file verification information" block) is ignored.
    while idx < len(lines) and len(reactions) < nprofiles_declared:
        line = lines[idx]
        if not line.strip():
            # skip blank lines (should not normally happen, but be tolerant)
            idx += 1
            continue
        # Reaction header. Tokenise on whitespace (column widths vary across tools).
        # Two SCALE dialects occur, distinguished by token count:
        #   verbose (TSUNAMI-3D): "<form> <reaction> <zaid> <mt>"                 (4 tokens)
        #   compact (TSUNAMI-1D): "<form> <reaction> <zaid> <mt> <unit> <value>"  (6 tokens)
        # The verbose form is followed by two metadata lines, a 5-scalar line, then
        # groupwise sensitivities AND errors. The compact form is followed by a
        # single summary line, then groupwise sensitivities ONLY (no error block);
        # every compact profile is a single-region system total.
        parts = line.split()
        if len(parts) not in (4, 6) or not (parts[2].isdigit() and parts[3].isdigit()):
            raise ValueError(f"Malformed reaction header at line {idx+1}: {line}")
        reaction_name = parts[1]
        zaid = int(parts[2])
        mt = int(parts[3])
        compact = len(parts) == 6
        idx += 1

        if compact:
            # unit is inline; compact profiles are region-integrated -> region 0.
            unit, region = int(parts[4]), 0
            # One summary line (integrated sensitivity etc.) - read and discard.
            if idx >= len(lines) or not _parse_scientific_numbers(lines[idx]):
                raise ValueError(f"Missing compact summary line at line {idx+1}")
            idx += 1
        else:
            # Two metadata lines follow the reaction header. KIKA's writer emits the
            # literal FIXED_LINE_1/FIXED_LINE_2 (all zeros), but standard SCALE/TSUNAMI
            # files put varying integer indices on the first line (e.g. "-1  0") and
            # may right-pad with trailing whitespace or glue on a label. Validate
            # structurally rather than by exact match, then discard.
            if idx >= len(lines):
                raise ValueError(f"Missing metadata line 1 after reaction header at line {idx+1}")
            meta1 = _META1_RE.match(lines[idx])
            if not meta1:
                raise ValueError(
                    f"Malformed metadata line 1 after reaction header at line {idx+1}: {lines[idx]!r}"
                )
            # (unit, region) indices: (0, 0) marks a region-integrated system total.
            unit, region = int(meta1.group(1)), int(meta1.group(2))
            idx += 1
            if idx >= len(lines):
                raise ValueError(f"Missing metadata line 2 after reaction header at line {idx+1}")
            # Two floats + two integers (optionally followed by a label); values unused.
            if len(lines[idx].split()) < 4:
                raise ValueError(
                    f"Malformed metadata line 2 after reaction header at line {idx+1}: {lines[idx]!r}"
                )
            idx += 1

            # Scalar values line (5 numbers) - read and discard.
            if idx >= len(lines):
                raise ValueError("Unexpected EOF reading scalar values line")
            scalar_nums = _parse_scientific_numbers(lines[idx])
            if len(scalar_nums) != 5:
                raise ValueError(f"Expected 5 scalar values, got {len(scalar_nums)} at line {idx+1}")
            idx += 1

        # Read groupwise sensitivities (reversed order) until we have ngroups numbers
        sens_vals: List[float] = []
        while idx < len(lines) and len(sens_vals) < ngroups_declared:
            nums = _parse_scientific_numbers(lines[idx])
            if not nums:
                break
            sens_vals.extend(nums)
            idx += 1
        if len(sens_vals) != ngroups_declared:
            raise ValueError("Sensitivity values count mismatch")

        # Read errors (reversed order) until we have ngroups numbers. The compact
        # dialect has no error block, so errors default to zeros.
        if compact:
            err_vals = [0.0] * ngroups_declared
        else:
            err_vals = []
            while idx < len(lines) and len(err_vals) < ngroups_declared:
                nums = _parse_scientific_numbers(lines[idx])
                if not nums:
                    break
                err_vals.extend(nums)
                idx += 1
            if len(err_vals) != ngroups_declared:
                raise ValueError("Error values count mismatch")

        # Reverse back to ascending energy order
        sens_vals.reverse()
        err_vals.reverse()

        # Construct reaction data (nuclide symbol & reaction name resolved in __post_init__).
        # Provide reaction_name so unknown MT numbers are preserved.
        reaction = SDFReactionData(zaid=zaid, mt=mt, sensitivity=sens_vals, error=err_vals, reaction_name=reaction_name, unit=unit, region=region)
        reactions.append(reaction)

    if len(reactions) != nprofiles_declared:
        # Not fatal, but signal inconsistency.
        raise ValueError(f"Profile count mismatch: declared {nprofiles_declared}, parsed {len(reactions)}")

    # Attempt to reconstruct original energy label from filename if it contains two scientific numbers
    # separated by '_' (e.g. 1.00e-11_1.96e+01) to keep filename stable.
    energy_label = f"{pert_energies[0]:.2E}_{pert_energies[-1]:.2E}"  # fallback
    fname = p.name
    m_energy = re.search(r"(\d\.\d+e[+-]?\d+_\d\.\d+e[+-]?\d+)", fname, re.IGNORECASE)
    if m_energy:
        energy_label = m_energy.group(1)

    sdf_obj = SDFData(title=title, energy=energy_label, pert_energies=pert_energies, r0=r0, e0=e0, data=reactions)
    return sdf_obj


def roundtrip_equal(path: str, tmp_dir: Optional[str] = None) -> bool:
    """Roundtrip parse and write a file; return True if identical.

    The output filename is constructed by ``SDFData.write_file``; we compare
    the text content of the newly written file with the original path.
    """
    import tempfile, filecmp, os
    sdf = read_sdf(path)
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="sdf_rt_")
    sdf.write_file(tmp_dir)
    # Determine produced filename
    produced = Path(tmp_dir) / f"{sdf.title}_{sdf.energy}.sdf".replace(' ', '_').replace('/', '_').replace('\\', '_')
    if not produced.exists():
        return False
    # Compare textual content
    with open(path, 'r') as f1, open(produced, 'r') as f2:
        return f1.read() == f2.read()
