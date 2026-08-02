"""Read SCALE Sensitivity Data Files into the KIKA SDF model.

The public model stores groupwise sensitivity uncertainties and response
uncertainty as absolute one-sigma standard deviations. Standard SCALE files
therefore use uncertainty_convention="absolute" by default.

Historical KIKA files used relative uncertainties and carry a distinctive
"MCNP to SCALE sdf" header. Read them explicitly with
uncertainty_convention="relative"; conversion occurs at the boundary.

External SCALE energies are converted from eV to internal MeV. Group data
remain the source of truth; all five reaction scalars are checked and any
discrepancy is logged with file, ZAID and MT context. Writing always produces
the SCALE-compatible absolute-uncertainty dialect.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
import logging
import re
import numpy as np

from .sdf import SDFData, SDFReactionData

logger = logging.getLogger(__name__)

# Magic header produced by historical KIKA writers; the title precedes it. When this
# matches we recover the bare title before migrating to the standard dialect.
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


def _parse_scientific_numbers(line: str) -> List[float]:
    return [float(x) for x in re.findall(FLOAT_PATTERN, line)]


def read_sdf(
    path: str,
    *,
    uncertainty_convention: Literal["absolute", "relative"] = "absolute",
) -> SDFData:
    """Parse an SDF file returning an ``SDFData`` instance.

    Parameters
    ----------
    path : str
        Path to SDF file.
    uncertainty_convention : {"absolute", "relative"}
        SCALE uses absolute standard deviations; "relative" supports files
        written by historical KIKA versions.
    """
    if uncertainty_convention not in {"absolute", "relative"}:
        raise ValueError("uncertainty_convention must be 'absolute' or 'relative'")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    with p.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    idx = 0
    if idx >= len(lines):
        raise ValueError("Empty SDF file")

    # Line 1: title. KIKA's writer appends a magic " MCNP to SCALE sdf <N>gr"
    # suffix; recover the bare title while applying the requested legacy conversion.
    # Standard SCALE files use a free-form first line, which we keep verbatim.
    header_match = HEADER_RE.match(lines[idx])
    if header_match and uncertainty_convention == "absolute":
        raise ValueError(
            "This file has the historical KIKA relative-uncertainty header. "
            "Read it with uncertainty_convention='relative'."
        )
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
    if uncertainty_convention == "relative":
        e0 = abs(r0) * abs(e0)
    else:
        e0 = abs(e0)
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

    # Standard SCALE files store eV; historical KIKA files stored internal MeV.
    if not header_match:
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
        compact = False
        try:
            # Compact TSUNAMI-1D headers end in ``ZAID MT unit value``.
            # Verbose SCALE headers end in ``ZAID MT`` and may use a reaction
            # label containing whitespace (for example "nu-bar total").
            if (
                len(parts) == 6
                and parts[-4].isdigit()
                and parts[-3].isdigit()
                and parts[-2].lstrip("+-").isdigit()
            ):
                float(parts[-1])
                zaid, mt = int(parts[-4]), int(parts[-3])
                reaction_name = " ".join(parts[1:-4])
                compact = True
            elif len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
                zaid, mt = int(parts[-2]), int(parts[-1])
                reaction_name = " ".join(parts[1:-2])
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"Malformed reaction header at line {idx+1}: {line}") from None
        if not reaction_name:
            raise ValueError(f"Missing reaction name at line {idx+1}: {line}")
        idx += 1

        scalar_nums = None
        if compact:
            # unit is inline; compact profiles are region-integrated -> region 0.
            unit, region = int(parts[-2]), 0
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

        sensitivity_array = np.asarray(sens_vals, dtype=float)
        raw_error = np.asarray(err_vals, dtype=float)
        if uncertainty_convention == "relative":
            error_array = np.abs(sensitivity_array) * np.abs(raw_error)
        else:
            error_array = np.abs(raw_error)
        err_vals = error_array.tolist()

        if scalar_nums is not None:
            integral = float(np.sum(sensitivity_array))
            integral_std = float(np.sqrt(np.sum(error_array ** 2)))
            sum_abs = float(np.sum(np.abs(sensitivity_array)))
            sum_error_abs = float(np.sum(error_array))
            # The scalar integral was evaluated before each group value was
            # rounded to ES14.6. Use its sign to reconstruct OSC; otherwise a
            # near-zero constrained-chi integral can flip sign after rounding.
            integral_for_sign = scalar_nums[0]
            if integral_for_sign > 0.0:
                osc_mask = sensitivity_array < 0.0
            elif integral_for_sign < 0.0:
                osc_mask = sensitivity_array > 0.0
            elif scalar_nums[3] < 0.0:
                # A constrained profile can have an exactly zero stored
                # integral. Its stored OSC sign identifies the selected side.
                osc_mask = sensitivity_array < 0.0
            elif scalar_nums[3] > 0.0:
                osc_mask = sensitivity_array > 0.0
            else:
                osc_mask = np.zeros_like(sensitivity_array, dtype=bool)
            osc = float(np.sum(sensitivity_array[osc_mask]))
            osc_std = float(np.sqrt(np.sum(error_array[osc_mask] ** 2)))

            mismatches = []
            derived = (integral, integral_std, sum_abs, osc, osc_std)
            names = (
                "integral",
                "integral uncertainty",
                "absolute sum",
                "oscillation",
                "oscillation uncertainty",
            )
            # Accumulating N group values rounded to ES14.6 can differ from the
            # independently rounded scalar by up to roughly 5e-7 times their
            # absolute sum. Values inside that envelope are formatting noise.
            value_atol = 5.0e-7 * sum_abs + 1.0e-15
            uncertainty_atol = 5.0e-7 * sum_error_abs + 1.0e-15
            atols = (
                value_atol,
                uncertainty_atol,
                value_atol,
                value_atol,
                uncertainty_atol,
            )
            for name, expected, stored, atol in zip(
                names,
                derived,
                scalar_nums,
                atols,
            ):
                stored_value = abs(stored) if "uncertainty" in name else stored
                if not np.isclose(
                    expected,
                    stored_value,
                    rtol=5e-7,
                    atol=atol,
                ):
                    mismatches.append(
                        f"{name} (groups={expected:.8E}, stored={stored_value:.8E})"
                    )
            if mismatches:
                logger.warning(
                    "SDF scalar mismatch in %s for ZAID=%s MT=%s: %s",
                    path,
                    zaid,
                    mt,
                    "; ".join(mismatches),
                )

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
    """Return whether parse/write preserves the normalized SDF values."""
    import tempfile

    first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    convention = "relative" if HEADER_RE.match(first_line) else "absolute"
    sdf = read_sdf(path, uncertainty_convention=convention)
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="sdf_rt_")
    sdf.write_file(tmp_dir)
    produced_name = f"{sdf.title}_{sdf.energy}.sdf"
    produced_name = produced_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    produced = Path(tmp_dir) / produced_name
    if not produced.exists():
        return False
    restored = read_sdf(str(produced))

    def reaction_key(reaction):
        return reaction.zaid, reaction.mt, reaction.unit or 0, reaction.region or 0

    expected_reactions = sorted(sdf.data, key=reaction_key)
    actual_reactions = sorted(restored.data, key=reaction_key)
    if len(expected_reactions) != len(actual_reactions):
        return False
    reactions_equal = all(
        reaction_key(expected) == reaction_key(actual)
        and np.allclose(expected.sensitivity, actual.sensitivity, rtol=1e-6, atol=1e-12)
        and np.allclose(expected.error, actual.error, rtol=1e-6, atol=1e-12)
        for expected, actual in zip(expected_reactions, actual_reactions)
    )
    return (
        reactions_equal
        and np.allclose(sdf.pert_energies, restored.pert_energies, rtol=1e-6)
        and np.isclose(sdf.r0 or 0.0, restored.r0 or 0.0, rtol=1e-6)
        and np.isclose(sdf.e0 or 0.0, restored.e0 or 0.0, rtol=1e-6)
    )
