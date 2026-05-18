import os
import warnings


_XSDIR_MAX_COL = 79  # '+' continuation must sit strictly before column 80


def _wrap_xsdir_fields(fields: list[str]) -> str:
    """
    Pack xsdir fields into physical lines whose length never exceeds
    _XSDIR_MAX_COL (79). Breaks only occur between fields. Non-final physical
    lines end with " +" so the '+' occupies column <= 79. Continuation lines
    start at column 1 (no indent).
    """
    lines: list[str] = []
    current = ""
    for i, field in enumerate(fields):
        is_last = i == len(fields) - 1
        limit = _XSDIR_MAX_COL if is_last else _XSDIR_MAX_COL - 2  # reserve " +"
        if not current:
            current = field
            if len(current) > limit:
                warnings.warn(
                    f"xsdir field too long to fit in {_XSDIR_MAX_COL} cols: {field!r}",
                    stacklevel=3,
                )
            continue
        trial = current + " " + field
        if len(trial) <= limit:
            current = trial
        else:
            lines.append(current + " +")
            current = field
    if current:
        lines.append(current)
    return "\n".join(lines)


def build_xsdir_line(
    zaid_with_ext: str,
    awr: float,
    file_ref: str,
    xss_len: int,
    kT_MeV: float,
    *,
    dir_field: str | int = 0,
    file_type: int = 1,
    address: int = 1,
    record_len: int = 0,
    entries_per_record: int = 0,
    has_ptable: bool = False,
) -> str:
    """
    Build one xsdir entry. Long entries are wrapped at field boundaries with
    a trailing " +" continuation marker and 10-space indent on subsequent
    physical lines so that no physical line exceeds 80 columns.

    Parameters
    ----------
    zaid_with_ext : str
        Full table name as it must appear in xsdir, e.g. "22048.01c".
    awr : float
        Atomic weight ratio (AWR).
    file_ref : str
        File reference in column 3. Can be a relative path or just a filename.
    xss_len : int
        Length of the XSS array (NXS(2)).
    kT_MeV : float
        Temperature in MeV (kT).

    Keyword-only parameters
    -----------------------
    dir_field : str | int, default 0
    file_type : int, default 1
        1 = text (sequential), 2 = binary (direct access).
    address : int, default 1
    record_len : int, default 0
    entries_per_record : int, default 0
    has_ptable : bool, default False
        Whether to append the trailing "ptable" token.

    Returns
    -------
    str
        The complete xsdir entry (may contain embedded "\n" for continuation).
    """
    fields = [
        zaid_with_ext,
        f"{awr:.6f}",
        file_ref,
        str(dir_field),
        str(file_type),
        str(address),
        str(xss_len),
        str(record_len),
        str(entries_per_record),
        f"{kT_MeV:.3E}",
    ]
    if has_ptable:
        fields.append("ptable")
    return _wrap_xsdir_fields(fields)


def write_xsdir_line(xsdir_path: str, line: str, mode: str = "w") -> None:
    """Write a single xsdir entry (possibly multi-line) to file."""
    with open(xsdir_path, mode) as fx:
        fx.write(line + "\n")


def create_xsdir_files_for_ace(
    ace_file_path: str,
    zaid: int,
    awr: float,
    xss_len: int,
    temperature_mev: float,
    sample_index: int,
    output_dir: str,
    master_xsdir_file: str = None,
    has_ptable: bool = False,
    per_sample_xsdir_path: str = None,
) -> None:
    """
    Create xsdir files for a perturbed ACE file.

    Creates both per-sample xsdir files and optionally updates a master xsdir
    file. Shared between ACE and ENDF perturbation workflows.

    Parameters
    ----------
    per_sample_xsdir_path : str, optional
        Full path to write the per-sample xsdir snippet. When None, the
        legacy layout `sample_dir/{base}_{sample_str}.xsdir` is used.
        Pass the new-layout path (e.g. `output_dir/ace/<NNNN>/xsdir/<ZAID>.<ext>.xsdir`)
        from pipelines that already provisioned a per-sample xsdir
        directory and want their xsdir to land there.
    """
    import shutil

    base, ace_ext = os.path.splitext(os.path.basename(ace_file_path))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.dirname(ace_file_path)

    if base.endswith(f"_{sample_str}"):
        base = base[:-len(f"_{sample_str}")]

    rel = os.path.relpath(ace_file_path, output_dir).replace(os.sep, "/")
    rel_path = f"../{rel}"

    xsdir_line = build_xsdir_line(
        zaid_with_ext=f"{zaid}{ace_ext}",
        awr=awr,
        file_ref=rel_path,
        xss_len=xss_len,
        kT_MeV=temperature_mev,
        has_ptable=has_ptable,
    )

    if per_sample_xsdir_path is None:
        xsdir_path = os.path.join(sample_dir, f"{base}_{sample_str}.xsdir")
    else:
        xsdir_path = per_sample_xsdir_path
        os.makedirs(os.path.dirname(xsdir_path) or ".", exist_ok=True)
    write_xsdir_line(xsdir_path, xsdir_line, mode="w")

    if master_xsdir_file:
        xsdir_dir = os.path.join(output_dir, "xsdir")
        os.makedirs(xsdir_dir, exist_ok=True)

        sample_tag = f"_{sample_index+1:04d}"
        orig_xs_base = os.path.splitext(os.path.basename(master_xsdir_file))[0]
        master_xs_name = orig_xs_base + sample_tag
        master_xs_path = os.path.join(xsdir_dir, master_xs_name)

        if os.path.exists(master_xs_path):
            source_xsdir = master_xs_path
        else:
            source_xsdir = master_xsdir_file

        with open(source_xsdir, 'r') as f:
            xs_lines = f.readlines()

        zaid_prefix = f"{zaid}{ace_ext}"
        line_found = False

        with open(master_xs_path, 'w') as f:
            i = 0
            # Continuation is signaled by a trailing '+' on the previous physical
            # line; leading whitespace on entry lines is allowed (and standard in
            # JENDL-style xsdirs) and must NOT be treated as a continuation marker.
            prev_continues = False
            while i < len(xs_lines):
                line = xs_lines[i]
                stripped = line.strip()
                first_token = stripped.split(None, 1)[0] if stripped else ""
                if first_token == zaid_prefix and not prev_continues:
                    f.write(xsdir_line + "\n")
                    line_found = True
                    this_continues = stripped.endswith("+")
                    i += 1
                    while i < len(xs_lines) and this_continues:
                        this_continues = xs_lines[i].rstrip("\n").rstrip().endswith("+")
                        i += 1
                    prev_continues = False
                else:
                    f.write(line)
                    if stripped:
                        prev_continues = stripped.endswith("+")
                    i += 1

            if not line_found:
                if xs_lines and not xs_lines[-1].endswith('\n'):
                    f.write('\n')
                f.write(xsdir_line + "\n")


def _parse_xsdir_entry(text: str) -> list[str]:
    """
    Parse an xsdir entry (possibly wrapped across physical lines with trailing
    '+' continuation markers) back into its list of fields.
    """
    joined_parts = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith("+"):
            stripped = stripped[:-1].rstrip()
        joined_parts.append(stripped)
    return " ".join(joined_parts).split()


def _read_ace_zaid(ace_path: str) -> int:
    """Read just the ZAID from the first line of an ACE file."""
    import re
    with open(ace_path, 'r') as f:
        line1 = f.readline()
    # 2.0.1 format: ZAID lives in cols 10-22; legacy: starts at col 0.
    zaid_field = line1[10:22] if "2.0.1" in line1[:10] else line1[:20]
    m = re.match(r'\s*(\d+)', zaid_field)
    if not m:
        raise ValueError(f"Could not parse ZAID from first line of {ace_path}: {line1!r}")
    return int(m.group(1))


def _rewrap_single_entry_file(path: str) -> bool:
    """
    Rewrite a per-sample .xsdir file that contains a single entry, re-wrapping
    it with the current rules. No-op if the file already complies.
    """
    with open(path, 'r') as f:
        content = f.read()
    fields = _parse_xsdir_entry(content)
    if not fields:
        return False
    new = _wrap_xsdir_fields(fields) + "\n"
    if new == content:
        return False
    with open(path, 'w') as f:
        f.write(new)
    return True


def _write_master_with_replacements(
    out_path: str,
    base_lines: list[str],
    replacements: dict[str, str],
) -> None:
    """
    Write a master xsdir file at `out_path` using `base_lines` as the
    starting point. For each `zaid_prefix -> new_entry` pair in
    `replacements`, replace the baseline's matching entry (and swallow its
    old continuation lines). All other baseline bytes are preserved.
    """
    used: set[str] = set()
    out: list[str] = []
    i = 0
    n = len(base_lines)
    # Continuation is signaled by a trailing '+' on the previous physical line;
    # leading whitespace on entry lines is allowed (and standard in JENDL-style
    # xsdirs) and must NOT be treated as a continuation marker.
    prev_continues = False
    while i < n:
        line = base_lines[i]
        stripped = line.strip()
        first_token = stripped.split(None, 1)[0] if stripped else ""
        matched_prefix = first_token if (first_token in replacements and not prev_continues) else None
        if matched_prefix is not None:
            out.append(replacements[matched_prefix] + "\n")
            used.add(matched_prefix)
            this_continues = stripped.endswith("+")
            i += 1
            while i < n and this_continues:
                this_continues = base_lines[i].rstrip("\n").rstrip().endswith("+")
                i += 1
            prev_continues = False
        else:
            out.append(line)
            if stripped:
                prev_continues = stripped.endswith("+")
            i += 1

    # Append any replacements whose zaid wasn't present in the baseline.
    missing = [p for p in replacements if p not in used]
    if missing:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        for p in missing:
            out.append(replacements[p] + "\n")

    with open(out_path, 'w') as f:
        f.writelines(out)


def _rewrite_xsdir_for_zaids(
    zaids,
    output_dir: str,
    xsdir_file: str | None,
    logger=None,
) -> int:
    """
    Regenerate xsdir output for the given zaids from scratch:

      1. Re-emit each per-sample .xsdir file (one entry, properly wrapped)
         under output_dir/ace/<temp>/<zaid>/<sample>/ — fields are taken
         from the existing per-sample .xsdir contents (no ACE parsing).

      2. For each sample, copy `xsdir_file` and substitute every perturbed
         zaid's entry with its wrapped version, writing the result to
         output_dir/xsdir/<baseline_basename>_<NNNN>. All non-perturbed lines
         in the baseline xsdir are preserved byte-for-byte.

    Returns the number of files written/modified.
    """
    zaids_set = {str(int(z)) for z in zaids}
    ace_root = os.path.join(output_dir, "ace")
    if not os.path.isdir(ace_root):
        if logger:
            logger.warning(f"  [WARN] [XSDIR] ACE root does not exist: {ace_root}")
        return 0

    # sample_index -> { zaid_prefix (e.g. "10010.81c") -> wrapped_entry }
    per_sample: dict[int, dict[str, str]] = {}
    count = 0

    for temp_str in os.listdir(ace_root):
        temp_dir = os.path.join(ace_root, temp_str)
        if not os.path.isdir(temp_dir):
            continue
        for zaid_str in os.listdir(temp_dir):
            if zaid_str not in zaids_set:
                continue
            zaid_dir = os.path.join(temp_dir, zaid_str)
            if not os.path.isdir(zaid_dir):
                continue
            for sample_str in os.listdir(zaid_dir):
                sample_dir = os.path.join(zaid_dir, sample_str)
                if not os.path.isdir(sample_dir):
                    continue
                try:
                    sample_index = int(sample_str)
                except ValueError:
                    continue
                for fname in os.listdir(sample_dir):
                    if not fname.endswith(".xsdir"):
                        continue
                    path = os.path.join(sample_dir, fname)
                    try:
                        with open(path, 'r') as f:
                            content = f.read()
                        fields = _parse_xsdir_entry(content)
                        if not fields:
                            continue
                        new_entry = _wrap_xsdir_fields(fields)
                        if new_entry + "\n" != content:
                            with open(path, 'w') as f:
                                f.write(new_entry + "\n")
                            count += 1
                        per_sample.setdefault(sample_index, {})[fields[0]] = new_entry
                    except Exception as e:
                        if logger:
                            logger.error(
                                f"  [ERROR] [XSDIR] Failed to rewrap {path}: {e}"
                            )

    if xsdir_file and os.path.isfile(xsdir_file):
        with open(xsdir_file, 'r') as f:
            base_lines = f.readlines()
        xsdir_out_dir = os.path.join(output_dir, "xsdir")
        os.makedirs(xsdir_out_dir, exist_ok=True)
        orig_base = os.path.splitext(os.path.basename(xsdir_file))[0]
        for sample_index, repls in sorted(per_sample.items()):
            out_name = f"{orig_base}_{sample_index:04d}"
            out_path = os.path.join(xsdir_out_dir, out_name)
            try:
                _write_master_with_replacements(out_path, base_lines, repls)
                count += 1
            except Exception as e:
                if logger:
                    logger.error(
                        f"  [ERROR] [XSDIR] Failed to write {out_path}: {e}"
                    )
    elif logger:
        logger.warning(
            f"  [WARN] [XSDIR] xsdir_file not provided or missing; per-sample "
            f"master xsdir files will NOT be regenerated."
        )

    if logger:
        logger.info(
            f"  [INFO] [XSDIR] only_xsdir: {count} file(s) written for "
            f"{len(per_sample)} sample(s), zaids {sorted(zaids_set)}"
        )
    return count
