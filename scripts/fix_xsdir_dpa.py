#!/usr/bin/env python3
"""Repair the master xsdir of every replica of an ensemble.

Four defects, all inherited from the xsdir the ensembles were generated against
(`xsdir_MCNPy/xsdir40-irdff2`), all fixed in one pass:

1. **The DPA-600 tables are missing.** The NRT-DPA-600 / ARC-DPA-600 entries
   (`.65y` / `.66y` for Fe-54/56/57/58, Ni-58/60/61/62/64, Cu-63/65) are not in
   the directory, so a run asking for them dies in `getxst`. The 22 entries are
   appended after the last one.

2. **The IRDFF-II `.34y` entries overflow the 80-column record.** All 70 of them
   run to 81..86 characters, which puts their `+` continuation marker past
   column 80, where it is not a marker at all -- it is a truncated entry.

3. **Both blocks point at the wrong depth.** This is the one that costs a run.
   MCNP resolves a relative entry **against the directory the xsdir sits in**
   (it prints the list it searched when it fails), and a replica master sits two
   directories below the run root:

       <run>/<set>/xsdir/xsdir40-irdff2_0001     <- the file
       <run>/IRDFF-II/dos-irdff2-2225.acef       <- ../../../IRDFF-II/...
       <run>/xsec/NRT-DPA-600/26054.65y          <- ../../../xsec/...

   The depth is **measured, not counted**: for each file, walk up until the
   directory turns up (:func:`prefix_for`), and stat every rewritten entry
   afterwards to prove it points at something. Counting by eye is what produced
   `../IRDFF-II`, then `../../IRDFF-II`, and two dead runs. When a directory is
   not found the entries keep the depth they already carry -- inventing one is
   the failure mode, not the fix.

4. **The file ends on an empty line**, which the append in (1) would leave
   sitting in the middle of the directory section.

WHAT IT TOUCHES. Only **full** xsdir files -- the per-replica masters under
`<set>/xsdir/xsdir40-irdff2_NNNN`. A full xsdir is recognised by its `directory`
keyword; the one-entry snippets under `<set>/ace/NNNN/xsdir/` are not
directories and are left alone.

IDEMPOTENT. The corrected file is *rebuilt* from the current one rather than
patched into it, so a file already correct produces itself, a file fixed by an
earlier version of this script converges on the same bytes, and a half-written
one is repaired instead of appended to twice. Each file is written to a sibling
temp and renamed, so a kill leaves either the old file or the new one, never
half of one. What landed is read back and checked -- byte for byte, plus no line
over 80 columns and no trailing blank -- and the pass stops on the first file
that disagrees.

Usage
-----
    python fix_xsdir_dpa.py <root> [<root> ...]            # ensayo
    python fix_xsdir_dpa.py <root> [<root> ...] --apply
    python fix_xsdir_dpa.py <root> --apply --no-irdff      # no tocar IRDFF-II
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# The two blocks, as fields.
#
# They are held as fields and not as text because the **relative path is not a
# property of the library, it is a property of where the xsdir sits**. Writing a
# depth into a constant is how this gets fixed once and broken again on the next
# layout; :func:`prefix_for` measures it per file instead.
# --------------------------------------------------------------------------

#: Every entry in both blocks shares these: dir_field, file_type, address, then
#: record length and entries per record.
HEAD = "0 1 1"
TAIL = "0 0"
#: A continuation line is indented as the shipped xsdir indents it.
CONT_INDENT = 10
#: `kika.ace.xsdir` writes to 79 and not 80 because the `+` must sit strictly
#: before column 80. 80 satisfies "no more than 80 characters" and still puts
#: the marker on the boundary; one space of margin is cheaper than finding out.
WRITE_COL = 79
#: The hard bar, asserted on what was written.
MAX_COL = 80
EXPECTED_DATAPATH = "/soft_snc/lib/ace"

#: The DPA-600 damage tables: (ZAID, AWR, sub-directory, XSS length). The file
#: name is the ZAID. Supplied by hand; the ensembles' xsdir has none of them.
DPA_DIR = "xsec"
DPA_KT = "5.170E-08"
DPA_ENTRIES: list[tuple[str, str, str, str]] = [
    ("26054.65y", "53.476250", "NRT-DPA-600", "658684"),
    ("26056.65y", "55.454400", "NRT-DPA-600", "1028364"),
    ("26057.65y", "56.446300", "NRT-DPA-600", "325726"),
    ("26058.65y", "57.435610", "NRT-DPA-600", "405364"),
    ("28058.65y", "57.438000", "NRT-DPA-600", "1060070"),
    ("28060.65y", "59.415950", "NRT-DPA-600", "656658"),
    ("28061.65y", "60.408000", "NRT-DPA-600", "279140"),
    ("28062.65y", "61.396350", "NRT-DPA-600", "215014"),
    ("28064.65y", "63.379000", "NRT-DPA-600", "205466"),
    ("29065.65y", "64.370040", "NRT-DPA-600", "347926"),
    ("29063.65y", "62.389010", "NRT-DPA-600", "547078"),
    ("26054.66y", "53.476250", "ARC-DPA-600", "658684"),
    ("26056.66y", "55.454400", "ARC-DPA-600", "1028364"),
    ("26057.66y", "56.446300", "ARC-DPA-600", "325726"),
    ("26058.66y", "57.435610", "ARC-DPA-600", "405364"),
    ("28058.66y", "57.438000", "ARC-DPA-600", "1060070"),
    ("28060.66y", "59.415950", "ARC-DPA-600", "656658"),
    ("28061.66y", "60.408000", "ARC-DPA-600", "279140"),
    ("28062.66y", "61.396350", "ARC-DPA-600", "215014"),
    ("28064.66y", "63.379000", "ARC-DPA-600", "205466"),
    ("29065.66y", "64.370040", "ARC-DPA-600", "347926"),
    ("29063.66y", "62.389010", "ARC-DPA-600", "547078"),
]

#: The IRDFF-II dosimetry tables, taken field for field from the shipped
#: `xsdir40-irdff2`: (ZAID, AWR, .acef file, XSS length).
IRDFF_DIR = "IRDFF-II"
IRDFF_KT = "2.5300E-08"
IRDFF_ENTRIES: list[tuple[str, str, str, str]] = [
    ("3000.34y", "6.758268", "dos-irdff2-300.acef", "4066"),
    ("3006.34y", "5.963450", "dos-irdff2-325.acef", "3600"),
    ("3007.34y", "6.955732", "dos-irdff2-328.acef", "1996"),
    ("5000.34y", "10.811028", "dos-irdff2-500.acef", "10550"),
    ("5010.34y", "9.926921", "dos-irdff2-525.acef", "10050"),
    ("5011.34y", "10.914730", "dos-irdff2-528.acef", "2440"),
    ("9019.34y", "18.835200", "dos-irdff2-925.acef", "126"),
    ("11023.34y", "22.792280", "dos-irdff2-1125.acef", "14062"),
    ("12000.34y", "24.096261", "dos-irdff2-1200.acef", "506"),
    ("12024.34y", "23.779000", "dos-irdff2-1225.acef", "470"),
    ("13027.34y", "26.749800", "dos-irdff2-1325.acef", "3112"),
    ("14000.34y", "27.844231", "dos-irdff2-1400.acef", "1144"),
    ("14028.34y", "27.736600", "dos-irdff2-1425.acef", "1028"),
    ("14029.34y", "28.727600", "dos-irdff2-1428.acef", "186"),
    ("15031.34y", "30.707700", "dos-irdff2-1525.acef", "676"),
    ("16000.34y", "31.789335", "dos-irdff2-1600.acef", "564"),
    ("16032.34y", "31.697400", "dos-irdff2-1625.acef", "528"),
    ("21045.34y", "44.570000", "dos-irdff2-2125.acef", "72614"),
    ("22000.34y", "47.455546", "dos-irdff2-2200.acef", "1278"),
    ("22046.34y", "45.557900", "dos-irdff2-2225.acef", "358"),
    ("22047.34y", "46.548400", "dos-irdff2-2228.acef", "394"),
    ("22048.34y", "47.536000", "dos-irdff2-2231.acef", "176"),
    ("23051.34y", "50.506300", "dos-irdff2-2328.acef", "532"),
    ("24000.34y", "51.549459", "dos-irdff2-2400.acef", "96"),
    ("25055.34y", "54.466100", "dos-irdff2-2525.acef", "57604"),
    ("26000.34y", "55.365407", "dos-irdff2-2600.acef", "2492"),
    ("26054.34y", "53.476200", "dos-irdff2-2625.acef", "53014"),
    ("26056.34y", "55.454400", "dos-irdff2-2631.acef", "190"),
    ("26058.34y", "57.435610", "dos-irdff2-2637.acef", "97396"),
    ("27059.34y", "58.426900", "dos-irdff2-2725.acef", "110322"),
    ("28000.34y", "58.189142", "dos-irdff2-2800.acef", "684"),
    ("28058.34y", "57.437700", "dos-irdff2-2825.acef", "420"),
    ("28060.34y", "59.416000", "dos-irdff2-2831.acef", "172"),
    ("29000.34y", "63.000149", "dos-irdff2-2900.acef", "62872"),
    ("29063.34y", "62.389400", "dos-irdff2-2925.acef", "105212"),
    ("29065.34y", "64.370000", "dos-irdff2-2931.acef", "136"),
    ("30000.34y", "64.816156", "dos-irdff2-3000.acef", "68842"),
    ("30064.34y", "63.380000", "dos-irdff2-3025.acef", "192"),
    ("30067.34y", "66.352200", "dos-irdff2-3034.acef", "95314"),
    ("30068.34y", "67.341340", "dos-irdff2-3037.acef", "24198"),
    ("33075.34y", "74.278000", "dos-irdff2-3325.acef", "122"),
    ("39089.34y", "88.142100", "dos-irdff2-3925.acef", "164"),
    ("40000.34y", "90.439988", "dos-irdff2-4000.acef", "140"),
    ("40090.34y", "89.132400", "dos-irdff2-4025.acef", "130"),
    ("41093.34y", "92.108300", "dos-irdff2-4125.acef", "258524"),
    ("42000.34y", "95.135446", "dos-irdff2-4200.acef", "272"),
    ("42092.34y", "91.117200", "dos-irdff2-4225.acef", "246"),
    ("45103.34y", "102.021000", "dos-irdff2-4525.acef", "204"),
    ("47109.34y", "107.969000", "dos-irdff2-4731.acef", "157450"),
    ("48000.34y", "111.445418", "dos-irdff2-4800.acef", "475966"),
    ("49000.34y", "113.831744", "dos-irdff2-4900.acef", "79292"),
    ("49113.34y", "111.934000", "dos-irdff2-4925.acef", "270088"),
    ("49115.34y", "113.917000", "dos-irdff2-4931.acef", "216950"),
    ("53127.34y", "125.814000", "dos-irdff2-5325.acef", "146"),
    ("57139.34y", "137.713000", "dos-irdff2-5728.acef", "141416"),
    ("59141.34y", "139.697000", "dos-irdff2-5925.acef", "176"),
    ("64000.34y", "155.901222", "dos-irdff2-6400.acef", "417430"),
    ("69169.34y", "167.483000", "dos-irdff2-6925.acef", "494"),
    ("73181.34y", "179.393000", "dos-irdff2-7328.acef", "153116"),
    ("74186.34y", "184.357000", "dos-irdff2-7443.acef", "96224"),
    ("79197.34y", "195.274000", "dos-irdff2-7925.acef", "65088"),
    ("80199.34y", "197.259000", "dos-irdff2-8034.acef", "248"),
    ("82204.34y", "202.220000", "dos-irdff2-8225.acef", "242"),
    ("83209.34y", "207.185200", "dos-irdff2-8325.acef", "1592"),
    ("90232.34y", "230.045000", "dos-irdff2-9040.acef", "185128"),
    ("92235.34y", "233.024800", "dos-irdff2-9228.acef", "92760"),
    ("92238.34y", "236.005800", "dos-irdff2-9237.acef", "556324"),
    ("93237.34y", "235.011800", "dos-irdff2-9346.acef", "79138"),
    ("94239.34y", "236.998600", "dos-irdff2-9437.acef", "127196"),
    ("95241.34y", "238.986000", "dos-irdff2-9543.acef", "40860"),
]

#: How each block is recognised in a file that already has it. Both halves of
#: each pair matter: the ZAID pattern alone would also catch a table served from
#: somewhere else entirely.
DPA_RE = re.compile(r"^\d+\.6[56]y$")
DPA_REF = "DPA-600/"
IRDFF_RE = re.compile(r"^\d+\.34y$")
IRDFF_REF = "IRDFF-II/dos-irdff2-"
DPA_ZAIDS = [z for z, _a, _s, _x in DPA_ENTRIES]
IRDFF_ZAIDS = [z for z, _a, _f, _x in IRDFF_ENTRIES]


def format_block(rows: list[tuple[str, str, str, str]], kt: str) -> list[str]:
    """Write `rows` as xsdir entries inside :data:`WRITE_COL` columns.

    `rows` are (ZAID, AWR, reference, XSS length). The aligned one-line layout
    is tried first because it is the most readable, then the kT is moved to a
    continuation line, then the padding goes. The first that fits wins; if none
    does, this raises rather than emit an entry whose `+` lands past column 80,
    which MCNP does not read as a continuation.
    """
    zw = max(len(r[0]) for r in rows) + 1
    aw = max(len(r[1]) for r in rows) + 1
    rw = max(len(r[2]) for r in rows)
    xw = max(len(r[3]) for r in rows)

    def build(pad: bool, wrap: bool) -> list[str]:
        out = []
        for zaid, awr, ref, xss in rows:
            if pad:
                entry = (f"{zaid:>{zw}}{awr:>{aw}} {ref:<{rw}} "
                         f"{HEAD} {xss:>{xw}} {TAIL}")
            else:
                entry = f"{zaid} {awr} {ref} {HEAD} {xss} {TAIL}"
            if wrap:
                out.append(entry + " +")
                out.append(" " * CONT_INDENT + kt)
            else:
                out.append(f"{entry} {kt}")
        return out

    widest = 0
    for pad, wrap in ((True, False), (True, True), (False, True)):
        lines = build(pad, wrap)
        widest = max(len(ln) for ln in lines)
        if widest <= WRITE_COL:
            return lines
    raise SystemExit(
        f"las entradas no caben en {WRITE_COL} columnas ({widest}) con la ruta "
        f"{rows[0][2]!r}. Habria que partirlas en mas de dos lineas fisicas y "
        f"este script no lo hace.")


def dpa_block(prefix: str) -> list[str]:
    return format_block(
        [(z, a, f"{prefix}/{sub}/{z}", x) for z, a, sub, x in DPA_ENTRIES], DPA_KT)


def irdff_block(prefix: str) -> list[str]:
    return format_block(
        [(z, a, f"{prefix}/{fn}", x) for z, a, fn, x in IRDFF_ENTRIES], IRDFF_KT)


def prefix_for(xsdir_dir: Path, directory: str, probe: str,
               forced: str | None) -> tuple[str | None, str]:
    """Where `directory` is, as a path relative to the xsdir's own directory.

    Walks up from the xsdir looking for `<ancestor>/<directory>/<probe>`.
    Returns (prefix, how); `prefix` is None when nothing was found.
    """
    if forced:
        return forced.rstrip("/"), "forzado"
    here = xsdir_dir.resolve()
    for up in range(0, 9):
        cand = here.joinpath(*([".."] * up), directory)
        try:
            if (cand / probe).exists():
                return os.path.relpath(cand, here).replace(os.sep, "/"), "medido"
        except OSError:
            pass
    return None, "no encontrado"


def current_prefix(lines: list[str], rx: re.Pattern, ref_mark: str,
                   directory: str) -> str | None:
    """The depth the file already carries, so it can be kept when nothing else
    is known. `../../IRDFF-II/x.acef` -> `../../IRDFF-II`."""
    for ln in lines:
        stripped = ln.strip()
        first = stripped.split(None, 1)[0] if stripped else ""
        if rx.match(first) and ref_mark in ln:
            ref = stripped.split()[2]
            head = ref.split("/" + directory + "/")[0]
            return f"{head}/{directory}" if head else directory
    return None


def strip_entries(lines: list[str], zaids: set[str], marks: tuple[str, ...]
                  ) -> tuple[list[str], int]:
    """Drop every entry for one of `zaids`, continuation lines included.

    Matching on the ZAID as well as on the reference is what makes the pass
    repair a *truncated* leftover: a half-written `26056.65y ... xsec/NRT-DPA-`
    no longer carries the marker, but it is still an entry for a table the block
    is about to declare. An entry may also be wrapped across physical lines with
    a trailing `+`; dropping only the first would leave an orphan continuation
    for the next entry to swallow.
    """
    kept: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        first = stripped.split(None, 1)[0] if stripped else ""
        if first in zaids or any(mk in lines[i] for mk in marks):
            continues = stripped.endswith("+")
            removed += 1
            i += 1
            while i < len(lines) and continues:
                continues = lines[i].rstrip().endswith("+")
                removed += 1
                i += 1
            continue
        kept.append(lines[i])
        i += 1
    return kept, removed


def replace_in_place(lines: list[str], rx: re.Pattern, ref_mark: str,
                     known: list[str], block: list[str]) -> tuple[list[str], int]:
    """Swap a block's entries for `block`, keeping the position of the first.

    Removing them and appending at the end would work for MCNP but would make
    every diff against the shipped xsdir unreadable. Raises if the file declares
    a table the block does not, because the alternative is dropping it silently.
    """
    idx = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        first = stripped.split(None, 1)[0] if stripped else ""
        if rx.match(first) and ref_mark in lines[i]:
            idx.append(i)
            continues = stripped.endswith("+")
            i += 1
            while i < len(lines) and continues:
                continues = lines[i].rstrip().endswith("+")
                idx.append(i)
                i += 1
            continue
        i += 1
    if not idx:
        return lines, 0

    found = [lines[j].strip().split(None, 1)[0] for j in idx
             if rx.match((lines[j].strip().split(None, 1) or [""])[0])]
    unknown = sorted(set(found) - set(known))
    if unknown:
        raise SystemExit(
            f"el fichero declara tablas que el bloque no trae: {unknown}. "
            f"Parado sin tocar nada -- sustituir el bloque las borraria.")

    at = idx[0]
    drop = set(idx)
    head = [ln for k, ln in enumerate(lines[:at]) if k not in drop]
    tail = [ln for k, ln in enumerate(lines[at:], start=at) if k not in drop]
    return head + block + tail, len(found)


def is_full_xsdir(lines: list[str]) -> bool:
    """True when the file carries the `directory` keyword.

    That keyword is what separates a real xsdir from the one-entry snippets the
    sampler drops next to each ACE file. Editing a snippet would put a library
    into something MCNP never reads as a directory.
    """
    return any(ln.strip().lower() == "directory" for ln in lines)


def datapath_of(lines: list[str]) -> str | None:
    for ln in lines[:20]:
        s = ln.strip()
        if s.upper().startswith("DATAPATH"):
            return s.split("=", 1)[1].strip() if "=" in s else s.split(None, 1)[-1].strip()
    return None


def refs_resolve(xsdir: Path, lines: list[str]) -> tuple[int, int, str | None]:
    """How many of the two blocks' entries point at a file that is there.

    The defect this pass exists to fix was a path that resolved nowhere, and
    MCNP only reports that at run time, one table at a time, after the job has
    been queued. Checking it here costs 92 stats.
    """
    here = xsdir.resolve().parent
    ok = bad = 0
    first_bad = None
    for ln in lines:
        stripped = ln.strip()
        first = stripped.split(None, 1)[0] if stripped else ""
        hit = ((IRDFF_RE.match(first) and IRDFF_REF in ln)
               or (DPA_RE.match(first) and DPA_REF in ln))
        if not hit:
            continue
        ref = stripped.split()[2]
        if (here / ref).exists():
            ok += 1
        else:
            bad += 1
            if first_bad is None:
                first_bad = ref
    return ok, bad, first_bad


def desired(path: Path, lines: list[str], do_irdff: bool,
            forced_dpa: str | None, forced_irdff: str | None):
    """The file as it should be, plus what was measured to get there.

    Built by rewriting rather than by patching, which is what makes the pass
    idempotent: a file already correct produces itself, and a half-written one
    is repaired instead of appended to twice.
    """
    info: dict[str, str] = {}

    dpa_pre, how = prefix_for(path.parent, DPA_DIR,
                              f"{DPA_ENTRIES[0][2]}/{DPA_ENTRIES[0][0]}", forced_dpa)
    if dpa_pre is None:
        dpa_pre = current_prefix(lines, DPA_RE, DPA_REF, DPA_DIR) or DPA_DIR
        how = "no encontrado, se deja como esta"
    info["dpa"] = f"{dpa_pre} ({how})"

    kept, _ = strip_entries(lines, set(DPA_ZAIDS), (DPA_REF,))

    n_irdff = 0
    if do_irdff:
        irdff_pre, how = prefix_for(path.parent, IRDFF_DIR, IRDFF_ENTRIES[0][2],
                                    forced_irdff)
        if irdff_pre is None:
            irdff_pre = current_prefix(kept, IRDFF_RE, IRDFF_REF, IRDFF_DIR)
            how = "no encontrado, se deja como esta"
        if irdff_pre:
            info["irdff"] = f"{irdff_pre} ({how})"
            kept, n_irdff = replace_in_place(kept, IRDFF_RE, IRDFF_REF,
                                             IRDFF_ZAIDS, irdff_block(irdff_pre))

    # The shipped xsdir ends on an empty line. Left alone it becomes a blank
    # line in the middle of the directory once the block is appended.
    while kept and not kept[-1].strip():
        kept.pop()
    block = dpa_block(dpa_pre)
    return kept + block, len(block), n_irdff, info


def fix_one(path: Path, apply: bool, do_irdff: bool = True,
            forced_dpa: str | None = None, forced_irdff: str | None = None):
    """Return (status, n_body_lines, datapath, n_irdff, info).

    status: 'ok' | 'fixed' | 'would-fix' | 'skip-snippet' | 'skip-empty'.
    """
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    lines = text.splitlines()
    if not lines:
        return "skip-empty", 0, None, 0, {}
    if not is_full_xsdir(lines):
        return "skip-snippet", 0, None, 0, {}

    dp = datapath_of(lines)
    want, n_dpa, n_irdff, info = desired(path, lines, do_irdff,
                                         forced_dpa, forced_irdff)
    out = "\n".join(want) + "\n"
    n_body = len(want) - n_dpa

    if out == text:
        return "ok", n_body, dp, n_irdff, info
    if not apply:
        return "would-fix", n_body, dp, n_irdff, info

    tmp = path.with_name(path.name + ".dpatmp")
    tmp.write_text(out, encoding="utf-8", errors="surrogateescape")
    os.replace(tmp, path)

    # Verify what landed, not what was intended: a short write here is a
    # directory MCNP will read and half-understand.
    back = path.read_text(encoding="utf-8", errors="surrogateescape")
    if back != out:
        raise SystemExit(f"VERIFY FAILED on {path} -- parado, nada mas tocado")
    bl = back.splitlines()
    declared = {ln.strip().split(None, 1)[0] for ln in bl if ln.strip()}
    missing = [z for z in DPA_ZAIDS if z not in declared]
    if missing:
        raise SystemExit(f"VERIFY FAILED on {path}: faltan tablas DPA {missing}")
    # The 80-column invariant is only ours to assert over what this pass wrote.
    # With --no-irdff the overlong IRDFF entries stay by request, and failing on
    # them would make the flag unusable.
    scope = bl if do_irdff else bl[-n_dpa:]
    offset = 0 if do_irdff else len(bl) - n_dpa
    long = [offset + k + 1 for k, ln in enumerate(scope) if len(ln) > MAX_COL]
    if long:
        raise SystemExit(
            f"VERIFY FAILED on {path}: {len(long)} linea(s) pasan de {MAX_COL} "
            f"columnas (la primera, la {long[0]})")
    if not bl[-1].strip():
        raise SystemExit(f"VERIFY FAILED (linea en blanco final) on {path}")
    return "fixed", n_body, dp, n_irdff, info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="directorios a recorrer (o ficheros sueltos)")
    p.add_argument("--apply", action="store_true",
                   help="escribe el cambio; sin esto no se modifica nada")
    p.add_argument("--no-irdff", action="store_true",
                   help="no tocar el bloque IRDFF-II .34y")
    p.add_argument("--dpa-rel", metavar="RUTA",
                   help="forzar el prefijo relativo de xsec/ en vez de medirlo")
    p.add_argument("--irdff-rel", metavar="RUTA",
                   help="forzar el prefijo relativo de IRDFF-II/ en vez de medirlo")
    p.add_argument("--quiet", action="store_true", help="solo el resumen")
    args = p.parse_args()
    do_irdff = not args.no_irdff

    targets: list[Path] = []
    for r in args.roots:
        rp = Path(r)
        if rp.is_file():
            targets.append(rp)
        elif rp.is_dir():
            for dirpath, _dirs, files in os.walk(rp):
                for f in sorted(files):
                    if f.endswith(".dpatmp"):
                        continue
                    targets.append(Path(dirpath) / f)
        else:
            print(f"⛔ no existe: {r}")
            return 2

    counts = {"ok": 0, "fixed": 0, "would-fix": 0, "skip-snippet": 0, "skip-empty": 0}
    odd_datapath: dict[str, int] = {}
    body_counts: set[int] = set()
    irdff_counts: set[int] = set()
    prefixes: dict[str, dict[str, int]] = {"dpa": {}, "irdff": {}}
    unresolved: list[tuple[Path, str]] = []
    n_resolved = 0

    for t in targets:
        try:
            status, n_body, dp, n_irdff, info = fix_one(
                t, args.apply, do_irdff, args.dpa_rel, args.irdff_rel)
        except UnicodeDecodeError:
            continue                      # binario (ACE, PENDF): no es un xsdir
        except IsADirectoryError:
            continue
        counts[status] += 1
        if status not in ("ok", "fixed", "would-fix"):
            continue
        body_counts.add(n_body)
        irdff_counts.add(n_irdff)
        if dp != EXPECTED_DATAPATH:
            odd_datapath[str(dp)] = odd_datapath.get(str(dp), 0) + 1
        for key, val in info.items():
            prefixes[key][val] = prefixes[key].get(val, 0) + 1
        if status in ("ok", "fixed"):
            ok, bad, first = refs_resolve(
                t, t.read_text(encoding="utf-8", errors="surrogateescape").splitlines())
            n_resolved += ok
            if bad and len(unresolved) < 5:
                unresolved.append((t, first or "?"))
        if not args.quiet and (counts["fixed"] + counts["would-fix"]) % 100 == 1:
            print(f"   {status:9s} {t}")

    n_touched = counts["fixed"] if args.apply else counts["would-fix"]
    print("\n-- resumen")
    print(f"   xsdir completos ya correctos       : {counts['ok']}")
    print(f"   {'CORREGIDOS' if args.apply else 'a corregir'}                         : {n_touched}")
    print(f"   snippets de una entrada (intactos) : {counts['skip-snippet']}")
    if counts["skip-empty"]:
        print(f"   vacios (ignorados)                 : {counts['skip-empty']}")
    print(f"   bloque DPA-600 ({len(DPA_ENTRIES)} entradas)       : anadido al final")
    for pre, n in sorted(prefixes["dpa"].items()):
        print(f"     prefijo {pre:<40} en {n} fichero(s)")
    if not do_irdff:
        print("   bloque IRDFF-II .34y               : NO tocado (--no-irdff)")
    elif irdff_counts == {0}:
        print("   bloque IRDFF-II .34y               : ninguno encontrado")
    else:
        print(f"   bloque IRDFF-II .34y               : "
              f"{sorted(irdff_counts)} entradas, <= {WRITE_COL} columnas")
        for pre, n in sorted(prefixes["irdff"].items()):
            print(f"     prefijo {pre:<40} en {n} fichero(s)")
    if n_resolved:
        print(f"   entradas que resuelven en disco    : {n_resolved}")
    if unresolved:
        print(f"   ⛔ {len(unresolved)}+ fichero(s) con entradas que NO resuelven:")
        for t, ref in unresolved:
            print(f"      {t}  ->  {ref}")
    if len(body_counts) > 1:
        print(f"   ⚠ los xsdir no tienen todos el mismo numero de lineas: "
              f"{sorted(body_counts)}")
    elif body_counts:
        n, = body_counts
        print(f"   lineas por xsdir                   : {n} + bloque DPA")
    for dp, n in odd_datapath.items():
        print(f"   ⚠ DATAPATH inesperado en {n} fichero(s): {dp!r}")
    if not args.apply:
        print("\n   Ensayo. Nada escrito. Repite con --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
