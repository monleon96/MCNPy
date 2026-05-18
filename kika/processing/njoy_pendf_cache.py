"""
Cache-backed NJOY PENDF generation for MF33 NC LTY=0 resolution.

Some libraries (e.g. JENDL Fe-56 MT=2) store MF33 covariance with NC LTY=0
sub-subsections (sum-rule across other MTs). Resolving them against
*relative* contributing covariances requires reconstructed pointwise σ(E),
i.e. a PENDF — the smooth-only MF3 in the original ENDF file is not enough.

This module provides:

  - ``mf33_needs_pendf``        — sentinel: True iff a section's MF33 needs σ(E).
  - ``get_or_create_pendf``     — returns the cached tape22 path, running NJOY
                                  only on cache miss. SHA256(ENDF) keyed.
  - ``read_pendf_mf3_sections`` — parse the cached tape22 into MT → MF3MT.

The default cache directory resolves via ``tempfile.gettempdir()`` so the
module works on any machine without machine-specific paths baked in.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

# LB types in MF33 NI sub-subsections that store *relative* covariances.
# Anything outside this set is absolute and needs σ(E) for diagonal use.
_RELATIVE_LB_TYPES = frozenset({5, 6, 8})


# Default cache directory — portable across Linux / macOS / Windows.
# Linux:   /tmp/kika_pendf_cache
# macOS:   /var/folders/.../T/kika_pendf_cache
# Windows: C:\Users\<user>\AppData\Local\Temp\kika_pendf_cache
DEFAULT_PENDF_CACHE_DIR = Path(tempfile.gettempdir()) / "kika_pendf_cache"


def mf33_needs_pendf(mf33_file, mt: int) -> bool:
    """True if MF33 for ``mt`` needs σ(E) (NC sum-rule or absolute NI LB)."""
    if mf33_file is None or mt not in mf33_file.sections:
        return False
    sec = mf33_file.sections[mt]
    for sub in sec.subsections:
        if getattr(sub, "nc", False) and (sub.nc_records or []):
            return True
        for ni in (sub.ni_records or []):
            lb = getattr(ni, "lb", None) or getattr(ni, "LB", None)
            if lb is None or lb not in _RELATIVE_LB_TYPES:
                return True
    return False


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA256 over the raw bytes — any byte edit invalidates the cache."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _pendf_cache_path(endf_path: Path, tolerance: float, cache_dir: Path) -> Path:
    """``cache_dir / {sha256}_{tolerance}.pendf`` — SHA256 over the ENDF bytes."""
    return cache_dir / f"{_sha256_of_file(endf_path)}_{tolerance}.pendf"


def get_or_create_pendf(
    endf_path: str | Path,
    *,
    tolerance: float,
    njoy_exe: str | Path,
    cache_dir: str | Path | None = None,
    timeout_s: float = 600.0,
    keep_njoy_io_dir: str | Path | None = None,
) -> Path:
    """Return a path to a cached PENDF (tape22) for ``endf_path``.

    On cache miss runs ``njoy_reconstruct_stream`` and atomically copies the
    PENDF bytes into the cache before the generator's TempDir is cleaned up.

    ``cache_dir=None`` uses ``DEFAULT_PENDF_CACHE_DIR`` (portable tempdir).

    If ``keep_njoy_io_dir`` is set, the RECONR input deck and stdout are
    copied there as ``<base>_recon.input`` / ``<base>_recon.output``. On
    cache hit, NJOY didn't run, so nothing is copied (look in
    ``cache_dir`` for the original I/O if needed). The same files are
    always saved next to the cached PENDF as ``<sha>_<tol>.input`` /
    ``.output`` so they are recoverable even after a hit.
    """
    endf_path = Path(endf_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_PENDF_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _pendf_cache_path(endf_path, tolerance, cache_dir)

    if target.is_file() and target.stat().st_size > 0:
        print(f"  PENDF cache hit: {target.name}")
        if keep_njoy_io_dir is not None:
            _copy_cached_recon_io(
                target, Path(keep_njoy_io_dir).expanduser(), endf_path,
            )
        return target

    print(f"  PENDF cache miss for {endf_path.name} — running NJOY reconr "
          f"(tolerance={tolerance})")

    # Local import — keeps module light when caller doesn't need NJOY.
    from kika.processing.njoy_reconstruct import njoy_reconstruct_stream

    src_pendf: Optional[Path] = None
    for kind, payload in njoy_reconstruct_stream(
        str(endf_path), str(njoy_exe),
        tolerance=tolerance, timeout_s=timeout_s,
    ):
        if kind == "pendf_path":
            # Copy synchronously while the generator's TempDir is still alive.
            src = Path(str(payload))
            tmp = target.with_suffix(".pendf.tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(target)
            src_pendf = target
            # Persist the deck + stdout next to the cached PENDF and, if
            # requested, also into the per-run keep_njoy_io_dir.
            workdir = src.parent
            _save_recon_io(workdir, target)
            if keep_njoy_io_dir is not None:
                _copy_cached_recon_io(
                    target, Path(keep_njoy_io_dir).expanduser(), endf_path,
                )
        elif kind == "warning":
            print(f"  NJOY: {payload}")

    if src_pendf is None or not target.is_file():
        raise RuntimeError(
            f"NJOY reconr did not emit a PENDF for {endf_path.name}"
        )
    print(f"  PENDF cached: {target.name} "
          f"({target.stat().st_size / 1024:.0f} KB)")
    return target


def _save_recon_io(workdir: Path, cached_pendf: Path) -> None:
    """Copy ``njoy.inp`` / ``njoy.out`` from the RECONR workdir next to the cache."""
    inp_src = workdir / "njoy.inp"
    out_src = workdir / "njoy.out"
    inp_dst = cached_pendf.with_suffix(".input")
    out_dst = cached_pendf.with_suffix(".output")
    try:
        if inp_src.is_file():
            shutil.copyfile(inp_src, inp_dst)
        if out_src.is_file():
            shutil.copyfile(out_src, out_dst)
    except OSError:
        pass


def _copy_cached_recon_io(
    cached_pendf: Path, dst_dir: Path, endf_path: Path,
) -> None:
    """Copy the cache's stored RECONR I/O into ``dst_dir`` for visibility."""
    inp_src = cached_pendf.with_suffix(".input")
    out_src = cached_pendf.with_suffix(".output")
    if not (inp_src.is_file() or out_src.is_file()):
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = endf_path.stem
    try:
        if inp_src.is_file():
            shutil.copyfile(inp_src, dst_dir / f"{base}_recon.input")
        if out_src.is_file():
            shutil.copyfile(out_src, dst_dir / f"{base}_recon.output")
    except OSError:
        pass


def read_pendf_mf3_sections(pendf_path: str | Path) -> Dict[int, object]:
    """Parse a cached tape22 and return the MT → MF3MT map."""
    from kika.endf import read_endf
    pendf = read_endf(str(pendf_path), mf_numbers=[3])
    mf3 = pendf.get_file(3)
    if mf3 is None or not getattr(mf3, "sections", None):
        raise RuntimeError(f"PENDF {pendf_path} has no MF3 sections")
    return {int(mt): sec for mt, sec in mf3.sections.items()}
