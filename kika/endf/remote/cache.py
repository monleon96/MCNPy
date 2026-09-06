"""Local file cache for downloaded ENDF files."""

import json
from datetime import datetime
from pathlib import Path

from .constants import get_cache_dir, normalize_library_name
from .exceptions import CacheError


class ENDFCache:
    """
    Manages local caching of downloaded ENDF files.

    Cache structure:
        {cache_dir}/
            {library}/
                {particle}/
                    {zaid}.endf
            cache_metadata.json
    """

    METADATA_FILE = "cache_metadata.json"

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the cache.

        Parameters
        ----------
        cache_dir : Path, optional
            Custom cache directory. If None, uses default or env var.
        """
        self.cache_dir = cache_dir or get_cache_dir()

    def _get_cache_path(self, zaid: int | str, library: str, particle: str = "n") -> Path:
        """Get the cache file path for a given isotope.

        ``zaid`` is the MCNP-style ZAID for a nuclide file; files that name no
        nuclide (thermal scattering) use the server's filename stem instead.
        """
        canonical_lib = normalize_library_name(library)
        return self.cache_dir / canonical_lib / particle / f"{zaid}.endf"

    def _get_metadata_path(self) -> Path:
        """Get the path to the metadata file."""
        return self.cache_dir / self.METADATA_FILE

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_metadata(self, metadata: dict) -> None:
        """Save cache metadata to disk."""
        metadata_path = self._get_metadata_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    def get(self, zaid: int | str, library: str, particle: str = "n") -> Path | None:
        """
        Get a cached file if it exists.

        Parameters
        ----------
        zaid : int
            ZAID number (e.g., 26056 for Fe-56)
        library : str
            Library name
        particle : str
            Particle type (default: "n" for neutron)

        Returns
        -------
        Path or None
            Path to cached file if it exists, None otherwise
        """
        cache_path = self._get_cache_path(zaid, library, particle)
        if cache_path.exists():
            return cache_path
        return None

    def put(
        self, zaid: int | str, library: str, content: bytes, particle: str = "n"
    ) -> Path:
        """
        Store content in the cache.

        Parameters
        ----------
        zaid : int
            ZAID number
        library : str
            Library name
        content : bytes
            File content to cache
        particle : str
            Particle type (default: "n")

        Returns
        -------
        Path
            Path to the cached file
        """
        cache_path = self._get_cache_path(zaid, library, particle)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cache_path.write_bytes(content)
        except OSError as e:
            raise CacheError(f"Failed to write cache file: {e}") from e

        # Update metadata
        metadata = self._load_metadata()
        canonical_lib = normalize_library_name(library)
        key = f"{canonical_lib}/{particle}/{zaid}"
        metadata[key] = {
            "zaid": zaid,
            "library": canonical_lib,
            "particle": particle,
            "size_bytes": len(content),
            "cached_at": datetime.now().isoformat(),
            "path": str(cache_path),
        }
        self._save_metadata(metadata)

        return cache_path

    def remove(
        self, zaid: int | str | None = None, library: str | None = None
    ) -> int:
        """
        Remove cached files.

        Parameters
        ----------
        zaid : int, optional
            Specific ZAID to remove. If None, removes all for the library.
        library : str, optional
            Specific library to remove. If None, removes all.

        Returns
        -------
        int
            Number of files removed
        """
        removed = 0
        metadata = self._load_metadata()
        keys_to_remove = []

        for key, info in metadata.items():
            should_remove = True

            if library is not None:
                canonical_lib = normalize_library_name(library)
                if info["library"] != canonical_lib:
                    should_remove = False

            if zaid is not None:
                if str(info["zaid"]) != str(zaid):
                    should_remove = False

            if should_remove:
                cache_path = Path(info["path"])
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        removed += 1
                    except OSError:
                        pass
                keys_to_remove.append(key)

        # Update metadata
        for key in keys_to_remove:
            del metadata[key]
        self._save_metadata(metadata)

        # Clean up empty directories
        self._cleanup_empty_dirs()

        return removed

    def _cleanup_empty_dirs(self) -> None:
        """Remove empty directories in the cache."""
        if not self.cache_dir.exists():
            return

        # Walk bottom-up to remove empty directories
        for dirpath in sorted(self.cache_dir.rglob("*"), reverse=True):
            if dirpath.is_dir():
                try:
                    dirpath.rmdir()  # Only succeeds if empty
                except OSError:
                    pass  # Directory not empty

    def get_info(self) -> dict:
        """
        Get cache information.

        Returns
        -------
        dict
            Cache statistics and file list
        """
        metadata = self._load_metadata()

        total_size = 0
        libraries = {}

        for key, info in metadata.items():
            lib = info["library"]
            size = info["size_bytes"]
            total_size += size

            if lib not in libraries:
                libraries[lib] = {"count": 0, "size_bytes": 0}
            libraries[lib]["count"] += 1
            libraries[lib]["size_bytes"] += size

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(metadata),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "libraries": libraries,
            "files": metadata,
        }


# Module-level cache instance
_cache = None


def get_cache() -> ENDFCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = ENDFCache()
    return _cache


def get_cache_info() -> dict:
    """Get cache information."""
    return get_cache().get_info()


def clear_cache(
    isotope: int | None = None, library: str | None = None
) -> int:
    """
    Clear cached files.

    Parameters
    ----------
    isotope : int, optional
        Specific ZAID to clear
    library : str, optional
        Specific library to clear

    Returns
    -------
    int
        Number of files removed
    """
    return get_cache().remove(zaid=isotope, library=library)
