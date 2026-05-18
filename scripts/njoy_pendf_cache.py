"""Backward-compatibility shim — the implementation lives in kika.processing."""
from kika.processing.njoy_pendf_cache import (  # noqa: F401
    DEFAULT_PENDF_CACHE_DIR,
    mf33_needs_pendf,
    get_or_create_pendf,
    read_pendf_mf3_sections,
)
