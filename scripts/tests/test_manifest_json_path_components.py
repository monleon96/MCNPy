"""The JSON loader path must not lose the per-point uncertainties.

``kika/exfor/io.py`` sets ``_raw_uncertainty_components = []`` for a dataset
read from JSON, meaning "no raw EXFOR columns — synthesize a ``DATA-ERR``
component from the per-point ``uncertainty_stat``".  The synthesis in
``apply_manifest_to_exfor`` used to be guarded by ``is None``, and ``[]`` is not
``None``: the synthesis was skipped, every ``column:`` lookup in the manifest
resolved to zeros, and for a ``derive_stat_only`` entry that cascaded into
σ_total = 0 → σ_sys capped to 0 → σ_stat pinned at the 1 % floor.

Found 2026-08-27 on Gkatis 27673002 (the only supplementary JSON in the
production corpus): its real 3.4-26 % per-point uncertainties were replaced by a
flat 1 % and its 1.566 % systematic by zero, in both the fit and the chi2.

The contract these tests pin: **``[]`` and ``None`` mean the same thing.**
"""
import os
from pathlib import Path

import numpy as np
import pytest

MICRO_MANIFEST = str(Path(__file__).parent / "data" / "uncertainty_manifest_micro.yaml")

# 10886002 in the micro manifest: column DATA-ERR, semantics total_per_point,
# derive_stat_only, and a flat 5 % correlated sys.
DATASET_ID = "10886002"
SYS_PCT = 5.0
# Declared totals, chosen to sit above the sys so no cap binds and the
# decomposition is exercised rather than the floor.
TOTAL_PCT = np.array([8.0, 12.0, 20.0, 9.0])
VALUES = np.array([0.30, 0.12, 0.05, 0.22])


def _build_exfor():
    from kika.exfor.angular_distribution import ExforAngularDistribution

    data = [
        {"angle": a, "cross_section": float(v),
         "uncertainty_stat": float(v * t / 100.0)}
        for a, v, t in zip((30.0, 60.0, 90.0, 120.0), VALUES, TOTAL_PCT)
    ]
    return ExforAngularDistribution(
        entry="10886", subentry="002", quantity="DA",
        citation={"authors": ["Smith"], "year": 1980},
        reaction={"notation": "26-FE-56(N,EL)26-FE-56"},
        facility={}, method={}, angle_frame="CM",
        units={"energy": "MeV", "angle": "deg", "cross_section": "b/sr"},
        _data_blocks=[{"value": 2.0, "data": data}],
    )


def _resolve(components):
    """Apply the manifest with the given ``uncertainty_components`` and read back."""
    os.environ["KIKA_UNCERTAINTY_MANIFEST_PATH"] = MICRO_MANIFEST
    from scripts import uncertainty_manifest as um

    um.load_manifest(MICRO_MANIFEST, force_reload=True)
    ad = _build_exfor()
    um.apply_manifest_to_exfor(ad, uncertainty_components=components)
    pts = [p for blk in ad._data_blocks for p in blk["data"]]
    v = np.array([p["cross_section"] for p in pts])
    return {
        "stat_pct": np.array([p["uncertainty_stat"] for p in pts]) / v * 100.0,
        "sys_pct": np.array([p.get("uncertainty_sys", 0.0) for p in pts]) / v * 100.0,
        "flag": ad.uncertainty_manifest_flag,
    }


@pytest.mark.parametrize("components", [None, []], ids=["None", "empty-list"])
def test_json_path_keeps_the_declared_uncertainties(components):
    """Neither sentinel may collapse σ_stat onto the floor or zero σ_sys."""
    r = _resolve(components)

    assert r["flag"] == "curated"

    # σ_sys is the manifest's flat 5 %, not zero.
    np.testing.assert_allclose(r["sys_pct"], SYS_PCT, rtol=1e-9)

    # σ_stat is the declared total decomposed against σ_sys, not the 1 % floor.
    expected_stat = np.sqrt(TOTAL_PCT ** 2 - SYS_PCT ** 2)
    np.testing.assert_allclose(r["stat_pct"], expected_stat, rtol=1e-9)
    assert not np.any(np.isclose(r["stat_pct"], 1.0, atol=1e-9)), \
        "σ_stat pinned at SIGMA_STAT_MIN_REL — the column lookup resolved to zeros"

    # And the total the lab reported is preserved.
    np.testing.assert_allclose(
        np.hypot(r["stat_pct"], r["sys_pct"]), TOTAL_PCT, rtol=1e-9)


def test_empty_list_and_none_are_equivalent():
    """The regression itself: the two sentinels must not diverge."""
    a, b = _resolve(None), _resolve([])
    np.testing.assert_allclose(a["stat_pct"], b["stat_pct"], rtol=1e-12)
    np.testing.assert_allclose(a["sys_pct"], b["sys_pct"], rtol=1e-12)


# ── The other half of the contract ───────────────────────────────────────────
# A dataset that declares NO per-point uncertainty at all must NOT get a
# synthetic column of zeros. Building one makes ``column: best_available``
# "succeed" with σ = 0, which triggers the same cascade the fix above removes
# and replaces the defaults block's 5 % σ_sys with 1 % σ_stat ⊕ 0 — strictly
# worse than the bug being fixed. Four production datasets are in this state
# (13511004 Perey 1991, 20482005, 10332004, 11638003); the first is the
# largest positive outlier in the corpus, so under-declaring it is expensive.

NO_UNC_DATASET = ("23059", "003")   # absent from the micro manifest -> defaults
DEFAULTS_SYS_PCT = 5.0


def _build_exfor_without_uncertainties():
    from kika.exfor.angular_distribution import ExforAngularDistribution

    entry, sub = NO_UNC_DATASET
    data = [{"angle": a, "cross_section": float(v), "uncertainty_stat": 0.0}
            for a, v in zip((30.0, 60.0, 90.0, 120.0), VALUES)]
    return ExforAngularDistribution(
        entry=entry, subentry=sub, quantity="DA",
        citation={"authors": ["NoErr"], "year": 1991},
        reaction={"notation": "26-FE-56(N,EL)26-FE-56"},
        facility={}, method={}, angle_frame="CM",
        units={"energy": "MeV", "angle": "deg", "cross_section": "b/sr"},
        _data_blocks=[{"value": 2.0, "data": data}],
    )


@pytest.mark.parametrize("components", [None, []], ids=["None", "empty-list"])
def test_dataset_with_no_declared_uncertainty_keeps_the_default_sys(components):
    os.environ["KIKA_UNCERTAINTY_MANIFEST_PATH"] = MICRO_MANIFEST
    from scripts import uncertainty_manifest as um

    um.load_manifest(MICRO_MANIFEST, force_reload=True)
    ad = _build_exfor_without_uncertainties()
    um.apply_manifest_to_exfor(ad, uncertainty_components=components)

    pts = [p for blk in ad._data_blocks for p in blk["data"]]
    v = np.array([p["cross_section"] for p in pts])
    sys_pct = np.array([p.get("uncertainty_sys", 0.0) for p in pts]) / v * 100.0

    np.testing.assert_allclose(sys_pct, DEFAULTS_SYS_PCT, rtol=1e-9), \
        "the defaults block's sigma_sys was capped away against a synthetic zero column"
