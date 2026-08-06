"""The phase 3d performance gate, recorded in P8 and enforced in P9.

**Why this exists.** Phase 3d rewrites the flat classes' method bodies to route
through the GNDS model: ``CrossSection.from_endf`` will build a ``ReactionSuite``
and project it back. ``kika/processing/reconstruct.py:274`` constructs one
``CrossSection`` per MT per call, and the cluster pipeline calls that per sample
per temperature — so a round trip that costs a few milliseconds each time is a
real cost, paid thousands of times, on a machine nobody is watching. The plan's
escape hatch is a lazy ``model`` property; this file is what decides whether it
is needed, instead of guessing after the fact.

**The gate: no measured path may regress by more than 20%.**

**This does not run in the ordinary suite, and the reason is a mistake made
here first.** The first version of this file ran its two cheap measurements in
the fast lane. In isolation they passed; inside the full suite all three failed,
because 1300 other tests were competing for the same cores. That is not a
regression, it is contention — and a timing gate that goes red on machine load
is worse than no gate, because it teaches you to ignore it. So every test here is
``perf``-marked **and** skipped unless ``RUN_PERF_GATE=1``. It is run
deliberately, on a quiet machine, as part of accepting P9.

**What is honest about wall-clock, and what is not.**

* Timings are **min of N**, not mean. The minimum is the least contaminated
  estimator here: interference from other processes can only ever make a run
  slower, so the smallest observation is the closest to the true cost. It does
  not rescue a machine that is busy for the whole measurement, which is why the
  opt-in above is doing the real work.
* **A path whose own run-to-run spread exceeds the tolerance is not gated, and
  says so.** Recording the baseline repeats each measurement five times and
  stores ``max / min`` over those trials. That ratio is the floor on what the
  measurement can resolve: a path that varies 1.25x between identical runs cannot
  detect a 1.20x regression, so the gate **skips** it and names the number rather
  than failing at random. Widening the tolerance to cover the noise instead would
  leave a test that is green whatever happens, which is the worst option of the
  three.

  This is not hypothetical. ``read_pendf_mf3_sections`` at three repeats spread
  **1.28x** — it is dominated by file I/O and page-cache state, not by anything
  phase 3d touches — which is what failed the first version of this gate. Ten
  repeats brings it to about 1.06x and back inside the budget. The recorded
  spreads are in the baseline file; check them before trusting a green run.
* The baseline records the **machine it was measured on** — platform, processor,
  Python version. Comparing a wall-clock number across machines is not evidence,
  so a mismatch **skips** rather than passing or failing. Saying "no regression"
  on a different CPU would be worse than saying nothing.
* ``reconstruct`` takes ~8 s per call and is measured twice, so this file costs
  around 40 s when it runs.

Run the gate with ``RUN_PERF_GATE=1 pytest kika/tests/test_performance_baseline.py``.
Regenerate the baseline with ``REGEN_PERF_BASELINE=1 pytest kika/tests/test_performance_baseline.py``,
and say in the commit message *why* — a baseline regenerated to make a red test
green is a deleted test with extra steps.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict

import pytest

BASELINE = Path(__file__).parent / "data" / "p9_performance_baseline.json"

#: Fail the increment above this. 20% is the plan's number.
TOLERANCE = 0.20

#: Below this, wall-clock noise dominates and a ratio means nothing.
NOISE_FLOOR_SECONDS = 5.0e-4


pytestmark = pytest.mark.perf


def _requireOptIn() -> None:
    """Wall-clock assertions are opt-in. See the module docstring for why."""
    if os.environ.get("RUN_PERF_GATE") != "1":
        pytest.skip(
            "set RUN_PERF_GATE=1 to run the phase 3d timing gate, on a quiet "
            "machine. Inside a loaded test suite these numbers measure "
            "contention, not the code."
        )


def _machine() -> Dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }


#: Trials used when recording, to measure each path's own noise.
SPREAD_TRIALS = 5


def _best(call: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def _bestAndSpread(call: Callable[[], object], repeats: int) -> tuple:
    """``(best, spread)`` — the fastest min-of-N, and how much the trials varied.

    ``spread`` is ``max / min`` over :data:`SPREAD_TRIALS` independent min-of-N
    measurements. It is the floor on what this path can detect: a 1.16x spread
    cannot resolve a 1.20x regression.
    """
    trials = [_best(call, repeats) for _ in range(SPREAD_TRIALS)]
    return min(trials), max(trials) / min(trials)


# ---------------------------------------------------------------------------
# The measured paths
# ---------------------------------------------------------------------------

def _measureFlatConstruction(tapePath: str) -> float:
    """``CrossSection.from_endf`` over every MF3 section — what 3d rewrites.


    The narrowest measurement of the change: no file reading, no physics, just
    the constructor whose body is about to grow a model round trip.
    """
    from kika.endf.read_endf import read_endf
    from kika.nuclear_data import CrossSection

    endf = read_endf(tapePath)
    sections = [endf.mf[3].mt[mt] for mt in sorted(endf.mf[3].mt)]
    return _bestAndSpread(lambda: [CrossSection.from_endf(s) for s in sections], repeats=20)


def _measurePendfRead(tapePath: str) -> float:
    """``read_pendf_mf3_sections`` — the boundary the desktop app calls."""
    from kika.processing.njoy_pendf_cache import read_pendf_mf3_sections

    return _bestAndSpread(lambda: read_pendf_mf3_sections(tapePath), repeats=10)


def _measureReconstruct(tapePath: str) -> float:
    """The hot path: one ``CrossSection`` per MT, per sample, per temperature."""
    from kika.endf.processing.reconstruct import reconstruct
    from kika.endf.read_endf import read_endf

    endf = read_endf(tapePath)
    mf2, mf3 = endf.mf[2].mt[151], endf.files.get(3)
    return _bestAndSpread(lambda: reconstruct(mf2, mf3), repeats=2)


#: name -> (callable, is_slow). Adding one here and regenerating extends the gate.
MEASUREMENTS = {
    "cross_section_from_endf_all_mts": (_measureFlatConstruction, False),
    "read_pendf_mf3_sections": (_measurePendfRead, False),
    "endf_reconstruct_adapter": (_measureReconstruct, True),
}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_record_the_baseline(micro_tape):
    """Writes the committed baseline. Skips unless ``REGEN_PERF_BASELINE=1``."""
    if os.environ.get("REGEN_PERF_BASELINE") != "1":
        pytest.skip("set REGEN_PERF_BASELINE=1 to re-record the phase 3d baseline")

    payload = {
        "machine": _machine(),
        "tolerance": TOLERANCE,
        "seconds": {},
        "spread": {},
    }
    for name, (measure, _) in MEASUREMENTS.items():
        best, spread = measure(str(micro_tape))
        payload["seconds"][name] = best
        payload["spread"][name] = spread
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Enforcing
# ---------------------------------------------------------------------------

def _baseline() -> dict:
    if not BASELINE.is_file():
        pytest.skip(f"no baseline recorded at {BASELINE}")
    return json.loads(BASELINE.read_text())


def test_the_baseline_records_the_machine_it_was_measured_on():
    """Without this, the numbers below are unfalsifiable."""
    data = _baseline()
    assert set(data["machine"]) == {"platform", "processor", "python"}
    assert set(data["seconds"]) == set(MEASUREMENTS)
    assert set(data["spread"]) == set(MEASUREMENTS), (
        "every measured path must record its own noise, or the gate cannot know "
        "whether it is able to detect the regression it claims to detect"
    )


def _checkOne(name: str, micro_tape) -> None:
    _requireOptIn()
    data = _baseline()
    if data["machine"] != _machine():
        pytest.skip(
            "the baseline was recorded on a different machine "
            f"({data['machine']['platform']}); a wall-clock comparison across "
            "machines is not evidence. Re-record with REGEN_PERF_BASELINE=1."
        )

    recorded = data["seconds"][name]
    tolerance = data.get("tolerance", TOLERANCE)
    if recorded < NOISE_FLOOR_SECONDS:
        pytest.skip(f"{name} baseline is {recorded:.2e} s — below the noise floor")

    spread = data["spread"][name]
    if spread - 1.0 >= tolerance:
        pytest.skip(
            f"{name} varies by {spread:.2f}x between identical runs on this "
            f"machine, so it cannot detect a {1 + tolerance:.2f}x regression. "
            f"Not gated — see the module docstring."
        )

    measure, _ = MEASUREMENTS[name]
    measured, _ = measure(str(micro_tape))
    ratio = measured / recorded
    assert ratio <= 1.0 + tolerance, (
        f"{name} regressed {ratio:.2f}x: {recorded:.4f} s -> {measured:.4f} s.\n"
        f"The phase 3d fallback is a lazy `model` property on the flat class — "
        f"keep the fast constructor and build the model on first access."
    )


@pytest.mark.parametrize(
    "name", [n for n, (_, slow) in MEASUREMENTS.items() if not slow]
)
def test_no_cheap_path_regressed(name, micro_tape):
    _checkOne(name, micro_tape)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", [n for n, (_, slow) in MEASUREMENTS.items() if slow]
)
def test_no_slow_path_regressed(name, micro_tape):  # noqa: D401
    """``reconstruct`` is ~11 s a call, so it is out of the fast lane. It is also
    the path that matters most: one ``CrossSection`` per MT, per sample, per
    temperature, on the cluster."""
    _checkOne(name, micro_tape)
