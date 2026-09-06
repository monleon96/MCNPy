"""A correlation of 1.0000002 is the file's rounding, not a reason to refuse.

Found on U-235 JEFF-4.0 MF33 the day ``perturbFromModel`` first ran on a full
tape: six correlations above 1 by 2e-7 and the pre-flight said the block
"cannot be sampled". ENDF writes six significant figures, so a stated 1.0
comes back as 1 + O(1e-6); the 2x2 minor is negative by round-off and the
definiteness check already handles that with ``clip``. Above
``CORRELATION_ROUNDOFF`` the refusal stands.
"""
from __future__ import annotations

import numpy as np

from kika.cov.conditioning import (BLOCKS, CORRELATION_ROUNDOFF, NOTE,
                                   inspect_blocks, inspect_matrix)


def _withCorrelation(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]])


def test_roundoff_above_one_is_a_note_and_definiteness_takes_over():
    report = inspect_matrix(_withCorrelation(1.0 + 2e-7))
    (bound,) = [f for f in report.findings if f.check == "correlation_bound"]
    assert bound.severity == NOTE and bound.evidence["roundoff"] is True
    assert report.samplable
    assert [f for f in report.findings if f.check == "definiteness"], \
        "the negative minor is reported where it belongs"
    plan = inspect_blocks({"c": _withCorrelation(1.0 + 2e-7)}).recommended_plan()
    assert plan.steps[0].remedy == "clip"


def test_a_real_excess_still_blocks():
    report = inspect_matrix(_withCorrelation(1.0 + 10 * CORRELATION_ROUNDOFF))
    (bound,) = [f for f in report.findings if f.check == "correlation_bound"]
    assert bound.severity == BLOCKS and bound.evidence["roundoff"] is False
    assert not report.samplable


def test_the_threshold_sits_far_above_what_a_file_produces():
    assert 2.1e-7 < CORRELATION_ROUNDOFF < 1e-3
