"""MF35 → ``CovarianceSuite``, and the honesty of the report that comes with it.

Two things are being tested, and the second is the one that would rot quietly.

**The shape.** An MF35 band is the covariance of an energy distribution
*restricted to a range of incident energy*, so it is a ``rowData`` slice with
``domainMin``/``domainMax`` — the range counterpart of the ``domainValue``
slice that carries an MF34 Legendre order. Modelling a band as a separate
quantity would be structurally valid and would say something false about what
the data is, exactly as it would for MF34. So the slice is asserted, and so is
the fact that the bands keep their own orders instead of being flattened
together.

**The report.** Adding 5 to ``SUPPORTED_MF`` converts a silent skip into a
silent *claim of coverage* unless something says which MF5 laws were only
passed through. Two tests below pin that: MF5 must be declared undecoded, its
verbatim partials must be named one by one, and the covariance redirect must
not claim MF5 belongs to the covarianceSuite — which is what the old
``- {1,2,3,4}`` branch said, and which was false.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import kika
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF35MT
from kika.endf.model_adapter.covariances import (
    INCIDENT_ENERGY_DIMENSION,
    energyDistributionHref,
)
from kika.endf.model_adapter.decode import COVARIANCE_MF, SUPPORTED_MF
from kika.endf.read_endf import read_endf


@pytest.fixture(scope="module")
def pfnsEndf(micro_pfns_tape):
    return read_endf(str(micro_pfns_tape))


@pytest.fixture(scope="module")
def pfnsSuite(pfnsEndf):
    return decodeCovarianceSuite(pfnsEndf, evaluation="micro-cf252")


# ---------------------------------------------------------------------------
# The shape
# ---------------------------------------------------------------------------

def test_every_band_becomes_its_own_section(pfnsEndf, pfnsSuite):
    suite, _ = pfnsSuite
    bands = pfnsEndf.mf[35].mt[18].subsections
    assert len(suite.covarianceSections) == len(bands) == 4
    assert [s.label for s in suite] == [f"MF35-MT18-band{i}" for i in range(4)]


def test_the_band_is_a_slice_range_on_the_incident_energy_axis(pfnsEndf, pfnsSuite):
    """The mapping this module exists to get right."""
    suite, _ = pfnsSuite
    bands = pfnsEndf.mf[35].mt[18].subsections

    for section, band in zip(suite, bands):
        link = section.rowData
        assert link.href == energyDistributionHref(18)
        assert link.ENDF_MFMT == "35/18"

        assert len(link.slices) == 1
        entry = list(link.slices)[0]
        assert entry.dimension == INCIDENT_ENERGY_DIMENSION
        assert entry.domainValue is None, "a band is a range, not a point"
        assert entry.domainMin == pytest.approx(band.e1)
        assert entry.domainMax == pytest.approx(band.e2)
        assert entry.domainUnit == "eV"
        assert link.incidentEnergyBand == (band.e1, band.e2)


def test_the_matrix_is_the_one_the_record_layer_produced(pfnsEndf, pfnsSuite):
    """The adapter re-expresses; it must not recompute or re-project."""
    suite, _ = pfnsSuite
    bands = pfnsEndf.mf[35].mt[18].subsections

    for section, band in zip(suite, bands):
        np.testing.assert_array_equal(section.form.matrix, band.matrix())
        np.testing.assert_array_equal(section.form.rowGrid, band.energy_grid())
        assert section.form.rowGrid.size == section.form.matrix.shape[0] + 1


def test_the_matrix_stays_absolute(pfnsSuite):
    """LB=7 is already the covariance of dimensionless group probabilities.

    Unlike MF34, which needs its MF4 to convert relative to absolute, there is
    nothing to convert here and no MF5 is consulted. A section that came back
    marked relative would mean somebody had added a conversion that has no
    quantity to divide by.
    """
    suite, _ = pfnsSuite
    assert all(s.form.isRelative is False for s in suite)


def test_bands_are_not_flattened_into_one_matrix(pfnsSuite):
    """No cross-band blocks exist, so nothing may invent them.

    A container that assembled every block at one dimension would be malformed
    the moment the bands differ in order — 84, 641, 641, 641, 641 on
    ENDF/B-VIII.1 U-235 — and would be so *silently*. The suite has no such
    property, and this test is what keeps anyone from adding one.
    """
    suite, _ = pfnsSuite
    assert all(s.columnData is None for s in suite), "invented a cross-band block"
    assert not hasattr(suite, "covariance_matrix")


def test_the_section_header_survives_as_provenance(pfnsEndf, pfnsSuite):
    suite, _ = pfnsSuite
    mf35 = pfnsEndf.mf[35].mt[18]
    for section in suite:
        assert section.provenance.mat == mf35._mat
        assert section.provenance.za == pytest.approx(mf35._za)
        assert section.provenance.awr == pytest.approx(mf35._awr)


def test_a_tape_with_no_covariance_at_all_is_still_reported(micro_tape):
    """The 'no MF33/34/35' loss must not disappear now that there are three."""
    endf = read_endf(str(micro_tape), mf_numbers=[3])
    _, report = decodeCovarianceSuite(endf)
    assert any("MF35" in message for message in report.losses)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_mf35_is_registered_as_a_covariance_file():
    assert 5 in SUPPORTED_MF and 35 in SUPPORTED_MF
    assert 35 in COVARIANCE_MF
    assert 5 not in COVARIANCE_MF, (
        "MF5 is reactionSuite content, not a covariance file; putting it here "
        "is what made the redirect loop tell users something false"
    )


def unsupported(suite) -> list[str]:
    return list(suite.report.unsupported)


def test_report_declares_mf5_partials_it_did_not_decode(micro_pfns_tape):
    """One line per verbatim subsection, naming the law and the count.

    Six LF=5 partials on Cf-252 MT455. Without these lines the report would say
    MF5 is supported and stop, which reads as "decoded" to every consumer.
    """
    suite = kika.read(str(micro_pfns_tape))
    messages = unsupported(suite)

    assert any("MF5 (energy distributions) is present and parsed" in m
               for m in messages)
    verbatim = [m for m in messages if "stored verbatim" in m]
    assert len(verbatim) == 6
    for index in range(6):
        assert any(f"MF5/MT455 partial {index} is stored verbatim: LF=5" in m
                   for m in verbatim)


def test_the_redirect_no_longer_claims_mf5_is_a_covariance(micro_pfns_tape):
    """The bug this phase exists to fix, pinned.

    ``decode.py`` used to redirect every supported MF outside {1,2,3,4} to
    ``decodeCovarianceSuite``. MF5 is the first supported MF that is neither
    one of those four nor a covariance, so the sentence became false the moment
    MF5 was registered.
    """
    suite = kika.read(str(micro_pfns_tape))
    for message in unsupported(suite):
        if message.startswith("MF5 ") or "MF5/" in message:
            assert "covarianceSuite" not in message, message


def test_report_is_clean_apart_from_the_declared_gaps(micro_pfns_tape):
    """Nothing else is lost, approximated or warned about on this tape.

    The fixture carries MF1/451, MF3/MT18, MF5/MT18+455 and MF35/MT18 and
    nothing else, so every one of those must land somewhere. If a future change
    starts dropping something, this is what says so rather than the count of
    unsupported nodes quietly going up.
    """
    suite = kika.read(str(micro_pfns_tape))
    report = suite.report

    assert report.losses == [], report.losses
    assert report.approximations == [], report.approximations
    assert report.warnings == [], report.warnings

    messages = unsupported(suite)
    assert len(messages) == 7, messages
    assert all(m.startswith("MF5") for m in messages), messages

    # And the MF35 redirect really was acted on rather than left standing.
    assert not any("covarianceSuite" in m for m in messages)
    assert len(suite.covarianceSuite.covarianceSections) == 4


def test_the_report_fields_are_the_ones_this_module_reads():
    """Guard against the accessor names drifting under these tests."""
    from kika.nuclear_data.model import ConversionReport

    names = {f.name for f in dataclasses.fields(ConversionReport)}
    assert {"warnings", "losses", "approximations", "unsupported"} <= names
