"""Regression tests for standard SCALE/TSUNAMI SDF dialect parsing.

These cover the forms found in DICE exports that KIKA's own writer does not emit:
trailing-whitespace padding on the fixed metadata line, varying integer unit/region
indices (including a trailing label glued to them), a k-eff line without an
uncertainty, per-region (non-(0,0)) profiles, and a trailing verification footer.
"""

import gzip

import pytest

from kika.sensitivities.sdf_parser import read_sdf


def _block(nuclide, reaction, zaid, mt, meta1, scalars, sens, err):
    """Build one SCALE-dialect reaction block (4-group)."""
    return "\n".join(
        [
            f"{nuclide:<13}{reaction:<17}{zaid:>5}{mt:>7}",
            meta1,
            "  0.000000E+00  0.000000E+00      0      0",
            scalars,
            sens,
            err,
        ]
    )


def _scale_sdf(keff_line: str) -> str:
    """A 4-group SCALE-dialect SDF with 3 profiles (2 region-integrated) + footer."""
    header = "\n".join(
        [
            "/some/scale/path/test.inp",
            "         4 number of neutron groups",
            "         3   number of sensitivity profiles          2 are region integrated",
            keff_line,
            "energy boundaries:",
            "  2.000000E+07  1.000000E+05  6.250000E-01  1.000000E-05  1.000000E-11",
        ]
    )
    scal = "  1.000000E-02  1.000000E-03  1.000000E-02  0.000000E+00  0.000000E+00"
    blocks = [
        # region-integrated (0,0), with trailing whitespace padding on meta1
        _block("fe-56", "elastic", 26056, 2, "      0      0        ", scal,
               "  4.000000E-03  3.000000E-03  2.000000E-03  1.000000E-03",
               "  1.000000E-04  1.000000E-04  1.000000E-04  1.000000E-04"),
        # per-region breakdown (unit -1) -> must be parsed but flagged non-integrated
        _block("u-235", "fission", 92235, 18, "     -1      0", scal,
               "  2.000000E-01  1.000000E-01  1.000000E-01  1.000000E-01",
               "  1.000000E-04  1.000000E-04  1.000000E-04  1.000000E-04"),
        # region-integrated (0,0) with a trailing label glued to the indices
        _block("u-235", "fission", 92235, 18, "      0      0benchmark-model", scal,
               "  2.500000E-01  1.500000E-01  5.000000E-02  5.000000E-02",
               "  1.000000E-04  1.000000E-04  1.000000E-04  1.000000E-04"),
    ]
    footer = "file verification information\n  checksum 0xdeadbeef"
    return header + "\n" + "\n".join(blocks) + "\n" + footer + "\n"


def test_scale_dialect_parses(tmp_path):
    p = tmp_path / "scale.sdf"
    p.write_text(_scale_sdf("  1.003930 +/-   0.000290  k-eff from the forward case"))
    sdf = read_sdf(str(p))

    assert sdf.r0 == pytest.approx(1.003930)
    assert sdf.e0 == pytest.approx(0.000290)
    assert len(sdf.pert_energies) == 5  # 4 groups + 1
    # 3 profiles parsed; the trailing footer must be ignored (declared count = 3).
    assert len(sdf.data) == 3

    # unit/region captured; (0,0) marks region-integrated system totals.
    region_integrated = [d for d in sdf.data if (d.unit, d.region) == (0, 0)]
    assert len(region_integrated) == 2
    assert {(d.zaid, d.mt) for d in region_integrated} == {(26056, 2), (92235, 18)}
    per_region = [d for d in sdf.data if (d.unit, d.region) != (0, 0)]
    assert len(per_region) == 1 and per_region[0].unit == -1


def test_scale_dialect_keff_without_uncertainty(tmp_path):
    p = tmp_path / "scale_nokeff.sdf"
    p.write_text(_scale_sdf("  1.000717    k-eff from the forward case"))
    sdf = read_sdf(str(p))
    assert sdf.r0 == pytest.approx(1.000717)
    assert sdf.e0 == 0.0
    assert len(sdf.data) == 3


def test_scale_dialect_gzip_roundtrip(tmp_path):
    """Files come from DICE gzipped; ensure gunzip -> parse works end to end."""
    raw = _scale_sdf("  1.003930 +/-   0.000290  k-eff from the forward case")
    gz = tmp_path / "scale.sdf.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(raw)
    plain = tmp_path / "scale.sdf"
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        plain.write_text(f.read())
    sdf = read_sdf(str(plain))
    assert len(sdf.data) == 3
