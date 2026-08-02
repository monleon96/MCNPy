from pathlib import Path

import numpy as np
import pytest

from kika.sensitivities.sdf import (
    SDFData,
    SDFReactionData,
    sensitivity_to_plot_data,
)
from kika.sensitivities.sdf_parser import read_sdf
from kika.sensitivities.sensitivity_processing import create_sdf_from_serpent


DATA = Path(__file__).parent / "data" / "sdf"


def test_reaction_absolute_contract_and_relative_constructor():
    reaction = SDFReactionData.from_relative_errors(
        zaid=26056,
        mt=2,
        sensitivity=[2.0, -4.0, 0.0],
        relative_error=[0.1, 0.25, 0.5],
    )
    assert reaction.error == pytest.approx([0.2, 1.0, 0.0])
    assert reaction.relative_error == pytest.approx([0.1, 0.25, 0.0])

    zero = SDFReactionData(
        zaid=26056, mt=2, sensitivity=[0.0], error=[0.3]
    )
    assert np.isinf(zero.relative_error[0])

    with pytest.raises(ValueError, match="same length"):
        SDFReactionData(zaid=26056, mt=2, sensitivity=[1.0], error=[])
    with pytest.raises(ValueError, match="non-negative"):
        SDFReactionData(zaid=26056, mt=2, sensitivity=[1.0], error=[-0.1])


def test_historical_kika_requires_explicit_relative_mode(tmp_path):
    legacy = DATA / "legacy_kika_relative.sdf"
    with pytest.raises(ValueError, match="uncertainty_convention='relative'"):
        read_sdf(str(legacy))

    sdf = read_sdf(str(legacy), uncertainty_convention="relative")
    assert sdf.r0 == pytest.approx(2.0)
    assert sdf.e0 == pytest.approx(0.2)
    assert sdf.relative_response_error == pytest.approx(0.1)
    assert sdf.data[0].sensitivity == pytest.approx([0.0, -2.0])
    assert sdf.data[0].error == pytest.approx([0.0, 0.5])

    sdf.write_file(str(tmp_path))
    migrated_path = next(tmp_path.glob("*.sdf"))
    assert "MCNP to SCALE sdf" not in migrated_path.read_text().splitlines()[0]
    migrated = read_sdf(str(migrated_path))
    assert migrated.pert_energies == pytest.approx(sdf.pert_energies)
    assert migrated.data[0].sensitivity == pytest.approx(sdf.data[0].sensitivity)
    assert migrated.data[0].error == pytest.approx(sdf.data[0].error)
    assert migrated.e0 == pytest.approx(sdf.e0)


def test_standard_writer_uses_ev_and_absolute_uncertainties(tmp_path):
    reaction = SDFReactionData(
        zaid=26056,
        mt=2,
        sensitivity=[1.0, -2.0],
        error=[0.2, 0.3],
    )
    sdf = SDFData(
        title="absolute",
        energy="test",
        pert_energies=[1.0e-11, 1.0e-3, 20.0],
        r0=3.0,
        e0=0.4,
        data=[reaction],
    )
    sdf.write_file(str(tmp_path))
    path = tmp_path / "absolute_test.sdf"
    text = path.read_text()
    assert "2.000000E+07" in text
    assert "+/-   4.000000E-01" in text

    restored = read_sdf(str(path))
    assert restored.pert_energies == pytest.approx(sdf.pert_energies)
    assert restored.data[0].error == pytest.approx([0.2, 0.3])
    assert restored.e0 == pytest.approx(0.4)


def test_plot_band_and_inelastic_grouping_use_absolute_sigma():
    plot, band = sensitivity_to_plot_data(
        [1.0, 2.0, 4.0],
        [2.0, -1.0],
        error=[0.5, 0.25],
        per_lethargy=False,
    )
    assert plot.y == pytest.approx([2.0, -1.0, -1.0])
    assert band.y_lower == pytest.approx([1.5, -1.25, -1.25])
    assert band.y_upper == pytest.approx([2.5, -0.75, -0.75])

    sdf = SDFData(
        title="group",
        energy="test",
        pert_energies=[1.0, 2.0, 4.0],
        data=[
            SDFReactionData(
                zaid=26056, mt=51, sensitivity=[1.0, -2.0], error=[0.3, 0.4]
            ),
            SDFReactionData(
                zaid=26056, mt=52, sensitivity=[-1.0, 3.0], error=[0.4, 0.3]
            ),
        ],
    )
    sdf.group_inelastic_reactions()
    grouped = next(reaction for reaction in sdf.data if reaction.mt == 4)
    assert grouped.sensitivity == pytest.approx([0.0, 1.0])
    assert grouped.error == pytest.approx([0.5, 0.5])


def test_scale_fixture_errors_are_absolute_magnitudes():
    sdf = read_sdf(str(DATA / "scale_dialect_eV.sdf"))
    assert sdf.e0 == pytest.approx(0.000290)
    for reaction in sdf.data:
        error = np.asarray(reaction.error)
        sensitivity = np.asarray(reaction.sensitivity)
        assert np.all(np.isfinite(error))
        assert np.all(error >= 0.0)
        nonzero = sensitivity != 0.0
        if np.any(nonzero):
            assert np.median(error[nonzero] / np.abs(sensitivity[nonzero])) < 0.1


def test_serpent_native_relative_errors_convert_at_sdf_boundary():
    class FakeSerpent:
        energy_grid = np.array([1.0e-11, 1.0, 20.0])
        data = {"sens_ratio": object()}
        responses = ["sens_ratio_BIN_0"]
        n_materials = 1
        n_nuclides = 1
        nuclides = [type("Nuclide", (), {"zai": 26056})()]
        perturbations = [type("Perturbation", (), {"index": 0, "mt": 2})()]

        def get_energy_dependent(self, response_name, mat, zai, mt):
            assert response_name == "sens_ratio_BIN_0"
            assert (mat, zai, mt) == (0, 0, 2)
            return (
                np.array([-2.0, 4.0]),
                np.array([0.1, 0.25]),
            )

    source = FakeSerpent()
    native_values, native_relative = source.get_energy_dependent(
        "sens_ratio_BIN_0", 0, 0, 2
    )
    sdf = create_sdf_from_serpent(
        source,
        "sens_ratio_BIN_0",
        "serpent",
        response_values=(2.0, 0.3),
    )

    assert native_values == pytest.approx([-2.0, 4.0])
    assert native_relative == pytest.approx([0.1, 0.25])
    assert sdf.data[0].sensitivity == pytest.approx([-2.0, 4.0])
    assert sdf.data[0].error == pytest.approx([0.2, 1.0])
    assert sdf.e0 == pytest.approx(0.3)
