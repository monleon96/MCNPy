import pytest

from kika.serpent.utils import parse_perturbation_label
from kika.serpent.sens import PertCategory


@pytest.mark.parametrize("label", [
    "inl scatt xs",
    "inel scatt xs",
    "inelastic scatt xs",
    "inl scatter xs",
])
def test_inelastic_scattering_labels_map_to_mt4(label):
    pert = parse_perturbation_label(label, index=0)
    assert pert.mt == 4
    assert pert.category == PertCategory.MT_XS
    assert "4" in (pert.short_label or "")


def test_total_and_mt_still_work():
    p_total = parse_perturbation_label("total xs", index=1)
    assert p_total.mt == 1
    assert p_total.category == PertCategory.MT_XS

    p_mt = parse_perturbation_label("mt 102 xs", index=2)
    assert p_mt.mt == 102
    assert p_mt.category == PertCategory.MT_XS
