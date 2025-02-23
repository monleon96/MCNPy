import pytest
from src.input.parse_input import read_mcnp, read_PERT


def test_input_parsing_single_card(tmp_path):
    # Test a simple input file with one PERT card.
    content = "PERT1:n CELL=1,2,3 MAT=5 RHO=0.95 METHOD=2 RXN=101 ERG=0.0 10.0\n"
    file = tmp_path / "input_test1.i"
    file.write_text(content)
    
    inst = read_mcnp(str(file))
    pert = inst.pert.perturbation.get(1)
    assert pert is not None
    assert pert.particle == "n"
    assert pert.cell == [1, 2, 3]
    assert pert.material == 5
    assert pert.rho == 0.95
    assert pert.method == 2
    assert pert.reaction == 101
    assert pert.energy == (0.0, 10.0)

def test_input_parsing_multiline_card(tmp_path):
    # Test a PERT card spanning multiple lines with '&' continuation.
    content = (
        "PERT2:f CELL=1,2,3 &\n"
        "MAT=5 RHO=1.0 &\n"
        "METHOD=2 RXN=102 ERG=0.5 9.5\n"
    )
    file = tmp_path / "input_test2.i"
    file.write_text(content)
    
    inst = read_mcnp(str(file))
    pert = inst.pert.perturbation.get(2)
    assert pert is not None
    assert pert.particle == "f"
    assert pert.cell == [1, 2, 3]
    assert pert.material == 5
    assert pert.rho == 1.0
    assert pert.method == 2
    assert pert.reaction == 102
    assert pert.energy == (0.5, 9.5)

def test_PERT_parsing_valid():
    # Test direct parsing of a valid PERT card.
    lines = ["PERT3:p CELL=4,5 MAT=7 RHO=0.88 METHOD=3 RXN=103 ERG=1.0 11.0"]
    pert_obj, new_index = read_PERT(lines, 0)
    assert pert_obj is not None
    assert pert_obj.id == 3
    assert pert_obj.particle == "p"
    assert pert_obj.cell == [4,5]
    assert pert_obj.material == 7
    assert pert_obj.rho == 0.88
    assert pert_obj.method == 3
    assert pert_obj.reaction == 103
    assert pert_obj.energy == (1.0, 11.0)
    assert new_index == 1

def test_PERT_parsing_invalid_header():
    # Test that a card with an invalid header returns None.
    lines = ["XYZ3:p CELL=4,5 MAT=7 RHO=0.88 METHOD=3 RXN=103 ERG=1.0 11.0"]
    pert_obj, new_index = read_PERT(lines, 0)
    assert pert_obj is None

def test_PERT_parsing_incomplete_erg():
    # Test a card with incomplete ERG values so that energy is set to None.
    lines = ["PERT4:a CELL=1 MAT=5 RHO=1.0 METHOD=1 RXN=100 ERG=5.0"]
    pert_obj, new_index = read_PERT(lines, 0)
    assert pert_obj is not None
    # Energy should be None because only one value is provided.
    assert pert_obj.energy is None

