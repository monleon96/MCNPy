import pytest
import kika
import os
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_sdf():

    # Generated filename
    test_file_name = 'test_1.00e+00_3.00e+00.sdf'
    
    try:
        reference_sdf_path = _DATA + '/sdf/ref_1.00e+00_3.00e+00.sdf'

        pertfile1 = _DATA + '/sdf/pertfile_26056_PERT.i'
        mctalfile1 = _DATA + '/sdf/pertfile_26056.m'
        pertfile2 = _DATA + '/sdf/pertfile_26054_PERT.i'
        mctalfile2 = _DATA + '/sdf/pertfile_26054.m'
        pertfile3 = _DATA + '/sdf/pertfile_1001_PERT.i'
        mctalfile3 = _DATA + '/sdf/pertfile_1001.m'
        pertfile4 = _DATA + '/sdf/pertfile_8016_PERT.i'
        mctalfile4 = _DATA + '/sdf/pertfile_8016.m'

        sensData1 = kika.compute_sensitivity(pertfile1, mctalfile1, 4, 26056, 'Fe56')
        sensData2 = kika.compute_sensitivity(pertfile2, mctalfile2, 4, 26054, 'Fe54')
        sensData3 = kika.compute_sensitivity(pertfile3, mctalfile3, 4, 1001, 'H1')
        sensData4 = kika.compute_sensitivity(pertfile4, mctalfile4, 4, 8016, 'O16')

        list_of_sens = [
            sensData1,
            sensData2,
            sensData3,
            sensData4
        ]

        sdf_data = kika.create_sdf_data(list_of_sens, energy='1.00e+00_3.00e+00', title='test')

        assert sdf_data.title == 'test'
        assert sdf_data.energy == '1.00e+00_3.00e+00'
        assert sdf_data.pert_energies == [1e-11, 3e-09, 7.5e-09, 1e-08, 2.53e-08, 3e-08, 4e-08, 5e-08, 7e-08, 1e-07, 1.5e-07, 2e-07, 2.25e-07, 2.5e-07, 2.75e-07, 3.25e-07, 3.5e-07, 3.75e-07, 4e-07, 6.25e-07, 1e-06, 1.77e-06, 3e-06, 4.75e-06, 6e-06, 8.1e-06, 1e-05, 3e-05, 0.0001, 0.00055, 0.003, 0.017, 0.025, 0.1, 0.4, 0.9, 1.4, 1.85, 2.354, 2.479, 3.0, 4.8, 6.434, 8.1873, 20.0]
        assert sdf_data.r0 == 1.06652e-09
        assert sdf_data.e0 == 0.0173
        assert sdf_data.data[2].mt == 4
        assert sdf_data.data[0].nuclide == 'Fe-56'
        assert sdf_data.data[18].zaid == 1001

        sdf_data.group_inelastic_reactions(replace=True)

        sdf_data.write_file()

        # Check that files have the same content line by line
        with open(reference_sdf_path, 'r') as f1, open(test_file_name, 'r') as f2:
            ref_lines = f1.readlines()
            test_lines = f2.readlines()
            
            # Check if files have same number of lines
            assert len(ref_lines) == len(test_lines), "Files have different number of lines"
            
            # Compare each line
            for i, (ref_line, test_line) in enumerate(zip(ref_lines, test_lines)):
                assert ref_line == test_line, f"Difference at line {i+1}: ref='{ref_line}', test='{test_line}'"
    
    except FileNotFoundError as e:
        pytest.fail(f"File not found: {e}")
    except IOError as e:
        pytest.fail(f"IO error when reading or writing files: {e}")
    except AssertionError as e:
        pytest.fail(f"Assertion failed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
    finally:
        # Clean up - delete the generated file after test completes
        if os.path.exists(test_file_name):
            os.remove(test_file_name)


def test_read_scale_dialect_sdf():
    """Read a standard SCALE/TSUNAMI SDF file (foreign dialect).

    Unlike KIKA's own writer output, this dialect has a free-form title line,
    lowercase ``e`` scientific notation, a ``k-eff`` line with plain decimals and
    a trailing comment, and energies in eV. The parser must accept all of these
    and normalise the grid to MeV. Fixture is a 3-reaction trim of the OECD
    SNEAK-LMFR-EXP-001-001 file (the SCALE 238-group structure in eV).
    """
    from kika.energy_grids.utils import _identify_energy_grid

    fixture = os.path.join(os.path.dirname(__file__), 'data', 'sdf',
                           'scale_dialect_eV.sdf')

    sdf = kika.read_sdf(fixture)

    # 238-group structure -> 239 boundaries; 3 reaction blocks kept.
    assert len(sdf.pert_energies) == 239
    assert len(sdf.data) == 3

    # Energies normalised from eV to MeV (top boundary 2e7 eV -> 20 MeV) and
    # recognised as the SCALE 238-group grid.
    assert sdf.pert_energies == sorted(sdf.pert_energies)  # ascending
    assert sdf.pert_energies[-1] == pytest.approx(20.0)
    assert sdf.pert_energies[0] == pytest.approx(1e-11)
    assert _identify_energy_grid(sdf.pert_energies) == 'SCALE238'

    # k-eff line parsed despite plain decimals + trailing comment.
    assert sdf.r0 == pytest.approx(1.003930)
    assert sdf.e0 == pytest.approx(0.000290)

    # Free-form title preserved verbatim (no magic suffix to strip).
    assert sdf.title.startswith('Warning, this file was generated')

    # Reaction headers tokenised; ZAID/MT read, nuclide derived from ZAID.
    assert [(r.zaid, r.mt) for r in sdf.data] == [(6000, 1), (6000, 2), (6000, 4)]
    assert sdf.data[0].nuclide == 'C-0'

    # Lowercase-``e`` sensitivities parsed to non-zero floats.
    assert any(np.any(np.asarray(r.sensitivity) != 0.0) for r in sdf.data)
