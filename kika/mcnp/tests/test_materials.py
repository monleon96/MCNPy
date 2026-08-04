import pytest
import os
import filecmp
import tempfile
from kika.mcnp.parse_input import read_mcnp
from kika.mcnp.pert_generator import perturb_material
from kika.mcnp.parse_materials import read_material
from kika.materials import Material

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_material_parsing():

    matfile = _DATA + "/mat/matfile_test_1.i"
    inputfile = _DATA + "/input/inputfile_test_1.i"

    try:

        # Input file with only materials
        materials = read_mcnp(matfile).materials

        # --- Test Material properties using by_id accessor ---
        assert materials.by_id[100000].libs.get('nlib') == '06c'
        assert materials.by_id[300000].libs.get('nlib') == '06c'
        assert materials.by_id[300000].libs.get('plib') == '02p'
        assert materials.by_id[100000].libs.get('plib') == None
        assert materials.by_id[300000].id == 300000
        assert materials.by_id[300000].nuclide['C13'].libs.get('nlib') == '03c'
        assert materials.by_id[300000].nuclide['C13'].libs.get('plib') == '04p'
        assert materials.by_id[300000].nuclide['C13'].zaid == 6013
        assert materials.by_id[300000].nuclide['C13'].element == 'C'
        assert materials.by_id[300000].nuclide['C13'].fraction == pytest.approx(0.0001022318)
        assert materials.by_id[300000].nuclide['Fe56'].fraction == pytest.approx(0.8810613)
        assert materials.by_id[300000].nuclide['Cr54'].fraction == pytest.approx(6.288632e-05)

        materials.by_id[200100].to_weight_fraction()
        materials.by_id[300000].to_weight_fraction()

        # Fractions are always stored as positive values
        assert materials.by_id[300000].nuclide['C13'].fraction == pytest.approx(2.397746932859688e-05)
        assert materials.by_id[300000].nuclide['Fe56'].fraction == pytest.approx(0.8888964955001305)
        assert materials.by_id[300000].nuclide['Cr54'].fraction == pytest.approx(6.118148592568498e-05)

        # Verify fraction_type changed
        assert materials.by_id[300000].fraction_type == 'wo'

        materials.by_id[200100].to_atomic_fraction()
        materials.by_id[300000].to_atomic_fraction()

        assert materials.by_id[300000].nuclide['C13'].fraction == pytest.approx(0.00010223180430777937)
        assert materials.by_id[300000].nuclide['Fe56'].fraction == pytest.approx(0.8810613371256076)
        assert materials.by_id[300000].nuclide['Cr54'].fraction == pytest.approx(6.288632264986424e-05)

        # Verify fraction_type changed back
        assert materials.by_id[300000].fraction_type == 'ao'

        # Verify MCNP output contains expected content
        mcnp_200100 = materials.by_id[200100].to_mcnp()
        assert 'm200100' in mcnp_200100
        assert 'nlib=06c' in mcnp_200100

        mcnp_300000 = materials.by_id[300000].to_mcnp()
        assert 'm300000' in mcnp_300000
        assert 'nlib=06c' in mcnp_300000
        assert 'plib=02p' in mcnp_300000

        # Complete input file
        materials = read_mcnp(inputfile).materials

        # --- Test Material properties ---
        assert materials.by_id[100000].libs.get('nlib') == '06c'
        assert materials.by_id[300000].libs.get('nlib') == '06c'
        assert materials.by_id[300000].libs.get('plib') == None
        assert materials.by_id[100000].libs.get('plib') == None
        assert materials.by_id[300000].id == 300000
        assert materials.by_id[300000].nuclide[6013].libs.get('nlib') == None
        assert materials.by_id[300000].nuclide[6013].libs.get('plib') == None
        assert materials.by_id[300000].nuclide[6013].zaid == 6013
        assert materials.by_id[300000].nuclide[6013].element == 'C'
        assert materials.by_id[300000].nuclide[6013].fraction == 0.0001022318
        assert materials.by_id[300000].nuclide[26056].fraction == 0.8810613
        assert materials.by_id[300000].nuclide[24054].fraction == 6.288632e-05

        materials.by_id[200100].to_weight_fraction()
        materials.by_id[300000].to_weight_fraction()

        # Fractions stored as positive values; fraction_type tracks the type
        assert materials.by_id[300000].nuclide[6013].fraction == pytest.approx(2.397746932859688e-05)
        assert materials.by_id[300000].nuclide[26056].fraction == pytest.approx(0.8888964955001305)
        assert materials.by_id[300000].nuclide[24054].fraction == pytest.approx(6.118148592568498e-05)

        # Verify MCNP output for weight fractions shows negative values
        mcnp_200100 = materials.by_id[200100].to_mcnp()
        assert 'm200100' in mcnp_200100
        assert '-' in mcnp_200100  # Weight fractions shown as negative in MCNP

        materials.by_id[200100].to_atomic_fraction()
        materials.by_id[300000].to_atomic_fraction()

        assert materials.by_id[300000].nuclide[6013].fraction == pytest.approx(0.00010223180430777937)
        assert materials.by_id[300000].nuclide[26056].fraction == pytest.approx(0.8810613371256076)
        assert materials.by_id[300000].nuclide[24054].fraction == pytest.approx(6.288632264986424e-05)

        # Verify MCNP output for atomic fractions shows positive values
        mcnp_300000 = materials.by_id[300000].to_mcnp()
        assert 'm300000' in mcnp_300000

    except FileNotFoundError as e:
        pytest.fail(f"Test file not found: {str(e)}")
    except ValueError as e:
        pytest.fail(f"Invalid data in material file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing material file: {str(e)}")


def test_mat_perturbation():
    # Define input file
    matfile = _DATA + "/mat/matfile_test_1.i"

    try:
        # Parse materials
        materials = read_mcnp(matfile).materials

        # Perturb material using new API (MaterialCollection, not file path)
        perturbed = perturb_material(materials, 300000, 26056, pert_mat_id=31,
                                     in_place=False, fraction_type='weight')

        # Verify perturbed material exists
        assert 31 in perturbed.by_id

        # Verify the perturbed material has the same nuclides
        original = materials.by_id[300000]
        pert_mat = perturbed.by_id[31]
        assert 26056 in [n.zaid for n in pert_mat.nuclide.values()]

    except FileNotFoundError as e:
        pytest.fail(f"Test file not found: {str(e)}")
    except ValueError as e:
        pytest.fail(f"Invalid data in material file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error in material perturbation test: {str(e)}")


def test_one_line_material_parsing():
    """Test parsing of materials formatted in a single line."""

    # Test case 1: Simple one-line material
    test_case_1 = ["m100 1001 0.66667 8016 0.33333"]
    mat_obj, idx = read_material(test_case_1, 0)

    assert mat_obj.id == 100
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].zaid == 1001
    assert mat_obj.nuclide[1001].fraction == 0.66667
    assert mat_obj.nuclide[8016].zaid == 8016
    assert mat_obj.nuclide[8016].fraction == 0.33333
    assert idx == 1

    # Test case 2: One-line material with negative fractions (weight fractions)
    # Fractions are stored as absolute values; fraction_type tracks the type
    test_case_2 = ["m200 1001 -0.112 8016 -0.888"]
    mat_obj, idx = read_material(test_case_2, 0)

    assert mat_obj.id == 200
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == 0.112  # abs value stored
    assert mat_obj.nuclide[8016].fraction == 0.888  # abs value stored
    assert mat_obj.fraction_type == 'wo'  # weight fraction type inferred

    # Test case 3: One-line material with material-level libraries
    test_case_3 = ["m300 nlib=70c plib=12p 1001 0.5 8016 0.5"]
    mat_obj, idx = read_material(test_case_3, 0)

    assert mat_obj.id == 300
    assert mat_obj.libs.get('nlib') == "70c"
    assert mat_obj.libs.get('plib') == "12p"
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == 0.5
    assert mat_obj.nuclide[8016].fraction == 0.5

    # Test case 4: One-line material with nuclide-level libraries
    test_case_4 = ["m400 1001.80c 0.6 8016.70c 0.4"]
    mat_obj, idx = read_material(test_case_4, 0)

    assert mat_obj.id == 400
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].libs.get('nlib') == "80c"
    assert mat_obj.nuclide[8016].libs.get('nlib') == "70c"
    assert mat_obj.nuclide[1001].fraction == 0.6
    assert mat_obj.nuclide[8016].fraction == 0.4

    # Test case 5: One-line material with both material and nuclide libraries
    test_case_5 = ["m500 nlib=70c 1001 0.4 8016.80c 0.6"]
    mat_obj, idx = read_material(test_case_5, 0)

    assert mat_obj.id == 500
    assert mat_obj.libs.get('nlib') == "70c"
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].libs.get('nlib') is None  # Uses material-level lib
    assert mat_obj.nuclide[8016].libs.get('nlib') == "80c"  # Overrides material-level lib
    assert mat_obj.nuclide[1001].fraction == 0.4
    assert mat_obj.nuclide[8016].fraction == 0.6

    # Test case 6: One-line material with comments
    test_case_6 = ["m600 1001 0.7 8016 0.3 $ Water material"]
    mat_obj, idx = read_material(test_case_6, 0)

    assert mat_obj.id == 600
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == 0.7
    assert mat_obj.nuclide[8016].fraction == 0.3

    # Test case 7: One-line material with line continuation
    test_case_7 = [
        "m700 1001 0.3 8016 0.4 &",
        "     24052 0.15 26056 0.15"
    ]
    mat_obj, idx = read_material(test_case_7, 0)

    assert mat_obj.id == 700
    assert len(mat_obj.nuclide) == 4
    assert mat_obj.nuclide[1001].fraction == 0.3
    assert mat_obj.nuclide[8016].fraction == 0.4
    assert mat_obj.nuclide[24052].fraction == 0.15
    assert mat_obj.nuclide[26056].fraction == 0.15
    assert idx == 2

    # Test case 8: Complex case with multiple libraries and photon libraries
    test_case_8 = [
        "m800 nlib=80c plib=12p 1001.81c 0.1 8016.80c 0.2 &",
        "     13027.70c 0.3 92235.80c 0.4"
    ]
    mat_obj, idx = read_material(test_case_8, 0)

    assert mat_obj.id == 800
    assert mat_obj.libs.get('nlib') == "80c"
    assert mat_obj.libs.get('plib') == "12p"
    assert len(mat_obj.nuclide) == 4
    assert mat_obj.nuclide[1001].libs.get('nlib') == "81c"
    assert mat_obj.nuclide[8016].libs.get('nlib') == "80c"
    assert mat_obj.nuclide[13027].libs.get('nlib') == "70c"
    assert mat_obj.nuclide[92235].libs.get('nlib') == "80c"
    assert mat_obj.nuclide[1001].fraction == 0.1
    assert mat_obj.nuclide[8016].fraction == 0.2
    assert mat_obj.nuclide[13027].fraction == 0.3
    assert mat_obj.nuclide[92235].fraction == 0.4


def test_natural_element_conversion():
    """Test conversion of natural elements to their isotopic compositions."""

    # Test case 1: Material with natural carbon (atomic fractions)
    mat = Material(id=900)
    mat.add_nuclide(6000, 1.0, fraction_type='ao')  # Natural carbon

    # Create copy to preserve the original
    original_mat = mat.copy(900)

    # Test expand_natural_elements method
    mat.expand_natural_elements()

    # Verify original material is preserved in our copy
    assert 6000 in [n.zaid for n in original_mat.nuclide.values()]
    assert len(original_mat.nuclide) == 1

    # Verify expanded material has isotopes instead of natural element
    zaids = [n.zaid for n in mat.nuclide.values()]
    assert 6000 not in zaids
    assert 6012 in zaids
    assert 6013 in zaids

    # Verify fractions sum to approximately the original value
    total_fraction = sum(nuclide.fraction for nuclide in mat.nuclide.values())
    assert abs(total_fraction - 1.0) < 1e-10

    # Test case 2: Material with natural iron (weight fractions)
    mat = Material(id=901)
    mat.add_nuclide(26000, 1.0, fraction_type='wo')  # Natural iron (weight fraction)

    # Test in-place conversion
    mat.expand_natural_elements()

    # Verify natural element is replaced with isotopes
    zaids = [n.zaid for n in mat.nuclide.values()]
    assert 26000 not in zaids
    assert 26054 in zaids
    assert 26056 in zaids
    assert 26057 in zaids
    assert 26058 in zaids

    # Verify fractions sum to approximately the original value (positive values)
    total_fraction = sum(nuclide.fraction for nuclide in mat.nuclide.values())
    assert abs(total_fraction - 1.0) < 1e-10

    # Test case 3: Specific ZAID conversion
    mat = Material(id=902)
    mat.add_nuclide(6000, 0.5, fraction_type='ao')    # Natural carbon
    mat.add_nuclide(8016, 0.5, fraction_type='ao')    # Oxygen-16 (specific isotope)

    # Convert only carbon
    mat.expand_natural_elements(elements='C')

    # Verify specific conversion
    zaids = [n.zaid for n in mat.nuclide.values()]
    assert 6000 not in zaids
    assert 6012 in zaids
    assert 6013 in zaids
    assert 8016 in zaids  # Unchanged

    # Test error cases
    mat = Material(id=903)
    mat.add_nuclide(8016, 1.0, fraction_type='ao')    # Not a natural element

    # Attempt to convert non-natural element should raise ValueError
    try:
        mat.expand_natural_elements(elements='O16')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_fraction_conversion_with_natural_elements():
    """Test atomic/weight fraction conversion with natural elements."""

    # Create material with natural elements (atomic fractions)
    mat = Material(id=904)
    mat.add_nuclide(6000, 0.3, fraction_type='ao')    # Natural carbon
    mat.add_nuclide(26000, 0.7, fraction_type='ao')   # Natural iron

    # Create a copy for later verification
    original_mat = mat.copy(904)

    # Convert to weight fractions (in-place)
    mat.to_weight_fraction()

    # Verify natural elements are preserved
    zaids = [n.zaid for n in mat.nuclide.values()]
    assert 6000 in zaids
    assert 26000 in zaids

    # Verify fraction_type changed
    assert mat.fraction_type == 'wo'

    # Fractions are stored as positive values
    assert all(nuclide.fraction > 0 for nuclide in mat.nuclide.values())

    # Calculate expected weight fractions:
    # C: 0.3 * 12.011 = 3.6033
    # Fe: 0.7 * 55.845 = 39.0915
    # Total: 42.6948
    # C weight fraction: 3.6033 / 42.6948 ≈ 0.0844
    # Fe weight fraction: 39.0915 / 42.6948 ≈ 0.9156
    expected_c_weight = 0.0844
    expected_fe_weight = 0.9156

    # Check actual weight fractions against calculated values
    assert abs(mat.nuclide[6000].fraction - expected_c_weight) < 1e-3
    assert abs(mat.nuclide[26000].fraction - expected_fe_weight) < 1e-3

    # Convert back to atomic fractions (in-place)
    mat.to_atomic_fraction()

    # Verify original atomic fractions are restored
    expected_c_atomic = 0.3
    expected_fe_atomic = 0.7
    assert abs(mat.nuclide[6000].fraction - expected_c_atomic) < 1e-10
    assert abs(mat.nuclide[26000].fraction - expected_fe_atomic) < 1e-10

    # Test conversion with expanded isotopes
    # Start with a fresh copy
    expanded_mat = original_mat.copy(904)
    expanded_mat.expand_natural_elements()

    # Convert to weight fractions (in-place)
    expanded_mat.to_weight_fraction()

    # Verify all fractions are positive
    assert all(nuclide.fraction > 0 for nuclide in expanded_mat.nuclide.values())

    # Calculate total weight fraction by element after expansion
    c_weight_sum = sum(nuclide.fraction for zaid_key, nuclide in expanded_mat.nuclide.items()
                     if nuclide.zaid // 1000 == 6)
    fe_weight_sum = sum(nuclide.fraction for zaid_key, nuclide in expanded_mat.nuclide.items()
                      if nuclide.zaid // 1000 == 26)

    # Verify expanded isotope weight fractions sum to the expected element weight fractions
    assert abs(c_weight_sum - expected_c_weight) < 1e-3
    assert abs(fe_weight_sum - expected_fe_weight) < 1e-3

    # Convert back to atomic fractions (in-place)
    expanded_mat.to_atomic_fraction()

    # Sum up atomic fractions by element
    c_atomic_sum = sum(nuclide.fraction for zaid_key, nuclide in expanded_mat.nuclide.items()
                     if nuclide.zaid // 1000 == 6)
    fe_atomic_sum = sum(nuclide.fraction for zaid_key, nuclide in expanded_mat.nuclide.items()
                      if nuclide.zaid // 1000 == 26)

    # Verify expanded isotope atomic fractions sum to the original atomic fractions
    assert abs(c_atomic_sum - expected_c_atomic) < 1e-10
    assert abs(fe_atomic_sum - expected_fe_atomic) < 1e-10


# ── Serpent material tests ──────────────────────────────────────────────────

# The PWRSphere input is resolved by the root conftest through $KIKA_TAPES;
# these three tests previously hardcoded its absolute path with no guard at
# all, so they errored rather than skipped on a machine without the mount.


def test_serpent_material_parsing(serpent_input):
    """Test parsing materials from a Serpent input file."""
    from kika.materials import MaterialCollection

    mc = MaterialCollection.from_serpent(str(serpent_input))

    # Should find 3 materials
    assert len(mc) == 3

    # Check by_name accessor
    assert set(mc.by_name.keys()) == {"AIR", "WATER", "SS304"}

    # AIR: atomic density
    air = mc.by_name["AIR"]
    assert air.name == "AIR"
    assert air.density == pytest.approx(5.3248e-05)
    assert air.density_unit == "atoms/b-cm"
    assert len(air.nuclide) == 2
    assert air.fraction_type == "ao"
    assert air.metadata["rgb"] == (0, 128, 0)

    # WATER: atomic density
    water = mc.by_name["WATER"]
    assert water.density == pytest.approx(7.18825e-02)
    assert water.density_unit == "atoms/b-cm"
    assert len(water.nuclide) == 4
    assert water.nuclide["H1"].fraction == pytest.approx(6.666630e-01)
    assert water.metadata["rgb"] == (0, 0, 255)

    # SS304: mass density (negative in Serpent → stored as g/cc)
    ss = mc.by_name["SS304"]
    assert ss.density == pytest.approx(7.85)
    assert ss.density_unit == "g/cc"
    assert len(ss.nuclide) == 36
    assert ss.nuclide["Fe56"].fraction == pytest.approx(0.8810613)
    assert ss.nuclide["C13"].fraction == pytest.approx(1.022318e-04)
    assert ss.metadata["rgb"] == (255, 0, 0)

    # All nuclides should have 06c library
    for mat in mc:
        for nuclide in mat.nuclide.values():
            assert nuclide.libs.get("nlib") == "06c"


def test_serpent_roundtrip(serpent_input):
    """Test parse → export → re-parse roundtrip."""
    from kika.materials import MaterialCollection, read_serpent_materials

    mc1 = MaterialCollection.from_serpent(str(serpent_input))
    text1 = mc1.to_serpent()

    # Re-parse from the exported text
    lines = [l + "\n" for l in text1.split("\n")]
    mc2 = read_serpent_materials(lines)
    text2 = mc2.to_serpent()

    # Exact text match
    assert text1 == text2

    # Verify density values preserved
    for m1, m2 in zip(mc1, mc2):
        assert m1.density == pytest.approx(m2.density)
        assert m1.density_unit == m2.density_unit
        assert m1.metadata.get("rgb") == m2.metadata.get("rgb")

        # All nuclide fractions match
        for sym in m1.nuclide:
            n1 = m1.nuclide[sym]
            n2 = m2.nuclide[n1.zaid]
            assert n1.fraction == pytest.approx(n2.fraction)


def test_serpent_mcnp_interop(serpent_input):
    """Test Serpent→MCNP export and fraction conversion."""
    from kika.materials import MaterialCollection

    mc = MaterialCollection.from_serpent(str(serpent_input))

    # MCNP export should work
    mcnp_text = mc.to_mcnp()
    assert "m1" in mcnp_text
    assert "KIKA_MAT_NAME: AIR" in mcnp_text
    assert "KIKA_DENSITY:" in mcnp_text

    # Fraction type conversion roundtrip
    ss = mc.by_name["SS304"]
    orig_fracs = {n.zaid: n.fraction for n in ss.nuclide.values()}

    ss.to_weight_fraction()
    assert ss.fraction_type == "wo"

    ss.to_atomic_fraction()
    assert ss.fraction_type == "ao"

    for n in ss.nuclide.values():
        assert n.fraction == pytest.approx(orig_fracs[n.zaid], rel=1e-6)
