import os
import pytest
import numpy as np
import pandas as pd
from kika.mcnp.parse_mctal import read_mctal

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def test_mctal_parsing():
    
    filename = _DATA + "/mctal/mctalfile_test_1.m"

    try:
        # Parse file (set pert=False to avoid TallyPert parsing)
        mctal = read_mctal(filename)

        # --- Test #sym:MCTAL variables ---
        # Header values
        assert mctal.code_name == "mcnp6"  
        assert mctal.ver == "6"              
        assert mctal.probid == "02/22/25 15:50:25"        
        assert mctal.knod == 11              
        assert mctal.nps == 100000                
        assert mctal.rnr == 140497964                

        # Problem ID
        assert mctal.problem_id == "attenuation Sphere for PWR"  

        # Tally header info
        assert mctal.ntal == 3              
        assert mctal.npert == 0             
        assert mctal.tally_numbers == [4, 14, 24]  

        # --- Test #sym:Tally 4 variables ---
        tally = mctal.tally.get(4)
        assert tally is not None, "Tally 4 not found"
        assert tally.tally_id == 4         
        assert tally.name == "shell Detector"         
        assert tally.energies == [0.1, 1.0, 3.0]  
        assert tally.results == [2.95887e-09, 8.25405e-09, 8.62196e-10]   
        assert tally.errors == [0.1038, 0.0765, 0.1932]       
        assert tally.integral_result == 1.20751e-08        
        assert tally.integral_error == 0.0597          
        assert tally.tfc_nps == [8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000, 96000, 100000]
        assert tally.tfc_results == [1.5011e-08, 1.20618e-08, 1.25793e-08, 1.30825e-08, 1.25922e-08, 1.23967e-08, 1.27288e-08, 1.26059e-08, 1.21273e-08, 1.25449e-08, 1.24136e-08, 1.21039e-08, 1.20751e-08]
        assert tally.tfc_errors == [0.17687, 0.137365, 0.117759, 0.0955509, 0.0871119, 0.081184, 0.078529, 0.075061, 0.0710826, 0.0677334, 0.0636706, 0.0610153, 0.059727]
        assert tally.tfc_fom == [1124.46, 930.741, 842.564, 955.983, 920.563, 883.742, 808.925, 774.51, 767.004, 760.409, 782.636, 781.449, 782.933]
        assert tally.perturbation == {}

        # --- Test #sym:Tally 14 variables ---
        tally = mctal.tally.get(14)
        assert tally is not None, "Tally 14 not found"
        assert tally.tally_id == 14         
        assert tally.name == "test Tally"         
        assert tally.energies == []  
        assert tally.results == [1.35263e-05]   
        assert tally.errors == [0.0023]       
        assert tally.integral_result == 1.35263e-05        
        assert tally.integral_error == 0.0023          
        assert tally.tfc_nps == [8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000, 96000, 100000]
        assert tally.tfc_results == [1.34675e-05, 1.35004e-05, 1.35215e-05, 1.34978e-05, 1.35318e-05, 1.35378e-05, 1.35273e-05, 1.35321e-05, 1.3526e-05, 1.35276e-05, 1.35138e-05, 1.35209e-05, 1.35263e-05]
        assert tally.tfc_errors == [0.00806785, 0.00569246, 0.00463665, 0.00402376, 0.00359537, 0.0032863, 0.00304824, 0.00285357, 0.00268642, 0.00254727, 0.0024327, 0.00232932, 0.00228126]
        assert tally.tfc_fom == [540427.0, 541977.0, 543477.0, 539084.0, 540408.0, 539326.0, 536871.0, 535891.0, 537002.0, 537655.0, 536119.0, 536193.0, 536679.0]
        assert tally.perturbation == {}   

    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
    except ValueError as e:
        pytest.fail(f"Invalid data in MCTAL file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing MCTAL file: {str(e)}")


def test_mctal_pert_parsing():
    
    filename = _DATA + "/pert/mctalfile_PERT_test_1.m"

    try:
        # Parse file (set pert=False to avoid TallyPert parsing)
        mctal = read_mctal(filename)

        # --- Test #sym:MCTAL variables ---
        # Header values
        assert mctal.code_name == "mcnp6"  
        assert mctal.ver == "6"              
        assert mctal.probid == "05/24/24 17:18:48"        
        assert mctal.knod == 11              
        assert mctal.nps == 10000000                
        assert mctal.rnr == 14044013305                

        # Problem ID
        assert mctal.problem_id == "attenuation Sphere for PWR"  

        # Tally header info
        assert mctal.ntal == 1              
        assert mctal.npert == 1760             
        assert mctal.tally_numbers == [4]  

        # --- Test #sym:Tally variables ---
        tally = mctal.tally[4].perturbation[174]
        assert tally.tally_id == 4         
        assert tally.name == "shell Detector"         
        assert tally.energies == [0.1, 1.0, 3.0]  
        assert tally.results == [2.65504e-09, 9.16002e-09, 1.06652e-09] 
        assert tally.errors == [0.0226, 0.0306, 0.0173]     
        assert tally.integral_result == 1.28816e-08       
        assert tally.integral_error == 0.0223         
        assert tally.tfc_nps == [512000, 1024000, 1536000, 2048000, 2560000, 3072000, 3584000, 4096000, 4608000, 5120000, 5632000, 6144000, 6656000, 7168000, 7680000, 8192000, 8704000, 9216000, 9728000, 10000000]
        assert tally.tfc_results == [1.41484e-08, 1.28689e-08, 1.34668e-08, 1.34216e-08, 1.31317e-08, 1.31887e-08, 1.30217e-08, 1.27547e-08, 1.27062e-08, 1.26384e-08, 1.26825e-08, 1.24796e-08, 1.26279e-08, 1.26995e-08, 1.27588e-08, 1.27599e-08, 1.27685e-08, 1.28999e-08, 1.28646e-08, 1.28816e-08]
        assert tally.tfc_errors == [0.0894553, 0.0775128, 0.0637202, 0.0529134, 0.0461339, 0.0415367, 0.0392754, 0.0360147, 0.0336184, 0.0313968, 0.0298067, 0.0285205, 0.0271316, 0.0264072, 0.0254242, 0.0243953, 0.0234621, 0.0235264, 0.0227834, 0.0223234]
        assert tally.tfc_fom == [1.91203, 1.27507, 1.25838, 1.36868, 1.4406, 1.48117, 1.41996, 1.47769, 1.5074, 1.55553, 1.56906, 1.57097, 1.60234, 1.57072, 1.58176, 1.61057, 1.6388, 1.53938, 1.55498, 1.57561]
        assert tally.perturbation == {}    

    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
    except ValueError as e:
        pytest.fail(f"Invalid data in MCTAL file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing MCTAL file: {str(e)}")


def test_mctal_multidim_parsing():

    filename = _DATA + "/mctal/mctalfile_test_2.m"

    try:
        # Parse file (set pert=False to avoid TallyPert parsing)
        mctal = read_mctal(filename)

        # --- Test #sym:MCTAL variables ---
        # Header values
        assert mctal.code_name == "mcnp6"  
        assert mctal.ver == "6"              
        assert mctal.probid == "03/21/25 09:04:37"        
        assert mctal.knod == 2              
        assert mctal.nps == 110079                
        assert mctal.rnr == 70134725                

        # Problem ID
        assert mctal.problem_id == "Example 3 MODEL"  

        # Tally header info
        assert mctal.ntal == 33              
        assert mctal.npert == 0             
        assert mctal.tally_numbers == [10014, 10024, 10034, 10044, 10054, 10064, 10074, 10084, 10094, 10104, 10114, 10124, 10134, 10144, 10154, 10164, 10174, 10184, 10194, 10204, 10214, 10224, 10234, 10244, 10254, 10264, 10274, 10284, 10294, 10304, 20014, 20024, 20234] 

        # --- Test #sym:Tally 10014 variables ---
        tally = mctal.tally.get(10014)
        assert mctal.tally.get(4) is None, "Tally 4 not found"
        assert tally.tally_id == 10014         
        assert tally.name == "Reaction rate 58Ni(n,p)  (Vol.) in (-1,+2)"         
        assert tally.energies == [0.1, 1.0, 3.0, 20.0]  
        assert tally.results == [0.0, 150578.0, 8901150.0, 51558600.0, 0.0, 111063.0, 66178300.0, 0.0, 0.0, 214811.0, 26666300.0, 104354000.0, 0.0, 173543.0, 43900100.0, 39612000.0, 0.0, 115372.0, 70412600.0, 177830000.0, 0.0, 74983.1, 33045500.0, 60255200.0, 0.0, 92540.3, 42515400.0, 56592200.0]   
        assert tally.errors == [0.0, 0.7105, 0.5135, 0.7182, 0.0, 0.7287, 0.6156, 0.0, 0.0, 0.3045, 0.3549, 0.3315, 0.0, 0.6447, 0.6623, 0.7261, 0.0, 0.3534, 0.3141, 0.3591, 0.0, 0.9804, 0.7646, 1.0, 0.0, 0.6652, 0.4196, 0.7575]     
        
        # Get the integral energy data
        integral_data = tally.get_integral_energy_data()
        
        # Expected result values
        expected_results = np.array([
            [[6.06103e+07]],
            [[6.62893e+07]],
            [[1.31235e+08]],
            [[8.36857e+07]],
            [[2.48358e+08]],
            [[9.33757e+07]],
            [[9.92002e+07]]
        ])
        
        # Expected error values
        expected_errors = np.array([
            [[0.6156]],
            [[0.6146]],
            [[0.283]],
            [[0.4887]],
            [[0.2721]],
            [[0.6997]],
            [[0.4681]]
        ])
        
        # Check that results match expected values
        assert 'Result' in integral_data
        assert 'Error' in integral_data
        assert np.allclose(integral_data['Result'], expected_results.reshape(integral_data['Result'].shape))
        assert np.allclose(integral_data['Error'], expected_errors.reshape(integral_data['Error'].shape))
        
        assert tally.get_dimensions() == {'cell': 1, 'segment': 7, 'multiplier': 1, 'energy': 4}
        
        # Test slices using np.allclose for float comparison
        slice_result, slice_error = tally.get_slice(energy=1, segment=1)
        assert np.allclose(slice_result, np.array([[111063.]]))
        assert np.allclose(slice_error, np.array([[0.7287]]))
        
        slice_result, slice_error = tally.get_slice(segment=2)
        expected_slice_result = np.array([[[0.00000e+00, 2.14811e+05, 2.66663e+07, 1.04354e+08]]])
        expected_slice_error = np.array([[[0.0, 0.3045, 0.3549, 0.3315]]])
        assert np.allclose(slice_result, expected_slice_result)
        assert np.allclose(slice_error, expected_slice_error)
        
        assert tally.tfc_nps == [110079]
        assert tally.tfc_results == [99200200.0]
        assert tally.tfc_errors == [0.468054]
        assert tally.perturbation == {}

    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
    except ValueError as e:
        pytest.fail(f"Invalid data in MCTAL file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing MCTAL file: {str(e)}")


def test_to_dataframe():
    """Test the to_dataframe method for simple and multidimensional tallies."""
    
    # Test with simple tally (energy-only)
    filename = _DATA + "/mctal/mctalfile_test_1.m"
    try:
        mctal = read_mctal(filename)
        
        # Test tally 4 (with energy bins)
        tally = mctal.tally.get(4)
        df = tally.to_dataframe()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Energy', 'Result', 'Error']
        assert len(df) == 3  # 3 energy bins
        
        # Check values
        assert np.allclose(df['Energy'].values, [0.1, 1.0, 3.0])
        assert np.allclose(df['Result'].values, [2.95887e-09, 8.25405e-09, 8.62196e-10])
        assert np.allclose(df['Error'].values, [0.1038, 0.0765, 0.1932])
        
        # Test tally 14 (no energy bins)
        tally = mctal.tally.get(14)
        df = tally.to_dataframe()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert 'Result' in df.columns
        assert 'Error' in df.columns
        assert len(df) == 1  # Single result
        
        # Check values
        assert np.isclose(df['Result'].values[0], 1.35263e-05)
        assert np.isclose(df['Error'].values[0], 0.0023)
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")

    # Test with multidimensional tally
    filename = _DATA + "/mctal/mctalfile_test_2.m"
    try:
        mctal = read_mctal(filename)
        tally = mctal.tally.get(10014)
        
        # Get the dataframe
        df = tally.to_dataframe()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert set(['Cell', 'Segment', 'Multiplier', 'Energy', 'Result', 'Error']).issubset(set(df.columns))
        
        # Check dimensions
        dimensions = tally.get_dimensions()
        expected_rows = dimensions['cell'] * dimensions['segment'] * dimensions['multiplier'] * dimensions['energy']
        assert len(df) == expected_rows
        
        # Verify some values from different segments
        segment0_energy1 = df[(df['Segment'] == 0) & (df['Energy'] == 1.0)]
        assert len(segment0_energy1) == 1
        assert np.isclose(segment0_energy1['Result'].values[0], 150578.0)
        assert np.isclose(segment0_energy1['Error'].values[0], 0.7105)
        
        segment1_energy1 = df[(df['Segment'] == 1) & (df['Energy'] == 1.0)]
        assert len(segment1_energy1) == 1
        assert np.isclose(segment1_energy1['Result'].values[0], 111063.0)
        assert np.isclose(segment1_energy1['Error'].values[0], 0.7287)
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
        
    # Test with perturbation data
    filename = _DATA + "/pert/mctalfile_PERT_test_1.m"
    try:
        mctal = read_mctal(filename)
        
        # Test perturbation to_dataframe
        tally = mctal.tally.get(4)
        pert = tally.perturbation[174]
        df = pert.to_dataframe()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Energy', 'Result', 'Error']
        assert len(df) == 3  # 3 energy bins
        
        # Check specific values
        assert np.allclose(df['Energy'].values, [0.1, 1.0, 3.0])
        assert np.allclose(df['Result'].values, [2.65504e-09, 9.16002e-09, 1.06652e-09])
        assert np.allclose(df['Error'].values, [0.0226, 0.0306, 0.0173])
        
        # Test perturbation collection to_dataframe
        pertcol_df = tally.perturbation.to_dataframe()
        assert isinstance(pertcol_df, pd.DataFrame)
        assert 'Perturbation' in pertcol_df.columns
        assert set([174]).issubset(set(pertcol_df['Perturbation'].unique()))
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")


def test_to_xarray():
    """Test the to_xarray method for simple and multidimensional tallies."""
    
    # Skip if xarray is not available
    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray not installed")
    
    # Test with simple tally (energy-only)
    filename = _DATA + "/mctal/mctalfile_test_1.m"
    try:
        mctal = read_mctal(filename)
        
        # Test tally 4 (with energy bins)
        tally = mctal.tally.get(4)
        ds = tally.to_xarray()
        
        # Check dataset structure
        assert isinstance(ds, xr.Dataset)
        assert 'result' in ds.data_vars
        assert 'error' in ds.data_vars
        assert 'energy' in ds.coords
        
        # Check dimensions - use .sizes instead of .dims to avoid FutureWarning
        assert set(ds.dims) == {'cell', 'energy'}
        assert ds.sizes['energy'] == 3
        
        # Check values
        assert np.allclose(ds.energy.values, [0.1, 1.0, 3.0])
        assert np.allclose(ds.result.values, [2.95887e-09, 8.25405e-09, 8.62196e-10])
        assert np.allclose(ds.error.values, [0.1038, 0.0765, 0.1932])
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")

    # Test with multidimensional tally
    filename = _DATA + "/mctal/mctalfile_test_2.m"
    try:
        mctal = read_mctal(filename)
        tally = mctal.tally.get(10014)
        
        # Get the xarray dataset
        ds = tally.to_xarray()
        
        # Check dataset structure
        assert isinstance(ds, xr.Dataset)
        assert 'result' in ds.data_vars
        assert 'error' in ds.data_vars
        
        # Check dimensions - use .sizes instead of .dims to avoid FutureWarning
        dimensions = tally.get_dimensions()
        assert set(ds.dims) == set(dimensions.keys())
        for dim, size in dimensions.items():
            assert ds.sizes[dim] == size
        
        # Check some values using xarray's selection
        result_value = ds.result.sel(energy=1.0, segment=0).values.item()
        error_value = ds.error.sel(energy=1.0, segment=0).values.item()
        assert np.isclose(result_value, 150578.0)
        assert np.isclose(error_value, 0.7105)
        
        result_value = ds.result.sel(energy=1.0, segment=1).values.item()
        error_value = ds.error.sel(energy=1.0, segment=1).values.item()
        assert np.isclose(result_value, 111063.0)
        assert np.isclose(error_value, 0.7287)
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")


def test_integral_energy_data():
    """Test methods for getting energy-integrated data."""
    
    filename = _DATA + "/mctal/mctalfile_test_2.m"
    try:
        mctal = read_mctal(filename)
        tally = mctal.tally.get(10014)
        
        # Test get_integral_energy_data
        integral_data = tally.get_integral_energy_data()
        
        # Check structure
        assert 'Result' in integral_data
        assert 'Error' in integral_data
        
        # Check shape matches expected dimensions (segment, cell, multiplier)
        dimensions = tally.get_dimensions()
        expected_shape = (dimensions['segment'], dimensions['cell'], dimensions['multiplier'])
        reshaped_data = integral_data['Result'].reshape(dimensions['segment'], dimensions['cell'], dimensions['multiplier'])
        
        # Check values from the reshaped array match expected values
        assert np.isclose(reshaped_data[0, 0, 0], 6.06103e+07)
        assert np.isclose(reshaped_data[1, 0, 0], 6.62893e+07)
        
        # Test get_integral_energy_dataframe
        df = tally.get_integral_energy_dataframe()
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert set(['Cell', 'Segment', 'Multiplier', 'Energy', 'Result', 'Error']).issubset(set(df.columns))
        
        # All rows should have Energy = 'Integral'
        assert all(df['Energy'] == 'Integral')
        
        # Should have one row per segment (since cell and multiplier are both 1)
        assert len(df) == dimensions['segment']
        
        # Check specific values
        segment0 = df[df['Segment'] == 0]
        assert np.isclose(segment0['Result'].values[0], 6.06103e+07)
        assert np.isclose(segment0['Error'].values[0], 0.6156)
        
        segment1 = df[df['Segment'] == 1]
        assert np.isclose(segment1['Result'].values[0], 6.62893e+07)
        assert np.isclose(segment1['Error'].values[0], 0.6146)
        
    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
