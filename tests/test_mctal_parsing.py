import pytest
import src

def test_mctal_parsing():
    
    filename = "tests/data/F4Tally_test1.m"

    try:
        # Parse file (set pert=False to avoid TallyPert parsing)
        mctal = src.read_mctal(filename, tally_ids=[4,14], tfc=True, pert=False)

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
        assert tally.pert_data == {}

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
        assert tally.pert_data == {}   

    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
    except ValueError as e:
        pytest.fail(f"Invalid data in MCTAL file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing MCTAL file: {str(e)}")


def test_mctal_pert_parsing():
    
    filename = "tests/data/F4PertTally_test1.m"

    try:
        # Parse file (set pert=False to avoid TallyPert parsing)
        mctal = src.read_mctal(filename, tally_ids=[4], tfc=True, pert=True)

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
        tally = mctal.tally[4].pert_data[174]
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
        assert tally.pert_data == {}    

    except FileNotFoundError:
        pytest.fail(f"Test file not found: {filename}")
    except ValueError as e:
        pytest.fail(f"Invalid data in MCTAL file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing MCTAL file: {str(e)}")
