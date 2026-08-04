"""
Script to compare angular distributions between perturbed ACE files and the original.

This script:
1. Loads the original (unperturbed) ACE file
2. Iterates through all 1024 perturbed ACE samples
3. Compares angular distributions at 2.85 MeV
4. Groups 8 samples per figure
5. Adds experimental EXFOR data points
6. Saves figures to the specified output directory
"""

import kika
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from kika.plotting import PlotBuilder
from kika.exfor import (
    load_all_exfor_data,
    extract_experiment_info,
    transform_lab_to_cm,
    jacobian_cm_to_lab
)

# Configuration
BASE_DIR = Path('/SCRATCH/users/monleon-de-la-jan/KIKA_LIB/JEFF40_LEG_ALL/ace/600/26056')
ORIGINAL_ACE = '/soft_snc/lib/ace/40/600/260560_40.06c'
OUTPUT_DIR = Path('/SCRATCH/users/monleon-de-la-jan/KIKA_LIB/JEFF40_LEG_ALL/figures')
EXFOR_DIR = '/share_snc/snc/JuanMonleon/EXFOR/data/'  # Directory containing EXFOR JSON files
ENERGY = 2.85  # MeV
MT = 2  # Elastic scattering
SAMPLES_PER_FIGURE = 8
TOTAL_SAMPLES = 1024

# Nuclear masses (atomic mass units)
from kika._constants import NEUTRON_MASS_AMU as M_NEUTRON
M_FE56 = 55.93494       # Fe-56 mass

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original ACE file
print(f"Loading original ACE file: {ORIGINAL_ACE}")
ace_original = kika.read_ace(ORIGINAL_ACE)
print("Original ACE file loaded successfully")

# Load EXFOR data
print(f"\nLoading EXFOR data from: {EXFOR_DIR}")
try:
    energy_data, sorted_energies = load_all_exfor_data(EXFOR_DIR)
    print(f"Successfully loaded EXFOR data. Available energies: {sorted(energy_data.keys())}")
except Exception as e:
    print(f"Warning: Could not load EXFOR data: {e}")
    energy_data = {}
    print("Continuing without EXFOR data...")

# Process samples in groups
for group_idx in range(0, TOTAL_SAMPLES, SAMPLES_PER_FIGURE):
    group_end = min(group_idx + SAMPLES_PER_FIGURE, TOTAL_SAMPLES)
    print(f"\nProcessing group {group_idx // SAMPLES_PER_FIGURE + 1}: samples {group_idx + 1} to {group_end}")
    
    # Create a new PlotBuilder for this group
    builder = PlotBuilder()
    
    # Add original ACE data (will be the same in all plots)
    ang_data_original = ace_original.to_plot_data(
        'angular',
        mt=MT,
        energy=ENERGY,
        label='Original (unperturbed)',
        color='black',
        linewidth=2.5,
        linestyle='-'
    )
    builder.add_data(ang_data_original)
    
    # Process each sample in this group
    samples_loaded = 0
    for sample_idx in range(group_idx + 1, group_end + 1):
        sample_name = f"{sample_idx:04d}"
        ace_file_path = BASE_DIR / sample_name / f"260560_40_{sample_name}.06c"
        
        if not ace_file_path.exists():
            print(f"  Warning: File not found - {ace_file_path}")
            continue
        
        try:
            # Load perturbed ACE file
            ace_perturbed = kika.read_ace(str(ace_file_path))
            
            # Create plot data for this sample
            ang_data_perturbed = ace_perturbed.to_plot_data(
                'angular',
                mt=MT,
                energy=ENERGY,
                label=f'Sample {sample_name}',
                linewidth=1.0,
                alpha=0.7
            )
            builder.add_data(ang_data_perturbed)
            samples_loaded += 1
            print(f"  Loaded sample {sample_name}")
            
        except Exception as e:
            print(f"  Error loading sample {sample_name}: {e}")
            continue
    
    # Build and save the plot if we have samples
    if samples_loaded > 0:
        # Build figure with PlotBuilder
        fig = (builder
               .set_labels(
                   title=f'Angular Distribution Comparison at {ENERGY} MeV\nSamples {group_idx + 1}-{group_end}',
                   x_label='cos(θ)',
                   y_label='Probability Density'
               )
               .set_legend(loc='best')
               .build())
        
        # Add EXFOR data points if available
        ax = fig.get_axes()[0]
        if ENERGY in energy_data:
            experiment_list = energy_data[ENERGY]
            # Use a color palette for experiments
            colors = plt.cm.Set1(np.linspace(0, 1, max(len(experiment_list), 3)))
            
            for i, (df_exp, meta_exp) in enumerate(experiment_list):
                try:
                    exp_label, year = extract_experiment_info(meta_exp)
                    
                    # Get data
                    angles = df_exp["angle"].values
                    dsig = df_exp["dsig"].values
                    err_stat = df_exp["error_stat"].values
                    data_frame = meta_exp['angle_frame']
                    
                    # Convert to CM frame if needed
                    if data_frame.upper() == 'LAB':
                        mu_lab = np.cos(np.deg2rad(angles))
                        mu_cm, dsig_cm = transform_lab_to_cm(mu_lab, dsig, M_NEUTRON, M_FE56)
                        mu_plot = mu_cm
                        dsig_plot = dsig_cm
                        alpha = M_NEUTRON / M_FE56
                        J = jacobian_cm_to_lab(mu_cm, alpha)
                        err_plot = err_stat / J
                    else:
                        # Data is already in CM frame
                        mu_plot = np.cos(np.deg2rad(angles))
                        dsig_plot = dsig
                        err_plot = err_stat
                    
                    # Plot experimental data
                    ax.errorbar(mu_plot, dsig_plot, yerr=err_plot,
                               fmt='o', color=colors[i], label=exp_label,
                               markersize=4, capsize=3, alpha=0.8)
                except Exception as e:
                    print(f"    Warning: Could not plot EXFOR data for experiment {i}: {e}")
            
            # Update legend to include EXFOR data
            ax.legend(loc='best', fontsize=8)
        
        # Save the figure
        output_filename = OUTPUT_DIR / f'AD_comparison_samples_{group_idx + 1:04d}_to_{group_end:04d}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved figure: {output_filename}")
        plt.close(fig)
    else:
        print(f"  No samples loaded for this group, skipping figure")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"Figures saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
