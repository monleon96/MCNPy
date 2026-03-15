"""
Generate explanatory figures for Chapter 3 of the thesis manuscript.

Produces two publication-quality figures:
1. overlap_weights_diagram.png — Gaussian resolution kernels and overlap weights
2. hybrid_correlation_method.png — 4-panel hybrid correlation blending illustration

Usage:
    python3 scripts/generate_chapter3_figures.py

Output directory: manuscript/CHAPTER-3/Figures/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

# Ensure kika is importable
_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# Output directory
OUTPUT_DIR = _kika_path / "manuscript" / "CHAPTER-3" / "Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style for manuscript figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color-blind friendly palette (same as kika plotting)
COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00",
           "#CC78BC", "#CA9161", "#FBAFE4", "#949494"]


def generate_overlap_weights_figure():
    """
    Figure 1: Overlap weights diagram.

    Shows how experiments with different TOF energy resolutions produce
    Gaussian kernels that overlap with energy bins, and how the CDF-based
    overlap weights are computed.
    """
    # Define a set of energy bins (representative subset of the union grid)
    bin_edges = np.array([1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60,
                          1.65, 1.70, 1.75, 1.80, 1.85, 1.90])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # Define 4 representative experiments with different resolutions
    experiments = [
        {"name": "Kinney (1976)",     "E": 1.48, "sigma_E": 0.015, "color": COLORS[0]},
        {"name": "Pirovano (2019)",   "E": 1.60, "sigma_E": 0.035, "color": COLORS[1]},
        {"name": "Smith (1980)",      "E": 1.72, "sigma_E": 0.050, "color": COLORS[2]},
        {"name": "Cierjacks (1978)",  "E": 1.55, "sigma_E": 0.070, "color": COLORS[3]},
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1.2, 1],
                             gridspec_kw={"hspace": 0.35})

    # --- Top panel: Gaussian resolution kernels + bin boundaries ---
    ax = axes[0]

    # Draw bin boundaries as vertical grey lines
    for edge in bin_edges:
        ax.axvline(edge, color="grey", lw=0.5, ls="-", alpha=0.3)

    # Shade alternating bins for visibility
    for i in range(n_bins):
        alpha_shade = 0.06 if i % 2 == 0 else 0.12
        ax.axvspan(bin_edges[i], bin_edges[i + 1], color="grey", alpha=alpha_shade)

    # Plot Gaussian kernels for each experiment
    e_fine = np.linspace(1.20, 2.00, 500)
    for exp in experiments:
        kernel = norm.pdf(e_fine, loc=exp["E"], scale=exp["sigma_E"])
        # Normalize to unit peak for visibility
        kernel = kernel / kernel.max()
        ax.plot(e_fine, kernel, color=exp["color"], lw=2.0,
                label=f'{exp["name"]}, $\\sigma_E$ = {exp["sigma_E"]*1000:.0f} keV')
        # Mark nominal energy
        ax.axvline(exp["E"], color=exp["color"], lw=1.0, ls=":", alpha=0.5)

    ax.set_xlabel("Neutron energy (MeV)")
    ax.set_ylabel("Normalized kernel amplitude")
    ax.set_title("(a) TOF energy-resolution kernels and energy bins")
    ax.set_xlim(1.25, 1.95)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True, fancybox=True)

    # Label a few bins
    for i in [3, 5, 7]:
        ax.text(bin_centers[i], 1.07, f"Bin {i+1}",
                ha="center", va="bottom", fontsize=8, color="grey")

    # --- Bottom panel: Overlap weights as stacked bar chart ---
    ax = axes[1]

    bar_width = 0.8 * (bin_edges[1] - bin_edges[0])
    bottom = np.zeros(n_bins)

    for exp in experiments:
        # Compute CDF-based overlap weights
        weights = np.zeros(n_bins)
        for i in range(n_bins):
            w = (norm.cdf(bin_edges[i + 1], loc=exp["E"], scale=exp["sigma_E"])
                 - norm.cdf(bin_edges[i], loc=exp["E"], scale=exp["sigma_E"]))
            weights[i] = w

        # Only show bins with weight > threshold
        weights[weights < 1e-3] = 0.0

        ax.bar(bin_centers, weights, width=bar_width, bottom=bottom,
               color=exp["color"], alpha=0.7, edgecolor="white", lw=0.5,
               label=exp["name"])
        bottom += weights

    ax.set_xlabel("Neutron energy (MeV)")
    ax.set_ylabel("Overlap weight $\\omega_{g,i}$")
    ax.set_title("(b) CDF-based overlap weights per experiment and bin")
    ax.set_xlim(1.25, 1.95)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=True, fancybox=True, ncol=2)

    # Add annotation explaining the mechanism
    ax.annotate(
        "Broader kernels $\\rightarrow$ more bins $\\rightarrow$ stronger correlations",
        xy=(1.60, ax.get_ylim()[1] * 0.85),
        fontsize=9, ha="center", style="italic", color="grey",
    )

    out_path = OUTPUT_DIR / "overlap_weights_diagram.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_hybrid_correlation_figure():
    """
    Figure 2: Hybrid correlation blending (4-panel).

    Shows:
    (a) Gaussian parametric correlation (smooth exponential decay)
    (b) Kernel-weight MC correlation (data-driven, noisier)
    (c) Per-bin reliability score alpha
    (d) Final hybrid blend
    """
    np.random.seed(42)
    n_bins = 20
    energies = np.linspace(0.85, 4.0, n_bins)

    # --- (a) Gaussian parametric correlation ---
    # Smooth exponential decay based on energy separation / mean resolution
    length_scale = 0.8  # MeV
    corr_gauss = np.exp(-0.5 * ((energies[:, None] - energies[None, :]) / length_scale) ** 2)

    # --- (b) KW correlation (simulated data-driven) ---
    # Simulate realistic KW correlations: strong for nearby bins sharing data,
    # weaker and noisier for distant bins
    rng = np.random.default_rng(42)
    n_mc = 500
    raw = rng.normal(size=(n_mc, n_bins))

    # Introduce correlations from shared experiments (nearest 3 bins share data)
    for i in range(n_bins):
        for j in range(max(0, i - 3), min(n_bins, i + 4)):
            if i != j:
                overlap = max(0, 1.0 - 0.3 * abs(i - j))
                raw[:, j] += overlap * raw[:, i]

    corr_kw_raw = np.corrcoef(raw.T)
    # Ensure diagonal = 1 and clip
    np.fill_diagonal(corr_kw_raw, 1.0)
    corr_kw = np.clip(corr_kw_raw, -1, 1)

    # --- (c) Per-bin alpha (reliability score) ---
    # Simulate: edges of the energy range have fewer overlapping experiments
    # Center has better coverage
    alpha_per_bin = np.zeros(n_bins)
    for i in range(n_bins):
        e = energies[i]
        # More data available in 1-2.5 MeV (Kinney) and 2.5-4 MeV (Smith)
        # Less at the boundaries and around 2.5 MeV transition
        if e < 1.0:
            alpha_per_bin[i] = 0.15
        elif e < 1.5:
            alpha_per_bin[i] = 0.70 + 0.1 * rng.random()
        elif e < 2.0:
            alpha_per_bin[i] = 0.80 + 0.1 * rng.random()
        elif e < 2.5:
            alpha_per_bin[i] = 0.65 + 0.1 * rng.random()
        elif e < 3.0:
            alpha_per_bin[i] = 0.55 + 0.1 * rng.random()
        elif e < 3.5:
            alpha_per_bin[i] = 0.45 + 0.1 * rng.random()
        else:
            alpha_per_bin[i] = 0.25 + 0.1 * rng.random()

    alpha_per_bin = np.clip(alpha_per_bin, 0, 1)

    # Pairwise alpha: conservative rule
    alpha_ij = np.minimum(alpha_per_bin[:, None], alpha_per_bin[None, :])

    # --- (d) Hybrid blend ---
    corr_hybrid = alpha_ij * corr_kw + (1.0 - alpha_ij) * corr_gauss
    corr_hybrid = (corr_hybrid + corr_hybrid.T) / 2.0
    np.fill_diagonal(corr_hybrid, 1.0)
    corr_hybrid = np.clip(corr_hybrid, -1, 1)

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    extent = [energies[0], energies[-1], energies[-1], energies[0]]
    cmap = "RdBu_r"
    vmin, vmax = -0.3, 1.0

    # Panel (a): Gaussian
    im0 = axes[0].imshow(corr_gauss, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect="equal", extent=extent, interpolation="nearest")
    axes[0].set_title("(a) Gaussian $\\mathbf{R}_{\\mathrm{Gauss}}$")
    axes[0].set_xlabel("Energy (MeV)")
    axes[0].set_ylabel("Energy (MeV)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Correlation")

    # Panel (b): KW
    im1 = axes[1].imshow(corr_kw, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect="equal", extent=extent, interpolation="nearest")
    axes[1].set_title("(b) Kernel-weight $\\mathbf{R}_{\\mathrm{KW}}$")
    axes[1].set_xlabel("Energy (MeV)")
    axes[1].set_ylabel("Energy (MeV)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label="Correlation")

    # Panel (c): Alpha bar chart
    bar_width = 0.7 * (energies[1] - energies[0])
    bar_colors = plt.cm.viridis(alpha_per_bin)
    axes[2].bar(energies, alpha_per_bin, width=bar_width,
                color=bar_colors, edgecolor="white", lw=0.5)
    axes[2].set_xlabel("Energy (MeV)")
    axes[2].set_ylabel("Reliability $\\alpha_i$")
    axes[2].set_title("(c) Per-bin reliability score")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlim(energies[0] - 0.2, energies[-1] + 0.2)
    # Add annotation
    axes[2].axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
    axes[2].text(energies[-1] + 0.1, 0.52, "$\\alpha = 0.5$",
                 fontsize=8, color="grey", va="bottom")

    # Panel (d): Hybrid blend
    im3 = axes[3].imshow(corr_hybrid, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect="equal", extent=extent, interpolation="nearest")
    axes[3].set_title("(d) Hybrid $\\mathbf{R}_{\\mathrm{hyb}}$")
    axes[3].set_xlabel("Energy (MeV)")
    axes[3].set_ylabel("Energy (MeV)")
    plt.colorbar(im3, ax=axes[3], shrink=0.8, label="Correlation")

    plt.tight_layout()

    out_path = OUTPUT_DIR / "hybrid_correlation_method.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating Chapter 3 explanatory figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    generate_overlap_weights_figure()
    generate_hybrid_correlation_figure()

    print()
    print("Done. All figures saved.")
