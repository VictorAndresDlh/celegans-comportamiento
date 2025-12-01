#!/usr/bin/env python
# coding: utf-8

"""
Methodology 6: Lévy Flight and Path Structure Analysis

This script implements the step length analysis from Moy et al. (2015) to
distinguish treatment effects on the N2 strain. It identifies turning events,
calculates step lengths between them, and fits a power-law distribution to
analyze the search strategy (Brownian vs. Lévy flight).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import powerlaw
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap

from utils_data import load_data_for_strain

# ===========================================================================
# PUBLICATION-QUALITY STYLING
# ===========================================================================

plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'patch.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True
})

# COMPREHENSIVE COLOR PALETTE
PRIMARY_COLORS = {
    'blue': '#2E86AB',
    'orange': '#FD7E14',
    'green': '#198754',
    'red': '#DC3545',
    'purple': '#6F42C1',
    'gray': '#6C757D',
    'yellow': '#FFC107',
    'teal': '#20C997',
    'pink': '#E91E63',
    'brown': '#795548',
    'cyan': '#17A2B8',
    'magenta': '#E83E8C'
}

def setup_plot_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to any matplotlib axis."""
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    return ax

# ===========================================================================
# DATA LOADING AND PROCESSING
# ===========================================================================

"""Data loading is centralized in `utils_data.load_data_for_strain`."""

# ===========================================================================
# STATISTICAL FUNCTIONS
# ===========================================================================

def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (q-values) for each test.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return p_values

    order = np.argsort(p_values)
    p_sorted = p_values[order]

    q_sorted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = p_sorted[i] * m / rank
        if q < prev:
            prev = q
        q_sorted[i] = prev

    q_values = np.empty_like(q_sorted)
    q_values[order] = np.minimum(q_sorted, 1.0)
    return q_values


def bootstrap_alpha_ci(step_lengths, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Bootstrap confidence interval for power-law alpha parameter.

    Returns:
        (alpha_lower, alpha_upper): confidence interval bounds
    """
    if len(step_lengths) < 10:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)

    def fit_alpha(data):
        try:
            fit = powerlaw.Fit(data, discrete=False, verbose=False)
            return fit.power_law.alpha
        except:
            return np.nan

    # Bootstrap resampling
    alphas = []
    for _ in range(n_bootstrap):
        sample = rng.choice(step_lengths, size=len(step_lengths), replace=True)
        alpha = fit_alpha(sample)
        if not np.isnan(alpha):
            alphas.append(alpha)

    if len(alphas) == 0:
        return (np.nan, np.nan)

    alphas = np.array(alphas)
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100

    return (np.percentile(alphas, lower_percentile),
            np.percentile(alphas, upper_percentile))


# ===========================================================================
# LÉVY FLIGHT ANALYSIS
# ===========================================================================

def get_steps_from_trajectory(trajectory, angle_threshold_deg=40):
    """
    Identifies turning events and returns the list of step lengths between them.
    A turning event is defined as a change in direction greater than the threshold.

    Note: WMicrotracker SMART samples at 1 Hz (1 frame = 1 second), so the data
    is already properly sampled according to Moy et al. (2015) specifications.
    No additional temporal sampling is needed.
    """
    if len(trajectory) < 3:
        return []

    # 1. Data is already sampled at 1 Hz by the WMicrotracker SMART hardware.
    # We use the raw trajectory points for turning event detection.
    points = trajectory

    # 2. Identify turning events
    vectors = np.diff(points, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # Unwrap angles to avoid artificial 360° jumps when crossing -π/π
    angles_unwrapped = np.unwrap(angles)
    turning_angles = np.diff(angles_unwrapped)
    turning_angles_deg = np.abs(np.rad2deg(turning_angles))

    angle_threshold = angle_threshold_deg
    turn_indices = np.where(turning_angles_deg > angle_threshold)[0] + 1

    # Always include the start and end points
    turn_indices = np.unique(np.concatenate(([0], turn_indices, [len(points)-1])))

    # 3. Calculate step lengths
    turning_points = points[turn_indices]
    step_lengths = np.sqrt(np.sum(np.diff(turning_points, axis=0)**2, axis=1))
    
    return step_lengths[step_lengths > 0] # Filter out zero-length steps

# ===========================================================================
# TRAJECTORY VISUALIZATIONS
# ===========================================================================

def create_velocity_colored_trajectories(treatment_data, output_dir, target_strain, max_trajectories=10):
    """Create trajectory plots colored by velocity (like Moy et al. 2015, Figure 6).

    Args:
        treatment_data: dict mapping treatment names to DataFrames
        output_dir: Path to save plots
        target_strain: Strain name for titles
        max_trajectories: Maximum number of trajectories to plot per treatment
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    print(f"\nGenerating velocity-colored trajectory visualizations...")

    for treatment, df in treatment_data.items():
        print(f"  Creating velocity plot for {treatment}...")

        # Get unique tracks
        track_ids = df['track_id'].unique()
        if len(track_ids) == 0:
            continue

        # Select representative trajectories
        selected_tracks = track_ids[:max_trajectories]

        # Create figure with subplots for individual trajectories
        n_tracks = len(selected_tracks)
        if n_tracks == 0:
            continue

        # Create a grid layout
        ncols = min(3, n_tracks)
        nrows = (n_tracks + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        if n_tracks == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_tracks > 1 else axes

        fig.suptitle(f'Velocity-Colored Trajectories: {treatment} ({target_strain})',
                    fontsize=16, fontweight='bold')

        for idx, track_id in enumerate(selected_tracks):
            ax = axes[idx]

            # Get trajectory data
            track_df = df[df['track_id'] == track_id].sort_values('frame')
            x = track_df['x'].values
            y = track_df['y'].values

            if len(x) < 2:
                ax.axis('off')
                continue

            # Calculate instantaneous velocity
            dx = np.diff(x)
            dy = np.diff(y)
            velocity = np.sqrt(dx**2 + dy**2)

            # Prepare line segments for coloring
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create colormap based on velocity (using plasma for better contrast on white background)
            norm = Normalize(vmin=0, vmax=np.percentile(velocity, 95))
            lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2)
            lc.set_array(velocity)

            # Plot
            line = ax.add_collection(lc)
            ax.plot(x[0], y[0], 'o', color='darkgreen',
                   markersize=4, label='Start', zorder=10, markeredgecolor='black', markeredgewidth=0.5)
            ax.plot(x[-1], y[-1], 's', color='darkred',
                   markersize=4, label='End', zorder=10, markeredgecolor='black', markeredgewidth=0.5)

            # Styling
            ax.set_xlim(x.min() - 50, x.max() + 50)
            ax.set_ylim(y.min() - 50, y.max() + 50)
            ax.set_aspect('equal')
            ax.invert_yaxis()

            setup_plot_style(ax,
                           title=f'Track {idx+1} (n={len(x)} points)',
                           xlabel='X Position (pixels)',
                           ylabel='Y Position (pixels)')

            # Add colorbar
            cbar = plt.colorbar(line, ax=ax, label='Velocity (pixels/frame)')
            cbar.ax.tick_params(labelsize=8)

            # Legend
            ax.legend(loc='upper right', fontsize=8)

        # Hide unused subplots
        for idx in range(n_tracks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = output_dir / f'velocity_trajectories_{treatment.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved to {plot_path}")

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    """Main execution function."""
    print("--- Starting Lévy Flight Analysis Methodology ---")
    root_data_dir = Path('Datos')
    base_output_dir = Path('results/Analysis/Levy_Flight')
    strains = ['N2', 'NL5901', 'UA44', 'TJ356', 'BR5270', 'BR5271', 'ROSELLA']

    # Collect all p-values for global FDR correction
    all_test_results = []

    for target_strain in strains:
        print(f"\n\n===================================================")
        print(f"=== Running Analysis for Strain: {target_strain} ===")
        print(f"===================================================\n")

        output_dir = base_output_dir / target_strain
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Data
        treatment_data = load_data_for_strain(root_data_dir, target_strain)

        if not treatment_data:
            print(f"No data found for strain {target_strain}. Skipping.")
            continue

        # 2. Create velocity-colored trajectory visualizations
        create_velocity_colored_trajectories(treatment_data, output_dir, target_strain)

        print("\n--- Lévy Flight Analysis Results ---")
        results_summary = []
        per_worm_results = []

        for treatment, df in treatment_data.items():
            print(f"\n--- Analyzing Treatment: {treatment} ---")

            # Per-worm analysis
            worm_alphas = []
            worm_Rs = []
            worm_ps = []
            all_step_lengths = []

            for track_id, worm_df in df.groupby('track_id'):
                trajectory = worm_df.sort_values('frame')[['x', 'y']].values
                steps = get_steps_from_trajectory(trajectory)

                if len(steps) >= 50:  # Minimum 50 steps for reliable fitting
                    try:
                        fit_worm = powerlaw.Fit(steps, discrete=False, verbose=False)
                        alpha_worm = fit_worm.power_law.alpha
                        R_worm, p_worm = fit_worm.distribution_compare('power_law', 'lognormal')

                        worm_alphas.append(alpha_worm)
                        worm_Rs.append(R_worm)
                        worm_ps.append(p_worm)

                        # Store per-worm result
                        per_worm_results.append({
                            'strain': target_strain,
                            'treatment': treatment,
                            'track_id': track_id,
                            'n_steps': len(steps),
                            'alpha': alpha_worm,
                            'R': R_worm,
                            'p_value': p_worm,
                            'is_levy': (R_worm > 0 and p_worm < 0.05)
                        })
                    except Exception as e:
                        print(f"    Warning: Could not fit worm {track_id}: {e}")

                if len(steps) > 0:
                    all_step_lengths.extend(steps)

            if len(all_step_lengths) < 50:
                print("  -> Not enough step data to perform analysis.")
                continue

            # Pooled analysis across all worms
            fit = powerlaw.Fit(all_step_lengths, discrete=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            # Bootstrap confidence intervals for alpha
            print("  -> Computing bootstrap confidence intervals...")
            alpha_ci_lower, alpha_ci_upper = bootstrap_alpha_ci(
                all_step_lengths,
                n_bootstrap=1000,
                confidence_level=0.95,
                random_state=42
            )

            # Per-worm statistics
            n_worms_tested = len(worm_alphas)
            if n_worms_tested > 0:
                n_levy_worms = sum(1 for i in range(len(worm_Rs))
                                  if worm_Rs[i] > 0 and worm_ps[i] < 0.05)
                percent_levy = (n_levy_worms / n_worms_tested) * 100
            else:
                n_levy_worms = 0
                percent_levy = 0.0

            results_summary.append({
                'treatment': treatment,
                'alpha': alpha,
                'alpha_ci_lower': alpha_ci_lower,
                'alpha_ci_upper': alpha_ci_upper,
                'xmin': xmin,
                'loglikelihood_ratio': R,
                'p_value': p,
                'n_worms_tested': n_worms_tested,
                'n_levy_worms': n_levy_worms,
                'percent_levy_worms': percent_levy,
                'n_total_steps': len(all_step_lengths)
            })

            # Store for global FDR correction
            all_test_results.append({
                'strain': target_strain,
                'treatment': treatment,
                'p_value': p,
                'R': R
            })

            print(f"  Pooled power-law fit (alpha): {alpha:.3f} [{alpha_ci_lower:.3f}, {alpha_ci_upper:.3f}]")
            print(f"  Log-likelihood ratio vs Lognormal: R={R:.3f}, p-value={p:.3f}")
            print(f"  Worms showing Lévy pattern: {n_levy_worms}/{n_worms_tested} ({percent_levy:.1f}%)")

            # --- CCDF Plot (Complementary Cumulative Distribution Function) ---
            # This is the standard visualization for Lévy flights (Moy et al. 2015, Figure 9)
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot CCDF instead of PDF - this is the standard for power-law distributions
            # Always plot empirical data
            fit.plot_ccdf(color=PRIMARY_COLORS['blue'], linewidth=2.5, ax=ax, label='Empirical Data')

            # Try to plot power-law fit (may fail for extreme distributions)
            try:
                fit.power_law.plot_ccdf(color=PRIMARY_COLORS['red'], linestyle='--', linewidth=2,
                                        ax=ax, label=f'Power-law fit (α={alpha:.2f})')
            except Exception as e:
                print(f"    Warning: Could not plot power-law fit: {e}")

            # Try to plot lognormal fit (may fail for extreme distributions)
            try:
                fit.lognormal.plot_ccdf(color=PRIMARY_COLORS['green'], linestyle='--', linewidth=2,
                                       ax=ax, label='Lognormal fit')
            except Exception as e:
                print(f"    Warning: Could not plot lognormal fit: {e}")

            # Apply publication styling
            setup_plot_style(ax,
                           title=f'Step Length Distribution: {treatment} ({target_strain})',
                           xlabel='Step Length (pixels)',
                           ylabel='P(X ≥ x) [CCDF]')

            # Set log-log scale (standard for power-law visualization)
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Legend with enhanced styling
            legend = ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)

            # Add annotation with statistical results
            annotation_text = (f'{percent_levy:.1f}% worms Lévy-like\n'
                             f'R = {R:.3f}, p = {p:.3f}\n'
                             f'α = {alpha:.2f} [{alpha_ci_lower:.2f}, {alpha_ci_upper:.2f}]')
            ax.text(0.02, 0.02, annotation_text,
                   transform=ax.transAxes, ha='left', va='bottom',
                   fontsize=9, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='black', alpha=0.9, linewidth=1.5))

            plt.tight_layout()
            plot_path = output_dir / f'levy_fit_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  CCDF plot saved to {plot_path}")

        if not results_summary:
            print(f"No results to save for strain {target_strain}.")
            continue

        # Save per-treatment summary
        results_df = pd.DataFrame(results_summary)
        treatment_order = [
            'Control', 'ETANOL',
            'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM',
            'CBDV 0.3uM', 'CBDV 3uM', 'CBDV 30uM',
            'Total_Extract_CBD'
        ]
        present_treatments = [t for t in treatment_order if t in results_df['treatment'].unique()]
        results_df['treatment'] = pd.Categorical(results_df['treatment'], categories=present_treatments, ordered=True)
        results_df = results_df.sort_values('treatment')

        summary_path = output_dir / 'levy_flight_summary.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"\nTreatment summary saved to {summary_path}")

        # Save per-worm results
        if per_worm_results:
            per_worm_df = pd.DataFrame([r for r in per_worm_results if r['strain'] == target_strain])
            per_worm_path = output_dir / 'levy_flight_per_worm.csv'
            per_worm_df.to_csv(per_worm_path, index=False)
            print(f"Per-worm results saved to {per_worm_path}")

    # Apply global FDR correction across all strains and treatments
    if all_test_results:
        print("\n\n===================================================")
        print("=== Applying FDR Correction Across All Tests ===")
        print("===================================================\n")

        test_df = pd.DataFrame(all_test_results)
        p_values = test_df['p_value'].values
        q_values = benjamini_hochberg(p_values, alpha=0.05)
        test_df['q_value_BH'] = q_values
        test_df['significant_FDR'] = q_values < 0.05

        # Save global FDR results
        fdr_path = base_output_dir / 'levy_flight_fdr_correction.csv'
        test_df.to_csv(fdr_path, index=False)
        print(f"Global FDR correction results saved to {fdr_path}")

        n_sig_uncorrected = (test_df['p_value'] < 0.05).sum()
        n_sig_corrected = test_df['significant_FDR'].sum()
        print(f"\nSignificant results (p < 0.05): {n_sig_uncorrected}/{len(test_df)}")
        print(f"Significant results (q < 0.05, FDR-corrected): {n_sig_corrected}/{len(test_df)}")
        print(f"False positives eliminated: {n_sig_uncorrected - n_sig_corrected}")

    print("\n--------------------------------------")

if __name__ == "__main__":
    main()
