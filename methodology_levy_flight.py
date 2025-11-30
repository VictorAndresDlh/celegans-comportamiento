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

from utils_data import load_data_for_strain

# ===========================================================================
# DATA LOADING AND PROCESSING
# ===========================================================================

"""Data loading is centralized in `utils_data.load_data_for_strain`."""

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
    turning_angles = np.diff(angles)
    # Normalize to [-π, π] range before taking absolute value
    turning_angles = (turning_angles + np.pi) % (2 * np.pi) - np.pi
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
# MAIN EXECUTION
# ===========================================================================

def main():
    """Main execution function."""
    print("--- Starting Lévy Flight Analysis Methodology ---")
    root_data_dir = Path('Datos')
    base_output_dir = Path('results/Analysis/Levy_Flight')
    strains = ['N2', 'NL5901', 'UA44', 'TJ356', 'BR5270', 'BR5271', 'ROSELLA']

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

        print("\n--- Lévy Flight Analysis Results ---")
        results_summary = []

        for treatment, df in treatment_data.items():
            print(f"\n--- Analyzing Treatment: {treatment} ---")
            
            all_step_lengths = []
            for _, worm_df in df.groupby('track_id'):
                trajectory = worm_df.sort_values('frame')[['x', 'y']].values
                steps = get_steps_from_trajectory(trajectory)
                if len(steps) > 0:
                    all_step_lengths.extend(steps)

            if len(all_step_lengths) < 50:
                print("  -> Not enough step data to perform analysis.")
                continue

            # Fit power-law distribution
            fit = powerlaw.Fit(all_step_lengths, discrete=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            results_summary.append({
                'treatment': treatment,
                'alpha': alpha,
                'xmin': xmin,
                'loglikelihood_ratio': R,
                'p_value': p
            })

            print(f"  Power-law fit (alpha): {alpha:.3f}")
            print(f"  Log-likelihood ratio vs Lognormal: R={R:.3f}, p-value={p:.3f}")

            # --- Distribution Plot ---
            fig, ax = plt.subplots(figsize=(8, 6))
            fit.plot_pdf(color='b', linewidth=2, ax=ax, label='Empirical Data')
            fit.power_law.plot_pdf(color='r', linestyle='--', ax=ax, label=f'Power-law fit (alpha={alpha:.2f})')
            fit.lognormal.plot_pdf(color='g', linestyle='--', ax=ax, label='Lognormal fit')
            
            ax.set_title(f'Step Length Distribution: {treatment} ({target_strain})')
            ax.set_xlabel('Step Length')
            ax.set_ylabel('Probability Density')
            ax.legend()
            plt.tight_layout()
            plot_path = output_dir / f'levy_fit_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  Distribution plot saved to {plot_path}")

        if not results_summary:
            print(f"No results to save for strain {target_strain}.")
            continue

        # 4. Save summary results
        results_df = pd.DataFrame(results_summary)
        treatment_order = [
            'Control',
            'ETANOL',
            'CBD 0.3uM',
            'CBD 3uM',
            'CBD 30uM',
            'CBDV 0.3uM',
            'CBDV 3uM',
            'CBDV 30uM',
            'Total_Extract_CBD'
        ]
        present_treatments = [t for t in treatment_order if t in results_df['treatment'].unique()]
        results_df['treatment'] = pd.Categorical(results_df['treatment'], categories=present_treatments, ordered=True)
        results_df = results_df.sort_values('treatment')
        
        summary_path = output_dir / 'levy_flight_summary.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"\nSummary results saved to {summary_path}")

    print("\n--------------------------------------")

if __name__ == "__main__":
    main()
