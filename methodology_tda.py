#!/usr/bin/env python
# coding: utf-8

"""
Methodology 1: Topological Data Analysis (TDA) for C. elegans Behavior

This script implements TDA to distinguish treatment effects on the N2 strain.
It uses persistent homology on trajectory data to generate topological features
and classifies treatments using an SVM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns

import gudhi
from gudhi.representations import Landscape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils_data import load_data_for_strain

# ===========================================================================
# DATA LOADING AND PROCESSING
# ===========================================================================

"""Data loading is centralized in `utils_data.load_data_for_strain`."""

def _compute_landscape_for_trajectory(args):
    """Helper for parallel TDA computation on a single trajectory.

    Args:
        args: tuple(window_length, trajectory, treatment, track_idx, n_tracks)
    Returns:
        (treatment, landscape_flat) or None if no valid H1 features.
    """
    window_length, trajectory, treatment, track_idx, n_tracks = args

    if len(trajectory) < window_length:
        return None

    trajectory = trajectory - np.mean(trajectory, axis=0)
    points = np.array([
        trajectory[j:j + window_length].flatten()
        for j in range(len(trajectory) - window_length + 1)
    ])
    if points.shape[0] == 0:
        return None

    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=np.inf)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    h1_diag = [p for (dim, p) in diag if dim == 1]
    if not h1_diag:
        return None

    landscape_computer = Landscape(num_landscapes=5, resolution=100)
    landscape = landscape_computer.fit_transform([np.array(h1_diag)])
    return treatment, landscape.flatten()


# ===========================================================================
# TDA IMPLEMENTATION
# ===========================================================================

def main():
    """Main execution function."""
    print("--- Starting TDA Methodology Script (Binary Classification) ---")
    root_data_dir = Path('Datos')
    base_output_dir = Path('results/Analysis/TDA')
    strains = ['N2', 'NL5901', 'UA44', 'TJ356', 'BR5270', 'BR5271', 'ROSELLA']

    for target_strain in strains:
        print(f"\n\n===================================================")
        print(f"=== Running Analysis for Strain: {target_strain} ===")
        print(f"===================================================\n")

        output_dir = base_output_dir / target_strain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Data
        treatment_data = load_data_for_strain(root_data_dir, target_strain)
        
        if 'Control' not in treatment_data:
            print(f"\nError: 'Control' group not found for strain {target_strain}. Cannot perform binary classification. Skipping.")
            continue

        # 2. Extract features for all treatments (with multiprocessing)
        print("\nExtracting topological features for all treatment groups...")
        features_by_treatment = defaultdict(list)
        window_length = 20

        # Build jobs for all trajectories
        jobs = []
        for treatment, df in treatment_data.items():
            n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
            print(f"  - Queuing treatment: {treatment} ({n_tracks} trajectories)")
            for track_idx, (_, worm_df) in enumerate(df.groupby('track_id'), start=1):
                trajectory = worm_df.sort_values('frame')[['x', 'y']].values
                jobs.append((window_length, trajectory, treatment, track_idx, n_tracks))

        if not jobs:
            print("No trajectories found for strain {target_strain}. Skipping.")
            continue

        total_jobs = len(jobs)
        print(f"\nTotal trajectories queued for TDA: {total_jobs}")

        processed = 0
        # Use a reasonable number of workers (up to CPU count)
        max_workers = os.cpu_count() or 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_compute_landscape_for_trajectory, job): job for job in jobs}
            for future in as_completed(future_to_job):
                result = future.result()
                processed += 1
                if processed % 10 == 0 or processed == total_jobs:
                    print(f"    · Global trajectories processed: {processed}/{total_jobs}")
                if result is None:
                    continue
                treatment, landscape_flat = result
                features_by_treatment[treatment].append(landscape_flat)

        # 3. Perform binary classification and save results/plots
        print("\n--- TDA Control vs. Treatment Binary Classification Results ---")
        results_summary = []
        control_features_list = features_by_treatment.get('Control', [])
        if not control_features_list:
            print(f"No control features found for strain {target_strain}. Skipping.")
            continue
        
        control_features = np.array(control_features_list)

        for treatment, treatment_features_list in features_by_treatment.items():
            if treatment == 'Control':
                continue

            print(f"\n--- Testing Control vs. {treatment} ---")
            if not treatment_features_list:
                print(f"  -> No features for treatment {treatment}. Skipping.")
                continue
            
            treatment_features = np.array(treatment_features_list)

            if len(control_features) == 0 or len(treatment_features) == 0:
                print("  -> Not enough data to perform classification.")
                continue

            X = np.vstack((control_features, treatment_features))
            y = np.array([0] * len(control_features) + [1] * len(treatment_features))

            # Ensure there's enough data for a split
            if np.sum(y==0) < 2 or np.sum(y==1) < 2:
                print("  -> Not enough samples in one of the classes for a stratified split.")
                continue

            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, target_names=['Control', treatment], output_dict=True, zero_division=0)

            results_summary.append({
                'treatment': treatment,
                'accuracy': accuracy,
                'precision_control': report_dict['Control']['precision'],
                'recall_control': report_dict['Control']['recall'],
                'f1_control': report_dict['Control']['f1-score'],
                'precision_treatment': report_dict[treatment]['precision'],
                'recall_treatment': report_dict[treatment]['recall'],
                'f1_treatment': report_dict[treatment]['f1-score'],
            })
            
            print(f"  Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred, target_names=['Control', treatment], zero_division=0))

            # --- Persistence Landscape Plot ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            fig.suptitle(f'Persistence Landscapes: Control vs. {treatment} ({target_strain})')
            
            # Plot for Control
            control_indices_in_test = idx_test[y_test == 0]
            for i in range(min(5, len(control_indices_in_test))):
                idx = control_indices_in_test[i]
                axes[0].plot(X[idx])
            axes[0].set_title('Control Landscapes')

            # Plot for Treatment
            treatment_indices_in_test = idx_test[y_test == 1]
            for i in range(min(5, len(treatment_indices_in_test))):
                idx = treatment_indices_in_test[i]
                axes[1].plot(X[idx])
            axes[1].set_title(f'{treatment} Landscapes')

            plot_path = output_dir / f'landscapes_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  Landscape plot saved to {plot_path}")

        if not results_summary:
            print(f"No results to save for strain {target_strain}.")
            continue

        # 4. Save summary results
        results_df = pd.DataFrame(results_summary)
        treatment_order = [
            'Control', 'ETANOL',
            'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'CBDV 3uM', 'CBDV 30uM',
            'Total_Extract_CBD'
        ]
        present_treatments = [t for t in treatment_order if t in results_df['treatment'].unique()]
        results_df['treatment'] = pd.Categorical(results_df['treatment'], categories=present_treatments, ordered=True)
        results_df = results_df.sort_values('treatment')
        
        summary_path = output_dir / 'classification_summary.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"\nSummary results saved to {summary_path}")

    print("\n-----------------------------------------------------------")

if __name__ == "__main__":
    # This script requires gudhi and scikit-learn.
    # You can install them using:
    # uv add gudhi scikit-learn
    main()
