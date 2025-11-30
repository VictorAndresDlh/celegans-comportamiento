#!/usr/bin/env python
# coding: utf-8

"""
Methodology 7: Unsupervised Behavioral State Discovery

This script implements an adapted version of the methodology from Jaime et al. (2021).
It uses unsupervised machine learning (PCA and GMM) on a rich set of
trajectory-based features to discover behavioral states and analyze how
treatments affect the distribution of these states.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from utils_data import load_data_for_strain

"""Data loading is centralized in `utils_data.load_data_for_strain`."""

# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

def extract_rich_features(trajectory):
    if len(trajectory) < 3:
        return None

    speeds = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
    vectors = np.diff(trajectory, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    turning_angles = np.diff(angles)
    turning_angles = (turning_angles + np.pi) % (2 * np.pi) - np.pi

    features = {}
    # Speed features
    features['speed_mean'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    features['speed_min'] = np.min(speeds)
    features['speed_25q'] = np.percentile(speeds, 25)
    features['speed_median'] = np.median(speeds)
    features['speed_75q'] = np.percentile(speeds, 75)
    features['speed_max'] = np.max(speeds)

    # Turning features
    features['turning_mean'] = np.mean(np.abs(turning_angles))
    features['turning_std'] = np.std(turning_angles)
    features['turning_max'] = np.max(np.abs(turning_angles))

    # Path features
    features['path_length'] = np.sum(speeds)
    features['displacement'] = np.linalg.norm(trajectory[-1] - trajectory[0])
    features['confinement_ratio'] = features['displacement'] / features['path_length'] if features['path_length'] > 0 else 0

    return features

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    print("--- Starting Unsupervised State Discovery Methodology ---")
    root_data_dir = Path('Datos')
    base_output_dir = Path('results/Analysis/Unsupervised_States')
    strains = ['N2', 'NL5901', 'UA44', 'TJ356', 'BR5270', 'BR5271', 'ROSELLA']

    for target_strain in strains:
        print(f"\n\n===================================================")
        print(f"=== Running Analysis for Strain: {target_strain} ===")
        print(f"===================================================\n")

        output_dir = base_output_dir / target_strain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Data and Extract Features
        treatment_data = load_data_for_strain(root_data_dir, target_strain)
        if not treatment_data:
            print(f"No data found for strain {target_strain}. Skipping.")
            continue

        all_features_list = []
        worm_to_treatment_map = {}
        for treatment, df in treatment_data.items():
            for _, worm_df in df.groupby('track_id'):
                trajectory = worm_df.sort_values('frame')[['x', 'y']].values
                features = extract_rich_features(trajectory)
                if features:
                    all_features_list.append(features)
                    worm_to_treatment_map[len(all_features_list)-1] = treatment

        if not all_features_list:
            print(f"No features extracted for strain {target_strain}. Skipping.")
            continue

        feature_df = pd.DataFrame(all_features_list).fillna(0)
        
        # 2. Unsupervised Learning Pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        print(f"\nPCA resulted in {X_pca.shape[1]} components for strain {target_strain}.")

        n_components = np.arange(2, 11)
        bics = []
        for n in n_components:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(X_pca)
            bics.append(gmm.bic(X_pca))
        
        best_n = n_components[np.argmin(bics)]
        print(f"Best number of states found for strain {target_strain}: {best_n}")

        gmm = GaussianMixture(n_components=best_n, random_state=42)
        states = gmm.fit_predict(X_pca)
        feature_df['state'] = states

        # 3. Analysis and Reporting
        print("\n--- Ethogram: Characterization of Behavioral States ---")
        ethogram_df = feature_df.groupby('state').mean()
        print(ethogram_df.round(2))
        ethogram_path = output_dir / 'ethogram.csv'
        ethogram_df.to_csv(ethogram_path)
        print(f"\nEthogram saved to {ethogram_path}")

        # Create and save ethogram heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(ethogram_df, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f'Behavioral State Ethogram ({target_strain})')
        plt.xlabel('Features')
        plt.ylabel('State')
        plt.tight_layout()
        heatmap_path = output_dir / 'ethogram_heatmap.png'
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Ethogram heatmap saved to {heatmap_path}")

        print("\n--- State Distribution by Treatment ---")
        state_counts_by_treatment = defaultdict(Counter)
        for i, state in enumerate(states):
            treatment = worm_to_treatment_map[i]
            state_counts_by_treatment[treatment][state] += 1

        distribution_summary = []
        for treatment, counts in state_counts_by_treatment.items():
            total_worms = sum(counts.values())
            dist_data = {'treatment': treatment}
            dist_data.update({f'state_{i}': (counts.get(i, 0)/total_worms)*100 for i in range(best_n)})
            distribution_summary.append(dist_data)
            
            print(f"\nTreatment: {treatment} ({total_worms} worms)")
            for state_num in range(best_n):
                percentage = (counts.get(state_num, 0) / total_worms) * 100 if total_worms > 0 else 0
                print(f"  - State {state_num}: {percentage:.1f}%")
        
        distribution_df = pd.DataFrame(distribution_summary).fillna(0)
        dist_path = output_dir / 'state_distribution.csv'
        distribution_df.to_csv(dist_path, index=False)
        print(f"\nState distribution data saved to {dist_path}")

        # Create and save state distribution bar plot
        treatment_order = [
            'Control', 'ETANOL',
            'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'CBDV 3uM', 'CBDV 30uM',
            'Total_Extract_CBD'
        ]
        present_treatments = [t for t in treatment_order if t in distribution_df['treatment'].unique()]
        distribution_df['treatment'] = pd.Categorical(distribution_df['treatment'], categories=present_treatments, ordered=True)
        distribution_df = distribution_df.sort_values('treatment')

        distribution_df.set_index('treatment').plot(kind='bar', stacked=True, figsize=(14, 8), cmap='tab20')
        plt.title(f'State Distribution by Treatment ({target_strain})')
        plt.ylabel('Percentage of Worms')
        plt.xlabel('Treatment')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        barchart_path = output_dir / 'state_distribution_barchart.png'
        plt.savefig(barchart_path)
        plt.close()
        print(f"State distribution bar chart saved to {barchart_path}")

    print("\n-----------------------------------------------------")

if __name__ == "__main__":
    main()
