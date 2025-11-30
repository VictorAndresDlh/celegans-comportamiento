#!/usr/bin/env python
# coding: utf-8

"""
Methodology 3: Machine Learning for High-Throughput Screening

This script implements a machine learning pipeline to classify C. elegans
behavior based on trajectory features. It distinguishes treatment effects
on the N2 strain by training a Random Forest classifier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from utils_data import load_data_for_strain

# ===========================================================================
# DATA LOADING AND PROCESSING
# ===========================================================================

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
    features['speed_mean'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    features['speed_median'] = np.median(speeds)
    features['turning_mean'] = np.mean(np.abs(turning_angles))
    features['turning_std'] = np.std(turning_angles)
    features['path_length'] = np.sum(speeds)
    features['displacement'] = np.linalg.norm(trajectory[-1] - trajectory[0])
    features['confinement_ratio'] = features['displacement'] / features['path_length'] if features['path_length'] > 0 else 0

    return features

def main():
    """Main execution function."""
    print("--- Starting ML Screening Methodology Script (Binary Classification) ---")
    root_data_dir = Path('Datos')
    base_output_dir = Path('results/Analysis/ML_Screening')
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

        # 2. Extract features for all treatments
        print("\nExtracting features for all treatment groups...")
        features_by_treatment = defaultdict(list)
        for treatment, df in treatment_data.items():
            print(f"  - Extracting features for {treatment}...")
            for _, worm_df in df.groupby('track_id'):
                trajectory = worm_df.sort_values('frame')[['x', 'y']].values
                features = extract_rich_features(trajectory)
                if features:
                    features_by_treatment[treatment].append(features)

        # 3. Perform binary classification
        print("\n--- Control vs. Treatment Binary Classification Results ---")
        results_summary = []
        control_features_list = features_by_treatment.get('Control', [])
        if not control_features_list:
            print(f"No control features found for strain {target_strain}. Skipping.")
            continue
        
        control_df = pd.DataFrame(control_features_list)
        feature_names = list(control_df.columns)

        for treatment, treatment_features_list in features_by_treatment.items():
            if treatment == 'Control':
                continue

            print(f"\n--- Testing Control vs. {treatment} ---")
            if not treatment_features_list:
                print(f"  -> No features for treatment {treatment}. Skipping.")
                continue
            
            treatment_df = pd.DataFrame(treatment_features_list)
            
            X_control = control_df.values
            X_treatment = treatment_df.values
            
            X = np.vstack((X_control, X_treatment))
            y = np.array([0] * len(X_control) + [1] * len(X_treatment))

            if len(X) == 0 or len(np.unique(y)) < 2:
                print("  -> Not enough data to perform classification.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Control', treatment], output_dict=True, zero_division=0)

            results_summary.append({
                'treatment': treatment,
                'accuracy': accuracy,
                'precision_control': report['Control']['precision'],
                'recall_control': report['Control']['recall'],
                'f1_control': report['Control']['f1-score'],
                'precision_treatment': report[treatment]['precision'],
                'recall_treatment': report[treatment]['recall'],
                'f1_treatment': report[treatment]['f1-score'],
            })

            print(f"  Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred, target_names=['Control', treatment], zero_division=0))

            importances = clf.feature_importances_
            feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_imp_df, palette='viridis')
            plt.title(f'Feature Importance: Control vs. {treatment} ({target_strain})')
            plt.tight_layout()
            plot_path = output_dir / f'feature_importance_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  Feature importance plot saved to {plot_path}")

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

    print("\n---------------------------------------------------------")
if __name__ == "__main__":
    main()
