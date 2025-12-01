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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

# Number of permutations for feature-level significance testing
N_PERMUTATIONS = 5000
FDR_ALPHA = 0.10

# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

def extract_rich_features(trajectory):
    """Extract comprehensive kinematic features from centroid trajectory.

    Expanded from 8 to ~20 features following best practices for
    centroid-based behavioral analysis.
    """
    if len(trajectory) < 3:
        return None

    # Basic kinematics
    vectors = np.diff(trajectory, axis=0)
    speeds = np.sqrt(np.sum(vectors**2, axis=1))
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    turning_angles = np.diff(angles)
    turning_angles = (turning_angles + np.pi) % (2 * np.pi) - np.pi

    # Angular acceleration
    angular_acceleration = np.diff(turning_angles) if len(turning_angles) > 1 else np.array([0])

    features = {}

    # Speed features (7 total)
    features['speed_mean'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    features['speed_median'] = np.median(speeds)
    features['speed_min'] = np.min(speeds)
    features['speed_max'] = np.max(speeds)
    features['speed_q25'] = np.percentile(speeds, 25)
    features['speed_q75'] = np.percentile(speeds, 75)

    # Turning features (5 total)
    features['turning_mean'] = np.mean(np.abs(turning_angles))
    features['turning_std'] = np.std(turning_angles)
    features['turning_max'] = np.max(np.abs(turning_angles))
    features['angular_accel_mean'] = np.mean(np.abs(angular_acceleration))
    features['angular_accel_std'] = np.std(angular_acceleration)

    # Path structure features (5 total)
    features['path_length'] = np.sum(speeds)
    features['displacement'] = np.linalg.norm(trajectory[-1] - trajectory[0])
    features['confinement_ratio'] = (
        features['displacement'] / features['path_length']
        if features['path_length'] > 0
        else 0
    )

    # Radius of gyration (measure of spatial spread)
    centroid = np.mean(trajectory, axis=0)
    distances_from_centroid = np.sqrt(np.sum((trajectory - centroid)**2, axis=1))
    features['radius_of_gyration'] = np.sqrt(np.mean(distances_from_centroid**2))

    # Tortuosity (path_length / displacement)
    features['tortuosity'] = (
        features['path_length'] / features['displacement']
        if features['displacement'] > 0
        else 0
    )

    # Temporal features (3 total)
    # Autocorrelation of speed at lag 1
    if len(speeds) > 1:
        speed_centered = speeds - np.mean(speeds)
        autocorr_lag1 = (
            np.correlate(speed_centered[:-1], speed_centered[1:], mode='valid')[0]
            / np.sum(speed_centered**2)
            if np.sum(speed_centered**2) > 0
            else 0
        )
        features['speed_autocorr_lag1'] = autocorr_lag1
    else:
        features['speed_autocorr_lag1'] = 0

    # Fraction of time moving (speed > threshold)
    speed_threshold = 0.01 * np.mean(speeds) if np.mean(speeds) > 0 else 0.01
    features['fraction_moving'] = np.sum(speeds > speed_threshold) / len(speeds)

    # Mean squared displacement (MSD) - indicator of diffusive behavior
    if len(trajectory) >= 10:
        lag = min(10, len(trajectory) // 2)
        msd_values = []
        for i in range(len(trajectory) - lag):
            displacement_lag = np.linalg.norm(trajectory[i + lag] - trajectory[i])
            msd_values.append(displacement_lag**2)
        features['msd_lag10'] = np.mean(msd_values)
    else:
        features['msd_lag10'] = 0

    return features


def _compute_group_difference(X, y):
    """Compute difference of means between treatment (1) and control (0) for each feature."""
    if X.size == 0:
        return np.array([])
    mask_control = y == 0
    mask_treatment = y == 1
    if mask_control.sum() == 0 or mask_treatment.sum() == 0:
        return np.zeros(X.shape[1])
    mean_control = X[mask_control].mean(axis=0)
    mean_treatment = X[mask_treatment].mean(axis=0)
    return mean_treatment - mean_control


def permutation_feature_test_blocked(X, y, experiment_ids, n_permutations=N_PERMUTATIONS, random_state=42):
    """Permutation test per feature with label shuffling within experiments (blocks).

    Args:
        X (ndarray): shape (n_samples, n_features)
        y (ndarray): binary labels (0=Control, 1=Treatment)
        experiment_ids (ndarray): block labels, same length as y
        n_permutations (int): number of permutations
        random_state (int): RNG seed

    Returns:
        p_values (ndarray): permutation p-value per feature
    """
    rng = np.random.default_rng(random_state)

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return np.ones(n_features)

    t_obs = np.abs(_compute_group_difference(X, y))
    counts = np.zeros_like(t_obs, dtype=int)

    # Pre-compute block indices
    blocks = {}
    for idx, exp_id in enumerate(experiment_ids):
        blocks.setdefault(exp_id, []).append(idx)
    block_indices = [np.array(idxs, dtype=int) for idxs in blocks.values()]

    for _ in range(n_permutations):
        y_perm = y.copy()
        for idxs in block_indices:
            if len(idxs) > 1:
                y_perm[idxs] = rng.permutation(y_perm[idxs])
        t_perm = np.abs(_compute_group_difference(X, y_perm))
        counts += (t_perm >= t_obs)

    p_values = (counts + 1) / (n_permutations + 1)
    return p_values


def benjamini_yekutieli(p_values, alpha=FDR_ALPHA):
    """Benjamini–Yekutieli FDR correction.

    Returns adjusted p-values (q-values) for each feature.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return p_values

    order = np.argsort(p_values)
    p_sorted = p_values[order]

    c_m = np.sum(1.0 / np.arange(1, m + 1))
    q_sorted = np.empty(m, dtype=float)

    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = p_sorted[i] * m * c_m / rank
        if q < prev:
            prev = q
        q_sorted[i] = prev

    q_values = np.empty_like(q_sorted)
    q_values[order] = np.minimum(q_sorted, 1.0)
    return q_values

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
                    # Preserve experiment_id for blocked permutation tests
                    features['experiment_id'] = worm_df['experiment_id'].iloc[0]
                    features_by_treatment[treatment].append(features)

        # 3. Perform binary classification
        print("\n--- Control vs. Treatment Binary Classification Results ---")
        results_summary = []
        control_features_list = features_by_treatment.get('Control', [])
        if not control_features_list:
            print(f"No control features found for strain {target_strain}. Skipping.")
            continue
        
        control_df = pd.DataFrame(control_features_list)
        # Separate experiment_id from numeric features
        if 'experiment_id' not in control_df.columns:
            print("  -> 'experiment_id' not found in control features. Skipping strain.")
            continue
        feature_names = [c for c in control_df.columns if c != 'experiment_id']

        for treatment, treatment_features_list in features_by_treatment.items():
            if treatment == 'Control':
                continue

            print(f"\n--- Testing Control vs. {treatment} ---")
            if not treatment_features_list:
                print(f"  -> No features for treatment {treatment}. Skipping.")
                continue
            
            treatment_df = pd.DataFrame(treatment_features_list)
            if 'experiment_id' not in treatment_df.columns:
                print("  -> 'experiment_id' not found in treatment features. Skipping.")
                continue

            X_control_full = control_df[feature_names].values
            X_treatment_full = treatment_df[feature_names].values
            exp_control = control_df['experiment_id'].values
            exp_treatment = treatment_df['experiment_id'].values

            X_full = np.vstack((X_control_full, X_treatment_full))
            y = np.array([0] * len(X_control_full) + [1] * len(X_treatment_full))
            experiment_ids = np.concatenate([exp_control, exp_treatment])

            if len(X_full) == 0 or len(np.unique(y)) < 2:
                print("  -> Not enough data to perform classification.")
                continue

            # 3a. Feature-level permutation tests with Benjamini–Yekutieli correction
            print("  -> Running permutation feature tests (blocked by experiment)...")
            p_perm = permutation_feature_test_blocked(
                X_full,
                y,
                experiment_ids,
                n_permutations=N_PERMUTATIONS,
                random_state=42,
            )
            q_by = benjamini_yekutieli(p_perm, alpha=FDR_ALPHA)
            selected_mask = q_by <= FDR_ALPHA
            n_significant = int(selected_mask.sum())

            # Save per-feature significance table
            significance_df = pd.DataFrame({
                'feature': feature_names,
                'p_permutation': p_perm,
                'q_by': q_by,
                'selected_fdr': selected_mask,
            })
            sig_path = output_dir / f'feature_significance_Control_vs_{treatment.replace(" ", "_")}.csv'
            significance_df.to_csv(sig_path, index=False)
            print(f"  -> Feature significance table saved to {sig_path}")

            if n_significant == 0:
                print("  -> No features passed FDR < 0.10. Using all features for classification (no selection).")
                selected_features = feature_names
            else:
                selected_features = [f for f, use in zip(feature_names, selected_mask) if use]
                print(f"  -> {n_significant} feature(s) selected under FDR < 0.10.")

            X_control = control_df[selected_features].values
            X_treatment = treatment_df[selected_features].values

            X = np.vstack((X_control, X_treatment))

            # k-fold cross-validation
            n_folds = 10
            min_class_size = min(np.sum(y == 0), np.sum(y == 1))
            if min_class_size < n_folds:
                print(f"  -> Not enough samples for {n_folds}-fold CV (min class size: {min_class_size}). Using {min_class_size} folds.")
                n_folds_actual = max(2, min_class_size)
            else:
                n_folds_actual = n_folds

            skf = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=42)
            cv_accuracies = []
            cv_f1_control = []
            cv_f1_treatment = []
            all_importances = []

            all_y_test = []
            all_y_pred = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                )
                clf.fit(X_train_scaled, y_train)

                y_pred = clf.predict(X_test_scaled)
                fold_accuracy = accuracy_score(y_test, y_pred)
                cv_accuracies.append(fold_accuracy)

                # Store for overall report
                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)

                # Per-fold metrics
                try:
                    report_dict = classification_report(
                        y_test,
                        y_pred,
                        target_names=['Control', treatment],
                        output_dict=True,
                        zero_division=0,
                    )
                    cv_f1_control.append(report_dict['Control']['f1-score'])
                    cv_f1_treatment.append(report_dict[treatment]['f1-score'])
                except:
                    cv_f1_control.append(0.0)
                    cv_f1_treatment.append(0.0)

                # Collect feature importances
                all_importances.append(clf.feature_importances_)

            # Aggregate CV results
            mean_accuracy = np.mean(cv_accuracies)
            std_accuracy = np.std(cv_accuracies)
            mean_f1_control = np.mean(cv_f1_control)
            mean_f1_treatment = np.mean(cv_f1_treatment)

            # Overall classification report
            all_y_test = np.array(all_y_test)
            all_y_pred = np.array(all_y_pred)
            overall_report = classification_report(
                all_y_test,
                all_y_pred,
                target_names=['Control', treatment],
                output_dict=True,
                zero_division=0,
            )

            # Average feature importances across folds
            importances = np.mean(all_importances, axis=0)
            importances_std = np.std(all_importances, axis=0)

            results_summary.append({
                'treatment': treatment,
                'accuracy_mean': mean_accuracy,
                'accuracy_std': std_accuracy,
                'precision_control': overall_report['Control']['precision'],
                'recall_control': overall_report['Control']['recall'],
                'f1_control': mean_f1_control,
                'precision_treatment': overall_report[treatment]['precision'],
                'recall_treatment': overall_report[treatment]['recall'],
                'f1_treatment': mean_f1_treatment,
                'n_significant_features_fdr': n_significant,
                'n_folds': n_folds_actual,
            })

            print(f"  Mean Accuracy ({n_folds_actual}-fold CV): {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            print(f"  Mean F1 (Control): {mean_f1_control:.3f}")
            print(f"  Mean F1 ({treatment}): {mean_f1_treatment:.3f}")
            # --- Feature importance plot with error bars (improved styling) ---
            feature_imp_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances,
                'importance_std': importances_std,
            })
            feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)

            # Categorize features by type for color coding
            def categorize_feature(name):
                if any(x in name for x in ['speed', 'velocity']):
                    return 'Speed'
                elif any(x in name for x in ['turning', 'angular']):
                    return 'Turning'
                elif any(x in name for x in ['radius', 'tortuosity', 'displacement', 'path']):
                    return 'Path Structure'
                elif any(x in name for x in ['autocorr', 'moving', 'msd']):
                    return 'Temporal'
                else:
                    return 'Other'

            feature_imp_df['category'] = feature_imp_df['feature'].apply(categorize_feature)
            category_colors = {
                'Speed': PRIMARY_COLORS['blue'],
                'Turning': PRIMARY_COLORS['orange'],
                'Path Structure': PRIMARY_COLORS['green'],
                'Temporal': PRIMARY_COLORS['purple'],
                'Other': PRIMARY_COLORS['gray']
            }
            colors = [category_colors[cat] for cat in feature_imp_df['category']]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'],
                   xerr=feature_imp_df['importance_std'], alpha=0.8, color=colors,
                   edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(feature_imp_df)))
            ax.set_yticklabels(feature_imp_df['feature'])

            setup_plot_style(ax,
                           title=f'Feature Importance: Control vs. {treatment}\n({target_strain}, {n_folds_actual}-fold CV)',
                           xlabel='Importance (mean ± std across folds)',
                           ylabel='')

            ax.invert_yaxis()

            # Add legend for categories (placed outside to avoid overlap)
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat)
                             for cat, color in category_colors.items()
                             if cat in feature_imp_df['category'].values]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                     title='Feature Category', frameon=True, shadow=True)

            plt.tight_layout()
            plot_path = output_dir / f'feature_importance_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Feature importance plot saved to {plot_path}")

            # --- Confusion Matrix (García-Garví et al. 2025, Figure 7) ---
            cm = confusion_matrix(all_y_test, all_y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))

            # Create heatmap with custom colors
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Count', rotation=270, labelpad=20, fontweight='bold')

            # Labels and ticks
            class_names = ['Control', treatment]
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=14, fontweight='bold')

            # Styling
            setup_plot_style(ax,
                           title=f'Confusion Matrix: Control vs. {treatment}\n({target_strain}, Accuracy={mean_accuracy:.1%})',
                           xlabel='Predicted Label',
                           ylabel='True Label')

            # Remove grid for confusion matrix
            ax.grid(False)

            # Add accuracy annotation
            annotation_text = (f'Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.3f}\n'
                             f'F1 (Control): {mean_f1_control:.3f}\n'
                             f'F1 ({treatment}): {mean_f1_treatment:.3f}')
            ax.text(1.45, 0.5, annotation_text,
                   transform=ax.transData, ha='left', va='center',
                   fontsize=9, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='black', alpha=0.9, linewidth=1.5))

            plt.tight_layout()
            cm_path = output_dir / f'confusion_matrix_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Confusion matrix saved to {cm_path}")

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
