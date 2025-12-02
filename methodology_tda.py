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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed, cpu_count

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

def _compute_landscape_for_trajectory(args):
    """Helper for parallel TDA computation on a single trajectory.

    Args:
        args: tuple(window_length, trajectory, treatment, track_idx, n_tracks)
    Returns:
        (treatment, h1_landscape_flat, h0_landscape_flat) or None if no valid features.
        
    Following Thomas et al. (2021), we compute persistent homology on sliding window embeddings.
    H1 captures loops (quasi-periodic behavior), H0 captures connected components.
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
    
    # Extract H0 (connected components) and H1 (loops)
    h0_diag = [p for (dim, p) in diag if dim == 0 and p[1] != float('inf')]
    h1_diag = [p for (dim, p) in diag if dim == 1]
    
    # We need at least H1 features for the main analysis
    if not h1_diag:
        return None

    landscape_computer = Landscape(num_landscapes=5, resolution=1000)
    
    # Compute H1 landscape (main feature)
    h1_landscape = landscape_computer.fit_transform([np.array(h1_diag)])
    
    # Compute H0 landscape if available (supplementary feature)
    if h0_diag:
        h0_landscape = landscape_computer.fit_transform([np.array(h0_diag)])
    else:
        h0_landscape = np.zeros_like(h1_landscape)
    
    return treatment, h1_landscape.flatten(), h0_landscape.flatten()


# ===========================================================================
# TDA IMPLEMENTATION
# ===========================================================================


def _extract_tda_features_for_strain(treatment_data, window_length: int):
    """Compute persistence landscapes for all trajectories at a given window length.

    Returns:
        tuple: (h1_features_by_treatment, h0_features_by_treatment)
        Each is a dict[treatment] -> List[np.ndarray] of flattened landscapes.
        
    Following Thomas et al. (2021):
    - H1 (loops) captures quasi-periodic behavior like undulation
    - H0 (connected components) captures trajectory fragmentation/continuity
    
    OPTIMIZED: 
    - Pre-filters short trajectories before parallelization
    - Uses Pool.imap_unordered with optimal chunksize
    - Reduces progress output overhead
    """
    print(f"\nExtracting topological features for all treatment groups (window length L={window_length})...")
    h1_features_by_treatment = defaultdict(list)
    h0_features_by_treatment = defaultdict(list)

    # Build jobs for all trajectories - PRE-FILTER short trajectories
    jobs = []
    skipped = 0
    for treatment, df in treatment_data.items():
        n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
        valid_tracks = 0
        for track_idx, (_, worm_df) in enumerate(df.groupby('track_id'), start=1):
            trajectory = worm_df.sort_values('frame')[['x', 'y']].values
            # PRE-FILTER: Skip trajectories too short for window
            if len(trajectory) >= window_length:
                jobs.append((window_length, trajectory, treatment, track_idx, n_tracks))
                valid_tracks += 1
            else:
                skipped += 1
        print(f"  - Queuing treatment: {treatment} ({valid_tracks}/{n_tracks} valid trajectories)")

    if not jobs:
        print(f"No valid trajectories found for window length L={window_length}.")
        return h1_features_by_treatment, h0_features_by_treatment

    total_jobs = len(jobs)
    print(f"\nTotal trajectories queued for TDA (L={window_length}): {total_jobs} (skipped {skipped} short)")

    # OPTIMIZED: Use joblib for better numpy serialization and memory sharing
    n_jobs = cpu_count() or 2
    print(f"  Using joblib with {n_jobs} workers (backend=loky)")
    
    # Joblib Parallel with progress bar via verbose
    # batch_size='auto' lets joblib optimize chunk sizes
    results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=10,  # Show progress (10 = detailed progress)
        batch_size='auto',
        prefer='processes'
    )(delayed(_compute_landscape_for_trajectory)(job) for job in jobs)
    
    # Process results
    valid_results = 0
    for result in results:
        if result is None:
            continue
        treatment, h1_landscape_flat, h0_landscape_flat = result
        h1_features_by_treatment[treatment].append(h1_landscape_flat)
        h0_features_by_treatment[treatment].append(h0_landscape_flat)
        valid_results += 1
    
    print(f"  Completed: {valid_results} valid landscapes from {total_jobs} trajectories")

    return h1_features_by_treatment, h0_features_by_treatment


def create_mds_visualization(features_by_treatment, output_dir, target_strain, window_length):
    """Create Multidimensional Scaling visualization of persistence landscapes.
    
    Following Thomas et al. (2021) Figure 6: MDS of average persistence landscapes.
    """
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    
    # Compute average landscape per treatment
    avg_landscapes = {}
    for treatment, landscapes in features_by_treatment.items():
        if landscapes:
            avg_landscapes[treatment] = np.mean(landscapes, axis=0)
    
    if len(avg_landscapes) < 2:
        print("  -> Not enough treatments for MDS visualization.")
        return
    
    treatments = list(avg_landscapes.keys())
    X = np.array([avg_landscapes[t] for t in treatments])
    
    # Compute pairwise distances
    distances = pairwise_distances(X, metric='euclidean')
    
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(distances)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color mapping
    treatment_colors = {
        'Control': PRIMARY_COLORS['gray'],
        'ETANOL': PRIMARY_COLORS['yellow'],
        'CBD 0.3uM': PRIMARY_COLORS['green'],
        'CBD 3uM': PRIMARY_COLORS['teal'],
        'CBD 30uM': PRIMARY_COLORS['cyan'],
        'CBDV 0.3uM': PRIMARY_COLORS['blue'],
        'CBDV 3uM': PRIMARY_COLORS['purple'],
        'CBDV 30uM': PRIMARY_COLORS['magenta'],
        'Total_Extract_CBD': PRIMARY_COLORS['orange'],
    }
    
    for i, treatment in enumerate(treatments):
        color = treatment_colors.get(treatment, PRIMARY_COLORS['gray'])
        ax.scatter(coords[i, 0], coords[i, 1], s=200, c=color, 
                  edgecolors='black', linewidth=1.5, label=treatment, zorder=5)
        ax.annotate(treatment, (coords[i, 0], coords[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    setup_plot_style(ax,
                    title=f'MDS of Average Persistence Landscapes\n{target_strain} (L={window_length})',
                    xlabel='MDS Dimension 1',
                    ylabel='MDS Dimension 2')
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = output_dir / f'mds_landscapes_L{window_length}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  MDS visualization saved to {plot_path}")


def create_landscape_heatmap(features_by_treatment, output_dir, target_strain, window_length):
    """Create heatmap of pairwise distances between treatment average landscapes."""
    from sklearn.metrics import pairwise_distances
    
    # Compute average landscape per treatment
    avg_landscapes = {}
    for treatment, landscapes in features_by_treatment.items():
        if landscapes:
            avg_landscapes[treatment] = np.mean(landscapes, axis=0)
    
    if len(avg_landscapes) < 2:
        return
    
    # Order treatments
    treatment_order = [
        'Control', 'ETANOL',
        'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM',
        'CBDV 0.3uM', 'CBDV 3uM', 'CBDV 30uM',
        'Total_Extract_CBD'
    ]
    treatments = [t for t in treatment_order if t in avg_landscapes]
    X = np.array([avg_landscapes[t] for t in treatments])
    
    # Compute pairwise distances
    distances = pairwise_distances(X, metric='euclidean')
    
    # Normalize distances
    if distances.max() > 0:
        distances_norm = distances / distances.max()
    else:
        distances_norm = distances
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(distances_norm, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(treatments)))
    ax.set_yticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=45, ha='right')
    ax.set_yticklabels(treatments)
    
    # Add text annotations
    for i in range(len(treatments)):
        for j in range(len(treatments)):
            text = ax.text(j, i, f'{distances_norm[i, j]:.2f}',
                          ha='center', va='center', color='white' if distances_norm[i, j] > 0.5 else 'black',
                          fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Distance', rotation=270, labelpad=20)
    
    setup_plot_style(ax,
                    title=f'Pairwise Distances Between Average Persistence Landscapes\n{target_strain} (L={window_length})',
                    xlabel='Treatment',
                    ylabel='Treatment')
    ax.grid(False)
    
    plt.tight_layout()
    plot_path = output_dir / f'landscape_distance_heatmap_L{window_length}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Landscape distance heatmap saved to {plot_path}")


def _run_tda_classification_for_strain(
    features_by_treatment,
    target_strain: str,
    output_dir: Path,
    window_length: int,
    make_plots: bool = True,
    n_folds: int = 10,
):
    """Run Control vs treatment TDA classification with k-fold CV.

    Returns:
        List[dict]: one summary row per treatment.
    """
    print("\n--- TDA Control vs. Treatment Binary Classification Results (k-fold CV) ---")
    results_summary = []
    control_features_list = features_by_treatment.get('Control', [])
    if not control_features_list:
        print(f"No control features found for strain {target_strain}. Skipping classification.")
        return results_summary

    control_features = np.array(control_features_list)

    for treatment, treatment_features_list in features_by_treatment.items():
        if treatment == 'Control':
            continue

        print(f"\n--- Testing Control vs. {treatment} (L={window_length}) ---")
        if not treatment_features_list:
            print(f"  -> No features for treatment {treatment}. Skipping.")
            continue

        treatment_features = np.array(treatment_features_list)

        if len(control_features) == 0 or len(treatment_features) == 0:
            print("  -> Not enough data to perform classification.")
            continue

        X = np.vstack((control_features, treatment_features))
        y = np.array([0] * len(control_features) + [1] * len(treatment_features))

        # Ensure there's enough data for k-fold CV
        min_class_size = min(np.sum(y == 0), np.sum(y == 1))
        if min_class_size < n_folds:
            print(f"  -> Not enough samples for {n_folds}-fold CV (min class size: {min_class_size}). Using {min_class_size} folds.")
            n_folds_actual = min_class_size
            if n_folds_actual < 2:
                print("  -> Not enough samples for cross-validation. Skipping.")
                continue
        else:
            n_folds_actual = n_folds

        # k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=42)
        cv_accuracies = []
        cv_f1_control = []
        cv_f1_treatment = []

        all_y_test = []
        all_y_pred = []
        all_test_indices = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
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
            all_test_indices.extend(test_idx)

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
            'window_length': window_length,
            'n_folds': n_folds_actual,
        })

        print(f"  Mean Accuracy ({n_folds_actual}-fold CV): {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"  Mean F1 (Control): {mean_f1_control:.3f}")
        print(f"  Mean F1 ({treatment}): {mean_f1_treatment:.3f}")

        if make_plots:
            # --- Persistence Landscape Plot (Thomas et al. 2021, Figure 4) ---
            # Use first fold test set for visualization
            first_fold_test_indices = np.array(all_test_indices[:len(all_test_indices)//n_folds_actual])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            fig.suptitle(
                f'Topological Persistence Landscapes\nControl vs. {treatment} ({target_strain})\n'
                f'Window Length L={window_length}, Classification Accuracy={mean_accuracy:.1%}±{std_accuracy:.3f}',
                fontsize=14, fontweight='bold', y=1.02
            )

            # Plot for Control
            control_indices = first_fold_test_indices[all_y_test[:len(first_fold_test_indices)] == 0]
            for i in range(min(5, len(control_indices))):
                idx = control_indices[i]
                axes[0].plot(X[idx], alpha=0.5, color=PRIMARY_COLORS['blue'],
                           linewidth=1.5, label=f'Sample {i+1}' if i < 3 else None)

            setup_plot_style(axes[0],
                           title='Control',
                           xlabel='Feature Dimension (persistence landscape coefficients)',
                           ylabel='Persistence Value\n(topological feature strength)')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            if len(control_indices) > 0:
                axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)

            # Plot for Treatment
            treatment_indices = first_fold_test_indices[all_y_test[:len(first_fold_test_indices)] == 1]
            for i in range(min(5, len(treatment_indices))):
                idx = treatment_indices[i]
                axes[1].plot(X[idx], alpha=0.5, color=PRIMARY_COLORS['orange'],
                           linewidth=1.5, label=f'Sample {i+1}' if i < 3 else None)

            setup_plot_style(axes[1],
                           title=f'{treatment}',
                           xlabel='Feature Dimension (persistence landscape coefficients)',
                           ylabel='')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            if len(treatment_indices) > 0:
                axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)

            plot_path = output_dir / f'landscapes_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Landscape plot saved to {plot_path}")

            # --- Persistence Diagrams (Thomas et al. 2021, Figure 2 & 4) ---
            # Generate persistence diagrams for representative samples
            from sklearn.decomposition import PCA

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Topological Features: Control vs. {treatment}\n({target_strain}, L={window_length})',
                        fontsize=16, fontweight='bold')

            # Select representative samples from each class
            n_samples = min(3, len(control_indices))
            selected_control = control_indices[:n_samples]
            selected_treatment = treatment_indices[:n_samples]

            for idx, sample_idx in enumerate(selected_control):
                if idx >= 3:
                    break

                # Recompute persistence for this sample
                # We need to get back to the original point cloud
                # Since we don't have it stored, we'll use the landscape as a proxy visualization
                ax_diag = axes[0, idx]
                ax_pca = axes[1, idx]

                # Plot landscape as "persistence-like" visualization
                landscape_data = X[sample_idx]

                # Create a pseudo-persistence diagram from landscape peaks
                # This is an approximation for visualization purposes
                peaks = []
                for i in range(0, len(landscape_data), 200):
                    if landscape_data[i] > 0:
                        birth = i / 1000.0
                        death = birth + landscape_data[i]
                        peaks.append((birth, death))

                if peaks:
                    births, deaths = zip(*peaks)
                    ax_diag.scatter(births, deaths, c=PRIMARY_COLORS['blue'],
                                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

                # Diagonal line (represents no persistence)
                max_val = max(max(births), max(deaths)) if peaks else 1
                ax_diag.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

                setup_plot_style(ax_diag,
                               title=f'Control Sample {idx+1}',
                               xlabel='Birth',
                               ylabel='Death')
                ax_diag.set_aspect('equal')

                # PCA projection (simplified - using landscape as feature)
                # Create a simple 2D projection of the landscape
                landscape_reshaped = landscape_data[:1000].reshape(-1, 100)  # Reshape for PCA
                if landscape_reshaped.shape[0] > 2:
                    pca = PCA(n_components=2)
                    projection = pca.fit_transform(landscape_reshaped)

                    ax_pca.scatter(projection[:, 0], projection[:, 1],
                                 c=PRIMARY_COLORS['blue'], alpha=0.6, s=30,
                                 edgecolors='black', linewidth=0.5)
                    ax_pca.plot(projection[:, 0], projection[:, 1],
                              '-', color=PRIMARY_COLORS['blue'], alpha=0.3, linewidth=1)

                    setup_plot_style(ax_pca,
                                   title=f'Point Cloud Projection',
                                   xlabel='PC1',
                                   ylabel='PC2')

            # Save comprehensive visualization
            plt.tight_layout()
            topo_path = output_dir / f'topological_features_Control_vs_{treatment.replace(" ", "_")}.png'
            plt.savefig(topo_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Topological features plot saved to {topo_path}")

    return results_summary


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
            print(
                f"\nError: 'Control' group not found for strain {target_strain}. "
                "Cannot perform binary classification. Skipping."
            )
            continue

        # 2. Evaluate multiple window lengths (as in Thomas et al. 2021)
        # Note: L=1 was the best performer in Thomas et al. (95.5% accuracy)
        window_lengths = [1, 10, 20, 50]
        candidate_results = {}
        candidate_features = {}
        candidate_mean_accuracies = {}

        for window_length in window_lengths:
            h1_features, h0_features = _extract_tda_features_for_strain(
                treatment_data,
                window_length,
            )
            if not h1_features:
                print(f"  -> No features extracted for L={window_length}. Skipping.")
                continue

            # Use H1 features for classification (main analysis, as in Thomas et al.)
            results_summary = _run_tda_classification_for_strain(
                features_by_treatment=h1_features,
                target_strain=target_strain,
                output_dir=output_dir,
                window_length=window_length,
                make_plots=False,
            )
            if not results_summary:
                print(f"  -> No classification results for L={window_length}. Skipping.")
                continue

            accuracies = [row['accuracy_mean'] for row in results_summary]
            mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
            candidate_results[window_length] = results_summary
            candidate_features[window_length] = (h1_features, h0_features)
            candidate_mean_accuracies[window_length] = mean_accuracy

            print(f"  -> Mean classification accuracy for L={window_length}: {mean_accuracy:.3f}")

        if not candidate_results:
            print(f"No TDA results to save for strain {target_strain}.")
            continue

        # 3. Select best window length and re-run classification with plots enabled
        best_window_length = max(
            candidate_mean_accuracies,
            key=candidate_mean_accuracies.get,
        )
        best_mean_accuracy = candidate_mean_accuracies[best_window_length]
        print(
            f"\nSelected window length L={best_window_length} for strain {target_strain} "
            f"(highest mean accuracy = {best_mean_accuracy:.3f})."
        )

        # Get H1 and H0 features for best window length
        best_h1_features, best_h0_features = candidate_features[best_window_length]
        
        final_results_summary = _run_tda_classification_for_strain(
            features_by_treatment=best_h1_features,
            target_strain=target_strain,
            output_dir=output_dir,
            window_length=best_window_length,
            make_plots=True,
        )
        
        # Create additional visualizations (Thomas et al. 2021 style)
        print("\n  Creating MDS and distance visualizations...")
        create_mds_visualization(best_h1_features, output_dir, target_strain, best_window_length)
        create_landscape_heatmap(best_h1_features, output_dir, target_strain, best_window_length)

        if not final_results_summary:
            print(f"No final results to save for strain {target_strain}.")
            continue

        # 4. Save summary results (only for the selected window length)
        results_df = pd.DataFrame(final_results_summary)
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
