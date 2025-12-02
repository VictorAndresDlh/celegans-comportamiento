import sys
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd


class Tee:
    """Simple tee stream to write simultaneously to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def analyze_tda():
    """
    Comprehensive analysis of TDA (Topological Data Analysis) methodology results showing ALL
    classification attempts across strains and treatments based on trajectory topology.
    
    Updated to reflect improvements based on Thomas et al. (2021):
    - H0 (connected components) and H1 (loops) homology features
    - 5000 coefficients from persistence landscapes (5 landscapes × 1000 resolution)
    - MDS visualization for treatment similarity
    - Distance heatmaps between treatment groups
    """
    base_analysis_dir = Path('results/Analysis/TDA')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive TDA (Topological Data Analysis) Analysis\n")
    print("This analysis shows topological feature-based SVM classification results for ALL strain/treatment combinations.\n")
    print("**Methodology (Based on Thomas et al. 2021):**")
    print("- **Sliding Window Embedding:** Transforms time series trajectories into point clouds")
    print("- **Persistent Homology:** Computes topological features:")
    print("  - H0: Connected components (overall trajectory structure)")
    print("  - H1: Loops/cycles (repetitive movement patterns)")
    print("- **Persistence Landscapes:** 5000 coefficients (5 landscapes × 1000 resolution)")
    print("- **SVM Classification:** Distinguishes treatments based on topological signatures\n")
    print("**TDA Performance Categories:**")
    print("- **Excellent (>80%):** Very strong topological signature, highly distinguishable trajectory shape")
    print("- **Good (65-80%):** Clear topological phenotype, reliably distinguishable global trajectory patterns")
    print("- **Moderate (55-65%):** Detectable topological changes, some classification ability")
    print("- **Poor (<55%):** Weak/no topological phenotype, trajectory shapes similar to control\n")

    all_results = []
    h0_results_available = False
    
    for strain in strains:
        summary_path = base_analysis_dir / strain / 'classification_summary.csv'
        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)
        summary_df['strain'] = strain
        all_results.append(summary_df)
        
        # Check if H0 results are available
        if 'h0_accuracy_mean' in summary_df.columns or 'feature_type' in summary_df.columns:
            h0_results_available = True

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Clean data: remove rows with nan treatments
        clean_df = final_df[final_df['treatment'].notna()].copy()

        # Handle both old (accuracy) and new (accuracy_mean) column names
        if 'accuracy_mean' in clean_df.columns:
            acc_col = 'accuracy_mean'
            acc_std_col = 'accuracy_std'
            has_std = True
        else:
            acc_col = 'accuracy'
            has_std = False

        # Check for H0/H1 separate columns
        has_h0 = 'h0_accuracy_mean' in clean_df.columns
        has_h1 = 'h1_accuracy_mean' in clean_df.columns

        # Add performance categories (TDA typically has lower accuracy than standard ML)
        def categorize_tda_performance(accuracy):
            if accuracy >= 0.8:
                return 'Excellent (>80%)'
            elif accuracy >= 0.65:
                return 'Good (65-80%)'
            elif accuracy >= 0.55:
                return 'Moderate (55-65%)'
            else:
                return 'Poor (<55%)'

        clean_df['performance'] = clean_df[acc_col].apply(categorize_tda_performance)

        # Sort by performance
        clean_df = clean_df.sort_values([acc_col], ascending=False)

        # Display complete results
        display_cols = ['strain', 'treatment', acc_col]
        if has_std:
            display_cols.append(acc_std_col)
        
        # Add H0/H1 columns if available
        if has_h0:
            display_cols.extend(['h0_accuracy_mean'])
        if has_h1:
            display_cols.extend(['h1_accuracy_mean'])
            
        display_cols.extend(['f1_control', 'f1_treatment', 'performance'])
        
        # Filter to existing columns
        display_cols = [col for col in display_cols if col in clean_df.columns]
        display_df = clean_df[display_cols].copy()

        rename_dict = {
            acc_col: 'Combined Acc.' if has_std else 'Accuracy',
            'f1_control': 'F1 (Control)',
            'f1_treatment': 'F1 (Treatment)',
            'performance': 'TDA Performance Category',
            'h0_accuracy_mean': 'H0 Acc.',
            'h1_accuracy_mean': 'H1 Acc.'
        }
        if has_std:
            rename_dict[acc_std_col] = 'Acc. (std)'
        
        rename_dict = {k: v for k, v in rename_dict.items() if k in display_df.columns}
        display_df.rename(columns=rename_dict, inplace=True)

        # Format accuracy columns
        for col in display_df.columns:
            if 'Acc' in col and 'std' not in col.lower():
                display_df[col] = display_df[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        if 'Acc. (std)' in display_df.columns:
            display_df['Acc. (std)'] = display_df['Acc. (std)'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        print("### Complete TDA Classification Results\n")
        print(display_df.to_markdown(index=False))

        # Summary statistics
        total_tests = len(clean_df)
        performance_counts = clean_df['performance'].value_counts()

        print(f"\n### TDA Performance Summary")
        print(f"- **Total strain/treatment combinations:** {total_tests}")
        for category in ['Excellent (>80%)', 'Good (65-80%)', 'Moderate (55-65%)', 'Poor (<55%)']:
            count = performance_counts.get(category, 0)
            percentage = count / total_tests * 100 if total_tests > 0 else 0
            print(f"- **{category}:** {count} cases ({percentage:.1f}%)")

        # Compare with random baseline (50% for binary classification)
        above_random = len(clean_df[clean_df[acc_col] > 0.5])
        print(f"- **Above random chance (>50%):** {above_random}/{total_tests} ({above_random/total_tests*100:.1f}%)")

        # H0 vs H1 Analysis if available
        if has_h0 and has_h1:
            print(f"\n### H0 vs H1 Homology Comparison")
            print("Comparing classification performance using different topological features:\n")
            
            h0_better = len(clean_df[clean_df['h0_accuracy_mean'] > clean_df['h1_accuracy_mean']])
            h1_better = len(clean_df[clean_df['h1_accuracy_mean'] > clean_df['h0_accuracy_mean']])
            equal = total_tests - h0_better - h1_better
            
            print(f"- **H0 (connected components) performs better:** {h0_better} cases ({h0_better/total_tests*100:.1f}%)")
            print(f"- **H1 (loops/cycles) performs better:** {h1_better} cases ({h1_better/total_tests*100:.1f}%)")
            print(f"- **Equal performance:** {equal} cases ({equal/total_tests*100:.1f}%)")
            
            # Best H0 performers
            if h0_better > 0:
                print(f"\n**Best H0 performers (trajectory structure most informative):**")
                h0_best = clean_df.nlargest(3, 'h0_accuracy_mean')[['strain', 'treatment', 'h0_accuracy_mean']]
                for _, row in h0_best.iterrows():
                    print(f"  - {row['strain']} + {row['treatment']}: {row['h0_accuracy_mean']:.1%}")
            
            # Best H1 performers
            if h1_better > 0:
                print(f"\n**Best H1 performers (cyclic patterns most informative):**")
                h1_best = clean_df.nlargest(3, 'h1_accuracy_mean')[['strain', 'treatment', 'h1_accuracy_mean']]
                for _, row in h1_best.iterrows():
                    print(f"  - {row['strain']} + {row['treatment']}: {row['h1_accuracy_mean']:.1%}")

        # Strain-wise performance breakdown
        print(f"\n### TDA Performance by Strain")
        for strain in clean_df['strain'].unique():
            strain_data = clean_df[clean_df['strain'] == strain]
            total_strain_tests = len(strain_data)
            avg_accuracy = strain_data[acc_col].mean()
            best_accuracy = strain_data[acc_col].max()
            best_treatment = strain_data.loc[strain_data[acc_col].idxmax(), 'treatment']

            print(f"\n**{strain}:**")
            print(f"- Tests: {total_strain_tests}")
            print(f"- Average TDA accuracy: {avg_accuracy:.1%}")
            print(f"- Best result: {best_accuracy:.1%} ({best_treatment})")
            print(f"- Above random: {len(strain_data[strain_data[acc_col] > 0.5])}/{total_strain_tests}")

            # H0/H1 comparison for this strain
            if has_h0 and has_h1:
                avg_h0 = strain_data['h0_accuracy_mean'].mean()
                avg_h1 = strain_data['h1_accuracy_mean'].mean()
                print(f"- Average H0 accuracy: {avg_h0:.1%}")
                print(f"- Average H1 accuracy: {avg_h1:.1%}")

            # Performance distribution for this strain
            strain_performance_counts = strain_data['performance'].value_counts()
            for perf_cat, count in strain_performance_counts.items():
                print(f"- {perf_cat}: {count}")

            # Report window lengths used for this strain (if available)
            if 'window_length' in strain_data.columns:
                unique_L = sorted(strain_data['window_length'].dropna().unique())
                if len(unique_L) == 1:
                    print(f"- Window length used (L): {int(unique_L[0])}")
                elif len(unique_L) > 1:
                    counts_L = strain_data['window_length'].value_counts().sort_index()
                    breakdown = ", ".join(
                        f"L={int(L)}: {count} tests"
                        for L, count in counts_L.items()
                    )
                    print(f"- Window lengths used: {breakdown}")

        # Global window-length usage (if available)
        if 'window_length' in clean_df.columns:
            print(f"\n### Window Length Selection Across All Strains")
            L_counts = clean_df['window_length'].value_counts().sort_index()
            for L, count in L_counts.items():
                pct = count / total_tests * 100 if total_tests > 0 else 0
                print(f"- **L={int(L)}:** {count} tests ({pct:.1f}%)")

        # Treatment effectiveness across all strains
        agg_dict = {acc_col: ['count', 'mean', 'max', 'std'], 'strain': lambda x: list(x)}
        if has_h0:
            agg_dict['h0_accuracy_mean'] = 'mean'
        if has_h1:
            agg_dict['h1_accuracy_mean'] = 'mean'
            
        treatment_summary = clean_df.groupby('treatment').agg(agg_dict).round(3)
        
        # Flatten column names
        new_cols = ['Tests', 'Avg Combined', 'Best Combined', 'Std Dev', 'Strains']
        if has_h0:
            new_cols.append('Avg H0')
        if has_h1:
            new_cols.append('Avg H1')
        treatment_summary.columns = new_cols
        
        treatment_summary = treatment_summary.reset_index()
        treatment_summary = treatment_summary.sort_values('Avg Combined', ascending=False)
        treatment_summary['Avg Combined'] = treatment_summary['Avg Combined'].map(lambda x: f"{x:.1%}")
        treatment_summary['Best Combined'] = treatment_summary['Best Combined'].map(lambda x: f"{x:.1%}")
        treatment_summary['Std Dev'] = treatment_summary['Std Dev'].map(lambda x: f"{x:.3f}")
        if 'Avg H0' in treatment_summary.columns:
            treatment_summary['Avg H0'] = treatment_summary['Avg H0'].map(lambda x: f"{x:.1%}")
        if 'Avg H1' in treatment_summary.columns:
            treatment_summary['Avg H1'] = treatment_summary['Avg H1'].map(lambda x: f"{x:.1%}")

        print(f"\n### Treatment Effectiveness (Ranked by Average TDA Performance)\n")
        display_treatment_cols = ['treatment', 'Tests', 'Avg Combined', 'Best Combined', 'Std Dev']
        if 'Avg H0' in treatment_summary.columns:
            display_treatment_cols.append('Avg H0')
        if 'Avg H1' in treatment_summary.columns:
            display_treatment_cols.append('Avg H1')
        print(treatment_summary[display_treatment_cols].to_markdown(index=False))

        # Consistency analysis - treatments that work well across multiple strains
        consistent_treatments = []
        for treatment in clean_df['treatment'].unique():
            treatment_data = clean_df[clean_df['treatment'] == treatment]
            if len(treatment_data) > 1:  # Multiple strains
                avg_acc = treatment_data[acc_col].mean()
                std_acc = treatment_data[acc_col].std()
                min_acc = treatment_data[acc_col].min()
                consistency_score = avg_acc - std_acc  # Penalize high variability

                consistent_treatments.append({
                    'Treatment': treatment,
                    'Strains': len(treatment_data),
                    'Avg Accuracy': f"{avg_acc:.1%}",
                    'Min Accuracy': f"{min_acc:.1%}",
                    'Std Dev': f"{std_acc:.3f}",
                    'Consistency Score': f"{consistency_score:.3f}"
                })

        if consistent_treatments:
            consistency_df = pd.DataFrame(consistent_treatments)
            consistency_df = consistency_df.sort_values('Consistency Score', ascending=False)

            print(f"\n### Treatment Consistency Across Strains (Multi-strain treatments only)\n")
            print(consistency_df.to_markdown(index=False))

        # Topological signature interpretation
        high_performers = clean_df[clean_df[acc_col] > 0.65]
        if not high_performers.empty:
            print(f"\n### Topological Signature Analysis")
            print(f"The following {len(high_performers)} strain/treatment combinations show strong topological signatures (>65% accuracy):")
            print(f"This suggests these treatments induce distinctive changes in the geometric/spatial patterns of worm trajectories.\n")

            for _, row in high_performers.iterrows():
                acc_str = f"{row[acc_col]:.1%}"
                if has_std:
                    acc_str += f" ± {row[acc_std_col]:.3f}"
                
                # Add H0/H1 info if available
                homology_info = ""
                if has_h0 and has_h1:
                    if row['h0_accuracy_mean'] > row['h1_accuracy_mean']:
                        homology_info = " [H0 dominant]"
                    else:
                        homology_info = " [H1 dominant]"
                
                print(f"- **{row['strain']} + {row['treatment']}:** {acc_str}{homology_info} (Strong topological phenotype)")

        else:
            print(f"\n### Topological Signature Analysis")
            print("No treatments achieved >65% TDA classification accuracy.")
            print("This suggests that most treatments do not significantly alter the global geometric patterns of trajectories,")
            print("or that the topological changes are too subtle for current persistent homology features to detect.")

        # Visualizations generated
        print(f"\n### Visualizations Generated\n")
        print("The following visualizations are available in each strain's output directory:")
        print("- `persistence_diagram_*.png`: Persistence diagrams showing H0 and H1 features")
        print("- `mds_plot_H1_L*.png`: MDS visualization of H1 persistence landscapes")
        print("- `mds_plot_H0_L*.png`: MDS visualization of H0 persistence landscapes")
        print("- `distance_heatmap_L*.png`: Heatmap of pairwise distances between treatments")
        print("- `confusion_matrix_*.png`: Classification confusion matrices")

    else:
        print("No TDA analysis results found. Check if the methodology has completed.")


if __name__ == '__main__':
    # Save a copy of the summary to disk while still printing to the terminal
    summaries_dir = Path('results/Analysis/Summaries')
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / 'tda_summary.md'

    with summary_path.open('w', encoding='utf-8') as f, redirect_stdout(Tee(sys.stdout, f)):
        analyze_tda()
