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
    """
    base_analysis_dir = Path('results/Analysis/TDA')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive TDA (Topological Data Analysis) Analysis\n")
    print("This analysis shows topological feature-based SVM classification results for ALL strain/treatment combinations.\n")
    print("**TDA Performance Categories:**")
    print("- **Excellent (>80%):** Very strong topological signature, highly distinguishable trajectory shape")
    print("- **Good (65-80%):** Clear topological phenotype, reliably distinguishable global trajectory patterns")
    print("- **Moderate (55-65%):** Detectable topological changes, some classification ability")
    print("- **Poor (<55%):** Weak/no topological phenotype, trajectory shapes similar to control\n")
    print("**Note:** TDA captures the 'shape' of behavioral trajectories through persistent homology, detecting global geometric patterns that other methods might miss.\n")

    all_results = []
    for strain in strains:
        summary_path = base_analysis_dir / strain / 'classification_summary.csv'
        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)
        summary_df['strain'] = strain
        all_results.append(summary_df)

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
        display_cols.extend(['f1_control', 'f1_treatment', 'performance'])
        display_df = clean_df[display_cols].copy()

        rename_dict = {
            acc_col: 'Accuracy (mean)' if has_std else 'Accuracy',
            'f1_control': 'F1 (Control)',
            'f1_treatment': 'F1 (Treatment)',
            'performance': 'TDA Performance Category'
        }
        if has_std:
            rename_dict[acc_std_col] = 'Accuracy (std)'
        display_df.rename(columns=rename_dict, inplace=True)

        display_df['Accuracy (mean)' if has_std else 'Accuracy'] = display_df['Accuracy (mean)' if has_std else 'Accuracy'].map(lambda x: f"{x:.1%}")
        if has_std:
            display_df['Accuracy (std)'] = display_df['Accuracy (std)'].map(lambda x: f"{x:.3f}")

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

            # Performance distribution for this strain
            strain_performance_counts = strain_data['performance'].value_counts()
            for perf_cat, count in strain_performance_counts.items():
                print(f"- {perf_cat}: {count}")

            # Report window lengths used for this strain (if available)
            if 'window_length' in strain_data.columns:
                unique_L = sorted(strain_data['window_length'].dropna().unique())
                if len(unique_L) == 1:
                    print(f"- Window length used (L): {unique_L[0]}")
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
        treatment_summary = clean_df.groupby('treatment').agg({
            acc_col: ['count', 'mean', 'max', 'std'],
            'strain': lambda x: list(x)
        }).round(3)
        treatment_summary.columns = ['Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev', 'Strains']
        treatment_summary = treatment_summary.reset_index()
        treatment_summary = treatment_summary.sort_values('Avg Accuracy', ascending=False)
        treatment_summary['Avg Accuracy'] = treatment_summary['Avg Accuracy'].map(lambda x: f"{x:.1%}")
        treatment_summary['Best Accuracy'] = treatment_summary['Best Accuracy'].map(lambda x: f"{x:.1%}")
        treatment_summary['Std Dev'] = treatment_summary['Std Dev'].map(lambda x: f"{x:.3f}")

        print(f"\n### Treatment Effectiveness (Ranked by Average TDA Performance)\n")
        print(treatment_summary[['treatment', 'Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev']].to_markdown(index=False))

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
                print(f"- **{row['strain']} + {row['treatment']}:** {acc_str} (Strong topological phenotype)")

        else:
            print(f"\n### Topological Signature Analysis")
            print("No treatments achieved >65% TDA classification accuracy.")
            print("This suggests that most treatments do not significantly alter the global geometric patterns of trajectories,")
            print("or that the topological changes are too subtle for current persistent homology features to detect.")

    else:
        print("No TDA analysis results found. Check if the methodology has completed.")

if __name__ == '__main__':
    # Save a copy of the summary to disk while still printing to the terminal
    summaries_dir = Path('results/Analysis/Summaries')
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / 'tda_summary.md'

    with summary_path.open('w', encoding='utf-8') as f, redirect_stdout(Tee(sys.stdout, f)):
        analyze_tda()