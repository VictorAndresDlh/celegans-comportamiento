import pandas as pd
from pathlib import Path

def analyze_tda():
    """
    Comprehensive analysis of TDA (Topological Data Analysis) methodology results showing ALL
    classification attempts across strains and treatments based on trajectory topology.
    """
    base_analysis_dir = Path('results/Analysis/TDA')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive TDA (Topological Data Analysis) Analysis\n")
    print("This analysis shows topological feature-based Random Forest classification results for ALL strain/treatment combinations.\n")
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

        clean_df['performance'] = clean_df['accuracy'].apply(categorize_tda_performance)

        # Sort by performance
        clean_df = clean_df.sort_values(['accuracy'], ascending=False)

        # Display complete results
        display_df = clean_df[['strain', 'treatment', 'accuracy', 'f1_control', 'f1_treatment', 'performance']].copy()
        display_df.rename(columns={
            'accuracy': 'Accuracy',
            'f1_control': 'F1 (Control)',
            'f1_treatment': 'F1 (Treatment)',
            'performance': 'TDA Performance Category'
        }, inplace=True)
        display_df['Accuracy'] = display_df['Accuracy'].map(lambda x: f"{x:.1%}")

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
        above_random = len(clean_df[clean_df['accuracy'] > 0.5])
        print(f"- **Above random chance (>50%):** {above_random}/{total_tests} ({above_random/total_tests*100:.1f}%)")

        # Strain-wise performance breakdown
        print(f"\n### TDA Performance by Strain")
        for strain in clean_df['strain'].unique():
            strain_data = clean_df[clean_df['strain'] == strain]
            total_strain_tests = len(strain_data)
            avg_accuracy = strain_data['accuracy'].mean()
            best_accuracy = strain_data['accuracy'].max()
            best_treatment = strain_data.loc[strain_data['accuracy'].idxmax(), 'treatment']

            print(f"\n**{strain}:**")
            print(f"- Tests: {total_strain_tests}")
            print(f"- Average TDA accuracy: {avg_accuracy:.1%}")
            print(f"- Best result: {best_accuracy:.1%} ({best_treatment})")
            print(f"- Above random: {len(strain_data[strain_data['accuracy'] > 0.5])}/{total_strain_tests}")

            # Performance distribution for this strain
            strain_performance_counts = strain_data['performance'].value_counts()
            for perf_cat, count in strain_performance_counts.items():
                print(f"- {perf_cat}: {count}")

        # Treatment effectiveness across all strains
        treatment_summary = clean_df.groupby('treatment').agg({
            'accuracy': ['count', 'mean', 'max', 'std'],
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
                avg_acc = treatment_data['accuracy'].mean()
                std_acc = treatment_data['accuracy'].std()
                min_acc = treatment_data['accuracy'].min()
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
        high_performers = clean_df[clean_df['accuracy'] > 0.65]
        if not high_performers.empty:
            print(f"\n### Topological Signature Analysis")
            print(f"The following {len(high_performers)} strain/treatment combinations show strong topological signatures (>65% accuracy):")
            print(f"This suggests these treatments induce distinctive changes in the geometric/spatial patterns of worm trajectories.\n")

            for _, row in high_performers.iterrows():
                print(f"- **{row['strain']} + {row['treatment']}:** {row['accuracy']:.1%} (Strong topological phenotype)")

        else:
            print(f"\n### Topological Signature Analysis")
            print("No treatments achieved >65% TDA classification accuracy.")
            print("This suggests that most treatments do not significantly alter the global geometric patterns of trajectories,")
            print("or that the topological changes are too subtle for current persistent homology features to detect.")

    else:
        print("No TDA analysis results found. Check if the methodology has completed.")

if __name__ == '__main__':
    analyze_tda()