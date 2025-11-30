import pandas as pd
from pathlib import Path

def analyze_ml_screening():
    """
    Comprehensive analysis of ML Screening methodology results showing ALL classification attempts
    across strains and treatments, correctly interpreting classification_summary.csv files.
    """
    base_analysis_dir = Path('results/Analysis/ML_Screening')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive ML Screening Analysis\n")
    print("This analysis shows Random Forest classification results for ALL strain/treatment combinations.\n")
    print("**Performance Categories:**")
    print("- **Excellent (>90%):** Very strong behavioral phenotype, highly distinguishable")
    print("- **Good (75-90%):** Clear behavioral phenotype, reliably distinguishable")
    print("- **Moderate (60-75%):** Detectable phenotype, some classification ability")
    print("- **Poor (<60%):** Weak/no phenotype, difficult to distinguish from control\n")

    all_results = []
    for strain in strains:
        summary_path = base_analysis_dir / strain / 'classification_summary.csv'
        if not summary_path.exists():
            print(f"Missing ML screening data for strain {strain}")
            continue

        summary_df = pd.read_csv(summary_path)
        summary_df['strain'] = strain
        all_results.append(summary_df)

    if not all_results:
        print("No ML Screening results found. Check if the methodology has completed.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # Add performance categories
    def categorize_performance(accuracy):
        if accuracy >= 0.9:
            return 'Excellent (>90%)'
        elif accuracy >= 0.75:
            return 'Good (75-90%)'
        elif accuracy >= 0.6:
            return 'Moderate (60-75%)'
        else:
            return 'Poor (<60%)'

    final_df['performance'] = final_df['accuracy'].apply(categorize_performance)

    # Sort by performance
    final_df = final_df.sort_values(['accuracy'], ascending=False)

    # Display complete results
    display_df = final_df[['strain', 'treatment', 'accuracy', 'f1_control', 'f1_treatment', 'performance']].copy()
    display_df.rename(columns={
        'accuracy': 'Accuracy',
        'f1_control': 'F1 (Control)',
        'f1_treatment': 'F1 (Treatment)',
        'performance': 'Performance Category'
    }, inplace=True)
    display_df['Accuracy'] = display_df['Accuracy'].map(lambda x: f"{x:.1%}")

    print("### Complete Classification Results\n")
    print(display_df.to_markdown(index=False))

    # Summary statistics
    total_tests = len(final_df)
    performance_counts = final_df['performance'].value_counts()

    print(f"\n### Performance Summary")
    print(f"- **Total strain/treatment combinations:** {total_tests}")
    for category in ['Excellent (>90%)', 'Good (75-90%)', 'Moderate (60-75%)', 'Poor (<60%)']:
        count = performance_counts.get(category, 0)
        percentage = count / total_tests * 100 if total_tests > 0 else 0
        print(f"- **{category}:** {count} cases ({percentage:.1f}%)")

    # Strain-wise performance breakdown
    print(f"\n### Performance by Strain")
    for strain in final_df['strain'].unique():
        strain_data = final_df[final_df['strain'] == strain]
        total_strain_tests = len(strain_data)
        avg_accuracy = strain_data['accuracy'].mean()
        best_accuracy = strain_data['accuracy'].max()
        best_treatment = strain_data.loc[strain_data['accuracy'].idxmax(), 'treatment']

        print(f"\n**{strain}:**")
        print(f"- Tests: {total_strain_tests}")
        print(f"- Average accuracy: {avg_accuracy:.1%}")
        print(f"- Best result: {best_accuracy:.1%} ({best_treatment})")

        # Performance distribution for this strain
        strain_performance_counts = strain_data['performance'].value_counts()
        for perf_cat, count in strain_performance_counts.items():
            print(f"- {perf_cat}: {count}")

    # Treatment effectiveness across all strains
    treatment_summary = final_df.groupby('treatment').agg({
        'accuracy': ['count', 'mean', 'max', 'std'],
        'strain': lambda x: list(x)
    }).round(3)
    treatment_summary.columns = ['Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev', 'Strains']
    treatment_summary = treatment_summary.reset_index()
    treatment_summary = treatment_summary.sort_values('Avg Accuracy', ascending=False)
    treatment_summary['Avg Accuracy'] = treatment_summary['Avg Accuracy'].map(lambda x: f"{x:.1%}")
    treatment_summary['Best Accuracy'] = treatment_summary['Best Accuracy'].map(lambda x: f"{x:.1%}")
    treatment_summary['Std Dev'] = treatment_summary['Std Dev'].map(lambda x: f"{x:.3f}")

    print(f"\n### Treatment Effectiveness (Ranked by Average Performance)\n")
    print(treatment_summary[['treatment', 'Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev']].to_markdown(index=False))

    # Highlight exceptional performers (>85%)
    exceptional_performers = final_df[final_df['accuracy'] > 0.85]
    if not exceptional_performers.empty:
        print(f"\n### Exceptional Performers (>85% Accuracy)\n")
        print("These strain/treatment combinations show exceptionally strong and distinctive behavioral phenotypes:\n")
        for _, row in exceptional_performers.iterrows():
            f1_balance = abs(row['f1_control'] - row['f1_treatment'])
            balance_desc = "balanced classification" if f1_balance < 0.3 else "control-dominant" if row['f1_control'] > row['f1_treatment'] else "treatment-dominant"
            print(f"- **{row['strain']} + {row['treatment']}:** {row['accuracy']:.1%} accuracy ({balance_desc})")

    # Analysis of poor performers for insights
    poor_performers = final_df[final_df['accuracy'] < 0.55]
    if not poor_performers.empty:
        print(f"\n### Analysis of Poor Classification Performance (<55%)\n")
        print("These combinations show limited phenotypic distinctiveness, suggesting:")
        print("1. Subtle treatment effects below current detection threshold")
        print("2. High individual variability masking treatment effects")
        print("3. Genetic background resistance to the treatment")
        print("4. Treatment concentrations outside effective range\n")

        poor_by_strain = poor_performers['strain'].value_counts()
        print("**Strains with most poor classifications:**")
        for strain, count in poor_by_strain.items():
            total_strain = len(final_df[final_df['strain'] == strain])
            print(f"- {strain}: {count}/{total_strain} treatments ({count/total_strain*100:.1f}%)")

    # Cross-strain treatment consistency analysis
    print(f"\n### Treatment Consistency Analysis\n")
    multi_strain_treatments = final_df.groupby('treatment').filter(lambda x: len(x) > 1)
    if not multi_strain_treatments.empty:
        consistency_analysis = []
        for treatment in multi_strain_treatments['treatment'].unique():
            treatment_data = final_df[final_df['treatment'] == treatment]

            avg_accuracy = treatment_data['accuracy'].mean()
            std_accuracy = treatment_data['accuracy'].std()
            min_accuracy = treatment_data['accuracy'].min()
            max_accuracy = treatment_data['accuracy'].max()
            strain_count = len(treatment_data)

            # Consistency score: high average with low variability
            consistency_score = avg_accuracy - std_accuracy if pd.notna(std_accuracy) else avg_accuracy

            consistency_analysis.append({
                'Treatment': treatment,
                'Strains': strain_count,
                'Avg Accuracy': f"{avg_accuracy:.1%}",
                'Std Dev': f"{std_accuracy:.3f}" if pd.notna(std_accuracy) else "N/A",
                'Range': f"{min_accuracy:.1%}-{max_accuracy:.1%}",
                'Consistency Score': f"{consistency_score:.3f}"
            })

        consistency_df = pd.DataFrame(consistency_analysis)
        consistency_df = consistency_df.sort_values('Consistency Score', ascending=False)

        print("**Most consistent treatments across strains:**\n")
        print(consistency_df.to_markdown(index=False))

    # Feature importance insights (based on F1 scores)
    print(f"\n### Classification Pattern Analysis\n")

    # Identify control-dominant vs treatment-dominant classifications
    control_dominant = final_df[final_df['f1_control'] > final_df['f1_treatment'] + 0.2]
    treatment_dominant = final_df[final_df['f1_treatment'] > final_df['f1_control'] + 0.2]

    print(f"**Control-dominant classifications:** {len(control_dominant)} cases")
    print("These suggest treatments create subtle changes, but control phenotype remains more distinctive.\n")

    print(f"**Treatment-dominant classifications:** {len(treatment_dominant)} cases")
    print("These suggest treatments create highly distinctive, consistent phenotypes.\n")

    if not treatment_dominant.empty:
        print("**Strongest treatment-dominant phenotypes:**")
        treatment_dominant_sorted = treatment_dominant.sort_values('f1_treatment', ascending=False).head(5)
        for _, row in treatment_dominant_sorted.iterrows():
            print(f"- {row['strain']} + {row['treatment']}: Treatment F1={row['f1_treatment']:.3f}, Control F1={row['f1_control']:.3f}")

if __name__ == '__main__':
    analyze_ml_screening()