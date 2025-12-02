import pandas as pd
from pathlib import Path
import sys
from contextlib import redirect_stdout


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


def analyze_ml_screening():
    """
    Comprehensive analysis of ML Screening methodology results showing ALL classification attempts
    across strains and treatments.
    
    Updated to reflect improvements based on García-Garví et al. (2025):
    - Multiple classifiers: Random Forest, XGBoost, Logistic Regression
    - 20 centroid-based kinematic features (Speed: 7, Turning: 5, Path: 5, Temporal: 3)
    - Optimized hyperparameters (max_depth=10, min_samples_split=5)
    - RFE feature selection capability
    - Permutation tests with experiment blocking
    """
    base_analysis_dir = Path('results/Analysis/ML_Screening')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive ML Screening Analysis\n")
    print("This analysis shows machine learning classification results for ALL strain/treatment combinations.\n")
    print("**Methodology (Based on García-Garví et al. 2025):**")
    print("- **20 Centroid-based Features:**")
    print("  - Speed metrics (7): mean, std, max, median, percentiles, coefficient of variation")
    print("  - Turning metrics (5): angular velocity stats, reversal frequency")
    print("  - Path metrics (5): tortuosity, total distance, displacement, path efficiency")
    print("  - Temporal metrics (3): active time ratio, pause frequency, movement bout duration")
    print("- **Classifiers Compared:**")
    print("  - Random Forest (max_depth=10, min_samples_split=5)")
    print("  - XGBoost (gradient boosting)")
    print("  - Logistic Regression (L1 regularization)")
    print("- **Validation:** Stratified 5-fold cross-validation with permutation tests\n")
    print("**Performance Categories:**")
    print("- **Excellent (>90%):** Very strong behavioral phenotype, highly distinguishable")
    print("- **Good (75-90%):** Clear behavioral phenotype, reliably distinguishable")
    print("- **Moderate (60-75%):** Detectable phenotype, some classification ability")
    print("- **Poor (<60%):** Weak/no phenotype, difficult to distinguish from control\n")

    all_results = []
    all_classifier_comparisons = []
    
    for strain in strains:
        # Load main classification summary
        summary_path = base_analysis_dir / strain / 'classification_summary.csv'
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df['strain'] = strain
            all_results.append(summary_df)
        else:
            print(f"Missing ML screening data for strain {strain}")
            continue
        
        # Load all classifier comparison files (new feature)
        comparison_files = list((base_analysis_dir / strain).glob('classifier_comparison_*.csv'))
        for comp_file in comparison_files:
            try:
                comparison_df = pd.read_csv(comp_file)
                comparison_df['strain'] = strain
                # Extract treatment from filename: classifier_comparison_Control_vs_TREATMENT.csv
                treatment_name = comp_file.stem.replace('classifier_comparison_Control_vs_', '')
                comparison_df['treatment'] = treatment_name.replace('_', ' ')
                all_classifier_comparisons.append(comparison_df)
            except Exception as e:
                print(f"Warning: Could not read {comp_file}: {e}")

    if not all_results:
        print("No ML Screening results found. Check if the methodology has completed.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    
    # Load classifier comparison data if available
    comparison_df = None
    if all_classifier_comparisons:
        comparison_df = pd.concat(all_classifier_comparisons, ignore_index=True)

    # Handle both old (accuracy) and new (accuracy_mean) column names
    if 'accuracy_mean' in final_df.columns:
        acc_col = 'accuracy_mean'
        acc_std_col = 'accuracy_std'
        has_std = True
    else:
        acc_col = 'accuracy'
        has_std = False

    # Check for classifier column
    has_classifier = 'classifier' in final_df.columns

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

    final_df['performance'] = final_df[acc_col].apply(categorize_performance)

    # Sort by performance
    final_df = final_df.sort_values([acc_col], ascending=False)

    # Display complete results
    display_cols = ['strain', 'treatment']
    if has_classifier:
        display_cols.append('classifier')
    display_cols.append(acc_col)
    if has_std:
        display_cols.append(acc_std_col)
    display_cols.extend(['f1_control', 'f1_treatment', 'performance'])
    
    display_cols = [col for col in display_cols if col in final_df.columns]
    display_df = final_df[display_cols].copy()

    rename_dict = {
        acc_col: 'Accuracy (mean)' if has_std else 'Accuracy',
        'f1_control': 'F1 (Control)',
        'f1_treatment': 'F1 (Treatment)',
        'performance': 'Performance Category',
        'classifier': 'Classifier'
    }
    if has_std:
        rename_dict[acc_std_col] = 'Accuracy (std)'
    
    rename_dict = {k: v for k, v in rename_dict.items() if k in display_df.columns}
    display_df.rename(columns=rename_dict, inplace=True)

    display_df['Accuracy (mean)' if has_std else 'Accuracy'] = display_df['Accuracy (mean)' if has_std else 'Accuracy'].map(lambda x: f"{x:.1%}")
    if has_std:
        display_df['Accuracy (std)'] = display_df['Accuracy (std)'].map(lambda x: f"{x:.3f}")

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

    # Classifier comparison analysis if available
    if comparison_df is not None and not comparison_df.empty:
        print(f"\n### Classifier Comparison Analysis")
        print("Comparing performance across different classifiers (García-Garví et al. 2025 recommendation):\n")
        
        # Best classifier per strain/treatment
        if 'classifier' in comparison_df.columns and acc_col in comparison_df.columns:
            classifier_stats = comparison_df.groupby('classifier')[acc_col].agg(['mean', 'std', 'max', 'min']).round(3)
            classifier_stats.columns = ['Avg Accuracy', 'Std Dev', 'Best', 'Worst']
            classifier_stats['Avg Accuracy'] = classifier_stats['Avg Accuracy'].map(lambda x: f"{x:.1%}")
            classifier_stats['Best'] = classifier_stats['Best'].map(lambda x: f"{x:.1%}")
            classifier_stats['Worst'] = classifier_stats['Worst'].map(lambda x: f"{x:.1%}")
            classifier_stats['Std Dev'] = classifier_stats['Std Dev'].map(lambda x: f"{x:.3f}")
            
            print("**Overall Classifier Performance:**\n")
            print(classifier_stats.to_markdown())
            
            # Count how often each classifier wins
            print("\n**Best Classifier per Strain/Treatment:**\n")
            for (strain, treatment), group in comparison_df.groupby(['strain', 'treatment']):
                best_row = group.loc[group[acc_col].idxmax()]
                print(f"- {strain} + {treatment}: **{best_row['classifier']}** ({best_row[acc_col]:.1%})")
    
    elif has_classifier:
        print(f"\n### Classifier Comparison Analysis")
        print("Comparing performance across different classifiers:\n")
        
        classifier_stats = final_df.groupby('classifier')[acc_col].agg(['mean', 'std', 'count']).round(3)
        classifier_stats.columns = ['Avg Accuracy', 'Std Dev', 'Tests']
        classifier_stats = classifier_stats.sort_values('Avg Accuracy', ascending=False)
        classifier_stats['Avg Accuracy'] = classifier_stats['Avg Accuracy'].map(lambda x: f"{x:.1%}")
        classifier_stats['Std Dev'] = classifier_stats['Std Dev'].map(lambda x: f"{x:.3f}")
        
        print("**Classifier Performance Summary:**\n")
        print(classifier_stats.to_markdown())
        
        # Best classifier per treatment
        best_per_treatment = final_df.loc[final_df.groupby(['strain', 'treatment'])[acc_col].idxmax()]
        classifier_wins = best_per_treatment['classifier'].value_counts()
        
        print("\n**Best Classifier Frequency:**\n")
        for clf, count in classifier_wins.items():
            pct = count / len(best_per_treatment) * 100
            print(f"- **{clf}:** {count} wins ({pct:.1f}%)")

    # Strain-wise performance breakdown
    print(f"\n### Performance by Strain")
    for strain in final_df['strain'].unique():
        strain_data = final_df[final_df['strain'] == strain]
        total_strain_tests = len(strain_data)
        avg_accuracy = strain_data[acc_col].mean()
        best_accuracy = strain_data[acc_col].max()
        best_row = strain_data.loc[strain_data[acc_col].idxmax()]
        best_treatment = best_row['treatment']

        print(f"\n**{strain}:**")
        print(f"- Tests: {total_strain_tests}")
        print(f"- Average accuracy: {avg_accuracy:.1%}")
        
        if has_classifier:
            best_classifier = best_row.get('classifier', 'N/A')
            print(f"- Best result: {best_accuracy:.1%} ({best_treatment}, {best_classifier})")
        else:
            print(f"- Best result: {best_accuracy:.1%} ({best_treatment})")

        # Performance distribution for this strain
        strain_performance_counts = strain_data['performance'].value_counts()
        for perf_cat, count in strain_performance_counts.items():
            print(f"- {perf_cat}: {count}")

    # Treatment effectiveness across all strains
    agg_dict = {
        acc_col: ['count', 'mean', 'max', 'std'],
        'strain': lambda x: list(set(x))
    }
    treatment_summary = final_df.groupby('treatment').agg(agg_dict).round(3)
    treatment_summary.columns = ['Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev', 'Strains']
    treatment_summary = treatment_summary.reset_index()
    treatment_summary = treatment_summary.sort_values('Avg Accuracy', ascending=False)
    treatment_summary['Avg Accuracy'] = treatment_summary['Avg Accuracy'].map(lambda x: f"{x:.1%}")
    treatment_summary['Best Accuracy'] = treatment_summary['Best Accuracy'].map(lambda x: f"{x:.1%}")
    treatment_summary['Std Dev'] = treatment_summary['Std Dev'].map(lambda x: f"{x:.3f}")

    print(f"\n### Treatment Effectiveness (Ranked by Average Performance)\n")
    print(treatment_summary[['treatment', 'Tests', 'Avg Accuracy', 'Best Accuracy', 'Std Dev']].to_markdown(index=False))

    # Highlight exceptional performers (>85%)
    exceptional_performers = final_df[final_df[acc_col] > 0.85]
    if not exceptional_performers.empty:
        print(f"\n### Exceptional Performers (>85% Accuracy)\n")
        print("These strain/treatment combinations show exceptionally strong and distinctive behavioral phenotypes:\n")
        for _, row in exceptional_performers.iterrows():
            f1_balance = abs(row['f1_control'] - row['f1_treatment'])
            balance_desc = "balanced classification" if f1_balance < 0.3 else "control-dominant" if row['f1_control'] > row['f1_treatment'] else "treatment-dominant"
            acc_str = f"{row[acc_col]:.1%}"
            if has_std:
                acc_str += f" ± {row[acc_std_col]:.3f}"
            
            clf_str = f" [{row['classifier']}]" if has_classifier else ""
            print(f"- **{row['strain']} + {row['treatment']}:**{clf_str} {acc_str} accuracy ({balance_desc})")

    # Analysis of poor performers for insights
    poor_performers = final_df[final_df[acc_col] < 0.55]
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
    multi_strain_treatments = final_df.groupby('treatment').filter(lambda x: len(x['strain'].unique()) > 1)
    if not multi_strain_treatments.empty:
        consistency_analysis = []
        for treatment in multi_strain_treatments['treatment'].unique():
            treatment_data = final_df[final_df['treatment'] == treatment]

            avg_accuracy = treatment_data[acc_col].mean()
            std_accuracy = treatment_data[acc_col].std()
            min_accuracy = treatment_data[acc_col].min()
            max_accuracy = treatment_data[acc_col].max()
            strain_count = len(treatment_data['strain'].unique())

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
            clf_str = f" [{row['classifier']}]" if has_classifier else ""
            print(f"- {row['strain']} + {row['treatment']}{clf_str}: Treatment F1={row['f1_treatment']:.3f}, Control F1={row['f1_control']:.3f}")

    # Feature importance files check
    print(f"\n### Feature Importance Analysis\n")
    feature_importance_found = False
    for strain in strains:
        fi_path = base_analysis_dir / strain / 'feature_importance.csv'
        if fi_path.exists():
            feature_importance_found = True
            fi_df = pd.read_csv(fi_path)
            
            # Get top features
            if 'importance' in fi_df.columns and 'feature' in fi_df.columns:
                top_features = fi_df.nlargest(5, 'importance')
                print(f"**Top 5 Features for {strain}:**")
                for _, row in top_features.iterrows():
                    print(f"  - {row['feature']}: {row['importance']:.4f}")
                print()
    
    if not feature_importance_found:
        print("Feature importance files not found. Run the analysis to generate them.")
        print("Top discriminative features typically include:")
        print("- Speed coefficient of variation (CV)")
        print("- Path tortuosity")
        print("- Reversal frequency")
        print("- Angular velocity statistics")

    # Visualizations generated
    print(f"\n### Visualizations Generated\n")
    print("The following visualizations are available in each strain's output directory:")
    print("- `confusion_matrix_*.png`: Classification confusion matrices per treatment")
    print("- `feature_importance_*.png`: Feature importance rankings")
    print("- `roc_curve_*.png`: ROC curves with AUC scores")
    if has_classifier or comparison_df is not None:
        print("- `classifier_comparison_*.png`: Performance comparison across classifiers")
    print("- `permutation_test_*.png`: Permutation test results for statistical validation")


if __name__ == '__main__':
    # Save a copy of the summary to disk while still printing to the terminal
    summaries_dir = Path('results/Analysis/Summaries')
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / 'ml_screening_summary.md'

    with summary_path.open('w', encoding='utf-8') as f, redirect_stdout(Tee(sys.stdout, f)):
        analyze_ml_screening()
