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


def analyze_levy_flight():
    """
    Comprehensive analysis of Lévy flight methodology results across all strains and treatments.
    Updated to reflect improvements based on Moy et al. (2015):
    - Per-worm statistics with strict Lévy criterion (1 < α ≤ 3)
    - Statistical criterion (R > 0, p < 0.05) vs strict criterion
    - Alpha distribution analysis across individual worms
    """
    base_analysis_dir = Path('results/Analysis/Levy_Flight')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive Lévy Flight Analysis\n")
    print("This analysis shows power-law vs lognormal distribution fitting results for ALL strain/treatment combinations.\n")
    print("**Interpretation (Based on Moy et al. 2015):**")
    print("- **Statistical Criterion:** R > 0 & p < 0.05 (power-law fits better than lognormal)")
    print("- **Strict Lévy Criterion:** R > 0 & p < 0.05 & **1 < α ≤ 3** (within theoretical Lévy range)")
    print("- **α (alpha):** Power-law exponent")
    print("  - α ≈ 2: Optimal foraging search strategy")
    print("  - 1 < α ≤ 3: Lévy flight range")
    print("  - α > 3: Transition to Brownian motion\n")

    all_results = []
    all_per_worm_results = []
    
    for strain in strains:
        # Load pooled summary
        summary_path = base_analysis_dir / strain / 'levy_flight_summary.csv'
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df['strain'] = strain
            all_results.append(summary_df)
        
        # Load per-worm results if available (new feature)
        per_worm_path = base_analysis_dir / strain / 'levy_flight_per_worm.csv'
        if per_worm_path.exists():
            per_worm_df = pd.read_csv(per_worm_path)
            per_worm_df['strain'] = strain
            all_per_worm_results.append(per_worm_df)

    if not all_results:
        print("No Lévy flight analysis results found. Check if the methodology has completed.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    
    # Load per-worm data if available
    per_worm_df = None
    if all_per_worm_results:
        per_worm_df = pd.concat(all_per_worm_results, ignore_index=True)

    # Normalize column names (handle both 'alpha' and 'alpha_pooled')
    if 'alpha_pooled' in final_df.columns and 'alpha' not in final_df.columns:
        final_df['alpha'] = final_df['alpha_pooled']
    
    # Add significance classification - Statistical criterion
    final_df['is_levy_statistical'] = final_df.apply(
        lambda row: row['loglikelihood_ratio'] > 0 and row['p_value'] < 0.05, axis=1
    )
    
    # Add strict Lévy criterion (1 < α ≤ 3)
    final_df['is_levy_strict'] = final_df.apply(
        lambda row: (row['loglikelihood_ratio'] > 0 and 
                    row['p_value'] < 0.05 and 
                    1 < row['alpha'] <= 3), axis=1
    )
    
    # Classification based on both criteria
    def classify_search_type(row):
        if row['is_levy_strict']:
            return 'Lévy (Strict)'
        elif row['is_levy_statistical']:
            return 'Lévy (Statistical only)'
        else:
            return 'Brownian'
    
    final_df['search_type'] = final_df.apply(classify_search_type, axis=1)

    # Sort by search type and p-value for better readability
    display_cols = ['strain', 'treatment', 'alpha', 'alpha_ci_lower', 'alpha_ci_upper',
                    'p_value', 'loglikelihood_ratio', 'search_type']
    
    # Add per-worm statistics if available (handle different column names)
    if 'percent_levy_statistical' in final_df.columns:
        display_cols.insert(-1, 'percent_levy_statistical')
    elif 'percent_levy_worms' in final_df.columns:
        display_cols.insert(-1, 'percent_levy_worms')
    if 'percent_levy_strict' in final_df.columns:
        display_cols.insert(-1, 'percent_levy_strict')
    elif 'percent_levy_worms_strict' in final_df.columns:
        display_cols.insert(-1, 'percent_levy_worms_strict')
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in final_df.columns]
    display_df = final_df[display_cols].copy()
    display_df = display_df.sort_values(['search_type', 'p_value'])

    # Rename columns for clarity
    rename_dict = {
        'alpha': 'α (exponent)',
        'alpha_pooled': 'α (exponent)',
        'alpha_ci_lower': 'α CI Lower',
        'alpha_ci_upper': 'α CI Upper',
        'p_value': 'p-value',
        'loglikelihood_ratio': 'R (Log-Likelihood)',
        'percent_levy_worms': '% Worms Statistical',
        'percent_levy_statistical': '% Worms Statistical',
        'percent_levy_worms_strict': '% Worms Strict',
        'percent_levy_strict': '% Worms Strict',
        'search_type': 'Search Pattern'
    }
    rename_dict = {k: v for k, v in rename_dict.items() if k in display_df.columns}
    display_df.rename(columns=rename_dict, inplace=True)

    print("### Complete Results Table\n")
    print(display_df.to_markdown(index=False))

    # Summary statistics
    total_tests = len(final_df)
    levy_strict = len(final_df[final_df['is_levy_strict']])
    levy_statistical = len(final_df[final_df['is_levy_statistical']])
    brownian_cases = total_tests - levy_statistical

    print(f"\n### Summary Statistics")
    print(f"- **Total strain/treatment combinations tested:** {total_tests}")
    print(f"- **Lévy patterns (Statistical: R>0, p<0.05):** {levy_statistical} ({levy_statistical/total_tests*100:.1f}%)")
    print(f"- **Lévy patterns (Strict: 1 < α ≤ 3):** {levy_strict} ({levy_strict/total_tests*100:.1f}%)")
    print(f"- **Standard Brownian motion:** {brownian_cases} ({brownian_cases/total_tests*100:.1f}%)")
    
    # Alpha value interpretation
    if levy_statistical > 0:
        levy_data = final_df[final_df['is_levy_statistical']]
        alpha_in_range = len(levy_data[(levy_data['alpha'] > 1) & (levy_data['alpha'] <= 3)])
        alpha_transition = len(levy_data[levy_data['alpha'] > 3])
        
        print(f"\n### Alpha Value Analysis (For Statistical Lévy Cases)")
        print(f"- **α in Lévy range (1 < α ≤ 3):** {alpha_in_range}")
        print(f"- **α in transition zone (α > 3):** {alpha_transition}")
        
        if alpha_transition > 0:
            print(f"\n⚠️ **Note:** {alpha_transition} case(s) with α > 3 show power-law distribution but are in the")
            print("   transition zone between Lévy and Brownian motion (Moy et al., 2015).")

    # Results by strain
    strain_summary = final_df.groupby('strain').agg({
        'is_levy_statistical': 'sum',
        'is_levy_strict': 'sum',
        'alpha': ['mean', 'std']
    }).round(3)
    strain_summary.columns = ['Lévy (Statistical)', 'Lévy (Strict)', 'α Mean', 'α Std']
    strain_summary['Total'] = final_df.groupby('strain').size()
    strain_summary['% Lévy Strict'] = (strain_summary['Lévy (Strict)'] / strain_summary['Total'] * 100).round(1)

    print(f"\n### Results by Strain\n")
    print(strain_summary.to_markdown())

    # Per-worm analysis if available
    if per_worm_df is not None and not per_worm_df.empty:
        print(f"\n### Per-Worm Alpha Distribution Analysis\n")
        print("Individual worm statistics provide more robust insights than pooled data (Moy et al., 2015):\n")
        
        # Per-worm summary by treatment
        per_worm_summary = per_worm_df.groupby(['strain', 'treatment']).agg({
            'alpha': ['mean', 'median', 'std', 'min', 'max'],
            'is_levy_statistical': 'sum' if 'is_levy_statistical' in per_worm_df.columns else 'count',
            'is_levy_strict': 'sum' if 'is_levy_strict' in per_worm_df.columns else 'count'
        }).round(3)
        
        # Count total worms per group
        worm_counts = per_worm_df.groupby(['strain', 'treatment']).size()
        
        print("**Per-Worm Alpha Statistics:**\n")
        for (strain, treatment), group in per_worm_df.groupby(['strain', 'treatment']):
            n_worms = len(group)
            alpha_mean = group['alpha'].mean()
            alpha_median = group['alpha'].median()
            alpha_std = group['alpha'].std()
            
            # Count worms in Lévy range
            if 'is_levy_strict' in group.columns:
                n_levy_strict = group['is_levy_strict'].sum()
            else:
                n_levy_strict = len(group[(group['alpha'] > 1) & (group['alpha'] <= 3)])
            
            pct_levy = n_levy_strict / n_worms * 100 if n_worms > 0 else 0
            
            print(f"- **{strain} + {treatment}:** n={n_worms}, α={alpha_mean:.2f}±{alpha_std:.2f} (median={alpha_median:.2f}), {pct_levy:.1f}% in Lévy range")

    # Treatment effectiveness for inducing Lévy-like search
    if levy_strict > 0:
        print(f"\n### Successful Lévy Flight Inductions (Strict Criterion)\n")

        levy_results = final_df[final_df['is_levy_strict']].copy()
        levy_results = levy_results.sort_values('p_value')

        print("**Significant Lévy-like search patterns with α in theoretical range:**\n")
        for _, row in levy_results.iterrows():
            alpha_optimal = "optimal" if 1.8 <= row['alpha'] <= 2.2 else "efficient"
            print(f"- **{row['strain']} + {row['treatment']}:** α={row['alpha']:.3f} ({alpha_optimal}), p={row['p_value']:.2e}, R={row['loglikelihood_ratio']:.3f}")

        # Treatment effectiveness
        levy_treatments = levy_results['treatment'].value_counts()
        print(f"\n### Most Effective Treatments for Inducing Lévy-like Search\n")
        for treatment, count in levy_treatments.items():
            affected_strains = levy_results[levy_results['treatment'] == treatment]['strain'].tolist()
            print(f"- **{treatment}:** {count} strain(s) ({', '.join(affected_strains)})")

    elif levy_statistical > 0:
        print(f"\n### Statistical Lévy Patterns (α outside strict range)\n")
        print("These cases show power-law distribution but α is outside the theoretical Lévy range (1-3):\n")
        
        levy_results = final_df[final_df['is_levy_statistical']].copy()
        for _, row in levy_results.iterrows():
            alpha_note = "transition zone" if row['alpha'] > 3 else "superdiffusive" if row['alpha'] < 1 else "valid"
            print(f"- **{row['strain']} + {row['treatment']}:** α={row['alpha']:.3f} ({alpha_note}), p={row['p_value']:.2e}")

    else:
        print(f"\n### No Significant Lévy Flight Patterns Detected")
        print("All strain/treatment combinations showed standard Brownian motion patterns.")
        print("This suggests that the tested cannabinoid treatments do not induce optimal foraging behaviors under the current experimental conditions.")

    # Analysis of near-significant cases
    near_significant = final_df[(final_df['loglikelihood_ratio'] > 0) & (final_df['p_value'] >= 0.05) & (final_df['p_value'] < 0.1)]
    if not near_significant.empty:
        print(f"\n### Near-Significant Cases (0.05 ≤ p < 0.1)\n")
        print("These cases show some evidence of Lévy-like behavior but don't reach statistical significance:\n")
        for _, row in near_significant.iterrows():
            print(f"- **{row['strain']} + {row['treatment']}:** α={row['alpha']:.3f}, p={row['p_value']:.3f}, R={row['loglikelihood_ratio']:.3f}")

    # Detailed strain analysis
    print(f"\n### Detailed Strain Analysis\n")
    for strain in strains:
        strain_data = final_df[final_df['strain'] == strain]
        if strain_data.empty:
            continue

        levy_strict_count = len(strain_data[strain_data['is_levy_strict']])
        levy_stat_count = len(strain_data[strain_data['is_levy_statistical']])
        total_count = len(strain_data)
        avg_alpha = strain_data['alpha'].mean()

        print(f"**{strain}:**")
        print(f"- Total treatments tested: {total_count}")
        print(f"- Average α: {avg_alpha:.3f}")
        print(f"- Lévy patterns (statistical): {levy_stat_count} ({levy_stat_count/total_count*100:.1f}%)")
        print(f"- Lévy patterns (strict 1<α≤3): {levy_strict_count} ({levy_strict_count/total_count*100:.1f}%)")

        if levy_strict_count > 0:
            best_levy = strain_data[strain_data['is_levy_strict']].sort_values('p_value').iloc[0]
            print(f"- Best strict Lévy: {best_levy['treatment']} (α={best_levy['alpha']:.3f}, p={best_levy['p_value']:.2e})")
        elif levy_stat_count > 0:
            best_levy = strain_data[strain_data['is_levy_statistical']].sort_values('p_value').iloc[0]
            print(f"- Best statistical Lévy: {best_levy['treatment']} (α={best_levy['alpha']:.3f}, p={best_levy['p_value']:.2e})")

        # Show most promising non-significant result
        non_levy = strain_data[~strain_data['is_levy_statistical']]
        if not non_levy.empty:
            promising = non_levy.sort_values('loglikelihood_ratio', ascending=False).iloc[0]
            if promising['loglikelihood_ratio'] > -2:
                print(f"- Most promising non-significant: {promising['treatment']} (R={promising['loglikelihood_ratio']:.3f})")
        print()

    # FDR correction results
    fdr_path = Path('results/Analysis/Levy_Flight/levy_flight_fdr_correction.csv')
    if fdr_path.exists():
        print(f"\n### FDR Correction Results (Benjamini-Hochberg)\n")
        fdr_df = pd.read_csv(fdr_path)

        n_sig_uncorrected = (fdr_df['p_value'] < 0.05).sum()
        n_sig_corrected = fdr_df['significant_FDR'].sum()
        false_positives_eliminated = n_sig_uncorrected - n_sig_corrected

        print(f"**Multiple comparison correction applied across {len(fdr_df)} total tests:**")
        print(f"- Significant before correction (p < 0.05): {n_sig_uncorrected}")
        print(f"- Significant after FDR correction (q < 0.05): {n_sig_corrected}")
        print(f"- Likely false positives eliminated: {false_positives_eliminated}\n")

        if n_sig_corrected > 0:
            print(f"**FDR-corrected significant results:**\n")
            sig_fdr = fdr_df[fdr_df['significant_FDR']].sort_values('q_value_BH')
            for _, row in sig_fdr.iterrows():
                # Check if alpha is in strict range
                alpha_note = "✓ strict" if 1 < row.get('alpha', 0) <= 3 else "transition zone"
                print(f"- **{row['strain']} + {row['treatment']}:** q={row['q_value_BH']:.3f}, R={row['R']:.3f} ({alpha_note})")
        else:
            print("**No results survived FDR correction.**")
            print("This suggests that all significant p-values were likely false positives due to multiple testing.")

    # Visualizations generated
    print(f"\n### Visualizations Generated\n")
    print("The following visualizations are available in each strain's output directory:")
    print("- `ccdf_*.png`: CCDF plots with power-law fits (pooled data)")
    print("- `alpha_histogram_*.png`: Distribution of α values across individual worms")
    print("- `alpha_boxplot.png`: Boxplot comparing α distributions between treatments")


if __name__ == '__main__':
    # Save a copy of the summary to disk while still printing to the terminal
    summaries_dir = Path('results/Analysis/Summaries')
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / 'levy_flight_summary.md'

    with summary_path.open('w', encoding='utf-8') as f, redirect_stdout(Tee(sys.stdout, f)):
        analyze_levy_flight()
