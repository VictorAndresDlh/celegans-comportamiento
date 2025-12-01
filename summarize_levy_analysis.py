import pandas as pd
from pathlib import Path

def analyze_levy_flight():
    """
    Comprehensive analysis of Lévy flight methodology results across all strains and treatments.
    Correctly interprets levy_flight_summary.csv files from each strain's analysis.
    """
    base_analysis_dir = Path('results/Analysis/Levy_Flight')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive Lévy Flight Analysis\n")
    print("This analysis shows power-law vs lognormal distribution fitting results for ALL strain/treatment combinations.\n")
    print("**Interpretation:**")
    print("- **R > 0 & p < 0.05:** Significant Lévy-like search (efficient foraging)")
    print("- **R ≤ 0 or p ≥ 0.05:** Standard Brownian motion (normal random walk)")
    print("- **α (alpha) 1-3:** Lévy flight range; α ≈ 2: optimal search strategy\n")

    all_results = []
    for strain in strains:
        summary_path = base_analysis_dir / strain / 'levy_flight_summary.csv'
        if not summary_path.exists():
            print(f"Missing Levy flight data for strain {strain}")
            continue

        summary_df = pd.read_csv(summary_path)
        summary_df['strain'] = strain
        all_results.append(summary_df)

    if not all_results:
        print("No Lévy flight analysis results found. Check if the methodology has completed.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # Add significance classification based on log-likelihood ratio and p-value
    final_df['search_type'] = final_df.apply(
        lambda row: 'Lévy-like' if (row['loglikelihood_ratio'] > 0 and row['p_value'] < 0.05) else 'Brownian', axis=1
    )

    # Sort by search type and p-value for better readability
    display_cols = ['strain', 'treatment', 'alpha', 'alpha_ci_lower', 'alpha_ci_upper',
                    'p_value', 'loglikelihood_ratio', 'percent_levy_worms', 'search_type']
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in final_df.columns]
    display_df = final_df[display_cols].copy()
    display_df = display_df.sort_values(['search_type', 'p_value'])

    # Rename columns for clarity
    rename_dict = {
        'alpha': 'α (exponent)',
        'alpha_ci_lower': 'α CI Lower',
        'alpha_ci_upper': 'α CI Upper',
        'p_value': 'p-value',
        'loglikelihood_ratio': 'R (Log-Likelihood Ratio)',
        'percent_levy_worms': '% Worms Lévy-like',
        'search_type': 'Search Pattern'
    }
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in display_df.columns}
    display_df.rename(columns=rename_dict, inplace=True)

    print("### Complete Results Table\n")
    print(display_df.to_markdown(index=False))

    # Summary statistics
    total_tests = len(final_df)
    levy_cases = len(final_df[final_df['search_type'] == 'Lévy-like'])
    brownian_cases = total_tests - levy_cases

    print(f"\n### Summary Statistics")
    print(f"- **Total strain/treatment combinations tested:** {total_tests}")
    print(f"- **Lévy-like search patterns:** {levy_cases} ({levy_cases/total_tests*100:.1f}%)")
    print(f"- **Standard Brownian motion:** {brownian_cases} ({brownian_cases/total_tests*100:.1f}%)")

    # Results by strain
    strain_summary = final_df.groupby('strain')['search_type'].value_counts().unstack(fill_value=0)
    strain_summary['Total'] = strain_summary.sum(axis=1)
    if 'Lévy-like' in strain_summary.columns:
        strain_summary['Lévy %'] = (strain_summary['Lévy-like'] / strain_summary['Total'] * 100).round(1)
    else:
        strain_summary['Lévy %'] = 0

    print(f"\n### Results by Strain\n")
    print(strain_summary.to_markdown())

    # Treatment effectiveness for inducing Lévy-like search
    if levy_cases > 0:
        print(f"\n### Successful Lévy Flight Inductions\n")

        levy_results = final_df[final_df['search_type'] == 'Lévy-like'].copy()
        levy_results = levy_results.sort_values('p_value')

        print("**Significant Lévy-like search patterns detected:**\n")
        for _, row in levy_results.iterrows():
            alpha_optimal = "optimal" if 1.8 <= row['alpha'] <= 2.2 else "sub-optimal" if row['alpha'] > 3 else "efficient"
            print(f"- **{row['strain']} + {row['treatment']}:** α={row['alpha']:.3f} ({alpha_optimal}), p={row['p_value']:.2e}, R={row['loglikelihood_ratio']:.3f}")

        # Treatment effectiveness
        levy_treatments = levy_results['treatment'].value_counts()
        print(f"\n### Most Effective Treatments for Inducing Lévy-like Search\n")
        for treatment, count in levy_treatments.items():
            affected_strains = levy_results[levy_results['treatment'] == treatment]['strain'].tolist()
            print(f"- **{treatment}:** {count} strain(s) ({', '.join(affected_strains)})")

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

        levy_count = len(strain_data[strain_data['search_type'] == 'Lévy-like'])
        total_count = len(strain_data)

        print(f"**{strain}:**")
        print(f"- Total treatments tested: {total_count}")
        print(f"- Lévy-like patterns: {levy_count} ({levy_count/total_count*100:.1f}%)")

        if levy_count > 0:
            best_levy = strain_data[strain_data['search_type'] == 'Lévy-like'].sort_values('p_value').iloc[0]
            print(f"- Best Lévy result: {best_levy['treatment']} (α={best_levy['alpha']:.3f}, p={best_levy['p_value']:.2e})")

        # Show most promising non-significant result
        promising = strain_data[strain_data['search_type'] == 'Brownian'].sort_values('loglikelihood_ratio', ascending=False).iloc[0]
        if promising['loglikelihood_ratio'] > -2:  # Show if not too negative
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
                print(f"- **{row['strain']} + {row['treatment']}:** q={row['q_value_BH']:.3f}, R={row['R']:.3f}")
        else:
            print("**No results survived FDR correction.**")
            print("This suggests that all significant p-values were likely false positives due to multiple testing.")

if __name__ == '__main__':
    analyze_levy_flight()