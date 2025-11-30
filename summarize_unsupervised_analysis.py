import pandas as pd
from pathlib import Path
import numpy as np

def analyze_unsupervised_states():
    """
    Comprehensive analysis of unsupervised state discovery results across ALL strains and treatments.
    Correctly interprets ethogram.csv and state_distribution.csv files from each strain's analysis.
    """
    base_analysis_dir = Path('results/Analysis/Unsupervised_States')
    strains = [p.name for p in base_analysis_dir.iterdir() if p.is_dir()]

    print("## Comprehensive Unsupervised State Analysis\n")
    print("This analysis examines behavioral state distributions across ALL strain/treatment combinations.\n")
    print("**State Interpretation:**")
    print("- **Pause State:** Lowest speed_mean - sedation, toxicity, or rest behavior")
    print("- **Active Cruise State:** Highest displacement - exploration, stimulation, or search behavior")
    print("- **Intermediate States:** Various locomotion and turning behaviors\n")

    all_strain_results = {}

    for strain in strains:
        ethogram_path = base_analysis_dir / strain / 'ethogram.csv'
        distribution_path = base_analysis_dir / strain / 'state_distribution.csv'

        if not (ethogram_path.exists() and distribution_path.exists()):
            print(f"Missing data for strain {strain}")
            continue

        # Read the actual data files
        ethogram_df = pd.read_csv(ethogram_path)
        dist_df = pd.read_csv(distribution_path)

        # Find pause state (minimum speed_mean) and cruise state (maximum displacement)
        pause_state_idx = ethogram_df['speed_mean'].idxmin()
        cruise_state_idx = ethogram_df['displacement'].idxmax()

        all_strain_results[strain] = {
            'ethogram': ethogram_df,
            'distribution': dist_df,
            'pause_state': pause_state_idx,
            'cruise_state': cruise_state_idx
        }

    if not all_strain_results:
        print("No unsupervised state analysis results found. Check if the methodology has completed.")
        return

    # Display state definitions by strain
    print("### State Definitions by Strain\n")
    state_definitions = []
    for strain, data in all_strain_results.items():
        ethogram = data['ethogram']
        pause_idx = data['pause_state']
        cruise_idx = data['cruise_state']

        state_definitions.append({
            'Strain': strain,
            'Pause State': f'State {pause_idx}',
            'Pause Speed': f"{ethogram.loc[pause_idx, 'speed_mean']:.3f}",
            'Cruise State': f'State {cruise_idx}',
            'Cruise Displacement': f"{ethogram.loc[cruise_idx, 'displacement']:.3f}",
            'Total States': len(ethogram)
        })

    state_def_df = pd.DataFrame(state_definitions)
    print(state_def_df.to_markdown(index=False))

    # Complete state distribution analysis
    print(f"\n### Complete State Distribution Analysis\n")

    all_treatment_data = []

    for strain, data in all_strain_results.items():
        dist_df = data['distribution']
        pause_state_col = f"state_{data['pause_state']}"
        cruise_state_col = f"state_{data['cruise_state']}"

        print(f"**{strain}:**\n")

        # Display complete treatment table for this strain
        strain_summary = []
        for _, row in dist_df.iterrows():
            treatment = row['treatment']
            pause_pct = row[pause_state_col] if pause_state_col in row else 0
            cruise_pct = row[cruise_state_col] if cruise_state_col in row else 0

            strain_summary.append({
                'Treatment': treatment,
                'Pause %': f"{pause_pct:.1f}%",
                'Cruise %': f"{cruise_pct:.1f}%"
            })

            # Store for cross-strain analysis
            all_treatment_data.append({
                'strain': strain,
                'treatment': treatment,
                'pause_percent': pause_pct,
                'cruise_percent': cruise_pct
            })

        strain_summary_df = pd.DataFrame(strain_summary)
        print(strain_summary_df.to_markdown(index=False))
        print()

    # Cross-treatment analysis
    if all_treatment_data:
        comparison_df = pd.DataFrame(all_treatment_data)

        print(f"### Treatment Effects Across All Strains\n")

        # Group by treatment to see consistency
        treatment_effects = []
        for treatment in comparison_df['treatment'].unique():
            treatment_data = comparison_df[comparison_df['treatment'] == treatment]

            avg_pause = treatment_data['pause_percent'].mean()
            std_pause = treatment_data['pause_percent'].std()
            avg_cruise = treatment_data['cruise_percent'].mean()
            std_cruise = treatment_data['cruise_percent'].std()
            n_strains = len(treatment_data)

            treatment_effects.append({
                'Treatment': treatment,
                'Strains Tested': n_strains,
                'Avg Pause %': f"{avg_pause:.1f} ± {std_pause:.1f}",
                'Avg Cruise %': f"{avg_cruise:.1f} ± {std_cruise:.1f}",
                'Pause Range': f"{treatment_data['pause_percent'].min():.1f}-{treatment_data['pause_percent'].max():.1f}%",
                'Cruise Range': f"{treatment_data['cruise_percent'].min():.1f}-{treatment_data['cruise_percent'].max():.1f}%"
            })

        treatment_df = pd.DataFrame(treatment_effects)
        treatment_df = treatment_df.sort_values('Strains Tested', ascending=False)
        print(treatment_df.to_markdown(index=False))

        # Control comparison analysis
        control_data = comparison_df[comparison_df['treatment'] == 'Control']
        if not control_data.empty:
            print(f"\n### Treatment vs Control Comparison\n")

            control_comparison = []
            for treatment in comparison_df['treatment'].unique():
                if treatment == 'Control':
                    continue

                treatment_data = comparison_df[comparison_df['treatment'] == treatment]

                # Find strains that have both control and treatment data
                common_strains = set(control_data['strain']) & set(treatment_data['strain'])

                if common_strains:
                    sedative_effect = 0  # Count of strains showing increased pause
                    stimulant_effect = 0  # Count of strains showing increased cruise

                    avg_pause_changes = []
                    avg_cruise_changes = []

                    for strain in common_strains:
                        control_pause = control_data[control_data['strain'] == strain]['pause_percent'].iloc[0]
                        treatment_pause = treatment_data[treatment_data['strain'] == strain]['pause_percent'].iloc[0]

                        control_cruise = control_data[control_data['strain'] == strain]['cruise_percent'].iloc[0]
                        treatment_cruise = treatment_data[treatment_data['strain'] == strain]['cruise_percent'].iloc[0]

                        if treatment_pause > control_pause:
                            sedative_effect += 1
                        if treatment_cruise > control_cruise:
                            stimulant_effect += 1

                        avg_pause_changes.append(treatment_pause - control_pause)
                        avg_cruise_changes.append(treatment_cruise - control_cruise)

                    avg_pause_delta = np.mean(avg_pause_changes)
                    avg_cruise_delta = np.mean(avg_cruise_changes)

                    control_comparison.append({
                        'Treatment': treatment,
                        'Strains': len(common_strains),
                        'Sedative Effect': f"{sedative_effect}/{len(common_strains)}",
                        'Stimulant Effect': f"{stimulant_effect}/{len(common_strains)}",
                        'Avg Pause Δ': f"{avg_pause_delta:+.1f}%",
                        'Avg Cruise Δ': f"{avg_cruise_delta:+.1f}%"
                    })

            if control_comparison:
                control_comp_df = pd.DataFrame(control_comparison)
                print(control_comp_df.to_markdown(index=False))

        # Detailed strain-by-strain control comparisons for key treatments
        key_treatments = ['CBD 30uM', 'CBD 3uM', 'CBDV 30uM', 'ETANOL']
        for treatment in key_treatments:
            treatment_data = comparison_df[comparison_df['treatment'] == treatment]
            if treatment_data.empty:
                continue

            print(f"\n### {treatment} Effects by Strain\n")

            strain_effects = []
            for strain in treatment_data['strain'].unique():
                strain_control = control_data[control_data['strain'] == strain]
                strain_treatment = treatment_data[treatment_data['strain'] == strain]

                if not strain_control.empty and not strain_treatment.empty:
                    control_pause = strain_control['pause_percent'].iloc[0]
                    control_cruise = strain_control['cruise_percent'].iloc[0]
                    treatment_pause = strain_treatment['pause_percent'].iloc[0]
                    treatment_cruise = strain_treatment['cruise_percent'].iloc[0]

                    strain_effects.append({
                        'Strain': strain,
                        'Control Pause %': f"{control_pause:.1f}%",
                        f'{treatment} Pause %': f"{treatment_pause:.1f}%",
                        'Pause Change': f"{treatment_pause - control_pause:+.1f}%",
                        'Control Cruise %': f"{control_cruise:.1f}%",
                        f'{treatment} Cruise %': f"{treatment_cruise:.1f}%",
                        'Cruise Change': f"{treatment_cruise - control_cruise:+.1f}%"
                    })

            if strain_effects:
                strain_effects_df = pd.DataFrame(strain_effects)
                print(strain_effects_df.to_markdown(index=False))

if __name__ == '__main__':
    analyze_unsupervised_states()