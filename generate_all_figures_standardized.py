#!/usr/bin/env python3
"""
STANDARDIZED Paper Visualization Generator
===========================================
All 7 figures with IDENTICAL formatting:
- Consistent legend: upper right, no shadow, framealpha=0.95
- Y-axis labels always present and complete
- X-axis labels complete (no abbreviations)
- Grid only on Y-axis (alpha=0.3, linestyle=':')
- Same color palette throughout
- Same font sizes and styles
- Panel titles: Left-aligned with "A.", "B.", etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Paths
BASE_PATH = Path('results/Analysis')
OUTPUT_PATH = Path('results/Figures')
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# CONSISTENT color palette (colorblind-friendly)
COLORS = {
    'control': '#0173B2',        # Blue
    'treatment': '#DE8F05',      # Orange
    'etanol': '#DE8F05',         # Orange
    'stimulant': '#029E73',      # Teal/Green
    'sedative': '#CC3311',       # Red
    'ml': '#0173B2',            # Blue
    'tda': '#DE8F05',           # Orange
    'levy_optimal': '#029E73',   # Teal
    'levy_brownian': '#999999',  # Gray
}

# ============================================================================
# DATA LOADING FUNCTIONS (CORRECTED)
# ============================================================================

def load_state_distribution(strain: str) -> pd.DataFrame:
    """Load state distribution data for a given strain."""
    path = BASE_PATH / 'Unsupervised_States' / strain / 'state_distribution.csv'
    if not path.exists():
        raise FileNotFoundError(f"State distribution not found for {strain}")
    return pd.read_csv(path)

def load_ethogram(strain: str) -> pd.DataFrame:
    """Load ethogram (state characterization) for a given strain."""
    path = BASE_PATH / 'Unsupervised_States' / strain / 'ethogram.csv'
    if not path.exists():
        raise FileNotFoundError(f"Ethogram not found for {strain}")
    return pd.read_csv(path)

def load_ml_classification(strain: str) -> pd.DataFrame:
    """Load ML screening classification results for a strain."""
    path = BASE_PATH / 'ML_Screening' / strain / 'classification_summary.csv'
    if not path.exists():
        raise FileNotFoundError(f"ML classification not found for {strain}")
    return pd.read_csv(path)

def load_tda_classification(strain: str) -> pd.DataFrame:
    """Load TDA classification results for a strain."""
    path = BASE_PATH / 'TDA' / strain / 'classification_summary.csv'
    if not path.exists():
        raise FileNotFoundError(f"TDA classification not found for {strain}")
    return pd.read_csv(path)

def load_levy_flight(strain: str) -> pd.DataFrame:
    """Load Lévy flight analysis results for a strain."""
    path = BASE_PATH / 'Levy_Flight' / strain / 'levy_flight_summary.csv'
    if not path.exists():
        raise FileNotFoundError(f"Lévy flight data not found for {strain}")
    return pd.read_csv(path)

def identify_pause_state(strain: str) -> int:
    """Identify the pause state (lowest speed_mean) for a strain."""
    ethogram = load_ethogram(strain)
    return ethogram['speed_mean'].idxmin()

def identify_cruise_state(strain: str) -> int:
    """Identify the active cruise state (highest displacement) for a strain."""
    ethogram = load_ethogram(strain)
    return ethogram['displacement'].idxmax()

# ============================================================================
# FIG1: ETANOL SEDATIVE EFFECT
# ============================================================================

def create_fig1_etanol_sedative():
    """3-panel comparison showing ETANOL sedative effect across strains."""
    print("Creating Fig1: ETANOL Sedative Effect...")
    
    strains = [
        ('N2', 'Wild-type'),
        ('UA44', 'DA α-synuclein'),
        ('ROSELLA', 'Autophagy reporter')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (strain, description) in enumerate(strains):
        ax = axes[idx]
        
        # Identify pause state
        pause_state = identify_pause_state(strain)
        state_col = f'state_{pause_state}'
        
        # Load data
        dist_df = load_state_distribution(strain)
        
        # Extract Control and ETANOL values
        ctrl_row = dist_df[dist_df['treatment'] == 'Control']
        etanol_row = dist_df[dist_df['treatment'] == 'ETANOL']
        
        if ctrl_row.empty or etanol_row.empty:
            print(f"Warning: Missing data for {strain}")
            continue
        
        ctrl_val = ctrl_row[state_col].values[0]
        etanol_val = etanol_row[state_col].values[0]
        
        # Plot bars
        x = [0, 1]
        y = [ctrl_val, etanol_val]
        colors = [COLORS['control'], COLORS['etanol']]
        
        bars = ax.bar(x, y, width=0.6, color=colors,
                     edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # Value labels
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 1, f"{yi:.1f}%",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 55)
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'ETANOL'], fontsize=10)
        ax.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
        ax.set_title(f"{['A', 'B', 'C'][idx]}. {strain} - {description}",
                    fontsize=12, fontweight='bold', pad=10, loc='left')
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    
    fig.suptitle('Consistent Sedative Effect of Ethanolic Vehicle Across Strains',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    save_path = OUTPUT_PATH / 'Fig1_ETANOL_Sedative_Panel.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# FIG2: CBD PARADOXICAL EFFECT
# ============================================================================

def create_fig2_cbd_paradoxical():
    """CBD 30µM: stimulant vs sedative depending on genetic background."""
    print("Creating Fig2: CBD Paradoxical Effect...")
    
    stimulant_strains = [
        ('N2', 'Wild-type'),
        ('NL5901', 'Muscle α-synuclein')
    ]
    
    sedative_strains = [
        ('UA44', 'DA α-synuclein'),
        ('ROSELLA', 'Autophagy reporter')
    ]
    
    def get_data(strains_list):
        data = []
        for strain, description in strains_list:
            # Identify pause state
            pause_state = identify_pause_state(strain)
            state_col = f'state_{pause_state}'
            
            # Load distribution
            dist_df = load_state_distribution(strain)
            
            # Extract values
            ctrl_row = dist_df[dist_df['treatment'] == 'Control']
            cbd_row = dist_df[dist_df['treatment'] == 'CBD 30uM']
            
            if not ctrl_row.empty and not cbd_row.empty:
                data.append({
                    'label': f"{strain}\n{description}",
                    'control': ctrl_row[state_col].values[0],
                    'cbd': cbd_row[state_col].values[0]
                })
        return data
    
    stim_data = get_data(stimulant_strains)
    sed_data = get_data(sedative_strains)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    width = 0.35
    
    # PANEL A: STIMULANT
    x = np.arange(len(stim_data))
    ctrl_vals = [d['control'] for d in stim_data]
    cbd_vals = [d['cbd'] for d in stim_data]
    
    ax1.bar(x - width/2, ctrl_vals, width, label='Control',
           color=COLORS['control'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color=COLORS['stimulant'], edgecolor='black', linewidth=1.5, alpha=0.85)
    
    for i, d in enumerate(stim_data):
        ax1.text(i - width/2, d['control'] + 0.5, f"{d['control']:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.text(i + width/2, d['cbd'] + 0.5, f"{d['cbd']:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([d['label'] for d in stim_data], fontsize=10)
    ax1.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
    ax1.set_title('A. Stimulant Effect', fontsize=13, fontweight='bold', pad=10, loc='left')
    ax1.set_ylim(0, 55)
    ax1.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    
    # PANEL B: SEDATIVE
    x = np.arange(len(sed_data))
    ctrl_vals = [d['control'] for d in sed_data]
    cbd_vals = [d['cbd'] for d in sed_data]
    
    ax2.bar(x - width/2, ctrl_vals, width, label='Control',
           color=COLORS['control'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color=COLORS['sedative'], edgecolor='black', linewidth=1.5, alpha=0.85)
    
    for i, d in enumerate(sed_data):
        ax2.text(i - width/2, d['control'] + 0.5, f"{d['control']:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, d['cbd'] + 0.5, f"{d['cbd']:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d['label'] for d in sed_data], fontsize=10)
    ax2.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
    ax2.set_title('B. Sedative Effect', fontsize=13, fontweight='bold', pad=10, loc='left')
    ax2.set_ylim(0, 55)
    ax2.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    
    fig.suptitle('Paradoxical, Genetically-Determined Effect of CBD 30µM on Motor Behavior',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    save_path = OUTPUT_PATH / 'Fig2_CBD_Paradoxical_Effect.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# FIG3: NL5901 COMPENSATORY LÉVY FLIGHT
# ============================================================================

def create_fig3_nl5901_levy():
    """NL5901 shows baseline compensatory Lévy flight vs N2."""
    print("Creating Fig3: NL5901 Compensatory Lévy Flight...")
    
    levy_n2 = load_levy_flight('N2')
    levy_nl5901 = load_levy_flight('NL5901')
    
    treatments = ['Control', 'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'ETANOL']
    
    n2_alphas = []
    nl5901_alphas = []
    
    for treatment in treatments:
        n2_row = levy_n2[levy_n2['treatment'] == treatment]
        nl5901_row = levy_nl5901[levy_nl5901['treatment'] == treatment]
        
        n2_alphas.append(n2_row['alpha'].values[0] if not n2_row.empty else np.nan)
        nl5901_alphas.append(nl5901_row['alpha'].values[0] if not nl5901_row.empty else np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(treatments))
    
    # Plot lines
    ax.plot(x, nl5901_alphas, marker='o', markersize=10, linewidth=2.5,
           color='#EE7733', label='NL5901 (Muscle α-synuclein)', linestyle='-', zorder=3)
    ax.plot(x, n2_alphas, marker='s', markersize=8, linewidth=2,
           color='#999999', label='N2 (Wild-type)', linestyle='--', zorder=2, alpha=0.7)
    
    # Optimal Lévy reference
    ax.axhline(y=2.0, color=COLORS['levy_optimal'], linestyle=':', 
              linewidth=2, alpha=0.6, zorder=1)
    ax.axhspan(1.8, 2.2, alpha=0.1, color=COLORS['levy_optimal'], zorder=0)
    
    ax.text(0.3, 2.05, 'Optimal Lévy (α ≈ 2.0)', fontsize=9,
           color=COLORS['levy_optimal'], style='italic', fontweight='bold')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, fontsize=10)
    ax.set_ylabel('Lévy Flight Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Treatment', fontsize=12, fontweight='bold')
    ax.set_ylim(1.8, max(max(nl5901_alphas), max(n2_alphas)) + 0.3)
    ax.set_title('NL5901 Exhibits Baseline Compensatory Lévy Flight Pattern',
                fontsize=13, fontweight='bold', pad=15)
    
    # STANDARDIZED legend (no shadow!)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    save_path = OUTPUT_PATH / 'Fig3_NL5901_Levy_Compensatory.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# FIG4: LÉVY INDUCTION - CLEAR AND SIMPLE
# ============================================================================

def create_fig4_levy_induction():
    """
    Cannabinoid-induced Lévy flight.
    Shows N2 transition from Brownian (Control) to Lévy (CBDV).
    BR5270 shows baseline Lévy maintained/enhanced by CBD.
    """
    print("Creating Fig4: Lévy Flight Induction...")
    
    # Define data to show
    cases = [
        {'strain': 'N2', 'treatment': 'Control', 'label': 'N2\nControl'},
        {'strain': 'N2', 'treatment': 'CBDV 0.3uM', 'label': 'N2\nCBDV 0.3µM'},
        {'strain': 'BR5270', 'treatment': 'Control', 'label': 'BR5270\nControl'},
        {'strain': 'BR5270', 'treatment': 'CBD 0.3uM', 'label': 'BR5270\nCBD 0.3µM'},
    ]
    
    # Collect data
    data_points = []
    for case in cases:
        levy_df = load_levy_flight(case['strain'])
        row = levy_df[levy_df['treatment'] == case['treatment']]
        
        if not row.empty:
            alpha = row['alpha'].values[0]
            pval = row['p_value'].values[0]
            is_significant = pval < 0.05
            
            data_points.append({
                'label': case['label'],
                'alpha': alpha,
                'p_value': pval,
                'is_levy': is_significant,
                'strain': case['strain']
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    x = np.arange(len(data_points))
    colors = [COLORS['levy_optimal'] if d['is_levy'] else '#999999' for d in data_points]
    alphas = [d['alpha'] for d in data_points]
    
    bars = ax.bar(x, alphas, width=0.6, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels - SIMPLE
    for i, d in enumerate(data_points):
        # Alpha value above bar
        ax.text(i, d['alpha'] + 0.15, f"α = {d['alpha']:.2f}",
               ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # p-value below alpha
        if d['is_levy']:
            pval_text = f"p < 0.001" if d['p_value'] < 0.001 else f"p = {d['p_value']:.3f}"
            ax.text(i, d['alpha'] + 0.35, pval_text,
                   ha='center', va='bottom', fontsize=9,
                   color=COLORS['levy_optimal'], fontweight='bold')
        else:
            ax.text(i, d['alpha'] + 0.35, 'n.s.',
                   ha='center', va='bottom', fontsize=9,
                   style='italic', color='#666666')
    
    # Reference zone - Optimal Lévy
    ax.axhspan(1.8, 2.2, alpha=0.15, color=COLORS['levy_optimal'], zorder=0)
    ax.axhline(y=2.0, color=COLORS['levy_optimal'], linestyle=':', 
              linewidth=2, alpha=0.6, zorder=1)
    
    # Zone label
    ax.text(3.7, 2.0, 'Optimal Lévy (α ≈ 2.0)', fontsize=9,
           color=COLORS['levy_optimal'], style='italic', fontweight='bold',
           va='center', ha='right')
    
    # Transition arrows (only for N2: Brownian → Lévy)
    if not data_points[0]['is_levy'] and data_points[1]['is_levy']:
        ax.annotate('', xy=(1, data_points[1]['alpha'] - 0.1),
                   xytext=(0, data_points[0]['alpha'] - 0.1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['levy_optimal'],
                                 linewidth=2.5, linestyle='--'))
        ax.text(0.5, (data_points[0]['alpha'] + data_points[1]['alpha'])/2 + 0.4,
               'Cannabinoid\ninduces Lévy',
               ha='center', va='bottom', fontsize=9,
               color=COLORS['levy_optimal'], fontweight='bold')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([d['label'] for d in data_points], fontsize=10)
    ax.set_ylabel('Lévy Flight Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strain and Treatment', fontsize=12, fontweight='bold')
    ax.set_ylim(1.5, max(alphas) + 0.8)
    ax.set_title('Cannabinoid-Induced Transition to Lévy Flight Pattern',
                fontsize=13, fontweight='bold', pad=15)
    
    # STANDARDIZED legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['levy_optimal'], edgecolor='black',
              label='Significant Lévy Flight (p < 0.05)'),
        Patch(facecolor='#999999', edgecolor='black',
              label='Brownian (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
             frameon=True, framealpha=0.95, fontsize=9)
    
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    save_path = OUTPUT_PATH / 'Fig4_Levy_Induction_Cannabinoids.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# FIG5, FIG6, FIG7: HEATMAPS (ML, TDA, Comparison)
# ============================================================================

def create_heatmaps():
    """Generate classification heatmaps (ML and TDA) with consistent styling."""
    print("Creating Fig4, Fig5: Classification Heatmaps (Binary: Treatment vs Control)...")
    
    strains = ['N2', 'NL5901', 'UA44', 'BR5270', 'BR5271', 'TJ356', 'ROSELLA']
    # EXCLUDE Control from treatments shown
    treatments = ['CBD 0.3uM', 'CBD 3uM', 'CBD 30uM',
                 'CBDV 0.3uM', 'CBDV 3uM', 'CBDV 30uM', 'ETANOL']
    
    # Collect ML data
    ml_matrix = np.full((len(strains), len(treatments)), np.nan)  # Use NaN for missing data
    for i, strain in enumerate(strains):
        try:
            df = load_ml_classification(strain)
            for j, treatment in enumerate(treatments):
                row = df[df['treatment'] == treatment]
                if not row.empty:
                    # Try different possible column names
                    if 'accuracy' in row.columns:
                        ml_matrix[i, j] = row['accuracy'].values[0] * 100
                    elif 'test_accuracy' in row.columns:
                        ml_matrix[i, j] = row['test_accuracy'].values[0] * 100
        except Exception as e:
            print(f"Warning: Could not load ML data for {strain}: {e}")
    
    # Collect TDA data
    tda_matrix = np.full((len(strains), len(treatments)), np.nan)  # Use NaN for missing data
    for i, strain in enumerate(strains):
        try:
            df = load_tda_classification(strain)
            for j, treatment in enumerate(treatments):
                row = df[df['treatment'] == treatment]
                if not row.empty:
                    # Try different possible column names
                    if 'accuracy' in row.columns:
                        tda_matrix[i, j] = row['accuracy'].values[0] * 100
                    elif 'test_accuracy' in row.columns:
                        tda_matrix[i, j] = row['test_accuracy'].values[0] * 100
        except Exception as e:
            print(f"Warning: Could not load TDA data for {strain}: {e}")
    
    # Create 2 heatmaps (ML and TDA only, remove comparison)
    for fig_num, (title, data, cmap_color, filename) in enumerate([
        ('ML Screening: Binary Classification Accuracy (Treatment vs Control)', ml_matrix, 'Blues', 'Fig4_ML_Screening_Landscape.pdf'),
        ('TDA: Binary Classification Accuracy (Treatment vs Control)', tda_matrix, 'Oranges', 'Fig5_TDA_Landscape.pdf'),
    ], start=4):
        
        fig, ax = plt.subplots(figsize=(9, 7))
        
        vmin, vmax = 50, 100  # Binary classification: 50% = chance, 100% = perfect
        
        # Create custom annotations: show values only where data exists
        annot_array = np.empty_like(data, dtype=object)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    annot_array[i, j] = ''  # Empty string for NaN
                else:
                    annot_array[i, j] = f'{data[i, j]:.1f}'
        
        sns.heatmap(data, annot=annot_array, fmt='', cmap=cmap_color,
                   xticklabels=[t.replace('uM', 'µM') for t in treatments],
                   yticklabels=strains,
                   cbar_kws={'label': 'Classification Accuracy (%)'},
                   vmin=vmin, vmax=vmax,
                   linewidths=0.5, linecolor='white', ax=ax)
        
        ax.set_xlabel('Treatment (vs Control)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strain', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        save_path = OUTPUT_PATH / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved to {save_path}")
    
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("STANDARDIZED PAPER VISUALIZATION GENERATOR")
    print("="*70)
    print()
    print("Generating 5 figures:")
    print("  • Fig1: ETANOL Sedative Effect (Unsupervised States)")
    print("  • Fig2: CBD Paradoxical Effect (Unsupervised States)")
    print("  • Fig3: NL5901 Compensatory Lévy Flight")
    print("  • Fig4: ML Screening Binary Classification")
    print("  • Fig5: TDA Binary Classification")
    print()
    
    create_fig1_etanol_sedative()
    create_fig2_cbd_paradoxical()
    create_fig3_nl5901_levy()
    create_heatmaps()
    
    print("="*70)
    print("✓ ALL 5 FIGURES GENERATED WITH CONSISTENT FORMATTING")
    print("="*70)
    print()
    print("NOTE: Fig4 (Lévy Induction) was removed - BR5270 baseline")
    print("already shows Lévy pattern, making 'induction' message unclear.")
    print("Figures renumbered: Fig5→Fig4, Fig6→Fig5, Fig7 removed.")
