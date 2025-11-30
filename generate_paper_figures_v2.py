#!/usr/bin/env python3
"""
Paper Visualization Generator - SIMPLIFIED VERSION
Generate clean, publication-quality figures for C. elegans cannabinoid behavioral analysis.

All figures share consistent styling:
- Colorblind-friendly palette
- Same typography (Arial/Helvetica)
- Consistent legend placement (upper right, no shadow)
- Grid on Y-axis only
- 300 DPI PDF + PNG output
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
BASE_PATH = Path(__file__).parent
ANALYSIS_PATH = BASE_PATH / 'Analysis'
OUTPUT_PATH = BASE_PATH / 'Figures'
OUTPUT_PATH.mkdir(exist_ok=True)

# Consistent color palette (colorblind-friendly)
COLORS = {
    'control': '#999999',        # Gray
    'treatment': '#DE8F05',      # Orange
    'stimulant': '#029E73',      # Teal/Green
    'sedative': '#CC3311',       # Red
    'ml': '#0173B2',            # Blue
    'tda': '#DE8F05',           # Orange
    'levy_optimal': '#029E73',   # Teal (optimal foraging)
    'levy_brownian': '#999999',  # Gray (random walk)
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_state_distribution(strain: str) -> pd.DataFrame:
    """Load unsupervised state distribution data for a given strain."""
    file_path = ANALYSIS_PATH / 'Unsupervised_States' / strain / 'state_distribution.csv'
    return pd.read_csv(file_path)

def load_levy_flight(strain: str) -> pd.DataFrame:
    """Load Lévy flight analysis data for a given strain."""
    file_path = ANALYSIS_PATH / 'Levy_Flight' / strain / 'levy_flight_summary.csv'
    return pd.read_csv(file_path)

def identify_pause_state(strain: str) -> str:
    """
    Identify pause state (state with highest population in control).
    Returns state column name like 'state_0', 'state_1', etc.
    """
    df = load_state_distribution(strain)
    ctrl_row = df[df['treatment'] == 'Control']
    
    if ctrl_row.empty:
        return 'state_0'  # Default fallback
    
    # Find state_X columns
    state_cols = [col for col in df.columns if col.startswith('state_')]
    
    # Get the state with max percentage in control
    ctrl_vals = ctrl_row[state_cols].values[0]
    max_idx = np.argmax(ctrl_vals)
    
    return state_cols[max_idx]

# ============================================================================
# FIG2: CBD PARADOXICAL EFFECT - SIMPLIFIED
# ============================================================================

def create_fig2_cbd_paradoxical():
    """
    CBD 30µM has opposite effects depending on genetic background:
    - Stimulant (↓ pause) in N2, NL5901
    - Sedative (↑ pause) in UA44, ROSELLA
    """
    print("Creating Fig2: CBD Paradoxical Effect...")
    
    # Define strains for each panel
    stimulant_strains = [
        ('N2', 'N2\nWild-type'),
        ('NL5901', 'NL5901\nMuscle α-syn')
    ]
    
    sedative_strains = [
        ('UA44', 'UA44\nDA α-syn'),
        ('ROSELLA', 'ROSELLA\nAutophagy')
    ]
    
    # Collect data
    def get_data(strains_list):
        data = []
        for strain, label in strains_list:
            df = load_state_distribution(strain)
            pause_col = identify_pause_state(strain)
            
            ctrl = df[df['treatment'] == 'Control']
            cbd = df[df['treatment'] == 'CBD 30uM']
            
            if not ctrl.empty and not cbd.empty:
                ctrl_val = ctrl[pause_col].values[0]
                cbd_val = cbd[pause_col].values[0]
                delta = ((cbd_val - ctrl_val) / ctrl_val) * 100
                
                data.append({
                    'label': label,
                    'control': ctrl_val,
                    'cbd': cbd_val,
                    'delta': delta
                })
        return data
    
    stim_data = get_data(stimulant_strains)
    sed_data = get_data(sedative_strains)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    width = 0.35
    
    # PANEL A: STIMULANT
    x = np.arange(len(stim_data))
    ctrl_vals = [d['control'] for d in stim_data]
    cbd_vals = [d['cbd'] for d in stim_data]
    
    ax1.bar(x - width/2, ctrl_vals, width, label='Control',
           color=COLORS['control'], edgecolor='black', linewidth=1.5, alpha=0.9)
    ax1.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color=COLORS['stimulant'], edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Labels
    for i, d in enumerate(stim_data):
        ax1.text(i - width/2, d['control'] + 0.5, f"{d['control']:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, d['cbd'] + 0.5, f"{d['cbd']:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
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
           color=COLORS['control'], edgecolor='black', linewidth=1.5, alpha=0.9)
    ax2.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color=COLORS['sedative'], edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Labels
    for i, d in enumerate(sed_data):
        ax2.text(i - width/2, d['control'] + 0.5, f"{d['control']:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width/2, d['cbd'] + 0.5, f"{d['cbd']:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d['label'] for d in sed_data], fontsize=10)
    ax2.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
    ax2.set_title('B. Sedative Effect', fontsize=13, fontweight='bold', pad=10, loc='left')
    ax2.set_ylim(0, 55)
    ax2.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    
    # Main title
    fig.suptitle('Paradoxical, Genetically-Determined Effect of CBD 30µM on Motor Behavior',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    save_path = OUTPUT_PATH / 'Fig2_CBD_Paradoxical_Effect.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# FIG4: LÉVY INDUCTION - SIMPLIFIED
# ============================================================================

def create_fig4_levy_induction():
    """
    Low-dose cannabinoids induce transition from Brownian (α~3) to Lévy (α~2) flight.
    Shows N2 + CBDV and BR5270 + CBD examples.
    """
    print("Creating Fig4: Lévy Flight Induction...")
    
    # Cases to compare (Control vs Treatment)
    cases = [
        {'strain': 'N2', 'ctrl_treat': 'Control', 'cbd_treat': 'CBDV 0.3uM',
         'label_ctrl': 'N2\nControl', 'label_cbd': 'N2\nCBDV 0.3µM'},
        {'strain': 'BR5270', 'ctrl_treat': 'Control', 'cbd_treat': 'CBD 0.3uM',
         'label_ctrl': 'BR5270\nControl', 'label_cbd': 'BR5270\nCBD 0.3µM'},
    ]
    
    # Collect data
    data_points = []
    for case in cases:
        df = load_levy_flight(case['strain'])
        
        # Control
        ctrl_row = df[df['treatment'] == case['ctrl_treat']]
        if not ctrl_row.empty:
            data_points.append({
                'label': case['label_ctrl'],
                'alpha': ctrl_row['alpha'].values[0],
                'p_value': ctrl_row['p_value'].values[0],
                'is_levy': ctrl_row['p_value'].values[0] < 0.05
            })
        
        # Treatment
        cbd_row = df[df['treatment'] == case['cbd_treat']]
        if not cbd_row.empty:
            data_points.append({
                'label': case['label_cbd'],
                'alpha': cbd_row['alpha'].values[0],
                'p_value': cbd_row['p_value'].values[0],
                'is_levy': cbd_row['p_value'].values[0] < 0.05
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(data_points))
    colors = [COLORS['levy_optimal'] if d['is_levy'] else COLORS['levy_brownian'] 
              for d in data_points]
    alphas = [d['alpha'] for d in data_points]
    
    bars = ax.bar(x, alphas, width=0.6, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels
    for i, d in enumerate(data_points):
        if d['is_levy']:
            ax.text(i, d['alpha'] + 0.1, 
                   f"α = {d['alpha']:.3f}\np < {d['p_value']:.0e}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i, d['alpha'] + 0.5, '✓ Lévy Flight',
                   ha='center', va='bottom', fontsize=9,
                   color=COLORS['levy_optimal'], fontweight='bold')
        else:
            ax.text(i, d['alpha'] + 0.1,
                   f"α = {d['alpha']:.3f}\nn.s.",
                   ha='center', va='bottom', fontsize=9,
                   style='italic', color='#666666')
            ax.text(i, d['alpha'] + 0.4, 'Brownian',
                   ha='center', va='bottom', fontsize=9,
                   color='#999999', style='italic')
    
    # Add reference zones
    ax.axhspan(1.8, 2.2, alpha=0.15, color=COLORS['levy_optimal'], zorder=0)
    ax.axhline(y=3.0, color='#CC3311', linestyle=':', linewidth=2, alpha=0.6)
    
    # Labels on zones
    ax.text(3.5, 2.0, 'Optimal Lévy\n(α ≈ 2.0)',
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=COLORS['levy_optimal'], linewidth=1.5, alpha=0.9))
    ax.text(0.5, 3.15, 'Brownian (α ≈ 3.0)',
           fontsize=9, color='#CC3311', style='italic')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([d['label'] for d in data_points], fontsize=10)
    ax.set_ylabel('Lévy Flight Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_ylim(1.5, 3.8)
    ax.set_title('Cannabinoid-Induced Transition from Brownian to Lévy Flight Pattern',
                fontsize=13, fontweight='bold', pad=15)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['levy_optimal'], edgecolor='black',
              label='Significant Lévy Flight (p < 0.05)'),
        Patch(facecolor=COLORS['levy_brownian'], edgecolor='black',
              label='Brownian (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
             frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    # Save
    save_path = OUTPUT_PATH / 'Fig4_Levy_Induction_Cannabinoids.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}\n")
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("REGENERATING FIG2 AND FIG4 - SIMPLIFIED VERSIONS")
    print("="*70)
    print()
    
    create_fig2_cbd_paradoxical()
    create_fig4_levy_induction()
    
    print("="*70)
    print("✓ FIGURES REGENERATED SUCCESSFULLY")
    print("="*70)
