#!/usr/bin/env python
# coding: utf-8

"""
Paper Visualization Generator - High Priority Figures
=====================================================

This script generates the 4 high-priority visualizations for the C. elegans
behavioral analysis paper. Designed for publication-quality output.

Author: Generated for paper final_paper_section_v10.md
Date: October 9, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Colorblind-friendly palette
COLORS = {
    'control': '#0173B2',      # Blue
    'treatment': '#DE8F05',    # Orange
    'etanol': '#DE8F05',       # Orange
    'stimulant': '#0173B2',    # Blue
    'sedative': '#CC3311',     # Red
    'tau_agg': '#CC3311',      # Red (diseased)
    'tau_nonagg': '#009988',   # Teal (healthy)
    'ml': '#0173B2',           # Blue
    'tda': '#DE8F05',          # Orange
    'levy_optimal': '#029E73', # Green
    'primary': ['#0173B2', '#DE8F05', '#029E73', '#CC3311', '#EE7733', '#999999']
}

# Paths
BASE_PATH = Path('results/Analysis')
OUTPUT_PATH = Path('results/Figures')
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATA LOADING UTILITIES
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
# VISUALIZATION 1: ETANOL SEDATIVE EFFECT PANEL
# ============================================================================

def create_etanol_sedative_panel(save_path: str = None):
    """
    Create a 3-panel figure showing consistent sedative effect of ETANOL
    across N2, UA44, and ROSELLA strains.
    
    This visualization supports Section 1.3, bullet 1 of the paper.
    """
    print("Creating VISUALIZATION 1: ETANOL Sedative Effect Panel...")
    
    # Define strains and their descriptions
    strains_info = {
        'N2': {
            'name': 'N2',
            'description': 'Wild-type',
            'pause_state': None,  # Will be identified
            'label': 'A'
        },
        'UA44': {
            'name': 'UA44',
            'description': 'Dopaminergic α-syn',
            'pause_state': None,
            'label': 'B'
        },
        'ROSELLA': {
            'name': 'ROSELLA',
            'description': 'Autophagy reporter',
            'pause_state': None,
            'label': 'C'
        }
    }
    
    # Collect data
    data_for_plot = []
    for strain_key, info in strains_info.items():
        # Identify pause state
        pause_state = identify_pause_state(strain_key)
        info['pause_state'] = pause_state
        state_col = f'state_{pause_state}'
        
        # Load distribution
        dist_df = load_state_distribution(strain_key)
        
        # Extract Control and ETANOL values
        control_row = dist_df[dist_df['treatment'] == 'Control']
        etanol_row = dist_df[dist_df['treatment'] == 'ETANOL']
        
        if control_row.empty or etanol_row.empty:
            print(f"Warning: Missing Control or ETANOL data for {strain_key}")
            continue
        
        control_val = control_row[state_col].values[0]
        etanol_val = etanol_row[state_col].values[0]
        
        data_for_plot.append({
            'strain': strain_key,
            'description': info['description'],
            'label': info['label'],
            'pause_state': pause_state,
            'control': control_val,
            'etanol': etanol_val,
            'change': etanol_val - control_val
        })
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Consistent Sedative Effect of Ethanolic Vehicle Across Strains', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    for idx, data in enumerate(data_for_plot):
        ax = axes[idx]
        
        # Data for this strain
        x = [0, 1]
        y = [data['control'], data['etanol']]
        
        # Create bars
        bars = ax.bar(x, y, width=0.6, 
                     color=[COLORS['control'], COLORS['etanol']],
                     edgecolor='black', linewidth=1.2, alpha=0.85)
        
        # Add value labels on top of bars
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi + 1, f"{yi:.1f}%", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting - ESTANDARIZADA para comparabilidad entre paneles
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 50)  # Escala estandarizada 0-50%
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'ETANOL'], fontsize=10)
        ax.set_ylabel('% Population in Pause State', fontsize=10)
        
        # Panel label - separado del título del strain
        ax.text(0.02, 0.98, data['label'], transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='black', linewidth=1.5))
        ax.set_title(f"{data['strain']} - {data['description']}", 
                    fontsize=10, pad=15)
        
        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig1_ETANOL_Sedative_Panel.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 3: CBD PARADOXICAL EFFECT (STIMULANT VS SEDATIVE)
# ============================================================================

def create_cbd_paradoxical_effect(save_path: str = None):
    """
    Create visualization showing the paradoxical, strain-dependent effect of CBD 30µM:
    - Stimulant in N2 and NL5901
    - Sedative in UA44 and ROSELLA
    
    KEY MESSAGE: CBD effect is genetically determined - same dose causes opposite effects.
    
    Paper quote: "CBD 30uM had a striking strain-dependent effect. In wild-type N2, 
    CBD 30uM had a mild stimulant effect. In stark contrast, the same compound had 
    a strong sedative effect in the UA44 and ROSELLA backgrounds."
    """
    print("Creating VISUALIZATION 3: CBD Paradoxical Effect (Stimulant vs Sedative)...")
    
    # Define strains and their characteristics
    strains_info = {
        'N2': {'description': 'Wild-type', 'category': 'Stimulant'},
        'NL5901': {'description': 'Muscle α-synuclein', 'category': 'Stimulant'},
        'UA44': {'description': 'DA α-synuclein', 'category': 'Sedative'},
        'ROSELLA': {'description': 'Autophagy reporter', 'category': 'Sedative'}
    }
    
    # Collect data
    data_points = []
    for strain, info in strains_info.items():
        pause_state = identify_pause_state(strain)
        dist_df = load_state_distribution(strain)
        
        # Get Control and CBD 30µM values
        control_row = dist_df[dist_df['treatment'] == 'Control']
        cbd_row = dist_df[dist_df['treatment'] == 'CBD 30uM']
        
        if not control_row.empty and not cbd_row.empty:
            state_col = f'state_{pause_state}'
            control_val = control_row[state_col].values[0]
            cbd_val = cbd_row[state_col].values[0]
            change = cbd_val - control_val
            
            data_points.append({
                'strain': strain,
                'description': info['description'],
                'category': info['category'],
                'control': control_val,
                'cbd_30': cbd_val,
                'change': change
            })
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Paradoxical, Genetically-Determined Effect of CBD 30µM on Motor Behavior', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Panel A: Stimulant effect (N2, NL5901)
    ax = axes[0]
    stimulant_data = [d for d in data_points if d['category'] == 'Stimulant']
    
    x = np.arange(len(stimulant_data))
    width = 0.35
    
    control_vals = [d['control'] for d in stimulant_data]
    cbd_vals = [d['cbd_30'] for d in stimulant_data]
    
    ax.bar(x - width/2, control_vals, width, label='Control', 
           color='#999999', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color='#029E73', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for i, (ctrl, cbd) in enumerate(zip(control_vals, cbd_vals)):
        ax.text(i - width/2, ctrl + 1, f'{ctrl:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, cbd + 1, f'{cbd:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add change indicator
        change = cbd - ctrl
        arrow_y = max(ctrl, cbd) + 3
        if change < 0:  # Decrease = stimulant (less paused)
            ax.annotate('', xy=(i, arrow_y), xytext=(i, arrow_y + 5),
                       arrowprops=dict(arrowstyle='<-', color='green', lw=2.5))
            ax.text(i, arrow_y + 6, f'{change:.1f}%\nStimulant', 
                   ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['strain']}\n{d['description']}" for d in stimulant_data], fontsize=9)
    ax.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 50)
    ax.set_title('A. Stimulant Effect', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Panel B: Sedative effect (UA44, ROSELLA)
    ax = axes[1]
    sedative_data = [d for d in data_points if d['category'] == 'Sedative']
    
    x = np.arange(len(sedative_data))
    
    control_vals = [d['control'] for d in sedative_data]
    cbd_vals = [d['cbd_30'] for d in sedative_data]
    
    ax.bar(x - width/2, control_vals, width, label='Control',
           color='#999999', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x + width/2, cbd_vals, width, label='CBD 30µM',
           color='#CC3311', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for i, (ctrl, cbd) in enumerate(zip(control_vals, cbd_vals)):
        ax.text(i - width/2, ctrl + 1, f'{ctrl:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, cbd + 1, f'{cbd:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add change indicator
        change = cbd - ctrl
        arrow_y = max(ctrl, cbd) + 3
        if change > 0:  # Increase = sedative (more paused)
            ax.annotate('', xy=(i, arrow_y + 5), xytext=(i, arrow_y),
                       arrowprops=dict(arrowstyle='<-', color='red', lw=2.5))
            ax.text(i, arrow_y + 6, f'+{change:.1f}%\nSedative', 
                   ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['strain']}\n{d['description']}" for d in sedative_data], fontsize=9)
    ax.set_ylabel('% Population in Pause State', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 50)
    ax.set_title('B. Sedative Effect', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig2_CBD_Paradoxical_Effect.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 3: NL5901 COMPENSATORY LÉVY FLIGHT
# ============================================================================

def create_nl5901_levy_comparison(save_path: str = None):
    """
    Create a line plot comparing Lévy flight exponents (alpha) between
    NL5901 (muscle synucleinopathy) and N2 (wild-type) across treatments.
    
    This visualization supports Section 2.3, final paragraph of the paper.
    """
    print("Creating VISUALIZATION 6: NL5901 Compensatory Lévy Flight...")
    
    # Load data
    levy_n2 = load_levy_flight('N2')
    levy_nl5901 = load_levy_flight('NL5901')
    
    # Define treatment order
    treatment_order = ['Control', 'CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'ETANOL']
    
    # Extract alpha values
    n2_alphas = []
    nl5901_alphas = []
    nl5901_pvals = []
    
    for treatment in treatment_order:
        n2_row = levy_n2[levy_n2['treatment'] == treatment]
        nl5901_row = levy_nl5901[levy_nl5901['treatment'] == treatment]
        
        if not n2_row.empty:
            n2_alphas.append(n2_row['alpha'].values[0])
        else:
            n2_alphas.append(np.nan)
        
        if not nl5901_row.empty:
            nl5901_alphas.append(nl5901_row['alpha'].values[0])
            nl5901_pvals.append(nl5901_row['p_value'].values[0])
        else:
            nl5901_alphas.append(np.nan)
            nl5901_pvals.append(np.nan)
    
    # Create figure - aumentado para mejor legibilidad
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    x = np.arange(len(treatment_order))
    
    # Plot lines
    ax.plot(x, nl5901_alphas, marker='o', markersize=10, linewidth=2.5,
           color='#EE7733', label='NL5901 (muscle α-synuclein)', 
           linestyle='-', zorder=3)
    ax.plot(x, n2_alphas, marker='s', markersize=8, linewidth=2,
           color='#999999', label='N2 (wild-type)',
           linestyle='--', zorder=2, alpha=0.7)
    
    # Optimal Lévy reference line
    ax.axhline(y=2.0, color=COLORS['levy_optimal'], linestyle=':', 
              linewidth=2, label='Optimal Lévy (α = 2.0)', zorder=1, alpha=0.6)
    
    # Lévy flight zone (shaded region)
    ax.axhspan(1.8, 2.2, alpha=0.1, color=COLORS['levy_optimal'], zorder=0)
    
    # Add optimal zone annotation
    ax.text(0.2, 2.0, 'Optimal Lévy', 
           fontsize=8, va='center', ha='left',
           color=COLORS['levy_optimal'], style='italic', 
           fontweight='bold', alpha=0.9,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=COLORS['levy_optimal'], alpha=0.7, linewidth=1))
    
    # Add stability band for NL5901 to highlight persistent pattern
    nl5901_mean = np.nanmean(nl5901_alphas)
    nl5901_std = np.nanstd(nl5901_alphas)
    ax.axhspan(nl5901_mean - nl5901_std, nl5901_mean + nl5901_std,
              alpha=0.12, color='#EE7733', zorder=0)
    
    # Annotate the stable pattern
    ax.text(2.5, nl5901_mean - nl5901_std - 0.08, 
           f'NL5901 stable: α={nl5901_mean:.2f}±{nl5901_std:.2f}',
           ha='center', va='top', fontsize=8.5,
           bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                    edgecolor='#EE7733', linewidth=1.5, alpha=0.95))
    
    # Highlight significant points for NL5901
    for i, (alpha, pval) in enumerate(zip(nl5901_alphas, nl5901_pvals)):
        if not np.isnan(pval) and pval < 0.05:
            significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            ax.text(x[i], alpha + 0.1, significance, ha='center', va='bottom',
                   fontsize=12, color='#EE7733', fontweight='bold')
    
    # Special annotation for Control (baseline Lévy)
    control_idx = 0
    control_alpha = nl5901_alphas[control_idx]
    control_pval = nl5901_pvals[control_idx]
    
    ax.annotate(f'Baseline Lévy\nα={control_alpha:.3f}, p<{control_pval:.1e}',
               xy=(control_idx, control_alpha), 
               xytext=(control_idx + 1.2, control_alpha - 0.15),
               fontsize=8, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEECC', 
                        edgecolor='#EE7733', linewidth=1.5, alpha=0.95),
               arrowprops=dict(arrowstyle='->', color='#EE7733', 
                             linewidth=1.5, shrinkA=0, shrinkB=5))
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(' ', '\n') for t in treatment_order], fontsize=9)
    ax.set_ylabel('α exponent (Lévy flight)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Treatment', fontsize=12, fontweight='bold')
    # Ajustar límite superior para que todos los puntos queden dentro
    y_max_data = max(max(nl5901_alphas), max(n2_alphas))
    ax.set_ylim(1.8, y_max_data + 0.3)  # Margen dinámico
    ax.set_title('NL5901 Exhibits Baseline Compensatory Lévy Flight Pattern',
                fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=9,
             framealpha=0.95, edgecolor='gray')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig3_NL5901_Levy_Compensatory.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 4: LÉVY FLIGHT INDUCTION BY CANNABINOIDS
# ============================================================================

def create_levy_induction_by_cannabinoids(save_path: str = None):
    """
    Create visualization showing cannabinoid-induced transition from Brownian to Lévy flight.
    
    KEY MESSAGE: Low-dose cannabinoids qualitatively alter exploratory strategy from 
    random (Brownian) to optimal (Lévy flight) pattern.
    
    Paper quote: "Certain low-dose cannabinoids can qualitatively alter the worms' 
    exploratory behavior, pushing it from a standard Brownian-like pattern towards 
    a more efficient, Lévy-like strategy."
    
    Data:
    - N2 + CBDV 0.3µM: α=2.675, p<0.001 (Lévy)
    - BR5270 + CBD 0.3µM: α=2.851, p<1e-38 (Lévy)
    - Controls: not significant (Brownian)
    """
    print("Creating VISUALIZATION 4: Lévy Flight Induction by Cannabinoids...")
    
    # Define cases to compare
    cases = [
        {'strain': 'N2', 'treatment': 'Control', 'label': 'N2\nControl', 'category': 'Brownian'},
        {'strain': 'N2', 'treatment': 'CBDV 0.3uM', 'label': 'N2\nCBDV 0.3µM', 'category': 'Lévy'},
        {'strain': 'BR5270', 'treatment': 'Control', 'label': 'BR5270\nControl', 'category': 'Brownian'},
        {'strain': 'BR5270', 'treatment': 'CBD 0.3uM', 'label': 'BR5270\nCBD 0.3µM', 'category': 'Lévy'},
    ]
    
    # Collect data
    data_points = []
    for case in cases:
        try:
            levy_df = load_levy_flight(case['strain'])
            row = levy_df[levy_df['treatment'] == case['treatment']]
            
            if not row.empty:
                alpha = row['alpha'].values[0]
                pval = row['p_value'].values[0]
                is_significant = pval < 0.05 if not np.isnan(pval) else False
                
                data_points.append({
                    'label': case['label'],
                    'alpha': alpha,
                    'p_value': pval,
                    'significant': is_significant,
                    'category': case['category'] if is_significant else 'Brownian'
                })
        except Exception as e:
            print(f"Warning: Could not load data for {case['strain']} {case['treatment']}: {e}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    x = np.arange(len(data_points))
    colors = [COLORS['levy_optimal'] if d['significant'] else '#999999' for d in data_points]
    alphas_display = [d['alpha'] if d['alpha'] < 10 else 3.5 for d in data_points]  # Cap display
    
    bars = ax.bar(x, alphas_display, width=0.6, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels
    for i, d in enumerate(data_points):
        if d['significant']:
            # Significant Lévy
            ax.text(i, d['alpha'] + 0.1, f"α = {d['alpha']:.3f}\np < {d['p_value']:.0e}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Add checkmark
            ax.text(i, d['alpha'] + 0.6, '✓ Lévy Flight', ha='center', va='bottom',
                   fontsize=9, color=COLORS['levy_optimal'], fontweight='bold')
        else:
            # Not significant / Brownian
            ax.text(i, alphas_display[i] + 0.1, f"α = {d['alpha']:.3f}\nn.s.",
                   ha='center', va='bottom', fontsize=9, style='italic', color='#666666')
            ax.text(i, alphas_display[i] + 0.5, 'Brownian', ha='center', va='bottom',
                   fontsize=9, color='#999999', style='italic')
    
    # Add Lévy flight optimal zone (1.8-2.2)
    ax.axhspan(1.8, 2.2, alpha=0.15, color=COLORS['levy_optimal'], zorder=0)
    ax.text(len(x) - 0.5, 2.0, 'Optimal Lévy\nzone (α ≈ 2.0)', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor=COLORS['levy_optimal'], linewidth=2, alpha=0.9))
    
    # Add Brownian reference line
    ax.axhline(y=3.0, color='#CC3311', linestyle=':', linewidth=2, alpha=0.6)
    ax.text(0.5, 3.1, 'Brownian (α ≈ 3.0)', fontsize=9, color='#CC3311', style='italic')
    
    # Add transition arrows
    # N2: Control → CBDV
    ax.annotate('', xy=(1, 2.7), xytext=(0, 3.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['levy_optimal'], 
                             linewidth=3, linestyle='--'))
    ax.text(0.5, 3.5, 'Cannabinoid\ninduces Lévy', ha='center', va='bottom',
           fontsize=8, color=COLORS['levy_optimal'], fontweight='bold')
    
    # BR5270: Control → CBD
    ax.annotate('', xy=(3, 2.85), xytext=(2, 3.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['levy_optimal'], 
                             linewidth=3, linestyle='--'))
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([d['label'] for d in data_points], fontsize=10)
    ax.set_ylabel('Lévy Flight Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_ylim(1.5, 4.0)
    ax.set_title('Cannabinoid-Induced Transition from Brownian to Lévy Flight Pattern',
                fontsize=13, fontweight='bold', pad=15)
    
    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['levy_optimal'], edgecolor='black', label='Significant Lévy Flight (p < 0.05)'),
        Patch(facecolor='#999999', edgecolor='black', label='Brownian (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig4_Levy_Induction_Cannabinoids.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 5: ML SCREENING CLASSIFICATION HEATMAP
# ============================================================================

def create_ml_screening_heatmap(save_path: str = None):
    """
    Create heatmap showing ML Screening classification accuracy across strains and treatments.
    """
    print("Creating VISUALIZATION 5: ML Screening Classification Landscape...")
    
    # Define strains
    strains = ['N2', 'NL5901', 'UA44', 'BR5270', 'BR5271', 'TJ356', 'ROSELLA']
    
    # Collect ML data
    ml_data = {}
    for strain in strains:
        try:
            ml_df = load_ml_classification(strain)
            ml_data[strain] = ml_df
        except FileNotFoundError:
            print(f"Warning: ML data not found for {strain}")
            ml_data[strain] = None
    
    # Get all unique treatments
    all_treatments = set()
    for df in ml_data.values():
        if df is not None:
            all_treatments.update(df['treatment'].values)
    
    # Filter to key treatments
    key_treatments = ['CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'CBDV 3uM', 
                     'CBDV 30uM', 'ETANOL', 'Total_Extract_CBD']
    treatments = [t for t in key_treatments if t in all_treatments]
    
    # Build matrix
    ml_matrix = np.full((len(treatments), len(strains)), np.nan)
    
    for i, treatment in enumerate(treatments):
        for j, strain in enumerate(strains):
            if ml_data[strain] is not None:
                ml_row = ml_data[strain][ml_data[strain]['treatment'] == treatment]
                if not ml_row.empty:
                    ml_matrix[i, j] = ml_row['accuracy'].values[0] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    im = ax.imshow(ml_matrix, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax.set_xticks(range(len(strains)))
    ax.set_xticklabels(strains, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(treatments)))
    ax.set_yticklabels([t.replace('_', ' ') for t in treatments], fontsize=11)
    ax.set_title('ML Screening: Kinematic Feature Classification Accuracy',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add values to cells
    for i in range(len(treatments)):
        for j in range(len(strains)):
            val = ml_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 60 else 'black'
                fontsize = 10 if val > 85 else 9
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=fontsize, color=color, fontweight='bold')
                # Highlight exceptional cases (>85%)
                if val > 85:
                    rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                        fill=False, edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                
                # Mark key paper case: NL5901 + CBD 3uM (95.2%)
                if strains[j] == 'NL5901' and treatments[i] == 'CBD 3uM':
                    ax.text(j - 0.38, i + 0.38, '★', ha='center', va='center',
                           fontsize=18, color='gold', fontweight='bold', zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Legend
    fig.text(0.5, 0.01, '★ = Key finding (NL5901 + CBD 3µM: 95.2%)', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig5_ML_Screening_Landscape.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 6: TDA CLASSIFICATION HEATMAP
# ============================================================================

def create_tda_classification_heatmap(save_path: str = None):
    """
    Create heatmap showing TDA classification accuracy across strains and treatments.
    """
    print("Creating VISUALIZATION 6: TDA Classification Landscape...")
    
    # Define strains
    strains = ['N2', 'NL5901', 'UA44', 'BR5270', 'BR5271', 'TJ356', 'ROSELLA']
    
    # Collect TDA data
    tda_data = {}
    for strain in strains:
        try:
            tda_df = load_tda_classification(strain)
            tda_data[strain] = tda_df
        except FileNotFoundError:
            print(f"Warning: TDA data not found for {strain}")
            tda_data[strain] = None
    
    # Get all unique treatments
    all_treatments = set()
    for df in tda_data.values():
        if df is not None:
            all_treatments.update(df['treatment'].values)
    
    # Filter to key treatments
    key_treatments = ['CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'CBDV 3uM', 
                     'CBDV 30uM', 'ETANOL', 'Total_Extract_CBD']
    treatments = [t for t in key_treatments if t in all_treatments]
    
    # Build matrix
    tda_matrix = np.full((len(treatments), len(strains)), np.nan)
    
    for i, treatment in enumerate(treatments):
        for j, strain in enumerate(strains):
            if tda_data[strain] is not None:
                tda_row = tda_data[strain][tda_data[strain]['treatment'] == treatment]
                if not tda_row.empty:
                    tda_matrix[i, j] = tda_row['accuracy'].values[0] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    im = ax.imshow(tda_matrix, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax.set_xticks(range(len(strains)))
    ax.set_xticklabels(strains, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(treatments)))
    ax.set_yticklabels([t.replace('_', ' ') for t in treatments], fontsize=11)
    ax.set_title('TDA: Topological Feature Classification Accuracy',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add values to cells
    for i in range(len(treatments)):
        for j in range(len(strains)):
            val = tda_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 60 else 'black'
                fontsize = 10 if val > 85 else 9
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=fontsize, color=color, fontweight='bold')
                # Highlight exceptional cases (>85%)
                if val > 85:
                    rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                        fill=False, edgecolor='black', linewidth=3)
                    ax.add_patch(rect)
                
                # Mark key paper case: UA44 + CBD 3uM (86.7%)
                if strains[j] == 'UA44' and treatments[i] == 'CBD 3uM':
                    ax.text(j - 0.38, i + 0.38, '★', ha='center', va='center',
                           fontsize=18, color='gold', fontweight='bold', zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Legend
    fig.text(0.5, 0.01, '★ = Key finding (UA44 + CBD 3µM: 86.7%)', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig6_TDA_Landscape.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# VISUALIZATION 7: METHOD COMPARISON (ML - TDA)
# ============================================================================

def create_method_comparison_heatmap(save_path: str = None):
    """
    Create heatmap showing difference in classification accuracy (ML - TDA).
    """
    print("Creating VISUALIZATION 7: Method Comparison (ML vs TDA)...")
    
    # Define strains
    strains = ['N2', 'NL5901', 'UA44', 'BR5270', 'BR5271', 'TJ356', 'ROSELLA']
    
    # Collect data
    ml_data = {}
    tda_data = {}
    
    for strain in strains:
        try:
            ml_data[strain] = load_ml_classification(strain)
        except FileNotFoundError:
            ml_data[strain] = None
        
        try:
            tda_data[strain] = load_tda_classification(strain)
        except FileNotFoundError:
            tda_data[strain] = None
    
    # Get all unique treatments
    all_treatments = set()
    for df in ml_data.values():
        if df is not None:
            all_treatments.update(df['treatment'].values)
    
    # Filter to key treatments
    key_treatments = ['CBD 0.3uM', 'CBD 3uM', 'CBD 30uM', 'CBDV 0.3uM', 'CBDV 3uM', 
                     'CBDV 30uM', 'ETANOL', 'Total_Extract_CBD']
    treatments = [t for t in key_treatments if t in all_treatments]
    
    # Build matrices
    ml_matrix = np.full((len(treatments), len(strains)), np.nan)
    tda_matrix = np.full((len(treatments), len(strains)), np.nan)
    
    for i, treatment in enumerate(treatments):
        for j, strain in enumerate(strains):
            if ml_data[strain] is not None:
                ml_row = ml_data[strain][ml_data[strain]['treatment'] == treatment]
                if not ml_row.empty:
                    ml_matrix[i, j] = ml_row['accuracy'].values[0] * 100
            
            if tda_data[strain] is not None:
                tda_row = tda_data[strain][tda_data[strain]['treatment'] == treatment]
                if not tda_row.empty:
                    tda_matrix[i, j] = tda_row['accuracy'].values[0] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Difference matrix
    diff_matrix = ml_matrix - tda_matrix
    
    # Heatmap
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-30, vmax=40)
    ax.set_xticks(range(len(strains)))
    ax.set_xticklabels(strains, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(treatments)))
    ax.set_yticklabels([t.replace('_', ' ') for t in treatments], fontsize=11)
    ax.set_title('Method Comparison: ML Advantage over TDA (Δ Accuracy)',
                fontsize=14, fontweight='bold', pad=15)
    
    # Add values to cells
    for i in range(len(treatments)):
        for j in range(len(strains)):
            val = diff_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 20 else 'black'
                ax.text(j, i, f'{val:+.0f}', ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference (ML - TDA, %)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Legend
    fig.text(0.5, 0.01, 'Red = TDA better | Blue = ML better', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    if save_path is None:
        save_path = OUTPUT_PATH / 'Fig7_Method_Comparison.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"✓ Saved to {save_path}")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all paper figures covering all 4 methodologies."""
    print("\n" + "="*70)
    print("PAPER VISUALIZATION GENERATOR - ALL METHODOLOGIES")
    print("="*70 + "\n")
    print("Generating 7 figures with consistent style:")
    print("  • Fig1-2: Unsupervised States")
    print("  • Fig3-4: Lévy Flight")
    print("  • Fig5: ML Screening")
    print("  • Fig6: TDA")
    print("  • Fig7: Method Comparison")
    print()
    print(f"Output directory: {OUTPUT_PATH.absolute()}\n")
    
    try:
        # METHODOLOGY 1: UNSUPERVISED STATES
        print("="*70)
        print("METHODOLOGY 1: UNSUPERVISED BEHAVIORAL STATES")
        print("="*70)
        fig1 = create_etanol_sedative_panel()
        print()
        
        fig2 = create_cbd_paradoxical_effect()
        print()
        
        # METHODOLOGY 2: LÉVY FLIGHT
        print("="*70)
        print("METHODOLOGY 2: LÉVY FLIGHT ANALYSIS")
        print("="*70)
        fig3 = create_nl5901_levy_comparison()
        print()
        
        fig4 = create_levy_induction_by_cannabinoids()
        print()
        
        # METHODOLOGY 3: ML SCREENING
        print("="*70)
        print("METHODOLOGY 3: ML SCREENING (Kinematic Features)")
        print("="*70)
        fig5 = create_ml_screening_heatmap()
        print()
        
        # METHODOLOGY 4: TDA
        print("="*70)
        print("METHODOLOGY 4: TDA (Topological Features)")
        print("="*70)
        fig6 = create_tda_classification_heatmap()
        print()
        
        # COMPARISON
        print("="*70)
        print("METHODOLOGY COMPARISON")
        print("="*70)
        fig7 = create_method_comparison_heatmap()
        print()
        
        print("="*70)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*70)
        print(f"\nFiles saved to: {OUTPUT_PATH.absolute()}")
        print("\nGenerated files (covering all 4 methodologies):")
        for f in sorted(OUTPUT_PATH.glob('Fig*.pdf')):
            print(f"  ✓ {f.name}")
        print("\nMethodology coverage:")
        print("  • Unsupervised States: Fig1, Fig2")
        print("  • Lévy Flight: Fig3, Fig4 (pending)")
        print("  • ML Screening: Fig5")
        print("  • TDA: Fig6")
        print("  • Comparison: Fig7")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
