#!/usr/bin/env python
# coding: utf-8

"""
C. elegans Trajectory Visualizer and Animator - Batch Processor
==============================================================
Creates animations and static visualizations of worm movement trajectories
from WMicrotracker SMART CSV data files for multiple experiments.

This script processes all CSV files within a specified data directory,
groups them by strain and treatment, and generates a set of visualizations
for each group.

Author: Behavioral Analysis Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
import warnings
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ===========================================================================
# PUBLICATION-QUALITY STYLING (matching Statistical_Analysis.py)
# ===========================================================================

plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'patch.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True
})

# COMPREHENSIVE COLOR PALETTE - PUBLICATION QUALITY
PRIMARY_COLORS = {
    'blue': '#2E86AB',
    'orange': '#FD7E14',
    'green': '#198754',
    'red': '#DC3545',
    'purple': '#6F42C1',
    'gray': '#6C757D',
    'yellow': '#FFC107',
    'teal': '#20C997',
    'pink': '#E91E63',
    'brown': '#795548',
    'cyan': '#17A2B8',
    'magenta': '#E83E8C'
}

# Strain colors
STRAIN_COLORS = {
    'N2': PRIMARY_COLORS['blue'],
    'NL5901': PRIMARY_COLORS['orange'],
    'TJ356': PRIMARY_COLORS['purple'],
    'UA44': PRIMARY_COLORS['green'],
    'ROSELLA': PRIMARY_COLORS['pink'],
    'BR5270': PRIMARY_COLORS['red'],
    'BR5271': PRIMARY_COLORS['teal']
}

# Compound colors
COMPOUND_COLORS = {
    'CONTROL': PRIMARY_COLORS['gray'],
    'CBD': PRIMARY_COLORS['green'],
    'CBDV': PRIMARY_COLORS['purple']
}

def setup_plot_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to any matplotlib axis."""
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    return ax

# ===========================================================================
# DATA LOADING AND PROCESSING
# ===========================================================================

def parse_filename(file_path):
    """Extracts strain and treatment from the filename AND directory structure."""
    file_name = Path(file_path).stem
    dir_path = Path(file_path).parent
    dir_name = dir_path.name
    
    # First, try to get strain from directory name (more reliable)
    strain = None
    for s in ['N2', 'NL5901', 'UA44', 'TJ356', 'BR5270', 'BR5271', 'ROSELLA']:
        if s in dir_name:
            strain = s
            break
    
    # If not found in directory, try filename
    if not strain:
        match = re.match(r'^\d+_(?P<strain>[a-zA-Z0-9]+)_', file_name)
        if match:
            strain = match.group('strain')
    
    # Extract treatment from filename
    treatment = None
    file_upper = file_name.upper()
    
    # Check for control first
    if 'CONTROL' in file_upper:
        treatment = 'Control'
    # Plain ethanol or ET 1:1000: if folder has treatment, use Total_Extract_<treatment>
    elif ('ETANOL' in file_upper) or ('ET' in file_upper and (
        '1000' in file_upper or '1:1000' in file_upper or '1.1000' in file_upper or '1.100' in file_upper
    )):
        # Extract treatment from folder name or files in folder
        folder_treatment = None
        dir_upper = dir_name.upper()
        
        # First, check folder name
        if '_CBD' in dir_upper and '_CBDV' not in dir_upper:
            folder_treatment = 'CBD'
        elif '_CBDV' in dir_upper:
            folder_treatment = 'CBDV'
        
        # If not found in folder name, check files in the folder for treatment patterns
        if not folder_treatment:
            try:
                csv_files = list(dir_path.glob('*.csv'))
                for csv_file in csv_files[:10]:  # Check first 10 files to avoid too much I/O
                    csv_name = csv_file.stem.upper()
                    if 'CBD' in csv_name and 'CBDV' not in csv_name:
                        folder_treatment = 'CBD'
                        break
                    elif 'CBDV' in csv_name:
                        folder_treatment = 'CBDV'
                        break
            except:
                pass
        
        if folder_treatment:
            if 'ETANOL' in file_upper:
                treatment = f'ETANOL'
            elif 'ET' in file_upper and (
                '1000' in file_upper or '1:1000' in file_upper or '1.1000' in file_upper or '1.100' in file_upper
            ):
                treatment = f'Total_Extract_{folder_treatment}'
        else:
            # Fallback for folders that don't contain CBD/CBDV treatments
            if 'ETANOL' in file_upper:
                treatment = 'ETANOL'  # Plain ethanol, not a total extract
            elif 'ET' in file_upper and (
                '1000' in file_upper or '1:1000' in file_upper or '1.1000' in file_upper or '1.100' in file_upper
            ):
                treatment = 'Ethanol 1:1000'  # Ethanol dilution, not a total extract
    # Check for specific compounds and doses
    elif 'CBDV' in file_upper:
        if '30UM' in file_upper or '30 UM' in file_upper:
            treatment = 'CBDV 30uM'
        elif '3UM' in file_upper and '30' not in file_upper and '0.3' not in file_upper and '0,3' not in file_upper:
            treatment = 'CBDV 3uM'
        elif '0,3' in file_upper or '0.3' in file_upper:
            treatment = 'CBDV 0.3uM'
        else:
            treatment = 'CBDV'
    elif 'CBD' in file_upper and 'CBDV' not in file_upper:
        if '30UM' in file_upper or '30 UM' in file_upper:
            treatment = 'CBD 30uM'
        elif '3UM' in file_upper and '30' not in file_upper and '0.3' not in file_upper and '0,3' not in file_upper:
            treatment = 'CBD 3uM'
        elif '0,3' in file_upper or '0.3' in file_upper:
            treatment = 'CBD 0.3uM'
        else:
            treatment = 'CBD'
    else:
        # Try to extract whatever is after the strain in the filename
        parts = file_name.split('_')
        if len(parts) > 2:
            treatment = '_'.join(parts[2:]).replace('_', ' ').strip()
        else:
            treatment = 'Unknown'
    
    # Log for debugging
    if strain and treatment:
        print(f"  Parsed: {file_name} (in {dir_name}) -> Strain: {strain}, Treatment: {treatment}")
    else:
        print(f"  WARNING: Could not parse {file_name} (in {dir_name})")
    
    return strain, treatment

def load_and_combine_tracking_data(file_paths):
    """Load and combine tracking data from a list of CSV files, ensuring unique worm IDs."""
    all_data = []
    for i, csv_path in enumerate(file_paths):
        try:
            data = pd.read_csv(csv_path, sep=';')
            # Create a unique prefix for IDs from this file
            data['ID'] = f"{i}_" + data['ID'].astype(str)
            
            # Add metadata
            data['file_name'] = Path(csv_path).name
            data['experiment_id'] = Path(csv_path).stem.split('_')[0]
            
            # Try to add treatment info if available in the data
            if 'TRATAMIENTO' in data.columns:
                data['treatment'] = data['TRATAMIENTO'].fillna(method='ffill')
            
            all_data.append(data)
        except Exception as e:
            print(f"Could not read {csv_path}: {e}")
    
    if not all_data:
        return None

    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Quality filtering - remove rows with missing position data
    combined_data = combined_data[combined_data['X'].notna() & combined_data['Y'].notna()]
    
    print(f"  Loaded {len(combined_data)} tracking records from {len(file_paths)} files")
    
    return combined_data

# ===========================================================================
# STATIC VISUALIZATIONS
# ===========================================================================

def create_static_trajectory_plot(data, exp_name, strain, treatment, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f'Worm Trajectories - {strain} - {treatment}', fontsize=16, fontweight='bold')

    # Complete Trajectories
    # Each unique ID corresponds to a worm/track in this context
    worm_ids = data['ID'].unique()
    valid_worm_ids = [worm_id for worm_id in worm_ids
                      if len(data[data['ID'] == worm_id]) > 5]
    
    trajectory_lengths_final = {}
    trajectory_data = {}
    
    for worm_id in valid_worm_ids:
        worm_data = data[data['ID'] == worm_id].sort_values('FRAME')
        x_vals, y_vals = worm_data['X'].values, worm_data['Y'].values
        
        if len(worm_data) > 5:
            dx, dy = np.diff(x_vals), np.diff(y_vals)
            path_length = np.sum(np.sqrt(dx**2 + dy**2))
            trajectory_lengths_final[worm_id] = path_length
            trajectory_data[worm_id] = (x_vals, y_vals)
    
    # Use vibrant colors for trajectories
    vibrant_colors = [
        PRIMARY_COLORS['blue'], PRIMARY_COLORS['orange'], PRIMARY_COLORS['green'],
        PRIMARY_COLORS['red'], PRIMARY_COLORS['purple'], PRIMARY_COLORS['teal'],
        PRIMARY_COLORS['pink'], PRIMARY_COLORS['cyan'], PRIMARY_COLORS['magenta']
    ]
    
    # Plot all trajectories with colors
    for i, worm_id in enumerate(trajectory_data):
        x_vals, y_vals = trajectory_data[worm_id]
        
        # Each trajectory gets a vibrant color
        color = vibrant_colors[i % len(vibrant_colors)]
        ax.plot(x_vals, y_vals, '-', color=color,
                alpha=0.7, linewidth=1.0)

    setup_plot_style(ax,
                     title='Complete Trajectories',
                     xlabel='X Position (pixels)',
                     ylabel='Y Position (pixels)')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Add text annotation with counts
    total_tracks = len(valid_worm_ids)
    total_worms = data['ID'].nunique()
    ax.text(0.02, 0.98, f'Tracks: {total_tracks} | Worms: {total_worms}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

    plt.tight_layout()
    output_path = Path(output_dir) / f'{exp_name}_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved summary plot to: {output_path}")
    plt.close(fig)

def create_velocity_analysis(data, exp_name, strain, output_dir):
    # Velocity analysis is now included in the summary plot above
    pass

# ===========================================================================
# ANIMATION
# ===========================================================================

def create_animation(data, exp_name, interval=50, fade_frames=30, output_dir='results/Trajectories/animations', skip_animation=False):
    """
    Create enhanced animation with appearing/disappearing traces and final highlight of longest paths.
    
    Parameters:
    - fade_frames: Number of frames for trace to fade in/out
    - skip_animation: If True, only create the final frame as a static image
    
    Real optimizations used:
    - Pre-computed trajectories to avoid repeated data filtering
    - Efficient numpy operations for frame indexing
    - Smaller figure size and optimized rendering
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if skip_animation:
        print(f"Skipping animation, creating final frame only...")
        return
    
    # Use smaller figure size for faster rendering
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set plot limits
    x_min, x_max = data['X'].min(), data['X'].max()
    y_min, y_max = data['Y'].min(), data['Y'].max()
    margin = 50
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Clean minimal style
    setup_plot_style(ax, 
                    title=f'Worm Movement Traces - {exp_name}',
                    xlabel='X Position (pixels)',
                    ylabel='Y Position (pixels)')
    
    # Get ALL frames and ALL valid worms - no sampling or limiting
    frames = sorted(data['FRAME'].unique())
    print(f"Processing {len(frames)} frames")
    
    valid_worm_ids = [worm_id for worm_id in data['ID'].unique() 
                      if len(data[data['ID'] == worm_id]) > 5]
    print(f"Processing {len(valid_worm_ids)} valid worms")
    
    # Optimization 3: Pre-compute all trajectories and segments
    print("Pre-computing trajectory data...")
    trajectory_lengths = {}
    full_trajectories = {}
    trajectory_frames = {}  # Store frame indices for each worm
    
    # Pre-filter data for valid worms to reduce repeated filtering
    valid_data = data[data['ID'].isin(valid_worm_ids)]
    
    for worm_id in valid_worm_ids:
        worm_data = valid_data[valid_data['ID'] == worm_id].sort_values('FRAME')
        if len(worm_data) > 5:
            x_vals, y_vals = worm_data['X'].values, worm_data['Y'].values
            frame_vals = worm_data['FRAME'].values
            
            # Store pre-computed values
            dx, dy = np.diff(x_vals), np.diff(y_vals)
            path_length = np.sum(np.sqrt(dx**2 + dy**2))
            trajectory_lengths[worm_id] = path_length
            full_trajectories[worm_id] = (x_vals, y_vals)
            trajectory_frames[worm_id] = frame_vals
    
    # Color mapping - vibrant colors for trajectories during animation
    vibrant_colors = [
        PRIMARY_COLORS['blue'], PRIMARY_COLORS['orange'], PRIMARY_COLORS['green'],
        PRIMARY_COLORS['red'], PRIMARY_COLORS['purple'], PRIMARY_COLORS['teal'],
        PRIMARY_COLORS['pink'], PRIMARY_COLORS['cyan'], PRIMARY_COLORS['magenta']
    ]
    
    # For fading: colors will fade from vibrant to gray
    gray_color = '#B0B0B0'

    # Initialize line objects for each worm
    worm_lines = {}
    worm_alphas = {}
    worm_active_frames = {}
    worm_base_colors = {}
    
    for i, worm_id in enumerate(valid_worm_ids):
        worm_data = data[data['ID'] == worm_id]
        first_frame = worm_data['FRAME'].min()
        last_frame = worm_data['FRAME'].max()
        worm_active_frames[worm_id] = (first_frame, last_frame)
        
        # Assign vibrant color to each worm
        base_color = vibrant_colors[i % len(vibrant_colors)]
        worm_base_colors[worm_id] = base_color
        
        # All worms start with same line width
        linewidth = 2.0
        
        line, = ax.plot([], [], '-', color=base_color, linewidth=linewidth, alpha=0)
        worm_lines[worm_id] = line
        worm_alphas[worm_id] = 0.0
    
    # Frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top', color='black',
                        bbox=dict(boxstyle='round', facecolor='white', 
                                edgecolor='black', alpha=0.8))
    
    # Add final frame indicator - no limit for production
    total_frames = len(frames) + 60  # Add extra frames for final display
    
    # Track which worms actually appeared during the animation
    worms_that_appeared = set()
    
    def animate(frame_idx):
        if frame_idx < len(frames):
            current_frame = frames[frame_idx]
            
            for worm_id in valid_worm_ids:
                first_frame, last_frame = worm_active_frames[worm_id]
                
                # Calculate alpha based on fade in/out
                if current_frame < first_frame:
                    alpha = 0.0
                elif current_frame <= first_frame + fade_frames:
                    # Fade in
                    alpha = (current_frame - first_frame) / fade_frames
                elif current_frame < last_frame - fade_frames:
                    # Full visibility
                    alpha = 1.0
                elif current_frame <= last_frame:
                    # Fade out
                    alpha = (last_frame - current_frame) / fade_frames
                else:
                    alpha = 0.0
                
                worm_alphas[worm_id] = alpha
                
                # Optimization: Use pre-computed trajectory data
                if worm_id in trajectory_frames:
                    frame_vals = trajectory_frames[worm_id]
                    # Find index where frames are <= current_frame
                    mask = frame_vals <= current_frame
                    if np.any(mask):
                        worms_that_appeared.add(worm_id)
                        x_vals, y_vals = full_trajectories[worm_id]
                        # Use boolean indexing for faster slicing
                        last_idx = np.where(mask)[0][-1] + 1
                        x_data = x_vals[:last_idx]
                        y_data = y_vals[:last_idx]
                        worm_lines[worm_id].set_data(x_data, y_data)
                    
                    # Apply alpha and color transition
                    if alpha < 1.0 and current_frame >= last_frame - fade_frames:
                        # During fade out, transition to gray
                        fade_factor = alpha
                        base_color = worm_base_colors[worm_id]
                        # Interpolate between base color and gray
                        import matplotlib.colors as mcolors
                        base_rgb = mcolors.to_rgb(base_color)
                        gray_rgb = mcolors.to_rgb(gray_color)
                        faded_rgb = tuple(base_rgb[i] * fade_factor + gray_rgb[i] * (1 - fade_factor) 
                                        for i in range(3))
                        worm_lines[worm_id].set_color(faded_rgb)
                    else:
                        # Normal color
                        worm_lines[worm_id].set_color(worm_base_colors[worm_id])
                    
                    worm_lines[worm_id].set_alpha(alpha * 0.8)  # Max alpha 0.8 for visibility
            
            frame_text.set_text(f'Frame: {current_frame}/{frames[-1]}')
        
        else:
            # Final frames - show all trajectories with same style
            transition_progress = (frame_idx - len(frames)) / 60
            
            for worm_id in valid_worm_ids:
                # Only show worms that actually appeared during the animation
                if worm_id not in worms_that_appeared:
                    continue
                    
                # Get full trajectory 
                x_data, y_data = full_trajectories[worm_id]
                worm_lines[worm_id].set_data(x_data, y_data)
                
                # Keep the vibrant colors but same style for all
                worm_lines[worm_id].set_linewidth(2.0)  # Same width for all
                worm_lines[worm_id].set_color(worm_base_colors[worm_id])  # Keep their colors!
                
                # All get same alpha - no highlighting
                target_alpha = 0.7  # Semi-transparent so overlaps are visible
                current_alpha = worm_alphas[worm_id] * 0.8
                new_alpha = current_alpha + (target_alpha - current_alpha) * transition_progress
                worm_lines[worm_id].set_alpha(new_alpha)
            
            frame_text.set_text(f'Complete Trajectories\n{len(worms_that_appeared)} worms tracked')
        
        return list(worm_lines.values()) + [frame_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                  interval=interval, blit=True, repeat=True)
    
    output_path = Path(output_dir) / f'{exp_name}_animation.gif'
    print(f"\nCreating animation with {len(frames)} frames...")
    print(f"Animating {len(valid_worm_ids)} worms...")
    
    try:
        # Try to save with Pillow (most compatible)
        print("Generating optimized GIF...")
        anim.save(output_path, writer='pillow', fps=15)
        print(f"Animation saved to: {output_path}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # If animation fails, at least save the final frame
        try:
            fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
            print(f"Saved final frame as static image: {output_path.with_suffix('.png')}")
        except:
            print("Failed to save any output")
    
    plt.close(fig)

# ===========================================================================
# PARALLEL WORKER
# ===========================================================================

def process_single_experiment(strain, treatment, file_paths, base_plot_dir, base_animation_dir, skip_animations=False):
    """Process a single (strain, treatment) experiment group.

    Returns a tuple: (exp_name, success: bool, details: str)
    """
    try:
        exp_name = f"{strain}_{treatment}"
        print(f"\n{'='*40}")
        print(f"Processing: {exp_name}")
        print(f"Files: {len(file_paths)}")
        print(f"{'='*40}")

        # Define output directories
        plot_output_dir = Path(base_plot_dir) / strain / treatment.replace(' ', '_').replace(':', '')
        animation_output_dir = Path(base_animation_dir) / strain / treatment.replace(' ', '_').replace(':', '')

        # Load and combine data
        combined_data = load_and_combine_tracking_data(file_paths)

        if combined_data is None or combined_data.empty:
            msg = f"ERROR: No data to process for {exp_name}"
            print(msg)
            return exp_name, False, msg

        # Create visualizations
        print(f"\n1. Creating trajectory analysis...")
        create_static_trajectory_plot(combined_data, exp_name, strain, treatment, plot_output_dir)

        print(f"\n2. Creating velocity analysis...")
        create_velocity_analysis(combined_data, exp_name, strain, plot_output_dir)

        if not skip_animations:
            print(f"\n3. Creating animation...")
            create_animation(combined_data, exp_name, interval=100, fade_frames=20,
                            output_dir=animation_output_dir)
        else:
            print(f"\n3. Skipping animation (animations disabled)")

        return exp_name, True, "OK"
    except Exception as exc:
        return f"{strain}_{treatment}", False, f"FAILED with error: {exc}"

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main(skip_animations=False, workers=None):
    """Main function to run the batch processing.
    
    Args:
        skip_animations: If True, skip animation generation for faster processing
        workers: Number of parallel worker processes to use (default: auto)
    """
    
    root_data_dir = Path('Datos')
    base_plot_dir = Path('results/Trajectories')
    base_animation_dir = Path('results/Trajectories/animations')

    print("="*60)
    print("C. ELEGANS BATCH TRAJECTORY VISUALIZER")
    if skip_animations:
        print("(ANIMATIONS DISABLED FOR FASTER PROCESSING)")
    print("="*60)
    
    # Find all CSV files and group them
    print("\nScanning for data files...")
    all_csv_files = list(root_data_dir.rglob('*.csv'))
    print(f"Found {len(all_csv_files)} CSV files")
    
    experiments = {}
    unparsed_files = []

    print("\nParsing file information...")
    for file_path in all_csv_files:
        strain, treatment = parse_filename(file_path)
        if strain and treatment:
            if strain not in experiments:
                experiments[strain] = {}
            if treatment not in experiments[strain]:
                experiments[strain][treatment] = []
            experiments[strain][treatment].append(file_path)
        else:
            unparsed_files.append(file_path)

    # Summary of what was found
    print("\n" + "-"*60)
    print("DATA SUMMARY:")
    total_experiments = sum(len(treatments) for treatments in experiments.values())
    print(f"Strains identified: {', '.join(sorted(experiments.keys()))}")
    print(f"Total unique experiments: {total_experiments}")
    print(f"Files that couldn't be parsed: {len(unparsed_files)}")
    
    if unparsed_files:
        print("\nUnparsed files:")
        for f in unparsed_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(unparsed_files) > 5:
            print(f"  ... and {len(unparsed_files) - 5} more")
    
    print("\nExperiment breakdown:")
    for strain in sorted(experiments.keys()):
        treatments_list = sorted(experiments[strain].keys())
        print(f"  {strain}: {', '.join(treatments_list)}")
    print("-"*60)

    if not experiments:
        print("\nERROR: No valid experiments found to process.")
        return

    # Build task list
    tasks = []
    for strain, treatments in experiments.items():
        for treatment, file_paths in treatments.items():
            tasks.append((strain, treatment, [str(p) for p in file_paths], str(base_plot_dir), str(base_animation_dir), bool(skip_animations)))

    # Determine number of workers
    total_tasks = len(tasks)
    if workers is None:
        env_workers = os.environ.get('BATCH_WORKERS')
        if env_workers and env_workers.isdigit():
            workers = int(env_workers)
        else:
            # Sensible default: up to 4 or CPU count, but not more than tasks
            cpu = os.cpu_count() or 2
            workers = min(4, cpu, total_tasks)
    else:
        workers = max(1, min(int(workers), total_tasks))

    print(f"\nRunning in parallel with workers={workers} (total tasks={total_tasks})")

    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_exp = {executor.submit(process_single_experiment, *t): f"{t[0]}_{t[1]}" for t in tasks}
        for future in as_completed(future_to_exp):
            exp_name = future_to_exp[future]
            try:
                result_exp, ok, details = future.result()
            except Exception as e:
                print(f"{exp_name}: FAILED with error: {e}")
                failed_count += 1
            else:
                if ok:
                    processed_count += 1
                    print(f"{result_exp}: DONE")
                else:
                    failed_count += 1
                    print(f"{result_exp}: {details}")

    # Final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE!")
    print("="*60)
    print(f"Successfully processed: {processed_count}/{total_experiments} experiments")
    if failed_count > 0:
        print(f"Failed: {failed_count} experiments")
    print(f"Output directory: {base_plot_dir.absolute()}")
    print("\nAll figures saved with publication-quality formatting.")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    skip_animations = '--no-animations' in sys.argv or '--fast' in sys.argv
    workers = None

    # Parse workers from CLI: --workers N, --workers=N, or -j N
    for i, arg in enumerate(sys.argv):
        if arg in ('--workers', '-j'):
            if i + 1 < len(sys.argv):
                try:
                    workers = int(sys.argv[i + 1])
                except ValueError:
                    pass
        elif arg.startswith('--workers='):
            try:
                workers = int(arg.split('=', 1)[1])
            except ValueError:
                pass

    if '--help' in sys.argv:
        print("Usage: python generate_trajectory_visualizations.py [options]")
        print("Options:")
        print("  --no-animations, --fast        Skip animations for faster processing")
        print("  --workers N | -j N             Number of parallel workers")
        print("  --help                         Show this help message")
        sys.exit(0)
    
    main(skip_animations=skip_animations, workers=workers)