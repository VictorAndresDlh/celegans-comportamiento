# Agent Context – C. elegans Cannabinoid Behavior Project

**Purpose of this document**

This file is a living summary for code agents working in this repository. It explains:
- What the data look like and where they live
- Which analysis methodologies are implemented and how they connect
- How results flow into paper figures and text
- Where to plug in new analyses or refactors safely

Update this document whenever you change the analysis flow, file locations, or figure-generation code.

---

## 1. Repository Overview

Top-level structure (only relevant pieces):

- `Datos/` – Raw trajectory CSVs from WMicrotracker SMARTx8 (per strain/treatment)
- `results/` – All derived outputs (numerical + figures), organized as:
  - `Analysis/` – Analysis outputs by methodology and strain
    - `Unsupervised_States/` – GMM-based state discovery
    - `Levy_Flight/` – Lévy flight / step-length fits
    - `ML_Screening/` – Kinematic ML classification
    - `TDA/` – Topological classification (persistent homology)
  - `Trajectories/` – Static plots and animations of raw trajectories (QA)
  - `Figures/` – Assembled figures for manuscript/summary
- `Documentation/` – Project-level docs and prompts for agents
- `Paper/` – Section text and any legacy figures for the manuscript
- Root scripts:
  - `generate_trajectory_visualizations.py` (antes `run_batch_analysis.py`)
  - `methodology_*.py` (one per methodology)
  - `summarize_*.py` (one per methodology family)
  - `generate_*.py` (figure generators)

All analysis is driven from `Datos/` and ends under `results/` (numerical outputs in `results/Analysis`, visuals in `results/Trajectories` and `results/Figures`).

---

## 2. Data Model

### 2.1 Raw data (`Datos/`)

- Source: WMicrotracker SMARTx8, tracking mode.
- Format: semicolon-separated CSVs with at least columns:
  - `ID` – worm identity within acquisition
  - `FRAME` – frame index (1 Hz sampling)
  - `X`, `Y` – centroid position (pixels or mm, consistent per experiment)
- Layout:
  - Top-level strain or experiment group folders, e.g.:
    - `Datos/N2_CBD/`, `Datos/N2_CBDV/`, `Datos/NL5901_CBD/`, `Datos/BR5270/`, etc.
  - Filenames encode experiment ID, strain, and treatment, e.g.:
    - `1551_N2_CONTROL.csv`
    - `1602_N2_30uMCBD.csv`
    - `1716_N2_30uMCBD.csv`
    - `1505_BR5270_ETANOL.csv`

### 2.2 Parsing strain and treatment

Multiple scripts share a common `parse_filename` implementation (duplicated in some files; refactor target):

- Strain detection priority:
  1. Directory name contains one of: `N2`, `NL5901`, `UA44`, `TJ356`, `BR5270`, `BR5271`, `ROSELLA`
  2. If not, regex on filename: `^<ID>_<strain>_...`

- Treatment detection logic (simplified summary):
  - If filename contains `CONTROL` → `"Control"`.
  - If it contains `ETANOL` or `ET` with 1:1000 variants:
    - Try to infer associated compound from parent folder or sibling files (CBD/CBDV) and classify as
      - `"ETANOL"` (plain vehicle) or
      - `"Total_Extract_CBD"` / `"Total_Extract_CBDV"` depending on context.
  - If it contains `CBDV`:
    - Map dose by `30uM`, `3uM`, `0.3uM` patterns → `"CBDV 30uM"`, `"CBDV 3uM"`, `"CBDV 0.3uM"`.
  - If it contains `CBD` (but not `CBDV`):
    - Analogous mapping → `"CBD 30uM"`, `"CBD 3uM"`, `"CBD 0.3uM"`.
  - Else: everything after strain in filename is treated as free-text treatment label or `"Unknown"`.

**Important for agents:** before changing naming conventions or paths in `Datos/`, update all `parse_filename` copies consistently, or centralize into a single importable helper.

---

## 3. Analysis Pipelines (Methodologies)

Each methodology is implemented as a standalone script named `methodology_*.py`. All of them:
- Walk `Datos/`
- Use `parse_filename` to get `(strain, treatment)`
- Load trajectories per `(strain, treatment)` into a combined `DataFrame`
- Compute per-worm features / objects
- Write results under `results/Analysis/<Methodology>/<Strain>/...`

### 3.1 Unsupervised Behavioral States

- Script: `methodology_unsupervised_states.py`
- Output root: `results/Analysis/Unsupervised_States/<Strain>/`

**Logic:**
1. Load all worms for a given `target_strain` grouped by `treatment`.
2. For each worm, compute a rich feature vector from its centroid trajectory:
   - Speed statistics (mean, std, min, quartiles, max)
   - Turning angle statistics (mean |angle|, std, max)
   - Path length, net displacement, confinement ratio.
3. Pool all worms (all treatments) for that strain.
4. Standardize features (`StandardScaler`).
5. Apply PCA keeping enough components to explain 95% variance.
6. Fit Gaussian Mixture Models (full covariance) with K ∈ [2, 10]; choose K minimizing BIC.
7. Assign a discrete state label to each worm.

**Key outputs:**
- `ethogram.csv` – rows = states, columns = feature means (state characterization).
- `ethogram_heatmap.png` – visual etogram.
- `state_distribution.csv` – percentage of worms (per treatment) in each state.
- `state_distribution_barchart.png` – stacked bar chart by treatment.

Downstream, `summarize_unsupervised_analysis.py` uses these to:
- Identify, per strain:
  - Pause state = state with minimal `speed_mean`.
  - Cruise/active state = state with maximal `displacement`.
- Compute treatment vs control deltas in % occupancy of these key states.

### 3.2 Lévy Flight / Path Structure

- Script: `methodology_levy_flight.py`
- Output root: `results/Analysis/Levy_Flight/<Strain>/`

**Logic:**
1. For each `(strain, treatment)` group:
   - Concatenate all worms.
   - For each worm:
     - Order by `FRAME`, get `[(X_i, Y_i)]` trajectory.
     - Compute inter-step vectors and heading angles.
     - Detect turning events where the absolute angle change between successive movement vectors exceeds a threshold (40°).
     - Define step lengths as distances between successive turning points.
2. Pool all step lengths across worms in that treatment.
3. Fit continuous power-law with `powerlaw.Fit`:
   - Estimate `alpha`, `xmin`.
   - Compare `power_law` vs `lognormal` using log-likelihood ratio `R` and p-value.
4. Save per-treatment summary and per-treatment plots.

**Key outputs:**
- `levy_flight_summary.csv` per strain with columns:
  - `treatment`, `alpha`, `xmin`, `loglikelihood_ratio`, `p_value`.
- `levy_fit_<treatment>.png` – empirical vs fitted distributions.

`summarize_levy_analysis.py` aggregates all strains and treatments, classifies patterns as:
- Lévy-like (R > 0 and p < 0.05)
- Brownian-like (else)

### 3.3 ML Screening (Kinematic Features)

- Script: `methodology_ml_screening.py`
- Output root: `results/Analysis/ML_Screening/<Strain>/`

**Logic:**
1. For a given `target_strain`, load data for all treatments including `Control`.
2. Per worm, extract a kinematic feature vector similar to the unsupervised case (without state labels).
3. For each treatment T ≠ Control:
   - Build a binary dataset: `Control` vs `T`.
   - Train/test split (70/30, stratified).
   - Scale features.
   - Train `RandomForestClassifier` (100 trees, `class_weight='balanced'`).
   - Evaluate accuracy and F1 per class.
   - Generate feature importance barplots.

**Key outputs:**
- `classification_summary.csv` per strain with columns e.g.:
  - `treatment`, `accuracy`, `f1_control`, `f1_treatment`, etc.
- `feature_importance_Control_vs_<treatment>.png`.

`summarize_ml_screening.py` reads all summaries and:
- Categorizes accuracy (Excellent/Good/Moderate/Poor).
- Highlights exceptional cases (e.g. NL5901 + CBD 3uM ≈ 95% accuracy).

### 3.4 Topological Data Analysis (TDA)

- Script: `methodology_tda.py`
- Output root: `results/Analysis/TDA/<Strain>/`

**Logic:**
1. For each strain and treatment group (including Control):
   - For each worm with length ≥ window length (20 points):
     - Center trajectory (subtract mean position).
     - Build sliding windows of size 20; each window is flattened into a 40D vector.
2. For each worm/treatment:
   - Treat the collection of window vectors as a point cloud in R^{40}.
   - Build a Vietoris–Rips complex up to dim 2 and compute persistence.
   - Extract H1 intervals and compute persistence landscapes (Gudhi `Landscape`).
   - Flatten landscapes into a vector of topological features.
3. For each treatment T ≠ Control:
   - Build binary dataset Control vs T using landscape features.
   - Scale features.
   - Train SVM (RBF kernel, `class_weight='balanced'`).
   - Evaluate accuracy and F1 per class.
   - Save example landscape plots.

**Key outputs:**
- `classification_summary.csv` per strain.
- `landscapes_Control_vs_<treatment>.png`.

`summarize_tda.py` aggregates performance and highlights conditions with strong topological phenotypes (e.g. UA44 + CBD 3uM ≈ high 80% accuracy).

---

## 4. Trajectory Visualization (Raw Data)

- Script: `generate_trajectory_visualizations.py` (antes `run_batch_analysis.py`)
- Role: visualization/QA of raw trajectories; not part of the formal four-method pipeline, but shares the same `parse_filename` logic.

**Key behavior:**
- Scans `Datos/` recursively for `.csv`.
- Groups files by `(strain, treatment)`.
- For each group:
  - Loads all files and ensures unique worm IDs per file.
  - Creates summary plot (`results/Trajectories/<Strain>/<Treatment>/..._summary.png`) with:
    - Overlaid worm trajectories.
    - Velocity histogram.
    - Mean velocity over time.
    - Text annotation summarizing `Tracks: N | Worms: M`.
  - Optionally creates GIF animations (`results/Trajectories/<Strain>/<Treatment>/animations/...gif`) unless `--no-animations`.
- Parallelized with `ProcessPoolExecutor` (CLI options `--fast`, `--no-animations`, `--workers`).

Use this script for quick visual sanity checks after any change to parsing or data loading logic.

---

## 5. Figure Generation for the Paper

Figures for `Paper/final_paper_section_v10.md` are generated from `Analysis/**` via several scripts.

### 5.1 High-level figure generators

- `generate_paper_figures_v2.py` – main, feature-complete generator.
- `generate_all_figures_standardized.py` – alternative with stricter style unification.
- `generate_paper_figures.py` – simplified/regeneration of selected figures.

**Common behavior:**
- Define paths:
  - Input: `results/Analysis/Unsupervised_States`, `results/Analysis/Levy_Flight`, `results/Analysis/ML_Screening`, `results/Analysis/TDA`.
  - Output: `results/Figures/` (primary) and/or `Paper/Figures/` (legacy).
- Use helper loaders:
  - `load_state_distribution(strain)` → `state_distribution.csv`.
  - `load_ethogram(strain)` → `ethogram.csv`.
  - `load_levy_flight(strain)` → `levy_flight_summary.csv`.
  - `load_ml_classification(strain)` → ML `classification_summary.csv`.
  - `load_tda_classification(strain)` → TDA `classification_summary.csv`.
- Implement reusable helpers:
  - `identify_pause_state(strain)` – index of state with min `speed_mean`.
  - `identify_cruise_state(strain)` – index of state with max `displacement`.

### 5.2 Core figures (as of now)

Typical final figure files (PDF+PNG) include:

- `Fig1_ETANOL_Sedative_Panel.*`
  - 3-panel barplot (N2, UA44, ROSELLA):
    - Y: % worms in pause state
    - X: Control vs ETANOL.

- `Fig2_CBD_Paradoxical_Effect.*`
  - Split in stimulant and sedative panels:
    - N2 & NL5901: CBD 30uM reduces pause occupancy (stimulant/anti-sedative).
    - UA44 & ROSELLA: CBD 30uM increases pause occupancy (sedative/toxic).

- `Fig3_NL5901_Levy_Compensatory.*`
  - Line plot of α exponent across treatments for N2 vs NL5901.
  - Shows NL5901’s stable Lévy-like pattern vs N2’s mostly Brownian.

- `Fig4_ML_Screening_Landscape.*`
  - Heatmap of ML classification accuracy for all (strain, treatment) pairs.
  - Highlights exceptional binary separations (e.g. NL5901 + CBD 3uM).

- `Fig5_TDA_Landscape.*`
  - Heatmap of TDA classification accuracy.
  - Highlights conditions where topology is especially informative (e.g. UA44 + CBD 3uM).

Some scripts also defined a `Fig4_Levy_Induction_Cannabinoids.*` figure; the latest standardized generator may have removed/renumbered it. Before changing figure numbering, check `final_paper_section_v10.md` for references.

---

## 6. Paper Text Linkage

- File: `Paper/final_paper_section_v10.md`

This is the narrative that:
- Describes acquisition (SMARTx8), strains, and treatment panel.
- Explains each methodology:
  - Unsupervised states → ethograms and state occupancy shifts.
  - Lévy flight analysis → α exponents and power-law vs lognormal tests.
  - ML screening → binary classification accuracy as a measure of phenotypic separability.
  - TDA → topological separability and when it outperforms kinematic ML.
- Cites concrete numeric results that should remain consistent with the outputs of:
  - `state_distribution.csv`
  - `levy_flight_summary.csv`
  - `ML_Screening/*/classification_summary.csv`
  - `TDA/*/classification_summary.csv`

**When editing analysis code:**
- If metrics or definitions change (e.g. new pause-state rule, different window length), either:
  - Update the paper text to match, **or**
  - Preserve backwards-compatible behavior for the existing version and add new outputs under new filenames.

---

## 7. Agent Guidelines for Future Changes

### 7.1 Safe refactors

Good, low-risk places to improve/organize:

- Centralize `parse_filename` into a single module (e.g. `utils_io.py`) and import it from all methodologies and `generate_trajectory_visualizations.py`.
- Factor repeated feature-extraction logic (currently duplicated between ML/Unsupervised) into a shared module.
- Add a thin orchestrator script (e.g. `run_all_analyses.py`) that:
  - Runs all `methodology_*.py` scripts for all strains.
  - Optionally runs `summarize_*.py` and `generate_paper_figures_v2.py`.

### 7.2 Things to keep stable (unless explicitly requested)

- Directory contracts:
  - Inputs: `Datos/**.csv`
  - Analysis outputs: `results/Analysis/<Methodology>/<Strain>/...`
  - Trajectory QA plots: `results/Trajectories/`
  - Final assembled figures: `results/Figures/` (and optionally `Paper/Figures/`)
- Filenames consumed by figure generators and summarizers:
  - `ethogram.csv`, `state_distribution.csv`
  - `levy_flight_summary.csv`
  - `classification_summary.csv` (both for ML and TDA)
- Semantics of key derived quantities:
  - Pause state definition: lowest `speed_mean` per strain.
  - Cruise/active state: highest `displacement` per strain.
  - Lévy vs Brownian classification: `R > 0` and `p < 0.05`.

### 7.3 How to extend

When adding a new analysis method or figure:

1. **New methodology script**
   - Follow pattern of `methodology_*.py` (standalone, per-strain, writes to `Analysis/<NewMethod>/<Strain>/`).
   - Reuse existing loaders and `parse_filename` where possible.
2. **Summarizer**
   - Mirror the `summarize_*.py` pattern: read all strains, build a single rich table, print markdown summaries.
3. **Figure generator**
   - Add new plotting functions to a `generate_*.py` file.
   - Use the same style defaults (fonts, colors) used by the existing figure scripts.

Update this `agent_context.md` with:
- New analysis outputs (paths, filenames, schema).
- Any new entrypoints for reproducing the paper (commands, prerequisites).

---

## 8. Reproduction Commands (for agents)

Assuming cwd = repository root (`Comportamiento/`), typical runs (all results under `results/`):

```bash
# 1) Batch raw trajectory visualizations (QA)
python generate_trajectory_visualizations.py --fast --workers 4

# 2) Run all methodologies (per-strain analyses)
python methodology_unsupervised_states.py
python methodology_levy_flight.py
python methodology_ml_screening.py
python methodology_tda.py

# 3) Summarize results
python summarize_unsupervised_analysis.py
python summarize_levy_analysis.py
python summarize_ml_screening.py
python summarize_tda.py

# 4) Generate paper figures (assembled panels)
python generate_paper_figures_v2.py
# or
python generate_all_figures_standardized.py
```

If you change any of these entrypoints or filenames, reflect that here.
