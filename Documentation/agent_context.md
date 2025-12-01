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
    - `Levy_Flight/` – Lévy flight / step-length fits
    - `ML_Screening/` – Kinematic ML classification
    - `TDA/` – Topological classification (persistent homology)
  - `Trajectories/` – Static plots and animations of raw trajectories (QA)
  - `Figures/` – Assembled figures for manuscript/summary
- `Documentation/` – Project-level docs and prompts for agents
- `Paper/` – Section text and any legacy figures for the manuscript
- Root scripts:
  - `generate_trajectory_visualizations.py` – Raw trajectory visualization
  - `methodology_*.py` (one per methodology)
  - `summarize_*.py` (one per methodology)

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

**Note**: Only methodologies applicable to centroid trajectory data (x, y coordinates) are implemented.

### 3.1 Lévy Flight / Path Structure

**Reference**: Moy et al. (2015) - Computational Methods for Tracking, Quantitative Assessment, and Visualization of C. elegans Locomotory Behavior

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

### 3.2 ML Screening (Kinematic Features)

**Reference**: García-Garví & Sánchez-Salmerón (2025) - High-throughput behavioral screening using machine learning

- Script: `methodology_ml_screening.py`
- Output root: `results/Analysis/ML_Screening/<Strain>/`

**Logic:**
1. For a given `target_strain`, load data for all treatments including `Control`.
2. Per worm, extract kinematic features from centroid trajectory:
   - Speed statistics (mean, std, median)
   - Turning angle statistics (mean, std)
   - Path structure (path length, displacement, confinement ratio)
3. For each treatment T ≠ Control:
   - Run block permutation tests (5000 permutations) per feature
   - Apply Benjamini-Yekutieli correction (FDR < 0.10)
   - Select significant features
   - Build binary dataset: `Control` vs `T`
   - Train/test split (70/30, stratified)
   - Scale features
   - Train `RandomForestClassifier` (100 trees, `class_weight='balanced'`)
   - Evaluate accuracy and F1 per class
   - Generate feature importance plots

**Key outputs:**
- `classification_summary.csv` per strain with columns:
  - `treatment`, `accuracy`, `f1_control`, `f1_treatment`, `n_significant_features_fdr`, etc.
- `feature_significance_Control_vs_<treatment>.csv` – permutation p-values and q-values
- `feature_importance_Control_vs_<treatment>.png`

`summarize_ml_screening.py` reads all summaries and:
- Categorizes accuracy (Excellent/Good/Moderate/Poor)
- Highlights exceptional cases (e.g. NL5901 + CBD 3uM with high accuracy)

### 3.3 Topological Data Analysis (TDA)

**Reference**: Thomas et al. (2021) - Topological Data Analysis of C. elegans Locomotion and Behavior

- Script: `methodology_tda.py`
- Output root: `results/Analysis/TDA/<Strain>/`

**Logic:**
1. For each strain, test multiple window lengths L ∈ {10, 20, 50}
2. For each treatment group (including Control):
   - For each worm with length ≥ window length:
     - Center trajectory (subtract mean position)
     - Build sliding windows of size L; each window is flattened into a 2L-dimensional vector
3. For each worm/treatment:
   - Treat the collection of window vectors as a point cloud in R^{2L}
   - Build a Vietoris–Rips complex up to dim 2 and compute persistence
   - Extract H₁ intervals and compute persistence landscapes (Gudhi `Landscape`)
   - Flatten landscapes into a vector of topological features
4. For each treatment T ≠ Control:
   - Build binary dataset Control vs T using landscape features
   - Scale features
   - Train SVM (RBF kernel, `class_weight='balanced'`)
   - Evaluate accuracy and F1 per class
5. Select window length with highest mean accuracy across all treatments
6. Re-run with best window length and save plots

**Key outputs:**
- `classification_summary.csv` per strain (includes `window_length` column)
- `landscapes_Control_vs_<treatment>.png`

`summarize_tda.py` aggregates performance and highlights conditions with strong topological phenotypes (e.g. UA44 + CBD 3uM with high accuracy).

---

## 4. Trajectory Visualization (Raw Data)

- Script: `generate_trajectory_visualizations.py`
- Role: visualization/QA of raw trajectories; not part of the formal analysis pipeline, but shares the same `parse_filename` logic.

**Key behavior:**
- Scans `Datos/` recursively for `.csv`
- Groups files by `(strain, treatment)`
- For each group:
  - Loads all files and ensures unique worm IDs per file
  - Creates summary plot (`results/Trajectories/<Strain>/<Treatment>/..._summary.png`) with:
    - Overlaid worm trajectories
    - Velocity histogram
    - Mean velocity over time
    - Text annotation summarizing `Tracks: N | Worms: M`
  - Optionally creates GIF animations (`results/Trajectories/<Strain>/<Treatment>/animations/...gif`) unless `--no-animations`
- Parallelized with `ProcessPoolExecutor` (CLI options `--fast`, `--no-animations`, `--workers`)

Use this script for quick visual sanity checks after any change to parsing or data loading logic.

---

## 5. Figure Generation

Figures are generated from analysis results in `results/Analysis/`. Figure generation scripts were removed during cleanup as they contained references to non-applicable methodologies.

New figures should be created by:
1. Loading results from `results/Analysis/<Methodology>/<Strain>/`
2. Using consistent styling (colorblind-friendly palette, clear labels)
3. Saving to `results/Figures/`

**Available data sources:**
- `results/Analysis/Levy_Flight/<Strain>/levy_flight_summary.csv`
- `results/Analysis/ML_Screening/<Strain>/classification_summary.csv`
- `results/Analysis/TDA/<Strain>/classification_summary.csv`

---

## 6. Paper Text Linkage

- File: `Paper/final_paper_section_v10.md`

This is the narrative that:
- Describes acquisition (SMARTx8), strains, and treatment panel
- Explains each methodology:
  - Lévy flight analysis → α exponents and power-law vs lognormal tests
  - ML screening → binary classification accuracy as a measure of phenotypic separability
  - TDA → topological separability and when it detects movement pattern changes
- Cites concrete numeric results that should remain consistent with the outputs of:
  - `levy_flight_summary.csv`
  - `ML_Screening/*/classification_summary.csv`
  - `TDA/*/classification_summary.csv`

**When editing analysis code:**
- If metrics or definitions change (e.g. different threshold angles, window lengths), either:
  - Update the paper text to match, **or**
  - Preserve backwards-compatible behavior for the existing version and add new outputs under new filenames.

---

## 7. Agent Guidelines for Future Changes

### 7.1 Safe refactors

Good, low-risk places to improve/organize:

- Centralize `parse_filename` into a single module (e.g. `utils_io.py`) and import it from all methodologies and `generate_trajectory_visualizations.py`
- Add a thin orchestrator script (e.g. `run_all_analyses.py`) that:
  - Runs all `methodology_*.py` scripts for all strains
  - Optionally runs `summarize_*.py` scripts

### 7.2 Things to keep stable (unless explicitly requested)

- Directory contracts:
  - Inputs: `Datos/**.csv`
  - Analysis outputs: `results/Analysis/<Methodology>/<Strain>/...`
  - Trajectory QA plots: `results/Trajectories/`
  - Final assembled figures: `results/Figures/`
- Filenames consumed by summarizers:
  - `levy_flight_summary.csv`
  - `classification_summary.csv` (both for ML and TDA)
- Semantics of key derived quantities:
  - Lévy vs Brownian classification: `R > 0` and `p < 0.05`
  - TDA window length selection: highest mean accuracy across treatments

### 7.3 How to extend

When adding a new analysis method:

1. **New methodology script**
   - Follow pattern of `methodology_*.py` (standalone, per-strain, writes to `Analysis/<NewMethod>/<Strain>/`)
   - Reuse existing loaders and `parse_filename` where possible
   - Verify methodology is applicable to centroid trajectory data only
2. **Summarizer**
   - Mirror the `summarize_*.py` pattern: read all strains, build a single rich table, print markdown summaries
3. **Figure generator**
   - Create new script for specific figure needs
   - Use consistent style (colorblind-friendly colors, clear fonts)

Update this `agent_context.md` with:
- New analysis outputs (paths, filenames, schema)
- Any new entrypoints for reproducing analyses

---

## 8. Reproduction Commands (for agents)

Assuming cwd = repository root, typical runs (all results under `results/`):

```bash
# 1) Batch raw trajectory visualizations (QA)
python generate_trajectory_visualizations.py --fast --workers 4

# 2) Run all methodologies (per-strain analyses)
python methodology_levy_flight.py
python methodology_ml_screening.py
python methodology_tda.py

# 3) Summarize results
python summarize_levy_analysis.py
python summarize_ml_screening.py
python summarize_tda.py
```

If you change any of these entrypoints or filenames, reflect that here.

---

## 9. Methodologies NOT Implemented

The following papers/methodologies were reviewed but **not implemented** because they require data beyond centroid trajectories:

- **Koren et al. (2015)** - SIFT visual features: Requires raw video frames, not applicable to trajectory data
- **Sohrabi et al. (2021)** - Parkinson's curling detection: Requires posture detection from images
- **Puchalt et al. (2022)** - Multi-view tracking: Hardware paper, not analysis methodology
- **Zhao et al. (2024)** - BAAIWorm: Brain-body simulator, not data analysis

**Previously implemented but removed:**

- **Unsupervised Behavioral States (GMM clustering)**: Originally implemented as `methodology_unsupervised_states.py` using Gaussian Mixture Models on kinematic features. Removed because it had no valid published reference for this specific approach (the initially cited Koren 2015 paper uses SIFT, not GMM on kinematics). The methodology was ad-hoc without formal validation and lacked statistical testing for state differences between treatments.

Only methodologies verified to work with centroid position data (x, y coordinates at 1 Hz) and backed by published, applicable references are implemented.
