# Proposed Methodologies for C. elegans Trajectory Data Analysis

## Introduction

This document outlines methodologies applicable to centroid trajectory data (x, y coordinates by frame) from C. elegans behavioral experiments. All methodologies listed here are verified to work with centroidal position data sampled at 1 Hz.

**Data specification**:
- Input: (x, y) position coordinates per frame
- Sampling rate: 1 Hz (WMicrotracker SMART)
- Format: CSV with columns: experiment_id, track_id, frame, x, y

**Papers reviewed and applicable**:
1. Topological Data Analysis - Thomas et al. (Frontiers in AI, 2021)
2. Lévy Flight Analysis - Moy et al. (PLOS ONE, 2015)
3. Machine Learning Screening - García-Garví & Sánchez-Salmerón (Scientific Reports, 2025)

---

## 1. Topological Data Analysis (TDA) for Behavior Classification

**Reference**: Thomas, A., Bates, K., Elchesen, A., Hartsock, I., Lu, H., & Bubenik, P. (2021). Topological Data Analysis of C. elegans Locomotion and Behavior. *Frontiers in Artificial Intelligence*, 4:668395.

### Paper Summary
Uses persistent homology on sliding window embeddings of centroid trajectories to detect topological features (loops) in movement patterns. Applies Vietoris-Rips filtration and persistence landscapes for classification.

### Why This Works With Centroid Data
- Paper explicitly uses centroid position data (x, y) or midline skeletons
- Sliding window embedding works with any time series
- Does not require morphological features

### Methodology for Our Data

**Preprocessing**:
- Center trajectories (subtract mean position) to remove spatial drift
- Create sliding window embeddings of length L (test L ∈ {10, 20, 50})

**Feature Extraction**:
1. For trajectory T = [(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)]
2. Create windows of length L: wᵢ = [xᵢ, yᵢ, xᵢ₊₁, yᵢ₊₁, ..., xᵢ₊ₗ₋₁, yᵢ₊ₗ₋₁]
3. Each window is a point in ℝ²ᴸ (e.g., L=20 → 40D space)
4. Compute Vietoris-Rips complex on point cloud
5. Extract H₁ (loops) persistence diagrams
6. Compute persistence landscapes (discretized to ~500-1000 points)

**Classification**:
- Support Vector Machine (SVM) with RBF kernel
- Hyperparameters: test multiple window lengths, select by cross-validation
- 70/30 train-test split, stratified by treatment

**Implementation Tools**:
- `gudhi` (Python): Rips complex, persistence, landscapes
- `scikit-learn`: SVM classifier, StandardScaler
- `numpy`: sliding windows, data manipulation

**Expected Outputs**:
- Persistence diagrams and barcodes per treatment
- Persistence landscapes (visualization)
- Classification accuracy (Control vs Treatment)
- Feature importance via landscape peak locations

---

## 2. Lévy Flight and Step Length Analysis

**Reference**: Moy, K., Li, W., Tran, H.P., Simonis, V., Story, E., Brandon, C., Furst, J., Raicu, D., & Kim, H. (2015). Computational Methods for Tracking, Quantitative Assessment, and Visualization of C. elegans Locomotory Behavior. *PLOS ONE*, 10(12):e0145870.

### Paper Summary
Analyzes search strategies by identifying turning events and fitting power-law distributions to step lengths. Distinguishes Lévy flight (power-law) from Brownian walk (exponential/lognormal) patterns.

### Why This Works With Centroid Data
- Paper explicitly uses centroid position (Cₓ, Cᵧ) for all calculations
- Only requires 2D position data, not morphology
- Verified compatible with 1 Hz sampling rate

### Methodology for Our Data

**Preprocessing**:
- Sample centroid coordinates at Δt = 1 second (already native rate)
- Ensure trajectories are sorted by frame

**Turning Event Detection**:
1. Compute movement vectors: vᵢ = (xᵢ₊₁ - xᵢ, yᵢ₊₁ - yᵢ)
2. Compute heading angles: φᵢ = arctan2(vᵢ_y, vᵢ_x)
3. Unwrap angles to avoid ±180° discontinuities: φ_unwrapped = unwrap(φ)
4. Compute angular changes: Δφᵢ = |φᵢ₊₁ - φᵢ|
5. Turning event if Δφᵢ > Θ (threshold = 40°)

**Step Length Calculation**:
- Step length Sⱼ = Euclidean distance between consecutive turning events
- Sⱼ = √[(Cₓⱼ - Cₓⱼ₋₁)² + (Cᵧⱼ - Cᵧⱼ₋₁)²]

**Power-Law Fitting**:
1. Pool all step lengths for a treatment group
2. Fit continuous power-law distribution (maximum likelihood)
3. Estimate parameters: α (exponent), xₘᵢₙ (minimum step length)
4. Compare power-law vs lognormal using log-likelihood ratio R and p-value

**Classification Criteria**:
- **Lévy-like**: R > 0 and p < 0.05 (power-law better than lognormal)
- **Brownian-like**: Otherwise
- **Optimal search**: α ≈ 2
- **Valid Lévy range**: 1 < α < 3

**Implementation Tools**:
- `powerlaw` (Python): Distribution fitting and comparison
- `numpy`: Vector calculations, angle unwrapping
- `matplotlib`: Distribution plots

**Expected Outputs**:
- α exponent per treatment
- Log-likelihood ratio R and p-value
- Step length distribution plots (empirical vs fitted)
- Classification summary (Lévy vs Brownian)

---

## 3. Machine Learning Screening with Kinematic Features

**Reference**: García-Garví, A. & Sánchez-Salmerón, A.J. (2025). High-throughput behavioral screening in Caenorhabditis elegans using machine learning for drug repurposing. *Scientific Reports*, 15:26140.

### Paper Summary
Uses machine learning classifiers on behavioral features to distinguish disease models from controls and evaluate drug recovery. Paper analyzes data from O'Brien et al. (2025) which used Tierpsy Tracker (256 features from skeleton). Proposes ML as alternative to statistical testing for detecting subtle patterns.

### Why This Works With Centroid Data (With Limitations)
- Core ML approach (Random Forest classification) is applicable
- Can extract kinematic features from centroid trajectories
- **Limitation**: Only ~8-10 features available vs 256 in paper (no skeleton morphology)
- Feature power is reduced but approach is valid

### Methodology for Our Data

**Feature Extraction from Centroid**:

From each trajectory, extract:

1. **Speed features** (4):
   - mean(speed), std(speed), median(speed), max(speed)
   - speed = ||[xᵢ₊₁-xᵢ, yᵢ₊₁-yᵢ]|| / Δt

2. **Turning angle features** (3):
   - mean(|turning_angle|), std(turning_angle), max(|turning_angle|)
   - turning_angle = angle between consecutive movement vectors

3. **Path structure features** (3):
   - path_length = Σ ||[xᵢ₊₁-xᵢ, yᵢ₊₁-yᵢ]||
   - net_displacement = ||[x_final - x_initial, y_final - y_initial]||
   - confinement_ratio = net_displacement / path_length

**Total: ~10 kinematic features**

**Statistical Validation** (following García-Garví/O'Brien approach):

1. **Permutation testing** (per feature):
   - Block permutation test (5000 permutations)
   - Blocks = experiment IDs (control for batch effects)
   - Compute p-value for each feature

2. **Multiple comparison correction**:
   - Apply Benjamini-Yekutieli correction (FDR < 0.10)
   - More conservative than Benjamini-Hochberg
   - Accounts for arbitrary feature dependencies

3. **Feature selection**:
   - Use only features with q_BY ≤ 0.10
   - If zero features pass: use all (report low power)

**Classification**:
- Random Forest classifier
- Hyperparameters: n_estimators=100, class_weight='balanced'
- Binary classification: Control vs each Treatment
- 70/30 train-test split, stratified
- Metrics: accuracy, precision, recall, F1-score

**Implementation Tools**:
- `scikit-learn`: RandomForestClassifier, StandardScaler, train_test_split
- `numpy`: Feature calculations, permutation tests
- `pandas`: Data aggregation per worm

**Expected Outputs**:
- Feature significance table (p-values, q-values, selected features)
- Classification accuracy per treatment
- Feature importance plots
- Recovery percentage (if framed as drug screening)

**Important Note**:
- With only 10 features vs 256, expect lower detection power
- May miss subtle phenotypes that require morphological features
- Best for detecting gross motor phenotypes (speed changes, confinement, etc.)

---

## Comparison and Recommendations

### When to Use Each Methodology

**TDA (Topological)**:
- Best for: Detecting periodic/rhythmic behaviors, complex movement patterns
- Strength: Captures global trajectory shape and periodicity
- Limitation: Computationally intensive, requires sufficient data per worm

**Lévy Flight**:
- Best for: Characterizing search strategies, area exploration patterns
- Strength: Biologically meaningful (foraging theory), simple interpretation
- Limitation: Requires long trajectories with many turning events

**ML Screening**:
- Best for: High-throughput screening, binary phenotype classification
- Strength: Handles non-linear patterns, provides statistical rigor
- Limitation: Limited features from centroid only, needs many samples

### Computational Considerations

**Sample size requirements**:
- TDA: ≥20 worms per treatment (for stable persistence features)
- Lévy: ≥50 step lengths per treatment (for power-law fitting)
- ML: ≥30 worms per class (for train/test split)

**Processing time** (approximate, single core):
- TDA: ~10-30 sec per worm (depends on window length)
- Lévy: <1 sec per worm
- ML: <1 sec for feature extraction + training

### Integration Strategy

Recommended pipeline for comprehensive analysis:

1. **Start with Lévy Flight**: Fast, provides overview of search behavior
2. **Run ML Screening**: Identifies which treatments have detectable effects
3. **Apply TDA to promising hits**: Deep dive into movement pattern changes

---

## Implementation Notes

### Data Quality Requirements

All methodologies require:
- Minimum trajectory length: 100 frames (Lévy), 20 frames (TDA/ML)
- No major tracking errors (large jumps, identity switches)
- Consistent frame rate across experiments

### Validation Recommendations

1. **Cross-validation**: 10-fold CV for TDA/ML classifiers
2. **Biological controls**: Include known phenotypes (e.g., unc mutants)
3. **Reproducibility**: Run on independent experimental replicates
4. **Visualization**: Always plot example trajectories and distributions

### Code Organization

Suggested structure:
```
utils_data.py          # Centralized data loading
methodology_levy_flight.py
methodology_tda.py
methodology_ml_screening.py
summarize_*.py         # Cross-methodology summaries
generate_figures.py    # Publication-ready plots
```

Each methodology script should:
- Load data via `utils_data.load_data_for_strain()`
- Process per strain independently
- Save results to `results/Analysis/<Methodology>/<Strain>/`
- Generate plots and CSV summaries
