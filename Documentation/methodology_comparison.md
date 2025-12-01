# Critical Methodology Comparison: Papers vs. Implementation

## Purpose

This document provides a **critical, detailed comparison** between published methodologies and their implementation in this repository. It identifies strengths, weaknesses, and deviations from source papers.

**Last Updated**: 2025-01-30

---

## 1. Topological Data Analysis (TDA)

### 1.1 Source Paper

**Thomas, A., Bates, K., Elchesen, A., Hartsock, I., Lu, H., & Bubenik, P. (2021)**
*Topological Data Analysis of C. elegans Locomotion and Behavior*
Frontiers in Artificial Intelligence, 4:668395

### 1.2 What the Paper Does

**Experimental setup**:
- Adult C. elegans immersed in **methylcellulose** at different viscosities (0.5%, 1%, 2%)
- 30 fps video acquisition
- Worms tracked using skeleton midline (100-dimensional time series per frame)
- Goal: Classify environmental viscosity from movement patterns

**Methodology**:
1. **Data preprocessing**: Extract midline skeleton → 100D time series
2. **Patch segmentation**: Divide videos into overlapping 300-frame patches (50% overlap)
3. **Sliding window embedding**: Window length L ∈ {1, 10, 20, 30} frames
   - Primary analysis uses L=20 (0.67 seconds at 30 fps = ~1 forward crawl period)
4. **Topological computation**:
   - Vietoris-Rips complex on embedded point cloud
   - Compute H₁ (degree 1) persistent homology
   - Extract persistence landscapes with depth 5, discretized at resolution ~1000 points
5. **Classification**:
   - Multiclass SVM (RBF kernel, C=10)
   - 10-fold cross-validation, 20 repetitions
   - Feature: Flattened persistence landscapes

**Window length selection**:
- Paper cross-validates L ∈ {1, 10, 20, 30}
- Finding: L=1 (raw data) gives **highest accuracy** (95.5%)
- But L=20 provides better **interpretability** (clearer PCA separation)
- **Trade-off acknowledged**: accuracy vs. interpretability

**Key results**:
- 95.5% accuracy distinguishing 3 viscosity classes (L=1)
- 83.4% accuracy with L=20

### 1.3 What Our Implementation Does

**File**: `methodology_tda.py`

**Data context** (CRITICAL DIFFERENCE):
- Adult C. elegans on **agar plates** treated with cannabinoids
- 1 Hz sampling (vs 30 Hz in paper)
- **Centroid tracking only** (x, y) → 2D time series (vs 100D skeleton)
- Goal: Classify treatment (Control vs CBD/CBDV/etc.)

**Implementation**:
1. **No patch segmentation**: Uses full trajectories
2. **Sliding window embedding**: L ∈ {10, 20, 50} frames
   - At 1 Hz: L=20 → 20 seconds (vs 0.67 seconds in paper)
3. **Centering**: Subtracts mean position (removes spatial drift)
4. **Topological computation**:
   - Vietoris-Rips with `max_edge_length=np.inf` (unbounded)
   - H₁ persistent homology
   - Landscape: `num_landscapes=5, resolution=100` (vs ~1000 in paper)
5. **Classification**:
   - Binary SVM per treatment (Control vs each treatment)
   - RBF kernel, C=1.0 (vs C=10 in paper)
   - 70/30 train-test split, single run (vs 20×10-fold CV in paper)
6. **Window selection**: Mean accuracy across treatments (highest wins)

### 1.4 Critical Analysis

#### ✅ Strengths

1. **Core pipeline intact**: Sliding window → Rips → H₁ → Landscapes → SVM
2. **Window length optimization**: Tests multiple L values and selects best
3. **Computational efficiency**: Parallel processing, reasonable discretization

#### ⚠️ Moderate Issues

1. **Temporal resolution mismatch**:
   - Paper: 30 Hz → L=20 ≈ 0.67s (one crawl cycle)
   - Ours: 1 Hz → L=20 ≈ 20s (biological meaning unclear)
   - **Impact**: Window length rationale breaks down

2. **Different window range**:
   - Paper: {1, 10, 20, 30}
   - Ours: {10, 20, 50}
   - Missing L=1 (raw data, paper's **best performer**)

3. **Landscape resolution**:
   - Paper: ~1000 points (fine discretization)
   - Ours: 100 points (10× coarser)
   - **Potential impact**: Loss of topological detail

4. **SVM hyperparameters**:
   - Paper: C=10 (stronger regularization control)
   - Ours: C=1.0 (default)
   - **Likely minor** but unexplained deviation

#### 🚨 Critical Issues

1. **No patch segmentation**:
   - Paper divides long videos into 300-frame patches
   - Ours uses full trajectories (variable length)
   - **Problem**: Some trajectories may be too short (<20 frames) or too long (>1000 frames)
   - **Impact**: Feature scale inconsistency across worms

2. **Validation scheme**:
   - Paper: 20 repetitions × 10-fold CV (robust estimate)
   - Ours: Single 70/30 split (high variance)
   - **Problem**: Accuracy estimates unreliable, no confidence intervals

3. **Biological context shift**:
   - Paper: Viscosity (physical impediment) affects movement **periodicity**
   - Ours: Cannabinoids (neurochemical) may affect **speed, pausing, exploration**
   - **Hypothesis violation**: TDA detects periodic loops; not clear cannabinoids alter periodicity

4. **Dimensionality reduction**:
   - Paper: 100D skeleton → rich shape information
   - Ours: 2D centroid → **only positional** info
   - **Critical**: Centroid loses posture (body curvature, head movements)
   - **Impact**: May miss phenotypes detectable by Thomas et al.

#### 📊 Quantitative Comparison

| Aspect | Thomas et al. 2021 | Our Implementation | Assessment |
|--------|-------------------|-------------------|------------|
| Input dimensions | 100D (skeleton) | 2D (centroid) | ⚠️ 50× reduction |
| Sampling rate | 30 Hz | 1 Hz | ⚠️ 30× reduction |
| Window length (time) | 0.67s | 20s | ❌ 30× longer |
| Window lengths tested | 1, 10, 20, 30 | 10, 20, 50 | ⚠️ Missing L=1 |
| Landscape resolution | ~1000 | 100 | ⚠️ 10× coarser |
| Validation | 20×10-fold CV | 1×70/30 split | ❌ Weak |
| SVM C parameter | 10 | 1.0 | ⚠️ Minor |

### 1.5 Recommendations

1. **Add L=1 test**: Paper's best performer
2. **Implement proper cross-validation**: At minimum 10-fold CV
3. **Consider patch segmentation**: Standardize trajectory lengths
4. **Document biological hypothesis**: Why expect topological changes with cannabinoids?
5. **Test on known phenotype**: Validate with unc mutants first

---

## 2. Lévy Flight Analysis

### 2.1 Source Paper

**Moy, K., Li, W., Tran, H.P., et al. (2015)**
*Computational Methods for Tracking, Quantitative Assessment, and Visualization of C. elegans Locomotory Behavior*
PLOS ONE, 10(12):e0145870

### 2.2 What the Paper Does

**Experimental setup**:
- Adult C. elegans on **agar plates** with/without food
- Centroid tracking at **1 Hz** (Δt = 1 second)
- Minimum 20 minutes recording
- Goal: Distinguish local search (Brownian) from global search (Lévy flight)

**Methodology** (cites Reynolds et al. 2007, Codling et al. 2008):
1. **Sampling**: Centroid (Cₓ, Cᵧ) at Δt = 1 sec intervals
2. **Turning event detection**:
   - Compute heading angle φ between consecutive positions
   - Turning event if |φ_current - φ_previous| > Θ (threshold)
   - Paper uses Θ = 40° (empirically determined)
3. **Step length calculation**:
   - Step = Euclidean distance between consecutive turning events
   - S_j = √[(Cₓⱼ - Cₓⱼ₋₁)² + (Cᵧⱼ - Cᵧⱼ₋₁)²]
4. **Power-law fitting**:
   - Use `powerlaw` library (Clauset et al. 2009 method)
   - Fit: P(step > s) ~ s^(-α) for s ≥ xₘᵢₙ
   - Maximum likelihood estimation of α and xₘᵢₙ
5. **Model comparison**:
   - Compare power-law vs lognormal via log-likelihood ratio R
   - p-value from Vuong's test
   - **Lévy-like if**: R > 0 and p < 0.05

**Key findings**:
- ~20% of food-deprived N2 show Lévy flight patterns
- α typically in range [1.5, 2.5] when Lévy-like

### 2.3 What Our Implementation Does

**File**: `methodology_levy_flight.py`

**Data context**:
- ✅ Same: Agar plates, 1 Hz sampling, centroid tracking
- Different: Cannabinoid treatments (not food deprivation)

**Implementation**:
1. **Sampling**: Uses raw 1 Hz data (correct)
2. **Turning event detection**:
   ```python
   vectors = np.diff(points, axis=0)
   angles = np.arctan2(vectors[:, 1], vectors[:, 0])
   angles_unwrapped = np.unwrap(angles)  # ← Key step
   turning_angles = np.diff(angles_unwrapped)
   turning_angles_deg = np.abs(np.rad2deg(turning_angles))
   turn_indices = np.where(turning_angles_deg > 40)[0] + 1
   ```
3. **Step lengths**: Euclidean distance (correct)
4. **Power-law fitting**: `powerlaw.Fit(all_step_lengths, discrete=False)`
5. **Model comparison**: R, p from `.distribution_compare('power_law', 'lognormal')`

### 2.4 Critical Analysis

#### ✅ Strengths

1. **High fidelity to paper**:
   - Identical sampling rate (1 Hz)
   - Same data type (centroid)
   - Same threshold (40°)
   - Same statistical test (R, p-value)

2. **Correct angle handling**:
   - Uses `np.unwrap()` to avoid ±180° discontinuities
   - This is **better** than simple modulo wrapping

3. **Proper method citation**:
   - Paper cites Reynolds 2007 and Codling 2008
   - We indirectly follow same algorithm

#### ⚠️ Moderate Issues

1. **Pooling across worms**:
   - Paper: Analyzes individual worms, reports % showing Lévy
   - Ours: Pools all step lengths per treatment group
   - **Impact**: Cannot report "% of worms with Lévy pattern"
   - **Consequence**: Lose individual-level heterogeneity

2. **No minimum step length filtering**:
   - Paper mentions xₘᵢₙ is estimated, but also discusses filtering very short steps
   - Ours: Uses all step lengths > 0
   - **Potential issue**: Very short steps (<1 pixel) may be noise

#### 🚨 Critical Issues

1. **Multiple comparisons problem**:
   - Paper: ~2-3 conditions tested (food vs no food, few genotypes)
   - Ours: 7 strains × 8 treatments ≈ **56 tests**
   - **No correction** for multiple comparisons (e.g., Bonferroni, FDR)
   - **Expected false positives**: 56 × 0.05 ≈ **2.8 "significant" results by chance**
   - **Critical flaw**: Cannot distinguish real Lévy patterns from Type I errors

2. **Biological context**:
   - Paper: Food deprivation → adaptive foraging strategy (well-established)
   - Ours: Cannabinoids → unknown effect on search strategy
   - **Problem**: Applying foraging theory to pharmacology unclear

3. **Sample size per treatment**:
   - Paper doesn't specify minimum, but emphasizes 20-minute recordings
   - Ours: Some treatments may have few worms or short trajectories
   - **Issue**: Power-law fitting unreliable with <50-100 step lengths

4. **No biological validation**:
   - Paper validates with food presence/absence (known to modulate search)
   - Ours: No positive control (e.g., food-deprived vs fed comparison)

#### 📊 Quantitative Comparison

| Aspect | Moy et al. 2015 | Our Implementation | Assessment |
|--------|----------------|-------------------|------------|
| Sampling rate | 1 Hz | 1 Hz | ✅ Identical |
| Threshold angle | 40° | 40° | ✅ Identical |
| Step length calc | Euclidean | Euclidean | ✅ Identical |
| Power-law method | MLE | MLE (powerlaw lib) | ✅ Identical |
| Model comparison | R, p-value | R, p-value | ✅ Identical |
| Angle handling | Not specified | `unwrap()` | ✅ Robust |
| Pooling strategy | Per worm | Per treatment | ⚠️ Different |
| Multiple comparison correction | None (2-3 tests) | None (56 tests) | ❌ Critical |

### 2.5 Recommendations

1. **Add multiple comparison correction**: Benjamini-Hochberg FDR at minimum
2. **Report per-worm statistics**: % of worms showing Lévy per treatment
3. **Minimum sample size filter**: Require ≥50 step lengths per analysis
4. **Positive control**: Include food-deprived vs fed condition
5. **Confidence intervals**: Bootstrap α estimates

---

## 3. Machine Learning Screening

### 3.1 Source Papers (Distinction Critical)

**Primary reference**:
**García-Garví, A. & Sánchez-Salmerón, A.J. (2025)**
*High-throughput behavioral screening in Caenorhabditis elegans using machine learning for drug repurposing*
Scientific Reports, 15:26140

**Data source cited by García-Garví**:
**O'Brien, T.J., Barlow, I.L., Feriani, L., & Brown, A.E. (2025)**
*High-throughput tracking enables systematic phenotyping and drug repurposing in C. elegans disease models*
eLife, 12:RP92491

⚠️ **Critical**: García-Garví **analyzes** O'Brien's data. O'Brien is the **original method developer**.

### 3.2 What O'Brien et al. 2025 Does (Cited by García-Garví)

**Experimental setup**:
- CRISPR-generated disease models (25 strains)
- Tierpsy Tracker: **256 features** from **skeleton tracking**
  - 90 kinematic features (speed, curvature, etc.)
  - 166 morphological features (length, area, body segments, etc.)
- High-throughput imaging platform (16 wells, 3 worms/well)

**Statistical validation pipeline**:
1. **Block permutation tests**: 100,000 permutations per feature
   - Blocks = experiments (control for batch effects)
2. **Benjamini-Yekutieli correction**: FDR < 0.10
   - More conservative than Benjamini-Hochberg
3. **Feature selection**: Only significant features used

**Critical finding reported**:
> "When analysis was expanded to 256 predefined features, **no hits were detected**"

**Why?** Stringent correction + 256 comparisons → low statistical power

### 3.3 What García-Garví et al. 2025 Proposes

**Contribution**: ML alternative to statistical testing

**Two approaches compared**:
1. **Traditional ML**: Random Forest on Tierpsy features (256 features)
2. **Deep Learning**: CNN-Transformer on video clips

**Findings**:
- Random Forest outperforms DL slightly
- ML can detect subtle patterns missed by p-value thresholding
- **But**: Still uses Tierpsy Tracker (256 features from skeleton)

### 3.4 What Our Implementation Does

**File**: `methodology_ml_screening.py`

**Data context** (CRITICAL DIFFERENCE):
- **Input**: Centroid trajectories (x, y) only
- **No skeleton**, no morphology

**Feature extraction**:
```python
features = {
    'speed_mean': np.mean(speeds),
    'speed_std': np.std(speeds),
    'speed_median': np.median(speeds),
    'turning_mean': np.mean(np.abs(turning_angles)),
    'turning_std': np.std(turning_angles),
    'path_length': np.sum(speeds),
    'displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
    'confinement_ratio': displacement / path_length
}
```
**Total: 8 features** (vs 256 in O'Brien/García-Garví)

**Statistical validation** (IMPLEMENTED):
1. Block permutation tests: 5000 permutations (vs 100,000 in O'Brien)
2. Benjamini-Yekutieli correction: FDR < 0.10
3. Feature selection: Uses only significant features
4. If zero significant: Uses all (reports low power)

**Classification**:
- Random Forest (n_estimators=100, class_weight='balanced')
- Binary: Control vs each treatment
- 70/30 train-test split
- Metrics: accuracy, precision, recall, F1

### 3.5 Critical Analysis

#### ✅ Strengths

1. **Statistical validation present**:
   - Contrary to initial documentation errors, code **does implement** permutation + BY
   - Follows O'Brien's statistical framework
   - Reports p-values and q-values

2. **Feature selection logic**:
   - Properly identifies significant features
   - Falls back gracefully if none pass threshold

3. **Block design**:
   - Uses `experiment_id` for blocked permutations
   - Controls batch effects

#### ⚠️ Moderate Issues

1. **Fewer permutations**:
   - O'Brien: 100,000
   - Ours: 5,000
   - **Impact**: Slightly less precise p-values, but likely adequate

2. **Train-test split**:
   - Ours: Single 70/30 split
   - Better: k-fold CV or repeated splits
   - **Impact**: Accuracy variance unquantified

#### 🚨 Critical Issues

1. **Feature count mismatch**:
   - O'Brien/García-Garví: **256 features** (skeleton morphology + kinematics)
   - Ours: **8 features** (centroid kinematics only)
   - **97% reduction in feature space**

2. **O'Brien's key finding applies to us**:
   - O'Brien: "With 256 features, **no hits detected** after statistical correction"
   - Implication: Even 256 features insufficient with rigorous stats
   - **Our situation**: Only 8 features → **even lower power**
   - **Expected outcome**: Most treatments show zero significant features

3. **Biological interpretation gap**:
   - Morphological changes (body length, curvature) often critical for phenotyping
   - Centroid alone misses:
     - Body posture
     - Head oscillations
     - Omega turns
     - Curling/coiling
   - **Impact**: Can only detect **gross motor phenotypes** (speed, wandering)

4. **No feature engineering**:
   - Missing from our 8:
     - Percentiles of speed (min, q25, q75, max)
     - Angular acceleration
     - Curvature proxies (turning radius)
     - Temporal features (autocorrelation, periodicity)
   - Even with centroid, could extract ~20-30 features

5. **Multiple comparison burden**:
   - 7 strains × 8 treatments × 8 features ≈ **448 feature tests**
   - BY correction very conservative here
   - **Expected**: Most features won't pass FDR < 0.10

#### 📊 Quantitative Comparison

| Aspect | O'Brien 2025 | García-Garví 2025 | Our Implementation | Assessment |
|--------|-------------|------------------|-------------------|------------|
| Input data | Skeleton | Skeleton (Tierpsy) | Centroid | ❌ Major diff |
| Features | 256 | 256 | **8** | ❌ 97% reduction |
| Permutations | 100,000 | N/A (uses ML) | 5,000 | ⚠️ Adequate |
| Correction | BY (FDR<0.10) | None (ML-based) | BY (FDR<0.10) | ✅ Matches O'Brien |
| Classifier | - | Random Forest | Random Forest | ✅ Matches García-Garví |
| Validation | Manual review | Train/val/test | Single 70/30 | ⚠️ Weaker |

### 3.6 Recommendations

1. **Acknowledge severe limitations**:
   - With 8 features vs 256, expect **very low detection power**
   - Document: "Best for gross motor changes only"

2. **Expand feature set** (still centroid-only):
   - Speed: add min, q25, q75, max (4 more)
   - Turning: add angular acceleration, max turning (2 more)
   - Path: add radius of gyration, fractal dimension (2 more)
   - Temporal: add autocorrelation lag-1 (1 more)
   - **Target: ~20 features** from centroid

3. **Positive controls**:
   - Include known phenotypes (unc mutants, daf-2, etc.)
   - Validate: Can we even detect gross motor defects?

4. **Cross-validation**:
   - Minimum: 5-fold CV
   - Better: Nested CV for hyperparameter tuning

5. **Consider alternative approaches**:
   - If skeleton data ever available: Re-run with Tierpsy features
   - Meanwhile: Document this as **preliminary/low-power** screen

---

## 4. Summary and Overall Assessment

### 4.1 Methodology Fidelity Ranking

1. **Lévy Flight** (90% fidelity)
   - ✅ Algorithm identical
   - ✅ Data type matches (centroid, 1 Hz)
   - ❌ Missing multiple comparison correction
   - ❌ Pooling strategy differs

2. **TDA** (70% fidelity)
   - ✅ Core pipeline preserved
   - ⚠️ Window lengths mismatch
   - ❌ Validation scheme weak
   - ❌ Dimensionality 50× lower (centroid vs skeleton)

3. **ML Screening** (60% fidelity)
   - ✅ Statistical validation implemented
   - ✅ Random Forest matches García-Garví
   - ❌ Feature count 97% lower (8 vs 256)
   - ❌ Biological relevance questionable

### 4.2 Critical Limitations Summary

#### Data Limitations (Shared)
- **Centroid-only tracking**: Loses morphology, posture, curvature
- **1 Hz sampling**: Adequate for long-timescale search, marginal for TDA
- **Agar environment**: Static (vs liquid in some papers)

#### Lévy Flight
- **56 tests without correction**: ~3 false positives expected
- **No per-worm analysis**: Can't report % showing Lévy
- **Biological hypothesis weak**: Cannabinoids ≠ foraging

#### TDA
- **30× longer windows**: Biological meaning lost
- **Single train-test split**: Unreliable accuracy
- **Missing L=1**: Paper's best performer
- **50× lower dimensionality**: May miss relevant topology

#### ML Screening
- **8 vs 256 features**: Catastrophic power loss
- **O'Brien precedent**: Even 256 features showed "no hits" with stats
- **Implication**: Most treatments will show **zero significant features**

### 4.3 Strengths Worth Preserving

1. **Lévy**: Clean implementation, correct angle handling
2. **TDA**: Proper window length optimization, parallelization
3. **ML**: Full statistical validation (permutation + BY + selection)

### 4.4 Recommendations for Revision

#### Priority 1 (Critical)
1. **Lévy**: Add Benjamini-Hochberg correction
2. **TDA**: Implement k-fold cross-validation
3. **ML**: Expand to ~20 features, document limitations
4. **All**: Add positive controls (unc mutants, food deprivation)

#### Priority 2 (Important)
1. **Lévy**: Report per-worm statistics
2. **TDA**: Test L=1, reduce landscape resolution mismatch
3. **ML**: Nested CV for hyperparameters
4. **All**: Confidence intervals / bootstrapped estimates

#### Priority 3 (Nice to have)
1. **TDA**: Patch segmentation for length standardization
2. **ML**: Feature importance stability (across CV folds)
3. **All**: Automated reporting (Markdown summaries)

---

## 5. Conclusions

### 5.1 Methodological Validity

All three methodologies are **algorithmically sound** when compared to their source papers. The core mathematical procedures (persistent homology, power-law fitting, Random Forest) are correctly implemented.

### 5.2 Biological Applicability

The **critical issue** is not implementation fidelity, but **data compatibility**:

- **Papers assume**: Rich data (skeleton, high frame rate)
- **We have**: Minimal data (centroid, 1 Hz)

This is not a code problem—it's a **fundamental limitation**.

### 5.3 Statistical Robustness

- **Lévy**: Needs multiple comparison correction (easy fix)
- **TDA**: Needs proper cross-validation (moderate fix)
- **ML**: Needs more features (hard fix without better tracking)

### 5.4 Expected Outcomes

**Realistic expectations given our data**:

1. **Lévy Flight**: May detect changes in search strategy (if cannabinoids affect exploration)
   - But: High false positive risk without correction
   - Solution: Apply FDR correction

2. **TDA**: Unclear if cannabinoids alter movement periodicity
   - Topology best suited for rhythmic behavior changes
   - May show null results (not a failure—just wrong tool for question)

3. **ML Screening**: Low power to detect subtle effects
   - Will reliably detect **only gross motor defects** (e.g., paralysis, hyperactivity)
   - Subtle phenotypes (body posture, turning style) → missed

### 5.5 Path Forward

**Option A** (Current data):
- Apply statistical corrections
- Add positive controls
- **Accept low sensitivity** for subtle phenotypes
- Document limitations clearly

**Option B** (Better data):
- Re-track with skeleton extraction (Tierpsy, OpenWorm, etc.)
- Higher frame rate acquisition (10-30 Hz)
- Re-run all methods with full features

**Option C** (Hybrid):
- Use current methods as **screening** (high specificity)
- Validate hits with detailed manual annotation
- Focus biological follow-up on strong signals only

---

**End of Document**
