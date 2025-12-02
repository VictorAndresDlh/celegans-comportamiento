# Results & Discussion: A Multi-Methodology, Multi-Strain Analysis of Cannabinoid-Induced Behavioral Phenotypes in *C. elegans*

## Introduction

The nematode *Caenorhabditis elegans* serves as a valuable model organism for neurobiology and drug discovery due to its genetic tractability, fully mapped connectome of 302 neurons, and well-characterized nervous system. Behavioral phenotyping through quantitative analysis of locomotion provides a critical readout for the effects of genetic mutations and pharmacological compounds. However, traditional methods rely on population averages of simple metrics such as mean speed or track length. This reductionist approach masks the diversity of behaviors within a population and fails to detect compounds that induce subtle or context-dependent changes.

This challenge becomes particularly relevant when studying cannabinoids, compounds known to have pleiotropic effects through multiple receptor systems. Simple metrics collapse this complexity into a single dimension, potentially missing therapeutic signals. Moreover, genetic background profoundly influences drug response, yet most studies examine only wild-type animals.

Here, we present a computational pipeline that provides high-dimensional characterization of cannabinoid-induced behavioral changes across multiple disease-relevant genetic backgrounds. The pipeline integrates three complementary approaches with varying degrees of adaptation from published methods:

1. **Lévy flight analysis (direct replication of Moy et al. [1]):** Our implementation closely follows the original methodology. Both studies use centroid tracking at 1 Hz, the same 40° turning threshold, and identical statistical tests (power-law vs. lognormal comparison). This method evaluates whether treatments alter the statistical structure of movement.

2. **Machine learning classification (adapted from García-Garví et al. [2]):** The original study uses 256 features extracted from skeleton tracking (Tierpsy Tracker). Our adaptation uses 20 centroid-derived kinematic features due to data constraints. The classification framework (Random Forest, XGBoost, cross-validation) follows the original methodology.

3. **Topological data analysis (adapted from Thomas et al. [3]):** The original study analyzes 100-dimensional skeleton data (body posture angles) at 30 fps. Our adaptation applies the same TDA pipeline (sliding window embedding → Vietoris-Rips → persistent homology → landscapes → SVM) to 2-dimensional centroid trajectories at 1 fps. This detects patterns in trajectory geometry rather than body posture cycles.

Each method addresses a different aspect of behavior, and together they provide a comprehensive phenotypic characterization.

---

## Materials and Methods

### Data Acquisition

Behavioral data was acquired using the WMicrotracker SMARTx8 system (Phylumtech S.A., Buenos Aires, Argentina). Adult *C. elegans* were cultivated on standard NGM plates seeded with *E. coli* OP50 under controlled conditions (20°C, 60% humidity). The system captures infrared images at 1 Hz frequency and reconstructs individual worm trajectories with sub-millimeter spatial resolution. Each trajectory consists of frame-by-frame (X, Y) coordinates in millimeters. Recording sessions lasted 10 minutes with 20-30 worms per condition.

### Strains

| Strain | Genetic Background | Disease Model |
|--------|-------------------|---------------|
| N2 | Wild-type (Bristol) | Reference |
| NL5901 | α-synuclein in muscle | Inclusion body myositis |
| UA44 | α-synuclein in dopaminergic neurons | Parkinson's disease |
| ROSELLA | Autophagy reporter (GFP/DsRed) | Autophagy dysfunction |
| BR5270 | Pro-aggregation Tau (K280del) | Tauopathy |
| BR5271 | Anti-aggregation Tau | Tauopathy control |
| TJ356 | DAF-16::GFP reporter | Insulin/IGF-1 signaling |

### Treatments

- **CBD:** 0.3µM, 3µM, 30µM
- **CBDV:** 0.3µM, 3µM, 30µM
- **ETANOL:** Vehicle control
- **Total Extract:** Full-spectrum botanical extract
- **Control:** Untreated

Exposure was from L4 stage for 24 hours before recording.

---

## 1. Lévy Flight Analysis

**Methodological fidelity: Direct replication.** Our implementation closely follows Moy et al. [1]. Both studies use centroid tracking at 1 Hz sampling, the same 40° turning threshold, and identical statistical framework (MLE fitting, power-law vs. lognormal comparison).

### 1.1. Theoretical Background

Lévy flight refers to a movement pattern where step lengths follow a power-law distribution. This pattern has been identified as optimal for searching sparse, randomly distributed resources [1]. The key signature is:

$$P(l) \propto l^{-\alpha}$$

where *l* is step length and α is the power-law exponent. When 1 < α ≤ 3, the movement qualifies as Lévy flight. When α > 3, movement transitions toward Brownian motion (random walk). An α ≈ 2 represents theoretically optimal search efficiency.

### 1.2. Computational Pipeline

The analysis followed the methodology of Moy et al. [1] and consisted of four stages:

**Stage 1: Trajectory Segmentation into Runs and Turns**

Each trajectory was decomposed into alternating "runs" (straight segments) and "turns" (direction changes). A turn was defined as any point where the angular change between consecutive movement vectors exceeded 40°.

The angle θ between consecutive movement vectors was calculated as:

$$\theta = \arccos\left(\frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}||\vec{v_2}|}\right)$$

where $\vec{v_1}$ and $\vec{v_2}$ are consecutive displacement vectors. Points with θ > 40° were marked as turn events.

**Stage 2: Run Length Extraction**

For each run segment (the path between consecutive turns), we calculated the Euclidean distance:

$$l = \sqrt{(x_{end} - x_{start})^2 + (y_{end} - y_{start})^2}$$

This produced a distribution of run lengths for each worm and, by pooling, for each treatment condition.

**Stage 3: Power-Law Fitting**

The power-law exponent α was estimated using Maximum Likelihood Estimation (MLE). MLE finds the α value that makes the observed data most probable given the power-law model. This approach is more robust than linear regression on log-transformed data.

A minimum run length threshold (x_min) was applied to exclude very short movements dominated by tracking noise. The threshold was determined by finding the x_min that minimizes the Kolmogorov-Smirnov statistic between the data and fitted model.

Confidence intervals for α were computed using bootstrap resampling (1000 iterations): the data was resampled with replacement, α was re-estimated for each resample, and the 2.5th and 97.5th percentiles defined the 95% CI.

**Stage 4: Statistical Validation**

Observing a power-law-like distribution is insufficient evidence; we must test whether power-law is significantly better than alternatives. We used a log-likelihood ratio test comparing power-law to lognormal (another heavy-tailed distribution common in biological data):

$$R = \log L_{power-law} - \log L_{lognormal}$$

- R > 0 with p < 0.05: Power-law is statistically superior
- R ≤ 0 or p ≥ 0.05: Cannot reject Brownian motion

**Multiple Comparison Correction**

With 47 tests performed, we applied Benjamini-Hochberg False Discovery Rate (FDR) correction at α = 0.05 to control for false positives.

**Per-Worm Analysis**

Following Moy et al. [1], we also computed α for each individual worm separately. This captures heterogeneity within treatment groups that population pooling obscures.

### 1.3. Results

#### 1.3.1. Summary Statistics

| Metric | Value |
|--------|-------|
| Total tests | 47 |
| Significant power-law (R>0, p<0.05) | 2 (4.3%) |
| Strict Lévy (1 < α ≤ 3) | 0 (0.0%) |
| Brownian-like | 45 (95.7%) |
| Surviving FDR correction | 14/47 |

The majority of cannabinoid treatments did not alter the statistical structure of movement from Brownian-like patterns.

#### 1.3.2. ROSELLA-Specific Response

Two conditions showed significant power-law patterns, both in the ROSELLA strain:

| Condition | α | 95% CI | R | p-value |
|-----------|---|--------|---|---------|
| ROSELLA + CBD 3µM | 3.176 | [2.47, 3.24] | 1.31 | 2.0×10⁻¹⁷ |
| ROSELLA + CBD 30µM | 3.044 | [2.96, 3.13] | 0.87 | 3.6×10⁻⁶ |

Both α values exceed 3.0, placing them in the transition zone between Lévy flight and Brownian motion. The movement shows more structure than pure random walk but does not achieve the efficiency characteristic of true Lévy flight (α ≤ 3).

**Interpretation:** The ROSELLA strain has altered autophagy pathways. Cannabinoids modulate autophagy through mTOR-independent mechanisms, and this interaction may affect neural circuits controlling movement decisions. The dose-independence (similar effects at 3µM and 30µM) suggests a threshold response.

#### 1.3.3. Per-Worm Heterogeneity

Population averages can mask important individual variation. Analysis of individual worms revealed:

| Condition | n worms | Mean α | Median α | % in Lévy range |
|-----------|---------|--------|----------|-----------------|
| UA44 Control | 438 | 6.16 ± 14.47 | 3.28 | 41.1% |
| BR5271 ETANOL | 115 | 4.48 ± 3.02 | 3.71 | 33.0% |
| ROSELLA Control | 302 | 6.92 ± 15.31 | 3.48 | 31.5% |

Despite population-level Brownian classification, 41.1% of individual UA44 worms showed α values within the Lévy range. This heterogeneity may reflect compensatory adaptations to dopaminergic dysfunction in subpopulations.

---

## 2. Machine Learning Classification

**Methodological fidelity: Adaptation.** García-Garví et al. [2] use 256 features from skeleton tracking (Tierpsy Tracker). Our implementation adapts their classification framework (Random Forest, XGBoost, stratified cross-validation) to 20 centroid-derived features. This reduction in feature space limits detection of subtle phenotypes but remains effective for gross motor changes.

### 2.1. Rationale

Classification accuracy provides an objective measure of phenotypic distinctiveness. If a classifier can reliably distinguish treated from control worms, the treatment produces a consistent behavioral signature. Accuracy near 50% (chance level for binary classification) indicates no distinguishable effect.

### 2.2. Computational Pipeline

The analysis followed García-Garví and Sánchez-Salmerón [2], with adaptations for centroid-based tracking. Note: García-Garví et al. use 256 features from skeleton tracking (Tierpsy Tracker); our implementation uses 20 centroid-derived features due to data constraints.

**Stage 1: Feature Extraction**

From each centroid trajectory, 20 kinematic features were computed:

*Speed Features (7):*
- Mean speed: $\bar{v} = \frac{1}{N}\sum_{i=1}^{N} v_i$ where $v_i = \frac{d_i}{\Delta t}$
- Speed standard deviation
- Maximum speed
- Median speed
- 25th and 75th percentile speeds
- Coefficient of variation: $CV = \frac{\sigma_v}{\bar{v}}$

*Turning Features (5):*
- Mean angular velocity: $\bar{\omega} = \frac{1}{N}\sum_{i=1}^{N} |\theta_i|$
- Angular velocity standard deviation
- Maximum angular velocity
- Reversal frequency (180° turns per minute)
- Turn rate (turns per unit distance)

*Path Features (5):*
- Tortuosity: $\tau = \frac{L}{D}$ where L is path length and D is net displacement
- Total distance traveled
- Net displacement (start to end)
- Path efficiency: $\eta = \frac{D}{L}$
- Radius of gyration: $R_g = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(r_i - \bar{r})^2}$

*Temporal Features (3):*
- Active time ratio (fraction of time with speed > threshold)
- Pause frequency (pauses per minute)
- Mean movement bout duration

**Stage 2: Data Normalization**

Features were standardized using z-score normalization:

$$z_i = \frac{x_i - \mu}{\sigma}$$

This ensures all features contribute equally regardless of their original scale.

**Stage 3: Classifier Training**

Three classifiers were compared:

1. **Random Forest (RF):** Ensemble of 100 decision trees. Each tree is trained on a bootstrap sample of the data and makes decisions based on feature thresholds. Final prediction is the majority vote. Parameters: max_depth=10, min_samples_split=5, class_weight='balanced'.

2. **XGBoost:** Gradient boosting algorithm that builds trees sequentially, with each tree correcting errors of the previous ensemble. Parameters: n_estimators=100, max_depth=10, learning_rate=0.1.

3. **Logistic Regression:** Linear model that estimates probability of class membership. L1 regularization was applied to prevent overfitting. Parameters: penalty='l1', solver='liblinear', class_weight='balanced'.

**Stage 4: Cross-Validation**

Stratified 10-fold cross-validation was used:
1. Data was divided into 10 equal parts (folds)
2. For each iteration, 9 folds were used for training, 1 for testing
3. This was repeated 10 times, each fold serving as test set once
4. Final accuracy is the mean across all folds

Stratification ensured each fold maintained the same class proportions as the full dataset. Cross-validation provides robust estimation of generalization performance without requiring additional statistical tests.

### 2.3. Results

#### 2.3.1. Classifier Comparison

| Classifier | Mean Accuracy | Std Dev | Best | Worst |
|------------|--------------|---------|------|-------|
| XGBoost | 76.2% | 0.123 | 95.1% | 53.2% |
| Random Forest | 67.2% | 0.101 | 95.2% | 52.2% |
| Logistic Regression | 57.7% | 0.090 | 72.5% | 27.9% |

XGBoost achieved the highest average accuracy and was the best-performing classifier in the majority of strain-treatment combinations. This suggests cannabinoid effects involve complex, non-linear interactions between movement features that gradient boosting captures more effectively than simpler methods.

#### 2.3.2. Exceptional Performers

Two combinations achieved >90% accuracy:

**NL5901 + CBD 3µM: 95.2% ± 0.003**

The muscle synucleinopathy model showed near-perfect classification with medium-dose CBD. This accuracy indicates that CBD 3µM produces a behavioral signature so consistent that 19 of every 20 worms are correctly classified.

The effect is dose-specific: CBD 0.3µM achieved only 56.1% and CBD 30µM achieved 70.2% in the same strain. The 3µM dose appears to hit a specific therapeutic window.

**UA44 + CBDV 3µM: 92.1% ± 0.008**

The dopaminergic synucleinopathy model showed similar exceptional performance with CBDV. The specificity to the Parkinson's disease model suggests CBDV may interact with dopaminergic or compensatory pathways.

#### 2.3.3. Strain Sensitivity

| Strain | Mean Accuracy | Most Distinguishable Condition |
|--------|--------------|-------------------------------|
| UA44 | 73.3% | CBDV 3µM (92.1%) |
| ROSELLA | 71.9% | CBD 3µM (75.5%) |
| NL5901 | 69.0% | CBD 3µM (95.2%) |
| N2 | 64.7% | CBD 30µM (76.5%) |
| BR5271 | 63.2% | ETANOL (72.2%)* |
| BR5270 | 57.1% | ETANOL (60.6%)* |
| TJ356 | 55.9% | ETANOL (59.0%)* |

*Note: For BR5270, BR5271, and TJ356, the vehicle (ETANOL) produced more distinguishable phenotypes than cannabinoid treatments. This suggests that in these strains, the ethanol vehicle itself induces detectable behavioral changes, while additional cannabinoid effects are minimal.

TJ356 consistently showed near-chance classification (55.9% average), indicating phenotypic rigidity. The DAF-16/FOXO pathway may buffer behavioral output against pharmacological perturbation.

#### 2.3.4. Treatment Rankings

| Treatment | Mean Accuracy | Best Result |
|-----------|--------------|-------------|
| CBD 3µM | 79.2% | 95.2% (NL5901) |
| CBD 30µM | 73.6% | 76.5% (N2) |
| CBDV 30µM | 72.9% | 83.7% (NL5901) |
| CBDV 3µM | 70.9% | 92.1% (UA44) |
| CBDV 0.3µM | 65.4% | 70.5% (UA44) |
| ETANOL | 62.8% | 72.2% (BR5271) |
| CBD 0.3µM | 60.5% | 75.4% (ROSELLA) |
| Total Extract | 59.4% | 68.2% (UA44) |

CBD 3µM emerged as the most effective treatment for inducing distinguishable kinematic phenotypes.

---

## 3. Topological Data Analysis

**Methodological fidelity: Significant adaptation.** Thomas et al. [3] analyze 100-dimensional skeleton data (body posture represented as 100 angles between midline segments) captured at 30 fps. Our implementation applies the same TDA pipeline to 2-dimensional centroid trajectories at 1 fps. Key differences:

| Aspect | Thomas et al. [3] | This study |
|--------|------------------|------------|
| Input data | Skeleton midline (100D) | Centroid position (2D) |
| Sampling rate | 30 fps | 1 fps |
| L=20 represents | 0.67 seconds | 20 seconds |
| Detects | Body posture cycles | Trajectory geometry |

The core TDA pipeline (sliding window → Vietoris-Rips → persistent homology → landscapes → SVM) remains identical. We added H0 homology (connected components) alongside the H1 (loops) used in the original study.

### 3.1. Rationale

Kinematic features capture local movement properties (speed, turns) but may miss changes in global path organization. Topological Data Analysis (TDA) captures geometric properties: the shapes, loops, and spatial arrangements that characterize different movement patterns. Two trajectories with identical speeds and turn rates can have fundamentally different topologies.

### 3.2. Computational Pipeline

The analysis followed Thomas et al. [3]:

**Stage 1: Sliding Window Embedding**

Raw trajectories are 2D (X, Y over time). To capture temporal structure, we used sliding window embedding to transform each trajectory into a higher-dimensional point cloud.

For a trajectory with T timepoints and window length L:
- Take consecutive windows of L points
- Flatten each window into a single vector of dimension 2L
- This produces (T - L + 1) points in 2L-dimensional space

Example: With L=10, each point represents 10 consecutive (X,Y) pairs, yielding a 20-dimensional vector. The collection of all such windows forms a point cloud that encodes the trajectory's temporal structure.

**Stage 2: Vietoris-Rips Complex Construction**

The point cloud was converted into a simplicial complex using the Vietoris-Rips construction:

1. Start with points as 0-simplices (vertices)
2. Connect points within distance ε to form 1-simplices (edges)
3. Fill triangles where all three edges exist (2-simplices)
4. Continue to higher dimensions as needed

The parameter ε was varied from 0 to ∞, and we tracked which topological features appeared and disappeared at each scale.

**Stage 3: Persistent Homology Computation**

Persistent homology tracks topological features across scales. Thomas et al. [3] focus on H1 (loops); we extended the analysis to include H0:

- **H1 (dimension 1):** Loops/cycles. These appear when edges form closed paths and disappear when the interior is filled. Long-lived H1 features indicate prominent cyclic or repetitive movement patterns. This is the primary feature used in Thomas et al. [3].

- **H0 (dimension 0):** Connected components (extension). Initially, each point is its own component. As ε increases, components merge. We included H0 to capture additional structural information about trajectory fragmentation.

Each feature has a birth time (ε at which it appears) and death time (ε at which it disappears). The persistence (death - birth) measures the feature's robustness.

Computation was performed using the GUDHI library with max_dimension=2.

**Stage 4: Persistence Landscapes**

Persistence diagrams (collections of birth-death pairs) are difficult to use directly in machine learning. We converted them to persistence landscapes, which are functional representations:

For each feature with birth b and death d, create a tent function centered at (b+d)/2 with height (d-b)/2. The landscape is the pointwise maximum of all tent functions.

We computed 5 landscapes at 1000-point resolution, yielding 5000-dimensional feature vectors per trajectory. These vectors encode all topological information in a format suitable for classification.

**Stage 5: Window Length Optimization**

The optimal window length L varies by strain. We tested L ∈ {1, 10, 20, 50} and selected the value maximizing classification accuracy for each strain.

**Stage 6: SVM Classification**

Support Vector Machine with RBF kernel was used for classification. SVM finds the hyperplane that maximally separates classes in the high-dimensional landscape space. Parameters: C=1.0, γ='scale'.

Stratified 10-fold cross-validation was applied as described in Section 2.2.

### 3.3. Results

#### 3.3.1. Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Excellent (>80%) | 7 | 17.9% |
| Good (65-80%) | 13 | 33.3% |
| Moderate (55-65%) | 9 | 23.1% |
| Poor (<55%) | 10 | 25.6% |
| Above chance (>50%) | 33/39 | 84.6% |

#### 3.3.2. Top Performers

| Strain | Treatment | TDA Accuracy | Window L |
|--------|-----------|--------------|----------|
| UA44 | CBD 30µM | 87.0% | 20 |
| UA44 | CBD 3µM | 86.7% | 20 |
| ROSELLA | CBD 0.3µM | 85.0% | 10 |
| ROSELLA | CBDV 30µM | 84.1% | 10 |
| ROSELLA | CBD 30µM | 82.5% | 10 |
| ROSELLA | CBD 3µM | 82.0% | 10 |
| UA44 | CBDV 3µM | 81.8% | 20 |

UA44 and ROSELLA dominated topological performance. UA44 achieved the single highest TDA accuracy (87.0% with CBD 30µM), indicating that CBD fundamentally reorganizes path geometry in the Parkinson's disease model.

#### 3.3.3. Optimal Window Lengths

For each strain, all four window lengths (L = 1, 10, 20, 50) were tested and the value maximizing mean classification accuracy was selected. Table 4 shows the optimal window length selected for each strain.

**Table 4. Optimal Window Length by Strain**

| Strain | Optimal L | Average TDA Accuracy | Best Treatment Result |
|--------|-----------|---------------------|----------------------|
| TJ356 | L = 1 | 57.6% | 61.0% (CBD 0.3µM) |
| ROSELLA | L = 10 | 77.5% | 85.0% (CBD 0.3µM) |
| N2 | L = 10 | 61.3% | 74.2% (CBDV 30µM) |
| UA44 | L = 20 | 70.1% | 87.0% (CBD 30µM) |
| NL5901 | L = 50 | 58.0% | 72.6% (CBD 3µM) |
| BR5270 | L = 50 | 54.8% | 63.7% (Total Extract) |
| BR5271 | L = 50 | 64.8% | 66.6% (CBD 0.3µM) |

The pattern of optimal window lengths reveals strain-specific behavioral timescales:

- **L = 1 (TJ356):** The DAF-16 reporter strain showed best discrimination at instantaneous timescales, suggesting rapid behavioral fluctuations.
- **L = 10 (ROSELLA, N2):** Medium-timescale patterns (~10 seconds) characterize autophagy reporter and wild-type strains.
- **L = 20 (UA44):** The dopaminergic Parkinson's model required longer windows, consistent with altered dopamine-dependent behavioral cycles.
- **L = 50 (NL5901, BR5270, BR5271):** Muscle synucleinopathy and Tau models showed long-range path organization, suggesting these pathologies affect extended behavioral sequences.

#### 3.3.4. Kinematic vs. Topological Comparison

| Strain | Treatment | ML Accuracy | TDA Accuracy | Dominant Method |
|--------|-----------|-------------|--------------|-----------------|
| NL5901 | CBD 3µM | 95.2% | 72.6% | Kinematic |
| UA44 | CBD 30µM | 73.7% | 87.0% | Topological |
| ROSELLA | CBD 0.3µM | 75.4% | 85.0% | Topological |

Different treatments manifest in different feature spaces. NL5901 effects are primarily kinematic (local movement changes), while UA44 and ROSELLA effects are primarily topological (global geometric changes). Comprehensive phenotyping requires both approaches.

---

## 4. Synthesis

### 4.1. Multi-Dimensional Phenotypic Fingerprint

Each strain-treatment combination produces a characteristic profile across three dimensions:

| Dimension | Best Performer | Value |
|-----------|---------------|-------|
| Foraging strategy (Lévy) | ROSELLA + CBD 3µM | α = 3.176 (p < 10⁻¹⁷) |
| Kinematic signature (ML) | NL5901 + CBD 3µM | 95.2% accuracy |
| Path geometry (TDA) | UA44 + CBD 30µM | 87.0% accuracy |

### 4.2. Key Findings

1. **Genetic background determines response.** CBD 3µM achieves 95.2% accuracy in NL5901 but only 55.8% in TJ356—a 40-point difference for the same treatment.

2. **Dose-response is non-linear.** Medium doses (3µM) often outperformed both low (0.3µM) and high (30µM) doses, suggesting a therapeutic window.

3. **Methods capture different phenotypes.** NL5901 shows strong kinematic but moderate topological effects; UA44 shows the opposite pattern.

4. **Individual heterogeneity is substantial.** 41% of UA44 worms show Lévy-like search despite population-level Brownian classification.

5. **Some strains show phenotypic rigidity.** TJ356 maintained stable behavior across all treatments, suggesting homeostatic buffering by the DAF-16/FOXO pathway.

### 4.3. Methodological Recommendations

1. Use multiple complementary methods; single-method approaches miss substantial signals.
2. Preserve individual-level data to capture heterogeneity.
3. Test multiple genetic backgrounds; wild-type studies miss important interactions.
4. Compare multiple classifiers; XGBoost generally outperformed alternatives.
5. Test multiple doses; non-linear dose-response requires dose ranging.

---

## Conclusions

Integration of Lévy flight analysis, machine learning classification, and topological data analysis across seven genetically diverse strains reveals that cannabinoid behavioral phenotypes are multi-dimensional, genetically contingent, and methodology-dependent.

Exceptional performers identified—NL5901 + CBD 3µM (95.2% kinematic), UA44 + CBD 30µM (87.0% topological), and ROSELLA + CBD (effects across all methods)—represent leads for mechanism-of-action studies.

The phenotypic fingerprint framework provides a template for comprehensive behavioral pharmacology that captures the full dimensionality of drug effects across genetic contexts.

---

## References

[1] Moy, K., et al. (2015). Computational methods for tracking, quantitative assessment, and visualization of *C. elegans* locomotory behavior. *PLOS ONE*, 10(12), e0145870.

[2] García-Garví, A., & Sánchez-Salmerón, A. J. (2025). High-throughput behavioral screening in *Caenorhabditis elegans* using machine learning for drug repurposing. *Scientific Reports*, 15, 26140.

[3] Thomas, A., et al. (2021). Topological data analysis of *C. elegans* locomotion and behavior. *Frontiers in Artificial Intelligence*, 4, 668395.

---

## Appendix A: Analysis Parameters

**Table A1. Configuration Parameters by Method**

| Method | Parameter | Value | Rationale |
|--------|-----------|-------|-----------|
| **Lévy Flight** | | | |
| | Turn threshold | 40° | Standard threshold [1] |
| | Bootstrap iterations | 1000 | Stable CI estimation |
| | x_min optimization | Automatic | KS-minimization |
| | FDR correction | Benjamini-Hochberg | α=0.05 |
| **ML Classification** | | | |
| | Number of features | 20 | Centroid-derived (vs. 256 in [2]) |
| | Cross-validation | Stratified 10-fold | Preserves class proportions |
| | RF trees | 100 | Performance/computation balance |
| | RF max_depth | 10 | Prevents overfitting |
| | XGBoost learning_rate | 0.1 | Conservative value |
| **TDA** | | | |
| | Window lengths tested | L ∈ {1, 10, 20, 50} | Strain-specific optimization |
| | Number of landscapes | 5 | Per Thomas et al. [3] |
| | Landscape resolution | 1000 points | High discrimination |
| | Feature vector dimension | 5000 | 5 landscapes × 1000 points |
| | Homology dimensions | H0, H1 | H1 from [3]; H0 extension |
| | SVM kernel | RBF | Non-linear boundaries |

**Window Length Selection for TDA**

The sliding window length L determines the temporal scale captured:
- L = 1: 1 second (instantaneous)
- L = 10: 10 seconds (short-term cycles)
- L = 20: 20 seconds (medium-term patterns)
- L = 50: 50 seconds (long-range organization)

For each strain, all four values were tested and the L maximizing cross-validated accuracy was selected.

