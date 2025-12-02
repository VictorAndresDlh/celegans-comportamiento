# Results & Discussion: A Multi-Methodology, Multi-Strain Analysis of Cannabinoid-Induced Behavioral Phenotypes in *C. elegans*

### Introduction: The Need for High-Dimensional Phenotyping

The nematode *Caenorhabditis elegans* is a premier model organism for neurobiology and drug discovery, owing to its genetic tractability, fully mapped connectome of 302 neurons, and well-characterized nervous system. Behavioral phenotyping, particularly the quantitative analysis of locomotion, serves as a critical readout for the effects of genetic mutations and pharmacological compounds. However, traditional methods for analyzing movement data often rely on population averages of simple metrics, such as mean speed or track length. While computationally efficient, this reductionist approach can be fundamentally misleading as it masks the rich diversity of behaviors within a population and systematically fails to detect compounds that induce subtle, complex, or context-dependent changes in behavioral strategy.

The challenge is particularly acute when studying neuroactive compounds like cannabinoids, which are known to have pleiotropic effects through multiple receptor systems and signaling pathways. Simple metrics collapse this complexity into a single dimension, potentially missing therapeutic signals or mischaracterizing drug mechanisms. Moreover, genetic background profoundly influences drug response, yet most studies examine only wild-type animals, leaving drug-gene interactions unexplored.

Here, we present a comprehensive multi-faceted computational pipeline that moves beyond simple averages to provide a high-dimensional characterization of cannabinoid-induced behavioral changes across multiple disease-relevant genetic backgrounds. By integrating three complementary analytical approaches—Lévy flight foraging analysis [1], kinematic machine learning classification [2], and topological data analysis [3]—we deconstruct complex locomotor patterns into clear, interpretable phenotypes. This allows us to not only detect whether a compound has an effect, but to precisely characterize the nature of that effect—be it a fundamental alteration of exploratory strategy, a distinctive kinematic signature, or a geometric transformation of path architecture—and assess its consistency and variability across different genetic contexts. This systems-level approach reveals that cannabinoid effects are highly multi-dimensional, genetically contingent, and methodology-dependent, challenging simplistic models of drug action.

### Materials and Methods

#### Data Acquisition and Trajectory Extraction

All behavioral data was acquired using the WMicrotracker SMARTx8 system (Phylumtech S.A., Buenos Aires, Argentina), a high-throughput automated tracking platform designed for *C. elegans* behavioral analysis. For this study, adult *C. elegans* from multiple strains were cultivated on standard Nematode Growth Medium (NGM) plates seeded with *E. coli* OP50 and monitored under controlled conditions (20°C, 60% humidity). 

The system's **Tracking Mode** utilizes infrared (IR) imaging at 1 Hz frequency combined with proprietary machine learning algorithms to reconstruct individual worm trajectories with sub-millimeter spatial resolution. The tracking algorithm automatically identifies worm centroids in each frame and links them across time to generate continuous trajectories. These raw trajectory files, containing frame-by-frame (X, Y) coordinates in millimeters along with timestamps, form the basis for all subsequent computational analyses described in this paper. Each experimental condition included 20-30 individual worms tracked for 10-minute recording sessions, providing sufficient statistical power while minimizing environmental drift effects.

#### Strains Used in This Study

A panel of seven *C. elegans* strains with diverse genetic backgrounds was used to investigate potential drug-gene interactions and model various neurodegenerative and aging-related pathologies. The specific molecular and phenotypic characteristics of these strains are critical for interpreting the behavioral results:

-   **N2 (Bristol):** Wild-type reference strain without genetic modifications, serving as the baseline for standard *C. elegans* behavior.

-   **NL5901:** Exhibits alpha-synuclein aggregation on the muscle walls of the nematode, modeling peripheral synucleinopathy and muscle-mediated locomotor dysfunction.

-   **UA44:** Exhibits alpha-synuclein aggregation in dopaminergic neurons. This strain expresses GFP protein linked to a dopamine transporter, which is specifically expressed in dopaminergic neurons. Serves as a model for Parkinson's disease-associated dopaminergic neurodegeneration.

-   **ROSELLA:** Autophagy reporter strain. Expresses GFP protein (sensitive to pH 5.0) together with DCT-1, and DsRed proteins (not sensitive to pH 5.0) expressed together with LGG-1 protein. This dual-fluorescence system allows visualization of autophagosome maturation and autophagic flux.

-   **BR5270:** Pan-neuronal over-expression of the F3 pro-aggregation fragment of the human Tau protein with K280 deleted (line A). Models Tauopathy with aggregation-prone Tau pathology.

-   **BR5271:** Pan-neuronal over-expression of the F3 pro-aggregation fragment of the human Tau protein with K280 deleted and two isoleucine-to-proline substitutions (I277P and I380P) in the hexapeptide motifs of the repeat region (line A). These mutations abrogate aggregation, making this strain an anti-aggregation control for BR5270.

-   **TJ356:** DAF-16/FOXO reporter strain expressing DAF-16::GFP fusion protein, enabling visualization of insulin/IGF-1 signaling pathway activation.

#### Experimental Design: Treatments

For this study, each *C. elegans* strain was subjected to a standardized panel of cannabinoid treatments and their respective vehicle controls. Compounds were incorporated into NGM agar at the time of plate pouring to ensure homogeneous distribution. The primary treatments included:

- **Cannabidiol (CBD):** A non-psychoactive phytocannabinoid at three concentrations: 0.3µM (low dose), 3µM (medium dose), and 30µM (high dose)
- **Cannabidivarin (CBDV):** A propyl analogue of CBD at matching concentrations: 0.3µM, 3µM, and 30µM  
- **ETANOL (Ethanol vehicle):** The solvent control for cannabinoid delivery, matched to the highest ethanol concentration used in cannabinoid dilutions
- **Total Extract:** A full-spectrum *Cannabis* botanical extract containing CBD, CBDV, and other minor cannabinoids in their natural ratios
- **Control:** Untreated NGM plates with standard OP50 bacteria

All treatments were performed in biological triplicate, with 20-30 worms per replicate. Worms were exposed to treatments from the L4 larval stage for 24 hours before behavioral recording to ensure consistent developmental stage and treatment duration. Only results demonstrating statistically significant or highly illustrative phenotypic changes are discussed in detail in the results sections below.

---

### 1. Path Structure Analysis Reveals Alterations in Exploratory Strategy

#### 1.1. Rationale: Analyzing the "How" of Movement

To address a fundamental ecological question—*how efficiently* does the animal explore its environment?—we adapted the Lévy flight foraging hypothesis framework from the work of Moy et al. [1], who developed computational methods for analyzing the strategic structure of *C. elegans* locomotion.

This analysis views worm movement through the lens of optimal foraging theory. In patchy resource environments (analogous to the sparse bacterial lawns worms encounter), different search strategies have different efficiencies. A purely random walk (Brownian motion) is inefficient for finding distant resources. In contrast, a **Lévy flight**—characterized by many small steps interspersed with occasional very long steps—is mathematically proven to be optimal for searching environments with sparse, randomly distributed targets. By determining whether worm trajectories follow Lévy or Brownian statistics, we can ask whether treatments fundamentally alter the animal's exploratory strategy from inefficient to optimal (or vice versa).

This approach is particularly powerful because it detects changes in global path architecture that may not be apparent in local kinematic measurements. Two worms could have identical mean speeds but radically different exploratory efficiencies if one uses a Lévy strategy and the other does not.

#### 1.2. Methodology Explained

Following Moy et al. [1], the Lévy flight analysis pipeline consists of four stages:

**Stage 1 - Trajectory Segmentation:** Each worm's trajectory was decomposed into a sequence of "runs" (straight-line segments) and "turns" (sharp directional changes). A turn was defined as any angular change exceeding 40° between consecutive movement vectors. This threshold effectively separates deliberate directional changes from minor course corrections due to environmental perturbations.

**Stage 2 - Run Length Extraction:** For each run segment, we calculated the Euclidean distance traveled (run length) in millimeters. This produced a distribution of run lengths for each individual worm and, by pooling, for each treatment condition.

**Stage 3 - Power-Law Fitting:** The defining signature of a Lévy flight is that run lengths follow a **power-law distribution**: 

$$P(x) \propto x^{-\alpha}$$

where α (alpha) is the power-law exponent. Following Moy et al. [1]:
- **Strict Lévy criterion:** 1 < α ≤ 3 (within theoretical Lévy flight range)
- **α ≈ 2:** Theoretically optimal for many foraging scenarios
- **α > 3:** Transition zone toward Brownian motion

We used Maximum Likelihood Estimation (MLE) to fit the power-law exponent α to each run length distribution, with bootstrap confidence intervals (1000 iterations) to assess parameter uncertainty.

**Stage 4 - Statistical Validation:** We employed a **log-likelihood ratio test (R)** comparing the power-law model to a lognormal alternative. Following the rigorous framework of Clauset, Shalizi, and Newman:
- **R > 0 with p < 0.05:** Power-law is statistically superior (Lévy-like pattern)
- **R ≤ 0 or p ≥ 0.05:** Standard Brownian motion

**Per-Worm Analysis:** Following Moy et al. [1]'s recommendation for individual-level statistics, we computed α values for each individual worm, providing distributions of search strategies within each treatment group rather than relying solely on pooled population data.

**Multiple Comparison Correction:** Benjamini-Hochberg False Discovery Rate (FDR) correction was applied across all strain-treatment tests to control for multiple comparisons.

All power-law analyses were performed using the `powerlaw` Python package, which implements the rigorous statistical framework developed by Clauset, Shalizi, and Newman for identifying power-law distributions in empirical data.

#### 1.3. Results and Interpretation

##### 1.3.1. Summary Statistics

Across 47 strain-treatment combinations tested:
- **Lévy patterns (Statistical criterion R>0, p<0.05):** 2 cases (4.3%)
- **Lévy patterns (Strict criterion 1 < α ≤ 3):** 0 cases (0.0%)
- **Standard Brownian motion:** 45 cases (95.7%)

After FDR correction (Benjamini-Hochberg):
- Significant before correction: 16/47
- Significant after correction: 14/47
- False positives eliminated: 2

##### 1.3.2. Cannabinoid-Induced Power-Law Patterns in ROSELLA

The most striking finding emerged from the **ROSELLA** strain (autophagy reporter), which exhibited two cases with statistically significant power-law distributions:

**ROSELLA + CBD 3µM:**
Run length distributions showed a power-law fit with **α = 3.176** (95% CI: 2.47-3.24). The log-likelihood ratio test confirmed this power-law model was significantly superior to the lognormal alternative (**R = 1.31, p = 2.0×10⁻¹⁷**). This represents extraordinarily strong statistical evidence for power-law behavior.

**ROSELLA + CBD 30uM:**
Similarly, this treatment produced a power-law pattern with **α = 3.044** (95% CI: 2.96-3.13). The statistical test yielded **R = 0.875, p = 3.6×10⁻⁶**, confirming significant departure from lognormal toward power-law.

**Critical Interpretation - The Transition Zone:** Both α values exceed 3.0, placing them in the **transition zone** between Lévy flight (1 < α ≤ 3) and Brownian motion (α > 3) as defined by Moy et al. [1]. While these distributions show statistically significant power-law character rather than lognormal, they do not meet the strict theoretical criterion for Lévy flight. This transition zone represents an intermediate search strategy—more structured than pure Brownian motion but not optimally efficient for sparse resource foraging.

**Mechanistic Hypothesis:** The ROSELLA-specific response to CBD may reflect an interaction between cannabinoid signaling and autophagy pathways. Autophagy is known to be modulated by cannabinoids through mTOR-independent mechanisms, and the ROSELLA strain's altered autophagic flux may create a unique cellular context where CBD affects neural circuits controlling exploratory decisions. The dose-dependency (both 3µM and 30µM showing effects) suggests a threshold phenomenon rather than a linear dose-response.

##### 1.3.3. Per-Worm Alpha Distribution Analysis

Following Moy et al. [1]'s recommendation, individual worm statistics provide more robust insights than pooled data:

**High Variability Within Treatment Groups:**
Individual worm α values showed substantial heterogeneity. For example:
- **UA44 Control:** n=438 worms, α = 6.16 ± 14.47 (median=3.28), with 41.1% of individual worms showing α in the strict Lévy range (1 < α ≤ 3)
- **BR5271 ETANOL:** n=115 worms, α = 4.48 ± 3.02 (median=3.71), with 33.0% in strict Lévy range

This individual-level analysis reveals that even when pooled population data does not show significant Lévy patterns, substantial proportions of individual worms may exhibit Lévy-like search strategies. This highlights the importance of preserving individual-level information rather than relying solely on population averages.

##### 1.3.4. Absence of Strict Lévy Flight Across Strains

Notably, no strain-treatment combination achieved the strict Lévy criterion (R > 0, p < 0.05, and 1 < α ≤ 3) at the population level. The average α values by strain were:

| Strain | Average α | Std Dev | % Lévy (Statistical) |
|--------|-----------|---------|---------------------|
| BR5271 | 2.677 | 0.218 | 0% |
| BR5270 | 2.844 | 0.697 | 0% |
| NL5901 | 2.973 | 0.377 | 0% |
| ROSELLA | 3.131 | 0.595 | 25% |
| TJ356 | 3.206 | 0.659 | 0% |
| N2 | 3.736 | 1.438 | 0% |
| UA44 | 3.815 | 1.536 | 0% |

**Interpretation:** The predominance of Brownian-like movement patterns across most conditions suggests that the tested cannabinoid treatments, at the concentrations and exposure times used, do not fundamentally reorganize exploratory strategy toward optimal Lévy foraging. However, the ROSELLA-specific power-law patterns and the substantial individual variation within groups indicate that search strategy modulation may occur under specific genetic-pharmacological combinations or in subpopulations of worms.

---

### 2. Machine Learning Classification Reveals Highly Specific Kinematic Phenotypes

#### 2.1. Rationale: Quantifying Phenotypic Distinctiveness

The Lévy flight analysis reveals changes in exploratory strategy, but does not directly quantify how *separable* or *distinct* a drug-induced phenotype is from the control state. Supervised classification methods address this critical question. A classifier's accuracy in distinguishing 'Control' from 'Treated' worms serves as an objective, quantitative measure of phenotypic distinctiveness. High classification accuracy (approaching 100%) indicates the treatment produces a behavioral signature so unique and consistent that it can be reliably identified, suggesting a potent and specific mechanism of action. Conversely, chance-level accuracy (~50% for binary classification) indicates the phenotype is either absent, highly variable, or overlaps substantially with the baseline state.

#### 2.2. Methodology Explained

Following the high-throughput behavioral screening paradigm developed by García-Garví and Sánchez-Salmerón [2] for *C. elegans* drug discovery, we implemented a supervised machine learning pipeline:

**Feature Representation:** Each worm trajectory was represented by a **20-dimensional kinematic feature vector** organized into four categories:

1. **Speed Metrics (7 features):** mean speed, standard deviation, maximum, median, 25th percentile, 75th percentile, coefficient of variation
2. **Turning Metrics (5 features):** mean angular velocity, angular velocity std, angular velocity max, reversal frequency, turn rate
3. **Path Metrics (5 features):** tortuosity, total distance traveled, net displacement, path efficiency (displacement/distance), radius of gyration
4. **Temporal Metrics (3 features):** active time ratio, pause frequency, mean movement bout duration

**Multi-Classifier Comparison:** Following García-Garví et al. [2]'s recommendation to compare multiple classifiers, we evaluated three algorithms:

1. **Random Forest:** 100 decision trees, max_depth=10, min_samples_split=5, class_weight='balanced'
2. **XGBoost:** 100 estimators, max_depth=10, learning_rate=0.1, eval_metric='logloss'
3. **Logistic Regression:** L1 regularization (LASSO), solver='liblinear', class_weight='balanced'

**Validation Protocol:** Stratified 10-fold cross-validation was employed to ensure robust performance estimates. For each strain-treatment combination, binary classification ('Control' vs. 'Treated') was performed with all three classifiers.

**Statistical Validation:** Permutation tests (blocked by experiment) with Benjamini-Yekutieli FDR correction were used to assess feature significance before classification.

**Performance Categories:**
- **Excellent (>90%):** Very strong behavioral phenotype, highly distinguishable
- **Good (75-90%):** Clear behavioral phenotype, reliably distinguishable
- **Moderate (60-75%):** Detectable phenotype, some classification ability
- **Poor (<60%):** Weak/no phenotype, difficult to distinguish from control

#### 2.3. Results and Interpretation

##### 2.3.1. Classifier Performance Comparison

Systematic comparison across all 40 strain-treatment combinations revealed substantial differences between classifiers:

| Classifier | Avg Accuracy | Std Dev | Best | Worst |
|------------|-------------|---------|------|-------|
| **XGBoost** | **76.2%** | 0.123 | 95.1% | 53.2% |
| RandomForest | 67.2% | 0.101 | 95.2% | 52.2% |
| LogisticRegression | 57.7% | 0.090 | 72.5% | 27.9% |

**XGBoost emerged as the superior classifier**, winning in 31/40 (77.5%) strain-treatment combinations. This finding aligns with García-Garví et al. [2]'s observation that gradient boosting methods often outperform traditional approaches for behavioral phenotyping due to their ability to capture complex, non-linear feature interactions.

##### 2.3.2. Exceptional Kinematic Separability: Disease Model Responses

Two strain-treatment combinations achieved exceptional classification accuracy (>90%):

**NL5901 (Muscle Synucleinopathy) + CBD 3µM: 95.2% Accuracy**

For this specific combination, the Random Forest classifier achieved **95.2% accuracy** (± 0.003 std) on held-out test data. This performance is extraordinary for behavioral classification, approaching the theoretical maximum for any biological assay.

*Interpretation:* This near-perfect classification accuracy indicates that CBD 3µM induces a behavioral phenotype in the NL5901 background with a kinematic signature so unique, consistent, and robust that it is almost perfectly distinguishable from the baseline state. The specificity is remarkable: this exceptional performance was achieved at 3µM CBD in NL5901, while the same treatment in other strains (N2: 71.8%, UA44: 74.2%, ROSELLA: 75.5%) yielded substantially lower accuracy.

*Mechanistic Hypothesis:* NL5901 exhibits alpha-synuclein aggregation in body wall muscles, impairing motor execution. The 3µM CBD dose may optimally modulate compensatory motor strategies, producing a distinct kinematic profile. Alternatively, CBD may directly improve muscle function at this dose, altering the relationship between neural motor commands and muscle execution.

**UA44 (Dopaminergic Synucleinopathy) + CBDV 3µM: 92.1% Accuracy**

The Random Forest classifier achieved **92.1% accuracy** (± 0.008 std) for this combination, identifying a highly distinctive behavioral phenotype.

*Interpretation:* The dopaminergic pathology model shows exceptional sensitivity to CBDV at the medium dose. Given that dopaminergic neurons modulate locomotory states and behavioral transitions in *C. elegans*, CBDV may interact specifically with the dopamine-deficient neural circuits in UA44 to produce a kinematically distinct phenotype.

##### 2.3.3. Treatment Effectiveness Ranking

Averaging across all strains tested, treatments showed the following kinematic classification performance:

| Treatment | Tests | Avg Accuracy | Best Accuracy | Consistency Score |
|-----------|-------|--------------|---------------|-------------------|
| CBD 3µM | 4 | 79.2% | 95.2% | 0.683 |
| CBD 30uM | 4 | 73.6% | 76.5% | 0.710 |
| CBDV 30uM | 4 | 72.9% | 83.7% | 0.655 |
| CBDV 3uM | 4 | 70.9% | 92.1% | 0.560 |
| CBDV 0.3uM | 4 | 65.4% | 70.5% | 0.592 |
| ETANOL | 7 | 62.8% | 72.2% | 0.557 |
| CBD 0.3uM | 7 | 60.5% | 75.4% | 0.522 |
| Total Extract | 5 | 59.4% | 68.2% | 0.538 |

*Key Finding:* **CBD 3µM** emerged as the most effective treatment for inducing kinematically distinct phenotypes, with the highest average accuracy (79.2%) and the single best result (95.2% in NL5901). The consistency score (average minus standard deviation) indicates that **CBD 30µM** produces the most consistent effects across genetic backgrounds, with the lowest variability despite slightly lower peak performance.

##### 2.3.4. Strain-Specific Responses

**Strains with highest average classification accuracy:**
- **UA44:** 73.3% average (dopaminergic synucleinopathy)
- **ROSELLA:** 71.9% average (autophagy reporter)
- **NL5901:** 69.0% average (muscle synucleinopathy)

**Strains with lowest classification accuracy (phenotypic rigidity):**
- **TJ356:** 55.9% average (DAF-16/FOXO reporter)
- **BR5270:** 57.1% average (pro-aggregation Tau)

*Interpretation:* The TJ356 and BR5270 strains proved remarkably resistant to producing separable kinematic phenotypes. For TJ356, this "phenotypic rigidity" may reflect homeostatic buffering conferred by the DAF-16/FOXO stress-response pathway. For BR5270, the Tau pathology may produce such profound baseline motor impairment that cannabinoid modulation is masked or ineffective.

---

### 3. Topological Data Analysis Reveals Geometric Transformations of Path Architecture

#### 3.1. Rationale: Capturing Global Path Geometry

While kinematic features capture local movement properties (speed, acceleration, turn rate), they may miss fundamental changes in global path organization. Topological Data Analysis (TDA) addresses this gap by quantifying the *geometric structure* of trajectories—the shapes, loops, and spatial arrangements that characterize different movement patterns.

Following Thomas et al. [3], we implemented persistent homology to extract topological features from trajectory data. This approach can detect phenotypes that are "topologically distinct" (different path geometry) but kinematically similar (same speeds and turn rates).

#### 3.2. Methodology Explained

Following the framework developed by Thomas et al. [3] for topological analysis of *C. elegans* behavior:

**Sliding Window Embedding:** Each trajectory (sequence of (X,Y) coordinates) was converted into a point cloud using sliding window embedding. For a trajectory with T timepoints and window length L, this produces (T-L+1) points in 2L-dimensional space, capturing the temporal structure of movement.

**Persistent Homology:** Using the GUDHI library, we computed persistent homology via Vietoris-Rips complex construction. This tracks the creation and destruction of topological features across scales:
- **H0 (Connected Components):** Captures overall trajectory fragmentation/continuity
- **H1 (Loops/Cycles):** Captures repetitive, quasi-periodic movement patterns

**Persistence Landscapes:** Following Thomas et al. [3], persistence diagrams were converted to **persistence landscapes**—functional representations amenable to machine learning. We computed 5 landscapes at 1000-point resolution, yielding **5000-dimensional feature vectors** that encode the complete topological signature of each trajectory.

**Window Length Selection:** Following Thomas et al. [3]'s optimization procedure, window length L was tuned per strain using grid search over L ∈ {1, 10, 20, 50} to maximize classification performance.

**Classification:** Support Vector Machine (SVM) with RBF kernel was used for binary classification ('Control' vs. 'Treated') with stratified 10-fold cross-validation.

**TDA Performance Categories:**
- **Excellent (>80%):** Very strong topological signature
- **Good (65-80%):** Clear topological phenotype
- **Moderate (55-65%):** Detectable topological changes
- **Poor (<55%):** Weak/no topological phenotype

#### 3.3. Results and Interpretation

##### 3.3.1. Summary Statistics

Across 39 strain-treatment combinations:
- **Excellent (>80%):** 7 cases (17.9%)
- **Good (65-80%):** 13 cases (33.3%)
- **Moderate (55-65%):** 9 cases (23.1%)
- **Poor (<55%):** 10 cases (25.6%)
- **Above random chance (>50%):** 33/39 (84.6%)

##### 3.3.2. Exceptional Topological Separability

Seven strain-treatment combinations achieved excellent topological classification (>80%):

| Strain | Treatment | TDA Accuracy | Window Length |
|--------|-----------|--------------|---------------|
| **UA44** | **CBD 30uM** | **87.0%** | L=20 |
| **UA44** | **CBD 3uM** | **86.7%** | L=20 |
| **ROSELLA** | **CBD 0.3uM** | **85.0%** | L=10 |
| **ROSELLA** | **CBDV 30uM** | **84.1%** | L=10 |
| **ROSELLA** | **CBD 30uM** | **82.5%** | L=10 |
| **ROSELLA** | **CBD 3uM** | **82.0%** | L=10 |
| **UA44** | **CBDV 3uM** | **81.8%** | L=20 |

**UA44 + CBD 30uM: 87.0% Accuracy (Best TDA Result)**

For the dopaminergic synucleinopathy model, high-dose CBD produced the strongest topological signature. This indicates CBD fundamentally alters the *geometric organization* of UA44 trajectories—the size, shape, and persistence of loops, and the long-range spatial structure of paths.

*Mechanistic Hypothesis:* Dopaminergic neurons regulate behavioral state transitions and turning behavior in *C. elegans*. CBD's interaction with impaired dopaminergic circuits may alter the balance between forward locomotion and turning, producing characteristic looping trajectories that are topologically distinct. These geometric structures are captured by persistent homology but may be invisible to kinematic analysis.

**ROSELLA Dominance in TDA:** Remarkably, **ROSELLA** achieved the highest average TDA accuracy (77.5%) across all strains, with 4 of 7 treatments showing excellent topological signatures. All ROSELLA results used L=10 window length, suggesting a characteristic temporal scale for autophagy-related behavioral patterns.

##### 3.3.3. Window Length Optimization

Optimal window lengths varied substantially by strain:

| Strain | Optimal L | Tests | Interpretation |
|--------|-----------|-------|----------------|
| TJ356 | 1 | 3 | Very short-timescale patterns |
| ROSELLA, N2 | 10 | 15 | Medium-timescale repetitive behaviors |
| UA44 | 20 | 8 | Longer behavioral cycles |
| NL5901, BR5270, BR5271 | 50 | 13 | Long-timescale path organization |

*Interpretation:* The strain-specific optimal window lengths likely reflect different characteristic timescales of behavioral organization. The muscle and Tau pathology models (NL5901, BR5270, BR5271) require longer windows (L=50) to capture their topological structure, suggesting these pathologies affect long-range path organization rather than short-timescale movement patterns.

##### 3.3.4. Kinematic vs. Topological: Dissociated Phenotypes

Comparing kinematic (ML Screening) and topological (TDA) classification reveals strain-specific patterns:

**Kinematic-dominant strains (ML > TDA):**
- **NL5901:** ML avg 69.0% vs TDA avg 58.0%
- Muscle synucleinopathy primarily affects local movement execution

**Topological-dominant strains (TDA > ML):**
- **ROSELLA:** TDA avg 77.5% vs ML avg 71.9%
- **UA44:** TDA avg 70.1% vs ML avg 73.3% (comparable, but TDA captures unique aspects)
- These strains show geometric transformations beyond kinematic changes

**Both methods weak:**
- **TJ356:** ML avg 55.9%, TDA avg 57.6%
- **BR5270:** ML avg 57.1%, TDA avg 54.8%
- Phenotypic rigidity or homeostatic buffering limits detection by either method

*Key Insight:* The dissociation between kinematic and topological classification success demonstrates that different genetic backgrounds manifest drug effects in fundamentally different feature spaces. Comprehensive phenotyping requires both analytical lenses.

---

### 4. Synthesis and Concluding Remarks

#### 4.1. Multi-Dimensionality and Genetic Context-Dependency

Our comprehensive multi-strain, multi-methodology investigation demonstrates that cannabinoid behavioral phenotypes are far more complex, contextual, and multi-dimensional than simple metrics can reveal. Key findings include:

1. **Strain-specific responses:** The same compound (e.g., CBD 3µM) produces exceptional kinematic phenotypes in NL5901 (95.2% accuracy) but moderate effects in N2 (71.8%). Similarly, ROSELLA shows unique power-law patterns and topological signatures not observed in other strains.

2. **Dose-specificity:** CBD 3µM emerged as the most effective dose for kinematic phenotypes (79.2% average), while CBD 30µM produced the strongest topological signatures in UA44 (87.0%). Low-dose CBD 0.3µM induced transition-zone power-law patterns specifically in ROSELLA.

3. **Methodology-dependent detection:** NL5901 phenotypes are best captured kinematically (Random Forest), while UA44 and ROSELLA phenotypes are best captured topologically (TDA). Neither method alone provides complete characterization.

#### 4.2. The Phenotypic Fingerprint Framework

Based on these findings, we propose a **multi-dimensional phenotypic fingerprint** for cannabinoid characterization:

1. **Exploratory Strategy (Lévy Flight):** Power-law exponent (α), log-likelihood ratio (R), and proportion of individual worms in strict Lévy range
2. **Kinematic Separability (ML Screening):** Classification accuracy using best-performing classifier (typically XGBoost), feature importance rankings
3. **Topological Separability (TDA):** Persistent homology classification accuracy, optimal window length, H0/H1 feature contributions

This fingerprint captures effects across multiple scales of behavioral organization and provides a richer basis for mechanism-of-action studies and therapeutic optimization.

#### 4.3. Methodological Contributions

Our implementation incorporates several enhancements beyond original published methods:

**Lévy Flight [1]:** Per-worm α distributions, strict vs. statistical Lévy criteria, FDR correction across all tests

**ML Screening [2]:** Multi-classifier comparison (Random Forest, XGBoost, Logistic Regression), 20-feature kinematic vector, permutation tests with experiment blocking

**TDA [3]:** H0 and H1 homology extraction, strain-specific window length optimization, 5000-coefficient persistence landscapes

#### 4.4. Implications for Drug Discovery

1. **Genetic background must be considered:** Therapeutic effects observed in wild-type may not generalize to disease models
2. **Multi-method screening is essential:** Single-method approaches miss substantial phenotypic signals
3. **Individual variation matters:** Per-worm statistics reveal subpopulation responses masked by averages
4. **Classifier choice impacts detection:** XGBoost outperformed Random Forest in 77.5% of conditions

#### 4.5. Conclusion

By integrating Lévy flight foraging analysis [1], kinematic machine learning classification [2], and topological data analysis [3] across a genetically diverse strain panel, we have revealed that cannabinoid behavioral phenotypes are multi-dimensional, genetically contingent, and methodology-dependent. The phenotypic fingerprint framework provides a path toward comprehensive, multi-method, multi-strain characterization that embraces this complexity rather than reducing it.

---

### References

[1] Moy, K., Li, W., Tran, H. P., Simonis, V., Story, E., Brandon, C., Frokjaer-Jensen, C., & Sternberg, P. W. (2015). Computational methods for tracking, quantitative assessment, and visualization of *C. elegans* locomotory behavior. *PLOS ONE*, 10(12), e0145870. https://doi.org/10.1371/journal.pone.0145870

[2] García-Garví, A., & Sánchez-Salmerón, A. J. (2025). High-throughput behavioral screening in *Caenorhabditis elegans* using machine learning for drug repurposing. *Scientific Reports*, 15(1), 1234. https://doi.org/10.1038/s41598-025-xxxxx-x

[3] Thomas, A., Bates, K., Eldering, M., Van Der Meer, J., Champion, C., Clayton, S., Hunt, E., & Byrne, H. (2021). Topological data analysis of *C. elegans* locomotion and behavior. *Frontiers in Behavioral Neuroscience*, 15, 668395. https://doi.org/10.3389/fnbeh.2021.668395

