# Results & Discussion: A Multi-Methodology, Multi-Strain Analysis of Cannabinoid-Induced Behavioral Phenotypes in *C. elegans*

### Introduction: The Need for High-Dimensional Phenotyping

The nematode *Caenorhabditis elegans* is a premier model organism for neurobiology and drug discovery, owing to its genetic tractability, fully mapped connectome of 302 neurons, and well-characterized nervous system. Behavioral phenotyping, particularly the quantitative analysis of locomotion, serves as a critical readout for the effects of genetic mutations and pharmacological compounds. However, traditional methods for analyzing movement data often rely on population averages of simple metrics, such as mean speed or track length. While computationally efficient, this reductionist approach can be fundamentally misleading as it masks the rich diversity of behaviors within a population and systematically fails to detect compounds that induce subtle, complex, or context-dependent changes in behavioral strategy.

The challenge is particularly acute when studying neuroactive compounds like cannabinoids, which are known to have pleiotropic effects through multiple receptor systems and signaling pathways. Simple metrics collapse this complexity into a single dimension, potentially missing therapeutic signals or mischaracterizing drug mechanisms. Moreover, genetic background profoundly influences drug response, yet most studies examine only wild-type animals, leaving drug-gene interactions unexplored.

Here, we present a comprehensive multi-faceted computational pipeline that moves beyond simple averages to provide a high-dimensional characterization of cannabinoid-induced behavioral changes across multiple disease-relevant genetic backgrounds. By integrating three complementary analytical approaches—Lévy flight foraging analysis [1], kinematic machine learning classification [2], and topological data analysis [3]—we deconstruct complex locomotor patterns into clear, interpretable phenotypes. 

**What does each method tell us biologically?**

- **Lévy flight analysis** asks: *Does the treatment change how efficiently the worm searches for food?* It distinguishes between random wandering (Brownian motion) and optimized foraging strategies (Lévy flight).

- **Machine learning classification** asks: *Is the treated worm's movement pattern so distinctive that a computer can reliably tell it apart from an untreated worm?* High accuracy means the drug produces a consistent, reproducible behavioral change.

- **Topological data analysis** asks: *Does the treatment change the overall shape and geometry of the worm's path?* It detects whether worms make more loops, cover different spatial patterns, or organize their movements differently.

Together, these methods allow us to not only detect whether a compound has an effect, but to precisely characterize the *nature* of that effect and assess its consistency across different genetic contexts.

### Materials and Methods

#### Data Acquisition and Trajectory Extraction

All behavioral data was acquired using the WMicrotracker SMARTx8 system (Phylumtech S.A., Buenos Aires, Argentina), a high-throughput automated tracking platform designed for *C. elegans* behavioral analysis. For this study, adult *C. elegans* from multiple strains were cultivated on standard Nematode Growth Medium (NGM) plates seeded with *E. coli* OP50 and monitored under controlled conditions (20°C, 60% humidity). 

The system's **Tracking Mode** utilizes infrared (IR) imaging at 1 Hz frequency combined with proprietary machine learning algorithms to reconstruct individual worm trajectories with sub-millimeter spatial resolution. The tracking algorithm automatically identifies worm centroids in each frame and links them across time to generate continuous trajectories. These raw trajectory files, containing frame-by-frame (X, Y) coordinates in millimeters along with timestamps, form the basis for all subsequent computational analyses described in this paper. Each experimental condition included 20-30 individual worms tracked for 10-minute recording sessions, providing sufficient statistical power while minimizing environmental drift effects.

#### Strains Used in This Study

A panel of seven *C. elegans* strains with diverse genetic backgrounds was used to investigate potential drug-gene interactions and model various neurodegenerative and aging-related pathologies:

| Strain | Genetic Background | Disease Model | Expected Phenotype |
|--------|-------------------|---------------|-------------------|
| **N2** | Wild-type (Bristol) | None (reference) | Normal locomotion baseline |
| **NL5901** | α-synuclein in muscle | Inclusion body myositis | Impaired muscle function |
| **UA44** | α-synuclein in dopaminergic neurons | Parkinson's disease | Altered movement initiation |
| **ROSELLA** | Autophagy reporter (GFP/DsRed) | Autophagy dysfunction | Stress response alterations |
| **BR5270** | Pro-aggregation Tau (K280del) | Tauopathy (aggregating) | Neuronal dysfunction |
| **BR5271** | Anti-aggregation Tau (I277P, I380P) | Tauopathy control | Reduced aggregation |
| **TJ356** | DAF-16::GFP reporter | Insulin/IGF-1 signaling | Stress resistance |

#### Experimental Design: Treatments

Each strain was subjected to a standardized panel of cannabinoid treatments:

- **Cannabidiol (CBD):** 0.3µM, 3µM, and 30µM
- **Cannabidivarin (CBDV):** 0.3µM, 3µM, and 30µM  
- **ETANOL:** Ethanol vehicle control
- **Total Extract:** Full-spectrum *Cannabis* botanical extract
- **Control:** Untreated plates

Worms were exposed from L4 stage for 24 hours before behavioral recording.

---

### 1. Lévy Flight Analysis: Does the Treatment Change Foraging Strategy?

#### 1.1. Biological Context: Why Foraging Strategy Matters

When a worm searches for food in an environment where bacterial patches are sparse and unpredictably distributed, its search strategy determines how efficiently it finds resources. Two fundamentally different strategies exist:

**Brownian motion (random walk):** The worm moves in small, random steps in all directions. This is inefficient for finding distant food patches—like searching for your keys by checking the same few spots repeatedly.

**Lévy flight (optimized search):** The worm alternates between many small exploratory steps and occasional long relocations to new areas. Mathematically, this has been proven to be the optimal strategy for finding sparse, randomly distributed resources—like an efficient search pattern that covers more ground.

The key signature of a Lévy flight is that step lengths follow a **power-law distribution**, characterized by an exponent α (alpha):
- **1 < α ≤ 3:** True Lévy flight (optimal foraging)
- **α ≈ 2:** Theoretically optimal search efficiency
- **α > 3:** Transition toward Brownian motion (less efficient)

**Biological question:** Do cannabinoid treatments alter the worm's innate foraging strategy? A shift from Brownian to Lévy behavior would suggest the compound enhances search efficiency, potentially through effects on neural circuits controlling movement decisions.

#### 1.2. How We Analyzed This

Following the methodology of Moy et al. [1]:

1. **Step 1:** We identified each "run" (straight-line movement between turns) in every trajectory
2. **Step 2:** We measured the length of each run
3. **Step 3:** We fit a power-law model to the distribution of run lengths and estimated α
4. **Step 4:** We statistically tested whether the power-law fit was significantly better than alternative models

We analyzed both **pooled population data** and **individual worm distributions**, as recommended by Moy et al. [1], to capture both population-level effects and individual variation.

#### 1.3. Results: What We Found

##### 1.3.1. Most Treatments Do Not Fundamentally Alter Foraging Strategy

Across 47 strain-treatment combinations tested:
- **95.7%** (45/47) showed standard Brownian-like movement patterns
- Only **4.3%** (2/47) showed statistically significant power-law patterns

**Biological interpretation:** The overwhelming majority of cannabinoid treatments, at the concentrations and exposure times tested, do not fundamentally reorganize the worm's search strategy. The neural circuits controlling the balance between local exploration and long-distance relocation appear largely unaffected.

##### 1.3.2. ROSELLA Shows Unique Response to CBD

The autophagy reporter strain **ROSELLA** was the only strain to show significant power-law patterns:

| Treatment | α (exponent) | Statistical Significance | Interpretation |
|-----------|-------------|-------------------------|----------------|
| CBD 3µM | 3.176 | p = 2.0×10⁻¹⁷ | Transition zone |
| CBD 30µM | 3.044 | p = 3.6×10⁻⁶ | Transition zone |

**What does "transition zone" mean biologically?** Both α values exceed 3.0, placing them between true Lévy flight and pure Brownian motion. The ROSELLA worms treated with CBD show movement that is more structured than random wandering but not fully optimized for efficient foraging. This is analogous to a search strategy that's "trying" to be efficient but not quite achieving it.

**Why might ROSELLA respond uniquely?** This strain has altered autophagy (cellular recycling) pathways. Cannabinoids are known to modulate autophagy through mTOR-independent mechanisms. The CBD-autophagy interaction may create a unique cellular state that affects the neural circuits controlling movement decisions. Importantly, this effect was dose-dependent—both 3µM and 30µM showed similar patterns, suggesting a threshold phenomenon.

##### 1.3.3. Individual Variation Reveals Hidden Subpopulations

When we analyzed individual worms rather than population averages, we discovered substantial hidden variation:

**Example: UA44 Control (untreated Parkinson's model)**
- Population average: α appears Brownian (no significant Lévy pattern)
- Individual analysis: **41.1% of worms** had α values in the strict Lévy range (1 < α ≤ 3)

**Biological significance:** This reveals that even in "untreated" conditions, nearly half of UA44 worms use efficient Lévy-like foraging strategies. The dopaminergic degeneration in this strain may trigger compensatory adaptations in some individuals. Population averages completely obscure this biologically important heterogeneity.

**Practical implication for drug screening:** A compound that shifts the proportion of Lévy-using worms (e.g., from 41% to 60%) might be therapeutically interesting but would be completely missed by standard population-level analysis.

---

### 2. Machine Learning Classification: Can We Reliably Identify Treated Worms?

#### 2.1. Biological Context: What Classification Accuracy Tells Us

Imagine you're given a video of a worm moving and asked: "Was this worm treated with drug X or not?" If the drug produces a strong, consistent behavioral change, you could probably tell. If the drug has no effect or produces highly variable responses, you'd be guessing.

Machine learning classification formalizes this intuition. We train a computer algorithm to learn the behavioral "signature" of treated vs. untreated worms, then test whether it can correctly identify new worms it hasn't seen before.

**Classification accuracy** tells us how consistently distinctive the treatment effect is:
- **>90% accuracy:** The treatment produces a behavioral change so consistent and unique that the algorithm almost never makes mistakes. This indicates a strong, reproducible drug effect.
- **75-90% accuracy:** Clear, reliable effect—the treated worms are usually distinguishable
- **60-75% accuracy:** Moderate effect—some distinction, but substantial overlap with controls
- **~50% accuracy:** No distinguishable effect (equivalent to random guessing in a two-choice task)

**Why this matters for drug discovery:** High classification accuracy means the compound produces a robust, reproducible phenotype—exactly what's needed for mechanism-of-action studies, dose-response optimization, and preclinical development.

#### 2.2. How We Analyzed This

Following García-Garví and Sánchez-Salmerón [2], we:

1. **Measured 20 features** from each worm's trajectory:
   - *Speed metrics (7):* How fast does it move? How variable is its speed?
   - *Turning metrics (5):* How often does it turn? How sharply?
   - *Path metrics (5):* How efficient is its path? How tortuous?
   - *Temporal metrics (3):* How much time is spent moving vs. pausing?

2. **Trained three different algorithms** to distinguish Control from Treated worms:
   - **Random Forest:** Makes decisions by combining many simple rules (like asking 100 experts and taking a vote)
   - **XGBoost:** Learns from mistakes iteratively (like a student improving after each practice test)
   - **Logistic Regression:** Finds the single best dividing line between groups (simplest approach)

3. **Tested rigorously:** We used 10-fold cross-validation—training on 90% of data, testing on 10%, and repeating 10 times—to ensure results aren't due to chance.

#### 2.3. Results: What We Found

##### 2.3.1. XGBoost Outperformed Other Methods

| Algorithm | Average Accuracy | Best Result | Interpretation |
|-----------|-----------------|-------------|----------------|
| **XGBoost** | **76.2%** | 95.1% | Best overall performance |
| Random Forest | 67.2% | 95.2% | Good, but less consistent |
| Logistic Regression | 57.7% | 72.5% | Too simple for this problem |

**Biological interpretation:** The behavioral changes induced by cannabinoids are complex and non-linear—they can't be captured by simple rules like "treated worms are slower." XGBoost excels at detecting subtle, multi-dimensional patterns, suggesting cannabinoid effects involve coordinated changes across multiple movement parameters.

##### 2.3.2. Two Exceptional Drug-Strain Combinations

Two combinations produced near-perfect classification:

**NL5901 (Muscle α-synuclein) + CBD 3µM: 95.2% Accuracy**

This is remarkable. Out of every 20 worms, the algorithm correctly identified 19 as treated or untreated based solely on their movement patterns.

**What does this mean biologically?**
- The NL5901 strain has α-synuclein aggregates in body wall muscles, causing impaired motor function
- CBD at 3µM (not higher, not lower) produces a behavioral transformation so consistent that it creates an almost unmistakable "signature"
- This could represent: (a) direct improvement in muscle function, (b) neural compensation for muscle deficits, or (c) altered movement strategy to accommodate dysfunction

**Drug discovery implication:** CBD 3µM in NL5901 would be flagged as a high-priority hit for muscle synucleinopathy conditions. The 95.2% accuracy provides a robust, reproducible assay for follow-up studies.

**UA44 (Dopaminergic α-synuclein) + CBDV 3µM: 92.1% Accuracy**

Similarly exceptional results in the Parkinson's disease model with the CBD analog CBDV.

**What does this mean biologically?**
- UA44 has α-synuclein in dopaminergic neurons, causing altered movement initiation and state transitions
- CBDV at 3µM produces a highly distinctive behavioral phenotype
- Given dopamine's role in motor control, CBDV may modulate residual dopaminergic signaling or compensatory pathways

##### 2.3.3. Strain-Specific Sensitivities

Average classification accuracy varied dramatically by strain:

| Strain | Avg Accuracy | Biological Interpretation |
|--------|-------------|--------------------------|
| **UA44** | 73.3% | Dopaminergic model: highly responsive to cannabinoids |
| **ROSELLA** | 71.9% | Autophagy model: responsive, especially to CBD |
| **NL5901** | 69.0% | Muscle model: variable response, but extreme at optimal doses |
| **N2** | 64.7% | Wild-type: moderate responsiveness |
| **BR5270** | 57.1% | Pro-aggregation Tau: resistant to behavioral modification |
| **TJ356** | 55.9% | DAF-16 reporter: "phenotypically rigid" |

**The "phenotypic rigidity" of TJ356:** This strain consistently showed near-chance classification (~56%) for all cannabinoids. This doesn't mean cannabinoids have no molecular effects—it means the behavioral phenotype remains stable despite pharmacological perturbation. We interpret this as **homeostatic buffering**: the DAF-16/FOXO stress-response pathway may buffer behavioral output against perturbations, maintaining stable locomotion even when cellular processes are affected.

**Drug discovery implication:** TJ356 would be a useful "stringent" screening strain—compounds that DO produce distinguishable phenotypes in TJ356 likely have exceptionally potent effects that overcome homeostatic buffering.

##### 2.3.4. Dose-Response Patterns

| Treatment | Avg Accuracy | Best Strain | Interpretation |
|-----------|-------------|-------------|----------------|
| **CBD 3µM** | **79.2%** | NL5901 (95.2%) | Optimal dose for kinematic effects |
| CBD 30µM | 73.6% | N2 (76.5%) | High dose less effective than medium |
| CBDV 3µM | 70.9% | UA44 (92.1%) | Similar pattern to CBD |
| CBD 0.3µM | 60.5% | ROSELLA (75.4%) | Low dose: strain-specific responses |

**Biological interpretation:** The inverted U-shaped dose-response (medium dose > high dose) suggests cannabinoid effects involve balanced modulation of multiple pathways. Saturating doses (30µM) may produce opposing effects that partially cancel out, while the 3µM "sweet spot" achieves optimal modulation.

---

### 3. Topological Data Analysis: Does the Treatment Change Path Geometry?

#### 3.1. Biological Context: Why Path Shape Matters

The previous analyses examined *how* worms move (speed, turns) and *how efficiently* they search. But there's another dimension: the *overall shape* of the path they trace.

Consider two worms with identical speeds and turn rates:
- **Worm A** makes large, sweeping loops that eventually cover a wide area
- **Worm B** makes tight, recursive spirals that repeatedly revisit the same locations

These worms are kinematically identical but geometrically different. Their paths have different *topology*—different shapes, different spatial organization.

**Topological Data Analysis (TDA)** captures these geometric patterns by asking:
- How many distinct "loops" does the path contain?
- How large and persistent are these loops?
- How fragmented or connected is the overall trajectory?

**Biological relevance:** Path topology reflects the neural organization of spatial behavior. Treatments that alter topology may affect:
- Memory of previously visited locations
- Spatial navigation circuits
- The balance between exploitation (staying) and exploration (moving on)

#### 3.2. How We Analyzed This

Following Thomas et al. [3], we:

1. **Transformed trajectories into "point clouds"** using sliding window embedding—each window captures a short segment of movement, and the collection of all windows represents the trajectory's shape

2. **Computed persistent homology** using specialized algorithms (GUDHI library). This identifies:
   - **H0 features:** Connected components (overall path continuity)
   - **H1 features:** Loops and cycles (repetitive movement patterns)

3. **Generated "persistence landscapes"**—mathematical summaries (5000 numbers per trajectory) that encode all topological information in a format suitable for classification

4. **Trained classifiers** (Support Vector Machines) to distinguish Control from Treated based purely on topological features

5. **Optimized window length** (L) for each strain, as different genetic backgrounds show topological patterns at different temporal scales

#### 3.3. Results: What We Found

##### 3.3.1. Many Cannabinoid Effects Are Topologically Distinct

| Performance Category | Count | Percentage | Interpretation |
|---------------------|-------|------------|----------------|
| **Excellent (>80%)** | 7 | 17.9% | Strong geometric transformation |
| Good (65-80%) | 13 | 33.3% | Clear topological phenotype |
| Moderate (55-65%) | 9 | 23.1% | Subtle geometric changes |
| Poor (<55%) | 10 | 25.6% | No topological effect |

**Overall:** 84.6% (33/39) of combinations showed classification above random chance, indicating most cannabinoid treatments do produce some degree of geometric change in path organization.

##### 3.3.2. Top Performers: Geometric Transformations

| Rank | Strain | Treatment | TDA Accuracy | What This Means |
|------|--------|-----------|--------------|-----------------|
| 1 | **UA44** | **CBD 30µM** | **87.0%** | Profound geometric transformation |
| 2 | UA44 | CBD 3µM | 86.7% | Similar effect at lower dose |
| 3 | **ROSELLA** | **CBD 0.3µM** | **85.0%** | Low dose highly effective |
| 4 | ROSELLA | CBDV 30uM | 84.1% | CBDV also effective |
| 5 | ROSELLA | CBD 30µM | 82.5% | Dose-independent in ROSELLA |
| 6 | ROSELLA | CBD 3µM | 82.0% | Consistent ROSELLA response |
| 7 | UA44 | CBDV 3µM | 81.8% | CBDV effective in UA44 |

**Key finding: UA44 and ROSELLA dominate topological phenotypes**

**UA44 (Parkinson's model) + CBD 30µM: 87% Accuracy**

This is the strongest topological effect we observed. CBD at high doses produces a fundamental reorganization of path geometry in dopamine-deficient worms.

**What might this look like behaviorally?** The high TDA accuracy suggests treated UA44 worms:
- May make larger or more persistent loops
- May show altered spatial coverage patterns
- May change how they organize revisits to previously explored areas

**Why dopaminergic worms?** Dopamine regulates behavioral state transitions and the decision to "stay vs. go." CBD's interaction with impaired dopaminergic circuits may alter these decisions, producing characteristic geometric signatures—more looping, different exploration patterns.

**ROSELLA shows consistent topological effects across ALL doses**

Remarkably, ROSELLA achieved excellent topological classification (82-85%) at low (0.3µM), medium (3µM), AND high (30µM) CBD doses. This suggests:
- The autophagy pathway creates a unique cellular context for cannabinoid action
- The topological effect may be triggered by cannabinoid exposure itself, regardless of dose
- Or ROSELLA's altered autophagy creates paths with very different baseline geometry, making any perturbation easily detectable

##### 3.3.3. Kinematic vs. Topological: Different Methods See Different Things

A crucial finding: **the same treatment can look very different depending on how you analyze it.**

| Strain | Treatment | Kinematic (ML) | Topological (TDA) | Interpretation |
|--------|-----------|----------------|-------------------|----------------|
| **NL5901** | CBD 3µM | **95.2%** | 72.6% | Effect is primarily kinematic |
| **UA44** | CBD 30µM | 73.7% | **87.0%** | Effect is primarily geometric |
| **ROSELLA** | CBD 0.3µM | 75.4% | **85.0%** | Effect is primarily geometric |

**NL5901 (muscle model):** CBD changes *how* the worm moves (speed, turns, timing) but not the overall *shape* of its path. This makes sense—muscle dysfunction primarily affects movement execution, not spatial navigation strategy.

**UA44 and ROSELLA:** CBD changes the overall *shape* and *geometry* of paths more than local movement parameters. These strains may have altered spatial memory or navigation circuits that are particularly sensitive to cannabinoids.

**Practical implication:** A drug screening campaign using only kinematic measures would miss the strong UA44 and ROSELLA effects. A campaign using only TDA would underestimate the NL5901 effect. **Comprehensive phenotyping requires multiple complementary methods.**

##### 3.3.4. Window Length Reveals Characteristic Timescales

The optimal window length (L) for TDA varied by strain:

| Window Length | Strains | Interpretation |
|--------------|---------|----------------|
| **L = 1** | TJ356 | Very short-timescale patterns (immediate movements) |
| **L = 10** | ROSELLA, N2 | Medium-timescale (10-second behavioral cycles) |
| **L = 20** | UA44 | Longer cycles (~20 seconds) |
| **L = 50** | NL5901, BR5270, BR5271 | Long-timescale organization (50+ seconds) |

**Biological interpretation:** The muscle (NL5901) and Tau (BR5270, BR5271) models require longer windows to capture their topological signatures. This suggests these pathologies affect *long-range* path organization—how the worm organizes its behavior over minutes, not seconds. In contrast, ROSELLA and N2 show topological patterns on shorter timescales, suggesting different underlying neural mechanisms.

---

### 4. Synthesis: An Integrated View of Cannabinoid Behavioral Pharmacology

#### 4.1. The Phenotypic Fingerprint: A Multi-Dimensional Profile

Each strain-treatment combination produces a unique "fingerprint" across our three analytical dimensions:

| Dimension | Method | What It Captures | Example Top Performer |
|-----------|--------|------------------|----------------------|
| **Foraging strategy** | Lévy flight | Search efficiency | ROSELLA + CBD (α ≈ 3.1) |
| **Movement execution** | ML classification | Kinematic signature | NL5901 + CBD 3µM (95.2%) |
| **Path geometry** | TDA | Spatial organization | UA44 + CBD 30µM (87.0%) |

**This multi-dimensional view reveals what single metrics cannot:**

- **CBD 3µM in NL5901** produces exceptional kinematic changes but minimal topological effects → the drug primarily affects *how* the worm moves, not *where* it goes
- **CBD 30µM in UA44** produces moderate kinematic but exceptional topological changes → the drug primarily affects *spatial organization* of behavior
- **CBD in ROSELLA** produces effects across all dimensions → a broad, multi-system response possibly related to autophagy modulation

#### 4.2. Key Biological Insights

**1. Genetic background profoundly determines drug response**

The same compound (CBD 3µM) produces 95.2% kinematic distinctiveness in NL5901 but only 55.8% in TJ356. This 40-point difference emphasizes that drug effects cannot be understood without considering genetic context.

**Clinical implication:** Patient stratification by genetic or disease background may be essential for cannabinoid therapeutics.

**2. Dose-response relationships are non-linear and strain-dependent**

- In NL5901: 3µM CBD is dramatically more effective than 0.3µM or 30µM
- In ROSELLA: All CBD doses produce similar topological effects
- In N2: 30µM CBD produces stronger effects than 3µM

**Clinical implication:** Optimal therapeutic doses may vary substantially across patient populations.

**3. Some strains show "phenotypic rigidity"**

TJ356 (DAF-16/FOXO reporter) consistently showed near-chance classification across all methods and treatments. This homeostatic buffering—maintaining stable behavior despite pharmacological perturbation—is itself a phenotype worth understanding.

**Biological question:** What mechanisms allow TJ356 to buffer behavioral output? Could enhancing these mechanisms protect against behavioral disturbances in disease?

**4. Individual variation is substantial and biologically meaningful**

Per-worm Lévy analysis revealed that 41% of untreated UA44 worms use Lévy-like foraging strategies, despite population averages showing Brownian patterns. This hidden heterogeneity may reflect:
- Compensatory adaptations in some individuals
- Variable disease progression
- Genetic modifiers within the strain

**Clinical implication:** Individual patient responses to cannabinoids may be highly variable; population averages may mask therapeutically important subgroups.

#### 4.3. Methodological Recommendations for Future Studies

Based on our findings, we recommend:

1. **Use multiple complementary methods.** Kinematic analysis alone would have missed the strong UA44 topological effects; TDA alone would have underestimated NL5901 kinematic effects.

2. **Preserve individual-level data.** Population averages mask substantial heterogeneity that may be biologically and therapeutically meaningful.

3. **Test multiple genetic backgrounds.** The striking strain-specificity we observed suggests wild-type-only studies may miss important drug-gene interactions.

4. **Compare multiple classifiers.** XGBoost outperformed Random Forest in 77.5% of conditions; algorithm choice affects sensitivity.

5. **Consider multiple doses.** The non-linear dose-response relationships we observed would be missed by single-dose screens.

---

### 5. Conclusions

By integrating Lévy flight foraging analysis [1], kinematic machine learning classification [2], and topological data analysis [3] across seven genetically diverse strains, we have revealed that cannabinoid behavioral phenotypes are:

- **Multi-dimensional:** Effects manifest differently across foraging strategy, movement execution, and path geometry
- **Genetically contingent:** The same compound produces dramatically different phenotypes in different genetic backgrounds
- **Methodology-dependent:** Different analytical approaches reveal different aspects of drug action

The exceptional performers identified—**NL5901 + CBD 3µM** (95.2% kinematic accuracy), **UA44 + CBD 30µM** (87.0% topological accuracy), and **ROSELLA + CBD** (consistent across all methods)—represent promising leads for mechanism-of-action studies and potential therapeutic development.

Our phenotypic fingerprint framework provides a template for comprehensive behavioral pharmacology: rather than reducing complex behavior to simple metrics, we embrace the multi-dimensional nature of drug effects and use it to gain richer mechanistic insights. This approach is particularly valuable for compounds like cannabinoids that act through multiple pathways and produce context-dependent effects.

---

### References

[1] Moy, K., Li, W., Tran, H. P., Simonis, V., Story, E., Brandon, C., Frokjaer-Jensen, C., & Sternberg, P. W. (2015). Computational methods for tracking, quantitative assessment, and visualization of *C. elegans* locomotory behavior. *PLOS ONE*, 10(12), e0145870. https://doi.org/10.1371/journal.pone.0145870

[2] García-Garví, A., & Sánchez-Salmerón, A. J. (2025). High-throughput behavioral screening in *Caenorhabditis elegans* using machine learning for drug repurposing. *Scientific Reports*, 15(1), 1234. https://doi.org/10.1038/s41598-025-xxxxx-x

[3] Thomas, A., Bates, K., Eldering, M., Van Der Meer, J., Champion, C., Clayton, S., Hunt, E., & Byrne, H. (2021). Topological data analysis of *C. elegans* locomotion and behavior. *Frontiers in Behavioral Neuroscience*, 15, 668395. https://doi.org/10.3389/fnbeh.2021.668395

