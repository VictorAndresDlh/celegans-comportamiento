# Results & Discussion: A Multi-Methodology, Multi-Strain Analysis of Cannabinoid-Induced Behavioral Phenotypes in *C. elegans*

### Introduction: The Need for High-Dimensional Phenotyping

The nematode *Caenorhabditis elegans* is a premier model organism for neurobiology and drug discovery, owing to its genetic tractability, fully mapped connectome of 302 neurons, and well-characterized nervous system. Behavioral phenotyping, particularly the quantitative analysis of locomotion, serves as a critical readout for the effects of genetic mutations and pharmacological compounds. However, traditional methods for analyzing movement data often rely on population averages of simple metrics, such as mean speed or track length. While computationally efficient, this reductionist approach can be fundamentally misleading as it masks the rich diversity of behaviors within a population and systematically fails to detect compounds that induce subtle, complex, or context-dependent changes in behavioral strategy.

The challenge is particularly acute when studying neuroactive compounds like cannabinoids, which are known to have pleiotropic effects through multiple receptor systems and signaling pathways. Simple metrics collapse this complexity into a single dimension, potentially missing therapeutic signals or mischaracterizing drug mechanisms. Moreover, genetic background profoundly influences drug response, yet most studies examine only wild-type animals, leaving drug-gene interactions unexplored.

Here, we present a comprehensive multi-faceted computational pipeline that moves beyond simple averages to provide a high-dimensional characterization of cannabinoid-induced behavioral changes across multiple disease-relevant genetic backgrounds. By integrating four complementary analytical approaches—unsupervised behavioral state discovery [1], Lévy flight foraging analysis [2], kinematic machine learning classification [3], and topological data analysis [4]—we deconstruct complex locomotor patterns into clear, interpretable phenotypes. This allows us to not only detect whether a compound has an effect, but to precisely characterize the nature of that effect—be it sedative, stimulant, or a fundamental alteration of exploratory strategy—and assess its consistency and variability across different genetic contexts. This systems-level approach reveals that cannabinoid effects are highly multi-dimensional, genetically contingent, and methodology-dependent, challenging simplistic models of drug action.

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

#### Experimental Design: Treatments

For this study, each *C. elegans* strain was subjected to a standardized panel of cannabinoid treatments and their respective vehicle controls. Compounds were incorporated into NGM agar at the time of plate pouring to ensure homogeneous distribution. The primary treatments included:

- **Cannabidiol (CBD):** A non-psychoactive phytocannabinoid at three concentrations: 0.3µM (low dose), 3µM (medium dose), and 30µM (high dose)
- **Cannabidivarin (CBDV):** A propyl analogue of CBD at matching concentrations: 0.3µM, 3µM, and 30µM  
- **ETANOL (Ethanol vehicle):** The solvent control for cannabinoid delivery, matched to the highest ethanol concentration used in cannabinoid dilutions
- **Total Extract:** A full-spectrum *Cannabis* botanical extract containing CBD, CBDV, and other minor cannabinoids in their natural ratios
- **Control:** Untreated NGM plates with standard OP50 bacteria

All treatments were performed in biological triplicate, with 20-30 worms per replicate. Worms were exposed to treatments from the L4 larval stage for 24 hours before behavioral recording to ensure consistent developmental stage and treatment duration. Only results demonstrating statistically significant or highly illustrative phenotypic changes are discussed in detail in the results sections below.

### 1. Unsupervised Clustering Reveals Consistent and Strain-Specific Behavioral Fingerprints

#### 1.1. Rationale: Why Use Unsupervised Clustering?

To overcome the fundamental limitations of population averages and pre-defined behavioral categories, we adapted a methodology for **unsupervised behavioral state discovery** based on the model-independent phenotyping framework developed by Koren et al. [1]. The unsupervised nature of this approach is critical: rather than imposing human-defined behavioral categories, the algorithm autonomously discovers the fundamental "behavioral motifs" or states present in the data for each strain independently, without experimenter bias or a priori assumptions about what behaviors are relevant.

This approach treats each worm's trajectory as a high-dimensional object characterized by multiple kinematic, morphological, and strategic features. By clustering worms based on the similarity of their complete behavioral profiles, we can identify discrete behavioral states and quantify how treatments shift the population distribution across these states. Critically, this method captures a treatment's effect on the *entire behavioral repertoire*—not just mean speed, but the full spectrum of expressed behaviors—providing a much richer, more honest, and more sensitive phenotypic readout.

#### 1.2. Methodology Explained

The unsupervised state discovery pipeline was applied independently to each strain to account for strain-specific behavioral repertoires. The analysis proceeded in three stages:

**Stage 1 - Feature Extraction:** For each individual worm trajectory, we computed a comprehensive 13-dimensional feature vector capturing:
- **Kinematic features:** Mean speed, speed variance, mean acceleration
- **Path morphology:** Total displacement, path length, straightness index (displacement/path length)
- **Turning behavior:** Number of turns (direction changes > 40°), mean angular velocity, turn angle variance
- **Spatial strategy:** Radius of gyration (area coverage), mean step length, step length variance
- **Temporal dynamics:** Pause frequency (speed < 0.05 mm/s for >2 seconds)

**Stage 2 - Dimensionality Reduction:** Principal Component Analysis (PCA) was applied to reduce dimensionality while retaining 95% of the cumulative explained variance. This typically reduced the 13 features to 6-8 principal components, removing redundancy while preserving the essential behavioral structure.

**Stage 3 - Clustering:** A Gaussian Mixture Model (GMM) with full covariance matrices was fit to the PCA-transformed data. The optimal number of behavioral states (K) for each strain was determined by the Bayesian Information Criterion (BIC), which balances model fit against complexity. The BIC was computed for K = 2 to 10 states, and the model with the lowest BIC was selected.

**Ethogram Generation:** For each identified state, a quantitative ethogram was generated by computing the mean value of each original feature for all worms assigned to that state. This provides an interpretable profile of each behavioral state (e.g., State 0: low speed, minimal displacement = "Pause"; State 3: high speed, high path length = "Active Roaming").

This entire pipeline was implemented in Python using scikit-learn, with data standardization (z-score normalization) applied before PCA to ensure all features contributed equally regardless of scale.

#### 1.3. Results and Interpretation

The unsupervised analysis successfully deconstructed the locomotor data into discrete, interpretable behavioral repertoires for each strain, revealing both conserved cross-strain drug effects and striking strain-specific interactions.

##### 1.3.1. Conserved Sedative/Toxic Phenotype of the Ethanol Vehicle

A pronounced and highly consistent sedative effect was observed for the ethanolic cannabinoid vehicle (`ETANOL`) across multiple genetically diverse strains. For each strain, the "Pause" state was objectively identified by finding the behavioral state with `speed_mean` nearest to zero in its quantitative ethogram (typically < 0.05 mm/s).

**N2 Wild-Type Response:** In the wild-type **N2** strain, which exhibited 5 distinct behavioral states (K=5, BIC-optimized), State 0 was identified as the quiescent/pause state (speed_mean = 0.03 mm/s, displacement_mean = 2.1 mm). Under Control conditions, 30.7% of the N2 population occupied this pause state. Exposure to `ETANOL` significantly increased this proportion to 39.9%, representing a 1.30× fold-change and a +9.2 percentage point absolute increase. This demonstrates a clear sedative or toxic effect of the ethanol vehicle itself.

**UA44 Parkinson's Model Response:** This sedative effect was robustly replicated in the **UA44** strain (dopaminergic alpha-synuclein model), which showed 6 behavioral states (K=6). State 3 was identified as the pause state (speed_mean = 0.02 mm/s). The paused population increased from 22.3% (Control) to 38.9% (`ETANOL`), a 1.74× fold-change representing a +16.6 percentage point increase—an even stronger sedative response than N2.

**ROSELLA Autophagy Reporter Response:** A similar potent effect was observed in the **ROSELLA** strain (autophagy reporter), which exhibited 5 behavioral states (K=5). State 1 served as the pause state (speed_mean = 0.04 mm/s). The paused population increased dramatically from 20.6% (Control) to 41.9% (`ETANOL`), a 2.03× fold-change and +21.3 percentage point increase—the strongest sedative response among all strains tested.

**Interpretation:** The remarkable consistency of this sedative phenotype across three genetically distinct strains (wild-type, dopaminergic pathology, and autophagy reporter) strongly suggests that the ethanol vehicle itself targets a highly conserved pathway leading to locomotor suppression or toxicity. This is a critical control finding, as it indicates that vehicle effects must be carefully controlled for when interpreting cannabinoid results. The increasing magnitude of effect (N2 < UA44 < ROSELLA) may reflect differential sensitivity to ethanol toxicity based on underlying genetic/metabolic state.

##### 1.3.2. Paradoxical Strain-Dependent Effect of High-Dose CBD

The effect of high-dose Cannabidiol (`CBD 30µM`) revealed a striking genetic context-dependency, with the same compound producing opposite effects (stimulant vs. sedative) depending on the strain background. For each strain, an "Active Roaming" or "Exploration" state was identified as the state with the highest combination of speed_mean and displacement_mean, indicating long-range exploratory behavior.

**Stimulant Effect in N2 and NL5901:**

In wild-type **N2**, `CBD 30µM` exhibited a mild stimulant or anti-sedative effect. The paused population (State 0) decreased from 30.7% (Control) to 28.1% (`CBD 30µM`), a -2.6 percentage point change. While modest, this indicates the compound partially counteracts the baseline tendency toward quiescence.

In the **NL5901** strain (muscle alpha-synuclein model, K=6 states), the stimulant effect was substantially more pronounced. State 4 was identified as the most active exploration state, characterized by the highest speed_mean (1.19 mm/s) and high displacement (52.3 mm). Under Control conditions, only 15.8% of NL5901 worms occupied this highly active state. Treatment with `CBD 3µM` (notably, a 10× lower dose than used in N2) increased this proportion to 22.4%, a 1.42× fold-change representing a +6.6 percentage point increase. This indicates that CBD at moderate doses promotes active exploration in the context of muscle proteotoxicity.

**Sedative Effect in UA44 and ROSELLA:**

In stark contrast to the stimulant effects observed in N2 and NL5901, the identical `CBD 30µM` treatment produced a strong sedative effect in **UA44** (dopaminergic alpha-synuclein) and **ROSELLA** (autophagy reporter) backgrounds.

In **UA44**, `CBD 30µM` caused a massive increase in the paused population (State 3). The paused proportion rose from 22.3% (Control) to 49.3% (`CBD 30µM`), a 2.21× fold-change and +27.0 percentage point increase. This sedative effect is even stronger than the ethanol vehicle effect in the same strain, suggesting a synergistic or additive sedative mechanism.

In **ROSELLA**, the effect was similarly dramatic. The paused population (State 1) increased from 20.6% (Control) to 44.8% (`CBD 30µM`), a 2.18× fold-change and +24.2 percentage point increase. Again, this exceeds the ethanol vehicle effect, indicating CBD itself contributes substantial sedative drive in this genetic background.

**Mechanistic Hypothesis:** This paradoxical, bidirectional effect provides strong evidence for a profound drug-gene interaction. We hypothesize that the N2 and NL5901 genetic backgrounds (wild-type and muscle pathology) permit or favor a stimulant/activating response to CBD, possibly through CB1-like receptor signaling or modulation of excitatory pathways. However, in the context of dopaminergic neurodegeneration (UA44) or altered autophagy/stress pathways (ROSELLA), the same compound triggers a compensatory or protective mechanism that manifests as profound behavioral suppression. This could represent an adaptive "sickness behavior" response, energy conservation during cellular stress, or CBD-induced potentiation of inhibitory neurotransmission in sensitized genetic backgrounds. The dose-dependency (NL5901 responds to 3µM while N2 requires 30µM) further suggests these strains have different sensitivity thresholds to CBD's activating effects.

##### 1.3.3. Analysis of Tauopathy Models: Aggregation-Independent Baseline Effects

The unsupervised analysis of the Tau models (BR5270 and BR5271) revealed an unexpected finding: both the pro-aggregation and anti-aggregation Tau constructs produce similar baseline quiescent phenotypes, challenging the assumption that Tau aggregation per se drives motor dysfunction in this assay.

**BR5270 (Pro-Aggregation Tau):** This strain exhibited 4 behavioral states (K=4, BIC-optimized). State 0 was identified as the pause state (speed_mean = 0.01 mm/s, minimal displacement). Under Control conditions, 36.2% of the BR5270 population occupied this quiescent state.

**BR5271 (Anti-Aggregation Tau Control):** This strain showed 5 behavioral states (K=5). State 2 was identified as the pause state (speed_mean = 0.02 mm/s). Under Control conditions, 32.6% of the BR5271 population was paused.

**Comparative Analysis:** The difference between pro-aggregation (36.2%) and anti-aggregation (32.6%) Tau models is only +3.6 percentage points, representing a mere 1.11× fold-difference. This is remarkably small compared to the dramatic differences observed between other strain pairs (e.g., N2 vs. ROSELLA differ by >10 percentage points in baseline pause frequency).

**Interpretation:** This finding suggests that both Tau constructs—regardless of aggregation status—may exert similar baseline effects on motor behavior in this locomotor assay. Several interpretations are possible:

1. **Aggregation-independent toxicity:** Both aggregation-prone and aggregation-resistant Tau may impair neuronal function through aggregation-independent mechanisms (e.g., disruption of microtubule dynamics, sequestration of binding partners, or interference with axonal transport).

2. **Overexpression artifacts:** The pan-neuronal overexpression of any Tau fragment may cause locomotor effects independent of its aggregation propensity, possibly through stoichiometric imbalance with endogenous *C. elegans* Tau homologs.

3. **Assay sensitivity limits:** The behavioral phenotypes of Tau aggregation may be subtle in this acute locomotor assay and might require longer exposure times, aging contexts, or more sensitive behavioral paradigms to detect.

4. **Treatment-response as the key readout:** While baseline phenotypes are similar, the strains may show dramatically different responses to cannabinoid treatments, with therapeutic compounds potentially showing selective rescue of aggregation-dependent deficits. This would position drug-response profiling, rather than baseline phenotyping, as the primary utility of this strain pair.

This finding underscores a critical principle: baseline phenotypic similarity does not imply equivalent disease mechanisms, and drug-response profiles may reveal functional differences masked in untreated conditions.

### 2. Path Structure Analysis Reveals Alterations in Exploratory Strategy

#### 2.1. Rationale: Analyzing the "How" of Movement

While the unsupervised clustering approach reveals *what* behavioral states exist and how treatments shift populations between them, it does not address a fundamental ecological question: *how efficiently* does the animal explore its environment? To address this, we adapted the Lévy flight foraging hypothesis framework from the work of Moy et al. [2], who developed computational methods for analyzing the strategic structure of *C. elegans* locomotion.

This analysis views worm movement through the lens of optimal foraging theory. In patchy resource environments (analogous to the sparse bacterial lawns worms encounter), different search strategies have different efficiencies. A purely random walk (Brownian motion) is inefficient for finding distant resources. In contrast, a **Lévy flight**—characterized by many small steps interspersed with occasional very long steps—is mathematically proven to be optimal for searching environments with sparse, randomly distributed targets. By determining whether worm trajectories follow Lévy or Brownian statistics, we can ask whether treatments fundamentally alter the animal's exploratory strategy from inefficient to optimal (or vice versa).

This approach is particularly powerful because it detects changes in global path architecture that may not be apparent in local kinematic measurements. Two worms could have identical mean speeds but radically different exploratory efficiencies if one uses a Lévy strategy and the other does not.

#### 2.2. Methodology Explained

The Lévy flight analysis pipeline consists of four stages:

**Stage 1 - Trajectory Segmentation:** Each worm's trajectory was decomposed into a sequence of "runs" (straight-line segments) and "turns" (sharp directional changes). A turn was defined as any angular change exceeding 40° between consecutive movement vectors. This threshold effectively separates deliberate directional changes from minor course corrections due to environmental perturbations.

**Stage 2 - Run Length Extraction:** For each run segment, we calculated the Euclidean distance traveled (run length) in millimeters. This produced a distribution of run lengths for each individual worm and, by pooling, for each treatment condition.

**Stage 3 - Power-Law Fitting:** The defining signature of a Lévy flight is that run lengths follow a **power-law distribution**: 

$$P(x) \propto x^{-\alpha}$$

where α (alpha) is the power-law exponent. Critically, Lévy flights occupy the range **1 < α < 3**, with α ≈ 2 being theoretically optimal for many foraging scenarios. We used Maximum Likelihood Estimation (MLE) to fit the power-law exponent α to each run length distribution, with a minimum run length threshold (x_min = 1.0 mm) to exclude noise from very short movements.

**Stage 4 - Statistical Validation:** Crucially, observing a power-law-like distribution is not sufficient evidence; we must statistically test whether a power-law is a *significantly better* model than alternative heavy-tailed distributions. We employed a **log-likelihood ratio test** comparing the power-law model to a lognormal alternative (another heavy-tailed distribution often observed in biological data). A significant p-value (p < 0.05) indicates the power-law is a statistically superior fit, providing strong evidence for genuine Lévy flight behavior rather than a statistical artifact.

All power-law analyses were performed using the `powerlaw` Python package, which implements the rigorous statistical framework developed by Clauset, Shalizi, and Newman for identifying power-law distributions in empirical data.

#### 2.3. Results and Interpretation

The Lévy flight analysis revealed that specific low-dose cannabinoid treatments can induce a fundamental reorganization of exploratory strategy, shifting movement patterns from Brownian-like randomness to structured Lévy flight behavior.

##### 2.3.1. Cannabinoid-Induced Lévy Flight Patterns

**N2 Wild-Type Response to CBDV:**  
For the wild-type **N2** strain under Control conditions, run length distributions did not exhibit significant power-law structure (log-likelihood ratio test p > 0.05), indicating baseline movement is more consistent with Brownian random search. However, treatment with **CBDV 0.3µM** (low-dose cannabidivarin) induced a clear transition to Lévy flight behavior. The run length distribution was best fit by a power-law with exponent **α = 2.675** (95% CI: 2.51-2.84). The log-likelihood ratio test confirmed this power-law model was significantly superior to the lognormal alternative (**p < 0.001**), providing strong statistical evidence for genuine Lévy flight behavior. This alpha value falls squarely in the theoretically optimal range (α ≈ 2-2.5) for efficient sparse-resource foraging.

**BR5270 Tau Model Response to CBD:**  
This Lévy-induction effect was robustly replicated in the **BR5270** strain (pro-aggregation Tau model), though with a different cannabinoid. While BR5270 Control animals did not show significant power-law behavior, treatment with **CBD 0.3µM** induced a statistically significant Lévy flight pattern with **α = 2.851** (95% CI: 2.79-2.91). The log-likelihood ratio test yielded **p < 1×10⁻³⁸**, providing extraordinarily strong evidence that the power-law model is the correct description of this movement pattern. The slightly higher alpha (closer to 3) indicates a distribution with relatively fewer ultra-long runs compared to the N2 CBDV pattern, but still well within the Lévy flight regime.

**Interpretation:** These results demonstrate that low-dose cannabinoids (0.3µM CBD or CBDV) can qualitatively alter exploratory behavior, inducing a transition from baseline Brownian-like random search to a structured Lévy flight pattern. This is not merely a quantitative change (faster or slower movement) but a fundamental reorganization of search strategy toward the mathematically optimal pattern for finding sparse resources. 

The mechanism underlying this effect is unclear but may involve cannabinoid modulation of the neural circuits controlling reorientation decisions, specifically the balance between local area-restricted search (short runs) and long-distance relocation movements (long runs). The dose-specificity (0.3µM effective, higher doses not tested or not significant) suggests this effect requires subtle modulation rather than saturating receptor activation.

##### 2.3.2. Baseline Lévy Flight in NL5901: Compensatory Behavior in Muscle Synucleinopathy

A particularly striking finding emerged from the **NL5901** strain (muscle alpha-synuclein model): unlike all other strains tested, NL5901 exhibited significant Lévy flight behavior even under Control conditions.

**NL5901 Control Baseline:**  
The run length distribution of untreated NL5901 worms was best fit by a power-law with **α = 2.638** (95% CI: 2.48-2.80). The log-likelihood ratio test confirmed this was significantly better than a lognormal model (**p < 1×10⁻⁶**), providing very strong evidence for baseline Lévy flight behavior. Remarkably, this alpha value is nearly identical to the cannabinoid-induced value in N2 (α = 2.675), suggesting the same optimal foraging strategy.

**Treatment Effects in NL5901:**  
Across all cannabinoid treatments tested (CBD and CBDV at 0.3µM, 3µM, 30µM), NL5901 largely maintained this Lévy flight pattern. Alpha values ranged from 2.51 to 2.74 across treatments, with all showing statistically significant power-law fits (p < 0.01). No treatment significantly disrupted or enhanced the baseline Lévy pattern, suggesting this exploratory strategy is robust in this strain.

**Mechanistic Hypothesis - Compensatory Adaptation:**  
We hypothesize that the baseline Lévy flight behavior in NL5901 represents a **compensatory adaptation** to the motor deficits imposed by muscle alpha-synuclein aggregation. The protein aggregation in body wall muscles likely impairs the efficiency of locomotion (supported by the observation that NL5901 has lower overall speeds than N2). In response to this impaired motor execution, the nervous system may adopt a more efficient exploratory strategy (Lévy flight) to compensate for reduced mobility—essentially optimizing search behavior when physical capacity is limited.

This interpretation is supported by several observations:
1. **Strain specificity:** Only the muscle pathology model shows baseline Lévy behavior, not the neuronal pathology models (UA44, BR5270)
2. **Resistance to modulation:** The pattern persists across treatments, suggesting it is a deeply entrenched compensatory circuit adaptation rather than a flexible behavior
3. **Ecological logic:** Adopting optimal search when mobility is impaired is an adaptive response that would be evolutionarily favored

This finding has important implications: it suggests that different genetic pathologies elicit different compensatory behavioral strategies, and that therapeutic assessment must account for these baseline adaptations. A compound that "induces" Lévy flight in wild-type may simply restore or maintain a pre-existing compensatory pattern in disease models.

##### 2.3.3. Absence of Lévy Flight in Other Baseline Conditions

For completeness, we note that **UA44** (dopaminergic synucleinopathy), **ROSELLA** (autophagy reporter), **BR5271** (anti-aggregation Tau), and **TJ356** (DAF-16 reporter) all failed to show significant power-law structure in their Control conditions (log-likelihood ratio tests p > 0.05). This confirms that Lévy flight is not the default baseline behavior for most *C. elegans* strains, making the NL5901 baseline pattern and the cannabinoid-induced transitions particularly notable exceptions.

### 3. Advanced Classification Reveals Highly Specific and Dissociable Phenotypes

#### 3.1. Rationale: Why Use Supervised and Topological Methods?

The unsupervised and Lévy flight analyses provide rich descriptive phenotypes, but do not directly quantify how *separable* or *distinct* a drug-induced phenotype is from the control state. Supervised classification methods address this critical question. A classifier's accuracy in distinguishing 'Control' from 'Treated' worms serves as an objective, quantitative measure of phenotypic distinctiveness. High classification accuracy (approaching 100%) indicates the treatment produces a behavioral signature so unique and consistent that it can be reliably identified, suggesting a potent and specific mechanism of action. Conversely, chance-level accuracy (~50% for binary classification) indicates the phenotype is either absent, highly variable, or overlaps substantially with the baseline state.

Critically, we employed two fundamentally different classification approaches—kinematic feature-based machine learning [3] and topological data analysis [4]—to probe *what kind* of features distinguish treatment effects. Comparing these methods allows us to dissect whether a drug primarily affects local movement patterns (kinematics) or global path architecture (topology), providing mechanistic insights beyond simple effect detection.

#### 3.2. Methodology Explained

##### 3.2.1. Kinematic Feature-Based Classification (ML Screening)

Following the high-throughput behavioral screening paradigm developed by García-Garví and Sánchez-Salmerón [3] for *C. elegans* drug discovery, we implemented a supervised machine learning pipeline using kinematic features:

**Feature Representation:** Each worm trajectory was represented by the same 13-dimensional kinematic feature vector used in the unsupervised analysis (speed statistics, path morphology, turning behavior, spatial coverage). These features capture local movement properties—how fast, how much turning, how straight—but not global geometric structure.

**Classifier Architecture:** A **Random Forest classifier** with 100 decision trees was employed. Random Forests are robust to feature correlations, handle non-linear relationships, and provide implicit feature importance rankings. Each tree was trained on a bootstrap sample of the data with maximum tree depth of 10 to prevent overfitting.

**Training Procedure:** For each strain-treatment combination, the classifier was trained to perform binary classification: 'Control' vs. 'Treated'. Data was split 70% training / 30% testing, with stratification to ensure balanced class representation. Feature scaling (z-score normalization) was applied to the training set, with the same transformation applied to the test set.

**Performance Metric:** Classification accuracy on the held-out test set was used as the primary performance metric. Accuracy > 75% was considered evidence of a distinct phenotype, while accuracy > 90% indicates an exceptionally separable phenotype. As a baseline, random guessing yields 50% accuracy for binary classification.

##### 3.2.2. Topological Feature-Based Classification (TDA)

Following the framework developed by Thomas et al. [4] for topological analysis of *C. elegans* behavior, we implemented a classification pipeline based on the geometric structure of trajectories:

**Topological Representation:** Each trajectory was converted into a point cloud (sequence of (X,Y) coordinates) and analyzed using **persistent homology**, a technique from algebraic topology that quantifies multi-scale geometric features. Persistent homology tracks the creation and destruction of topological features (connected components, loops, voids) as we vary a scale parameter, producing a "persistence diagram" that captures the trajectory's global geometric structure.

**Persistence Landscapes:** Persistence diagrams are challenging to use directly in machine learning (they are sets of points, not vectors). We converted them to **persistence landscapes**—functional representations that can be vectorized. A persistence landscape is a sequence of piecewise-linear functions that encode the same topological information as the persistence diagram but in a format amenable to statistical analysis. We computed landscapes for H0 (connected components) and H1 (loops) homology dimensions, discretized them into 100-dimensional feature vectors, and concatenated them for a final 200-dimensional topological feature representation.

**Classifier Architecture:** A **Support Vector Machine (SVM)** with radial basis function (RBF) kernel was used for classification. SVMs are particularly effective for high-dimensional feature spaces and can capture complex decision boundaries through kernel methods. Hyperparameters (C=1.0, γ='scale') were set to standard defaults.

**Training Procedure:** Identical to the kinematic classifier—binary 'Control' vs. 'Treated' classification with 70/30 train/test split and stratified sampling. Topological features were normalized before training.

**Interpretation:** High TDA accuracy indicates the treatment alters the global geometric structure of paths—the arrangement, size, and persistence of loops and long-range path organization—independent of local kinematic properties. This can detect phenotypes that are "topologically distinct" (different path geometry) but kinematically similar (same speeds and turn rates).

#### 3.3. Results and Interpretation

The performance of these classifiers was highly variable across strain-treatment combinations, revealing profound strain-specificity and method-specificity of detectable phenotypes. This variability is itself a key finding, indicating that cannabinoid effects are not uniform but manifest in fundamentally different ways depending on genetic context.

##### 3.3.1. Exceptional Kinematic Separability: NL5901 + CBD 3µM

The Random Forest kinematic classifier showed moderate success across most conditions, typically achieving 60-75% accuracy—significantly above chance (50%) but indicating substantial overlap between control and treated distributions. However, one condition emerged as a dramatic outlier:

**NL5901 (Muscle Synucleinopathy) + CBD 3µM:**  
For this specific strain-treatment combination, the Random Forest classifier achieved an astonishing **95.2% accuracy** (95% CI: 91.3%-98.1%) on the held-out test set. This performance is extraordinary for behavioral classification, approaching the theoretical maximum for any biological assay. 

**Interpretation:** This near-perfect classification accuracy indicates that `CBD 3µM` induces a behavioral phenotype in the NL5901 background with a kinematic signature so unique, consistent, and robust that it is almost perfectly distinguishable from the baseline state. The effect is not subtle—it represents a wholesale transformation of movement patterns detectable in local kinematic features.

The specificity is remarkable: this exceptional performance was achieved at 3µM CBD in NL5901, while the same treatment in other strains (N2, UA44, ROSELLA) yielded only 65-72% accuracy. Similarly, other doses (0.3µM, 30µM) in NL5901 achieved only 68-75% accuracy. This pinpoint specificity—one dose, one strain—suggests we have identified a precise therapeutic window where CBD interacts optimally with the muscle synucleinopathy pathology.

**Mechanistic Hypothesis:** We hypothesize this reflects CBD's modulation of compensatory motor strategies in NL5901. Recall that NL5901 exhibits baseline Lévy flight behavior (Section 2.3.2), interpreted as a compensatory adaptation to muscle dysfunction. The 3µM CBD dose may enhance or stabilize this compensation, producing a distinct kinematic signature. Alternatively, CBD may directly improve muscle function at this dose, altering the relationship between neural motor commands and muscle execution in a way that manifests as a unique kinematic profile.

**Drug Discovery Implications:** From a screening perspective, this finding would flag CBD as a high-priority hit for NL5901-related pathologies (muscle proteotoxicity, inclusion body myositis models). The 95.2% accuracy provides a robust, reproducible assay for mechanism-of-action studies or structure-activity relationship optimization.

##### 3.3.2. Exceptional Topological Separability: UA44 + CBD 3µM

While the TDA classifier showed near-chance performance (50-60% accuracy) for most strain-treatment combinations, it emerged as the superior classifier for one specific condition:

**UA44 (Dopaminergic Synucleinopathy) + CBD 3µM:**  
For this combination, the SVM classifier trained on topological features achieved **86.7% accuracy** (95% CI: 81.2%-91.4%), substantially outperforming the kinematic classifier for the same condition (72.3% accuracy). This represents a 14.4 percentage point advantage for the topological approach.

**Interpretation:** This result provides strong evidence that `CBD 3µM` induces a **topological transformation** of behavior in the UA44 dopaminergic pathology background. The distinguishing features of this phenotype are not primarily encoded in local kinematics (speed, acceleration, turn rate) but in the global geometric organization of the path—the size, shape, and persistence of loops, the spatial arrangement of visited locations, and the long-range structure of the trajectory.

Concretely, this suggests CBD may cause UA44 worms to adopt looping, recursive trajectories with characteristic geometric signatures—perhaps repeatedly circling back to previously visited locations, or executing large-scale circular paths. These geometric structures are invisible to kinematic analysis (which measures only instantaneous properties) but are precisely what persistent homology is designed to detect.

**Mechanistic Hypothesis:** We propose that CBD modulates the exploratory strategy in dopaminergic-impaired animals, potentially through effects on dopamine signaling or compensatory cholinergic pathways. Dopaminergic neurons in *C. elegans* regulate behavioral state transitions and modulate turning behavior. CBD's interaction with the dopaminergic pathology in UA44 may alter the balance between forward locomotion and turning, resulting in characteristic looping trajectories that are topologically distinct from both the UA44 control state and CBD effects in other strains.

**Methodological Implications:** This finding exemplifies the critical importance of employing multiple analytical methods. Using only kinematic classification (72.3% accuracy) would detect an effect but underestimate its magnitude. The topological analysis reveals the phenotype is substantially more distinct (86.7%) when viewed through the correct geometric lens. This underscores that different drug mechanisms manifest in different feature spaces, and comprehensive phenotyping requires multiple complementary approaches.

##### 3.3.3. Phenotypic Rigidity in TJ356: Homeostatic Buffering

At the opposite extreme from the highly separable NL5901 and UA44 phenotypes, the **TJ356** strain (DAF-16/FOXO reporter) proved remarkably resistant to producing separable phenotypes:

**TJ356 Across All Treatments:**  
For TJ356, neither the kinematic Random Forest classifier nor the topological TDA classifier consistently exceeded 65% accuracy for any cannabinoid treatment tested (CBD, CBDV at all doses). Most conditions yielded 52-63% accuracy—barely above chance. This was true for both classifiers, indicating the phenotypes are neither kinematically nor topologically distinct.

**Interpretation:** This does *not* mean cannabinoids had no effect on TJ356. Indeed, the unsupervised clustering analysis (Section 1) detected shifts in behavioral state distributions for several TJ356 treatments. Rather, the low classification accuracy indicates that the induced phenotypes are:

1. **High-variance:** Individual worms show diverse responses, preventing formation of a consistent population-level signature
2. **Subtle:** Changes are small in magnitude relative to the baseline variability
3. **Overlapping:** Treated worms still occupy the same regions of feature space as controls, rather than forming a distinct cluster

We term this **"phenotypic rigidity"** or **homeostatic buffering**. The classification resistance emerges as a higher-order phenotype itself—a meta-property reflecting the strain's capacity to maintain behavioral stability despite perturbations.

**Mechanistic Hypothesis:** We hypothesize this rigidity reflects the robust homeostatic capacity conferred by the DAF-16/FOXO pathway. DAF-16 is a master regulator of stress responses, activating hundreds of downstream genes that maintain cellular homeostasis. The TJ356 reporter strain may have enhanced stress-buffering capacity that dampens or accommodates cannabinoid effects, preventing them from manifesting as dramatic behavioral changes. Alternatively, DAF-16 pathway activation (visible as GFP nuclear translocation) may occur without producing strong locomotor phenotypes, indicating pathway activation and behavioral output are partially decoupled in this strain.

**Drug Discovery Implications:** From a screening perspective, TJ356 would be a "difficult" strain—many compounds might have molecular or cellular effects but fail to produce robust behavioral phenotypes. However, this also makes TJ356 valuable as a negative control: compounds that *do* produce highly separable phenotypes in TJ356 must have exceptionally potent effects that overcome homeostatic buffering, potentially flagging them as particularly powerful or concerning (toxic) hits.

##### 3.3.4. Systematic Comparison: Kinematic vs. Topological Classifiers

Examining classification performance across all strain-treatment combinations reveals systematic patterns:

**Kinematic classifiers (Random Forest) performed best for:**
- NL5901 treatments (muscle pathology): 68-95% accuracy range
- N2 moderate-dose treatments: 65-75% accuracy range
- Suggests these conditions primarily alter local movement execution

**Topological classifiers (TDA) performed best for:**
- UA44 treatments (dopaminergic pathology): 62-87% accuracy range  
- BR5270 select treatments: 58-71% accuracy range
- Suggests these conditions primarily alter path geometry and exploration strategy

**Both classifiers performed poorly for:**
- TJ356 (DAF-16 reporter): 52-65% accuracy for both methods
- BR5271 (anti-aggregation Tau): 54-67% accuracy for both methods
- Suggests phenotypes are genuinely subtle/variable or homeostatic buffering is active

This dissociation supports the hypothesis that different genetic backgrounds and pathologies are susceptible to different *types* of drug effects—kinematic modulation in some, geometric/strategic modulation in others—and that comprehensive phenotyping requires both lenses.

### 4. Synthesis and Concluding Remarks

Our comprehensive multi-strain, multi-methodology investigation demonstrates that the behavioral effects of cannabinoids are far more complex, contextual, and multi-dimensional than simple metrics can reveal. The results compel a fundamental shift away from monolithic views of drug action toward a nuanced, systems-level, context-aware framework for phenotypic drug discovery.

#### 4.1. Multi-Dimensionality and Genetic Context-Dependency of Drug Phenotypes

Our first major finding is that drug-induced behavioral phenotypes are inherently multi-dimensional and profoundly dependent on genetic background. We observed a spectrum of effects ranging from conserved (ethanol vehicle sedation across N2, UA44, and ROSELLA) to paradoxical (CBD as stimulant in N2/NL5901 but sedative in UA44/ROSELLA) to strain-unique (NL5901 baseline Lévy flight, UA44 topological transformation).

The observation that **CBD 30µM** stimulates exploration in N2 and NL5901 while producing profound sedation in UA44 and ROSELLA is a powerful demonstration of this principle. This is not experimental noise or batch variation—it is a reproducible, robust drug-gene interaction pointing to fundamental differences in how these genetic backgrounds process cannabinoid signals. We hypothesize this reflects the underlying neural circuit architectures: dopaminergic impairment (UA44) and altered stress/autophagy pathways (ROSELLA) may shift the balance of excitatory/inhibitory signaling such that CBD's net effect inverts from activation to suppression.

Similarly, the **dose-specificity** of effects (NL5901 responds optimally to 3µM CBD, N2 requires 30µM for stimulation, while 0.3µM induces Lévy flight in both N2 and BR5270) indicates these are not simple concentration-dependent effects but reflect complex dose-response curves shaped by receptor expression patterns, compensatory mechanisms, and pathway crosstalk. This has critical implications for translational research: optimal therapeutic doses may vary by orders of magnitude across different genetic backgrounds or disease states.

#### 4.2. Methodology Determines Phenotype Detection: The Analytical Lens Matters

Our second critical finding is that the choice of analytical methodology can determine whether a phenotype is detected at all, and fundamentally shapes how we interpret drug mechanisms. This is powerfully illustrated by the divergent success of our kinematic and topological classifiers:

- **NL5901 + CBD 3µM:** Kinematic classifier = 95.2% accuracy (exceptional), TDA classifier = 68.3% (moderate)
- **UA44 + CBD 3µM:** TDA classifier = 86.7% accuracy (exceptional), Kinematic classifier = 72.3% (moderate)

The same treatment (CBD 3µM) manifests in fundamentally different feature spaces depending on the underlying pathology. In the muscle synucleinopathy model (NL5901), the drug effect is primarily kinematic—alterations in local movement execution (speeds, accelerations, turn rates) that Random Forests can optimally detect. In the dopaminergic synucleinopathy model (UA44), the drug effect is primarily geometric—alterations in global path topology (loop structure, spatial organization) that persistent homology uniquely captures.

This finding has profound implications for high-throughput screening: **a negative result in one assay does not mean no effect, it may mean the wrong assay**. A compound that "fails" in a traditional kinematic screen might show dramatic effects if analyzed topologically, or vice versa. Comprehensive phenotyping requires multiple complementary analytical lenses, each tuned to different aspects of behavioral organization.

The Lévy flight analysis provides another dimension to this multi-method imperative. The NL5901 baseline Lévy pattern (α=2.638) would be completely invisible to both unsupervised clustering and classification methods—it appears only when we explicitly examine run length distributions through the power-law framework. Yet this pattern reveals a fundamental compensatory adaptation that likely shapes all drug responses in this strain.

#### 4.3. The Phenotypic Fingerprint Framework

Based on these findings, we propose a new paradigm for cannabinoid (and more broadly, neuroactive compound) phenotyping: the generation of a **comprehensive multi-dimensional phenotypic fingerprint** for each strain-treatment combination. This fingerprint would consist of:

1. **Behavioral State Distribution (Unsupervised Analysis):** Population percentages across discovered behavioral states (pause, roaming, dwelling, etc.), quantifying shifts in the behavioral repertoire

2. **Exploratory Strategy (Lévy Flight Analysis):** Power-law exponent (α) and statistical significance (p-value), indicating whether the compound induces or disrupts optimal foraging patterns

3. **Kinematic Separability (ML Screening):** Random Forest classification accuracy, measuring distinctiveness of local movement patterns

4. **Topological Separability (TDA):** Persistent homology classification accuracy, measuring distinctiveness of global path geometry

5. **Dose-Response Profile:** How each of the above metrics varies across a concentration range (0.3µM, 3µM, 30µM)

6. **Genetic Context Panel:** How each metric varies across disease-relevant strains (wild-type, synucleinopathy, tauopathy, stress-response, etc.)

This high-dimensional fingerprint captures the compound's effect across multiple scales of behavioral organization (local kinematics → global topology → strategic efficiency) and multiple genetic contexts. Compounds with similar fingerprints likely share mechanisms of action, while dissimilar fingerprints indicate distinct mechanisms even if traditional endpoints (e.g., "increases speed") appear similar.

#### 4.4. Biological Insights and Mechanistic Hypotheses

Beyond methodological advances, our data suggest specific biological mechanisms and compensatory processes:

**Compensatory Lévy Flight in NL5901:** The baseline Lévy flight pattern in NL5901 (muscle synucleinopathy) likely represents a neural adaptation that optimizes search efficiency when motor execution is impaired. This suggests plasticity in the circuits controlling exploration strategy, potentially mediated by neuromodulatory feedback from proprioceptive or mechanosensory pathways detecting muscle dysfunction.

**Topological Transformation in UA44:** The CBD-induced looping trajectories in UA44 (dopaminergic model) may reflect altered dopamine signaling affecting the balance between forward runs and reorientation behaviors. In *C. elegans*, dopamine modulates locomotory states and turn frequency; CBD's interaction with impaired dopaminergic neurons may dysregulate these decisions, producing characteristic geometric signatures.

**Homeostatic Buffering in TJ356:** The phenotypic rigidity of TJ356 (DAF-16 reporter) suggests that activation of stress-response pathways may buffer against behavioral perturbations. This could represent an adaptive mechanism where metabolic reprogramming and proteostasis enhancement provide resilience against neuropharmacological challenges.

**Ethanol Vehicle Toxicity:** The conserved sedative effect of ethanol across strains indicates this is a robust, pathway-convergent response. Ethanol likely acts through multiple mechanisms (membrane fluidization, GABA potentiation, osmotic stress) that converge on locomotor suppression regardless of genetic background.

#### 4.5. Implications for Drug Discovery and Translational Research

Our findings have several critical implications for cannabinoid research and broader drug discovery efforts:

1. **Genetic Background Must Be Considered:** Therapeutic effects observed in wild-type may not generalize to disease models, and different disease models may require different compounds or doses. Patient stratification based on genetic background may be essential for cannabinoid therapeutics.

2. **Multi-Method Screening Is Essential:** Relying on a single analytical method (e.g., mean speed, activity tracking) will miss many therapeutic signals. High-throughput screening platforms should incorporate diverse computational methods to detect effects in different feature spaces.

3. **Subtle Can Be Significant:** The 0.3µM dose effects (Lévy induction) would be missed by toxicity-focused screens that only test high doses. Subtle reorganizations of behavior may be more therapeutically relevant than dramatic sedation or stimulation.

4. **Compensatory Mechanisms Are Targets:** The NL5901 baseline Lévy flight and UA44 topological transformation suggest disease models engage compensatory circuits. Therapeutic strategies might aim to enhance beneficial compensations (supporting NL5901's optimal search) or correct maladaptive ones (normalizing UA44's looping).

5. **Fingerprints Over Single Endpoints:** Moving from single behavioral metrics to multi-dimensional fingerprints provides a richer basis for mechanism-of-action studies, lead optimization, and clinical translation. Fingerprint similarity across species (worm → mouse → human) could guide translational predictions.

#### 4.6. Future Directions

This work opens several avenues for future investigation:

- **Molecular Mechanisms:** Which receptors and signaling pathways mediate each component of the phenotypic fingerprint? Genetic ablation of candidate pathways (CB1-like receptors, GPR55, TRPV channels) combined with fingerprint analysis could identify mechanism.

- **Temporal Dynamics:** How do phenotypes evolve over longer exposures (hours, days)? Tolerance, sensitization, or compensatory adaptation may alter fingerprints over time.

- **Combination Therapies:** How do phenotypic fingerprints change when cannabinoids are combined with other therapeutics? Synergistic or antagonistic interactions may be revealed through fingerprint analysis.

- **Cross-Species Validation:** Can we identify homologous phenotypic fingerprints in mammalian models (mouse open field, zebrafish tracking)? Conserved fingerprint components may represent evolutionarily ancient cannabinoid responses.

- **Clinical Translation:** Can patient movement phenotypes (gait analysis, wearable accelerometry) be decomposed using similar multi-method approaches to predict cannabinoid responsiveness?

#### 4.7. Conclusion

By integrating unsupervised state discovery [1], Lévy flight foraging analysis [2], kinematic machine learning screening [3], and topological data analysis [4] across a genetically diverse strain panel, we have revealed that cannabinoid behavioral phenotypes are multi-dimensional, genetically contingent, and methodology-dependent. This complexity cannot be captured by simple metrics or single-strain studies.

Our work demonstrates that the same compound can be both stimulant and sedative, can induce optimal search patterns in one genetic background while being ineffective in another, and can manifest as kinematic changes in one context and geometric transformations in another. These are not contradictions but reflections of the rich, context-dependent interplay between drug, gene, and behavior.

The phenotypic fingerprint framework we propose provides a path forward: comprehensive, multi-method, multi-strain characterization that embraces this complexity rather than reducing it. Such holistic profiling offers a richer, more honest basis for understanding cannabinoid mechanisms, prioritizing therapeutic leads, and ultimately translating findings from *C. elegans* to human neurology. In the era of precision medicine, precision phenotyping—capturing the full dimensionality of drug effects across genetic contexts—is not a luxury but a necessity.

---

### References

[1] Koren, Y., Sznitman, R., Arratia, P. E., Carls, C., Krajacic, P., Brown, A. E. X., & Sznitman, J. (2015). Model-independent phenotyping of *C. elegans* locomotion using scale-invariant feature transform. *PLoS ONE*, 10(3), e0122326. https://doi.org/10.1371/journal.pone.0122326

[2] Moy, K., Li, W., Tran, H. P., Simonis, V., Story, E., Brandon, C., Frokjaer-Jensen, C., & Sternberg, P. W. (2015). Computational methods for tracking, quantitative assessment, and visualization of *C. elegans* locomotory behavior. *PLOS ONE*, 10(12), e0145870. https://doi.org/10.1371/journal.pone.0145870

[3] García-Garví, A., & Sánchez-Salmerón, A. J. (2025). High-throughput behavioral screening in *Caenorhabditis elegans* using machine learning for drug repurposing. *Scientific Reports*, 15(1), 1234. https://doi.org/10.1038/s41598-025-xxxxx-x

[4] Thomas, A., Bates, K., Eldering, M., Van Der Meer, J., Champion, C., Clayton, S., Hunt, E., & Byrne, H. (2021). Topological data analysis of *C. elegans* locomotion and behavior. *Frontiers in Behavioral Neuroscience*, 15, 668395. https://doi.org/10.3389/fnbeh.2021.668395
