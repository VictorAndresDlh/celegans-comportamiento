## Comprehensive ML Screening Analysis

This analysis shows machine learning classification results for ALL strain/treatment combinations.

**Methodology (Based on García-Garví et al. 2025):**
- **20 Centroid-based Features:**
  - Speed metrics (7): mean, std, max, median, percentiles, coefficient of variation
  - Turning metrics (5): angular velocity stats, reversal frequency
  - Path metrics (5): tortuosity, total distance, displacement, path efficiency
  - Temporal metrics (3): active time ratio, pause frequency, movement bout duration
- **Classifiers Compared:**
  - Random Forest (max_depth=10, min_samples_split=5)
  - XGBoost (gradient boosting)
  - Logistic Regression (L1 regularization)
- **Validation:** Stratified 5-fold cross-validation with permutation tests

**Performance Categories:**
- **Excellent (>90%):** Very strong behavioral phenotype, highly distinguishable
- **Good (75-90%):** Clear behavioral phenotype, reliably distinguishable
- **Moderate (60-75%):** Detectable phenotype, some classification ability
- **Poor (<60%):** Weak/no phenotype, difficult to distinguish from control

### Complete Classification Results

| strain   | treatment         | Classifier   | Accuracy (mean)   |   Accuracy (std) |   F1 (Control) |   F1 (Treatment) | Performance Category   |
|:---------|:------------------|:-------------|:------------------|-----------------:|---------------:|-----------------:|:-----------------------|
| NL5901   | CBD 3uM           | RandomForest | 95.2%             |            0.003 |       0.975667 |         0        | Excellent (>90%)       |
| UA44     | CBDV 3uM          | RandomForest | 92.1%             |            0.008 |       0.958573 |         0.130864 | Excellent (>90%)       |
| NL5901   | CBDV 30uM         | RandomForest | 83.7%             |            0.015 |       0.909592 |         0.190105 | Good (75-90%)          |
| NL5901   | nan               | RandomForest | 79.5%             |            0.022 |       0.882343 |         0.200535 | Good (75-90%)          |
| N2       | CBD 30uM          | RandomForest | 76.5%             |            0.01  |       0.861947 |         0.213635 | Good (75-90%)          |
| ROSELLA  | CBD 3uM           | RandomForest | 75.5%             |            0.02  |       0.82074  |         0.611477 | Good (75-90%)          |
| ROSELLA  | CBD 0.3uM         | RandomForest | 75.4%             |            0.021 |       0.832478 |         0.534026 | Good (75-90%)          |
| UA44     | CBD 3uM           | RandomForest | 74.2%             |            0.017 |       0.842736 |         0.274221 | Moderate (60-75%)      |
| ROSELLA  | CBD 30uM          | RandomForest | 74.1%             |            0.014 |       0.817335 |         0.5561   | Moderate (60-75%)      |
| UA44     | CBD 30uM          | RandomForest | 73.7%             |            0.022 |       0.836413 |         0.329732 | Moderate (60-75%)      |
| BR5271   | ETANOL            | RandomForest | 72.2%             |            0.032 |       0.815114 |         0.431367 | Moderate (60-75%)      |
| N2       | CBD 3uM           | RandomForest | 71.8%             |            0.011 |       0.823407 |         0.296639 | Moderate (60-75%)      |
| ROSELLA  | CBDV 30uM         | RandomForest | 71.0%             |            0.026 |       0.821007 |         0.230041 | Moderate (60-75%)      |
| UA44     | CBDV 0.3uM        | RandomForest | 70.5%             |            0.018 |       0.814219 |         0.283685 | Moderate (60-75%)      |
| UA44     | CBDV 30uM         | RandomForest | 70.4%             |            0.023 |       0.820142 |         0.165842 | Moderate (60-75%)      |
| NL5901   | CBD 30uM          | RandomForest | 70.2%             |            0.045 |       0.815885 |         0.186601 | Moderate (60-75%)      |
| ROSELLA  | CBDV 3uM          | RandomForest | 69.5%             |            0.024 |       0.766263 |         0.562136 | Moderate (60-75%)      |
| ROSELLA  | ETANOL            | RandomForest | 69.2%             |            0.023 |       0.765141 |         0.552197 | Moderate (60-75%)      |
| UA44     | CBD 0.3uM         | RandomForest | 69.1%             |            0.018 |       0.774139 |         0.511483 | Moderate (60-75%)      |
| ROSELLA  | CBDV 0.3uM        | RandomForest | 68.5%             |            0.028 |       0.777325 |         0.459331 | Moderate (60-75%)      |
| UA44     | Total_Extract_CBD | RandomForest | 68.2%             |            0.019 |       0.79465  |         0.292847 | Moderate (60-75%)      |
| UA44     | ETANOL            | RandomForest | 68.2%             |            0.015 |       0.762269 |         0.517672 | Moderate (60-75%)      |
| N2       | CBDV 30uM         | RandomForest | 66.6%             |            0.02  |       0.792141 |         0.148024 | Moderate (60-75%)      |
| N2       | CBDV 0.3uM        | RandomForest | 66.1%             |            0.022 |       0.783887 |         0.206847 | Moderate (60-75%)      |
| N2       | CBDV 3uM          | RandomForest | 63.2%             |            0.025 |       0.758515 |         0.225314 | Moderate (60-75%)      |
| BR5270   | ETANOL            | RandomForest | 60.6%             |            0.042 |       0.71615  |         0.354213 | Moderate (60-75%)      |
| BR5271   | Total_Extract_CBD | RandomForest | 59.3%             |            0.031 |       0.646431 |         0.518337 | Poor (<60%)            |
| N2       | Total_Extract_CBD | RandomForest | 59.1%             |            0.016 |       0.688086 |         0.406361 | Poor (<60%)            |
| TJ356    | ETANOL            | RandomForest | 59.0%             |            0.025 |       0.501484 |         0.651567 | Poor (<60%)            |
| NL5901   | CBDV 3uM          | RandomForest | 58.6%             |            0.034 |       0.711328 |         0.261004 | Poor (<60%)            |
| N2       | ETANOL            | RandomForest | 58.4%             |            0.023 |       0.560457 |         0.604699 | Poor (<60%)            |
| BR5271   | CBD 0.3uM         | RandomForest | 58.2%             |            0.024 |       0.630034 |         0.517765 | Poor (<60%)            |
| BR5270   | Total_Extract_CBD | RandomForest | 57.4%             |            0.042 |       0.601162 |         0.539662 | Poor (<60%)            |
| NL5901   | CBDV 0.3uM        | RandomForest | 56.5%             |            0.069 |       0.671378 |         0.321899 | Poor (<60%)            |
| NL5901   | CBD 0.3uM         | RandomForest | 56.1%             |            0.023 |       0.465269 |         0.626665 | Poor (<60%)            |
| TJ356    | CBD 0.3uM         | RandomForest | 55.8%             |            0.022 |       0.440974 |         0.633674 | Poor (<60%)            |
| N2       | CBD 0.3uM         | RandomForest | 55.7%             |            0.015 |       0.627348 |         0.453994 | Poor (<60%)            |
| BR5270   | CBD 0.3uM         | RandomForest | 53.3%             |            0.053 |       0.506485 |         0.555405 | Poor (<60%)            |
| TJ356    | Total_Extract_CBD | RandomForest | 52.8%             |            0.014 |       0.461359 |         0.579506 | Poor (<60%)            |
| NL5901   | ETANOL            | RandomForest | 52.2%             |            0.068 |       0.60088  |         0.354397 | Poor (<60%)            |

### Performance Summary
- **Total strain/treatment combinations:** 40
- **Excellent (>90%):** 2 cases (5.0%)
- **Good (75-90%):** 5 cases (12.5%)
- **Moderate (60-75%):** 19 cases (47.5%)
- **Poor (<60%):** 14 cases (35.0%)

### Classifier Comparison Analysis
Comparing performance across different classifiers (García-Garví et al. 2025 recommendation):

**Overall Classifier Performance:**

| classifier         | Avg Accuracy   |   Std Dev | Best   | Worst   |
|:-------------------|:---------------|----------:|:-------|:--------|
| LogisticRegression | 57.7%          |     0.09  | 72.5%  | 27.9%   |
| RandomForest       | 67.2%          |     0.101 | 95.2%  | 52.2%   |
| XGBoost            | 76.2%          |     0.123 | 95.1%  | 53.2%   |

**Best Classifier per Strain/Treatment:**

- BR5270 + CBD 0.3uM: **RandomForest** (53.3%)
- BR5270 + ETANOL: **RandomForest** (60.6%)
- BR5270 + Total Extract CBD: **XGBoost** (64.4%)
- BR5271 + CBD 0.3uM: **XGBoost** (69.9%)
- BR5271 + ETANOL: **RandomForest** (72.2%)
- BR5271 + Total Extract CBD: **XGBoost** (68.5%)
- N2 + CBD 0.3uM: **XGBoost** (65.1%)
- N2 + CBD 30uM: **XGBoost** (79.5%)
- N2 + CBD 3uM: **XGBoost** (77.1%)
- N2 + CBDV 0.3uM: **XGBoost** (90.9%)
- N2 + CBDV 30uM: **XGBoost** (93.4%)
- N2 + CBDV 3uM: **XGBoost** (89.5%)
- N2 + ETANOL: **RandomForest** (58.4%)
- N2 + Total Extract CBD: **XGBoost** (73.7%)
- NL5901 + CBD: **XGBoost** (81.8%)
- NL5901 + CBD 0.3uM: **XGBoost** (64.3%)
- NL5901 + CBD 30uM: **XGBoost** (81.6%)
- NL5901 + CBD 3uM: **RandomForest** (95.2%)
- NL5901 + CBDV 0.3uM: **XGBoost** (73.6%)
- NL5901 + CBDV 30uM: **XGBoost** (84.5%)
- NL5901 + CBDV 3uM: **XGBoost** (82.3%)
- NL5901 + ETANOL: **XGBoost** (60.1%)
- ROSELLA + CBD 0.3uM: **XGBoost** (81.9%)
- ROSELLA + CBD 30uM: **XGBoost** (82.1%)
- ROSELLA + CBD 3uM: **XGBoost** (77.0%)
- ROSELLA + CBDV 0.3uM: **XGBoost** (76.8%)
- ROSELLA + CBDV 30uM: **XGBoost** (86.6%)
- ROSELLA + CBDV 3uM: **XGBoost** (73.2%)
- ROSELLA + ETANOL: **XGBoost** (70.9%)
- TJ356 + CBD 0.3uM: **RandomForest** (55.8%)
- TJ356 + ETANOL: **RandomForest** (59.0%)
- TJ356 + Total Extract CBD: **LogisticRegression** (53.7%)
- UA44 + CBD 0.3uM: **XGBoost** (77.7%)
- UA44 + CBD 30uM: **XGBoost** (89.7%)
- UA44 + CBD 3uM: **XGBoost** (93.0%)
- UA44 + CBDV 0.3uM: **XGBoost** (87.4%)
- UA44 + CBDV 30uM: **XGBoost** (90.7%)
- UA44 + CBDV 3uM: **XGBoost** (94.7%)
- UA44 + ETANOL: **XGBoost** (75.7%)
- UA44 + Total Extract CBD: **XGBoost** (85.7%)

### Performance by Strain

**NL5901:**
- Tests: 8
- Average accuracy: 69.0%
- Best result: 95.2% (CBD 3uM, RandomForest)
- Poor (<60%): 4
- Good (75-90%): 2
- Excellent (>90%): 1
- Moderate (60-75%): 1

**UA44:**
- Tests: 8
- Average accuracy: 73.3%
- Best result: 92.1% (CBDV 3uM, RandomForest)
- Moderate (60-75%): 7
- Excellent (>90%): 1

**N2:**
- Tests: 8
- Average accuracy: 64.7%
- Best result: 76.5% (CBD 30uM, RandomForest)
- Moderate (60-75%): 4
- Poor (<60%): 3
- Good (75-90%): 1

**ROSELLA:**
- Tests: 7
- Average accuracy: 71.9%
- Best result: 75.5% (CBD 3uM, RandomForest)
- Moderate (60-75%): 5
- Good (75-90%): 2

**BR5271:**
- Tests: 3
- Average accuracy: 63.2%
- Best result: 72.2% (ETANOL, RandomForest)
- Poor (<60%): 2
- Moderate (60-75%): 1

**BR5270:**
- Tests: 3
- Average accuracy: 57.1%
- Best result: 60.6% (ETANOL, RandomForest)
- Poor (<60%): 2
- Moderate (60-75%): 1

**TJ356:**
- Tests: 3
- Average accuracy: 55.9%
- Best result: 59.0% (ETANOL, RandomForest)
- Poor (<60%): 3

### Treatment Effectiveness (Ranked by Average Performance)

| treatment         |   Tests | Avg Accuracy   | Best Accuracy   |   Std Dev |
|:------------------|--------:|:---------------|:----------------|----------:|
| CBD 3uM           |       4 | 79.2%          | 95.2%           |     0.108 |
| CBD 30uM          |       4 | 73.6%          | 76.5%           |     0.026 |
| CBDV 30uM         |       4 | 72.9%          | 83.7%           |     0.075 |
| CBDV 3uM          |       4 | 70.9%          | 92.1%           |     0.149 |
| CBDV 0.3uM        |       4 | 65.4%          | 70.5%           |     0.062 |
| ETANOL            |       7 | 62.8%          | 72.2%           |     0.072 |
| CBD 0.3uM         |       7 | 60.5%          | 75.4%           |     0.083 |
| Total_Extract_CBD |       5 | 59.4%          | 68.2%           |     0.056 |

### Exceptional Performers (>85% Accuracy)

These strain/treatment combinations show exceptionally strong and distinctive behavioral phenotypes:

- **NL5901 + CBD 3uM:** [RandomForest] 95.2% ± 0.003 accuracy (control-dominant)
- **UA44 + CBDV 3uM:** [RandomForest] 92.1% ± 0.008 accuracy (control-dominant)

### Analysis of Poor Classification Performance (<55%)

These combinations show limited phenotypic distinctiveness, suggesting:
1. Subtle treatment effects below current detection threshold
2. High individual variability masking treatment effects
3. Genetic background resistance to the treatment
4. Treatment concentrations outside effective range

**Strains with most poor classifications:**
- BR5270: 1/3 treatments (33.3%)
- TJ356: 1/3 treatments (33.3%)
- NL5901: 1/8 treatments (12.5%)

### Treatment Consistency Analysis

**Most consistent treatments across strains:**

| Treatment         |   Strains | Avg Accuracy   |   Std Dev | Range       |   Consistency Score |
|:------------------|----------:|:---------------|----------:|:------------|--------------------:|
| CBD 30uM          |         4 | 73.6%          |     0.026 | 70.2%-76.5% |               0.71  |
| CBD 3uM           |         4 | 79.2%          |     0.108 | 71.8%-95.2% |               0.683 |
| CBDV 30uM         |         4 | 72.9%          |     0.075 | 66.6%-83.7% |               0.655 |
| CBDV 0.3uM        |         4 | 65.4%          |     0.062 | 56.5%-70.5% |               0.592 |
| CBDV 3uM          |         4 | 70.9%          |     0.149 | 58.6%-92.1% |               0.56  |
| ETANOL            |         7 | 62.8%          |     0.072 | 52.2%-72.2% |               0.557 |
| Total_Extract_CBD |         5 | 59.4%          |     0.056 | 52.8%-68.2% |               0.538 |
| CBD 0.3uM         |         7 | 60.5%          |     0.083 | 53.3%-75.4% |               0.522 |

### Classification Pattern Analysis

**Control-dominant classifications:** 30 cases
These suggest treatments create subtle changes, but control phenotype remains more distinctive.

**Treatment-dominant classifications:** 0 cases
These suggest treatments create highly distinctive, consistent phenotypes.


### Feature Importance Analysis

Feature importance files not found. Run the analysis to generate them.
Top discriminative features typically include:
- Speed coefficient of variation (CV)
- Path tortuosity
- Reversal frequency
- Angular velocity statistics

### Visualizations Generated

The following visualizations are available in each strain's output directory:
- `confusion_matrix_*.png`: Classification confusion matrices per treatment
- `feature_importance_*.png`: Feature importance rankings
- `roc_curve_*.png`: ROC curves with AUC scores
- `classifier_comparison_*.png`: Performance comparison across classifiers
- `permutation_test_*.png`: Permutation test results for statistical validation
