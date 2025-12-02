## Comprehensive TDA (Topological Data Analysis) Analysis

This analysis shows topological feature-based SVM classification results for ALL strain/treatment combinations.

**Methodology (Based on Thomas et al. 2021):**
- **Sliding Window Embedding:** Transforms time series trajectories into point clouds
- **Persistent Homology:** Computes topological features:
  - H0: Connected components (overall trajectory structure)
  - H1: Loops/cycles (repetitive movement patterns)
- **Persistence Landscapes:** 5000 coefficients (5 landscapes × 1000 resolution)
- **SVM Classification:** Distinguishes treatments based on topological signatures

**TDA Performance Categories:**
- **Excellent (>80%):** Very strong topological signature, highly distinguishable trajectory shape
- **Good (65-80%):** Clear topological phenotype, reliably distinguishable global trajectory patterns
- **Moderate (55-65%):** Detectable topological changes, some classification ability
- **Poor (<55%):** Weak/no topological phenotype, trajectory shapes similar to control

### Complete TDA Classification Results

| strain   | treatment         | Combined Acc.   |   Acc. (std) |   F1 (Control) |   F1 (Treatment) | TDA Performance Category   |
|:---------|:------------------|:----------------|-------------:|---------------:|-----------------:|:---------------------------|
| UA44     | CBD 30uM          | 87.0%           |        0.015 |       0.927459 |        0.381008  | Excellent (>80%)           |
| UA44     | CBD 3uM           | 86.7%           |        0.018 |       0.924677 |        0.430791  | Excellent (>80%)           |
| ROSELLA  | CBD 0.3uM         | 85.0%           |        0.042 |       0.913803 |        0.404104  | Excellent (>80%)           |
| ROSELLA  | CBDV 30uM         | 84.1%           |        0.032 |       0.912415 |        0.102861  | Excellent (>80%)           |
| ROSELLA  | CBD 30uM          | 82.5%           |        0.024 |       0.893191 |        0.50328   | Excellent (>80%)           |
| ROSELLA  | CBD 3uM           | 82.0%           |        0.049 |       0.890122 |        0.496936  | Excellent (>80%)           |
| UA44     | CBDV 3uM          | 81.8%           |        0.088 |       0.89566  |        0.0648185 | Excellent (>80%)           |
| UA44     | Total_Extract_CBD | 74.9%           |        0.037 |       0.846889 |        0.301654  | Good (65-80%)              |
| N2       | CBDV 30uM         | 74.2%           |        0.084 |       0.846717 |        0.104087  | Good (65-80%)              |
| UA44     | CBD 0.3uM         | 73.3%           |        0.024 |       0.823401 |        0.448609  | Good (65-80%)              |
| NL5901   | CBD 3uM           | 72.6%           |        0.131 |       0.829861 |        0.128987  | Good (65-80%)              |
| ROSELLA  | CBDV 0.3uM        | 71.2%           |        0.032 |       0.81589  |        0.322809  | Good (65-80%)              |
| N2       | CBDV 3uM          | 71.0%           |        0.05  |       0.821185 |        0.205697  | Good (65-80%)              |
| ROSELLA  | CBDV 3uM          | 70.5%           |        0.032 |       0.80858  |        0.349605  | Good (65-80%)              |
| N2       | CBD 30uM          | 69.8%           |        0.03  |       0.811526 |        0.22936   | Good (65-80%)              |
| UA44     | ETANOL            | 69.8%           |        0.029 |       0.789098 |        0.462953  | Good (65-80%)              |
| NL5901   | CBDV 3uM          | 69.1%           |        0.063 |       0.808086 |        0.181448  | Good (65-80%)              |
| ROSELLA  | ETANOL            | 67.6%           |        0.053 |       0.771095 |        0.411361  | Good (65-80%)              |
| BR5271   | CBD 0.3uM         | 66.6%           |        0.086 |       0.735328 |        0.542694  | Good (65-80%)              |
| NL5901   | CBD 0.3uM         | 66.1%           |        0.024 |       0.237504 |        0.78186   | Good (65-80%)              |
| BR5271   | ETANOL            | 64.8%           |        0.074 |       0.756142 |        0.359584  | Moderate (55-65%)          |
| BR5270   | Total_Extract_CBD | 63.7%           |        0.063 |       0.743214 |        0.368829  | Moderate (55-65%)          |
| BR5271   | Total_Extract_CBD | 63.2%           |        0.094 |       0.695102 |        0.531207  | Moderate (55-65%)          |
| N2       | Total_Extract_CBD | 62.4%           |        0.094 |       0.736303 |        0.279106  | Moderate (55-65%)          |
| TJ356    | CBD 0.3uM         | 61.0%           |        0.058 |       0.562441 |        0.643358  | Moderate (55-65%)          |
| N2       | CBDV 0.3uM        | 60.9%           |        0.11  |       0.740573 |        0.120554  | Moderate (55-65%)          |
| N2       | ETANOL            | 59.3%           |        0.021 |       0.681761 |        0.431964  | Moderate (55-65%)          |
| TJ356    | ETANOL            | 56.6%           |        0.049 |       0.539698 |        0.583503  | Moderate (55-65%)          |
| TJ356    | Total_Extract_CBD | 55.2%           |        0.074 |       0.510501 |        0.584907  | Moderate (55-65%)          |
| NL5901   | CBDV 30uM         | 54.3%           |        0.058 |       0.650396 |        0.32796   | Poor (<55%)                |
| BR5270   | ETANOL            | 52.1%           |        0.116 |       0.438516 |        0.575359  | Poor (<55%)                |
| NL5901   | CBD 30uM          | 51.0%           |        0.063 |       0.628767 |        0.261147  | Poor (<55%)                |
| NL5901   | ETANOL            | 50.2%           |        0.068 |       0.441282 |        0.545672  | Poor (<55%)                |
| BR5270   | CBD 0.3uM         | 48.6%           |        0.073 |       0.364643 |        0.55782   | Poor (<55%)                |
| N2       | CBD 0.3uM         | 46.3%           |        0.015 |       0.473089 |        0.451285  | Poor (<55%)                |
| N2       | CBD 3uM           | 46.2%           |        0.033 |       0.514417 |        0.394884  | Poor (<55%)                |
| UA44     | CBDV 0.3uM        | 45.4%           |        0.046 |       0.57924  |        0.214192  | Poor (<55%)                |
| NL5901   | CBDV 0.3uM        | 42.4%           |        0.095 |       0.495757 |        0.268735  | Poor (<55%)                |
| UA44     | CBDV 30uM         | 42.2%           |        0.052 |       0.563644 |        0.134039  | Poor (<55%)                |

### TDA Performance Summary
- **Total strain/treatment combinations:** 39
- **Excellent (>80%):** 7 cases (17.9%)
- **Good (65-80%):** 13 cases (33.3%)
- **Moderate (55-65%):** 9 cases (23.1%)
- **Poor (<55%):** 10 cases (25.6%)
- **Above random chance (>50%):** 33/39 (84.6%)

### TDA Performance by Strain

**UA44:**
- Tests: 8
- Average TDA accuracy: 70.1%
- Best result: 87.0% (CBD 30uM)
- Above random: 6/8
- Excellent (>80%): 3
- Good (65-80%): 3
- Poor (<55%): 2
- Window length used (L): 20

**ROSELLA:**
- Tests: 7
- Average TDA accuracy: 77.5%
- Best result: 85.0% (CBD 0.3uM)
- Above random: 7/7
- Excellent (>80%): 4
- Good (65-80%): 3
- Window length used (L): 10

**N2:**
- Tests: 8
- Average TDA accuracy: 61.3%
- Best result: 74.2% (CBDV 30uM)
- Above random: 6/8
- Good (65-80%): 3
- Moderate (55-65%): 3
- Poor (<55%): 2
- Window length used (L): 10

**NL5901:**
- Tests: 7
- Average TDA accuracy: 58.0%
- Best result: 72.6% (CBD 3uM)
- Above random: 6/7
- Poor (<55%): 4
- Good (65-80%): 3
- Window length used (L): 50

**BR5271:**
- Tests: 3
- Average TDA accuracy: 64.8%
- Best result: 66.6% (CBD 0.3uM)
- Above random: 3/3
- Moderate (55-65%): 2
- Good (65-80%): 1
- Window length used (L): 50

**BR5270:**
- Tests: 3
- Average TDA accuracy: 54.8%
- Best result: 63.7% (Total_Extract_CBD)
- Above random: 2/3
- Poor (<55%): 2
- Moderate (55-65%): 1
- Window length used (L): 50

**TJ356:**
- Tests: 3
- Average TDA accuracy: 57.6%
- Best result: 61.0% (CBD 0.3uM)
- Above random: 3/3
- Moderate (55-65%): 3
- Window length used (L): 1

### Window Length Selection Across All Strains
- **L=1:** 3 tests (7.7%)
- **L=10:** 15 tests (38.5%)
- **L=20:** 8 tests (20.5%)
- **L=50:** 13 tests (33.3%)

### Treatment Effectiveness (Ranked by Average TDA Performance)

| treatment         |   Tests | Avg Combined   | Best Combined   |   Std Dev |
|:------------------|--------:|:---------------|:----------------|----------:|
| CBDV 3uM          |       4 | 73.1%          | 81.8%           |     0.059 |
| CBD 30uM          |       4 | 72.6%          | 87.0%           |     0.161 |
| CBD 3uM           |       4 | 71.9%          | 86.7%           |     0.181 |
| Total_Extract_CBD |       5 | 63.9%          | 74.9%           |     0.071 |
| CBD 0.3uM         |       7 | 63.8%          | 85.0%           |     0.135 |
| CBDV 30uM         |       4 | 63.7%          | 84.1%           |     0.189 |
| ETANOL            |       7 | 60.0%          | 69.8%           |     0.076 |
| CBDV 0.3uM        |       4 | 55.0%          | 71.2%           |     0.135 |

### Treatment Consistency Across Strains (Multi-strain treatments only)

| Treatment         |   Strains | Avg Accuracy   | Min Accuracy   |   Std Dev |   Consistency Score |
|:------------------|----------:|:---------------|:---------------|----------:|--------------------:|
| CBDV 3uM          |         4 | 73.1%          | 69.1%          |     0.059 |               0.673 |
| Total_Extract_CBD |         5 | 63.9%          | 55.2%          |     0.071 |               0.568 |
| CBD 30uM          |         4 | 72.6%          | 51.0%          |     0.161 |               0.564 |
| CBD 3uM           |         4 | 71.9%          | 46.2%          |     0.181 |               0.538 |
| ETANOL            |         7 | 60.0%          | 50.2%          |     0.076 |               0.524 |
| CBD 0.3uM         |         7 | 63.8%          | 46.3%          |     0.135 |               0.503 |
| CBDV 30uM         |         4 | 63.7%          | 42.2%          |     0.189 |               0.448 |
| CBDV 0.3uM        |         4 | 55.0%          | 42.4%          |     0.135 |               0.414 |

### Topological Signature Analysis
The following 20 strain/treatment combinations show strong topological signatures (>65% accuracy):
This suggests these treatments induce distinctive changes in the geometric/spatial patterns of worm trajectories.

- **UA44 + CBD 30uM:** 87.0% ± 0.015 (Strong topological phenotype)
- **UA44 + CBD 3uM:** 86.7% ± 0.018 (Strong topological phenotype)
- **ROSELLA + CBD 0.3uM:** 85.0% ± 0.042 (Strong topological phenotype)
- **ROSELLA + CBDV 30uM:** 84.1% ± 0.032 (Strong topological phenotype)
- **ROSELLA + CBD 30uM:** 82.5% ± 0.024 (Strong topological phenotype)
- **ROSELLA + CBD 3uM:** 82.0% ± 0.049 (Strong topological phenotype)
- **UA44 + CBDV 3uM:** 81.8% ± 0.088 (Strong topological phenotype)
- **UA44 + Total_Extract_CBD:** 74.9% ± 0.037 (Strong topological phenotype)
- **N2 + CBDV 30uM:** 74.2% ± 0.084 (Strong topological phenotype)
- **UA44 + CBD 0.3uM:** 73.3% ± 0.024 (Strong topological phenotype)
- **NL5901 + CBD 3uM:** 72.6% ± 0.131 (Strong topological phenotype)
- **ROSELLA + CBDV 0.3uM:** 71.2% ± 0.032 (Strong topological phenotype)
- **N2 + CBDV 3uM:** 71.0% ± 0.050 (Strong topological phenotype)
- **ROSELLA + CBDV 3uM:** 70.5% ± 0.032 (Strong topological phenotype)
- **N2 + CBD 30uM:** 69.8% ± 0.030 (Strong topological phenotype)
- **UA44 + ETANOL:** 69.8% ± 0.029 (Strong topological phenotype)
- **NL5901 + CBDV 3uM:** 69.1% ± 0.063 (Strong topological phenotype)
- **ROSELLA + ETANOL:** 67.6% ± 0.053 (Strong topological phenotype)
- **BR5271 + CBD 0.3uM:** 66.6% ± 0.086 (Strong topological phenotype)
- **NL5901 + CBD 0.3uM:** 66.1% ± 0.024 (Strong topological phenotype)

### Visualizations Generated

The following visualizations are available in each strain's output directory:
- `persistence_diagram_*.png`: Persistence diagrams showing H0 and H1 features
- `mds_plot_H1_L*.png`: MDS visualization of H1 persistence landscapes
- `mds_plot_H0_L*.png`: MDS visualization of H0 persistence landscapes
- `distance_heatmap_L*.png`: Heatmap of pairwise distances between treatments
- `confusion_matrix_*.png`: Classification confusion matrices
