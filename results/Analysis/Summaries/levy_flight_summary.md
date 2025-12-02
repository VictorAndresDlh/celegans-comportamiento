## Comprehensive Lévy Flight Analysis

This analysis shows power-law vs lognormal distribution fitting results for ALL strain/treatment combinations.

**Interpretation (Based on Moy et al. 2015):**
- **Statistical Criterion:** R > 0 & p < 0.05 (power-law fits better than lognormal)
- **Strict Lévy Criterion:** R > 0 & p < 0.05 & **1 < α ≤ 3** (within theoretical Lévy range)
- **α (alpha):** Power-law exponent
  - α ≈ 2: Optimal foraging search strategy
  - 1 < α ≤ 3: Lévy flight range
  - α > 3: Transition to Brownian motion

### Complete Results Table

| strain   | treatment         |   α (exponent) |   α CI Lower |   α CI Upper |     p-value |   R (Log-Likelihood) |   % Worms Statistical |   % Worms Strict | Search Pattern          |
|:---------|:------------------|---------------:|-------------:|-------------:|------------:|---------------------:|----------------------:|-----------------:|:------------------------|
| NL5901   | CBD 0.3uM         |        2.8764  |      2.85293 |      9.65769 | 7.47811e-18 |         -85.6528     |              2.62431  |         13.674   | Brownian                |
| UA44     | ETANOL            |        2.48684 |      2.37497 |      5.02755 | 2.67249e-12 |         -62.9745     |              4.01606  |         14.0562  | Brownian                |
| BR5271   | Control           |        2.45047 |      2.30988 |      4.18266 | 1.96343e-09 |         -46.9057     |              0.990099 |         30.198   | Brownian                |
| NL5901   | Control           |        2.73517 |      2.66747 |      7.32331 | 4.35477e-09 |         -31.6519     |              2.82258  |         15.3226  | Brownian                |
| NL5901   | ETANOL            |        2.59931 |      2.56256 |      5.88947 | 4.83451e-08 |         -37.2616     |              4.24242  |         15.1515  | Brownian                |
| N2       | Total_Extract_CBD |        2.85081 |      2.62519 |     13.1738  | 1.8014e-05  |         -22.5623     |              2.45399  |         14.7239  | Brownian                |
| N2       | CBD 30uM          |        2.58334 |      2.55105 |      4.44456 | 1.87842e-05 |         -23.6036     |              1.12994  |         31.0734  | Brownian                |
| BR5270   | Control           |        2.40676 |      2.36795 |      5.95239 | 4.40252e-05 |         -20.6267     |              5.42636  |         16.2791  | Brownian                |
| TJ356    | Total_Extract_CBD |        2.94068 |      2.87503 |      3.35476 | 0.000139507 |         -14.9687     |              2.48756  |         16.4179  | Brownian                |
| ROSELLA  | ETANOL            |        2.51435 |      2.44001 |      7.05429 | 0.000450202 |         -13.039      |              1.92308  |          7.69231 | Brownian                |
| UA44     | CBD 3uM           |        3.25124 |      3.10508 |      3.38878 | 0.00243632  |          -7.35979    |              1.75439  |          1.75439 | Brownian                |
| TJ356    | ETANOL            |        4.18544 |      2.55116 |      4.44128 | 0.00336293  |          -9.0321     |              3.06122  |          8.67347 | Brownian                |
| TJ356    | Control           |        2.75649 |      2.6603  |      2.79797 | 0.0218846   |          -4.93955    |              1.83486  |         14.6789  | Brownian                |
| NL5901   | CBDV 30uM         |        3.0943  |      2.99183 |      4.32477 | 0.0418747   |          -5.92396    |              1.36986  |         31.5068  | Brownian                |
| BR5271   | Total_Extract_CBD |        2.53121 |      2.49346 |      2.64661 | 0.0506536   |          -4.02525    |              6.25     |          9.72222 | Brownian                |
| N2       | CBD 3uM           |        3.62165 |      3.48482 |      5.18956 | 0.0526161   |          -5.35216    |              2.21239  |         28.3186  | Brownian                |
| N2       | CBDV 0.3uM        |        2.95873 |      2.48635 |      3.10991 | 0.056193    |          -3.40002    |              1.85185  |          3.7037  | Brownian                |
| ROSELLA  | CBDV 0.3uM        |        3.0139  |      2.91758 |      3.32891 | 0.0641059   |          -3.52919    |              2.77778  |         11.1111  | Brownian                |
| NL5901   | CBD 30uM          |        2.55108 |      2.46822 |      2.64563 | 0.0752338   |          -3.561      |              0        |         23.4043  | Brownian                |
| UA44     | CBD 30uM          |        2.21174 |      2.17215 |      9.39472 | 0.0836146   |          -1.09418    |              4.34783  |          6.52174 | Brownian                |
| NL5901   | CBDV 0.3uM        |        3.00344 |      2.83675 |      3.14057 | 0.0986129   |          -2.49934    |              3.65854  |         18.2927  | Brownian                |
| N2       | CBD 0.3uM         |        4.25725 |      2.71447 |      4.50569 | 0.100929    |          -0.214153   |              1.02041  |         23.4694  | Brownian                |
| NL5901   | nan               |        3.58057 |      3.34573 |      6.82238 | 0.114934    |          -2.99493    |              2.63158  |         15.7895  | Brownian                |
| TJ356    | CBD 0.3uM         |        2.94096 |      2.84504 |      3.0097  | 0.114942    |          -2.04389    |              2.7907   |         10.6977  | Brownian                |
| NL5901   | CBDV 3uM          |        2.76974 |      2.54692 |      2.98596 | 0.135658    |          -1.84316    |              0        |         16.129   | Brownian                |
| BR5271   | CBD 0.3uM         |        2.88414 |      2.6787  |      5.74773 | 0.140671    |          -0.281996   |              5.82524  |          5.82524 | Brownian                |
| BR5271   | ETANOL            |        2.84226 |      2.76039 |      3.06511 | 0.151217    |          -2.79169    |              0        |         33.0435  | Brownian                |
| N2       | CBDV 3uM          |        3.07131 |      2.94921 |      3.24401 | 0.158705    |           0.0253431  |              4.25532  |          7.44681 | Brownian                |
| BR5270   | CBD 0.3uM         |        2.61786 |      2.54786 |      2.89644 | 0.165631    |          -2.36944    |             10        |          6.25    | Brownian                |
| ROSELLA  | CBDV 30uM         |        3.52313 |      3.1593  |      4.33217 | 0.261379    |          -1.93269    |              0        |         25       | Brownian                |
| UA44     | CBD 0.3uM         |        7.37341 |      2.31048 |      8.86559 | 0.268098    |          -0.752071   |              1.95122  |         19.0244  | Brownian                |
| BR5270   | Total_Extract_CBD |        3.88177 |      3.74751 |      4.54604 | 0.406056    |          -0.926657   |              0        |         12.8205  | Brownian                |
| N2       | ETANOL            |        7.17831 |      3.03245 |      8.40526 | 0.409655    |          -0.0706883  |              2.5729   |         15.952   | Brownian                |
| ROSELLA  | CBD 0.3uM         |        3.22318 |      3.07206 |      3.40678 | 0.461727    |          -0.437834   |              5.55556  |          5.55556 | Brownian                |
| UA44     | Total_Extract_CBD |        3.44466 |      3.23467 |      3.85734 | 0.490873    |          -0.612778   |              1.02041  |         22.449   | Brownian                |
| UA44     | CBDV 3uM          |        2.94833 |      2.58206 |      3.21492 | 0.552079    |          -0.375139   |              0        |         40       | Brownian                |
| UA44     | Control           |        4.2357  |      3.97285 |      5.28484 | 0.575265    |          -0.377022   |              0.456621 |         41.0959  | Brownian                |
| UA44     | CBDV 30uM         |        3.85678 |      3.46091 |      4.21554 | 0.590494    |          -0.375682   |              0        |         32.2581  | Brownian                |
| N2       | Control           |        4.31752 |      2.79013 |      5.40699 | 0.610492    |          -0.283578   |              1.9337   |         25.5525  | Brownian                |
| ROSELLA  | CBDV 3uM          |        2.31166 |      2.25783 |      2.35249 | 0.622061    |          -0.262458   |              3.77358  |         21.6981  | Brownian                |
| BR5270   | ETANOL            |        2.47074 |      2.42708 |      2.53588 | 0.686678    |          -0.214855   |              3.26087  |          7.6087  | Brownian                |
| N2       | CBDV 30uM         |        2.78611 |      2.66436 |      2.95856 | 0.727039    |          -0.100904   |              3.84615  |         26.9231  | Brownian                |
| ROSELLA  | Control           |        4.24508 |      2.91656 |      4.57408 | 0.73983     |           0.00962306 |              0.993377 |         31.457   | Brownian                |
| NL5901   | CBD 3uM           |        3.54598 |      2.68692 |      4.48126 | 0.832525    |          -0.038011   |              0        |          8.33333 | Brownian                |
| UA44     | CBDV 0.3uM        |        4.52423 |      2.6545  |      5.03037 | 0.883397    |          -0.020837   |              0        |         19.5122  | Brownian                |
| ROSELLA  | CBD 3uM           |        3.1764  |      2.46948 |      3.23738 | 1.99832e-17 |           1.31016    |              3.48837  |          1.16279 | Lévy (Statistical only) |
| ROSELLA  | CBD 30uM          |        3.04382 |      2.96426 |      3.13021 | 3.58905e-06 |           0.874569   |              2.1978   |          4.3956  | Lévy (Statistical only) |

### Summary Statistics
- **Total strain/treatment combinations tested:** 47
- **Lévy patterns (Statistical: R>0, p<0.05):** 2 (4.3%)
- **Lévy patterns (Strict: 1 < α ≤ 3):** 0 (0.0%)
- **Standard Brownian motion:** 45 (95.7%)

### Alpha Value Analysis (For Statistical Lévy Cases)
- **α in Lévy range (1 < α ≤ 3):** 0
- **α in transition zone (α > 3):** 2

⚠️ **Note:** 2 case(s) with α > 3 show power-law distribution but are in the
   transition zone between Lévy and Brownian motion (Moy et al., 2015).

### Results by Strain

| strain   |   Lévy (Statistical) |   Lévy (Strict) |   α Mean |   α Std |   Total |   % Lévy Strict |
|:---------|---------------------:|----------------:|---------:|--------:|--------:|----------------:|
| BR5270   |                    0 |               0 |    2.844 |   0.697 |       4 |               0 |
| BR5271   |                    0 |               0 |    2.677 |   0.218 |       4 |               0 |
| N2       |                    0 |               0 |    3.736 |   1.438 |       9 |               0 |
| NL5901   |                    0 |               0 |    2.973 |   0.377 |       9 |               0 |
| ROSELLA  |                    2 |               0 |    3.131 |   0.595 |       8 |               0 |
| TJ356    |                    0 |               0 |    3.206 |   0.659 |       4 |               0 |
| UA44     |                    0 |               0 |    3.815 |   1.536 |       9 |               0 |

### Per-Worm Alpha Distribution Analysis

Individual worm statistics provide more robust insights than pooled data (Moy et al., 2015):

**Per-Worm Alpha Statistics:**

- **BR5270 + CBD 0.3uM:** n=80, α=52.81±206.32 (median=14.14), 6.2% in Lévy range
- **BR5270 + Control:** n=129, α=34.21±77.40 (median=7.10), 16.3% in Lévy range
- **BR5270 + ETANOL:** n=92, α=31.82±60.37 (median=11.58), 7.6% in Lévy range
- **BR5270 + Total_Extract_CBD:** n=78, α=28.16±101.90 (median=4.69), 12.8% in Lévy range
- **BR5271 + CBD 0.3uM:** n=103, α=26.75±40.69 (median=11.99), 5.8% in Lévy range
- **BR5271 + Control:** n=202, α=14.01±45.12 (median=4.10), 30.2% in Lévy range
- **BR5271 + ETANOL:** n=115, α=4.48±3.02 (median=3.71), 33.0% in Lévy range
- **BR5271 + Total_Extract_CBD:** n=144, α=40.79±113.09 (median=10.32), 9.7% in Lévy range
- **N2 + CBD 0.3uM:** n=294, α=18.10±80.73 (median=4.15), 23.5% in Lévy range
- **N2 + CBD 30uM:** n=177, α=7.92±17.35 (median=3.58), 31.1% in Lévy range
- **N2 + CBD 3uM:** n=226, α=6.59±9.98 (median=3.74), 28.3% in Lévy range
- **N2 + CBDV 0.3uM:** n=54, α=28.61±46.93 (median=9.16), 3.7% in Lévy range
- **N2 + CBDV 30uM:** n=52, α=24.96±73.16 (median=4.70), 26.9% in Lévy range
- **N2 + CBDV 3uM:** n=94, α=18.43±33.13 (median=6.70), 7.4% in Lévy range
- **N2 + Control:** n=724, α=10.51±39.13 (median=3.93), 25.6% in Lévy range
- **N2 + ETANOL:** n=583, α=23.50±60.01 (median=6.36), 16.0% in Lévy range
- **N2 + Total_Extract_CBD:** n=163, α=30.11±133.14 (median=7.41), 14.7% in Lévy range
- **NL5901 + CBD:** n=38, α=15.32±26.41 (median=5.69), 15.8% in Lévy range
- **NL5901 + CBD 0.3uM:** n=724, α=34.04±133.40 (median=7.33), 13.7% in Lévy range
- **NL5901 + CBD 30uM:** n=47, α=53.49±278.39 (median=4.14), 23.4% in Lévy range
- **NL5901 + CBD 3uM:** n=24, α=17.43±30.72 (median=6.56), 8.3% in Lévy range
- **NL5901 + CBDV 0.3uM:** n=82, α=17.48±34.61 (median=4.81), 18.3% in Lévy range
- **NL5901 + CBDV 30uM:** n=73, α=21.03±109.19 (median=3.58), 31.5% in Lévy range
- **NL5901 + CBDV 3uM:** n=62, α=42.38±231.92 (median=4.95), 16.1% in Lévy range
- **NL5901 + Control:** n=248, α=21.95±72.35 (median=5.46), 15.3% in Lévy range
- **NL5901 + ETANOL:** n=165, α=33.89±107.35 (median=6.88), 15.2% in Lévy range
- **ROSELLA + CBD 0.3uM:** n=54, α=33.74±59.32 (median=16.79), 5.6% in Lévy range
- **ROSELLA + CBD 30uM:** n=91, α=41.62±59.66 (median=18.59), 4.4% in Lévy range
- **ROSELLA + CBD 3uM:** n=86, α=21.14±23.40 (median=14.17), 1.2% in Lévy range
- **ROSELLA + CBDV 0.3uM:** n=108, α=20.14±95.67 (median=5.21), 11.1% in Lévy range
- **ROSELLA + CBDV 30uM:** n=56, α=4.81±5.64 (median=3.71), 25.0% in Lévy range
- **ROSELLA + CBDV 3uM:** n=106, α=25.72±62.44 (median=6.74), 21.7% in Lévy range
- **ROSELLA + Control:** n=302, α=6.92±15.31 (median=3.48), 31.5% in Lévy range
- **ROSELLA + ETANOL:** n=104, α=24.95±33.90 (median=10.80), 7.7% in Lévy range
- **TJ356 + CBD 0.3uM:** n=215, α=36.49±93.46 (median=10.06), 10.7% in Lévy range
- **TJ356 + Control:** n=218, α=20.76±66.91 (median=5.90), 14.7% in Lévy range
- **TJ356 + ETANOL:** n=196, α=31.82±52.94 (median=12.21), 8.7% in Lévy range
- **TJ356 + Total_Extract_CBD:** n=201, α=20.07±43.51 (median=5.48), 16.4% in Lévy range
- **UA44 + CBD 0.3uM:** n=205, α=20.09±41.71 (median=6.92), 19.0% in Lévy range
- **UA44 + CBD 30uM:** n=46, α=21.16±35.58 (median=10.29), 6.5% in Lévy range
- **UA44 + CBD 3uM:** n=57, α=21.55±24.94 (median=15.50), 1.8% in Lévy range
- **UA44 + CBDV 0.3uM:** n=41, α=39.58±112.03 (median=4.41), 19.5% in Lévy range
- **UA44 + CBDV 30uM:** n=31, α=9.54±27.36 (median=3.74), 32.3% in Lévy range
- **UA44 + CBDV 3uM:** n=45, α=6.90±22.33 (median=3.15), 40.0% in Lévy range
- **UA44 + Control:** n=438, α=6.16±14.47 (median=3.28), 41.1% in Lévy range
- **UA44 + ETANOL:** n=249, α=29.27±113.25 (median=6.20), 14.1% in Lévy range
- **UA44 + Total_Extract_CBD:** n=98, α=8.99±19.38 (median=4.07), 22.4% in Lévy range

### Statistical Lévy Patterns (α outside strict range)

These cases show power-law distribution but α is outside the theoretical Lévy range (1-3):

- **ROSELLA + CBD 3uM:** α=3.176 (transition zone), p=2.00e-17
- **ROSELLA + CBD 30uM:** α=3.044 (transition zone), p=3.59e-06

### Detailed Strain Analysis

**TJ356:**
- Total treatments tested: 4
- Average α: 3.206
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)

**ROSELLA:**
- Total treatments tested: 8
- Average α: 3.131
- Lévy patterns (statistical): 2 (25.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Best statistical Lévy: CBD 3uM (α=3.176, p=2.00e-17)
- Most promising non-significant: Control (R=0.010)

**NL5901:**
- Total treatments tested: 9
- Average α: 2.973
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Most promising non-significant: CBD 3uM (R=-0.038)

**N2:**
- Total treatments tested: 9
- Average α: 3.736
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Most promising non-significant: CBDV 3uM (R=0.025)

**BR5271:**
- Total treatments tested: 4
- Average α: 2.677
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Most promising non-significant: CBD 0.3uM (R=-0.282)

**BR5270:**
- Total treatments tested: 4
- Average α: 2.844
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Most promising non-significant: ETANOL (R=-0.215)

**UA44:**
- Total treatments tested: 9
- Average α: 3.815
- Lévy patterns (statistical): 0 (0.0%)
- Lévy patterns (strict 1<α≤3): 0 (0.0%)
- Most promising non-significant: CBDV 0.3uM (R=-0.021)


### FDR Correction Results (Benjamini-Hochberg)

**Multiple comparison correction applied across 47 total tests:**
- Significant before correction (p < 0.05): 16
- Significant after FDR correction (q < 0.05): 14
- Likely false positives eliminated: 2

**FDR-corrected significant results:**

- **NL5901 + CBD 0.3uM:** q=0.000, R=-85.653 (transition zone)
- **ROSELLA + CBD 3uM:** q=0.000, R=1.310 (transition zone)
- **UA44 + ETANOL:** q=0.000, R=-62.975 (transition zone)
- **BR5271 + Control:** q=0.000, R=-46.906 (transition zone)
- **NL5901 + Control:** q=0.000, R=-31.652 (transition zone)
- **NL5901 + ETANOL:** q=0.000, R=-37.262 (transition zone)
- **ROSELLA + CBD 30uM:** q=0.000, R=0.875 (transition zone)
- **N2 + Total_Extract_CBD:** q=0.000, R=-22.562 (transition zone)
- **N2 + CBD 30uM:** q=0.000, R=-23.604 (transition zone)
- **BR5270 + Control:** q=0.000, R=-20.627 (transition zone)
- **TJ356 + Total_Extract_CBD:** q=0.001, R=-14.969 (transition zone)
- **ROSELLA + ETANOL:** q=0.002, R=-13.039 (transition zone)
- **UA44 + CBD 3uM:** q=0.009, R=-7.360 (transition zone)
- **TJ356 + ETANOL:** q=0.011, R=-9.032 (transition zone)

### Visualizations Generated

The following visualizations are available in each strain's output directory:
- `ccdf_*.png`: CCDF plots with power-law fits (pooled data)
- `alpha_histogram_*.png`: Distribution of α values across individual worms
- `alpha_boxplot.png`: Boxplot comparing α distributions between treatments
