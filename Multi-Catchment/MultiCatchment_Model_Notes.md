# Multi-Catchment LSTM Model - Development Notes

## Overview
This model extends the single-catchment LSTM (Havelse) to a generalized multi-catchment model trained on 6 Danish catchments simultaneously. The goal is a single model that can predict discharge across catchments with different characteristics.

## Catchment Data

| Catchment | Rows | Date Range | Area (ha) | Key Notes |
|-----------|------|------------|-----------|-----------|
| Group1 | 5,521 | 2011-2026 | 2,914 | CSV uses commas (others use semicolons). Trimmed to end of 2024 due to discharge outlier/measurement error in later data. |
| Group2 | 7,452 | 2003-2023 | 14,000 | Same as Havelse from single-catchment model. Column names differ (ETp vs ETP). |
| Group3 | 13,881 | 1988-2025 | 11,863 | Longest record. 1,102 missing discharge values. |
| Group5 | 8,358 | 2003-2025 | 120,300 | Largest catchment by far. 177 missing discharge, 101 missing ETP. |
| Group7 | 6,493 | 2008-2025 | 4,842 | Clean data, no missing values. |
| Group11 | 5,886 | 2010-2026 | 48,900 | Has temperature column (not used). 1,119 missing discharge. |

## Data Standardization (A_ReadSeries.py)

The raw data had several inconsistencies that needed resolving:
- **Separators**: Group1 uses commas, all others use semicolons. Script auto-detects by trying semicolon first, falling back to comma.
- **Column names**: ETP vs ETp capitalization, different column ordering. All normalized to: discharge, precipitation, ETp.
- **Extra columns**: Group11 has temperature - dropped to keep all catchments consistent.
- **Different date ranges**: Left as-is. Each catchment is split independently into train/val/test.
- **Missing data (NaN)**: Handled via linear interpolation in B_lstm_forecaster.py.
- **Group1 outlier**: Trimmed Group1 data to end of 2024 to remove a measurement error spike.

Only precipitation and ETp are available across all 6 catchments (no groundwater or temperature), so the feature set is more limited than the single-catchment model.

## Feature Engineering

### Dynamic inputs (per timestep) — current set (Run 6)
- precip_7d: 7-day rolling mean of precipitation (short-term triggering signal)
- precip_30d: 30-day rolling mean of precipitation (antecedent moisture)
- precip_90d: 90-day rolling mean of precipitation (longer-term wetness)
- precip_surplus: precipitation - ETp (net water balance)

Raw `precipitation` and raw `ETp` were dropped in Run 6 based on the single-catchment feature analysis — the rolling means already encode the precipitation signal in a smoother form, and ETp's contribution is captured through `precip_surplus`.

Log1p transformation applied to precip_7d, precip_30d, and precip_90d based on histogram analysis (all right-skewed). Precip_surplus is NOT log-transformed because it contains negative values.

### Static inputs (per catchment) — current set (Run 6)
Added to help the model differentiate between catchments:
- area_ha
- mean_elevation_m
- mean_slope_percent
- rural_percent
- urban_percent
- nature_percent
- lake_percent
- longest_flow_path_km

Initially `mean_elevation_m` was excluded (Denmark is flat), but it was added back in Run 6 along with `longest_flow_path_km`. The flow-path length proved especially helpful for differentiating catchment response times (notably Group1).

Static attributes are encoded through a small embedding network (Linear(n_static,16) -> ReLU -> Linear(16,16)) and concatenated to the dynamic inputs at every timestep.

## Training Approaches

Two training approaches from Lecture 10 were implemented and compared:

### Training Approach 2 (Full sequence per catchment)
Each epoch loops through all catchments, computes MSE for each one individually using the full training timeseries, then averages the losses across catchments. This gives each catchment equal weight regardless of time series length (avoids bias toward Group3 with 13,881 rows vs Group1 with 5,521).

Each epoch:
1. Forward pass on each catchment separately (6 forward passes)
2. Compute MSE per catchment (excluding 182-day warmup)
3. Average losses across all 6 catchments
4. Single backpropagation and weight update

### Training Approach 3 (Random window sampling)
Instead of feeding full timeseries, each step randomly samples a fixed-length window (182 days warmup + 4 years training = 1,642 days) from each catchment. Windows are stacked into a batch with shape (n_catchments, 1642, n_features) and processed in a single forward pass.

Each epoch:
1. Randomly sample one window from each catchment's training data
2. Stack windows into a batched tensor
3. Single batched forward pass through all catchments at once
4. Compute MSE (excluding warmup)
5. Backprop and weight update
6. Resample new windows for next epoch (acts as data augmentation)

Benefits over Approach 2:
- True parallel processing - one forward pass instead of 6
- Data augmentation - model sees different time combinations each epoch
- Fixed sequence length regardless of catchment data length
- Reduced overfitting

### Scaling (both approaches)
Z-score normalization (mean/std) computed globally across ALL catchments' training data, then applied to each catchment. Static attributes scaled separately with their own mean/std.

### Train/Val/Test Split
65% / 25% / 10% per catchment (same as single-catchment model).

### Learning Rate
LR search conducted (10 epochs each across range 1e-5 to 5e-2). Optimal LR found at 1e-2. Implemented CyclicLR triangular scheduler cycling between base_lr=1e-3 and max_lr=1e-2 with step_size_up=4 (full cycle = 8 epochs). This helps the model escape local minima.

### Other hyperparameters
- Hidden size: 64
- LSTM layers: 1
- Dropout: 0.2
- Warmup period: 182 days (6 months)
- Epochs: 500
- Optimizer: Adam
- Gradient clipping: max norm 1.0

## Model Evolution & Results

### Run 1: Baseline
- **Forecaster**: B_lstm_forecaster.py (Approach 2, full sequences)
- **Features**: precipitation, ETp only (2 features)
- **Static**: None
- **LR**: Fixed at 1e-3
- **Dropout**: 0
- **Group1 trim**: No

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -4.03 | -1.12 |
| Group2 | 0.04 | 0.34 |
| Group3 | 0.69 | 0.37 |
| Group5 | -0.10 | -0.35 |
| Group7 | 0.61 | 0.57 |
| Group11 | 0.76 | -0.70 |

### Run 2: Added engineered features
- **Forecaster**: B_lstm_forecaster.py (Approach 2)
- **Features**: 5 features (added precip_7d, precip_30d, precip_surplus)
- **Static**: None
- **LR**: Fixed at 1e-3
- **Dropout**: 0
- **Group1 trim**: No
- **Transformations**: Log1p applied to precipitation, ETp, precip_7d, precip_30d

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -2.46 | -0.72 |
| Group2 | -0.00 | 0.35 |
| Group3 | 0.75 | 0.45 |
| Group5 | 0.35 | -0.07 |
| Group7 | 0.60 | 0.59 |
| Group11 | 0.74 | -1.38 |

### Run 3: Added CyclicLR scheduler
- **Forecaster**: B_lstm_forecaster_scheduler.py (Approach 2)
- **Features**: 5 features
- **Static**: None
- **LR**: CyclicLR, base_lr=1e-3, max_lr=1e-2, step_size_up=4
- **Dropout**: 0
- **Group1 trim**: No

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -1.79 | -0.42 |
| Group2 | 0.11 | 0.47 |
| Group3 | 0.76 | 0.29 |
| Group5 | 0.19 | -0.10 |
| Group7 | 0.54 | 0.36 |
| Group11 | 0.66 | -0.83 |

### Run 4: Added static attributes + dropout
- **Forecaster**: B_lstm_forecaster_scheduler.py (Approach 2)
- **Features**: 5 dynamic + 6 static attributes
- **Static**: area_ha, mean_slope_percent, rural/urban/nature/lake percent (embedded via FC network)
- **LR**: CyclicLR, base_lr=1e-3, max_lr=1e-2
- **Dropout**: 0.2
- **Group1 trim**: No

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -0.31 | -0.16 |
| Group2 | 0.74 | **0.86** |
| Group3 | 0.79 | 0.44 |
| Group5 | 0.77 | 0.49 |
| Group7 | 0.72 | 0.66 |
| Group11 | 0.70 | 0.13 |

**Static attributes were a major breakthrough** - they allow the model to differentiate between catchments. Group2 and Group5 saw the largest jumps.

### Run 5: Random window sampling + Group1 trim
- **Forecaster**: B_lstm_forecaster_random_windows.py (Approach 3)
- **Features**: 5 dynamic + 6 static attributes
- **Static**: Same as Run 4
- **LR**: CyclicLR, base_lr=1e-3, max_lr=1e-2
- **Dropout**: 0.2
- **Group1 trim**: Yes - data trimmed to end of 2024 to remove outlier
- **Window size**: 182-day warmup + 4 years = 1,642 days per sample

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | **0.63** | -0.53 |
| Group2 | 0.73 | 0.81 |
| Group3 | 0.80 | 0.45 |
| Group5 | **0.84** | 0.38 |
| Group7 | **0.78** | **0.79** |
| Group11 | 0.66 | 0.38 |

**Random window sampling combined with the Group1 trim produced the most balanced model.** All 6 catchments achieved validation NSE > 0.6 for the first time. Training and validation loss curves track closely together throughout training with no visible overfitting gap.

### Run 6: Refined feature set
- **Forecaster**: B_lstm_forecaster_random_windows.py (Approach 3)
- **Features**: 4 dynamic features — dropped raw `precipitation` and raw `ETp` based on single-catchment feature analysis; added `precip_90d` (90-day rolling mean). Final set: precip_7d, precip_30d, precip_90d, precip_surplus.
- **Static**: 8 attributes — added `mean_elevation_m` back in and added `longest_flow_path_km`. Final set: area_ha, mean_elevation_m, mean_slope_percent, rural_percent, urban_percent, nature_percent, lake_percent, longest_flow_path_km.
- **LR**: CyclicLR, base_lr=1e-3, max_lr=1e-2 (unchanged)
- **Dropout**: 0.2 (unchanged)
- **Group1 trim**: Yes (unchanged)
- **Window size**: 182-day warmup + 4 years (unchanged)
- **Log1p transformation**: Applied to precip_7d, precip_30d, precip_90d (precip_surplus excluded due to negative values)

| Catchment | Test NSE |
|-----------|----------|
| Group1 | **0.66** |
| Group2 | 0.80 |
| Group3 | 0.41 |
| Group5 | 0.34 |
| Group7 | **0.79** |
| Group11 | 0.34 |

**Group1 finally cracked positive test NSE (-0.53 → 0.66)**, while the other catchments held roughly steady (within ±0.05 of Run 5). Training loss bottoms out lower than Run 5 (~0.15–0.20). The likely driver is `longest_flow_path_km`, which gives the model a direct signal for catchment response time — particularly informative for Group1 (the smallest catchment). Removing the raw precipitation/ETp signals in favor of only smoothed/derived features simplified the input space without losing information, since the rolling means already encode that signal.

### Run 7: Bug fix — NaN masking now actually works + seeded for reproducibility
- **Forecaster**: B_lstm_forecaster_random_windows.py (Approach 3)
- **Features**: Same as Run 6 (4 dynamic + 8 static)
- **Hyperparameters**: All same as Run 6
- **Bug fix**: In `load_catchment`, interpolation is now applied only to input feature columns. Discharge labels keep their NaN values intact, so the `~torch.isnan(label)` masks in `mse()` and `nse()` actually filter out interpolated points.
  - Previously: `data_in.interpolate(method='linear', inplace=True)` filled the entire DataFrame including discharge — masks were no-ops, model was being scored against ~10 months of fabricated linear discharge in Group3 and similar gaps in Group11.
- **Reproducibility**: `torch.manual_seed(42)` and `np.random.seed(42)` set at top of script. Re-runs are now deterministic.

| Catchment | Train NSE | Val NSE | Test NSE |
|-----------|-----------|---------|----------|
| Group1 | 0.66 | 0.79 | 0.46 |
| Group2 | 0.79 | 0.76 | **0.84** |
| Group3 | 0.82 | 0.84 | **0.82** |
| Group5 | 0.77 | 0.81 | 0.27 |
| Group7 | 0.87 | 0.76 | **0.85** |
| Group11 | 0.73 | 0.66 | 0.30 |

**Group3 jumped from 0.41 → 0.82 on test** — the model wasn't actually performing worse than the others; it was being unfairly penalized for failing to match an interpolated discharge segment that wasn't a real observation. Train ≈ val ≈ test now (all in the 0.82–0.84 range), which is a much healthier signal.

**Group2 and Group7 also improved slightly** (+0.04, +0.06) despite having clean discharge data — likely just better luck from the seeded random window draws.

**Group5 and Group11 dropped slightly** (-0.07, -0.04). These catchments have NaN gaps in their discharge that were previously contributing easy-to-predict interpolated values to the score. With those removed, the metric is now reflecting the model's true performance on the harder real observations. Group5 in particular continues to under-predict the major peak events in its test period — likely a distribution-shift problem where the test years are unusually wet relative to the training period.

**Group1 dropped from 0.66 → 0.46.** This is partly the fix and partly run-to-run variance. Group1 has the shortest record (after the 2024 trim) so it's most sensitive to which 4-year window happens to get sampled.

## Summary of Improvements

| Change | Biggest impact |
|--------|----------------|
| Engineered features (Run 2) | Group5 validation: -0.10 → 0.35 |
| CyclicLR (Run 3) | Group1 validation: -2.46 → -1.79 |
| Static attributes (Run 4) | Group5 validation: 0.19 → 0.77 |
| Random windows + Group1 trim (Run 5) | Group1 validation: -0.31 → 0.63 |
| Refined features + flow path / elevation (Run 6) | Group1 test: -0.53 → 0.66 |
| NaN masking bug fix + seeded (Run 7) | Group3 test: 0.41 → 0.82 |

The biggest improvements came from adding **static attributes** (Run 4), switching to **random window sampling** combined with **trimming Group1's outlier** (Run 5), **refining the feature set** with `longest_flow_path_km` and `precip_90d` (Run 6), and finally fixing the **NaN masking bug** so discharge gaps no longer count toward the score (Run 7).

## Key Observations (as of Run 7)

- **Group2, Group3, Group7** are now the strongest test performers (0.82–0.85). All three have relatively clean discharge data, and their train/val/test NSEs all track within ~0.10 of each other — a healthy generalization signature.
- **Group3** went from looking like a weak catchment (0.41 test in Run 6) to one of the strongest (0.82) once the NaN masking was fixed. The earlier "weakness" was an artifact of being scored against ~10 months of interpolated discharge.
- **Group2** (Havelse, the original single-catchment model) consistently performs well across runs. Test NSE 0.84 in Run 7.
- **Group7** delivers the strongest test performance (0.85 in Run 7) but shows some overfitting (train 0.87 vs val 0.76 vs test 0.85 — val being the weakest is unusual and may reflect unusual conditions in its specific validation window).
- **Group1** test NSE bounces between 0.46–0.66 across runs. Smallest dataset after the 2024 trim, so most sensitive to random window sampling. Even with seeding, this catchment will be hardest to make rock-solid.
- **Group5** (largest catchment) and **Group11** are the new problem children. Both have train ≈ val NSE around 0.7–0.85 but test NSE in the 0.27–0.30 range. Now that NaN masking is honest, this gap reflects real distribution shift between their training periods and test periods. Group5 in particular under-predicts major peak events visible in the test plot.
- Training and validation loss curves now show a visible gap (train consistently lower) — a more honest picture of generalization than the previous "tracks together" appearance, which was partially flattered by the model fitting interpolated values trivially.

## Leave-One-Catchment-Out (LOCO) Cross-Validation

LOCO is the proper test of generalization to *unseen* catchments: for each fold, train on 5 catchments and evaluate on the 6th. Same model architecture and hyperparameters as Run 7, same NaN masking fix, same seeded training. Implemented in `B_lstm_forecaster_random_windows_LOCO.py`.

### Results

| Held-out catchment | Test NSE |
|--------------------|----------|
| Group1 | -2.55 |
| Group2 | -2.04 |
| Group11 | -1.22 |
| Group5 | -0.12 |
| Group7 | +0.10 |
| Group3 | +0.47 |

Training and validation loss curves looked similar across folds (train ~0.12–0.20, val ~0.22–0.34) — the model successfully fits the 5 training catchments in each fold. The held-out catchment never enters the loss function during training.

### Static-attribute analysis: testing the "extremes fail" hypothesis

Initial intuition: catchments at the extremes of the static-attribute distribution should fail worst because the embedding has no nearby training neighbors to interpolate from. The data does **not** support this:

| Catchment | NSE | Dist from mean (z-space) | Max \|z\| | Nearest-neighbor dist | Most extreme attribute |
|-----------|-----|--------------------------|-----------|------------------------|------------------------|
| Group1 | -2.55 | 2.08 | 1.13 | 1.92 | rural% (-1.13) |
| Group2 | -2.04 | 2.47 | 1.52 | 2.71 | rural% (+1.52) |
| Group11 | -1.22 | **1.65** | 1.17 | 1.92 | lake% (-1.17) |
| Group5 | -0.12 | **3.47** | **1.90** | **3.98** | area, flow path (+1.90) |
| Group7 | +0.10 | 2.99 | 1.67 | 3.75 | urban% (-1.67) |
| Group3 | +0.47 | 2.43 | 1.35 | 3.18 | slope (+1.35) |

All three extremeness metrics correlate **positively** with LOCO NSE (r = +0.53 to +0.77), the opposite of the prediction. The catchment most isolated in attribute space (Group5) achieves middling LOCO performance, while one of the catchments closest to its neighbors (Group11) performs poorly.

### Tentative pattern (limited data)

A weaker pattern is visible if you look at *which kind* of attribute is extreme:

- **Catchments that fail (Group1, Group2, Group11)** are most extreme on a **land-use** attribute (rural, nature, lake).
- **Catchments that work (Group3, Group5, Group7)** are most extreme on a **topographic** attribute (slope, elevation, area, flow path).

A coherent hypothesis: the model has learned that topographic attributes meaningfully shape hydrologic response (small catchments respond faster, etc.) and can extrapolate when held-out catchments are topographically distinctive. It treats land-use attributes more like an arbitrary catchment ID — useful for fitting the trained catchments, useless for predicting unseen ones with different land-use mixes.

**Caveat:** with N=6 catchments and 8 attributes, this is barely above the noise floor. A Pearson r of 0.77 is not statistically meaningful at N=6 (95% CI for the null hypothesis r=0 is roughly ±0.81). Treat as a hypothesis worth testing, not a finding.

### Broader takeaway

LOCO test NSEs range from -2.55 to +0.47 with no clean attribute-space explanation for which catchments fail. With only 6 catchments, the static-attribute embedding has insufficient data to learn smooth attribute-to-behavior mappings — published regional LSTM hydrology models (e.g. Kratzert et al. 2019) typically use 500+ catchments to make static attributes generalize meaningfully. The N=6 setup cannot distinguish "genuine attribute-driven differences" from "arbitrary catchment-specific behavior the model memorized."

**The joint multi-catchment model from Run 7 is therefore a *joint training* model, not a *regional generalization* model.** It is a useful tool for predicting the 6 specific catchments it was trained on (test NSEs 0.27–0.85) but cannot be relied on to predict an unseen catchment from its static attributes alone.

### Diagnostic experiments worth running (not yet attempted)

- **Drop land-use attributes entirely.** Train with only topographic attributes (area, elevation, slope, flow path). If LOCO numbers improve, that supports the topographic-vs-land-use hypothesis. If they don't change, it confirms N=6 is the dominant constraint.
- **Engineered land-use ratios** (e.g. `urban / (rural + nature)`) — fewer, more physically meaningful land-use signals.
- **PCA the static attributes to 2–3 components** — forces the embedding to use only directions of meaningful variance, reduces over-parameterization.

A "dominant land-use" binary categorical was considered and rejected: all 6 catchments are rural-dominated (rural% is the largest category in every catchment), so a categorical of the dominant class would produce identical values for every catchment.

## Script Structure

| Script | Purpose |
|--------|---------|
| A_ReadSeries.py | Data loading, standardization, feature engineering, Group1 trim, visualization |
| B_lstm_forecaster.py | Data loading functions, scaling, loss functions (used as import by other scripts). Also contains a full Approach 2 training loop. |
| B_lstm_forecaster_scheduler.py | Training script with LR search + CyclicLR scheduler (Approach 2) |
| B_lstm_forecaster_random_windows.py | Training script with random window sampling + CyclicLR scheduler (Approach 3) |
| B_lstm_forecaster_random_windows_LOCO.py | LOCO cross-validation: trains 6 separate models, each holding out one catchment as the test set |
| C_ModelEvaluation.py | Load saved weights, compute NSE per catchment, generate diagnostic plots |
| model.py | LSTM architecture with optional static attribute embedding |

## Potential Future Improvements

- Add temperature data where available (Group11 already has it)
- Add groundwater data where available
- Sample multiple windows per catchment per batch (currently 1) for more stable gradients
- Experiment with model size (nhidden=32 vs 64)
- Try different window sizes (e.g. 2 years vs 4 years of training per window)
- Investigate why some catchments have weak test performance despite strong validation (Group5, Group11 — likely distribution shift in test period; cross-check via shared chronological test window)
- LOCO diagnostic experiments: drop land-use attributes; engineered land-use ratios; PCA-compressed static inputs (see LOCO section)