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

## Summary of Improvements

| Change | Biggest impact |
|--------|----------------|
| Engineered features (Run 2) | Group5 validation: -0.10 → 0.35 |
| CyclicLR (Run 3) | Group1 validation: -2.46 → -1.79 |
| Static attributes (Run 4) | Group5 validation: 0.19 → 0.77 |
| Random windows + Group1 trim (Run 5) | Group1 validation: -0.31 → 0.63 |
| Refined features + flow path / elevation (Run 6) | Group1 test: -0.53 → 0.66 |

The biggest improvements came from adding **static attributes** (Run 4), switching to **random window sampling** combined with **trimming Group1's outlier** (Run 5), and **refining the feature set** with `longest_flow_path_km` and `precip_90d` (Run 6).

## Key Observations

- **Group1** went from failing on test (-0.53 in Run 5) to its best result yet (0.66 in Run 6) once `longest_flow_path_km` was added. The model now captures the small catchment's fast response.
- **Group2** (Havelse) consistently performs well across all runs - this is the same catchment the original single-catchment model was built for. Test NSE ~0.80 in Run 6.
- **Group5** (largest catchment) achieved the best validation NSE in Run 5 (0.84). It went from failing completely in Run 1 to being one of the best-performing on validation, though test NSE remains modest (~0.34).
- **Group7** continues to deliver the strongest test performance (0.79 in both Run 5 and Run 6).
- **Group3** still has weaker test performance despite strong validation - the test period may represent hydrological conditions not seen during training.
- **Group11** consistently shows signs of overfitting (better validation than test) - more data cleaning may help.
- Training and validation loss curves track together in Run 5 and Run 6, indicating the random window approach reduced overfitting compared to earlier runs.

## Script Structure

| Script | Purpose |
|--------|---------|
| A_ReadSeries.py | Data loading, standardization, feature engineering, Group1 trim, visualization |
| B_lstm_forecaster.py | Data loading functions, scaling, loss functions (used as import by other scripts). Also contains a full Approach 2 training loop. |
| B_lstm_forecaster_scheduler.py | Training script with LR search + CyclicLR scheduler (Approach 2) |
| B_lstm_forecaster_random_windows.py | Training script with random window sampling + CyclicLR scheduler (Approach 3) |
| C_ModelEvaluation.py | Load saved weights, compute NSE per catchment, generate diagnostic plots |
| model.py | LSTM architecture with optional static attribute embedding |

## Potential Future Improvements

- Add temperature data where available (Group11 already has it)
- Add groundwater data where available
- Sample multiple windows per catchment per batch (currently 1) for more stable gradients
- Cross-validation: train on 5 catchments, test on the 6th to evaluate true generalization to unseen catchments
- Experiment with model size (nhidden=32 vs 64)
- Try different window sizes (e.g. 2 years vs 4 years of training per window)
- Investigate why some catchments have weak test performance despite strong validation