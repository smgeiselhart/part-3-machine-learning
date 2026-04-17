# Multi-Catchment LSTM Model - Development Notes

## Overview
This model extends the single-catchment LSTM (Havelse) to a generalized multi-catchment model trained on 6 Danish catchments simultaneously. The goal is a single model that can predict discharge across catchments with different characteristics.

## Catchment Data

| Catchment | Rows | Date Range | Area (ha) | Key Notes |
|-----------|------|------------|-----------|-----------|
| Group1 | 5,521 | 2011-2026 | 2,914 | CSV uses commas (others use semicolons). Has a discharge outlier - needs investigation. |
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

Only precipitation and ETp are available across all 6 catchments (no groundwater or temperature), so the feature set is more limited than the single-catchment model.

## Feature Engineering

### Dynamic inputs (per timestep)
- precipitation (raw)
- ETp (raw)
- precip_7d: 7-day rolling mean of precipitation (short-term triggering signal)
- precip_30d: 30-day rolling mean of precipitation (antecedent moisture)
- precip_surplus: precipitation - ETp (net water balance)

Log1p transformation applied to precipitation, ETp, precip_7d, and precip_30d based on histogram analysis (all right-skewed). Precip_surplus is NOT log-transformed because it contains negative values.

### Static inputs (per catchment)
Added later to help the model differentiate between catchments:
- area_ha
- mean_slope_percent
- rural_percent
- urban_percent
- nature_percent
- lake_percent

Mean elevation was excluded - all catchments are in Denmark where elevation doesn't vary much.

Static attributes are encoded through a small embedding network (Linear(6,16) -> ReLU -> Linear(16,16)) and concatenated to the dynamic inputs at every timestep.

## Training Approach

Following **Training Approach 2** from Lecture 10: compute MSE for each catchment individually, then average across catchments. This gives each catchment equal weight regardless of time series length (avoids bias toward Group3 with 13,881 rows vs Group1 with 5,521).

Each epoch:
1. Forward pass on each catchment separately
2. Compute MSE per catchment (excluding 182-day warmup)
3. Average losses across all 6 catchments
4. Single backpropagation and weight update

### Scaling
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
- Epochs: 500 (model converges around epoch 200, best weights saved automatically)
- Optimizer: Adam
- Gradient clipping: max norm 1.0

## Model Evolution & Results

### Run 1: Baseline (2 features, fixed LR 1e-3, no static)
Features: precipitation, ETp only. No dropout.

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -4.03 | -1.12 |
| Group2 | 0.04 | 0.34 |
| Group3 | 0.69 | 0.37 |
| Group5 | -0.10 | -0.35 |
| Group7 | 0.61 | 0.57 |
| Group11 | 0.76 | -0.70 |

### Run 2: Added engineered features (5 features, fixed LR 1e-3, no static)
Added precip_7d, precip_30d, precip_surplus. Log-transformed skewed inputs.

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -2.46 | -0.72 |
| Group2 | -0.00 | 0.35 |
| Group3 | 0.75 | 0.45 |
| Group5 | 0.35 | -0.07 |
| Group7 | 0.60 | 0.59 |
| Group11 | 0.74 | -1.38 |

### Run 3: Added CyclicLR scheduler (5 features, CyclicLR, no static)
Same features, added CyclicLR scheduler with base_lr=1e-3, max_lr=1e-2.

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -1.79 | -0.42 |
| Group2 | 0.11 | 0.47 |
| Group3 | 0.76 | 0.29 |
| Group5 | 0.19 | -0.10 |
| Group7 | 0.54 | 0.36 |
| Group11 | 0.66 | -0.83 |

### Run 4: Added static attributes (5 features, CyclicLR, 6 static attributes, dropout 0.2)
Added catchment properties as static inputs through embedding network.

| Catchment | Val NSE | Test NSE |
|-----------|---------|----------|
| Group1 | -0.31 | -0.16 |
| Group2 | 0.74 | **0.86** |
| Group3 | **0.79** | 0.44 |
| Group5 | **0.77** | 0.49 |
| Group7 | **0.72** | 0.66 |
| Group11 | 0.70 | 0.13 |

**Static attributes were the single biggest improvement** - they allow the model to differentiate between catchments. Group2 and Group5 saw the largest jumps.

## Key Observations

- **Group1** remains problematic across all runs. Likely caused by a discharge outlier/measurement error visible in the data overview plots. This should be investigated and potentially clipped.
- **Group2** (Havelse) performs best overall, which makes sense as it's the same catchment the original single-catchment model was built for.
- **Group3** has strong validation but weaker test performance - the test period may have different hydrological conditions than training.
- **Group11** shows signs of overfitting (good validation, poor test) - more data cleaning may help (1,119 missing discharge values).
- Training and validation loss curves diverge around epoch 200, indicating the model has converged by that point. Extra epochs beyond ~300 are unnecessary.

## Script Structure

| Script | Purpose |
|--------|---------|
| A_ReadSeries.py | Data loading, standardization, feature engineering, visualization |
| B_lstm_forecaster.py | Data loading functions, scaling, loss functions (also used as import by other scripts) |
| B_lstm_forecaster_scheduler.py | Main training script with LR search and CyclicLR scheduler |
| C_ModelEvaluation.py | Load saved weights, compute NSE per catchment, generate diagnostic plots |
| model.py | LSTM architecture with optional static attribute embedding |

## Potential Future Improvements

- Fix Group1 discharge outlier (clip or remove)
- Add temperature data where available
- Add groundwater data where available
- Implement Training Approach 3 (random window sampling) from Lecture 10 for faster training and better generalization
- Cross-validation: train on 5 catchments, test on the 6th to evaluate true generalization
- Experiment with model size (nhidden=32 vs 64) to address overfitting
