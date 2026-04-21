# Machine Learning for Rainfall-Runoff Modelling

---

## Part 1: The Example Scripts

### Overall Concept

Unlike physics-based models (HBV, NAM, TOPMODEL) which encode equations describing how water moves through a catchment, ML models have **no physics built in**. Instead:
- You feed it input data (rain, ETp) and target output (discharge)
- It adjusts thousands of internal weights during training to minimise prediction error
- It learns statistical relationships from data — it has no concept of soil, groundwater, or a river
- If a physical pattern (e.g. rain fills soil before becoming runoff) exists consistently in the training data, the model will learn to replicate it without being told

| | Physics-based | LSTM/ML |
|---|---|---|
| Needs lots of data? | No | Yes |
| Works outside training range? | Better | Risky |
| Interpretable? | Yes | Hard |
| Captures complex patterns? | Limited | Strong |

---

### A_ReadSeries.py — Data Preparation

Reads raw hourly CSV files for rain, reference evapotranspiration (refet), and flow. Key steps:
1. Parses datetime columns and resamples all series to daily resolution
2. Merges the three series into a single aligned dataframe on a common date index
3. Fills missing values via linear interpolation
4. Saves the result as `dataframe.csv` (or `.pkl`) for use by the model scripts

---

### B_lstm_forecaster.py — LSTM Model

#### What it does
Trains a Long Short-Term Memory (LSTM) neural network to predict daily discharge from daily rainfall. Plots training/validation loss curves and predicted vs observed flow for both periods.

#### How an LSTM works
An LSTM processes the input sequence **one timestep at a time**, carrying a hidden state (memory) forward. At each step it decides:
- What to forget from previous memory
- What new information to add
- What to output

This makes it well-suited to time series where the past matters — e.g. yesterday's rain affects today's flow.

#### Key hyperparameters (lines 52–59)
| Parameter | Value | Meaning |
|---|---|---|
| `index_warmup` | 50 | First 50 steps excluded from loss — model needs time to initialise its memory |
| `index_validation` | 1398 | Train on days 0–1397, validate on 1398 onward |
| `epochs` | 500 | Number of full passes through the training data |
| `ninputs` | 1 | Number of input features (only rain here) |
| `nhidden` | 32 | Number of parallel memory units per LSTM layer |
| `nlayers` | 1 | Number of stacked LSTM layers |

#### Data flow
1. Load CSV → separate into `inputs` (rain) and `labels` (flow)
2. Scale both to [0, 1] using min-max scaling (computed on training data only)
3. Feed full training sequence into LSTM → get prediction → compute MSE loss
4. Backpropagation updates weights to reduce loss
5. After each epoch, evaluate on validation data (no gradient updates)
6. Save best model weights based on lowest validation loss

#### Loss function: MSE
Mean Squared Error — penalises large errors heavily. Computed excluding the warmup period.

#### Where to add complexity
| What | Where in code |
|---|---|
| More input features (ETp, temperature) | `load_datafile()` line 15, update `ninputs` line 57 |
| Bigger/deeper LSTM | `nhidden` and `nlayers` lines 58–59 |
| Different architecture | `model.py` + line 76 |
| Dropout regularisation | `model.py` lines 24/34 (already built in, currently disabled) |
| Different loss function (e.g. NSE) | `mse()` function line 45, used on lines 93/105 |
| Learning rate / scheduler | `optimizer` line 80 |

---

### C_rrformer_forecaster.py — Transformer Model (RRFormer)

#### What it does
Same goal as the LSTM (predict discharge from rain), but uses a **Transformer** architecture. Rather than processing the sequence step-by-step, the Transformer sees the entire lookback window at once and uses **attention** to learn which days matter most.

#### Key difference from LSTM
The attention mechanism assigns a weight to every day in the lookback window for each prediction. The model learns that, e.g., last week's rain matters a lot but so does a wet spell 3 months ago — without being told which days to look at. This is a dynamic, learned lookback within the fixed window.

#### Key hyperparameters (lines 68–81)
| Parameter | Value | Meaning |
|---|---|---|
| `window_len` | 90 | Days of rainfall context fed to the encoder |
| `pred_len` | 7 | Number of days predicted at once (multi-step output) |
| `batch_size` | 64 | Number of windows processed per weight update |
| `d_model` | 32 | Internal embedding dimension |
| `nhead` | 4 | Number of parallel attention heads |
| `num_enc_layers` | 2 | Transformer encoder depth |
| `num_dec_layers` | 2 | Transformer decoder depth |
| `dropout` | 0.1 | Fraction of neurons randomly disabled during training |

#### Loss function: NSE (Nash-Sutcliffe Efficiency)
More standard in hydrology than MSE. NSE = 1 is perfect, NSE = 0 means the model is no better than predicting the mean flow. The script minimises `MSE / variance`, which is equivalent to maximising NSE.

#### Sliding window approach
Unlike the LSTM which sees the full sequence at once, the Transformer uses a sliding window: each training sample is a 90-day rainfall window paired with the last 7 days of flow as the target. Windows are shuffled and fed in batches during training.

---

## Part 2: Our Project (Catchment: Havelse a, Stro — CatchmentML)

### Catchment Characteristics (from Presentation 2)

| Property | Value |
|---|---|
| Area | 140 km² |
| Average slope | 0.0013 m/m (very flat) |
| Rainfall-to-runoff lag | ~2 days (cross-correlation peak at lag k=4 on 12h data) |
| Runoff coefficient | ~18.7% (only 1 in 5 mm of rain reaches the river) |
| Shallow groundwater time constant (LS1) | ~81 days |
| Deep groundwater time constant (LS2) | ~451 days |

### Hydrological Character
- **Slow-responding, storage-dominated, groundwater-controlled**
- Very flat catchment means water moves slowly through the subsurface
- Strong groundwater-river interaction — baseflow dominates the hydrograph
- Snowmelt is relevant (temperatures below freezing observed, snow module needed in physics-based model)
- Low flashiness — monthly aggregation performs better than daily in physics-based models, suggesting smooth dynamics

### Implications for ML Model Design

**Lookback window (seqlength)**
A 365-day lookback is well justified:
- LS1 drains over ~81 days → shallow groundwater memory
- LS2 drains over ~451 days → deep groundwater memory exceeds a full year
- Snowpack adds seasonal memory (winter accumulation → spring melt)
- With ~20 years of data (~7450 rows in `daily_means.csv`), there is sufficient data to support long sequences

A 730-day lookback could also be tested given the 451-day deep groundwater constant.

**Input features**
The dataset (`daily_means.csv`) contains:
- `precipitation` — primary driver
- `ETp` — evapotranspiration demand (important for soil moisture dynamics)
- `discharge` — target variable

Adding ETp as a second input feature is a natural first improvement over the example scripts (which only use rain).

**Data file**
`Part 3/CatchmentML/daily_means.csv`
- Separator: `;`
- Columns: `date`, `discharge`, `precipitation`, `ETp`
- ~7450 rows (~2003–2022)
- Already preprocessed — `A_ReadSeries.py` is not needed

### Notes / Decisions
- Loss function: use **NSE** (more standard in hydrology, used in C script) rather than MSE
- Train/validation split: TBD based on data range
- Warmup period: likely needs to be longer than the example (50 days) given the long catchment memory — consider 365 days

---

## Part 3: Adapting B_lstm_forecaster.py for Havelse A

### Changes made to the example script

**Data loading**
- Removed `A_ReadSeries.py` step — `daily_means.csv` is already clean and daily-resolution
- Updated `pd.read_csv` to use `sep=';'` and column names `precipitation`/`discharge` instead of `rain`/`flow`
- Added `data_in.interpolate(method='linear', inplace=True)` after loading — fills any NaN gaps in all columns before tensor conversion, preventing gradient corruption during backpropagation

**Train/validation split**
- Replaced hardcoded `index_validation = 1398` with `int(len(data_in) * 0.7)` computed inside `load_datafile()` — makes the split portable across catchments with different data lengths
- 70/30 split gives train: 2003–2017, val: 2018–2023 (~5215 / 2236 days)
- `index_validation` is now returned from `load_datafile()` so the rest of the script uses the correct value

**NaN-safe scaling**
- Replaced `torch.amin/amax` with `data[~torch.isnan(data)].min()/.max()` — prevents NaN propagation through min-max scaling if any gaps remain

**NaN-safe loss function**
- Added boolean mask `~torch.isnan(label)` in `mse()` — excludes any timesteps with missing discharge from the loss calculation (46-day gap in 2006)

**Input features**
- Added ETp as a second input: `np.stack([precipitation, ETp], axis=1)` and `ninputs = 2`
- ETp helps the model learn soil moisture dynamics and seasonal baseflow patterns

**Gradient clipping** *(to be added)*
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()` — prevents gradient explosions observed at ~epoch 200 which corrupt weights

### Key observations from first runs
- Without interpolation of the 2006 gap: model hidden state collapses at that point and predicts a constant ~2.2 mm/d for the rest of training
- Adding ETp: removes the spurious second line on the rainfall plot (fix: index input tensor as `inputs[0,:,0]` to extract precipitation only for plotting)
- Gradient explosion still present — loss spike recovers but saved weights are suboptimal; gradient clipping is the fix

---

## Part 4: Feature Engineering and Model Improvements (Lecture 8)

### Data preparation — A_ReadSeries.py

A separate data preparation script was created to compile all features before training. This keeps `B_lstm_forecaster.py` clean.

**Pipeline:**
```
CSV12H_Clean.csv  ──┐
                    ├──► A_ReadSeries.py ──► ML_dataframe.csv ──► B_lstm_forecaster.py
daily_means.csv   ──┘
```

**Steps in A_ReadSeries.py:**
1. Load `CSV12H_Clean.csv` (12-hourly), extract `temp` column only, resample to daily mean with `.resample('D').mean()`
2. Load `daily_means.csv` and join on date index
3. Compute engineered features (see below)
4. Save as `ML_dataframe.csv`

---

### Features added

| Feature | Formula | Hydrological reasoning |
|---|---|---|
| `precip_30d` | 30-day rolling mean of precipitation | Captures antecedent moisture — wet soil produces more runoff from a given rain event |
| `precip_surplus` | `precipitation - ETp` | Daily net water balance — positive means catchment is gaining water, negative means drying out |
| `temp` | Daily mean temperature (°C) | Continuous signal for seasonal behaviour and snow dynamics |
| `melt` | `(temp > 0).astype(float)` | Binary on/off indicator for snowmelt conditions |

**On including both `temp` and `melt`:** They are not perfectly redundant — `temp` gives a continuous signal while `melt` gives a clean threshold. Feature importance analysis (next lecture) will determine if both are needed.

**Total inputs:** 6 (`precipitation`, `ETp`, `precip_30d`, `precip_surplus`, `temp`, `melt`)

---

### Scaling — switched from min-max to z-score

**Min-max scaling:** compresses data to [0, 1] but preserves the shape of the distribution (skew remains)

**Z-score scaling:** `z = (x - mean) / std` — centres data at 0 with unit standard deviation. Better for variables that can be negative (`precip_surplus`, `temp`) which don't fit naturally in [0, 1].

Scaling parameters (mean, std) are always computed on **training data only** and reused for validation — prevents data leakage.

---

### Residual analysis

Two diagnostic plots added after training:

**Histogram of residuals** — should be roughly bell-shaped and centred at 0
- Before feature engineering: slight right skew (model underpredicted peaks)
- After z-score scaling: more symmetric, closer to normal distribution
- Log-transform of discharge was considered but not applied — residuals were already close to normal

**Residuals over time** — should fluctuate randomly around zero with no visible pattern

**ACF of residuals** — bars outside 95% confidence band indicate the model is missing temporal structure at that lag

---

### Regularisation — dropout enabled

Training and validation loss diverged (overfitting) after switching to z-score scaling. Fix: enable dropout in `model.py` (was already built in but commented out).

- Dropout value: `0.4` (40% of neurons randomly disabled each training step)
- Dropout is only active during `model.train()` — switched off during `model.eval()` for predictions
- Note: dropout makes training loss artificially higher than validation loss (neurons are dropped during training but not evaluation), so some gap between the two curves is expected and normal

---

### Training configuration (current)

| Parameter | Value | Reason |
|---|---|---|
| `epochs` | 1000 | Sufficient for convergence at this learning rate |
| `lr` | `1e-3` | Lowered from `5e-3` to reduce loss spikes |
| `nhidden` | 64 | Unchanged |
| `dropout` | 0.2 | Balances regularisation without suppressing learning |
| `index_warmup` | 182 | Half a year for LSTM memory initialisation |

**Note on learning rate tuning:** reducing to `5e-4` with 1500 epochs made oscillation worse due to interaction with dropout — reverted to `1e-3` / 1000 epochs.

**Note on architecture tuning:** a second LSTM layer (`nlayers = 2`) was tested on the 9-feature model and did not improve performance.

---

## Part 5: Feature Analysis — D_lstm_shapley.py

### What it does

Quantifies each input feature's contribution to the model's predictions using two complementary methods from the `captum` library:

1. **ShapleyValueSampling** — Global feature importance. Wraps the model in a `ScalarWrapper` that averages predictions over all post-warmup timesteps, then computes Shapley values by systematically permuting features against a baseline. Produces a single importance percentage per feature.

2. **IntegratedGradients** — Temporal feature importance. Computes attributions for each individual output timestep by integrating gradients along a path from the baseline to the actual input. Shows how feature importance varies over time (e.g. precipitation matters more during storm events).

### Baseline design

A mixed baseline is used:
- Precipitation-type features and melt → set to **0** (no forcing)
- ETp, temperature, groundwater → set to **training mean** (average conditions)

This represents "average meteorological forcing" — the reference state against which each variable's contribution is measured. The completeness property guarantees: `sum(attributions) = F(x) - F(baseline)`.

### 9-feature model results

| Feature | Shapley % |
|---|---|
| Precip | -0.5% |
| ETp | 2.8% |
| Precip 30d | 14.2% |
| Precip 7d | -44.9% |
| Precip 90d | **-146.8%** |
| Precip surplus | **206.9%** |
| Temp | 21.6% |
| Groundwater | -33.2% |
| Melt | **80.1%** |

**Interpretation:** Severe multicollinearity between precipitation-derived features. Negative percentages exceeding -100% and positives exceeding +200% indicate features are fighting each other — the model exploits correlated redundancy to fit the data, but individual attributions are physically meaningless. This motivated feature reduction.

### Intermediate feature-reduction experiments (6-feature, original 7-feature)

Earlier iterations tried removing the **derived** features (dropping precip_7d, precip_90d, precip_surplus for the 6-feature model, then adding precip_7d back for a 7-feature variant). Those results have been removed from these notes — the training runs were based on `B_lstm_forecaster.py` with random seeds commented out (weights non-reproducible between runs) and Shapley normalization computed on signed values (percentages exceeded ±100% when attributions cancelled). Both methodology bugs were fixed before the final analysis below. The reported direction of travel (keep engineered features, drop raw precipitation and ETp) still holds.

---

## Part 6: Model Comparison — DM_Test.py

### Diebold-Mariano test

Compares the predictive accuracy of two models using their validation-period predictions and observations (saved as `.npy` files by `C_ModelEvaluation.py`).

**Method:**
1. Compute squared forecast errors for each model: `e1 = (obs - pred1)^2`, `e2 = (obs - pred2)^2`
2. Compute loss differential series: `d = e1^2 - e2^2`
3. Test whether the mean of `d` is significantly different from zero using a z-test with Newey-West variance estimate (accounts for serial correlation in errors)
4. Two-sided test: H0 = both models have equal predictive accuracy

**Interpretation:**
- If `p < 0.05` and `dm_stat > 0` → model 2 (pred2) is significantly better
- If `p < 0.05` and `dm_stat < 0` → model 1 (pred1) is significantly better
- If `p >= 0.05` → no significant difference

### Results

Earlier DM comparisons (9-feature vs 6-feature, 9-feature vs original 7-feature) have been removed alongside those model runs — the predictions used in those tests came from models trained without reproducible seeds and are no longer authoritative. Rerun against the current 7-feature try2 predictions when needed.

**Note on statistical vs practical significance:** With ~1800+ validation timesteps, the DM test has high statistical power and can detect even small systematic differences in squared errors. A statistically significant result does not imply a practically meaningful difference — especially worth keeping in mind when comparing models whose NSE scores differ by less than ~0.05.

---

## Part 7: Model Comparison Summary

### Performance

| Model | Val NSE | Test NSE | Shapley interpretability |
|---|---|---|---|
| 9-feature | 0.836 | 0.811 | Broken under signed normalization (negatives past -100%) |
| 7-feature try2 | 0.787 | 0.835 | Clean (bounded 0–100%, sums to 100%) |

### Residual diagnostics

The remaining models (9-feature and 7-feature try2) both show strong residual autocorrelation (ACF significant out to ~30 lags), indicating both underfit slow dynamics — likely the deep groundwater component (LS2 = 451 days). Residual histograms are roughly centred at zero with a slight right skew (underprediction of peaks).

### 7-feature model (try2) — alternative feature selection

This is the final model. The approach was to remove the **raw daily precipitation and ETp** from the 9-feature set — reasoning that these are already encoded in their derived aggregates (rolling means, surplus) and are therefore redundant. (Earlier experiments removed the *derived* features instead; those are documented in the intermediate-experiments note above.)

**Features removed:** `precipitation`, `ETp`
**Remaining 7 features:** precip_30d, precip_7d, precip_90d, precip_surplus, temp, groundwater, melt

**Note on re-run:** This analysis was redone after two methodology issues were identified:
1. `B_lstm_forecaster.py` had the random seeds commented out, so weights were not reproducible between runs. Comparing results across team members revealed mismatched models — a mid-experiment retrain had overwritten `weights.csv` with a different model, making the original attribution figures irreproducible.
2. Shapley normalization used signed values (`feature_importance / feature_importance.sum()`), which produces percentages exceeding ±100% when positive and negative attributions cancel in the denominator. Replaced with `abs(...) / abs(...).sum()` so percentages are bounded 0–100% and sum to 100% — standard practice for magnitude-of-influence reporting. Sign information is lost, but all numbers become interpretable.

Both issues were fixed (seeds set to 42; abs normalization), the model was retrained from scratch, and the feature analysis was rerun. Results below are from that rerun.

| Feature | Shapley % |
|---|---|
| Precip 30d | 2.7% |
| Precip 7d | 31.8% |
| Precip 90d | 23.4% |
| Precip surplus | 14.7% |
| Temp | 9.6% |
| Groundwater | 5.5% |
| Melt | 12.2% |

**Performance:** Val NSE = 0.787, Test NSE = 0.835

**Interpretation:**
- All percentages are now bounded 0–100% and sum to 100%, making the ranking directly interpretable as share-of-total-magnitude
- Short-term (precip_7d, 31.8%) and long-term (precip_90d, 23.4%) precipitation aggregates dominate — consistent with the catchment's dual timescale behaviour (quick response from recent rainfall, slow response from antecedent moisture)
- Precip_30d contributes only 2.7%, effectively redundant once precip_7d and precip_90d are present — a clear candidate for removal if further simplification is wanted
- Groundwater (5.5%) is surprisingly low; precip_90d likely already encodes the antecedent moisture signal that groundwater would otherwise carry
- Melt (12.2%) and temp (9.6%) together account for snow dynamics — melt being higher than temp suggests the model uses the explicit melt indicator rather than inferring it from temperature
- Shapley and the IG temporal panel give slightly different stories: Shapley ranks precip_7d first, while IG shows precip_surplus dominating throughout. This reflects the methods measuring different things — precip_surplus is always nonzero, so its per-timestep attribution is consistently large, while precip_7d spikes only during/after rainfall events. Both views are self-consistent
- **Test NSE 0.835 exceeds the 9-feature model's 0.811**, indicating that removing raw precipitation and ETp — which duplicate information already in the engineered features — slightly improves generalisation as well as interpretability

---

### Conclusions

1. Dropping the raw daily precipitation and ETp (7-feature try2) while keeping the engineered aggregates produced both the best test performance (NSE 0.835, exceeding the 9-feature model's 0.811) and the cleanest feature attributions — the engineered aggregates already carry the relevant rainfall signal, and the raw daily inputs were adding noise
2. The Shapley ranking (precip_7d 31.8%, precip_90d 23.4%, precip_surplus 14.7%) tells a physically coherent story: short-term triggering plus long-term antecedent moisture dominate runoff, consistent with the catchment's dual-timescale storage-dominated behaviour
3. Precip_30d is effectively redundant once precip_7d and precip_90d are both present (2.7% Shapley contribution) — candidate for removal if further simplification is wanted
4. Methodology lesson: seed every training script from day one and normalize Shapley attributions by `abs(...) / abs(...).sum()` rather than `x / x.sum()`. The original 6-feature/7-feature experiments were invalidated by both bugs — non-reproducible weights meant we could not replicate earlier figures, and signed normalization produced percentages past ±100% that obscured the real ranking
5. The persistent residual autocorrelation in all models suggests the biggest remaining improvement lies in capturing slow groundwater dynamics, potentially through longer sequence lengths or architectural changes, rather than further feature engineering
