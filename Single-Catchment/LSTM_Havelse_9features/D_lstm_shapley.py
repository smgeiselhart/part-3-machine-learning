import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import (ShapleyValueSampling,
                         IntegratedGradients)
from B_lstm_forecaster import load_datafile, scale_series, unscale_series, nse, mse
from model import LSTMModel

#Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Utility functions defined in script B
# =============================================================================


# =============================================================================
# Scalar wrapper (used for ShapleyValueSampling only)
# =============================================================================
# IntegratedGradients uses target=t to select a specific output timestep,
# so it does not need a wrapper — the raw model is used directly.

class ScalarWrapper(torch.nn.Module):
    def __init__(self, base_model, warmup):
        super().__init__()
        self.base_model = base_model
        self.warmup = warmup

    def forward(self, x):
        out = self.base_model(x)                # (batch, T)
        return out[:, self.warmup:].mean(dim=1) # (batch,)


# =============================================================================
# Hyperparameters - UPDATE TO MATCH BEST CONFIGURATION
# =============================================================================

index_warmup     = 182 #half a year 
ninputs          = 9
nhidden          = 64
nlayers          = 1

# =============================================================================
# Data loading and scaling
# =============================================================================

inputs_train, inputs_val, inputs_test, labels_train, labels_val, labels_test, index_validation = load_datafile(
    'Data/LSTM_dataframe.csv')

inputs_train, inputscales = scale_series(inputs_train)
labels_train, labelscales = scale_series(labels_train)
inputs_val,   inputscales = scale_series(inputs_val, inputscales)
labels_val,   labelscales = scale_series(labels_val, labelscales)
inputs_test, inputscales = scale_series(inputs_test, inputscales)
labels_test, labelscales = scale_series(labels_test, labelscales)

#Join all datasets (train, val, test) back together to ensure all get a warmup period
inputs_all = torch.cat([inputs_train, inputs_val, inputs_test], dim=1)
labels_all = torch.cat([labels_train, labels_val, labels_test], dim=1)

n_train = inputs_train.shape[1]
n_val = inputs_val.shape[1]

# =============================================================================
# Load model and weights
# =============================================================================

model = LSTMModel(ninputs, nhidden, 1, nlayers, 0)
model.load_state_dict(torch.load('weights.csv'))
model.eval()

# =============================================================================
# Baseline (shared by both methods)
# =============================================================================
# Training-data feature means broadcast to the validation length.
# Represents "average meteorological forcing" — the reference state against
# which each variable's contribution is measured.
# Completeness guarantee: sum(attributions) = F(x) - F(baseline)

## Mixed baseline:
#   Precip (0), Precip 30d (2), Precip 7d (3), Precip 90d (4), Melt (8) → 0 (no forcing)
#   ETp (1), Precip surplus (5), Temp (6), Groundwater (7) → training mean
train_mean = inputs_train.mean(dim=1, keepdim=True)       # (1, 1, 9)
baseline   = torch.zeros_like(inputs_val)                 # start from all zeros

#Set training mean for: ETp(1), precip_surplus(5), temp(6), groundwater(7)
for col in [1, 5, 6, 7]:
    baseline[:, :, col] = train_mean[:, :, col].expand_as(baseline[:, :, col])
baseline = baseline.detach()


feature_names = ['Precip', 'Etp', 'Precip 30d', 'Precip 7d', 'Precip 90d', 'Precip surplus', 'Temp', 'Groundwater', 'Melt']
T_val = inputs_val.shape[1]

# =============================================================================
# ShapleyValueSampling — Global Feature Importance
# =============================================================================

wrapped = ScalarWrapper(model, index_warmup) #the wrapper will compute the average of the flows predicted by the LSTM model

feature_mask = torch.zeros(1, T_val, ninputs, dtype=torch.long)
for i in range(ninputs):
    feature_mask[:, :, i] = i  # group 0 = Rain, 1 = PET, 2 = Rain 30d

svs = ShapleyValueSampling(wrapped)

attributions = svs.attribute(
    inputs_val,
    baselines=baseline,
    feature_mask=feature_mask,
    n_samples=200,
    show_progress=True,
)
feature_importance      = attributions[0, 0, :].detach().numpy()  # (9,)
feature_importance_norm = np.abs(feature_importance) / np.abs(feature_importance).sum()

print("\nGlobal feature importance (ShapleyValueSampling, normalised):")
for name, imp in zip(feature_names, feature_importance_norm):
    print(f"  {name:<22s}: {imp:.2%}")

# =============================================================================
# IntegratedGradients — Temporal Output Importance
# =============================================================================

ig = IntegratedGradients(model)

shap_step   = 1
out_indices = list(range(index_warmup, T_val, shap_step))
n_out       = len(out_indices)

temporal_importance = np.zeros((n_out, 9))
convergence_deltas  = np.zeros(n_out)

print(f"\nComputing IntegratedGradients for {n_out} output timesteps...")
for i, t in enumerate(out_indices):
    if i % 50 == 0:
        print(f"  timestep {i}/{n_out}")

    attr_t, delta = ig.attribute(
        inputs_val,
        baselines=baseline,
        target=t,
        n_steps=200,
        method='gausslegendre',
        return_convergence_delta=True,
    )
    temporal_importance[i] = attr_t[:, :t, :].abs().mean(dim=(0, 1)).detach().numpy()
    convergence_deltas[i]  = delta.abs().item()

print(f"  Mean convergence delta: {convergence_deltas.mean():.5f} "
      f"(should be close to 0)")

row_sums = temporal_importance.sum(axis=1, keepdims=True)
temporal_importance_norm = np.where(
    row_sums > 0, temporal_importance / row_sums, 0.0
)

# =============================================================================
# Prepare series for input and discharge panels (unscale to physical units)
# =============================================================================

# Observed inputs in original units: shape (T_val, 3)
inputs_val_phys = unscale_series(
    inputs_val[0, :, :], inputscales
).detach().numpy()

# Observed and predicted discharge in original units: shape (T_val,)
with torch.no_grad():
    pred_all = model(inputs_all)
    pred_val = pred_all[:, n_train: n_train + n_val]

flow_obs  = unscale_series(labels_val[0, :], labelscales).detach().numpy()
flow_pred = unscale_series(pred_val[0, :],   labelscales).detach().numpy()

t_axis = list(range(T_val))

# =============================================================================
# Plot — 4 panels
# =============================================================================

fig, axes = plt.subplots(
    nrows=4, figsize=(12, 14),
    gridspec_kw={'height_ratios': [0.9, 1.4, 1.2, 1.2]}
)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#17becf']

# Share x-axis across the three time series panels (rows 1, 2, 3)
axes[2].sharex(axes[1])
axes[3].sharex(axes[1])

# --- Panel 1: Global Feature Importance (ShapleyValueSampling) ---
ax = axes[0]
bars = ax.bar(feature_names, feature_importance_norm, color=colors,
              edgecolor='white', linewidth=0.8, width=0.5)
ax.set_ylabel('Normalised\nImportance')
ax.set_title('(a) Global Feature Importance (ShapleyValueSampling)',
             fontsize=11, fontweight='bold', loc='left')
y_max = max(feature_importance_norm) * 1.3
y_min = min(feature_importance_norm) * 1.3 if min(feature_importance_norm) < 0 else 0
ax.set_ylim(y_min, y_max)
ax.axhline(0, color = 'black', linewidth = 0.5)
ax.set_yticks([])
ax.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, feature_importance_norm):
    if val >= 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else: 
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.01, 
        f'{val:.1%}', ha='center', va='top', fontsize=11, fontweight='bold')

# --- Panel 2: Temporal Output Importance (IntegratedGradients) ---
ax2 = axes[1]
for i, (name, color) in enumerate(zip(feature_names, colors)):
    ax2.plot(out_indices, temporal_importance_norm[:, i],
             label=name, color=color, linewidth=1.5)
ax2.set_ylabel('Mean |Attribution|')
ax2.set_title('(b) Temporal Output Importance (IntegratedGradients)',
              fontsize=11, fontweight='bold', loc='left')
ax2.legend(loc='upper right', framealpha=0.8)
ax2.spines[['top', 'right']].set_visible(False)
plt.setp(ax2.get_xticklabels(), visible=False)

# --- Panel 3: Observed input series ---
ax3 = axes[2]
# Rain as bars (standard hydrological convention), PET and rain_month as lines
ax3.bar(t_axis, inputs_val_phys[:, 0], color=colors[0],
        label=feature_names[0], width=1.0, alpha=0.75)
for i in range(1,9):
    ax3.plot(t_axis, inputs_val_phys[:, i], color = colors[i],
             label = feature_names[i], linewidth = 1.4)
ax3.set_ylabel('mm/day')
ax3.set_title('(c) Observed Meteorological Inputs',
              fontsize=11, fontweight='bold', loc='left')
ax3.legend(loc='upper right', framealpha=0.8, fontsize = 8, ncol = 3)
ax3.spines[['top', 'right']].set_visible(False)
plt.setp(ax3.get_xticklabels(), visible=False)

# --- Panel 4: Observed and predicted discharge ---
ax4 = axes[3]
ax4.plot(t_axis, flow_obs,  color='#2d2d2d', label='Observed',
         linewidth=1.5)
ax4.plot(t_axis, flow_pred, color='#C44E52', label='Predicted',
         linewidth=1.5, linestyle='--')
ax4.set_xlabel('Validation Timestep')
ax4.set_ylabel('Discharge (mm/day)')
ax4.set_title('(d) Observed vs Predicted Discharge',
              fontsize=11, fontweight='bold', loc='left')
ax4.legend(loc='upper right', framealpha=0.8)
ax4.spines[['top', 'right']].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig('figures/feature_importance_ig.png', dpi=150)
plt.show()