# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:16:18 2026

@author: olegh
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import (ShapleyValueSampling,
                         IntegratedGradients)

from model import LSTMModel

# =============================================================================
# Utility functions
# =============================================================================

def load_datafile(datafile, index_validation):
    data_in = pd.read_pickle(datafile)
    rain_month = data_in['rain'].rolling(window='30D').mean()
    inputs = np.stack([
        data_in['rain'].to_numpy(),
        data_in['PET'].to_numpy(),
        rain_month.to_numpy()
    ], axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(data_in['flow'].to_numpy(), dtype=torch.float32)
    inputs = inputs.unsqueeze(0)          # (1, T, 3)
    labels = labels.unsqueeze(0)          # (1, T)
    inputs_train = inputs[:, :index_validation, :]
    inputs_val   = inputs[:, index_validation:, :]
    labels_train = labels[:, :index_validation]
    labels_val   = labels[:, index_validation:]
    return inputs_train, inputs_val, labels_train, labels_val


def scale_series(data, minmaxscale=None):
    if not minmaxscale:
        minimum = torch.amin(data, dim=1)
        maximum = torch.amax(data, dim=1)
        minmaxscale = (minimum, maximum)
    data_scaled = (data - minmaxscale[0]) / (minmaxscale[1] - minmaxscale[0])
    return data_scaled, minmaxscale


def unscale_series(data, minmaxscale):
    return data * (minmaxscale[1] - minmaxscale[0]) + minmaxscale[0]


def mse(pred, label):
    return torch.mean(torch.pow(label - pred, 2), dim=(0, 1))


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
# Hyperparameters
# =============================================================================

index_warmup     = 50
index_validation = 1398
epochs           = 500
ninputs          = 3
nhidden          = 32
nlayers          = 1

# =============================================================================
# Data loading and scaling
# =============================================================================

inputs_train, inputs_val, labels_train, labels_val = load_datafile(
    'dataframe.pkl', index_validation
)

inputs_train, inputscales = scale_series(inputs_train)
labels_train, labelscales = scale_series(labels_train)
inputs_val,   inputscales = scale_series(inputs_val, inputscales)
labels_val,   labelscales = scale_series(labels_val, labelscales)

# =============================================================================
# Load model and weights
# =============================================================================

model = LSTMModel(ninputs, nhidden, 1, nlayers, 0)
model.load_state_dict(torch.load('weights.pkl'))
model.eval()

# =============================================================================
# Baseline (shared by both methods)
# =============================================================================
# Training-data feature means broadcast to the validation length.
# Represents "average meteorological forcing" — the reference state against
# which each variable's contribution is measured.
# Completeness guarantee: sum(attributions) = F(x) - F(baseline)

# Mixed baseline:
#   Rain (v=0)           → 0  (no rainfall as reference)
#   PET  (v=1)           → training mean  (climatological average)
#   Rain 30-day (v=2)    → 0  (no antecedent rainfall as reference)
train_mean = inputs_train.mean(dim=1, keepdim=True)       # (1, 1, 3)
baseline   = torch.zeros_like(inputs_val)                 # start from all zeros
baseline[:, :, 1] = train_mean[:, :, 1].expand_as(        # overwrite PET column only
    baseline[:, :, 1])
baseline = baseline.detach()


feature_names = ['Rain', 'PET', 'Rain (30-day mean)']
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
feature_importance      = attributions[0, 0, :].detach().numpy()  # (3,)
feature_importance_norm = feature_importance / feature_importance.sum()

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

temporal_importance = np.zeros((n_out, 3))
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
    pred_val = model(inputs_val)

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
colors = ['#4C72B0', '#DD8452', '#55A868']

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
ax.set_ylim(0, max(feature_importance_norm) * 1.3)
ax.set_yticks([])
ax.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, feature_importance_norm):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

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
ax3.plot(t_axis, inputs_val_phys[:, 1], color=colors[1],
         label=feature_names[1], linewidth=1.4)
ax3.plot(t_axis, inputs_val_phys[:, 2], color=colors[2],
         label=feature_names[2], linewidth=1.4)
ax3.set_ylabel('mm/day')
ax3.set_title('(c) Observed Meteorological Inputs',
              fontsize=11, fontweight='bold', loc='left')
ax3.legend(loc='upper right', framealpha=0.8)
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
plt.savefig('feature_importance_ig.png', dpi=150)
plt.show()
