import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from B_lstm_forecaster import load_catchment, scale_series, mse, feature_cols, catchments, datafolder
from model import LSTMModel

#Figures and weights directory 
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(weights_dir, exist_ok=True)

#################################################################
# Hyperparameters
window_warmup = 182           # 6 months
window_train = 365 * 4        # 4 years of training
window_total = window_warmup + window_train  # total window length per batch

epochs = 500
ninputs = len(feature_cols)
nhidden = 64
nlayers = 1
n_static = 8

#################################################################
# Load and scale data (same as before)
catchment_data = [load_catchment(c) for c in catchments]
all_inputs_train = torch.cat([c['inputs_train'] for c in catchment_data], dim=1)
all_labels_train = torch.cat([c['labels_train'] for c in catchment_data], dim=1)
_, inputscales = scale_series(all_inputs_train)
_, labelscales = scale_series(all_labels_train)

for c in catchment_data:
    c['inputs_train'], _ = scale_series(c['inputs_train'], inputscales)
    c['inputs_val'], _ = scale_series(c['inputs_val'], inputscales)
    c['inputs_test'], _ = scale_series(c['inputs_test'], inputscales)
    c['labels_train'], _ = scale_series(c['labels_train'], labelscales)
    c['labels_val'], _ = scale_series(c['labels_val'], labelscales)
    c['labels_test'], _ = scale_series(c['labels_test'], labelscales)

# Scale static attributes
all_static = torch.stack([c['static'] for c in catchment_data])
static_mean = all_static.mean(dim=0)
static_std = all_static.std(dim=0)
for c in catchment_data:
    c['static_scaled'] = (c['static'] - static_mean) / static_std

# Pre-stack all static attributes as a (n_catchments, n_static) tensor for batching
all_static_scaled = torch.stack([c['static_scaled'] for c in catchment_data])

#################################################################
# Model + optimizer + scheduler
model = LSTMModel(ninputs, nhidden, 1, nlayers, 0.2, n_static=n_static)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=1e-2,
    step_size_up=4, mode='triangular', cycle_momentum=False
)

best_val_loss = torch.tensor(float('inf'), dtype=torch.float32)
history = {'train_loss': [], 'val_loss': [], 'lr': []}

#################################################################
# Training loop with random window sampling
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Sample one random window from each catchment
    batch_inputs = []
    batch_labels = []
    for c in catchment_data:
        T = c['inputs_train'].shape[1]
        start = torch.randint(0, T - window_total, (1,)).item()
        batch_inputs.append(c['inputs_train'][:, start:start+window_total, :])
        batch_labels.append(c['labels_train'][:, start:start+window_total])

    # Stack into a batched tensor (n_catchments, window_total, n_features)
    batch_inputs = torch.cat(batch_inputs, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)

    # Single forward pass for the whole batch
    pred = model(batch_inputs, all_static_scaled)
    loss = mse(pred[:, window_warmup:], batch_labels[:, window_warmup:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    history['train_loss'].append(loss.item())
    history['lr'].append(scheduler.get_last_lr()[0])

    # Validation (same as before - full sequence per catchment)
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for c in catchment_data:
            inputs_trainval = torch.cat([c['inputs_train'], c['inputs_val']], dim=1)
            pred_trainval = model(inputs_trainval, c['static_scaled'].unsqueeze(0))
            pred_val = pred_trainval[:, c['inputs_train'].shape[1]:]
            total_val_loss += mse(pred_val, c['labels_val']).item()

    avg_val_loss = total_val_loss / len(catchment_data)
    history['val_loss'].append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(weights_dir, 'weights_random_windows.csv'))

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: train={loss.item():.4f}, val={avg_val_loss:.4f}')

#################################################################
# Plot training history
fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(history['train_loss'], label='Training loss')
axes[0].plot(history['val_loss'], label='Validation loss')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[1].plot(history['lr'], color='tab:orange')
axes[1].set_ylabel('Learning Rate')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, 'lstm_training_history_random_windows.png'), dpi=150)
plt.show()