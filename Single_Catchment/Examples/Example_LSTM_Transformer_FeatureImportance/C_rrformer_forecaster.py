import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rrformer_model_v3 import RRFormer


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_features(datafile):
    """Load rainfall and flow"""
    d         = pd.read_pickle(datafile)
    rain      = d['rain'].to_numpy()
    rain_feat = np.stack([rain], axis=1).astype(np.float32)
    flow      = d['flow'].to_numpy().astype(np.float32)
    return rain_feat, flow


def scale(x, vmin=None, vmax=None):
    if vmin is None:
        vmin = x.min(axis=0)
        vmax = x.max(axis=0)
    return (x - vmin) / (vmax - vmin + 1e-8), vmin, vmax


def unscale(x_sc, vmin, vmax):
    return x_sc * (vmax - vmin) + vmin


def make_windows(rain_sc, flow_sc, window_len, pred_len, stride=1):
    """
    Slide a window of length window_len over the series.

    src   : rain_sc[i : i+window_len]               (window_len, n_feat)
    label : flow_sc[i+window_len-pred_len : i+window_len]   (pred_len,)

    No tgt is created — the model uses learned query embeddings instead.
    """
    N = len(flow_sc)
    src_l, lbl_l = [], []

    for i in range(0, N - window_len, stride):
        src_l.append(rain_sc[i : i + window_len])
        lbl_l.append(flow_sc[i + window_len - pred_len : i + window_len])

    return (torch.from_numpy(np.array(src_l)),
            torch.from_numpy(np.array(lbl_l)))


def nse_loss(pred, label, eps=0.1):
    """Smoothed NSE loss — minimising ≡ maximising Nash-Sutcliffe Efficiency."""
    mse = torch.mean((pred - label) ** 2)
    var = torch.mean((label - label.mean()) ** 2) + eps
    return mse / var


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

index_validation = 1398
window_len = 90     # days of rainfall context given to the encoder
pred_len   = 7      # prediction horizon (days)

epochs     = 200
batch_size = 64

ninputs         = 1 #    # number of input features
d_model         = 32
nhead           = 4
num_enc_layers  = 2
num_dec_layers  = 2
dim_feedforward = 64
dropout         = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# Load and scale data
# ─────────────────────────────────────────────────────────────────────────────

rain_feat, flow = load_features('dataframe.pkl')

rain_train = rain_feat[:index_validation]
flow_train = flow[:index_validation]
rain_val   = rain_feat[index_validation:]
flow_val   = flow[index_validation:]

rain_train_sc, rain_vmin, rain_vmax = scale(rain_train)
flow_train_sc, flow_vmin, flow_vmax = scale(flow_train)
rain_val_sc,  *_ = scale(rain_val,  rain_vmin, rain_vmax)
flow_val_sc,  *_ = scale(flow_val,  flow_vmin, flow_vmax)

# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window datasets
# ─────────────────────────────────────────────────────────────────────────────

src_train, lbl_train = make_windows(rain_train_sc, flow_train_sc, window_len, pred_len)
src_val,   lbl_val   = make_windows(rain_val_sc,   flow_val_sc,   window_len, pred_len)

# ─────────────────────────────────────────────────────────────────────────────
# Model, optimiser, training loop
# ─────────────────────────────────────────────────────────────────────────────

model = RRFormer(ninputs, d_model, nhead, num_enc_layers, num_dec_layers,
                 dim_feedforward, window_len, pred_len, dropout)

optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float('inf')
history       = {'train_loss': [], 'val_loss': []}
n_train       = src_train.shape[0]

for epoch in range(epochs):
    print(epoch)
    model.train()
    epoch_loss = 0.0
    perm = torch.randperm(n_train)

    for start in range(0, n_train, batch_size):
        idx  = perm[start : start + batch_size]
        optimizer.zero_grad()
        pred = model(src_train[idx])
        loss = nse_loss(pred, lbl_train[idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)

    history['train_loss'].append(epoch_loss / n_train)

    model.eval()
    with torch.no_grad():
        val_loss = nse_loss(model(src_val), lbl_val).item()
    history['val_loss'].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'rrformer_weights.pkl')

# ─────────────────────────────────────────────────────────────────────────────
# Training curve
# ─────────────────────────────────────────────────────────────────────────────

plt.figure()
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'],   label='Validation')
plt.xlabel('Epoch'); plt.ylabel('NSE loss')
plt.title('RR-Former training history')
plt.legend(); plt.tight_layout()
plt.savefig('rrformer_training_history.png', dpi=150)

# ─────────────────────────────────────────────────────────────────────────────
# Reload best checkpoint and reconstruct full time series
# ─────────────────────────────────────────────────────────────────────────────

model = RRFormer(ninputs, d_model, nhead, num_enc_layers, num_dec_layers,
                 dim_feedforward, window_len, pred_len, dropout)
model.load_state_dict(torch.load('rrformer_weights.pkl'))
model.eval()


def reconstruct_series(rain_sc, flow_vmin, flow_vmax):
    """Reconstruct full predicted flow by sliding non-overlapping pred_len windows."""
    N         = len(rain_sc)
    flow_pred = np.full(N, np.nan, dtype=np.float32)

    with torch.no_grad():
        i = 0
        while i + window_len <= N:
            src  = torch.from_numpy(rain_sc[i : i + window_len]).unsqueeze(0)
            pred = model(src)[0].numpy()                      # (pred_len,)
            flow_pred[i + window_len - pred_len : i + window_len] = pred
            i   += pred_len

    return unscale(flow_pred, flow_vmin, flow_vmax)


for split, rain_sc, flow_raw, title in [
    ('train', rain_train_sc, flow_train, 'RR-Former — Training period'),
    ('val',   rain_val_sc,   flow_val,   'RR-Former — Validation period'),
]:
    flow_pred_raw = reconstruct_series(rain_sc, flow_vmin, flow_vmax)
    rain_raw      = unscale(rain_sc, rain_vmin, rain_vmax)

    fig, ax = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
    ax[0].plot(rain_raw[:, 0], label='Rain (daily)')
    ax[0].set_ylabel('Rainfall'); ax[0].legend()
    ax[1].plot(flow_raw,      label='Observed')
    ax[1].plot(flow_pred_raw, label='Predicted', linestyle='--')
    ax[1].set_ylabel('Runoff'); ax[1].legend()
    ax[1].set_xlabel('Time step (days)')
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(f'rrformer_{split}_predictions.png', dpi=150)
