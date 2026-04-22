#This script uses 5 catchments to train and tests on one catchment
#LOCO = leave one catchment out 

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


from B_lstm_forecaster import load_catchment, scale_series, unscale_series, mse, nse, feature_cols, catchments, datafolder
from model import LSTMModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#Create a figures, weights, and data directory 
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(weights_dir, exist_ok=True)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

#Add seed for reproducibility 
torch.manual_seed(42)
np.random.seed(42)

def run_fold(test_catchment_name):
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
    # Load all 6 catchments and split into 5 training and 1 test 
    catchment_data = [load_catchment(c) for c in catchments]
    train_catchments = [c for c in catchment_data if c['name'] != test_catchment_name]
    test_catchment = next(c for c in catchment_data if c['name'] == test_catchment_name)
    
    #Resplit the training catchment: 70% training, 30% validation 
    #They were imported with 65/25/10 split, but here a whole dataset for one catchment is used for test 
    val_frac = 0.30 
    for c in train_catchments: 
        full_inputs = torch.cat([c['inputs_train'], c['inputs_val'], c['inputs_test']], dim=1)
        full_labels = torch.cat([c['labels_train'], c['labels_val'], c['labels_test']], dim=1)
        T = full_inputs.shape[1]
        n_train = int(T * (1 - val_frac))
        c['inputs_train'] = full_inputs[:, :n_train, :]
        c['inputs_val']   = full_inputs[:, n_train:, :]
        c['labels_train'] = full_labels[:, :n_train]
        c['labels_val']   = full_labels[:, n_train:]

    #Held-out test catchment: full series, no split 
    test_inputs = torch.cat([test_catchment['inputs_train'], test_catchment['inputs_val'], test_catchment['inputs_test']], dim=1)
    test_labels = torch.cat([test_catchment['labels_train'], test_catchment['labels_val'], test_catchment['labels_test']], dim=1)

    #Compute input and label scales from the 5 training catchments ONLY 
    all_inputs_train = torch.cat([c['inputs_train'] for c in train_catchments], dim=1)
    all_labels_train = torch.cat([c['labels_train'] for c in train_catchments], dim=1)
    _, inputscales = scale_series(all_inputs_train)
    _, labelscales = scale_series(all_labels_train)

    #Apply the scale to the 5 training catchments 
    for c in train_catchments:
        c['inputs_train'], _ = scale_series(c['inputs_train'], inputscales)
        c['inputs_val'], _ = scale_series(c['inputs_val'], inputscales)
        c['labels_train'], _ = scale_series(c['labels_train'], labelscales)
        c['labels_val'], _ = scale_series(c['labels_val'], labelscales)

    #Apply this same training scale to the held-out test catchment 
    test_inputs_scaled, _ = scale_series(test_inputs, inputscales)
    test_labels_scaled, _ = scale_series(test_labels, labelscales)

    # Scale static attributes
    all_static = torch.stack([c['static'] for c in train_catchments])
    static_mean = all_static.mean(dim=0)
    static_std = all_static.std(dim=0)
    for c in train_catchments:
        c['static_scaled'] = (c['static'] - static_mean) / static_std
    test_static_scaled = (test_catchment['static'] - static_mean) / static_std

    # Pre-stack all static attributes as a (n_catchments, n_static) tensor for batching
    all_static_scaled = torch.stack([c['static_scaled'] for c in train_catchments])

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
        for c in train_catchments:
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
            for c in train_catchments:
                inputs_trainval = torch.cat([c['inputs_train'], c['inputs_val']], dim=1)
                pred_trainval = model(inputs_trainval, c['static_scaled'].unsqueeze(0))
                pred_val = pred_trainval[:, c['inputs_train'].shape[1]:]
                total_val_loss += mse(pred_val, c['labels_val']).item()

        avg_val_loss = total_val_loss / len(train_catchments)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, f'weights_LOCO_{test_catchment_name}.csv'))

        if epoch % 100 == 0:
            print(f'[{test_catchment_name}] Epoch {epoch}: train={loss.item():.4f}, val={avg_val_loss:.4f}')

    #################################################################
    # Load best weights for this fold and evaluate on held-out test catchment
    model.load_state_dict(torch.load(os.path.join(weights_dir, f'weights_LOCO_{test_catchment_name}.csv')))
    model.eval()
    with torch.no_grad():
        pred_test = model(test_inputs_scaled, test_static_scaled.unsqueeze(0))

    pred_phys = unscale_series(pred_test[0, window_warmup:], labelscales).numpy()
    obs_phys  = unscale_series(test_labels_scaled[0, window_warmup:], labelscales).numpy()
    test_nse  = nse(obs_phys, pred_phys)
    print(f'[{test_catchment_name}] Test NSE: {test_nse:.3f}')
    
    
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
    fig.savefig(os.path.join(figures_dir, f'lstm_training_history_LOCO_{test_catchment_name}.png'), dpi=150)

    plt.close(fig)

    return test_nse 

if __name__ == '__main__':
    fold_order = ['Group2'] + [c for c in catchments if c != 'Group2']
    results = {}
    for test_name in fold_order:
        print(f'\n=== LOCO fold: test = {test_name} ===')
        results[test_name] = run_fold(test_name)

    df = pd.DataFrame({'catchment': list(results.keys()),
                       'test_nse':  list(results.values())})
    print('\n' + df.to_string(index=False))
    df.to_csv(os.path.join(data_dir, 'loco_cv_results.csv'), index=False)