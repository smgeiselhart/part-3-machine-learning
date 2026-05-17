import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

from model import LSTMModel
import sklearn
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def scale_series(data, meanstdscale=None):
    if not meanstdscale:
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        meanstdscale = (mean, std)
    data_scaled = (data - meanstdscale[0]) / meanstdscale[1]
    return (data_scaled, meanstdscale)

def unscale_series(data, meanstdscale):
    mean = meanstdscale[0]
    std = meanstdscale[1]
    data_unscaled = data * std + mean
    return (data_unscaled)

def mse(pred, label):
    residual = torch.pow(label - pred, 2)
    mse = torch.mean(residual, dim=(0, 1))
    return (mse)

def mse_masked(pred, label):
    mask = ~torch.isnan(label)
    #print(label)
    #print("Valid points:", mask.sum().item())

    pred_safe = torch.where(mask, pred, torch.zeros_like(pred))
    label_safe = torch.where(mask, label, torch.zeros_like(label))

    residual = (pred_safe - label_safe) ** 2 * mask.float()

    n_valid = mask.sum(dim=1, keepdim=True).float()
    n_valid = torch.clamp(n_valid, min=1.0)

    return (residual.sum(dim=1, keepdim=True) / n_valid).mean()

def nse(pred, label):
    mask = ~torch.isnan(label)
    p = pred[mask]
    l = label[mask]
    return 1 - torch.sum((p - l)**2) / torch.sum((l - torch.mean(l))**2)



def scale_series2(data, minmaxscale=None):
    if minmaxscale is None:  # also fix: `not minmaxscale` fails on tensors
        # Use nan-safe min/max along the time dimension
        minimum = torch.tensor(np.nanmin(data.numpy(), axis=1), dtype=data.dtype)
        maximum = torch.tensor(np.nanmax(data.numpy(), axis=1), dtype=data.dtype)
        minmaxscale = (minimum.unsqueeze(1), maximum.unsqueeze(1))
    data_scaled = (data - minmaxscale[0]) / (minmaxscale[1] - minmaxscale[0])
    return data_scaled, minmaxscale

def scale_labels(data, meanstdscale=None):
    if meanstdscale is None:
        arr = data.numpy()
        mean = np.nanmean(arr)
        std  = np.nanstd(arr)
        meanstdscale = (
            torch.tensor(mean, dtype=data.dtype),
            torch.tensor(std,  dtype=data.dtype),
        )
    data_scaled = (data - meanstdscale[0]) / meanstdscale[1]
    return data_scaled, meanstdscale

class ModelDesign:
    def __init__(self,datafile,listInput,model,nameSave='weight', lr=1e-3, epochs=1000, index_warmup=300,index_validation=11000,transforms=None):
        self.datafile=datafile
        self.listInput=listInput
        self.model = model
        self.epochs = epochs
        self.index_warmup = index_warmup
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.nameSave=nameSave
        self.modelPlot=None
        self.index_validation=index_validation
        self.transforms = transforms or {}  # e.g. {'precip': torch.sqrt}

    
    def apply_transforms(self, inputs):
        inputs = inputs.clone()
        for name, fn in self.transforms.items():
            if name in self.listInput:
                idx = self.listInput.index(name)
                inputs[:, :, idx] = fn(inputs[:, :, idx].clamp(min=0))
        return inputs
        

    def load(self):
        data = pd.read_csv(self.datafile, index_col=0, parse_dates=True)
        data.interpolate(method='linear', inplace=True)  # ← add this

        inputs = np.stack([
            data[elt].to_numpy() for elt in self.listInput
            
        ], axis=1)

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(data['discharge'].to_numpy(), dtype=torch.float32)

        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        n_steps = labels.shape[1]
        idx = min(self.index_validation, n_steps)
        print(labels)
        return (
            inputs[:, :idx, :],
            inputs[:, idx:, :],
            labels[:, :idx],
            labels[:, idx:]
        )

    def scale(self, inputs_train, inputs_val, labels_train, labels_val):
        inputs_train, self.input_scale = scale_series(inputs_train)
        labels_train, self.label_scale = scale_labels(labels_train)   # ← changed

        inputs_val, _ = scale_series(inputs_val, self.input_scale)
        labels_val, _ = scale_labels(labels_val, self.label_scale)    # ← changed

        return inputs_train, inputs_val, labels_train, labels_val
    
    def train(self, inputs_train, labels_train, inputs_val, labels_val):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")

            # TRAIN
            self.model.train()
            self.optimizer.zero_grad()

            pred_train = self.model(inputs_train)
            loss_train = mse_masked(
                pred_train[:, self.index_warmup:],
                labels_train[:, self.index_warmup:]
            )
            print(loss_train)
            loss_train.backward()
            self.optimizer.step()

            # VALIDATION
            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(inputs_val)
                loss_val = mse_masked(
                    pred_val[:, self.index_warmup:],
                    labels_val[:, self.index_warmup:]
                )

            self.history['train_loss'].append(loss_train.item())
            self.history['val_loss'].append(loss_val.item())

            # SAVE BEST
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                torch.save(self.model.state_dict(), self.nameSave+'.pkl')
        
        return self.history
    
    def predict(self, inputs):
        self.model.load_state_dict(torch.load(self.nameSave + '.pkl'))
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)
        
    def _load_full(self):
        data = pd.read_csv(self.datafile, index_col=0, parse_dates=True)
        data.interpolate(method='linear', inplace=True)  # ← add this
        inputs = np.stack([data[elt].to_numpy() for elt in self.listInput], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(data['discharge'].to_numpy(), dtype=torch.float32).unsqueeze(0)
        return inputs, None, labels, None
    
    def predict_full(self, inputs_full, labels_full):
        """
        Run inference on the entire dataset.
        Refits scalers on the full data (no leakage concern for plotting purposes).
        """
        from sklearn.preprocessing import StandardScaler

        inputs = self.apply_transforms(inputs_full.clone())

        # Refit scalers on full data just for plotting
        T, F = inputs.squeeze(0).shape[0], inputs.squeeze(0).shape[1]
        inp_np  = inputs.squeeze(0).numpy()          # (T, F)
        lab_np  = labels_full.squeeze(0).numpy()     # (T,)

        inp_scaler = StandardScaler().fit(inp_np)
        lab_scaler = StandardScaler().fit(lab_np.reshape(-1, 1))

        inputs_scaled = torch.tensor(
            inp_scaler.transform(inp_np), dtype=torch.float32
        ).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(inputs_scaled).squeeze()

        preds = lab_scaler.inverse_transform(
            preds_scaled.numpy().reshape(-1, 1)
        ).flatten()

        return preds
    
    def cross_validate_sliding3(self, n_folds=5, segment_warmup=90,index_test=0.9,n_passes=3):
        """
        Sliding-window CV: validation window moves right each fold.
        Everything outside the val window is trained on, split into
        separate segments each with their own warmup period.
        
        Fold k: segments before val + segments after val → train
                segment k → val
        """
        inputs_full, _, labels_full, _ = self._load_full()
        n_steps   = labels_full.shape[1]
        test_start = int(n_steps * index_test)
        fold_size = test_start // (n_folds + 1)  # uses test_start, not n_steps
        inputs_full = inputs_full[:, :test_start, :]
        labels_full = labels_full[:, :test_start]

        original_weights = copy.deepcopy(self.model.state_dict())
        original_lr      = self.optimizer.param_groups[0]['lr']

        results = []

        for fold in range(n_folds):
            val_start = (fold + 1) * fold_size  # leave one segment at the start
            val_end   = val_start + fold_size
            if val_end > test_start:
                break

            print(f"\n{'='*60}")
            print(f"Fold {fold}: val[{val_start}:{val_end}]")
            print(f"{'='*60}")

            # ── Build list of training segments (before + after val) ──
            train_segments = []

            # Segments before val window
            seg_start = 0
            while seg_start + fold_size <= val_start:
                seg_end = min(seg_start + fold_size, val_start)
                train_segments.append((seg_start, seg_end, 'pre'))
                seg_start += fold_size

            # Segments after val window
            seg_start = val_end
            while seg_start < test_start-1:
                seg_end = min(seg_start + fold_size, test_start)
                train_segments.append((seg_start, seg_end, 'post'))
                seg_start += fold_size

            

            print(f"  Training segments: {[(s,e) for s,e,_ in train_segments]}")

            # ── Reset model for this fold ──
            self.model.load_state_dict(copy.deepcopy(original_weights))
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=original_lr
            )

            # ── Scale using only pre-val training data to avoid leakage ──
            pre_val_inputs = inputs_full[:, :val_start, :]
            pre_val_labels = labels_full[:, :val_start]
            pre_val_inputs = self.apply_transforms(pre_val_inputs)
            _, self.input_scale = scale_series(pre_val_inputs)
            _, self.label_scale = scale_labels(pre_val_labels)

            # ── Train on each segment independently ──
            fold_name  = f"{self.nameSave}_cv_fold{fold}"
            orig_name  = self.nameSave
            self.nameSave = fold_name
            fold_history = []

            for pass_idx in range(n_passes):
                print(f"  Pass {pass_idx+1}/{n_passes}")
                pass_losses = []

                for seg_idx, (seg_start, seg_end, pos) in enumerate(train_segments):
                    seg_len = seg_end - seg_start
                    if seg_len <= segment_warmup:
                        continue

                    inp_seg = self.apply_transforms(
                        inputs_full[:, seg_start:seg_end, :].clone()
                    )
                    lab_seg = labels_full[:, seg_start:seg_end].clone()
                    inp_seg, _ = scale_series(inp_seg, self.input_scale)
                    lab_seg, _ = scale_labels(lab_seg, self.label_scale)

                    seg_losses = []
                    for epoch in range(self.epochs):
                        self.model.train()
                        self.optimizer.zero_grad()
                        pred = self.model(inp_seg)
                        """loss = mse_masked(
                            pred[:, segment_warmup:],
                            lab_seg[:, segment_warmup:]
                        )"""
                        
                        loss = -nse(pred[:, segment_warmup:], lab_seg[:, segment_warmup:])
                        loss.backward()
                        self.optimizer.step()
                        seg_losses.append(loss.item())

                    pass_losses.append({
                        'seg_idx':   seg_idx,
                        'seg_start': seg_start,
                        'seg_end':   seg_end,
                        'pos':       pos,
                        'losses':    seg_losses,
                    })

                fold_history.append({
                    'pass_idx':    pass_idx,
                    'segments':    pass_losses,
                    'final_losses': [s['losses'][-1] for s in pass_losses],
                })
            # ── Evaluate on validation segment ──
            inp_val = self.apply_transforms(
                inputs_full[:, val_start:val_end, :].clone()
            )
            lab_val = labels_full[:, val_start:val_end].clone()

            inp_val, _ = scale_series(inp_val, self.input_scale)
            lab_val, _ = scale_labels(lab_val, self.label_scale)

            self.model.eval()
            with torch.no_grad():
                pred_val   = self.model(inp_val)
                val_loss   = -nse(
                    pred_val[:, segment_warmup:],
                    lab_val[:, segment_warmup:]
                )

            # ── Full prediction for plotting ──
            # ── Full prediction using THIS fold's scalers ──
            inp_full = self.apply_transforms(inputs_full.clone())
            inp_full_scaled, _ = scale_series(inp_full, self.input_scale)

            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(inp_full_scaled)

            preds_full = unscale_series(pred_scaled.squeeze(-1), self.label_scale).squeeze(0).numpy()

            results.append({
                'fold':          fold,
                'val_start':     val_start,
                'val_end':       val_end,
                'train_segments': train_segments,
                'val_loss':      val_loss.item(),
                'preds_full':    preds_full,
                'fold_history':   fold_history,  
            })

            self.nameSave  = orig_name
            print(f"  → val_loss = {val_loss.item():.4f}")
            torch.save(
                self.model.state_dict(),
                f"{orig_name}_fold{fold}_other.pkl"  # ← orig_name, not self.nameSave
            )       

        self.plot_cv_results_sliding(inputs_full, labels_full, results)
        summary = pd.DataFrame([{
            'fold':           r['fold'],
            'val_start':      r['val_start'],
            'val_end':        r['val_end'],
            'val_loss':       r['val_loss'],
            'train_segments': str(r['train_segments']),  # stringify for CSV
        } for r in results])
        summary.to_csv(f"{self.nameSave}_cv_results.csv", index=False)
        print(f"Saved CV summary → {self.nameSave}_cv_results.csv")

        # ── Save model weights per fold ──
        
        return results
    
    def plot_cv_test(self, results, index_test=0.90):
        import matplotlib.patches as mpatches

        inputs_full, _, labels_full, _ = self._load_full()
        n_steps    = labels_full.shape[1]
        test_start = int(n_steps * index_test)

        inputs_test = inputs_full[:, test_start:, :]
        labels_test = labels_full[:, test_start:]

        data     = pd.read_csv(self.datafile, index_col=0, parse_dates=True)
        dates    = data.index[test_start:]
        observed = labels_test.squeeze(0).numpy()

        colors = plt.cm.tab10.colors
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{self.nameSave} — Per-fold predictions on test set", fontsize=13)

        rainfall_col = self.listInput[0]
        rainfall = data[rainfall_col].to_numpy()[test_start:]
        axes[0].bar(dates, rainfall, color='steelblue', width=1.0, alpha=0.6, label=rainfall_col)
        axes[0].invert_yaxis()
        axes[0].set_ylabel('Rainfall')
        axes[0].legend(loc='upper right')

        axes[1].plot(dates, observed, color='black', lw=1.5, label='Observed', zorder=10)
        legend_handles = [mpatches.Patch(color='black', label='Observed')]
        nse_scores = []

        for res in results:
            fold      = res['fold']
            color     = colors[fold % len(colors)]
            fold_file = f"{self.nameSave}_fold{fold}_other.pkl"

            if not os.path.exists(fold_file):
                print(f"  Warning: {fold_file} not found, skipping fold {fold}")
                continue

            # ── Refit scalers (pre-val only) ──
            val_start = res['val_start']
            pre_val_inputs = self.apply_transforms(inputs_full[:, :val_start, :].clone())
            pre_val_labels = labels_full[:, :val_start]
            _, self.input_scale = scale_series(pre_val_inputs)
            _, self.label_scale = scale_labels(pre_val_labels)

            # ── Predict on test ──
            inp_test_t = self.apply_transforms(inputs_test.clone())
            inp_test_scaled, _ = scale_series(inp_test_t, self.input_scale)
            self.model.load_state_dict(torch.load(fold_file, map_location='cpu'))
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(inp_test_scaled)

            pred = unscale_series(pred_scaled.squeeze(-1), self.label_scale).squeeze(0).numpy()

            # ── Correct NSE (mask both) ──
            mask  = ~np.isnan(observed) & ~np.isnan(pred)
            nse   = 1 - np.sum((pred[mask] - observed[mask])**2) / np.sum((observed[mask] - np.mean(observed[mask]))**2)
            nse_scores.append((fold, nse))

            axes[1].plot(dates, pred, color=color, lw=1.2, alpha=0.85)
            legend_handles.append(mpatches.Patch(
                color=color,
                label=f'Fold {fold} — val[{val_start}:{res["val_end"]}]  NSE={nse:.3f}'
            ))

        axes[1].set_ylabel('Discharge')
        axes[1].set_xlabel('Date')
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        axes[1].legend(handles=legend_handles, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{self.nameSave}_test_predictions.png", dpi=150, bbox_inches='tight')
        plt.show()

        print("\nTest NSE per fold:")
        for fold, nse in nse_scores:
            print(f"  Fold {fold}: NSE = {nse:.4f}")
        print(f"  Mean NSE: {np.mean([n for _, n in nse_scores]):.4f}")
        return nse_scores
    
    def plot_fold_performance(self, results, index_test=0.90):
        inputs_full, _, labels_full, _ = self._load_full()
        n_steps    = labels_full.shape[1]
        test_start = int(n_steps * index_test)

        inputs_test = inputs_full[:, test_start:, :]
        labels_test = labels_full[:, test_start:]
        obs_test    = labels_test.squeeze(0).numpy()
        

        def nse_numpy(pred, obs):
            mask = ~np.isnan(obs) & ~np.isnan(pred)
            p, o = pred[mask], obs[mask]
            return 1 - np.sum((p - o)**2) / np.sum((o - np.mean(o))**2)

        fold_ids, train_nses, val_nses, test_nses = [], [], [], []

        for res in results:
            fold      = res['fold']
            fold_file = f"{self.nameSave}_fold{fold}_other.pkl"

            if not os.path.exists(fold_file):
                print(f"  Warning: {fold_file} not found, skipping fold {fold}")
                continue

            # ── Refit fold scalers (pre-val only) ──
            val_start = res['val_start']
            val_end   = res['val_end']
            pre_val_inputs = self.apply_transforms(inputs_full[:, :val_start, :].clone())
            pre_val_labels = labels_full[:, :val_start]
            _, self.input_scale = scale_series(pre_val_inputs)
            _, self.label_scale = scale_labels(pre_val_labels)

            

            # ── Load fold weights ──
            self.model.load_state_dict(torch.load(fold_file, map_location='cpu'))
            self.model.eval()

            def predict_period(inp):
                inp_t = self.apply_transforms(inp.clone())
                inp_scaled, _ = scale_series(inp_t, self.input_scale)
                with torch.no_grad():
                    pred_scaled = self.model(inp_scaled)
                print(f"pred_scaled shape: {pred_scaled.shape}")
                print(f"label_scale mean shape: {self.label_scale[0].shape}")
                return unscale_series(pred_scaled.squeeze(-1), self.label_scale).squeeze(0).numpy()

            # ── Training NSE (all pre+post segments combined) ──
            train_preds, train_obs = [], []
            for seg_start, seg_end, _ in res['train_segments']:
                p = predict_period(inputs_full[:, seg_start:seg_end, :])
                o = labels_full[:, seg_start:seg_end].squeeze(0).numpy()
                train_preds.append(p)
                train_obs.append(o)
            train_nse = nse_numpy(np.concatenate(train_preds), np.concatenate(train_obs))

            # ── Validation NSE ──
            val_pred = predict_period(inputs_full[:, val_start:val_end, :])
            val_obs  = labels_full[:, val_start:val_end].squeeze(0).numpy()
            val_nse  = nse_numpy(val_pred, val_obs)

            # ── Test NSE ──
            test_pred = predict_period(inputs_test)
            test_nse  = nse_numpy(test_pred, obs_test)

            fold_ids.append(fold)
            train_nses.append(train_nse)
            val_nses.append(val_nse)
            test_nses.append(test_nse)
            

            print(f"Fold {fold}: train={train_nse:.3f}  val={val_nse:.3f}  test={test_nse:.3f}")

        # ── Plot ──
        x = np.arange(len(fold_ids))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, train_nses, width, label='Train NSE', color='steelblue', alpha=0.8)
        ax.bar(x,         val_nses,   width, label='Val NSE',   color='coral',     alpha=0.8)
        ax.bar(x + width, test_nses,  width, label='Test NSE',  color='seagreen',  alpha=0.8)

        ax.axhline(np.mean(val_nses),  color='coral',     lw=1.2, ls='--', alpha=0.7, label=f'Mean val={np.mean(val_nses):.3f}')
        ax.axhline(np.mean(test_nses), color='seagreen',  lw=1.2, ls='--', alpha=0.7, label=f'Mean test={np.mean(test_nses):.3f}')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
        ax.set_ylabel('NSE')
        ax.set_title(f'{self.nameSave} — NSE per fold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.nameSave}_fold_nse.png", dpi=150, bbox_inches='tight')
        plt.show()
    

listInput_friend = ['precip_30d', 'precip_7d','precip_90d','precip_surplus',
                    'temp', 'groundwater', 'melt']

model_friend = LSTMModel(
    input_size=7, hidden_size=64, output_size=1,
    num_layers=1, dropout=0.2   # ← her dropout
)

# Load data once just to get index_validation
data_temp = pd.read_csv('Data/LSTM_dataframe.csv', index_col=0, parse_dates=True)
index_validation_friend = int(len(data_temp) * 0.65)
del data_temp

model_friend_design = ModelDesign(
    'Data/LSTM_dataframe.csv',
    listInput_friend,
    model_friend,
    nameSave='Cross_Validation_model',
    lr=1e-3,
    epochs=200,
    index_warmup=90,
    index_validation=index_validation_friend,
    transforms={
        'precip_30d':    lambda x: torch.log1p(x),
        'precip_7d':     lambda x: torch.log1p(x),
        'precip_30d':lambda x: torch.log1p(x)
    }
)

# ── Run sliding CV ──────────────────────────────────────────────────────
results = model_friend_design.cross_validate_sliding3(
    n_folds=5,
    segment_warmup=90,
    index_test=0.90
)
model_friend_design.plot_fold_performance(results, index_test=0.90)
nse_scores = model_friend_design.plot_cv_test(results, index_test=0.90)