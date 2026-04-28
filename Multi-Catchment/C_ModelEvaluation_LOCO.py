#This script recreates the LOCO model from the saved parameters 
#Plots predictions for an independent series and calculates metrics 
#Do this only for Group 2 to compare to other models created throughout the course

from B_lstm_forecaster import load_catchment, scale_series, unscale_series, nse, mse
from B_lstm_forecaster import catchments, datafolder, feature_cols
from model import LSTMModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf 
import datetime as dt 
import os 
import pandas as pd 

#Create folder where all figures are saved, same with weights 
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


#Hyperparameters - these MUST match training 
ninputs = len(feature_cols)
nhidden = 64
nlayers = 1

#Load all catchments 
catchment_data = [load_catchment(c) for c in catchments]
test_catchment_name = 'Group2'
test_catchment = next(c for c in catchment_data if c['name'] == test_catchment_name)
train_catchments = [c for c in catchment_data if c['name'] != test_catchment_name]

# Match the log-space training: log-transform labels before scaling
non_zero = 1e-3
for c in train_catchments:
    c['labels_train'] = torch.log(c['labels_train'] + non_zero)
    c['labels_val']   = torch.log(c['labels_val']   + non_zero)
    c['labels_test']  = torch.log(c['labels_test']  + non_zero)
test_catchment['labels_train'] = torch.log(test_catchment['labels_train'] + non_zero)
test_catchment['labels_val']   = torch.log(test_catchment['labels_val']   + non_zero)
test_catchment['labels_test']  = torch.log(test_catchment['labels_test']  + non_zero)

#Set number of static inputs 
n_static = 8

#Compute global scaling from all training data (must match training data)
#This is for the 5 training catchments only (do not want to leak Group 2)
all_inputs_train = torch.cat([c['inputs_train'] for c in train_catchments], dim=1)
all_labels_train = torch.cat([c['labels_train'] for c in train_catchments], dim=1)
_, inputscales = scale_series(all_inputs_train, per_feature = True)
_, labelscales = scale_series(all_labels_train)

#Apply the scale to Group 2 catchment  
test_catchment['inputs_train'], _ = scale_series(test_catchment['inputs_train'], inputscales)
test_catchment['inputs_val'],   _ = scale_series(test_catchment['inputs_val'],   inputscales)
test_catchment['inputs_test'],  _ = scale_series(test_catchment['inputs_test'],  inputscales)
test_catchment['labels_train'], _ = scale_series(test_catchment['labels_train'], labelscales)
test_catchment['labels_val'],   _ = scale_series(test_catchment['labels_val'],   labelscales)
test_catchment['labels_test'],  _ = scale_series(test_catchment['labels_test'],  labelscales)

#Scale the static attributes used as input  
all_static = torch.stack([c['static'] for c in train_catchments])
_, staticscales = scale_series(all_static, per_feature=True)
test_catchment['static_scaled'], _ = scale_series(test_catchment['static'], staticscales)

#Load the trained model (done in B script - LOCO)
model = LSTMModel(ninputs,nhidden,1,nlayers,0, n_static = n_static)
#Load the Weights - pick which .csv based on the forecaster used!! or use the logspace weights 
model.load_state_dict(torch.load(os.path.join(weights_dir, f'weights_LOCO_{test_catchment_name}_logspace.csv')))

model.eval()

#Evaluate each Group 2 catchment 
c = test_catchment 
name = c['name']
print(f'\n=== LOCO evaluation: {name} (held-out) ===')

#Join all datasets back together to ensure all get a warm up period 
inputs_all = torch.cat([c['inputs_train'], c['inputs_val'], c['inputs_test']], dim = 1)
labels_all = torch.cat([c['labels_train'], c['labels_val'], c['labels_test']], dim = 1)

#Run model on full sequence 
with torch.no_grad():
    pred_all = model(inputs_all, c['static_scaled'].unsqueeze(0))

#In LOCO, the model never trained on or validated against ANY part of this catchment.
#The whole held-out series is the test set, so the conventional reporting is one NSE
#computed on the full series (excluding warmup), not three NSEs on arbitrary slices.
window_warmup = 182  # must match training script

#Full held-out series, post-warmup -- this is the headline LOCO result
pred_full = pred_all[:, window_warmup:]
labels_full = labels_all[:, window_warmup:]
rain_full = inputs_all[0, window_warmup:, 0]

#Last-10% slice -- only kept so we can compare apples-to-apples against the
#single-catchment / non-LOCO models that were tested on the same time window
n_train = c['inputs_train'].shape[1]
n_val = c['inputs_val'].shape[1]
pred_test = pred_all[:, n_train+n_val:]

#Unscale output - use when not in log tranformed space!
# flowpred_full = unscale_series(pred_full[0,:], labelscales).numpy()
# flowobs_full  = unscale_series(labels_full[0,:], labelscales).numpy()
# rainseries_full = unscale_series(rain_full, inputscales).numpy()
# flowpred_test = unscale_series(pred_test[0,:], labelscales).numpy()
# flowobs_test  = unscale_series(c['labels_test'][0,:], labelscales).numpy()
# rainseries_test = unscale_series(c['inputs_test'][0,:,0], inputscales).numpy()

#Unscale output - use when in log transformed space!
flowpred_full = np.exp(unscale_series(pred_full[0,:], labelscales).numpy()) - non_zero
flowobs_full  = np.exp(unscale_series(labels_full[0,:], labelscales).numpy()) - non_zero
rainseries_full = unscale_series(inputs_all, inputscales)[0, window_warmup:, 0].numpy()

flowpred_test = np.exp(unscale_series(pred_test[0,:], labelscales).numpy()) - non_zero
flowobs_test  = np.exp(unscale_series(c['labels_test'][0,:], labelscales).numpy()) - non_zero
rainseries_test = unscale_series(c['inputs_test'], inputscales)[0, :, 0].numpy()

######### EVALUATION ###########
#Headline LOCO result: NSE on the full held-out catchment, post-warmup
nse_full = nse(flowobs_full, flowpred_full)
print(f'LOCO Test NSE (full held-out series, post-warmup): {nse_full:.3f}')

fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(rainseries_full, linewidth=0.5)
ax[0].set_ylabel('Rainfall [mm/d]')
ax[0].set_title(f'{name} LOCO - Held-out catchment (NSE = {nse_full:.3f})')
ax[1].plot(flowobs_full, label='Observed')
ax[1].plot(flowpred_full, label='Predicted')
ax[1].set_ylabel('Flow [mm/d]')
ax[1].set_xlabel('Time [days]')
ax[1].legend()
fig.savefig(os.path.join(figures_dir, f'{name}_LOCO_predictions_logspace.png'), dpi=150)
plt.close()

#Cross-model comparison: NSE and plot on the last-10% slice that other course models were tested on
nse_test = nse(flowobs_test, flowpred_test)
print(f'Cross-model comparison NSE (last 10% slice): {nse_test:.3f}')

fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(rainseries_test, linewidth=0.5)
ax[0].set_ylabel('Rainfall [mm/d]')
ax[0].set_title(f'{name} LOCO - Test period for cross-model comparison (NSE = {nse_test:.3f})')
ax[1].plot(flowobs_test, label='Observed')
ax[1].plot(flowpred_test, label='Predicted')
ax[1].set_ylabel('Flow [mm/d]')
ax[1].set_xlabel('Time [days]')
ax[1].legend()
fig.savefig(os.path.join(figures_dir, f'{name}_LOCO_test_predictions_logspace.png'), dpi=150)
plt.close()

#Check Residuals on the full held-out series (must happen in transformed space)
residuals = np.log(flowobs_full + non_zero) - np.log(flowpred_full + non_zero)

fig, ax = plt.subplots()
ax.hist(residuals, bins=50)
ax.axvline(0, color='red', linestyle='--', label='Zero')
ax.set_xlabel('Log-Transformed Residual [mm/d]')
ax.set_ylabel('No. of occurences')
ax.set_title(f'{name} LOCO - Residual Histogram (full held-out series)')
ax.legend()
fig.savefig(os.path.join(figures_dir, f'{name}_LOCO_residuals_histogram_logspace.png'), dpi=150)
plt.close()

fig, ax = plt.subplots()
ax.plot(residuals, color='steelblue', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Time [days]')
ax.set_ylabel('Log-Transformed Residual [mm/d]')
ax.set_title(f'{name} LOCO - Residuals over time')
fig.savefig(os.path.join(figures_dir, f'{name}_LOCO_Residuals_timeseries_logspace.png'), dpi=150)
plt.close()

fig, ax = plt.subplots()
plot_acf(residuals, lags=60, ax=ax)
ax.set_xlabel('Lag [days]')
ax.set_ylabel('Autocorrelation')
ax.set_title(f'{name} - ACF of Residuals (LOCO)')
fig.savefig(os.path.join(figures_dir, f'{name}_LOCO_residuals_acf_logspace.png'), dpi=150)
plt.close()

#Save Group 2 test predictions in m3/s for model comparison 
group2_df = pd.read_csv(os.path.join(datafolder, 'Group2', 'LSTM_dataframe.csv'), sep=',', index_col=0, parse_dates=True)
n = len(group2_df)
test_index = int(n * 0.90)
dates_test = group2_df.index[test_index:]

CATCHMENT_AREA_M2 = 140_000_000  # 140 km² = 14,000 ha

test_predictions = pd.DataFrame({
    'predicted_m3s': flowpred_test * CATCHMENT_AREA_M2 / 1000 / 86400,
    'observed_m3s':  flowobs_test  * CATCHMENT_AREA_M2 / 1000 / 86400,
}, index=dates_test)
test_predictions.index.name = 'date'
#test_predictions.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data/Group2/Group2_LOCO_test_predictions.csv'))
test_predictions.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'Data/Group2/Group2_LOCO_logspace_test_predictions.csv'))
print(f'\nSaved: Data/Group2/Group2_LOCO_test_predictions.csv')


