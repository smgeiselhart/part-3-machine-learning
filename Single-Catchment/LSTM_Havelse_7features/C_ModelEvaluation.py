#This script recreates the model from the saved parameters 
#Plots predictions for an independent series and calculates metrics 

from B_lstm_forecaster import load_datafile, scale_series, unscale_series, nse, mse
from model import LSTMModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf 
import datetime as dt 
import os 
import pandas as pd 

#Create folder where all figures are saved 
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

#Hyperparameters - these MUST match training 
ninputs = 7
nhidden = 64
nlayers = 1

#Load and scale data 
inputs_train,inputs_val,inputs_test,labels_train,labels_val,labels_test, index_validation = load_datafile('Data/LSTM_dataframe.csv')
inputs_train,inputscales = scale_series(inputs_train)
labels_train,labelscales = scale_series(labels_train)
inputs_val,inputscales = scale_series(inputs_val,inputscales)
labels_val,labelscales = scale_series(labels_val,labelscales)
inputs_test,inputscales = scale_series(inputs_test,inputscales)
labels_test,labelscales = scale_series(labels_test,labelscales)

#Join all datasets (train, val, test) back together to ensure all get a warmup period
inputs_all = torch.cat([inputs_train, inputs_val, inputs_test], dim=1)
labels_all = torch.cat([labels_train, labels_val, labels_test], dim=1)

#Run model on the full sequence
model = LSTMModel(ninputs,nhidden,1,nlayers,0)
model.load_state_dict(torch.load('weights.csv'))
model.eval()
with torch.no_grad():
    pred_all = model(inputs_all)

#Slice predictions into each dataset 
n_train = inputs_train.shape[1]
n_val = inputs_val.shape[1]
n_test = inputs_test.shape[1]

pred_train = pred_all[:, :n_train]
pred_val = pred_all[:, n_train:n_train+n_val]
pred_test = pred_all[:, n_train+n_val:]

#Unscale output
#training
flowpred_train = unscale_series(pred_train[0,:],labelscales).numpy()
flowobs_train = unscale_series(labels_train[0,:],labelscales).numpy()

#Validation 
flowpred_val = unscale_series(pred_val[0,:],labelscales).numpy()
flowobs_val = unscale_series(labels_val[0,:],labelscales).numpy()

#Test 
flowpred_test = unscale_series(pred_test[0,:],labelscales).numpy()
flowobs_test = unscale_series(labels_test[0,:],labelscales).numpy()

#Save predictions to be used in Diebold Mariano Test (separate script)
np.save('predictions_val.npy', flowpred_val)
np.save('observations_val.npy', flowobs_val)

######### EVALUATION ###########
#Calculate the NSE on training 
nse_train = nse(flowobs_train, flowpred_train)
print(f'Training NSE: {nse_train:3f}')

#want to plot raw precip for each period to show with predictions 
raw_data = pd.read_csv('Data/LSTM_dataframe.csv', index_col=0, parse_dates=True)
precip_daily = raw_data['precipitation'].values
precip_train = precip_daily[:n_train]
precip_val   = precip_daily[n_train:n_train+n_val]
precip_test  = precip_daily[n_train+n_val:]

#Add date slices to include in figures (showing no. days currently)
dates_train = raw_data.index[:n_train]
dates_val   = raw_data.index[n_train:n_train+n_val]
dates_test  = raw_data.index[n_train+n_val:]

#PLot training predictions
fig,ax = plt.subplots(nrows=2)
ax[0].plot(dates_train, precip_train)
ax[0].set_ylabel('Rainfall [mm/d]')
ax[0].set_title(f'Training period (NSE = {nse_train:.3f})')
ax[1].plot(dates_train, flowobs_train, label='Observed')
ax[1].plot(dates_train, flowpred_train, label='Predicted')
ax[1].set_ylabel('Flow [mm/d]')
ax[1].set_xlabel('Date')
ax[1].legend()
fig.savefig(os.path.join(figures_dir, 'lstm_training_predictions.png'), dpi=150)

#Calculate NSE on validation set, 1 = perfect, 0 = no better than predicting the mean
nse_val = nse(flowobs_val, flowpred_val)
print(f'Validation NSE: {nse_val:3f}')

#PLot Validation predictions 
fig,ax = plt.subplots(nrows=2)
ax[0].plot(dates_val, precip_val)
ax[0].set_ylabel('Rainfall [mm/d]')
ax[0].set_title(f'Validation period (NSE = {nse_val:.3f})')
ax[1].plot(dates_val, flowobs_val, label='Observed')
ax[1].plot(dates_val, flowpred_val, label='Predicted')
ax[1].set_ylabel('Flow [mm/d]')
ax[1].set_xlabel('Date')
ax[1].legend()
fig.savefig(os.path.join(figures_dir, 'lstm_validation_predictions.png'), dpi=150)

#Calcualte NSE on test period 
nse_test = nse(flowobs_test, flowpred_test)
print(f'Test NSE: {nse_test:.3f}')

#Plot Test Predictions 
fig,ax = plt.subplots(nrows=2)
ax[0].plot(dates_test, precip_test)
ax[0].set_ylabel('Rainfall [mm/d]')
ax[0].set_title(f'Test period (NSE = {nse_test:.3f})')
ax[1].plot(dates_test, flowobs_test, label='Observed')
ax[1].plot(dates_test, flowpred_test, label='Predicted')
ax[1].set_ylabel('Flow [mm/d]')
ax[1].set_xlabel('Date')
ax[1].legend()
fig.savefig(os.path.join(figures_dir, 'lstm_test_predictions.png'), dpi=150)

#Check Residuals 
residuals = flowobs_val - flowpred_val
var_names = ['precip_30d', 'precip_7d', 'precip_90d', 'precip_surplus', 'temp', 'groundwater', 'melt']

#Plot histogram of residuals on validation period to confirm if normally distributed
fig, ax = plt.subplots()
ax.hist(residuals, bins=50)
ax.axvline(0, color='red', linestyle='--', label='Zero')
ax.set_xlabel('Residual [mm/d]')
ax.set_ylabel('No. of occurences')
ax.legend()
fig.savefig(os.path.join(figures_dir, 'Histogram_residuals.png'), dpi=150)

#Plot residuals over time to check for white noise
fig, ax = plt.subplots()
ax.plot(residuals, color = 'steelblue', linewidth = 0.5)
ax.axhline(0, color = 'red', linestyle = '--', linewidth =1)
ax.set_xlabel('Time [days]')
ax.set_ylabel('Residual [mm/d]')
ax.set_title('Residuals over time')
fig.savefig(os.path.join(figures_dir, 'Residuals_timeseries.png'), dpi=150)

#PLot autocorrelation function 
fig, ax = plt.subplots()
plot_acf(residuals, lags=60, ax=ax)
ax.set_xlabel('Lag [days]')
ax.set_ylabel('Autocorrelation')
ax.set_title('ACF of residuals')
fig.savefig(os.path.join(figures_dir, 'residuals_acf.png'), dpi=150)

#Plot residuals vs each input feature
#A correlation between residuals and an input means the model isn't fully extracting that input's signal. 
#Computed on the validation period
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
axes = axes.flatten()
for i, fname in enumerate(var_names):
    feature_val = unscale_series(inputs_val[0,:,i], inputscales).numpy()
    r = np.corrcoef(feature_val, residuals)[0, 1]
    axes[i].scatter(feature_val, residuals, alpha=0.3, s=8)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[i].set_xlabel(fname)
    axes[i].set_ylabel('Residual [mm/d]')
    axes[i].set_title(f'{fname}  (r = {r:.3f})')
axes[7].axis('off')   # only 7 features, hide the 8th panel
fig.suptitle('Validation residuals vs input features', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, 'residuals_vs_inputs.png'), dpi=150)

#Create box & whisker plot of features 
data_bw = inputs_train.squeeze(0).numpy() #shape = (timesteps, 7)
fig, ax = plt.subplots(figsize = (10,5))
bp = ax.boxplot(data_bw, tick_labels = var_names, patch_artist = True, showfliers = True)
colors = ['#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#17becf']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
fig.savefig(os.path.join(figures_dir, 'boxandwhisker.png'), dpi=150)

#Combined plot: train, val, test predictions stacked vertically
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))

axes[0].plot(dates_train, flowobs_train, label='Observed', linewidth=0.8)
axes[0].plot(dates_train, flowpred_train, label='Predicted', linewidth=0.8)
axes[0].set_ylabel('Flow [mm/d]')
axes[0].set_title(f'Training (NSE = {nse_train:.3f})')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

axes[1].plot(dates_val, flowobs_val, label='Observed', linewidth=0.8)
axes[1].plot(dates_val, flowpred_val, label='Predicted', linewidth=0.8)
axes[1].set_ylabel('Flow [mm/d]')
axes[1].set_title(f'Validation (NSE = {nse_val:.3f})')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

axes[2].plot(dates_test, flowobs_test, label='Observed', linewidth=0.8)
axes[2].plot(dates_test, flowpred_test, label='Predicted', linewidth=0.8)
axes[2].set_ylabel('Flow [mm/d]')
axes[2].set_xlabel('Date')
axes[2].set_title(f'Test (NSE = {nse_test:.3f})')
axes[2].legend(loc='upper right')
axes[2].grid(alpha=0.3)

fig.suptitle('LSTM 7-feature - Predictions across all periods', y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, 'lstm_all_periods_predictions.png'), dpi=150)

plt.show()

#Save test predictions in m3/s for model comparison 
CATCHMENT_AREA_M2 = 140000000 #140km2 = 14,000 ha 

test_predictions = pd.DataFrame({
    'predicted_m3s': flowpred_test * CATCHMENT_AREA_M2 / 1000 / 86400,
    'observed_m3s': flowobs_test * CATCHMENT_AREA_M2 / 1000 / 86400,
}, index=dates_test)
test_predictions.index.name = 'date'
test_predictions.to_csv('Data/test_predictions_run1.csv')



