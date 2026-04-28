#This script recreates the model from the saved parameters 
#Plots predictions for an independent series and calculates metrics 

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

#Set number of static inputs 
n_static = 8

#Compute global scaling from all training data (must match training data)
all_inputs_train = torch.cat([c['inputs_train'] for c in catchment_data], dim=1)
all_labels_train = torch.cat([c['labels_train'] for c in catchment_data], dim=1)
_, inputscales = scale_series(all_inputs_train, per_feature = True)
_, labelscales = scale_series(all_labels_train)

#Scale each catchment 
for c in catchment_data:
    c['inputs_train'], _ = scale_series(c['inputs_train'], inputscales)
    c['inputs_val'], _ = scale_series(c['inputs_val'], inputscales)
    c['inputs_test'], _ = scale_series(c['inputs_test'], inputscales)
    c['labels_train'], _ = scale_series(c['labels_train'], labelscales)
    c['labels_val'], _ = scale_series(c['labels_val'], labelscales)
    c['labels_test'], _ = scale_series(c['labels_test'], labelscales)

#Scale the static attributes used as input  
all_static = torch.stack([c['static'] for c in catchment_data])
_, staticscales = scale_series(all_static, per_feature=True)

for c in catchment_data:
    c['static_scaled'], _ = scale_series(c['static'], staticscales)

#Load the trained model (done in B script)
model = LSTMModel(ninputs,nhidden,1,nlayers,0, n_static = n_static)
#Load the Weights - pick which .csv based on the forecaster used!! 
#model.load_state_dict(torch.load(os.path.join(weights_dir, 'weights.csv')))
#model.load_state_dict(torch.load(os.path.join(weights_dir, 'weights_scheduler.csv')))
model.load_state_dict(torch.load(os.path.join(weights_dir, 'weights_random_windows.csv')))
model.eval()

#Evaluate each catchment
for c in catchment_data: 
    name = c['name']
    print(f'\n=== {name} ===')

    #Join all datasets back together to ensure all get a warm up period 
    inputs_all = torch.cat([c['inputs_train'], c['inputs_val'], c['inputs_test']], dim = 1)
    labels_all = torch.cat([c['labels_train'], c['labels_val'], c['labels_test']], dim = 1)

    #Run model on full sequence 
    with torch.no_grad():
        pred_all = model(inputs_all, c['static_scaled'].unsqueeze(0))

    #Slice predictions into each dataset 
    n_train = c['inputs_train'].shape[1]
    n_val = c['inputs_val'].shape[1]

    pred_train = pred_all[:, :n_train]
    pred_val = pred_all[:, n_train:n_train+n_val]
    pred_test = pred_all[:, n_train+n_val:]

    #Unscale output
    #training
    flowpred_train = unscale_series(pred_train[0,:],labelscales).numpy()
    flowobs_train = unscale_series(c['labels_train'][0,:],labelscales).numpy()
    rainseries_train = unscale_series(c['inputs_train'], inputscales)[0, :, 0].numpy()

    #Validation 
    flowpred_val = unscale_series(pred_val[0,:],labelscales).numpy()
    flowobs_val = unscale_series(c['labels_val'][0,:],labelscales).numpy()
    rainseries_val = unscale_series(c['inputs_val'], inputscales)[0, :, 0].numpy()


    #Test 
    flowpred_test = unscale_series(pred_test[0,:],labelscales).numpy()
    flowobs_test = unscale_series(c['labels_test'][0,:],labelscales).numpy()
    rainseries_test = unscale_series(c['inputs_test'], inputscales)[0, :, 0].numpy()


    ######### EVALUATION ###########
    #Calculate the NSE on training 
    nse_train = nse(flowobs_train, flowpred_train)
    print(f'Training NSE: {nse_train:3f}')

    #PLot training predictions
    fig,ax = plt.subplots(nrows=2)
    ax[0].plot(rainseries_train)
    ax[0].set_ylabel('Rainfall [mm/d]')
    ax[0].set_title(f'{name} - Training (NSE = {nse_train:.3f})')
    ax[1].plot(flowobs_train, label='Observed')
    ax[1].plot(flowpred_train, label='Predicted')
    ax[1].set_ylabel('Flow [mm/d]')
    ax[1].set_xlabel('Time [days]')
    ax[1].legend()
    fig.savefig(os.path.join(figures_dir, f'{name}_training_predictions.png'), dpi=150)
    plt.close()

    #Calculate NSE on validation set, 1 = perfect, 0 = no better than predicting the mean
    nse_val = nse(flowobs_val, flowpred_val)
    print(f'Validation NSE: {nse_val:3f}')

    #PLot Validation predictions 
    fig,ax = plt.subplots(nrows=2, figsize = (12, 6))
    ax[0].plot(rainseries_val, linewidth = 0.5)
    ax[0].set_ylabel('Rainfall [mm/d]')
    ax[0].set_title(f'{name} - Validation (NSE = {nse_val:.3f})')
    ax[1].plot(flowobs_val, label='Observed')
    ax[1].plot(flowpred_val, label='Predicted')
    ax[1].set_ylabel('Flow [mm/d]')
    ax[1].set_xlabel('Time [days]')
    ax[1].legend()
    fig.savefig(os.path.join(figures_dir, f'{name}_validation_predictions.png'), dpi=150)
    plt.close()

    #Calcualte NSE on test period 
    nse_test = nse(flowobs_test, flowpred_test)
    print(f'Test NSE: {nse_test:.3f}')

    #Plot Test Predictions 
    fig,ax = plt.subplots(nrows=2, figsize = (12, 6))
    ax[0].plot(rainseries_test, linewidth = 0.5)
    ax[0].set_ylabel('Rainfall [mm/d]')
    ax[0].set_title(f'{name} - Test (NSE = {nse_test:.3f})')
    ax[1].plot(flowobs_test, label='Observed')
    ax[1].plot(flowpred_test, label='Predicted')
    ax[1].set_ylabel('Flow [mm/d]')
    ax[1].set_xlabel('Time [days]')
    ax[1].legend()
    fig.savefig(os.path.join(figures_dir, f'{name}_test_predictions.png'), dpi=150)
    plt.close()

    #Check Residuals 
    #Plot histogram of residuals on validation period to confirm if normally distributed
    residuals = flowobs_val - flowpred_val
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50)
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    ax.set_xlabel('Residual [mm/d]')
    ax.set_ylabel('No. of occurences')
    ax.set_title(f'{name} - Residual Histogram')
    ax.legend()
    fig.savefig(os.path.join(figures_dir, f'{name}_residuals_histogram.png'), dpi=150)
    plt.close()

    #Plot residuals over time to check for white noise
    fig, ax = plt.subplots()
    ax.plot(residuals, color = 'steelblue', linewidth = 0.5)
    ax.axhline(0, color = 'red', linestyle = '--', linewidth =1)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Residual [mm/d]')
    ax.set_title('Residuals over time')
    fig.savefig(os.path.join(figures_dir, f'{name}_Residuals_timeseries.png'), dpi=150)
    plt.close()

    #PLot autocorrelation function 
    fig, ax = plt.subplots()
    plot_acf(residuals, lags=60, ax=ax)
    ax.set_xlabel('Lag [days]')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{name} - ACF of Residuals')
    fig.savefig(os.path.join(figures_dir, f'{name}_residuals_acf.png'), dpi=150)
    plt.close()

    #Save Group 2 test predictions in m3/s for model comparison 
    if name == 'Group2':
        group2_df = pd.read_csv(os.path.join(datafolder, 'Group2', 'LSTM_dataframe.csv'),
                            sep=',', index_col=0, parse_dates=True)
        n = len(group2_df)
        test_index = int(n*0.90)
        dates_test = group2_df.index[test_index:]

        CATCHMENT_AREA_M2 = 140_000_000  # 140 km² = 14,000 ha

        test_predictions = pd.DataFrame({
            'predicted_m3s': flowpred_test * CATCHMENT_AREA_M2 / 1000 / 86400,
            'observed_m3s': flowobs_test * CATCHMENT_AREA_M2 / 1000 / 86400,
        }, index=dates_test)

        test_predictions.index.name = 'date'
        test_predictions.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'Data/Group2/Group2_test_predictions.csv'))
      


