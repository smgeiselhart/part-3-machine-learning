#This script compares two models with the Diebold Mariano Test: 
    #Original model with 9 features 
    #Simplified model with 7 features, updated based on feature analysis 
#The C script in each model folder saves the predictions and observations to a numpy array that is then loaded here
#The B and C script of each model was run 5 times WITHOUT the seed and manually saved to different npy files that are read here 

import numpy as np 
from scipy import stats 

N_RUNS = 5

#Load observations 
obs = np.load('LSTM_Havelse_9features/observations_val.npy') #can be from either model 

d_per_run = []
for run_idx in range (1, N_RUNS + 1):
    pred_9feat = np.load(f'LSTM_Havelse_9features/predictions_val_run{run_idx}.npy')
    pred_7feat = np.load(f'LSTM_Havelse_7features/predictions_val_run{run_idx}.npy')
    d_run = (obs - pred_9feat) ** 2 - (obs - pred_7feat) ** 2
    d_per_run.append(d_run)

d = np.mean(d_per_run, axis = 0)

#Define the Diebold Mariano test function 
#H0: 7-feature model is not worse than 9-feature model 
def diebold_mariano(d, h):

    n = len(d)
    mean_d = np.mean(d)

    #Bartlett kernel 
    gamma_0 = np.mean((d - mean_d) ** 2)
    gamma_sum = 0.0
    for lag in range (1, h + 1):
        gamma_lag = np.mean((d[lag:] - mean_d) * (d[: -lag] - mean_d))
        weight = 1.0 - lag / (h+1)
        gamma_sum += weight * gamma_lag 

    s_hat = gamma_0 + 2 * gamma_sum 
    dm_stat = mean_d / np.sqrt(s_hat / n)
    p_value = stats.norm.cdf(dm_stat)
    
    return dm_stat, p_value 

#using h = 80 based on HBV model parameter calibration (k1_ls1)
dm_stat, p_value = diebold_mariano(d, h = 80)
print(f'DM statistic: {dm_stat:.3f}')
print(f'P-value: {p_value:.4f}')

if p_value < 0.05:
    print('9-feature model is significantly better, removing features hurt the model')
else: 
    print('9-feature model is NOT significantly better, removing the features did NOT hurt the model')