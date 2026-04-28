#This script compares two models with teh Diebold Mariano Test: 
    #Original model with 9 features 
    #Simplified model with 7 features, updated based on feature analysis 
#The C script in each model folder saves the predictions and observations to a numpy array that is then loaded here

import numpy as np 
from scipy import stats 


#Load both sets of predictions and one set of observations (obsv are the same for both)
pred_9feat = np.load('LSTM_Havelse_9features/predictions_test.npy')
pred_7feat = np.load('LSTM_Havelse_7features/predictions_test.npy')
obs = np.load('LSTM_Havelse_9features/observations_test.npy') #can be from either model 

#Define the Diebold Mariano test function 
#H0: 7-feature model is not worse than 9-feature model 
#using h = 80 based on HBV model parameter calibration (k1_ls1)
def diebold_mariano(obs, pred1, pred2, h=80):
    e1 = obs - pred1
    e2 = obs - pred2
    d = e1 ** 2 - e2 **2

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

dm_stat, p_value = diebold_mariano(obs, pred_9feat, pred_7feat)
print(f'DM statistic: {dm_stat:.3f}')
print(f'P-value: {p_value:.4f}')

if p_value < 0.05:
    print('9-feature model is significantly better, removing features hurt the model')
else: 
    print('9-feature model is NOT signficiantly better, removing the features did NOT hurt the model')