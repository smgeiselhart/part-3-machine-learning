#This script compares two models with teh Diebold Mariano Test: 
    #Original model with 9 features 
    #Simplified model with 6 features, updated based on feature analysis 
#The C script in each model folder saves the predictions and observations to a numpy array that is then loaded here

import numpy as np 
from scipy import stats 


#Load both sets of predictions and one set of observations (obsv are the same for both)
pred_9feat = np.load('LSTM_Havelse_9features/predictions_val.npy')
pred_7feat = np.load('LSTM_Havelse_7features_try2/predictions_val.npy')
obs = np.load('LSTM_Havelse_9features/observations_val.npy') #can be from either model 

#Define the Diebold Mariano test function 
#H0 = both models have equal predictive accuracy 
def diebold_mariano(obs, pred1, pred2, h=1):
    e1 = obs - pred1
    e2 = obs - pred2
    d = e1 ** 2 - e2 **2

    n = len(d)
    mean_d = np.mean(d)

    #Newey-West variance estimate 
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1,h):
        gamma_sum += np.cov(d[k:], d[:-k])[0, 1]
    var_d = (gamma_0 + 2 * gamma_sum) / n

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value 

dm_stat, p_value = diebold_mariano(obs, pred_9feat, pred_7feat)
print(f'DM statistic: {dm_stat:.3f}')
print(f'P-value: {p_value:.4f}')

if p_value < 0.05:
    if dm_stat > 0:
        print('7-feature model is significantly better')
    else: 
        print('9-feature model is significantly better')
else: 
    print('No significant difference between models')