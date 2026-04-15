#import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

#####################
#read daily_means file (date, discharge, precip, ETP)
df_daily = pd.read_csv(os.path.join(datafolder,'daily_means.csv'),sep=';', index_col = 0, parse_dates = True)

#read CSV12H_Clean file (12h date, temp, humid, wind, radia, etc.)
#reading this in to get the temp column to use to determine if snow is melting 
df_12h = pd.read_csv(os.path.join(datafolder, 'CSV12H_Clean.csv'), sep =',', index_col = 0, parse_dates = True)

#extract temp column only and resample to daily means 
temp_daily = df_12h[['temp']].resample('D').mean()

#read groundwater.csv from HIP model to include as an input 
df_groundwater = pd.read_csv(os.path.join(datafolder, 'groundwater.csv'), sep = ';', index_col = 0, parse_dates = True, decimal = ',')
groundwater_daily = df_groundwater.resample('D').mean()

#join daily_means data, temp data, and groundwater data to one dataframe 
data_all_daily = df_daily.join(temp_daily)
data_all_daily = data_all_daily.join(groundwater_daily)

#Add columns with rolling mean of precipitation 
#this captures how wet the catchment was in the past 
#Starting with 30-day 
data_all_daily['precip_30d'] = data_all_daily['precipitation'].rolling(window = 30, min_periods = 1, center = True).mean()

#Adding 7-day window 
data_all_daily['precip_7d'] = data_all_daily['precipitation'].rolling(window = 7, min_periods = 1, center = True).mean()

#Adding 90-day window 
data_all_daily['precip_90d'] = data_all_daily['precipitation'].rolling(window = 90, min_periods = 1, center = True).mean()

#Add column with daily surplus = how much water is added vs lost each day 
#helps clarify relationship between precip and ETP, which can get confusing 
#positive = wetter conditions, negative = drying out
data_all_daily['precip_surplus'] = data_all_daily['precipitation'] - data_all_daily['ETp']

#Add column with a binary melt indicator to account for snow melt 
#if temp is above freezing: 1 (melting possible), if below: 0
data_all_daily['melt'] = (data_all_daily['temp'] > 0).astype(float)

#Save new data to a csv file in "data" folder 
data_all_daily.to_csv('data/LSTM_dataframe.csv', sep = ',')

#Plot timeseries of inputs 
fig, axes = plt.subplots(nrows = 7, figsize = (12, 10), sharex = True)
features = ['precipitation', 'ETp', 'precip_30d', 'precip_surplus', 'temp', 'groundwater', 'melt']
ylabels = ['Precip [mm/d]', 'ETp [mm/d]', '30d mean [mm/d]', 'Surplus [mm/d]', 'Temp [C]', 'GW level [m]', 'Melt [0/1]']

for i, (feat, label) in enumerate(zip(features, ylabels)): 
    axes[i].plot(data_all_daily.index, data_all_daily[feat], linewidth = 0.5)
    axes[i].set_ylabel(label)

axes[-1].set_xlabel('Date')
fig.suptitle('Input features over time')
plt.tight_layout()
plt.savefig('figures/input_features.png', dpi = 150)

#Plot histogram of all raw inputs
#Generally want inuts to be normally distributed
#Will be log transformed in Script B if needed 
var_names = ['precipitation', 'ETp', 'precip_30d', 'precip_7d', 'precip_90d', 'precip_surplus', 'temp','groundwater', 'melt']
fig, axes = plt.subplots(nrows = 1, ncols = 9, figsize = (24,4))
for i, name in enumerate(var_names):
    axes[i].hist(data_all_daily[name].dropna(), bins = 50)
    axes[i].set_title(name)
axes[0].set_ylabel('Frequency')
fig.suptitle('Raw Input Distributions')
fig.tight_layout()
plt.savefig('figures/rawdata_histograms.png', dpi = 150)

plt.show()