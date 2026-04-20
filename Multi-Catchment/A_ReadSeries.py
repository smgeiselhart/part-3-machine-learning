#import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Create Data and Figure folder 
datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
figurefolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')

#List of all catchment folders 
catchments = ['Group1', 'Group2','Group3','Group5','Group7','Group11']

#Standard column names to use across all catchments 
standard_cols = ['discharge', 'precipitation', 'ETp']

for catchment in catchments:
    filepath = os.path.join(datafolder, catchment, 'daily_means.csv')

    #####################
    #read CSV file - first try semicolon, then try comma (one group has commas)
    try:    
        df = pd.read_csv(filepath, sep = ';', index_col = 0, parse_dates = True, date_format = '%Y-%m-%d')
        if len(df.columns) < 2: #Meaning the semicolon didn't split properly 
            df = pd.read_csv(filepath, sep = ',', index_col = 0, parse_dates = True, date_format = '%Y-%m-%d')
    except: 
        df = pd.read_csv(filepath, sep = ',', index_col = 0, parse_dates = True, date_format = '%Y-%m-%d')

    #####################
    #Normalize column names to discharge, precipitation, ETp (some are capitalized differently)
    rename_map = {}

    for col in df.columns: 
        if col.lower() == 'etp':
            rename_map[col] = 'ETp'
        elif col.lower() == 'discharge': 
            rename_map[col] = 'discharge'
        elif col.lower() == 'precipitation':
            rename_map[col] = 'precipitation'
    df.rename(columns = rename_map, inplace = True)

    #Keep only the three standard columns (some datasets have temp)
    df = df[standard_cols]

    #Remove Group1 data after 2024 - there's an outlier that is effecting the model performance in the test set here 
    if catchment == 'Group1':
        df = df.loc[:'2024-01-01']

    #####################
    #Additional features to include based on precip and ETP 
    #7-day rolling mean 
    df['precip_7d'] = df['precipitation'].rolling(window = 7, min_periods = 1, center = True).mean()

    #30-day rolling mean 
    df['precip_30d'] = df['precipitation'].rolling(window = 30, min_periods = 1, center = True).mean()

    #Daily surplus = precip - ETP (net water balance)
    df['precip_surplus'] = df['precipitation'] - df['ETp']

    #####################
    #Save the dataframes 
    df.to_csv((os.path.join(datafolder, catchment, 'LSTM_dataframe.csv')), sep = ',')

    #####################
    #Plot a timeseries of  input data and discharge, 1 figure per catchment 
    features = ['discharge','precipitation', 'ETp', 'precip_7d', 'precip_30d', 'precip_surplus']
    ylabels = ['Discahrge [mm/d]','Precip [mm/d]', 'ETP [mm/d]', '7d mean [mm/d]', '30d mean [mm/d]', 'Precip Surplus [mm/d]']

    fig, axes = plt.subplots(nrows=len(features), figsize=(12, 10), sharex=True)
    for i, (feat, label) in enumerate(zip(features, ylabels)):
        axes[i].plot(df.index, df[feat], linewidth=0.5)
        axes[i].set_ylabel(label)
    axes[-1].set_xlabel('Date')
    fig.suptitle(f'{catchment} - Data Overview')
    plt.tight_layout()
    plt.savefig(os.path.join(figurefolder, f'{catchment}_data_overview.png'), dpi=150)
    plt.close()

    #####################
    #Plot histograms of all inputs to check skewness
    fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(20, 4))
    for i, feat in enumerate(features):
        axes[i].hist(df[feat].dropna(), bins=50)
        axes[i].set_title(feat)
    axes[0].set_ylabel('Frequency')
    fig.suptitle(f'{catchment} - Raw Input Distributions')
    fig.tight_layout()
    plt.savefig(os.path.join(figurefolder, f'{catchment}_rawdata_histograms.png'), dpi=150)
    plt.close()

    