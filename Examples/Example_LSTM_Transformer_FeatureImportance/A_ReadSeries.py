#import packages
import os
import datetime as dt
import numpy as np
import pandas as pd

#before executing in Spyder - right click on the tab with this script, choose "set console working directory", then restart the console
#the working directory in Spyder is now the directory where the script is located and you can use file path references relative to this location

datafolder=os.path.relpath(r'.\Data')
#####################
#read rainfall file
#define fields in file and datatype for each field
headers = ['date','time', 'value']
dtypes = {'date': 'str','date': 'str', 'value': float}
#read file
rain=pd.read_csv(os.path.join(datafolder,'BKLK.csv'),sep=',',skiprows=1,header=None,names=headers,dtype=dtypes)
#concatenate the date and the time into one string for each time step, convert into a time variable that Python can understand, and the save as the first column of the dataframe
datetimestring=rain.iloc[:,0]+' '+rain.iloc[:,1]
rain.iloc[:,0]=pd.to_datetime(datetimestring,format='%Y-%m-%d %H:%M:%S')
#the file contains -999 to indicate missing values. we will mark this as nan in our dataframe
rain['value']=rain['value'].replace(-999,np.nan)
#the file does not contain any observations until 2017-04-02 (you can see this by opening the dataframe in Spyder's variable explorer) - clip the period without data
rain=rain.iloc[np.min(np.where(rain['date']>=dt.datetime(year=2017,month=4,day=2))):,:]
####
#aggregate hourly to daily values - set hours, minutes, seconds to 0 and then group by date
rain['date']=[dt.datetime(x.year,x.month,x.day) for x in rain['date']]
#we assume missing values are 0 when summing up to daily values below. make sure that your datafile does not have long gaps before doing this, otherwise you will create 0 observations where it may have rained in reality
rain['value']=rain['value'].replace(np.nan,0) 
rain=rain.groupby('date').sum()
#create a continuous time index, again assuming that any missing periods are simply 0. if you have a file with many missing periods, you may be better off discarding it
rain=rain.resample('D', origin='start').fillna(0)
rain['date']=rain.index #group by converts the date column to an index. put it back into a date column and then recreate simple integer index
rain=rain.reset_index(drop=True)
#####################
#read evaporation file
#define fields in file and datatype for each field
headers = ['date','value']
dtypes = {'date': 'str', 'value': float}
#read file
refet=pd.read_csv(os.path.join(datafolder,'327501.csv'),sep=';',skiprows=1,header=None,names=headers,dtype=dtypes)
refet.iloc[:,0]=pd.to_datetime(refet.iloc[:,0],format='%d-%m-%Y %H:%M')
#convert from mm/step (with step=1month) to mm/day and the resample to days, assuming that each value corresponds to accumulated evaporation in previous month
#calculate number of days for each month in the datafile
steplength=(refet.date.diff().to_numpy()/1000000000/60/60/24).astype(np.float32)
steplength[0]=31
#divide monthly values by number of days to get evaporation per day
refet['value']=refet['value']/steplength 
refet.index=refet['date'] #resample operation below requires date index - create this index, then resample, then revert back to normal integer index
#create a series of daily values (where the evaporation  for all days in the same month is the same)
refet=refet.resample('D', origin='start').bfill()
refet['date']=refet.index
refet=refet.reset_index(drop=True)
#####################
#read flow series
#define fields in file and datatype for each field
headers = ['date','value']
dtypes = {'date': 'str', 'value': float}
#read file
flow=pd.read_csv(os.path.join(datafolder,'C2-Q.txt'),sep=';',skiprows=3,header=None,names=headers,dtype=dtypes)
flow.iloc[:,0]=pd.to_datetime(flow.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
flow['date']=[dt.datetime(x.year,x.month,x.day) for x in flow['date']] #remove time information from the flow dates
flow.iloc[:,1]=flow.iloc[:,1]*86400.0 #convert m3/s to m3/d
#####################
#Combine the 3 series into one dataframe with the same time index
#rain series is the shortest series, find dates where it starts and ends
startdate=rain['date'][np.min(np.where(np.logical_not(rain['value'].isna()))[0])]
enddate=rain['date'][rain.shape[0]-1]
#clip the refet and flow series
refet=refet.iloc[np.min(np.where(refet['date']>=startdate)):np.max(np.where(refet['date']<=enddate)),:]
flow=flow.iloc[np.min(np.where(flow['date']>=startdate)):np.max(np.where(flow['date']<=enddate)),:]
#####################
#ccombine textfiles into one dataframes
data_all=pd.merge(refet,rain,how='outer',on='date')
data_all=pd.merge(data_all,flow,how='outer',on='date')
data_all.rename(columns={'value_x':'PET','value_y':'rain','value':'flow'},inplace=True)
data_all.set_index('date',inplace=True)
#####################
#fill remaining missing values by linear interpolation
#this requires that the data have been properly checked before and that there is no bigger gaps that need to be treated manually!
data_all.interpolate(method='linear',inplace=True)
#####################
#save dataframe as pickled file
data_all.to_pickle('dataframe.pkl')

####################
#plot time series
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(data_all['rain']);ax[0].set_ylabel('P [mm/d]')
ax[1].plot(data_all['flow'],label='obs');ax[1].set_ylabel('Q [m3/d]')
ax[1].set_xlabel('Time [days]')
ax[1].legend()







