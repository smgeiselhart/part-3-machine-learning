import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import datetime as dt 
import matplotlib.pyplot as plt

#Create folder where all figures are saved 
figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

from model import LSTMModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#Function that loads the datafile
#Returns training and validation inputs and labels
#Also returns the index where validation period starts (at 65% through dataset)
#Also interpolates any NaN values to avoid corrupting model. Loss is not computed on these. 
def load_datafile(datafile):
    data_in=pd.read_csv(datafile, sep=',', index_col = 0, parse_dates = True)
   
    #Interpolate all NaN values 
    data_in.interpolate(method = 'linear', inplace = True)

    #Set training (first 65%), validation (next 25%) and test (last 10%) sets
    index_validation = int(len(data_in) * 0.65)
    index_test = int(len(data_in) * 0.90)

    inputs = np.stack([data_in['precip_30d'].to_numpy(), 
                       data_in['precip_7d'].to_numpy(),
                       data_in['precip_90d'].to_numpy(),
                       data_in['precip_surplus'].to_numpy(),
                       data_in['temp'].to_numpy(), 
                       data_in['groundwater'].to_numpy(),
                       data_in['melt'].to_numpy()], 
                       axis =1)    
    
    #Apply log1p transformation to skewed inputs (precip 30d, 7d, 90d)
    #see histograms of raw input data created in Script A
    log_cols = [0, 1, 2]
    for col in log_cols: 
        inputs[:, col] = np.log1p(inputs[:, col])

    # inputs = np.stack([data_in['rain'].to_numpy(),rain_month.to_numpy(),rain_3month.to_numpy()],axis=1)    
    inputs = torch.tensor(inputs,dtype=torch.float32)
    
    labels = torch.tensor(data_in['discharge'].to_numpy(), dtype=torch.float32)
   
    #add batch dimension
    inputs = inputs.unsqueeze(0)
   
    #labels can have batch dimension, we don't need feature dimension
    labels = labels.unsqueeze(0)
    

    inputs_train = inputs[:,:index_validation,:]
    inputs_val = inputs[:,index_validation:index_test,:]
    inputs_test = inputs[:,index_test:,:]
    labels_train = labels[:,:index_validation]
    labels_val = labels[:,index_validation:index_test]
    labels_test = labels[:,index_test:]
    return(inputs_train,inputs_val,inputs_test,labels_train,labels_val, labels_test,index_validation)
    
def scale_series(data,zscale=None):
    if not zscale:
        mean = data[~torch.isnan(data)].mean()
        std = data[~torch.isnan(data)].std()
        zscale = (mean, std)
    data_scaled = (data - zscale[0])/zscale[1]
    return(data_scaled,zscale)

def unscale_series(data,zscale):
    data_unscaled = data*zscale[1] + zscale[0]
    return(data_unscaled)

def mse(pred,label):
    mask = ~torch.isnan(label)
    residual = torch.pow(label[mask]-pred[mask],2)
    mse = torch.mean(residual)
    return(mse)

def nse(obs,pred):
    mask = ~np.isnan(obs)
    obs, pred = obs[mask], pred[mask]
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

#Wrapping in if statement so this doesn't run when imported to C script 
if __name__ == '__main__':
    #################################################################
    #define time periods number of epochs to train and LSTM hyperparameters
    #index where validation starts computed in load_datafile function
    index_warmup = 182 #time step index where warm up period ends (6 mo)
    #
    epochs = 1000
    #
    ninputs = 7 #no. of input variables
    nhidden = 64 #no. of hidden states, i.e. "parallel" LSTM cells
    nlayers = 1 #no. of LSTM layers, i.e. LSTM cells in sequence
    
    #################################################################
    #load the input data, see script A for details on creation of this file 
    inputs_train,inputs_val,inputs_test,labels_train,labels_val,labels_test,index_validation = load_datafile('Data/LSTM_dataframe.csv')

    #Plot histograms of log transformed inputs 
    var_names = ['precip_30d', 'precip_7d', 'precip_90d']
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16,6))
    for i, name in enumerate(var_names):
        axes[i].hist(inputs_train[0, :, i].numpy(), bins = 50)
        axes[i].set_title(name)
    axes[0].set_ylabel('Frequency')
    fig.suptitle('Log Transformed Input Distributions')
    fig.tight_layout()
    plt.savefig('figures/transformed_inputs_hist.png', dpi = 150)
    
    #################################################################
    #Scale data. For machine learning models, inputs and labels should be scaled. The scaling values should be computed on the training data only
    inputs_train,inputscales = scale_series(inputs_train)
    labels_train,labelscales = scale_series(labels_train)

    #for the validation data we provide the scaling values as input
    inputs_val,inputscales = scale_series(inputs_val,inputscales)
    labels_val,labelscales = scale_series(labels_val,labelscales)

    #Same for the test data 
    inputs_test,inputscales = scale_series(inputs_test,inputscales)
    labels_test,labelscales = scale_series(labels_test,labelscales)

    #################################################################
    #Create model object. The model architecture is defined in model.py
    #Added dropout of 20% (last value here) 
    model = LSTMModel(ninputs,nhidden,1,nlayers,0.2)

    #################################################################
    #Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #initialize best validation loss and training history
    best_val_loss = torch.tensor(float('inf'),dtype=torch.float32)
    history = {'train_loss': np.array([]),'val_loss': np.array([])}

    #set number of timesteps in the training set 
    #Using this to use as a warm up for the validaiton period 
    n_train = inputs_train.shape[1]

    #now train the model
    for epoch in range(epochs):
        print(epoch)
        #put model into training mode and reset gradients
        model.train() #in ML models you can include options that should only happen during training - e.g. randomly resetting some of the parameters to reduce overfitting (dropout)
        optimizer.zero_grad()
        #predict the flow series and compute MSE, excluding the warm up period
        pred_train = model(inputs_train)
        loss_train = mse(pred_train[:,index_warmup:],labels_train[:,index_warmup:])
        #Perform backpropagation to compute gradients of loss function with respect to weights of the neural network
        loss_train.backward()
        #Cap the gradient magnitude to prevent large steps from being taken 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #Update weights
        optimizer.step()
        #save training loss
        history['train_loss'] = np.append(history['train_loss'],loss_train.detach().cpu().numpy())
        #put model into evaluation model, i.e. features like dropout will be inactive because we now want to generate the best possible prediciton
        model.eval()
        #predict the validation series, using the training series as warm up 
        with torch.no_grad():
            inputs_trainval = torch.cat([inputs_train, inputs_val], dim=1)
            pred_trainval = model(inputs_trainval)
            pred_val = pred_trainval[:, n_train:]
            loss_validation = mse(pred_val, labels_val) 
        #save validation loss
        history['val_loss'] = np.append(history['val_loss'],loss_validation.detach().cpu().numpy())
        #if the validation loss has improved, saved the neural network parameters
        if loss_validation < best_val_loss:
            best_val_loss = loss_validation
            torch.save(model.state_dict(), 'weights.csv')

    #plot how training and validation loss change during training
    #the loss function for the validation data flattens out, so we have likely converged
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], label='Training loss')
    ax.plot(history['val_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.legend()
    fig.savefig(os.path.join(figures_dir, 'lstm_training_history.png'), dpi=150)

    plt.show()