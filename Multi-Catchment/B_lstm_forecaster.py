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

#Same for weights
weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(weights_dir, exist_ok=True)


from model import LSTMModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

catchments = ['Group1', 'Group2', 'Group3', 'Group5', 'Group7', 'Group11']
datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

#Feature columns to use as inputs 
#Not including precip or ETP based on feature analysis of single-catchment 
feature_cols = ['precip_7d', 'precip_30d', 'precip_90d','precip_surplus']

#Reads LSTM_dataframe csv for each catchment, interpolates NaN values 
#Stacks the feature columns into an input tensor and extracts discharge as the label tensor 
#Splits into train (65%), val (25%), and test (10%)
def load_catchment(catchment_name):
    
    #Load one catchment's dataframe and split it into train/val/test sets
    datafile = os.path.join(datafolder, catchment_name, 'LSTM_dataframe.csv')
    data_in=pd.read_csv(datafile, sep=',', index_col = 0, parse_dates = True)
   
    #Load static catchment properties 
    properties_file = os.path.join(datafolder, catchment_name, 'catchment_properties.csv')
    properties = pd.read_csv(properties_file, sep = ';', index_col = 0, skipinitialspace = True)

    static_cols = ['area_ha', 'mean_elevation_m', 'mean_slope_percent', 'rural_percent', 
                   'urban_percent', 'nature_percent', 'lake_percent', 'longest_flow_path_km']
    static = torch.tensor([float(properties.loc[col, 'value']) for col in static_cols], dtype = torch.float32)

    # interpolate the NaN values in input columns only
    data_in[feature_cols] = data_in[feature_cols].interpolate(method='linear')
    # leave discharge raw — keep NaNs
    labels = torch.tensor(data_in['discharge'].to_numpy(), dtype=torch.float32).unsqueeze(0)

    #Set training (first 65%), validation (next 25%) and test (last 10%) sets
    index_validation = int(len(data_in) * 0.65)
    index_test = int(len(data_in) * 0.90)
    
    inputs = np.stack([data_in[col].to_numpy() for col in feature_cols], axis =1)    

    #Apply log1p transformation to skewed inputs (precip7d, 30d, 90d)
    #see histograms of raw input data created in Script A
    log_cols = [0, 1, 2]
    for col in log_cols: 
        inputs[:, col] = np.log1p(inputs[:, col])

    inputs = torch.tensor(inputs,dtype=torch.float32).unsqueeze(0)  #(1, T, F)
    labels = torch.tensor(data_in['discharge'].to_numpy(), dtype=torch.float32).unsqueeze(0) #(1, T)

    return{
        'name': catchment_name,
        'static': static,   #shape: (8,)
        'inputs_train': inputs[:,:index_validation,:],
        'inputs_val': inputs[:,index_validation:index_test,:],
        'inputs_test' : inputs[:,index_test:,:],
        'labels_train' : labels[:,:index_validation],
        'labels_val' : labels[:,index_validation:index_test],
        'labels_test' : labels[:,index_test:]}
    
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
    nse = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
    return(nse)

#Wrapping in if statement so this doesn't run when imported to Cross Valid 
if __name__ == '__main__':
    #################################################################
    #define time periods number of epochs to train and LSTM hyperparameters
    #index where validation starts computed in load_catchment function
    index_warmup = 182 #time step index where warm up period ends (6 mo)
    
    epochs = 500  
    ninputs = len(feature_cols) #no. of input variables
    nhidden = 64 #no. of hidden states, i.e. "parallel" LSTM cells
    nlayers = 1 #no. of LSTM layers, i.e. LSTM cells in sequence
    
    #################################################################
    #load all catchments 
    catchment_data = [load_catchment(c) for c in catchments]
    
    #################################################################
    #Scale data. For machine learning models, inputs and labels should be scaled. The scaling values should be computed on the training data only
    #Compute global scaling from ALL training data (model sees all catchments on same scale)
    all_inputs_train = torch.cat([c['inputs_train'] for c in catchment_data], dim = 1)
    all_labels_train = torch.cat([c['labels_train'] for c in catchment_data], dim = 1)

    _, inputscales = scale_series(all_inputs_train)
    _, labelscales = scale_series(all_labels_train)

    #Scale each catchment using the global stats 
    for c in catchment_data:
        c['inputs_train'], _ = scale_series(c['inputs_train'], inputscales)
        c['inputs_val'], _ = scale_series(c['inputs_val'], inputscales)
        c['inputs_test'], _ = scale_series(c['inputs_test'], inputscales)
        c['labels_train'], _ = scale_series(c['labels_train'], labelscales)
        c['labels_val'], _ = scale_series(c['labels_val'], labelscales)
        c['labels_test'], _ = scale_series(c['labels_test'], labelscales)

    #Scale static features per-column (each attribute has different units, so one mean/std per column)
    all_static = torch.stack([c['static'] for c in catchment_data])  #(n_catchments, 8)
    static_mean = all_static.mean(dim = 0)
    static_std = all_static.std(dim = 0)

    for c in catchment_data:
        c['static_scaled'] = (c['static'] - static_mean) / static_std

    #################################################################
    #Create model object. The model architecture is defined in model.py
    #Last value here shows the dropout rate (%)
    model = LSTMModel(ninputs,nhidden,1,nlayers,0, n_static = 8)

    #################################################################
    #Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #initialize best validation loss and training history
    best_val_loss = torch.tensor(float('inf'),dtype=torch.float32)
    history = {'train_loss': [],'val_loss': []}

    #train the model
    for epoch in range(epochs):
       
        #put model into training mode and reset gradients
        model.train() #in ML models you can include options that should only happen during training - e.g. randomly resetting some of the parameters to reduce overfitting (dropout)
        optimizer.zero_grad()
        
        #predict the flow series and compute MSE, excluding the warm up period
        #Accumulate the loss across all catchments
        total_train_loss = 0.0
        for c in catchment_data:
            pred = model(c['inputs_train'], static = c['static_scaled'].unsqueeze(0))
            loss = mse(pred[:,index_warmup:],c['labels_train'][:,index_warmup:])
            total_train_loss = total_train_loss + loss
        
        #Average loss across all catchments, then backpropogation 
        avg_train_loss = total_train_loss / len(catchment_data)
        avg_train_loss.backward()

        #Cap the gradient magnitude to prevent large steps from being taken 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       
        #Update weights
        optimizer.step()
       
        #save training loss
        history['train_loss'].append(avg_train_loss.item())
        
        #put model into evaluation model, i.e. features like dropout will be inactive because we now want to generate the best possible prediciton
        #Run model on validation period 
        model.eval()
       
        #predict the validation series, using the training series as warm up 
        total_val_loss = 0

        with torch.no_grad():
            for c in catchment_data: 
                inputs_trainval = torch.cat([c['inputs_train'], c['inputs_val']], dim=1)
                pred_trainval = model(inputs_trainval, static = c['static_scaled'].unsqueeze(0))
                pred_val = pred_trainval[:, c['inputs_train'].shape[1]:]
                total_val_loss += mse(pred_val, c['labels_val']).item()
        
        avg_val_loss = total_val_loss / len(catchment_data)
        
        #save validation loss
        history['val_loss'].append(avg_val_loss)

        #if the validation loss has improved, saved the neural network parameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, 'weights.csv'))

        if epoch % 100 == 0: 
            print(f'Epoch {epoch}')

    #plot how training and validation loss change during training
    #If validation loss curve flattens out, we have likely converged 
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], label='Training loss')
    ax.plot(history['val_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.legend()
    fig.savefig(os.path.join(figures_dir, 'lstm_training_history.png'), dpi=150)

    plt.show()