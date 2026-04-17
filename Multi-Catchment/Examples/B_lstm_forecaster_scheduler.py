import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from model import LSTMModel


def load_datafile(datafile, index_validation):
    data_in = pd.read_pickle(datafile)
    inputs = np.stack([data_in['rain'].to_numpy()], axis=1)
    # inputs = np.stack([data_in['rain'].to_numpy(),rain_month.to_numpy(),rain_3month.to_numpy()],axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    labels = torch.tensor(data_in['flow'].to_numpy(), dtype=torch.float32)
    # add batch dimension
    inputs = inputs.unsqueeze(0)
    # labels can have batch dimension, we don't need feature dimension
    labels = labels.unsqueeze(0)

    inputs_train = inputs[:, :index_validation, :]
    inputs_val = inputs[:, index_validation:, :]
    labels_train = labels[:, :index_validation]
    labels_val = labels[:, index_validation:]
    return (inputs_train, inputs_val, labels_train, labels_val)


def scale_series(data, minmaxscale=None):
    if not minmaxscale:
        minimum = torch.amin(data, dim=1)
        maximum = torch.amax(data, dim=1)
        minmaxscale = (minimum, maximum)
    data_scaled = (data - minmaxscale[0]) / (minmaxscale[1] - minmaxscale[0])
    return (data_scaled, minmaxscale)


def unscale_series(data, minmaxscale):
    minimum = minmaxscale[0]
    maximum = minmaxscale[1]
    data_unscaled = data * (maximum - minimum) + minimum
    return data_unscaled


def mse(pred, label):
    residual = torch.pow(label - pred, 2)
    mse_val = torch.mean(residual, dim=(0, 1))
    return mse_val


#################################################################
# define time periods number of epochs to train and LSTM hyperparameters
index_warmup = 50  # time step index where warm up period ends
index_validation = 1398  # index where validation period starts

epochs = 500

ninputs = 1  # no. of input variables
nhidden = 32  # no. of hidden states, i.e. "parallel" LSTM cells
nlayers = 1  # no. of LSTM layers, i.e. LSTM cells in sequence

#################################################################
# load the input data - it contains rainfall, evaporation and flow for 4 years
# during loading, we also compute monthly and 3 monthly rainfall to make it easier for the LSTM to learn patterns
inputs_train, inputs_val, labels_train, labels_val = load_datafile(
    'dataframe.pkl', index_validation
)

#################################################################
# Scale data. For machine learning models, inputs and labels should be scaled.
# The scaling values should be computed on the training data only
inputs_train, inputscales = scale_series(inputs_train)
labels_train, labelscales = scale_series(labels_train)
# for the validation data we provide the scaling values as input
inputs_val, inputscales = scale_series(inputs_val, inputscales)
labels_val, labelscales = scale_series(labels_val, labelscales)

#################################################################
# Create model object. The model architecture is defined in model.py
model = LSTMModel(ninputs, nhidden, 1, nlayers, 0)

#################################################################
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#################################################################
# Triangular cyclical learning rate scheduler.
# step_size_up=4 -> LR increases for 4 epochs, then decreases for 4 epochs (full cycle = 8 epochs).
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-3,
    step_size_up=4,
    mode='triangular',
    cycle_momentum=False
)

# initialize best validation loss and training history
best_val_loss = torch.tensor(float('inf'), dtype=torch.float32)
history = {'train_loss': np.array([]), 'val_loss': np.array([]), 'lr': np.array([])}

# now train the model
for epoch in range(epochs):
    print(epoch)
    # put model into training mode and reset gradients
    model.train()
    optimizer.zero_grad()
    # predict the flow series and compute MSE, excluding the warm up period
    pred_train = model(inputs_train)
    loss_train = mse(pred_train[:, index_warmup:], labels_train[:, index_warmup:])
    # backpropagation
    loss_train.backward()
    # update weights
    optimizer.step()
    # advance the triangular LR schedule
    scheduler.step()
    # save training loss and current learning rate
    history['train_loss'] = np.append(
        history['train_loss'], loss_train.detach().cpu().numpy()
    )
    history['lr'] = np.append(history['lr'], scheduler.get_last_lr()[0])
    # put model into evaluation mode
    model.eval()
    # predict the validation series
    with torch.no_grad():
        pred_val = model(inputs_val)
        loss_validation = mse(
            pred_val[:, index_warmup:], labels_val[:, index_warmup:]
        )
    # save validation loss
    history['val_loss'] = np.append(
        history['val_loss'], loss_validation.detach().cpu().numpy()
    )
    # if the validation loss has improved, save the neural network parameters
    if loss_validation < best_val_loss:
        best_val_loss = loss_validation
        torch.save(model.state_dict(), 'weights.pkl')

# plot how training and validation loss change during training
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'], label='val')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[1].plot(history['lr'], color='tab:orange')
axes[1].set_ylabel('Learning Rate')
axes[1].set_xlabel('Epoch')
plt.tight_layout()

################################################################
# recreate the model from the saved parameters and plot predictions
# for an independent series (here: using validation period as pseudo-test)
model = LSTMModel(ninputs, nhidden, 1, nlayers, 0)
model.load_state_dict(torch.load('weights.pkl'))

model.eval()
with torch.no_grad():
    pred_val = model(inputs_val)
rainseries = unscale_series(inputs_val[0, :], inputscales).numpy()
flowpred = unscale_series(pred_val[0, :], labelscales).numpy()
flowobs = unscale_series(labels_val[0, :], labelscales).numpy()

fig, ax = plt.subplots(nrows=2)
ax[0].plot(rainseries)
ax[1].plot(flowobs)
ax[1].plot(flowpred)

# check what the model predicts during training
with torch.no_grad():
    pred_train = model(inputs_train)
rainseries = unscale_series(inputs_train[0, :], inputscales).numpy()
flowpred = unscale_series(pred_train[0, :], labelscales).numpy()
flowobs = unscale_series(labels_train[0, :], labelscales).numpy()

fig, ax = plt.subplots(nrows=2)
ax[0].plot(rainseries)
ax[1].plot(flowobs)
ax[1].plot(flowpred)
