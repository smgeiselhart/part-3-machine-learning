import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


from B_lstm_forecaster import load_catchment, scale_series, mse, feature_cols, catchments, datafolder
from model import LSTMModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(weights_dir, exist_ok=True)

#################################################################
# define time periods number of epochs to train and LSTM hyperparameters
index_warmup = 182  # time step index where warm up period ends

epochs = 500

ninputs = len(feature_cols)  # no. of input variables
nhidden = 64  # no. of hidden states, i.e. "parallel" LSTM cells
nlayers = 1  # no. of LSTM layers, i.e. LSTM cells in sequence

#################################################################
#Load and scale data 
#Inputs and labels are scaled on the training data only 
# and provided as input to validation period 
catchment_data = [load_catchment(c) for c in catchments]
all_inputs_train = torch.cat([c['inputs_train'] for c in catchment_data], dim = 1)
all_labels_train = torch.cat([c['labels_train'] for c in catchment_data], dim = 1)
_, inputscales = scale_series(all_inputs_train, per_feature = True)
_, labelscales = scale_series(all_labels_train)

for c in catchment_data: 
    c['inputs_train'], _ = scale_series(c['inputs_train'], inputscales)
    c['inputs_val'], _ = scale_series(c['inputs_val'], inputscales)
    c['inputs_test'], _ = scale_series(c['inputs_test'], inputscales)
    c['labels_train'], _ = scale_series(c['labels_train'], labelscales)
    c['labels_val'], _ = scale_series(c['labels_val'], labelscales)
    c['labels_test'], _ = scale_series(c['labels_test'], labelscales)

#Scale static attributes 
all_static = torch.stack([c['static'] for c in catchment_data]) #(n_catchments=6, n_static=8)
_, staticscales = scale_series(all_static, per_feature=True)

for c in catchment_data: 
    c['static_scaled'], _ = scale_series(c['static'], staticscales)

#################################################################
#Step 1: Learning Rate Search 
#Test a range of learning rates with 10 epochs each 
#Need to comment this section out after range has been found 
#################################################################
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
final_losses = []
n_static = 8

for lr in learning_rates: 
    # Create model object. The model architecture is defined in model.py
    model = LSTMModel(ninputs, nhidden, 1, nlayers, 0.2, n_static = n_static)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
         #put model into training mode and reset gradients 
        model.train()
         #Initialize optimizer
        optimizer.zero_grad()
        total_loss = 0.0

        for c in catchment_data: 
            pred = model(c['inputs_train'], c['static_scaled'].unsqueeze(0))
            loss = mse(pred[:, index_warmup:], c['labels_train'][:, index_warmup:])
            total_loss += loss 
        avg_loss = total_loss / len(catchment_data)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    final_losses.append(avg_loss.item())
    print(f'LR = {lr:.0e}: final loss = {avg_loss.item():.4f}')

# #Plot learning rate search 
fig, ax = plt.subplots()
ax.plot(learning_rates, final_losses, marker='o')
ax.set_xscale('log')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Training Loss (after 10 epochs)')
ax.set_title('Learning Rate Search')
fig.savefig(os.path.join(figures_dir, 'lr_search.png'), dpi=150)
plt.show()

#################################################################
#Step 2: Train with CyclicLR Scheduler 
model = LSTMModel(ninputs, nhidden, 1, nlayers, 0.2, n_static = n_static)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Triangular cyclical learning rate scheduler.
# base_lr and max_lr should be based on the LR search results (above)
# step_size_up=4 -> LR increases for 4 epochs, then decreases for 4 epochs (full cycle = 8 epochs).
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-3,   #Update based on LR search 
    max_lr=1e-2,    #Update based on LR search 
    step_size_up=4,
    mode='triangular',
    cycle_momentum=False
)

# initialize best validation loss and training history
best_val_loss = torch.tensor(float('inf'), dtype=torch.float32)
history = {'train_loss': [], 'val_loss': [], 'lr': []}

# now train the model
for epoch in range(epochs):
    print(epoch)
    
    # put model into training mode and reset gradients
    model.train()
    optimizer.zero_grad()
    
    total_train_loss = 0.0 
    for c in catchment_data:
        pred = model(c['inputs_train'], c['static_scaled'].unsqueeze(0))
        loss = mse(pred[:, index_warmup:], c['labels_train'][:, index_warmup:])
        total_train_loss += loss

    avg_train_loss = total_train_loss / len(catchment_data)
    avg_train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    #Update weights 
    optimizer.step()

    # advance the triangular LR schedule
    scheduler.step()

    # save training loss and current learning rate
    history['train_loss'].append(avg_train_loss.item())
    history['lr'].append(scheduler.get_last_lr()[0])
    
    #put model into evaluation mode
    model.eval()
    total_val_loss = 0.0

    # predict the validation series
    with torch.no_grad():
        for c in catchment_data: 
            inputs_trainval = torch.cat([c['inputs_train'], c['inputs_val']], dim=1)
            pred_trainval = model(inputs_trainval, c['static_scaled'].unsqueeze(0))
            pred_val = pred_trainval[:, c['inputs_train'].shape[1]:]
            total_val_loss += mse(pred_val, c['labels_val']).item()

    # save validation loss
    avg_val_loss = total_val_loss / len(catchment_data)
    history['val_loss'].append(avg_val_loss)
    
    # if the validation loss has improved, save the neural network parameters
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(weights_dir, 'weights_scheduler.csv'))
    
    if epoch % 100 == 0: 
        print(f'Epoch {epoch}: train={avg_train_loss.item():.4f}, val={avg_val_loss:.4f}')


# Plot training history with LR 
fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(history['train_loss'], label='Training loss')
axes[0].plot(history['val_loss'], label='Validation loss')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[1].plot(history['lr'], color='tab:orange')
axes[1].set_ylabel('Learning Rate')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, 'lstm_training_history_scheduler.png'), dpi=150)
plt.show()
