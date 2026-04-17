import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    NN model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, n_static = 0):
        super(LSTMModel, self).__init__()
        self.model_name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.n_static = n_static 

        #Add static embedding: small network that encodes catchment properties 
        if n_static > 0:
            self.static_embedding = nn.Sequential(
                nn.Linear(n_static, 16),
                nn.ReLU(), 
                nn.Linear(16,16)
            )
            lstm_input_size = input_size + 16
        else:
            lstm_input_size = input_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        #Adding the dropout because model showed signs of overfitting
        #This randomly switches off a fraction of neurons during training 
        #fraction switched off specified in B script
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, static = None):
        # Pass the full sequence through the LSTM in one call.
        # out shape: (batch_size, seq_len, hidden_size)
        # _ contains the final (h_n, c_n) states, not needed here
        if self.n_static > 0 and static is not None: 
            #Encode static attributes into a 16-dim vector 
            static_emb = self.static_embedding(static)  #batch, 16
            
            #Repeat for every timestep and concatenate with dynamic inputs 
            static_emb = static_emb.unsqueeze(1).expand(-1, x.size(1), -1)  #batch, seq_len, 16
            x = torch.cat([x, static_emb], dim = 2)     #batch, seq_len, input_size+16

        out, _ = self.lstm(x)
        
        out = self.dropout(out)
        predictions = self.fc(out)       # → (batch_size, seq_len, output_size)
        predictions = predictions.squeeze(2)  # → (batch_size, seq_len)
        return predictions
