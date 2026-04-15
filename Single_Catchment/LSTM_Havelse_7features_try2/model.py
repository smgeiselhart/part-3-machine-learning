import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    NN model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.model_name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
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

    def forward(self, x):
        # Pass the full sequence through the LSTM in one call.
        # out shape: (batch_size, seq_len, hidden_size)
        # _ contains the final (h_n, c_n) states, not needed here
        out, _ = self.lstm(x)
        
        out = self.dropout(out)
        predictions = self.fc(out)       # → (batch_size, seq_len, output_size)
        predictions = predictions.squeeze(2)  # → (batch_size, seq_len)
        return predictions
