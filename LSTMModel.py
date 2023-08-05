import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def _init_(self, input_size=54, hidden_size=1, num_layers=2, num_stocks=0.2, dropout_prob=54):
        super(LSTMModel, self)._init_()
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_stocks)

        # ReLU activation
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply ReLU activation and dropout
        out = self.relu(out)
        out = self.dropout(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

# Define hyperparameters
# input_size = 54  # Number of features (stocks)
# hidden_size = 128
# num_layers = 2
# dropout_prob = 0.2
# num_stocks = 54  # Number of output stocks

# # Create the LSTM model
# model = LSTMModel(input_size, hidden_size, num_layers, num_stocks, dropout_prob)

# # Print the model architecture
# print(model)