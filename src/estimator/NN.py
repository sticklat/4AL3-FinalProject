import torch
import torch.nn as nn

class DistanceRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_hidden_layers=2, activation=nn.ReLU):
        """
        input_dim: total number of features including one-hot label
        hidden_dim: size of each hidden layer
        num_hidden_layers: number of hidden layers (default: 2)
        activation: activation function class to use (default: nn.ReLU)
        """
        super(DistanceRegressor, self).__init__()
        
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        
        # Output layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)