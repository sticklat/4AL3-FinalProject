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
        
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.estimator(x)