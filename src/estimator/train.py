import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

from NN import DistanceRegressor


def train_distance_model(features, distances,device="cpu", epochs=50, batch_size=64, lr=1e-3, hidden_dim=64, num_hiddenLyr=2, activation=nn.ReLU, verbose=False):
    """
    features: tensor of shape [N, 4] -> xmin, xmax, ymin, ymax
    labels: tensor of shape [N] -> label_id as integer
    """

    

    X = features
    y = distances  # distance targets

    # Dataset and split
    dataset = TensorDataset(X, y)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator(device=device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_ds, batch_size=batch_size, generator=torch.Generator(device=device))

    # Initialize model
    model = DistanceRegressor(input_dim=X.shape[1], hidden_dim=hidden_dim, num_hidden_layers=num_hiddenLyr, activation=activation)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        if verbose:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model