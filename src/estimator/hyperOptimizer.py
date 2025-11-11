import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split


from train import train_distance_model



if __name__ == "__main__":
    # Set device to cuda if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
    
    
    
    # Load dataset
    dataset = torch.load("dataset/vehicle_dataset.pt",map_location=device)
    X = dataset["features"]
    y = dataset["distances"]
    
    
    # Quick test training
    # (features, distances,device="cpu", epochs=50, batch_size=64, lr=1e-3, hidden_dim=64, num_hiddenLyr=2, activation=nn.ReLU, verbose=False)
    model = train_distance_model(X, y, device=device, epochs=100, num_hiddenLyr=4, hidden_dim=150, activation=nn.LeakyReLU, verbose=True)
    
    model.eval()

    with torch.no_grad():
        sample_preds = model(X[:5])
        print("Sample Predictions:", sample_preds.squeeze().cpu().numpy())
        print("Actual Distances:", y[:5].squeeze().cpu().numpy())