import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

from NN import DistanceRegressor
from VehicleAnnotationDataset import VehicleAnnotationDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
        total_samples += input.size(0)
    avg_loss = running_loss / total_samples
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        outputs = model(input)
        loss = criterion(outputs, target)
        running_loss += loss.item() * input.size(0)
        total_samples += input.size(0)
    avg_loss = running_loss / total_samples
    return avg_loss

def train_distance_model(train_loader,val_loader,device="cpu", epochs=50, lr=1e-3, input_dim=4,
                         hidden_dim=64, num_hiddenLyr=2, activation=nn.ReLU, verbose=False, return_losses=False):
    """
    features: tensor of shape [N, 4] -> xmin, xmax, ymin, ymax
    labels: tensor of shape [N] -> label_id as integer
    return_losses: if True, also returns train_losses and val_losses lists
    """


    # Initialize model, loss function, and optimizer
    model = DistanceRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hiddenLyr, activation=activation).to(device)
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if return_losses:
        return model, train_losses, val_losses
    return model

if __name__ == "__main__":
    # Hyperparameters
    epochs = 15
    batch_size = 256
    lr = 1e-3
    
    
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'Available GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        
    # Load data from file into VehicleAnnotationDataset
    # Load on CPU first to avoid DataLoader worker process issues
    dataset_dict = torch.load('src/estimator/dataset/vehicle_dataset_ext.pt')
    features = dataset_dict['features']
    targets = dataset_dict['targets']
    dataset = VehicleAnnotationDataset(features, targets)
    
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    # Create DataLoaders
    # Note: num_workers must be 0 when using GPU tensors to avoid CUDA initialization errors in worker processes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    
    
    # Train the model
    model = train_distance_model(train_loader, val_loader, input_dim=features.shape[1], device=device, epochs=epochs, lr=lr, verbose=True)
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, nn.MSELoss(), device)
    print(f'Test MSE Loss: {test_loss:.4f}')
    
    # Try 10 random samples from test set
    model.eval()
    with torch.no_grad():
        for i in range(10):
            idx = torch.randint(0, len(test_ds), (1,)).item()
            input, target = test_ds[idx]
            input = input.to(device).unsqueeze(0)  # Add batch dimension
            prediction = model(input)
            print(f'Sample {i+1}: True Distance = {target.item():.2f}, Predicted Distance = {prediction.item():.2f}')
    
    
    