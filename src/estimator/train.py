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

@torch.no_grad()
def compute_depth_metrics(model, dataloader, device):
    """
    Compute depth estimation metrics:
    - RMSE: Root mean squared error (m)
    - iRMSE: Root mean squared error of inverse depth (1/km)
    - SILog: Scale invariant logarithmic error (log(m)*100)
    - sqErrorRel: Relative squared error (percent)
    - absErrorRel: Relative absolute error (percent)
    - MSE: Mean squared error (m²)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        outputs = model(input)
        all_predictions.append(outputs.squeeze())
        all_targets.append(target.squeeze())
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    # Clamp predictions to avoid division by zero or negative values
    predictions = torch.clamp(predictions, min=1e-6)
    targets = torch.clamp(targets, min=1e-6)
    
    # MSE (Mean Squared Error in m²)
    mse = ((predictions - targets) ** 2).mean()
    
    # RMSE (Root Mean Squared Error in m)
    rmse = torch.sqrt(mse)
    
    # iRMSE (Inverse RMSE in 1/km)
    # Convert from meters to kilometers for inverse depth
    inv_predictions = 1.0 / (predictions / 1000.0)  # Convert to 1/km
    inv_targets = 1.0 / (targets / 1000.0)
    irmse = torch.sqrt(((inv_predictions - inv_targets) ** 2).mean())
    
    # SILog (Scale Invariant Logarithmic Error)
    log_predictions = torch.log(predictions)
    log_targets = torch.log(targets)
    si_log = torch.sqrt(((log_predictions - log_targets) ** 2).mean() - (log_predictions - log_targets).mean() ** 2)
    si_log_percent = si_log * 100
    
    # Relative Squared Error (percent)
    sq_error_rel = (((predictions - targets) ** 2).sum() / ((targets) ** 2).sum() * 100)
    
    # Relative Absolute Error (percent)
    abs_error_rel = ((torch.abs(predictions - targets)).sum() / (torch.abs(targets)).sum() * 100)
    
    metrics = {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'irmse': irmse.item(),
        'silog': si_log_percent.item(),
        'sq_error_rel': sq_error_rel.item(),
        'abs_error_rel': abs_error_rel.item()
    }
    
    return metrics

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
    epochs = 300
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
    train_dict = torch.load('src/estimator/dataset/vehicle_dataset_ext.pt')
    features = train_dict['features']
    targets = train_dict['targets']
    dataset = VehicleAnnotationDataset(features, targets)
    
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Create DataLoaders
    # Note: num_workers must be 0 when using GPU tensors to avoid CUDA initialization errors in worker processes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    
    
    
    # Train the model
    model = train_distance_model(train_loader, val_loader, input_dim=features.shape[1], device=device, epochs=epochs, lr=lr, verbose=True)
    
    # Load test dataset
    test_dict = torch.load('src/estimator/dataset/vehicle_dataset_ext_test.pt')
    test_features = test_dict['features']
    test_targets = test_dict['targets']
    test_ds = VehicleAnnotationDataset(test_features, test_targets)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Evaluate on test set with comprehensive metrics
    print("\n=== Test Set Evaluation ===")
    test_metrics = compute_depth_metrics(model, test_loader, device)
    print(f"Test MSE:            {test_metrics['mse']:.4f} m²")
    print(f"Test iRMSE:          {test_metrics['irmse']:.4f} (1/km)")
    print(f"Test RMSE:           {test_metrics['rmse']:.4f} m")
    print(f"Test SILog:          {test_metrics['silog']:.4f} (log(m)*100)")
    print(f"Test Sq Error Rel:   {test_metrics['sq_error_rel']:.2f} %")
    print(f"Test Abs Error Rel:  {test_metrics['abs_error_rel']:.2f} %")
    
    # Try 10 random samples from test set
    print("\n=== Sample Predictions ===")
    model.eval()
    with torch.no_grad():
        for i in range(10):
            idx = torch.randint(0, len(test_ds), (1,)).item()
            input, target = test_ds[idx]
            input = input.to(device).unsqueeze(0)  # Add batch dimension
            prediction = model(input)
            error = abs(prediction.item() - target.item())
            error_pct = (error / target.item()) * 100
            print(f'Sample {i+1}: True = {target.item():.2f} m, Predicted = {prediction.item():.2f} m, Error = {error:.2f} m ({error_pct:.1f}%)')
            
    # Save the trained model
    os.makedirs('src/estimator/saved_models', exist_ok=True)
    model_path = 'src/estimator/saved_models/distance_regressor.pt'
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')
    
    
    