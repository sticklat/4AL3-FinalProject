import os
import json
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from pathlib import Path
import itertools
import random

from train import train_distance_model


def setup_output_dirs():
    """Create output directories for models and logs."""
    base_dir = Path("hyperopt_results")
    base_dir.mkdir(exist_ok=True)
    
    models_dir = base_dir / "models"
    logs_dir = base_dir / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    return base_dir, models_dir, logs_dir


def get_hyperparameter_grid():
    """Define hyperparameter search space."""
    return {
        'epochs': [50, 100, 150, 200, 300],
        'num_hidden_layers': [2, 3, 4, 5, 6, 7, 8],
        'hidden_dim': [64, 128, 256, 512, 1024],
        'activation': [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU],
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    }


def get_config_id(config):
    """Generate a unique ID for a configuration."""
    return f"epochs{config['epochs']}_layers{config['num_hidden_layers']}_dim{config['hidden_dim']}_act{config['activation'].__name__}_lr{config['learning_rate']:.0e}"


def train_and_evaluate(X, y, device, config, base_dir, models_dir, logs_dir, verbose=False):
    """
    Train a model with given hyperparameters and save results.
    
    Returns:
        dict: Contains final_val_loss, config_id, and other metrics
    """
    config_id = get_config_id(config)
    
    try:
        # Train model with loss tracking
        model, train_losses, val_losses = train_distance_model(
            X, y,
            device=device,
            epochs=config['epochs'],
            batch_size=64,
            lr=config['learning_rate'],
            hidden_dim=config['hidden_dim'],
            num_hiddenLyr=config['num_hidden_layers'],
            activation=config['activation'],
            verbose=verbose,
            return_losses=True
        )
        
        # Save model
        model_path = models_dir / f"{config_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save loss history
        loss_data = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
        }
        loss_df = pd.DataFrame(loss_data)
        loss_path = logs_dir / f"{config_id}_losses.csv"
        loss_df.to_csv(loss_path, index=False)
        
        # Get final validation loss
        final_val_loss = val_losses[-1]
        
        result = {
            'config_id': config_id,
            'epochs': config['epochs'],
            'num_hidden_layers': config['num_hidden_layers'],
            'hidden_dim': config['hidden_dim'],
            'activation': config['activation'].__name__,
            'learning_rate': config['learning_rate'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': final_val_loss,
            'min_train_loss': min(train_losses),
            'min_val_loss': min(val_losses),
            'model_path': str(model_path),
            'loss_path': str(loss_path),
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
        }
        
        if verbose:
            print(f"✓ {config_id}: Val Loss = {final_val_loss:.6f}")
        
        return result
        
    except Exception as e:
        result = {
            'config_id': config_id,
            'epochs': config['epochs'],
            'num_hidden_layers': config['num_hidden_layers'],
            'hidden_dim': config['hidden_dim'],
            'activation': config['activation'].__name__,
            'learning_rate': config['learning_rate'],
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }
        print(f"✗ {config_id}: Failed - {str(e)}")
        return result


def random_search_hyperopt(X, y, device, num_iterations=100, seed=42):
    """
    Perform random search over hyperparameter space.
    
    Args:
        num_iterations: Number of random configurations to try (default: 100)
    """
    base_dir, models_dir, logs_dir = setup_output_dirs()
    
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    grid = get_hyperparameter_grid()
    
    # Calculate total possible configurations for reference
    total_configs = 1
    for values in grid.values():
        total_configs *= len(values)
    
    print(f"Total possible configurations: {total_configs}")
    
    # Generate random configurations
    configs_to_try = []
    for _ in range(num_iterations):
        config = {
            'epochs': random.choice(grid['epochs']),
            'num_hidden_layers': random.choice(grid['num_hidden_layers']),
            'hidden_dim': random.choice(grid['hidden_dim']),
            'activation': random.choice(grid['activation']),
            'learning_rate': random.choice(grid['learning_rate']),
        }
        configs_to_try.append(config)
    
    print(f"Running random search with {len(configs_to_try)} configurations")
    
    # Results tracking
    results = []
    results_log = base_dir / "results.jsonl"
    summary_log = base_dir / "summary.csv"
    
    try:
        for i, config in enumerate(configs_to_try, 1):
            print(f"\n[{i}/{len(configs_to_try)}] Training with config:")
            print(f"  Epochs: {config['epochs']}")
            print(f"  Hidden Layers: {config['num_hidden_layers']}")
            print(f"  Hidden Dim: {config['hidden_dim']}")
            print(f"  Activation: {config['activation'].__name__}")
            print(f"  Learning Rate: {config['learning_rate']}")
            
            result = train_and_evaluate(X, y, device, config, base_dir, models_dir, logs_dir, verbose=False)
            results.append(result)
            
            # Save result to JSONL for progress tracking
            with open(results_log, 'a') as f:
                json.dump(result, f)
                f.write('\n')
            
            # Update summary CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(summary_log, index=False)
            
            print(f"  Final Val Loss: {result.get('final_val_loss', 'N/A')}")
            
    except KeyboardInterrupt:
        print("\n\n=== Interrupted by user ===")
        print(f"Completed {len(results)} configurations before interruption.")
    
    finally:
        # Final summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        results_df = pd.DataFrame(results)
        
        # Filter successful runs
        successful = results_df[results_df['status'] == 'success']
        
        if len(successful) > 0:
            successful_sorted = successful.sort_values('final_val_loss')
            
            print(f"\nSuccessful runs: {len(successful)}/{len(results)}")
            print("\nTop 5 configurations:")
            print(successful_sorted[['config_id', 'final_train_loss', 'final_val_loss', 'min_val_loss']].head().to_string())
            
            best = successful_sorted.iloc[0]
            print(f"\nBest configuration:")
            print(f"  Config ID: {best['config_id']}")
            print(f"  Final Train Loss: {best['final_train_loss']:.6f}")
            print(f"  Final Val Loss: {best['final_val_loss']:.6f}")
            print(f"  Min Val Loss: {best['min_val_loss']:.6f}")
            print(f"  Model saved to: {best['model_path']}")
            print(f"  Loss history saved to: {best['loss_path']}")
        
        failed = results_df[results_df['status'] == 'failed']
        if len(failed) > 0:
            print(f"\nFailed runs: {len(failed)}")
        
        print(f"\nAll results saved to: {base_dir}")
        print(f"  - Summary: {summary_log}")
        print(f"  - Detailed results: {results_log}")
        print(f"  - Models: {models_dir}")
        print(f"  - Loss histories: {logs_dir}")


if __name__ == "__main__":
    # Set device to cuda if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    torch.set_default_device(device)
    print(f"Using device: {torch.get_default_device()}")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = torch.load("dataset/vehicle_dataset.pt", map_location=device)
    X = dataset["features"]
    y = dataset["distances"]
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Run hyperparameter optimization
    # Adjust num_iterations to control how many random configs to try
    random_search_hyperopt(X, y, device, num_iterations=3000)
    
    print(f"\nEnd time: {datetime.now().isoformat()}")
