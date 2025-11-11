"""
Hyperoptimization Results Visualization Script

This script provides comprehensive visualizations for hyperparameter optimization results.
It generates plots for:
- Best performing models and their metrics
- Loss curves for individual models
- Parameter space exploration
- Hyperparameter importance analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from typing import Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class HyperoptVisualizer:
    """Visualizer for hyperoptimization results."""
    
    def __init__(self, results_dir: str = "./hyperopt_results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Path to the hyperopt_results directory
        """
        self.results_dir = Path(results_dir)
        self.summary_path = self.results_dir / "summary.csv"
        self.logs_dir = self.results_dir / "logs"
        
        # Load the summary data
        self.df = pd.read_csv(self.summary_path)
        print(f"✓ Loaded {len(self.df)} hyperopt results")
        
    def plot_best_models(self, top_n: int = 10, save_path: Optional[str] = None) -> None:
        """
        Plot the top N best performing models based on min validation loss.
        
        Args:
            top_n: Number of top models to display
            save_path: Path to save the figure (optional)
        """
        top_models = self.df.nsmallest(top_n, 'min_val_loss')[
            ['config_id', 'min_val_loss', 'min_train_loss', 'epochs', 
             'num_hidden_layers', 'hidden_dim', 'activation', 'learning_rate']
        ].reset_index(drop=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Validation Loss Comparison
        ax = axes[0]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_models)))
        bars1 = ax.barh(range(len(top_models)), top_models['min_val_loss'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_models)))
        ax.set_yticklabels(top_models['config_id'], fontsize=9)
        ax.set_xlabel('Minimum Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Models by Validation Loss', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, top_models['min_val_loss'])):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
        
        # Plot 2: Train vs Validation Loss
        ax = axes[1]
        x = np.arange(len(top_models))
        width = 0.35
        ax.bar(x - width/2, top_models['min_train_loss'], width, label='Min Train Loss', 
               alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, top_models['min_val_loss'], width, label='Min Val Loss', 
               alpha=0.8, color='#A23B72')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Train vs Validation Loss (Top Models)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(range(1, len(top_models) + 1))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved best models plot to {save_path}")
        plt.show()
        
    def plot_hyperparameter_impact(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the impact of different hyperparameters on validation loss.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Filter out extreme outliers for better visualization
        df_filtered = self.df[self.df['min_val_loss'] < self.df['min_val_loss'].quantile(0.95)]
        
        # Plot 1: Epochs vs Validation Loss
        ax = axes[0, 0]
        for epochs in sorted(df_filtered['epochs'].unique()):
            data = df_filtered[df_filtered['epochs'] == epochs]['min_val_loss']
            ax.scatter([epochs] * len(data), data, alpha=0.6, s=50, label=f'{epochs}')
        ax.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax.set_ylabel('Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Impact of Epochs on Validation Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Hidden Dimension vs Validation Loss
        ax = axes[0, 1]
        for dim in sorted(df_filtered['hidden_dim'].unique()):
            data = df_filtered[df_filtered['hidden_dim'] == dim]['min_val_loss']
            ax.scatter([dim] * len(data), data, alpha=0.6, s=50)
        ax.set_xlabel('Hidden Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel('Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Impact of Hidden Dimension on Validation Loss', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Number of Layers vs Validation Loss
        ax = axes[1, 0]
        for layers in sorted(df_filtered['num_hidden_layers'].unique()):
            data = df_filtered[df_filtered['num_hidden_layers'] == layers]['min_val_loss']
            ax.scatter([layers] * len(data), data, alpha=0.6, s=50)
        ax.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
        ax.set_ylabel('Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Impact of Layer Count on Validation Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Activation Function Impact
        ax = axes[1, 1]
        activations = df_filtered['activation'].unique()
        activation_means = [df_filtered[df_filtered['activation'] == act]['min_val_loss'].mean() 
                           for act in activations]
        colors = plt.cm.Set3(np.linspace(0, 1, len(activations)))
        ax.bar(activations, activation_means, color=colors, alpha=0.8)
        ax.set_ylabel('Mean Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Activation Function Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved hyperparameter impact plot to {save_path}")
        plt.show()
        
    def plot_learning_rate_impact(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the impact of learning rate on model performance.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Filter outliers
        df_filtered = self.df[self.df['min_val_loss'] < self.df['min_val_loss'].quantile(0.95)]
        
        # Plot 1: Learning Rate vs Validation Loss
        ax = axes[0]
        scatter = ax.scatter(df_filtered['learning_rate'], df_filtered['min_val_loss'], 
                           c=df_filtered['epochs'], cmap='viridis', s=100, alpha=0.6)
        ax.set_xlabel('Learning Rate (log scale)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Learning Rate Impact on Validation Loss', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epochs', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate distribution by activation
        ax = axes[1]
        for activation in df_filtered['activation'].unique():
            data = df_filtered[df_filtered['activation'] == activation]
            ax.scatter(data['learning_rate'], data['min_val_loss'], 
                      label=activation, alpha=0.6, s=80)
        ax.set_xlabel('Learning Rate (log scale)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Min Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Learning Rate by Activation Function', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved learning rate impact plot to {save_path}")
        plt.show()
        
    def plot_loss_curves(self, config_ids: Optional[List[str]] = None, 
                        save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss curves for specific models.
        
        Args:
            config_ids: List of config IDs to plot (if None, plots top 6 models)
            save_path: Path to save the figure (optional)
        """
        if config_ids is None:
            # Select top 6 models
            config_ids = self.df.nsmallest(6, 'min_val_loss')['config_id'].tolist()
        
        n_models = len(config_ids)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, config_id in enumerate(config_ids):
            ax = axes[idx]
            
            # Find the loss file
            loss_path = self.logs_dir / f"{config_id}_losses.csv"
            
            if loss_path.exists():
                df_losses = pd.read_csv(loss_path)
                ax.plot(df_losses['epoch'], df_losses['train_loss'], 
                       label='Train Loss', linewidth=2, color='#2E86AB', marker='o', markersize=3)
                ax.plot(df_losses['epoch'], df_losses['val_loss'], 
                       label='Val Loss', linewidth=2, color='#A23B72', marker='s', markersize=3)
                ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
                ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
                ax.set_title(config_id, fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Loss file not found\nfor {config_id}', 
                       ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved loss curves plot to {save_path}")
        plt.show()
        
    def plot_2d_parameter_space(self, param1: str = 'epochs', param2: str = 'hidden_dim',
                               save_path: Optional[str] = None) -> None:
        """
        Plot 2D parameter space colored by validation loss.
        
        Args:
            param1: First parameter to plot (x-axis)
            param2: Second parameter to plot (y-axis)
            save_path: Path to save the figure (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter outliers for better visualization
        df_filtered = self.df[self.df['min_val_loss'] < self.df['min_val_loss'].quantile(0.95)]
        
        scatter = ax.scatter(df_filtered[param1], df_filtered[param2],
                           c=df_filtered['min_val_loss'], cmap='RdYlGn_r', 
                           s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(param1.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(param2.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{param1.title()} vs {param2.title()} (colored by Val Loss)', 
                    fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Min Validation Loss', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 2D parameter space plot to {save_path}")
        plt.show()
        
    def print_summary_statistics(self) -> None:
        """Print summary statistics of the hyperopt results."""
        print("\n" + "="*60)
        print("HYPEROPTIMIZATION RESULTS SUMMARY".center(60))
        print("="*60)
        
        print(f"\nTotal configurations tested: {len(self.df)}")
        print(f"\nBest Model: {self.df.loc[self.df['min_val_loss'].idxmin(), 'config_id']}")
        print(f"  - Min Validation Loss: {self.df['min_val_loss'].min():.6f}")
        print(f"  - Min Train Loss: {self.df.loc[self.df['min_val_loss'].idxmin(), 'min_train_loss']:.6f}")
        
        print(f"\nValidation Loss Statistics:")
        print(f"  - Mean: {self.df['min_val_loss'].mean():.6f}")
        print(f"  - Median: {self.df['min_val_loss'].median():.6f}")
        print(f"  - Std Dev: {self.df['min_val_loss'].std():.6f}")
        print(f"  - Min: {self.df['min_val_loss'].min():.6f}")
        print(f"  - Max: {self.df['min_val_loss'].max():.6f}")
        
        print(f"\nParameter Ranges:")
        print(f"  - Epochs: {self.df['epochs'].min()} - {self.df['epochs'].max()}")
        print(f"  - Layers: {self.df['num_hidden_layers'].min()} - {self.df['num_hidden_layers'].max()}")
        print(f"  - Hidden Dim: {self.df['hidden_dim'].min()} - {self.df['hidden_dim'].max()}")
        print(f"  - Learning Rates: {sorted(self.df['learning_rate'].unique())}")
        print(f"  - Activations: {sorted(self.df['activation'].unique())}")
        
        print(f"\nTop 5 Best Models:")
        top_5 = self.df.nsmallest(5, 'min_val_loss')[['config_id', 'min_val_loss', 'min_train_loss']]
        for idx, row in enumerate(top_5.itertuples(), 1):
            print(f"  {idx}. {row.config_id}")
            print(f"     Val Loss: {row.min_val_loss:.6f}, Train Loss: {row.min_train_loss:.6f}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Initialize visualizer
    viz = HyperoptVisualizer(results_dir="./hyperopt_results")
    
    # Print summary statistics
    viz.print_summary_statistics()
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    viz.plot_best_models(top_n=15, save_path="hyperopt_result_graphs/best_models.png")
    viz.plot_hyperparameter_impact(save_path="hyperopt_result_graphs/hyperparameter_impact.png")
    viz.plot_learning_rate_impact(save_path="hyperopt_result_graphs/learning_rate_impact.png")
    viz.plot_loss_curves(save_path="hyperopt_result_graphs/loss_curves.png")
    viz.plot_2d_parameter_space(param1='epochs', param2='num_hidden_layers',
                               save_path="hyperopt_result_graphs/parameter_space_epochs_layers.png")
    viz.plot_2d_parameter_space(param1='hidden_dim', param2='learning_rate',
                               save_path="hyperopt_result_graphs/parameter_space_dim_lr.png")
    
    print("\n✓ All visualizations completed!")
