"""
Visualization module for federated learning results.

Creates plots and figures for analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedVisualizer:
    """Visualize federated learning results."""
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_training_history(self, history: Dict, metric: str = "loss"):
        """
        Plot training history over rounds.
        
        Args:
            history: History dictionary from server
            metric: Metric to plot ("loss", "accuracy", etc.)
        """
        if metric not in history:
            logger.warning(f"Metric {metric} not found in history")
            return
        
        values = history[metric]
        rounds = range(1, len(values) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, values, marker='o', linewidth=2, markersize=6)
        plt.xlabel("Round", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f"Training {metric.capitalize()} Over Rounds", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / f"training_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_convergence(self, history: Dict):
        """
        Plot convergence of loss and accuracy.
        
        Args:
            history: History dictionary from server
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if "loss" in history:
            loss = history["loss"]
            ax1.plot(range(1, len(loss) + 1), loss, marker='o', color='#e74c3c', linewidth=2)
            ax1.set_xlabel("Round", fontsize=11)
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.set_title("Loss Convergence", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if "accuracy" in history:
            accuracy = history["accuracy"]
            ax2.plot(range(1, len(accuracy) + 1), accuracy, marker='o', color='#27ae60', linewidth=2)
            ax2.set_xlabel("Round", fontsize=11)
            ax2.set_ylabel("Accuracy", fontsize=11)
            ax2.set_title("Accuracy Improvement", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "convergence.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix: Dict, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix dictionary
            title: Plot title
        """
        cm = np.array([
            [confusion_matrix.get("TP", 0), confusion_matrix.get("FP", 0)],
            [confusion_matrix.get("FN", 0), confusion_matrix.get("TN", 0)]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Positive', 'Negative'],
                   yticklabels=['Positive', 'Negative'])
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict):
        """
        Plot comparison of different metrics.
        
        Args:
            metrics: Metrics dictionary
        """
        metric_names = ["accuracy", "precision", "recall", "f1", "specificity"]
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.ylabel("Score", fontsize=12)
        plt.title("Model Performance Metrics", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_client_distribution(self, client_sizes: Dict):
        """
        Plot data distribution across clients.
        
        Args:
            client_sizes: Dictionary mapping client_id to dataset size
        """
        client_ids = sorted(client_sizes.keys())
        sizes = [client_sizes[cid] for cid in client_ids]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([str(cid) for cid in client_ids], sizes, color='#3498db', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(size)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel("Client ID", fontsize=12)
        plt.ylabel("Dataset Size (samples)", fontsize=12)
        plt.title("Data Distribution Across Clients", fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "client_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_strategy_comparison(self, strategies_results: Dict[str, Dict]):
        """
        Compare results across different FL strategies.
        
        Args:
            strategies_results: Dictionary mapping strategy names to results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        strategy_names = list(strategies_results.keys())
        metrics = ["accuracy", "precision", "f1", "auc_roc"]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = []
            
            for strategy in strategy_names:
                if "final_metrics" in strategies_results[strategy]:
                    val = strategies_results[strategy]["final_metrics"].get(metric, 0)
                    values.append(val)
                else:
                    values.append(0)
            
            colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12'][:len(strategy_names)]
            bars = ax.bar(strategy_names, values, color=colors, alpha=0.7, edgecolor='black')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f"{metric.capitalize()} Comparison", fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "strategy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_privacy_utility_tradeoff(self, privacy_levels: List, accuracies: List):
        """
        Plot privacy-utility tradeoff.
        
        Args:
            privacy_levels: List of epsilon values
            accuracies: List of corresponding accuracies
        """
        plt.figure(figsize=(10, 6))
        plt.plot(privacy_levels, accuracies, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        plt.xlabel("Privacy Budget (ε)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Privacy-Utility Tradeoff", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "privacy_utility_tradeoff.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()


def create_summary_report(results: Dict, output_dir: str = "results/metrics"):
    """
    Create a summary report of experimental results.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Summary report saved to {output_path / 'summary.json'}")


if __name__ == "__main__":
    # Test visualizer
    visualizer = FederatedVisualizer()
    
    # Test data
    history = {
        "loss": [0.8, 0.7, 0.6, 0.5, 0.4],
        "accuracy": [0.6, 0.65, 0.70, 0.75, 0.78]
    }
    
    visualizer.plot_convergence(history)
    
    metrics = {
        "accuracy": 0.78,
        "precision": 0.75,
        "recall": 0.80,
        "f1": 0.77,
        "specificity": 0.76
    }
    
    visualizer.plot_metrics_comparison(metrics)
    
    logger.info("Visualization tests completed")
