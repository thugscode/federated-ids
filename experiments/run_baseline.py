"""
Baseline centralized learning experiment.

Trains a centralized model on all data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from pathlib import Path
import json

from model import NeuralNetwork
from dataset import load_processed_data, DataLoader
from evaluate import ModelEvaluator
from visualize import FederatedVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline_experiment(data_dir: str = "data/processed", 
                           output_dir: str = "results",
                           epochs: int = 50,
                           batch_size: int = 32,
                           learning_rate: float = 0.01):
    """
    Run centralized baseline experiment.
    
    Args:
        data_dir: Path to preprocessed data
        output_dir: Output directory for results
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    logger.info("="*60)
    logger.info("BASELINE CENTRALIZED LEARNING EXPERIMENT")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)
    
    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Test data: {X_test.shape}")
    
    # Create model
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim=input_dim, hidden_dims=[128, 64], 
                         output_dim=1, learning_rate=learning_rate)
    
    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    train_losses = []
    
    for epoch in range(epochs):
        # Create data loader
        loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in loader:
            y_batch = y_batch.reshape(-1, 1)
            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    evaluator.print_metrics(metrics)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "baseline_centralized",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_data_size": len(X_train),
        "test_data_size": len(X_test),
        "train_losses": [float(l) for l in train_losses],
        "final_metrics": metrics
    }
    
    with open(output_path / "metrics" / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path / 'metrics' / 'baseline_results.json'}")
    
    # Visualize
    visualizer = FederatedVisualizer(output_dir=str(output_path / "plots"))
    visualizer.plot_metrics_comparison(metrics)
    
    logger.info("="*60)
    logger.info("BASELINE EXPERIMENT COMPLETED")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline centralized learning experiment")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
