"""
Federated Learning with Differential Privacy experiment.

Implements federated learning with differential privacy (FedDP).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from pathlib import Path
import json

from model import NeuralNetwork
from dataset import load_processed_data, FederatedDataset
from client import FederatedClient, ClientManager
from server import FederatedServer
from strategies import FedDP
from evaluate import ModelEvaluator
from visualize import FederatedVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_feddp_experiment(data_dir: str = "data/processed",
                        output_dir: str = "results",
                        num_clients: int = 10,
                        num_rounds: int = 20,
                        local_epochs: int = 5,
                        batch_size: int = 32,
                        learning_rate: float = 0.01,
                        epsilon: float = 1.0,
                        delta: float = 1e-5,
                        partition: str = "non-iid",
                        alpha: float = 0.1):
    """
    Run Federated Learning with Differential Privacy experiment.
    
    Args:
        data_dir: Path to preprocessed data
        output_dir: Output directory for results
        num_clients: Number of clients
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per client
        batch_size: Batch size for training
        learning_rate: Learning rate
        epsilon: Privacy budget (lower = more private)
        delta: Privacy delta parameter
        partition: Data partition method ("iid" or "non-iid")
        alpha: Dirichlet parameter for non-iid partitioning
    """
    logger.info("="*60)
    logger.info("FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY (FedDP)")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)
    
    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Test data: {X_test.shape}")
    
    # Create federated dataset
    logger.info(f"Creating federated dataset with {num_clients} clients ({partition} partitioning)...")
    fed_dataset = FederatedDataset(X_train, y_train, num_clients=num_clients,
                                   partition_method=partition, alpha=alpha)
    
    client_sizes = fed_dataset.get_client_sizes()
    logger.info(f"Client dataset sizes: {client_sizes}")
    
    # Create global model
    input_dim = X_train.shape[1]
    global_model = NeuralNetwork(input_dim=input_dim, hidden_dims=[128, 64],
                                output_dim=1, learning_rate=learning_rate)
    
    # Create strategy and server
    strategy = FedDP(num_clients, epsilon=epsilon, delta=delta)
    server = FederatedServer(global_model, strategy, num_clients)
    
    logger.info(f"Privacy settings: ε={epsilon}, δ={delta}")
    
    # Create clients
    client_manager = ClientManager()
    for client_id in range(num_clients):
        X_client, y_client = fed_dataset.get_client_data(client_id)
        
        client = FederatedClient(
            client_id=client_id,
            model=NeuralNetwork(input_dim=input_dim, hidden_dims=[128, 64],
                              output_dim=1, learning_rate=learning_rate),
            X_train=X_client,
            y_train=y_client,
            X_test=X_test,
            y_test=y_test,
            batch_size=batch_size
        )
        client_manager.add_client(client)
    
    # Federated training
    logger.info(f"Starting federated training for {num_rounds} rounds...")
    all_clients = client_manager.get_all_clients()
    
    for round_num in range(num_rounds):
        round_result = server.federated_round(all_clients, fraction=1.0, 
                                             num_local_epochs=local_epochs)
    
    # Evaluate final model
    logger.info("\nFinal Evaluation:")
    evaluator = ModelEvaluator()
    final_metrics = evaluator.evaluate_model(server.get_model(), X_test, y_test)
    evaluator.print_metrics(final_metrics)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = server.get_history()
    results = {
        "experiment": "feddp",
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "delta": delta,
        "partition": partition,
        "alpha": alpha,
        "client_sizes": client_sizes,
        "training_history": {
            "loss": [float(l) for l in history["loss"]],
            "accuracy": [float(a) for a in history["accuracy"]]
        },
        "final_metrics": final_metrics,
        "privacy_info": {
            "epsilon": epsilon,
            "delta": delta,
            "mechanism": "Laplace",
            "description": "Noise added at aggregation step"
        }
    }
    
    with open(output_path / "metrics" / "feddp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path / 'metrics' / 'feddp_results.json'}")
    
    # Visualize
    visualizer = FederatedVisualizer(output_dir=str(output_path / "plots"))
    visualizer.plot_convergence(history)
    visualizer.plot_metrics_comparison(final_metrics)
    visualizer.plot_client_distribution(client_sizes)
    
    logger.info("="*60)
    logger.info("FedDP EXPERIMENT COMPLETED")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FedDP federated learning experiment")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--partition", type=str, default="non-iid", choices=["iid", "non-iid"])
    parser.add_argument("--alpha", type=float, default=0.1)
    
    args = parser.parse_args()
    
    run_feddp_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        delta=args.delta,
        partition=args.partition,
        alpha=args.alpha
    )
