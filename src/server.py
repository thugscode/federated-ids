"""
Flower federated learning server for intrusion detection.

Orchestrates federated learning rounds using Flower framework.
Implements FedAvg strategy with metrics aggregation and logging.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, FitRes, Parameters
from typing import List, Tuple, Dict, Optional
import logging
import json
from pathlib import Path
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from clients using weighted averaging.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}
    
    # Extract total examples and metrics
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Initialize aggregated metrics
    aggregated = {
        "loss": 0.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
    }
    
    # Per-class F1 scores aggregation
    per_class_f1_sums = {}
    
    # Aggregate each metric
    for num_examples, client_metrics in metrics:
        weight = num_examples / total_examples if total_examples > 0 else 0
        
        # Weighted aggregation
        if "loss" in client_metrics:
            aggregated["loss"] += weight * client_metrics["loss"]
        if "accuracy" in client_metrics:
            aggregated["accuracy"] += weight * client_metrics["accuracy"]
        if "macro_f1" in client_metrics:
            aggregated["macro_f1"] += weight * client_metrics["macro_f1"]
        if "weighted_f1" in client_metrics:
            aggregated["weighted_f1"] += weight * client_metrics["weighted_f1"]
        
        # Per-class F1 aggregation
        for key, value in client_metrics.items():
            if key.startswith("f1_class_"):
                if key not in per_class_f1_sums:
                    per_class_f1_sums[key] = 0.0
                per_class_f1_sums[key] += weight * value
    
    # Add per-class F1 to aggregated metrics
    aggregated.update(per_class_f1_sums)
    
    return aggregated


def create_fedavg_strategy(
    min_fit_clients: int = 5,
    min_available_clients: int = 5,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_evaluate_clients: int = 5,
    num_rounds: int = 50
) -> FedAvg:
    """
    Create a FedAvg strategy with specified configuration.
    
    Args:
        min_fit_clients: Minimum clients to participate in training
        min_available_clients: Minimum clients available
        fraction_fit: Fraction of clients to participate in training
        fraction_evaluate: Fraction of clients to participate in evaluation
        min_evaluate_clients: Minimum clients for evaluation
        num_rounds: Number of federated rounds
        
    Returns:
        Configured FedAvg strategy
    """
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=None,  # Will be set by server
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    logger.info("FedAvg strategy created:")
    logger.info(f"  Fraction fit: {fraction_fit}")
    logger.info(f"  Min fit clients: {min_fit_clients}")
    logger.info(f"  Min available clients: {min_available_clients}")
    logger.info(f"  Evaluate metrics aggregation: Yes")
    
    return strategy


class MetricsLogger:
    """
    Logger for tracking federated learning metrics across rounds.
    """
    
    def __init__(self, output_dir: Path = Path("results/metrics")):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "timestamp": datetime.now().isoformat(),
            "rounds": []
        }
        
        logger.info(f"MetricsLogger initialized - output dir: {self.output_dir}")
    
    def log_round(self, round_num: int, metrics: Dict, fit_metrics: Dict = None):
        """
        Log metrics for a single round.
        
        Args:
            round_num: Current round number
            metrics: Evaluation metrics dict
            fit_metrics: Optional training metrics dict
        """
        round_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "eval_metrics": metrics,
        }
        
        if fit_metrics:
            round_data["fit_metrics"] = fit_metrics
        
        self.history["rounds"].append(round_data)
        
        # Save to JSON after each round
        self.save()
        
        # Log to console
        logger.info(f"Round {round_num} completed:")
        if metrics:
            loss = metrics.get('loss', None)
            if loss is not None:
                logger.info(f"  Eval Loss: {loss:.4f}")
            
            accuracy = metrics.get('accuracy', None)
            if accuracy is not None:
                logger.info(f"  Eval Accuracy: {accuracy:.4f}")
            
            macro_f1 = metrics.get('macro_f1', None)
            if macro_f1 is not None:
                logger.info(f"  Eval Macro F1: {macro_f1:.4f}")
            
            weighted_f1 = metrics.get('weighted_f1', None)
            if weighted_f1 is not None:
                logger.info(f"  Eval Weighted F1: {weighted_f1:.4f}")
    
    def save(self, filename: str = "federated_metrics.json"):
        """
        Save metrics history to JSON file.
        
        Args:
            filename: Name of output file
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


def run_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 50,
    min_fit_clients: int = 5,
    min_available_clients: int = 5,
    num_client_cpus: int = 1,
    num_client_gpus: float = 0.0,
):
    """
    Run Flower federated learning server.
    
    Args:
        server_address: Server address (host:port)
        num_rounds: Number of federated rounds
        min_fit_clients: Minimum clients for training
        min_available_clients: Minimum available clients
        num_client_cpus: CPU resources per client
        num_client_gpus: GPU resources per client
    """
    
    # Create strategy
    strategy = create_fedavg_strategy(
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        num_rounds=num_rounds
    )
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger()
    
    # Configure server
    config = fl.server.ServerConfig(
        num_rounds=num_rounds,
        round_timeout=600,
    )
    
    logger.info(f"Starting Flower server on {server_address}")
    logger.info(f"Configuration:")
    logger.info(f"  Number of rounds: {num_rounds}")
    logger.info(f"  Min fit clients: {min_fit_clients}")
    logger.info(f"  Min available clients: {min_available_clients}")
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flower server for federated IDS learning"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of federated rounds"
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=5,
        help="Minimum clients for training"
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=5,
        help="Minimum available clients"
    )
    parser.add_argument(
        "--num_client_cpus",
        type=int,
        default=1,
        help="CPU resources per client"
    )
    parser.add_argument(
        "--num_client_gpus",
        type=float,
        default=0.0,
        help="GPU resources per client"
    )
    
    args = parser.parse_args()
    
    run_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        num_client_cpus=args.num_client_cpus,
        num_client_gpus=args.num_client_gpus,
    )
