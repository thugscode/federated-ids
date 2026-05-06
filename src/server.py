"""
Federated learning server implementation.

Handles server-side aggregation and model coordination.
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated learning server."""
    
    def __init__(self, model, strategy, num_clients: int):
        """
        Initialize server.
        
        Args:
            model: Initial model
            strategy: Aggregation strategy
            num_clients: Number of clients
        """
        self.model = model
        self.strategy = strategy
        self.num_clients = num_clients
        self.round = 0
        self.history = {
            "loss": [],
            "accuracy": [],
            "client_metrics": []
        }
        
        logger.info(f"Server initialized with {num_clients} clients")
        logger.info(f"Aggregation strategy: {strategy.__class__.__name__}")
    
    def aggregate_models(self, client_models: List[Tuple[List, List]], 
                        client_sizes: List[int]) -> Tuple[List, List]:
        """
        Aggregate client models using the selected strategy.
        
        Args:
            client_models: List of (weights, biases) tuples from clients
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated (weights, biases)
        """
        aggregated_weights, aggregated_biases = self.strategy.aggregate(client_models, client_sizes)
        return aggregated_weights, aggregated_biases
    
    def broadcast_model(self) -> Tuple[List, List]:
        """
        Broadcast current model to clients.
        
        Returns:
            Current (weights, biases)
        """
        weights, biases = self.model.get_weights()
        logger.info(f"Broadcasting global model (Round {self.round})")
        return weights, biases
    
    def federated_round(self, clients, fraction: float = 1.0, 
                       num_local_epochs: int = 5) -> Dict:
        """
        Execute one federated learning round.
        
        Args:
            clients: List of FederatedClient objects
            fraction: Fraction of clients to participate
            num_local_epochs: Number of local training epochs
            
        Returns:
            Dictionary with round statistics
        """
        self.round += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Federated Round {self.round}")
        logger.info(f"{'='*60}")
        
        # Select clients
        num_selected = max(1, int(len(clients) * fraction))
        selected_clients = np.random.choice(clients, num_selected, replace=False)
        logger.info(f"Selected {num_selected}/{len(clients)} clients")
        
        # Broadcast model
        global_weights, global_biases = self.broadcast_model()
        
        # Local training
        client_models = []
        client_sizes = []
        round_metrics = []
        
        for client in selected_clients:
            # Set global weights
            client.set_weights(global_weights, global_biases)
            
            # Local update
            weights, biases, loss = client.local_update(num_epochs=num_local_epochs)
            client_models.append((weights, biases))
            client_sizes.append(client.get_dataset_size())
            
            # Evaluate
            metrics = client.evaluate()
            metrics["client_id"] = client.client_id
            metrics["loss"] = loss
            round_metrics.append(metrics)
            
            logger.info(f"  Client {client.client_id}: Loss={loss:.4f}, Acc={metrics.get('accuracy', 0):.4f}")
        
        # Aggregate models
        aggregated_weights, aggregated_biases = self.aggregate_models(client_models, client_sizes)
        self.model.set_weights(aggregated_weights, aggregated_biases)
        
        # Compute statistics
        avg_loss = np.mean([m["loss"] for m in round_metrics])
        avg_accuracy = np.mean([m["accuracy"] for m in round_metrics])
        avg_f1 = np.mean([m["f1"] for m in round_metrics])
        
        logger.info(f"\nRound {self.round} Summary:")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        logger.info(f"  Avg Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Avg F1-Score: {avg_f1:.4f}")
        
        # Store history
        self.history["loss"].append(avg_loss)
        self.history["accuracy"].append(avg_accuracy)
        self.history["client_metrics"].append(round_metrics)
        
        return {
            "round": self.round,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "f1": avg_f1,
            "num_clients": num_selected,
            "client_metrics": round_metrics
        }
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate global model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # Binary classification metrics
        TP = np.sum((predictions == 1) & (y_test == 1))
        TN = np.sum((predictions == 0) & (y_test == 0))
        FP = np.sum((predictions == 1) & (y_test == 0))
        FN = np.sum((predictions == 0) & (y_test == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": TP,
            "tn": TN,
            "fp": FP,
            "fn": FN
        }
        
        logger.info(f"Global model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    def save_checkpoint(self, output_dir: str):
        """Save server state and model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save history
        with open(output_path / f"history_round_{self.round}.json", "w") as f:
            json.dump({
                "loss": [float(l) for l in self.history["loss"]],
                "accuracy": [float(a) for a in self.history["accuracy"]]
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved to {output_path}")
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
    
    def get_model(self):
        """Get current global model."""
        return self.model
    
    def get_round(self) -> int:
        """Get current round number."""
        return self.round


if __name__ == "__main__":
    from model import NeuralNetwork
    from strategies import FedAvg
    
    # Create model
    model = NeuralNetwork(input_dim=78, hidden_dims=[128, 64], output_dim=1)
    
    # Create strategy
    strategy = FedAvg(num_clients=10)
    
    # Create server
    server = FederatedServer(model, strategy, num_clients=10)
    logger.info(f"Server created successfully")
