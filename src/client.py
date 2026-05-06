"""
Flower federated learning client for intrusion detection.

Implements IDSClient which wraps the IDSNet model and local dataset,
managing the training and evaluation workflow for federated learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import flwr as fl
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from model import IDSNet


class IDSClient(fl.client.NumPyClient):
    """
    Flower federated learning client for IDS task.
    
    Wraps IDSNet model and local dataset. Implements the NumPyClient interface
    for federated learning with Flower framework.
    
    Key responsibilities:
    - get_parameters(): Return current model weights as numpy arrays
    - fit(): Train locally for local_epochs and return updated weights
    - evaluate(): Evaluate model on test set and return loss + metrics
    
    Attributes:
        model: IDSNet neural network instance
        trainloader: DataLoader for training data
        valloader: DataLoader for validation data
        device: torch.device (cpu or cuda)
        epochs: Number of local training epochs
        learning_rate: Optimizer learning rate
        criterion: Loss function (CrossEntropyLoss)
    """
    
    def __init__(
        self,
        model: IDSNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """
        Initialize Flower IDS client.
        
        Args:
            model: IDSNet model instance
            X_train: Training features (n_samples, 77)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to use (cpu or cuda)
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Criterion for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        self.valloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        logger.info(f"Client initialized")
        logger.info(f"  Training samples: {len(self.train_dataset)}")
        logger.info(f"  Validation samples: {len(self.val_dataset)}")
        logger.info(f"  Local epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model parameters as a list of numpy arrays.
        
        Called by Flower server to retrieve the current model weights
        for aggregation across clients.
        
        Args:
            config: Configuration dict from server (unused)
            
        Returns:
            List of model parameter numpy arrays
        """
        return [val.cpu().detach().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of numpy arrays.
        
        Called by Flower server after aggregation to update the model
        with the newly aggregated weights.
        
        Args:
            parameters: List of numpy arrays to set as model weights
        """
        params_dict = zip(self.model.parameters(), parameters)
        for param, np_param in params_dict:
            param.data = torch.from_numpy(np_param).float().to(self.device)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model locally for local_epochs.
        
        Called by Flower server to perform local training on the client's data.
        
        Args:
            parameters: Current model parameters from server
            config: Configuration from server (e.g., local epochs)
            
        Returns:
            Tuple of:
            - updated model parameters as numpy arrays
            - number of training examples used
            - metrics dict containing training loss
        """
        # Update model with aggregated parameters
        self.set_parameters(parameters)
        
        # Get local epochs from config if provided
        local_epochs = config.get("local_epochs", self.epochs)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train for local_epochs
        logger.info(f"Starting local training for {local_epochs} epochs")
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            self.model.train()
            for batch_idx, (features, labels) in enumerate(self.trainloader):
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
            
            avg_epoch_loss = epoch_loss / len(self.train_dataset)
            total_loss += avg_epoch_loss
            
            logger.info(f"Epoch {epoch + 1}/{local_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Calculate average loss over all epochs
        avg_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
        
        logger.info(f"Local training completed - Average loss: {avg_loss:.4f}")
        
        return (
            self.get_parameters(config={}),
            len(self.train_dataset),
            {"loss": float(avg_loss)}
        )
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on validation data.
        
        Called by Flower server to assess model performance on the client's
        validation set.
        
        Args:
            parameters: Current model parameters from server
            config: Configuration from server
            
        Returns:
            Tuple of:
            - validation loss
            - number of validation examples
            - metrics dict with loss and per-class F1 scores
        """
        # Update model with aggregated parameters
        self.set_parameters(parameters)
        
        # Evaluate in no_grad mode
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.valloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                # Accumulate loss and predictions
                val_loss += loss.item() * features.size(0)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss
        avg_val_loss = val_loss / len(self.val_dataset) if len(self.val_dataset) > 0 else 0.0
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute per-class F1 scores
        from sklearn.metrics import f1_score, accuracy_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Macro F1 (average of F1 for each class)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Weighted F1 (weighted by support)
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Per-class F1 scores
        per_class_f1 = {}
        for class_idx in sorted(set(all_labels)):
            binary_labels = (all_labels == class_idx).astype(int)
            binary_predictions = (all_predictions == class_idx).astype(int)
            class_f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
            per_class_f1[f"f1_class_{class_idx}"] = float(class_f1)
        
        metrics = {
            "loss": float(avg_val_loss),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }
        metrics.update(per_class_f1)
        
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}, "
                   f"Macro F1: {macro_f1:.4f}")
        
        return avg_val_loss, len(self.val_dataset), metrics


def create_client(
    client_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int = 77,
    n_classes: int = 15,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> IDSClient:
    """
    Factory function to create an IDS client.
    
    Args:
        client_id: Client identifier for logging
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_features: Number of input features
        n_classes: Number of output classes
        epochs: Local training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cpu or cuda)
        
    Returns:
        Configured IDSClient instance
    """
    # Create model
    model = IDSNet(n_features=n_features, n_classes=n_classes, dropout=0.3)
    
    # Create client
    client = IDSClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    logger.info(f"Created client {client_id}")
    return client


if __name__ == "__main__":
    # Example usage: Create a dummy client for testing
    logger.info("Testing IDSClient")
    
    # Create dummy data
    X_train = np.random.randn(100, 77).astype(np.float32)
    y_train = np.random.randint(0, 15, 100)
    X_val = np.random.randn(50, 77).astype(np.float32)
    y_val = np.random.randint(0, 15, 50)
    
    # Create client
    client = create_client(
        client_id=0,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        device="cpu"
    )
    
    # Test get_parameters
    params = client.get_parameters(config={})
    logger.info(f"Got {len(params)} parameter tensors")
    
    # Test fit
    logger.info("Testing fit...")
    new_params, num_examples, metrics = client.fit(params, config={"local_epochs": 2})
    logger.info(f"Fit completed: {num_examples} examples, metrics: {metrics}")
    
    # Test evaluate
    logger.info("Testing evaluate...")
    val_loss, num_val_examples, eval_metrics = client.evaluate(new_params, config={})
    logger.info(f"Evaluation completed: {num_val_examples} examples, loss: {val_loss:.4f}")
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    logger.info("✓ IDSClient tests passed")
