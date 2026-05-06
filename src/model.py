"""
PyTorch neural network models for intrusion detection.

Provides a feedforward IDS network with batch normalization and dropout,
along with training and evaluation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSNet(nn.Module):
    """
    Feedforward neural network for intrusion detection.
    
    Architecture: input (77) -> 128 -> 64 -> 32 -> n_classes
    - ReLU activations
    - Dropout: 0.3 after each hidden layer
    - BatchNorm after each layer
    
    Args:
        n_features: Number of input features (default: 77 for CICIDS2017)
        n_classes: Number of output classes (default: 15 for CICIDS2017)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, n_features: int = 77, n_classes: int = 15, dropout: float = 0.3):
        super(IDSNet, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_rate = dropout
        
        # Input -> 128
        self.fc1 = nn.Linear(n_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        
        # 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        # 64 -> 32
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout)
        
        # 32 -> n_classes
        self.fc4 = nn.Linear(32, n_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
        logger.info(f"IDSNet initialized: {n_features} -> 128 -> 64 -> 32 -> {n_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        # Layer 1: input -> 128
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: 128 -> 64
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3: 64 -> 32
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output layer: 32 -> n_classes
        x = self.fc4(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels (argmax of logits).
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted probabilities (softmax of logits).
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Softmax probabilities of shape (batch_size, n_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


def train_one_epoch(
    model: IDSNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: IDSNet model instance
        dataloader: Training data loader
        optimizer: Optimizer (e.g., Adam, SGD)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run on (cpu or cuda)
        
    Returns:
        Average loss over the epoch
    """
    model.train()  # Set to training mode (enables dropout, batch norm updates)
    
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Move to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        n_batches += 1
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    logger.info(f"Epoch completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate(
    model: IDSNet,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate model on a dataloader and compute per-class metrics.
    
    Returns:
        Dictionary with keys:
        - 'accuracy': Overall accuracy
        - 'macro_precision': Macro-averaged precision
        - 'macro_recall': Macro-averaged recall
        - 'macro_f1': Macro-averaged F1-score
        - 'weighted_precision': Weighted precision
        - 'weighted_recall': Weighted recall
        - 'weighted_f1': Weighted F1-score
        - 'per_class': Dict of per-class metrics {class_idx: {'precision', 'recall', 'f1'}}
        - 'confusion_matrix': Confusion matrix
        - 'predictions': All predictions
        - 'true_labels': All true labels
        
    Args:
        model: IDSNet model instance
        dataloader: Evaluation data loader
        device: Device to run on
        label_names: Optional list of class names for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()  # Set to evaluation mode (disables dropout, uses running batch norm)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            # Move to device
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
            
            # Collect predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Compute macro-averaged metrics
    macro_precision = precision_score(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    macro_recall = recall_score(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Compute weighted-averaged metrics
    weighted_precision = precision_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    weighted_recall = recall_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    weighted_f1 = f1_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Compute per-class metrics
    per_class_metrics = {}
    unique_classes = sorted(set(all_labels) | set(all_predictions))
    
    for class_idx in unique_classes:
        binary_labels = (all_labels == class_idx).astype(int)
        binary_predictions = (all_predictions == class_idx).astype(int)
        
        class_name = label_names[class_idx] if label_names else f"Class {class_idx}"
        
        per_class_metrics[class_idx] = {
            'name': class_name,
            'precision': precision_score(binary_labels, binary_predictions, zero_division=0),
            'recall': recall_score(binary_labels, binary_predictions, zero_division=0),
            'f1': f1_score(binary_labels, binary_predictions, zero_division=0),
            'support': np.sum(all_labels == class_idx)
        }
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'n_samples': len(all_labels)
    }
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"  Weighted F1-Score: {weighted_f1:.4f}")
    logger.info(f"  Samples evaluated: {len(all_labels)}")
    
    return results


def print_evaluation_metrics(results: Dict, label_names: Optional[List[str]] = None):
    """
    Pretty-print evaluation metrics from evaluate() output.
    
    Args:
        results: Dictionary returned by evaluate()
        label_names: Optional list of class names
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:          {results['accuracy']:.4f}")
    print(f"  Macro Precision:   {results['macro_precision']:.4f}")
    print(f"  Macro Recall:      {results['macro_recall']:.4f}")
    print(f"  Macro F1-Score:    {results['macro_f1']:.4f}")
    print(f"  Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"  Weighted Recall:   {results['weighted_recall']:.4f}")
    print(f"  Weighted F1-Score: {results['weighted_f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 74)
    
    for class_idx in sorted(results['per_class'].keys()):
        metrics = results['per_class'][class_idx]
        class_name = metrics['name'][:28]
        print(f"{class_name:<30} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} "
              f"{metrics['support']:<8d}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = IDSNet(n_features=77, n_classes=15, dropout=0.3)
    model.to(device)
    
    # Create dummy data
    X_dummy = torch.randn(100, 77)
    y_dummy = torch.randint(0, 15, (100,))
    dataset = torch.utils.data.TensorDataset(X_dummy, y_dummy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Test forward pass
    print("\nForward pass test:")
    logits = model(X_dummy[:10].to(device))
    print(f"  Input shape: {X_dummy[:10].shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output logits: {logits[0].detach().cpu().numpy()}")
    
    # Test prediction
    print("\nPrediction test:")
    predictions = model.predict(X_dummy[:10].to(device))
    print(f"  Predictions: {predictions.cpu().numpy()}")
    
    # Test probabilities
    print("\nProbability test:")
    probs = model.predict_proba(X_dummy[:10].to(device))
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Max probability: {probs.max().item():.4f}")
    print(f"  Min probability: {probs.min().item():.4f}")
    
    # Test training
    print("\nTraining test:")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    print(f"  Training loss: {avg_loss:.4f}")
    
    # Test evaluation
    print("\nEvaluation test:")
    eval_results = evaluate(model, dataloader, device)
    print_evaluation_metrics(eval_results)
