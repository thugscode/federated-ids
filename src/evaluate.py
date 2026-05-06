"""
Evaluation module for federated learning models.

Provides comprehensive evaluation metrics and analysis.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Binary classification metrics
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate and false negative rate
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "tp": int(TP),
            "tn": int(TN),
            "fp": int(FP),
            "fn": int(FN),
            "total_samples": len(y_true)
        }
        
        # Add AUC-ROC if probabilities available
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, y_proba)
                metrics["auc_roc"] = float(auc)
            except Exception as e:
                logger.warning(f"Could not compute AUC-ROC: {e}")
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as dictionary
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        return {
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN)
        }
    
    @staticmethod
    def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Per-class metrics
        """
        classes = np.unique(y_true)
        metrics = {}
        
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
            FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[int(cls)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(y_true_binary))
            }
        
        return metrics
    
    @staticmethod
    def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Comprehensive evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Compute metrics
        metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)
        metrics["confusion_matrix"] = ModelEvaluator.compute_confusion_matrix(y_test, y_pred)
        metrics["per_class"] = ModelEvaluator.compute_per_class_metrics(y_test, y_pred)
        
        return metrics
    
    @staticmethod
    def save_metrics(metrics: Dict, output_dir: str, filename: str = "metrics.json"):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            output_dir: Output directory
            filename: Output filename
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / filename, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path / filename}")
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Print metrics in readable format."""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        
        logger.info(f"Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"Precision:  {metrics['precision']:.4f}")
        logger.info(f"Recall:     {metrics['recall']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"F1-Score:   {metrics['f1']:.4f}")
        logger.info(f"FPR:        {metrics['fpr']:.4f}")
        logger.info(f"FNR:        {metrics['fnr']:.4f}")
        
        if "auc_roc" in metrics:
            logger.info(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
        
        cm = metrics.get("confusion_matrix", {})
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TP: {cm.get('TP', 0)}, TN: {cm.get('TN', 0)}")
        logger.info(f"  FP: {cm.get('FP', 0)}, FN: {cm.get('FN', 0)}")
        
        logger.info("="*60 + "\n")


class FederatedEvaluator:
    """Evaluate federated learning results."""
    
    @staticmethod
    def evaluate_round(server, clients: list, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate global model and client models for a round.
        
        Args:
            server: Federated server
            clients: List of clients
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        results = {}
        
        # Evaluate global model
        evaluator = ModelEvaluator()
        global_metrics = evaluator.evaluate_model(server.get_model(), X_test, y_test)
        results["global"] = global_metrics
        
        # Evaluate client models
        results["clients"] = {}
        for client in clients:
            client_metrics = client.evaluate()
            results["clients"][client.client_id] = client_metrics
        
        return results
    
    @staticmethod
    def compare_strategies(results_dict: Dict) -> Dict:
        """
        Compare results across different strategies.
        
        Args:
            results_dict: Dictionary mapping strategy names to results
            
        Returns:
            Comparison summary
        """
        comparison = {}
        
        for strategy_name, results in results_dict.items():
            if "final_metrics" in results:
                metrics = results["final_metrics"]
                comparison[strategy_name] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1": metrics.get("f1", 0),
                    "auc_roc": metrics.get("auc_roc", 0)
                }
        
        return comparison


if __name__ == "__main__":
    # Test evaluator
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8, 0.7, 0.2, 0.95, 0.6])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    evaluator.print_metrics(metrics)
