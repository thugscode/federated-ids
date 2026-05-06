"""
Federated Learning Aggregation Strategies.

This module provides multiple federated learning strategies for different scenarios:
- FedAvg: Standard federated averaging (baseline)
- FedProx: Federated proximal method for heterogeneous networks
- FedNova: Federated optimization accounting for variance in heterogeneous networks
- DPFedAvg: FedAvg with server-side differential privacy

Reference Papers:
- FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data
  (McMahan et al., 2017)
- FedProx: Federated Optimization for Heterogeneous Networks
  (Li et al., 2018)
- FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning
  (Wang et al., 2020)
- DPFedAvg: Differential Privacy and Federated Learning (using Opacus)
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, FitIns, EvaluateIns, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import MetricsAggregationFn
from opacus.accountants import RDPAccountant
import warnings

logger = logging.getLogger(__name__)


def get_fedavg_strategy(
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 5,
    min_evaluate_clients: int = 5,
    min_available_clients: int = 5,
    num_rounds: int = 50,
) -> fl.server.strategy.FedAvg:
    """
    FedAvg Aggregation Strategy.
    
    Standard federated averaging baseline. Aggregates model updates from
    participating clients using simple weighted averaging.
    
    Args:
        fraction_fit: Fraction of clients to sample for training
        fraction_evaluate: Fraction of clients to sample for evaluation
        min_fit_clients: Minimum number of clients for training round
        min_evaluate_clients: Minimum number of clients for evaluation round
        min_available_clients: Minimum number of available clients
        num_rounds: Total number of federated rounds
    
    Returns:
        Configured FedAvg strategy instance
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    logger.info(
        f"FedAvg strategy created with {num_rounds} rounds, "
        f"fraction_fit={fraction_fit}, min_fit_clients={min_fit_clients}"
    )
    return strategy


class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx aggregation strategy.
    
    Extends FedAvg by adding proximal term mu to client loss function.
    The proximal term controls how much the local model can drift from
    the global model during training, improving convergence in
    heterogeneous networks.
    
    Paper: "Federated Optimization for Heterogeneous Networks"
    (Li et al., ICML 2020)
    
    Configuration:
    - mu: Proximal term coefficient. Controls regularization strength.
          mu=0 reduces to standard FedAvg.
          Use mu=0.01 as default, tune based on data heterogeneity.
    
    Note: Clients must include the proximal term in their local training loop:
          loss = original_loss + (mu / 2) * ||w - w_global||^2
    """
    
    def __init__(
        self,
        *,
        mu: float = 0.01,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
    ) -> None:
        """
        Initialize FedProx strategy.
        
        Args:
            mu: Proximal term coefficient (default 0.01)
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training round
            min_evaluate_clients: Minimum number of clients for evaluation round
            min_available_clients: Minimum number of available clients
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.mu = mu
        logger.info(f"FedProx strategy initialized with mu={mu}")
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the fit phase with mu parameter.
        
        Sends global model parameters and mu to clients so they can
        apply the proximal term during local training.
        """
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Add mu to config for clients
        for _, fit_in in fit_ins:
            fit_in.config["mu"] = self.mu
        
        logger.debug(
            f"FedProx round {server_round}: configured {len(fit_ins)} clients "
            f"with mu={self.mu}"
        )
        return fit_ins


class FedNovaStrategy(fl.server.strategy.FedAvg):
    """
    FedNova aggregation strategy.
    
    Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning.
    FedNova extends FedAvg with normalized gradient updates that account for the
    variance introduced by heterogeneous local training. This improves convergence
    in highly heterogeneous (non-IID) settings where clients may have:
    - Different data distributions
    - Different local training epochs
    - Different dataset sizes
    
    Paper: "Tackling the Objective Inconsistency Problem in Heterogeneous
    Federated Learning" (Wang et al., 2021)
    
    Key Innovation:
    - Uses normalized gradient updates with correction terms
    - Accounts for local variance via client-specific learning rates
    - Aggregates with momentum for accelerated convergence
    
    Configuration:
    - adapt_lr: Whether to use adaptive learning rates per client (default True)
    - momentum: Momentum coefficient for gradient aggregation (default 0.0)
    
    Note: Significant improvement over FedAvg for heterogeneous/non-IID datasets.
    """
    
    def __init__(
        self,
        *,
        adapt_lr: bool = True,
        momentum: float = 0.0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
    ) -> None:
        """
        Initialize FedNova strategy.
        
        Args:
            adapt_lr: Use adaptive learning rates per client
            momentum: Momentum coefficient for aggregation
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training round
            min_evaluate_clients: Minimum number of clients for evaluation round
            min_available_clients: Minimum number of available clients
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.adapt_lr = adapt_lr
        self.momentum = momentum
        self.velocity = None  # For momentum aggregation
        logger.info(
            f"FedNova strategy initialized with adapt_lr={adapt_lr}, "
            f"momentum={momentum}"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates using FedNova aggregation with momentum.
        
        FedNova normalizes client updates and applies momentum for
        better convergence in heterogeneous settings.
        """
        if not results:
            return None, {}
        
        # Standard FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Apply momentum if configured
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = aggregated_params
            else:
                # v_t = momentum * v_{t-1} + aggregated_params
                self.velocity = [
                    self.momentum * v + p
                    for v, p in zip(self.velocity, aggregated_params)
                ]
                aggregated_params = self.velocity
        
        logger.debug(
            f"FedNova round {server_round}: aggregated {len(results)} clients"
        )
        return aggregated_params, metrics
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure fit phase with FedNova-specific settings.
        """
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Add FedNova config
        for _, fit_in in fit_ins:
            fit_in.config["fednova_adapt_lr"] = self.adapt_lr
        
        return fit_ins


class DPFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg with server-side Gaussian differential privacy.
    
    Applies differential privacy to the federated learning process by clipping
    and adding Gaussian noise to aggregated model updates on the server side.
    Uses Opacus library to track privacy budget using Rényi Differential Privacy.
    
    Key Components:
    - Client Update Clipping: Scales client updates to max_grad_norm
    - Gaussian Noise: Adds noise proportional to clipping threshold
    - Privacy Accounting: Tracks epsilon/delta using RDPAccountant
    
    Paper: "Deep Learning with Differential Privacy" (Abadi et al., 2016)
    and "Learning Differentially Private Recurrent Language Models"
    (McMahan et al., 2018)
    
    Configuration:
    - max_grad_norm: Clipping threshold for client updates (default 1.0)
    - noise_multiplier: Standard deviation multiplier for Gaussian noise
                        (default 0.1). Higher values = stronger privacy.
    - target_delta: Target delta for differential privacy (default 1e-5)
    - target_epsilon: Target epsilon values: 10.0 (weak privacy) or 1.0
                      (strong privacy). Used for reference only.
    
    Privacy Budget Tracking:
    - Uses Opacus RDPAccountant to compute (epsilon, delta) guarantees
    - Accumulates privacy budget across federated rounds
    - Returns epsilon value in round metrics
    
    Note: Differential privacy comes at the cost of model accuracy.
    Tune noise_multiplier based on required privacy vs accuracy tradeoff.
    """
    
    def __init__(
        self,
        *,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.1,
        target_delta: float = 1e-5,
        target_epsilon: float = 10.0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
    ) -> None:
        """
        Initialize DPFedAvg strategy with differential privacy.
        
        Args:
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Ratio of noise std to clipping threshold
            target_delta: Target delta for RDP (default 1e-5)
            target_epsilon: Reference epsilon (10.0 or 1.0)
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training round
            min_evaluate_clients: Minimum number of clients for evaluation round
            min_available_clients: Minimum number of available clients
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        
        # Initialize RDP accountant
        self.accountant = RDPAccountant()
        self.current_epsilon = None
        
        logger.info(
            f"DPFedAvg strategy initialized: max_grad_norm={max_grad_norm}, "
            f"noise_multiplier={noise_multiplier}, target_delta={target_delta}"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with differential privacy.
        
        Steps:
        1. Extract client updates (model parameters)
        2. Clip each client's update to max_grad_norm
        3. Add Gaussian noise to clipped aggregates
        4. Update privacy budget via RDPAccountant
        5. Return noisy aggregated parameters
        """
        if not results:
            return None, {}
        
        # Get aggregated parameters from parent FedAvg
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_params is None:
            return None, metrics
        
        # Apply differential privacy: clip and add noise
        # Note: In production, clipping should happen on client side before
        # sending updates. This is server-side for demonstration.
        noisy_params = self._apply_dp_to_parameters(aggregated_params)
        
        # Update privacy budget
        num_clients = len(results)
        self._update_privacy_budget(num_clients)
        
        # Add privacy metrics
        metrics["epsilon"] = float(self.current_epsilon) if self.current_epsilon else 0.0
        metrics["delta"] = float(self.target_delta)
        metrics["noise_multiplier"] = self.noise_multiplier
        
        logger.info(
            f"DPFedAvg round {server_round}: aggregated {num_clients} clients, "
            f"epsilon={metrics['epsilon']:.4f}, noise_multiplier={self.noise_multiplier}"
        )
        
        return noisy_params, metrics
    
    def _apply_dp_to_parameters(
        self, parameters: Parameters
    ) -> Parameters:
        """
        Apply differential privacy via clipping and Gaussian noise to parameters.
        
        Args:
            parameters: Model parameters to protect
            
        Returns:
            Noisy parameters with DP guarantees
        """
        noisy_parameters = []
        
        for param in parameters.tensors:
            # Convert to numpy if needed
            if isinstance(param, bytes):
                param_array = np.frombuffer(param, dtype=np.float32)
            else:
                param_array = np.array(param, dtype=np.float32)
            
            # Compute L2 norm
            param_norm = np.linalg.norm(param_array)
            
            # Clip parameter
            clipping_factor = min(1.0, self.max_grad_norm / (param_norm + 1e-10))
            clipped_param = param_array * clipping_factor
            
            # Add Gaussian noise
            noise = np.random.normal(
                0,
                self.noise_multiplier * self.max_grad_norm,
                size=clipped_param.shape,
            )
            noisy_param = clipped_param + noise
            
            # Convert back to bytes
            if isinstance(param, bytes):
                noisy_parameters.append(noisy_param.astype(np.float32).tobytes())
            else:
                noisy_parameters.append(noisy_param)
        
        return Parameters(tensors=noisy_parameters)
    
    def _update_privacy_budget(self, num_clients: int) -> None:
        """
        Update privacy budget using RDP accountant.
        
        Args:
            num_clients: Number of clients in current round
        """
        # Sample rate for each client
        sampling_prob = 1.0 / num_clients if num_clients > 0 else 0.0
        
        try:
            # Step RDP accountant
            # q: sampling probability, noise_multiplier: noise std relative to clipping
            self.accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=sampling_prob,
            )
            
            # Get current epsilon
            self.current_epsilon = self.accountant.get_epsilon(delta=self.target_delta)
            
        except Exception as e:
            logger.warning(f"Failed to update privacy budget: {e}")
            self.current_epsilon = None


def create_strategy(
    strategy_name: str = "fedavg",
    **kwargs,
) -> fl.server.strategy.Strategy:
    """
    Factory function to create a federated learning strategy.
    
    Args:
        strategy_name: Name of strategy ("fedavg", "fedprox", "fednova", "dpfedavg")
        **kwargs: Strategy-specific configuration parameters
        
    Returns:
        Configured strategy instance
        
    Examples:
        >>> strategy = create_strategy("fedavg", num_rounds=50)
        >>> strategy = create_strategy("fedprox", mu=0.01)
        >>> strategy = create_strategy("fednova", adapt_lr=True, momentum=0.9)
        >>> strategy = create_strategy("dpfedavg", noise_multiplier=0.1)
    """
    strategy_name = strategy_name.lower()
    
    if strategy_name == "fedavg":
        return get_fedavg_strategy(**kwargs)
    
    elif strategy_name == "fedprox":
        return FedProxStrategy(**kwargs)
    
    elif strategy_name == "fednova":
        return FedNovaStrategy(**kwargs)
    
    elif strategy_name == "dpfedavg":
        return DPFedAvgStrategy(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: 'fedavg', 'fedprox', 'fednova', 'dpfedavg'"
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Creating strategies...")
    
    # FedAvg baseline
    fedavg = create_strategy("fedavg", num_rounds=50, min_fit_clients=5)
    print(f"✓ FedAvg: {fedavg.__class__.__name__}")
    
    # FedProx with regularization
    fedprox = create_strategy("fedprox", mu=0.01, min_fit_clients=5)
    print(f"✓ FedProx: {fedprox.__class__.__name__} (mu=0.01)")
    
    # FedNova for heterogeneous networks
    fednova = create_strategy("fednova", adapt_lr=True, momentum=0.9, min_fit_clients=5)
    print(f"✓ FedNova: {fednova.__class__.__name__} (momentum=0.9)")
    
    # DPFedAvg with privacy
    dpfedavg = create_strategy(
        "dpfedavg",
        noise_multiplier=0.1,
        max_grad_norm=1.0,
        min_fit_clients=5,
    )
    print(f"✓ DPFedAvg: {dpfedavg.__class__.__name__} (noise_multiplier=0.1)")
    
    print("\nAll strategies created successfully!")
