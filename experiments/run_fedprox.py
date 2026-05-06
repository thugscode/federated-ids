"""
FedProx Experiment - Federated Proximal Method for Heterogeneous Networks.

Runs federated learning with FedProx (proximal regularization term).
Improves convergence on non-IID heterogeneous networks.

Paper: "Federated Optimization for Heterogeneous Networks" (Li et al., ICML 2020)

Usage:
    python experiments/run_fedprox.py                    # Default (mu=0.01)
    python experiments/run_fedprox.py strategy.mu=0.05   # Higher regularization
    python experiments/run_fedprox.py data.partition=family  # Heterogeneous data
"""

import subprocess
import sys

if __name__ == "__main__":
    # Build command with FedProx specific config
    cmd = [
        sys.executable,
        "experiments/run.py",
        "strategy.name=fedprox",
        "strategy.mu=0.01",
        "experiment.name=fedprox_heterogeneous",
        "experiment.description=FedProx - Federated Proximal for heterogeneous networks (mu=0.01)",
    ]
    
    # Add any additional arguments passed from command line
    cmd.extend(sys.argv[1:])
    
    # Run the main experiment script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
