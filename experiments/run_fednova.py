"""
FedNova Experiment - Federated Learning for Non-IID Data.

Runs federated learning with FedNova aggregation strategy.
FedNova tackling the objective inconsistency problem in heterogeneous networks
with normalized gradient updates and momentum acceleration.

Paper: "Tackling the Objective Inconsistency Problem in Heterogeneous
Federated Learning" (Wang et al., 2021)

Usage:
    python experiments/run_fednova.py                        # Default
    python experiments/run_fednova.py data.partition=family  # Highly heterogeneous
    python experiments/run_fednova.py strategy.momentum=0.9  # With momentum
"""

import subprocess
import sys

if __name__ == "__main__":
    # Build command with FedNova specific config
    cmd = [
        sys.executable,
        "experiments/run.py",
        "strategy.name=fednova",
        "strategy.adapt_lr=true",
        "strategy.momentum=0.0",
        "experiment.name=fednova_noniid",
        "experiment.description=FedNova - Federated learning for heterogeneous/non-IID networks",
    ]
    
    # Add any additional arguments passed from command line
    cmd.extend(sys.argv[1:])
    
    # Run the main experiment script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
