"""
FedAvg Baseline Experiment - Standard Federated Averaging.

Runs federated learning with FedAvg aggregation (baseline).
Useful for comparison with other strategies.

Usage:
    python experiments/run_fedavg.py                    # Default config
    python experiments/run_fedavg.py data.partition=day # Override partition
    python experiments/run_fedavg.py training.num_rounds=100
"""

import subprocess
import sys

if __name__ == "__main__":
    # Build command with FedAvg specific config
    cmd = [
        sys.executable,
        "experiments/run.py",
        "strategy.name=fedavg",
        "experiment.name=fedavg_baseline",
        "experiment.description=FedAvg baseline - standard federated averaging",
    ]
    
    # Add any additional arguments passed from command line
    cmd.extend(sys.argv[1:])
    
    # Run the main experiment script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
