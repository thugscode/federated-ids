"""
DPFedAvg Experiment - Federated Learning with Differential Privacy.

Runs federated learning with server-side differential privacy via:
- Gradient clipping (max_grad_norm)
- Gaussian noise injection (noise_multiplier)
- Privacy budget tracking (Opacus RDPAccountant)

Paper: "Deep Learning with Differential Privacy" (Abadi et al., 2016)

Usage:
    python experiments/run_dp.py                                  # Default (noise_multiplier=1.0)
    python experiments/run_dp.py strategy.noise_multiplier=0.1   # Stronger privacy
    python experiments/run_dp.py strategy.noise_multiplier=10.0  # Weaker privacy (higher accuracy)
"""

import subprocess
import sys

if __name__ == "__main__":
    # Build command with DPFedAvg specific config
    cmd = [
        sys.executable,
        "experiments/run.py",
        "strategy.name=dpfedavg",
        "strategy.noise_multiplier=1.0",
        "strategy.max_grad_norm=1.0",
        "experiment.name=dpfedavg_privacy",
        "experiment.description=DPFedAvg with differential privacy (noise_multiplier=1.0)",
    ]
    
    # Add any additional arguments passed from command line
    cmd.extend(sys.argv[1:])
    
    # Run the main experiment script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
