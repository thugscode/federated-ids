# Federated Learning Experiments with Hydra Configuration

This directory contains scripts for running federated learning experiments with different strategies, data partitions, and configurations. All experiments use Hydra for flexible configuration management.

## Quick Start

### Run Default Experiment
```bash
python experiments/run.py
```

### Run Specific Strategy
```bash
python experiments/run_fedavg.py      # FedAvg baseline
python experiments/run_fedprox.py     # FedProx (heterogeneous networks)
python experiments/run_fednova.py     # FedNova (non-IID data)
python experiments/run_dp.py          # DPFedAvg (differential privacy)
```

## Hydra Configuration Override

Override any config parameter from the command line:

```bash
# Change partition strategy
python experiments/run.py data.partition=day

# Change number of rounds
python experiments/run.py training.num_rounds=100

# Multiple overrides
python experiments/run.py data.partition=family training.local_epochs=5 strategy.name=fednova

# Use preset config
python experiments/run.py --config-name=heterogeneous
python experiments/run.py --config-name=privacy
```

## Experiment Scripts

### `run.py` (Main Entry Point)
Universal experiment runner with Hydra integration. Loads configuration, runs federated learning simulation, saves results as JSON.

**Features:**
- Config loading via Hydra
- Data partitioning (IID, temporal, attack family)
- Client creation and dataset splitting
- Metrics aggregation and logging
- Results saved to `results/metrics/{experiment_name}_results.json`

**Usage:**
```bash
# Full control via config
python experiments/run.py strategy.name=fedprox strategy.mu=0.05

# Use config file
python experiments/run.py --config-name=heterogeneous

# Mix config file + overrides
python experiments/run.py --config-name=privacy strategy.noise_multiplier=0.5
```

### `run_fedavg.py`
Runs FedAvg (baseline federated averaging). Useful for comparison.

```bash
python experiments/run_fedavg.py data.partition=day
```

### `run_fedprox.py`
Runs FedProx with proximal regularization (mu=0.01). Good for heterogeneous networks.

```bash
python experiments/run_fedprox.py strategy.mu=0.05    # Increase regularization
python experiments/run_fedprox.py data.partition=family  # Attack specialization
```

### `run_fednova.py`
Runs FedNova for non-IID/heterogeneous data. Includes momentum acceleration.

```bash
python experiments/run_fednova.py strategy.momentum=0.9
python experiments/run_fednova.py data.partition=family
```

### `run_dp.py`
Runs DPFedAvg with server-side differential privacy.

```bash
python experiments/run_dp.py strategy.noise_multiplier=0.1   # Stronger privacy
python experiments/run_dp.py strategy.noise_multiplier=10.0  # Weaker privacy
```

## Configuration Files

Configuration files are in `configs/` directory and use YAML format.

### `configs/default.yaml`
Default configuration for all experiments. Includes:
- **Data**: partition method, number of clients, seed
- **Training**: local epochs, batch size, learning rate, num rounds
- **Strategy**: aggregation strategy (fedavg, fedprox, fednova, dpfedavg) + strategy-specific params
- **Experiment**: output directory, logging verbosity, checkpointing

### `configs/heterogeneous.yaml`
Preset for high-heterogeneity scenarios (attack-family partition with FedProx).

### `configs/privacy.yaml`
Preset for differential privacy experiments (DPFedAvg configuration).

## Configuration Structure

```yaml
data:
  partition: "iid"           # iid | day | family
  n_clients: 5
  seed: 42

training:
  local_epochs: 3
  batch_size: 512
  learning_rate: 0.001
  num_rounds: 50

strategy:
  name: "fedavg"             # fedavg | fedprox | fednova | dpfedavg
  mu: 0.01                   # FedProx: proximal coefficient
  noise_multiplier: 1.0      # DPFedAvg: noise std / clipping
  max_grad_norm: 1.0         # DPFedAvg: clipping threshold
  momentum: 0.0              # FedNova: momentum for aggregation
  adapt_lr: true             # FedNova: adaptive learning rates

experiment:
  name: "federated_ids_default"
  output_dir: "results/metrics"
  verbose: true
```

## Data Partitions

### `iid` (Independent & Identically Distributed)
- Random shuffle and equal split
- **Heterogeneity**: Low (EMD ≈ 0.0068)
- **Best for**: Baseline comparisons, testing strategies

### `day` (Temporal Partitioning)
- Clients grouped by day of week
- Each client has different attack rate distribution
- **Heterogeneity**: Medium (EMD ≈ 0.0122)
- **Best for**: Realistic network scenarios

### `family` (Attack Family Specialization)
- Client 0: DDoS, Client 1: Brute Force, Client 2: Web Attacks, etc.
- **Heterogeneity**: High (EMD ≈ 0.0785)
- **Best for**: Testing heterogeneous/non-IID strategies

## Aggregation Strategies

### FedAvg (Baseline)
Standard federated averaging. Simple weighted aggregation.

```bash
python experiments/run.py strategy.name=fedavg
```

### FedProx (Heterogeneous Networks)
Adds proximal regularization term: `loss + (mu/2) * ||w - w_global||²`

Configuration:
- `mu`: Proximal coefficient (default 0.01)

```bash
python experiments/run.py strategy.name=fedprox strategy.mu=0.01
```

### FedNova (Non-IID Optimization)
Normalized gradient updates accounting for local variance. Improves convergence on non-IID data.

Configuration:
- `adapt_lr`: Adaptive learning rates (default true)
- `momentum`: Momentum for aggregation (default 0.0)

```bash
python experiments/run.py strategy.name=fednova strategy.momentum=0.9
```

### DPFedAvg (Differential Privacy)
FedAvg with server-side Gaussian DP (clipping + noise injection).

Configuration:
- `max_grad_norm`: Clipping threshold (default 1.0)
- `noise_multiplier`: Noise std relative to clipping (default 1.0)
- `target_delta`: Delta for RDP accounting (default 1e-5)

Privacy tradeoff:
- Higher `noise_multiplier` → Stronger privacy, lower accuracy
- Lower `noise_multiplier` → Weaker privacy, higher accuracy

```bash
python experiments/run.py strategy.name=dpfedavg strategy.noise_multiplier=0.1
```

## Output

Results are saved to `results/metrics/{experiment_name}_results.json` with:
- Experiment configuration
- Round-by-round metrics (loss, accuracy, macro F1, weighted F1)
- Final performance summary
- Timing information
- Strategy-specific metrics (epsilon for DPFedAvg, etc.)

## Example Workflows

### Compare FedAvg vs FedProx on Heterogeneous Data
```bash
# FedAvg baseline
python experiments/run.py strategy.name=fedavg data.partition=family

# FedProx with regularization
python experiments/run_fedprox.py data.partition=family strategy.mu=0.01

# Compare results in results/metrics/
```

### Test Privacy-Utility Tradeoff
```bash
# Strong privacy (noise_multiplier=0.1)
python experiments/run_dp.py strategy.noise_multiplier=0.1

# Moderate privacy (noise_multiplier=1.0)
python experiments/run_dp.py strategy.noise_multiplier=1.0

# Weak privacy (noise_multiplier=10.0)
python experiments/run_dp.py strategy.noise_multiplier=10.0
```

### Tune FedNova for Non-IID Networks
```bash
# Without momentum
python experiments/run_fednova.py strategy.momentum=0.0

# With momentum acceleration
python experiments/run_fednova.py strategy.momentum=0.9
```

## Hydra Features

### Override Groups
```bash
# Use a different config as base
python experiments/run.py --config-name=heterogeneous
```

### Help
```bash
# List all overridable parameters
python experiments/run.py --info=defaults

# Get detailed help
python experiments/run.py --help
```

### Job Directories
Hydra creates `outputs/` directory with timestamped job logs and config snapshots:
```
outputs/
  2024-05-07/
    12-34-56/
      .hydra/
        config.yaml        # Resolved configuration
        job.log            # Experiment log
```

## Results Analysis

All results are JSON files in `results/metrics/`:

```json
{
  "experiment_name": "fedavg_baseline",
  "strategy": "fedavg",
  "partition": "iid",
  "n_clients": 5,
  "num_rounds": 50,
  "duration_seconds": 123.45,
  "final_loss": 1.234,
  "final_accuracy": 0.678,
  "final_macro_f1": 0.645,
  "final_weighted_f1": 0.670,
  "round_history": [
    {
      "round": 1,
      "loss": 2.345,
      "accuracy": 0.456,
      "macro_f1": 0.401,
      "weighted_f1": 0.420
    },
    ...
  ]
}
```

Use these results to:
- Plot convergence curves
- Compare strategy performance
- Analyze privacy-utility tradeoffs
- Validate heterogeneity handling

## Notes

- All experiments use PyTorch with CPU by default (set `training.device=cuda` for GPU)
- Results are deterministic when using the same seed
- Config overrides follow Hydra dot-notation (`.` for nested access)
- For help with Hydra: https://hydra.cc/docs
