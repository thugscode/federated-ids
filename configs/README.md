# Hydra Configuration Files

This directory contains YAML configuration files for federated learning experiments using Hydra.

## Files

### `default.yaml`
Base configuration for all experiments. Defines:
- Data partitioning (IID, temporal, attack family)
- Training hyperparameters (epochs, batch size, learning rate)
- Strategy selection and strategy-specific parameters
- Experiment metadata and output paths

**Use when**: You want full control over experiment parameters or starting from scratch.

```bash
python experiments/run.py                    # Uses default.yaml
python experiments/run.py training.num_rounds=100  # Override specific params
```

### `heterogeneous.yaml`
Preset configuration for high-heterogeneity scenarios.

**Configuration**:
- **Partition**: "family" (attack specialization) - highest heterogeneity (EMD ≈ 0.0785)
- **Strategy**: FedProx with mu=0.01
- **Training**: 5 local epochs, 100 rounds

**Best for**: Testing strategies on non-IID heterogeneous networks

**Use when**: Evaluating how strategies handle data heterogeneity

```bash
python experiments/run.py --config-name=heterogeneous
python experiments/run.py --config-name=heterogeneous strategy.mu=0.05  # Stronger reg
```

### `privacy.yaml`
Preset configuration for differential privacy experiments.

**Configuration**:
- **Partition**: "iid" (baseline for privacy testing)
- **Strategy**: DPFedAvg with noise_multiplier=0.1
- **Training**: 3 local epochs, 100 rounds

**Best for**: Privacy-utility tradeoff evaluation

**Use when**: Testing differential privacy performance

```bash
python experiments/run.py --config-name=privacy
python experiments/run.py --config-name=privacy strategy.noise_multiplier=1.0
```

## Configuration Structure

```yaml
data:
  partition: string          # Data partitioning method
  n_clients: int             # Number of clients
  seed: int                  # Random seed
  raw_data_dir: string       # Path to raw CICIDS2017 CSVs
  processed_data_dir: string # Path to processed data

training:
  local_epochs: int          # Epochs per client per round
  batch_size: int            # Batch size for training
  learning_rate: float       # Adam optimizer learning rate
  num_rounds: int            # Total federated rounds
  device: string             # cpu | cuda | auto

strategy:
  name: string               # Strategy: fedavg | fedprox | fednova | dpfedavg
  
  # Strategy-specific parameters
  mu: float                  # FedProx: proximal coefficient
  max_grad_norm: float       # DPFedAvg: clipping threshold
  noise_multiplier: float    # DPFedAvg: noise std / threshold
  target_delta: float        # DPFedAvg: delta for RDP
  momentum: float            # FedNova: momentum for aggregation
  adapt_lr: bool             # FedNova: adaptive learning rates

experiment:
  name: string               # Experiment name
  description: string        # Experiment description
  output_dir: string         # Output directory
  verbose: bool              # Verbose logging
  save_checkpoints: bool     # Save model checkpoints
  seed: int                  # Reproducibility seed
```

## Usage Patterns

### Pattern 1: Start from Default, Override Specific Params
```bash
python experiments/run.py training.num_rounds=200 strategy.name=fedprox
```

### Pattern 2: Use Preset Config, Override One Parameter
```bash
python experiments/run.py --config-name=heterogeneous strategy.mu=0.05
```

### Pattern 3: Use Preset Config As-Is
```bash
python experiments/run.py --config-name=privacy
```

### Pattern 4: Use Strategy-Specific Script with Config
```bash
python experiments/run_fednova.py --config-name=heterogeneous
```

## Data Partitioning Methods

| Method | Heterogeneity | EMD | Use Case |
|--------|---------------|-----|----------|
| `iid` | Low | 0.0068 | Baseline, privacy testing |
| `day` | Medium | 0.0122 | Temporal/realistic scenarios |
| `family` | High | 0.0785 | Non-IID stress testing |

## Strategy Guide

| Strategy | Heterogeneity | Privacy | Use Case | Config |
|----------|---------------|---------|----------|--------|
| **FedAvg** | Standard baseline | None | Basic federated learning | `name: fedavg` |
| **FedProx** | Good for non-IID | None | Heterogeneous networks | `name: fedprox, mu: 0.01` |
| **FedNova** | Excellent for non-IID | None | Non-IID data handling | `name: fednova, momentum: 0.9` |
| **DPFedAvg** | Standard | Gaussian DP | Privacy-preserving FL | `name: dpfedavg, noise_multiplier: 1.0` |

## Advanced: Creating New Config Files

### Example: Custom Low-Epochs, High-Rounds Config
Create `configs/quick_convergence.yaml`:

```yaml
defaults:
  - default

data:
  partition: "iid"
  n_clients: 5

training:
  local_epochs: 1      # Quick local updates
  num_rounds: 200      # More global rounds
  learning_rate: 0.01

experiment:
  name: "quick_convergence"
```

Usage:
```bash
python experiments/run.py --config-name=quick_convergence
```

### Example: Differential Privacy with Different Noise Levels
Create `configs/privacy_weak.yaml`:

```yaml
defaults:
  - privacy

strategy:
  noise_multiplier: 10.0  # Weaker privacy, higher accuracy
```

Usage:
```bash
python experiments/run.py --config-name=privacy_weak
```

## Hydra Features

### Config Composition
Use `defaults` to build configs from other configs:

```yaml
defaults:
  - default

strategy:
  mu: 0.05  # Override specific value
```

### Override Priority
Command-line overrides > Config file values > Defaults

```bash
# This takes precedence:
python experiments/run.py --config-name=heterogeneous strategy.mu=0.1
```

### List All Parameters
```bash
python experiments/run.py --cfg job
```

### Job Directory Structure
Hydra saves outputs with config snapshot:
```
outputs/
  2024-05-07/
    12-34-56/
      .hydra/
        config.yaml        # Resolved config
        job_name           # Job info
```

## Tips

1. **For reproducibility**: Always specify seed in config or override
2. **For comparison**: Save configs before running to track experiments
3. **For debugging**: Use `verbose: true` in config
4. **For parameter search**: Create preset configs for different scenarios
5. **For documentation**: Use descriptive experiment names

## See Also

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Structured Configs](https://hydra.cc/docs/structured_config/intro/)
- [Advanced Patterns](https://hydra.cc/docs/patterns/configuring_experiments/)
