# Federated Learning for Intrusion Detection System (IDS)

A comprehensive implementation of federated learning approaches for detecting network intrusions using the CICIDS2017 dataset.

## Project Overview

This project implements and compares multiple federated learning strategies for intrusion detection:
- **Baseline**: Centralized learning on all data
- **FedAvg**: Federated Averaging aggregation
- **FedProx**: Federated Proximal algorithm for heterogeneous data
- **FedDP**: Federated learning with Differential Privacy

## Project Structure

```
federated-ids/
├── data/
│   ├── raw/              # CICIDS2017 CSV files (8 daily files)
│   └── processed/        # Preprocessed parquet files
├── src/
│   ├── preprocess.py     # Data preprocessing pipeline
│   ├── dataset.py        # Dataset and federated dataset classes
│   ├── model.py          # Neural network models
│   ├── strategies.py     # Aggregation strategies (FedAvg, FedProx, FedDP)
│   ├── client.py         # Federated client implementation
│   ├── server.py         # Federated server implementation
│   ├── evaluate.py       # Evaluation metrics and utilities
│   └── visualize.py      # Visualization functions
├── experiments/
│   ├── run_baseline.py   # Baseline centralized learning
│   ├── run_fedavg.py     # FedAvg experiment
│   ├── run_fedprox.py    # FedProx experiment
│   └── run_dp.py         # FedDP experiment
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_baseline.ipynb          # Baseline model training
│   ├── 03_federation.ipynb        # Federated learning demo
│   └── 04_analysis.ipynb          # Results analysis and comparison
├── results/
│   ├── metrics/          # JSON files with experiment results
│   └── plots/            # Generated figures and plots
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

### Prerequisites
- Python 3.9+ (tested on 3.13)
- pip package manager

### 1. Clone and Navigate to Project

```bash
cd /path/to/federated-ids
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Prepare Data

Place the 8 daily CICIDS2017 CSV files in `data/raw/`:
```
data/raw/
├── monday.csv
├── monday_plus.csv
├── tuesday.csv
├── tuesday_plus.csv
├── wednesday.csv
├── wednesday_plus.csv
├── thursday.csv
├── thursday_plus.csv
├── friday.csv
└── friday_plus.csv
```

### 5. Preprocess Data

```bash
python src/preprocess.py
```

This:
- Loads and merges all 8 CSV files
- Removes infinite and duplicate values
- Encodes attack types as labels
- Scales features
- Saves to `data/processed/`

## Quick Start: Run Experiments

### Option 1: Use Hydra Configuration (Recommended)

All experiments use Hydra for flexible configuration management.

#### Basic Usage

```bash
# Default configuration (FedAvg, IID, 5 clients, 50 rounds)
python experiments/run.py

# Override single parameter
python experiments/run.py data.partition=day

# Multiple overrides
python experiments/run.py strategy.name=fedprox strategy.mu=0.05 training.num_rounds=100
```

#### Use Preset Configs

```bash
# High heterogeneity scenario (attack-family partition, FedProx)
python experiments/run.py --config-name=heterogeneous

# Differential privacy scenario
python experiments/run.py --config-name=privacy
```

#### Strategy-Specific Scripts

```bash
# FedAvg baseline
python experiments/run_fedavg.py

# FedProx (heterogeneous networks)
python experiments/run_fedprox.py data.partition=family

# FedNova (non-IID data)
python experiments/run_fednova.py strategy.momentum=0.9

# DPFedAvg (differential privacy)
python experiments/run_dp.py strategy.noise_multiplier=0.1
```

### Option 2: Configure via YAML

Edit `configs/default.yaml` to customize:
- Data partitioning (iid, day, family)
- Number of clients (n_clients)
- Training parameters (local_epochs, batch_size, num_rounds)
- Strategy (fedavg, fedprox, fednova, dpfedavg)
- Strategy-specific parameters

```bash
# Run with custom config
python experiments/run.py --config-name=default
```

### Option 3: Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Available notebooks in notebooks/:
# - 01_eda.ipynb: Exploratory data analysis
# - 02_baseline.ipynb: Baseline training
# - 03_federation.ipynb: Federated learning demo
# - 04_analysis.ipynb: Results analysis
# - 05_dataset_testing.ipynb: Dataset partitioning validation
# - 06_model_training.ipynb: Model training validation
```

## Example Workflows

### Compare Strategies on Non-IID Data

```bash
# FedAvg baseline
python experiments/run_fedavg.py data.partition=family

# FedProx with regularization
python experiments/run_fedprox.py data.partition=family strategy.mu=0.01

# FedNova with momentum
python experiments/run_fednova.py data.partition=family strategy.momentum=0.9

# Results saved to results/metrics/{experiment_name}_results.json
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

### Custom Experiment

```bash
# Create custom config in configs/my_config.yaml
# Then run:
python experiments/run.py --config-name=my_config
```

## Results

Experiment results are saved to `results/metrics/`:
- JSON files with full configuration
- Round-by-round metrics (loss, accuracy, F1 scores)
- Final performance summary
- Privacy metrics for DPFedAvg (epsilon, delta)

## Architecture & Modules

### Core Modules (src/)

#### `preprocess.py`
Data preprocessing pipeline with 7 tasks:
- Load and merge 8 daily CICIDS2017 CSV files
- Remove infinite values (Flow_Bytes/s, Flow_Packets/s)
- Remove duplicates
- Parse and unify timestamps
- Encode attack types as integer labels (0-14)
- Standard scale numeric features
- Save to parquet format for fast loading

**Usage**: `python src/preprocess.py`

#### `dataset.py`
Dataset partitioning for federated learning scenarios:
- `partition_iid()`: Random shuffle + equal split (heterogeneity ≈ 0.0068)
- `partition_by_day()`: Temporal grouping with day-specific attack rates (heterogeneity ≈ 0.0122)
- `partition_by_attack_family()`: Specialize clients by attack type (heterogeneity ≈ 0.0785)
- `compute_emd()`: Measure data heterogeneity via Wasserstein distance
- `analyze_partitions()`: Statistics on partition quality

**Heterogeneity Measurement**: Pairwise Earth Mover Distance (EMD) between client label distributions

#### `model.py`
PyTorch neural network for IDS classification:
- `IDSNet`: 77 → 128 → 64 → 32 → 15 (with BatchNorm, ReLU, Dropout)
- `train_one_epoch()`: Local training with Adam optimizer
- `evaluate()`: Comprehensive metrics (accuracy, macro F1, weighted F1, per-class F1)
- `print_evaluation_metrics()`: Pretty-print results

**Architecture**: 77 input features → 15 attack classes

#### `client.py`
Flower-based federated learning client:
- `IDSClient`: Extends `fl.client.NumPyClient`
  - `get_parameters()`: Extract model weights
  - `set_parameters()`: Update from global model
  - `fit()`: Local training
  - `evaluate()`: Validation on local data
- `create_client()`: Factory function for client instantiation

**Integration**: Seamlessly integrates with Flower framework for federated orchestration

#### `server.py`
Flower-based federated learning server:
- `weighted_average()`: Aggregates client metrics by dataset size
- `create_fedavg_strategy()`: Creates FedAvg strategy
- `MetricsLogger`: Tracks metrics across rounds and saves to JSON
- `run_server()`: Main server orchestration

**Features**: Metrics aggregation, round logging, JSON persistence

#### `strategies.py`
Multiple federated learning aggregation strategies:

1. **FedAvg** (Baseline)
   - Standard weighted averaging
   - Paper: McMahan et al., 2017
   
2. **FedProx** (Heterogeneous Networks)
   - Adds proximal regularization: `loss + (mu/2) * ||w - w_global||²`
   - Better convergence on non-IID data
   - Paper: Li et al., ICML 2020
   - Config: `mu` (default 0.01)
   
3. **FedNova** (Non-IID Optimization)
   - Normalized gradient updates
   - Momentum-accelerated aggregation
   - Handles varying local epochs/dataset sizes
   - Paper: Wang et al., 2021
   - Config: `adapt_lr`, `momentum`
   
4. **DPFedAvg** (Differential Privacy)
   - Server-side Gaussian DP: clipping + noise injection
   - Privacy budget tracking via Opacus RDPAccountant
   - Paper: Abadi et al., 2016
   - Config: `max_grad_norm`, `noise_multiplier`, `target_delta`

**Factory**: `create_strategy(strategy_name, **kwargs)` for unified interface

#### `evaluate.py`
Evaluation utilities:
- Classification metrics (accuracy, precision, recall, F1)
- Per-class performance analysis
- Confusion matrix computation
- Macro and weighted averaging

#### `visualize.py`
Visualization functions:
- Convergence curves (loss, accuracy)
- Metrics comparison
- Client data distribution
- Per-class performance plots

### Experiment Scripts (experiments/)

- `run.py`: Universal Hydra-based experiment runner (main entry point)
- `run_fedavg.py`: FedAvg baseline experiments
- `run_fedprox.py`: FedProx heterogeneous network experiments
- `run_fednova.py`: FedNova non-IID experiments
- `run_dp.py`: DPFedAvg privacy experiments

All scripts use Hydra for configuration management with command-line overrides.

### Configuration Files (configs/)

- `default.yaml`: Base configuration for all experiments
- `heterogeneous.yaml`: Preset for high-heterogeneity scenarios
- `privacy.yaml`: Preset for differential privacy experiments

## Data Partitioning Strategies

| Strategy | Partition | Heterogeneity | EMD | Use Case |
|----------|-----------|---------------|-----|----------|
| **IID** | Random shuffle | Low | 0.0068 | Baseline, privacy testing |
| **Temporal** | Day-based | Medium | 0.0122 | Realistic network scenarios |
| **Attack Family** | By attack type | High | 0.0785 | Non-IID stress testing |

## Configuration Reference

### Command-Line Overrides (Hydra Dot Notation)

```bash
# Data configuration
data.partition=day              # iid | day | family
data.n_clients=10              # Number of clients
data.seed=42                   # Random seed

# Training configuration
training.local_epochs=5        # Local epochs per round
training.batch_size=256        # Batch size
training.learning_rate=0.001   # Adam learning rate
training.num_rounds=100        # Total federated rounds
training.device=cuda           # cpu | cuda | auto

# Strategy configuration
strategy.name=fedprox          # fedavg | fedprox | fednova | dpfedavg
strategy.mu=0.01              # FedProx: proximal coefficient
strategy.noise_multiplier=1.0  # DPFedAvg: noise std / clipping
strategy.max_grad_norm=1.0     # DPFedAvg: clipping threshold
strategy.momentum=0.9          # FedNova: momentum
strategy.adapt_lr=true         # FedNova: adaptive LR

# Experiment configuration
experiment.name=my_exp         # Experiment name
experiment.output_dir=results/metrics  # Output directory
experiment.verbose=true        # Verbose logging
experiment.seed=42             # Reproducibility
```

### Example Configurations

```bash
# IID, FedAvg (baseline)
python experiments/run.py

# Non-IID (family), FedProx with regularization
python experiments/run.py data.partition=family strategy.name=fedprox strategy.mu=0.05

# High heterogeneity with FedNova and momentum
python experiments/run.py data.partition=family strategy.name=fednova strategy.momentum=0.9

# Differential privacy with strong privacy
python experiments/run.py strategy.name=dpfedavg strategy.noise_multiplier=0.1

# Custom: 20 clients, 200 rounds, high learning rate
python experiments/run.py data.n_clients=20 training.num_rounds=200 training.learning_rate=0.01
```

### `evaluate.py`
Evaluation utilities:
- `ModelEvaluator`: Comprehensive metrics computation
- `FederatedEvaluator`: Evaluate federated results
- Metrics: accuracy, precision, recall, F1, specificity, AUC-ROC, FPR, FNR

### `visualize.py`
Visualization functions:
- Training convergence plots
- Confusion matrices
- Metrics comparison
- Strategy comparison
- Privacy-utility tradeoff analysis

## Key Features

### Data Handling
- ✅ Automatic handling of encoding errors
- ✅ Missing value detection and removal
- ✅ Infinite value handling
- ✅ Duplicate detection and removal
- ✅ Timestamp parsing and unification
- ✅ Automatic feature scaling

### Federated Learning
- ✅ IID and Non-IID data partitioning (Dirichlet)
- ✅ Multiple aggregation strategies
- ✅ Differential privacy support
- ✅ Heterogeneous client data support
- ✅ Communication-efficient training

### Evaluation & Analysis
- ✅ Comprehensive evaluation metrics
- ✅ Per-class performance analysis
- ✅ Confusion matrix computation
- ✅ Multi-strategy comparison
- ✅ Convergence analysis
- ✅ Client performance tracking

### Visualization
- ✅ Training loss and accuracy curves
- ✅ Confusion matrices
- ✅ Metrics comparison charts
- ✅ Client distribution visualization
- ✅ Strategy performance radar charts

## Configuration

### Hyperparameters

**Neural Network:**
- Input dimensions: 78 (number of features in CICIDS2017)
- Hidden layers: [128, 64]
- Output dimension: 1 (binary classification)
- Activation: ReLU
- Learning rate: 0.01

**Federated Learning:**
- Number of clients: 10
- Number of rounds: 20
- Local epochs per client: 5
- Batch size: 32
- Partition: Non-IID with α=0.1

**Differential Privacy (FedDP):**
- Epsilon (ε): Privacy budget (1.0)
- Delta (δ): Privacy failure probability (1e-5)
- Mechanism: Laplace noise addition

## Results

### Expected Performance
- Baseline Accuracy: ~95%
- FedAvg Accuracy: ~93-94%
- FedProx Accuracy: ~93-95%
- FedDP Accuracy: ~88-92% (depending on privacy level)

### Output Files
- `results/metrics/baseline_results.json`
- `results/metrics/fedavg_results.json`
- `results/metrics/fedprox_results.json`
- `results/metrics/feddp_results.json`
- `results/plots/convergence.png`
- `results/plots/metrics_comparison.png`
- `results/plots/strategy_comparison.png`

## Dataset Information

**CICIDS2017:**
- 8 days of network traffic (Monday to Friday with Saturday/Sunday variants)
- 78 features per flow
- Binary classification (Normal/Attack)
- Imbalanced dataset (~80% normal, ~20% attack)
- Approximately 2.8M total flows

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- NumPy, Pandas: Data processing
- Scikit-learn: Machine learning utilities
- Matplotlib, Seaborn: Visualization
- PyArrow: Parquet file support

## Troubleshooting

### Virtual Environment Issues
```bash
# If activation fails
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate.bat  # Windows

# If still having issues, recreate environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Import Errors
```bash
# Ensure you're in the right directory
cd /path/to/federated-ids

# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data Processing Issues
```bash
# Verify data is in correct location
ls data/raw/ | wc -l  # Should show 8 or 10 CSV files

# Re-run preprocessing
python src/preprocess.py

# Check output
ls data/processed/ | head -5
```

### Out of Memory
```bash
# Reduce batch size
python experiments/run.py training.batch_size=256

# Reduce number of clients
python experiments/run.py data.n_clients=3

# Reduce number of rounds
python experiments/run.py training.num_rounds=20
```

### Hydra Configuration Issues
```bash
# View resolved configuration
python experiments/run.py --cfg job

# List all parameters
python experiments/run.py --info=defaults

# Check config directory
ls -la configs/
```

### Slow Training
```bash
# Use fewer clients and rounds for testing
python experiments/run.py data.n_clients=2 training.num_rounds=5

# Reduce epochs
python experiments/run.py training.local_epochs=1
```

## Project Layout

```
federated-ids/
├── data/
│   ├── raw/              # CICIDS2017 CSV files (place here)
│   └── processed/        # Preprocessed parquet (auto-generated)
├── src/                  # Core implementation
│   ├── preprocess.py     # Data preprocessing
│   ├── dataset.py        # Partitioning & dataloaders
│   ├── model.py          # IDSNet neural network
│   ├── client.py         # Flower client (NumPyClient)
│   ├── server.py         # Flower server & aggregation
│   ├── strategies.py     # FedAvg, FedProx, FedNova, DPFedAvg
│   ├── evaluate.py       # Metrics computation
│   └── visualize.py      # Plotting utilities
├── experiments/          # Experiment runners
│   ├── run.py            # Main Hydra entry point
│   ├── run_fedavg.py     # FedAvg wrapper
│   ├── run_fedprox.py    # FedProx wrapper
│   ├── run_fednova.py    # FedNova wrapper
│   ├── run_dp.py         # DPFedAvg wrapper
│   └── README.md         # Detailed experiment guide
├── configs/              # Hydra configuration files
│   ├── default.yaml      # Base configuration
│   ├── heterogeneous.yaml # High-heterogeneity preset
│   ├── privacy.yaml      # Privacy preset
│   └── README.md         # Configuration guide
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_federation.ipynb
│   ├── 04_analysis.ipynb
│   ├── 05_dataset_testing.ipynb
│   └── 06_model_training.ipynb
├── results/              # Experiment outputs
│   ├── metrics/          # JSON results
│   └── plots/            # Generated plots
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Learning Resources

### Federated Learning Papers
- **FedAvg**: McMahan et al., ICML 2017 — "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **FedProx**: Li et al., MLSys 2020 — "Federated Optimization in Heterogeneous Networks"
- **FedNova**: Wang et al., ICLR 2021 — "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning"

### Differential Privacy
- **Deep Learning with DP**: Abadi et al., CCS 2016
- **Opacus**: Meta's differential privacy library (https://opacus.ai/)

### Flower Framework
- Documentation: https://flower.ai/
- GitHub: https://github.com/adap/flower

### CICIDS2017 Dataset
- Paper: Sharafaldin et al., 2018 — "Toward Generating a Dataset for High Accuracy Intrusion Detection Systems"
- Download: https://www.unb.ca/cic/datasets/ids-2017.html

## Next Steps

1. **Understand the Codebase**
   - Review `src/dataset.py` for data partitioning concepts
   - Study `src/model.py` for neural network architecture
   - Explore `src/strategies.py` for aggregation algorithms

2. **Run Experiments**
   - Start with FedAvg baseline: `python experiments/run.py`
   - Compare on non-IID data: `python experiments/run.py data.partition=family`
   - Test different strategies systematically

3. **Analyze Results**
   - Results saved to `results/metrics/*.json`
   - Use notebooks for visualization and analysis
   - Compare strategy performance

4. **Extend the Project**
   - Add new strategies (FedProxFT, Scaffold, etc.)
   - Implement client-side DP
   - Add Byzantine-robust aggregation
   - Integrate with real Flower gRPC servers
   - Support model heterogeneity

## References

### Key Papers
- McMahan, H. B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." ICML, 2017.
- Li, T., et al. "Federated Optimization in Heterogeneous Networks." MLSys, 2020.
- Wang, J., et al. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning." ICLR, 2021.
- Abadi, M., et al. "Deep Learning with Differential Privacy." CCS, 2016.
- Sharafaldin, I., et al. "Toward Generating a Dataset for High Accuracy Intrusion Detection Systems." CIC, 2018.

### Frameworks & Libraries
- **Flower**: Federated Learning Framework (https://flower.ai/)
- **Opacus**: Differential Privacy Library (https://opacus.ai/)
- **PyTorch**: Deep Learning Framework (https://pytorch.org/)

## Citation

If you use this project, please cite:

```bibtex
@software{federated_ids_2026,
  title={Federated Learning for Intrusion Detection System},
  author={Shailesh Kumar Sharma},
  year={2026},
  url={https://github.com/yourusername/federated-ids}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

## Support

- 📖 Check [experiments/README.md](experiments/README.md) for experiment documentation
- ⚙️ Check [configs/README.md](configs/README.md) for configuration help
- 🐍 Run `python experiments/run.py --help` for Hydra help
- 📝 Review notebooks for examples

## Changelog

### v1.0 (Current)
- ✅ Flower-based federated learning framework
- ✅ Four aggregation strategies (FedAvg, FedProx, FedNova, DPFedAvg)
- ✅ Three data partitioning strategies (IID, Temporal, Attack Family)
- ✅ Hydra configuration management
- ✅ Comprehensive metrics and evaluation
- ✅ Privacy budget tracking (Opacus)
- ✅ Jupyter notebooks for exploration and analysis
