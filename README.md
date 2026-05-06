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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

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

### 3. Preprocess Data

Run the preprocessing pipeline to clean and prepare the data:

```bash
python src/preprocess.py data/raw data/processed
```

This will:
- Load all 8 CSV files and merge them
- Remove infinite values in Flow_Bytes/s and Flow_Packets/s
- Remove duplicate rows
- Parse and unify timestamps
- Encode attack types as integer labels
- Apply standard scaling to numeric features
- Save processed data as parquet files

## Quick Start

### Option 1: Run Experiments from Command Line

#### Baseline (Centralized Learning)
```bash
python experiments/run_baseline.py \
    --data_dir data/processed \
    --output_dir results \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.01
```

#### FedAvg
```bash
python experiments/run_fedavg.py \
    --data_dir data/processed \
    --output_dir results \
    --num_clients 10 \
    --num_rounds 20 \
    --local_epochs 5 \
    --partition non-iid \
    --alpha 0.1
```

#### FedProx
```bash
python experiments/run_fedprox.py \
    --data_dir data/processed \
    --output_dir results \
    --num_clients 10 \
    --num_rounds 20 \
    --local_epochs 5 \
    --mu 0.01
```

#### FedDP (with Differential Privacy)
```bash
python experiments/run_dp.py \
    --data_dir data/processed \
    --output_dir results \
    --num_clients 10 \
    --num_rounds 20 \
    --epsilon 1.0 \
    --delta 1e-5
```

### Option 2: Use Jupyter Notebooks

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

2. **Baseline Model Training**
   ```bash
   jupyter notebook notebooks/02_baseline.ipynb
   ```

3. **Federated Learning Demo**
   ```bash
   jupyter notebook notebooks/03_federation.ipynb
   ```

4. **Results Analysis**
   ```bash
   jupyter notebook notebooks/04_analysis.ipynb
   ```

## Module Documentation

### `preprocess.py`
Data preprocessing module with 7 main tasks:
- **Task 1**: Load and merge 8 daily CSV files
- **Task 2**: Remove infinite values in Flow_Bytes/s and Flow_Packets/s
- **Task 3**: Remove duplicate rows
- **Task 4**: Parse and unify timestamps
- **Task 5**: Encode Attack_Type as integer labels
- **Task 6**: Standard scale numeric features
- **Task 7**: Save processed data as parquet

### `dataset.py`
Dataset management classes:
- `CICIDSDataset`: Single dataset wrapper
- `FederatedDataset`: Distributes data across clients (IID/Non-IID)
- `DataLoader`: Batching utility
- `load_processed_data()`: Load preprocessed parquet files

### `model.py`
Neural network implementations:
- `NeuralNetwork`: Feedforward network with configurable hidden layers
- `DeepNetwork`: Network with dropout capability
- Forward pass, backward pass, and weight updates

### `strategies.py`
Aggregation strategies:
- `FedAvg`: Weighted averaging of client weights
- `FedProx`: Handles heterogeneous data with proximal term
- `FedDP`: Differential privacy with Laplace noise
- `StrategyFactory`: Factory for creating strategies

### `client.py`
Federated client implementation:
- `FederatedClient`: Handles local training and evaluation
- `ClientManager`: Manages multiple clients
- Local model updates and client selection

### `server.py`
Federated server implementation:
- `FederatedServer`: Coordinates federated learning rounds
- Model aggregation and broadcasting
- Global model evaluation and checkpointing

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

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Reduce number of clients: `--num_clients 5`
- Process data in chunks

### Slow Training
- Use fewer rounds: `--num_rounds 10`
- Reduce local epochs: `--local_epochs 3`
- Enable GPU acceleration (modify model.py)

### Missing Data Files
- Ensure all 8 CSV files are in `data/raw/`
- Check file encoding (UTF-8 or Latin-1)
- Verify column names match expected format

## References

### Federated Learning
- McMahan, H. B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." ICML, 2017. (FedAvg)
- Li, T., et al. "Federated optimization in heterogeneous networks." MLSys, 2020. (FedProx)

### Differential Privacy
- Dwork, C., & Roth, A. "The Algorithmic Foundations of Differential Privacy." FnT TCS, 2014.
- Abadi, M., et al. "Deep Learning with Differential Privacy." CCS, 2016.

### IDS/Intrusion Detection
- Sharafaldin, I., et al. "Toward Generating a Dataset for High Accuracy Intrusion Detection Systems." 2018. (CICIDS2017)

## Author

Federated IDS Project - 2024

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open an GitHub issue or contact the maintainers.
