# FedBoost-MVS

Federated XGBoost with Minimal Variance Sampling (MVS) using Flower framework.

## Installation

```bash
pip install -e .
```

## Running the Simulation

### Run on GPU (default)

The default configuration uses GPU acceleration:

```bash
flwr run
```

This configuration uses:
- `tree-method = "gpu_hist"` - GPU-accelerated histogram tree method
- `device = "cuda"` - CUDA device
- `num-gpus = 1.0` - Each client gets one full GPU

**Output**: Results are automatically saved to `results/run_YYYYMMDD_HHMMSS/` with:
- `final_model.json` - The trained global model
- `config.json` - Run configuration
- `metadata.json` - Run metadata (timestamp, params, rounds)

### Run on CPU

To run on CPU instead, modify `pyproject.toml`:

```toml
# XGBoost parameters
params.tree-method = "hist"  # Change from "gpu_hist" to "hist"
# params.device = "cuda"      # Comment out or remove this line

# Local simulation federation
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0  # Change from 1.0 to 0.0
```

Then run:

```bash
flwr run
```

## Configuration

### GPU Resource Allocation

The simulation engine assigns GPU resources to each client:

- `num-gpus = 1.0` - One full GPU per client (runs 1 client at a time per GPU)
- `num-gpus = 0.5` - Half GPU per client (runs 2 clients per GPU)
- `num-gpus = 0.25` - Quarter GPU per client (runs 4 clients per GPU)
- `num-gpus = 0.0` - No GPU (CPU-only mode)

**Note:** For XGBoost with `gpu_hist`, using `num-gpus = 1.0` is recommended to avoid memory contention.

### Simulation Parameters

Edit `pyproject.toml` to adjust:

- `num-server-rounds` - Number of federated learning rounds
- `fraction-train` - Fraction of clients participating in training
- `fraction-evaluate` - Fraction of clients participating in evaluation
- `local-epochs` - Number of local training rounds per client
- `options.num-supernodes` - Total number of federated clients

## Project Structure

```
src/fedboost_mvs/
├── client_app.py   # Flower ClientApp for local training
├── server_app.py   # Flower ServerApp for aggregation
└── task.py         # Data loading and utilities

experiments/        # Experiment configurations and scripts
└── README.md       # Guide for organizing experiments

results/            # Experimental results and findings
├── README.md       # Guide for documenting results
└── summary.md      # Summary of all findings
```

## Running Experiments

1. Create a new experiment directory in `experiments/`
2. Configure and run your experiment
3. Save results to `results/` with the same experiment ID
4. Document findings in `results/summary.md`
