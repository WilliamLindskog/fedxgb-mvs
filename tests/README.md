# Tests

This directory contains test suites for the FedBoost-MVS system.

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_federated_run.py -v
```

Run specific test:
```bash
pytest tests/test_federated_run.py::test_federated_run_completes -v
```

## Test Coverage

### `test_federated_run.py`

- **`test_federated_run_completes`**: Quick end-to-end test that runs a minimal federated learning simulation (1 round, 2 clients, CPU) and verifies:
  - Simulation completes successfully
  - Results directory is created with timestamped run folder
  - All required files are generated (model, config, metadata)
  - Metadata contains expected values

- **`test_federated_run_cpu`**: Tests federated learning with CPU-specific configuration (2 rounds, 3 clients):
  - Validates CPU tree method is used (`hist`)
  - Ensures no CUDA device is configured
  - Verifies model training completes successfully

- **`test_federated_run_gpu`**: Tests federated learning with GPU-specific configuration (2 rounds, 3 clients):
  - Automatically skipped if GPU is not available
  - Validates GPU tree method is used (`gpu_hist`)
  - Ensures CUDA device is configured
  - Verifies model training completes successfully on GPU

- **`test_cpu_vs_gpu_performance`**: Performance benchmark comparing CPU vs GPU training:
  - Runs 3 rounds with 5 clients on both CPU and GPU
  - Measures and compares execution time
  - Calculates speedup factor and time saved
  - Saves performance comparison to `results/cpu_gpu_performance.json`
  - Automatically skipped if GPU is not available
  - Asserts GPU is faster than or equal to CPU

- **`test_cpu_configuration`**: Validates CPU experiment configuration files

- **`test_gpu_configuration`**: Validates GPU experiment configuration files

- **`test_project_structure`**: Checks that all expected project files exist

- **`test_server_app_imports`**: Verifies server app can be imported

- **`test_client_app_imports`**: Verifies client app can be imported

- **`test_task_module`**: Validates task module has required functions

## Running Specific Tests

```bash
# Run only CPU test
pytest tests/test_federated_run.py::test_federated_run_cpu -v

# Run only GPU test (if GPU available)
pytest tests/test_federated_run.py::test_federated_run_gpu -v

# Run CPU vs GPU performance comparison (if GPU available)
pytest tests/test_federated_run.py::test_cpu_vs_gpu_performance -v -s

# Run both CPU and GPU tests
pytest tests/test_federated_run.py::test_federated_run_cpu tests/test_federated_run.py::test_federated_run_gpu -v
```

**Note**: Use `-s` flag with performance test to see detailed timing output in real-time.

## Adding New Tests

When adding new functionality:

1. Create a new test file: `test_<feature>.py`
2. Import required modules in `conftest.py` if needed
3. Follow the naming convention: `test_<what_you_are_testing>`
4. Include docstrings explaining what the test validates

## CI/CD Integration

These tests can be integrated into a CI/CD pipeline (GitHub Actions, etc.) to ensure code quality on every commit.

Example GitHub Actions workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e .
      - run: pip install pytest
      - run: pytest tests/ -v
```
