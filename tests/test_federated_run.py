"""Test suite for fedboost-mvs federated learning system."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import xgboost as xgb


def _gpu_available():
    """Check if GPU is available for XGBoost."""
    try:
        # Try to build a DMatrix with GPU
        import numpy as np
        data = np.random.rand(10, 5)
        labels = np.random.randint(2, size=10)
        dtrain = xgb.DMatrix(data, label=labels)
        
        # Try to train with GPU
        params = {"tree_method": "gpu_hist", "device": "cuda"}
        xgb.train(params, dtrain, num_boost_round=1)
        return True
    except Exception:
        return False


def _run_federated_simulation(config_content, timeout=300):
    """Helper function to run a federated simulation with given config.
    
    Args:
        config_content: String content for pyproject.toml
        timeout: Timeout in seconds for the simulation
        
    Returns:
        tuple: (result, latest_run_path, execution_time)
    """
    project_root = Path(__file__).parent.parent
    original_config = project_root / "pyproject.toml"
    backup_config = project_root / "pyproject.toml.test_backup"
    
    # Save original pyproject.toml
    if original_config.exists():
        original_config.rename(backup_config)
    
    try:
        # Write test configuration
        with open(original_config, "w") as f:
            f.write(config_content)
        
        # Run flwr simulation and measure time
        start_time = time.time()
        result = subprocess.run(
            ["flwr", "run"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        execution_time = time.time() - start_time
        
        # Find the latest run directory
        results_dir = project_root / "results"
        run_dirs = list(results_dir.glob("run_*"))
        latest_run = max(run_dirs, key=lambda p: p.name) if run_dirs else None
        
        return result, latest_run, execution_time
        
    finally:
        # Restore original configuration
        if backup_config.exists():
            if original_config.exists():
                original_config.unlink()
            backup_config.rename(original_config)


def test_federated_run_completes():
    """Test that a basic federated run completes successfully (CPU)."""
    # Create a minimal test configuration
    test_config = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedboost-mvs"
version = "0.1.0"
description = "Federated XGBoost with Minimal Variance Sampling (MVS)"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "flwr[simulation]>=1.23.0,<2.0.0",
    "flwr-datasets>=0.5.0",
    "xgboost>=2.0.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/fedboost_mvs"]

[tool.flwr.app]
publisher = "WilliamLindskog"

[tool.flwr.app.components]
serverapp = "fedboost_mvs.server_app:app"
clientapp = "fedboost_mvs.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 1
fraction-train = 0.1
fraction-evaluate = 0.1
local-epochs = 1

# XGBoost parameters - CPU
params.objective = "binary:logistic"
params.eta = 0.1
params.max-depth = 4
params.eval-metric = "auc"
params.nthread = 2
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "hist"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 2
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0
"""
    
    result, latest_run, execution_time = _run_federated_simulation(test_config)
    
    # Check that the command completed successfully
    assert result.returncode == 0, f"flwr run failed with error:\n{result.stderr}"
    
    # Check that results directory was created
    assert latest_run is not None, "No run directory was created"
    
    # Check that required files exist
    assert (latest_run / "final_model.json").exists(), "Model file not created"
    assert (latest_run / "config.json").exists(), "Config file not created"
    assert (latest_run / "metadata.json").exists(), "Metadata file not created"
    
    # Validate metadata content
    with open(latest_run / "metadata.json") as f:
        metadata = json.load(f)
    
    assert "timestamp" in metadata
    assert metadata["num_rounds"] == 1
    assert metadata["fraction_train"] == 0.1
    assert "params" in metadata
    assert metadata["params"]["tree_method"] == "hist"
    
    print(f"✓ CPU federated run completed successfully")
    print(f"✓ Results saved to: {latest_run}")


def test_federated_run_cpu():
    """Test federated run with CPU configuration."""
    test_config = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedboost-mvs"
version = "0.1.0"
description = "Federated XGBoost with Minimal Variance Sampling (MVS)"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "flwr[simulation]>=1.23.0,<2.0.0",
    "flwr-datasets>=0.5.0",
    "xgboost>=2.0.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/fedboost_mvs"]

[tool.flwr.app]
publisher = "WilliamLindskog"

[tool.flwr.app.components]
serverapp = "fedboost_mvs.server_app:app"
clientapp = "fedboost_mvs.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 2
fraction-train = 0.1
fraction-evaluate = 0.1
local-epochs = 1

# XGBoost parameters - CPU
params.objective = "binary:logistic"
params.eta = 0.1
params.max-depth = 4
params.eval-metric = "auc"
params.nthread = 2
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "hist"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0
"""
    
    result, latest_run, execution_time = _run_federated_simulation(test_config, timeout=300)
    
    assert result.returncode == 0, f"CPU federated run failed:\n{result.stderr}"
    assert latest_run is not None, "No run directory created"
    assert (latest_run / "final_model.json").exists()
    
    # Validate CPU-specific configuration
    with open(latest_run / "metadata.json") as f:
        metadata = json.load(f)
    
    assert metadata["params"]["tree_method"] == "hist", "Should use CPU tree method"
    assert "device" not in metadata["params"] or metadata["params"].get("device") != "cuda"
    assert metadata["num_rounds"] == 2
    
    print(f"✓ CPU configuration test passed")


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
def test_federated_run_gpu():
    """Test federated run with GPU configuration."""
    test_config = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedboost-mvs"
version = "0.1.0"
description = "Federated XGBoost with Minimal Variance Sampling (MVS)"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "flwr[simulation]>=1.23.0,<2.0.0",
    "flwr-datasets>=0.5.0",
    "xgboost>=2.0.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/fedboost_mvs"]

[tool.flwr.app]
publisher = "WilliamLindskog"

[tool.flwr.app.components]
serverapp = "fedboost_mvs.server_app:app"
clientapp = "fedboost_mvs.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 2
fraction-train = 0.1
fraction-evaluate = 0.1
local-epochs = 1

# XGBoost parameters - GPU
params.objective = "binary:logistic"
params.eta = 0.1
params.max-depth = 4
params.eval-metric = "auc"
params.nthread = 2
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "gpu_hist"
params.device = "cuda"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 1.0
"""
    
    result, latest_run, execution_time = _run_federated_simulation(test_config, timeout=300)
    
    assert result.returncode == 0, f"GPU federated run failed:\n{result.stderr}"
    assert latest_run is not None, "No run directory created"
    assert (latest_run / "final_model.json").exists()
    
    # Validate GPU-specific configuration
    with open(latest_run / "metadata.json") as f:
        metadata = json.load(f)
    
    assert metadata["params"]["tree_method"] == "gpu_hist", "Should use GPU tree method"
    assert metadata["params"]["device"] == "cuda", "Should use CUDA device"
    assert metadata["num_rounds"] == 2
    
    print(f"✓ GPU configuration test passed")


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
def test_cpu_vs_gpu_performance():
    """Test that GPU training is faster than CPU training."""
    base_config = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedboost-mvs"
version = "0.1.0"
description = "Federated XGBoost with Minimal Variance Sampling (MVS)"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "flwr[simulation]>=1.23.0,<2.0.0",
    "flwr-datasets>=0.5.0",
    "xgboost>=2.0.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/fedboost_mvs"]

[tool.flwr.app]
publisher = "WilliamLindskog"

[tool.flwr.app.components]
serverapp = "fedboost_mvs.server_app:app"
clientapp = "fedboost_mvs.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-train = 0.2
fraction-evaluate = 0.1
local-epochs = 1

# XGBoost parameters
params.objective = "binary:logistic"
params.eta = 0.1
params.max-depth = 6
params.eval-metric = "auc"
params.nthread = 4
params.num-parallel-tree = 1
params.subsample = 1
"""

    # CPU Configuration
    cpu_config = base_config + """
params.tree-method = "hist"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 5
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0
"""

    # GPU Configuration
    gpu_config = base_config + """
params.tree-method = "hist"
params.device = "cuda"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 5
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 1.0
"""

    print("\n" + "="*70)
    print("CPU vs GPU Performance Comparison")
    print("="*70)
    
    # Run CPU benchmark
    print("\n[1/2] Running CPU benchmark (3 rounds, 5 clients)...")
    cpu_result, cpu_run, cpu_time = _run_federated_simulation(cpu_config, timeout=600)
    assert cpu_result.returncode == 0, f"CPU run failed:\n{cpu_result.stderr}"
    print(f"✓ CPU training completed in {cpu_time:.2f} seconds")
    
    # Run GPU benchmark
    print("\n[2/2] Running GPU benchmark (3 rounds, 5 clients)...")
    gpu_result, gpu_run, gpu_time = _run_federated_simulation(gpu_config, timeout=600)
    assert gpu_result.returncode == 0, f"GPU run failed:\n{gpu_result.stderr}"
    print(f"✓ GPU training completed in {gpu_time:.2f} seconds")
    
    # Compare performance
    speedup = cpu_time / gpu_time
    print("\n" + "="*70)
    print("Results:")
    print(f"  CPU Time:  {cpu_time:.2f}s")
    print(f"  GPU Time:  {gpu_time:.2f}s")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"  Time Saved: {cpu_time - gpu_time:.2f}s ({((cpu_time - gpu_time) / cpu_time * 100):.1f}%)")
    print("="*70)
    
    # Assert GPU is faster (with some tolerance for overhead)
    # GPU should be at least slightly faster, but allow for initialization overhead
    assert gpu_time <= cpu_time, f"GPU ({gpu_time:.2f}s) should be faster than or equal to CPU ({cpu_time:.2f}s)"
    
    # Save performance comparison results
    comparison_file = Path(__file__).parent.parent / "results" / "cpu_gpu_performance.json"
    comparison_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_time_seconds": cpu_time,
        "gpu_time_seconds": gpu_time,
        "speedup": speedup,
        "time_saved_seconds": cpu_time - gpu_time,
        "percentage_improvement": ((cpu_time - gpu_time) / cpu_time * 100),
        "cpu_run_dir": str(cpu_run),
        "gpu_run_dir": str(gpu_run),
    }
    
    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Performance comparison saved to: {comparison_file}\n")


def test_cpu_configuration():
    """Test that CPU configuration is valid."""
    project_root = Path(__file__).parent.parent
    cpu_config = project_root / "experiments" / "exp_001_cpu_gpu_benchmark" / "config_cpu.toml"
    
    assert cpu_config.exists(), "CPU config file not found"
    
    with open(cpu_config) as f:
        content = f.read()
        assert "tree-method = \"hist\"" in content
        assert "num-gpus = 0.0" in content or "# options.backend.client-resources.num-gpus = 0.0" in content


def test_gpu_configuration():
    """Test that GPU configuration is valid."""
    project_root = Path(__file__).parent.parent
    gpu_config = project_root / "experiments" / "exp_001_cpu_gpu_benchmark" / "config_gpu.toml"
    
    assert gpu_config.exists(), "GPU config file not found"
    
    with open(gpu_config) as f:
        content = f.read()
        assert "tree-method = \"gpu_hist\"" in content
        assert "device = \"cuda\"" in content
        assert "num-gpus = 1.0" in content or "# options.backend.client-resources.num-gpus = 1.0" in content


def test_project_structure():
    """Test that the project has the expected structure."""
    project_root = Path(__file__).parent.parent
    
    # Check source files
    assert (project_root / "src" / "fedboost_mvs" / "__init__.py").exists()
    assert (project_root / "src" / "fedboost_mvs" / "client_app.py").exists()
    assert (project_root / "src" / "fedboost_mvs" / "server_app.py").exists()
    assert (project_root / "src" / "fedboost_mvs" / "task.py").exists()
    
    # Check configuration
    assert (project_root / "pyproject.toml").exists()
    
    # Check documentation
    assert (project_root / "README.md").exists()
    
    # Check experiment structure
    assert (project_root / "experiments").exists()
    assert (project_root / "results").exists()


def test_server_app_imports():
    """Test that server_app can be imported without errors."""
    from fedboost_mvs import server_app
    
    assert hasattr(server_app, "app")
    assert server_app.app is not None


def test_client_app_imports():
    """Test that client_app can be imported without errors."""
    from fedboost_mvs import client_app
    
    assert hasattr(client_app, "app")
    assert client_app.app is not None


def test_task_module():
    """Test that task module has required functions."""
    from fedboost_mvs import task
    
    assert hasattr(task, "load_data")
    assert hasattr(task, "replace_keys")
    assert callable(task.load_data)
    assert callable(task.replace_keys)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
