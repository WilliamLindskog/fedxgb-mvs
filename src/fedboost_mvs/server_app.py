"""fedboost-mvs: Flower ServerApp for federated XGBoost aggregation."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from fedboost_mvs.task import replace_keys

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated XGBoost training with bagging aggregation."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load global model into booster
    bst.load_model(global_model)

    # Save model
    model_path = output_dir / "final_model.json"
    print(f"\nSaving final model to {model_path}...")
    bst.save_model(str(model_path))
    
    # Save run configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(dict(context.run_config), f, indent=2)
    
    # Save run metadata
    metadata = {
        "timestamp": timestamp,
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "params": params,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
