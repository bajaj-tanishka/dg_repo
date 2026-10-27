
import json
from pathlib import Path
import torch

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# Print the saved model configuration
config_path = ARTIFACT_DIR / "model_config.json"

print("\n========== MODEL CONFIG ==========")
with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

print(json.dumps(config, indent=2))

# Inspect checkpoint parameter names and tensor shapes
checkpoint_path = ARTIFACT_DIR / "model.pt"

print("\n========== CHECKPOINT ==========")

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

if isinstance(checkpoint, dict):
    # Handle common checkpoint wrappers
    for key in ("state_dict", "model_state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            print("Checkpoint wrapper:", key)
            break

    for name, value in checkpoint.items():
        if hasattr(value, "shape"):
            print(f"{name}: {tuple(value.shape)}")
        else:
            print(f"{name}: {type(value).__name__}")
else:
    print("Unexpected checkpoint type:", type(checkpoint).__name__)
