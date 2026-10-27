
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Resolve paths relative to this file, not the terminal's working directory.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"



import torch
import torch.nn as nn


class DGSenseTemporalGRU(nn.Module):
    def __init__(
        self,
        input_dim=288,
        hidden_dim=128,
        num_layers=2,
        num_classes=14,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)

        if use_attention:
            self.attn = nn.Linear(output_dim, 1)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, input_dim)
        gru_output, _ = self.gru(x)

        if self.use_attention:
            # One attention score per time step
            attention_scores = self.attn(gru_output)
            attention_weights = torch.softmax(
                attention_scores, dim=1
            )

            # Weighted temporal representation
            context = torch.sum(
                gru_output * attention_weights, dim=1
            )
        else:
            context = gru_output.mean(dim=1)

        context = self.layer_norm(context)
        context = self.dropout(context)

        return self.classifier(context)



def read_labels(labels_path, config=None):
    import json
    from pathlib import Path

    labels_path = Path(labels_path)

    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Format used by DGSense labels.json:
    # {"classes": [{"id": 0, "canonical_label": "RAIN", ...}, ...]}
    if isinstance(data, dict) and isinstance(data.get("classes"), list):
        entries = data["classes"]

        # Keep only classes that the trained model supports.
        supported = [
            item for item in entries
            if isinstance(item, dict)
            and item.get("is_model_supported") is True
        ]

        if not supported:
            raise ValueError(
                "No model-supported classes found in labels.json."
            )

        # Sort by the model's original class IDs.
        supported.sort(key=lambda item: item.get("id", -1))

        ids = [item.get("id") for item in supported]

        if ids != list(range(len(supported))):
            raise ValueError(
                f"Supported class IDs must be consecutive from 0. Found: {ids}"
            )

        labels = [item.get("canonical_label") for item in supported]

    # Also support a plain list of label strings.
    elif isinstance(data, list):
        labels = data

    # Support {"labels": [...]} or {"classes": [...]}.
    elif isinstance(data, dict) and isinstance(data.get("labels"), list):
        labels = data["labels"]

    else:
        raise ValueError(
            "Unsupported labels.json format. Expected a classes list "
            "containing class objects or a list of label strings."
        )

    if (
        not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
    ):
        raise ValueError(
            "Labels must contain non-empty strings."
        )

    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate class labels found.")

    return labels



def load_dgs_model():
    config_path = ARTIFACT_DIR / "model_config.json"
    labels_path = ARTIFACT_DIR / "labels.json"
    checkpoint_path = ARTIFACT_DIR / "model.pt"

    for path in (config_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    labels = read_labels(labels_path, config)

    input_dim = int(config.get("input_dim", 144))
    sequence_length = int(config.get("sequence_length", 30))
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 2))
    num_classes = int(config.get("num_classes", len(labels)))
    dropout = float(config.get("dropout", 0.35))

    if input_dim != 144 or sequence_length != 30:
        raise ValueError(
            f"Unexpected input shape: {sequence_length} frames, "
            f"{input_dim} features. Verify the training feature schema."
        )

    if len(labels) != num_classes:
        raise ValueError(
            f"Label count ({len(labels)}) does not match "
            f"model class count ({num_classes})."
        )

    architecture = config["architecture"]

    model = DGSenseTemporalGRU(
        input_dim=architecture["input_dim"],
        hidden_dim=architecture["hidden_dim"],
        num_layers=architecture["num_layers"],
        num_classes=architecture["num_classes"],
        dropout=architecture["dropout"],
        bidirectional=architecture["bidirectional"],
        use_attention=architecture["use_attention"],
    )

    # This checkpoint should contain a state_dict, as saved by the notebook.
    # Only load checkpoints from sources you trust.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("model.pt does not contain a state dictionary.")

    # Strict loading catches architecture/key/shape mismatches.
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model, labels, config



def predict_features(model, labels, features):
    """
    Predict from 30 frames of normalized landmarks.
    Input:  (30, 144)
    Model:  (1, 30, 288) when delta features are enabled.
    """
    features = np.asarray(features, dtype=np.float32)

    if features.shape != (30, 144):
        raise ValueError(
            f"Expected feature shape (30, 144), got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Input contains NaN or infinite values.")

    config_path = ARTIFACT_DIR / "model_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    use_delta = config["architecture"].get("use_delta", True)

    if use_delta:
        delta = np.zeros_like(features, dtype=np.float32)
        delta[1:] = features[1:] - features[:-1]
        features = np.concatenate([features, delta], axis=1)

    expected_dim = config["architecture"]["input_dim"]
    if features.shape != (30, expected_dim):
        raise ValueError(
            f"Expected (30, {expected_dim}), got {features.shape}"
        )

    x = torch.from_numpy(features).unsqueeze(0)
    model.eval()

    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=-1)[0]
        class_id = int(probabilities.argmax().item())

    return {
        "label": labels[class_id],
        "class_id": class_id,
        "confidence": float(probabilities[class_id].item()),
        "probabilities": {
            labels[i]: float(probabilities[i].item())
            for i in range(len(labels))
        },
    }

