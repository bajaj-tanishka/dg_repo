"""
DGSense Temporal Model Training Script (synthetic smoke-trainer)

NOTE: This script trains on CALIBRATED SYNTHETIC data and exists only as a fast
      end-to-end smoke test of the training loop. It does NOT produce a model that
      works with real webcam input.

Real training happens in `notebooks/DGSense_Training.ipynb`, which streams real
      DGS footage (RWTH-PHOENIX-Weather 2014T) from HuggingFace, extracts MediaPipe
      landmarks in memory, and exports artifacts into `backend/artifacts/`.

Samples recorded through the Sign Recorder (uploaded to the `recorded_samples`
      SQLite table) can be exported and merged into real training later.
"""

import math
import random
import json
from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from backend.model_service import DGSenseTemporalGRU, normalize_frame, to_model_input
from backend.config import settings

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSES = [
    "HELLO", "THANK_YOU", "SORRY", "YES", "NO",
    "PLEASE", "GOOD_MORNING", "GOODBYE", "HELP", "STOP"
]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASSES)}

def generate_trajectory(label: str, signer_id: str, sample_idx: int) -> np.ndarray:
    """
    Generates a 30-frame sequence (144 features/frame) exhibiting the distinct
    kinematic and spatial signature of the specified German Sign Language sign.
    """
    seq = np.zeros((30, 144), dtype=np.float32)
    
    # Signer-specific tempo and spatial offset
    speed = 0.85 + (hash(signer_id) % 30) / 100.0
    pos_x_offset = np.random.uniform(-0.05, 0.05)
    pos_y_offset = np.random.uniform(-0.05, 0.05)
    scale = np.random.uniform(0.9, 1.1)

    for t in range(30):
        # Progress parameter 0.0 to 1.0
        p = (t / 29.0) * speed

        # Base torso references (shoulders 126..131, elbows 132..137, wrists 138..143)
        seq[t, 126] = 0.35 + pos_x_offset # left shoulder
        seq[t, 127] = 0.50 + pos_y_offset
        seq[t, 129] = 0.65 + pos_x_offset # right shoulder
        seq[t, 130] = 0.50 + pos_y_offset

        # Dominant hand base (wrist is landmark 0, indices 63, 64, 65)
        # We model the distinct kinematic trajectory of each DGS gesture:
        if label == "HELLO":
            # Lateral wave: periodic horizontal oscillation of right hand near head
            wx = 0.65 + 0.12 * math.sin(3.5 * math.pi * p) * scale + pos_x_offset
            wy = 0.32 + 0.03 * math.cos(3.5 * math.pi * p) * scale + pos_y_offset
            wz = 0.05 + 0.02 * math.sin(2.0 * math.pi * p)
            # Open palm shape: fingers extended upward
            hand_shape_spread = 0.06

        elif label == "THANK_YOU":
            # Chin to forward: starts at chin (y=0.40, z=0.1) and pushes outward towards partner (y=0.55, z=-0.2)
            wx = 0.52 + 0.02 * p + pos_x_offset
            wy = 0.38 + 0.18 * p * scale + pos_y_offset
            wz = 0.10 - 0.25 * p * scale
            hand_shape_spread = 0.04

        elif label == "SORRY":
            # Circular rubbing motion over upper sternum
            radius = 0.06 * scale
            angle = 3.0 * math.pi * p
            wx = 0.50 + radius * math.cos(angle) + pos_x_offset
            wy = 0.48 + radius * math.sin(angle) + pos_y_offset
            wz = 0.08
            hand_shape_spread = 0.02 # closed/cupped hand

        elif label == "YES":
            # Fist nodding up and down repeatedly at wrist
            nod = math.sin(4.0 * math.pi * p)
            wx = 0.60 + pos_x_offset
            wy = 0.55 + 0.08 * nod * scale + pos_y_offset
            wz = 0.05 + 0.04 * nod
            hand_shape_spread = 0.015 # fist

        elif label == "NO":
            # Side-to-side wagging / snapping
            wag = math.sin(5.0 * math.pi * p)
            wx = 0.58 + 0.10 * wag * scale + pos_x_offset
            wy = 0.52 + pos_y_offset
            wz = 0.05
            hand_shape_spread = 0.03

        elif label == "PLEASE":
            # Gentle downward stroke on upper chest
            wx = 0.48 + 0.02 * math.sin(math.pi * p) + pos_x_offset
            wy = 0.42 + 0.20 * p * scale + pos_y_offset # downward
            wz = 0.06
            hand_shape_spread = 0.05 # flat open hand

        elif label == "GOOD_MORNING":
            # Rising arc from lower chest upward towards face (sun rising)
            wx = 0.50 + 0.12 * p * scale + pos_x_offset
            wy = 0.65 - 0.35 * p * scale + pos_y_offset # rising
            wz = 0.05 + 0.05 * math.sin(math.pi * p)
            hand_shape_spread = 0.05

        elif label == "GOODBYE":
            # Waving parting motion: open hand flexes and oscillates horizontally
            wave = math.sin(3.0 * math.pi * p)
            wx = 0.70 + 0.10 * wave * scale + pos_x_offset
            wy = 0.35 + 0.04 * math.sin(2.0 * math.pi * p) + pos_y_offset
            wz = 0.08
            hand_shape_spread = 0.06

        elif label == "HELP":
            # Two-handed upward support: dominant hand sits in other hand and lifts upward
            lift = p * scale
            wx = 0.50 + pos_x_offset
            wy = 0.65 - 0.20 * lift + pos_y_offset # upward lift
            wz = 0.08
            hand_shape_spread = 0.04

            # Supporting left hand active as well (indices 0..62)
            for lm in range(21):
                b_l = lm * 3
                seq[t, b_l] = 0.48 + (lm % 5) * 0.015 + pos_x_offset
                seq[t, b_l + 1] = wy + 0.04 + (lm // 5) * 0.01
                seq[t, b_l + 2] = wz - 0.02

        elif label == "STOP":
            # Decisive downward halt strike onto open palm
            if p < 0.6:
                # Coming down fast
                strike_p = p / 0.6
                wx = 0.52 + pos_x_offset
                wy = 0.35 + 0.25 * (strike_p ** 2) * scale + pos_y_offset
                wz = 0.10 - 0.05 * strike_p
            else:
                # Static halt hold
                wx = 0.52 + pos_x_offset
                wy = 0.60 + pos_y_offset
                wz = 0.05
            hand_shape_spread = 0.05

        # Populate dominant hand (indices 63..125) with 21 landmarks relative to wrist
        seq[t, 63] = wx # wrist x
        seq[t, 64] = wy # wrist y
        seq[t, 65] = wz # wrist z
        seq[t, 141] = wx # right wrist pose slot
        seq[t, 142] = wy
        seq[t, 143] = wz

        # 20 finger joints relative to wrist
        for lm in range(1, 21):
            finger_idx = (lm - 1) // 4
            joint_idx = (lm - 1) % 4
            dist = (joint_idx + 1) * 0.025 * scale
            angle = (finger_idx - 2) * 0.25

            base = 63 + lm * 3
            noise = np.random.normal(0, 0.005)
            seq[t, base] = wx + dist * math.sin(angle) * hand_shape_spread * 15.0 + noise
            seq[t, base + 1] = wy - dist * math.cos(angle) + noise
            seq[t, base + 2] = wz + noise * 0.5

        # Also populate left-hand slots (indices 0..62) if single-hand sign,
        # so model recognizes whether user signs with left or right hand!
        if label != "HELP":
            for lm in range(21):
                base_r = 63 + lm * 3
                base_l = lm * 3
                # mirror x across center 0.5
                seq[t, base_l] = 1.0 - seq[t, base_r]
                seq[t, base_l + 1] = seq[t, base_r + 1]
                seq[t, base_l + 2] = seq[t, base_r + 2]
            seq[t, 138] = 1.0 - wx
            seq[t, 139] = wy
            seq[t, 140] = wz

    return seq

class SignDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]], augment: bool = True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, lbl = self.samples[idx]
        seq = np.copy(seq)
        if self.augment:
            # Subtle temporal/spatial jitter. Features are wrist-normalized (can be
            # negative), so do NOT clip to [0, 1] here.
            noise = np.random.normal(0, 0.008, seq.shape).astype(np.float32)
            seq = seq + noise
            if np.random.rand() > 0.5:
                stretch = np.random.uniform(0.95, 1.05)
                seq = seq * stretch
        return torch.from_numpy(seq), torch.tensor(lbl, dtype=torch.long)

def train():
    print("[DGSense Training] WARNING: synthetic smoke-trainer - use the Colab notebook")
    print("[DGSense Training] (notebooks/DGSense_Training.ipynb) for real DGS data.")
    print("[DGSense Training] Building multi-signer synthetic dataset for 10 classes...")
    signers = ["signer_01", "signer_02", "signer_03", "signer_04", "signer_05"]
    
    train_samples = []
    val_samples = []
    test_samples = []

    for class_name in CLASSES:
        lbl_idx = LABEL2ID[class_name]
        for s_idx, signer in enumerate(signers):
            # 20 samples per signer per class = 100 samples per class = 1000 total samples
            for i in range(20):
                seq = generate_trajectory(class_name, signer, i)
                # Apply the same transforms used at inference (model_service.predict) so
                # train and serve see identical features.
                seq = np.stack([normalize_frame(frame) for frame in seq]).astype(np.float32)
                if input_dim == 2 * 144:
                    seq = to_model_input(seq)
                if s_idx in (0, 1, 2): # signers 01, 02, 03 -> train
                    train_samples.append((seq, lbl_idx))
                elif s_idx == 3: # signer 04 -> val
                    val_samples.append((seq, lbl_idx))
                else: # signer 05 -> test (signer-independent)
                    test_samples.append((seq, lbl_idx))

    print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    train_loader = DataLoader(SignDataset(train_samples, augment=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(SignDataset(val_samples, augment=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(SignDataset(test_samples, augment=False), batch_size=32, shuffle=False)

    # Match the architecture currently declared in artifacts/model_config.json so the
    # produced checkpoint always loads in the backend (144 raw vs 288 velocity-stacked).
    model_config_path = settings.ARTIFACTS_DIR / "model_config.json"
    arch = {}
    if model_config_path.exists():
        with open(model_config_path, "r", encoding="utf-8") as f:
            arch = json.load(f).get("architecture", {})
    input_dim = arch.get("input_dim", 144)
    use_attention = arch.get("use_attention", False)
    print(f"[DGSense Training] arch: input_dim={input_dim} use_attention={use_attention}")

    model = DGSenseTemporalGRU(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=10,
        dropout=0.25,
        bidirectional=True,
        use_attention=use_attention,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_weights = None

    print("[DGSense Training] Training BiGRU temporal network...")
    for epoch in range(1, 36):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_samples)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
        val_loss /= len(val_samples)
        val_acc = correct / len(val_samples)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/35] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Test set evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    test_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"\n[Test Results] Test Accuracy: {test_acc:.2%}, Macro F1: {macro_f1:.4f}")

    # Save checkpoint to backend/artifacts/model.pt
    ckpt_path = settings.ARTIFACTS_DIR / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved optimal model weights to: {ckpt_path} ({ckpt_path.stat().st_size / 1024:.1f} KB)")

    # Update training_report.json
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True)
    report_payload = {
        "model_name": "DGSenseTemporalGRU",
        "version": "2.1.0",
        "schema_version": "dgsense-v1",
        "training_source": "synthetic smoke-trainer (not real DGS - retrain via Colab notebook)",
        "split_strategy": "Signer-Independent (Train: ['signer_01', 'signer_02', 'signer_03'], Val: signer_04, Test: signer_05)",
        "test_accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "supported_classes": CLASSES,
        "timestamp": "2026-10-26T13:50:00Z"
    }
    with open(settings.ARTIFACTS_DIR / "training_report.json", "w") as f:
        json.dump(report_payload, f, indent=2)

    with open(settings.ARTIFACTS_DIR / "class_metrics.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print("[DGSense Training] Complete!")

if __name__ == "__main__":
    train()
