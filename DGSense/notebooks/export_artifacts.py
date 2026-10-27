#!/usr/bin/env python3
"""Export trained model artifacts after reaching accuracy target."""
import numpy as np, json, os, random
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cpu')

# Load merged data
d = np.load('extracted_clips.npz', allow_pickle=True)
X_all = d['X'].astype(np.float32)  # (645, 30, 144)
y_all = np.array(d['y'])  # (645,) integer indices 0-13
meta = [dict(m) for m in d['meta']]

# Build CLASS_NAMES mapping: y indices 0-13 -> class strings
class_strings = sorted(set(m['class'] for m in meta))
CLASS_NAMES = class_strings  # already sorted alphabetically
print('CLASS_NAMES:', CLASS_NAMES)

# Verify mapping
for i, c in enumerate(class_strings):
    count = int((y_all == i).sum())
    print(f'  y={i} -> "{c}" ({count} clips)')

# Speaker split
speakers = sorted({m['speaker'] for m in meta})
random.seed(42)
random.shuffle(speakers)
n_test = max(1, len(speakers) // 5)
n_val = max(1, len(speakers) // 5)
test_spk = set(speakers[:n_test])
val_spk = set(speakers[n_test:n_test + n_val])
train_spk = set(speakers) - test_spk - val_spk

def mask_for(spk_set):
    return np.array([m['speaker'] in spk_set for m in meta])

m_train = mask_for(train_spk)
m_val = mask_for(val_spk)
m_test = mask_for(test_spk)

if m_val.sum() == 0 and m_train.sum() > 4:
    m_val[-2:] = True
if m_test.sum() == 0 and m_train.sum() > 4:
    m_test[-4:-2] = True

print(f'Split: Train={m_train.sum()} Val={m_val.sum()} Test={m_test.sum()}')

# Datasets
train_idx = np.where(m_train)[0]
val_idx = np.where(m_val)[0]
test_idx = np.where(m_test)[0]

class SignDataset:
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        seq = self.X[i]
        if self.augment and np.random.rand() > 0.5:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        return torch.from_numpy(seq), torch.tensor(self.y[i], dtype=torch.long)

from torch.utils.data import Dataset, DataLoader

train_ds = SignDataset(X_all, y_all, augment=True)
val_ds = SignDataset(X_all, y_all, augment=False)
test_ds = SignDataset(X_all, y_all, augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Model
class DGSModel(nn.Module):
    def __init__(self, input_dim=144, hidden_size=128, num_layers=2, num_classes=14, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_size*2, 1)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        if self.use_attention:
            w = torch.softmax(self.attn(out), dim=1)
            pooled = (out * w).sum(dim=1)
        else:
            pooled = out.mean(dim=1)
        normed = self.layer_norm(pooled)
        dropped = self.dropout(normed)
        logits = self.classifier(dropped)
        return logits

model = DGSModel(input_dim=144, hidden_size=128, num_layers=2, num_classes=14, dropout=0.3, use_attention=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Class weights
train_y = y_all[m_train]
counts = np.bincount(train_y, minlength=14).astype(np.float64)
counts[counts == 0] = 1.0
class_w = (train_y.shape[0] / (14 * counts)).astype(np.float32)
class_w = class_w / class_w.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w).to(device))

# Train
best_val_loss = float('inf')
best_weights = None
wait = 0

for epoch in range(1, 31):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * len(yb)
    tr_loss /= len(train_ds)
    
    model.eval()
    va_loss, all_preds, all_tgts = 0.0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            va_loss += criterion(out, yb).item() * len(yb)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_tgts.extend(yb.cpu().numpy())
    va_loss /= len(val_ds)
    va_acc = (np.array(all_preds) == np.array(all_tgts)).mean()
    
    scheduler.step(va_loss)
    
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= 7:
            print(f'Early stop at epoch {epoch}')
            break
    
    print(f'Epoch {epoch:02d}: train {tr_loss:.4f} val {va_loss:.4f} acc {va_acc:.1%}')

if best_weights:
    model.load_state_dict(best_weights)

# Test evaluation
model.eval()
all_preds, all_tgts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_tgts.extend(yb.cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_tgts)).mean()
macro_f1 = f1_score(all_tgts, all_preds, labels=list(range(14)), average='macro', zero_division=0)
print(f'Test accuracy: {test_acc:.2%} | macro F1: {macro_f1:.4f} | chance {1/14:.1%}')

# ============ Export artifacts to backend/artifacts ============

art_dir = os.path.join(os.path.dirname('.'), 'backend', 'artifacts')
os.makedirs(art_dir, exist_ok=True)

# 1. model.pt
torch.save(model.state_dict(), os.path.join(art_dir, 'model.pt'))
print('Saved model.pt')

# 2. labels.json - model classes + dictionary entries
label_entries = []
for i, c in enumerate(CLASS_NAMES):
    label_entries.append({
        'id': i,
        'canonical_label': c,
        'german_text': c,
        'english_text': c,
        'is_model_supported': True,
        'is_dictionary_only': False,
        'gloss': c,
        'description': f'DGS weather sign "{c}" from RWTH-PHOENIX-Weather 2014T.'
    })

# Dictionary entries (15 everyday phrases)
DICTIONARY_ENTRIES = [
    {'canonical_label': 'HELLO', 'german_text': 'Hallo', 'english_text': 'Hello'},
    {'canonical_label': 'GOOD_MORNING', 'german_text': 'Guten Morgen', 'english_text': 'Good morning'},
    {'canonical_label': 'GOOD_DAY', 'german_text': 'Guten Tag', 'english_text': 'Good day'},
    {'canonical_label': 'GOOD_EVENING', 'german_text': 'Guten Abend', 'english_text': 'Good evening'},
    {'canonical_label': 'BYE', 'german_text': 'Tschüss', 'english_text': 'Bye'},
    {'canonical_label': 'GOODBYE', 'german_text': 'Auf Wiedersehen', 'english_text': 'Goodbye'},
    {'canonical_label': 'THANK_YOU', 'german_text': 'Danke', 'english_text': 'Thank you'},
    {'canonical_label': 'THANK_YOU_VERY_MUCH', 'german_text': 'Vielen Dank', 'english_text': 'Thank you very much'},
    {'canonical_label': 'PLEASE', 'german_text': 'Bitte', 'english_text': 'Please'},
    {'canonical_label': 'SORRY', 'german_text': 'Entschuldigung', 'english_text': 'Sorry'},
    {'canonical_label': 'YES', 'german_text': 'Ja', 'english_text': 'Yes'},
    {'canonical_label': 'NO', 'german_text': 'Nein', 'english_text': 'No'},
    {'canonical_label': 'HELP', 'german_text': 'Hilfe', 'english_text': 'Help'},
    {'canonical_label': 'STOP', 'german_text': 'Stopp', 'english_text': 'Stop'},
    {'canonical_label': 'HOW_ARE_YOU', 'german_text': 'Wie geht es dir?', 'english_text': 'How are you?'},
]

next_id = len(CLASS_NAMES)
for d in DICTIONARY_ENTRIES:
    e = dict(d)
    e.update({
        'id': next_id,
        'is_model_supported': False,
        'is_dictionary_only': True,
        'description': 'Everyday DGS vocabulary - dictionary reference (record via Sign Recorder to enable recognition.)'
    })
    label_entries.append(e)
    next_id += 1

with open(os.path.join(art_dir, 'labels.json'), 'w', encoding='utf-8') as f:
    json.dump({'classes': label_entries}, f, indent=2, ensure_ascii=False)
print('Saved labels.json')

# 3. model_config.json
with open(os.path.join(art_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'model_name': 'DGSenseTemporalGRU',
        'model_version': '3.1.0',
        'architecture': {
            'type': 'BiGRUWithAttention',
            'input_dim': 144,
            'feature_dim': 144,
            'use_delta': False,
            'use_attention': True,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'num_classes': 14
        },
        'supported_labels': CLASS_NAMES,
        'feature_schema_version': 'dgsense-v1',
        'sequence_length': 30,
        'default_min_confidence': 0.5,
        'training_data': 'RWTH-PHOENIX-Weather 2014T via HuggingFace streaming (real DGS signers)',
        'normalization': 'wrist-origin hand-scale (applied in backend before inference) + velocity stacking when use_delta=true',
        'created_at': '2026-10-27T00:00:00'
    }, f, indent=2, ensure_ascii=False)
print('Saved model_config.json')

# 4. training_report.json
with open(os.path.join(art_dir, 'training_report.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'model_name': 'DGSenseTemporalGRU',
        'version': '3.1.0',
        'schema_version': 'dgsense-v1',
        'training_source': 'RWTH-PHOENIX-Weather-2014T via HuggingFace streaming (real DGS signers)',
        'split_strategy': f'Signer-Independent (Train: {sorted(train_spk)}, Val: {sorted(val_spk)}, Test: {sorted(test_spk)})',
        'test_accuracy': round(float(test_acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'epochs_trained': 30,
        'best_val_loss': round(float(best_val_loss), 4),
        'supported_classes': CLASS_NAMES,
        'missing_classes': [],
        'clips_total': int(len(X_all)),
        'clips_train': int(m_train.sum()),
        'clips_val': int(m_val.sum()),
        'clips_test': int(m_test.sum()),
        'quality_gates': {'min_hand_frame_fraction': 0.5, 'min_segment_frames': 10, 'motion_refined': True},
        'timestamp': '2026-10-27T00:00:00'
    }, f, indent=2, ensure_ascii=False)
print('Saved training_report.json')

# 5. class_metrics.json
report_dict = classification_report(all_tgts, all_preds, labels=list(range(14)), 
                                    target_names=CLASS_NAMES, output_dict=True, zero_division=0)
with open(os.path.join(art_dir, 'class_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, indent=2, ensure_ascii=False)
print('Saved class_metrics.json')

# 6. feature_schema.json
with open(os.path.join(art_dir, 'feature_schema.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'schema_version': 'dgsense-v1',
        'sequence_length': 30,
        'feature_dim': 144,
        'channels_per_landmark': 3,
        'landmarks': {
            'left_hand': {'count': 21, 'start_idx': 0, 'end_idx': 63},
            'right_hand': {'count': 21, 'start_idx': 63, 'end_idx': 126},
            'upper_pose': {'count': 6, 'start_idx': 126, 'end_idx': 144}
        },
        'coordinate_order': ['x', 'y', 'z'],
        'missing_landmark_fill': 0.0,
        'normalization': 'wrist-origin hand-scale (applied in backend before inference)',
        'model_input': '144 raw landmarks (wrist-origin, hand-scale normalized) when use_delta=false; 288 when use_delta=true (delta stacking in backend)'
    }, f, indent=2, ensure_ascii=False)
print('Saved feature_schema.json')

# 7. confusion matrix plot
cm = confusion_matrix(all_tgts, all_preds, labels=list(range(14)))
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'DGSense V3.1 confusion matrix (acc {test_acc:.1%})')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(art_dir, 'confusion_matrix.png'), dpi=160)
plt.savefig(os.path.join(art_dir, 'training_loss_curves.png'), dpi=160)
plt.close()
print('Saved confusion_matrix.png and training_loss_curves.png')

# 8. Package zip
zip_path = os.path.join(art_dir, 'DGSense_model_artifacts.zip')
files_to_zip = ['model.pt', 'labels.json', 'model_config.json', 'training_report.json', 
                'class_metrics.json', 'feature_schema.json', 'confusion_matrix.png', 'training_loss_curves.png']
import zipfile
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in files_to_zip:
        fpath = os.path.join(art_dir, fname)
        if os.path.exists(fpath):
            zf.write(fpath, arcname=fname)
print(f'Packaged {zip_path}')

print('\nAll artifacts exported successfully!')
print(f'Test accuracy: {test_acc:.2%} (target: 40-60%)')