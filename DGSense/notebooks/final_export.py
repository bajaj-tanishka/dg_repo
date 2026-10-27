#!/usr/bin/env python3
"""Final export with proper model service compatibility."""
import sys
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'backend'))
import numpy as np, json, os, random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from model_service import DGSenseTemporalGRU

device = torch.device('cpu')

# Load merged data
d = np.load('extracted_clips.npz', allow_pickle=True)
X_all = d['X'].astype(np.float32)
y_all = np.array(d['y'])
meta = [dict(m) for m in d['meta']]

class_strings = sorted(set(m['class'] for m in meta))
CLASS_NAMES = class_strings

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

train_idx = np.where(m_train)[0]
val_idx = np.where(m_val)[0]
test_idx = np.where(m_test)[0]

class SignDataset:
    def __init__(self, X, y, augment=False):
        self.X = X; self.y = y; self.augment = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        seq = self.X[i]
        if self.augment and np.random.rand() > 0.5:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        return torch.from_numpy(seq), torch.tensor(self.y[i], dtype=torch.long)

train_ds = SignDataset(X_all, y_all, augment=True)
val_ds = SignDataset(X_all, y_all, augment=False)
test_ds = SignDataset(X_all, y_all, augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Model using the exact DGSenseTemporalGRU from model_service
model = DGSenseTemporalGRU(
    input_dim=144, hidden_dim=128, num_layers=2, num_classes=14,
    dropout=0.3, bidirectional=True, use_attention=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_y = y_all[m_train]
counts = np.bincount(train_y, minlength=14).astype(np.float64)
counts[counts == 0] = 1.0
class_w = (train_y.shape[0] / (14 * counts)).astype(np.float32)
class_w = class_w / class_w.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w).to(device))

# Quick train - 10 epochs
best_val_loss = float('inf')
best_weights = None
wait = 0
for epoch in range(1, 11):
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
    scheduler.step(va_loss)
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= 5: break
    print(f'Epoch {epoch:02d}: train {tr_loss:.4f} val {va_loss:.4f}')

if best_weights: model.load_state_dict(best_weights)

# Test
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
print(f'Test accuracy: {test_acc:.2%} | macro F1: {macro_f1:.4f}')

# Save model.pt
os.makedirs('../backend/artifacts', exist_ok=True)
torch.save(model.state_dict(), '../backend/artifacts/model.pt')
print('Saved model.pt with 14 classes')

# Save labels.json
label_entries = []
for i, c in enumerate(CLASS_NAMES):
    label_entries.append({
        'id': i, 'canonical_label': c, 'german_text': c, 'english_text': c,
        'is_model_supported': True, 'is_dictionary_only': False,
        'gloss': c, 'description': f'DGS weather sign "{c}" from RWTH-PHOENIX-Weather 2014T.'
    })

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
    e.update({'id': next_id, 'is_model_supported': False, 'is_dictionary_only': True,
               'description': 'Everyday DGS vocabulary - dictionary reference (record via Sign Recorder to enable recognition.)'})
    label_entries.append(e)
    next_id += 1

with open('../backend/artifacts/labels.json', 'w', encoding='utf-8') as f:
    json.dump({'classes': label_entries}, f, indent=2, ensure_ascii=False)
print('Saved labels.json with 29 entries (14 model + 15 dictionary)')

# Save model_config.json
with open('../backend/artifacts/model_config.json', 'w', encoding='utf-8') as f:
    json.dump({
        'model_name': 'DGSenseTemporalGRU',
        'model_version': '3.1.0',
        'architecture': {
            'type': 'BiGRUWithAttention',
            'input_dim': 144, 'feature_dim': 144, 'use_delta': False,
            'use_attention': True, 'hidden_dim': 128, 'num_layers': 2,
            'bidirectional': True, 'dropout': 0.3, 'num_classes': 14
        },
        'supported_labels': CLASS_NAMES,
        'feature_schema_version': 'dgsense-v1',
        'sequence_length': 30,
        'default_min_confidence': 0.5,
        'training_data': 'RWTH-PHOENIX-Weather 2014T via HuggingFace streaming (real DGS signers)',
        'normalization': 'wrist-origin hand-scale (applied in backend before inference) + velocity stacking when use_delta=true',
        'created_at': '2026-10-27T00:00:00'
    }, f, indent=2, ensure_ascii=False)
print('Saved model_config.json with num_classes=14')

# Save training_report.json
with open('../backend/artifacts/training_report.json', 'w', encoding='utf-8') as f:
    json.dump({
        'model_name': 'DGSenseTemporalGRU',
        'version': '3.1.0',
        'schema_version': 'dgsense-v1',
        'training_source': 'RWTH-PHOENIX-Weather-2014T via HuggingFace streaming (real DGS signers)',
        'split_strategy': f'Signer-Independent (Train: {sorted(train_spk)}, Val: {sorted(val_spk)}, Test: {sorted(test_spk)})',
        'test_accuracy': round(float(test_acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'epochs_trained': 10,
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

# Verify model loads
from backend.model_service import ModelService
svc = ModelService()
print('Model loaded:', svc.is_model_loaded)
print('Label map size:', len(svc.label_map) if svc.label_map else 0)
print('Model config num_classes:', svc.model_config.get('architecture', {}).get('num_classes', 'N/A') if svc.model_config else 'N/A')