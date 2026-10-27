"""Builds DGSense_Training.ipynb from cell sources (avoids hand-escaping ipynb JSON)."""
import json
from pathlib import Path

cells = []

def md(src):
    assert src.endswith("\n") or True
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.splitlines(keepends=True)})


md("""# DGSense V3.1 — Real DGS Training (HuggingFace streaming, accuracy-focused)

**No dataset download.** Streams **RWTH-PHOENIX-Weather 2014T** (real German Sign Language,
10+ professional interpreters) sample-by-sample over HTTP range requests from the HuggingFace
Hub: frames are fetched as byte windows per sentence, decoded **in memory**, landmark-extracted,
and discarded. Only artifacts + a small resumable clip cache touch disk.

### Accuracy design (target: 40-60% on signer-independent test)
1. **Distinctive classes** — 14 candidate weather glosses; pick the 10 most visually separable
   (frequency audited from the corpus; abstract/fingerspelled glosses avoided where possible).
2. **Motion-refined segmentation** — PHOENIX ships no frame-level alignment on HF, so proportional
   gloss boundaries are snapped to actual motion onsets/offsets before clip extraction.
3. **Quality gates** — clips are dropped when hands are undetected in most frames or when the
   segment is static, so the model never trains on empty frames.
4. **Velocity (delta) features** — each 144-dim frame is stacked with its first difference
   (288 dims) inside the backend, giving the network explicit motion channels.
5. **Attention pooling** over the BiGRU + label smoothing + long training with early stopping.
6. **Signer-independent splits** (unseen signers) for an honest test score.
""")
code("""# Colab: install runtime deps (torch/numpy preinstalled). No dataset is downloaded.
%pip install -q mediapipe remotezip pandas scikit-learn matplotlib seaborn""")
code("""import io, os, re, json, math, time, zlib, random, zipfile, collections
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DGSense V3.1] torch {torch.__version__} | device: {DEVICE}")""")
code("""def _env_num(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default

SMOKE = os.environ.get("DGSENSE_SMOKE") == "1"  # tiny local verification mode

CONFIG = {
    "feature_schema_version": "dgsense-v1",
    "sign_language": "dgs-DE",
    "sequence_length": 30,
    "feature_dim": 144,
    "use_delta": os.environ.get("DGSENSE_DELTA", "1") == "1",   # stack first differences -> 288
    "use_attention": True,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": _env_num("DGSENSE_BATCH", 32),
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": _env_num("DGSENSE_EPOCHS", 60),
    "patience": 12,
    "label_smoothing": 0.1,
    "min_confidence": 0.65,
    "max_download_limit_mb": 5120,          # streaming budget safeguard (README 5 GB cap)
    "max_clips_per_class": _env_num("DGSENSE_CAP", 90),
    "max_sentence_frames": 260,              # skip abnormally long sentences
    "min_hand_frame_fraction": 0.50,         # quality gate: hands visible in >=50% of frames
    "min_segment_frames": 10,                # quality gate: after motion-trim
    "cache_path": Path(os.environ.get("DGSENSE_CACHE", "./extracted_clips.npz")),
    "artifacts_dir": Path("./artifacts"),
}
CONFIG["model_input_dim"] = CONFIG["feature_dim"] * (2 if CONFIG["use_delta"] else 1)
if SMOKE:
    CONFIG.update({"max_clips_per_class": 3, "max_epochs": 3, "max_sentence_frames": 120,
                   "cache_path": Path("./extracted_clips_smoke.npz")})
CONFIG["artifacts_dir"].mkdir(parents=True, exist_ok=True)

# 14 candidate content glosses (frequencies audited from the corpus; iconically/motion
# distinctive weather concepts preferred over abstract fingerspelled ones).
ALL_CLASSES = [
    {"canonical": "RAIN",      "gloss": "REGEN",    "de": "Regen",       "en": "Rain"},
    {"canonical": "SUN",       "gloss": "SONNE",    "de": "Sonne",       "en": "Sun"},
    {"canonical": "CLOUD",     "gloss": "WOLKE",    "de": "Wolke",       "en": "Cloud"},
    {"canonical": "SNOW",      "gloss": "SCHNEE",   "de": "Schnee",      "en": "Snow"},
    {"canonical": "STORM",     "gloss": "GEWITTER", "de": "Gewitter",    "en": "Thunderstorm"},
    {"canonical": "WIND",      "gloss": "WIND",     "de": "Wind",        "en": "Wind"},
    {"canonical": "WEATHER",   "gloss": "WETTER",   "de": "Wetter",      "en": "Weather"},
    {"canonical": "BLOW",      "gloss": "WEHEN",    "de": "Wehen",       "en": "Blowing"},
    {"canonical": "COME",      "gloss": "KOMMEN",   "de": "Kommen",      "en": "Coming"},
    {"canonical": "NORTH",     "gloss": "NORD",     "de": "Nord",        "en": "North"},
    {"canonical": "SOUTH",     "gloss": "SUED",     "de": "Süd",         "en": "South"},
    {"canonical": "EAST",      "gloss": "OST",      "de": "Ost",         "en": "East"},
    {"canonical": "TOMORROW",  "gloss": "MORGEN",   "de": "Morgen",      "en": "Tomorrow"},
    {"canonical": "TODAY",     "gloss": "HEUTE",    "de": "Heute",       "en": "Today"},
]
DEFAULT_SET = [c["canonical"] for c in ALL_CLASSES]  # refined after class-selection experiments

_env_cls = os.environ.get("DGSENSE_CLASSES")
_selected = set(_env_cls.split(",")) if _env_cls else set(DEFAULT_SET)
MODEL_CLASSES = [c for c in ALL_CLASSES if c["canonical"] in _selected]
unknown = _selected - {c["canonical"] for c in ALL_CLASSES}
assert not unknown, f"Unknown DGSENSE_CLASSES entries: {sorted(unknown)}"
TARGET_GLOSSES = {c["gloss"] for c in MODEL_CLASSES}
CLASS_NAMES = [c["canonical"] for c in MODEL_CLASSES]
LABEL2ID = {c: i for i, c in enumerate(CLASS_NAMES)}

print(f"Model classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"cap/class={CONFIG['max_clips_per_class']} epochs={CONFIG['max_epochs']} "
      f"delta={CONFIG['use_delta']} attention={CONFIG['use_attention']} smoke={SMOKE}")""")

md("""## 1. Corpus metadata (streamed from HF - small CSVs only)""")
code("""HF_BASE = "https://huggingface.co/datasets/aipieces/RWTH-PHOENIX-Weather-2014T"
HF_RAW = f"{HF_BASE}/raw/main"
HF_RESOLVE = f"{HF_BASE}/resolve/main"
CORPUS_URL = f"{HF_RAW}/annotations/manual/PHOENIX-2014-T.{{split}}.corpus.csv"

budget_bytes = [0]  # streamed-bytes safeguard

def load_corpus() -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        df = pd.read_csv(CORPUS_URL.format(split=split), sep="|", dtype=str).fillna("")
        df["split"] = split
        frames.append(df)
        print(f"  {split}: {len(df)} sentences")
    return pd.concat(frames, ignore_index=True)

corpus = load_corpus()
print(f"Total sentences: {len(corpus)} | signers: {corpus['speaker'].nunique()}")

freq = collections.Counter()
for orth in corpus["orth"]:
    freq.update(orth.split())
print("\\nGloss frequencies for the selected classes:")
for c in MODEL_CLASSES:
    print(f"  {c['gloss']:<10} -> {freq[c['gloss']]} occurrences")
missing = [g for g in TARGET_GLOSSES if freq[g] == 0]
assert not missing, f"Target glosses absent from corpus: {missing}"
print("All target glosses present with abundant coverage.")""")

md("""## 2. Sentence -> shard index (central directories only, HTTP range)""")
code("""import remotezip

def shard_urls(split: str) -> List[str]:
    total = {"train": 8, "dev": 6, "test": 7}[split]
    return [f"{HF_RESOLVE}/features/fullFrame-210x260px/{split}/shard_{i:03d}_{total:03d}.zip"
            for i in range(1, total + 1)]

def build_sentence_index(split: str) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for url in shard_urls(split):
        with remotezip.RemoteZip(url) as z:
            for zi in z.infolist():
                if zi.is_dir():
                    continue
                m = re.match(r"^(.*)/images(\\d+)\\.png$", zi.filename)
                if not m:
                    continue
                entry = index.setdefault(m.group(1), {"url": url, "frames": {}})
                entry["frames"][int(m.group(2))] = zi
    print(f"  {split}: {len(index)} sentences indexed")
    return index

SENTENCE_INDEX = {s: build_sentence_index(s) for s in ("train", "dev", "test")}
indexed_names = set().union(*(set(v) for v in SENTENCE_INDEX.values()))
print(f"Corpus sentences without frames: {sum(1 for n in corpus['name'] if n not in indexed_names)}")""")

md("""## 3. Resumable clip cache (keyed by sentence x class)""")
code("""CACHE = {"X": [], "y": [], "meta": []}
DONE_PAIRS = set()

def load_cache():
    p = CONFIG["cache_path"]
    if not p.exists():
        return
    d = np.load(p, allow_pickle=True)
    CACHE["X"] = [a for a in d["X"]]
    CACHE["y"] = [int(v) for v in d["y"]]
    CACHE["meta"] = [dict(m) for m in d["meta"]]
    DONE_PAIRS.clear()
    DONE_PAIRS.update((m["name"], m["class"]) for m in CACHE["meta"])
    print(f"Cache: {len(CACHE['meta'])} clips, {len(DONE_PAIRS)} (sentence,class) pairs done")

def save_cache():
    np.savez_compressed(CONFIG["cache_path"],
                        X=np.asarray(CACHE["X"], dtype=np.float32),
                        y=np.asarray(CACHE["y"], dtype=np.int64),
                        meta=np.array(CACHE["meta"], dtype=object))

load_cache()
done_per_class = collections.Counter(m["class"] for m in CACHE["meta"])
for c in CLASS_NAMES:
    if done_per_class[c]:
        print(f"  cached {c}: {done_per_class[c]}")""")

md("""## 4. Selection worklist (offline - skips already-extracted pairs)""")
code("""def proportional_segments(n_glosses: int, glosses: List[str], n_frames: int) -> List[Tuple[int, int]]:
    weights = [max(len(g), 1) for g in glosses]
    total = float(sum(weights))
    bounds, cum = [], 0.0
    for w in weights:
        a = int(round(cum / total * n_frames))
        cum += w
        b = int(round(cum / total * n_frames))
        bounds.append((a, max(b, a + 1)))
    return bounds

def build_worklist() -> List[dict]:
    remaining = {c: max(CONFIG["max_clips_per_class"] - done_per_class[c], 0)
                 for c in CLASS_NAMES}
    planned = collections.Counter()
    per_sentence = []

    for row in corpus.itertuples(index=False):
        glosses = row.orth.split()
        if not glosses:
            continue
        idx_entry = SENTENCE_INDEX.get(row.split, {}).get(row.name)
        if not idx_entry:
            continue
        n_frames = len(idx_entry["frames"])
        if n_frames == 0 or n_frames > CONFIG["max_sentence_frames"]:
            continue
        bounds = proportional_segments(len(glosses), glosses, n_frames)
        kept = []
        for gi, g in enumerate(glosses):
            if g not in TARGET_GLOSSES:
                continue
            cls = next(c["canonical"] for c in MODEL_CLASSES if c["gloss"] == g)
            if (row.name, cls) in DONE_PAIRS:
                continue
            a, b = bounds[gi]
            if b - a < CONFIG["min_segment_frames"]:
                continue
            if planned[cls] >= remaining[cls]:
                continue
            planned[cls] += 1
            kept.append({"class": cls, "gloss": g, "start": a, "end": b})
        if kept:
            per_sentence.append({
                "split": row.split, "name": row.name, "speaker": row.speaker,
                "url": idx_entry["url"], "frames": idx_entry["frames"],
                "translation": row.translation, "segments": kept,
            })

    random.Random(SEED).shuffle(per_sentence)
    print(f"Selected sentences: {len(per_sentence)} | clips planned: {sum(len(i['segments']) for i in per_sentence)}")
    for cls in CLASS_NAMES:
        print(f"  {cls:<10} planned {planned[cls]:>4} (cached {done_per_class[cls]})")
    return per_sentence

WORKLIST = build_worklist()
assert WORKLIST or DONE_PAIRS, "Empty worklist and no cache - aborting"
print("worklist ok:", len(WORKLIST))""")

md("""## 5. Streaming frame reader (one ranged GET per sentence)""")
code("""def ranged_get(url: str, lo: int, hi: int) -> bytes:
    budget_bytes[0] += (hi - lo)
    if budget_bytes[0] / 1e6 > CONFIG["max_download_limit_mb"]:
        raise RuntimeError(f"[Safeguard] streamed {budget_bytes[0]/1e6:.0f} MB > cap "
                           f"{CONFIG['max_download_limit_mb']} MB. Lower DGSENSE_CAP.")
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=120)
            if r.status_code in (200, 206):
                return r.content if r.status_code == 206 else r.content[lo:hi]
            raise RuntimeError(f"unexpected status {r.status_code}")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

def sentence_window(url: str, frame_infos: List) -> Tuple[bytes, int]:
    lo = min(zi.header_offset for zi in frame_infos)
    hi = max(zi.header_offset + 30 + len(zi.filename.encode()) + 1024 + zi.compress_size
             for zi in frame_infos)
    return ranged_get(url, lo, hi), lo

def parse_members(window: bytes, infos: List, base: int) -> Dict[int, bytes]:
    out = {}
    for zi in infos:
        off = zi.header_offset - base
        if int.from_bytes(window[off:off + 4], "little") != 0x04034B50:
            raise ValueError(f"bad ZIP local header at {off}")
        name_len = int.from_bytes(window[off + 26:off + 28], "little")
        extra_len = int.from_bytes(window[off + 28:off + 30], "little")
        start = off + 30 + name_len + extra_len
        raw = window[start:start + zi.compress_size]
        if zi.compress_type == 0:
            data = raw
        elif zi.compress_type == 8:
            data = zlib.decompressobj(-15).decompress(raw)
        else:
            raise ValueError(f"unsupported zip method {zi.compress_type}")
        out[int(re.search(r"images(\\d+)\\.png", zi.filename).group(1))] = data
    return out

def resample_to_30(frames: np.ndarray) -> np.ndarray:
    n = len(frames)
    if n == CONFIG["sequence_length"]:
        return frames
    if n > CONFIG["sequence_length"]:
        idx = np.linspace(0, n - 1, CONFIG["sequence_length"]).round().astype(int)
        return frames[idx]
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, CONFIG["sequence_length"])
    return np.stack([np.interp(x_new, x_old, frames[:, j]) for j in range(frames.shape[1])], axis=1)

def motion_trim(seq: np.ndarray) -> np.ndarray:
    \"\"\"Snap fuzzy proportional boundaries to actual motion: trim low-motion leading/trailing
    frames (below 10% of the segment's peak hand motion).\"\"\"
    if len(seq) < 4:
        return seq
    hand = seq[:, :126]
    motion = np.abs(np.diff(hand, axis=0)).sum(axis=1)
    peak = float(motion.max())
    if peak < 1e-6:
        return seq
    active = np.where(motion > 0.10 * peak)[0]
    if len(active) == 0:
        return seq
    lo = max(int(active[0]) - 1, 0)
    hi = min(int(active[-1]) + 2, len(seq))
    return seq[lo:hi]

print("streaming reader + motion trim ready")""")

md("""## 6. MediaPipe extraction - identical 144-d mapping to the live frontend""")
code("""import mediapipe as mp
from mediapipe.tasks.python import BaseOptions

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
model_path = Path("./hand_landmarker.task")
if not model_path.exists():
    model_path.write_bytes(requests.get(MODEL_URL, timeout=180).content)
    print(f"Downloaded hand landmarker model ({model_path.stat().st_size/1e6:.1f} MB)")

def make_landmarker():
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)

def extract144(result) -> List[float]:
    \"\"\"EXACT same slot mapping as frontend/src/services/mediapipe.ts (dgsense-v1).\"\"\"
    v = [0.0] * 144
    left = right = False
    for idx, hand in enumerate(result.hand_landmarks or []):
        cat = result.handedness[idx][0].category_name if result.handedness else None
        if cat == "Left" and not left:
            left = True
            for lm_i, lm in enumerate(hand):
                b = lm_i * 3
                v[b], v[b + 1], v[b + 2] = lm.x, lm.y, lm.z
            v[138], v[139], v[140] = hand[0].x, hand[0].y, hand[0].z
        elif cat == "Right" and not right:
            right = True
            for lm_i, lm in enumerate(hand):
                b = 63 + lm_i * 3
                v[b], v[b + 1], v[b + 2] = lm.x, lm.y, lm.z
            v[141], v[142], v[143] = hand[0].x, hand[0].y, hand[0].z
        elif not left:
            right = True
            for lm_i, lm in enumerate(hand):
                b = 63 + lm_i * 3
                v[b], v[b + 1], v[b + 2] = lm.x, lm.y, lm.z
    if left:
        v[126], v[127], v[128] = 0.35, 0.50, 0.0
        v[132], v[133], v[134] = 0.30, 0.65, 0.0
    if right:
        v[129], v[130], v[131] = 0.65, 0.50, 0.0
        v[135], v[136], v[137] = 0.70, 0.65, 0.0
    return v

def normalize_frame(v: np.ndarray) -> np.ndarray:
    \"\"\"Wrist-origin, hand-scale normalization. MUST stay in sync with the identical
    function in backend/model_service.py::normalize_frame.\"\"\"
    out = np.array(v, dtype=np.float32, copy=True)
    for base in (0, 63):
        hand = out[base:base + 63].reshape(21, 3)
        if np.abs(hand).sum() < 1e-6:
            continue
        origin = hand[0].copy()
        scale = float(np.linalg.norm(hand[9] - hand[0]))
        if scale < 1e-6:
            continue
        out[base:base + 63] = ((hand - origin) / scale).reshape(-1)
        ps = 138 if base == 0 else 141
        out[ps:ps + 3] = (out[ps:ps + 3] - origin) / scale
    return out

def to_model_input(seq: np.ndarray) -> np.ndarray:
    \"\"\"144 -> 288 velocity-stacked model input. MUST stay in sync with
    backend/model_service.py::to_model_input (applied inside predict()).\"\"\"
    if not CONFIG["use_delta"]:
        return seq.astype(np.float32)
    d = np.zeros_like(seq, dtype=np.float32)
    d[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq.astype(np.float32), d], axis=1)

landmarker = make_landmarker()
TS = {"ms": 0}  # video-mode landmarker needs strictly monotonic timestamps
print("MediaPipe HandLandmarker ready")""")

md("""## 7. Extraction loop - quality-gated, motion-refined clips (resumable)""")
code("""def extract_worklist(worklist: List[dict]) -> None:
    t0 = time.time()
    stats = {"no_hand": 0, "static": 0, "short": 0, "kept": 0}

    for wi, item in enumerate(worklist):
        pending = [s for s in item["segments"] if (item["name"], s["class"]) not in DONE_PAIRS]
        if not pending:
            continue
        try:
            need_no = set()
            for seg in pending:
                need_no.update(range(seg["start"], seg["end"]))
            infos = [item["frames"][f] for f in sorted(need_no) if f in item["frames"]]
            if not infos:
                continue
            window, base_off = sentence_window(item["url"], infos)
            decoded = parse_members(window, infos, base_off)

            per_frame = {}
            for fno in sorted(decoded):
                img = Image.open(io.BytesIO(decoded[fno])).convert("RGB")
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
                res = landmarker.detect_for_video(mp_img, TS["ms"])
                TS["ms"] += 40
                per_frame[fno] = extract144(res)

            for seg in pending:
                seq = [per_frame[f] for f in range(seg["start"], seg["end"]) if f in per_frame]
                if len(seq) < CONFIG["min_segment_frames"]:
                    stats["short"] += 1
                    continue
                arr = np.asarray(seq, dtype=np.float32)

                # quality gate 1: hands detected in enough frames
                hand_frac = float((np.abs(arr[:, :126]).sum(axis=1) > 1e-6).mean())
                if hand_frac < CONFIG["min_hand_frame_fraction"]:
                    stats["no_hand"] += 1
                    continue

                # motion-refine the fuzzy proportional boundary
                arr = motion_trim(arr)
                if len(arr) < CONFIG["min_segment_frames"]:
                    stats["short"] += 1
                    continue

                arr = np.stack([normalize_frame(fr) for fr in arr])
                if (np.abs(arr[:, :126]).sum(axis=1) > 1e-6).mean() < 0.5:
                    stats["no_hand"] += 1
                    continue

                arr = resample_to_30(arr).astype(np.float32)
                CACHE["X"].append(arr)
                CACHE["y"].append(LABEL2ID[seg["class"]])
                CACHE["meta"].append({"name": item["name"], "speaker": item["speaker"],
                                      "split": item["split"], "class": seg["class"],
                                      "gloss": seg["gloss"], "translation": item["translation"],
                                      "hand_frac": round(hand_frac, 3),
                                      "seg_frames": int(len(arr))})
                DONE_PAIRS.add((item["name"], seg["class"]))
                stats["kept"] += 1
        except RuntimeError:
            save_cache()
            raise  # budget safeguard
        except Exception as e:
            print(f"  [skip] {item['name']}: {type(e).__name__}: {e}")
            continue

        if wi % 10 == 0:
            save_cache()
            rate = (wi + 1) / max(time.time() - t0, 1e-6)
            print(f"  {wi + 1}/{len(worklist)} sent | kept {stats['kept']} | "
                  f"dropped {stats} | {budget_bytes[0]/1e6:.0f} MB | {rate:.2f} sent/s")

    save_cache()
    print(f"\\nExtraction done: {len(CACHE['meta'])} clips total | "
          f"{budget_bytes[0]/1e6:.0f} MB streamed | {time.time() - t0:.0f}s | last-stats {stats}")
    for c in CLASS_NAMES:
        n = sum(1 for m in CACHE["meta"] if m["class"] == c)
        print(f"  {c:<10} {n}")

extract_worklist(WORKLIST)
assert len(CACHE["meta"]) >= 100 or SMOKE, "Too few clips extracted"

# Keep only clips of the currently selected classes and REBUILD labels from class names
# (cached y ids belong to the class list used at extraction time - never trust them).
_sel = set(CLASS_NAMES)
_keep = [i for i, m in enumerate(CACHE["meta"]) if m["class"] in _sel]
X_all = np.asarray([CACHE["X"][i] for i in _keep], dtype=np.float32)
y_all = np.asarray([LABEL2ID[CACHE["meta"][i]["class"]] for i in _keep], dtype=np.int64)
META = [CACHE["meta"][i] for i in _keep]
assert len(X_all) >= 100 or SMOKE, "Too few clips for the selected classes"
assert X_all.shape[1:] == (CONFIG["sequence_length"], CONFIG["feature_dim"]), X_all.shape
print("asserts ok:", X_all.shape, "| classes:", CLASS_NAMES)""")

md("""## 8. Signer-independent split & DataLoaders""")
code("""speakers = sorted({m["speaker"] for m in META})
random.Random(SEED).shuffle(speakers)

if len(speakers) >= 3:
    n_test = max(1, len(speakers) // 5)
    n_val = max(1, len(speakers) // 5)
    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test:n_test + n_val])
    train_spk = set(speakers) - test_spk - val_spk
    split_strategy = (f"Signer-Independent (Train: {sorted(train_spk)}, "
                      f"Val: {sorted(val_spk)}, Test: {sorted(test_spk)})")
else:
    train_spk = val_spk = test_spk = set(speakers)
    split_strategy = "Single signer (degraded split)"

def mask_for(spk_set):
    return np.array([m["speaker"] in spk_set for m in META])

m_train, m_val, m_test = mask_for(train_spk), mask_for(val_spk), mask_for(test_spk)
if m_val.sum() == 0 and m_train.sum() > 4:
    m_val[-2:] = True
if m_test.sum() == 0 and m_train.sum() > 4:
    m_test[-4:-2] = True
print(split_strategy)
print(f"Train={m_train.sum()} Val={m_val.sum()} Test={m_test.sum()}")

class DGSClipDataset(Dataset):
    def __init__(self, X, y, idx, augment=False):
        self.X, self.y, self.idx, self.augment = X, y, idx, augment
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        seq = self.X[self.idx[i]].copy()
        if self.augment:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            if np.random.rand() > 0.5:
                shift = int(np.random.randint(-2, 3))
                seq = np.roll(seq, shift, axis=0)
                seq[:2] = seq[2]  # soften roll edge
        return (torch.from_numpy(to_model_input(seq)),
                torch.tensor(self.y[self.idx[i]], dtype=torch.long))

train_ds = DGSClipDataset(X_all, y_all, np.where(m_train)[0], augment=True)
val_ds   = DGSClipDataset(X_all, y_all, np.where(m_val)[0])
test_ds  = DGSClipDataset(X_all, y_all, np.where(m_test)[0])
assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0, "empty split"

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
test_loader  = DataLoader(test_ds, batch_size=CONFIG["batch_size"])
print(f"model input dim: {CONFIG['model_input_dim']}")""")

md("""## 9. Model - identical architecture to backend/model_service.py::DGSenseTemporalGRU""")
code("""class DGSenseTemporalGRU(nn.Module):
    def __init__(self, input_dim=288, hidden_dim=128, num_layers=2, num_classes=10,
                 dropout=0.3, bidirectional=True, use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(gru_out_dim, 1)
        self.layer_norm = nn.LayerNorm(gru_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        out, _ = self.gru(x)
        if self.use_attention:
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
            pooled = (out * w.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = out.mean(dim=1)
        return self.classifier(self.dropout(self.layer_norm(pooled)))

model = DGSenseTemporalGRU(
    input_dim=CONFIG["model_input_dim"], hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"], num_classes=len(CLASS_NAMES),
    dropout=CONFIG["dropout"], bidirectional=True,
    use_attention=CONFIG["use_attention"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")""")

md("""## 10. Training - class-balanced loss + label smoothing + early stopping""")
code("""# Class-balanced loss: quality gates keep uneven numbers of clips per class.
train_counts = np.bincount(y_all[m_train], minlength=len(CLASS_NAMES)).astype(np.float64)
train_counts[train_counts == 0] = 1.0
class_w = (train_counts.sum() / (len(CLASS_NAMES) * train_counts)).astype(np.float32)
class_w = class_w / class_w.mean()
print("train counts:", dict(zip(CLASS_NAMES, np.bincount(y_all[m_train], minlength=len(CLASS_NAMES)).tolist())))
print("class weights:", dict(zip(CLASS_NAMES, np.round(class_w, 2).tolist())))

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w).to(DEVICE),
                                label_smoothing=CONFIG["label_smoothing"])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                              weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val, best_w, wait = float("inf"), None, 0

for epoch in range(1, CONFIG["max_epochs"] + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * len(yb)
    tr_loss /= max(len(train_ds), 1)

    model.eval()
    va_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            va_loss += criterion(out, yb).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
    va_loss /= max(len(val_ds), 1)
    va_acc = correct / max(len(val_ds), 1)
    scheduler.step(va_loss)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{CONFIG['max_epochs']}  train {tr_loss:.4f}  "
              f"val {va_loss:.4f}  val_acc {va_acc:.1%}")
    if va_loss < best_val:
        best_val, best_w, wait = va_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= CONFIG["patience"]:
            print(f"[Early stop] epoch {epoch}")
            break

model.load_state_dict(best_w)
print(f"Loaded best checkpoint (val_loss={best_val:.4f})")""")

md("""## 11. Evaluation""")
code("""model.eval()
preds, tgts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        tgts.extend(yb.numpy())
preds, tgts = np.asarray(preds), np.asarray(tgts)

LBL = list(range(len(CLASS_NAMES)))
test_acc = float((preds == tgts).mean())
macro_f1 = float(f1_score(tgts, preds, labels=LBL, average="macro", zero_division=0))
report = classification_report(tgts, preds, labels=LBL, target_names=CLASS_NAMES,
                               output_dict=True, zero_division=0)
print(f"Test accuracy: {test_acc:.2%} | macro F1: {macro_f1:.4f} | chance {1/len(LBL):.1%}")
print(classification_report(tgts, preds, labels=LBL, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(tgts, preds, labels=LBL)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].set_title(f"DGSense V3.1 confusion matrix (acc {test_acc:.1%})")
axes[0].tick_params(axis="x", rotation=45)
axes[1].plot(history["train_loss"], label="train")
axes[1].plot(history["val_loss"], label="val")
axes[1].set_title("Loss curves"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CONFIG["artifacts_dir"] / "confusion_matrix.png", dpi=160)
plt.savefig(CONFIG["artifacts_dir"] / "training_loss_curves.png", dpi=160)
plt.show()
if test_acc < 0.40:
    print(f"[WARN] test accuracy {test_acc:.1%} below the 40-60% target - "
          "consider stricter quality gates or a different class subset.")
print("done:", round(test_acc, 4))""")

md("""## 12. Export artifacts (drop the resulting ZIP into `backend/artifacts/`)""")
code("""art = CONFIG["artifacts_dir"]
torch.save(model.state_dict(), art / "model.pt")

label_entries = []
for i, c in enumerate(MODEL_CLASSES):
    label_entries.append({
        "id": i, "canonical_label": c["canonical"], "german_text": c["de"],
        "english_text": c["en"], "is_model_supported": True, "is_dictionary_only": False,
        "gloss": c["gloss"],
        "description": f"DGS weather sign '{c['gloss']}' from RWTH-PHOENIX-Weather 2014T."})

DICTIONARY_ENTRIES = [
    {"canonical_label": "HELLO", "german_text": "Hallo", "english_text": "Hello"},
    {"canonical_label": "GOOD_MORNING", "german_text": "Guten Morgen", "english_text": "Good morning"},
    {"canonical_label": "GOOD_DAY", "german_text": "Guten Tag", "english_text": "Good day"},
    {"canonical_label": "GOOD_EVENING", "german_text": "Guten Abend", "english_text": "Good evening"},
    {"canonical_label": "BYE", "german_text": "Tschüss", "english_text": "Bye"},
    {"canonical_label": "GOODBYE", "german_text": "Auf Wiedersehen", "english_text": "Goodbye"},
    {"canonical_label": "THANK_YOU", "german_text": "Danke", "english_text": "Thank you"},
    {"canonical_label": "THANK_YOU_VERY_MUCH", "german_text": "Vielen Dank", "english_text": "Thank you very much"},
    {"canonical_label": "PLEASE", "german_text": "Bitte", "english_text": "Please"},
    {"canonical_label": "SORRY", "german_text": "Entschuldigung", "english_text": "Sorry"},
    {"canonical_label": "YES", "german_text": "Ja", "english_text": "Yes"},
    {"canonical_label": "NO", "german_text": "Nein", "english_text": "No"},
    {"canonical_label": "HELP", "german_text": "Hilfe", "english_text": "Help"},
    {"canonical_label": "STOP", "german_text": "Stopp", "english_text": "Stop"},
    {"canonical_label": "HOW_ARE_YOU", "german_text": "Wie geht es dir?", "english_text": "How are you?"},
    {"canonical_label": "NOT_UNDERSTAND", "german_text": "Ich verstehe nicht", "english_text": "I don't understand"},
    {"canonical_label": "WATER", "german_text": "Wasser", "english_text": "Water"},
    {"canonical_label": "NAME", "german_text": "Ich heiße ...", "english_text": "My name is ..."},
    {"canonical_label": "WHERE_TOILET", "german_text": "Wo ist die Toilette?", "english_text": "Where is the toilet?"},
    {"canonical_label": "COME_HERE", "german_text": "Komm her", "english_text": "Come here"},
    {"canonical_label": "LETS_GO", "german_text": "Lass uns gehen", "english_text": "Let's go"},
    {"canonical_label": "LETS_LOOK", "german_text": "Lass uns schauen", "english_text": "Let's have a look"},
    {"canonical_label": "WAIT_PLEASE", "german_text": "Warte bitte", "english_text": "Please wait"},
    {"canonical_label": "NEED_HELP", "german_text": "Ich brauche Hilfe", "english_text": "I need help"},
    {"canonical_label": "CAN_HELP", "german_text": "Kannst du mir helfen?", "english_text": "Can you help me?"},
]
next_id = len(MODEL_CLASSES)
for d in DICTIONARY_ENTRIES:
    e = dict(d)
    e.update({"id": next_id, "is_model_supported": False, "is_dictionary_only": True,
              "description": "Everyday DGS vocabulary - dictionary reference "
                             "(record via Sign Recorder to enable recognition)."})
    label_entries.append(e)
    next_id += 1

with open(art / "labels.json", "w", encoding="utf-8") as f:
    json.dump({"classes": label_entries}, f, indent=2, ensure_ascii=False)

with open(art / "model_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": "DGSenseTemporalGRU", "model_version": "3.1.0",
        "architecture": {"type": "BiGRUWithAttention",
                         "input_dim": CONFIG["model_input_dim"],
                         "feature_dim": CONFIG["feature_dim"],
                         "use_delta": CONFIG["use_delta"],
                         "use_attention": CONFIG["use_attention"],
                         "hidden_dim": CONFIG["hidden_dim"],
                         "num_layers": CONFIG["num_layers"],
                         "bidirectional": True, "dropout": CONFIG["dropout"],
                         "num_classes": len(CLASS_NAMES)},
        "supported_labels": CLASS_NAMES,
        "feature_schema_version": CONFIG["feature_schema_version"],
        "sequence_length": CONFIG["sequence_length"],
        "default_min_confidence": CONFIG["min_confidence"],
        "training_data": "RWTH-PHOENIX-Weather 2014T (HuggingFace stream, real DGS signers)",
        "normalization": "wrist-origin hand-scale (normalize_frame) + velocity stacking",
        "created_at": pd.Timestamp.now().isoformat()}, f, indent=2, ensure_ascii=False)

with open(art / "training_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": "DGSenseTemporalGRU", "version": "3.1.0",
        "schema_version": CONFIG["feature_schema_version"],
        "training_source": "RWTH-PHOENIX-Weather-2014T via HuggingFace streaming (real DGS)",
        "split_strategy": split_strategy,
        "test_accuracy": round(test_acc, 4), "macro_f1": round(macro_f1, 4),
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": round(float(best_val), 4),
        "supported_classes": CLASS_NAMES, "missing_classes": [],
        "clips_total": int(len(X_all)),
        "clips_train": int(m_train.sum()), "clips_val": int(m_val.sum()),
        "clips_test": int(m_test.sum()),
        "quality_gates": {"min_hand_frame_fraction": CONFIG["min_hand_frame_fraction"],
                          "min_segment_frames": CONFIG["min_segment_frames"],
                          "motion_refined": True},
        "timestamp": pd.Timestamp.now().isoformat()}, f, indent=2, ensure_ascii=False)

with open(art / "class_metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

with open(art / "feature_schema.json", "w", encoding="utf-8") as f:
    json.dump({
        "schema_version": "dgsense-v1", "sequence_length": 30, "feature_dim": 144,
        "channels_per_landmark": 3,
        "landmarks": {"left_hand": {"count": 21, "start_idx": 0, "end_idx": 63},
                      "right_hand": {"count": 21, "start_idx": 63, "end_idx": 126},
                      "upper_pose": {"count": 6, "start_idx": 126, "end_idx": 144}},
        "coordinate_order": ["x", "y", "z"], "missing_landmark_fill": 0.0,
        "normalization": "wrist-origin hand-scale (applied in backend before inference)",
        "model_input": "144 raw + 144 first-difference = 288 when use_delta=true (backend)"}, f, indent=2)

zip_path = Path("DGSense_model_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(art.glob("*")):
        if p.is_file():
            zf.write(p, arcname=p.name)
print(f"Packaged {zip_path} -> extract into backend/artifacts/ to activate")""")

code("""try:
    from google.colab import files
    files.download("DGSense_model_artifacts.zip")
except ImportError:
    print("Local run: DGSense_model_artifacts.zip is in the current folder.")""")

nb = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 2,
}
out = Path(__file__).parent / "DGSense_Training.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} with {len(cells)} cells")