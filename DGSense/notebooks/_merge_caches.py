"""Merge extraction caches with (name, class) dedup.

Job A writes extracted_clips.npz (all 14 classes), job B writes
extracted_clips_b.npz (7-class shard, overlaps A). Duplicated clips are
identified by (name, class) - the same key DONE_PAIRS uses during extraction.

Usage: python _merge_caches.py
Result: extracted_clips.npz (deduped, primary), backup _pre_merge_backup.npz
"""
import shutil
import numpy as np

PRIMARY = "extracted_clips.npz"
SECONDARY = "extracted_clips_b.npz"

a = np.load(PRIMARY, allow_pickle=True)
b = np.load(SECONDARY, allow_pickle=True)

aX, ay, ameta = list(a["X"]), list(a["y"]), [dict(m) for m in a["meta"]]
bX, by, bmeta = list(b["X"]), list(b["y"]), [dict(m) for m in b["meta"]]

seen = set()
dup = 0
for i, m in enumerate(bmeta):
    key = (m["name"], m["class"])
    if key in seen:
        dup += 1
        continue
    # also skip if already present from A
    if any((m["name"], m["class"]) == (am["name"], am["class"]) for am in ameta):
        dup += 1
        continue
    seen.add(key)
    aX.append(bX[i])
    ay.append(by[i])
    ameta.append(m)

shutil.copyfile(PRIMARY, "_pre_merge_backup.npz")
np.savez_compressed(PRIMARY, X=np.asarray(aX, dtype=np.float32),
                    y=np.asarray(ay, dtype=np.int64),
                    meta=np.asarray(ameta, dtype=object))

import collections
counts = collections.Counter(m["class"] for m in ameta)
print(f"merged: A={len(bmeta) and ''}{len(a['y'])} + B={len(bmeta)} -> "
      f"{len(ameta)} unique (dropped {dup} dups)")
print(dict(sorted(counts.items())))
