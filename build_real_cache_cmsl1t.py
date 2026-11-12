"""
Full-scale cache builder for the CMS L1T anomaly-detection dataset
(CERN/anomaly_detection_cmsl1t), producing compact .npz feature caches
analogous to the old data/real_cache/jetht_features.npz.

Reads parquet shards directly via pyarrow (memory_map=False) rather than
through datasets.load_dataset(), to avoid the mmap-related instability
observed earlier (posix_madvise error / drive eject) when the HF cache
lives on an NTFS-formatted external volume. Processes shard-by-shard and
concatenates only the small extracted feature arrays, not raw tables, to
keep peak memory low even for the 6.2M-row ZB_run396102 train split.

Place this file at the repo root (alongside main.py, validate_extraction.py).
Run from the repo root: python3 build_real_cache_cmsl1t.py

Output caches (fixed-length feature vectors, matching validate_extraction.py's
schema):
  data/real_cache/zb396102_train.npz   (ZB_run396102, train split)
  data/real_cache/zb396102_valid.npz   (ZB_run396102, validation split)
  data/real_cache/zb398183_train.npz   (ZB_run398183, train split -- second
                                         real run, for cross-run generalization)
  data/real_cache/<signal>_valid.npz   (one or more real BSM signal configs,
                                         for real anomaly injection material)

Each .npz contains:
  features: (N, 12) float64  -- see src/feature_extraction_cmsl1t.py for columns
  pileup:   (N,) float32     -- nPV_True
  label:    (N,) int16       -- 0 = zero-bias, negative = simulated background,
                                 positive = signal
"""
import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from src.feature_extraction_cmsl1t import extract_features_batch

REPO = "CERN/anomaly_detection_cmsl1t"
OUT_DIR = "data/real_cache"


def build_cache(config: str, split: str, out_name: str, row_group_batch: bool = True):
    """
    Downloads (or reuses cached) parquet shards for one HF config/split,
    extracts features shard-by-shard, and saves one concatenated .npz.
    """
    pattern = f"data/{config}/{split}-*"
    print(f"\n=== {config} / {split} ===")
    print(f"Resolving snapshot (pattern: {pattern}) ...")
    local_dir = snapshot_download(REPO, repo_type="dataset", allow_patterns=[pattern])

    shard_files = sorted(glob.glob(f"{local_dir}/data/{config}/{split}-*.parquet"))
    if not shard_files:
        print(f"  No shards found for {config}/{split} -- skipping.")
        return

    print(f"  Found {len(shard_files)} shard(s).")

    feat_chunks, pileup_chunks, label_chunks = [], [], []
    total_rows = 0
    t0 = time.time()

    for i, shard_path in enumerate(shard_files):
        pf = pq.ParquetFile(shard_path)
        if row_group_batch:
            # Process one row-group at a time to bound peak memory on large shards.
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx)
                feats, pileup, labels = extract_features_batch(table)
                feat_chunks.append(feats)
                pileup_chunks.append(pileup)
                label_chunks.append(labels)
                total_rows += len(feats)
        else:
            table = pq.read_table(shard_path, memory_map=False)
            feats, pileup, labels = extract_features_batch(table)
            feat_chunks.append(feats)
            pileup_chunks.append(pileup)
            label_chunks.append(labels)
            total_rows += len(feats)

        elapsed = time.time() - t0
        print(f"  shard {i+1}/{len(shard_files)} done "
              f"({os.path.basename(shard_path)}), "
              f"running total rows={total_rows}, elapsed={elapsed:.1f}s")

    features = np.concatenate(feat_chunks, axis=0)
    pileup = np.concatenate(pileup_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    np.savez(out_path, features=features, pileup=pileup, label=labels)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved {out_path}: {features.shape[0]} rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    # --- Real zero-bias background, first run (primary reference/burn-in) ---
    build_cache("ZB_run396102", "train", "zb396102_train.npz")
    build_cache("ZB_run396102", "valid", "zb396102_valid.npz")

    # --- Real zero-bias background, second run (cross-run generalization) ---
    build_cache("ZB_run398183", "train", "zb398183_train.npz")

    # --- A couple of real BSM signal samples, for real anomaly injection ---
    build_cache("VBFHto2Tau_Par-MH-125", "valid", "vbfhto2tau_valid.npz")
    build_cache("SUSYGluGluToBBHToBB_Par-M-350", "valid", "susy_m350_valid.npz")

    print("\nAll requested caches built.")
