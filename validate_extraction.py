"""
Cross-checks the vectorized extractor against a naive row-by-row extraction
on a small slice, mirroring the bit-identical validation pattern used for
the PyROOT cache rebuild (validate_pyroot_cache.py).

This version reads parquet directly via pyarrow (memory_map=False) instead
of going through datasets' load_dataset(), which memory-maps files and has
been observed to trigger posix_madvise errors (and a drive eject) when the
Hugging Face cache lives on an NTFS-formatted external volume on macOS.

Place this file at the repo root (alongside main.py, explore_cmsl1t.py).
Run from the repo root: python3 validate_extraction.py
"""
import glob

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from src.feature_extraction_cmsl1t import extract_features_batch

REPO = "CERN/anomaly_detection_cmsl1t"
N = 2000


def row_reference(row):
    def lead(lst, scale=1.0, offset=0.0):
        return (max(lst) - offset) * scale if len(lst) > 0 else 0.0

    def unwrap(lst, scale=1.0):
        return lst[0] * scale

    return np.array([
        len(row["jets_jetIEt"]),
        len(row["egammas_egIEt"]),
        len(row["muons_muonIEt"]),
        len(row["taus_tauIEt"]),
        lead(row["jets_jetIEt"], 0.5),
        lead(row["egammas_egIEt"], 0.5),
        lead(row["muons_muonIEt"], 0.5, offset=1.0),
        lead(row["taus_tauIEt"], 0.5),
        unwrap(row["HT_Et"], 0.5),
        unwrap(row["MET_Et"], 0.5),
        unwrap(row["MHT_Et"], 0.5),
        unwrap(row["ET_Et"], 0.5),
    ], dtype=np.float64)


if __name__ == "__main__":
    # Resolve (or reuse cached) local snapshot without going through
    # datasets' mmap-based loader at all.
    local_dir = snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=["data/ZB_run396102/train-*"],
    )

    parquet_files = sorted(glob.glob(f"{local_dir}/data/ZB_run396102/train-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No train parquet files found under {local_dir}/data/ZB_run396102/")

    print(f"Reading from: {parquet_files[0]}")
    table = pq.read_table(parquet_files[0], memory_map=False).slice(0, N)

    vec_features, vec_pileup, vec_labels = extract_features_batch(table)

    rows = table.to_pylist()
    ref_features = np.stack([row_reference(row) for row in rows])

    max_abs_diff = np.max(np.abs(vec_features - ref_features))
    print(f"Max abs diff (vectorized vs row-by-row), N={N}: {max_abs_diff}")
    assert max_abs_diff == 0.0, "MISMATCH: vectorized extraction diverges from reference"
    print("MATCH: vectorized extraction is bit-identical to row-by-row reference.")

    print("\nSample rows (vectorized):")
    print(vec_features[:3])
    print("Pileup sample:", vec_pileup[:3])
    print("Labels sample:", vec_labels[:3])
