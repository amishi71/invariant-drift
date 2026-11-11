"""
build_l1_ordered_cache.py -- chronological feature cache for one CERN/anomaly_detection_cmsl1t config.

The dataset's own splits are random permutations, so this reads every shard of the config (all splits),
keeps the ordering columns, and sorts events by (run, orbit, bx).

    python3 scripts/build_l1_ordered_cache.py --config ZB_run396102 --out data/real_cache/zb396102_ordered.npz
    python3 scripts/build_l1_ordered_cache.py --config ZB_run396102 --max-row-groups 1 --smoke-rows 100000 --out /tmp/l1_smoke.npz

Output arrays: features (N,12) float64, pileup, label, run, lumi, event, bx, orbit, time, order.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, os.getcwd())
from src.feature_extraction_cmsl1t import extract_features_batch  # noqa: E402

REPO = "CERN/anomaly_detection_cmsl1t"
KEYS = ["run", "lumi", "event", "bx", "orbit", "time", "order"]


def col(table, name):
    return np.asarray(table.column(name).fill_null(0).to_numpy())


def runs_of_constant(x):
    return int(1 + np.count_nonzero(np.diff(x) != 0)) if len(x) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-row-groups", type=int, default=0, help="0 = all (smoke test: 1)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="keep only the first N rows of each row group")
    args = ap.parse_args()

    print("Resolving snapshot for %s ..." % args.config)
    local_dir = snapshot_download(REPO, repo_type="dataset",
                                  allow_patterns=["data/%s/*" % args.config])
    shards = sorted(glob.glob("%s/data/%s/*.parquet" % (local_dir, args.config)))
    if not shards:
        sys.exit("no parquet shards found for %s under %s" % (args.config, local_dir))
    print("Found %d shard(s)" % len(shards))

    feats_l, pile_l, lab_l = [], [], []
    keys_l = {k: [] for k in KEYS}
    n_rg, t0 = 0, time.time()
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for g in range(pf.num_row_groups):
            if args.max_row_groups and n_rg >= args.max_row_groups:
                break
            table = pf.read_row_group(g)
            if args.smoke_rows:
                table = table.slice(0, args.smoke_rows)
            f, p, l = extract_features_batch(table)
            if len(f) != table.num_rows:
                sys.exit("extract_features_batch dropped rows (%d of %d); cannot align ordering keys"
                         % (len(f), table.num_rows))
            feats_l.append(f); pile_l.append(p); lab_l.append(l)
            for k in KEYS:
                keys_l[k].append(col(table, k))
            n_rg += 1
            print("  %s rg %d: +%d rows (total %d, %.0fs)" % (
                os.path.basename(sp), g, len(f), sum(len(x) for x in feats_l), time.time() - t0), flush=True)
            del table
        if args.max_row_groups and n_rg >= args.max_row_groups:
            break

    F = np.concatenate(feats_l); P = np.concatenate(pile_l); L = np.concatenate(lab_l)
    K = {k: np.concatenate(v) for k, v in keys_l.items()}
    del feats_l, pile_l, lab_l, keys_l

    idx = np.lexsort((K["bx"], K["orbit"], K["run"]))   # primary key last: run, then orbit, then bx
    F, P, L = F[idx], P[idx], L[idx]
    K = {k: v[idx] for k, v in K.items()}

    print("\nIntegrity checks")
    print("  rows %d, features %s, NaN %d, inf %d" % (len(F), F.shape, int(np.isnan(F).sum()), int(np.isinf(F).sum())))
    print("  distinct runs: %s" % np.unique(K["run"]).tolist())
    print("  lumi range %d..%d, monotone: %s" % (K["lumi"].min(), K["lumi"].max(), bool(np.all(np.diff(K["lumi"].astype(np.int64)) >= 0))))
    print("  orbit monotone: %s" % bool(np.all(np.diff(K["orbit"].astype(np.float64)) >= 0)))
    print("  time monotone: %s" % bool(np.all(np.diff(K["time"].astype(np.float64)) >= 0)))
    pair = K["orbit"].astype(np.uint64) * np.uint64(4096) + K["bx"].astype(np.uint64)
    print("  duplicate (orbit,bx) pairs: %d" % (len(pair) - len(np.unique(pair))))
    pr = runs_of_constant(P)
    vals, cnt = np.unique(P, return_counts=True)
    exp = len(P) * (1.0 - float(((cnt / len(P)) ** 2).sum()))
    print("  pileup constant runs: %d (shuffled would give ~%.0f) -> %s" % (
        pr, exp, "TIME-ORDERED" if pr < 0.2 * exp else "still looks shuffled"))
    q = max(len(P) // 4, 1)
    print("  mean pileup by quarter: " + ", ".join("%.2f" % P[i * q:(i + 1) * q].mean() for i in range(4)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, features=F, pileup=P, label=L, **K)
    print("Wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
