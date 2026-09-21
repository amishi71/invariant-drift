"""
scripts/aggregate_matched_far_baseline.py

Aggregates multiple seeds' matched_far_baseline_comparison.py output into
mean +/- std per detector, for a defensible multi-seed headline number.

Usage:
    python3 scripts/aggregate_matched_far_baseline.py results/matched_far_multiseed/seed*.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Per-seed JSON files (glob expanded by shell).")
    args = parser.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No files matched -- check the glob pattern.")

    print(f"Aggregating {len(paths)} seed file(s)\n")

    by_detector = defaultdict(lambda: {
        "achieved_far": [], "miss_rate": [], "latency_mean": [], "chosen_param": [],
    })

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for name, row in data["results"].items():
            by_detector[name]["achieved_far"].append(row["achieved_far"])
            by_detector[name]["miss_rate"].append(row["miss_rate"])
            if row["latency_mean"] is not None:
                by_detector[name]["latency_mean"].append(row["latency_mean"])
            by_detector[name]["chosen_param"].append(row["chosen_param"])

    print(f"{'detector':<14} {'n_seeds':>8} {'achieved_FAR':>16} {'miss_rate':>16} {'latency_mean':>18}")
    print("-" * 78)
    # Fixed, sensible order rather than dict/glob order
    for name in ["CUSUM", "Page-Hinkley", "ADWIN", "KSWIN"]:
        if name not in by_detector:
            continue
        vals = by_detector[name]
        n = len(vals["achieved_far"])
        far_mean, far_std = np.mean(vals["achieved_far"]), np.std(vals["achieved_far"])
        miss_mean, miss_std = np.mean(vals["miss_rate"]), np.std(vals["miss_rate"])
        if vals["latency_mean"]:
            lat_mean, lat_std = np.mean(vals["latency_mean"]), np.std(vals["latency_mean"])
            lat_str = f"{lat_mean:8.1f}+/-{lat_std:<7.1f}"
        else:
            lat_str = "no successful detections"
        print(f"{name:<14} {n:8d} "
              f"{far_mean:8.3f}+/-{far_std:<7.3f} "
              f"{miss_mean:8.3f}+/-{miss_std:<7.3f} "
              f"{lat_str:>18}")

    print("\nNOTE: all detectors should show achieved_FAR close to the same value")
    print("(the shared target) with a small std -- that's what makes latency and")
    print("miss_rate comparable across detectors. A detector with a much higher")
    print("std on achieved_FAR across seeds means its threshold search is unstable")
    print("and its latency/miss_rate numbers should be treated with more caution.")


if __name__ == "__main__":
    main()
