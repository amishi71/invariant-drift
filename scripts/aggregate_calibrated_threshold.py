"""
scripts/aggregate_calibrated_threshold.py

Aggregates multiple seeds' calibrated_threshold_experiment.py output
(from calibrated_threshold_multiseed.py) into mean +/- std per target_far,
matching the reporting convention already used in
docs/REAL_DATA_VALIDATION.md's other multi-seed tables.

Usage:
    python3 scripts/aggregate_calibrated_threshold.py results/calibrated_multiseed/seed*.json
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

    print(f"Aggregating {len(paths)} seed file(s): {[p for p in paths]}\n")

    by_target = defaultdict(lambda: {"chosen_h": [], "bootstrap_far": [], "held_out_far": []})

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for row in data["results"]:
            t = row["target_far"]
            by_target[t]["chosen_h"].append(row["chosen_h"])
            by_target[t]["bootstrap_far"].append(row["bootstrap_measured_far"])
            by_target[t]["held_out_far"].append(row["held_out_false_alarm_rate"])

    print(f"{'target_far':>10} {'n_seeds':>8} {'chosen_h':>18} {'bootstrap_FAR':>18} {'held_out_FAR':>18} {'ratio':>8}")
    print("-" * 90)
    for target in sorted(by_target):
        vals = by_target[target]
        n = len(vals["chosen_h"])
        h_mean, h_std = np.mean(vals["chosen_h"]), np.std(vals["chosen_h"])
        boot_mean, boot_std = np.mean(vals["bootstrap_far"]), np.std(vals["bootstrap_far"])
        held_mean, held_std = np.mean(vals["held_out_far"]), np.std(vals["held_out_far"])
        ratio = held_mean / boot_mean if boot_mean > 0 else float("nan")
        print(f"{target:10.2f} {n:8d} "
              f"{h_mean:8.2f}+/-{h_std:<7.2f} "
              f"{boot_mean:8.3f}+/-{boot_std:<7.3f} "
              f"{held_mean:8.3f}+/-{held_std:<7.3f} "
              f"{ratio:8.2f}")

    print("\nNOTE: ratio = held_out_FAR / bootstrap_FAR. Ratio near 1.0 with a small")
    print("std means calibration is reliable and consistent across seeds at that")
    print("target. A ratio meaningfully above 1.0, or a large std, means seed=0's")
    print("single-seed reading (in docs/REAL_DATA_VALIDATION.md) was optimistic")
    print("and the writeup should report the multi-seed range instead.")


if __name__ == "__main__":
    main()
