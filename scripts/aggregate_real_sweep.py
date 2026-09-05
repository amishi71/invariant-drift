"""
scripts/aggregate_real_sweep.py

Aggregates real_data_sweep.py output across multiple VAE seeds into
mean +/- std, matching the reporting convention used for the synthetic
radiation-damage/masked-channel multi-seed results (see README's
"finding #7" and space-case-study docstring: report a mean+/-std range,
not a single-run figure, whenever more than one seed exists).

Usage:
    python3 scripts/aggregate_real_sweep.py results/real_seeds/*.json
"""
import glob
import json
import sys

import numpy as np


def load_all(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def agg(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "values": values}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": values}


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("results/real_seeds/*.json"))
    if not paths:
        print("No result files found. Pass paths explicitly or populate "
              "results/real_seeds/.")
        sys.exit(1)

    runs = load_all(paths)
    print(f"Aggregating {len(runs)} seed(s): {paths}\n")

    summary = {}

    # --- ARL check ---
    summary["real_arl_check"] = {
        "arl_mean": agg([r["real_arl_check"]["arl_mean"] for r in runs]),
        "false_alarm_rate": agg([r["real_arl_check"]["false_alarm_rate"] for r in runs]),
    }

    # --- Masked-channel: same label set across runs ---
    summary["real_masked_channel"] = {}
    for label in runs[0]["real_masked_channel"]:
        summary["real_masked_channel"][label] = {
            "miss_rate": agg([r["real_masked_channel"][label]["miss_rate"] for r in runs]),
            "latency_mean": agg([r["real_masked_channel"][label]["latency_mean"] for r in runs]),
        }

    # --- Radiation-damage ---
    summary["real_radiation_damage"] = {}
    for label in runs[0]["real_radiation_damage"]:
        summary["real_radiation_damage"][label] = {
            "miss_rate": agg([r["real_radiation_damage"][label]["miss_rate"] for r in runs]),
            "latency_mean": agg([r["real_radiation_damage"][label]["latency_mean"] for r in runs]),
        }

    def fmt(d, pct=False):
        if d["mean"] is None:
            return "N/A"
        m, s = (d["mean"] * 100, d["std"] * 100) if pct else (d["mean"], d["std"])
        unit = "%" if pct else ""
        return f"{m:.1f}{unit} +/- {s:.1f}{unit}  (n={len(d['values'])}, values={d['values']})"

    print("=== Real-data ARL check (no injection) ===")
    print(f"  ARL:        {fmt(summary['real_arl_check']['arl_mean'])}")
    print(f"  FA_rate:    {fmt(summary['real_arl_check']['false_alarm_rate'], pct=True)}")

    print("\n=== Masked-channel (real+injected) ===")
    for label, d in summary["real_masked_channel"].items():
        print(f"  {label}")
        print(f"    miss_rate:     {fmt(d['miss_rate'], pct=True)}")
        print(f"    latency_mean:  {fmt(d['latency_mean'])}")

    print("\n=== Radiation-damage (real+injected) ===")
    for label, d in summary["real_radiation_damage"].items():
        print(f"  {label}")
        print(f"    miss_rate:     {fmt(d['miss_rate'], pct=True)}")
        print(f"    latency_mean:  {fmt(d['latency_mean'])}")

    with open("results/real_data_sweep_aggregated.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nWritten to results/real_data_sweep_aggregated.json")


if __name__ == "__main__":
    main()
