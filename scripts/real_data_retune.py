"""
scripts/real_data_retune.py

Quick hyperparameter retune experiment: does adjusting CUSUM's (k, h) fix
the real-data false-alarm-rate gap found in the multi-seed validation
(README "Real-Data Substrate Validation" -- FA_rate=66.7%+/-19.7% real
vs ~17-33% synthetic at the default k=0.5,h=8.0)?

Detector hyperparameters don't change the underlying residuals -- only
how they're thresholded -- so this reuses ONE real burn-in/calibration
and ONE set of cached residual windows (stable + masked-channel +
radiation-damage) across the entire (k, h) grid. No VAE retraining per
grid point, so this is fast even though it's a real grid search.

Reports, for each (k, h):
  - ARL / false-alarm rate on stable (no-injection) real windows
  - masked-channel miss_rate (default two-sided config only -- the
    one-sided variant is a separate, already-documented finding)
  - radiation-damage miss_rate

Goal: find an h that brings real FA-rate down toward the synthetic
~17-33% range without destroying detection sensitivity. This is a
diagnostic sweep on a SINGLE seed -- confirm any promising candidate
across multiple VAE seeds (same pattern as real_data_sweep.py's --seed)
before citing a specific number in the paper.

Usage:
    python3 scripts/real_data_retune.py
    python3 scripts/real_data_retune.py --h-values 8,10,12,16,20,24,32 --k-values 0.3,0.5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import _cache_residual_trials
from src.detectors.cusum import CUSUMDetector
from src import evaluation
from src.real_pipeline import build_real_calibration, score_real_segment
from src.drift_sim.real_data_injection import (
    real_masked_channel_stream,
    real_radiation_damage_stream,
)


def stable_windows(model, scaler, calib, features, pileup, n_jet,
                    n_windows, window_size, start_base):
    """Precompute residual arrays for stable (no-injection) real windows --
    shared across every (k, h) combo since residuals don't depend on
    detector hyperparameters, only on the VAE/calibration (fixed here).
    """
    windows = []
    for w in range(n_windows):
        start = start_base + w * window_size
        _, residuals = score_real_segment(
            model, scaler, calib, features, pileup, n_jet, start, window_size,
        )
        windows.append(residuals)
    return windows


def arl_for_config(windows, factory, burn_in_residuals):
    run_lengths, n_false_alarms = [], 0
    for residuals in windows:
        detector = factory(burn_in_residuals)
        result = evaluation.run_detector_on_residuals(detector, residuals)
        if result.detected:
            n_false_alarms += 1
            run_lengths.append(result.detection_index + 1)
        else:
            run_lengths.append(len(residuals))
    return {
        "arl_mean": float(np.mean(run_lengths)),
        "false_alarm_rate": n_false_alarms / len(windows),
    }


def miss_rate_for_config(cache, factory, burn_in_residuals):
    n_missed, latencies = 0, []
    for residuals, onset in cache:
        detector = factory(burn_in_residuals)
        r = evaluation.run_detector_on_residuals(detector, residuals, true_onset_event=onset)
        if r.detected and r.latency is not None:
            latencies.append(r.latency)
        else:
            n_missed += 1
    return {
        "miss_rate": n_missed / len(cache),
        "latency_mean": float(np.mean(latencies)) if latencies else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000)
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--n-arl-windows", type=int, default=20,
                         help="More windows than real_data_sweep.py's 10 -- cheap "
                              "here since no VAE retraining happens per grid point.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-damage", type=float, default=3000.0)
    parser.add_argument("--k-values", type=str, default="0.5",
                         help="Comma-separated k values to grid over.")
    parser.add_argument("--h-values", type=str, default="8,10,12,16,20,24,32",
                         help="Comma-separated h values to grid over.")
    parser.add_argument("--out", type=str, default="results/real_data_retune.json")
    args = parser.parse_args()

    k_values = [float(x) for x in args.k_values.split(",")]
    h_values = [float(x) for x in args.h_values.split(",")]

    print(f"[1/3] Real burn-in: {args.n_burn_in} events, training proxy VAE "
          f"({args.vae_epochs} epochs)...")
    model, scaler, calib, burn_in_residuals, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False,
    )

    arl_start = args.n_burn_in
    arl_block = args.n_arl_windows * args.n_events
    masked_start = arl_start + arl_block
    masked_block = args.n_trials * args.n_events
    rad_start = masked_start + masked_block
    rad_block = args.n_trials * args.n_events

    pool_needed = rad_start + rad_block
    if pool_needed > len(features):
        print(f"WARNING: needs {pool_needed} events, pool has {len(features)}. "
              f"Reduce --n-arl-windows/--n-trials/--n-events.")

    print(f"\n[2/3] Precomputing shared residual windows/caches "
          f"({args.n_arl_windows} stable windows, {args.n_trials} masked-channel "
          f"trials, {args.n_trials} radiation-damage trials)...")
    windows = stable_windows(
        model, scaler, calib, features, pileup, n_jet,
        args.n_arl_windows, args.n_events, arl_start,
    )
    masked_cache = _cache_residual_trials(
        model, scaler, calib,
        lambda t: real_masked_channel_stream(
            features, pileup, n_jet,
            start=masked_start + t * args.n_events, n_events=args.n_events,
            changepoint_event=args.n_events // 2, drop_fraction=0.4,
        ),
        args.n_events, args.n_trials, args.seed,
    )
    rad_cache = _cache_residual_trials(
        model, scaler, calib,
        lambda t: real_radiation_damage_stream(
            features, pileup, n_jet,
            start=rad_start + t * args.n_events, n_events=args.n_events,
            tau_damage=args.tau_damage,
        ),
        args.n_events, args.n_trials, args.seed,
    )

    print(f"\n[3/3] Grid over k in {k_values}, h in {h_values} "
          f"({len(k_values) * len(h_values)} configs)...")
    print(f"{'k':>5} {'h':>6} {'ARL':>10} {'FA_rate':>9} "
          f"{'masked_miss':>12} {'rad_miss':>10}")

    results = []
    for k in k_values:
        for h in h_values:
            factory = lambda ref, k=k, h=h: CUSUMDetector(ref, k=k, h=h)
            arl_result = arl_for_config(windows, factory, burn_in_residuals)
            masked_result = miss_rate_for_config(masked_cache, factory, burn_in_residuals)
            rad_result = miss_rate_for_config(rad_cache, factory, burn_in_residuals)
            row = {
                "k": k, "h": h,
                "arl_mean": arl_result["arl_mean"],
                "false_alarm_rate": arl_result["false_alarm_rate"],
                "masked_channel_miss_rate": masked_result["miss_rate"],
                "radiation_damage_miss_rate": rad_result["miss_rate"],
            }
            results.append(row)
            print(f"{k:5.2f} {h:6.1f} {row['arl_mean']:10.1f} "
                  f"{row['false_alarm_rate']:9.2f} "
                  f"{row['masked_channel_miss_rate']:12.2f} "
                  f"{row['radiation_damage_miss_rate']:10.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"grid": results, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nDone. Results written to {args.out}")
    print(f"\nNOTE: single seed (seed={args.seed}) -- confirm any promising "
          f"(k, h) candidate across multiple seeds (same pattern as "
          f"real_data_sweep.py's --seed sweep) before citing it in the paper.")


if __name__ == "__main__":
    main()
