"""
scripts/calibrated_threshold_experiment.py

Compares the existing manual (k, h) grid search in real_data_retune.py
against CUSUMDetector.calibrate_h()'s empirically-calibrated threshold,
on the SAME real burn-in and the SAME cached stable/masked-channel/
radiation-damage windows -- so any difference is attributable to the
calibration method, not to different data.

This directly answers the open question in docs/REAL_DATA_VALIDATION.md:
does calibrating h against the empirical (heavy-tailed) real residual
distribution close the false-alarm-rate gap that manual (k, h) retuning
couldn't fully close?

Usage:
    python3 scripts/calibrated_threshold_experiment.py
    python3 scripts/calibrated_threshold_experiment.py --target-far 0.02 0.05 0.10
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
from src.real_pipeline import build_real_calibration
from src.drift_sim.real_data_injection import (
    real_masked_channel_stream,
    real_radiation_damage_stream,
)

# Reuse the exact same window/cache builders real_data_retune.py already
# has -- import them directly rather than duplicating the logic.
from scripts.real_data_retune import (
    stable_windows,
    arl_for_config,
    miss_rate_for_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000)
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--n-arl-windows", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-damage", type=float, default=3000.0)
    parser.add_argument("--k", type=float, default=0.5,
                         help="k held fixed while h is calibrated -- matches "
                              "real_data_retune.py's default k grid.")
    parser.add_argument("--target-far", type=float, nargs="+", default=[0.02, 0.05, 0.10],
                         help="One or more target false-alarm rates to calibrate h against.")
    parser.add_argument("--h-step", type=float, default=1.0,
                         help="Granularity of calibrate_h's internal h search grid. "
                              "Lower (e.g. 0.25) resolves cases where the coarse "
                              "default (1.0) makes multiple targets collapse onto "
                              "the same chosen h.")
    parser.add_argument("--block-size", type=int, default=1,
                         help="calibrate_h's bootstrap block size. 1 = i.i.d. "
                              "resampling (destroys any real autocorrelation/"
                              "clustering in burn-in residuals). >1 resamples "
                              "contiguous chunks instead, preserving local "
                              "clustering if it exists.")
    parser.add_argument("--out", type=str, default="results/calibrated_threshold_experiment.json")
    args = parser.parse_args()

    print(f"[1/3] Real burn-in: {args.n_burn_in} events, training proxy VAE...")
    model, scaler, calib, burn_in_residuals, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False,
    )

    arl_start = args.n_burn_in
    MAX_ARL_WINDOWS_RESERVED = 200  # >= any --n-arl-windows you'll ever pass
    arl_block = MAX_ARL_WINDOWS_RESERVED * args.n_events
    masked_start = arl_start + arl_block
    masked_block = args.n_trials * args.n_events
    rad_start = masked_start + masked_block

    print(f"\n[2/3] Precomputing shared residual windows/caches "
          f"(identical to real_data_retune.py's, for apples-to-apples comparison)...")
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

    print(f"\n[3/3] Calibrating h for target FAR in {args.target_far} "
          f"(k={args.k} fixed)...\n")
    print(f"{'target_far':>11} {'chosen_h':>9} {'measured_FAR':>13} "
          f"{'masked_miss':>12} {'rad_miss':>10}")

    results = []
    for target_far in args.target_far:
        detector, measured_far = CUSUMDetector.calibrate_h(
            burn_in_residuals, k=args.k, target_far=target_far,
            window_size=args.n_events, n_bootstrap=200, seed=args.seed,
            h_grid=np.arange(2.0, 40.0, args.h_step),
            block_size=args.block_size,
        )
        factory = lambda ref, h=detector.h, k=args.k: CUSUMDetector(ref, k=k, h=h)

        arl_result = arl_for_config(windows, factory, burn_in_residuals)
        masked_result = miss_rate_for_config(masked_cache, factory, burn_in_residuals)
        rad_result = miss_rate_for_config(rad_cache, factory, burn_in_residuals)

        row = {
            "target_far": target_far,
            "k": args.k,
            "chosen_h": detector.h,
            "bootstrap_measured_far": measured_far,
            "held_out_arl_mean": arl_result["arl_mean"],
            "held_out_false_alarm_rate": arl_result["false_alarm_rate"],
            "masked_channel_miss_rate": masked_result["miss_rate"],
            "radiation_damage_miss_rate": rad_result["miss_rate"],
        }
        results.append(row)
        print(f"{target_far:11.2f} {detector.h:9.2f} {measured_far:13.3f} "
              f"{masked_result['miss_rate']:12.2f} {rad_result['miss_rate']:10.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)

    print(f"\nDone. Results written to {args.out}")
    print(f"\nCompare 'held_out_false_alarm_rate' here against the manual grid's")
    print(f"'false_alarm_rate' column in results/real_data_retune.json at a similar")
    print(f"h -- if calibrate_h's held-out FAR tracks target_far more closely than")
    print(f"the manual grid's FAR tracked the Gaussian-theory expectation for that")
    print(f"h, that's the evidence the empirical-calibration approach is doing its")
    print(f"job. As with real_data_retune.py: this is a single-seed diagnostic --")
    print(f"confirm across multiple seeds before citing a specific number.")


if __name__ == "__main__":
    main()
