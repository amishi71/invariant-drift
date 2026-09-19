"""
scripts/validation_split_threshold_experiment.py

The 6-seed calibrate_h investigation (see docs/REAL_DATA_VALIDATION.md)
found that bootstrap-based h calibration systematically underestimates
true held-out false-alarm rate by 3-5x, regardless of burn-in size, and
that the gap is NOT a small-sample artifact that more data fixes -- it
looks like a fundamental limit of resampling from a finite burn-in pool:
bootstrapping can't manufacture tail behavior it has never observed.

This script tests the natural fix: instead of bootstrapping PSEUDO-streams
from burn-in, search h directly against a GENUINE, contiguous, held-out
real-data segment -- reserved purely for calibration and never reused for
final evaluation. If the earlier gap really was a bootstrap-specific
tail-extrapolation problem, calibrating against real windows should close
it. If the gap persists even here, something else is going on (e.g. the
calibration-validation split itself isn't a representative sample of what
the true test-time background looks like).

Three real-data blocks, in order, at FIXED offsets (not derived from any
`--n-*` argument the user might vary -- this project already got bitten
once by an accidental offset/argument coupling; see
real_data_retune.py / calibrated_threshold_experiment.py's
MAX_ARL_WINDOWS_RESERVED fix):
  1. CALIBRATION-VALIDATION windows: h is searched against these directly.
  2. HELD-OUT windows: completely separate from (1), never used to pick h.
     This is the honest, out-of-sample check.
  3. masked-channel / radiation-damage trial caches, as in the other
     real-data scripts.

Usage:
    python3 scripts/validation_split_threshold_experiment.py --target-far 0.02 0.05 0.10
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


def collect_windows(model, scaler, calib, features, pileup, n_jet,
                     n_windows, window_size, start_base):
    """Same pattern as real_data_retune.py's stable_windows() -- precompute
    residual arrays for a block of real windows starting at start_base.
    """
    windows = []
    for w in range(n_windows):
        start = start_base + w * window_size
        _, residuals = score_real_segment(
            model, scaler, calib, features, pileup, n_jet, start, window_size,
        )
        windows.append(residuals)
    return windows


def far_at_h(windows, burn_in_residuals, k, h, two_sided=True):
    """Fraction of real windows that produce at least one alarm, using the
    project's actual evaluation semantics (evaluation.run_detector_on_residuals
    -- stops at first alarm, same as everywhere else in this project).
    """
    n_alarms = 0
    for residuals in windows:
        detector = CUSUMDetector(burn_in_residuals, k=k, h=h, two_sided=two_sided)
        result = evaluation.run_detector_on_residuals(detector, residuals)
        if result.detected:
            n_alarms += 1
    return n_alarms / len(windows)


def search_h_on_validation(validation_windows, burn_in_residuals, k, target_far, h_grid):
    """Smallest h (most sensitive) whose FAR on the validation windows is
    at or below target_far. Same greedy-smallest-h logic as
    CUSUMDetector.calibrate_h(), but measured against real windows instead
    of a bootstrap.
    """
    chosen_h = float(h_grid[-1])
    achieved_far = None
    for h in h_grid:
        far = far_at_h(validation_windows, burn_in_residuals, k, float(h))
        if far <= target_far:
            chosen_h = float(h)
            achieved_far = far
            break
    if achieved_far is None:
        achieved_far = far_at_h(validation_windows, burn_in_residuals, k, chosen_h)
        import warnings
        warnings.warn(
            f"No h in the given grid achieved target_far={target_far} on the "
            f"validation windows; using largest grid value h={chosen_h} "
            f"(validation FAR={achieved_far:.3f}). Extend h_grid's upper bound.",
            stacklevel=2,
        )
    return chosen_h, achieved_far


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000,
                         help="Kept at this project's usual default -- the "
                              "validation-split approach doesn't need the "
                              "inflated burn-in the bootstrap approach needed, "
                              "since it never resamples burn-in for tail "
                              "estimation in the first place.")
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--n-validation-windows", type=int, default=300)
    parser.add_argument("--n-held-out-windows", type=int, default=300)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-damage", type=float, default=3000.0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--h-step", type=float, default=0.25)
    parser.add_argument("--target-far", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--out", type=str, default="results/validation_split_experiment.json")
    args = parser.parse_args()

    # Fixed reservations -- deliberately NOT derived from any --n-* argument,
    # so varying window counts never re-creates the earlier offset-coupling
    # bug (see module docstring).
    MAX_VALIDATION_WINDOWS_RESERVED = 300
    MAX_HELDOUT_WINDOWS_RESERVED = 300

    print(f"[1/4] Real burn-in: {args.n_burn_in} events, training proxy VAE...")
    model, scaler, calib, burn_in_residuals, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False,
    )

    validation_start = args.n_burn_in
    held_out_start = validation_start + MAX_VALIDATION_WINDOWS_RESERVED * args.n_events
    masked_start = held_out_start + MAX_HELDOUT_WINDOWS_RESERVED * args.n_events
    masked_block = args.n_trials * args.n_events
    rad_start = masked_start + masked_block

    pool_needed = rad_start + args.n_trials * args.n_events
    if pool_needed > len(features):
        print(f"WARNING: needs {pool_needed} events, pool has {len(features)}. "
              f"Reduce --n-validation-windows/--n-held-out-windows/--n-trials/--n-events.")

    print(f"\n[2/4] Precomputing {args.n_validation_windows} calibration-validation "
          f"windows and {args.n_held_out_windows} SEPARATE held-out windows...")
    validation_windows = collect_windows(
        model, scaler, calib, features, pileup, n_jet,
        args.n_validation_windows, args.n_events, validation_start,
    )
    held_out_windows = collect_windows(
        model, scaler, calib, features, pileup, n_jet,
        args.n_held_out_windows, args.n_events, held_out_start,
    )

    print(f"\n[3/4] Precomputing masked-channel / radiation-damage trial caches...")
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

    def miss_rate_for_config(cache, k, h):
        n_missed = 0
        for residuals, onset in cache:
            detector = CUSUMDetector(burn_in_residuals, k=k, h=h)
            r = evaluation.run_detector_on_residuals(detector, residuals, true_onset_event=onset)
            if not r.detected:
                n_missed += 1
        return n_missed / len(cache)

    h_grid = np.arange(2.0, 40.0, args.h_step)

    print(f"\n[4/4] Searching h against VALIDATION windows, then checking "
          f"against SEPARATE held-out windows for target FAR in {args.target_far}...\n")
    print(f"{'target_far':>11} {'chosen_h':>9} {'validation_FAR':>15} "
          f"{'held_out_FAR':>13} {'masked_miss':>12} {'rad_miss':>10}")

    results = []
    for target_far in args.target_far:
        chosen_h, validation_far = search_h_on_validation(
            validation_windows, burn_in_residuals, args.k, target_far, h_grid,
        )
        held_out_far = far_at_h(held_out_windows, burn_in_residuals, args.k, chosen_h)
        masked_miss = miss_rate_for_config(masked_cache, args.k, chosen_h)
        rad_miss = miss_rate_for_config(rad_cache, args.k, chosen_h)

        row = {
            "target_far": target_far,
            "k": args.k,
            "chosen_h": chosen_h,
            "validation_far": validation_far,
            "held_out_far": held_out_far,
            "masked_channel_miss_rate": masked_miss,
            "radiation_damage_miss_rate": rad_miss,
        }
        results.append(row)
        print(f"{target_far:11.2f} {chosen_h:9.2f} {validation_far:15.3f} "
              f"{held_out_far:13.3f} {masked_miss:12.2f} {rad_miss:10.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)

    print(f"\nDone. Results written to {args.out}")
    print(f"\nCompare 'held_out_far' against 'target_far' directly -- if they're")
    print(f"close (unlike calibrate_h's bootstrap, which was consistently 3-5x")
    print(f"off), the validation-split approach avoids the tail-extrapolation")
    print(f"problem. This is a single-seed run -- confirm across multiple seeds")
    print(f"(same pattern as calibrated_threshold_multiseed.py) before citing a")
    print(f"specific number.")


if __name__ == "__main__":
    main()
