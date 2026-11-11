"""
scripts/synthetic_far_windows.py

Closes the comparison the README/docs currently get wrong. The "real 67% vs
synthetic ~17-33%" false-alarm gap compared 5000-event real windows against
2000-event synthetic windows with n=6 trials. False-alarm probability grows
with window length, so those numbers are not comparable.

This script measures the SYNTHETIC pipeline's false-alarm probability for
CUSUM (two-sided, k fixed) at several h values, at both 2000- and 5000-event
windows, over many independent no-drift windows, and prints it beside an
ideal i.i.d. Gaussian Monte Carlo reference computed in the same run. It also
reports the held-out residual mean/std per window, because small offsets from
(0, 1) inflate false alarms sharply (see the printed reference rows).

Design notes:
  - One 5000-event window supplies both window lengths: the probability of an
    alarm within the first T events is the fraction of windows whose first
    alarm index is < T.
  - Window streams use seeds 100000 + 10000*seed + w, so windows never repeat
    across --seed values (as long as --n-windows < 10000).
  - The detector reference is the burn-in residuals (as everywhere in the
    repo). Windows are fresh streams, so this is a held-out measurement for
    the synthetic pipeline -- the like-for-like counterpart of the real-data
    ARL check.

Usage:
    python3 scripts/synthetic_far_windows.py
    python3 scripts/synthetic_far_windows.py --n-windows 300 --seed 1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import build_calibration, score_stream
from src.stream_loader import synthetic_object_stream
from src.detectors.cusum import CUSUMDetector
from src import evaluation


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def gaussian_first_alarm(h, k, t_max, trials, rng, mean=0.0, std=1.0):
    """First-alarm index (inf if none within t_max) for two-sided CUSUM on
    i.i.d. N(mean, std^2) data, vectorised over trials."""
    sp = np.zeros(trials)
    sn = np.zeros(trials)
    first = np.full(trials, np.inf)
    for t in range(t_max):
        z = mean + std * rng.standard_normal(trials)
        sp = np.maximum(0.0, sp + z - k)
        sn = np.maximum(0.0, sn - z - k)
        newly = ((sp >= h) | (sn >= h)) & np.isinf(first)
        first[newly] = t
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-burn-in", type=int, default=3000)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--window-size", type=int, default=5000)
    p.add_argument("--report-sizes", type=int, nargs="+", default=[2000, 5000])
    p.add_argument("--h-values", type=float, nargs="+", default=[8.0, 10.0, 11.0])
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--mc-trials", type=int, default=20000)
    p.add_argument("--out", type=str, default="results/synthetic_far_windows.json")
    args = p.parse_args()
    assert max(args.report_sizes) <= args.window_size

    model, scaler, calib, burn_resid = build_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False)

    print(f"\nScoring {args.n_windows} fresh no-drift windows of {args.window_size} events...")
    windows = []
    for w in range(args.n_windows):
        stream = synthetic_object_stream(args.window_size, seed=100000 + 10000 * args.seed + w)
        _, resid, _ = score_stream(model, scaler, calib, stream, args.window_size)
        windows.append(resid)
    w_means = np.array([r.mean() for r in windows])
    w_stds = np.array([r.std() for r in windows])
    pooled = np.concatenate(windows)
    print(f"Held-out residuals: pooled mean={pooled.mean():+.4f} std={pooled.std():.4f}; "
          f"window means {w_means.mean():+.4f} +/- {w_means.std():.4f}; "
          f"window stds {w_stds.mean():.4f} +/- {w_stds.std():.4f}")

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"\n{'h':>5} {'T':>6} {'synthetic FAR':>14} {'95% CI':>16} {'ideal Gaussian':>15}")
    for h in args.h_values:
        first_idx = []
        for resid in windows:
            det = CUSUMDetector(burn_resid, k=args.k, h=h)
            r = evaluation.run_detector_on_residuals(det, resid)
            first_idx.append(r.detection_index if r.detected else np.inf)
        first_idx = np.array(first_idx, dtype=float)
        g_first = gaussian_first_alarm(h, args.k, max(args.report_sizes), args.mc_trials, rng)
        for T in args.report_sizes:
            k_alarm = int(np.sum(first_idx < T))
            lo, hi = wilson(k_alarm, args.n_windows)
            g = float(np.mean(g_first < T))
            rows.append({"h": h, "k": args.k, "window": T, "n_windows": args.n_windows,
                         "synthetic_far": k_alarm / args.n_windows,
                         "ci95": [lo, hi], "gaussian_mc_far": g})
            print(f"{h:5.1f} {T:6d} {k_alarm / args.n_windows:14.3f} "
                  f"[{lo:5.3f},{hi:5.3f}] {g:15.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows,
                   "heldout_residuals": {"pooled_mean": float(pooled.mean()),
                                         "pooled_std": float(pooled.std()),
                                         "window_mean_sd": float(w_means.std()),
                                         "window_std_mean": float(w_stds.mean())},
                   "config": vars(args)}, f, indent=2, default=str)
    print(f"\nWritten to {args.out}")
    print("Compare to real: use the SAME window length (5000) and h; real windows are "
          "sequential 5000-event segments (real_data_sweep.py / real_data_retune.py).")


if __name__ == "__main__":
    main()
