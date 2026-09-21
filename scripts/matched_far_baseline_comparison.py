"""
scripts/matched_far_baseline_comparison.py

Produces the headline comparison this project's README/paper is currently
missing: detection latency and miss rate for CUSUM vs. Page-Hinkley vs.
ADWIN vs. KSWIN, all tuned to the SAME target false-alarm rate first, on
the same drift scenario. main.py's existing Component 1 sweep runs all
five detectors, but each at its own default (k,h)/(delta,lam)/(delta)/
(alpha) operating point -- comparing their raw detection latencies at
different false-alarm rates isn't a fair comparison. This fixes that by
searching each detector's threshold parameter (on stable, no-drift
synthetic windows) until it hits --target-far, THEN comparing.

Runs on SYNTHETIC data only (fast, no real-data VAE retraining, and
consistent with main.py's own default fast-path convention). This is a
legitimate first cut for a "preliminary" headline number -- the natural
follow-up (not done here) is repeating this on real CMS Open Data, the
way the CUSUM-only false-alarm-rate work was extended earlier.

CAVEAT ON PARAMETER GRIDS: Page-Hinkley's lam, ADWIN's delta, and KSWIN's
alpha grids below are set from this project's documented defaults and the
KSWIN retuning finding already in docs/FINDINGS.md (smaller alpha = fewer
false alarms). I have not seen src/detectors/page_hinkley.py, adwin.py, or
kswin.py directly in this session, so the exact monotonic direction and
sensible range for each parameter is inferred, not confirmed against
source. If the search hits the edge of a grid without reaching
--target-far, it will print a warning -- widen that detector's grid (or
share the relevant detector file) and rerun rather than trusting an
edge-of-grid result.

Usage:
    python3 scripts/matched_far_baseline_comparison.py --target-far 0.05
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import build_calibration, _collect
from src.stream_loader import synthetic_object_stream
from src.drift_sim.gradual import misspecified_gradual_stream
from src.proxy_vae import anomaly_score
from src.detectors.cusum import CUSUMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.detectors.adwin import ADWINDetector
from src.detectors.kswin import KSWINDetector
from src import evaluation


def build_stable_windows(model, scaler, calib, n_windows, window_size, seed_base):
    """No-drift synthetic windows, used only to measure/search false-alarm rate."""
    windows = []
    for w in range(n_windows):
        feats, pileup, n_jet, lumi, _ = _collect(
            synthetic_object_stream(window_size, seed=seed_base + w), window_size,
        )
        scores = anomaly_score(model, scaler, feats)
        residuals = calib.residual(scores, pileup, n_jet, lumi)
        windows.append(residuals)
    return windows


def build_drift_trials(model, scaler, calib, n_trials, n_events, onset_frac, bias_rate, seed_base):
    """misspecified_gradual_stream trials with known onset -- same scenario
    main.py's Component 2 demo already uses, so results are comparable to
    existing project numbers.
    """
    trials = []
    onset_event = int(onset_frac * n_events)
    for t in range(n_trials):
        feats, pileup, n_jet, lumi, _ = _collect(
            misspecified_gradual_stream(
                n_events, bias_onset_frac=onset_frac, bias_rate=bias_rate,
                seed=seed_base + t * 137,
            ),
            n_events,
        )
        scores = anomaly_score(model, scaler, feats)
        residuals = calib.residual(scores, pileup, n_jet, lumi)
        trials.append((residuals, onset_event))
    return trials


def far_at_config(windows, burn_in_residuals, factory):
    n_alarms = 0
    for residuals in windows:
        detector = factory(burn_in_residuals)
        result = evaluation.run_detector_on_residuals(detector, residuals)
        if result.detected:
            n_alarms += 1
    return n_alarms / len(windows)


def search_param(factory_builder, param_grid, windows, burn_in_residuals, target_far, name,
                  descending=False):
    """Finds the MOST SENSITIVE parameter value that still meets target_far.

    'Most sensitive' depends on the detector: for CUSUM/Page-Hinkley, a
    LARGER threshold is more conservative (fewer alarms), so the most
    sensitive value satisfying the target is the SMALLEST one that works
    -- searched ascending (descending=False). For ADWIN/KSWIN, it's the
    OPPOSITE: a SMALLER delta/alpha is more conservative, so the most
    sensitive value satisfying the target is the LARGEST one that works
    -- searched descending (descending=True). Getting this backwards means
    the search stops at the first (trivially compliant, overly
    conservative) grid point instead of actually finding the boundary --
    which silently produces a much stricter, unfairly slower operating
    point for that detector.
    """
    grid = list(param_grid)
    if descending:
        grid = grid[::-1]
    chosen = grid[-1]
    achieved = None
    for p in grid:
        far = far_at_config(windows, burn_in_residuals, lambda ref, p=p: factory_builder(ref, p))
        if far <= target_far:
            chosen, achieved = p, far
            break
    if achieved is None:
        achieved = far_at_config(windows, burn_in_residuals, lambda ref, p=chosen: factory_builder(ref, p))
        warnings.warn(
            f"{name}: no parameter in the given grid achieved target_far="
            f"{target_far}; using grid edge {chosen} (achieved FAR={achieved:.3f}). "
            f"Widen this detector's grid and rerun before trusting this result.",
        )
    return chosen, achieved


def eval_on_trials(trials, burn_in_residuals, factory):
    n_missed = 0
    latencies = []
    for residuals, onset in trials:
        detector = factory(burn_in_residuals)
        r = evaluation.run_detector_on_residuals(detector, residuals, true_onset_event=onset)
        if r.detected and r.latency is not None:
            latencies.append(r.latency)
        else:
            n_missed += 1
    return {
        "miss_rate": n_missed / len(trials),
        "latency_mean": float(np.mean(latencies)) if latencies else None,
        "latency_std": float(np.std(latencies)) if latencies else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000)
    parser.add_argument("--n-events", type=int, default=2000)
    parser.add_argument("--n-stable-windows", type=int, default=300)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onset-frac", type=float, default=0.3)
    parser.add_argument("--bias-rate", type=float, default=0.5)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="results/matched_far_baseline_comparison.json")
    args = parser.parse_args()

    print(f"[1/3] Synthetic burn-in: {args.n_burn_in} events, training proxy VAE...")
    model, scaler, calib, burn_in_residuals = build_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False,
    )

    print(f"\n[2/3] Building {args.n_stable_windows} stable windows (FAR search) "
          f"and {args.n_trials} drift trials (misspecified_gradual, matches "
          f"Component 2's existing scenario)...")
    stable_windows = build_stable_windows(
        model, scaler, calib, args.n_stable_windows, args.n_events, seed_base=5000 + args.seed,
    )
    drift_trials = build_drift_trials(
        model, scaler, calib, args.n_trials, args.n_events,
        args.onset_frac, args.bias_rate, seed_base=9000 + args.seed,
    )

    detectors = {
        "CUSUM": {
            "builder": lambda ref, h: CUSUMDetector(ref, k=0.5, h=h),
            "grid": np.arange(2.0, 40.0, 0.25),
            "descending": False,  # larger h = fewer alarms -> smallest-that-works is most sensitive
        },
        "Page-Hinkley": {
            "builder": lambda ref, lam: PageHinkleyDetector(ref, delta=0.5, lam=lam),
            "grid": np.arange(2.0, 150.0, 0.5),  # finer step -- 2.0 overshot past the target region
            "descending": False,  # larger lam = fewer alarms, same direction as CUSUM's h
        },
        "ADWIN": {
            "builder": lambda ref, delta: ADWINDetector(ref, delta=delta),
            "grid": np.logspace(-5, -1, 80),
            "descending": True,  # SMALLER delta = fewer alarms -> largest-that-works is most sensitive
        },
        "KSWIN": {
            "builder": lambda ref, alpha: KSWINDetector(ref, alpha=alpha, window_size=200, stat_size=40),
            "grid": np.logspace(-6, -1, 15),  # KSWIN is ~1000x slower than the others
            "descending": True,  # SMALLER alpha = fewer alarms -> largest-that-works is most sensitive
        },
    }

    print(f"\n[3/3] Tuning each detector to target_far={args.target_far}, then "
          f"measuring latency/miss-rate on the SAME drift scenario...\n")
    print(f"{'detector':<14} {'tuned_param':>12} {'achieved_FAR':>13} "
          f"{'miss_rate':>10} {'latency_mean':>13} {'latency_std':>12}")
    print("-" * 80)

    results = {}
    for name, spec in detectors.items():
        chosen_param, achieved_far = search_param(
            spec["builder"], spec["grid"], stable_windows, burn_in_residuals,
            args.target_far, name, descending=spec["descending"],
        )
        eval_result = eval_on_trials(
            drift_trials, burn_in_residuals,
            lambda ref, p=chosen_param, b=spec["builder"]: b(ref, p),
        )
        results[name] = {"chosen_param": chosen_param, "achieved_far": achieved_far, **eval_result}
        lat_mean = eval_result["latency_mean"]
        lat_std = eval_result["latency_std"]
        print(f"{name:<14} {chosen_param:12.5g} {achieved_far:13.3f} "
              f"{eval_result['miss_rate']:10.2f} "
              f"{lat_mean if lat_mean is not None else float('nan'):13.1f} "
              f"{lat_std if lat_std is not None else float('nan'):12.1f}")

    print(f"\nAll detectors matched to ~{args.target_far} FAR on {args.n_stable_windows} "
          f"stable synthetic windows; latency/miss-rate measured on the same "
          f"{args.n_trials} misspecified_gradual trials. PRELIMINARY: synthetic "
          f"data, single seed -- confirm on real data and across seeds before "
          f"treating this as a final headline number.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
