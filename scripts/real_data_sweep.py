"""
scripts/real_data_sweep.py

Runs the masked-channel and radiation-damage detection sweeps on REAL CMS
Open Data (record 30558, cached in data/real_cache/jetht_features.npz),
through this project's actual shared evaluation.run_detector_on_residuals
path -- per main.py::run_masked_channel_onesided_case_study's docstring,
NOT a standalone diagnostic with different semantics (that mistake cost a
full debugging cycle once already; see handover notes).

Also reproduces the plain real-data ARL/false-alarm check (no injection,
genuinely untouched real background) as a saved, rerunnable script instead
of the ad hoc one-off that produced ARL=2624.8, FA_rate=0.60 previously.

Non-overlapping real segments substitute for "seeds" here: real data has
no reseedable RNG, so where the synthetic case studies vary a VAE/stream
seed across trials, this script instead advances the start index into the
real feature pool by one window per trial. The burn-in, ARL-check block,
masked-channel block, and radiation-damage block are all disjoint slices
of the pool so no window is reused across checks.

Usage (from repo root, with venv active):
    python3 scripts/real_data_sweep.py
    python3 scripts/real_data_sweep.py --n-trials 15 --n-events 5000
"""
import argparse
import json
import os
import sys

# Allow `from main import ...` / `from src...` when run as scripts/real_data_sweep.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import _cache_residual_trials, DETECTOR_SPECS
from src.detectors.cusum import CUSUMDetector
from src import evaluation
from src.real_pipeline import build_real_calibration, score_real_segment
from src.drift_sim.real_data_injection import (
    real_masked_channel_stream,
    real_radiation_damage_stream,
)


def real_arl_check(model, scaler, calib, burn_in_residuals, features, pileup, n_jet,
                    n_windows, window_size, start_base):
    """No-injection ARL/false-alarm check on genuinely untouched real data.
    Mirrors evaluation.average_run_length's exact logic (mean run length to
    a false alarm, censored at window_size if none occurs), fed real
    non-overlapping windows instead of a synthetic stable_residual_factory.
    """
    detector_factory = DETECTOR_SPECS["CUSUM"]
    run_lengths = []
    n_false_alarms = 0
    for w in range(n_windows):
        start = start_base + w * window_size
        _, residuals = score_real_segment(
            model, scaler, calib, features, pileup, n_jet, start, window_size,
        )
        detector = detector_factory(burn_in_residuals)
        result = evaluation.run_detector_on_residuals(detector, residuals)
        if result.detected:
            n_false_alarms += 1
            run_lengths.append(result.detection_index + 1)
        else:
            run_lengths.append(window_size)
    return {
        "arl_mean": float(np.mean(run_lengths)),
        "arl_std": float(np.std(run_lengths)),
        "false_alarm_rate": n_false_alarms / n_windows,
        "n_windows": n_windows,
        "window_size": window_size,
    }


def run_case_study(cache, burn_in_residuals, detector_factories):
    """Shared scoring loop -- same pattern as main.py's case-study
    functions: for each detector, run every cached (residuals, onset)
    trial through evaluation.run_detector_on_residuals and report
    miss_rate / latency stats.
    """
    results = {}
    for label, factory in detector_factories.items():
        latencies, n_missed = [], 0
        for residuals, onset in cache:
            detector = factory(burn_in_residuals)
            r = evaluation.run_detector_on_residuals(detector, residuals, true_onset_event=onset)
            if r.detected and r.latency is not None:
                latencies.append(r.latency)
            else:
                n_missed += 1
        results[label] = {
            "n_trials": len(cache),
            "n_missed": n_missed,
            "miss_rate": n_missed / len(cache),
            "latency_mean": float(np.mean(latencies)) if latencies else None,
            "latency_median": float(np.median(latencies)) if latencies else None,
        }
        print(f"      {label:42s} miss_rate={results[label]['miss_rate']:.2f} "
              f"latency_mean={results[label]['latency_mean']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000)
    parser.add_argument("--n-events", type=int, default=5000,
                         help="Window size for each ARL/trial segment.")
    parser.add_argument("--n-arl-windows", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=10,
                         help="Non-overlapping real segments per detection sweep.")
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-damage", type=float, default=3000.0,
                         help="Default matches real_radiation_damage_stream's own "
                              "default (verified against radiation_damage.py's "
                              "5%% detectability-floor onset rule). Pass --tau-damage "
                              "400.0 instead if you want a direct comparison against "
                              "main.py's synthetic run_space_case_study, which "
                              "overrides tau_damage=400.0.")
    parser.add_argument("--verbose-vae", action="store_true")
    parser.add_argument("--out", type=str, default="results/real_data_sweep.json")
    args = parser.parse_args()

    print(f"[1/4] Real burn-in: {args.n_burn_in} events, training proxy VAE "
          f"({args.vae_epochs} epochs)...")
    model, scaler, calib, burn_in_residuals, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, args.verbose_vae,
    )

    # Disjoint blocks of the real pool: burn-in, then ARL-check windows,
    # then masked-channel trials, then radiation-damage trials -- nothing
    # is reused across checks.
    arl_start = args.n_burn_in
    arl_block = args.n_arl_windows * args.n_events

    masked_start = arl_start + arl_block
    masked_block = args.n_trials * args.n_events

    rad_start = masked_start + masked_block
    rad_block = args.n_trials * args.n_events

    pool_needed = rad_start + rad_block
    if pool_needed > len(features):
        print(f"WARNING: needs {pool_needed} events, pool has {len(features)}. "
              f"Reduce --n-trials/--n-arl-windows/--n-events.")

    print(f"\n[2/4] Real-data ARL check (no injection, {args.n_arl_windows} "
          f"non-overlapping {args.n_events}-event windows)...")
    arl_result = real_arl_check(
        model, scaler, calib, burn_in_residuals, features, pileup, n_jet,
        args.n_arl_windows, args.n_events, arl_start,
    )
    print(f"      ARL={arl_result['arl_mean']:.1f}  "
          f"FA_rate={arl_result['false_alarm_rate']:.2f}")

    print(f"\n[3/4] Masked-channel sweep on real+injected data "
          f"({args.n_trials} non-overlapping real segments)...")
    masked_cache = _cache_residual_trials(
        model, scaler, calib,
        lambda t: real_masked_channel_stream(
            features, pileup, n_jet,
            start=masked_start + t * args.n_events, n_events=args.n_events,
            changepoint_event=args.n_events // 2, drop_fraction=0.4,
        ),
        args.n_events, args.n_trials, args.seed,
    )
    onesided_factory = lambda ref: CUSUMDetector(
        ref, k=0.1, h=16.0, two_sided=False, transform=lambda x: -np.asarray(x),
    )
    masked_results = run_case_study(masked_cache, burn_in_residuals, {
        "CUSUM (default, two-sided)": DETECTOR_SPECS["CUSUM"],
        "CUSUM (one-sided, masked-channel-tuned)": onesided_factory,
    })

    print(f"\n[4/4] Radiation-damage sweep on real+injected data "
          f"(tau_damage={args.tau_damage}, {args.n_trials} non-overlapping "
          f"real segments)...")
    rad_cache = _cache_residual_trials(
        model, scaler, calib,
        lambda t: real_radiation_damage_stream(
            features, pileup, n_jet,
            start=rad_start + t * args.n_events, n_events=args.n_events,
            tau_damage=args.tau_damage,
        ),
        args.n_events, args.n_trials, args.seed,
    )
    rad_results = run_case_study(rad_cache, burn_in_residuals, {
        "CUSUM": DETECTOR_SPECS["CUSUM"],
        "BOCPD": DETECTOR_SPECS["BOCPD"],
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "real_arl_check": arl_result,
            "real_masked_channel": masked_results,
            "real_radiation_damage": rad_results,
            "config": vars(args),
        }, f, indent=2, default=str)
    print(f"\nDone. Results written to {args.out}")


if __name__ == "__main__":
    main()