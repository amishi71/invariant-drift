"""
scripts/score_gate_baseline.py

Supplies the numbers the paper's masked-channel claim (Figure 1) is missing:
what a one-sided "flag if the score is HIGH" gate does on a masked-channel
failure, compared with residual-based change-point detectors, with the
outcome decomposed so that false alarms and missed detections are no longer
lumped together.

Why this exists. The thesis is that a failure which pushes the anomaly
score DOWN is invisible to a one-sided high-score gate but visible to a
two-sided residual monitor. The existing results do not yet quantify either
half: (a) no run reports what the one-sided gate actually does, and (b) the
repo's single-shot "miss rate" (evaluation.run_detector_on_residuals via
detection_latency) counts an alarm BEFORE the true onset as a miss, so the
reported miss rates mix false alarms with failures to detect.

For each drop fraction and each trial (changepoint at n_events // 2) this
script reports:

  Event-level gate (a fixed threshold at the --gate-quantile of burn-in
    scores, flagging events whose score exceeds it):
      flag rate before onset, flag rate after onset, and the mean residual
      shift in sigma units. If the post-onset flag rate is not higher than
      the pre-onset rate, the gate cannot alert on this failure.

  Stream-level detectors, each reported three ways:
      pre_onset_alarm   an alarm raised before the true onset (false alarm)
      single_shot_miss  the repo's existing definition (pre-onset alarm OR
                        no alarm counts as a miss) -- kept for comparability
      fresh_detect/delay  the detector is restarted at the onset on the
                        post-onset segment only; reports detection rate and
                        delay from onset. This isolates detection ability
                        from pre-onset false alarms. (A freshly reset CUSUM
                        is the favourable case; a deployed detector may
                        carry non-zero state into the change.)

  Detectors:
      CUSUM up-only on the residual            the stream-level analogue of
                                               a one-sided high-score gate
      CUSUM two-sided on the residual          repo default (k=0.5, h=8)
      CUSUM two-sided on raw log-score         NO covariate regression --
                                               the covariate-unaware baseline
      CUSUM down-only on the residual          oracle failure direction
                                               (k=0.1, h=16, as in main.py)

Caveats to carry into the write-up:
  - The gate threshold and every detector reference are computed from the
    burn-in sample, which is in-sample for the VAE and regression. Absolute
    false-alarm numbers inherit that optimism; see the calibration notes.
  - In this scenario pileup/multiplicity are stationary, so the regression
    residual has little to add over the raw log-score. The comparison of
    those two rows is a sanity check, not a test of the regression's value;
    that needs a scenario with moving covariates.
  - Trial seeds are shared across drop fractions (paired comparison).

Usage:
    python3 scripts/score_gate_baseline.py
    python3 scripts/score_gate_baseline.py --drop-fractions 0.15 0.4 --n-trials 50
    python3 scripts/score_gate_baseline.py --source real --n-burn-in 20000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from main import _collect, build_calibration, score_stream
from src.proxy_vae import anomaly_score
from src.stream_loader import synthetic_object_stream
from src.detectors.cusum import CUSUMDetector
from src.drift_sim.abrupt import masked_channel_stream
from src import evaluation


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def make_suite(burn_resid, burn_logscore):
    neg = lambda x: -np.asarray(x)
    return {
        "CUSUM up-only, residual (high-score-gate analogue)": dict(
            series="resid", ref=burn_resid,
            factory=lambda ref: CUSUMDetector(ref, k=0.5, h=8.0, two_sided=False)),
        "CUSUM two-sided, residual (default k=0.5,h=8)": dict(
            series="resid", ref=burn_resid,
            factory=lambda ref: CUSUMDetector(ref, k=0.5, h=8.0, two_sided=True)),
        "CUSUM two-sided, raw log-score (no regression)": dict(
            series="logscore", ref=burn_logscore,
            factory=lambda ref: CUSUMDetector(ref, k=0.5, h=8.0, two_sided=True)),
        "CUSUM down-only, residual (oracle direction k=0.1,h=16)": dict(
            series="resid", ref=burn_resid,
            factory=lambda ref: CUSUMDetector(ref, k=0.1, h=16.0, two_sided=False, transform=neg)),
    }


def decompose(factory, ref, series, onset):
    d = factory(ref)
    r = evaluation.run_detector_on_residuals(d, series, true_onset_event=onset)
    pre_alarm = bool(r.detected and r.detection_index < onset)
    single_detect = bool(r.detected and r.latency is not None)
    d2 = factory(ref)
    r2 = evaluation.run_detector_on_residuals(d2, series[onset:], true_onset_event=0)
    return {
        "pre_onset_alarm": pre_alarm,
        "single_shot_miss": not single_detect,
        "fresh_detect": bool(r2.detected),
        "fresh_delay": int(r2.latency) if (r2.detected and r2.latency is not None) else None,
    }


def log_score(scores):
    return np.log(np.clip(np.asarray(scores, dtype=np.float64), 1e-12, None))


def run_condition(trials, burn_scores, burn_resid, onset, gate_q):
    """trials: list of (scores, residuals) arrays, one per trial."""
    thr = float(np.quantile(burn_scores, gate_q))
    suite = make_suite(burn_resid, log_score(burn_scores))

    gate_pre, gate_post, shift = [], [], []
    per_det = {name: [] for name in suite}
    for scores, resid in trials:
        gate_pre.append(float(np.mean(scores[:onset] > thr)))
        gate_post.append(float(np.mean(scores[onset:] > thr)))
        shift.append(float(np.mean(resid[onset:]) - np.mean(resid[:onset])))
        for name, spec in suite.items():
            series = resid if spec["series"] == "resid" else log_score(scores)
            per_det[name].append(decompose(spec["factory"], spec["ref"], series, onset))

    n = len(trials)
    out = {
        "n_trials": n,
        "gate": {
            "threshold_quantile": gate_q,
            "flag_rate_pre_mean": float(np.mean(gate_pre)),
            "flag_rate_post_mean": float(np.mean(gate_post)),
            "frac_trials_post_flag_rate_higher": float(np.mean(np.array(gate_post) > np.array(gate_pre))),
            "mean_residual_shift_sigma": float(np.mean(shift)),
        },
        "detectors": {},
    }
    for name, rows in per_det.items():
        def rate(key):
            k = int(sum(bool(r[key]) for r in rows))
            lo, hi = wilson(k, n)
            return {"rate": k / n, "ci95": [lo, hi]}
        delays = [r["fresh_delay"] for r in rows if r["fresh_delay"] is not None]
        out["detectors"][name] = {
            "pre_onset_alarm": rate("pre_onset_alarm"),
            "single_shot_miss": rate("single_shot_miss"),
            "fresh_detect": rate("fresh_detect"),
            "fresh_delay_mean": float(np.mean(delays)) if delays else None,
            "fresh_delay_median": float(np.median(delays)) if delays else None,
        }
    return out


def print_condition(label, res):
    g = res["gate"]
    print(f"\n=== {label}  (n_trials={res['n_trials']}) ===")
    print(f"  mean residual shift after onset: {g['mean_residual_shift_sigma']:+.3f} sigma")
    print(f"  one-sided score gate (q={g['threshold_quantile']}): flag rate "
          f"{g['flag_rate_pre_mean']:.4f} before -> {g['flag_rate_post_mean']:.4f} after onset; "
          f"post > pre in {100*g['frac_trials_post_flag_rate_higher']:.0f}% of trials")
    print(f"  {'detector':<58} {'pre-alarm':>9} {'1-shot miss':>11} {'fresh det':>9} {'delay(med)':>10}")
    for name, d in res["detectors"].items():
        dm = d["fresh_delay_median"]
        print(f"  {name:<58} {d['pre_onset_alarm']['rate']:9.2f} "
              f"{d['single_shot_miss']['rate']:11.2f} {d['fresh_detect']['rate']:9.2f} "
              f"{'-' if dm is None else f'{dm:.0f}':>10}")


def synthetic_conditions(args):
    model, scaler, calib, burn_resid = build_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False)
    feats, *_ = _collect(synthetic_object_stream(args.n_burn_in, seed=args.seed), args.n_burn_in)
    burn_scores = anomaly_score(model, scaler, feats)
    onset = args.n_events // 2
    seed_base = 20000 + 1000 * args.seed
    for drop in args.drop_fractions:
        trials = []
        for t in range(args.n_trials):
            stream = masked_channel_stream(
                args.n_events, changepoint_event=onset, drop_fraction=drop, seed=seed_base + t)
            scores, resid, _ = score_stream(model, scaler, calib, stream, args.n_events)
            trials.append((scores, resid))
        yield f"synthetic, drop_fraction={drop}", drop, trials, burn_scores, burn_resid, onset


def real_conditions(args):
    from src.real_pipeline import build_real_calibration
    from src.drift_sim.real_data_injection import real_masked_channel_stream
    model, scaler, calib, burn_resid, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False)
    burn_scores = anomaly_score(model, scaler, features[:args.n_burn_in])
    onset = args.n_events // 2
    n_slots = (len(features) - args.n_burn_in) // args.n_events
    if n_slots < args.n_trials:
        raise SystemExit(f"Pool has only {n_slots} disjoint {args.n_events}-event slots; "
                         f"reduce --n-trials.")
    rng = np.random.default_rng(args.seed)
    starts = [args.n_burn_in + int(s) * args.n_events
              for s in rng.choice(n_slots, size=args.n_trials, replace=False)]
    for drop in args.drop_fractions:
        trials = []
        for start in starts:
            stream = real_masked_channel_stream(
                features, pileup, n_jet, start=start, n_events=args.n_events,
                changepoint_event=onset, drop_fraction=drop)
            scores, resid, _ = score_stream(model, scaler, calib, stream, args.n_events)
            trials.append((scores, resid))
        yield f"real JetHT, drop_fraction={drop}", drop, trials, burn_scores, burn_resid, onset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--drop-fractions", type=float, nargs="+", default=[0.15, 0.4])
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-events", type=int, default=5000)
    p.add_argument("--n-burn-in", type=int, default=3000)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-quantile", type=float, default=0.99)
    p.add_argument("--out", type=str, default="results/score_gate_baseline.json")
    args = p.parse_args()

    gen = synthetic_conditions(args) if args.source == "synthetic" else real_conditions(args)
    results = {}
    for label, drop, trials, burn_scores, burn_resid, onset in gen:
        res = run_condition(trials, burn_scores, burn_resid, onset, args.gate_quantile)
        results[label] = res
        print_condition(label, res)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nWritten to {args.out}")
    print("NOTE: thresholds/references are in-sample (burn-in). Treat absolute false-alarm "
          "rates as optimistic; the comparisons between rows are the point.")


if __name__ == "__main__":
    main()
