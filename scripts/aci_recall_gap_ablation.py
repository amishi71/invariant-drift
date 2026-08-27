"""ACI recall-gap ablation: does the sliding calibration buffer explain
the adaptive-vs-fixed-threshold recall/FPR gap seen in main.py's
Component 2 output, or is it a genuine precision/recall trade-off?

Runs THREE variants on the identical score/label stream:
  1. "sliding"  -- ACI with adapt_calibration_buffer=True (main.py's default)
  2. "frozen"   -- ACI with adapt_calibration_buffer=False (buffer pinned
                   to burn-in; alpha_t still adapts from miscoverage)
  3. "fixed"    -- naive frozen threshold (burn-in quantile, no adaptation
                   at all -- same baseline main.py already compares against)

If "frozen" behaves like "fixed" (recall/FPR close to the naive baseline)
and "sliding" is what differs, the sliding buffer is the driver. If
"frozen" is closer to "sliding" than to "fixed", the alpha_t adaptation
itself (not the buffer) drives the gap, and it's a genuine adaptive-
threshold trade-off rather than a buffer artifact.

Run: python scripts/aci_recall_gap_ablation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.stream_loader import synthetic_object_stream
from src.proxy_vae import train_proxy_vae, anomaly_score
from src.residual import CalibrationModel
from src.drift_sim.gradual import misspecified_gradual_stream
from src.conformal.aci import AdaptiveConformalThreshold
from src import evaluation


def _collect(stream, n):
    feats, pileup, n_jet, lumi, labels = [], [], [], [], []
    for i, event in enumerate(stream):
        if i >= n:
            break
        feats.append(event["features"])
        pileup.append(event["pileup"])
        n_jet.append(event.get("n_jet", np.nan))
        lumi.append(event.get("lumi", np.nan))
        labels.append(event.get("true_label", 0))
    return (
        np.array(feats), np.array(pileup), np.array(n_jet),
        np.array(lumi), np.array(labels),
    )


def run_aci_variant(aci, scores, labels, delayed_lag=300):
    """Same feedback-loop structure as main.py's run_component_2: live
    Zero-Bias feedback on confirmed-background events, plus a simulated
    delayed-offline-validation channel with a fixed lag.
    """
    n = len(scores)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        s, lab = float(scores[i]), int(labels[i])
        flags[i] = aci.decide(s)
        if lab == 0:
            aci.update(s)
        j = i - delayed_lag
        if j >= 0 and int(labels[j]) == 0:
            aci.update_delayed(float(scores[j]), true_label_background=True)
    return flags


def main():
    seed = 0
    n_burn_in = 2500
    n_events = 1500
    alpha_target = 0.02

    print(f"[1/3] Burn-in: {n_burn_in} stable events, training proxy VAE...")
    feats, pileup, n_jet, lumi, _ = _collect(
        synthetic_object_stream(n_burn_in, seed=seed), n_burn_in,
    )
    model, scaler, _ = train_proxy_vae(feats, epochs=50, seed=seed)
    burn_in_scores = anomaly_score(model, scaler, feats)
    calib = CalibrationModel.fit(burn_in_scores, pileup, n_jet, lumi)
    fixed_threshold = float(np.quantile(burn_in_scores, 1 - alpha_target))

    print(f"[2/3] Scoring the misspecified-gradual scenario "
          f"({n_events} events, same as main.py's Component 2 demo)...")
    onset_frac, bias_rate = 0.3, 0.5  # match main.py's run_component_2 exactly
    ev_feats, ev_pileup, ev_n_jet, ev_lumi, labels = _collect(
        misspecified_gradual_stream(
            n_events, bias_onset_frac=onset_frac, bias_rate=bias_rate, seed=9000 + seed,
        ),
        n_events,
    )
    scores = anomaly_score(model, scaler, ev_feats)
    true_labels = labels.astype(bool)

    print("[3/3] Running sliding / frozen / fixed variants...\n")

    aci_sliding = AdaptiveConformalThreshold.from_burn_in(
        burn_in_scores, alpha_target=alpha_target, gamma=0.01, calibration_window=2000,
        adapt_calibration_buffer=True,
    )
    flags_sliding = run_aci_variant(aci_sliding, scores, labels)

    aci_frozen = AdaptiveConformalThreshold.from_burn_in(
        burn_in_scores, alpha_target=alpha_target, gamma=0.01, calibration_window=2000,
        adapt_calibration_buffer=False,
    )
    flags_frozen = run_aci_variant(aci_frozen, scores, labels)

    flags_fixed = scores > fixed_threshold

    def metrics(flags, name):
        tp = int(np.sum(flags & true_labels))
        fp = int(np.sum(flags & ~true_labels))
        fn = int(np.sum(~flags & true_labels))
        tn = int(np.sum(~flags & ~true_labels))
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        return {"name": name, "recall": recall, "fpr": fpr, "precision": precision,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    results = [
        metrics(flags_sliding, "sliding (main.py default)"),
        metrics(flags_frozen, "frozen (buffer pinned to burn-in)"),
        metrics(flags_fixed, "fixed (no adaptation at all)"),
    ]

    print(f"{'variant':<34} {'recall':>8} {'FPR':>8} {'precision':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['name']:<34} {r['recall']:>8.4f} {r['fpr']:>8.4f} {r['precision']:>10.4f}")

    print()
    sliding, frozen, fixed = results
    dist_frozen_to_sliding = abs(frozen["recall"] - sliding["recall"]) + abs(frozen["fpr"] - sliding["fpr"])
    dist_frozen_to_fixed = abs(frozen["recall"] - fixed["recall"]) + abs(frozen["fpr"] - fixed["fpr"])
    if dist_frozen_to_fixed < dist_frozen_to_sliding:
        verdict = ("frozen tracks FIXED more closely than sliding -> the SLIDING BUFFER "
                   "is the main driver of the recall/FPR gap, not the alpha_t adaptation itself.")
    else:
        verdict = ("frozen tracks SLIDING more closely than fixed -> the alpha_t adaptation "
                   "itself (not the buffer) drives the gap -- this looks like a genuine "
                   "precision/recall trade-off from adapting the threshold, not a buffer artifact.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()