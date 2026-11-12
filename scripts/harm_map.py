#!/usr/bin/env python3
"""Harm map (Phase 4, minimal): does an injected calorimeter-scale drop cost signal efficiency or bandwidth?
Calibrate on the CALIB run (as in the cross-run experiments). Background = random events of the MONITOR run.
Thresholds on the raw VAE score at fixed healthy background rates. Signal = simulated caches (PROXY only).
Pre-registered harm criterion (docs/prereg.md), at 1% healthy background rate:
  harmful if recalibrated-threshold signal efficiency drops >10% relative OR frozen-threshold background
  rate changes >20% relative.
Quick test:  python3 scripts/harm_map.py --bkg-events 200000 --max-sig 20000 --vae-epochs 5
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.calibration_split import build_split_calibration, score_chunked
from src.injection_sets import INJECTION_SETS, inject_masked_cols

SIGNALS = {"susy_m350": "data/real_cache/susy_m350_valid.npz",
           "vbfhto2tau": "data/real_cache/vbfhto2tau_valid.npz"}


def load(path):
    z = np.load(path)
    f = z["features"]
    nj = z["n_jet"] if "n_jet" in z.files else f[:, 0].copy()
    return f, z["pileup"], nj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor-cache", default="data/real_cache/zb396102_ordered.npz")
    p.add_argument("--calib-cache", default="data/real_cache/zb398183_ordered.npz")
    p.add_argument("--bkg-events", type=int, default=3_000_000)
    p.add_argument("--max-sig", type=int, default=0, help="subsample each signal to N events (0 = all)")
    p.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.001])
    p.add_argument("--primary-rate", type=float, default=0.01)
    p.add_argument("--drops", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.25, 0.40])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--design", default="pu_spline")
    p.add_argument("--out-dir", default="results/harm_map")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    cols = INJECTION_SETS["l1_calo"]
    rng = np.random.default_rng(args.seed)

    CF, CP, CJ = load(args.calib_cache)
    L = len(CF) // 4
    print(f"calibrating on {args.calib_cache} (first {L} events)...", flush=True)
    sc = build_split_calibration(CF[:L], CP[:L], CJ[:L], mode="three_way", seed=args.seed,
                                 vae_epochs=args.vae_epochs, design=args.design, n_knots=8)
    del CF, CP, CJ
    score = lambda X: score_chunked(sc.model, sc.scaler, X)

    BF, _, _ = load(args.monitor_cache)
    B = BF[rng.choice(len(BF), size=min(args.bkg_events, len(BF)), replace=False)]
    del BF
    sig = {}
    for name, path in SIGNALS.items():
        X, _, _ = load(path)
        if args.max_sig and len(X) > args.max_sig:
            X = X[rng.choice(len(X), size=args.max_sig, replace=False)]
        sig[name] = X
    names = ["nJet", "nEG", "nMu", "nTau", "jetET", "egET", "muET", "tauET", "HT", "MET", "MHT", "ETT"]
    print("sanity (column medians) -- signals should look like same-layout, higher-energy events:")
    print("  " + " ".join(f"{n:>7}" for n in names))
    for lab, X in [("bkg", B)] + list(sig.items()):
        print(f"  {np.median(X, axis=0).round(1).astype(str).tolist()}  <- {lab}")
    print(f"  background events: {len(B)}; signal events: { {k: len(v) for k, v in sig.items()} } "
          f"[{time.time() - t0:.0f}s]", flush=True)

    s_b0 = score(B)
    thr0 = {r: float(np.quantile(s_b0, 1 - r)) for r in args.rates}
    s_s0 = {k: score(v) for k, v in sig.items()}
    eff0 = {(k, r): float((s_s0[k] > thr0[r]).mean()) for k in sig for r in args.rates}
    for k in sig:
        print("  healthy signal efficiency " + k + ": " +
              ", ".join(f"rate {r:g}: {eff0[(k, r)]:.3f}" for r in args.rates))

    rows = []
    for drop in args.drops:
        s_bi = score(inject_masked_cols(B, drop, cols, 0.5))
        s_si = {k: score(inject_masked_cols(v, drop, cols, 0.5)) for k, v in sig.items()}
        for r in args.rates:
            thr_re = float(np.quantile(s_bi, 1 - r))
            rate_frozen = float((s_bi > thr0[r]).mean())
            for k in sig:
                e0 = eff0[(k, r)]
                ef = float((s_si[k] > thr0[r]).mean())
                er = float((s_si[k] > thr_re).mean())
                se = float(np.sqrt(max(er * (1 - er), 1e-12) / len(s_si[k])))
                rows.append({"drop": drop, "rate": r, "signal": k, "eff_healthy": e0, "eff_frozen": ef,
                             "eff_recal": er, "eff_recal_se": se, "bkg_rate_frozen": rate_frozen,
                             "rate_change_rel": rate_frozen / r - 1, "eff_recal_rel": er / e0 - 1,
                             "eff_frozen_rel": ef / e0 - 1})
        print(f"  drop {drop:g} done [{time.time() - t0:.0f}s]", flush=True)

    print("\n" + "=" * 118)
    print("HARM MAP (signal = simulated, PROXY). rel = change relative to healthy. Harm criterion applies at "
          f"rate {args.primary_rate:g}: recalibrated eff drop >10% OR frozen bkg rate change >20%")
    print(f"{'rate':>7}{'drop':>6}  {'signal':<12}{'bkg rate frozen':>16}{'rate chg':>10}{'eff healthy':>12}"
          f"{'eff frozen':>11}{'eff frozen rel':>15}{'eff recal':>10}{'eff recal rel':>14}  harm?")
    for r in args.rates:
        for drop in args.drops:
            for k in sig:
                x = next(q for q in rows if q["rate"] == r and q["drop"] == drop and q["signal"] == k)
                harm = ""
                if abs(r - args.primary_rate) < 1e-12:
                    flags = []
                    if abs(x["rate_change_rel"]) > 0.20: flags.append("bandwidth")
                    if x["eff_recal_rel"] < -0.10: flags.append("efficiency")
                    harm = "HARMFUL (" + "+".join(flags) + ")" if flags else "harmless"
                print(f"{r:>7g}{drop:>6g}  {k:<12}{x['bkg_rate_frozen']:>16.4f}{x['rate_change_rel']:>+10.1%}"
                      f"{x['eff_healthy']:>12.3f}{x['eff_frozen']:>11.3f}{x['eff_frozen_rel']:>+15.1%}"
                      f"{x['eff_recal']:>10.3f}{x['eff_recal_rel']:>+14.1%}  {harm}")
    path = os.path.join(args.out_dir, "harm_map.json")
    json.dump({"args": vars(args), "rows": rows}, open(path, "w"))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
