#!/usr/bin/env python3
"""Lumisection-scale stability and detectability (Phase 3, minimal version).

Calibrate (VAE, spline regression, reference) on the start of the CALIB run, as in the cross-run protocol.
On the MONITOR run, compute per-lumisection statistics of the healthy stream, then of the same stream with
a masked-drop injection (l1_calo, requantize 0.5) applied to every event. Reports:
  * LS-to-LS spread of each statistic vs the pure sampling expectation, and lag autocorrelations;
  * per-LS shift of each statistic under each drop, divided by the healthy LS-to-LS spread (d').
Statistics: residual mean, raw log-score mean, and tail fractions above the healthy q90/q99/q99.9 of the
raw log-score (quantiles from a large healthy sample of the calibration run).
Tail fractions are score-tail PROXIES on a sampled event set, not CMS trigger accept rates.
Quick test:  python3 scripts/ls_stability.py --max-ls 40 --vae-epochs 5
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.calibration_split import build_split_calibration, residualize, score_chunked
from src.injection_sets import INJECTION_SETS, inject_masked_cols

LEVELS = (0.90, 0.99, 0.999)
STATS = ["resid_mean_z", "logscore_mean_z"] + [f"tail_q{100 * lv:g}" for lv in LEVELS]


def load(path):
    z = np.load(path)
    f, pu = z["features"], z["pileup"]
    nj = z["n_jet"] if "n_jet" in z.files else f[:, 0].copy()
    lumi = z["lumi"] if "lumi" in z.files else None
    run = z["run"] if "run" in z.files else None
    return f, pu, nj, lumi, run


def ls_stats(inv, n_ls, resid, logs, ref, qs):
    n = np.bincount(inv, minlength=n_ls).astype(np.float64)
    mean = lambda x: np.bincount(inv, weights=x, minlength=n_ls) / n
    rz = (resid - ref["r_mean"]) / ref["r_std"]
    lz = (logs - ref["l_mean"]) / ref["l_std"]
    out = {"n": n, "resid_mean_z": mean(rz), "logscore_mean_z": mean(lz)}
    out["resid_std"] = np.sqrt(np.maximum(mean(rz ** 2) - out["resid_mean_z"] ** 2, 0))
    out["logscore_std"] = np.sqrt(np.maximum(mean(lz ** 2) - out["logscore_mean_z"] ** 2, 0))
    for lv, q in qs.items():
        out[f"tail_q{100 * lv:g}"] = mean((logs > q).astype(np.float64))
    return out


def sampling_sd(st, name):
    n = st["n"]
    if name == "resid_mean_z":
        return float(np.sqrt(np.mean(st["resid_std"] ** 2 / n)))
    if name == "logscore_mean_z":
        return float(np.sqrt(np.mean(st["logscore_std"] ** 2 / n)))
    p = st[name]
    return float(np.sqrt(np.mean(p * (1 - p) / n)))


def acf(x, lag):
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    d = float((x * x).sum())
    return float((x[:-lag] * x[lag:]).sum() / d) if d > 0 and len(x) > lag else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor-cache", default="data/real_cache/zb396102_ordered.npz")
    p.add_argument("--calib-cache", default="data/real_cache/zb398183_ordered.npz")
    p.add_argument("--max-ls", type=int, default=0, help="use only the first N lumisections (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--design", default="pu_spline")
    p.add_argument("--n-knots", type=int, default=8)
    p.add_argument("--ref-sample", type=int, default=400_000)
    p.add_argument("--drops", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.25, 0.40])
    p.add_argument("--out-dir", default="results/ls_stability")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    cols = INJECTION_SETS["l1_calo"]

    # ---- calibration on the first quarter of the calibration run (as in the cross-run experiments) ----
    CF, CP, CJ, _, _ = load(args.calib_cache)
    L = len(CF) // 4
    CF, CP, CJ = CF[:L], CP[:L], CJ[:L]
    print(f"calibrating on {args.calib_cache} (first {L} events)...", flush=True)
    sc = build_split_calibration(CF, CP, CJ, mode="three_way", seed=args.seed, vae_epochs=args.vae_epochs,
                                 design=args.design, n_knots=args.n_knots)
    reserve = 40_000
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(np.arange(reserve, L), size=min(args.ref_sample, L - reserve), replace=False)
    lref = np.log(np.clip(score_chunked(sc.model, sc.scaler, CF[idx]), 1e-12, None))
    qs = {lv: float(np.quantile(lref, lv)) for lv in LEVELS}
    ref = {"r_mean": float(sc.ref_residuals.mean()), "r_std": float(sc.ref_residuals.std()),
           "l_mean": float(lref.mean()), "l_std": float(lref.std())}
    print(f"  healthy log-score quantiles from {len(lref)} events: {qs}  [{time.time() - t0:.0f}s]", flush=True)
    del CF, CP, CJ

    # ---- monitored run, grouped by lumisection ----
    F, P, J, lumi, run = load(args.monitor_cache)
    if lumi is None:
        raise SystemExit("monitor cache has no 'lumi' array")
    key = lumi.astype(np.int64) if run is None else run.astype(np.int64) * 1_000_000 + lumi.astype(np.int64)
    if args.max_ls:
        keep = np.isin(key, np.unique(key)[:args.max_ls])
        F, P, J, key = F[keep], P[keep], J[keep], key[keep]
    ukeys, inv = np.unique(key, return_inverse=True)
    n_ls = len(ukeys)
    print(f"monitoring {args.monitor_cache}: {len(F)} events in {n_ls} lumisections "
          f"(median {int(np.median(np.bincount(inv)))} events/LS)", flush=True)

    scores, resid = residualize(sc, F, P, J)
    healthy = ls_stats(inv, n_ls, resid, np.log(np.clip(scores, 1e-12, None)), ref, qs)
    pu_ls = np.bincount(inv, weights=P.astype(np.float64), minlength=n_ls) / healthy["n"]
    print(f"  healthy LS stats done [{time.time() - t0:.0f}s]. Pileup per LS: "
          f"{pu_ls.min():.1f}..{pu_ls.max():.1f}", flush=True)

    # ---- healthy stability table ----
    long_sd, short_sd = {}, {}
    print("\n" + "=" * 110)
    print("HEALTHY LS-LEVEL STABILITY (monitored run, calibration from the other run)")
    print(f"{'statistic':<18}{'LS mean':>10}{'long-term SD':>14}{'short-term SD':>15}{'sampling SD':>13}"
          f"{'short/sampling':>16}{'acf1':>7}{'acf5':>7}{'acf10':>7}{'total drift':>13}")
    for name in STATS:
        x = healthy[name]
        long_sd[name] = float(x.std())
        short_sd[name] = float(np.diff(x).std() / np.sqrt(2)) if n_ls > 2 else float("nan")
        samp = sampling_sd(healthy, name)
        drift = float(np.polyfit(np.arange(n_ls), x, 1)[0] * n_ls)
        print(f"{name:<18}{x.mean():>10.4f}{long_sd[name]:>14.4f}{short_sd[name]:>15.4f}{samp:>13.4f}"
              f"{short_sd[name] / samp:>16.2f}{acf(x, 1):>7.2f}{acf(x, 5):>7.2f}{acf(x, 10):>7.2f}{drift:>13.4f}")
    print("short/sampling >> 1 means LS-to-LS variation is not sampling noise (systematic); ~1 means "
          "statistics-limited. Large acf = neighbouring LS are not independent.")

    # ---- injected: per-LS shift relative to healthy LS-to-LS spread ----
    table = {}
    for drop in args.drops:
        Finj = inject_masked_cols(F, drop, cols, 0.5)
        s_i, r_i = residualize(sc, Finj, P, J)
        del Finj
        inj = ls_stats(inv, n_ls, r_i, np.log(np.clip(s_i, 1e-12, None)), ref, qs)
        table[drop] = {}
        for name in STATS:
            delta = float(np.mean(inj[name] - healthy[name]))
            dl, ds = abs(delta) / long_sd[name], abs(delta) / short_sd[name]
            table[drop][name] = {"delta": delta, "d_long": dl, "d_short": ds,
                                 "ls_to_3sigma": (3.0 / dl) ** 2 if dl > 0 else float("inf")}
        print(f"  drop {drop:g} done [{time.time() - t0:.0f}s]", flush=True)

    print("\n" + "=" * 110)
    print("PER-LS DETECTABILITY: mean shift under injection divided by healthy LS-to-LS SD (d')")
    print(f"{'drop':>6}  " + "".join(f"{n:>26}" for n in STATS))
    print(f"{'':>6}  " + "".join(f"{'shift | d_long | d_short':>26}" for _ in STATS))
    for drop in args.drops:
        row = f"{drop:>6g}  "
        for name in STATS:
            t = table[drop][name]
            row += f"{t['delta']:>+8.3f}|{t['d_long']:>5.1f}|{t['d_short']:>5.1f}   "
        print(row)
    print(f"\nApprox. LS needed to reach 3 sigma if LS statistics were independent: (3/d_long)^2 "
          f"(run has {n_ls} LS; neighbouring-LS correlation makes this optimistic)")
    print(f"{'drop':>6}  " + "".join(f"{n:>26}" for n in STATS))
    for drop in args.drops:
        row = f"{drop:>6g}  "
        for name in STATS:
            v = table[drop][name]["ls_to_3sigma"]
            row += f"{('>%d' % n_ls) if v > n_ls else format(v, '.1f'):>26}"
        print(row)

    # ---- save ----
    out = {"monitor": args.monitor_cache, "calib": args.calib_cache, "n_ls": n_ls, "quantiles": qs,
           "healthy": {k: v.tolist() for k, v in healthy.items()}, "pileup_per_ls": pu_ls.tolist(),
           "long_sd": long_sd, "short_sd": short_sd,
           "detectability": {str(k): v for k, v in table.items()}}
    path = os.path.join(args.out_dir, "ls_stability.json")
    json.dump(out, open(path, "w"))
    print(f"\nwrote {path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        ax[0].plot(healthy["resid_mean_z"], lw=0.8)
        ax[0].set_xlabel("lumisection index"); ax[0].set_ylabel("mean residual (z)")
        ax[0].set_title("Healthy per-LS mean residual")
        w = 0.8 / len(STATS)
        for i, name in enumerate(STATS):
            ax[1].bar(np.arange(len(args.drops)) + i * w, [max(table[d][name]["d_long"], 1e-3) for d in args.drops],
                      w, label=name)
        ax[1].set_xticks(np.arange(len(args.drops)) + 0.4); ax[1].set_xticklabels([f"{d:g}" for d in args.drops])
        ax[1].set_yscale("log"); ax[1].axhline(3, color="k", ls="--", lw=0.8)
        ax[1].set_xlabel("injected drop"); ax[1].set_ylabel("d' = shift / LS-to-LS SD")
        ax[1].set_title("Per-LS detectability by statistic"); ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "ls_stability.png"), dpi=150)
        print("wrote ls_stability.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
