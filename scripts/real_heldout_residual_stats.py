"""
scripts/real_heldout_residual_stats.py

Answers the question the previous results left open: why does CUSUM false-alarm
more often on real windows than on synthetic ones, and is it (a) a small offset
or scale change between the calibration residuals and held-out residuals, (b)
heavy tails, (c) autocorrelation, or (d) slow non-stationarity across the data?

It scores every full window of the pool that follows the burn-in with a frozen
VAE and regression, then reports, per calibration mode:

  * Held-out residual mean, std and excess kurtosis (pooled and per window).
  * The held-out offset and scale RELATIVE TO THE REFERENCE the detector uses:
    z-mean = (held-out mean - ref mean) / ref std, z-std = held-out std / ref std.
    These two numbers are what drives false alarms.
  * Whether window means vary more than i.i.d. noise allows (ratio >> 1 means
    slow drift or long-range correlation), plus the within-window ACF at several
    lags, a linear trend of window mean against position, and the correlation of
    window mean with window mean pileup.
  * Whether held-out pileup stays inside the range the regression was fit on.
  * Empirical CUSUM false-alarm probability (all windows, and by position) next to
    the Gaussian Monte Carlo prediction made from the z-mean/z-std alone. If the
    two agree, offset and scale explain the false alarms; if empirical is higher,
    tails, autocorrelation or non-stationarity matter.

Modes: "insample" (the original pipeline: VAE, regression and reference all on one
block) and "three_way" (VAE / regression / reference on three disjoint blocks).
Both modes are scored on the SAME windows, so the difference between them is the
effect of the in-sample calibration. A synthetic pool (--source synthetic) is a
stationary control: a correct procedure should give nominal numbers there.

Note: scripts/background_far_vs_distance.py already exists in the repo and may
overlap with the position analysis here; compare the two.

Usage:
    python3 scripts/real_heldout_residual_stats.py
    python3 scripts/real_heldout_residual_stats.py --source synthetic --synthetic-events 1500000
    python3 scripts/real_heldout_residual_stats.py --plot
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.calibration_split import MODES, build_split_calibration, residualize, reserved_events
from src.matched_far import (
    RealPool, SyntheticPool, cusum_runmax, first_passage, gaussian_far, wilson, zscore,
)


def excess_kurtosis_rows(x):
    m = x.mean(axis=1, keepdims=True)
    d = x - m
    m2 = (d ** 2).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    return m4 / np.maximum(m2 ** 2, 1e-300) - 3.0


def within_window_acf(R, lags):
    d = R - R.mean(axis=1, keepdims=True)
    denom = (d ** 2).sum(axis=1)
    out = {}
    for lag in lags:
        if lag >= R.shape[1]:
            continue
        num = (d[:, :-lag] * d[:, lag:]).sum(axis=1)
        out[lag] = float(np.mean(num / np.maximum(denom, 1e-300)))
    return out


def analyse_mode(mode, F, P, J, args):
    sc = build_split_calibration(
        F, P, J, mode=mode, n_vae=args.n_vae, n_reg=args.n_reg, n_ref=args.n_ref,
        seed=args.seed, vae_epochs=args.vae_epochs, verbose=False)
    reserve = reserved_events(args.n_vae, args.n_reg, args.n_ref)
    W = args.window
    n_win = (len(F) - reserve) // W
    if args.max_windows:
        n_win = min(n_win, args.max_windows)
    if n_win < 20:
        raise SystemExit("Fewer than 20 windows available; reduce --window or burn-in sizes.")
    lo, hi = reserve, reserve + n_win * W
    print(f"[{mode}] scoring {n_win} windows of {W} events "
          f"({hi - lo:,} events) with the frozen VAE/regression...")
    _, resid = residualize(sc, F[lo:hi], P[lo:hi], J[lo:hi])
    R = resid.reshape(n_win, W)
    PU = P[lo:hi].reshape(n_win, W)

    ref_mean, ref_std = float(sc.ref_residuals.mean()), float(sc.ref_residuals.std())
    w_mean, w_std = R.mean(axis=1), R.std(axis=1)
    w_kurt = excess_kurtosis_rows(R)
    pooled_mean, pooled_std = float(R.mean()), float(R.std())
    within_sd = float(np.sqrt(np.mean(w_std ** 2)))
    iid_sd_of_means = within_sd / np.sqrt(W)
    ratio = float(w_mean.std(ddof=1) / iid_sd_of_means)

    idx = np.arange(n_win)
    lr = stats.linregress(idx, w_mean)
    pu_w = PU.mean(axis=1)
    pu_corr = float(np.corrcoef(pu_w, w_mean)[0, 1]) if pu_w.std() > 0 else float("nan")

    b0, b1 = sc.blocks["regression"]
    burn_pu = P[b0:b1]
    p_lo, p_hi = np.percentile(burn_pu, [1, 99])
    frac_events_outside = float(np.mean((P[lo:hi] < burn_pu.min()) | (P[lo:hi] > burn_pu.max())))
    frac_windows_outside = float(np.mean((pu_w < p_lo) | (pu_w > p_hi)))

    z_mean = (pooled_mean - ref_mean) / ref_std
    z_std = pooled_std / ref_std

    # Empirical vs Gaussian-predicted false alarms.
    z = zscore(R, sc.ref_residuals)
    far_rows = []
    for h in args.h_values:
        for T in args.far_windows:
            if T > W:
                continue
            fp = first_passage(cusum_runmax(z[:, :T], args.k, "two"), [h])[:, 0]
            alarm = fp < T
            k_alarm, n = int(alarm.sum()), len(alarm)
            lo_ci, hi_ci = wilson(k_alarm, n)
            thirds = np.array_split(idx, 3)
            by_pos = [float(alarm[t].mean()) for t in thirds]
            rho, p = stats.spearmanr(idx, alarm.astype(float)) if alarm.any() and not alarm.all() \
                else (float("nan"), float("nan"))
            gauss = gaussian_far(h, args.k, T, mean=z_mean, std=z_std,
                                 trials=args.mc_trials, seed=args.seed)
            ideal = gaussian_far(h, args.k, T, trials=args.mc_trials, seed=args.seed)
            far_rows.append({
                "h": h, "T": T, "empirical": k_alarm / n, "ci95": [lo_ci, hi_ci],
                "n_windows": n, "by_position_thirds": by_pos,
                "spearman_position_vs_alarm": [float(rho), float(p)],
                "gaussian_pred_from_offset_scale": gauss, "gaussian_ideal_0_1": ideal})

    res = {
        "mode": mode, "n_windows": n_win, "window": W,
        "blocks": {k: list(v) for k, v in sc.blocks.items()},
        "reference": {"mean": ref_mean, "std": ref_std},
        "heldout_pooled": {"mean": pooled_mean, "std": pooled_std,
                           "excess_kurtosis_mean_of_windows": float(w_kurt.mean()),
                           "excess_kurtosis_pooled": float(stats.kurtosis(R.ravel()))},
        "relative_to_reference": {"z_mean": float(z_mean), "z_std": float(z_std)},
        "window_means": {"sd": float(w_mean.std(ddof=1)), "iid_expected_sd": float(iid_sd_of_means),
                         "ratio_observed_to_iid": ratio},
        "acf_within_window": within_window_acf(R, [1, 2, 5, 10, 50, 100, 500]),
        "trend_window_mean": {"slope_per_window": float(lr.slope),
                              "slope_per_million_events": float(lr.slope * 1e6 / W),
                              "r": float(lr.rvalue), "p": float(lr.pvalue)},
        "corr_window_mean_vs_window_pileup": pu_corr,
        "pileup_coverage": {"regression_block_range": [float(burn_pu.min()), float(burn_pu.max())],
                            "frac_events_outside_range": frac_events_outside,
                            "frac_window_means_outside_p1_p99": frac_windows_outside},
        "cusum_far": far_rows,
        "per_window": {"mean": w_mean.tolist(), "std": w_std.tolist(),
                       "excess_kurtosis": w_kurt.tolist(), "pileup_mean": pu_w.tolist()},
    }
    return res


def print_mode(r):
    print(f"\n=== {r['mode']} ({r['n_windows']} windows of {r['window']}) ===")
    ref, hp, rel = r["reference"], r["heldout_pooled"], r["relative_to_reference"]
    print(f"  reference residual (block {r['blocks']['reference']}):  mean {ref['mean']:+.4f}  std {ref['std']:.4f}")
    print(f"  held-out residual, pooled:                   mean {hp['mean']:+.4f}  std {hp['std']:.4f}  "
          f"excess kurtosis {hp['excess_kurtosis_pooled']:+.3f}")
    print(f"  relative to reference (what CUSUM sees):     z-mean {rel['z_mean']:+.4f}  z-std {rel['z_std']:.4f}")
    wm = r["window_means"]
    print(f"  window means: sd {wm['sd']:.4f} vs {wm['iid_expected_sd']:.4f} expected if i.i.d. "
          f"(ratio {wm['ratio_observed_to_iid']:.2f}; >> 1 means slow drift / long-range correlation)")
    print("  within-window ACF:  " + "  ".join(f"lag{k}={v:+.3f}" for k, v in r["acf_within_window"].items()))
    t = r["trend_window_mean"]
    print(f"  trend of window mean over the pool: {t['slope_per_million_events']:+.4f} per million events "
          f"(r={t['r']:+.3f}, p={t['p']:.3g})")
    print(f"  corr(window mean residual, window mean pileup) = {r['corr_window_mean_vs_window_pileup']:+.3f}")
    pc = r["pileup_coverage"]
    print(f"  pileup coverage: regression saw {pc['regression_block_range'][0]:.0f}-"
          f"{pc['regression_block_range'][1]:.0f}; {100*pc['frac_events_outside_range']:.1f}% of held-out "
          f"events and {100*pc['frac_window_means_outside_p1_p99']:.1f}% of window means fall outside")
    print(f"  {'h':>5} {'T':>6} {'empirical FAR [95% CI]':>26} {'Gauss pred (offset+scale)':>26} "
          f"{'Gauss ideal(0,1)':>17}  by position (thirds)")
    for f in r["cusum_far"]:
        print(f"  {f['h']:5.1f} {f['T']:6d} {f['empirical']:9.3f} [{f['ci95'][0]:.3f},{f['ci95'][1]:.3f}]"
              f"{f['gaussian_pred_from_offset_scale']:19.3f}{f['gaussian_ideal_0_1']:18.3f}  "
              + " / ".join(f"{x:.2f}" for x in f["by_position_thirds"]))


def make_plot(results, path, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for r in results:
        x = np.arange(r["n_windows"]) * W / 1e6
        axes[0].plot(x, r["per_window"]["mean"], ".", ms=3, label=r["mode"])
        axes[0].axhline(r["reference"]["mean"], lw=0.8, ls="--")
        axes[1].plot(x, r["per_window"]["std"], ".", ms=3, label=r["mode"])
        axes[1].axhline(r["reference"]["std"], lw=0.8, ls="--")
    axes[2].plot(x, results[0]["per_window"]["pileup_mean"], ".", ms=3, color="k")
    axes[0].set_ylabel("window mean residual"); axes[1].set_ylabel("window std")
    axes[2].set_ylabel("window mean pileup"); axes[2].set_xlabel("events after burn-in (millions)")
    axes[0].legend(); axes[0].set_title("Held-out residual by position in the pool (dashed = reference)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["real", "synthetic"], default="real")
    p.add_argument("--cache-path", default="data/real_cache/jetht_features.npz")
    p.add_argument("--synthetic-events", type=int, default=1_500_000)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=["insample", "three_way"])
    p.add_argument("--n-vae", type=int, default=20_000)
    p.add_argument("--n-reg", type=int, default=10_000)
    p.add_argument("--n-ref", type=int, default=10_000)
    p.add_argument("--window", type=int, default=5000)
    p.add_argument("--max-windows", type=int, default=0, help="0 = all windows in the pool")
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--h-values", type=float, nargs="+", default=[8.0, 10.0, 11.0])
    p.add_argument("--far-windows", type=int, nargs="+", default=[2000, 5000])
    p.add_argument("--mc-trials", type=int, default=10_000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default="results/real_heldout_residual_stats.json")
    args = p.parse_args()

    pool = RealPool(args.cache_path) if args.source == "real" else SyntheticPool(args.synthetic_events)
    F, P, J = pool.region(0, 1)
    print(f"Pool: {pool.name}, {len(F):,} events.")

    results = [analyse_mode(m, F, P, J, args) for m in args.modes]
    for r in results:
        print_mode(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nWritten to {args.out}")
    if args.plot:
        png = os.path.splitext(args.out)[0] + ".png"
        make_plot(results, png, args.window)
        print(f"Plot written to {png}")

    print("\nHow to read this:")
    print("  - If z-mean/z-std differ from 0/1 and 'Gauss pred' ~ 'empirical', an offset or scale")
    print("    change explains the false alarms; the three_way mode should shrink it.")
    print("  - If empirical >> 'Gauss pred', look at the ratio (non-stationarity), ACF (correlation)")
    print("    and kurtosis (tails). A large ratio or a significant trend means the pool is not stationary.")
    print("  - On --source synthetic everything should look nominal; if not, the procedure is at fault.")


if __name__ == "__main__":
    main()
