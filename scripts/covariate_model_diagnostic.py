"""
scripts/covariate_model_diagnostic.py

Step 1 of the covariate-model experiment (cheap; run BEFORE any matched-FAR rerun).

Question. On real data, window means of the calibration residual vary ~4x more than
i.i.d. noise allows and correlate with pileup, at about half the size of the
masked-channel signal. Is that a covariate-model problem (fixable with a better
pileup/n_jet regression) or slow variation the covariates cannot explain?

Method. Train ONE proxy VAE (block A) and score every event of the region ONCE. Then,
on those same frozen scores, fit each calibration design (src/residual_models.py) from
each calibration source and measure the held-out residual on the same monitoring
windows. Only the regression changes between rows, so differences are due to the
covariate model and the fit sample, not to VAE training noise.

Calibration sources
  insample   regression and reference on block A (the original pipeline)
  three_way  regression on block B, reference on block C (contiguous, causal)
  span       regression and reference on two disjoint random samples of events spread
             across the region, taken from slots that belong to NO monitoring window
             (non-causal: an offline-calibration variant; label it as such)

Per row it reports (windows of --window events, default 5000 as in the ledger):
  z-mean / z-std     held-out residual relative to the detector reference
  ratio              sd of window means / what i.i.d. noise would give (target < 1.5)
  corr_pu, corr_nj   corr of window-mean residual with window-mean pileup / n_jet
  lag1               lag-1 autocorrelation of the window-mean sequence (>0 = slow drift)
  binned_sd          sd across pileup-decile means of the held-out residual (z units)
  FAR(h=8)           empirical two-sided CUSUM k=0.5 false-alarm probability within the
                     window (ideal i.i.d. Gaussian: about 0.41 at 5000 events)
  shift, SNR         mean residual shift caused by a 40% masked drop (z units) and
                     shift / sd-of-window-means (higher = easier to detect)
  in_range           fraction of held-out events inside the fit sample's pileup range

Reading the result
  * ratio falls toward 1 and SNR rises  -> the slow variation was a covariate problem;
    proceed to the matched-FAR rerun (scripts/matched_far_heldout.py --design ...).
  * ratio stays well above 1 for every design, including span  -> the variation is not
    explained by pileup/n_jet; the real-data power ceiling is set by something else
    (prescales, trigger mix, detector conditions). Report that; do not keep tuning.
  * improvement only in "span"  -> causal burn-ins cannot see the full pileup range;
    report span as offline calibration.

Usage (macOS, venv root):
    python3 scripts/covariate_model_diagnostic.py                      # whole real pool
    python3 scripts/covariate_model_diagnostic.py --n-regions 3 --region 0   # faster
    python3 scripts/covariate_model_diagnostic.py --source synthetic   # stationary control
    python3 scripts/covariate_model_diagnostic.py --plot
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.matched_far import (RealPool, SyntheticPool, cusum_runmax, first_passage,
                             gaussian_far, inject_masked, wilson, zscore)
from src.residual_models import DESIGNS, fit_calibration

SOURCES = ("insample", "three_way", "span")


# ----------------------------------------------------------------------------
# Pure-numpy analysis (unit-tested without torch)
# ----------------------------------------------------------------------------
def residuals_chunked(calib, S, P, J, lo, hi, chunk=1_000_000):
    out = np.empty(hi - lo)
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        out[i - lo:j - lo] = calib.residual(S[i:j], P[i:j], J[i:j], np.ones(j - i))
    return out


def fit_source(design, source, S, P, J, blocks, span_idx, n_knots):
    """Returns (calib, ref_residuals, fit_pileup)."""
    if source == "span":
        reg, ref = span_idx
    else:
        reg = np.arange(*blocks[source]["regression"])
        ref = np.arange(*blocks[source]["reference"])
    calib = fit_calibration(design, S[reg], P[reg], J[reg], np.ones(len(reg)), n_knots=n_knots)
    ref_res = calib.residual(S[ref], P[ref], J[ref], np.ones(len(ref)))
    return calib, ref_res, P[reg]


def window_metrics(R, PU, NJ, ref_res, fit_pu, S_inj_resid=None, R_same=None,
                   far_h=8.0, far_k=0.5, bins=10):
    """R, PU, NJ: (n_win, W) held-out residuals / pileup / n_jet. ref_res: reference
    residuals the detector standardizes by. S_inj_resid, R_same: residuals of the
    drop-injected windows and of the same windows without injection."""
    ref_mean, ref_std = float(ref_res.mean()), float(ref_res.std())
    Z = (R - ref_mean) / ref_std
    n_win, W = Z.shape
    wm = Z.mean(axis=1)
    within_sd = float(np.sqrt(np.mean(Z.std(axis=1) ** 2)))
    iid_sd = within_sd / np.sqrt(W)
    sd_wm = float(wm.std(ddof=1))
    pu_w, nj_w = PU.mean(axis=1), NJ.mean(axis=1)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    lag1 = corr(wm[:-1], wm[1:])
    # pileup-binned mean of the held-out residual (event level)
    pu_flat, z_flat = PU.ravel(), Z.ravel()
    edges = np.unique(np.quantile(pu_flat, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(pu_flat, edges[1:-1]), 0, len(edges) - 2)
    bin_means = np.array([z_flat[which == b].mean() for b in range(len(edges) - 1)])
    bin_pu = np.array([pu_flat[which == b].mean() for b in range(len(edges) - 1)])
    lo, hi = fit_pu.min(), fit_pu.max()
    fp = first_passage(cusum_runmax(Z, far_k, "two"), [far_h])[:, 0]
    k_alarm = int((fp < W).sum())
    out = {
        "n_windows": int(n_win), "window": int(W),
        "ref_mean": ref_mean, "ref_std": ref_std,
        "z_mean": float(Z.mean()), "z_std": float(Z.std()),
        "excess_kurtosis": float(stats.kurtosis(z_flat)),
        "window_mean_sd": sd_wm, "iid_expected_sd": float(iid_sd),
        "ratio": float(sd_wm / iid_sd),
        "corr_pu": corr(pu_w, wm), "corr_nj": corr(nj_w, wm), "lag1_window_means": lag1,
        "binned_sd": float(bin_means.std()), "binned_max_abs": float(np.abs(bin_means).max()),
        "binned": {"pileup": bin_pu.tolist(), "z_mean": bin_means.tolist()},
        "far": k_alarm / n_win, "far_ci95": list(wilson(k_alarm, n_win)),
        "in_range": float(np.mean((pu_flat >= lo) & (pu_flat <= hi))),
        "window_means": wm.tolist(), "window_pileup": pu_w.tolist(),
    }
    if S_inj_resid is not None and R_same is not None:
        shift = float((S_inj_resid.mean() - R_same.mean()) / ref_std)
        out["shift"] = shift
        out["snr"] = float(abs(shift) / sd_wm)
        out["snr_iid"] = float(abs(shift) / iid_sd)
    return out


# ----------------------------------------------------------------------------
# VAE-dependent part (needs torch; kept out of the pure analysis above)
# ----------------------------------------------------------------------------
def build_scores(F, P, J, args):
    from src.calibration_split import score_chunked
    from src.proxy_vae import train_proxy_vae
    a = (0, args.n_vae)
    print(f"Training the proxy VAE on block A ({args.n_vae:,} events, {args.vae_epochs} epochs)...",
          flush=True)
    model, scaler, hist = train_proxy_vae(F[a[0]:a[1]], epochs=args.vae_epochs,
                                          seed=args.seed, verbose=False)
    print(f"  final VAE loss {float(hist[-1]):.4f}. Scoring {len(F):,} events once...", flush=True)
    return model, scaler, score_chunked(model, scaler, F)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["real", "synthetic"], default="real")
    p.add_argument("--cache-path", default="data/real_cache/jetht_features.npz")
    p.add_argument("--synthetic-events", type=int, default=1_500_000)
    p.add_argument("--n-regions", type=int, default=1, help="1 = whole pool (comparable to ledger 6.3)")
    p.add_argument("--region", type=int, default=0)
    p.add_argument("--designs", nargs="+", choices=list(DESIGNS), default=list(DESIGNS))
    p.add_argument("--sources", nargs="+", choices=list(SOURCES), default=["insample", "three_way", "span"])
    p.add_argument("--n-knots", type=int, default=8)
    p.add_argument("--n-vae", type=int, default=20_000)
    p.add_argument("--n-reg", type=int, default=10_000)
    p.add_argument("--n-ref", type=int, default=10_000)
    p.add_argument("--window", type=int, default=5000)
    p.add_argument("--max-windows", type=int, default=0, help="0 = all windows in the region")
    p.add_argument("--n-inject", type=int, default=100, help="windows used for the masked-drop shift")
    p.add_argument("--drop", type=float, default=0.4)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc-trials", type=int, default=10_000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default="results/covariate_model_diagnostic.json")
    args = p.parse_args()

    pool = RealPool(args.cache_path) if args.source == "real" else SyntheticPool(args.synthetic_events)
    F, P, J = pool.region(args.region, args.n_regions)
    P, J = np.asarray(P, dtype=np.float64), np.asarray(J, dtype=np.float64)
    W = args.window
    reserve = args.n_vae + args.n_reg + args.n_ref
    n_slots = (len(F) - reserve) // W
    if args.max_windows:
        n_slots = min(n_slots, args.max_windows + 8)
    n_span_slots = int(np.ceil((args.n_reg + args.n_ref) / W)) + 1
    if n_slots < n_span_slots + 20:
        raise SystemExit("Too few windows; reduce --window or burn-in sizes.")
    print(f"Pool {pool.name}: {len(F):,} events, {n_slots} slots of {W}; "
          f"{n_span_slots} slots held out as the 'span' fit/reference pool.")

    rng = np.random.default_rng(7000 + args.seed)
    span_slots = np.sort(rng.choice(n_slots, n_span_slots, replace=False))
    win_slots = np.setdiff1d(np.arange(n_slots), span_slots)
    ev = lambda slots: (reserve + slots[:, None] * W + np.arange(W)[None, :])
    span_pool = ev(span_slots).ravel()
    pick = rng.choice(len(span_pool), args.n_reg + args.n_ref, replace=False)
    span_idx = (np.sort(span_pool[pick[:args.n_reg]]), np.sort(span_pool[pick[args.n_reg:]]))

    model, scaler, S = build_scores(F, P, J, args)
    a = (0, args.n_vae)
    blocks = {"insample": {"regression": a, "reference": a},
              "three_way": {"regression": (a[1], a[1] + args.n_reg),
                            "reference": (a[1] + args.n_reg, a[1] + args.n_reg + args.n_ref)}}

    Widx = ev(win_slots)                         # (n_win, W) event indices of monitoring windows
    PU, NJ = P[Widx], J[Widx]
    inj_rows = np.sort(rng.choice(len(win_slots), min(args.n_inject, len(win_slots)), replace=False))
    from src.calibration_split import score_chunked
    F_inj = inject_masked(F[Widx[inj_rows]], args.drop)
    S_inj = score_chunked(model, scaler, F_inj.reshape(-1, F.shape[1])).reshape(len(inj_rows), W)

    ideal = gaussian_far(8.0, 0.5, W, trials=args.mc_trials, seed=args.seed)
    print(f"Ideal i.i.d. Gaussian CUSUM (k=0.5, h=8) false-alarm probability within {W}: {ideal:.3f}\n")

    rows = []
    for source in args.sources:
        for design in args.designs:
            calib, ref_res, fit_pu = fit_source(design, source, S, P, J, blocks, span_idx, args.n_knots)
            R = residuals_chunked(calib, S, P, J, reserve, reserve + n_slots * W).reshape(n_slots, W)[win_slots]
            Rinj = calib.residual(S_inj.ravel(), PU[inj_rows].ravel(), NJ[inj_rows].ravel(),
                                  np.ones(S_inj.size))
            m = window_metrics(R, PU, NJ, ref_res, fit_pu, S_inj_resid=Rinj, R_same=R[inj_rows])
            m.update(source=source, design=design, ideal_far=ideal)
            rows.append(m)

    hdr = (f"{'source':<10}{'design':<16}{'z-mean':>8}{'z-std':>7}{'ratio':>7}{'corr_pu':>8}{'corr_nj':>8}"
           f"{'lag1':>6}{'bin_sd':>7}{'FAR(h=8)':>10}{'shift':>8}{'SNR':>6}{'in_rng':>7}")
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(f"{m['source']:<10}{m['design']:<16}{m['z_mean']:>+8.3f}{m['z_std']:>7.3f}{m['ratio']:>7.2f}"
              f"{m['corr_pu']:>+8.2f}{m['corr_nj']:>+8.2f}{m['lag1_window_means']:>+6.2f}"
              f"{m['binned_sd']:>7.3f}{m['far']:>10.3f}{m['shift']:>+8.3f}{m['snr']:>6.2f}{m['in_range']:>7.2f}")
    print(f"\nTargets: ratio < 1.5, |corr_pu| ~ 0, FAR(h=8) near {ideal:.2f}, SNR as high as possible "
          f"(i.i.d. ceiling = shift / iid_sd = {np.mean([m['snr_iid'] for m in rows]):.1f}).")

    base = [m for m in rows if m["design"] == "quad"]
    best = min(rows, key=lambda m: m["ratio"])
    print(f"\nBaseline (quad) ratios: " + ", ".join(f"{m['source']} {m['ratio']:.2f}" for m in base))
    print(f"Lowest ratio overall: {best['source']} / {best['design']} = {best['ratio']:.2f}")
    if best["ratio"] > 2.5:
        print("VERDICT: window means stay far from i.i.d. for every design -> the slow variation is not "
              "explained by pileup/n_jet. Do not rerun matched-FAR expecting a large gain.")
    elif best["ratio"] < 1.5:
        print("VERDICT: a covariate model removes most of the slow variation -> proceed to the matched-FAR rerun.")
    else:
        print("VERDICT: partial improvement; the matched-FAR rerun is worth doing but expect a moderate gain.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "config": vars(args), "ideal_far": ideal}, f, indent=2, default=str)
    print(f"Written to {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        srcs = args.sources
        fig, axes = plt.subplots(2, len(srcs), figsize=(5 * len(srcs), 7), squeeze=False)
        for c, source in enumerate(srcs):
            for m in (r for r in rows if r["source"] == source):
                axes[0, c].plot(m["binned"]["pileup"], m["binned"]["z_mean"], "o-", ms=3, label=m["design"])
                axes[1, c].plot(m["window_means"], ".", ms=2, alpha=0.5, label=m["design"]) \
                    if m["design"] in ("quad", best["design"]) else None
            axes[0, c].set_title(f"{source}: held-out residual vs pileup (decile means)")
            axes[0, c].set_xlabel("pileup"); axes[0, c].set_ylabel("mean z"); axes[0, c].axhline(0, c="k", lw=0.5)
            axes[1, c].set_title(f"{source}: window means by position"); axes[1, c].set_xlabel("window")
        axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=130)
        print(f"Plot written to {png}")


if __name__ == "__main__":
    main()
