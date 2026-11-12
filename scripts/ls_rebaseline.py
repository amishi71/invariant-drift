#!/usr/bin/env python3
"""Fixed-reference vs re-baselined LS-level CUSUM, offline from ls_stability.json (no VAE).
Healthy LS series are bootstrapped (iid; acf ~ 0). Injected LS = healthy draw + measured shift (shift-only model).
h is set per policy so healthy P(alarm within --tcal LS) = --alpha. Detection = P(alarm within --horizon LS)."""
import argparse, json, numpy as np

NOM = {"tail_q90": 0.1, "tail_q99": 0.01, "tail_q99.9": 0.001}
STATS = ["resid_mean_z", "logscore_mean_z", "tail_q90", "tail_q99"]

def get_shift(v):
    if "delta" in v: return float(v["delta"])
    if "shift" in v: return float(v["shift"])
    for k in v:
        if "shift" in k: return float(v[k])
    raise KeyError(f"no shift key among {list(v)}")

def paths_max(Z, k):
    N, T = Z.shape
    sp = np.zeros(N); sn = np.zeros(N); run = np.zeros(N); m = np.empty((N, T))
    for t in range(T):
        sp = np.maximum(0, sp + Z[:, t] - k); sn = np.maximum(0, sn - Z[:, t] - k)
        run = np.maximum(run, np.maximum(sp, sn)); m[:, t] = run
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tcal", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--warm", type=int, nargs="+", default=[1, 5, 30])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for path in a.json:
        d = json.load(open(path)); H = d["healthy"]
        drops = sorted(d["detectability"], key=float)
        print("\n" + "=" * 110 + f"\n{path}  ({d['n_ls']} LS)  k={a.k}  FAR budget {a.alpha:g} per {a.tcal} LS  detection horizon {a.horizon} LS")
        for s in STATS:
            x = np.asarray(H[s], float); sd = float(d["long_sd"][s]); ref0 = NOM.get(s, 0.0)
            shifts = {dr: get_shift(d["detectability"][dr][s]) for dr in drops}
            print(f"\n[{s}]  healthy offset vs nominal reference = {(x.mean()-ref0)/sd:+.1f} LS-SD")
            print(f"{'policy':<14}{'h':>9}{'FAR@H':>7}" + "".join(f"{'drop '+dr:>11}" for dr in drops))
            absorbed = []
            for pol in [-1, 0] + a.warm:
                W = rng.integers(0, len(x), (a.n, pol)) if pol > 0 else None
                ref = np.full(a.n, ref0) if pol <= 0 else x[W].mean(1)
                refc = np.full(a.n, x.mean()) if pol == -1 else ref   # strict: h set on a run with zero offset
                mh = paths_max((x[rng.integers(0, len(x), (a.n, a.tcal))] - refc[:, None]) / sd, a.k)
                h = float(np.quantile(mh[:, -1], 1 - a.alpha))
                if pol == -1:
                    mh = paths_max((x[rng.integers(0, len(x), (a.n, a.tcal))] - ref[:, None]) / sd, a.k)
                far = float((mh[:, a.horizon - 1] > h).mean())
                row, ab = [], []
                for dr in drops:
                    sh = shifts[dr]
                    mf = paths_max((x[rng.integers(0, len(x), (a.n, a.horizon))] + sh - ref[:, None]) / sd, a.k)
                    row.append(float((mf[:, -1] > h).mean()))
                    if pol > 0:
                        ref_a = (x[W] + sh).mean(1)  # fault already present during warm-up
                        ma = paths_max((x[rng.integers(0, len(x), (a.n, a.horizon))] + sh - ref_a[:, None]) / sd, a.k)
                        ab.append(float((ma[:, -1] > h).mean()))
                name = {-1: "fixed-strict", 0: "fixed-oracle"}.get(pol, f"rebase n={pol}")
                print(f"{name:<14}{h:>9.1f}{100*far:>6.1f}%" + "".join(f"{100*r:>10.0f}%" for r in row))
                if pol > 0: absorbed.append((name, ab))
            print("  fault ALREADY PRESENT during warm-up (absorption):")
            for name, ab in absorbed:
                print(f"{name:<14}{'':>16}" + "".join(f"{100*r:>10.0f}%" for r in ab))

main()
