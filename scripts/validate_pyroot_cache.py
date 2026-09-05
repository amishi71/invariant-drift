"""
scripts/validate_pyroot_cache.py

Compares the PyROOT/RDataFrame-derived feature cache
(data/real_cache/jetht_features_pyroot.npz, from
scripts/build_real_cache_pyroot.py) against the existing uproot-derived
cache (data/real_cache/jetht_features.npz, from src/stream_loader.py's
real_object_stream) to confirm the rebuild reproduces the same features
from the same underlying ROOT files -- not just "runs without error."

Run this in the MAIN project venv (not venv-root) -- it's pure numpy,
no ROOT import needed.

Usage:
    python3 scripts/validate_pyroot_cache.py
    python3 scripts/validate_pyroot_cache.py \\
        --uproot-cache data/real_cache/jetht_features.npz \\
        --pyroot-cache data/real_cache/jetht_features_pyroot.npz
"""
import argparse

import numpy as np

DEFAULT_FEATURE_NAMES = [
    "jet1_pt", "jet1_eta", "jet1_phi", "jet2_pt", "jet2_eta", "jet2_phi",
    "n_jet", "met_pt", "met_phi", "ht", "n_muon", "n_electron",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uproot-cache", type=str,
                         default="data/real_cache/jetht_features.npz")
    parser.add_argument("--pyroot-cache", type=str,
                         default="data/real_cache/jetht_features_pyroot.npz")
    parser.add_argument("--atol", type=float, default=1e-3,
                         help="Absolute tolerance for float comparison -- "
                              "loose enough to absorb float32(ROOT)/float64 "
                              "(uproot->numpy) rounding, tight enough to catch "
                              "a real logic mismatch.")
    args = parser.parse_args()

    up = np.load(args.uproot_cache)
    pr = np.load(args.pyroot_cache)

    print(f"uproot cache:  {args.uproot_cache}")
    print(f"  features: {up['features'].shape}, pileup: {up['pileup'].shape}, "
          f"n_jet: {up['n_jet'].shape}")
    print(f"pyroot cache:  {args.pyroot_cache}")
    print(f"  features: {pr['features'].shape}, pileup: {pr['pileup'].shape}, "
          f"n_jet: {pr['n_jet'].shape}")

    n_up, n_pr = len(up["features"]), len(pr["features"])
    if n_up != n_pr:
        print(f"\nWARNING: event counts differ (uproot={n_up}, pyroot={n_pr}). "
              f"This means the two extraction paths selected a different "
              f"number of events -- check whether both read the SAME file "
              f"list, in the SAME order, with the SAME nJet>=2 filter and "
              f"min_jet_pt, before trusting a row-by-row comparison below. "
              f"Comparing only the first {min(n_up, n_pr)} events as a "
              f"partial check.")
    n = min(n_up, n_pr)

    print(f"\n=== Per-feature comparison (first {n} events) ===")
    print(f"{'feature':12s} {'max_abs_diff':>14s} {'mean_abs_diff':>14s} {'within_atol':>12s}")
    all_ok = True
    for i, name in enumerate(DEFAULT_FEATURE_NAMES):
        a = up["features"][:n, i]
        b = pr["features"][:n, i]
        diff = np.abs(a - b)
        max_diff, mean_diff = diff.max(), diff.mean()
        ok = max_diff <= args.atol
        all_ok &= ok
        print(f"{name:12s} {max_diff:14.6f} {mean_diff:14.6f} {str(ok):>12s}")

    pileup_diff = np.abs(up["pileup"][:n] - pr["pileup"][:n])
    n_jet_diff = np.abs(up["n_jet"][:n] - pr["n_jet"][:n])
    print(f"{'pileup':12s} {pileup_diff.max():14.6f} {pileup_diff.mean():14.6f} "
          f"{str(pileup_diff.max() <= args.atol):>12s}")
    print(f"{'n_jet':12s} {n_jet_diff.max():14.6f} {n_jet_diff.mean():14.6f} "
          f"{str(n_jet_diff.max() <= args.atol):>12s}")
    all_ok &= pileup_diff.max() <= args.atol
    all_ok &= n_jet_diff.max() <= args.atol

    print()
    if n_up == n_pr and all_ok:
        print("PASS: event counts match and every feature agrees within "
              "tolerance. The PyROOT extraction reproduces the uproot "
              "extraction faithfully -- safe to treat as a drop-in "
              "replacement for data/real_cache/jetht_features.npz.")
    elif all_ok:
        print("PARTIAL PASS: features agree within tolerance on the "
              "overlapping events, but event COUNTS differ -- resolve why "
              "before treating this as validated (see the warning above).")
    else:
        print("FAIL: at least one feature disagrees beyond tolerance. Do "
              "NOT treat the PyROOT cache as validated yet -- check the "
              "flagged feature(s) above against build_real_cache_pyroot.py's "
              "column definitions (e.g. HT's min_jet_pt cut, or a jet1/jet2 "
              "ordering difference between uproot's and ROOT's jet arrays).")


if __name__ == "__main__":
    main()
