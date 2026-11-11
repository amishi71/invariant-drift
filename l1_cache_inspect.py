"""
l1_cache_inspect.py -- read-only look at the L1 feature caches in data/real_cache.

For each cache it prints the arrays inside, feature statistics, NaN/inf counts, and whether the rows look
time-ordered or randomly shuffled (the dataset's provided split is a random permutation, so a cache built
straight from it is NOT chronological).

    python3 l1_cache_inspect.py
    python3 l1_cache_inspect.py data/real_cache/zb396102_train.npz
"""
import glob
import os
import sys

import numpy as np


def runs_of_constant(x):
    """Number of maximal runs of equal consecutive values."""
    if len(x) == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(x) != 0))


def inspect(path):
    print("=" * 100)
    print("%s  (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    z = np.load(path, allow_pickle=False)
    for k in z.files:
        a = z[k]
        line = "  %-12s shape %-14s dtype %-9s" % (k, a.shape, a.dtype)
        if a.dtype.kind in "fiu" and a.size:
            af = a.astype(np.float64)
            line += " min %.4g max %.4g mean %.4g" % (np.nanmin(af), np.nanmax(af), np.nanmean(af))
        print(line)
    if "features" in z.files:
        F = z["features"].astype(np.float64)
        print("  features: %d rows x %d cols; NaN %d, inf %d, all-zero rows %d" % (
            F.shape[0], F.shape[1], int(np.isnan(F).sum()), int(np.isinf(F).sum()),
            int((F == 0).all(axis=1).sum())))
        print("  per-column mean/std: " + "  ".join(
            "c%d %.3g/%.3g" % (j, np.nanmean(F[:, j]), np.nanstd(F[:, j])) for j in range(F.shape[1])))
    if "pileup" in z.files:
        p = z["pileup"].astype(np.float64)
        n = len(p)
        runs = runs_of_constant(p)
        print("  pileup: %d distinct values; %d constant runs in file order" % (len(np.unique(p)), runs))
        # Time-ordered data with a per-lumi-section pileup has FEW long runs; a random shuffle has about
        # n * (1 - sum(freq^2)) runs, i.e. close to n.
        vals, cnt = np.unique(p, return_counts=True)
        expected_random = n * (1.0 - float(((cnt / n) ** 2).sum()))
        verdict = "looks TIME-ORDERED (few long runs)" if runs < 0.2 * expected_random else \
                  "looks RANDOMLY SHUFFLED (about as many runs as a shuffle would give)"
        print("  expected runs if shuffled ~%.0f, observed %d -> %s" % (expected_random, runs, verdict))
        if n > 1000:
            q = n // 4
            print("  mean pileup by quarter of file: " + ", ".join("%.2f" % p[i * q:(i + 1) * q].mean() for i in range(4)))
    for k in ("lumi", "time", "orbit", "run", "order"):
        if k in z.files:
            a = z[k].astype(np.float64)
            mono = bool(np.all(np.diff(a) >= 0)) if len(a) > 1 else True
            print("  %s: monotone non-decreasing in file order: %s" % (k, mono))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("data/real_cache/*.npz"))
    if not paths:
        sys.exit("no .npz files found; run from the repo root")
    for p in paths:
        try:
            inspect(p)
        except Exception as e:  # noqa
            print("%s: could not read (%s: %s)" % (p, type(e).__name__, e))


if __name__ == "__main__":
    main()
