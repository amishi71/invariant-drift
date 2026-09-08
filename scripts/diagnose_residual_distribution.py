"""
scripts/diagnose_residual_distribution.py

Tests the hypothesis that real background residuals are more
autocorrelated or heavier-tailed than synthetic ones -- which would
explain why no (k,h) retune closes the FA-rate gap.

Usage:
    python3 scripts/diagnose_residual_distribution.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from main import build_calibration
from src.real_pipeline import build_real_calibration


def acf(x: np.ndarray, max_lag: int = 5) -> list:
    x = x - x.mean()
    denom = np.sum(x ** 2)
    return [float(np.sum(x[:-lag] * x[lag:]) / denom) for lag in range(1, max_lag + 1)]


def report(name: str, residuals: np.ndarray):
    print(f"\n=== {name} (n={len(residuals)}) ===")
    print(f"  mean={residuals.mean():.4f}  std={residuals.std():.4f}")
    print(f"  skewness:        {stats.skew(residuals):.4f}")
    print(f"  excess kurtosis: {stats.kurtosis(residuals):.4f}  "
          f"(0=Gaussian, >0=heavier tails than Gaussian)")
    lags = acf(residuals, max_lag=5)
    print(f"  ACF lag 1-5:     {[f'{v:.4f}' for v in lags]}")


def main():
    print("[1/2] Building synthetic burn-in (seed=0)...")
    _, _, _, synth_residuals = build_calibration(
        n_burn_in=3000, seed=0, vae_epochs=40, verbose=False,
    )

    print("[2/2] Building real burn-in (seed=0)...")
    _, _, _, real_residuals, _, _, _ = build_real_calibration(
        n_burn_in=3000, seed=0, vae_epochs=40, verbose=False,
    )

    report("Synthetic burn-in residuals", synth_residuals)
    report("Real burn-in residuals", real_residuals)

    print("\n=== Interpretation ===")
    print("If real ACF lag-1 is meaningfully larger than synthetic's,")
    print("that supports the autocorrelation hypothesis: CUSUM's false-")
    print("alarm-rate theory assumes i.i.d. noise, and positive")
    print("autocorrelation alone inflates false alarms at any (k,h).")
    print("If real excess kurtosis is meaningfully larger than synthetic's,")
    print("that supports the heavy-tails hypothesis instead (or as well).")


if __name__ == "__main__":
    main()
