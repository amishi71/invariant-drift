"""
scripts/diagnose_residual_distribution_robust.py

Cheap follow-up to diagnose_residual_distribution.py: CalibrationModel.fit()
already supports robust=True (MAD-based sigma_resid instead of plain std),
but no script has tried it. Before trusting calibrate_h's bootstrap-based
threshold calibration as the fix for the false-alarm-rate gap, check
whether this free flag already shrinks the excess-kurtosis gap between
synthetic and real residuals.

Uses build_calibration()/build_real_calibration()'s EXISTING return values
directly -- no new kwargs needed. Both already hand back the trained
model/scaler; refitting with robust=True only requires recomputing scores
through that same model and re-running CalibrationModel.fit() with
robust=True, using the exact same burn-in covariates each function already
used internally.

Usage:
    python3 scripts/diagnose_residual_distribution_robust.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from main import build_calibration, _collect
from src.stream_loader import synthetic_object_stream
from src.real_pipeline import build_real_calibration
from src.proxy_vae import anomaly_score
from src.residual import CalibrationModel


def refit_robust_synthetic(model, scaler, n_burn_in, seed, score_transform="log"):
    """Regenerate the SAME synthetic burn-in stream build_calibration()
    used internally (identical seed -> identical stream by construction),
    then refit with robust=True using the already-trained model/scaler.
    """
    feats, pileup, n_jet, lumi, _ = _collect(
        synthetic_object_stream(n_burn_in, seed=seed), n_burn_in,
    )
    scores = anomaly_score(model, scaler, feats)
    calib_robust = CalibrationModel.fit(
        scores, pileup, n_jet, lumi, robust=True, score_transform=score_transform,
    )
    return calib_robust.residual(scores, pileup, n_jet, lumi)


def refit_robust_real(model, scaler, features, pileup, n_jet, n_burn_in,
                       score_transform="log"):
    """build_real_calibration() returns the FULL real feature pool, not
    just burn-in -- slice the first n_burn_in ourselves (same slice it
    used internally) and refit with robust=True.
    """
    feats = features[:n_burn_in]
    pu = pileup[:n_burn_in]
    nj = n_jet[:n_burn_in]
    lumi = np.ones(n_burn_in)  # matches build_real_calibration's convention

    scores = anomaly_score(model, scaler, feats)
    calib_robust = CalibrationModel.fit(
        scores, pu, nj, lumi, robust=True, score_transform=score_transform,
    )
    return calib_robust.residual(scores, pu, nj, lumi)


def report(name: str, residuals: np.ndarray):
    print(f"\n=== {name} (n={len(residuals)}) ===")
    print(f"  mean={residuals.mean():.4f}  std={residuals.std():.4f}")
    print(f"  skewness:        {stats.skew(residuals):.4f}")
    print(f"  excess kurtosis: {stats.kurtosis(residuals):.4f}  "
          f"(0=Gaussian, >0=heavier tails than Gaussian)")


def main():
    n_burn_in = 3000
    seed = 0
    vae_epochs = 40

    print("[1/2] Synthetic burn-in (seed=0)...")
    model_s, scaler_s, calib_s, synth_residuals = build_calibration(
        n_burn_in=n_burn_in, seed=seed, vae_epochs=vae_epochs, verbose=False,
    )
    synth_robust_residuals = refit_robust_synthetic(
        model_s, scaler_s, n_burn_in, seed,
        score_transform=calib_s.score_transform,
    )

    print("[2/2] Real burn-in (seed=0)...")
    model_r, scaler_r, calib_r, real_residuals, features, pileup, n_jet = \
        build_real_calibration(n_burn_in, seed, vae_epochs, False)
    real_robust_residuals = refit_robust_real(
        model_r, scaler_r, features, pileup, n_jet, n_burn_in,
        score_transform=calib_r.score_transform,
    )

    report("Synthetic, std-based (current default)", synth_residuals)
    report("Synthetic, MAD-based (robust=True)", synth_robust_residuals)
    report("Real, std-based (current default)", real_residuals)
    report("Real, MAD-based (robust=True)", real_robust_residuals)

    print("\n=== Interpretation ===")
    print("Compare excess kurtosis: std-based vs MAD-based, within each")
    print("substrate. If MAD-based kurtosis on REAL residuals drops")
    print("substantially toward the synthetic number, robust=True alone")
    print("closes a meaningful chunk of the gap for free -- consider")
    print("making it the default in build_real_calibration.")
    print("If it barely moves, the heavy tails are dense (not a few")
    print("outliers dragging up the std estimate), and calibrate_h's")
    print("empirical-bootstrap approach remains the right tool -- just")
    print("make sure --n-arl-windows is large enough to actually verify")
    print("the target FAR it's calibrating against (see terminal output")
    print("from calibrated_threshold_experiment.py: 20 windows only")
    print("resolves FAR to steps of 0.05).")


if __name__ == "__main__":
    main()
