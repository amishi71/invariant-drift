"""Real-data equivalent of main.py's build_calibration + score_stream,
using the cached real feature pool (data/real_cache/jetht_features.npz)
instead of synthetic_object_stream. Reuses train_proxy_vae, anomaly_score,
and CalibrationModel UNCHANGED -- only the data source differs.
"""
import numpy as np
from src.proxy_vae import train_proxy_vae, anomaly_score
from src.residual import CalibrationModel
from src.drift_sim.real_data_injection import load_real_feature_pool


def build_real_calibration(n_burn_in: int, seed: int, vae_epochs: int, verbose: bool,
                            cache_path: str = "data/real_cache/jetht_features.npz"):
    features, pileup, n_jet = load_real_feature_pool(cache_path)
    if n_burn_in > len(features):
        raise ValueError(f"n_burn_in={n_burn_in} exceeds real pool size {len(features)}")

    # Burn-in: first n_burn_in real events, treated as the "correctly
    # calibrated" reference period (same convention as the synthetic
    # burn-in -- an assumption, not a verified claim that these
    # specific events are drift-free; real JetHT is presumed stable
    # over this short a window, not verified independently here).
    feats = features[:n_burn_in]
    pu = pileup[:n_burn_in]
    nj = n_jet[:n_burn_in]
    lumi = np.ones(n_burn_in)  # no real lumi trend available, see module docstring

    model, scaler, history = train_proxy_vae(feats, epochs=vae_epochs, seed=seed, verbose=verbose)
    scores = anomaly_score(model, scaler, feats)
    calib = CalibrationModel.fit(scores, pu, nj, lumi)
    residuals = calib.residual(scores, pu, nj, lumi)

    print(f"[real burn-in] {n_burn_in} events, VAE loss={history[-1]:.4f}, "
          f"residual mean={residuals.mean():.3f} std={residuals.std():.3f}")
    return model, scaler, calib, residuals, features, pileup, n_jet


def score_real_segment(model, scaler, calib, features, pileup, n_jet, start, n_events):
    """Scores a real segment [start:start+n_events) -- no drift injection,
    for stability/ARL checks on genuinely untouched real data."""
    feats = features[start:start + n_events]
    pu = pileup[start:start + n_events]
    nj = n_jet[start:start + n_events]
    lumi = np.ones(n_events)
    scores = anomaly_score(model, scaler, feats)
    residuals = calib.residual(scores, pu, nj, lumi)
    return scores, residuals
