"""Burn-in construction with explicit control over which events each stage sees.

The calibration pipeline has three stages that each consume burn-in events:

  (1) train the proxy VAE,
  (2) fit the calibration regression of log-score on (pileup, mult, lumi),
  (3) freeze the reference residual statistics (mean/std) that detectors
      standardize by and that thresholds are calibrated against.

The original pipeline (src/real_pipeline.build_real_calibration and
main.build_calibration) uses the SAME events for all three. Residuals on those
events are in-sample twice over: the VAE has seen them, and the regression was
fit to them (mean 0 and std 1 by construction). Held-out events are scored by
the same frozen VAE and regression but are not forced to have mean 0 / std 1,
and the detector's reference statistics never saw how the held-out residual
actually behaves.

This module offers two modes so the effect can be measured directly on the same
data:

  "insample"   VAE, regression and reference all use block A (the original
               behaviour, but with an explicit, shared block size).
  "three_way"  VAE trains on block A; the regression is fit on block B (scored
               by the frozen VAE, so out-of-sample for the VAE); the reference
               statistics come from block C (out-of-sample for both). Blocks are
               contiguous and disjoint, in time order, as in a deployment where
               calibration happens before monitoring starts.

A third mode, "span", exists for the covariate-model experiment (see
src/residual_models.py): VAE trains on block A (as in "three_way"), but the
regression and the reference statistics come from two DISJOINT random samples of
events spread across the whole region (drawn by the caller from events that are not
in any monitoring window). Its pileup coverage matches the held-out windows, at the
price of being non-causal (a deployed monitor would not have future events). Report
it as an offline-calibration variant next to the causal modes.

Nothing here modifies the existing pipeline; it is additive. With the default
design="quad" the behaviour is unchanged.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from src.proxy_vae import train_proxy_vae, anomaly_score
from src.residual import CalibrationModel
from src.residual_models import fit_calibration

MODES = ("insample", "three_way", "span")


@dataclass
class SplitCalibration:
    model: object
    scaler: object
    calib: object  # CalibrationModel (design "quad") or FlexCalibrationModel
    mode: str
    blocks: Dict[str, Tuple[int, int]]
    ref_scores: np.ndarray
    ref_residuals: np.ndarray
    reserved_end: int = 0
    info: dict = field(default_factory=dict)

    @property
    def ref_logscores(self) -> np.ndarray:
        return np.log(np.clip(self.ref_scores, 1e-12, None))


def score_chunked(model, scaler, features: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(features), dtype=np.float64)
    for i in range(0, len(features), chunk):
        out[i:i + chunk] = anomaly_score(model, scaler, features[i:i + chunk])
    return out


def residualize(sc: SplitCalibration, features: np.ndarray, pileup: np.ndarray,
                n_jet: np.ndarray, chunk: int = 500_000):
    """Scores arbitrary events with the frozen VAE and returns (scores, residuals)
    using the frozen regression. lumi is held at 1.0 as everywhere in this
    project's pool-based pipelines."""
    n = len(features)
    scores = np.empty(n, dtype=np.float64)
    resid = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):  # chunked: the design matrix for 8.8M events is ~1 GB
        s = anomaly_score(sc.model, sc.scaler, features[i:i + chunk])
        scores[i:i + chunk] = s
        resid[i:i + chunk] = sc.calib.residual(
            s, pileup[i:i + chunk], n_jet[i:i + chunk], np.ones(len(s)))
    return scores, resid


def reserved_events(n_vae: int, n_reg: int, n_ref: int) -> int:
    """Events reserved at the start of a region for burn-in, in BOTH modes, so
    that monitoring windows are identical across modes (paired comparison)."""
    return n_vae + n_reg + n_ref


def build_split_calibration(
    features: np.ndarray, pileup: np.ndarray, n_jet: np.ndarray,
    mode: str = "three_way", n_vae: int = 20_000, n_reg: int = 10_000,
    n_ref: int = 10_000, seed: int = 0, vae_epochs: int = 40,
    start: int = 0, verbose: bool = False,
    design: str = "quad", n_knots: int = 8, span_idx=None,
) -> SplitCalibration:
    """span_idx: for mode "span" only, a pair (reg_idx, ref_idx) of disjoint index
    arrays into `features`, drawn by the caller from events outside every window
    that will be monitored."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "span":
        if span_idx is None:
            raise ValueError('mode "span" needs span_idx=(reg_idx, ref_idx)')
        reg_idx, ref_idx = (np.sort(np.asarray(i, dtype=np.int64)) for i in span_idx)
        if len(np.intersect1d(reg_idx, ref_idx)) or reg_idx.max() >= len(features) \
                or ref_idx.max() >= len(features):
            raise ValueError("span_idx arrays must be disjoint and inside the region")
    a = (start, start + n_vae)
    if mode == "span":
        b = (int(reg_idx.min()), int(reg_idx.max()) + 1)  # extents only; for reporting
        c = (int(ref_idx.min()), int(ref_idx.max()) + 1)
    elif mode == "three_way":
        b = (a[1], a[1] + n_reg)
        c = (b[1], b[1] + n_ref)
    else:
        b = c = a
    # "span" reserves the same leading events as "three_way" so that monitoring
    # windows are identical across modes (paired comparison); the contiguous B and C
    # blocks are simply left unused in that mode.
    reserved_end = start + reserved_events(n_vae, n_reg, n_ref)
    if reserved_end > len(features):
        raise ValueError(
            f"Need {reserved_end - start} burn-in events from index {start}, "
            f"but only {len(features) - start} are available.")

    model, scaler, history = train_proxy_vae(
        features[a[0]:a[1]], epochs=vae_epochs, seed=seed, verbose=verbose)

    def sl(arr, blk):
        return arr[blk[0]:blk[1]]

    if mode == "span":
        scores_b = score_chunked(model, scaler, features[reg_idx])
        pu_b, nj_b = pileup[reg_idx], n_jet[reg_idx]
        scores_c = score_chunked(model, scaler, features[ref_idx])
        pu_c, nj_c = pileup[ref_idx], n_jet[ref_idx]
    else:
        scores_b = score_chunked(model, scaler, sl(features, b))
        pu_b, nj_b = sl(pileup, b), sl(n_jet, b)
        scores_c = score_chunked(model, scaler, sl(features, c))
        pu_c, nj_c = sl(pileup, c), sl(n_jet, c)
    calib = fit_calibration(design, scores_b, pu_b, nj_b, np.ones(len(scores_b)),
                            n_knots=n_knots)
    ref_resid = calib.residual(scores_c, pu_c, nj_c, np.ones(len(scores_c)))

    return SplitCalibration(
        model=model, scaler=scaler, calib=calib, mode=mode,
        blocks={"vae": a, "regression": b, "reference": c},
        ref_scores=scores_c, ref_residuals=ref_resid, reserved_end=reserved_end,
        info={"vae_final_loss": float(history[-1]) if len(history) else None,
              "design": design,
              "ref_mean": float(ref_resid.mean()), "ref_std": float(ref_resid.std())},
    )
