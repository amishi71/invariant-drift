"""Applies the SAME drift-injection logic used in abrupt.py and
radiation_damage.py, but onto a pool of REAL CMS Open Data features
(cached via data/real_cache/jetht_features.npz) instead of the
synthetic generator.

Design: real Open Data gives no ground-truth drift onset (a real quiet
run has no labeled "calibration broke here" event), so ARL/latency/
coverage/FDR metrics -- which all require a known onset -- still need
injected, controlled drift. This module reuses the exact bias functions
already validated in abrupt.py/radiation_damage.py against synthetic
data, applying them to real feature ARRAYS directly rather than
regenerating features from scratch. This is the same "real substrate,
synthetic controlled perturbation" pattern used in the Kepler PDC
papers cited in this project's related work (Stumpe et al. 2012, 2014).

IMPORTANT SCOPE NOTE: only feature-level drift (masked-channel,
radiation-damage-style decay) is supported here, not the lumi-trend
gradual scenarios in gradual.py -- a single real Open Data record has
no meaningful instantaneous-luminosity trend to perturb (see README/
docs for why). gradual.py's scenarios remain synthetic-only.
"""

import numpy as np
from typing import Iterator, Dict, Optional, Tuple


def load_real_feature_pool(
    cache_path: str = "data/real_cache/jetht_features.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the cached real feature pool. Returns (features, pileup, n_jet)."""
    data = np.load(cache_path)
    return data["features"], data["pileup"], data["n_jet"]


def real_segment_stream(
    features: np.ndarray,
    pileup: np.ndarray,
    n_jet: np.ndarray,
    start: int,
    n_events: int,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, object]]:
    """Yields n_events real events starting at index `start` in the
    pooled arrays, in the same dict shape synthetic_object_stream uses
    (so score_stream/_collect work unmodified). `seed` is accepted for
    call-signature parity with the synthetic scenario functions but is
    NOT used to generate anything here -- real data has no RNG; it's
    only used, if at all, by the drift wrappers below for reproducible
    injected noise.
    """
    end = start + n_events
    if end > len(features):
        raise ValueError(
            f"Requested real segment [{start}:{end}] exceeds pool size "
            f"{len(features)}. Use a smaller n_events or start earlier."
        )
    for i in range(start, end):
        yield {
            "features": features[i].copy(),
            "pileup": float(pileup[i]),
            "n_jet": float(n_jet[i]),
            "lumi": 1.0,  # no real lumi-trend available; held constant, see module docstring
            "feature_names": None,
        }


def real_masked_channel_stream(
    features: np.ndarray,
    pileup: np.ndarray,
    n_jet: np.ndarray,
    start: int,
    n_events: int,
    changepoint_event: int,
    drop_fraction: float = 0.4,
    affected_features: tuple = (0, 3, 9, 7),  # jet1_pt, jet2_pt, ht, met_pt
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Real-data equivalent of drift_sim.abrupt.masked_channel_stream:
    identical multiplicative-drop injection logic, applied to a real
    feature segment instead of synthetic. Ground truth (true_label,
    true_onset_event) is synthetic metadata about WHERE the injection
    happens, same as the synthetic version -- this is standard practice
    for controlled-perturbation studies on real data (cf. Kepler PDC).
    """
    for i, event in enumerate(real_segment_stream(
        features, pileup, n_jet, start, n_events, seed=seed,
    )):
        if i >= changepoint_event:
            feats = event["features"].copy()
            for col in affected_features:
                feats[col] = feats[col] * (1.0 - drop_fraction)
            event["features"] = feats
        event["true_label"] = 0 if i < changepoint_event else 1
        event["true_onset_event"] = changepoint_event
        yield event


def real_radiation_damage_stream(
    features: np.ndarray,
    pileup: np.ndarray,
    n_jet: np.ndarray,
    start: int,
    n_events: int,
    gain0: float = 1.0,
    tau_damage: float = 3000.0,
    affected_features: tuple = (0, 3, 9, 7),
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Real-data equivalent of drift_sim.radiation_damage.
    radiation_damage_stream -- IDENTICAL formula and onset-labeling rule,
    applied to a real feature segment instead of synthetic:
    gain_factor(i) = gain0 * exp(-i / tau_damage), permanent from event 0,
    no reset. true_label/true_onset_event use the SAME detectability-floor
    convention as the synthetic version (5% cumulative gain loss), NOT an
    arbitrary placeholder -- see radiation_damage.py's own docstring for
    why this floor, not a discrete physical onset, is the defensible
    reference point for latency metrics here.
    """
    detectability_floor = 0.05
    onset = int(-tau_damage * np.log(1.0 - detectability_floor))

    for i, event in enumerate(real_segment_stream(
        features, pileup, n_jet, start, n_events, seed=seed,
    )):
        gain_factor = gain0 * np.exp(-i / tau_damage)
        feats = event["features"].copy()
        for col in affected_features:
            feats[col] = feats[col] * gain_factor
        event["features"] = feats
        event["true_label"] = 0 if i < onset else 1
        event["true_onset_event"] = onset
        event["gain_factor"] = float(gain_factor)
        yield event
