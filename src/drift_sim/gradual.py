"""Gradual drift: pileup evolving across an LHC fill.

Instantaneous luminosity (and with it, pileup) falls across a fill as
bunch currents burn off. A standard, widely-used approximate model for
this "leveling"/burn-off curve (see e.g. CMS/ATLAS luminosity public
results) is a hyperbolic decay:

    pileup(t) = pileup0 / (1 + t / tau) ** power

with tau a characteristic decay time and power ~ O(1). This module
provides TWO distinct scenarios built on that same mechanism, matching the
abstract's "simulated gradual drift (pileup evolving across a fill)" and
its separate "robustness when the assumed drift model is misspecified"
check:

1. `nominal_gradual_stream` -- pileup (and the luminosity-trend covariate
   fed to residual.py) evolve via the decay law, and the underlying
   object-generation physics (stream_loader._synthesize_object_features)
   is untouched. Since residual.py's calibration model was fit as a
   function OF pileup/mult/lumi (not of wall-clock time), this is exactly
   the covariate movement the calibration is supposed to already absorb.
   Expected detector behavior: residual stays ~flat -- this is a targeted
   FALSE-ALARM stress test, not a drift-detection test.

2. `misspecified_gradual_stream` -- the SAME pileup/lumi decay law, but
   with an additional slow, physically-motivated bias layered on top of
   the generated objects (a jet-response/HT scale that itself drifts
   across the fill, e.g. from a slowly shifting calorimeter calibration)
   that is NOT a function of pileup alone. This is a genuine calibration
   break riding on top of otherwise-normal fill evolution -- the harder
   and more realistic failure mode than an abrupt jump. Expected detector
   behavior: residual should show a growing trend once the bias magnitude
   exceeds detector sensitivity.
"""

from typing import Iterator, Optional

import numpy as np

from src.stream_loader import synthetic_object_stream


def _pileup_decay(pileup0: float, t: np.ndarray, tau: float, power: float) -> np.ndarray:
    return pileup0 / (1.0 + t / tau) ** power


def _make_condition_fn(pileup0: float, tau: float, power: float, n_jet_base: float):
    def condition_fn(rng, i):
        pileup = np.array([_pileup_decay(pileup0, float(i), tau, power)])
        pileup = np.clip(pileup, 5.0, None)
        lumi = pileup / pileup0  # lumi tracks pileup by construction of the decay model
        n_jet = np.array([max(2.0, rng.poisson(lam=n_jet_base + 0.05 * pileup[0]))])
        return {"pileup": pileup, "n_jet": n_jet, "lumi": lumi}
    return condition_fn


def nominal_gradual_stream(
    n_events: int,
    pileup0: float = 45.0,
    tau: float = 4000.0,
    power: float = 0.7,
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Pileup/lumi decay only; object-generation physics unchanged.
    Ground-truth label is always 0 (no genuine calibration break) --
    included for interface symmetry with misspecified_gradual_stream /
    abrupt.py, used only in evaluation.py, never fed to a detector.
    """
    condition_fn = _make_condition_fn(pileup0, tau, power, n_jet_base=3.0)
    for i, event in enumerate(synthetic_object_stream(
        n_events, seed=seed, condition_fn=condition_fn,
    )):
        event["true_label"] = 0
        yield event


def misspecified_gradual_stream(
    n_events: int,
    pileup0: float = 45.0,
    tau: float = 4000.0,
    power: float = 0.7,
    bias_onset_frac: float = 0.3,
    bias_rate: float = 0.02,
    bias_features: tuple = (0, 3, 9),  # jet1_pt, jet2_pt, ht (see DEFAULT_FEATURE_NAMES)
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Same pileup/lumi decay as nominal_gradual_stream, PLUS a linearly
    growing additive bias on jet pT / HT starting at bias_onset_frac of the
    way through the stream, growing at bias_rate (in feature units per
    event, e.g. GeV/event) after onset. This represents a real, independent
    detector-response drift (e.g. jet energy scale) that the calibration
    model was never told about -- ground truth true_label flips to 1 at
    the onset event for evaluation purposes only.
    """
    onset_event = int(bias_onset_frac * n_events)

    def feature_bias_fn(rng, i, features):
        if i < onset_event:
            return features
        magnitude = bias_rate * (i - onset_event)
        biased = features.copy()
        for col in bias_features:
            biased[:, col] = biased[:, col] + magnitude
        return biased

    condition_fn = _make_condition_fn(pileup0, tau, power, n_jet_base=3.0)
    for i, event in enumerate(synthetic_object_stream(
        n_events, seed=seed, condition_fn=condition_fn, feature_bias_fn=feature_bias_fn,
    )):
        event["true_label"] = 0 if i < onset_event else 1
        event["true_onset_event"] = onset_event
        yield event
