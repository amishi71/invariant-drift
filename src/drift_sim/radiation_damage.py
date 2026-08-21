"""Permanent, monotonic gain degradation -- the calibration-drift mechanism
long-duration space-based particle/photon detectors actually experience
(accumulated radiation damage degrading silicon sensor response; CCD/CMOS
sensitivity loss over a mission lifetime), as opposed to gradual.py's
LHC-fill-specific pileup evolution.

This is intentionally NOT just a relabeling of gradual.py's mechanism.
Three structural differences make it a genuinely distinct stress test, not
a cosmetic rename:

1. **Never resets.** LHC pileup evolution (gradual.py) resets at the start
   of every fill; a real fill-to-fill "burn-in" recalibration can track it.
   Radiation damage is cumulative and permanent -- there is no reset event,
   which is exactly the harder monitoring regime (a detector cannot be
   "refilled" to reset its dose history).
2. **No abrupt component.** Unlike abrupt.py's masked-channel scenario,
   there's no single failure event -- degradation is smooth from the first
   event onward, testing whether a detector calibrated once at mission
   start can catch a threat that never produces a sharp edge to lock onto.
3. **Operating covariates stay stable.** Pileup/multiplicity/luminosity
   are LHC-specific concepts with no space-detector analog; here they're
   held at fixed nominal values (see `_stable_condition_fn`) and the drift
   is injected purely as a feature-level bias, isolating "does the
   calibration-relationship-monitoring approach generalize to a mechanism
   with no LHC-specific covariate structure at all" from any pileup-
   regression-specific behavior.

Physical motivation (real citations, not asserted): accumulated-dose
degradation of tracker signal response is directly documented for
long-duration space silicon detectors -- see the LHCb Tracker Turicensis
radiation-damage monitoring note (CERN-LHCb-DP-2018-003, arXiv:1809.05063)
for a collider-adjacent tracker example, and Liu et al., "Simulation of
Radiation Damage for Silicon Drift Detector" (Sensors 19(8):1767, 2019,
doi:10.3390/s19081767) for deep-space silicon drift detector degradation
modeling specifically. The general pattern of a space-instrument
calibration pipeline needing to detect and correct slow, permanent
systematic drift is also documented for space photometry (Kepler's
Presearch Data Conditioning module; Smith et al. 2012, Stumpe et al. 2012,
2014) -- a different sensor technology and a different drift mechanism
(pointing/thermal systematics, not radiation dose), but the same class of
problem this module targets: a monitoring layer that has to catch slow,
non-resetting systematic drift with no natural checkpoint to recalibrate
against.

Model: gain(t) = gain0 * exp(-fluence(t) / tau_damage), with fluence(t)
proportional to elapsed event count (a stand-in for accumulated dose /
mission elapsed time). Exponential decay is a deliberately simple
approximation -- real NIEL-scaled damage models include annealing effects
and are far more detailed (see the Sensors 2019 citation above for a
proper treatment) -- adequate here because the point is to stress-test the
DETECTION methodology against a permanent-monotonic drift SHAPE, not to
produce a physically precise damage curve.
"""

from typing import Iterator, Optional

import numpy as np

from src.stream_loader import synthetic_object_stream, synthetic_burnin_conditions


def _stable_condition_fn():
    """Operating covariates held at fixed nominal values throughout --
    see module docstring point 3. Unlike gradual.py/abrupt.py, pileup here
    is NOT the thing drifting; it's deliberately inert."""
    def condition_fn(rng, i):
        cond = synthetic_burnin_conditions(rng, 1)
        return cond
    return condition_fn


def radiation_damage_stream(
    n_events: int,
    gain0: float = 1.0,
    tau_damage: float = 3000.0,
    affected_features: tuple = (0, 3, 9, 7),  # jet1_pt, jet2_pt, ht, met_pt -- generic "signal amplitude" proxies
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Monotonic, permanent, multiplicative gain decay from event 0 onward
    -- no changepoint_event parameter, because there isn't one; drift is
    present from the very first event and never stops or resets.
    ground-truth true_label is 1 from a configurable point once the
    cumulative degradation exceeds a detectability floor (5% gain loss),
    for evaluation.py's latency metrics -- NOT because the physical
    mechanism has a discrete onset (it doesn't), but because "detection
    latency" needs SOME reference point, and "when the effect becomes
    non-negligible" is the natural, defensible choice here (as opposed to
    abrupt.py/gradual.py's misspecified scenario, where the onset is a
    genuine mechanism change and the label is unambiguous).
    """
    detectability_floor = 0.05  # 5% cumulative gain loss

    def onset_event_for(tau):
        # gain(t)/gain0 = exp(-t/tau) = 1 - floor  =>  t = -tau * ln(1-floor)
        return int(-tau * np.log(1.0 - detectability_floor))

    onset = onset_event_for(tau_damage)

    def feature_bias_fn(rng, i, features):
        gain_factor = gain0 * np.exp(-i / tau_damage)
        biased = features.copy()
        for col in affected_features:
            biased[:, col] = biased[:, col] * gain_factor
        return biased

    for i, event in enumerate(synthetic_object_stream(
        n_events, seed=seed, condition_fn=_stable_condition_fn(), feature_bias_fn=feature_bias_fn,
    )):
        event["true_label"] = 0 if i < onset else 1
        event["true_onset_event"] = onset
        event["gain_factor"] = float(gain0 * np.exp(-i / tau_damage))
        yield event
