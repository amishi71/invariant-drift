"""Abrupt drift: subdetector conditions changing mid-run, e.g. masked
readout channels.

Unlike gradual.py's slow fill-evolution scenarios, this models a step
change at a specific, known (for evaluation only) event index: a fraction
of calorimeter/tracker readout channels going dead or masked, which
manifests as a sudden partial loss of reconstructed energy/objects rather
than a smooth trend. Two independent failure modes are provided since real
masked-channel incidents don't all look the same:

- `masked_channel_stream`: an abrupt DROP in reconstructed HT/MET/jet pT
  (channels stop contributing energy) -- the more common failure mode
  (dead front-end boards, HV trips).
- `multiplicity_step_stream`: an abrupt CHANGE in object multiplicity
  (e.g. a masked region causes jets to be under-counted or double-counted
  at a boundary) with energy scale left alone -- exercises detectors on a
  covariate-shape change rather than a magnitude drop.

Both step changes are instantaneous and sustained (not a single-event
blip) starting at `changepoint_event`, which is the realistic behavior of
a channel actually going dead (it stays dead) as opposed to a transient
readout glitch.
"""

from typing import Iterator, Optional

import numpy as np

from src.stream_loader import synthetic_object_stream, synthetic_burnin_conditions


def _stable_condition_fn(n_jet_base: float = 3.0):
    def condition_fn(rng, i):
        cond = synthetic_burnin_conditions(rng, 1)
        return cond
    return condition_fn


def masked_channel_stream(
    n_events: int,
    changepoint_event: int,
    drop_fraction: float = 0.15,
    affected_features: tuple = (0, 3, 9, 7),  # jet1_pt, jet2_pt, ht, met_pt
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Sustained multiplicative drop (masked channels stop contributing
    energy) on energy-like features, starting exactly at changepoint_event.
    Pileup/lumi conditions are left at stable burn-in levels throughout --
    this isolates the abrupt-hardware-failure signal from any fill
    evolution, matching the abstract's framing of gradual vs. abrupt as
    two separate injected scenarios.
    """
    def feature_bias_fn(rng, i, features):
        if i < changepoint_event:
            return features
        biased = features.copy()
        for col in affected_features:
            biased[:, col] = biased[:, col] * (1.0 - drop_fraction)
        return biased

    for i, event in enumerate(synthetic_object_stream(
        n_events, seed=seed,
        condition_fn=_stable_condition_fn(),
        feature_bias_fn=feature_bias_fn,
    )):
        event["true_label"] = 0 if i < changepoint_event else 1
        event["true_onset_event"] = changepoint_event
        yield event


def multiplicity_step_stream(
    n_events: int,
    changepoint_event: int,
    n_jet_delta: float = -1.5,
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Sustained step change in jet multiplicity (n_jet feature and the
    n_jet covariate fed to residual.py's regression both shift), energy
    scale untouched. n_jet_delta is added to the Poisson mean used for
    n_jet generation from `changepoint_event` onward; negative values
    model a masked region silently dropping jets, positive values model
    spurious noise-jet reconstruction from a misbehaving channel.
    """
    def condition_fn(rng, i):
        cond = synthetic_burnin_conditions(rng, 1)
        if i >= changepoint_event:
            lam = max(0.5, 3.0 + 0.05 * cond["pileup"][0] + n_jet_delta)
            cond["n_jet"] = np.array([max(2.0, rng.poisson(lam=lam))])
        return cond

    for i, event in enumerate(synthetic_object_stream(
        n_events, seed=seed, condition_fn=condition_fn,
    )):
        event["true_label"] = 0 if i < changepoint_event else 1
        event["true_onset_event"] = changepoint_event
        yield event
