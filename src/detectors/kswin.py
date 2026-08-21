"""Wraps river's KSWIN (Kolmogorov-Smirnov WINdowing, Raab et al. 2020) for
the update/is_ready/evaluate_drift interface shared by every detector here.

KSWIN maintains a fixed-size sliding window and repeatedly runs a KS test
between a recent sub-sample and the rest of the window -- distribution-free
like ADWIN, but window-based rather than adaptively-sized, and directly
sensitive to changes in the full shape of the residual's distribution
(not just its mean), which is a useful property here since a stale
calibration can manifest as a variance/shape change before the mean visibly
moves.

alpha default (1e-3, not river's own default of 0.005): river's KSWIN runs
a FRESH KS-test on every single `update()` call once the sliding window
first fills (confirmed from river's source, not just the docstring) -- it
is not "one test per window", it's one test per event. Over a stream of a
few thousand events that's a few thousand repeated hypothesis tests with
no multiple-testing correction, so alpha=0.01 (or even river's own default
0.005) produces a false-alarm rate close to 100% over a stream of this
length -- verified empirically during development (10/10 trials false-
alarmed within 1500 events at alpha=0.01; see README's "Known issues"
section). alpha=1e-3 was tuned to bring KSWIN's false-alarm rate over a
~1500-2000 event run down to roughly the same order of magnitude as the
other four detectors' ARL-derived rates, while still detecting an actual
shift within a few dozen events -- it is an empirical operating-point
choice for this project's event-count scale, not a principled default;
retune it (river's own guidance: "should be set below 0.01") if you run
KSWIN over streams of very different length.
"""

import numpy as np


class KSWINDetector:
    def __init__(
        self,
        reference_data=None,
        alpha: float = 1e-3,
        window_size: int = 200,
        stat_size: int = 40,
        seed: int = 0,
    ):
        from river import drift
        self.detector = drift.KSWIN(
            alpha=alpha, window_size=window_size, stat_size=stat_size, seed=seed,
        )
        self.n_events = 0
        self._last_detected = False

    def update(self, x: float) -> None:
        self.detector.update(float(x))
        self.n_events += 1
        self._last_detected = bool(self.detector.drift_detected)

    def is_ready(self) -> bool:
        return self.n_events > 0

    def evaluate_drift(self) -> dict:
        return {
            "ready": True,
            "kswin_detected": self._last_detected,
            "shift_detected": bool(self._last_detected),
        }
