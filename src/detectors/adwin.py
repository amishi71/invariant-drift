"""Wraps river's ADWIN (Adaptive Windowing) for the update/is_ready/
evaluate_drift interface shared by every detector in this project.

Adapted from cms-streaming-shift-detection's src/detector.py::ADWINDetector.
That version fed ADWIN log(m_jj), matching CUSUM/Page-Hinkley's log-space
convention for a strictly-positive skewed observable. This version feeds
ADWIN the calibration residual directly (no transform) -- see cusum.py's
docstring for why: the residual is already signed and approximately
standardized, so log-space is inapplicable, not just unnecessary.

ADWIN's false-alarm guarantee is distribution-free (Hoeffding-bound based),
governed by the `delta` confidence parameter -- it doesn't require any
particular marginal shape for the input the way CUSUM/Page-Hinkley's
Gaussian-standardization framing implicitly does, which makes it a useful
model-free cross-check on the same residual stream.
"""

import numpy as np


class ADWINDetector:
    def __init__(self, reference_data=None, delta: float = 0.002):
        from river import drift
        self.detector = drift.ADWIN(delta=delta)
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
            "adwin_detected": self._last_detected,
            "estimated_width": getattr(self.detector, "width", None),
            "shift_detected": bool(self._last_detected),
        }
