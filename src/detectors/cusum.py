"""Sequential change-point detector via the classic (Page) CUSUM statistic.

Adapted from cms-streaming-shift-detection's src/detector.py::CUSUMDetector.

Project A applied CUSUM in log-space because its observable (dijet mass)
is strictly positive and heavily right-skewed (skew ~2.07, kurtosis ~6.81).
The observable here is the calibration residual from src/residual.py:
    r_t = (anomaly_score_t - E[score | pileup_t, mult_t, lumi_t]) / scale_t
which is already a *signed*, approximately standardized quantity by
construction (see residual.py) -- log-space is not just unnecessary here,
it is inapplicable (log of a signed value is undefined). So the transform
is now a pluggable hook (`transform`, default identity) rather than a
hardcoded log(); pass transform=np.log only if you point this detector at
a strictly-positive, skewed covariate (e.g. raw pileup) instead of the
residual.

Everything else about the CUSUM logic -- and the underlying ARL-vs-latency
tradeoff theory that governs threshold `h` -- is unchanged.
"""

from typing import Callable, Optional

import numpy as np


class CUSUMDetector:
    """Two-sided Page-CUSUM change-point detector on a scalar stream.

    S+_n = max(0, S+_{n-1} + z_n - k)
    S-_n = max(0, S-_{n-1} - z_n - k)
    where z_n = (transform(x_n) - mu0) / sigma0 is the standardized
    observation (mu0, sigma0 frozen from burn-in reference data), k is the
    reference/slack value (in sigma units), and h is the alarm threshold.

    Two-sided by default (S+ catches upward drift, S- catches downward
    drift) because a calibration residual can go stale in either direction
    -- e.g. an anomaly-score model that starts systematically
    over-predicting under otherwise-nominal conditions is just as much a
    calibration failure as under-predicting. Project A's CUSUM was
    one-sided (S+ only) because a resonance signal only ever pulls m_jj
    mass upward; that asymmetry doesn't hold here.

    Unlike repeated static two-sample tests, CUSUM's false-alarm behavior
    is governed by Average Run Length (ARL) theory: threshold h controls a
    genuine tradeoff between mean time-to-false-alarm (under H0) and mean
    detection delay (under H1), rather than accumulating look-elsewhere
    risk with every new evaluation. h must still be tuned empirically
    against real background (Zero-Bias) data -- see evaluation.py's
    average_run_length().
    """

    def __init__(
        self,
        reference_data,
        k: float = 0.5,
        h: float = 8.0,
        two_sided: bool = True,
        reset_after_alarm: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        reference_data = np.asarray(reference_data, dtype=np.float64)
        self.transform = transform if transform is not None else (lambda x: x)
        ref = self.transform(reference_data)
        self.mu0 = float(np.mean(ref))
        self.sigma0 = float(np.std(ref))
        if self.sigma0 <= 0:
            raise ValueError(
                "Reference data has zero variance after transform; CUSUM "
                "standardization is undefined. Check burn-in data / transform."
            )
        self.k = k                  # slack, in standardized (sigma) units
        self.h = h                  # alarm threshold, in standardized (sigma) units -- TUNE THIS
        self.two_sided = two_sided
        self.reset_after_alarm = reset_after_alarm

        self.S_pos = 0.0
        self.S_neg = 0.0
        self.n_events = 0

    def update(self, x: float) -> None:
        z = (float(self.transform(np.array([x]))[0]) - self.mu0) / self.sigma0
        self.S_pos = max(0.0, self.S_pos + z - self.k)
        if self.two_sided:
            self.S_neg = max(0.0, self.S_neg - z - self.k)
        self.n_events += 1

    def is_ready(self) -> bool:
        # No warm-up window beyond the frozen reference stats -- can
        # evaluate from the very first streamed event.
        return self.n_events > 0

    def evaluate_drift(self) -> dict:
        s_pos_at_eval = self.S_pos
        s_neg_at_eval = self.S_neg
        shift_detected = (s_pos_at_eval >= self.h) or (
            self.two_sided and s_neg_at_eval >= self.h
        )
        direction = None
        if shift_detected:
            direction = "up" if s_pos_at_eval >= self.h else "down"
            if self.reset_after_alarm:
                if s_pos_at_eval >= self.h:
                    self.S_pos = 0.0
                if self.two_sided and s_neg_at_eval >= self.h:
                    self.S_neg = 0.0
        return {
            "ready": True,
            "cusum_stat_pos": s_pos_at_eval,
            "cusum_stat_neg": s_neg_at_eval if self.two_sided else None,
            "threshold": self.h,
            "shift_detected": bool(shift_detected),
            "direction": direction,
        }
