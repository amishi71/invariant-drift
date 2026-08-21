"""Adaptive Conformal Inference (ACI) for a one-sided anomaly-score
decision threshold.

Reference: Gibbs & Candès, "Adaptive Conformal Inference Under Distribution
Shift" (2021), arXiv:2106.00170.

Why ACI here: standard (split) conformal prediction assumes exchangeable
calibration/test data. That assumption is false at the LHC by construction
-- beam current decay and pileup drift mean the anomaly-score distribution
at hour 8 of a fill is not exchangeable with hour 1. ACI drops the
exchangeability requirement and instead tracks a target miscoverage rate
online, adjusting the *effective* alpha (and therefore the decision
threshold) up or down based on whether recent control-sample points are
being flagged/covered at the target rate.

Framing as a one-sided threshold rather than a two-sided interval: this is
an anomaly-score gate (flag if score > threshold), not a regression
interval, so "coverage" here means "the fraction of TRUE-BACKGROUND
(Zero-Bias control) events whose score falls below the threshold" and
"miscoverage" means a background control event exceeded the threshold --
i.e. would have been a false positive at the current operating point. ACI
adapts alpha_t to keep that empirical false-positive rate on control data
near the nominal target, exactly as the abstract describes: "recalibrates
decision thresholds on the fly using miscoverage feedback from Zero-Bias
control data and delayed offline validation."

Update rule (Gibbs & Candès, eq. 2):
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
where err_t = 1{control point at t exceeded the threshold} (miscoverage).
If a miscoverage just happened, alpha_t decreases -> the next threshold is
the quantile at a SMALLER alpha, i.e. a HIGHER (safer) quantile -> the
interval widens / the gate becomes stricter. If coverage has been
consistently fine, alpha_t drifts up toward looser thresholds, which is
what keeps the false-positive rate calibrated rather than monotonically
conservative.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class AdaptiveConformalThreshold:
    alpha_target: float = 0.01
    gamma: float = 0.01
    calibration_window: int = 2000
    alpha_min: float = 1e-4
    alpha_max: float = 0.5
    delayed_gamma: Optional[float] = None  # defaults to gamma if None

    _alpha_t: float = field(init=False)
    _calib_scores: Deque[float] = field(init=False)
    _history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._alpha_t = self.alpha_target
        self._calib_scores = deque(maxlen=self.calibration_window)
        if self.delayed_gamma is None:
            self.delayed_gamma = self.gamma

    @classmethod
    def from_burn_in(
        cls, burn_in_scores: np.ndarray, alpha_target: float = 0.01,
        gamma: float = 0.01, calibration_window: int = 2000,
    ) -> "AdaptiveConformalThreshold":
        obj = cls(alpha_target=alpha_target, gamma=gamma, calibration_window=calibration_window)
        for s in np.asarray(burn_in_scores, dtype=np.float64):
            obj._calib_scores.append(float(s))
        return obj

    def current_threshold(self) -> float:
        """The (1 - alpha_t) empirical quantile of the sliding calibration
        buffer. A sliding buffer (not a single frozen burn-in set) is used
        deliberately -- see module docstring: exchangeability doesn't hold
        here, so the reference distribution the threshold is drawn from
        must itself be allowed to track slow, LEGITIMATE condition changes
        (this is separate from, and complementary to, the alpha_t
        adaptation, which tracks the *coverage rate*, not the score
        distribution directly).
        """
        if len(self._calib_scores) == 0:
            raise RuntimeError("No calibration scores available yet; call from_burn_in() first.")
        q = 1.0 - self._clipped_alpha()
        return float(np.quantile(np.array(self._calib_scores), q))

    def decide(self, anomaly_score: float) -> bool:
        """True if anomaly_score should be flagged (exceeds current threshold)."""
        return anomaly_score > self.current_threshold()

    def update(self, zero_bias_score: float) -> dict:
        """Online update from a fresh Zero-Bias control-stream score (known
        background by construction of the Zero-Bias trigger path). Call
        this on every Zero-Bias event; it both adapts alpha_t and pushes
        the score onto the sliding calibration buffer.
        """
        threshold = self.current_threshold()
        err = 1.0 if zero_bias_score > threshold else 0.0
        self._alpha_t += self.gamma * (self.alpha_target - err)
        self._alpha_t = self._clip(self._alpha_t)
        self._calib_scores.append(float(zero_bias_score))
        record = {"threshold": threshold, "miscoverage": bool(err), "alpha_t": self._alpha_t}
        self._history.append(record)
        return record

    def update_delayed(self, offline_score: float, true_label_background: bool) -> dict:
        """Incorporate delayed, offline-validated feedback (the abstract's
        "delayed offline validation"): once a batch of events has been
        offline-confirmed as background or not, feed each confirmed
        BACKGROUND event through the same ACI update using
        `delayed_gamma` (defaults to the same step size as the live
        Zero-Bias feedback, but exposed separately since delayed feedback
        typically arrives in bursts and a practitioner may want to
        down-weight it, e.g. delayed_gamma = gamma / 10, to avoid
        over-reacting to a single offline-validation batch).

        Confirmed non-background (true_label_background=False) events are
        skipped -- they aren't control points and would bias the
        miscoverage estimate if included (a genuine anomaly correctly
        exceeding the threshold is not a miscoverage event).
        """
        if not true_label_background:
            return {"skipped": True, "reason": "not a background-confirmed control point"}
        threshold = self.current_threshold()
        err = 1.0 if offline_score > threshold else 0.0
        self._alpha_t += self.delayed_gamma * (self.alpha_target - err)
        self._alpha_t = self._clip(self._alpha_t)
        record = {
            "threshold": threshold, "miscoverage": bool(err),
            "alpha_t": self._alpha_t, "delayed": True,
        }
        self._history.append(record)
        return record

    def _clip(self, a: float) -> float:
        return float(np.clip(a, self.alpha_min, self.alpha_max))

    def _clipped_alpha(self) -> float:
        return self._clip(self._alpha_t)

    @property
    def alpha_t(self) -> float:
        return self._alpha_t

    def empirical_coverage(self, last_n: Optional[int] = None) -> float:
        """1 - empirical miscoverage rate over recent update() calls
        (both live and delayed), for evaluation.py's coverage metric."""
        hist = self._history if last_n is None else self._history[-last_n:]
        if not hist:
            return float("nan")
        miscov = np.mean([1.0 if h.get("miscoverage") else 0.0 for h in hist if not h.get("skipped")])
        return 1.0 - float(miscov)
