"""Page-Hinkley test: sequential detector for a shift in the mean of a stream.

Adapted from cms-streaming-shift-detection's src/detector.py::PageHinkleyDetector.
Same "frozen reference mean" discipline as that implementation, and the same
bug note applies (kept verbatim because it is load-bearing, not historical
color): letting the reference mean adapt to every post-burn-in event
self-cancels a sustained low-level shift, because the running mean chases
the shifted mean before the cumulative statistic can build up. mu0 (and
sigma0) are frozen at burn-in and never updated online.

Log-space is now a pluggable `transform` hook (default identity) instead of
hardcoded, for the same reason as cusum.py: the calibration residual this
detector watches (src/residual.py) is a signed, already-standardized
quantity, not a strictly-positive skewed one.
"""

from typing import Callable, Optional

import numpy as np


class PageHinkleyDetector:
    """Tracks cumulative deviation from a frozen baseline mean, with a
    running minimum subtracted off. Two-sided: also tracks the mirrored
    cumulative sum to catch a downward shift, for the same reason CUSUM
    here is two-sided (see cusum.py docstring) -- a calibration residual
    can drift low as easily as high.
    """

    def __init__(
        self,
        reference_data,
        delta: float = 0.5,
        lam: float = 10.0,
        two_sided: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        reference_data = np.asarray(reference_data, dtype=np.float64)
        self.transform = transform if transform is not None else (lambda x: x)
        ref = self.transform(reference_data)
        self.mu0 = float(np.mean(ref))
        self.sigma0 = float(np.std(ref))  # frozen baseline scale, same as CUSUM
        if self.sigma0 <= 0:
            raise ValueError(
                "Reference data has zero variance after transform; "
                "Page-Hinkley standardization is undefined."
            )
        self.delta = delta          # dead-band, in sigma0 units
        self.lam = lam              # alarm threshold -- TUNE empirically
        self.two_sided = two_sided

        self.n_events = 0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.cumulative_sum_neg = 0.0
        self.min_cumulative_sum_neg = 0.0

    def update(self, x: float) -> None:
        z_raw = float(self.transform(np.array([x]))[0])
        self.n_events += 1
        z = (z_raw - self.mu0) / self.sigma0  # frozen mu0, not a running mean
        self.cumulative_sum += z - self.delta
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        if self.two_sided:
            # Mirrors the up-side exactly with w_n = -z_n: this is the same
            # cumsum-minus-running-MINIMUM construction (see module
            # docstring's CUSUM equivalence), not a running maximum. An
            # earlier version of this file tracked a running maximum here,
            # which made ph_stat_down grow ~linearly with n (driven by the
            # -delta drift term alone) regardless of any real shift --
            # verified against a direct CUSUM cross-check during
            # development; see README's "Known issues" section.
            self.cumulative_sum_neg += -z - self.delta
            self.min_cumulative_sum_neg = min(
                self.min_cumulative_sum_neg, self.cumulative_sum_neg
            )

    def is_ready(self) -> bool:
        return self.n_events > 0

    def evaluate_drift(self) -> dict:
        ph_stat_up = self.cumulative_sum - self.min_cumulative_sum
        shift_up = ph_stat_up >= self.lam
        ph_stat_down = None
        shift_down = False
        if self.two_sided:
            ph_stat_down = self.cumulative_sum_neg - self.min_cumulative_sum_neg
            shift_down = ph_stat_down >= self.lam
        shift_detected = shift_up or shift_down
        direction = "up" if shift_up else ("down" if shift_down else None)
        return {
            "ready": True,
            "ph_stat_up": ph_stat_up,
            "ph_stat_down": ph_stat_down,
            "threshold": self.lam,
            "shift_detected": bool(shift_detected),
            "direction": direction,
        }
