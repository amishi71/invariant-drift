"""Online, single-pass statistical accumulators.

`OnlineMoments` is copied verbatim from cms-streaming-shift-detection
(src/kinematics.py) -- it is generic (Welford/Terriberry streaming
mean/variance/skewness/kurtosis) and has no dependency on the dijet-mass
observable that project was built around, so it transfers directly.
The dijet-mass-specific helpers (compute_dijet_mass, compute_delta_eta)
from that module are NOT reused here: this project's observable is a
calibration residual over trigger objects, not a resonance mass.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WelfordAccumulator:
    """Online, single-pass running statistics tracker using Welford's algorithm.

    Avoids catastrophic numerical cancellation of standard sum-of-squares formulas.
    """

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Incrementally updates running mean and M2 moment with an array of new observations."""
        values = np.asarray(values, dtype=np.float64)
        for x in values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Returns sample variance s^2."""
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        """Returns sample standard deviation s."""
        return np.sqrt(self.variance)


class OnlineMoments:
    """Streaming mean, variance, skewness, kurtosis via Welford/Terriberry.

    Single-pass, numerically stable. Used throughout this project wherever a
    "frozen burn-in" statistic is needed (residual calibration, detector
    reference stats) without materializing the whole burn-in array.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0

    def update(self, x):
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.mean += delta_n
        self.M4 += (term1 * delta_n2 * (self.n**2 - 3 * self.n + 3)
                    + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3)
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

    def update_batch(self, values):
        for x in values:
            self.update(float(x))

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_dev(self):
        return self.variance ** 0.5

    @property
    def skewness(self):
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return (self.n ** 0.5) * self.M3 / (self.M2 ** 1.5)

    @property
    def kurtosis(self):
        """Excess kurtosis (0 for a Gaussian)."""
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return (self.n * self.M4) / (self.M2 ** 2) - 3.0
