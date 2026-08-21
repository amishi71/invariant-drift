"""The calibration residual: the single scalar quantity every detector in
src/detectors/ monitors.

From the abstract: "it tracks the anomaly score jointly with its main
driving covariates (pileup, object multiplicity) through a scalar residual
calibrated against expected luminosity trends."

Design
------
During a burn-in period known to reflect a *correctly calibrated* trigger
(e.g. the start of a fill, or a period cross-checked offline), fit a
regression of the anomaly score onto its main driving covariates:

    E[score | pileup, mult, lumi] = f(pileup, mult, lumi; theta)

using a small polynomial feature basis (quadratic in pileup, since
anomaly-score-vs-pileup response is typically smooth but non-linear;
linear in multiplicity and in the luminosity-trend term). Freeze theta
(same "frozen reference" discipline as the CUSUM/Page-Hinkley burn-in
statistics -- see cusum.py) and the residual scale sigma_resid from the
burn-in fit residuals.

At run time, the residual for a new event is

    r_t = (score_t - f(pileup_t, mult_t, lumi_t; theta)) / sigma_resid

If the calibration still holds, r_t behaves like a mean-zero,
unit-scale, trend-free sequence, REGARDLESS of how pileup/multiplicity/
luminosity are moving on their own -- a normal luminosity burn-off across
a fill should NOT by itself make r_t drift, because that expected
covariate movement is already baked into f(.). A change-point in r_t
means the *relationship* between the anomaly score and its covariates
has changed, which is precisely "the calibration no longer holds" -- not
just "conditions changed" (conditions always change; that's what f(.) is
for).

Score transform (log by default): a reconstruction-error-type anomaly
score (as produced by proxy_vae.py) is strictly positive and right-skewed
by construction -- it behaves roughly like a sum of squared
approximately-Gaussian terms, i.e. chi-squared-shaped, not Gaussian.
Regressing the raw score and standardizing the residual does NOT fix
this: the conditional distribution of score given (pileup, mult, lumi)
is still right-skewed after subtracting a linear-in-covariates
prediction, which quietly breaks the near-Gaussian assumption CUSUM's
and Page-Hinkley's ARL guarantees are calibrated against (empirically,
this produced false-alarm rates far above their theoretical values
during development of this project -- see README's "Known issues /
design decisions" section). The fix mirrors what
cms-streaming-shift-detection did for its own strictly-positive, skewed
observable (dijet mass): fit the regression and compute the residual in
log-score space by default (`score_transform="log"`). This is a property
of the score itself, not a per-detector choice, so it lives here in the
calibration model rather than as a transform hook on individual
detectors (contrast with detectors/cusum.py's `transform` argument, which
still defaults to identity because IT expects to receive this already-
log-transformed, already-symmetrized residual, not the raw score).

This is also exactly what lets drift_sim/gradual.py test two distinct
scenarios with the same underlying pileup-evolution mechanism: pileup
evolving *as f(.) assumes* (residual should stay flat -- a clean
false-alarm test) vs. a *misspecified* evolution law (residual should
drift -- the abstract's "robustness when the assumed drift model is
misspecified").
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_SCORE_TRANSFORMS = {
    "identity": lambda s: s,
    "log": lambda s: np.log(np.clip(s, 1e-12, None)),
}


def _design_matrix(pileup: np.ndarray, mult: np.ndarray, lumi: np.ndarray) -> np.ndarray:
    """Polynomial feature basis: [1, pileup, pileup^2, mult, lumi]."""
    pileup = np.asarray(pileup, dtype=np.float64)
    mult = np.asarray(mult, dtype=np.float64)
    lumi = np.asarray(lumi, dtype=np.float64)
    return np.column_stack([
        np.ones_like(pileup), pileup, pileup ** 2, mult, lumi,
    ])


@dataclass
class CalibrationModel:
    """Frozen burn-in regression of anomaly score on (pileup, multiplicity,
    luminosity trend), plus the residual scale. Everything here is set
    once by `fit()` and never updated online -- the whole point is that
    the detectors downstream are watching for the moment this frozen model
    stops being a good fit.
    """

    theta: Optional[np.ndarray] = None
    sigma_resid: Optional[float] = None
    robust: bool = False
    n_burn_in: int = field(default=0)
    score_transform: str = "log"

    @classmethod
    def fit(
        cls,
        score: np.ndarray,
        pileup: np.ndarray,
        mult: np.ndarray,
        lumi: np.ndarray,
        robust: bool = False,
        score_transform: str = "log",
    ) -> "CalibrationModel":
        """Least-squares fit of the burn-in regression, in
        `score_transform` space (see module docstring for why "log" is the
        default for a reconstruction-error-type score).

        :param robust: if True, use the median absolute deviation (scaled
            by 1/0.6745 to be a consistent estimator of sigma under
            Normality) instead of the plain standard deviation for
            sigma_resid. More resistant to a handful of burn-in outliers,
            at the cost of some efficiency if burn-in truly is clean.
        :param score_transform: "log" (default) or "identity". Use
            "identity" if `score` is already a signed/symmetric quantity
            (e.g. you're calibrating directly on some other detector's
            output rather than the raw VAE reconstruction error).
        """
        if score_transform not in _SCORE_TRANSFORMS:
            raise ValueError(f"Unknown score_transform '{score_transform}'; "
                              f"choose from {list(_SCORE_TRANSFORMS)}.")
        score_t = _SCORE_TRANSFORMS[score_transform](np.asarray(score, dtype=np.float64))
        X = _design_matrix(pileup, mult, lumi)
        if X.shape[0] < X.shape[1] + 5:
            raise ValueError(
                f"Need at least {X.shape[1] + 5} burn-in events to fit the "
                f"calibration model safely; got {X.shape[0]}."
            )
        theta, _, _, _ = np.linalg.lstsq(X, score_t, rcond=None)
        resid = score_t - X @ theta
        if robust:
            mad = np.median(np.abs(resid - np.median(resid)))
            sigma = float(mad / 0.6745) if mad > 0 else float(np.std(resid))
        else:
            sigma = float(np.std(resid))
        if sigma <= 0:
            raise ValueError(
                "Burn-in residual scale is zero or negative; burn-in data "
                "may be degenerate (e.g. constant score)."
            )
        return cls(theta=theta, sigma_resid=sigma, robust=robust,
                   n_burn_in=X.shape[0], score_transform=score_transform)

    def predict(self, pileup, mult, lumi) -> np.ndarray:
        """Predicted score, IN score_transform SPACE (log-space by
        default) -- not raw score units. This is what `residual()`
        compares against; it is not meant to be exponentiated back into
        raw-score units by callers, since sigma_resid is likewise a
        log-space scale.
        """
        if self.theta is None:
            raise RuntimeError("CalibrationModel has not been fit yet.")
        X = _design_matrix(pileup, mult, lumi)
        return X @ self.theta

    def residual(self, score, pileup, mult, lumi) -> np.ndarray:
        """Standardized scalar residual r_t (see module docstring).
        Accepts RAW score (same units as what proxy_vae.anomaly_score
        returns) -- applies score_transform internally, so callers never
        need to remember to transform it themselves.
        """
        score_t = _SCORE_TRANSFORMS[self.score_transform](np.asarray(score, dtype=np.float64))
        expected = self.predict(pileup, mult, lumi)
        return (score_t - expected) / self.sigma_resid

    def residual_one(self, score: float, pileup: float, mult: float, lumi: float) -> float:
        """Convenience scalar version for event-by-event streaming use."""
        return float(self.residual(
            np.array([score]), np.array([pileup]), np.array([mult]), np.array([lumi]),
        )[0])
