"""Alternative calibration regressions (covariate models) for the residual.

Why this exists. The original calibration is a quadratic in pileup plus linear
n_jet and lumi (src/residual.py). On the real JetHT pool, held-out window means of
the residual vary about 4x more than i.i.d. noise allows and correlate with
pileup, at roughly half the size of the masked-channel signal (about -0.11 sigma).
This module offers richer pileup/n_jet models so the question "is that slow
variation a covariate-model problem?" can be answered by experiment.

Every model here has the same interface as `CalibrationModel` (fit / predict /
residual / residual_one, plus theta and sigma_resid), so it can be dropped into
`calibration_split.build_split_calibration(design=...)` without touching the
detectors. The default design "quad" is the ORIGINAL model, bit for bit, so
nothing changes unless a new design is requested.

Designs
-------
quad            [1, pu, pu^2, n_jet, lumi]   (original; delegates to residual.py)
pu_spline       piecewise-linear spline in pileup (n_knots interior knots at
                quantiles of the fit sample) + linear n_jet (+ lumi if it varies)
pu_spline_nj    pu_spline, but n_jet enters as one indicator per observed level
pu_spline_nj_x  pu_spline_nj + a pileup x n_jet interaction

Extrapolation. Outside the pileup range seen in the fit sample the spline is held
constant ("clamp", the default) instead of extrapolating a polynomial; the quadratic
can run away outside its fit range, which matters when the burn-in is a narrow time
slice and 9% of held-out window means fall outside it. `extrapolate="linear"` keeps
the end-segment slope instead.

Causality. Whether a model is deployable depends on WHICH events it is fit on, not
on the functional form. A spline fit on a burn-in that spans the pileup range uses
data the detector would not have at the start of a run; report such fits as an
offline calibration (see calibration_split mode "span"), next to the causal ones.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.residual import CalibrationModel, _SCORE_TRANSFORMS, _design_matrix

DESIGNS = ("quad", "pu_spline", "pu_spline_nj", "pu_spline_nj_x")


@dataclass
class FlexCalibrationModel:
    design: str = "pu_spline"
    theta: Optional[np.ndarray] = None
    sigma_resid: Optional[float] = None
    robust: bool = False
    n_burn_in: int = 0
    score_transform: str = "log"
    n_knots: int = 8
    extrapolate: str = "clamp"
    # frozen at fit time
    pu_mu: float = 0.0
    pu_sd: float = 1.0
    pu_lo: float = 0.0
    pu_hi: float = 0.0
    knots: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nj_lo: int = 0
    nj_hi: int = 0
    use_lumi: bool = False

    # ------------------------------------------------------------------ design
    def _pu_basis(self, pileup: np.ndarray) -> np.ndarray:
        p = np.asarray(pileup, dtype=np.float64)
        if self.extrapolate == "clamp":
            p = np.clip(p, self.pu_lo, self.pu_hi)
        u = (p - self.pu_mu) / self.pu_sd
        cols = [u] + [np.maximum(0.0, u - k) for k in self.knots]
        return np.column_stack(cols), u

    def _X(self, pileup, mult, lumi) -> np.ndarray:
        pileup = np.asarray(pileup, dtype=np.float64)
        mult = np.asarray(mult, dtype=np.float64)
        lumi = np.asarray(lumi, dtype=np.float64)
        B, u = self._pu_basis(pileup)
        cols = [np.ones(len(pileup)), B]
        nj = np.clip(np.rint(mult), self.nj_lo, self.nj_hi)
        if self.design == "pu_spline":
            cols.append(mult[:, None])
        else:
            levels = np.arange(self.nj_lo + 1, self.nj_hi + 1)
            if len(levels):
                cols.append((nj[:, None] == levels[None, :]).astype(np.float64))
            if self.design == "pu_spline_nj_x":
                cols.append((u * (nj - self.nj_lo))[:, None])
        if self.use_lumi:
            cols.append(lumi[:, None])
        return np.column_stack(cols)

    # --------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, score, pileup, mult, lumi, design: str = "pu_spline",
            n_knots: int = 8, robust: bool = False, score_transform: str = "log",
            extrapolate: str = "clamp"):
        if design not in DESIGNS:
            raise ValueError(f"Unknown design {design!r}; choose from {DESIGNS}.")
        if score_transform not in _SCORE_TRANSFORMS:
            raise ValueError(f"Unknown score_transform {score_transform!r}.")
        if extrapolate not in ("clamp", "linear"):
            raise ValueError("extrapolate must be 'clamp' or 'linear'.")
        y = _SCORE_TRANSFORMS[score_transform](np.asarray(score, dtype=np.float64))
        pileup = np.asarray(pileup, dtype=np.float64)
        mult = np.asarray(mult, dtype=np.float64)
        lumi = np.asarray(lumi, dtype=np.float64)

        if design == "quad":
            base = CalibrationModel.fit(score, pileup, mult, lumi, robust=robust,
                                        score_transform=score_transform)
            return cls(design="quad", theta=base.theta, sigma_resid=base.sigma_resid,
                       robust=robust, n_burn_in=base.n_burn_in,
                       score_transform=score_transform)

        m = cls(design=design, robust=robust, score_transform=score_transform,
                n_knots=n_knots, extrapolate=extrapolate)
        m.pu_mu = float(pileup.mean())
        m.pu_sd = float(pileup.std()) or 1.0
        m.pu_lo, m.pu_hi = (float(v) for v in np.percentile(pileup, [0.5, 99.5]))
        u = (np.clip(pileup, m.pu_lo, m.pu_hi) - m.pu_mu) / m.pu_sd
        qs = np.linspace(0, 1, n_knots + 2)[1:-1]
        m.knots = np.unique(np.quantile(u, qs))
        njr = np.rint(mult)
        m.nj_lo, m.nj_hi = int(njr.min()), int(njr.max())
        m.use_lumi = bool(lumi.std() > 0)

        X = m._X(pileup, mult, lumi)
        if X.shape[0] < X.shape[1] + 5:
            raise ValueError(f"Need at least {X.shape[1] + 5} fit events for design "
                             f"{design!r}; got {X.shape[0]}.")
        m.theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ m.theta
        if robust:
            mad = np.median(np.abs(resid - np.median(resid)))
            m.sigma_resid = float(mad / 0.6745) if mad > 0 else float(np.std(resid))
        else:
            m.sigma_resid = float(np.std(resid))
        if m.sigma_resid <= 0:
            raise ValueError("Fit residual scale is zero or negative.")
        m.n_burn_in = X.shape[0]
        return m

    # --------------------------------------------------------------- inference
    def predict(self, pileup, mult, lumi) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("Model has not been fit yet.")
        if self.design == "quad":
            return _design_matrix(pileup, mult, lumi) @ self.theta
        return self._X(pileup, mult, lumi) @ self.theta

    def residual(self, score, pileup, mult, lumi) -> np.ndarray:
        y = _SCORE_TRANSFORMS[self.score_transform](np.asarray(score, dtype=np.float64))
        return (y - self.predict(pileup, mult, lumi)) / self.sigma_resid

    def residual_one(self, score: float, pileup: float, mult: float, lumi: float) -> float:
        return float(self.residual(np.array([score]), np.array([pileup]),
                                   np.array([mult]), np.array([lumi]))[0])


def fit_calibration(design, score, pileup, mult, lumi, n_knots: int = 8, **kw):
    """Single entry point. design="quad" returns the ORIGINAL CalibrationModel
    object unchanged; any other design returns a FlexCalibrationModel."""
    if design == "quad":
        return CalibrationModel.fit(score, pileup, mult, lumi,
                                    **{k: v for k, v in kw.items()
                                       if k in ("robust", "score_transform")})
    return FlexCalibrationModel.fit(score, pileup, mult, lumi, design=design,
                                    n_knots=n_knots, **kw)
