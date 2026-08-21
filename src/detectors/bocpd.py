"""Bayesian Online Change-Point Detection (BOCPD).

Reference: Adams & MacKay, "Bayesian Online Changepoint Detection" (2007),
arXiv:0710.3742. Conjugate-update formulas for the Normal / Normal-Inverse-
Gamma model follow Murphy, "Conjugate Bayesian analysis of the Gaussian
distribution" (2007) -- the standard reference for these closed forms.

Model: the calibration residual within a stable "regime" (run) is modeled
as i.i.d. Normal(mu, sigma^2) with unknown mean and variance, under a
Normal-Inverse-Gamma conjugate prior NIG(mu0, kappa0, alpha0, beta0). A
regime change resets to a fresh draw from that prior. BOCPD maintains the
full posterior distribution over the current "run length" r_t (events
since the last changepoint) and updates it online in closed form -- no
MCMC, no retraining.

Why this on top of CUSUM/Page-Hinkley: those are frequentist sequential
tests that answer "has *a* shift happened" via a fixed threshold on an
accumulated statistic. BOCPD instead maintains a full posterior over *when*
the last regime change was, which (a) gives a calibrated probability rather
than a threshold-crossing event, and (b) adapts its within-regime noise
model (unknown sigma, not just unknown mean) as each run accumulates
evidence -- relevant here because the residual's variance itself can
change as a calibration goes stale, not just its mean.
"""

from typing import Optional

import numpy as np
from scipy.special import gammaln


def _log_student_t_pdf(x, df, loc, scale):
    """Log-density of a (possibly vectorized) Student-t distribution.

    Implemented directly (not via scipy.stats.t.logpdf) so it is trivially
    vectorized over arrays of (df, loc, scale) -- one entry per active
    run-length hypothesis -- without per-hypothesis Python-level calls.
    """
    z = (x - loc) / scale
    return (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
        - ((df + 1.0) / 2.0) * np.log1p((z * z) / df)
    )


class BOCPD:
    """Bayesian Online Change-Point Detection with a Normal-Inverse-Gamma
    conjugate model and a constant (geometric) hazard function.

    Interface mirrors the other detectors here (update / is_ready /
    evaluate_drift) so it drops into the same evaluation harness.
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        hazard_lambda: float = 250.0,
        mu0: Optional[float] = None,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: Optional[float] = None,
        prune_threshold: float = 1e-4,
        changepoint_prob_threshold: float = 0.5,
        max_run_length: int = 5000,
        r_min: int = 4,
        warm_up_events: Optional[int] = None,
    ):
        """
        :param reference_data: burn-in data used to set mu0/beta0 if not
            given explicitly (empirical mean / variance of the residual
            under known-stable conditions).
        :param hazard_lambda: expected run length under the prior (constant
            hazard H = 1/hazard_lambda at every step). This is the one
            parameter with a direct physical reading: "how many events do
            we expect between genuine calibration breaks, a priori."
        :param mu0, kappa0, alpha0, beta0: Normal-Inverse-Gamma prior
            hyperparameters. kappa0 is prior confidence in mu0 (in units of
            "equivalent prior observations"); alpha0/beta0 set the prior on
            the noise variance (alpha0=2 gives a prior with a finite mean
            IG(2, beta0) = beta0, a weakly-informative default).
        :param prune_threshold: drop run-length hypotheses whose posterior
            mass falls below this. Keeps the run-length distribution's
            support bounded in a genuinely online (unbounded-stream)
            setting -- the naive algorithm's support grows by one each
            step, which is unbounded memory/compute otherwise.
        :param changepoint_prob_threshold: flag a change when
            P(r_t <= r_min | x_1:t) exceeds this.
        :param max_run_length: hard cap on tracked run length regardless of
            pruning (defensive bound).
        :param r_min: window used for the detection statistic (see
            evaluate_drift's docstring for why P(r_t = 0) alone is the
            wrong criterion, and P(r_t <= r_min) is used instead).
        :param warm_up_events: is_ready() (and therefore any consumer that
            gates evaluate_drift() on it, per this project's shared
            detector convention) returns False until this many events have
            been streamed. Defaults to 5 * r_min if not given. This is NOT
            optional bookkeeping: P(r_t <= r_min) is trivially/mechanically
            1.0 for the first r_min events (the run length literally cannot
            exceed the number of events seen so far), so without a warm-up
            gate the detector "detects a changepoint" on every single run,
            unconditionally, within the first few events -- caught during
            development as a real bug (see tests/test_bocpd.py and
            README's "Known issues" section), not a hypothetical concern.
        """
        if reference_data is not None and len(reference_data) > 0:
            ref = np.asarray(reference_data, dtype=np.float64)
            if mu0 is None:
                mu0 = float(np.mean(ref))
            if beta0 is None:
                # alpha0=2 => prior mean of the IG(alpha0, beta0) variance
                # distribution is beta0 / (alpha0 - 1) = beta0. Set beta0 to
                # the empirical variance so the prior mean variance matches
                # burn-in.
                beta0 = float(np.var(ref)) if np.var(ref) > 0 else 1.0
        else:
            mu0 = 0.0 if mu0 is None else mu0
            beta0 = 1.0 if beta0 is None else beta0

        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.prune_threshold = prune_threshold
        self.changepoint_prob_threshold = changepoint_prob_threshold
        self.max_run_length = max_run_length
        self.r_min = r_min
        self.warm_up_events = warm_up_events if warm_up_events is not None else 5 * r_min

        # Run-length posterior and per-hypothesis NIG parameters. Index i
        # in these arrays corresponds to run length r=i.
        self.log_R = np.array([0.0])  # log P(r_0 = 0) = log(1) at t=0
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

        self.n_events = 0
        self._last_changepoint_prob = 0.0
        self._last_young_run_prob = 0.0
        self._last_map_run_length = 0

    def update(self, x: float) -> None:
        x = float(x)

        # Student-t predictive df/loc/scale for each currently active
        # run-length hypothesis (Murphy 2007, eq. for NIG posterior
        # predictive): x | data ~ t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        log_pred = _log_student_t_pdf(x, df, self.mu, scale)

        log_H = np.log(self.hazard)
        log_1mH = np.log1p(-self.hazard)

        # Growth: R_t(r+1) = R_{t-1}(r) * pred(r) * (1 - H)
        log_growth = self.log_R + log_pred + log_1mH
        # Changepoint: R_t(0) = sum_r R_{t-1}(r) * pred(r) * H
        log_cp = _logsumexp(self.log_R + log_pred + log_H)

        new_log_R = np.concatenate([[log_cp], log_growth])
        new_log_R -= _logsumexp(new_log_R)  # normalize

        # NIG parameter update (Murphy 2007) for each surviving hypothesis,
        # PLUS a fresh r=0 hypothesis reset to the prior.
        new_mu = np.concatenate([[self.mu0], (self.kappa * self.mu + x) / (self.kappa + 1.0)])
        new_kappa = np.concatenate([[self.kappa0], self.kappa + 1.0])
        new_alpha = np.concatenate([[self.alpha0], self.alpha + 0.5])
        new_beta = np.concatenate([
            [self.beta0],
            self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
        ])

        # Prune negligible-mass hypotheses to keep support bounded.
        R = np.exp(new_log_R)
        keep = R >= self.prune_threshold
        keep[0] = True  # always keep the just-reset r=0 hypothesis
        if keep.sum() > self.max_run_length:
            # keep the top-`max_run_length` by mass if pruning alone
            # isn't enough (defensive; rare in practice).
            top_idx = np.argsort(R)[::-1][: self.max_run_length]
            mask = np.zeros_like(keep)
            mask[top_idx] = True
            keep = mask

        self.log_R = new_log_R[keep] - _logsumexp(new_log_R[keep])
        self.mu, self.kappa, self.alpha, self.beta = (
            new_mu[keep], new_kappa[keep], new_alpha[keep], new_beta[keep],
        )

        R_norm = np.exp(self.log_R)
        self._last_changepoint_prob = float(R_norm[0])
        self._last_young_run_prob = float(R_norm[: self.r_min + 1].sum())
        self._last_map_run_length = int(np.argmax(R_norm))
        self.n_events += 1

    def is_ready(self) -> bool:
        # See __init__'s warm_up_events docstring: below this many events,
        # P(r_t <= r_min) is trivially 1.0 regardless of the data.
        return self.n_events > self.warm_up_events

    def evaluate_drift(self) -> dict:
        """Flags a change using P(r_t <= r_min), NOT P(r_t = 0) alone.

        Why: once the run-length posterior concentrates on a single
        dominant hypothesis r* (which happens quickly in practice), the
        one-step update gives
            R_t(0)     ~= R_{t-1}(r*) * pred_{r*}(x_t) * H
            R_t(r*+1)  ~= R_{t-1}(r*) * pred_{r*}(x_t) * (1 - H)
        i.e. the SAME predictive-likelihood factor pred_{r*}(x_t) appears
        in both, so it cancels out of their ratio: R_t(0) / R_t(r*+1) ->
        H / (1-H) essentially independent of how surprising x_t was. In
        other words P(r_t = 0) alone converges to (approximately) the bare
        hazard rate and stops tracking the data at all once the posterior
        is confident about the current run -- verified directly during
        development (see tests/test_bocpd.py and the PR/commit history):
        on an injected mean shift, P(r_t=0) stayed pinned at ~hazard while
        map_run_length correctly collapsed to a small value within a
        couple of events. That collapse -- posterior mass moving onto
        SMALL run lengths generally, not specifically r=0 at one instant
        -- is the real signature of a changepoint just having occurred,
        which is exactly what P(r_t <= r_min) measures.
        """
        young_prob = self._last_young_run_prob
        shift_detected = young_prob >= self.changepoint_prob_threshold
        return {
            "ready": True,
            "changepoint_prob": self._last_changepoint_prob,  # P(r=0) -- diagnostic only, see docstring
            "young_run_prob": young_prob,                      # P(r<=r_min) -- the actual detection statistic
            "map_run_length": self._last_map_run_length,
            "threshold": self.changepoint_prob_threshold,
            "r_min": self.r_min,
            "shift_detected": bool(shift_detected),
            "n_active_hypotheses": int(len(self.log_R)),
        }


def _logsumexp(log_values: np.ndarray) -> float:
    m = np.max(log_values)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(log_values - m)))
