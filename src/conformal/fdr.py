"""Online false discovery rate (FDR) control over windowed batches: LORD
and SAFFRON.

Why this on top of ACI: ACI (aci.py) controls a *per-event* miscoverage
rate, which is the right tool for keeping the anomaly-score threshold
calibrated. But triggers process millions of events/second, so even a
well-calibrated per-event false-positive rate produces a large absolute
number of flagged events -- single-event error control alone doesn't say
anything about the rate of "batches that look like a genuine drift" vs.
"batches that just had a few of the expected false positives." That's a
multiple-testing problem over a stream of windowed-batch hypothesis tests
("does this window's flag rate exceed what the calibrated threshold
predicts"), which is what online FDR control is for.

References:
- Foster & Stine, "alpha-investing: A procedure for sequential control of
  expected false discoveries" (2008) -- the generalized alpha-investing
  framework both algorithms below are instances of.
- Javanmard & Montanari, "Online Rules for Control of False Discovery
  Rate and False Discovery Exceedance" (2018), arXiv:1502.06197 -- LORD.
- Ramdas, Yang, Wainwright, Jordan, "SAFFRON: an adaptive algorithm for
  online control of the false discovery rate" (2018), arXiv:1802.09098.

HONESTY NOTE on fidelity: both algorithms below are implemented as valid
instances of the Foster-Stine generalized alpha-investing recursion (full
wealth W_0, pay alpha_i on a test, get alpha back in full on a rejection,
pay a small "rent" alpha_i/(1-alpha_i) on a non-rejection -- this specific
accounting is what makes the wealth process a supermartingale under the
global null, which is the actual source of the FDR-control guarantee).
This gives a procedure that controls mFDR (marginal FDR, the standard
guarantee in this literature) under the same p-value-validity assumptions
as the source papers. The wealth-ALLOCATION rule (the gamma_i sequence,
and for SAFFRON, the candidate-discarding mechanism) follows the papers'
core ideas, but the exact index bookkeeping in this implementation is a
simplified, self-consistent variant rather than a byte-exact port of
either paper's numbered algorithm -- cross-check against the paper before
treating this as a reference implementation of a specific published
recursion. The validity argument (supermartingale wealth process =>
mFDR control) holds regardless of that simplification.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.stats import binomtest


def _gamma_sequence(n: int) -> np.ndarray:
    """Nonincreasing, summable sequence with sum_i gamma_i = 1 over the
    first n terms: gamma_i ∝ 1 / (i * log(max(i, 2))^2), i = 1..n.
    Normalized numerically over a finite horizon `n` -- legitimate here
    because both algorithms are run "over windowed batches" (the
    abstract's phrasing), i.e. a bounded number of sequential tests per
    run, not a literal infinite stream.
    """
    i = np.arange(1, n + 1, dtype=np.float64)
    raw = 1.0 / (i * np.log(np.maximum(i, 2.0)) ** 2)
    return raw / raw.sum()


@dataclass
class _AlphaInvestingBase:
    """Shared Foster-Stine wealth bookkeeping for LORD and SAFFRON."""

    alpha: float = 0.05
    w0_fraction: float = 0.5
    max_tests: int = 10_000

    wealth: float = field(init=False)
    gamma: np.ndarray = field(init=False)
    n_tests: int = field(default=0, init=False)
    rejections: List[int] = field(default_factory=list, init=False)
    history: List[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.wealth = self.alpha * self.w0_fraction
        self.gamma = _gamma_sequence(self.max_tests)

    def _gamma_i(self, k: int) -> float:
        idx = min(k, self.max_tests) - 1
        return float(self.gamma[idx])

    def _spend(self, alpha_i: float, rejected: bool) -> None:
        if alpha_i <= 0.0:
            return
        if rejected:
            self.wealth += self.alpha - alpha_i
            # self.n_tests is already incremented (1-indexed test count) by
            # the time _spend runs, but `rejections` is consumed elsewhere
            # (evaluation.py's evaluate_online_fdr) as 0-indexed positions
            # into an array of per-window labels/p-values -- store it
            # 0-indexed here so `rejections` and `test_index` (which stays
            # 1-indexed, for human-readable records) don't silently
            # disagree. Caught during development via a genuine
            # IndexError on the very last window (test_index == array
            # length); see tests/test_fdr.py's regression test and
            # README's "Known issues" section.
            self.rejections.append(self.n_tests - 1)
        else:
            self.wealth -= alpha_i / (1.0 - alpha_i)
        self.wealth = max(self.wealth, 0.0)


@dataclass
class LORD(_AlphaInvestingBase):
    """LORD-style online FDR control (see module docstring for the
    fidelity note). Test level for hypothesis i is alpha_i = gamma_i *
    W_{i-1}: proportional to the current wealth, allocated via the fixed
    summable gamma sequence.
    """

    def test(self, p_value: float) -> dict:
        self.n_tests += 1
        alpha_i = self._gamma_i(self.n_tests) * self.wealth
        rejected = p_value <= alpha_i
        self._spend(alpha_i, rejected)
        record = {
            "test_index": self.n_tests, "p_value": p_value,
            "alpha_i": alpha_i, "rejected": bool(rejected), "wealth_after": self.wealth,
        }
        self.history.append(record)
        return record


@dataclass
class SAFFRON(_AlphaInvestingBase):
    """SAFFRON-style online FDR control: adaptively discards p-values above
    a candidate threshold lambda from the wealth-consuming test sequence
    (core SAFFRON mechanism -- see module docstring for fidelity note on
    the simplified candidate-indexing scheme used here), which improves
    power over LORD when a meaningful fraction of tested windows are
    genuine discoveries (i.e. p_i tends to be small, not uniform, under
    H1) by not "spending" a wealth-allocation slot on windows that already
    look null.
    """

    lambda_threshold: float = 0.5
    n_candidates: int = field(default=0, init=False)

    def test(self, p_value: float) -> dict:
        self.n_tests += 1
        is_candidate = p_value <= self.lambda_threshold
        if not is_candidate:
            record = {
                "test_index": self.n_tests, "p_value": p_value,
                "alpha_i": 0.0, "rejected": False, "wealth_after": self.wealth,
                "candidate": False,
            }
            self.history.append(record)
            return record

        self.n_candidates += 1
        alpha_i = self._gamma_i(self.n_candidates) * (1.0 - self.lambda_threshold) * self.wealth
        rejected = p_value <= alpha_i
        self._spend(alpha_i, rejected)
        record = {
            "test_index": self.n_tests, "p_value": p_value,
            "alpha_i": alpha_i, "rejected": bool(rejected), "wealth_after": self.wealth,
            "candidate": True,
        }
        self.history.append(record)
        return record


def windowed_batch_pvalues(
    flags: np.ndarray, window_size: int, nominal_rate: float,
) -> np.ndarray:
    """Converts a stream of per-event boolean anomaly flags into one
    one-sided p-value per non-overlapping window, testing:
        H0: this window's flag rate == nominal_rate (the ACI-calibrated
            target false-positive rate)
        H1: this window's flag rate is elevated (excess flagging,
            consistent with genuine drift rather than expected background
            false positives)
    via the exact binomial test (scipy.stats.binomtest, alternative
    'greater'). This is what turns "does this window look drifted" into
    the p-value sequence LORD/SAFFRON consume, matching the abstract's
    "online false discovery rate control ... evaluated over windowed
    batches."
    """
    flags = np.asarray(flags, dtype=bool)
    n_windows = len(flags) // window_size
    pvals = np.empty(n_windows, dtype=np.float64)
    for w in range(n_windows):
        chunk = flags[w * window_size:(w + 1) * window_size]
        k = int(chunk.sum())
        result = binomtest(k, window_size, nominal_rate, alternative="greater")
        pvals[w] = result.pvalue
    return pvals
