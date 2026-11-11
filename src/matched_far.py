"""Tools for comparing change-point detectors at a MATCHED, HELD-OUT false-alarm
rate.

Why this exists. Detectors run at different operating points have different
false-alarm rates, so their detection rates are not comparable (an
oracle-direction CUSUM at k=0.1, h=16 "detects" 94% of masked-channel trials
but also alarms before onset in 46% of drift-free windows). The fair comparison
fixes the false-alarm probability over a window of T events, calibrates each
detector's threshold to it on one set of drift-free windows, verifies the
achieved rate on a second, independent set, and only then measures detection on
windows with injected drift.

Key trick. For CUSUM-type statistics the first alarm for threshold h is the
first time the running maximum of the statistic reaches h. One pass over a
window therefore yields the first-alarm time for EVERY h in a grid
(`cusum_runmax` + `first_passage`), exactly matching `CUSUMDetector` semantics
(alarm when S >= h, first alarm only). The same holds for any statistic whose
alarm is "statistic >= threshold" (for example BOCPD's young-run probability).

Everything here is numpy-only except the optional BOCPD/ADWIN helpers, which
import the repo's own detector classes.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

AFFECTED_FEATURES_JETHT = (0, 3, 9, 7)  # jet1_pt, jet2_pt, ht, met_pt (as in abrupt.py)
AFFECTED_FEATURES_L1 = (4, 8, 9)        # lead jet Et, HT, MET (CMS L1T layout, src/feature_extraction_cmsl1t.py)
AFFECTED_FEATURES = AFFECTED_FEATURES_JETHT


def set_affected_features_for_cache(cache_path: str) -> tuple:
    """L1T caches carry a 'run' array; the JetHT cache does not. Selects the injected columns."""
    global AFFECTED_FEATURES
    with np.load(cache_path) as z:
        is_l1 = "run" in z.files
    AFFECTED_FEATURES = AFFECTED_FEATURES_L1 if is_l1 else AFFECTED_FEATURES_JETHT
    return AFFECTED_FEATURES


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def zscore(series: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Standardize by the reference mean/std (population std, as CUSUMDetector)."""
    return (series - float(np.mean(ref))) / float(np.std(ref))


# ----------------------------------------------------------------------------
# Windows and injection
# ----------------------------------------------------------------------------
def take_windows(arr: np.ndarray, starts: Sequence[int], T: int) -> np.ndarray:
    idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(T)[None, :]
    return arr[idx]


def inject_masked(feats: np.ndarray, drop: float) -> np.ndarray:
    """Multiplicative drop on energy-like features from the first event of each
    window (fresh start at onset). Same math as abrupt.masked_channel_stream /
    real_data_injection.real_masked_channel_stream after the changepoint.
    feats: (n_windows, T, n_features)."""
    out = np.array(feats, dtype=np.float64, copy=True)
    cols = list(AFFECTED_FEATURES)
    out[..., cols] = out[..., cols] * (1.0 - drop)
    return out


def inject_radiation(feats: np.ndarray, tau: float, gain0: float = 1.0) -> np.ndarray:
    """gain(i) = gain0 * exp(-i / tau) from the first event of each window. Same
    math as radiation_damage_stream / real_radiation_damage_stream."""
    out = np.array(feats, dtype=np.float64, copy=True)
    T = out.shape[1]
    gain = gain0 * np.exp(-np.arange(T) / tau)
    cols = list(AFFECTED_FEATURES)
    out[..., cols] = out[..., cols] * gain[None, :, None]
    return out


# ----------------------------------------------------------------------------
# CUSUM first-passage machinery
# ----------------------------------------------------------------------------
def cusum_runmax(z: np.ndarray, k: float, mode: str = "two") -> np.ndarray:
    """Running maximum of the CUSUM statistic for standardized z (n_windows, T).

    mode: "two"  -> max(S+, S-)  (two-sided, as CUSUMDetector(two_sided=True))
          "up"   -> S+ only      (watches upward shifts)
          "down" -> S- only      (watches downward shifts; equals the repo's
                                  one-sided detector on the negated series)
    Returns an array of the same shape, non-decreasing along axis 1."""
    if mode not in ("two", "up", "down"):
        raise ValueError(mode)
    n, T = z.shape
    sp = np.zeros(n)
    sn = np.zeros(n)
    run = np.zeros(n)
    out = np.empty((n, T))
    for t in range(T):
        zt = z[:, t]
        if mode in ("two", "up"):
            sp = np.maximum(0.0, sp + zt - k)
        if mode in ("two", "down"):
            sn = np.maximum(0.0, sn - zt - k)
        run = np.maximum(run, np.maximum(sp, sn))
        out[:, t] = run
    return out


def first_passage(runmax: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """For each window and each threshold, the first index t with runmax[t] >=
    threshold, or T if it never happens. runmax must be non-decreasing per row."""
    params = np.asarray(params, dtype=np.float64)
    n, _ = runmax.shape
    out = np.empty((n, len(params)), dtype=np.int64)
    for i in range(n):
        out[i] = np.searchsorted(runmax[i], params, side="left")
    return out


def calibrate_param(fp_cal: np.ndarray, T: int, target: float):
    """Index of the MOST SENSITIVE parameter whose calibration false-alarm
    probability (fraction of windows with an alarm inside T events) is <= target.
    Parameters must be ordered from most to least sensitive (for CUSUM: h
    ascending). Returns (index or None, far-per-parameter array)."""
    far = (fp_cal < T).mean(axis=0)
    ok = np.where(far <= target)[0]
    if len(ok) == 0:
        return None, far
    return int(ok[0]), far


def outcome_counts(fp: np.ndarray, j: int, T: int):
    """(n_alarm, n_windows, delays) for parameter index j."""
    col = fp[:, j]
    alarmed = col < T
    return int(alarmed.sum()), len(col), col[alarmed].astype(int).tolist()


# ----------------------------------------------------------------------------
# Optional detectors using the repo's own classes (heavier; opt-in)
# ----------------------------------------------------------------------------
def bocpd_young_prob_paths(series: np.ndarray, ref: np.ndarray, **bocpd_kwargs) -> np.ndarray:
    """Per-window path of BOCPD's detection statistic P(r_t <= r_min), 0 before the
    detector is ready. Feed the result to np.maximum.accumulate + first_passage with
    thresholds in (0, 1]."""
    from src.detectors.bocpd import BOCPD
    n, T = series.shape
    paths = np.zeros((n, T))
    for i in range(n):
        det = BOCPD(ref, **bocpd_kwargs)
        for t in range(T):
            det.update(float(series[i, t]))
            if det.is_ready():
                paths[i, t] = det.evaluate_drift()["young_run_prob"]
    return paths


def adwin_first_alarm(series: np.ndarray, delta: float) -> np.ndarray:
    """First alarm index per window for river's ADWIN via the repo wrapper
    (T if none)."""
    from src.detectors.adwin import ADWINDetector
    n, T = series.shape
    out = np.full(n, T, dtype=np.int64)
    for i in range(n):
        det = ADWINDetector(None, delta=delta)
        for t in range(T):
            det.update(float(series[i, t]))
            if det.evaluate_drift()["shift_detected"]:
                out[i] = t
                break
    return out


# ----------------------------------------------------------------------------
# Reference: ideal Gaussian false-alarm probability
# ----------------------------------------------------------------------------
def gaussian_far(h: float, k: float, T: int, mean: float = 0.0, std: float = 1.0,
                 trials: int = 20_000, seed: int = 0, mode: str = "two") -> float:
    """P(alarm within T events) for CUSUM on i.i.d. N(mean, std^2) standardized
    data (mean/std are the held-out residual's offset/scale RELATIVE to the
    reference). Monte Carlo."""
    rng = np.random.default_rng(seed)
    sp = np.zeros(trials)
    sn = np.zeros(trials)
    hit = np.zeros(trials, dtype=bool)
    for _ in range(T):
        z = mean + std * rng.standard_normal(trials)
        if mode in ("two", "up"):
            sp = np.maximum(0.0, sp + z - k)
        if mode in ("two", "down"):
            sn = np.maximum(0.0, sn - z - k)
        hit |= (sp >= h) | (sn >= h)
    return float(hit.mean())


# ----------------------------------------------------------------------------
# Pool sources: a contiguous stream of events, split into independent regions
# ----------------------------------------------------------------------------
class RealPool:
    """The cached real JetHT feature pool. Region i of n is the i-th contiguous
    equal slice, so different seeds use disjoint, differently-located data."""

    def __init__(self, cache_path: str = "data/real_cache/jetht_features.npz"):
        from src.drift_sim.real_data_injection import load_real_feature_pool
        self.features, self.pileup, self.n_jet = load_real_feature_pool(cache_path)
        print("RealPool: injecting drift into feature columns %s" % (set_affected_features_for_cache(cache_path),))
        self.name = "real"

    def region(self, idx: int, n_regions: int):
        L = len(self.features) // n_regions
        s = idx * L
        return self.features[s:s + L], self.pileup[s:s + L], self.n_jet[s:s + L]


class SyntheticPool:
    """Stationary i.i.d. synthetic events (the repo's own generator, vectorised).
    A control: with stationary data a correct procedure must give nominal FAR."""

    def __init__(self, region_events: int = 1_300_000, seed_base: int = 900_000):
        self.region_events = region_events
        self.seed_base = seed_base
        self.name = "synthetic"

    def region(self, idx: int, n_regions: int):
        from src.stream_loader import synthetic_burnin_conditions, _synthesize_object_features
        rng = np.random.RandomState(self.seed_base + idx)
        cond = synthetic_burnin_conditions(rng, self.region_events)
        feats = _synthesize_object_features(rng, cond["pileup"], cond["n_jet"])
        return feats, cond["pileup"], cond["n_jet"]
