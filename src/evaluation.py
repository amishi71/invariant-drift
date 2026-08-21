"""Grades detector/ACI/FDR behavior against ground truth.

As in cms-streaming-shift-detection's evaluation philosophy: detectors and
the ACI/FDR layer only ever see the residual/anomaly-score stream. Ground-
truth labels (from drift_sim's true_label / true_onset_event) are used
here, in evaluation code only, to measure detection latency and false-
alarm behavior -- never fed to a detector or to the ACI/FDR update calls.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Component 1: change-point detector evaluation (ARL, latency, false-alarm
# rate).
# --------------------------------------------------------------------------

@dataclass
class DetectionResult:
    detected: bool
    detection_index: Optional[int]
    n_events_run: int
    true_onset_event: Optional[int] = None
    latency: Optional[int] = None  # detection_index - true_onset_event, if both known


def run_detector_on_residuals(
    detector, residuals: np.ndarray, true_onset_event: Optional[int] = None,
) -> DetectionResult:
    """Streams a precomputed residual array through a detector's
    update/is_ready/evaluate_drift interface (shared by every detector in
    src/detectors/) and returns the first detection, if any.
    """
    for i, r in enumerate(residuals):
        detector.update(float(r))
        if detector.is_ready():
            result = detector.evaluate_drift()
            if result["shift_detected"]:
                latency = None
                if true_onset_event is not None and i >= true_onset_event:
                    latency = i - true_onset_event
                return DetectionResult(True, i, i + 1, true_onset_event, latency)
    return DetectionResult(False, None, len(residuals), true_onset_event, None)


def average_run_length(
    detector_factory: Callable[[], object],
    stable_residual_factory: Callable[[int, int], np.ndarray],
    n_trials: int = 20,
    max_events: int = 20_000,
) -> dict:
    """ARL: mean number of events to a FALSE alarm under stable (no-drift)
    conditions, across n_trials independent draws.

    :param stable_residual_factory: callable(seed, n_events) -> residual
        array under stable/no-drift conditions.
    """
    run_lengths = []
    n_false_alarms = 0
    for trial in range(n_trials):
        detector = detector_factory()
        residuals = stable_residual_factory(trial, max_events)
        result = run_detector_on_residuals(detector, residuals)
        if result.detected:
            n_false_alarms += 1
            run_lengths.append(result.detection_index + 1)
        else:
            run_lengths.append(max_events)  # censored at max_events
    return {
        "arl_mean": float(np.mean(run_lengths)),
        "arl_std": float(np.std(run_lengths)),
        "false_alarm_rate": n_false_alarms / n_trials,
        "n_trials": n_trials,
        "max_events": max_events,
    }


def detection_latency(
    detector_factory: Callable[[], object],
    drift_residual_factory: Callable[[int, int], "tuple[np.ndarray, int]"],
    n_trials: int = 20,
    max_events: int = 20_000,
) -> dict:
    """Mean/median detection latency (events after true onset) across
    n_trials, plus miss rate.

    :param drift_residual_factory: callable(seed, n_events) ->
        (residual_array, true_onset_event).
    """
    latencies = []
    n_missed = 0
    for trial in range(n_trials):
        detector = detector_factory()
        residuals, onset = drift_residual_factory(trial, max_events)
        result = run_detector_on_residuals(detector, residuals, true_onset_event=onset)
        if result.detected and result.latency is not None:
            latencies.append(result.latency)
        else:
            n_missed += 1
    out = {
        "n_trials": n_trials, "n_missed": n_missed,
        "miss_rate": n_missed / n_trials,
    }
    if latencies:
        out["latency_mean"] = float(np.mean(latencies))
        out["latency_median"] = float(np.median(latencies))
        out["latency_std"] = float(np.std(latencies))
    else:
        out["latency_mean"] = out["latency_median"] = out["latency_std"] = None
    return out


# --------------------------------------------------------------------------
# Component 2: ACI / online-FDR evaluation (empirical coverage, online FDR,
# detection efficiency vs. a fixed-threshold baseline).
# --------------------------------------------------------------------------

def evaluate_aci_coverage(aci_history: List[dict], alpha_target: float) -> dict:
    """Empirical coverage of an AdaptiveConformalThreshold run vs. the
    nominal target, from its accumulated update-call history.
    """
    live = [h for h in aci_history if not h.get("skipped")]
    if not live:
        return {"empirical_miscoverage": None, "nominal_alpha": alpha_target}
    miscov = np.mean([1.0 if h["miscoverage"] else 0.0 for h in live])
    return {
        "empirical_miscoverage": float(miscov),
        "nominal_alpha": alpha_target,
        "n_points": len(live),
        "gap": float(miscov - alpha_target),
    }


def evaluate_online_fdr(fdr_controller, true_window_labels: np.ndarray) -> dict:
    """Empirical false discovery proportion of a LORD/SAFFRON run's
    rejections against ground-truth window labels (1 = window genuinely
    overlaps a true drift period, 0 = pure background window), plus power
    (fraction of true-drift windows correctly rejected).
    """
    rejections = set(fdr_controller.rejections)
    n_rejections = len(rejections)
    if n_rejections == 0:
        false_discoveries = 0
        fdp = 0.0
    else:
        false_discoveries = sum(1 for r in rejections if true_window_labels[r] == 0)
        fdp = false_discoveries / n_rejections

    n_true_drift_windows = int(np.sum(true_window_labels == 1))
    true_positives = sum(1 for r in rejections if true_window_labels[r] == 1)
    power = (true_positives / n_true_drift_windows) if n_true_drift_windows > 0 else None

    return {
        "n_tests": fdr_controller.n_tests,
        "n_rejections": n_rejections,
        "false_discovery_proportion": fdp,
        "power": power,
        "final_wealth": fdr_controller.wealth,
    }


def detection_efficiency_vs_fixed_threshold(
    scores: np.ndarray,
    true_labels: np.ndarray,
    adaptive_flags: np.ndarray,
    fixed_threshold: float,
) -> dict:
    """Compares the ACI-adaptive flagging decisions against a naive fixed
    threshold (frozen at burn-in, never updated) on the same score stream
    -- the abstract's explicit baseline comparison for the conformal
    layer's "detection efficiency."
    """
    fixed_flags = scores > fixed_threshold
    true_labels = np.asarray(true_labels, dtype=bool)

    def _metrics(flags):
        flags = np.asarray(flags, dtype=bool)
        tp = int(np.sum(flags & true_labels))
        fp = int(np.sum(flags & ~true_labels))
        fn = int(np.sum(~flags & true_labels))
        tn = int(np.sum(~flags & ~true_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "false_positive_rate": fpr}

    return {
        "adaptive": _metrics(adaptive_flags),
        "fixed_threshold": _metrics(fixed_flags),
        "fixed_threshold_value": float(fixed_threshold),
    }
