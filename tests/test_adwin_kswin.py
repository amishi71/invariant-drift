import numpy as np
import pytest

from src.detectors.adwin import ADWINDetector
from src.detectors.kswin import KSWINDetector


class TestADWIN:
    def test_no_false_alarm_on_pure_noise(self):
        rng = np.random.RandomState(0)
        d = ADWINDetector(delta=0.002)
        stream = rng.normal(0, 1, 2000)
        detected = False
        for x in stream:
            d.update(x)
            if d.evaluate_drift()["shift_detected"]:
                detected = True
                break
        assert not detected

    def test_detects_mean_shift(self):
        rng = np.random.RandomState(1)
        d = ADWINDetector(delta=0.002)
        stream = np.concatenate([rng.normal(0, 1, 500), rng.normal(3.0, 1, 500)])
        detected_after = False
        for i, x in enumerate(stream):
            d.update(x)
            if i >= 500 and d.evaluate_drift()["shift_detected"]:
                detected_after = True
                break
        assert detected_after


class TestKSWIN:
    def test_default_alpha_has_reasonable_false_alarm_rate(self):
        """Regression guard for a real finding: river's KSWIN retests on
        EVERY update once its window fills (confirmed from river's
        source), so river's own default alpha=0.005 (and the even looser
        0.01) gives a false-alarm rate near 100% over a stream of a few
        thousand events -- not a rare edge case, reproduced in 10/10
        trials during development. This project's default (alpha=1e-3,
        see kswin.py's docstring) should stay well below that.
        """
        n_trials = 10
        n_alarms = 0
        for t in range(n_trials):
            d = KSWINDetector(alpha=1e-3, window_size=200, stat_size=40, seed=t)
            stream = np.random.RandomState(100 + t).normal(0, 1, 1500)
            for x in stream:
                d.update(x)
                if d.evaluate_drift()["shift_detected"]:
                    n_alarms += 1
                    break
        assert n_alarms / n_trials < 0.5

    def test_detects_mean_shift(self):
        rng = np.random.RandomState(2)
        d = KSWINDetector(alpha=0.01, window_size=100, stat_size=30, seed=0)
        stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(3.0, 1, 300)])
        detected_after = False
        for i, x in enumerate(stream):
            d.update(x)
            if i >= 300 and d.evaluate_drift()["shift_detected"]:
                detected_after = True
                break
        assert detected_after

    def test_not_ready_before_first_update(self):
        d = KSWINDetector()
        assert d.is_ready() is False
