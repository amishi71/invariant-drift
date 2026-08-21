import numpy as np
import pytest

from src.detectors.bocpd import BOCPD


@pytest.fixture
def reference():
    rng = np.random.RandomState(0)
    return rng.normal(0, 1, 500)


class TestBOCPD:
    def test_low_false_alarm_rate_on_pure_noise(self, reference):
        """Gates on is_ready() before trusting evaluate_drift(), same
        contract as evaluation.py's run_detector_on_residuals -- without
        that gate, P(r_t<=r_min) is mechanically 1.0 for the first few
        events regardless of data (a real bug caught during development;
        see bocpd.py's warm_up_events docstring and README's "Known
        issues" section), which would make this test fail unconditionally
        and for the wrong reason."""
        rng = np.random.RandomState(1)
        n_alarms = 0
        n_trials = 15
        for trial in range(n_trials):
            d = BOCPD(reference, hazard_lambda=250.0, changepoint_prob_threshold=0.6)
            stream = rng.normal(0, 1, 800)
            for x in stream:
                d.update(x)
                if d.is_ready() and d.evaluate_drift()["shift_detected"]:
                    n_alarms += 1
                    break
        assert n_alarms / n_trials < 0.5

    def test_evaluate_drift_before_warm_up_is_trivially_saturated(self, reference):
        """Documents the mechanical P(r<=r_min)=1.0 startup behavior
        directly (rather than letting it be an unexplained surprise): this
        is exactly why is_ready() / warm_up_events exists."""
        d = BOCPD(reference, r_min=4)
        d.update(0.0)
        assert d.is_ready() is False
        assert d.evaluate_drift()["young_run_prob"] == pytest.approx(1.0)

    def test_detects_mean_shift(self, reference):
        rng = np.random.RandomState(2)
        d = BOCPD(reference, hazard_lambda=250.0, changepoint_prob_threshold=0.5)
        stream = np.concatenate([rng.normal(0, 1, 200), rng.normal(4.0, 1, 200)])
        detected_after_shift = False
        for i, x in enumerate(stream):
            d.update(x)
            if i >= 200 and d.is_ready() and d.evaluate_drift()["shift_detected"]:
                detected_after_shift = True
                break
        assert detected_after_shift

    def test_run_length_posterior_sums_to_one(self, reference):
        d = BOCPD(reference)
        rng = np.random.RandomState(3)
        for x in rng.normal(0, 1, 100):
            d.update(x)
            assert np.exp(d.log_R).sum() == pytest.approx(1.0, abs=1e-6)

    def test_not_ready_before_first_update(self, reference):
        d = BOCPD(reference)
        assert d.is_ready() is False

    def test_pruning_keeps_support_bounded(self, reference):
        d = BOCPD(reference, prune_threshold=1e-3, max_run_length=200)
        rng = np.random.RandomState(4)
        for x in rng.normal(0, 1, 2000):
            d.update(x)
        assert len(d.log_R) <= 200
