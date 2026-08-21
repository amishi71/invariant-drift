import numpy as np
import pytest

from src.detectors.cusum import CUSUMDetector
from src.detectors.page_hinkley import PageHinkleyDetector


@pytest.fixture
def reference():
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, 2000)


def _run_to_alarm(detector, stream, max_events=None):
    for i, x in enumerate(stream):
        if max_events is not None and i >= max_events:
            return None
        detector.update(x)
        if detector.evaluate_drift()["shift_detected"]:
            return i
    return None


class TestCUSUM:
    def test_false_alarm_rate_matches_arl_theory_on_pure_noise(self, reference):
        """NOT a zero-tolerance test: with k=0.5, h=10.0, Siegmund's ARL0
        approximation gives ARL0 ~= 44,000 (one-sided) / ~22,000
        (two-sided), so across 30 trials x 2000 events = 60,000 total
        test-events, ~2-3 false alarms are EXPECTED by chance, not a bug.
        Asserting a hard zero across a handful of trials is a statistical
        test-design error (caught during development -- see README's
        "Known issues" section); the correct check is that the empirical
        rate over many trials stays in the right ballpark, generously
        bounded, rather than exactly zero.
        """
        rng = np.random.RandomState(1)
        n_trials = 30
        n_events = 2000
        n_alarms = 0
        for trial in range(n_trials):
            d = CUSUMDetector(reference, k=0.5, h=10.0, two_sided=True)
            stream = rng.normal(0, 1, n_events)
            if _run_to_alarm(d, stream) is not None:
                n_alarms += 1
        # Generous upper bound: theory predicts ~2-3 alarms; even 10x that
        # would indicate a real problem.
        assert n_alarms <= 15, (
            f"{n_alarms}/{n_trials} trials false-alarmed on pure noise -- "
            f"theory predicts ~2-3, this many suggests a real miscalibration."
        )

    def test_detects_upward_shift(self, reference):
        rng = np.random.RandomState(2)
        d = CUSUMDetector(reference, k=0.5, h=8.0, two_sided=True)
        stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(3.0, 1, 300)])
        idx = _run_to_alarm(d, stream)
        assert idx is not None
        assert idx >= 300  # shouldn't fire before the injected shift

    def test_detects_downward_shift_two_sided(self, reference):
        rng = np.random.RandomState(3)
        d = CUSUMDetector(reference, k=0.5, h=8.0, two_sided=True)
        stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(-3.0, 1, 300)])
        idx = _run_to_alarm(d, stream)
        assert idx is not None
        assert idx >= 300

    def test_rejects_zero_variance_reference(self):
        with pytest.raises(ValueError):
            CUSUMDetector(np.ones(100), k=0.5, h=8.0)

    def test_not_ready_before_first_update(self, reference):
        d = CUSUMDetector(reference)
        assert d.is_ready() is False


class TestPageHinkley:
    """Includes a regression test for a real bug found during development:
    the two-sided down branch tracked a running MAXIMUM of the negated
    cumulative sum instead of a running MINIMUM, which made the down-side
    statistic grow ~linearly with event count (driven by the -delta drift
    term alone) regardless of any real shift -- i.e. guaranteed false
    alarms within a few dozen events on pure noise. See git history / the
    comment in src/detectors/page_hinkley.py for the fix.
    """

    def test_false_alarm_rate_matches_arl_theory_on_pure_noise(self, reference):
        """Same statistical (not zero-tolerance) design as
        TestCUSUM's equivalent test, and for the same reason: PH's
        statistic is provably identical to CUSUM's for k=delta (see
        test_matches_cusum_equivalence below), so it has the same ARL0
        and the same handful-of-alarms-in-many-trials is expected. This
        specific test is also the regression guard for the real bug found
        during development (down-side running MAX instead of MIN, which
        caused ~100% false-alarm rate within ~20-45 events, not an
        occasional late alarm) -- that bug would blow this bound by
        orders of magnitude, not marginally.
        """
        rng = np.random.RandomState(1)
        n_trials = 30
        n_events = 2000
        n_alarms = 0
        for trial in range(n_trials):
            d = PageHinkleyDetector(reference, delta=0.5, lam=10.0, two_sided=True)
            stream = rng.normal(0, 1, n_events)
            if _run_to_alarm(d, stream) is not None:
                n_alarms += 1
        assert n_alarms <= 15, (
            f"{n_alarms}/{n_trials} trials false-alarmed on pure noise -- "
            f"theory predicts ~2-3. This many strongly suggests the "
            f"down-side running-max/min regression is back."
        )

    def test_down_side_stat_matches_cusum_construction(self, reference):
        """ph_stat_down should equal cumulative_sum_neg minus its running
        MINIMUM (not maximum) at every step -- direct structural check."""
        rng = np.random.RandomState(4)
        d = PageHinkleyDetector(reference, delta=0.5, lam=1e9, two_sided=True)  # lam huge: never resets
        running_min = 0.0
        cum = 0.0
        for x in rng.normal(0, 1, 200):
            d.update(x)
            z = (x - d.mu0) / d.sigma0
            cum += -z - d.delta
            running_min = min(running_min, cum)
            expected_stat = cum - running_min
            assert d.cumulative_sum_neg == pytest.approx(cum)
            assert d.min_cumulative_sum_neg == pytest.approx(running_min)
            actual_stat = d.evaluate_drift()["ph_stat_down"]
            assert actual_stat == pytest.approx(expected_stat, abs=1e-8)

    def test_detects_upward_shift(self, reference):
        rng = np.random.RandomState(2)
        d = PageHinkleyDetector(reference, delta=0.5, lam=10.0, two_sided=True)
        stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(3.0, 1, 300)])
        idx = _run_to_alarm(d, stream)
        assert idx is not None
        assert idx >= 300

    def test_detects_downward_shift(self, reference):
        rng = np.random.RandomState(5)
        d = PageHinkleyDetector(reference, delta=0.5, lam=10.0, two_sided=True)
        stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(-3.0, 1, 300)])
        idx = _run_to_alarm(d, stream)
        assert idx is not None
        assert idx >= 300
        result = d.evaluate_drift()
        assert result["direction"] == "down"

    def test_matches_cusum_equivalence(self, reference):
        """Well-known equivalence: PH's up-side stat (cumsum minus running
        min) equals CUSUM's S+ recursion exactly, for k=delta, same input.
        """
        from src.detectors.cusum import CUSUMDetector

        rng = np.random.RandomState(6)
        stream = rng.normal(0, 1, 300)
        ph = PageHinkleyDetector(reference, delta=0.5, lam=1e9, two_sided=False)
        cu = CUSUMDetector(reference, k=0.5, h=1e9, two_sided=False)
        for x in stream:
            ph.update(x)
            cu.update(x)
            ph_stat = ph.cumulative_sum - ph.min_cumulative_sum
            assert ph_stat == pytest.approx(cu.S_pos, abs=1e-8)
