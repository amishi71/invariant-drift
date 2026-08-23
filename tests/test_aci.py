import numpy as np
import pytest

from src.conformal.aci import AdaptiveConformalThreshold


class TestACI:
    def test_tracks_target_miscoverage_under_stationary_noise(self):
        rng = np.random.RandomState(0)
        burn_in = rng.gamma(2.0, 1.0, 500)
        aci = AdaptiveConformalThreshold.from_burn_in(
            burn_in, alpha_target=0.05, gamma=0.02, calibration_window=1000,
        )
        stream = rng.gamma(2.0, 1.0, 5000)
        for x in stream:
            aci.update(float(x))
        coverage = aci.empirical_coverage(last_n=3000)
        # Should be close to 1 - alpha_target = 0.95, well within a
        # generous tolerance given stochastic online adaptation.
        assert coverage == pytest.approx(0.95, abs=0.03)

    def test_alpha_t_stays_within_bounds(self):
        rng = np.random.RandomState(1)
        burn_in = rng.normal(0, 1, 200)
        aci = AdaptiveConformalThreshold.from_burn_in(burn_in, alpha_target=0.1, gamma=0.5)
        # Deliberately adversarial stream (all huge miscoverages) to stress alpha_t
        for _ in range(500):
            aci.update(1e6)
        assert aci.alpha_min <= aci.alpha_t <= aci.alpha_max

    def test_delayed_update_skips_non_background(self):
        rng = np.random.RandomState(2)
        burn_in = rng.normal(0, 1, 200)
        aci = AdaptiveConformalThreshold.from_burn_in(burn_in, alpha_target=0.05)
        record = aci.update_delayed(5.0, true_label_background=False)
        assert record.get("skipped") is True

    def test_widens_after_miscoverage(self):
        """After a control point exceeds the threshold, alpha_t should
        decrease (per the ACI update rule), which pushes the threshold
        up (stricter / wider) on the next call -- verify the direction of
        the very first adaptation step explicitly."""
        rng = np.random.RandomState(3)
        burn_in = rng.normal(0, 1, 500)
        aci = AdaptiveConformalThreshold.from_burn_in(burn_in, alpha_target=0.05, gamma=0.1)
        alpha_before = aci.alpha_t
        threshold_before = aci.current_threshold()
        aci.update(threshold_before + 10.0)  # force a miscoverage
        assert aci.alpha_t < alpha_before

    def test_frozen_buffer_does_not_grow(self):
        """adapt_calibration_buffer=False: alpha_t still adapts, but the
        calibration buffer used for the quantile stays exactly at its
        burn-in size -- the ablation control for
        scripts/aci_recall_gap_ablation.py."""
        rng = np.random.RandomState(4)
        burn_in = rng.normal(0, 1, 300)
        aci = AdaptiveConformalThreshold.from_burn_in(
            burn_in, alpha_target=0.05, gamma=0.1, adapt_calibration_buffer=False,
        )
        alpha_before = aci.alpha_t
        for x in rng.normal(0, 1, 200):
            aci.update(float(x))
        assert len(aci._calib_scores) == 300  # unchanged from burn-in
        assert aci.alpha_t != alpha_before  # alpha_t still adapted

    def test_sliding_buffer_grows_by_default(self):
        rng = np.random.RandomState(5)
        burn_in = rng.normal(0, 1, 300)
        aci = AdaptiveConformalThreshold.from_burn_in(burn_in, alpha_target=0.05, gamma=0.1)
        for x in rng.normal(0, 1, 200):
            aci.update(float(x))
        assert len(aci._calib_scores) == 500  # grew by the 200 live updates