import numpy as np
import pytest

from src.residual import CalibrationModel


def _make_burn_in(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    pileup = rng.uniform(20, 50, n)
    mult = rng.poisson(3, n).astype(np.float64)
    lumi = rng.uniform(0.5, 1.5, n)
    # A positive, right-skewed "score" whose mean depends smoothly on the
    # covariates -- mimics a VAE reconstruction error's shape.
    true_mean = 5.0 + 0.1 * pileup + 0.02 * pileup ** 2 / 10 + 0.5 * mult + 2.0 * lumi
    score = rng.gamma(shape=4.0, scale=true_mean / 4.0, size=n)
    return score, pileup, mult, lumi


class TestCalibrationModel:
    def test_fit_produces_standardized_burn_in_residual(self):
        score, pileup, mult, lumi = _make_burn_in()
        calib = CalibrationModel.fit(score, pileup, mult, lumi)
        resid = calib.residual(score, pileup, mult, lumi)
        assert resid.mean() == pytest.approx(0.0, abs=1e-6)
        assert resid.std() == pytest.approx(1.0, abs=1e-6)

    def test_log_transform_reduces_skew_vs_identity(self):
        """The whole point of defaulting to score_transform='log': verify
        it actually reduces skew relative to the identity transform on a
        right-skewed (gamma-like) score, on held-out data."""
        from scipy.stats import skew

        score, pileup, mult, lumi = _make_burn_in(n=3000, seed=1)
        n_train = 2000
        calib_log = CalibrationModel.fit(
            score[:n_train], pileup[:n_train], mult[:n_train], lumi[:n_train],
            score_transform="log",
        )
        calib_id = CalibrationModel.fit(
            score[:n_train], pileup[:n_train], mult[:n_train], lumi[:n_train],
            score_transform="identity",
        )
        resid_log = calib_log.residual(score[n_train:], pileup[n_train:], mult[n_train:], lumi[n_train:])
        resid_id = calib_id.residual(score[n_train:], pileup[n_train:], mult[n_train:], lumi[n_train:])
        assert abs(skew(resid_log)) < abs(skew(resid_id))

    def test_rejects_insufficient_burn_in(self):
        score, pileup, mult, lumi = _make_burn_in(n=3)
        with pytest.raises(ValueError):
            CalibrationModel.fit(score, pileup, mult, lumi)

    def test_robust_scale_less_sensitive_to_outliers(self):
        score, pileup, mult, lumi = _make_burn_in(n=1000, seed=2)
        score_with_outliers = score.copy()
        score_with_outliers[:5] *= 50  # inject a handful of extreme outliers

        calib_plain = CalibrationModel.fit(score_with_outliers, pileup, mult, lumi, robust=False)
        calib_robust = CalibrationModel.fit(score_with_outliers, pileup, mult, lumi, robust=True)
        # Robust (MAD-based) sigma should be smaller than the outlier-
        # inflated plain std -- that's the entire point of robust=True.
        assert calib_robust.sigma_resid < calib_plain.sigma_resid

    def test_predict_before_fit_raises(self):
        calib = CalibrationModel()
        with pytest.raises(RuntimeError):
            calib.predict(np.array([30.0]), np.array([3.0]), np.array([1.0]))
