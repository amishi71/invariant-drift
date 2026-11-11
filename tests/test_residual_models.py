"""Tests for src/residual_models.py, the span mode, and the covariate diagnostic's metrics."""
import sys
import types

import numpy as np
import pytest

from src.residual import CalibrationModel
from src.residual_models import DESIGNS, FlexCalibrationModel, fit_calibration


def _data(n=20000, seed=0, pu_lo=10, pu_hi=60):
    rng = np.random.default_rng(seed)
    pu = rng.integers(pu_lo, pu_hi, n).astype(float)
    nj = rng.integers(2, 9, n).astype(float)
    f = 0.5 * np.sin(pu / 12.0) + 0.1 * nj          # non-quadratic pileup response
    y = f + 0.3 * rng.standard_normal(n)
    return np.exp(y), pu, nj


def test_quad_design_is_the_original_model_exactly():
    s, pu, nj = _data()
    lum = np.ones(len(s))
    old = CalibrationModel.fit(s, pu, nj, lum)
    new = fit_calibration("quad", s, pu, nj, lum)
    assert isinstance(new, CalibrationModel)
    assert np.array_equal(old.theta, new.theta) and old.sigma_resid == new.sigma_resid
    flex = FlexCalibrationModel.fit(s, pu, nj, lum, design="quad")
    assert np.allclose(flex.residual(s, pu, nj, lum), old.residual(s, pu, nj, lum))


@pytest.mark.parametrize("design", ["pu_spline", "pu_spline_nj", "pu_spline_nj_x"])
def test_spline_beats_quadratic_on_nonquadratic_response(design):
    s, pu, nj = _data()
    lum = np.ones(len(s))
    quad = fit_calibration("quad", s, pu, nj, lum)
    spl = fit_calibration(design, s, pu, nj, lum)
    assert spl.sigma_resid < quad.sigma_resid
    r = spl.residual(s, pu, nj, lum)
    assert abs(r.mean()) < 0.02 and abs(r.std() - 1) < 0.05


def test_clamp_extrapolation_is_bounded_and_linear_is_not():
    s, pu, nj = _data(pu_lo=10, pu_hi=40)
    lum = np.ones(len(s))
    far = np.array([500.0])
    m = fit_calibration("pu_spline", s, pu, nj, lum)
    edge = np.array([float(np.percentile(pu, 99.5))])
    assert abs(m.predict(far, np.array([5.0]), np.ones(1))[0]
               - m.predict(edge, np.array([5.0]), np.ones(1))[0]) < 1e-9
    ml = FlexCalibrationModel.fit(s, pu, nj, lum, design="pu_spline", extrapolate="linear")
    assert abs(ml.predict(far, np.array([5.0]), np.ones(1))[0]) > abs(m.predict(far, np.array([5.0]), np.ones(1))[0]) - 1e-12


def test_unseen_njet_levels_do_not_crash():
    s, pu, nj = _data()
    m = fit_calibration("pu_spline_nj_x", s, pu, nj, np.ones(len(s)))
    out = m.residual(s[:5], pu[:5], np.array([0, 1, 50, 99, 3.0]), np.ones(5))
    assert np.all(np.isfinite(out))


def test_bad_design_and_too_few_events():
    s, pu, nj = _data(n=100)
    with pytest.raises(ValueError):
        fit_calibration("nope", s, pu, nj, np.ones(100))
    with pytest.raises(ValueError):
        fit_calibration("pu_spline_nj_x", s[:10], pu[:10], nj[:10], np.ones(10))


def _stub_vae(monkeypatch):
    mod = types.ModuleType("src.proxy_vae")
    mod.train_proxy_vae = lambda x, epochs=1, seed=0, verbose=False: (None, None, np.array([0.0]))
    mod.anomaly_score = lambda m, sc, x: np.exp(0.01 * x[:, 0] + 0.001 * x[:, 1] ** 2)
    monkeypatch.setitem(sys.modules, "src.proxy_vae", mod)
    sys.modules.pop("src.calibration_split", None)


def test_span_mode_and_default_unchanged(monkeypatch):
    _stub_vae(monkeypatch)
    from src.calibration_split import build_split_calibration
    rng = np.random.default_rng(0)
    n = 6000
    F = rng.normal(size=(n, 12)); pu = rng.integers(10, 50, n).astype(float); nj = rng.integers(2, 8, n).astype(float)
    kw = dict(n_vae=1000, n_reg=800, n_ref=800)
    base = build_split_calibration(F, pu, nj, mode="three_way", **kw)
    assert isinstance(base.calib, CalibrationModel) and base.info["design"] == "quad"
    idx = rng.permutation(np.arange(3000, n))
    sp = build_split_calibration(F, pu, nj, mode="span", design="pu_spline",
                                 span_idx=(idx[:800], idx[800:1600]), **kw)
    assert sp.reserved_end == base.reserved_end and isinstance(sp.calib, FlexCalibrationModel)
    with pytest.raises(ValueError):
        build_split_calibration(F, pu, nj, mode="span", **kw)
    with pytest.raises(ValueError):
        build_split_calibration(F, pu, nj, mode="span", span_idx=(idx[:800], idx[:800]), **kw)


def test_window_metrics_iid_and_slow_drift():
    import importlib.util, os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "covariate_model_diagnostic.py")
    spec = importlib.util.spec_from_file_location("cov_diag", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    window_metrics = mod.window_metrics
    rng = np.random.default_rng(1)
    n, W = 150, 2000
    PU = np.tile(rng.integers(20, 40, n)[:, None], (1, W)).astype(float)
    NJ = np.full((n, W), 4.0)
    iid = rng.standard_normal((n, W))
    m = window_metrics(iid, PU, NJ, rng.standard_normal(20000), PU[:, 0])
    assert 0.8 < m["ratio"] < 1.25 and abs(m["corr_pu"]) < 0.25
    drift = iid + 0.15 * ((PU - 30) / 6)            # pileup-tied window offsets
    m2 = window_metrics(drift, PU, NJ, rng.standard_normal(20000), PU[:, 0])
    assert m2["ratio"] > 2 and m2["corr_pu"] > 0.5
