"""Tests for src/matched_far.py and src/calibration_split.py."""
import numpy as np
import pytest

from src import evaluation
from src.detectors.cusum import CUSUMDetector
from src.drift_sim.real_data_injection import (
    real_masked_channel_stream,
    real_radiation_damage_stream,
)
from src.matched_far import (
    calibrate_param, cusum_runmax, first_passage, gaussian_far, inject_masked,
    inject_radiation, outcome_counts, take_windows, wilson, zscore,
)


def _detector_first_alarm(series, ref, k, h, mode):
    if mode == "two":
        det = CUSUMDetector(ref, k=k, h=h, two_sided=True)
    elif mode == "up":
        det = CUSUMDetector(ref, k=k, h=h, two_sided=False)
    else:  # "down": the repo's one-sided detector on the negated series
        det = CUSUMDetector(ref, k=k, h=h, two_sided=False,
                            transform=lambda x: -np.asarray(x))
    r = evaluation.run_detector_on_residuals(det, series)
    return r.detection_index if r.detected else len(series)


@pytest.mark.parametrize("mode", ["two", "up", "down"])
@pytest.mark.parametrize("k", [0.1, 0.5])
def test_first_passage_matches_cusum_detector(mode, k):
    rng = np.random.default_rng(0)
    ref = rng.normal(0.2, 1.3, 4000)
    n, T = 12, 600
    series = rng.normal(0.2, 1.3, (n, T))
    series[::2, 300:] += rng.choice([-1.0, 1.0]) * 0.8  # shift half the windows
    h_grid = np.array([2.0, 3.5, 5.0, 8.0, 12.0])
    fp = first_passage(cusum_runmax(zscore(series, ref), k, mode), h_grid)
    for i in range(n):
        for j, h in enumerate(h_grid):
            assert fp[i, j] == _detector_first_alarm(series[i], ref, k, h, mode)


def test_first_passage_is_monotone_in_threshold():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, (20, 400))
    h_grid = np.arange(1.0, 15.0, 0.5)
    fp = first_passage(cusum_runmax(z, 0.5, "two"), h_grid)
    assert np.all(np.diff(fp, axis=1) >= 0)  # higher threshold, never earlier


def test_calibrate_param_picks_most_sensitive_meeting_target():
    T = 100
    # 10 windows; param 0 alarms in 5, param 1 in 2, param 2 in 0.
    fp = np.full((10, 3), T)
    fp[:5, 0] = 10
    fp[:2, 1] = 10
    j, far = calibrate_param(fp, T, 0.2)
    assert j == 1 and np.allclose(far, [0.5, 0.2, 0.0])
    j, _ = calibrate_param(fp, T, 0.0)
    assert j == 2
    j, _ = calibrate_param(np.zeros((4, 2), dtype=int), T, 0.1)  # always alarms
    assert j is None


def test_outcome_counts():
    fp = np.array([[5], [100], [7], [100]])
    k, n, delays = outcome_counts(fp, 0, 100)
    assert (k, n, sorted(delays)) == (2, 4, [5, 7])


def test_wilson_basic():
    lo, hi = wilson(6, 10)
    assert 0.30 < lo < 0.32 and 0.83 < hi < 0.84
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan for n=0


def test_gaussian_far_matches_known_values():
    # h=8, k=0.5, two-sided, 5000 events: about 0.41 (Monte Carlo elsewhere in the repo notes)
    assert 0.37 < gaussian_far(8.0, 0.5, 5000, trials=4000) < 0.45


def test_inject_masked_matches_real_stream():
    rng = np.random.default_rng(2)
    T = 50
    feats = rng.normal(100, 20, (T, 12))
    pu, nj = np.full(T, 30.0), np.full(T, 3.0)
    drop = 0.4
    streamed = np.array([e["features"] for e in real_masked_channel_stream(
        feats, pu, nj, start=0, n_events=T, changepoint_event=0, drop_fraction=drop)])
    assert np.allclose(inject_masked(feats[None], drop)[0], streamed)


def test_inject_radiation_matches_real_stream():
    rng = np.random.default_rng(3)
    T = 60
    feats = rng.normal(100, 20, (T, 12))
    pu, nj = np.full(T, 30.0), np.full(T, 3.0)
    streamed = np.array([e["features"] for e in real_radiation_damage_stream(
        feats, pu, nj, start=0, n_events=T, tau_damage=40.0)])
    assert np.allclose(inject_radiation(feats[None], 40.0)[0], streamed)


def test_injection_does_not_mutate_input_and_only_touches_affected_columns():
    feats = np.ones((2, 10, 12))
    before = feats.copy()
    out = inject_masked(feats, 0.5)
    assert np.array_equal(feats, before)
    changed = np.where(~np.isclose(out[0, 0], 1.0))[0].tolist()
    assert sorted(changed) == [0, 3, 7, 9]


def test_take_windows_shapes_and_content():
    arr = np.arange(100)
    w = take_windows(arr, [10, 50], 5)
    assert w.shape == (2, 5) and w[1].tolist() == [50, 51, 52, 53, 54]


# ----- split builder (needs torch) -------------------------------------------
def test_split_builder_blocks_and_modes():
    pytest.importorskip("torch")
    from src.calibration_split import build_split_calibration, reserved_events
    from src.matched_far import SyntheticPool

    feats, pu, nj = SyntheticPool(region_events=6000).region(0, 1)
    kw = dict(n_vae=1500, n_reg=800, n_ref=800, vae_epochs=2, seed=0)
    ins = build_split_calibration(feats, pu, nj, mode="insample", **kw)
    three = build_split_calibration(feats, pu, nj, mode="three_way", **kw)

    assert ins.blocks["vae"] == ins.blocks["regression"] == ins.blocks["reference"]
    a, b, c = three.blocks["vae"], three.blocks["regression"], three.blocks["reference"]
    assert a[1] == b[0] and b[1] == c[0]  # contiguous, disjoint, in time order
    assert ins.reserved_end == three.reserved_end == reserved_events(1500, 800, 800)
    # In-sample reference is standardized by construction; three-way need not be.
    assert abs(ins.ref_residuals.mean()) < 1e-6 and abs(ins.ref_residuals.std() - 1) < 1e-6
    assert np.all(np.isfinite(three.ref_residuals))
