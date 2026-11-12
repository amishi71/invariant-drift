"""Injection sets, forward/cross-run protocols and direction check of matched_far_heldout.py."""
import importlib.util
import json
import os
import sys
import types
from argparse import Namespace

import numpy as np
import pytest

from src.injection_sets import (INJECTION_SETS, detect_layout, inject_masked_cols, inject_radiation_cols,
                                resolve_injection_cols)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _stub_vae_everywhere(monkeypatch):
    """The script imports the VAE module at load time; these tests never need the real network."""
    _stub_vae(monkeypatch)


def _load_script():
    spec = importlib.util.spec_from_file_location("mfh", os.path.join(ROOT, "scripts", "matched_far_heldout.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cache(path, n, seed, l1=True):
    r = np.random.default_rng(seed)
    F = (np.rint(np.abs(r.normal(20, 10, (n, 12))) * 2) / 2).astype(np.float32)
    F[:, 0] = r.integers(0, 8, n)
    arrs = dict(features=F, pileup=np.full(n, 64.0, np.float32), n_jet=F[:, 0].copy())
    if l1:
        arrs.update(lumi=np.sort(r.integers(0, 50, n)).astype(np.uint32))
    np.savez(path, **arrs)


# ---------------------------------------------------------------- injection sets
def test_layout_detection_and_auto_resolution(tmp_path):
    a, b = str(tmp_path / "l1.npz"), str(tmp_path / "jet.npz")
    _cache(a, 100, 0, l1=True)
    _cache(b, 100, 0, l1=False)
    assert detect_layout(a) == "l1" and detect_layout(b) == "jetht"
    assert resolve_injection_cols("auto", a) == INJECTION_SETS["l1_legacy"]
    assert resolve_injection_cols("auto", b) == INJECTION_SETS["jetht"]
    assert resolve_injection_cols("auto", None) == INJECTION_SETS["jetht"]
    assert resolve_injection_cols("l1_calo", b) == INJECTION_SETS["l1_calo"]
    with pytest.raises(ValueError):
        resolve_injection_cols("nope")


def test_two_pools_do_not_clash_because_nothing_is_global(tmp_path):
    a, b = str(tmp_path / "l1.npz"), str(tmp_path / "jet.npz")
    _cache(a, 100, 0, l1=True)
    _cache(b, 100, 0, l1=False)
    first = resolve_injection_cols("auto", b)
    resolve_injection_cols("auto", a)            # building an L1 selection must not change the JetHT one
    assert resolve_injection_cols("auto", b) == first


def test_l1_calo_scales_energies_together_and_leaves_counts_and_muons():
    F = np.full((3, 20, 12), 10.0)
    out = inject_masked_cols(F, 0.4, INJECTION_SETS["l1_calo"])
    untouched = [0, 1, 2, 3, 6]
    assert np.allclose(out[..., untouched], 10.0)
    assert np.allclose(out[..., list(INJECTION_SETS["l1_calo"])], 6.0)
    legacy = inject_masked_cols(F, 0.4, INJECTION_SETS["l1_legacy"])
    assert np.allclose(legacy[..., [5, 7, 10, 11]], 10.0)          # the legacy set leaves these unscaled


def test_requantize_lands_on_the_lattice_and_only_touches_scaled_columns():
    rng = np.random.default_rng(0)
    F = np.rint(rng.random((4, 50, 12)) * 100) / 2                  # 0.5 GeV lattice everywhere
    cols = INJECTION_SETS["l1_calo"]
    raw = inject_masked_cols(F, 0.37, cols)
    q = inject_masked_cols(F, 0.37, cols, requantize=0.5)
    assert not np.allclose(raw[..., list(cols)] * 2, np.rint(raw[..., list(cols)] * 2))   # off-lattice without it
    assert np.allclose(q[..., list(cols)] * 2, np.rint(q[..., list(cols)] * 2))            # on-lattice with it
    rad = inject_radiation_cols(F, 300.0, cols, requantize=0.5)
    assert np.allclose(rad[..., list(cols)] * 2, np.rint(rad[..., list(cols)] * 2))
    assert np.array_equal(q[..., [0, 6]], F[..., [0, 6]])


# ------------------------------------------------------------------ window placement
def _args(**kw):
    base = dict(n_cal=40, n_val=40, n_eval=20, n_tune=20, time_split="none", forward_boundary=0.5)
    base.update(kw)
    return Namespace(**base)


def test_default_placement_is_exactly_the_original_formula():
    m = _load_script()
    n_slots = 300
    cal, val, ev, tune, free = m.place_windows(_args(), np.random.default_rng(1007), n_slots)
    perm = np.random.default_rng(1007).permutation(n_slots)
    need = 100
    assert np.array_equal(cal, perm[:40]) and np.array_equal(val, perm[40:80]) and np.array_equal(ev, perm[80:100])
    assert np.array_equal(tune, perm[need:need + 20]) and np.array_equal(free, perm[need + 20:])


def test_forward_placement_calibrates_on_the_past_and_monitors_the_future():
    m = _load_script()
    n_slots = 200
    cal, val, ev, tune, free = m.place_windows(_args(time_split="forward"), np.random.default_rng(3), n_slots)
    boundary = 100
    assert cal.max() < boundary
    assert min(val.min(), ev.min(), tune.min()) >= boundary
    allw = np.concatenate([cal, val, ev, tune])
    assert len(np.unique(allw)) == len(allw) == 120
    assert len(np.intersect1d(allw, free)) == 0 and len(free) == n_slots - 120


def test_forward_placement_refuses_when_the_halves_are_too_small():
    m = _load_script()
    with pytest.raises(SystemExit, match="forward"):
        m.place_windows(_args(time_split="forward", n_cal=150), np.random.default_rng(0), 200)


# ---------------------------------------------------------------------- end to end
def _stub_vae(monkeypatch):
    mod = types.ModuleType("src.proxy_vae")
    mod.DEFAULT_FEATURE_NAMES = [f"f{i}" for i in range(12)]
    mod.train_proxy_vae = lambda x, epochs=1, seed=0, verbose=False: (None, None, np.array([0.1]))
    mod.anomaly_score = lambda m, sc, x: np.exp(0.01 * (x[:, 4] + x[:, 8] + x[:, 9]) + 0.02 * x[:, 0])
    monkeypatch.setitem(sys.modules, "src.proxy_vae", mod)
    for name in ("src.calibration_split", "src.stream_loader"):
        sys.modules.pop(name, None)


def _run(m, monkeypatch, capsys, out, extra):
    argv = ["x", "--source", "real", "--seeds", "0", "--window", "500", "--n-vae", "500", "--n-reg", "500",
            "--n-ref", "500", "--n-cal", "40", "--n-val", "40", "--n-eval", "20", "--n-tune", "20",
            "--drops", "0.4", "--taus", "400", "--modes", "three_way", "--print-target", "0.1",
            "--out-dir", out] + extra
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    return capsys.readouterr().out


def test_cross_run_forward_tag_and_direction_check_end_to_end(tmp_path, monkeypatch, capsys):
    _stub_vae(monkeypatch)
    m = _load_script()
    a, b = str(tmp_path / "runA.npz"), str(tmp_path / "runB.npz")
    _cache(a, 60000, 1)
    _cache(b, 80000, 2)
    out = str(tmp_path / "res")
    text = _run(m, monkeypatch, capsys, out,
                ["--cache-path", b, "--calib-cache", a, "--calib-n-regions", "1", "--n-regions", "1",
                 "--time-split", "forward", "--inject-set", "l1_calo", "--requantize", "0.5", "--tag", "x_cross"])
    d = json.load(open(os.path.join(out, "x_cross_seed0.json")))
    assert d["protocol"]["calibration_from"] == a and d["protocol"]["time_split"] == "forward"
    assert d["protocol"]["inject_cols"] == list(INJECTION_SETS["l1_calo"]) and d["protocol"]["requantize"] == 0.5
    shift = d["mode_info"]["three_way"]["inject_shift_z"]
    assert set(shift) == set(d["scenarios"]) and all(np.isfinite(v) for v in shift.values())
    assert "DIRECTION CHECK" in text and "SELECTED ON SEPARATE TUNING WINDOWS" in text
    assert "eval min" in text and "seed range" in text
    assert "oracle direction" not in text


def test_default_run_has_no_protocol_surprises_and_distinct_tag(tmp_path, monkeypatch, capsys):
    _stub_vae(monkeypatch)
    m = _load_script()
    b = str(tmp_path / "runB.npz")
    _cache(b, 80000, 2)
    out = str(tmp_path / "res")
    _run(m, monkeypatch, capsys, out, ["--cache-path", b, "--n-regions", "1", "--tag", "mine"])
    assert os.path.exists(os.path.join(out, "mine_seed0.json")) and not os.path.exists(os.path.join(out, "real_seed0.json"))
    d = json.load(open(os.path.join(out, "mine_seed0.json")))
    assert d["protocol"]["time_split"] == "none" and d["protocol"]["calibration_from"] == "same region"
    assert d["protocol"]["inject_cols"] == list(INJECTION_SETS["l1_legacy"])      # auto on an L1 cache
