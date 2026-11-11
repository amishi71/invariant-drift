"""--n-tune: tuning windows must not change cal/val/eval results and must feed the k-selection table."""
import importlib.util
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_vae(monkeypatch):
    mod = types.ModuleType("src.proxy_vae")
    mod.DEFAULT_FEATURE_NAMES = ["jet1_pt", "jet1_eta", "jet1_phi", "jet2_pt", "jet2_eta", "jet2_phi",
                                 "n_jet", "met_pt", "met_phi", "ht", "n_muon", "n_electron"]
    mod.train_proxy_vae = lambda x, epochs=1, seed=0, verbose=False: (None, None, np.array([0.1]))
    mod.anomaly_score = lambda m, sc, x: np.exp(0.005 * (x[:, 0] + x[:, 3] + x[:, 9] + x[:, 7]) + 0.01 * x[:, 6])
    monkeypatch.setitem(sys.modules, "src.proxy_vae", mod)
    sys.modules.pop("src.calibration_split", None)
    sys.modules.pop("src.stream_loader", None)


def _load():
    spec = importlib.util.spec_from_file_location("mfh", os.path.join(ROOT, "scripts", "matched_far_heldout.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(m, tmp_path, n_tune, monkeypatch, capsys):
    out = str(tmp_path / f"nt{n_tune}")
    argv = ["x", "--source", "synthetic", "--synthetic-region-events", "80000", "--seeds", "0",
            "--n-regions", "1", "--window", "500", "--n-vae", "500", "--n-reg", "500", "--n-ref", "500",
            "--n-cal", "40", "--n-val", "40", "--n-eval", "20", "--drops", "0.4", "--taus", "400",
            "--modes", "three_way", "--print-target", "0.1", "--n-tune", str(n_tune), "--out-dir", out]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    text = capsys.readouterr().out
    with open(os.path.join(out, "synthetic_seed0.json")) as f:
        return json.load(f), text


def test_tuning_windows_leave_eval_unchanged_and_feed_selection(monkeypatch, tmp_path, capsys):
    _stub_vae(monkeypatch)
    m = _load()
    d0, t0 = _run(m, tmp_path, 0, monkeypatch, capsys)
    d1, t1 = _run(m, tmp_path, 20, monkeypatch, capsys)
    key = lambda d: [(r["family"], r["k"], r["target"], r["param"], r["val_alarms"],
                      json.dumps(r["scenarios"], sort_keys=True)) for r in d["records"]]
    assert key(d0) == key(d1)
    assert "SELECTED ON SEPARATE TUNING WINDOWS" not in t0
    assert "SELECTED ON SEPARATE TUNING WINDOWS" in t1
    rec = next(r for r in d1["records"] if r["param"] is not None and r["family"].startswith("CUSUM"))
    assert set(rec["tune"]) == set(d1["scenarios"]) and all(v["n"] == 20 for v in rec["tune"].values())
