import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seed(path, adwin_lat):
    row = lambda lat, cens: {"chosen_param": 1, "achieved_far": 0.05, "miss_rate": 0.02, "latency_mean": lat,
                             "latency_std": 1.0, "latency_censored_mean": cens, "val_far": 0.04,
                             "val_alarms": 8, "val_n": 200, "val_far_ci95": [0.02, 0.08]}
    ev = lambda lat: {"miss_rate": 0.02, "latency_mean": lat, "latency_std": 1.0, "latency_censored_mean": lat + 5}
    sweep = {k: {"h": 9.0, "achieved_far": 0.05, "val_far": 0.04, "val_alarms": 8, "val_n": 200,
                 "tune": ev(300 + 10 * i), "eval": ev(310 + 10 * i)} for i, k in enumerate(["0.25", "0.5", "1"])}
    json.dump({"results": {"CUSUM": row(318, 320), "ADWIN": row(adwin_lat, adwin_lat + 5)},
               "cusum_k": {"sweep": sweep, "selected_k": 0.25}, "config": {}}, open(path, "w"))


def test_aggregator_runs_and_reports_paired_difference(tmp_path):
    for i, lat in enumerate([340, 350]):
        _seed(tmp_path / f"seed{i}.json", lat)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "aggregate_baseline_oos.py"),
                          str(tmp_path / "seed*.json")], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "FAR (fresh, pooled)" in out.stdout and "k selected per seed: 0.25, 0.25" in out.stdout
    assert "ADWIN slower in 2/2 seeds" in out.stdout
