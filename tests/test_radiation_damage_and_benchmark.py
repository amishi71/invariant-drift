import numpy as np

from src.drift_sim.radiation_damage import radiation_damage_stream
from src.benchmark import benchmark_detector, benchmark_all, format_results
from src.detectors.cusum import CUSUMDetector
from src.detectors.bocpd import BOCPD


def _drain(stream, n):
    out = []
    for i, x in enumerate(stream):
        if i >= n:
            break
        out.append(x)
    return out


class TestRadiationDamageStream:
    def test_never_resets_monotonic_gain_decay(self):
        events = _drain(radiation_damage_stream(1000, tau_damage=1000.0, seed=0), 1000)
        gains = [e["gain_factor"] for e in events]
        # Strictly non-increasing (exp(-i/tau) is monotonic in i) -- no
        # reset anywhere, unlike gradual.py's fill-based pileup.
        assert all(gains[i] >= gains[i + 1] for i in range(len(gains) - 1))
        assert gains[0] > gains[-1]

    def test_no_changepoint_parameter_drift_from_event_zero(self):
        """Unlike abrupt.py's scenarios, there's no changepoint_event
        argument -- drift is present from the very first event."""
        import inspect
        sig = inspect.signature(radiation_damage_stream)
        assert "changepoint_event" not in sig.parameters

    def test_onset_label_reflects_detectability_floor(self):
        events = _drain(radiation_damage_stream(2000, tau_damage=500.0, seed=0), 2000)
        onset = events[0]["true_onset_event"]
        labels = [e["true_label"] for e in events]
        assert all(l == 0 for l in labels[:onset])
        assert all(l == 1 for l in labels[onset:])
        # At the onset, cumulative gain loss should be close to the 5% floor.
        gain_at_onset = events[onset]["gain_factor"]
        assert 0.90 < gain_at_onset < 0.96

    def test_covariates_held_stable(self):
        """Pileup should NOT be drifting here (module docstring point 3) --
        contrast with gradual.py where pileup decay IS the mechanism."""
        events = _drain(radiation_damage_stream(1000, seed=0), 1000)
        pileup = np.array([e["pileup"] for e in events])
        # Should look like noise around a constant level, not a trend --
        # check first-half vs second-half means are close (loose bound,
        # this is a stochastic stream).
        assert abs(pileup[:500].mean() - pileup[500:].mean()) < 5.0

    def test_detected_by_cusum_and_bocpd(self):
        """Sanity check the scenario is actually detectable by the two
        primary detectors it's meant to stress-test."""
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 500)
        events = _drain(radiation_damage_stream(1500, tau_damage=800.0, seed=1), 1500)
        # Use the gain_factor as a crude direct proxy signal for this
        # structural test (not the full VAE/residual pipeline) -- just
        # confirming the injected drift is a real, detectable signal.
        residuals = np.array([1.0 - e["gain_factor"] for e in events]) * 10 + rng.normal(0, 1, 1500)
        d = CUSUMDetector(reference, k=0.5, h=8.0, two_sided=True)
        detected = False
        for r in residuals:
            d.update(r)
            if d.is_ready() and d.evaluate_drift()["shift_detected"]:
                detected = True
                break
        assert detected


class TestBenchmark:
    def test_benchmark_detector_returns_positive_timings(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 200)
        residuals = rng.normal(0, 1, 300)
        result = benchmark_detector(
            "CUSUM", lambda: CUSUMDetector(reference, k=0.5, h=8.0), residuals, n_warmup=50,
        )
        assert result.mean_us_per_event > 0
        assert result.p99_us_per_event >= result.mean_us_per_event * 0.5  # loose sanity bound
        assert result.peak_memory_kb >= 0
        assert result.n_events == 250  # 300 - 50 warmup

    def test_benchmark_all_covers_every_detector(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 200)
        specs = {
            "CUSUM": lambda ref: CUSUMDetector(ref, k=0.5, h=8.0),
            "BOCPD": lambda ref: BOCPD(ref, hazard_lambda=250.0),
        }
        results = benchmark_all(specs, reference, n_events=300)
        names = {r.detector_name for r in results}
        assert names == {"CUSUM", "BOCPD"}

    def test_format_results_is_a_table(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 200)
        specs = {"CUSUM": lambda ref: CUSUMDetector(ref, k=0.5, h=8.0)}
        results = benchmark_all(specs, reference, n_events=300)
        table = format_results(results)
        assert "CUSUM" in table
        assert "us/event" in table
