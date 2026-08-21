import numpy as np

from src.drift_sim.gradual import nominal_gradual_stream, misspecified_gradual_stream
from src.drift_sim.abrupt import masked_channel_stream, multiplicity_step_stream


def _drain(stream, n):
    return list(itertools_islice(stream, n))


def itertools_islice(iterable, n):
    out = []
    for i, x in enumerate(iterable):
        if i >= n:
            break
        out.append(x)
    return out


class TestGradual:
    def test_nominal_all_labels_zero(self):
        events = _drain(nominal_gradual_stream(200, seed=0), 200)
        assert all(e["true_label"] == 0 for e in events)

    def test_pileup_decays_over_stream(self):
        events = _drain(nominal_gradual_stream(500, pileup0=45.0, tau=1000, power=0.7, seed=0), 500)
        pileup = [e["pileup"] for e in events]
        assert pileup[0] > pileup[-1]  # decays across the fill
        assert pileup[-1] >= 5.0  # respects the floor clip

    def test_misspecified_onset_label_flip(self):
        n = 400
        events = _drain(misspecified_gradual_stream(n, bias_onset_frac=0.5, seed=0), n)
        labels = [e["true_label"] for e in events]
        onset = events[0]["true_onset_event"]
        assert onset == n // 2
        assert all(l == 0 for l in labels[:onset])
        assert all(l == 1 for l in labels[onset:])

    def test_misspecified_bias_grows_after_onset(self):
        """Compares WINDOW-AVERAGED jet1_pt (not single events): a single
        event's jet1_pt is a noisy gamma draw with std comparable to or
        larger than the injected bias itself at these settings, so a
        single-event comparison is dominated by sampling noise regardless
        of whether the bias mechanism is correct (caught during
        development as a test-design bug, not a drift_sim bug -- see
        README's "Known issues" section). Averaging over a window isolates
        the actual injected trend from per-event noise.
        """
        n = 2000
        events = _drain(misspecified_gradual_stream(
            n, bias_onset_frac=0.5, bias_rate=0.05, bias_features=(0,), seed=1,
        ), n)
        onset = events[0]["true_onset_event"]
        window = 100
        jet1_pt_early_post = np.mean(
            [e["features"][0] for e in events[onset:onset + window]]
        )
        jet1_pt_late_post = np.mean(
            [e["features"][0] for e in events[n - window:n]]
        )
        assert jet1_pt_late_post > jet1_pt_early_post  # growing bias, on average


class TestAbrupt:
    def test_masked_channel_step_at_changepoint(self):
        n = 400
        cp = 200
        events = _drain(masked_channel_stream(n, changepoint_event=cp, drop_fraction=0.3, seed=0), n)
        labels = [e["true_label"] for e in events]
        assert all(l == 0 for l in labels[:cp])
        assert all(l == 1 for l in labels[cp:])

    def test_masked_channel_actually_drops_energy(self):
        n = 400
        cp = 200
        events_drift = _drain(
            masked_channel_stream(n, changepoint_event=cp, drop_fraction=0.5,
                                   affected_features=(9,), seed=0), n,
        )
        ht_pre = np.mean([e["features"][9] for e in events_drift[:cp]])
        ht_post = np.mean([e["features"][9] for e in events_drift[cp:]])
        assert ht_post < ht_pre  # HT (index 9) dropped after the changepoint

    def test_multiplicity_step_changes_n_jet_mean(self):
        n = 600
        cp = 300
        events = _drain(multiplicity_step_stream(n, changepoint_event=cp, n_jet_delta=-2.0, seed=0), n)
        n_jet_pre = np.mean([e["n_jet"] for e in events[:cp]])
        n_jet_post = np.mean([e["n_jet"] for e in events[cp:]])
        assert n_jet_post < n_jet_pre

    def test_labels_and_onset_consistent(self):
        n = 300
        cp = 150
        events = _drain(multiplicity_step_stream(n, changepoint_event=cp, seed=0), n)
        for i, e in enumerate(events):
            assert e["true_label"] == (0 if i < cp else 1)
            assert e["true_onset_event"] == cp
