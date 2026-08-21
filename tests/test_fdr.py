import numpy as np
import pytest

from src.conformal.fdr import LORD, SAFFRON, windowed_batch_pvalues


class TestLORDSAFFRON:
    def test_lord_few_rejections_under_global_null(self):
        """Under the global null (all p-values genuinely Uniform(0,1)),
        LORD should reject only rarely -- this is the core validity
        property (controls mFDR at the nominal alpha)."""
        rng = np.random.RandomState(0)
        p_values = rng.uniform(0, 1, 500)
        lord = LORD(alpha=0.1, max_tests=500)
        for p in p_values:
            lord.test(float(p))
        # Rejections under the null should be a small minority of tests.
        assert len(lord.rejections) < 0.2 * 500

    def test_saffron_few_rejections_under_global_null(self):
        rng = np.random.RandomState(1)
        p_values = rng.uniform(0, 1, 500)
        saffron = SAFFRON(alpha=0.1, max_tests=500)
        for p in p_values:
            saffron.test(float(p))
        assert len(saffron.rejections) < 0.2 * 500

    def test_lord_rejects_strong_signal(self):
        """Small p-values (strong alternative) should accumulate wealth
        and eventually get rejected -- sanity check that the procedure has
        power, not just that it's conservative."""
        p_values = np.concatenate([
            np.random.RandomState(2).uniform(0, 1, 50),
            np.full(50, 1e-6),  # obviously non-null block
        ])
        lord = LORD(alpha=0.1, max_tests=len(p_values))
        for p in p_values:
            lord.test(float(p))
        assert len(lord.rejections) > 0
        assert all(r >= 50 for r in lord.rejections[:1])  # first rejection in the signal block

    def test_saffron_discards_non_candidates(self):
        saffron = SAFFRON(alpha=0.1, lambda_threshold=0.5, max_tests=100)
        record = saffron.test(0.9)  # p > lambda -> not a candidate
        assert record["candidate"] is False
        assert record["alpha_i"] == 0.0
        assert record["rejected"] is False

    def test_wealth_never_negative(self):
        rng = np.random.RandomState(3)
        p_values = rng.uniform(0, 1, 1000)
        lord = LORD(alpha=0.1, max_tests=1000)
        for p in p_values:
            lord.test(float(p))
            assert lord.wealth >= 0.0

    def test_rejections_are_zero_indexed(self):
        """Regression test for a real off-by-one bug found during
        development: `rejections` was storing self.n_tests (1-indexed,
        incremented before use), while evaluation.py's evaluate_online_fdr
        uses entries in `rejections` directly as 0-indexed positions into
        a same-length array of per-window ground-truth labels -- a
        rejection on the very LAST test produced an IndexError (test_index
        == array length, one past the last valid index). Force a
        rejection on the very first test and confirm it's recorded as
        index 0, not 1.
        """
        lord = LORD(alpha=0.5, w0_fraction=1.0, max_tests=10)
        lord.test(1e-9)  # trivially small p-value -- should reject test 1
        assert lord.rejections == [0]

    def test_rejection_on_last_test_is_in_bounds(self):
        """Direct regression test for the exact IndexError scenario: a
        rejection on the final test of a run must be a valid index into
        an array of length == max_tests."""
        n = 20
        p_values = np.full(n, 1e-9)  # every test should reject
        lord = LORD(alpha=0.5, w0_fraction=1.0, max_tests=n)
        for p in p_values:
            lord.test(float(p))
        dummy_labels = np.zeros(n, dtype=int)
        for r in lord.rejections:
            assert 0 <= r < n
            dummy_labels[r]  # must not raise IndexError


class TestWindowedBatchPvalues:
    def test_uniform_under_null_rate(self):
        rng = np.random.RandomState(0)
        nominal_rate = 0.05
        flags = rng.uniform(0, 1, 5000) < nominal_rate
        pvals = windowed_batch_pvalues(flags, window_size=50, nominal_rate=nominal_rate)
        # Under the true null, p-values should be roughly uniform -- mean
        # around 0.5, not concentrated near 0.
        assert 0.3 < pvals.mean() < 0.7

    def test_elevated_rate_gives_small_pvalues(self):
        rng = np.random.RandomState(1)
        nominal_rate = 0.02
        flags = rng.uniform(0, 1, 2000) < 0.3  # much higher than nominal
        pvals = windowed_batch_pvalues(flags, window_size=50, nominal_rate=nominal_rate)
        assert pvals.mean() < 0.05

    def test_output_length(self):
        flags = np.zeros(505, dtype=bool)
        pvals = windowed_batch_pvalues(flags, window_size=50, nominal_rate=0.05)
        assert len(pvals) == 10  # floor(505/50)
