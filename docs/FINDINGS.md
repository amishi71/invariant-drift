# Findings and design decisions

Bugs and empirical findings caught by actually running `main.py` end-to-end
and noticing numbers that didn't match theory — e.g. CUSUM's ARL0 for
k=0.5, h=10 should be ~22,000, not ~20 — then tracing each one down.

## Bugs

**Score log-transform (`residual.py`).** The raw VAE reconstruction error
is right-skewed (skew ~1.35 empirically), being a sum-of-squares-type
quantity. Regressing the raw score and standardizing the residual doesn't
fix this — the conditional distribution stays skewed after de-meaning,
which quietly breaks CUSUM/Page-Hinkley's Gaussian-ARL assumptions
(false-alarm rate hit ~100% within a few hundred events instead of the
theoretical ARL0 in the thousands). Fixed by fitting the calibration
regression in log-score space by default (`score_transform="log"`), the
same approach Project A used for its own skewed observable (dijet mass).

**Page-Hinkley two-sided bug (`page_hinkley.py`).** The down-side branch
tracked a running maximum of the negated cumulative sum instead of a
running minimum. Since the cumulative sum has a `-delta` drift term
regardless of any real shift, `max - current` grows roughly linearly with
event count on its own — guaranteed false alarms within 20-45 events on
pure noise (reproduced in 10/10 trials before the fix). Fixed to mirror
the up-side construction; verified with a direct step-by-step equivalence
to CUSUM's S+ recursion (`tests/test_cusum_page_hinkley.py`).

**BOCPD detection criterion (`bocpd.py`).** `P(r_t = 0)` turns out to be
the wrong statistic to threshold: once the run-length posterior
concentrates on one dominant hypothesis (which happens quickly), the
predictive-likelihood term cancels out of the R(0)/R(r*+1) ratio, so
`P(r_t=0)` converges toward the bare hazard rate regardless of how
surprising the new data point is. Verified directly — an injected 4-sigma
shift left `P(r=0)` pinned near the hazard rate while `map_run_length`
correctly collapsed to 3-5. Fixed by thresholding `P(r_t <= r_min)`
instead (mass on small run lengths generally, not r=0 specifically). That
introduced a second issue: `P(r_t <= r_min)` is mechanically 1.0 for the
first `r_min` events no matter what the data looks like, fixed with a
`warm_up_events` gate on `is_ready()`, same convention every other
detector here uses.

**KSWIN's default alpha (`kswin.py`).** river's KSWIN reruns a fresh
KS-test on every event once its window fills (confirmed from river's
source, not just the "should be set below 0.01" hint in its docs) — over
a stream of a few thousand events that's a few thousand repeated
hypothesis tests with no multiple-testing correction, so river's own
default (0.005) gives a false-alarm rate near 100% (10/10 trials).
Retuned to `1e-3` here, which brings the false-alarm rate down to the
same order of magnitude as the other four detectors at this project's
event-count scale while still catching a real shift within a few dozen
events. This is an empirical operating point for this specific event
budget, not a principled default — retune if your stream length changes.

**LORD/SAFFRON rejection indexing (`fdr.py`).** `rejections` stored
1-indexed test numbers, but `evaluation.py`'s `evaluate_online_fdr` uses
those values directly as 0-indexed positions into the per-window
ground-truth label array — a rejection on the last test of a run threw an
`IndexError`. Off-by-one fix; regression test in `tests/test_fdr.py`.

## Design decisions and empirical findings

**`masked_channel_stream` doesn't register as anomalous to the VAE at
moderate settings.** A uniform multiplicative drop in HT/MET/jet-pT
(channels going quiet) pushes events toward a region the VAE reconstructs
well — background naturally includes low-activity events — so the raw
anomaly score actually falls rather than rises, up to a drop fraction of
about 0.7 (only an extreme ~0.9 drop clearly registers). Component 2's
`AdaptiveConformalThreshold.decide()` is a one-sided "flag if score is
high" gate, which structurally can't catch this. Component 1's two-sided
residual detectors can — a systematic negative residual is exactly what
their down-side branch exists for. This is why Component 2's demo in
`main.py` uses `misspecified_gradual_stream` (an additive, score-raising
bias) instead of `masked_channel_stream`: it's the right shape of failure
for a one-sided threshold, and `masked_channel_stream` is a genuinely
useful stress test for Component 1, not a redundant scenario. The two
components aren't redundant with each other — this is a concrete case
where one catches what the other structurally can't.

**Adaptive threshold recall vs. the fixed baseline.** In the default run
(`main.py`'s `detection_efficiency_vs_fixed_threshold` output), the
adaptive ACI threshold shows lower recall than the naive frozen threshold
on the misspecified-gradual scenario, despite both being well-calibrated
on background (ACI's empirical miscoverage tracks its 0.02 target
closely). A three-way ablation (`scripts/aci_recall_gap_ablation.py`:
sliding-buffer ACI vs. buffer-frozen ACI vs. a naive fixed threshold, all
on the same score/label stream) isolates the cause: freezing the
calibration buffer barely moves recall (sliding=0.186, frozen=0.205), and
both stay well below the naive fixed threshold (0.427). The sliding
buffer isn't the driver — `alpha_t`'s online adaptation of the target
miscoverage rate is what produces the recall drop. This is a genuine
precision/recall trade-off inherent to adaptive thresholding under this
feedback scheme, not something to fix.

**`n_jet` as the multiplicity covariate.** `residual.py`'s calibration
regression and `masked_channel_stream`/`multiplicity_step_stream` all use
jet count as the multiplicity signal. Real AXOL1TL/CICADA-style monitoring
would likely track several object multiplicities jointly (jets, muons,
electrons); this build simplifies to one. Extending it means editing
`residual.py::_design_matrix`.

**BOCPD underperforms CUSUM on continuous, non-resetting drift**
(`drift_sim/radiation_damage.py`, the space case study in `main.py`) —
confirmed across five independently trained VAEs, not one run. On the
permanent monotonic gain-decay scenario, CUSUM missed 0/75 trials across
all five seeds (miss_rate=0.000, std=0.000). BOCPD's miss rate varies a
lot by seed (0.267-0.867, mean=0.587, std=0.208), and when it does fire it
tends to do so either very early (<150 events) or not at all — no stable
middle latency. The direction of this result (CUSUM reliable, BOCPD not,
on this drift type) is solid; the specific BOCPD miss-rate number isn't a
fixed constant and should be reported as a mean ± std range, not a single
figure. Plausible mechanism, not proven: BOCPD's Normal-Inverse-Gamma
model assumes roughly constant parameters within a "run," so continuous
drift can get absorbed into an inflating variance estimate instead of
triggering a run-length reset, while CUSUM's fixed-slope accumulation has
no equivalent escape hatch. This finding has since reproduced on real data
(see [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md)).

**BOCPD and KSWIN fail the project's own throughput bar**
(`src/benchmark.py`). CUSUM/Page-Hinkley/ADWIN run at single-digit
microseconds per event; BOCPD is roughly two orders of magnitude slower
(vectorized NIG updates over many active run-length hypotheses), KSWIN
about three orders of magnitude slower (a fresh KS-test every event — see
its own docstring). If a "millions of events per second" claim goes in
the paper, these two detectors don't meet it as implemented.

**One-sided CUSUM for masked-channel: a real improvement, smaller than it
first looked** (`main.py::run_masked_channel_onesided_case_study`). The
default two-sided CUSUM (k=0.5, h=8.0) misses the masked-channel scenario
73-93% of the time across seeds — the residual shift here (~0.1-0.2 sigma)
is real but an order of magnitude smaller than what k=0.5 is tuned to
catch efficiently, and no two-sided (k, h) setting found by grid search
catches it without collapsing ARL0 to something unusable. A one-sided
CUSUM (k=0.1, h=16.0, watching only the known failure direction) looked
like a clean fix in an isolated diagnostic script (miss rate 0.0-0.075).
It's not that clean once run through this project's actual shared
evaluation function, `evaluation.run_detector_on_residuals()`, which stops
at the first alarm anywhere in a stream and counts an early false alarm as
a full miss of the real shift. Because the one-sided config is
meaningfully more sensitive (ARL0 ~3500, meaning a real chance of a
pre-changepoint false alarm within a 2500-event pre-changepoint window),
this single-shot evaluation penalizes it more than the less sensitive
default. Net result: one-sided still beats two-sided at every seed
checked, but nowhere near the initial "near-total detection" estimate.
That's a genuine methodological finding, not a solved problem — a more
sensitive, correctly-directed detector still trades against false-alarm
risk under strict single-shot detection semantics, and that trade-off is
real. This finding has since reproduced on real data (n=6 seeds; see
[REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md)).
## Matched-FAR baseline comparison (CUSUM vs. Page-Hinkley vs. ADWIN vs. KSWIN)

At matched ~5% false-alarm rate on synthetic gradual-drift data (6 seeds,
50 trials/seed), ADWIN detects 17% faster than CUSUM (264 vs. 318 mean
events to detection) while also achieving a lower miss rate (1.3% vs.
3.7%) -- consistent across every individual seed, not just on average.
Page-Hinkley tracks CUSUM almost exactly, as expected from their proven
mathematical equivalence. KSWIN is slowest on both axes; its false-alarm
match was also the least precise of the four (see caveat in the full
writeup). Full methodology, a real search-direction bug caught and fixed
during development, and reproduction commands:
[docs/BASELINE_COMPARISON.md](BASELINE_COMPARISON.md).
