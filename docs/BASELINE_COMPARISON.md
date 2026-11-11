# Matched false-alarm-rate baseline comparison

Sequential change-point detectors are only comparable to each other at a
shared false-alarm rate -- a detector run at a stricter threshold will
always look slower and more reliable than one run at a looser threshold,
regardless of anything about its underlying detection quality. `main.py`'s
Component 1 sweep runs CUSUM, Page-Hinkley, ADWIN, and KSWIN at their own
independent default operating points (`h=8.0`, `lam=10.0`, `delta=0.002`,
`alpha=1e-3` respectively) -- not matched to a common false-alarm rate, so
their detection latencies in that sweep aren't directly comparable to each
other. This page reports a matched-FAR comparison built specifically to
close that gap: `scripts/matched_far_baseline_comparison.py`.

## Method

For each detector, its threshold parameter is searched (on synthetic
no-drift "stable" windows) until it hits a target false-alarm rate
(0.05 here), then detection latency and miss rate are measured on the
*same* `misspecified_gradual_stream` drift trials -- the same scenario
`main.py`'s Component 2 demo already uses.

**A real bug caught during development, worth documenting:** the search
originally assumed that ascending a detector's threshold parameter always
*decreases* its false-alarm rate -- true for CUSUM's `h` and
Page-Hinkley's `lam` (larger threshold = more conservative), but the
*opposite* for ADWIN's `delta` and KSWIN's `alpha` (smaller = more
conservative). Applying the same ascending search to all four meant
ADWIN and KSWIN's search stopped at the very first, most-conservative
grid point tried (since it already trivially satisfied `far <= target`),
silently evaluating them near 0% FAR instead of the intended 5% --
unfairly inflating their apparent latency relative to CUSUM/Page-Hinkley.
Fixed by making search direction explicit per detector
(`descending=True` for ADWIN/KSWIN, `False` for CUSUM/Page-Hinkley).

## Results (6 seeds, 50 drift trials/seed, target FAR = 0.05)

Rerun after fixing a seed-overlap bug (see "Correction" below). Thresholds were searched on 100
stable windows per seed and the false-alarm rate in the table is measured on those same windows,
so it is **in-sample**.

| Detector      | Achieved FAR (search windows) | Miss rate       | Mean latency, detected trials (events) |
| ------------- | ----------------------------- | --------------- | -------------------------------------- |
| CUSUM (k=0.5) | 0.048 ± 0.004                 | 0.050 ± 0.043   | 318.2 ± 19.9                           |
| Page-Hinkley  | 0.047 ± 0.005                 | 0.050 ± 0.043   | 318.5 ± 19.4                           |
| ADWIN         | 0.047 ± 0.005                 | 0.020 ± 0.023   | 342.9 ± 11.5                           |
| KSWIN         | 0.023 ± 0.009                 | 0.007 ± 0.009   | 443.3 ± 40.2                           |

(± is the standard deviation across the 6 seeds.)

**What the data support**

- **CUSUM is not slower than ADWIN here; if anything it is faster.** ADWIN's mean latency was higher
  than CUSUM's in every one of the 6 seeds, by about 8% on average (343 vs. 318 events), though the
  gap was small in two seeds.
- **The reliability difference is not established.** ADWIN missed 6 of 300 drift trials and CUSUM 15
  of 300 (Fisher exact p ≈ 0.07 on the pooled counts). CUSUM's per-seed miss rate ranges from 0 to
  12%, so this is suggestive at best, and it is not a reason to prefer ADWIN.
- **Latency is averaged over detected trials only.** A detector that misses more trials is not
  penalized in that column. The out-of-sample script below also reports censored latency, which counts
  a miss as the full remaining stream.
- **Page-Hinkley tracks CUSUM** (318.5 vs. 318.2, identical in 5 of 6 seeds), as expected, since their
  up-side recursions are mathematically equivalent (docs/FINDINGS.md). Treat it as redundant.
- **KSWIN is slowest** and was evaluated at a stricter false-alarm budget (0.023 vs. about 0.047 for the
  others), so its low miss rate is partly bought by that stricter operating point.

**Correction.** An earlier version of this page reported that ADWIN detected 17% faster than CUSUM
(264 vs. 318 events) with a lower miss rate. That did not survive a rerun in which the stable-window
seeds no longer overlap across seeds (the earlier code let seeds 0 and 1 share 99 of 100 stable windows).
I attribute the change to that fix, but I have not isolated it by rerunning the old code, so this page
does not claim a mechanism beyond the timing.

**Remaining limitations**

- **In-sample false-alarm matching.** Every detector's threshold and the reported FAR use the same
  stable windows, so achieved FAR looks closer to the target than an independent measurement would.
- **CUSUM's reference value k was fixed at 0.5**, so "CUSUM" here is not tuned while ADWIN's delta is
  searched. A different k could move CUSUM's latency either way.
- **Synthetic data only**, one scenario (`misspecified_gradual_stream`), 6 seeds. Not run on real CMS
  Open Data.

## Out-of-sample version (`scripts/matched_far_baseline_comparison.py`, current)

The same script now also (a) measures each detector's false-alarm rate on fresh stable windows that no
threshold was chosen on (`--n-val-windows`, default 200); (b) matches CUSUM's h to the target for each
k in `--cusum-ks` (default 0.25, 0.5, 1.0) and selects k on a separate set of drift trials
(`--n-tune-trials`), never on the evaluation trials; and (c) reports censored latency. The original
four entries keep their windows, grids and evaluation trials, so the table above is reproducible with
the same command. **The out-of-sample results are not recorded on this page yet.** Quote them from the
aggregate output below once run.

## Reproduce

```bash
python3 scripts/matched_far_baseline_multiseed.py
python3 scripts/aggregate_matched_far_baseline.py results/matched_far_multiseed/seed*.json
python3 scripts/aggregate_baseline_oos.py results/matched_far_multiseed/seed*.json
```

The first aggregate script reproduces the table above. The second adds fresh-window FAR, censored
latency, the CUSUM k sweep and a paired per-seed ADWIN-vs-CUSUM(tuned k) comparison.

## What's next

- Record the out-of-sample results above and restate the ADWIN-vs-CUSUM comparison against CUSUM at its
  tuned k.
- Tighten KSWIN's search grid so its achieved FAR matches the target as closely as the others.
- Extend this comparison to real CMS Open Data, following the pattern in `docs/REAL_DATA_VALIDATION.md`.
