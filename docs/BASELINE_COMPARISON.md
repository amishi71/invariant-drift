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

| Detector      | Achieved FAR    | Miss rate       | Mean latency (events) |
| ------------- | ---------------- | ---------------- | ----------------------- |
| CUSUM         | 0.045 ± 0.005    | 0.037 ± 0.033     | 317.9 ± 20.7            |
| Page-Hinkley  | 0.033 ± 0.009    | 0.030 ± 0.028     | 318.8 ± 21.0            |
| **ADWIN**     | **0.048 ± 0.004**| **0.013 ± 0.015** | **263.5 ± 15.9**        |
| KSWIN         | 0.030 ± 0.013    | 0.007 ± 0.009     | 452.6 ± 43.0            |

**Headline finding:** at matched ~5% false-alarm rate, **ADWIN detects
17% faster than CUSUM** (264 vs. 318 events) while also achieving a
*lower* miss rate (1.3% vs. 3.7%), consistently across all 6 seeds
individually, not just on average. CUSUM and Page-Hinkley track each
other almost exactly at every seed -- expected, since `docs/FINDINGS.md`
already established their up-side recursions are mathematically
equivalent; any small divergence between them here is grid-resolution
noise (CUSUM's `h` grid steps by 0.25, Page-Hinkley's `lam` grid by 0.5),
not a real behavioral difference.

**Caveat, stated plainly:** ADWIN's `achieved_FAR` (0.048 ± 0.004) is the
tightest match to the 0.05 target of any detector. Page-Hinkley (0.033)
and especially KSWIN (0.030 ± 0.013, the largest spread of the four)
landed further from target -- meaning KSWIN's low miss rate is partly
earned by being evaluated at a stricter, more conservative false-alarm
budget than CUSUM/ADWIN actually got. The ADWIN-vs-CUSUM comparison is
the cleanest, most defensible claim from this experiment; the KSWIN
comparison is directionally consistent with its known throughput/
reliability weaknesses elsewhere in this project, but not yet as tightly
matched.

**Scope:** synthetic data only, one scenario (`misspecified_gradual_stream`),
6 seeds. Not yet run on real CMS Open Data.

## Reproduce

```bash
python3 scripts/matched_far_baseline_multiseed.py
python3 scripts/aggregate_matched_far_baseline.py results/matched_far_multiseed/seed*.json
```

## What's next

- Tighten KSWIN's search grid so its achieved FAR matches target as
  precisely as ADWIN's does, removing the one caveat above.
- Extend this comparison to real CMS Open Data, following the same
  pattern used for the CUSUM-only false-alarm-rate work in
  `docs/REAL_DATA_VALIDATION.md`.
