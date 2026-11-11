# Real-data validation

**Scope (read first).** Real-data runs use CMS Open Data record 30558
(2016 JetHT NanoAOD: offline-reconstructed jet/MET/HT/multiplicity
features, 8.8M cached events). JetHT is a prescaled, jet-trigger-selected
stream, not zero-bias, and the features are not Level-1 trigger objects.
The VAE is an AXOL1TL-style proxy: no AXOL1TL or CICADA data, inputs, or
scores are used. Only CUSUM (default, retuned, one-sided, empirically
calibrated) and BOCPD (radiation-damage case only) have been run on real
data. Page-Hinkley, ADWIN, and KSWIN (the matched-FAR comparison is
synthetic-only) and ACI/LORD/SAFFRON have not.

After the real-data pivot, the CUSUM/BOCPD pipeline was run on real CMS
Open Data (record 30558, JetHT dataset, 8.8M cached events) across six
independently trained proxy VAEs (seeds 0-5), using `scripts/real_data_sweep.py` and the
same `evaluation.run_detector_on_residuals` path the synthetic results
use — not a standalone diagnostic (this matters; see the one-sided-CUSUM
note in [FINDINGS.md](FINDINGS.md)). Each seed used ten non-overlapping
5000-event real segments (the burn-in and the segments are the same real
events for every seed -- only the VAE seed varies, so the +/- below reflects
VAE variability, not sampling variability over data), with drift injected via
`real_masked_channel_stream`/`real_radiation_damage_stream`
(`src/drift_sim/real_data_injection.py`) — same injection math as the
synthetic generators, just applied to real feature arrays. This mirrors
the "real substrate, synthetic controlled perturbation" approach used in
the Kepler PDC papers (Stumpe et al. 2012, 2014).

## Results (mean ± std across 6 seeds)

| Check                                           | Result                  |
| ----------------------------------------------- | ----------------------- |
| ARL, no injection (default CUSUM, k=0.5, h=8.0) | ARL = 2991.5 ± 695.2    |
| False-alarm rate, no injection                  | 66.7% ± 19.7%           |
| Masked-channel, default CUSUM                   | miss rate 70.0% ± 16.3% |
| Masked-channel, one-sided CUSUM (k=0.1, h=16.0) | miss rate 53.3% ± 12.5% |
| Radiation-damage, CUSUM                         | miss rate 10.0% ± 10.0% |
| Radiation-damage, BOCPD                         | miss rate 38.3% ± 19.5% |

"Miss rate" is the repo's single-shot definition: an alarm raised *before*
the true onset (event 2500 for masked-channel) counts as a miss, so these
figures include pre-onset false alarms. Radiation-damage rows use
`tau_damage=3000` (onset at event ~153); the synthetic radiation-damage
figures quoted elsewhere use `tau_damage=400` (onset at event ~20) and are
not like-for-like; see the `tau_damage=400` real run below.

**What holds up on real data:** the masked-channel failure mode (default
CUSUM missing most injections) and its partial fix (one-sided CUSUM) both
reproduce in the same direction and roughly the same magnitude as the
synthetic runs (67-93% synthetic vs. 70.0% ± 16.3% real for the default
detector; both at `drop_fraction=0.4`). The radiation-damage story holds too — CUSUM is far more
reliable than BOCPD on this drift type in both substrates (real: 0.0% vs.
28.3% at `tau_damage=400`, 10.0% vs. 38.3% at `tau_damage=3000`;
synthetic: 0.000 vs. 0.587 at `tau_damage=400`). The ordering holds in all
three, but BOCPD's real miss rate is highly seed-dependent (0-60% across
VAE seeds), so its magnitude is not stable.

**What does not transfer:** the false-alarm rate under the default
(k=0.5, h=8.0) settings on real background is 66.7% ± 19.7% over
5000-event windows (53.8% across seeds 0-3 in the 20-window retune run).
The earlier comparison figure ("roughly 17-33% in the synthetic sweep")
came from 2000-event windows with n=6 trials and is not comparable. At
matched window length, with 300 fresh windows
(`scripts/synthetic_far_windows.py`, seed 0, 3000-event in-sample
reference), the synthetic pipeline gives 28.7% (95% CI 23.8-34.0%) over
5000 events and 11.7% (8.5-15.8%) over 2000. Ideal i.i.d. Gaussian
residuals give 40.8% and 18.6% (Monte Carlo, 20,000 trials; Siegmund's
approximation gives a two-sided ARL0 of roughly 9,600 events). So at
matched window length the real-data rate is roughly 1.9-2.3x the
synthetic rate and 1.3-1.6x the ideal-Gaussian rate. The real figures rest
on only 10-20 windows (the same windows for every VAE seed), so their
uncertainty is wide (6/10 windows: 95% CI 31-83%). The synthetic rate sits
*below* the Gaussian ideal because its held-out residuals have mean +0.066
but std 0.977; the corresponding held-out statistics on real windows have
not been measured. See below for retuning and for what the residual
diagnostics do and do not show.

## CUSUM (k, h) retune

`scripts/real_data_retune.py` grid-searched `h` (holding `k=0.5`) against
real background, reusing cached residuals across the grid rather than
retraining the VAE per point. Across seeds 0-3:

| Config        | FA rate       | masked-channel miss rate | radiation-damage miss rate |
| ------------- | ------------- | ------------------------ | -------------------------- |
| h=8 (default) | 53.8% ± 11.4% | 80.0% ± 10.0%            | 2.5% ± 4.3%                |
| h=10          | 8.75% ± 6.5%  | 92.5% ± 8.3%             | 15.0% ± 11.2%              |

`h=10` cuts the false-alarm rate about 6x, but masked-channel detection
(already the harder case) gets worse and radiation-damage, previously
near-perfect, degrades too. `h≥12` drives FA rate to zero but takes
detection down with it.

A joint `(k, h)` search does better. At `k=0.2, h=24` (seeds 0-3):

| Config               | FA rate       | masked-channel miss rate | radiation-damage miss rate |
| -------------------- | ------------- | ------------------------ | -------------------------- |
| k=0.5, h=8 (default) | 53.8% ± 11.4% | 77.5% ± 8.3%             | 7.5% ± 4.3%                |
| k=0.2, h=24          | 33.8% ± 7.4%  | 82.5% ± 13.0%            | **0.0% ± 0.0%**            |

False-alarm rate drops from 54% to 34% and radiation-damage detection
becomes perfectly reliable, at the cost of a small, noisy bump in
masked-channel miss rate. Real improvement, but not a full fix.

## Root cause

`scripts/diagnose_residual_distribution.py` checks two candidate
explanations for why the gap resists closing: real residuals being more
autocorrelated, or more heavy-tailed, than the synthetic generator
produces — either would break CUSUM's implicit i.i.d.-Gaussian assumption
and inflate false alarms regardless of threshold tuning. Measured on
burn-in residuals:

| Metric                | Synthetic | Real   |
| --------------------- | --------- | ------ |
| Lag-1 autocorrelation | 0.005     | -0.018 |
| Excess kurtosis       | 0.072     | 0.452  |

Lag-1 autocorrelation is negligible in both cases. Excess kurtosis is
higher on real data (0.45 vs 0.07), but 0.45 is a mild departure from
Gaussian, and both numbers are measured on the *burn-in* residuals, which
are in-sample for the VAE and the calibration regression (mean 0 and std 1
by construction). These diagnostics therefore do not establish heavy tails
as the cause of the real-data false-alarm excess: they do not examine
held-out windows, and ACF is only checked to lag 5. The cause is open. A
small held-out residual offset (e.g. mean +0.1, or std x1.05) would by
itself reproduce the observed rates at h=8, 10 and 11 (simulation), so the
next step is to measure held-out per-window residual mean/std before
choosing a fix (empirical-threshold calibration, robust scaling, or a
three-way burn-in split). Note that an offset alone does not imply
inflation: on the synthetic pipeline the held-out mean offset is real
(+0.066, consistent across 300 windows) but is outweighed by a std below 1
(0.977), giving a net false-alarm rate below the Gaussian ideal. Batch 2 measured
these held-out statistics (see "Batch 2" below): the real-data excess is a
4% scale mismatch plus between-window variation tied to pileup, not heavy
tails.

## Masked-channel: one-sided score gate vs. residual detectors

`scripts/score_gate_baseline.py` splits each trial (changepoint at event
2500 of 5000) into an alarm *before* onset (false alarm) and detection
*after* onset by a detector restarted at the onset ("fresh"; delay from
onset). The single-shot miss rate used elsewhere lumps the two together.
Thresholds and detector references come from the burn-in sample, which is
in-sample for the VAE and regression, so absolute false-alarm rates are
optimistic. The pre-onset alarm rate is the null rate (an alarm within 2500
events with no drift) against which post-onset detection should be read.
P-values below are unpaired Fisher tests, approximate.

Rows show pre-onset alarm rate / post-onset detection rate (median delay
in events). Synthetic: 50 trials, 3000-event burn-in, one VAE seed.

| Detector                                              | drop 0.15            | drop 0.4             |
| ----------------------------------------------------- | -------------------- | -------------------- |
| CUSUM up-only on residual (analogue of a high-score gate) | 0.12 / 0.02 (68)  | 0.12 / 0.02 (2491)   |
| CUSUM two-sided on residual (k=0.5, h=8)              | 0.24 / 0.14 (1167)   | 0.24 / 0.36 (963)    |
| CUSUM two-sided on raw log-score (no regression)      | 0.22 / 0.14 (1167)   | 0.22 / 0.46 (1167)   |
| CUSUM down-only on residual, oracle direction (k=0.1, h=16) | 0.46 / 0.80 (576) | 0.46 / 0.94 (389)  |

Event-level gate (flag if score > 99th percentile of burn-in scores): flag
rate 0.0094 -> 0.0086 (drop 0.15) and 0.0094 -> 0.0094 (drop 0.4). Mean
residual shift after onset: -0.064 sigma and -0.108 sigma.

Real JetHT: 30 random disjoint segments, 20,000-event burn-in, one VAE
seed, drop 0.4.

| Detector                                              | pre-alarm / detect (delay) |
| ----------------------------------------------------- | -------------------------- |
| CUSUM up-only on residual                             | 0.17 / 0.10 (1122)         |
| CUSUM two-sided on residual (k=0.5, h=8)              | 0.30 / 0.63 (936)          |
| CUSUM two-sided on raw log-score (no regression)      | 0.63 / 0.83 (532)          |
| CUSUM down-only on residual, oracle direction (k=0.1, h=16) | 0.83 / 1.00 (214)    |

Event-level gate: flag rate 0.0111 -> 0.0125; mean residual shift -0.113 sigma.

Reading these:

- **A one-sided score gate cannot alert on this failure.** Synthetic
  post-onset flag rate is unchanged or lower, and the up-only CUSUM's
  post-onset alarm rate (2%) is at or below its null rate (12%). On real
  data the flag rate rises slightly (1.11% -> 1.25%); detecting a change
  that small would take on the order of 50,000 events per arm.
- **The default two-sided residual CUSUM does not demonstrably detect it.**
  Synthetic, drop 0.4: detection 36% (CI 24-50%) vs null 24% (CI 14-37%),
  p=0.28; drop 0.15: detection 14% is below the null rate. On real data
  detection (63%) exceeds the null rate (30%), p=0.02, but the null rate is
  itself high.
- **(Superseded by the matched-FAR comparison below, where direction-blind
  two-sided detection also works.) Only the small-slack, oracle-direction
  CUSUM detects reliably**
  (synthetic 94% vs 46% null, p<1e-6; real 100% vs 83% null, p~0.05), and
  its pre-onset alarm rate is 46% (synthetic) to 83% (real) per 2500
  events. At h=16 it is unusable as a standalone monitor. Detectors with
  different false-alarm rates are not comparable; the next step is
  matched-FAR calibration on held-out windows.
- **Does the regression help?** *Retracted.* At identical (k, h) the
  residual's pre-onset false-alarm rate was 30% vs 63% for raw log-score
  (p=0.02), but those are different false-alarm rates, so the comparison
  was unfair. At matched false-alarm rate the regression shows no
  consistent advantage (see the matched-FAR comparison below).

## Radiation damage at tau_damage=400 (real)

Real JetHT, `tau_damage=400` (onset at event ~20), six VAE seeds over the
same ten segments (`scripts/real_data_sweep.py --tau-damage 400`): CUSUM
miss rate 0.0% ± 0.0% (mean latency 612.9 ± 99.1 events after onset);
BOCPD 28.3% ± 25.4% (latency 1904 ± 788; per-seed miss rates 0.0, 0.6, 0.1,
0.5, 0.5, 0.0). Synthetic at the same decay constant: CUSUM 0.0%, BOCPD
58.7%. The CUSUM-more-reliable ordering holds, but BOCPD's magnitude varies
widely across VAE seeds. CUSUM's detection at ~600 events comes when the
gain is down to roughly 20% of nominal (about an 80% loss), a very large
drift, and the comparison is not matched at a common false-alarm rate: any
post-onset alarm counts, and CUSUM's real false-alarm rate is 54-67% per
5000 events.

## Batch 2: held-out calibration and matched false-alarm comparison

Scripts: `scripts/real_heldout_residual_stats.py` and
`scripts/matched_far_heldout.py` (logs in `results/logs/`: `hs_synthetic`,
`hs_real`, `mfh_synthetic`, `mfh_real`, `mfh_real_adwin`, `mfh_real_bocpd`).
"insample" is the original calibration (VAE, regression and reference on
one block); "three_way" uses disjoint VAE / regression / reference blocks.
Both are scored on the same windows. Everything below uses a 20,000-event
VAE block and 10,000-event regression and reference blocks.

### Held-out residual statistics over the whole pool

Whole pool after burn-in, windows of 5000 events (real: 1755 windows;
synthetic stationary control: 292). z-mean and z-std are the held-out
residual's offset and scale relative to the detector's reference.

| | synthetic, insample | synthetic, three_way | real, insample | real, three_way |
| --- | --- | --- | --- | --- |
| z-mean / z-std | +0.005 / 0.996 | +0.014 / 0.995 | +0.012 / 1.040 | -0.043 / 1.025 |
| excess kurtosis | -0.075 | -0.075 | +0.49 | +0.49 |
| window-mean sd vs i.i.d. expectation | 1.01x | 1.01x | 3.99x | 3.93x |
| corr(window mean residual, window mean pileup) | +0.09 | +0.10 | -0.18 | +0.49 |
| CUSUM FAR, h=8, 5000 ev: empirical / Gaussian prediction / ideal | .414 / .387 / .406 | .394 / .390 / .406 | .658 / .613 / .406 | .677 / .586 / .406 |
| h=10 | .068 / .063 / .066 | .065 / .063 / .066 | .177 / .137 / .066 | .232 / .128 / .066 |
| h=11 | .031 / .024 / .025 | .031 / .024 / .025 | .083 / .056 / .025 | .123 / .054 / .025 |

- **The control passes.** On stationary synthetic data the procedure is
  nominal: z about (0, 1), window means behave i.i.d., no trend (p=0.59), and
  empirical false-alarm rates match the Gaussian prediction.
- **Real data: scale mismatch.** The held-out std is 1.04x the in-sample
  reference (which is 1.000 by construction). That 4% alone moves the h=8,
  5000-event false-alarm probability from 41% to about 61% (Gaussian
  prediction), most of the observed 66%.
- **Real data: slow variation between windows.** Window means vary about
  4x more than i.i.d. noise allows (sd 0.058 sigma), with no within-window
  autocorrelation at any lag up to 500, so it is between-window variation,
  not short-range correlation. There is a small downward trend (-0.002 sigma
  per million events, p=0.002) but no monotonic false-alarm trend across
  thirds of the pool.
- **Real data: pileup.** 9.2% of window means and 2.9% of events fall
  outside the pileup range the regression was fit on, and the residual's
  dependence on pileup is unstable: window-mean correlation with pileup is
  -0.18 under the in-sample fit and +0.49 under the three-way fit. The
  quadratic regression fit on a short contiguous burn-in does not capture
  the pileup dependence on held-out data.
- **Tails are not the cause.** Excess kurtosis 0.49 is mild; the earlier
  "heavy tails" explanation is withdrawn.
- **The three-way split did not lower the uncalibrated false-alarm rate**
  (67.7% vs 65.8% at h=8). It fixes the scale (z-std 1.025) but introduces an
  offset (z-mean -0.043), because the 10,000-event reference block is a
  narrow time slice that does not represent the pool (its mean is -0.0004 vs
  -0.0445 held-out).

### Matched held-out false-alarm comparison

Three seeds, each on its own disjoint region of the pool. Per seed: 300
calibration windows choose each threshold so that P(alarm within 2500
events) <= target; 300 validation windows (never used to choose it) measure
the achieved rate; 200 evaluation windows with drift injected from event 0
measure detection by a freshly started detector. Detection below is at
target 0.05, pooled over seeds (600 evaluation windows per scenario).

**Calibration works.** Achieved validation false-alarm rates fall within
about 1-2 points of target for nearly every configuration, on synthetic and
real data, in both modes. Some are unachievable: two-sided k=0.1 on the raw
log-score at every target, and on the residual at target 0.02 (three_way).
On real data, slow variation of the residual mean (0.058 sigma between
windows) exceeds a 0.1-sigma slack, so a small-slack CUSUM cannot reach low
false-alarm rates at any finite threshold.

Detection probability, three_way / insample (95% CIs in the logs):

| Detector (k) | masked 0.15 | masked 0.4 | radiation tau=400 | radiation tau=3000 |
| --- | --- | --- | --- | --- |
| **Real data** | | | | |
| two-sided, residual (0.1) | 30 / 27% | 20 / 38% | 100 / 100% | 28 / 37% |
| two-sided, residual (0.25) | 16 / 13% | 16 / 18% | 100 / 100% | 16 / 15% |
| two-sided, residual (0.5) | 8 / 6% | 7 / 6% | 100 / 100% | 6 / 6% |
| two-sided, raw log-score (0.25) | 30 / 31% | 36 / 36% | 100 / 100% | 32 / 34% |
| down-only, residual, oracle (0.1) | 50 / 54% | 67 / 61% | 4 / 9% | 60 / 57% |
| up-only, residual (0.1), high-score-gate analogue | 2 / 1% | 1 / 1% | 99 / 100% | 3 / 2% |
| ADWIN on residual | 3 / 3% | 3 / 3% | 100 / 100% | 7 / 9% |
| BOCPD on residual (three_way only) | 2% | 0% | 0% | 2% |
| **Synthetic control** | | | | |
| two-sided, residual (0.1) | 52 / 40% | 99 / 93% | 100 / 100% | 91 / 82% |
| down-only, residual, oracle (0.1) | 61 / 58% | 99 / 98% | 22 / 21% | 94 / 91% |
| up-only, residual (0.1) | 0 / 0% | 0 / 0% | 100 / 100% | 0 / 0% |

Achieved false-alarm rates for the real-data rows are 4-7% for the CUSUM
rows at target 0.05, and 2-3% for ADWIN and BOCPD, whose threshold grids
are coarser, so their detection is, if anything, slightly understated.

- **A one-sided high-score gate is blind to moderate failures.** Up-only
  detection is 0-3% (null about 5%) for both channel drops and for slow
  decay (tau=3000), synthetic and real, with tight intervals.
- **The score's response changes sign with severity.** Moderate or slow
  degradation lowers the score (down-only detects, up-only is blind). Severe,
  fast damage (tau=400) raises it (up-only 99-100%, down-only 4-9%). This
  holds on synthetic and real data. A fixed-direction monitor, whether a gate
  or an oracle-direction CUSUM, is therefore unsafe; the direction-blind
  two-sided detector caught both regimes.
- **Direction-blind detection is strong on synthetic data and modest on real
  data.** Synthetic: 93-99% for a 40% drop at k=0.1. Real: 20-38% for a 40%
  drop at k=0.1 (null 4-7%), 16-18% at k=0.25, and 6-7% at k=0.5, which is
  indistinguishable from the null. Median delays are about 900-1400 events of
  the 2500-event window. The limit is nuisance variation: the masked-channel
  shift (-0.11 sigma) is about twice the between-window sd of the mean
  (0.058 sigma) and the same size as the seed-to-seed reference offsets
  (-0.03 to +0.05 sigma).
- **Oracle direction raises real-data power where the direction is right**
  (masked 0.4: 61-67%; masked 0.15: 50-54%; tau=3000: 57-60%) and fails where
  it is wrong.
- **The regression shows no consistent advantage at matched false-alarm
  rate.** Two-sided k=0.25, masked 0.4: raw log-score 36% vs residual 16-18%.
  Down-only k=0.1: residual 61-67% vs raw 28-36%. Synthetic: equal.
- **BOCPD (default prior and hazard, threshold tuned) is not competitive**:
  0-2% everywhere, including tau=400 where two-sided CUSUM and ADWIN detect
  100%. ADWIN detects masked-channel at about the null (3%) and slow decay
  weakly (7-9%).
- **Earlier radiation-damage numbers were not false-alarm matched.** CUSUM's
  10% miss rate at tau=3000 and BOCPD's 28-59% miss rates came from
  detectors running at high false-alarm rates. At matched false-alarm rate,
  CUSUM's tau=3000 detection is 6-37% (two-sided) and BOCPD's is 2%. The
  CUSUM-over-BOCPD ordering holds more strongly; the absolute CUSUM
  reliability claim does not.
- **The three-way split gives no consistent gain at matched false-alarm
  rate** (two-sided k=0.1, masked 0.4: 20% vs 38%; down-only: 67% vs 61%).
  Thresholds recalibrated on held-out windows already absorb the in-sample
  optimism, so the protocol, not the split, is what fixes calibration.

Caveats: three VAE seeds; the same 200 evaluation windows per seed are used
across scenarios; every k is reported, so selecting the best k would select on
the evaluation windows; "oracle direction" assumes the failure direction is
known; ADWIN's delta grid is coarse; BOCPD uses default hyperparameters.

## Extraction backend: uproot vs. PyROOT

Feature extraction from real CMS Open Data has two implementations — the
original `uproot`-based path (`src/stream_loader.py::real_object_stream` /
`_compute_object_features` / `_compute_pileup`) and a PyROOT/RDataFrame
rebuild (`scripts/build_real_cache_pyroot.py`), covering the same features
(jet1/jet2 kinematics, n_jet, MET, HT, n_muon, n_electron) and the same
`nJet >= 2` selection. `scripts/validate_pyroot_cache.py` confirms they're
bit-identical: max_abs_diff=0.0 across all 12 features and both metadata
arrays, over all 8,815,092 cached events. One gotcha for anyone
reproducing this: `ROOT.ROOT.EnableImplicitMT()` needs to stay off during
extraction — RDataFrame's multithreaded execution doesn't guarantee
`AsNumpy()`'s row order matches sequential file/event order, which broke
the row-by-row comparison until it was disabled. PyROOT is now the default
cache source (`data/real_cache/jetht_features.npz`); the uproot cache is
kept around for reference (`data/real_cache/jetht_features_uproot.npz`).

## Data provenance

The five real CMS Open Data files used here (record 30558,
`/JetHT/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD`, DOI
`10.7483/OPENDATA.CMS.8ALJ.MQSO`) were checked, not just downloaded and
trusted:

- **File integrity**: every local file's size and adler32 checksum match
  CERN's record metadata exactly (`scripts/verify_real_data_provenance.py`).
- **Content authenticity**: the record declares Run2016H, run numbers
  281613-284044. The actual `run` branch values inside the files
  (283876-284044) fall inside that range, confirming genuine CMS collision
  data rather than something synthetic or misattributed. The EDM
  provenance objects (`edm::ProcessHistory`, `edm::ProcessConfiguration`,
  etc., visible as ROOT warnings when reading the files) confirm these are
  real CMSSW-produced NanoAOD files.

Reproduce with:
```bash
cernopendata-client get-metadata --recid 30558 > /tmp/record_30558.json
python3 scripts/verify_real_data_provenance.py
```

## Full reproduction

```bash
for s in 0 1 2 3 4 5; do
  python3 scripts/real_data_sweep.py --seed $s --out results/real_seeds/seed${s}.json
done
python3 scripts/aggregate_real_sweep.py results/real_seeds/*.json

# (k,h) retune sweep and root-cause diagnosis
python3 scripts/real_data_retune.py --k-values 0.2,0.3,0.4,0.5 --h-values 6,8,10,12,16,20,24
python3 scripts/diagnose_residual_distribution.py

# ACI recall-gap ablation
python3 scripts/aci_recall_gap_ablation.py

# PyROOT extraction (run in a SEPARATE venv with ROOT installed — see
# scripts/build_real_cache_pyroot.py's docstring for setup, since ROOT
# isn't pip-installable into a normal project venv)
python3 scripts/build_real_cache_pyroot.py
python3 scripts/validate_pyroot_cache.py
```

## What's stubbed / still needs validation

- `real_object_stream()`'s live-XRootD/EOS path (`stream_loader.py`) is
  untested against a live endpoint — all real-data work in this repo uses
  a locally cached feature pool pulled via `cernopendata-client`, not a
  live stream.
- The proxy VAE's feature set (jet1/jet2 kinematics, MET, HT, n_jet,
  n_muon, n_electron) is a reasonable AXOL1TL-object analog but hasn't
  been checked against a real anomaly-detection baseline — see
  `docs/ADVISOR_NOTES.md` for the scope discussion behind this.
- Detector hyperparameters (Page-Hinkley's `lam`, BOCPD's `hazard_lambda`,
  KSWIN's `alpha`) are tuned against this project's synthetic
  event-count scale and haven't been checked against real background.
  CUSUM's default (k=0.5, h=8.0) has been checked (see above): its ARL is
  the right order of magnitude on real data, but its false-alarm rate
  isn't; a joint `(k, h)` retune narrows the gap without closing it, and
  the residual gap's cause is not yet established (real burn-in residuals
  have mildly higher excess kurtosis, 0.45 vs 0.07, but the diagnostics only
  cover in-sample burn-in residuals; see "Root cause" above).
  Calibrating the threshold against the empirical real residual
  distribution directly, instead of an implicitly Gaussian one, is the
  natural next step but isn't implemented here.
- Real-data validation draws from a single Open Data record (one run
  range, 283876-284044, a narrow slice of Run2016H) — generalization
  across run periods or datasets is untested.
- The gradual-drift scenarios (`drift_sim/gradual.py`, lumi-trend based)
  are still synthetic-only — a single Open Data record has no meaningful
  instantaneous-luminosity trend to perturb (see
  `src/drift_sim/real_data_injection.py`'s docstring for why).