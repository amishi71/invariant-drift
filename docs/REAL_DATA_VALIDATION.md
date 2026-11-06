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
not like-for-like. A real run at `tau_damage=400` is pending.

**What holds up on real data:** the masked-channel failure mode (default
CUSUM missing most injections) and its partial fix (one-sided CUSUM) both
reproduce in the same direction and roughly the same magnitude as the
synthetic runs (67-93% synthetic vs. 70.0% ± 16.3% real for the default
detector; both at `drop_fraction=0.4`). The radiation-damage story holds too — CUSUM is far more
reliable than BOCPD on this drift type in both substrates in direction
(10.0% vs. 38.3% real miss rate at `tau_damage=3000`; 0.000 vs. 0.587
synthetic at `tau_damage=400` -- different drift speeds, so compare the
direction, not the magnitudes).

**What may not transfer:** the false-alarm rate under the default
(k=0.5, h=8.0) settings on real background is 66.7% ± 19.7% over
5000-event windows. The earlier comparison figure ("roughly 17-33% in the
synthetic sweep") came from 2000-event windows with n=6 trials and is not
comparable. For ideal i.i.d. Gaussian residuals, the false-alarm
probability at k=0.5, h=8 is about 41% over 5000 events and about 19% over
2000 (Monte Carlo, 20,000 trials; Siegmund's approximation gives a
two-sided ARL0 of roughly 9,600 events). So the real-data excess over the
Gaussian expectation is roughly 1.3-1.6x, not the 2-4x the earlier
comparison implied. The synthetic pipeline's own false-alarm rate at
5000-event windows has not been measured yet -- run
`scripts/synthetic_far_windows.py`. See below for retuning and for what
the residual diagnostics do and do not show.

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
three-way burn-in split).

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