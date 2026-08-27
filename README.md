# Invariant Drift

**Calibration-Residual Monitoring for Learned Anomaly Triggers**

*(Repo/package name stays `invariant-drift` — lowercase, kebab-case, for
imports and the URL. This is the project's actual name for the README,
paper title, and CV/portfolio line.)*

Two components for detecting when a learned anomaly trigger's calibration
has silently gone stale, evaluated on CMS/LHC Open Data:

1. **Sequential change-point detection** on a scalar calibration residual
   (anomaly score jointly with pileup/multiplicity, calibrated against
   expected luminosity trends) -- CUSUM, Page-Hinkley, and BOCPD
   (implemented here), compared against ADWIN and KSWIN (river).
2. **Adaptive conformal inference (ACI)** for online threshold
   recalibration from Zero-Bias control feedback + delayed offline
   validation, plus **online false discovery rate control** (LORD,
   SAFFRON) over windowed batches of the resulting decisions.

Full abstract: [`docs/ABSTRACT.md`](docs/ABSTRACT.md). Advisor-facing scope
note (why AXOL1TL-proxy, not CICADA): [`docs/ADVISOR_NOTES.md`](docs/ADVISOR_NOTES.md).

Reuses `CUSUMDetector`/`PageHinkleyDetector`'s core logic (frozen-reference
discipline) and the XRootD streaming/retry infrastructure from
[cms-streaming-shift-detection](https://github.com/amishi71/cms-streaming-shift-detection),
this project's predecessor (dijet-mass resonance-shift detection). See each
file's module docstring for exactly what was reused verbatim vs. adapted
vs. new.

## Layout

```
src/
  proxy_vae.py          AXOL1TL-proxy VAE on offline jet/MET/HT/multiplicity features
  residual.py            calibration residual (score vs pileup/mult/lumi regression)
  stream_loader.py       real CMS Open Data (XRootD) + synthetic dev/test stream
  kinematics.py           streaming mean/variance/skew/kurtosis (copied from Project A)
  detectors/
    cusum.py, page_hinkley.py, bocpd.py, adwin.py, kswin.py
  conformal/
    aci.py               adaptive conformal threshold
    fdr.py                LORD, SAFFRON, windowed_batch_pvalues
  drift_sim/
    gradual.py            pileup evolving across a fill (nominal + misspecified)
    abrupt.py              masked readout channels / multiplicity step
  evaluation.py           ARL, latency, false-alarm rate, coverage, online FDR
main.py                   end-to-end orchestration; `python main.py --help`
tests/                    pytest suite, 58 tests (see "Known issues" -- several
                           are regression tests for real bugs found below)
```

## Running it

```
pip install -r requirements.txt
python main.py                          # small defaults, ~20-30s on CPU
python main.py --n-burn-in 5000 --n-events 3000 --n-trials 15 --vae-epochs 80
pytest tests/ -v
```

`main.py` runs entirely on **synthetic data** by default -- see
`src/stream_loader.py`'s module docstring for why (the sandbox this was
built in has no network path to CERN's XRootD/EOS endpoints, only package
registries). `real_object_stream()` in that file is the direct real-data
path, adapted from Project A's retry/circuit-breaker logic, and should work
once pointed at real file lists from an environment with EOS access -- see
[`data/README.md`](data/README.md) -- but it has not been exercised against
a live endpoint here.

## Known issues / design decisions found during development

Left in deliberately, not swept under the rug -- these were real bugs
caught by actually running the code and cross-checking it against theory,
not hypothetical concerns:

- **Score log-transform (residual.py).** The raw VAE reconstruction-error
  score is right-skewed (skew ~1.35 empirically) since it's a sum-of-
  squares-type quantity. Regressing the raw score and standardizing the
  residual does NOT fix this -- the conditional distribution is still
  skewed after de-meaning, which silently breaks CUSUM/Page-Hinkley's
  Gaussian-ARL assumptions (empirically: false-alarm rate ~100% within a
  few hundred events instead of the theoretical ARL0 in the thousands).
  Fixed by fitting the calibration regression in log-score space by
  default (`score_transform="log"`), mirroring what Project A did for its
  own skewed observable (dijet mass).
- **Page-Hinkley two-sided bug (page_hinkley.py).** The down-side branch
  tracked a running MAXIMUM of the negated cumulative sum instead of a
  running MINIMUM. Since the cumulative sum has a `-delta` drift term
  regardless of any real shift, `max - current` grows ~linearly with event
  count on its own -- guaranteed false alarms within ~20-45 events on pure
  noise, reproduced in 10/10 trials before the fix. Fixed to mirror the
  up-side construction exactly; verified against a direct step-by-step
  equivalence to CUSUM's S+ recursion (`tests/test_cusum_page_hinkley.py`).
- **BOCPD detection criterion (bocpd.py).** `P(r_t = 0)` is the wrong
  statistic to threshold: once the run-length posterior concentrates on a
  single dominant hypothesis (which happens quickly), the predictive-
  likelihood term cancels out of the R(0)/R(r*+1) ratio, so P(r_t=0)
  converges to ~the bare hazard rate regardless of how surprising the new
  data point is -- verified directly (an injected 4-sigma shift left
  P(r=0) pinned at ~hazard while `map_run_length` correctly collapsed to
  3-5). Fixed to threshold `P(r_t <= r_min)` instead (mass on *small* run
  lengths generally, not r=0 specifically). This introduced a second
  issue -- `P(r_t <= r_min)` is mechanically 1.0 for the first `r_min`
  events regardless of data -- fixed with a `warm_up_events` gate on
  `is_ready()`, the same convention every detector here already uses.
- **KSWIN's default alpha (kswin.py).** river's KSWIN reruns a fresh
  KS-test on *every single event* once its window fills (confirmed from
  river's source, not just its docstring's "should be set below 0.01"
  hint) -- over a stream of a few thousand events that's a few thousand
  repeated hypothesis tests with no multiple-testing correction, so
  river's own default (0.005) gives a false-alarm rate near 100% (10/10
  trials, reproduced empirically). Retuned the default here to `1e-3`,
  which brings the false-alarm rate over this project's event-count scale
  down to the same order of magnitude as the other four detectors while
  still detecting a real shift within a few dozen events. This is an
  empirical operating point for *this* event budget, not a principled
  default -- retune if your stream length changes substantially.
- **LORD/SAFFRON rejection indexing (fdr.py).** `rejections` was storing
  1-indexed test numbers, while `evaluation.py`'s `evaluate_online_fdr`
  uses those values directly as 0-indexed positions into an array of
  per-window ground-truth labels -- a rejection on the very last test of
  a run produced an `IndexError` (one past the last valid index).
  Straightforward off-by-one fix; regression test in `tests/test_fdr.py`.
- **`masked_channel_stream` doesn't register as anomalous to the VAE at
  moderate settings (abrupt.py / component 2 design in main.py).** This
  isn't a bug so much as a real empirical finding worth flagging: a
  uniform multiplicative *drop* in HT/MET/jet-pT (channels going quiet)
  pushes events toward a region the VAE reconstructs *well* -- background
  naturally includes low-activity events -- so the raw anomaly score
  actually *falls* rather than rises at drop_fraction up to ~0.7 (only an
  extreme ~0.9 drop clearly registers). Component 2's `AdaptiveConformal
  Threshold.decide()` is a one-sided "flag if score is high" gate, which
  structurally cannot catch this failure mode -- but Component 1's
  two-sided residual detectors can (a systematic *negative* residual is
  exactly what their down-side branch is for). This is why Component 2's
  demo in `main.py` uses `misspecified_gradual_stream` (an additive,
  score-*raising* bias) instead: it's the right shape of failure for a
  one-sided trigger-decision threshold, and `masked_channel_stream` is a
  genuine, useful stress test for Component 1 specifically, not a
  redundant scenario. Both components existing independently isn't
  incidental -- this is a concrete case where the residual-based detector
  catches something the threshold-based one structurally can't.
- **Adaptive threshold's recall vs. the fixed baseline (main.py's
  `detection_efficiency_vs_fixed_threshold` output).** In the default run,
  the adaptive ACI threshold shows *lower* recall than the naive frozen
  threshold on the misspecified-gradual scenario, despite both being
  well-calibrated on background (ACI's empirical miscoverage tracks its
  0.02 target closely). Plausible explanation, not yet fully isolated: ACI's
  sliding calibration buffer absorbs some of the pre-onset portion of the
  live test stream (which has its own sampling variability relative to the
  original burn-in set) into its quantile estimate, which can push its
  threshold slightly higher than the frozen burn-in-only quantile. Worth
  a dedicated ablation (fix the buffer to burn-in only vs. let it slide)
  before drawing conclusions for the writeup -- flagged here rather than
  quietly resolved by picking whichever run looked better.
- **`n_jet` as the multiplicity covariate.** `residual.py`'s calibration
  regression and `masked_channel_stream`/`multiplicity_step_stream` all
  use jet count as *the* multiplicity signal. Real AXOL1TL/CICADA-style
  monitoring would likely track multiple object multiplicities (jets,
  muons, electrons) jointly -- simplified to one for this build; the
  regression's design matrix (`residual.py::_design_matrix`) is the place
  to extend this.
- **BOCPD underperforms CUSUM on continuous, non-resetting drift
  (`drift_sim/radiation_damage.py`, `main.py`'s space case study).** Not a
  bug -- a real, reportable finding. On the permanent monotonic gain-decay
  scenario, CUSUM detects reliably within budget; BOCPD frequently misses
  or fires very late. Plausible mechanism: BOCPD's Normal-Inverse-Gamma
  model assumes roughly-constant parameters within a "run" -- continuous
  drift can get absorbed into an inflating variance estimate rather than
  triggering a run-length reset, while CUSUM's fixed-slope accumulation
  has no equivalent escape hatch. Stated as plausible, not proven; worth a
  dedicated ablation (does forcing a tighter prior on beta0 change this?)
  before it's a strong claim in a paper.
- **BOCPD and KSWIN fail the project's own throughput bar
  (`src/benchmark.py`).** CUSUM/Page-Hinkley/ADWIN run at single-digit
  microseconds/event; BOCPD is ~2 orders of magnitude slower (vectorized
  NIG updates over many active run-length hypotheses), KSWIN ~3 orders of
  magnitude slower (a fresh KS-test every event, see its own docstring).
  If "millions of events per second" is a claim the paper makes, these
  two detectors don't meet it as implemented -- report this plainly.

None of the above were caught by writing the tests first and hoping --
they were caught by running `main.py` end-to-end, noticing numbers that
didn't match textbook theory (CUSUM ARL0 for k=0.5,h=10 should be ~22,000,
not ~20), and tracing each one down before packaging this up.

## What's stubbed / needs real data to validate properly

- `real_object_stream()` (stream_loader.py) needs a real file list (see
  `data/README.md`) and a network path to CERN's XRootD redirector --
  untested here, adapted from Project A's working implementation.
- The proxy VAE's feature set (jet1/jet2 kinematics, MET, HT, n_jet,
  n_muon, n_electron) is a reasonable AXOL1TL-object analog but hasn't
  been validated against a real anomaly-detection baseline -- see
  `docs/ADVISOR_NOTES.md` for the scope discussion this rests on.
- Detector hyperparameters (CUSUM's `h`, Page-Hinkley's `lam`, BOCPD's
  `hazard_lambda`, KSWIN's `alpha`) are tuned against this project's
  *synthetic* event-count scale (~1500-3000 events/run). Re-tune against
  real Zero-Bias background before trusting ARL numbers for the actual
  writeup -- `evaluation.average_run_length()` is built for exactly this.
