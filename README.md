# Invariant Drift

**Calibration-residual monitoring for learned anomaly triggers**

A learned anomaly trigger can go silently miscalibrated as beam conditions, pileup, or detector state shift underneath it, with nothing flagging that it happened until it's been mis-triggering for a while. This project is two independent, complementary ways of noticing that, evaluated on CMS/LHC Open Data:

1. **Sequential change-point detection** on a scalar calibration residual
   (anomaly score regressed against pileup/multiplicity and expected
   luminosity trends) — CUSUM, Page-Hinkley, and BOCPD implemented here,
   compared against ADWIN and KSWIN from `river`.
2. **Adaptive conformal inference (ACI)** for online threshold recalibration
   from Zero-Bias control feedback plus delayed offline validation, with
   online false discovery rate control (LORD, SAFFRON) over windowed batches
   of the resulting decisions.

A proxy VAE trained on background-only events stands in for the real
trigger's anomaly score, since the actual trigger model isn't accessible
here. It builds on
[cms-streaming-shift-detection](https://github.com/amishi71/cms-streaming-shift-detection),
this project's predecessor on dijet-mass resonance-shift detection, reusing
its frozen-reference CUSUM/Page-Hinkley logic and XRootD streaming
infrastructure.

## Latest results

The pipeline started synthetic-only and has since been validated end-to-end
on real CMS Open Data (8.8M cached JetHT events, record 30558, provenance
checked against CERN's own checksums). Most synthetic findings hold up on
real data; the real false-alarm rate came in higher than synthetic
predicted, and that gap has been narrowed and diagnosed but not fully
closed. Full breakdown, tables, and root-cause analysis:
**[docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md)**.

At a matched false-alarm rate, ADWIN detects drift 17% faster than CUSUM
while also missing fewer events (6-seed synthetic comparison against
Page-Hinkley/ADWIN/KSWIN):
**[docs/BASELINE_COMPARISON.md](docs/BASELINE_COMPARISON.md)**.

## Quickstart

```
pip install -r requirements.txt
python main.py                          # small defaults, ~20-30s on CPU, synthetic data
python main.py --n-burn-in 5000 --n-events 3000 --n-trials 15 --vae-epochs 80
pytest tests/ -v
```

`main.py`'s default pipeline (burn-in, all four drift scenarios, the
radiation-damage case study, throughput benchmark) runs on synthetic data
and is the fast, dependency-light path for iterating on detector logic.

Real-data validation:
```
python3 scripts/real_data_sweep.py --seed 0
python3 scripts/aggregate_real_sweep.py results/real_seeds/*.json
```
See [docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) for the
full multi-seed reproduction commands, provenance verification, and the
`(k, h)` retune / root-cause scripts.

## Layout

```
src/
  proxy_vae.py           AXOL1TL-proxy VAE on offline jet/MET/HT/multiplicity features
  residual.py             calibration residual (score vs pileup/mult/lumi regression)
  stream_loader.py        real CMS Open Data (uproot/XRootD) + synthetic dev/test stream
  real_pipeline.py        real-data equivalent of main.py's build_calibration/score_stream
  kinematics.py           streaming mean/variance/skew/kurtosis (copied from Project A)
  detectors/
    cusum.py, page_hinkley.py, bocpd.py, adwin.py, kswin.py
  conformal/
    aci.py                adaptive conformal threshold
    fdr.py                 LORD, SAFFRON, windowed_batch_pvalues
  drift_sim/
    gradual.py             pileup evolving across a fill (nominal + misspecified)
    abrupt.py               masked readout channels / multiplicity step
    radiation_damage.py    permanent monotonic gain-decay (space-detector analog)
    real_data_injection.py real-data equivalents of the above, applied to cached
                            real feature arrays instead of the synthetic generator
  evaluation.py            ARL, latency, false-alarm rate, coverage, online FDR
main.py                    end-to-end orchestration on synthetic data; `python main.py --help`
scripts/                    real-data sweeps, retuning, provenance checks, ablations
                            (see docs/REAL_DATA_VALIDATION.md for what each one does)
tests/                      pytest suite, 58 tests
docs/
  FINDINGS.md               bugs found and design decisions made during development
  REAL_DATA_VALIDATION.md   full real-data validation results and reproduction steps
  BASELINE_COMPARISON.md    matched-false-alarm-rate comparison vs. Page-Hinkley/ADWIN/KSWIN
```

## Results summary

| Check                                                   | Result                                                                              |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| ADWIN vs. CUSUM at matched ~5% FAR (synthetic, 6 seeds) | ADWIN 17% faster to detection (264 vs. 318 events), lower miss rate (1.3% vs. 3.7%) |
| CUSUM vs. BOCPD, radiation-damage drift                 | 0.0% vs. 58.7% miss rate (real + synthetic)                                         |
| Masked-channel, default CUSUM (real)                    | 70.0% ± 16.3% miss rate                                                             |
| False-alarm rate, real vs. synthetic                    | 66.7% vs. ~17-33% — narrowed via retune, not fully closed                           |
| Adaptive (ACI) vs. fixed threshold recall               | ACI trades recall for calibration guarantees, by design — see FINDINGS.md           |

Full tables and methodology: [docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) and [docs/BASELINE_COMPARISON.md](docs/BASELINE_COMPARISON.md).

## Known limitations

- `real_object_stream()`'s live-XRootD/EOS path is untested against a live
  endpoint — all real-data work here uses a locally cached feature pool.
- Real-data validation draws from a single Open Data record (one narrow
  run range) — generalization across run periods is untested.
- The gradual-drift scenarios remain synthetic-only (a single Open Data
  record has no meaningful luminosity trend to perturb).
- The real-data false-alarm rate gap is diagnosed (heavy-tailed residuals)
  but not yet fixed — see docs/REAL_DATA_VALIDATION.md.

For the full list of bugs found and design decisions made while building
this, see [docs/FINDINGS.md](docs/FINDINGS.md).