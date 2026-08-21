# Abstract

**Author:** Amishi Agrawal

Learned anomaly triggers such as CMS AXOL1TL and CICADA are calibrated
against a fixed background model, but there is no standard way to check in
real time whether that calibration still holds as detector conditions
change. I present two components that address this from different angles.

The first is a sequential change-point detector that monitors trigger
calibration health online. Instead of watching the output threshold alone,
it tracks the anomaly score jointly with its main driving covariates
(pileup, object multiplicity) through a scalar residual calibrated against
expected luminosity trends. I implemented CUSUM and Bayesian Online
Change-Point Detection (BOCPD) for this and compared them against standard
drift detectors (ADWIN, Page-Hinkley, KSWIN).

The second is an adaptive conformal inference (ACI) layer that recalibrates
decision thresholds on the fly using miscoverage feedback from Zero-Bias
control data and delayed offline validation. This matters because standard
conformal prediction assumes exchangeable data, which does not hold at the
LHC due to beam current decay and pileup drift. Since triggers process
millions of events per second, single-event error control is not
sufficient on its own, so I added online false discovery rate control
(LORD, SAFFRON) evaluated over windowed batches.

I evaluated both components on public CMS and LHC Open Data, using
simulated gradual drift (pileup evolving across a fill) and abrupt shifts
(subdetector conditions changing, e.g. masked readout channels). For the
change-point detector, I report Average Run Length, detection latency, and
false-alarm rate; for the conformal layer, I report empirical coverage,
online FDR, and detection efficiency against a fixed-threshold baseline. I
also examine robustness when the assumed drift model is misspecified.

Together, these provide a model-agnostic way to detect when a trigger's
background assumptions break down, and to correct for it without
retraining.

---

See [`README.md`](../README.md) for how each piece of this maps to code,
and [`ADVISOR_NOTES.md`](ADVISOR_NOTES.md) for the AXOL1TL-proxy scope
decision this build rests on (CICADA's raw L1-calorimeter-image inputs
aren't in public CMS Open Data; AXOL1TL's object-level inputs have a
usable offline analog, which is what `src/proxy_vae.py` targets).
