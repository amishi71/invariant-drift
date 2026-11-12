# Pre-registration (written 2026-11-12, before baseline results exist)

Monitors (same streams, same injections, matched held-out FAR 0.05 per T=2500):
 M1 raw log-score CUSUM; M2 spline-residual CUSUM;
 M3 tail-rate CUSUM on exceedance indicators at healthy quantiles q90, q99, q99.9.
Injections: l1_calo masked drop of 5, 10, 15, 25, 40%, seeds 0-3, cross-run B->A and A->B.
Protocol: calibrate on drift-free windows, verify FAR on independent windows, h-max 400.
Cells still unachievable at cap 400 are reported as detector failures, not dropped.

Hypotheses:
 A: for some injected fault, M1 or M2 has power >= 0.8 while every M3 variant has power < 0.5
    at matched FAR.
 B: the best M3 variant matches or beats M1/M2 on every fault tested.
 C: methodology results stand regardless (cap artifact, in-sample FAR understated ~1.6x).
Harm (if Phase 4 is reached): a fault is harmful if signal efficiency at a threshold recalibrated
 to a fixed 1% healthy background rate drops >10% relative, or the frozen-threshold background
 rate changes >20% relative. Signal efficiency is on simulated samples (proxy only).
No change to faults, magnitudes, FAR or thresholds after results are seen.
Deviations will be logged here with a reason.

## Deviations and additions (logged 2026-11-12, after event-scale results were seen)
- k grid for the tail-rate comparison extended to 0.01 and 0.02 after observing that tail shifts are small per event.
  The first tail run (k >= 0.05) is superseded by results/tail_h400_smallk.
- Added lumisection-scale stability/detectability (scripts/ls_stability.py) and a harm map (scripts/harm_map.py)
  after seeing event-scale results. Neither was part of the original comparison.
- The raw and residual detection numbers for the 15% and 40% drops (cross B->A) were known before this file was committed.
- The harm criterion is applied as written; its "bandwidth" flag also counts a background-rate DECREASE, which is
  not a bandwidth problem. Frozen and recalibrated signal efficiencies are reported separately.

## Robustness gates (logged 2026-11-12, BEFORE running them)
- Gate 1 (strict transfer, results/strict): "fixed reference transfers" is accepted only if achieved held-out FAR
  is <= 2x target (<= 10% at the 5% target) in BOTH directions. Otherwise it is reported as falsified.
- Gate 2 (fine signed LS grid, results/ls_fine): for every drop 5-40%, at least one statistic has |shift|/LS-SD >= 3.
  Any grid drop where a statistic is < 3 is reported as that statistic's blind spot (signed shifts read, not d').
- Gate 3 (harm map, 20 VAE seeds): "invisible harm" = |bkg-rate change| <= 5% AND frozen efficiency loss >= 20% at
  the 1% operating rate. Claimed as a finding only if it occurs in >= 70% of seeds at some drop >= 25% (per signal).
  Otherwise reported as "k of 20 VAE realisations".
