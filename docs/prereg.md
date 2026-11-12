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
