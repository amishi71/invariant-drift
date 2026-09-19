"""
scripts/background_far_vs_distance.py

The validation-split threshold experiment found a flat ~2x gap between
false-alarm rate measured on windows adjacent to burn-in (validation) vs.
windows further downstream (held-out) -- at EVERY target_far tested, not
worsening toward more extreme targets like the earlier bootstrap gap did.
That flat-ratio signature doesn't look like a tail-extrapolation problem;
it looks like the two blocks themselves have different baseline alarm
rates.

This script tests the natural hypothesis directly: does spontaneous
(zero-injected-drift) false-alarm rate at a FIXED h rise as you move
further from the burn-in reference period? If so, real background is
naturally drifting away from its own "stable" reference within the pool
-- exactly the phenomenon this project exists to detect, just showing up
inside supposedly drift-free control data rather than only in the
injected scenarios.

Usage:
    python3 scripts/background_far_vs_distance.py --h 12.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.cusum import CUSUMDetector
from src import evaluation
from src.real_pipeline import build_real_calibration, score_real_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-burn-in", type=int, default=3000)
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--n-windows-per-block", type=int, default=300)
    parser.add_argument("--n-blocks", type=int, default=8,
                         help="How many successive blocks to measure, "
                              "stepping further from burn-in each time.")
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--h", type=float, required=True,
                         help="Fixed threshold to test across all blocks "
                              "(use the chosen_h from a validation-split run "
                              "you want to explain, e.g. 12.0 for target=0.02).")
    args = parser.parse_args()

    print(f"[1/2] Real burn-in: {args.n_burn_in} events, training proxy VAE...")
    model, scaler, calib, burn_in_residuals, features, pileup, n_jet = build_real_calibration(
        args.n_burn_in, args.seed, args.vae_epochs, False,
    )

    print(f"\n[2/2] Measuring spontaneous FAR at h={args.h} across "
          f"{args.n_blocks} blocks of {args.n_windows_per_block} windows each, "
          f"stepping {args.n_windows_per_block * args.n_events} events per block...\n")
    print(f"{'block':>6} {'events_from_burnin_end':>24} {'FAR':>8}")

    block_size_events = args.n_windows_per_block * args.n_events
    fars = []
    for b in range(args.n_blocks):
        block_start = args.n_burn_in + b * block_size_events
        n_alarms = 0
        for w in range(args.n_windows_per_block):
            start = block_start + w * args.n_events
            _, residuals = score_real_segment(
                model, scaler, calib, features, pileup, n_jet, start, args.n_events,
            )
            detector = CUSUMDetector(burn_in_residuals, k=args.k, h=args.h)
            result = evaluation.run_detector_on_residuals(detector, residuals)
            if result.detected:
                n_alarms += 1
        far = n_alarms / args.n_windows_per_block
        fars.append(far)
        print(f"{b:6d} {block_start - args.n_burn_in:24d} {far:8.3f}")

    print("\n=== Interpretation ===")
    print("If FAR rises roughly monotonically across blocks, background is")
    print("naturally drifting away from the burn-in reference as you move")
    print("further into the pool -- a real, reportable finding about the")
    print("'stable burn-in' assumption, not a threshold-calibration bug.")
    print("If FAR bounces around with no trend, the validation-vs-held-out")
    print("gap was likely a one-off two-block difference, not a systematic")
    print("drift-with-distance effect -- worth re-checking with a different")
    print("seed or a different pair of blocks before concluding either way.")


if __name__ == "__main__":
    main()
