"""
scripts/calibrated_threshold_multiseed.py

Runs calibrated_threshold_experiment.py across multiple seeds (same
convention as real_data_sweep.py's --seed sweep) to confirm the
n_burn_in>=20000 calibration finding isn't a seed=0 fluke before it goes
into docs/REAL_DATA_VALIDATION.md as a settled result.

Retrains the proxy VAE from scratch per seed (same as every other
multi-seed script in this project) -- this is NOT fast. At n_burn_in=20000
and n_arl_windows=300 per seed, expect this to take a while across 6 seeds;
consider running it in the background (see usage note below) rather than
waiting on it in the foreground.

Each seed's full results (all three target_far values) are written to
their own file, matching aggregate_real_sweep.py's glob-based aggregation
pattern already used elsewhere in this project.

Usage:
    python3 scripts/calibrated_threshold_multiseed.py
    python3 scripts/calibrated_threshold_multiseed.py --seeds 0 1 2 3 4 5

    # Run in the background so a closed terminal/laptop sleep doesn't kill it:
    nohup python3 scripts/calibrated_threshold_multiseed.py > results/multiseed_run.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--n-burn-in", type=int, default=20000,
                         help="Fixed at the value the seed=0 investigation "
                              "found necessary for reliable calibration.")
    parser.add_argument("--n-arl-windows", type=int, default=300)
    parser.add_argument("--h-step", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=1,
                         help="Kept at 1 (default) -- block resampling was "
                              "tested at seed=0 and made calibration worse, "
                              "not better; see docs/REAL_DATA_VALIDATION.md.")
    parser.add_argument("--target-far", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--out-dir", type=str, default="results/calibrated_multiseed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        out_path = os.path.join(args.out_dir, f"seed{seed}.json")
        print(f"\n{'='*60}")
        print(f"Seed {seed} -> {out_path}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "scripts/calibrated_threshold_experiment.py",
            "--target-far", *[str(t) for t in args.target_far],
            "--n-arl-windows", str(args.n_arl_windows),
            "--n-burn-in", str(args.n_burn_in),
            "--h-step", str(args.h_step),
            "--block-size", str(args.block_size),
            "--seed", str(seed),
            "--out", out_path,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: seed {seed} exited with code {result.returncode}, "
                  f"continuing with remaining seeds")

    print(f"\nAll seeds done. Aggregate with:")
    print(f"  python3 scripts/aggregate_calibrated_threshold.py {args.out_dir}/seed*.json")


if __name__ == "__main__":
    main()
