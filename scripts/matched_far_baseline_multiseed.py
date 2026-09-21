"""
scripts/matched_far_baseline_multiseed.py

Runs matched_far_baseline_comparison.py across multiple seeds to confirm
the single-seed finding (ADWIN fastest at matched FAR but nonzero miss
rate; CUSUM/Page-Hinkley equivalent and perfectly reliable; KSWIN slowest
on both axes) before it goes into a resume bullet or paper as settled.

A single seed's 2% miss rate for ADWIN/KSWIN is one trial out of 50 --
thin evidence on its own. This confirms whether that pattern holds.

Usage:
    python3 scripts/matched_far_baseline_multiseed.py
    python3 scripts/matched_far_baseline_multiseed.py --seeds 0 1 2 3 4 5

    # Run in the background:
    nohup python3 scripts/matched_far_baseline_multiseed.py > results/matched_far_multiseed_run.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--n-stable-windows", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="results/matched_far_multiseed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        out_path = os.path.join(args.out_dir, f"seed{seed}.json")
        print(f"\n{'='*60}")
        print(f"Seed {seed} -> {out_path}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "scripts/matched_far_baseline_comparison.py",
            "--target-far", str(args.target_far),
            "--n-stable-windows", str(args.n_stable_windows),
            "--n-trials", str(args.n_trials),
            "--seed", str(seed),
            "--out", out_path,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: seed {seed} exited with code {result.returncode}, "
                  f"continuing with remaining seeds")

    print(f"\nAll seeds done. Aggregate with:")
    print(f"  python3 scripts/aggregate_matched_far_baseline.py {args.out_dir}/seed*.json")


if __name__ == "__main__":
    main()
