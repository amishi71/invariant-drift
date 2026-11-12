#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p results/logs_h400 results/ls_fine results/ls_seeds results/ls_swap results/harm_seeds
python3 scripts/ls_stability.py --drops 0.05 0.075 0.09 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.4 --out-dir results/ls_fine > results/logs_h400/ls_fine.log 2>&1
for s in 1 2 3 4 5 6 7 8 9; do
  python3 scripts/ls_stability.py --seed $s --out-dir results/ls_seeds/s$s > results/logs_h400/ls_seed$s.log 2>&1
done
for s in 0 1 2 3 4; do
  python3 scripts/ls_stability.py --seed $s --monitor-cache data/real_cache/zb398183_ordered.npz --calib-cache data/real_cache/zb396102_ordered.npz --out-dir results/ls_swap/s$s > results/logs_h400/ls_swap$s.log 2>&1
done
for s in $(seq 0 19); do
  python3 scripts/harm_map.py --seed $s --out-dir results/harm_seeds/s$s > results/logs_h400/harm_seed$s.log 2>&1
done
echo done > results/logs_h400/robustness_done.txt
