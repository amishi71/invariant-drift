#!/usr/bin/env bash
# run_l1_validity.sh -- the L1 validity matrix: do the held-out results survive realistic injections,
# forward-in-time calibration and cross-run calibration?
#
#   bash scripts/run_l1_validity.sh            run every experiment (finished seeds are skipped), then summarise
#   bash scripts/run_l1_validity.sh summary    reprint the key tables from the logs, run nothing
#   bash scripts/run_l1_validity.sh run forward_396102 cross_A_to_B     only the named experiments
#   DRY=1 bash scripts/run_l1_validity.sh      print the commands, run nothing
#
# Run from the repo root with the venv active, after applying 0009-l1-validity.patch.
# Each experiment writes results/l1_validity/<name>/ and results/logs/l1v_<name>.log.
# Long runs:  nohup bash scripts/run_l1_validity.sh > results/logs/l1_validity.out 2>&1 &

set -u
MODE="${1:-run}"
shift 2>/dev/null || true
ONLY="$*"
A=data/real_cache/zb396102_ordered.npz
B=data/real_cache/zb398183_ordered.npz
OUT=results/l1_validity_h400
LOG=results/logs_h400
mkdir -p "$OUT" "$LOG"

COMMON="--source real --design pu_spline --modes three_way span --n-tune 150 --k-values 0.05 0.1 0.25 0.5 --h-max 400 --skip-existing"
CALO="--inject-set l1_calo --requantize 0.5"

EXPERIMENTS="repro_396102 repro_398183 legacy_rq_396102 calo_norq_396102 calo_rq_396102 calo_rq_398183 forward_396102 forward_398183 cross_A_to_B cross_B_to_A"

args_for() {
  case "$1" in
    # same protocol and injection as the earlier runs: must reproduce your previous numbers
    repro_396102)     echo "--cache-path $A --n-regions 4 --seeds 0 1 2 3" ;;
    repro_398183)     echo "--cache-path $B --n-regions 4 --seeds 0 1 2 3" ;;
    # injection realism, one change at a time (interleaved windows)
    legacy_rq_396102) echo "--cache-path $A --n-regions 4 --seeds 0 1 2 3 --inject-set l1_legacy --requantize 0.5" ;;
    calo_norq_396102) echo "--cache-path $A --n-regions 4 --seeds 0 1 2 3 --inject-set l1_calo" ;;
    calo_rq_396102)   echo "--cache-path $A --n-regions 4 --seeds 0 1 2 3 $CALO" ;;
    calo_rq_398183)   echo "--cache-path $B --n-regions 4 --seeds 0 1 2 3 $CALO" ;;
    # calibrate on the first half of a run, monitor the second half (two halves = two seeds)
    forward_396102)   echo "--cache-path $A --n-regions 2 --seeds 0 1 --time-split forward $CALO" ;;
    forward_398183)   echo "--cache-path $B --n-regions 2 --seeds 0 1 --time-split forward $CALO" ;;
    # VAE, regression and reference from one run; every window from the other
    cross_A_to_B)     echo "--cache-path $B --calib-cache $A --n-regions 4 --seeds 0 1 2 3 $CALO" ;;
    cross_B_to_A)     echo "--cache-path $A --calib-cache $B --n-regions 4 --seeds 0 1 2 3 $CALO" ;;
  esac
}

wanted() { [ -z "$ONLY" ] && return 0; for o in $ONLY; do [ "$o" = "$1" ] && return 0; done; return 1; }

run_one() {
  name="$1"
  cmd="python3 scripts/matched_far_heldout.py $COMMON --tag $name --out-dir $OUT/$name $(args_for "$name")"
  echo "[run] $name"; echo "      $cmd"
  [ "${DRY:-0}" = "1" ] && return
  t0=$(date +%s)
  bash -c "$cmd" > "$LOG/l1v_$name.log" 2>&1
  rc=$?
  echo "[$( [ $rc -eq 0 ] && echo ok || echo FAIL )] $name ($(( $(date +%s) - t0 ))s, exit $rc)"
  [ $rc -ne 0 ] && tail -n 6 "$LOG/l1v_$name.log" | sed 's/^/      | /'
}

summary() {
  echo; echo "================ L1 validity summary ================"
  for name in $EXPERIMENTS; do
    wanted "$name" || continue
    f="$LOG/l1v_$name.log"; [ -s "$f" ] || continue
    echo; echo "---- $name ----"
    grep -E "^(three_way|span) +[-+]" "$f" | head -4
    n=$(grep -n "SELECTED ON SEPARATE" "$f" | head -1 | cut -d: -f1)
    if [ -n "$n" ]; then
      sed -n "$((n+1)),\$p" "$f" | grep -E "^detector|CUSUM two-sided, residual" | cut -c1-260
    fi
    echo "  achieved FAR @0.05 (two-sided residual):"
    grep -E "^CUSUM two-sided, residual" "$f" | head -8 | cut -c1-120
  done
  echo
  echo "Read this against docs/L1_EVALUATION_PROTOCOLS.md. 'repro_*' must match your earlier numbers;"
  echo "calo_* are the realistic injections; forward_* and cross_* are the deployability tests."
}

case "$MODE" in
  summary) summary ;;
  run) for e in $EXPERIMENTS; do wanted "$e" && run_one "$e"; done; summary ;;
  *) echo "usage: bash scripts/run_l1_validity.sh [run|summary] [experiment ...]"; exit 1 ;;
esac
