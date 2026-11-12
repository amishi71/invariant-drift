#!/usr/bin/env bash
# final_run.sh -- one pass over everything finished in this project.
#   bash scripts/final_run.sh show     reprint results from existing outputs (about 1-2 min)
#   bash scripts/final_run.sh full     rerun anything missing, rerun the diagnostics, then show
set -u
MODE="${1:-show}"
LOG=results/logs/final; mkdir -p "$LOG" results
export HF_HOME="${HF_HOME:-$HOME/Desktop/Developer/hf_cache}"
A=data/real_cache/zb396102_ordered.npz
B=data/real_cache/zb398183_ordered.npz
hdr() { echo; echo "=================== $1 ==================="; }
tuned_block() {  # k chosen on separate tuning windows, reported on evaluation windows
  f="$1"; [ -s "$f" ] || { echo "(missing $f)"; return; }
  sed -n '/SELECTED ON SEPARATE/,$p' "$f" | grep -E "^detector|two-sided" | cut -c1-235
}

hdr "1. STATE OF THE REPO"
echo "branch: $(git branch --show-current)"; git --no-pager log --oneline -14; git status --short | grep -v '^??' || true

hdr "2. TESTS"
python3 -m pytest -q 2>&1 | tail -n 3

hdr "3. L1 ORDERED CACHES (chronological?)"
for f in $A $B; do
  [ -e "$f" ] || { echo "missing $f"; continue; }
  python3 l1_cache_inspect.py "$f" | grep -E "^data/|features: [0-9]|distinct values|expected runs|mean pileup|monotone"
done

hdr "4. BASELINE (synthetic, out-of-sample FAR, 6 seeds)"
python3 scripts/aggregate_baseline_oos.py results/matched_far_multiseed_oos/seed*.json

if [ "$MODE" = "full" ]; then
  hdr "5a. RERUN: JetHT held-out, k chosen on tuning windows (skips finished seeds)"
  python3 scripts/matched_far_heldout.py --n-tune 200 --include-adwin --skip-existing \
    --out-dir results/matched_far_heldout_tuned > "$LOG/heldout_jetht_tuned.log" 2>&1 && echo ok || echo FAILED
  hdr "5b. RERUN: L1 held-out with tuned k, both runs"
  for f in zb396102 zb398183; do
    python3 scripts/matched_far_heldout.py --source real --cache-path data/real_cache/${f}_ordered.npz \
      --n-regions 4 --seeds 0 1 2 3 --design pu_spline --modes three_way span --n-tune 200 \
      --k-values 0.05 0.1 0.25 0.5 --skip-existing \
      --out-dir results/matched_far_heldout_l1_${f}_tuned > "$LOG/heldout_l1_${f}_tuned.log" 2>&1 && echo "ok $f" || echo "FAILED $f"
  done
  hdr "5c. RERUN: L1 validity matrix (10 experiments) + forward cross-run B to A"
  bash scripts/run_l1_validity.sh run > "$LOG/l1_validity_run.out" 2>&1 && echo ok || echo FAILED
  python3 scripts/matched_far_heldout.py --source real --design pu_spline --modes three_way span \
    --n-tune 150 --k-values 0.05 0.1 0.25 0.5 --skip-existing \
    --cache-path $A --calib-cache $B --n-regions 2 --seeds 0 1 --time-split forward \
    --inject-set l1_calo --requantize 0.5 --tag cross_fwd_B_to_A \
    --out-dir results/l1_validity/cross_fwd_B_to_A > "$LOG/l1v_cross_fwd_B_to_A.log" 2>&1 && echo "ok cross_fwd_B_to_A" || echo "FAILED cross_fwd_B_to_A"
  hdr "5d. RERUN: covariate diagnostics (copies outputs aside, restores the tracked PNG)"
  python3 scripts/covariate_model_diagnostic.py --plot > "$LOG/diag_jetht.log" 2>&1
  cp results/covariate_model_diagnostic.json results/final_diag_jetht.json
  cp results/covariate_model_diagnostic.png results/final_diag_jetht.png
  for r in 0 1 2 3; do
    python3 scripts/covariate_model_diagnostic.py --source real --cache-path $A --n-regions 4 --region $r --plot > "$LOG/diag_l1_396102_r$r.log" 2>&1
    cp results/covariate_model_diagnostic.json results/final_diag_l1_396102_r$r.json
    cp results/covariate_model_diagnostic.png results/final_diag_l1_396102_r$r.png
  done
  git checkout results/covariate_model_diagnostic.png 2>/dev/null || true
fi

hdr "6. COVARIATE DIAGNOSTIC VERDICTS"
for f in "$LOG"/diag_*.log; do [ -s "$f" ] && { echo "-- $(basename $f)"; grep -E "^Pool real|^Baseline|^Lowest|^VERDICT" "$f"; }; done
[ -z "$(ls $LOG/diag_*.log 2>/dev/null)" ] && echo "(run 'full' to regenerate; earlier results are in results/covariate_model_diagnostic.json and results/l1_covariate_diagnostic_region0.png)"

hdr "7. HELD-OUT, k CHOSEN ON TUNING WINDOWS: JetHT"
tuned_block results/logs/heldout_jetht_tuned.log
hdr "7. HELD-OUT, k CHOSEN ON TUNING WINDOWS: L1 run 396102"
tuned_block results/logs/heldout_l1_zb396102_tuned.log
hdr "7. HELD-OUT, k CHOSEN ON TUNING WINDOWS: L1 run 398183"
tuned_block results/logs/heldout_l1_zb398183_tuned.log

hdr "8. L1 VALIDITY MATRIX (repro, injections, forward, cross-run)"
bash scripts/run_l1_validity.sh summary
echo; echo "-- forward cross-run (model from the other run, thresholds from the first half)"
for n in cross_fwd_A_to_B cross_fwd_B_to_A; do echo "== $n"; tuned_block results/logs/l1v_$n.log; done

hdr "9. FILES"
ls results/l1_validity | tr '\n' ' '; echo
echo "Done: $(date)"
