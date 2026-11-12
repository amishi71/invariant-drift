"""
scripts/matched_far_heldout.py

Batch 2 centrepiece. Decides whether residual monitoring detects a masked-channel
(or radiation-damage) failure once every detector is held to the SAME false-alarm
budget, measured on held-out windows.

Protocol (per seed; T = --window events):
  1. The pool is cut into --n-regions contiguous regions; seed s uses region
     s % n-regions. Different seeds therefore use DIFFERENT, non-overlapping data
     and a different burn-in location (the first events of its own region). Keep
     len(--seeds) <= --n-regions for fully independent seeds.
  2. Burn-in blocks at the start of the region (same events reserved in both modes):
       insample    VAE, regression and reference all on one block (original behaviour)
       three_way   VAE | regression | reference on three disjoint blocks
  3. Randomly chosen, disjoint T-event windows from the rest of the region:
       calibration windows  (drift-free)  choose each detector's threshold so that
                                          P(alarm within T events) <= target FAR
       validation windows   (drift-free)  measure the ACHIEVED false-alarm rate at
                                          that threshold, on windows never used to
                                          choose it
       evaluation windows   (drift injected from event 0) measure detection
                                          probability and delay from a freshly
                                          started detector (fresh start at onset)
  4. Every detector/variant is compared at every target FAR.

Scenarios: masked-channel (multiplicative drop of jet-pT/HT/MET) and radiation
damage (gain(i) = exp(-i / tau)), both applied from the first event of each
evaluation window. The pre-onset false-alarm rate (the 'null') is the validation
false-alarm rate; detection is read against it. No detector is ever scored on a
mixture of false alarms and detections.

Detectors: CUSUM variants (two-sided / down-only / up-only, on the calibration
residual and on the raw log-score with NO regression) at several slack values k;
optionally ADWIN (--include-adwin) and BOCPD (--include-bocpd; slow).

Caveats to carry into the write-up:
  - Direction-aware ("down-only") variants use oracle knowledge of the failure
    direction; "up-only" is the stream analogue of a one-sided high-score gate.
  - Reading detection power at the best k per detector selects on the evaluation
    windows; the table reports every k so the selection is visible.
  - With n windows, an achieved FAR has a binomial uncertainty (~ +/-0.03 at 0.05
    for 300 windows): compare to the target within the printed 95% interval.

Usage:
    python3 scripts/matched_far_heldout.py --seeds 0 1 2
    python3 scripts/matched_far_heldout.py --source synthetic --seeds 0 1 2   # control
    python3 scripts/matched_far_heldout.py --include-adwin
    python3 scripts/matched_far_heldout.py --aggregate-only --seeds 0 1 2

Forward-in-time protocol (calibrate on the past, monitor the future; needs a region that holds both):
    python3 scripts/matched_far_heldout.py --source real --cache-path data/real_cache/zb396102_ordered.npz \\
        --n-regions 2 --design pu_spline --time-split forward --n-tune 150 --tag l1_396102_forward \\
        --inject-set l1_calo --requantize 0.5 --out-dir results/l1_forward
Cross-run protocol (VAE, regression and reference from run A; every window from run B):
    python3 scripts/matched_far_heldout.py --source real --cache-path data/real_cache/zb398183_ordered.npz \\
        --calib-cache data/real_cache/zb396102_ordered.npz --n-regions 4 --design pu_spline --n-tune 150 \\
        --tag l1_cross_396102_to_398183 --inject-set l1_calo --requantize 0.5 --out-dir results/l1_cross

Choose k on separate tuning windows (adds a 'k selected on tuning' table):
    python3 scripts/matched_far_heldout.py --n-tune 200 --out-dir results/matched_far_heldout_tuned

Covariate-model experiment (one --out-dir per design so results never mix):
    python3 scripts/matched_far_heldout.py --design pu_spline --modes insample three_way span \\
        --out-dir results/matched_far_heldout_pu_spline
Mode "span" fits the regression and the reference on random events spread across the
region, taken from slots that belong to no monitoring window (non-causal; report it as
an offline-calibration variant).
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calibration_split import MODES, build_split_calibration, residualize, reserved_events
from src.residual_models import DESIGNS
from src.matched_far import (
    RealPool, SyntheticPool, adwin_first_alarm, bocpd_young_prob_paths, calibrate_param,
    cusum_runmax, first_passage, outcome_counts,
    take_windows, wilson, zscore,
)

from src.injection_sets import (INJECTION_SETS, inject_masked_cols, inject_radiation_cols,
                                resolve_injection_cols)

CUSUM_VARIANTS = [
    ("CUSUM two-sided, residual", "resid", "two"),
    ("CUSUM down-only, residual (fixed direction)", "resid", "down"),
    ("CUSUM up-only, residual (fixed direction)", "resid", "up"),
    ("CUSUM two-sided, raw log-score (no regression)", "logscore", "two"),
    ("CUSUM down-only, raw log-score (fixed direction)", "logscore", "down"),
]


def scenario_list(args):
    return ([(f"masked drop={d:g}", "masked", d) for d in args.drops]
            + [(f"radiation tau={t:g}", "radiation", t) for t in args.taus])


def score_windows(sc, fw, pw, jw):
    n, T, nf = fw.shape
    scores, resid = residualize(sc, fw.reshape(-1, nf), pw.reshape(-1), jw.reshape(-1))
    return {"resid": resid.reshape(n, T),
            "logscore": np.log(np.clip(scores.reshape(n, T), 1e-12, None))}


def make_record(mode, family, k, target, params, fp, T, scen_names, extra=None, tune_names=()):
    j, far_cal = calibrate_param(fp["cal"], T, target)
    rec = {"mode": mode, "family": family, "k": k, "target": target}
    if extra:
        rec.update(extra)
    if j is None:
        rec.update(param=None, cal_far=None, val_alarms=None, val_n=None, scenarios={})
        return rec
    a, n, _ = outcome_counts(fp["val"], j, T)
    scen = {}
    for name in scen_names:
        d, nn, delays = outcome_counts(fp[name], j, T)
        scen[name] = {"detect": d, "n": nn, "delays": delays}
    tune = {}
    for name in tune_names:  # separate drift-injected windows used ONLY to choose k
        d, nn, _ = outcome_counts(fp["tune:" + name], j, T)
        tune[name] = {"detect": d, "n": nn}
    rec.update(param=float(params[j]), cal_far=float(far_cal[j]),
               val_alarms=a, val_n=n, scenarios=scen, tune=tune)
    return rec


def place_windows(args, rng, n_slots):
    """Slot indices (relative to the first monitorable slot) for cal / val / eval / tune windows, plus the
    slots left free (source for mode "span").

    time_split "none"   : the original placement, one random permutation of all slots (so cal/val/eval are
                          IDENTICAL to earlier runs). Calibration, validation and evaluation windows are
                          interleaved in time.
    time_split "forward": slots before the boundary hold the calibration windows (they set h); validation
                          (false-alarm check), evaluation and tuning windows come from slots AFTER it. That is
                          the deployment order: set thresholds on the past, monitor the future."""
    need = args.n_cal + args.n_val + args.n_eval
    if args.time_split == "none":
        perm = rng.permutation(n_slots)
        cal = perm[:args.n_cal]
        val = perm[args.n_cal:args.n_cal + args.n_val]
        ev = perm[args.n_cal + args.n_val:need]
        tune = perm[need:need + args.n_tune]
        free = perm[need + args.n_tune:]
        return cal, val, ev, tune, free
    boundary = int(args.forward_boundary * n_slots)
    late_need = args.n_val + args.n_eval + args.n_tune
    if boundary < args.n_cal or n_slots - boundary < late_need:
        raise SystemExit(
            f"--time-split forward with boundary {args.forward_boundary:g} leaves {boundary} early slots "
            f"(need {args.n_cal} calibration windows) and {n_slots - boundary} late slots (need {late_need} "
            f"validation+evaluation+tuning windows). Use fewer windows, a larger region (smaller --n-regions) "
            f"or a different --forward-boundary.")
    early = rng.permutation(boundary)
    late = boundary + rng.permutation(n_slots - boundary)
    cal = early[:args.n_cal]
    val = late[:args.n_val]
    ev = late[args.n_val:args.n_val + args.n_eval]
    tune = late[args.n_val + args.n_eval:late_need]
    free = np.concatenate([early[args.n_cal:], late[late_need:]])
    return cal, val, ev, tune, free


def run_seed(pool, seed, args, calib_pool=None):
    t0 = time.time()
    T = args.window
    F, P, J = pool.region(seed % args.n_regions, args.n_regions)
    inj_cols = resolve_injection_cols(args.inject_set, None if pool.name == "synthetic" else args.cache_path)
    print(f"  injection set {args.inject_set!r} -> columns {tuple(inj_cols)}, requantize {args.requantize}", flush=True)
    if calib_pool is not None:
        # Cross-run: VAE, regression and reference come from ANOTHER run; every window comes from this one.
        CF, CP, CJ = calib_pool.region(args.calib_region, args.calib_n_regions)
        reserve = 0                      # nothing in this region is spent on calibration blocks
        reserve_c = reserved_events(args.n_vae, args.n_reg, args.n_ref)
    else:
        CF, CP, CJ = F, P, J
        reserve = reserved_events(args.n_vae, args.n_reg, args.n_ref)
        reserve_c = reserve
    n_slots = (len(F) - reserve) // T
    need = args.n_cal + args.n_val + args.n_eval
    if n_slots < need + args.n_tune:
        raise SystemExit(
            f"Region has {n_slots} usable {T}-event slots but {need} are needed "
            f"(n_cal+n_val+n_eval+n_tune). Reduce those, reduce --n-regions, or use a larger pool.")
    rng = np.random.default_rng(1000 + seed)
    cal_sl, val_sl, eval_sl, tune_sl, free_slots = place_windows(args, rng, n_slots)
    cal_s, val_s, eval_s = (reserve + cal_sl * T, reserve + val_sl * T, reserve + eval_sl * T)
    tune_s = reserve + tune_sl * T
    win = {name: (take_windows(F, s, T), take_windows(P, s, T), take_windows(J, s, T))
           for name, s in (("cal", cal_s), ("val", val_s), ("eval", eval_s))}
    if args.strict_transfer:
        if calib_pool is None:
            raise SystemExit("--strict-transfer needs --calib-cache")
        n_slots_c = (len(CF) - reserve_c) // T
        if n_slots_c < args.n_cal:
            raise SystemExit(f"calibration region has {n_slots_c} slots after burn-in, need {args.n_cal}")
        cs = reserve_c + np.random.default_rng(3000 + seed).permutation(n_slots_c)[:args.n_cal] * T
        win["cal"] = (take_windows(CF, cs, T), take_windows(CP, cs, T), take_windows(CJ, cs, T))

    scen = scenario_list(args)
    scen_names = [s[0] for s in scen]
    tune_names = scen_names if args.n_tune > 0 else []
    if args.n_tune > 0:
        win["tune"] = (take_windows(F, tune_s, T), take_windows(P, tune_s, T), take_windows(J, tune_s, T))
    h_grid = np.arange(args.h_min, args.h_max + 1e-9, args.h_step)
    records, mode_info = [], {}

    span_idx = None
    if "span" in args.modes:
        n_span = args.n_reg + args.n_ref
        if calib_pool is not None:
            pool_idx = np.arange(reserve_c, len(CF))      # any event of the calibration run past its blocks
        else:
            if len(free_slots) * T < n_span:
                raise SystemExit(f"mode 'span' needs {n_span} events from slots outside every window, "
                                 f"but only {len(free_slots) * T} exist. Reduce --n-cal/--n-val/--n-eval.")
            pool_idx = (reserve + free_slots[:, None] * T + np.arange(T)[None, :]).ravel()
        pick = np.random.default_rng(2000 + seed).choice(len(pool_idx), n_span, replace=False)
        span_idx = (pool_idx[pick[:args.n_reg]], pool_idx[pick[args.n_reg:]])

    for mode in args.modes:
        print(f"  [seed {seed} | {mode} | design {args.design}] training VAE and building "
              f"calibration...", flush=True)
        sc = build_split_calibration(
            CF, CP, CJ, mode=mode, n_vae=args.n_vae, n_reg=args.n_reg, n_ref=args.n_ref,
            seed=seed, vae_epochs=args.vae_epochs, verbose=False,
            design=args.design, n_knots=args.n_knots,
            span_idx=span_idx if mode == "span" else None)
        series = {"cal": score_windows(sc, *win["cal"]), "val": score_windows(sc, *win["val"])}
        ef, ep, ej = win["eval"]
        for name, kind, val in scen:
            inj = (inject_masked_cols(ef, val, inj_cols, args.requantize) if kind == "masked"
                   else inject_radiation_cols(ef, val, inj_cols, requantize=args.requantize))
            series[name] = score_windows(sc, inj, ep, ej)
        if args.n_tune > 0:
            tf, tp, tj = win["tune"]
            for name, kind, val in scen:
                inj = (inject_masked_cols(tf, val, inj_cols, args.requantize) if kind == "masked"
                       else inject_radiation_cols(tf, val, inj_cols, requantize=args.requantize))
                series["tune:" + name] = score_windows(sc, inj, tp, tj)
        ref = {"resid": sc.ref_residuals, "logscore": sc.ref_logscores}
        ref_sd = float(sc.ref_residuals.std())
        v = series["val"]["resid"]
        mode_info[mode] = {
            "ref_mean": float(sc.ref_residuals.mean()), "ref_std": ref_sd,
            "val_resid_mean": float(v.mean()), "val_resid_std": float(v.std()),
            "val_z_mean": float((v.mean() - sc.ref_residuals.mean()) / ref_sd),
            "val_z_std": float(v.std() / ref_sd),
            "vae_final_loss": sc.info.get("vae_final_loss"), "design": args.design}
        # Which way does each injection move the residual? Measured on the tuning windows when they exist
        # (diagnostic only; it never feeds back into any detector).
        pre = "tune:" if args.n_tune > 0 else ""
        mode_info[mode]["inject_shift_z"] = {
            name: float((series[pre + name]["resid"].mean() - sc.ref_residuals.mean()) / ref_sd)
            for name, _kind, _val in scen}
        mode_info[mode]["inject_shift_source"] = "tune" if args.n_tune > 0 else "eval"
        mi = mode_info[mode]
        print(f"      held-out validation residual vs reference: z-mean {mi['val_z_mean']:+.3f}, "
              f"z-std {mi['val_z_std']:.3f}", flush=True)
        print("      injection shift (z, + = residual rises): "
              + ", ".join(f"{n} {v:+.2f}" for n, v in mi["inject_shift_z"].items()), flush=True)

        # --- CUSUM family -----------------------------------------------------
        for family, ser, md in CUSUM_VARIANTS:
            for k in args.k_values:
                z = {s: zscore(series[s][ser], ref[ser]) for s in series}
                fp = {s: first_passage(cusum_runmax(z[s], k, md), h_grid) for s in series}
                for target in args.targets:
                    records.append(make_record(mode, family, k, target, h_grid, fp, T, scen_names,
                                               tune_names=tune_names))

        # --- ADWIN (optional) -------------------------------------------------
        if args.include_adwin:
            print(f"      ADWIN grid ({len(args.adwin_deltas)} deltas)...", flush=True)
            deltas = sorted(args.adwin_deltas, reverse=True)  # most -> least sensitive
            fp = {s: np.stack([adwin_first_alarm(series[s]["resid"], d) for d in deltas], axis=1)
                  for s in series}
            for target in args.targets:
                records.append(make_record(mode, "ADWIN on residual", None, target, deltas, fp, T, scen_names))

        # --- BOCPD (optional, slow) -------------------------------------------
        if args.include_bocpd:
            print("      BOCPD paths (slow)...", flush=True)
            thetas = np.concatenate([np.linspace(0.1, 0.9, 17), 1 - np.logspace(-1.3, -4, 12)])
            thetas = np.unique(np.round(thetas, 6))
            fp = {s: first_passage(np.maximum.accumulate(
                bocpd_young_prob_paths(series[s]["resid"], ref["resid"]), axis=1), thetas)
                for s in series}
            for target in args.targets:
                records.append(make_record(mode, "BOCPD on residual", None, target, thetas, fp, T, scen_names))

    print(f"  [seed {seed}] done in {time.time() - t0:.0f}s", flush=True)
    return {"seed": seed, "region": seed % args.n_regions, "source": pool.name,
            "n_windows": {"cal": args.n_cal, "val": args.n_val, "eval": args.n_eval, "tune": args.n_tune},
            "scenarios": scen_names, "mode_info": mode_info, "records": records,
            "protocol": {"time_split": args.time_split, "forward_boundary": args.forward_boundary,
                         "calibration_from": args.calib_cache or "same region",
                         "inject_set": args.inject_set, "inject_cols": list(inj_cols),
                         "requantize": args.requantize},
            "config": vars(args)}


# ----------------------------------------------------------------------------
# Aggregation and reporting
# ----------------------------------------------------------------------------
def fmt_rate(k, n):
    if n == 0 or k is None:
        return "   n/a"
    lo, hi = wilson(k, n)
    return f"{100*k/n:4.0f}% [{100*lo:2.0f},{100*hi:3.0f}]"


def aggregate(seed_files):
    agg = defaultdict(lambda: {"val_alarms": 0, "val_n": 0, "per_seed_far": [], "params": [],
                               "unachievable": 0, "n_seeds": 0,
                               "tune": defaultdict(lambda: {"detect": 0, "n": 0}),
                               "scen": defaultdict(
                                   lambda: {"detect": 0, "n": 0, "delays": [], "per_seed": []})})
    modeinfo = defaultdict(list)
    scen_names = None
    for path in seed_files:
        with open(path) as f:
            d = json.load(f)
        scen_names = d["scenarios"]
        for mode, mi in d["mode_info"].items():
            modeinfo[mode].append(mi)
        for r in d["records"]:
            a = agg[(r["mode"], r["family"], r["k"], r["target"])]
            a["n_seeds"] += 1
            if r["param"] is None:
                a["unachievable"] += 1
                continue
            a["params"].append(r["param"])
            a["val_alarms"] += r["val_alarms"]
            a["val_n"] += r["val_n"]
            a["per_seed_far"].append(r["val_alarms"] / r["val_n"])
            for name, t in (r.get("tune") or {}).items():
                a["tune"][name]["detect"] += t["detect"]
                a["tune"][name]["n"] += t["n"]
            for name, s in r["scenarios"].items():
                b = a["scen"][name]
                b["detect"] += s["detect"]
                b["n"] += s["n"]
                b["delays"] += s["delays"]
                b["per_seed"].append(s["detect"] / s["n"])
    return agg, modeinfo, scen_names


def print_report(agg, modeinfo, scen_names, args):
    print("\n" + "=" * 100)
    print("HELD-OUT OFFSET/SCALE OF THE RESIDUAL RELATIVE TO THE DETECTOR REFERENCE "
          "(validation windows, mean over seeds)")
    for mode, lst in modeinfo.items():
        zm = np.mean([m["val_z_mean"] for m in lst])
        zs = np.mean([m["val_z_std"] for m in lst])
        print(f"  {mode:<10} z-mean {zm:+.3f}   z-std {zs:.3f}   (seeds: {len(lst)})  "
              f"[ideal: 0 / 1]")

    modes = list(modeinfo)
    keys = sorted({(k[1], k[2]) for k in agg}, key=lambda t: (t[0], -1 if t[1] is None else t[1]))
    print("\n" + "=" * 100)
    print("ACHIEVED HELD-OUT FALSE-ALARM RATE vs TARGET  (P(alarm within "
          f"{args.window} events), pooled over seeds; windows never used for calibration)")
    head = f"{'detector':<52}{'k':>5} {'mode':<10}" + "".join(f"{'target '+format(t,'g'):>20}" for t in args.targets)
    print(head)
    for fam, k in keys:
        for mode in modes:
            row = f"{fam:<52}{'-' if k is None else format(k, 'g'):>5} {mode:<10}"
            for t in args.targets:
                a = agg.get((mode, fam, k, t))
                if a is None or a["val_n"] == 0:
                    row += f"{'unachievable':>20}"
                else:
                    row += f"{fmt_rate(a['val_alarms'], a['val_n']):>20}"
            print(row)

    pt = args.print_target
    print("\n" + "=" * 100)
    print(f"DETECTION PROBABILITY WITHIN {args.window} EVENTS AT TARGET FALSE-ALARM RATE {pt:g} "
          f"(pooled over seeds; detection % [95% CI] / median delay in events)")
    print(f"{'detector':<52}{'k':>5} {'mode':<10}" + "".join(f"{s:>26}" for s in scen_names))
    for fam, k in keys:
        for mode in modes:
            a = agg.get((mode, fam, k, pt))
            row = f"{fam:<52}{'-' if k is None else format(k, 'g'):>5} {mode:<10}"
            for s in scen_names:
                if a is None or a["val_n"] == 0 or s not in a["scen"]:
                    row += f"{'n/a':>26}"
                    continue
                b = a["scen"][s]
                med = f"{np.median(b['delays']):.0f}" if b["delays"] else "-"
                row += f"{fmt_rate(b['detect'], b['n']) + ' / ' + med:>26}"
            print(row)
    print("\nRead detection against the achieved FAR above, not the target alone. A detector whose "
          "detection interval overlaps its own false-alarm rate (the null) has not been shown to detect. "
          "The bracketed Wilson intervals treat windows as independent; window means are correlated and "
          "regions of one run are not independent seeds, so they are optimistic (see the seed ranges in the "
          "selected-k table).")


def print_direction_check(modeinfo):
    """Which way each injection moves the calibration residual (mean over seeds, z units). The 'up-only' and
    'down-only' detectors have a FIXED direction; this table says which one points the right way for this
    dataset and injection. The sign is not a property of the detector: it was negative for the JetHT masked
    drop and is positive for the L1 one."""
    rows = [(mode, lst) for mode, lst in modeinfo.items() if lst and "inject_shift_z" in lst[0]]
    if not rows:
        return
    print("\n" + "=" * 100)
    print("DIRECTION CHECK: mean shift of the residual under each injection (z units, mean over seeds)")
    names = list(rows[0][1][0]["inject_shift_z"])
    print(f"{'mode':<10}" + "".join(f"{n:>36}" for n in names))
    for mode, lst in rows:
        cells = []
        for n in names:
            v = float(np.mean([m["inject_shift_z"][n] for m in lst]))
            cells.append(f"{v:+.2f} ({'up-only' if v > 0 else 'down-only'} is right)")
        print(f"{mode:<10}" + "".join(f"{c:>36}" for c in cells))
    print("A fixed-direction detector pointing the other way is expected to detect nothing; compare "
          "one-sided rows against this table, and treat the two-sided detector as the deployable headline.")


def print_selected_k(agg, modeinfo, scen_names, args):
    """One k per (mode, detector family), chosen on the TUNING windows only (mean detection
    over scenarios, pooled over seeds), then reported on the EVALUATION windows. The 'best k on
    evaluation' column shows how much the old, visible-selection reading would have flattered."""
    pt = args.print_target
    print("\n" + "=" * 100)
    print(f"k SELECTED ON SEPARATE TUNING WINDOWS, REPORTED ON EVALUATION WINDOWS "
          f"(target FAR {pt:g}, window {args.window}, {args.n_tune} tuning windows per seed)")
    print(f"{'detector':<52}{'mode':<10}{'k(tune)':>8}{'tune mean':>10}"
          + "".join(f"{s:>22}" for s in scen_names)
          + f"{'eval mean':>10}{'eval min':>9}{'seed range of mean':>20}{'best-k eval mean':>18}")
    fams = sorted({(k[1]) for k in agg if k[1].startswith("CUSUM")})
    for fam in fams:
        for mode in modeinfo:
            cands = []
            dropped = []
            for (m, f, k, t), a in agg.items():
                if (m, f, t) != (mode, fam, pt) or k is None:
                    continue
                if a["val_n"] == 0 or a["unachievable"]:
                    dropped.append((k, a["unachievable"], a["n_seeds"]))
                    continue
                if not a["tune"] or any(a["tune"][s]["n"] == 0 for s in scen_names):
                    continue
                tune_mean = np.mean([a["tune"][s]["detect"] / a["tune"][s]["n"] for s in scen_names])
                eval_mean = np.mean([a["scen"][s]["detect"] / a["scen"][s]["n"] for s in scen_names])
                cands.append((tune_mean, -k, k, eval_mean, a))
            if dropped:
                print("  [" + fam + " | " + mode + "] k excluded from selection (calibration unachievable on some seed): "
                      + ", ".join(f"k={kk:g} ({u}/{n} seeds)" for kk, u, n in sorted(dropped)))
            if not cands:
                print(f"{fam:<52}{mode:<10}{'n/a (calibration unachievable for every k)':>60}")
                continue
            tm, _, k_sel, em, a = max(cands, key=lambda c: (c[0], c[1]))
            cells = "".join(f"{fmt_rate(a['scen'][s]['detect'], a['scen'][s]['n']):>22}" for s in scen_names)
            best_eval = max(c[3] for c in cands)
            worst = min(a["scen"][s]["detect"] / a["scen"][s]["n"] for s in scen_names)
            ps = [a["scen"][s]["per_seed"] for s in scen_names]
            if all(len(x) == len(ps[0]) for x in ps) and len(ps[0]) > 1:
                per_seed_mean = np.mean(np.array(ps), axis=0)
                rng_txt = f"{100*per_seed_mean.min():.0f}-{100*per_seed_mean.max():.0f}% ({len(ps[0])} seeds)"
            else:
                rng_txt = "n/a"
            print(f"{fam:<52}{mode:<10}{k_sel:>8g}{100*tm:>9.0f}%{cells}{100*em:>9.0f}%{100*worst:>8.0f}%"
                  f"{rng_txt:>20}{100*best_eval:>17.0f}%")
    print("\n'eval mean' averages over scenarios and is inflated when several scenarios saturate near 100%; "
          "'eval min' is the worst scenario and 'seed range' is the spread of the per-seed means (the more "
          "honest uncertainty: the bracketed Wilson intervals treat windows as independent, but window "
          "means are correlated, so they are optimistic). If 'best-k eval mean' is much higher than "
          "'eval mean', the per-k table would have overstated power by picking k on the evaluation windows.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["real", "synthetic"], default="real")
    p.add_argument("--cache-path", default="data/real_cache/jetht_features.npz")
    p.add_argument("--synthetic-region-events", type=int, default=2_200_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-regions", type=int, default=3)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=["insample", "three_way"])
    p.add_argument("--design", choices=list(DESIGNS), default="quad",
                   help="covariate model for the calibration regression (src/residual_models.py); "
                        "'quad' is the original")
    p.add_argument("--n-knots", type=int, default=8, help="interior pileup knots for spline designs")
    p.add_argument("--n-vae", type=int, default=20_000)
    p.add_argument("--n-reg", type=int, default=10_000)
    p.add_argument("--n-ref", type=int, default=10_000)
    p.add_argument("--window", type=int, default=2500)
    p.add_argument("--n-cal", type=int, default=300)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-tune", type=int, default=0,
                   help="drift-injected tuning windows per seed, used only to choose k (0 = off). "
                        "Taken from slots no other window uses; cal/val/eval are unchanged.")
    p.add_argument("--time-split", choices=["none", "forward"], default="none",
                   help="none = windows interleaved in time (original); forward = calibration windows come "
                        "from slots BEFORE --forward-boundary, validation/evaluation/tuning windows after it")
    p.add_argument("--forward-boundary", type=float, default=0.5,
                   help="fraction of the region (in time) before which calibration windows are drawn")
    p.add_argument("--calib-cache", default=None,
                   help="CROSS-RUN: take the VAE/regression/reference from this cache (another run) and all "
                        "windows from --cache-path")
    p.add_argument("--calib-region", type=int, default=0)
    p.add_argument("--strict-transfer", action="store_true",
                   help="CROSS-RUN with no healthy data from the monitored run: cal windows (which set h) come "
                        "from the CALIBRATION run after its burn-in blocks; val/eval/tune stay on the monitored run")
    p.add_argument("--calib-n-regions", type=int, default=4,
                   help="split of the calibration cache; only the start of region --calib-region is used")
    p.add_argument("--inject-set", choices=["auto"] + sorted(INJECTION_SETS), default="auto",
                   help="columns the injections scale (src/injection_sets.py). auto = jetht for JetHT caches, "
                        "l1_legacy (4, 8, 9) for L1 caches; l1_calo scales every calorimeter energy together")
    p.add_argument("--requantize", type=float, default=0.0,
                   help="round injected columns back onto this lattice (0.5 for the L1 caches; 0 = off)")
    p.add_argument("--tag", default=None,
                   help="prefix of the per-seed result files (default: the --source value). Use a distinct tag "
                        "per dataset, e.g. l1_396102, so results are never mixed up")
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--k-values", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--targets", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--print-target", type=float, default=0.05)
    p.add_argument("--drops", type=float, nargs="+", default=[0.15, 0.4])
    p.add_argument("--taus", type=float, nargs="+", default=[400.0, 3000.0])
    p.add_argument("--h-min", type=float, default=1.0)
    p.add_argument("--h-max", type=float, default=80.0)
    p.add_argument("--h-step", type=float, default=0.25)
    p.add_argument("--include-adwin", action="store_true")
    p.add_argument("--adwin-deltas", type=float, nargs="+",
                   default=[0.1, 0.03, 0.01, 0.003, 1e-3, 3e-4, 1e-4, 1e-5])
    p.add_argument("--include-bocpd", action="store_true")
    p.add_argument("--out-dir", default="results/matched_far_heldout")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if args.print_target not in args.targets:
        args.targets = sorted(set(args.targets) | {args.print_target})
    if len(args.seeds) > args.n_regions and not args.aggregate_only:
        print(f"WARNING: {len(args.seeds)} seeds but only {args.n_regions} regions; some seeds "
              f"will reuse the same data region (not independent). Raise --n-regions.")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.source
    if args.calib_cache and args.source != "real":
        raise SystemExit("--calib-cache needs --source real (the cache of the monitored run)")
    if not args.aggregate_only:
        pool = RealPool(args.cache_path) if args.source == "real" else SyntheticPool(args.synthetic_region_events)
        calib_pool = RealPool(args.calib_cache) if args.calib_cache else None
        print(f"Pool: {pool.name}. Seeds {args.seeds}; window {args.window}; "
              f"cal/val/eval windows {args.n_cal}/{args.n_val}/{args.n_eval}; modes {args.modes}")
        for seed in args.seeds:
            path = os.path.join(args.out_dir, f"{tag}_seed{seed}.json")
            if args.skip_existing and os.path.exists(path):
                print(f"seed {seed}: {path} exists, skipping")
                continue
            print(f"seed {seed}:", flush=True)
            res = run_seed(pool, seed, args, calib_pool)
            with open(path, "w") as f:
                json.dump(res, f, default=str)
            print(f"  wrote {path}")

    files = [os.path.join(args.out_dir, f"{tag}_seed{s}.json") for s in args.seeds]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("No per-seed result files found to aggregate.")
    agg, modeinfo, scen_names = aggregate(files)
    print(f"\nAggregating {len(files)} seed file(s) from {args.out_dir}")
    print_report(agg, modeinfo, scen_names, args)
    print_direction_check(modeinfo)
    if any(a["tune"] for a in agg.values()):
        print_selected_k(agg, modeinfo, scen_names, args)


if __name__ == "__main__":
    main()
