"""
scripts/aggregate_baseline_oos.py

Aggregates matched_far_baseline_comparison.py output that includes the out-of-sample fields
(val_far, censored latency, CUSUM k sweep). The older aggregate_matched_far_baseline.py still
works on the same files for the four original columns; this one adds what it cannot show.

Usage:
    python3 scripts/aggregate_baseline_oos.py results/matched_far_multiseed/seed*.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.matched_far import wilson

ORDER = ["CUSUM", "Page-Hinkley", "ADWIN", "KSWIN"]


def ms(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (float(np.mean(x)), float(np.std(x))) if x else (float("nan"), float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    paths = sorted({q for pat in args.files for q in glob.glob(pat)})
    if not paths:
        raise SystemExit("No files matched.")
    data = [json.load(open(q)) for q in paths]
    print(f"Aggregating {len(paths)} seed file(s)\n")

    det = defaultdict(lambda: defaultdict(list))
    for d in data:
        for name, r in d["results"].items():
            det[name]["far"].append(r["achieved_far"])
            det[name]["miss"].append(r["miss_rate"])
            det[name]["lat"].append(r["latency_mean"])
            det[name]["cens"].append(r.get("latency_censored_mean"))
            if "val_alarms" in r:
                det[name]["va"].append(r["val_alarms"])
                det[name]["vn"].append(r["val_n"])

    print(f"{'detector':<14}{'FAR (search)':>16}{'FAR (fresh, pooled)':>26}{'miss rate':>16}"
          f"{'latency (detected)':>22}{'latency (censored)':>22}")
    print("-" * 116)
    for name in [n for n in ORDER if n in det] + [n for n in det if n not in ORDER]:
        v = det[name]
        fm, fs = ms(v["far"]); mm, msd = ms(v["miss"]); lm, ls = ms(v["lat"]); cm, cs = ms(v["cens"])
        if v["vn"]:
            a, n = sum(v["va"]), sum(v["vn"])
            lo, hi = wilson(a, n)
            val = f"{a / n:.3f} [{lo:.3f},{hi:.3f}]"
        else:
            val = "n/a"
        print(f"{name:<14}{fm:8.3f}+/-{fs:<6.3f}{val:>26}{mm:8.3f}+/-{msd:<6.3f}"
              f"{lm:11.1f}+/-{ls:<8.1f}{cm:11.1f}+/-{cs:<8.1f}")
    print("\n'FAR (search)' is in-sample: the threshold was searched on those windows. 'FAR (fresh)' is on "
          "windows no threshold was chosen on; it is the number to quote. 'latency (censored)' counts a "
          "missed trial as the full remaining stream, so a detector is not rewarded for missing hard trials.")

    ck = [d["cusum_k"] for d in data if d.get("cusum_k")]
    if ck:
        print("\nCUSUM reference value k (h matched to the same target FAR; k chosen on separate tuning trials)")
        ks = list(ck[0]["sweep"].keys())
        print(f"{'k':>6}{'h (mean)':>10}{'FAR fresh':>12}{'eval miss':>12}{'eval latency':>16}{'eval censored':>16}{'tune censored':>16}")
        for k in ks:
            rows = [c["sweep"][k] for c in ck if k in c["sweep"]]
            hm = np.mean([r["h"] for r in rows])
            va = [r.get("val_alarms") for r in rows if "val_alarms" in r]
            vn = [r.get("val_n") for r in rows if "val_n" in r]
            vf = f"{sum(va) / sum(vn):.3f}" if vn else "n/a"
            em = np.mean([r["eval"]["miss_rate"] for r in rows])
            el = ms([r["eval"]["latency_mean"] for r in rows])
            ec = ms([r["eval"]["latency_censored_mean"] for r in rows])
            tc = ms([r["tune"]["latency_censored_mean"] for r in rows])
            print(f"{k:>6}{hm:10.2f}{vf:>12}{em:12.3f}{el[0]:9.1f}+/-{el[1]:<5.1f}{ec[0]:9.1f}+/-{ec[1]:<5.1f}{tc[0]:9.1f}+/-{tc[1]:<5.1f}")
        sel = [c["selected_k"] for c in ck]
        print("k selected per seed:", ", ".join(f"{s:g}" for s in sel))

        pairs = []
        for d in data:
            if d.get("cusum_k") and "ADWIN" in d["results"] and "latency_censored_mean" in d["results"]["ADWIN"]:
                c = d["cusum_k"]
                pairs.append((d["results"]["ADWIN"]["latency_censored_mean"],
                              c["sweep"][f"{c['selected_k']:g}"]["eval"]["latency_censored_mean"]))
        if pairs:
            diff = np.array([a - c for a, c in pairs])
            print(f"\nADWIN minus CUSUM(tuned k), censored latency per seed: "
                  + ", ".join(f"{x:+.1f}" for x in diff)
                  + f"   mean {diff.mean():+.1f}   (positive = ADWIN slower; ADWIN slower in "
                  f"{int((diff > 0).sum())}/{len(diff)} seeds)")


if __name__ == "__main__":
    main()
