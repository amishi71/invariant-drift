#!/usr/bin/env python3
"""Flag CUSUM threshold results that hit or sit within 10% of --h-max. Usage: python3 scripts/audit_cap.py results/l1_validity/*"""
import glob, json, sys
from collections import defaultdict
for d in sys.argv[1:]:
    files = sorted(glob.glob(d + "/*seed*.json"))
    if not files:
        continue
    unach, near, tot, hmax = defaultdict(int), defaultdict(int), defaultdict(int), None
    hs = defaultdict(list)
    for f in files:
        j = json.load(open(f)); hmax = (j.get("config") or {}).get("h_max", hmax)
        for r in j["records"]:
            if not str(r["family"]).startswith("CUSUM") or abs(r["target"] - 0.05) > 1e-9:
                continue
            key = (r["mode"], r["family"], r["k"]); tot[key] += 1
            if r["param"] is None:
                unach[key] += 1
            else:
                hs[key].append(r["param"])
                if hmax and r["param"] >= 0.9 * hmax:
                    near[key] += 1
    bad = [k for k in tot if unach[k] or near[k]]
    print(f"\n{d}  (h_max={hmax}, {len(files)} seeds): {len(bad)}/{len(tot)} CUSUM cells unachievable or within 10% of the cap")
    for k in sorted(bad, key=lambda x: (x[0], x[1], x[2])):
        print(f"   {k[0]:<10}{k[1]:<46}k={k[2]:<5g} unachievable {unach[k]}/{tot[k]}  near-cap {near[k]}  h(seeds)={[round(x,1) for x in hs[k]]}")
