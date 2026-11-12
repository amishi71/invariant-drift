"""headline_table.py -- selected-k table restricted to k values with a valid false-alarm rate.
   python3 scripts/headline_table.py [results/l1_validity]"""
import glob, json, math, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/l1_validity"
TARGET, KEY = 0.05, "0.15"
norm = lambda s: re.sub(r"[^a-z0-9.]+", " ", str(s).lower())


def wilson(a, n, z=1.96):
    p = a / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kind_of(fam):
    f = norm(fam)
    if "two" not in f:
        return None
    if "raw" in f or "log" in f:
        return "raw"
    return "resid" if "resid" in f else None


for exp_dir in sorted(glob.glob(ROOT + "/*")):
    exp = exp_dir.rstrip("/").split("/")[-1]
    if exp.endswith("_old"):
        continue
    files = sorted(glob.glob(exp_dir + "/*seed*.json"))
    if not files:
        continue
    acc = {}
    for p in files:
        d = json.load(open(p))
        for r in d["records"]:
            kind = kind_of(r["family"])
            if kind is None or abs(r["target"] - TARGET) > 1e-9:
                continue
            a = acc.setdefault((kind, r["mode"], round(r["k"], 4)),
                               {"seeds": 0, "ok": 0, "va": 0, "vn": 0, "td": 0, "tn": 0, "ev": {}, "s15": []})
            a["seeds"] += 1
            if r.get("cal_far") is None:
                continue
            a["ok"] += 1
            a["va"] += r.get("val_alarms", 0); a["vn"] += r.get("val_n", 0)
            for v in r.get("tune", {}).values():
                a["td"] += v["detect"]; a["tn"] += v["n"]
            for name, v in r.get("scenarios", {}).items():
                e = a["ev"].setdefault(name, [0, 0]); e[0] += v["detect"]; e[1] += v["n"]
                if KEY in name:
                    a["s15"].append(100.0 * v["detect"] / max(v["n"], 1))
    print("\n== %s" % exp)
    for kind in ("raw", "resid"):
        for mode in ("three_way", "span"):
            ks = {k: a for (kd, m, k), a in acc.items() if kd == kind and m == mode}
            if not ks:
                continue
            elig, excl = [], []
            for k, a in sorted(ks.items()):
                if a["ok"] < a["seeds"]:
                    excl.append("%g(unach %d/%d)" % (k, a["seeds"] - a["ok"], a["seeds"])); continue
                if a["vn"] == 0:
                    excl.append("%g(no val)" % k); continue
                lo, hi = wilson(a["va"], a["vn"])
                if lo > TARGET:
                    excl.append("%g(FAR %.0f%%)" % (k, 100.0 * a["va"] / a["vn"])); continue
                elig.append((a["td"] / max(a["tn"], 1), k, a, lo, hi))
            tag = "%-5s %-9s" % (kind, mode)
            if not elig:
                print("  %s no k with a valid FAR | excluded: %s" % (tag, ", ".join(excl))); continue
            _, k, a, lo, hi = max(elig, key=lambda t: (t[0], t[1]))
            ev15 = [v for n, v in a["ev"].items() if KEY in n]
            det15 = 100.0 * sum(v[0] for v in ev15) / max(sum(v[1] for v in ev15), 1)
            mn = 100.0 * min(v[0] / max(v[1], 1) for v in a["ev"].values())
            print("  %s k=%-5g FAR %4.1f%% [%4.1f,%4.1f]  drop0.15 %3.0f%% (per seed %s)  worst scenario %3.0f%% | excluded: %s"
                  % (tag, k, 100.0 * a["va"] / a["vn"], 100 * lo, 100 * hi, det15,
                     "/".join("%.0f" % x for x in a["s15"]), mn, ", ".join(excl) or "none"))
