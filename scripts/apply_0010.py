#!/usr/bin/env python3
"""0010 part 1: honest k-selection reporting and run log. Run from the repo root."""
import py_compile, re, shutil, sys, tempfile
from pathlib import Path

p = Path("scripts/matched_far_heldout.py")
s = p.read_text()

def once(pattern):
    m = list(re.finditer(pattern, s))
    if len(m) != 1:
        sys.exit(f"ABORT: anchor matched {len(m)} times (need 1): {pattern[:70]!r}")
    return m[0]

m1 = once(r'(?P<ind>[ ]+)inj_cols = resolve_injection_cols\(args\.inject_set, [^\n]*\n')
m2 = once(r'(?P<ind>[ ]+)cands = \[\]\n')
m3 = once(r'(?P<ind>[ ]+)if \(m, f, t\) != \(mode, fam, pt\) or k is None or a\["val_n"\] == 0 or\s*a\["unachievable"\]:\n[ ]+continue\n')
m4 = once(r'(?P<ind>[ ]+)if not cands:\n')

ind = m1.group("ind")
ins1 = (f'{ind}print(f"  injection set {{args.inject_set!r}} -> columns {{tuple(inj_cols)}}, '
        f'requantize {{args.requantize}}", flush=True)\n')
i3 = m3.group("ind")
new3 = (f'{i3}if (m, f, t) != (mode, fam, pt) or k is None:\n{i3}    continue\n'
        f'{i3}if a["val_n"] == 0 or a["unachievable"]:\n'
        f'{i3}    dropped.append((k, a["unachievable"], a["n_seeds"]))\n{i3}    continue\n')
i4 = m4.group("ind")
ins4 = (f'{i4}if dropped:\n'
        f'{i4}    print("  [" + fam + " | " + mode + "] k excluded from selection (calibration unachievable on some seed): "\n'
        f'{i4}          + ", ".join(f"k={{kk:g}} ({{u}}/{{n}} seeds)" for kk, u, n in sorted(dropped)))\n')

edits = [(m1.end(), m1.end(), ins1),
         (m2.end(), m2.end(), f'{m2.group("ind")}dropped = []\n'),
         (m3.start(), m3.end(), new3),
         (m4.start(), m4.start(), ins4)]
out = s
for a, b, txt in sorted(edits, reverse=True):
    out = out[:a] + txt + out[b:]

tmp = Path(tempfile.mkdtemp()) / "x.py"
tmp.write_text(out)
py_compile.compile(str(tmp), doraise=True)
shutil.copy(p, str(p) + ".orig")
p.write_text(out)
print("patched", p, "(backup:", str(p) + ".orig)")
