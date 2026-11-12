import sys, numpy as np
for p in sys.argv[1:]:
    with np.load(p) as z:
        lumi = z["lumi"]
    u, c = np.unique(lumi, return_counts=True)
    print(f"{p}: {len(u)} lumisections ({u.min()}..{u.max()}, gaps: {int((np.diff(u) > 1).sum())}); events/LS min {c.min()} median {int(np.median(c))} max {c.max()}; LS with <5000 events: {int((c < 5000).sum())}")
