"""Real-data touchpoint: fit residual.py's calibration regression on a
real (not synthetic) NanoAOD file, and report whether the burn-in
residual looks standardized (mean~0, std~1) the way it does on synthetic
data. This is deliberately NOT the full pipeline (no proxy VAE training,
no detectors, no drift injection) -- it answers exactly one question:
does the calibration regression's core assumption (a fittable, roughly
symmetric relationship between anomaly score and pileup/multiplicity/
luminosity) hold on real CMS data, or does something about real data
break an assumption that only looked fine on synthetic data.

Since there's no real "anomaly score" model trained on real data yet
(the proxy VAE here was trained on synthetic burn-in), this script uses
a simple physically-motivated stand-in score (HT, the scalar sum of jet
pT) purely to check the REGRESSION MACHINERY and residual behavior
against real covariates -- not to claim HT itself is a good anomaly
score. That's a real, useful, narrower claim than "the whole pipeline
works on real data," and it's the right scope for a single-file
touchpoint.

Run: python scripts/real_data_touchpoint.py data/jetht_files.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.stream_loader import real_object_stream, DEFAULT_FEATURE_NAMES
from src.residual import CalibrationModel


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/real_data_touchpoint.py <file_list_path>")
        print("  e.g.: python scripts/real_data_touchpoint.py data/jetht_files.txt")
        sys.exit(1)

    file_list_path = sys.argv[1]
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print(f"[1/3] Streaming up to {n_events} real events from {file_list_path}...")
    ht_idx = DEFAULT_FEATURE_NAMES.index("ht")
    scores, pileup, n_jet = [], [], []
    for i, event in enumerate(real_object_stream(file_list_path=file_list_path)):
        if i >= n_events:
            break
        scores.append(event["features"][ht_idx])  # HT as the stand-in score, see module docstring
        pileup.append(event["pileup"])
        n_jet.append(event["features"][6])  # n_jet is index 6 in DEFAULT_FEATURE_NAMES
    scores, pileup, n_jet = np.array(scores), np.array(pileup), np.array(n_jet)
    lumi = np.ones_like(pileup)  # no luminosity-trend covariate available from a single file

    print(f"      Got {len(scores)} real events. "
          f"HT: mean={scores.mean():.2f} std={scores.std():.2f}  "
          f"pileup (PV_npvsGood): mean={pileup.mean():.2f} std={pileup.std():.2f}")

    print("[2/3] Fitting CalibrationModel on real data...")
    calib = CalibrationModel.fit(scores, pileup, n_jet, lumi)
    residual = calib.residual(scores, pileup, n_jet, lumi)

    print("[3/3] Real-data residual check:")
    print(f"      mean={residual.mean():.4f}  std={residual.std():.4f}  "
          f"(target: ~0, ~1 -- same as synthetic burn-in)")
    from scipy.stats import skew, shapiro
    print(f"      skew={skew(residual):.4f}  "
          f"(compare to synthetic: ~-0.4 after log-transform, see README)")
    if len(residual) <= 5000:
        print(f"      Shapiro-Wilk normality p-value={shapiro(residual).pvalue:.6f} "
              f"(informational only -- large n makes this over-sensitive)")

    print("\nInterpretation: mean/std close to 0/1 and skew in a similar "
          "ballpark to the synthetic case means the regression machinery "
          "and log-transform assumption hold up on real data. A large "
          "mean offset or wildly different skew would mean something about "
          "real HT/pileup/n_jet's relationship differs from what the "
          "synthetic generator assumed -- worth digging into before "
          "trusting the full pipeline on real data.")


if __name__ == "__main__":
    main()