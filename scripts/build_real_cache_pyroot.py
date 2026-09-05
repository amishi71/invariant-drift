"""
scripts/build_real_cache_pyroot.py

PyROOT/RDataFrame-native rebuild of the feature extraction currently done
via uproot in src/stream_loader.py's real_object_stream() /
_compute_object_features() / _compute_pileup(). Reproduces the IDENTICAL
feature definitions and column order (see DEFAULT_FEATURE_NAMES in
src/proxy_vae.py and the column_stack order in _compute_object_features):

    jet1_pt, jet1_eta, jet1_phi, jet2_pt, jet2_eta, jet2_phi,
    n_jet, met_pt, met_phi, ht, n_muon, n_electron

Selection: nJet >= 2 (same as _compute_object_features's mask).
HT: sum of Jet_pt for jets with pt >= min_jet_pt (same as the uproot
version's `ak.sum(pt[pt >= min_jet_pt], axis=1)`).

Output: an .npz with the SAME keys/shapes as data/real_cache/
jetht_features.npz (features, pileup, n_jet), so nothing downstream
(src/real_pipeline.py, src/drift_sim/real_data_injection.py, the sweep
scripts) needs to change -- only the extraction backend differs.

IMPORTANT: run this in the ROOT-enabled venv (venv-root), NOT the main
project venv (venv) -- see the version-mismatch note from setup. The
main venv never needs to import ROOT; it only ever reads the .npz this
script produces.

Usage (from repo root, with venv-root active and thisroot.sh sourced):
    python3 scripts/build_real_cache_pyroot.py
    python3 scripts/build_real_cache_pyroot.py \\
        --file-list data/jetht_files_multi.txt \\
        --out data/real_cache/jetht_features_pyroot.npz
"""
import argparse
import os

import numpy as np
import ROOT


DEFAULT_FEATURE_NAMES = [
    "jet1_pt", "jet1_eta", "jet1_phi", "jet2_pt", "jet2_eta", "jet2_phi",
    "n_jet", "met_pt", "met_phi", "ht", "n_muon", "n_electron",
]


def load_file_list(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_dataframe(file_list: list, min_jet_pt: float) -> "ROOT.RDataFrame":
    """Constructs the RDataFrame with the same event selection and derived
    columns as _compute_object_features/_compute_pileup, using RDataFrame's
    native RVec indexing/Sum instead of awkward array operations.
    """
    chain = ROOT.TChain("Events")
    for fpath in file_list:
        chain.Add(fpath)

    df = ROOT.RDataFrame(chain)
    df = df.Filter("nJet >= 2", "at least 2 jets, matches uproot mask")
    df = df.Define("jet1_pt", "Jet_pt[0]")
    df = df.Define("jet1_eta", "Jet_eta[0]")
    df = df.Define("jet1_phi", "Jet_phi[0]")
    df = df.Define("jet2_pt", "Jet_pt[1]")
    df = df.Define("jet2_eta", "Jet_eta[1]")
    df = df.Define("jet2_phi", "Jet_phi[1]")
    # HT: sum of jet pt above threshold, matching
    # ak.sum(pt[pt >= min_jet_pt], axis=1) in _compute_object_features.
    df = df.Define("ht", f"Sum(Jet_pt[Jet_pt >= {min_jet_pt}f])")
    return df


def extract_arrays(df) -> dict:
    """AsNumpy pulls everything in one pass -- RDataFrame's lazy execution
    means the Filter/Define chain above only actually runs here.
    """
    columns = DEFAULT_FEATURE_NAMES[:6] + ["nJet", "MET_pt", "MET_phi", "ht",
                                            "nMuon", "nElectron", "PV_npvsGood"]
    # Rename to match: n_jet<-nJet, met_pt<-MET_pt, met_phi<-MET_phi,
    # n_muon<-nMuon, n_electron<-nElectron -- AsNumpy returns the RDF
    # column names verbatim, so map them back to DEFAULT_FEATURE_NAMES
    # ourselves rather than renaming columns in RDataFrame itself.
    result = df.AsNumpy(columns=columns)
    return result


def assemble(result: dict) -> tuple:
    """Builds (features, pileup, n_jet) in the exact shape/order
    load_real_feature_pool()/real_pipeline.py expect.
    """
    features = np.column_stack([
        result["jet1_pt"], result["jet1_eta"], result["jet1_phi"],
        result["jet2_pt"], result["jet2_eta"], result["jet2_phi"],
        result["nJet"], result["MET_pt"], result["MET_phi"],
        result["ht"], result["nMuon"], result["nElectron"],
    ]).astype(np.float64)
    pileup = np.asarray(result["PV_npvsGood"], dtype=np.float64)
    n_jet = np.asarray(result["nJet"], dtype=np.float64)
    return features, pileup, n_jet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", type=str,
                         default="data/jetht_files_multi.txt")
    parser.add_argument("--min-jet-pt", type=float, default=20.0,
                         help="Matches src/stream_loader.py's real_object_stream default.")
    parser.add_argument("--out", type=str,
                         default="data/real_cache/jetht_features_pyroot.npz")
    args = parser.parse_args()

    # NOTE: implicit MT deliberately NOT enabled -- RDataFrame's AsNumpy row
    # order is not guaranteed to match sequential file/event order under IMT,
    # which breaks row-by-row validation against the uproot-derived cache.
    # ROOT.ROOT.EnableImplicitMT()

    print(f"[1/3] Loading file list from {args.file_list}...")
    file_list = load_file_list(args.file_list)
    print(f"      {len(file_list)} files")

    print(f"[2/3] Building RDataFrame (nJet>=2 filter, min_jet_pt={args.min_jet_pt})...")
    df = build_dataframe(file_list, args.min_jet_pt)
    n_before = df.Count()  # lazy; triggers the read below

    print("      Running RDataFrame event loop (this reads every file once)...")
    result = extract_arrays(df)
    features, pileup, n_jet = assemble(result)

    print(f"[3/3] Extracted {len(features)} events "
          f"(RDataFrame reported {n_before.GetValue()} passing the nJet>=2 filter)")
    print(f"      features shape: {features.shape}")
    print(f"      feature means:  {dict(zip(DEFAULT_FEATURE_NAMES, features.mean(axis=0).round(3)))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, features=features, pileup=pileup, n_jet=n_jet)
    print(f"\nDone. Written to {args.out}")
    print("\nNext: run scripts/validate_pyroot_cache.py (in the MAIN venv, "
          "not venv-root) to compare this against the existing uproot-derived "
          "cache before treating it as a drop-in replacement.")


if __name__ == "__main__":
    main()
