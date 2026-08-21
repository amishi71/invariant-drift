"""Streaming ingestion of trigger-object features: real CMS Open Data, and a
synthetic generator for local development.

IMPORTANT -- network reality of where this file was built: the sandbox this
project was developed in only has network access to package registries
(pypi, github, npm, etc.), not to CERN's XRootD/EOS endpoints. `real_object_stream`
below is a direct adaptation of cms-streaming-shift-detection's
src/stream_loader.py (same retry/circuit-breaker discipline, same
"no silent fallback -- raise if the file list is missing" philosophy) and
should work as-is once pointed at real file lists from an environment that
*does* have EOS/XRootD access, but it has NOT been exercised against a live
XRootD endpoint here. `synthetic_object_stream` is a clearly-separate,
clearly-labeled generator (not a fallback hidden inside the real path) used
so that every other module in this project (proxy_vae, residual, detectors,
drift_sim, evaluation, main.py) has been built and tested end-to-end against
*something*. Swap `real_object_stream` in once you have local EOS access --
its output shape (a dict of named features per event) matches
`synthetic_object_stream` exactly, so nothing downstream needs to change.

Feature set (offline analog of AXOL1TL's L1-object inputs; see proxy_vae.py):
leading/subleading jet (pt, eta, phi), n_jet, MET (pt, phi), HT (scalar sum
jet pt), n_muon, n_electron. Pileup proxy: PV_npvsGood (good-vertex count),
the standard reconstructed-data stand-in for pileup when MC truth
(Pileup_nTrueInt) isn't available, which is the case for real collision
data.
"""

import os
import time
from typing import Dict, Iterator, Optional

import numpy as np

from src.proxy_vae import DEFAULT_FEATURE_NAMES

# --------------------------------------------------------------------------
# Real CMS Open Data path -- adapted from cms-streaming-shift-detection.
# Retry/circuit-breaker logic copied near-verbatim (it's generic XRootD
# robustness code, not specific to the dijet-mass observable that project
# was built around).
# --------------------------------------------------------------------------

FATJET_LIKE_BRANCHES = [
    "Jet_pt", "Jet_eta", "Jet_phi", "nJet",
    "MET_pt", "MET_phi",
    "PV_npvsGood",
    "nMuon", "nElectron",
]

_TRANSIENT_ERROR_SUBSTRINGS = (
    "timed out", "timeout", "connection reset", "socket error",
    "connection refused", "temporarily unavailable",
)
_MAX_RETRIES = 3
_RETRY_BACKOFF_SEC = 2.0
_MAX_CONSECUTIVE_FAILURES = 5


def _is_transient(exc) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_ERROR_SUBSTRINGS)


def _resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str) or os.path.exists(path_str):
        return path_str
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = os.path.join(repo_root, path_str)
    return candidate if os.path.exists(candidate) else path_str


def _load_file_list(path_str: str):
    resolved = _resolve_path(path_str)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Required file list not found at '{path_str}' (resolved: '{resolved}'). "
            f"Generate it with `cernopendata-client get-file-locations --recid <ID>` "
            f"before streaming real data. See data/README.md."
        )
    with open(resolved, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not files:
        raise ValueError(f"File list at '{resolved}' is empty.")
    return files


def _compute_object_features(batch, min_jet_pt: float = 20.0) -> np.ndarray:
    """Compute the per-event feature vector (see DEFAULT_FEATURE_NAMES)
    from a NanoAOD-style awkward batch, for events with >=2 jets."""
    import awkward as ak

    mask = batch["nJet"] >= 2
    sel = batch[mask]
    if len(sel) == 0:
        return np.empty((0, len(DEFAULT_FEATURE_NAMES)), dtype=np.float64)

    pt, eta, phi = sel["Jet_pt"], sel["Jet_eta"], sel["Jet_phi"]
    jet1_pt, jet2_pt = ak.to_numpy(pt[:, 0]), ak.to_numpy(pt[:, 1])
    jet1_eta, jet2_eta = ak.to_numpy(eta[:, 0]), ak.to_numpy(eta[:, 1])
    jet1_phi, jet2_phi = ak.to_numpy(phi[:, 0]), ak.to_numpy(phi[:, 1])
    n_jet = ak.to_numpy(sel["nJet"])
    ht = ak.to_numpy(ak.sum(pt[pt >= min_jet_pt], axis=1))
    met_pt = ak.to_numpy(sel["MET_pt"])
    met_phi = ak.to_numpy(sel["MET_phi"])
    n_muon = ak.to_numpy(sel["nMuon"])
    n_electron = ak.to_numpy(sel["nElectron"])

    return np.column_stack([
        jet1_pt, jet1_eta, jet1_phi, jet2_pt, jet2_eta, jet2_phi,
        n_jet, met_pt, met_phi, ht, n_muon, n_electron,
    ]).astype(np.float64)


def _compute_pileup(batch) -> np.ndarray:
    import awkward as ak
    mask = batch["nJet"] >= 2
    sel = batch[mask]
    return ak.to_numpy(sel["PV_npvsGood"]).astype(np.float64)


def real_object_stream(
    file_list_path: str = "data/jetht_files.txt",
    step_size: int = 10_000,
    delay_sec: float = 0.0,
    min_jet_pt: float = 20.0,
) -> Iterator[Dict[str, float]]:
    """Real CMS Open Data event stream. Yields dicts:
    {"features": np.ndarray[12], "pileup": float, "feature_names": [...]}.

    No synthetic fallback -- raises immediately if the file list is
    missing/empty, same discipline as Project A. Requires `uproot` and
    `awkward` (listed in requirements.txt) and a working XRootD/EOS network
    path, neither of which is available in the sandbox this file was
    authored in -- see module docstring.
    """
    import uproot

    files = _load_file_list(file_list_path)
    failed = 0
    consecutive_failures = 0

    for fpath in files:
        attempt = 0
        while True:
            try:
                for batch in uproot.iterate(
                    f"{fpath}:Events", FATJET_LIKE_BRANCHES,
                    step_size=step_size, library="ak",
                ):
                    feats = _compute_object_features(batch, min_jet_pt=min_jet_pt)
                    pileup = _compute_pileup(batch)
                    for row, pu in zip(feats, pileup):
                        yield {
                            "features": row,
                            "pileup": float(pu),
                            "feature_names": DEFAULT_FEATURE_NAMES,
                        }
                consecutive_failures = 0
                break
            except Exception as e:
                if "failed to close file" in str(e).lower():
                    print(f"Warning: post-read close error on {fpath}: {e}. "
                          f"File was already fully read; not retrying.")
                    consecutive_failures = 0
                    break
                attempt += 1
                if _is_transient(e) and attempt <= _MAX_RETRIES:
                    print(f"Warning: transient error on {fpath} "
                          f"(attempt {attempt}/{_MAX_RETRIES}): {e}. Retrying...")
                    time.sleep(_RETRY_BACKOFF_SEC * attempt)
                    continue
                failed += 1
                consecutive_failures += 1
                print(f"Warning: error reading {fpath}: {e}. Skipping file "
                      f"(after {attempt} attempt(s)).")
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive files failed the same way "
                        f"(latest: {e}). Looks like a redirector/connectivity outage -- "
                        f"check XRootD connectivity before retrying."
                    )
                break
        if delay_sec > 0:
            time.sleep(delay_sec)

    if files and failed == len(files):
        raise RuntimeError(
            f"All {len(files)} files in '{file_list_path}' failed to read. "
            f"Check XRootD connectivity/auth and the file list contents."
        )


# --------------------------------------------------------------------------
# Synthetic path -- new, used by main.py / tests / drift_sim in this
# sandbox. Explicitly separate from real_object_stream (see module
# docstring): never silently substituted.
# --------------------------------------------------------------------------

def synthetic_burnin_conditions(rng: np.random.RandomState, n: int) -> Dict[str, np.ndarray]:
    """Draws n events' worth of "stable, correctly-calibrated" trigger
    conditions: pileup, multiplicity, and a luminosity-trend covariate,
    with realistic-scale LHC Run 2/3 values.

    pileup: ~Gamma-ish around 30-40 true interactions/crossing (Run 2/3
        typical), multiplicity: jet/object counts consistent with that
        pileup level, lumi: instantaneous luminosity in stable units
        (normalized to O(1) near fill start, matching residual.py's use
        of it purely as a trend covariate rather than a physical unit).
    """
    pileup = rng.gamma(shape=9.0, scale=4.0, size=n)          # mean ~36
    pileup = np.clip(pileup, 5.0, 80.0)
    n_jet = rng.poisson(lam=3.0 + 0.05 * pileup, size=n).astype(np.float64)
    n_jet = np.clip(n_jet, 2, None)
    lumi = np.ones(n) * 1.0  # stable-fill normalization; gradual.py perturbs this
    return {"pileup": pileup, "n_jet": n_jet, "lumi": lumi}


def _synthesize_object_features(
    rng: np.random.RandomState, pileup: np.ndarray, n_jet: np.ndarray,
) -> np.ndarray:
    """Generates a realistic-shaped object-feature vector conditioned on
    pileup/multiplicity -- higher pileup means somewhat higher HT/MET
    tails and more jets, mirroring real detector behavior, so the proxy
    VAE has genuine pileup-dependent structure to learn (and so residual.py
    has something real to regress out)."""
    n = len(pileup)
    jet1_pt = rng.gamma(shape=4.0, scale=25.0 + 0.3 * pileup, size=n)
    jet2_pt = jet1_pt * rng.uniform(0.4, 0.9, size=n)
    jet1_eta = rng.normal(0, 1.5, size=n)
    jet2_eta = rng.normal(0, 1.5, size=n)
    jet1_phi = rng.uniform(-np.pi, np.pi, size=n)
    jet2_phi = rng.uniform(-np.pi, np.pi, size=n)
    ht = jet1_pt + jet2_pt + rng.gamma(shape=2.0, scale=10.0 + 0.1 * pileup, size=n)
    met_pt = rng.gamma(shape=2.0, scale=8.0 + 0.05 * pileup, size=n)
    met_phi = rng.uniform(-np.pi, np.pi, size=n)
    n_muon = rng.poisson(lam=0.3, size=n).astype(np.float64)
    n_electron = rng.poisson(lam=0.2, size=n).astype(np.float64)

    return np.column_stack([
        jet1_pt, jet1_eta, jet1_phi, jet2_pt, jet2_eta, jet2_phi,
        n_jet, met_pt, met_phi, ht, n_muon, n_electron,
    ]).astype(np.float64)


def synthetic_object_stream(
    n_events: int,
    seed: Optional[int] = None,
    condition_fn=None,
    feature_bias_fn=None,
) -> Iterator[Dict[str, object]]:
    """Synthetic per-event stream with the SAME output shape as
    real_object_stream: {"features", "pileup", "n_jet", "lumi",
    "feature_names"}.

    :param condition_fn: optional callable(rng, i) -> dict with keys
        "pileup", "n_jet", "lumi" (each a length-1 array/scalar) for event
        index i, overriding the default stable-burn-in conditions. This is
        the hook drift_sim/gradual.py and drift_sim/abrupt.py use to inject
        realistic COVARIATE drift (pileup evolving across a fill, an abrupt
        multiplicity step) while reusing this same feature-synthesis logic.
    :param feature_bias_fn: optional callable(rng, i, features) -> features,
        applied AFTER _synthesize_object_features. Unlike condition_fn (which
        moves pileup/mult/lumi -- covariates the calibration model is
        supposed to already account for), this represents a change in the
        *relationship* between conditions and objects itself -- e.g. a jet
        energy scale that quietly drifts, or a masked readout channel that
        suppresses HT/MET independent of pileup. This is what makes a
        calibration genuinely go stale (see residual.py's module docstring)
        rather than just reflecting expected covariate movement.
    """
    rng = np.random.RandomState(seed)
    for i in range(n_events):
        if condition_fn is not None:
            cond = condition_fn(rng, i)
            pileup, n_jet, lumi = cond["pileup"], cond["n_jet"], cond["lumi"]
        else:
            cond = synthetic_burnin_conditions(rng, 1)
            pileup, n_jet, lumi = cond["pileup"], cond["n_jet"], cond["lumi"]

        pileup = np.atleast_1d(pileup).astype(np.float64)
        n_jet = np.atleast_1d(n_jet).astype(np.float64)
        lumi = np.atleast_1d(lumi).astype(np.float64)

        feats = _synthesize_object_features(rng, pileup, n_jet)
        if feature_bias_fn is not None:
            feats = feature_bias_fn(rng, i, feats)
        yield {
            "features": feats[0],
            "pileup": float(pileup[0]),
            "n_jet": float(n_jet[0]),
            "lumi": float(lumi[0]),
            "feature_names": DEFAULT_FEATURE_NAMES,
        }
