"""Injection column sets and quantization-aware injections, independent of any pool object.

Why this exists. The drift injections scale a set of feature columns. Which columns are
"energy-like" depends on the cache layout, and on the CMS L1 caches the legacy JetHT indices would hit
object counts, so the choice has to be explicit and per cache. The previous mechanism kept it in a module
global that was rewritten whenever a pool was built (two pools in one process clashed). Here the set is
resolved from a name and the cache file, and passed to the injection functions explicitly.

Layouts
  jetht   (12 cols)  0 jet1_pt 1 jet1_eta 2 jet1_phi 3 jet2_pt 4 jet2_eta 5 jet2_phi 6 n_jet 7 met_pt
                     8 met_phi 9 ht 10 n_muon 11 n_electron
  l1      (12 cols)  0 n_jets 1 n_egammas 2 n_muons 3 n_taus 4 lead jet ET 5 lead e/gamma ET
                     6 lead muon ET 7 lead tau ET 8 HT 9 MET 10 MHT 11 ETT          (ET in GeV, 0.5 GeV steps)

Sets
  jetht       (0, 3, 9, 7)               the original JetHT injection (jet1_pt, jet2_pt, ht, met_pt)
  l1_legacy   (4, 8, 9)                  what the first L1 runs used: lead jet ET, HT, MET only. MHT, ETT and the
                                         other lead-object ETs stay unscaled, which breaks their correlations
                                         with the scaled ones.
  l1_calo     (4, 5, 7, 8, 9, 10, 11)    every calorimeter-derived energy scaled together (a gain/calibration
                                         drift moves all of them); muon ET and the object counts are untouched.

Limits that remain: object COUNTS and sub-leading objects are not recomputed (a real energy-scale shift would
push jets across the object thresholds), so this is a feature-level injection, not a hardware-level one.

`requantize` (e.g. 0.5) rounds the scaled columns back onto the hardware lattice after scaling. Use it on L1
caches: without it the scaled values sit off the 0.5 GeV lattice, which is itself a signature the proxy VAE
could pick up.
"""
from typing import Optional, Sequence

import numpy as np

INJECTION_SETS = {
    "jetht": (0, 3, 9, 7),
    "l1_legacy": (4, 8, 9),
    "l1_calo": (4, 5, 7, 8, 9, 10, 11),
}

_L1_KEYS = ("lumi", "orbit", "run", "bx")


def detect_layout(cache_path: str) -> str:
    """'l1' if the cache carries any time key written by the L1 cache builders, else 'jetht'."""
    with np.load(cache_path) as z:
        return "l1" if any(k in z.files for k in _L1_KEYS) else "jetht"


def resolve_injection_cols(name: str = "auto", cache_path: Optional[str] = None) -> tuple:
    """Columns for an injection set. 'auto' keeps the previous behaviour: JetHT indices for a JetHT cache,
    the legacy L1 set for an L1 cache. Synthetic pools (no cache) use the JetHT layout."""
    if name == "auto":
        layout = detect_layout(cache_path) if cache_path else "jetht"
        name = "l1_legacy" if layout == "l1" else "jetht"
    if name not in INJECTION_SETS:
        raise ValueError(f"unknown injection set {name!r}; choose from {sorted(INJECTION_SETS)} or 'auto'")
    return INJECTION_SETS[name]


def _requantize(x: np.ndarray, step: float) -> np.ndarray:
    return np.rint(x / step) * step


def inject_masked_cols(feats: np.ndarray, drop: float, cols: Sequence[int], requantize: float = 0.0) -> np.ndarray:
    """Multiplicative drop on `cols` from the first event of every window. feats: (n_windows, T, n_features)."""
    out = np.array(feats, dtype=np.float64, copy=True)
    cols = list(cols)
    out[..., cols] = out[..., cols] * (1.0 - drop)
    if requantize:
        out[..., cols] = _requantize(out[..., cols], requantize)
    return out


def inject_radiation_cols(feats: np.ndarray, tau: float, cols: Sequence[int], gain0: float = 1.0,
                          requantize: float = 0.0) -> np.ndarray:
    """gain(i) = gain0 * exp(-i / tau) on `cols`, i = event index inside the window."""
    out = np.array(feats, dtype=np.float64, copy=True)
    cols = list(cols)
    gain = gain0 * np.exp(-np.arange(out.shape[1]) / tau)
    out[..., cols] = out[..., cols] * gain[None, :, None]
    if requantize:
        out[..., cols] = _requantize(out[..., cols], requantize)
    return out
