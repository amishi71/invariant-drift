"""
Vectorized feature extraction for the CMS L1T anomaly-detection dataset
(CERN/anomaly_detection_cmsl1t), producing a fixed-length scalar feature
vector per event analogous to the old 12-feature JetHT cache.

Unit conversions per the dataset card's documented scale factors:
  jets/egammas/taus Et: x0.5 GeV
  muon Et: (raw - 1) x 0.5 GeV  [hardware zero means "no muon"]
  energy sums (ET/HT/MET/MHT): x0.5 GeV

Place this file at: src/feature_extraction_cmsl1t.py

NOTE: pyarrow.compute has no list_max function. Per-row max over a jagged
list column is computed here via list_flatten + list_parent_indices +
group_by(...).aggregate([('val', 'max')]), which is the correct vectorized
approach in pyarrow >= ~10. Rows with empty lists get 0.0 (no group entry).
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def _lead_or_zero(list_array: pa.ListArray, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Max value per row of a jagged int list column, 0.0 for empty lists."""
    n = len(list_array)
    result = np.zeros(n, dtype=np.float64)

    flat = pc.list_flatten(list_array)
    if len(flat) == 0:
        return result

    parent_indices = pc.list_parent_indices(list_array)
    tbl = pa.table({"val": flat, "parent": parent_indices})
    grouped = tbl.group_by("parent").aggregate([("val", "max")])

    parents = grouped["parent"].to_numpy(zero_copy_only=False)
    maxes = grouped["val_max"].to_numpy(zero_copy_only=False).astype(np.float64)

    result[parents] = (maxes - offset) * scale
    return result


def _count(list_array: pa.ListArray) -> np.ndarray:
    return pc.list_value_length(list_array).to_numpy(zero_copy_only=False).astype(np.float64)


def _unwrap_singleton(list_array: pa.ListArray, scale: float = 1.0) -> np.ndarray:
    """Energy-sum columns are single-element lists; take element 0."""
    flat = pc.list_flatten(list_array)
    arr = flat.to_numpy(zero_copy_only=False).astype(np.float64)
    return arr * scale


def extract_features_batch(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        features: (N, 12) float64 array
        pileup:   (N,) float32/float64 array (nPV_True)
        labels:   (N,) int16 array
    """
    n_jets = _count(table["jets_jetIEt"])
    n_egammas = _count(table["egammas_egIEt"])
    n_muons = _count(table["muons_muonIEt"])
    n_taus = _count(table["taus_tauIEt"])

    lead_jet_et = _lead_or_zero(table["jets_jetIEt"], scale=0.5)
    lead_eg_et = _lead_or_zero(table["egammas_egIEt"], scale=0.5)
    lead_muon_et = _lead_or_zero(table["muons_muonIEt"], scale=0.5, offset=1.0)
    lead_tau_et = _lead_or_zero(table["taus_tauIEt"], scale=0.5)

    ht = _unwrap_singleton(table["HT_Et"], scale=0.5)
    met = _unwrap_singleton(table["MET_Et"], scale=0.5)
    mht = _unwrap_singleton(table["MHT_Et"], scale=0.5)
    ett = _unwrap_singleton(table["ET_Et"], scale=0.5)

    features = np.stack([
        n_jets, n_egammas, n_muons, n_taus,
        lead_jet_et, lead_eg_et, lead_muon_et, lead_tau_et,
        ht, met, mht, ett,
    ], axis=1)

    pileup = table["nPV_True"].to_numpy(zero_copy_only=False)
    labels = table["label"].to_numpy(zero_copy_only=False)

    return features, pileup, labels
