# data/

This directory holds **file lists** (plain text, one XRootD/EOS path per
line, `#`-prefixed comments allowed) for `src/stream_loader.py`'s
`real_object_stream()` -- not the data files themselves.

## Generating a file list

CMS Open Data is indexed at [opendata.cern.ch](https://opendata.cern.ch/).
The standard tool for resolving a record to concrete file paths is
`cernopendata-client`:

```
pip install cernopendata-client
cernopendata-client get-file-locations --recid <RECID> --protocol xrootd \
    > data/jetht_files.txt
```

Pick a JetHT (or equivalent trigger-stream) NanoAOD record for the run
period you want -- search opendata.cern.ch for "NanoAOD" datasets from the
relevant CMS Run 2/3 era. `real_object_stream()` expects the file list at
`data/jetht_files.txt` by default (override via its `file_list_path`
argument).

## Why this is empty in this build

The environment this project was built in has network access to package
registries (PyPI, GitHub, npm) only -- not to CERN's XRootD/EOS
redirectors. `real_object_stream()` in `src/stream_loader.py` is written
and structurally ready (retry/circuit-breaker logic adapted from
cms-streaming-shift-detection, same "raise immediately if the file list is
missing, no silent fallback" discipline) but has not been exercised
against a live XRootD endpoint. Once you have a real file list here and a
working EOS network path, it should work as-is -- its output shape exactly
matches `synthetic_object_stream()` in the same file, so nothing else in
the pipeline (`main.py`, `residual.py`, the detectors, `evaluation.py`)
needs to change to switch from synthetic to real data.

## Required branches

`real_object_stream()` reads (see `FATJET_LIKE_BRANCHES` in
`stream_loader.py`): `Jet_pt`, `Jet_eta`, `Jet_phi`, `nJet`, `MET_pt`,
`MET_phi`, `PV_npvsGood`, `nMuon`, `nElectron`. All are standard NanoAOD
branches; no custom skimming needed beyond a `nJet >= 2` event selection
(applied in `_compute_object_features`).
