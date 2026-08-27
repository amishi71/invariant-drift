# data/

This directory holds:
- `jetht_files.txt` -- a file list (plain text) pointing `src/stream_loader.py`'s
  `real_object_stream()` at real NanoAOD file(s), and
- `real/` -- the actual downloaded ROOT file(s) (gitignored, see below).

## Status in this repo

A real-data touchpoint has been done (see `scripts/real_data_touchpoint.py`
and the project handover, section 6.4): CMS Open Data record 30558
(`/JetHT/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD`) was downloaded and
used to fit `residual.py`'s `CalibrationModel` on real HT scores, which
surfaced and led to a fix for a real edge case (exact-zero scores breaking
the log-transform -- see README.md's "Known issues").

This was a narrow, deliberate touchpoint: it exercises the regression/
residual machinery only, not the full pipeline (proxy VAE training, all 5
detectors, ACI/FDR). `real_object_stream()`'s full-pipeline path is written
and structurally ready but has not been run at that scale -- see "What's
stubbed" in README.md.

## `data/real/` is gitignored

The downloaded ROOT file is ~2.1GB, too large to commit. `data/jetht_files.txt`
is tracked and points at it, but a fresh clone will NOT have the file itself
-- you need to re-download it (see below) before `scripts/real_data_touchpoint.py`
will run.

## Re-downloading the real-data file

`cernopendata-client`'s plain `pip install` only supports the `http` and
`xrootd` protocols -- **not** `https` (that needs the `[xrootd]` extra, which
needs a system XRootD install). Use `http`:

```bash
pip install cernopendata-client
cernopendata-client download-files --recid 30558 --protocol http --filter-range 1-1
```

Note: this client version has no `--output-dir` flag -- it downloads into
`./30558/` in your current directory. Move it into `data/real/` yourself:

```bash
mkdir -p data/real
mv 30558 data/real/
```

`data/jetht_files.txt` already points at this path by default.

## Generating a file list for a different record

To point at a different CMS Open Data record instead, search
[opendata.cern.ch](https://opendata.cern.ch/) for a JetHT (or equivalent
trigger-stream) NanoAOD record for the run period you want, then:

```bash
cernopendata-client download-files --recid <RECID> --protocol http --filter-range <RANGE>
```

and update `data/jetht_files.txt` (or pass a different path via
`real_object_stream()`'s `file_list_path` argument).

## Required branches

`real_object_stream()` reads (see `FATJET_LIKE_BRANCHES` in
`stream_loader.py`): `Jet_pt`, `Jet_eta`, `Jet_phi`, `nJet`, `MET_pt`,
`MET_phi`, `PV_npvsGood`, `nMuon`, `nElectron`. All are standard NanoAOD
branches; no custom skimming needed beyond a `nJet >= 2` event selection
(applied in `_compute_object_features`).

## Full-pipeline real data (not yet done)

Running the *entire* pipeline (proxy VAE, all 5 detectors, ACI/FDR) against
real data at scale, and any CERN openlab access, are still open -- see
README.md's "What's stubbed" and the handover's section 7 ("Data strategy").
