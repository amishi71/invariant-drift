"""Per-run JetHT cache from the NanoAOD files in data/real_multi (new definitions, see assumptions below).
Features follow the 12-col 'jetht' layout: jet1 pt/eta/phi, jet2 pt/eta/phi, nJet, MET pt/phi, HT, nMuon, nElectron.
Assumptions: HT = sum Jet_pt with pt>30 & |eta|<2.5; pileup = PV_npvs; missing jets padded with 0."""
import glob, numpy as np, awkward as ak, uproot

BR = ["run", "luminosityBlock", "event", "nJet", "Jet_pt", "Jet_eta", "Jet_phi",
      "MET_pt", "MET_phi", "nMuon", "nElectron", "PV_npvs"]
feats, pu, run, lumi, evt = [], [], [], [], []

def lead(a, n):
    return ak.to_numpy(ak.fill_none(ak.pad_none(a, 2, axis=1), 0.0)[:, n]).astype(np.float32)

for f in sorted(glob.glob("data/real_multi/*.root")):
    print(f.split("/")[-1], flush=True)
    for a in uproot.iterate(f + ":Events", BR, step_size="150 MB", library="ak"):
        pt, eta, phi = a["Jet_pt"], a["Jet_eta"], a["Jet_phi"]
        ht = ak.to_numpy(ak.sum(pt[(pt > 30) & (abs(eta) < 2.5)], axis=1)).astype(np.float32)
        X = np.stack([lead(pt, 0), lead(eta, 0), lead(phi, 0), lead(pt, 1), lead(eta, 1), lead(phi, 1),
                      ak.to_numpy(a["nJet"]).astype(np.float32), ak.to_numpy(a["MET_pt"]).astype(np.float32),
                      ak.to_numpy(a["MET_phi"]).astype(np.float32), ht,
                      ak.to_numpy(a["nMuon"]).astype(np.float32), ak.to_numpy(a["nElectron"]).astype(np.float32)], axis=1)
        feats.append(X); pu.append(ak.to_numpy(a["PV_npvs"]).astype(np.float32))
        run.append(ak.to_numpy(a["run"]).astype(np.uint32)); lumi.append(ak.to_numpy(a["luminosityBlock"]).astype(np.uint32))
        evt.append(ak.to_numpy(a["event"]).astype(np.uint64))

F, P, R, L, E = (np.concatenate(x) for x in (feats, pu, run, lumi, evt))
o = np.lexsort((E, L, R))
np.savez("data/real_cache/jetht_runs.npz", features=F[o], pileup=P[o], n_jet=F[o][:, 6].copy(),
         run_id=R[o], lumi_id=L[o], event_id=E[o])
print(F.shape, "runs:", len(np.unique(R)))
