# Advisor note: AXOL1TL-proxy scope (CICADA dropped) -- SIGN-OFF PENDING

**Status: drafted, not yet confirmed with advisor.** Flagging this
explicitly rather than letting it sit implicit in the code, per the
project's own build-order plan (this was slated as step 1, ahead of the
proxy VAE itself, precisely because getting the framing wrong is a real
risk to catch early rather than after the writeup is built on top of it).

## The issue

Both AXOL1TL and CICADA (the abstract's two named CMS learned-anomaly-
trigger examples) run on **Level-1 trigger primitives** -- raw calorimeter
regions and L1 muon/jet objects computed in custom hardware, not the
offline-reconstructed objects (PAT jets, PF candidates, etc.) that make up
public CMS/LHC Open Data. Neither trigger's actual input representation is
directly available outside CMS's own trigger system.

- **CICADA** consumes a regular calorimeter-tower grid as a 2D image and
  uses a convolutional architecture built specifically for that spatial
  structure. There is no faithful public-data analog of a calorimeter-tower
  image at that granularity -- reconstructing one from offline objects
  would mean inventing a representation CICADA was never trained on, which
  would make any comparison to CICADA's own reported performance
  meaningless rather than approximate.
- **AXOL1TL** consumes a flatter, object-level input set (L1 jets, MET,
  multiplicities). Offline NanoAOD has a genuine, if imperfect, structural
  analog: reconstructed jet kinematics, MET, HT, and object multiplicities.
  The representations differ in resolution and latency-driven
  approximations, but the *shape* of the input (a feature vector over
  physics objects, not an image) is comparable.

## Proposed framing (what's implemented)

Drop CICADA from direct reproduction scope. Build `src/proxy_vae.py` as an
**AXOL1TL-proxy**: a VAE trained on offline-reconstructed jet/MET/HT/
multiplicity features (background-only, mirroring AXOL1TL's own Zero-Bias
training), with reconstruction error standing in for AXOL1TL's anomaly
score. Proposed abstract-facing language for the eventual writeup:

> Anomaly scores are computed with a variational autoencoder trained on
> offline-reconstructed jet and event-level objects (jet kinematics, MET,
> HT, object multiplicities) as a proxy for AXOL1TL's L1-object inputs,
> since raw L1 trigger primitives are not part of public CMS Open Data.
> The calibration-drift-detection methodology evaluated here -- the
> change-point detectors and the conformal/FDR layer -- is agnostic to
> which upstream anomaly-score model produces the monitored scalar, so
> this substitution affects the absolute anomaly-score scale but not the
> methodology being evaluated.

## Why this needs an explicit yes before the writeup leans on it

- It changes what can honestly be claimed: "we evaluated AXOL1TL's
  calibration drift" is not the same claim as "we evaluated a proxy
  model's calibration drift, in AXOL1TL's style." The abstract as
  currently drafted names both AXOL1TL and CICADA directly as the systems
  the drift-detection machinery is built around -- worth confirming
  whether the advisor wants the abstract's own language softened to match
  ("anomaly triggers such as AXOL1TL" framing already helps, but the
  results section will need the same care).
- CICADA's absence should probably be a stated limitation, not a silent
  scope narrowing discovered by a reader comparing the abstract's framing
  to the actual repo.
- If there's institutional access to L1 trigger primitives or CICADA's
  actual training data through the collaboration (rather than public Open
  Data), the "proxy" framing may be unnecessary and this whole note moot
  -- worth checking before the proxy framing gets load-bearing in the
  writeup.

## What ships either way

Everything downstream of the anomaly score -- `residual.py`'s calibration
regression, all five Component-1 detectors, and the ACI/FDR layer in
`conformal/` -- takes a scalar anomaly score as an opaque input. If the
proxy framing changes or a real AXOL1TL score becomes available, only
`proxy_vae.py` (and the feature-extraction half of `stream_loader.py`)
would need to change; nothing else in the pipeline assumes anything about
how the score was produced.
