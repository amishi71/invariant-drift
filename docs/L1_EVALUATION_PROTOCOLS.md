# Evaluating the monitor on real CMS L1 zero-bias data: protocols and caveats

This page defines what each L1 result in the repository means. It contains no results; quote numbers from the
run logs (`results/l1_validity/<tag>/`) and say which protocol produced them.

## Data
`CERN/anomaly_detection_cmsl1t`, zero-bias runs 396102 and 398183, all three splits per run, sorted by
(lumi section, orbit, bunch crossing) by `scripts/build_l1_ordered_cache.py`. The dataset's own split is a random
permutation, so only the ordered caches are chronological. Pileup is `nPV_True`, one value per lumi section.

## Protocols (`scripts/matched_far_heldout.py`)
| flag | calibration windows (set h) | validation windows (false-alarm check) | evaluation / tuning windows | what it shows |
|---|---|---|---|---|
| default (`--time-split none`) | random slots of the region | random slots of the same region | random slots of the same region | calibration holds when calibration and monitoring data are interleaved in time |
| `--time-split forward` | slots before `--forward-boundary` | slots after it | slots after it | calibrate on the past, monitor the future inside one run |
| `--calib-cache A --cache-path B` | windows of run B | windows of run B | windows of run B | VAE, regression and reference learned on run A only; everything monitored is run B |

Only the second and third protocols support a deployability claim. The first is the original design and stays
for comparison.

## Injections (`src/injection_sets.py`)
Feature-level, multiplicative, applied to cached features. `--inject-set` chooses the columns, `--requantize 0.5`
rounds the scaled values back onto the 0.5 GeV lattice.
* `l1_legacy` (4, 8, 9): lead jet ET, HT, MET only. Leaves MHT, ETT and the other lead-object ETs unscaled, which breaks
  their correlations; the shift it produces is large (about +1.5 sigma at a 40% drop), so detection is easy.
* `l1_calo` (4, 5, 7, 8, 9, 10, 11): every calorimeter-derived energy scaled together; muon ET and counts untouched.
* Object counts and sub-leading objects are not recomputed. This is not a hardware-level injection.
Report power for `l1_calo` with `--requantize 0.5` as the realistic case and `l1_legacy` as the easy case, and say so.

## Direction
"up-only" and "down-only" CUSUMs have a fixed direction. Which one is correct depends on the dataset and the injection
(the masked-drop shift is negative on JetHT and positive on L1). The report prints a direction check; the two-sided
detector is the deployable headline, and one-sided rows are diagnostics.

## Uncertainty
Wilson intervals treat windows as independent. Window means are correlated (the standard deviation of window means is 1.5 to 3 times the
i.i.d. value in the diagnostics), and the regions of one run are not independent seeds. The selected-k table therefore
also prints the spread of the per-seed means; quote that range next to any pooled interval.
