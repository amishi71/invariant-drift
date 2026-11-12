import numpy as np
from src.matched_far import zscore

def tail_indicator_z(score_windows, ref_scores, level):
    """Standardized exceedance indicator for a tail-rate CUSUM.
    score_windows: (n_windows, T), the same series the raw log-score monitor sees.
    ref_scores: the same healthy reference that monitor uses.
    level: 0.90, 0.99 or 0.999 -> healthy quantile."""
    q = np.quantile(ref_scores, level)
    ind = (score_windows > q).astype(np.float64)
    ref_ind = (ref_scores > q).astype(np.float64)
    return zscore(ind, ref_ind)
