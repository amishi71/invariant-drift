import numpy as np
from src.injection_sets import INJECTION_SETS, inject_masked_cols

def test_l1_calo_scales_seven_leaves_five():
    cols = INJECTION_SETS["l1_calo"]
    assert tuple(cols) == (4, 5, 7, 8, 9, 10, 11)
    f = np.full((3, 10, 12), 10.0)
    out = inject_masked_cols(f, 0.4, cols, 0.0)
    untouched = [c for c in range(12) if c not in cols]
    assert untouched == [0, 1, 2, 3, 6]
    assert np.allclose(out[..., list(cols)], 6.0)
    assert np.allclose(out[..., untouched], 10.0)
