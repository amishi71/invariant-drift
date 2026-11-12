import numpy as np
from src.rate_monitors import tail_indicator_z

def test_healthy_mean_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(200_000)
    z = tail_indicator_z(rng.standard_normal((50, 2500)), ref, 0.99)
    assert abs(z.mean()) < 0.05

def test_rate_increase_moves_z_up():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(200_000)
    z = tail_indicator_z(rng.standard_normal((50, 2500)) + 1.0, ref, 0.99)
    assert z.mean() > 0.5
