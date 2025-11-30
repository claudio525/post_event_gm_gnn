"""
Test the Loth and Baker 2013 correlation model against benchmark data.

However, due to differences the interpolation of the B-coefficients,
the results are not exactly the same, hence the large tolerance.
"""
import itertools

import scipy.io as io
import numpy as np
from numpy.testing import assert_allclose

import post_event_gm_gnn as pg

DISTANCES = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]


def test_lb13_benchmark():
    """
    Test the single IM Loth and Baker 2013 correlation model
    python implementation against the MATLAB benchmark data.
    """
    bench_data = io.loadmat("./bench_data.mat")["results"]

    max_diff = -np.inf
    max_diff_index = None
    for i, T_i in enumerate(PERIODS):
        for j, T_j in enumerate(PERIODS):
            rho = pg.lb13.get_correlations(T_i, T_j, DISTANCES)

            diff = np.abs(rho - bench_data[:, i, j])
            if diff.max() > max_diff:
                max_diff = diff.max()
                max_diff_index = (i, j, diff.argmax())

            assert_allclose(rho, bench_data[:, i, j], atol=0.06)

    print(f"Max diff: {max_diff} at index {max_diff_index}")

def test_lb13_vec_benchmark():
    """
    Test the vectorized Loth and Baker 2013 correlation model
    python implementation against the MATLAB benchmark data.
    """
    bench_data = io.loadmat("./bench_data.mat")["results"]

    period_combs = np.array(list(itertools.product(PERIODS, repeat=2)))

    rho = pg.lb13.get_correlations_vec(period_combs[:, 0], period_combs[:, 1], DISTANCES)
    rho = rho.T.reshape(bench_data.shape)

    assert_allclose(rho, bench_data, atol=0.06)

    print("Vectorized - Max diff: ", np.abs(rho - bench_data).max())
  
        

