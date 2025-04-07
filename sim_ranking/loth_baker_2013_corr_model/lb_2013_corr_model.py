import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

# Short range coregionalization matrix, B1
B1 = np.array(
    [
        [0.29, 0.25, 0.23, 0.23, 0.18, 0.1, 0.06, 0.06, 0.06],
        [0.25, 0.30, 0.2, 0.16, 0.1, 0.04, 0.03, 0.04, 0.05],
        [0.23, 0.20, 0.27, 0.18, 0.1, 0.03, 0, 0.01, 0.02],
        [0.23, 0.16, 0.18, 0.31, 0.22, 0.14, 0.08, 0.07, 0.07],
        [0.18, 0.10, 0.1, 0.22, 0.33, 0.24, 0.16, 0.13, 0.12],
        [0.10, 0.04, 0.03, 0.14, 0.24, 0.33, 0.26, 0.21, 0.19],
        [0.06, 0.03, 0, 0.08, 0.16, 0.26, 0.37, 0.3, 0.26],
        [0.06, 0.04, 0.01, 0.07, 0.13, 0.21, 0.3, 0.28, 0.24],
        [0.06, 0.05, 0.02, 0.07, 0.12, 0.19, 0.26, 0.24, 0.23],
    ]
)

# Long range coregionalization matrix, B2
B2 = np.array(
    [
        [0.47, 0.4, 0.43, 0.35, 0.27, 0.15, 0.13, 0.09, 0.12],
        [0.4, 0.42, 0.37, 0.25, 0.15, 0.03, 0.04, 0, 0.03],
        [0.43, 0.37, 0.45, 0.36, 0.26, 0.15, 0.09, 0.05, 0.08],
        [0.35, 0.25, 0.36, 0.42, 0.37, 0.29, 0.2, 0.16, 0.16],
        [0.27, 0.15, 0.26, 0.37, 0.48, 0.41, 0.26, 0.21, 0.21],
        [0.15, 0.03, 0.15, 0.29, 0.41, 0.55, 0.37, 0.33, 0.32],
        [0.13, 0.04, 0.09, 0.2, 0.26, 0.37, 0.51, 0.49, 0.49],
        [0.09, 0.00, 0.05, 0.16, 0.21, 0.33, 0.49, 0.62, 0.6],
        [0.12, 0.03, 0.08, 0.16, 0.21, 0.32, 0.49, 0.6, 0.68],
    ]
)

# Nugget effect coregionalization matrix, B3
B3 = np.array(
    [
        [
            0.240000000000000,
            0.219983028675722,
            0.209991239369580,
            0.0899940658151642,
            -0.0199982490874490,
            0.0100004273375877,
            0.0299729607606612,
            0.0200291990885140,
            0.00995702711846606,
        ],
        [
            0.219983028675722,
            0.280000000000000,
            0.199999710563431,
            0.0400020556476041,
            -0.0500003168664929,
            -5.02841885169300e-07,
            0.0100141900421009,
            0.00994747690890486,
            -0.00996663511790072,
        ],
        [
            0.209991239369580,
            0.199999710563431,
            0.280000000000000,
            0.0500007637487926,
            -0.0600002196848805,
            -1.80663938055364e-07,
            0.0399992445541918,
            0.0299487635912810,
            0.0100035235224244,
        ],
        [
            0.0899940658151642,
            0.0400020556476041,
            0.0500007637487926,
            0.270000000000000,
            0.139999321454879,
            0.0499996979574019,
            0.0499981807238188,
            0.0499227563681531,
            0.0399858842409999,
        ],
        [
            -0.0199982490874490,
            -0.0500003168664929,
            -0.0600002196848805,
            0.139999321454879,
            0.190000000000000,
            0.0700000354290215,
            0.0499897414826383,
            0.0499443288879162,
            0.0499652709189241,
        ],
        [
            0.0100004273375877,
            -5.02841885169300e-07,
            -1.80663938055364e-07,
            0.0499996979574019,
            0.0700000354290215,
            0.120000000000000,
            0.0799859494118349,
            0.0699172702775962,
            0.0599608152312721,
        ],
        [
            0.0299729607606612,
            0.0100141900421009,
            0.0399992445541918,
            0.0499981807238188,
            0.0499897414826383,
            0.0799859494118349,
            0.120000000000000,
            0.0997643834727755,
            0.0800031285024676,
        ],
        [
            0.0200291990885140,
            0.00994747690890486,
            0.0299487635912810,
            0.0499227563681531,
            0.0499443288879162,
            0.0699172702775962,
            0.0997643834727755,
            0.100000000000000,
            0.0896690207890228,
        ],
        [
            0.00995702711846606,
            -0.00996663511790072,
            0.0100035235224244,
            0.0399858842409999,
            0.0499652709189241,
            0.0599608152312721,
            0.0800031285024676,
            0.0896690207890228,
            0.0900000000000000,
        ],
    ]
)

T_list = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0001])

B_coeff1_interpolator = RegularGridInterpolator((T_list, T_list), B1, bounds_error=True, method="linear")
B_coeff2_interpolator = RegularGridInterpolator((T_list, T_list), B2, bounds_error=True, method="linear")
B_coeff3_interpolator = RegularGridInterpolator((T_list, T_list), B3, bounds_error=True, method="linear")

def get_correlations(T1: float, T2: float, h: np.ndarray[float]) -> np.ndarray[float]:
    """
    Computes the spatial cross-correlation
    as per the Loth & Baker model
    
    Parameters
    ----------
    T1: float
        The first period
    T2: float
        The second period
    h: np.ndarray   
        The site-to-site distances
    
    Returns
    -------
    rho: np.ndarray
        The spatial cross-correlation
    """
    if min(T1, T2) < 0.01 or max(T1, T2) > 10:
        raise ValueError(
            "The periods must be between 0.01 and 10 seconds"
        )
    
    # Find the interval containing each input period
    index_1 = np.flatnonzero(T_list <= T1)[-1]
    index_2 = np.flatnonzero(T_list <= T2)[-1]

    # Interpolate each coregionalization matrix coefficient
    if index_1 == index_2:
        # Period pair is close to the diagonal. Uses a careful interpolation to 
        # keep the ridgeo on the diagonal and avoid creating saddle points 
        B_coeff1 = _interpolate_B_coefficients(T1, T2, index_1, index_2, B1)
        B_coeff2 = _interpolate_B_coefficients(T1, T2, index_1, index_2, B2)
        B_coeff3 = _interpolate_B_coefficients(T1, T2, index_1, index_2, B3)
    else:
        B_coeff1 = B_coeff1_interpolator(np.array([T1, T2]))
        B_coeff2 = B_coeff2_interpolator(np.array([T1, T2]))
        B_coeff3 = B_coeff3_interpolator(np.array([T1, T2]))

    # t = RegularGridInterpolator((T_list[index_1:index_1+2], T_list[index_2:index_2+2]), B1[index_1:index_1+2, index_2:index_2+2], bounds_error=True, method="linear")
    # t(np.array([T1, T2]))

    # TI, TJ = np.meshgrid(T_list, T_list, indexing="ij")
    # t = LinearNDInterpolator(np.stack([TJ.ravel(), TI.ravel()], axis=1), B1.ravel())
    # t(np.array([T1, T2]))

    # t2 = griddata(np.stack([TI.ravel(), TJ.ravel()], axis=1), B1.ravel(), (T1, T2), method="linear")


    # Compute the correlation
    rho = B_coeff1 * np.exp(-3 * h / 20.0) + B_coeff2 * np.exp(-3 * h / 70.0) + np.where(np.isclose(h, 0.0), B_coeff3, 0.0)
    return rho


def _interpolate_B_coefficients(T1: float, T2:float, index_1: int, index_2: int, B: np.ndarray):
    # Take just the adjacent cells of the B matrix
    T1_vals = T_list[index_1:index_1 + 2]
    T2_vals = T_list[index_2:index_2 + 2]
    B_vals = B[index_1:index_1 + 2, index_2:index_2 + 2]

    # Interpolate along the diagonal
    T_avg_targ = np.mean([T1, T2])

    T_avg_vals = np.mean([T1_vals, T2_vals], axis=0)
    B_avg_vals = np.diagonal(B_vals)
    B_diag_val = np.interp(
        T_avg_targ, T_avg_vals, B_avg_vals
    )

    # Interpolate between the diagonal and the corner value of B
    T_diff_targ = np.abs(T1 - T2)

    T_diff_vals = [0, np.abs(np.min(T1_vals) - np.max(T2_vals))]
    B_diff_vals = [B_diag_val, B_vals[0, 1]]
    B_coeff = np.interp(
        T_diff_targ, T_diff_vals, B_diff_vals
    )

    return B_coeff




