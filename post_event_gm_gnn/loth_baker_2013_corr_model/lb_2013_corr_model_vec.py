import numpy as np

import ml_tools as mlt

from .lb_2013_corr_model import (
    B1,
    B2,
    B3,
    T_list,
    B_coeff1_interpolator,
    B_coeff2_interpolator,
    B_coeff3_interpolator,
)


def get_correlations(
    T1: np.ndarray[float], T2: np.ndarray[float], h: np.ndarray[float]
) -> np.ndarray[float]:
    """
    Computes the spatial cross-correlation for each period pair,
    and all distances in h.

    Parameters
    ----------
    T1: np.ndarray[float]
        The first periods
    T2: np.ndarray[float]
        The second periods
        Has to be the same size as T1
    h: np.ndarray[float]
        The site-to-site distances
    Returns
    -------
    rho: np.ndarray[float]
        The spatial cross-correlation
        Shape: [T1.size, h.size]
    """

    # Check if input periods are within valid range
    if np.any(T1 < 0.01) or np.any(T1 > 10) or np.any(T2 < 0.01) or np.any(T2 > 10):
        raise ValueError("All periods must be between 0.01 and 10 seconds")
    # Check input dimensions
    if T1.ndim != 1 or T2.ndim != 1 or h.ndim != 1:
        raise ValueError("Input arrays T1, T2, and h must be 1-dimensional")

    # Check that arrays have the same length
    if not (T1.size == T2.size):
        raise ValueError("Input arrays T1 and T2 must have the same length")

    ind1 = mlt.array_utils.find_nearest_smaller_vec(T_list, T1)
    ind2 = mlt.array_utils.find_nearest_smaller_vec(T_list, T2)

    # Interpolate each coregionalization matrix coefficient
    B_coeff1 = np.full(T1.shape[0], np.nan)
    B_coeff2 = np.full(T1.shape[0], np.nan)
    B_coeff3 = np.full(T1.shape[0], np.nan)

    equal_indices_mask = ind1 == ind2

    # Spatial correlation only
    if equal_indices_mask.any():
        B_coeff1[equal_indices_mask] = _interpolate_B_coefficients(
            T1[equal_indices_mask],
            T2[equal_indices_mask],
            ind1[equal_indices_mask],
            ind2[equal_indices_mask],
            B1,
        )
        B_coeff2[equal_indices_mask] = _interpolate_B_coefficients(
            T1[equal_indices_mask],
            T2[equal_indices_mask],
            ind1[equal_indices_mask],
            ind2[equal_indices_mask],
            B2,
        )
        B_coeff3[equal_indices_mask] = _interpolate_B_coefficients(
            T1[equal_indices_mask],
            T2[equal_indices_mask],
            ind1[equal_indices_mask],
            ind2[equal_indices_mask],
            B3,
        )

    # Spatial cross correlation
    B_coeff1[~equal_indices_mask] = B_coeff1_interpolator(np.stack([T1, T2], axis=1))[
        ~equal_indices_mask
    ]
    B_coeff2[~equal_indices_mask] = B_coeff2_interpolator(np.stack([T1, T2], axis=1))[
        ~equal_indices_mask
    ]
    B_coeff3[~equal_indices_mask] = B_coeff3_interpolator(np.stack([T1, T2], axis=1))[
        ~equal_indices_mask
    ]

    # Compute the correlation
    rho = (
        B_coeff1[:, None] * np.exp(-3 * h / 20.0)
        + B_coeff2[:, None] * np.exp(-3 * h / 70.0)
        + np.where(np.isclose(h, 0.0), B_coeff3[:, None], 0.0)
    )
    return rho


def _interpolate_B_coefficients(
    T1: np.ndarray[float],
    T2: np.ndarray[float],
    ind1: np.ndarray[int],
    ind2: np.ndarray[int],
    B: np.ndarray,
):
    # Take just the adjacent cells of the B matrix
    T1_vals, T2_vals, B_vals = [], [], []
    for i in range(T1.size):
        T1_vals.append(T_list[ind1[i] : ind1[i] + 2])
        T2_vals.append(T_list[ind2[i] : ind2[i] + 2])
        B_vals.append(B[ind1[i] : ind1[i] + 2, ind2[i] : ind2[i] + 2])
    T1_vals = np.stack(T1_vals, axis=0)
    T2_vals = np.stack(T2_vals, axis=0)
    B_vals = np.stack(B_vals, axis=0)

    # Interpolate along the diagonal
    T_avg_targ = np.mean(np.stack([T1, T2], axis=0), axis=0)

    T_avg_vals = np.mean(np.stack([T1_vals, T2_vals], axis=-1), axis=2)
    B_avg_vals = np.diagonal(B_vals, axis1=1, axis2=2)
    B_diag_val = [
        np.interp(T_avg_targ[i], T_avg_vals[i], B_avg_vals[i]) for i in range(T1.size)
    ]

    # Interpolate between the diagonal and the corner value of B
    T_diff_targ = np.abs(T1 - T2)

    T_diff_vals = np.stack(
        [np.zeros(T1.size), np.abs(np.min(T1_vals, axis=1) - np.max(T2_vals, axis=1))],
        axis=1,
    )

    B_diff_vals = np.stack([B_diag_val, B_vals[:, 0, 1]], axis=1)
    B_coeff = [
        np.interp(T_diff_targ[i], T_diff_vals[i], B_diff_vals[i])
        for i in range(T1.size)
    ]

    return B_coeff
