import numpy as np


def similiarity_score(rs_obs: np.ndarray, rs_sim: np.ndarray) -> float:
    # Compute the residual
    res = np.log(rs_obs) - np.log(rs_sim)
    res_area = np.trapz(np.abs(res))

    return res_area * (-1 / 30) + 1 if res_area < 30 else 0

def similiarity_score_t(rs_obs: np.ndarray, rs_sim: np.ndarray):
    # Compute the residual
    res = np.log(rs_obs[np.newaxis, :]) - np.log(rs_sim)
    res_area = np.trapz(np.abs(res), axis=1)

    return np.where(res_area < 30, res_area * (-1 / 30) + 1,0)