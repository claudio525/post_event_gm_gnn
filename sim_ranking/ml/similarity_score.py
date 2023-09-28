import numpy as np

def similiarity_score(rs_obs: np.ndarray, rs_sim: np.ndarray):
    """
    Computes the similarity score for response spectrum
    based on area under the (absolute) residual curve

    Parameters
    ----------
    rs_obs: array of floats
        Observed response spectrum
        Format [n_periods]
    rs_sim: array of floats
        Simulation realisation response spectra
        Format [n_realisations, n_periods]

    Returns
    -------
    array of floats
        Similarity score for each realisation
        Format [n_realisations]
    """
    # Compute the residual
    res = np.log(rs_obs[np.newaxis, :]) - np.log(rs_sim)
    res_area = np.trapz(np.abs(res), axis=1)

    return np.where(res_area < 30, res_area * (-1 / 30) + 1,0)