from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import lb_2013_corr_model as lb13

R_EARTH = 6378.139

def calculate_distance_matrix(
    stations: Sequence[str],
    locations_df: pd.DataFrame,
    site_lon_col: str = "lon",
    site_lat_col: str = "lat",
):
    """
    Given a set of stations and their locations (in lat, lon format),
    calculate the matrix containing
    the pairwise distance

    Parameters
    ----------
    stations: Sequence[str]
        List of the station names
    locations_df: pd.DataFrame
        Locations of each of the stations (in lat, lon)
    site_lat_col: str
    site_lon_col: str
    """
    dist_matrix = (
        haversine_distances(
            np.radians(locations_df.loc[stations, [site_lat_col, site_lon_col]].values)
        )
        * R_EARTH
    )
    return pd.DataFrame(data=dist_matrix, index=stations, columns=stations)

def get_pSA_period(im: str):
    if im.startswith("pSA"):
        return float(im.split("_")[-1])
    return None

def get_im_corr(im: str, sites: np.ndarray, dist_matrix: pd.DataFrame):
    """
    Gets the correlation for the specified IM
    and the specified sites. If sites are not specified,
    all sites in the distance matrix are used.
    """
    T = get_pSA_period(im)

    cur_dist_matrix = dist_matrix.loc[sites, sites]

    corr_matrix = lb13.get_correlations(T, T, cur_dist_matrix.values)
    corr_matrix = corr_matrix.reshape(cur_dist_matrix.shape)
    # np.fill_diagonal(corr_matrix, 1.0)
    assert np.allclose(np.diagonal(corr_matrix), 1.0)
    return pd.DataFrame(
        index=cur_dist_matrix.index,
        data=corr_matrix,
        columns=cur_dist_matrix.columns,
    )

def compute_cond_lnIM_dist(
    site_id: str,
    gm_params_df: pd.DataFrame,
    obs_lnIM_series: pd.Series,
    R: pd.DataFrame,
    allow_self: bool = False,
):
    """
    Computes the lnIM distribution for a site of interest
    conditioned on observations (from the same event);
    with the marginal distribution given by an
    empirical GMM

    Parameters
    ----------
    site_id: string
        Site of interest
    gm_params_df: dataframe
        GM parameters for the event and relevant sites
        from an empirical GMM

        Index must be the sites (Site of interest + Observation sites)
        Columns are [mu, sigma_total, sigma_between, sigma_within]
    obs_lnIM_series: series
        lnIM values at the observation sites
    R: dataframe
        Correlation matrix for all
        relevant sites (site of interest + observation sites)
    allow_self: bool, optional
        If True, allows the site of interest to be used
        as an observation site for itself.
        Default is False.

    Returns
    -------
    cond_lnIM_mu: float
        The conditional mean estimation of lnIM
        at the site of interest
    cond_lnIM_sigma
        The conditional sigma estimation of lnIM
        at the site of interest
    """
    obs_stations = obs_lnIM_series.index.values.astype(str)

    # Sanity checks
    assert np.all(np.isin(obs_stations, gm_params_df.index))
    assert allow_self or site_id in gm_params_df.index and site_id not in obs_stations

    # Relevant stations (Observation sites & Sites of interest)
    # rel_stations = obs_stations if allow_self else np.concatenate(([site_id], obs_stations))
    rel_stations = np.concatenate(([site_id], obs_stations))

    if allow_self and site_id in obs_stations:
        gm_params_df = gm_params_df.loc[obs_stations]
    else:
        gm_params_df = gm_params_df.loc[rel_stations]

    # Compute covariance matrix of within-event residuals
    # C_c(i,j) = rho_{i,j} * \delta_{W_i} * \delta_{W_j}
    # Equation 4 in Bradley 2014
    C_c = pd.DataFrame(
        data=np.einsum(
            "i, ij, j -> ij",
            gm_params_df.loc[obs_stations].sigma_within.values,
            R.loc[obs_stations, obs_stations].values,
            gm_params_df.loc[obs_stations].sigma_within.values,
        ),
        index=obs_stations,
        columns=obs_stations,
    )
    # Compute the inverse covariance matrix
    C_c_inv = np.linalg.inv(C_c)

    # Sanity check
    assert np.all(
        np.isclose(
            np.diag(C_c.values),
            gm_params_df.loc[obs_stations, "sigma_within"].values ** 2,
        )
    )

    # Compute the total residual
    total_residual = (
        obs_lnIM_series.loc[obs_stations] - gm_params_df.loc[obs_stations, "mu"]
    )

    # Compute the between event-residual using the observation stations
    # First part of Equation 3 numerator is just row-wise sum of inverse C_c
    numerator = np.einsum("ki, i -> ", C_c_inv, total_residual)
    denom = np.sum(
        (1 / gm_params_df.loc[obs_stations].sigma_between.values**2)
        + np.sum(C_c_inv, axis=1)
    )
    between_residual = numerator / denom

    # Compute the within-event residual
    within_residual = total_residual - between_residual

    # Define the within-event residual distribution
    # Equation 5 in Bradley 2014
    within_residual_cov = np.full(
        (rel_stations.size, rel_stations.size), fill_value=np.nan
    )
    within_residual_cov[1:, 1:] = C_c
    within_residual_cov[0, 0:] = within_residual_cov[0:, 0] = (
        R.loc[rel_stations, site_id].values
        * gm_params_df.loc[site_id, "sigma_within"]
        * gm_params_df.loc[rel_stations, "sigma_within"].values
    )
    within_residual_cov = pd.DataFrame(
        data=within_residual_cov, index=rel_stations, columns=rel_stations
    )
    # Sanity check, diagonal terms are just sigma_within**2
    assert np.all(
        np.isclose(
            np.diag(within_residual_cov.values),
            gm_params_df.loc[rel_stations, "sigma_within"].values ** 2,
        )
    )

    # Define the conditional within-event distribution
    cond_within_residual_mu = np.einsum(
        "i, ij, j -> ",
        within_residual_cov.values[0, 1:],
        C_c_inv,
        within_residual.values,
    )
    cond_within_residual_sigma = np.sqrt(
        max(
            gm_params_df.loc[site_id, "sigma_within"] ** 2
            - np.einsum(
                "i, ij, j -> ",
                within_residual_cov.values[0, 1:],
                C_c_inv,
                within_residual_cov.values[1:, 0],
            ),
            0.0,
        )
    )

    # Define the conditional lnIM distriubtion
    cond_lnIM_mu = (
        gm_params_df.loc[site_id, "mu"] + between_residual + cond_within_residual_mu
    )
    cond_lnIM_sigma = cond_within_residual_sigma

    return cond_lnIM_mu, cond_lnIM_sigma



nzmgdb_ffp = Path("/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v4.3/custom/mod_ground_motion_im_table_rotd50_flat.csv")
emp_gm_params_ffp = Path("/Users/claudy/dev/work/data/sim_ranking/emp_gm_params/nzgmdb_v4p3/emp_gm_params.parquet")
site_table_ffp = Path("/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v4.3/Tables/site_table.csv")

site_df = pd.read_csv(site_table_ffp, index_col="sta")
obs_df = pd.read_csv(nzmgdb_ffp, dtype={"evid": str, "loc": str}, index_col="record_id")
emp_gm_params = pd.read_parquet(emp_gm_params_ffp)

event = "3468575"
im = "pSA_0.1"
site = "CCCC"
im_cols = [f"{im}_mean", f"{im}_std_Total", f"{im}_std_Inter", f"{im}_std_Intra"]

event_emp_gm_params = emp_gm_params.loc[emp_gm_params.event_id == event].set_index("site_id")
event_emp_gm_params = event_emp_gm_params[im_cols].rename(columns=dict(zip(im_cols, ["mu", "sigma_total", "sigma_between", "sigma_within"])))

obs_lnIM_series = np.log(obs_df.loc[obs_df.evid == event].set_index("sta")[im])
# Drop sites for which we don't have empirical GM parameters
obs_lnIM_series = obs_lnIM_series[obs_lnIM_series.index.isin(event_emp_gm_params.index)]
# Remove the location of interest from the observations
obs_lnIM_series = obs_lnIM_series[obs_lnIM_series.index != site]

sites = np.unique(np.concatenate((event_emp_gm_params.index.values.astype(str), obs_lnIM_series.index.values.astype(str))))
dist_matrix = calculate_distance_matrix(sites, site_df)
R = get_im_corr(im, sites, dist_matrix)

cond_lnIM_mu, cond_lnIM_sigma = compute_cond_lnIM_dist(site, event_emp_gm_params, obs_lnIM_series, R)
