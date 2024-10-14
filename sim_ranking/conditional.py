import multiprocessing as mp
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import labelled_data_array as lda
from tqdm import tqdm

import ml_tools as mlt

from .data_classes import ObservedData, LBSiteCorrelationData
from . import ml
from . import data
from . import utils


def run_cim_for_CV_GNN(
    gnn_cv_results_dir: Path,
    emp_gm_params_ffp: Path,
    n_procs: int = 1,
    include_train: bool = False,
):
    """
    Runs the empirical conditional IM method
    for each CV iteration GNN result.

    Parameters
    ----------
    gnn_cv_results_dir: Path
        Directory containing the GNN CV results
    emp_gm_params_ffp: Path
        Path to the empirical GM parameters
    n_procs: int
        Number of processes to use
    include_train: bool, optional
        Also run for the training data
    """
    cv_dirs = [
        cur_dir
        for cur_dir in gnn_cv_results_dir.iterdir()
        if cur_dir.is_dir() and cur_dir.name.startswith("cv_")
    ]

    for cur_dir in tqdm(cv_dirs, desc="Processing CV results"):
        run_cim_for_GNN(
            cur_dir,
            emp_gm_params_ffp,
            on_train=False,
            n_procs=n_procs,
            verbose=False,
            cv_iter=cur_dir.stem,
        )
        if include_train:
            run_cim_for_GNN(
                cur_dir,
                emp_gm_params_ffp,
                on_train=True,
                n_procs=n_procs,
                verbose=False,
                cv_iter=cur_dir.stem,
            )

    # Create combination of validation results
    results_df = pd.concat(
        [
            pd.read_parquet(cur_dir / "cim_results" / "val_results.parquet")
            for cur_dir in cv_dirs
        ],
        axis=0,
    )

    (comb_out_dir := gnn_cv_results_dir / "cim_results").mkdir(
        parents=False, exist_ok=True
    )
    results_df.to_parquet(comb_out_dir / "val_results.parquet")


def run_cim_for_GNN(
    gnn_result_dir: Path,
    emp_gm_params_ffp: Path,
    on_train: bool = False,
    n_procs: int = 1,
    verbose: bool = True,
    cv_iter: int = None,
):
    """
    Runs the empirical conditional IM method
    for the same scenarios as the specificed
    GNN results.

    Results are stored in the GNN result directory.

    Parameters
    ----------
    gnn_result_dir: Path
        Directory containing the GNN results
    emp_gm_params_ffp: Path
        Path to the empirical GM parameters
    on_train: bool, optional
        If True then runs for the training data,
        otherwise runs for the validation data
    n_procs: int, optional
    verbose: bool, optional
    cv_iter: int, optional
        CV iteration number.
        Only relevant when this is being run for CV results.
    """
    run_config = ml.RunConfig.from_yaml(gnn_result_dir / "run_config.yaml")
    nzgmdb_flat_ffp = run_config.obs_data_ffp

    # Get the events
    gnn_results = (
        pd.read_parquet(gnn_result_dir / "train_results.parquet")
        if on_train
        else pd.read_parquet(gnn_result_dir / "val_results.parquet")
    )
    events = gnn_results.event_id.unique().astype(str)

    # Load the observed data
    obs_data = data.load_obs_nzgmdb(nzgmdb_flat_ffp)

    # Load GM params
    gm_params_df = pd.read_parquet(emp_gm_params_ffp)

    # Get the correlation data
    dist_matrix = utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    corr_data = LBSiteCorrelationData.from_dist_matrix(dist_matrix, run_config.ims)

    # Get the sites of interest for each event
    event_int_sites = gnn_results.groupby("event_id").site_int.unique().to_dict()

    results = []
    if n_procs == 1:
        for cur_event in tqdm(events, desc="Processing events", disable=not verbose):
            cur_result = run_event_cim(
                cur_event,
                event_int_sites[cur_event],
                obs_data,
                gm_params_df.loc[gm_params_df.event_id == cur_event].set_index(
                    "site_id"
                ),
                corr_data,
                dist_matrix,
                run_config.ims,
                20,
                True if on_train else False,
            )
            results.append(cur_result)
    else:
        with mp.Pool(n_procs) as p:
            results = p.starmap(
                run_event_cim,
                [
                    (
                        cur_event,
                        event_int_sites[cur_event],
                        obs_data,
                        gm_params_df.loc[gm_params_df.event_id == cur_event].set_index(
                            "site_id"
                        ),
                        corr_data,
                        dist_matrix,
                        run_config.ims,
                        20,
                        True if on_train else False,
                    )
                    for cur_event in events
                ],
            )

    result_df = pd.concat(results, axis=0)
    result_df["cv_iter"] = cv_iter

    (out_dir := gnn_result_dir / "cim_results").mkdir(parents=False, exist_ok=True)
    prefix = "train" if on_train else "val"
    result_df.to_parquet(out_dir / f"{prefix}_results.parquet")


def _get_im_name_mapping_dict(im: str):
    """Helper function"""
    return {
        f"{im}_mean": "mu",
        f"{im}_std_Total": "sigma_total",
        f"{im}_std_Inter": "sigma_between",
        f"{im}_std_Intra": "sigma_within",
    }


def run_event_cim(
    event: str,
    int_sites: np.ndarray[str],
    obs_data: ObservedData,
    gm_params_df: pd.DataFrame,
    corr_data: LBSiteCorrelationData,
    dist_matrix: pd.DataFrame,
    ims: list[str],
    n_obs_sites: int,
    allow_int_as_obs: bool = False,
):
    """
    Computes the conditional IM distribution for
    the given event and sites of interest.

    Parameters
    ----------
    event: str
        Event ID
    int_sites: array of strings
        Site for which to compute the conditional
         IM distribution
    obs_data: ObservedData
        Available observed data
    gm_params_df: dataframe
        Ground motion parameters
        Index has to be site_id
    corr_data: LBSiteCorrelationData
        Correlation data
    dist_matrix: dataframe
        Distance matrix
    ims: list of strings
        IMs for which to compute the conditional
         IM distributions
    n_obs_sites: int
        Number of observation sites to use
    allow_int_as_obs: bool, optional
        If True then allows sites of interest to be used
        as observation sites for the conditional IM distribution

    Returns
    -------
    dataframe:
        The conditional IM distribution
        for the sites of interest and specified IMs
    """
    obs_sites = obs_data.event_sites[event]
    if not allow_int_as_obs:
        obs_sites = obs_sites[~np.isin(obs_sites, int_sites)]

    # Get the mask for the observations sites,
    # based on the observation sites selection algorithm
    obs_site_mask = get_observation_sites_mask(
        n_obs_sites, int_sites, obs_sites, dist_matrix
    )

    # Get the observed data for the observation sites
    obs_df = obs_data.get_event_data(event, obs_sites)[ims]
    obs_df_nan_mask = obs_df.isna()
    assert obs_df.index.is_unique
    assert obs_site_mask.columns.equals(obs_df_nan_mask.index)

    # Create the combined observation site mask
    # which varies with period due to the application
    # of fmin filtering
    obs_site_mask = lda.LabelledDataArray(
        values=(obs_site_mask.values[:, :, None] & ~obs_df_nan_mask.values[None, :, :]),
        axis_labels=(int_sites, obs_sites, ims),
        axis_names=("int_sites", "obs_sites", "ims"),
    )
    assert np.all(obs_site_mask.labels["obs_sites"] == obs_sites)

    result_cols = (
        ims
        + [f"{cur_im}_cond_mean" for cur_im in ims]
        + [f"{cur_im}_cond_std" for cur_im in ims]
    )
    result_df = pd.DataFrame(index=int_sites, columns=result_cols, dtype=float)
    result_df["n_obs_sites"] = -1
    # result_df["obs_sites"] = None
    result_df["obs_sites"] = [
        obs_site_mask.sel[cur_site, :, :]
        .index[np.any(obs_site_mask.sel[cur_site, :, :], axis=1)]
        .values
        for cur_site in int_sites
    ]
    for cur_site_int in int_sites:
        for cur_im in ims:
            # Get the observation sites
            cur_obs_sites = obs_sites[
                obs_site_mask.sel[cur_site_int, :, cur_im].values
            ].tolist()
            cur_rel_sites = cur_obs_sites + [cur_site_int]

            # No observation sites,
            # use marginal distribution
            if len(cur_obs_sites) == 0:
                result_df.loc[cur_site_int, f"{cur_im}_cond_mean"] = gm_params_df.loc[
                    cur_site_int, f"{cur_im}_mean"
                ]
                result_df.loc[cur_site_int, f"{cur_im}_cond_std"] = gm_params_df.loc[
                    cur_site_int, f"{cur_im}_std_Total"
                ]
                result_df.loc[cur_site_int, f"{cur_im}_n_obs_sites"] = 0
                continue

            # Get the correlation matrix
            cur_R = corr_data.corr_data.sel[:, :, cur_im].loc[
                cur_rel_sites, cur_rel_sites
            ]

            # Get the GM parameters for the relevant sites
            # and put into correct format
            cur_mapping_dict = _get_im_name_mapping_dict(cur_im)
            cur_gm_params_df = gm_params_df.loc[
                cur_rel_sites, cur_mapping_dict.keys()
            ].rename(columns=cur_mapping_dict)

            # Get the observed data
            cur_obs_data = np.log(obs_df.loc[cur_obs_sites, cur_im])
            assert cur_obs_data.isna().sum() == 0

            # Compute the conditional IM distribution
            cond_lnIM_mu, cond_lnIM_sigma = compute_cond_lnIM_dist(
                cur_site_int, cur_gm_params_df, cur_obs_data, cur_R
            )

            result_df.loc[cur_site_int, f"{cur_im}_cond_mean"] = cond_lnIM_mu
            result_df.loc[cur_site_int, f"{cur_im}_cond_std"] = cond_lnIM_sigma
            result_df.loc[cur_site_int, f"{cur_im}_n_obs_sites"] = len(cur_obs_sites)

        result_df.at[cur_site_int, "n_obs_sites"] = result_df.at[
            cur_site_int, "obs_sites"
        ].size

    # Add the actual observed IMs at the site of interests
    result_df.loc[int_sites, ims] = np.log(
        obs_data.get_event_data(event, int_sites).loc[int_sites, ims]
    )

    result_df["site_int"] = result_df.index.values.astype(str)
    result_df["event_id"] = event

    # Fix the index
    result_df.index = mlt.array_utils.numpy_str_join(
        "_", event, result_df.index.values.astype(str)
    )

    return result_df


def get_observation_sites_mask(
    n_obs_sites: int,
    int_sites: np.ndarray,
    obs_sites: np.ndarray,
    distance_matrix: pd.DataFrame,
):
    """
    Returns the observation site filter for
    each site of interest using nearest neighbor.

    Parameters
    ----------
    n_obs_sites
    int_sites
    obs_sites
    distance_matrix

    Returns
    -------

    """
    # Less observation sites than required
    # i.e. use all available
    if obs_sites.size <= n_obs_sites:
        obs_station_mask_df = pd.DataFrame(
            index=int_sites, columns=obs_sites, data=True
        )
    # More than required use nearest neighbor to select
    else:
        # Get observation sites such that minimum number of observations sites is satisfied
        neigh = NearestNeighbors(
            n_neighbors=n_obs_sites + 1, radius=150, metric="precomputed", n_jobs=1
        )
        neigh.fit(distance_matrix.loc[obs_sites, obs_sites])
        n_neigh_ind = neigh.kneighbors(
            distance_matrix.loc[int_sites, obs_sites], return_distance=False
        )

        # Convert to a mask
        n_neigh_mask = np.zeros((int_sites.size, obs_sites.size), dtype=bool)
        np.put_along_axis(n_neigh_mask, n_neigh_ind, True, axis=1)

        # Combine & Create dataframe
        obs_station_mask_df = pd.DataFrame(
            index=int_sites, columns=obs_sites, data=n_neigh_mask
        )

    # Don't include stations of interest
    int_obs_sites = int_sites[np.isin(int_sites, obs_sites)]
    for cur_station in int_obs_sites:
        obs_station_mask_df.loc[cur_station, cur_station] = False

    return obs_station_mask_df


def compute_cond_lnIM_dist(
    site_id: str,
    gm_params_df: pd.DataFrame,
    obs_lnIM_series: pd.Series,
    R: pd.DataFrame,
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
    assert site_id in gm_params_df.index and site_id not in obs_stations

    # Relevant stations (Observation sites & Sites of interest)
    rel_stations = np.concatenate(([site_id], obs_stations))
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

    if np.all(gm_params_df.sigma_between == 0):
        # Between-event residual is zero
        # when the GM parameters are computed
        # from simulation realisations
        between_residual = 0.0
    else:
        # Compute the between event-residual using the observation stations
        # First part of Equation 3 numerator is just row-wise sum of inverse C_c
        numerator = np.einsum("ki, i -> ", C_c_inv, total_residual)
        denom = np.sum(
            (1 / gm_params_df.loc[obs_stations].sigma_between.values ** 2)
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
        gm_params_df.loc[site_id, "sigma_within"] ** 2
        - np.einsum(
            "i, ij, j -> ",
            within_residual_cov.values[0, 1:],
            C_c_inv,
            within_residual_cov.values[1:, 0],
        )
    )

    # Define the conditional lnIM distriubtion
    cond_lnIM_mu = (
        gm_params_df.loc[site_id, "mu"] + between_residual + cond_within_residual_mu
    )
    cond_lnIM_sigma = cond_within_residual_sigma

    return cond_lnIM_mu, cond_lnIM_sigma
