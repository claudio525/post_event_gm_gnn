import multiprocessing as mp
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

import labelled_data_array as lda
from tqdm import tqdm

import ml_tools as mlt

from .data_classes import ObservedData, DynamicLBSiteCorrelationsData
from . import ml
from . import data
from . import utils
from . import constants


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
    # corr_data = LBSiteCorrelationData.from_dist_matrix(dist_matrix, constants.PSA_KEYS)
    corr_data = DynamicLBSiteCorrelationsData(dist_matrix)

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
                constants.PSA_KEYS,
                20,
                None,
                True,
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
                        constants.PSA_KEYS,
                        20,
                        None,
                        True,
                    )
                    for cur_event in events
                ],
            )

    result_df = pd.concat(results, axis=0)
    result_df["cv_iter"] = cv_iter

    (out_dir := gnn_result_dir / "cim_results").mkdir(parents=False, exist_ok=True)
    prefix = "train" if on_train else "val"
    result_df.to_parquet(out_dir / f"{prefix}_results.parquet")


def predict_event_cIM(
    event_id: str,
    sites_df: pd.DataFrame,
    obs_data: ObservedData,
    obs_sites: np.ndarray[str],
    gm_params_df: pd.DataFrame,
    int_sites: np.ndarray[str],
    output_ffp: Path,
    allow_self: bool = False,
):
    """
    Predict conditional IM distributions for a given event.

    Parameters:
    -----------
    event_id : str
        Identifier for the event to predict.
    sites_df : pd.DataFrame
        Sites for which to predict the conditional IM distributions.
    obs_data : ObservedData
        Observation data used to compute the conditional IM distributions.
    gm_params_df : pd.DataFrame
        Marginal GM parameters for all relevant sites.
    int_sites : np.ndarray[str]
        Sites for which to predict the conditional IM distributions.
    output_ffp : Path
        File path to save the resulting DataFrame in Parquet format.
    allow_self : bool, optional
        If True, allows the site of interest to be used
        as an observation site for itself. Default is False.
    """
    assert np.all(gm_params_df["event_id"].values == event_id), "Mismatch in event_id"
    all_sites = np.union1d(int_sites, obs_sites)
    assert np.all(
        mlt.array_utils.pandas_isin(all_sites, gm_params_df["site_id"])
    ), "Missing GM parameters"

    # Compute distance matrix and correlations
    comb_site_df = pd.concat(
        [
            obs_data.site_df[["lon", "lat"]],
            sites_df.loc[
                ~np.isin(sites_df.index, obs_data.site_df.index),
                ["lon", "lat"],
            ],
        ],
        axis=0,
    )
    print(f"Computing distance matrix for {len(all_sites)} sites")
    dist_matrix = utils.calculate_distance_matrix(all_sites, comb_site_df)
    corr_data = DynamicLBSiteCorrelationsData(dist_matrix)

    # Compute conditional IM distributions
    print(f"Computing conditional IM distributions for {len(int_sites)} sites")
    result_df = run_event_cim(
        event_id,
        int_sites,
        obs_data,
        gm_params_df.set_index("site_id"),
        corr_data,
        dist_matrix,
        constants.PSA_KEYS,
        20,
        obs_sites=obs_sites,
        allow_int_as_obs=True,
        verbose=True,
        allow_self=allow_self,
    )

    result_df[["lon", "lat"]] = comb_site_df.loc[
        result_df["site_int"], ["lon", "lat"]
    ].values
    result_df.to_parquet(output_ffp)


def _get_im_name_mapping_dict(im: str):
    """Helper function"""
    return {
        f"{im}_mean": "mu",
        f"{im}_std_Total": "sigma_total",
        f"{im}_std_Inter": "sigma_between",
        f"{im}_std_Intra": "sigma_within",
    }


def run_event_cim(
    event_id: str,
    int_sites: np.ndarray[str],
    obs_data: ObservedData,
    gm_params_df: pd.DataFrame,
    corr_data: DynamicLBSiteCorrelationsData,
    dist_matrix: pd.DataFrame,
    ims: list[str],
    n_obs_sites: int,
    obs_sites: np.ndarray[str] = None,
    allow_int_as_obs: bool = False,
    verbose: bool = False,
    allow_self: bool = False,
):
    """
    Computes the conditional IM distribution for
    the given event and sites of interest.

    Parameters
    ----------
    event_id: str
        Event ID
    int_sites: array of strings
        Site for which to compute the conditional
         IM distribution
    obs_data: ObservedData
        Available observed data
    gm_params_df: dataframe
        Ground motion parameters
        Index has to be site_id
    corr_data: DynamicLBSiteCorrelationsData
        Correlation data
    dist_matrix: dataframe
        Distance matrix
    ims: list of strings
        IMs for which to compute the conditional
         IM distributions
    n_obs_sites: int
        Number of observation sites to use
    obs_sites: array of strings, optional
        Sites to use as observation sites.
        If None, then uses all available sites
        for the event.
    allow_int_as_obs: bool, optional
        If True then allows sites of interest to be used
        as observation sites for other sites of interest
        when computing the conditional IM distribution
        Default is False.
    verbose: bool, optional
        If True, then prints progress
    allow_self: bool, optional
        If True, then allows the site of interest to be used
        as an observation site for itself.
        Default is False.

    Returns
    -------
    dataframe:
        The conditional IM distribution
        for the sites of interest and specified IMs
    """
    assert (allow_self and allow_int_as_obs) or not allow_self

    obs_sites = obs_data.event_sites[event_id] if obs_sites is None else obs_sites
    if not allow_int_as_obs:
        obs_sites = obs_sites[~np.isin(obs_sites, int_sites)]

    # Get the mask for the observations sites,
    # based on the observation sites selection algorithm
    obs_site_mask = get_observation_sites_mask(
        n_obs_sites,
        int_sites,
        obs_sites,
        dist_matrix,
        allow_self=allow_self,
    )

    # Get the observed data for the observation sites
    obs_df = obs_data.get_event_data(event_id, obs_sites)[ims]
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
    result_df["obs_sites"] = [
        obs_site_mask.sel[cur_site, :, :]
        .index[np.any(obs_site_mask.sel[cur_site, :, :], axis=1)]
        .values
        for cur_site in int_sites
    ]
    for cur_site_int in tqdm(int_sites, disable=not verbose):
        for cur_im in ims:
            # Get the observation sites
            cur_obs_sites = obs_sites[
                obs_site_mask.sel[cur_site_int, :, cur_im].values
            ].tolist()
            
            # Get all relevant sites
            if allow_self and cur_site_int in cur_obs_sites:
                cur_rel_sites = cur_obs_sites
            else:
                cur_rel_sites = cur_obs_sites + [cur_site_int]

            # No observation sites, use marginal distribution
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
            cur_R = corr_data.get_im_corr(cur_im, cur_rel_sites)

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
                cur_site_int,
                cur_gm_params_df,
                cur_obs_data,
                cur_R,
                allow_self=allow_self,
            )

            result_df.loc[cur_site_int, f"{cur_im}_cond_mean"] = cond_lnIM_mu
            result_df.loc[cur_site_int, f"{cur_im}_cond_std"] = cond_lnIM_sigma
            result_df.loc[cur_site_int, f"{cur_im}_n_obs_sites"] = len(cur_obs_sites)

            assert (
                not allow_self
                or cur_site_int not in obs_sites
                or (
                    np.isclose(cond_lnIM_sigma, 0.0, atol=1e-4)
                    and np.isclose(cond_lnIM_mu, cur_obs_data.loc[cur_site_int])
                )
            )

        result_df.at[cur_site_int, "n_obs_sites"] = result_df.at[
            cur_site_int, "obs_sites"
        ].size

    # Add the actual observed IMs at the site of interests
    int_sites_with_obs = int_sites[np.isin(int_sites, obs_data.event_sites[event_id])]
    result_df.loc[int_sites_with_obs, ims] = np.log(
        obs_data.get_event_data(event_id, int_sites_with_obs).loc[
            int_sites_with_obs, ims
        ]
    )

    result_df["site_int"] = result_df.index.values.astype(str)
    result_df["event_id"] = event_id

    # Fix the index
    result_df.index = mlt.array_utils.numpy_str_join(
        "_", event_id, result_df.index.values.astype(str)
    )

    return result_df


def get_observation_sites_mask(
    n_obs_sites: int,
    int_sites: np.ndarray,
    obs_sites: np.ndarray,
    distance_matrix: pd.DataFrame,
    allow_self: bool = False,
):
    """
    Returns the observation site filter for
    each site of interest using nearest neighbor.

    Parameters
    ----------
    n_obs_sites: int
        Number of observation sites to use.
    int_sites: np.ndarray
        Array of sites of interest.
    obs_sites: np.ndarray
        Array of observation sites.
    distance_matrix: pd.DataFrame
        DataFrame containing the distance matrix between sites.
    allow_self: bool, optional
        If True, allows the site of interest to be used
        as an observation site for itself.
        Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean values indicating the
        selected observation sites for each site of interest.
    """
    # Less observation sites than required
    # i.e. use all available
    if obs_sites.size <= n_obs_sites:
        obs_station_mask_df = pd.DataFrame(
            index=int_sites, columns=obs_sites, data=True
        )
    # More than required use nearest neighbor to select
    else:
        # Get observation sites such that
        # minimum number of observations sites is satisfied
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
    if not allow_self:
        int_obs_sites = int_sites[np.isin(int_sites, obs_sites)]
        for cur_station in int_obs_sites:
            obs_station_mask_df.loc[cur_station, cur_station] = False

    return obs_station_mask_df


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
