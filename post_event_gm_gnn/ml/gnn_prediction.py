"""
Module for performing predictions using an already trained GNN
for post-event GM estimation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
from tqdm import tqdm

import ml_tools as mlt


from . import data as ml_data
from . import features
from . import gnn_gm
from .. import utils
from .. import constants


def predict_event(
    model_dir: Path,
    event_id: str,
    event_info: pd.Series,
    pred_site_df: pd.DataFrame,
    obs_site_df: pd.DataFrame,
    obs_event_site_df: pd.DataFrame,
    obs_im_data: pd.DataFrame,
    emp_gm_params: pd.DataFrame = None,
    obs_emp_res_df: pd.DataFrame = None,
    allow_self: bool = True,
    verbose: bool = True,
    device: str = None,
):
    """
    Perform predictions using a trained GNN model for a specific event.

    Parameters
    ----------
    model_dir : Path
        Directory containing the trained model and configuration files.
    event_id : str
        Identifier for the event.
    event_info : pd.Series
        Series containing information about the event.
    pred_site_df : pd.DataFrame
        DataFrame containing prediction site information.
        Also needs to include the required event-site information, such as rrup

        Required columns:
        - lon: Longitude of the site
        - lat: Latitude of the site
        - vs30: Vs30 value of the site in m/s
        - z1p0: Z1.0 value of the site in metres
        - z2p5: Z2.5 value of the site in kilometres
        - rrup: R_Rup in kilometres

    obs_site_df : pd.DataFrame
        DataFrame containing observation site information.
    obs_event_site_df : pd.DataFrame
        DataFrame containing event-site information for observation sites.
    obs_im_data : pd.DataFrame
        DataFrame containing intensity measure (IM) data for observation sites.
    emp_gm_params : pd.DataFrame, optional
        DataFrame containing empirical GM parameters for the event and all sites.
        Only used if run_config.use_emp_gm_model is True.
    obs_emp_res_df : pd.DataFrame, optional
        DataFrame containing empirical GM residuals for the observation sites.
        Only used if run_config.use_emp_gm_model is True.
    allow_self : bool
        Whether to allow the prediction site to be one of the observation sites.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the prediction results with columns for event_id, site_int,
        obs_sites, predicted IMs, and their standard deviations.
    """
    run_config = gnn_gm.RunConfig.from_yaml(model_dir / "run_config.yaml")
    if run_config.use_emp_gm_model and emp_gm_params is None:
        raise ValueError(
            "emp_gm_params must be provided if run_config.use_emp_gm_model is True"
        )

    model = torch.load(model_dir / "model.pt")
    model.to(device or run_config.device)
    model.eval()

    # Scale the IM data
    obs_im_data = obs_im_data.loc[
        obs_im_data.event_id == event_id, run_config.ims + ["event_id", "site_id"]
    ].copy()
    obs_im_data[run_config.ims] = (
        np.log(obs_im_data[run_config.ims])
        - run_config.im_scale_params["mean"][run_config.ims]
    ) / run_config.im_scale_params["std"][run_config.ims]

    obs_sites = obs_im_data.site_id.values.astype(str)
    int_sites = pred_site_df.index.values.astype(str)
    all_sites = np.unique(np.concatenate((obs_sites, int_sites)))

    # Create combined site dataframe
    comb_site_df = pd.concat(
        (
            obs_site_df[["lon", "lat", "vs30", "z1p0", "z2p5"]].copy(),
            pred_site_df.loc[
                ~np.isin(pred_site_df.index, obs_site_df.index),
                ["lon", "lat", "vs30", "z1p0", "z2p5"],
            ],
        ),
        axis=0,
    )
    assert np.all(
        mlt.array_utils.pandas_isin(all_sites, comb_site_df.index.values.astype(str))
    )

    pred_event_site_df = pred_site_df.loc[:, "rrup"].copy().to_frame()
    pred_event_site_df["event_id"] = event_id
    pred_event_site_df["site_id"] = pred_event_site_df.index
    pred_event_site_df.index = mlt.array_utils.numpy_str_join(
        "_",
        pred_event_site_df["event_id"].values.astype(str),
        pred_event_site_df["site_id"].values.astype(str),
    )

    comb_event_site_df = obs_event_site_df.loc[
        obs_event_site_df.event_id == event_id
    ].copy()
    comb_event_site_df = pd.concat(
        (
            comb_event_site_df,
            pred_event_site_df.loc[
                ~np.isin(pred_site_df.index, comb_event_site_df.site_id)
            ],
        ),
        axis=0,
    )

    if verbose:
        print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(all_sites, comb_site_df).astype(
        np.float32
    )

    event_sites = {event_id: all_sites}
    event_int_sites = {event_id: int_sites}

    if verbose:
        print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites,
        event_info.to_frame().T,
        comb_site_df,
        comb_event_site_df,
        run_config,
        constants.SCALAR_FEATURE_KEYS,
        dist_matrix,
        verbose=verbose,
    )

    # Site combinations
    if verbose:
        print("Computing site combinations")
    event_site_combs, event_used_sites = ml_data.compute_site_combinations(
        event_sites,
        event_int_sites,
        [event_id],
        dist_matrix,
        obs_sites,
        int_sites,
        run_config.max_dist,
        run_config.closest_max_dist,
        run_config.max_n_obs_sites,
        run_config.min_n_obs_sites,
        allow_self=allow_self,
    )
    site_combs, used_sites = event_site_combs[event_id], event_used_sites[event_id]

    if verbose:
        print("Creating scalar feature dataframes")
    event_scalar_feature_dfs, _ = ml_data.create_event_scalar_feature_dfs(
        event_used_sites, scalar_features, event_site_combs
    )
    scalar_feature_df = event_scalar_feature_dfs[event_id]

    # Generate the graph data
    graph_data = []
    for cur_int_site_ix in np.unique(site_combs[:, 0]):
        cur_graph_data = _create_graph_data(
            event_id,
            cur_int_site_ix,
            site_combs,
            used_sites,
            scalar_feature_df,
            obs_im_data.set_index("site_id"),
            run_config,
            obs_emp_res_df=(
                obs_emp_res_df.loc[obs_emp_res_df.event_id == event_id].set_index(
                    "site_id"
                )
                if obs_emp_res_df is not None
                else None
            ),
        )
        graph_data.append(cur_graph_data)

    result_df = _run_prediction(
        model,
        graph_data,
        run_config,
        emp_gm_params=emp_gm_params,
        verbose=verbose,
        device=device,
    )

    # Add site information
    result_df = pd.merge(
        result_df, comb_site_df, left_on="site_int", right_index=True, how="left"
    )
    return result_df


def _create_graph_data(
    event_id: str,
    site_int_ix: int,
    site_combs: np.ndarray[float],
    sites: np.ndarray[str],
    scalar_feature_df: pd.DataFrame,
    im_data: pd.DataFrame,
    run_config: gnn_gm.RunConfig,
    obs_emp_res_df: pd.DataFrame = None,
):
    """
    Creates the graph data for the specified scenario

    Note: sites must match the sites used for site_combs!
    """
    cur_site_int = sites[site_int_ix]
    cur_obs_sites = sites[site_combs[site_combs[:, 0] == site_int_ix, 1]]
    cur_site_combs = mlt.array_utils.numpy_str_join("_", cur_site_int, cur_obs_sites)

    # Create the site_int node features
    site_int_features = scalar_feature_df.loc[
        cur_site_combs, run_config.graph_feature_keys["site_int"]
    ].values[0]

    # Get the residuals or IM values
    if run_config.use_emp_gm_model:
        obs_sites_im_values = (
            obs_emp_res_df.loc[cur_obs_sites, run_config.ims].replace(np.nan, 99).values
        )
    else:
        obs_sites_im_values = (
            im_data.loc[cur_obs_sites, run_config.ims].replace(np.nan, 99).values
        )

    if (
        run_config.graph_feature_keys["site_obs"] is not None
        and len(run_config.graph_feature_keys["site_obs"]) > 0
    ):
        # Create the site_obs node features
        obs_sites_features = scalar_feature_df.loc[
            cur_site_combs,
            run_config.graph_feature_keys["site_obs"],
        ].values
        # Add the IM values
        obs_sites_features = np.concatenate(
            (
                obs_sites_features,
                obs_sites_im_values,
            ),
            axis=1,
        )
    else:
        obs_sites_features = obs_sites_im_values

    # Create the edge features
    edge_features = scalar_feature_df.loc[
        cur_site_combs, run_config.graph_feature_keys["edge"]
    ].values

    graph_data = gdata.HeteroData()
    graph_data["site_int"].x = torch.tensor(site_int_features, dtype=torch.float32)[
        None, :
    ]
    graph_data["site_obs"].x = torch.tensor(obs_sites_features, dtype=torch.float32)

    graph_data["site_obs", "informs", "site_int"].edge_index = torch.tensor(
        [[ix, 0] for ix, _ in enumerate(cur_obs_sites)],
        dtype=torch.long,
    ).T
    graph_data["site_obs", "informs", "site_int"].edge_attr = torch.tensor(
        edge_features, dtype=torch.float32
    )

    graph_data["site_obs", "self_loop", "site_obs"].edge_index = torch.tensor(
        [[ix, ix] for ix in range(len(cur_obs_sites))], dtype=torch.long
    ).T

    assert (
        graph_data["site_obs", "informs", "site_int"].edge_index.shape[1]
        == graph_data["site_obs", "informs", "site_int"].edge_attr.shape[0]
    )

    graph_data["metadata"] = {
        "sc_id": f"{event_id}_{cur_site_int}",
        "event": event_id,
        "site_int": cur_site_int,
        "obs_sites": cur_obs_sites,
    }

    return graph_data


def _run_prediction(
    model: nn.Module,
    graph_data: list[gdata.HeteroData],
    run_config: gnn_gm.RunConfig,
    emp_gm_params: pd.DataFrame = None,
    verbose: bool = True,
    device: str = None,
):
    pred_im_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
    pred_im_std_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_std")
    pred_res_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_res")

    results = []
    loader = gloader.DataLoader(graph_data, batch_size=1024, shuffle=False)
    for cur_batch in tqdm(loader, desc="Running predictions", disable=not verbose):
        cur_batch = cur_batch.to(device or run_config.device)

        # Get predictions
        torch_pred_mean, torch_pred_ln_std, *_ = model(cur_batch)
        pred_std = torch.exp(torch_pred_ln_std).cpu().numpy(force=True)
        pred_mean = torch_pred_mean.cpu().numpy(force=True)

        cur_result_df = pd.DataFrame(
            {
                "event_id": cur_batch["metadata"]["event"],
                "site_int": cur_batch["metadata"]["site_int"],
                "obs_sites": cur_batch["metadata"]["obs_sites"],
            }
        )
        cur_result_df.index = mlt.array_utils.numpy_str_join(
            "_",
            cur_result_df["event_id"].values.astype(str),
            cur_result_df["site_int"].values.astype(str),
        )
        cur_result_df["n_obs_sites"] = [
            cur_obs_sites.size for cur_obs_sites in cur_result_df.obs_sites
        ]

        # GNN Residual Model
        if run_config.use_emp_gm_model:
            cur_result_df.loc[:, pred_res_keys] = pred_mean
            cur_result_df.loc[:, pred_im_std_keys] = pred_std

            # IM value is simply GMM mean + GNN predicted residual
            pred_ln_im_mean = (
                emp_gm_params.loc[
                    cur_result_df.index, utils.get_emp_gm_mean_im_keys(run_config.ims)
                ].values
                + pred_mean
            )
            cur_result_df.loc[:, pred_im_keys] = pred_ln_im_mean
        # GNN IM Model
        else:
            # Revert the IM scaling
            if run_config.scale_IMs:
                pred_ln_im_mean, pred_ln_im_std = gnn_gm.revert_im_scaling(
                    pred_mean, run_config, pred_std
                )

            cur_result_df[pred_im_keys] = pred_ln_im_mean
            cur_result_df[pred_im_std_keys] = pred_ln_im_std

        results.append(cur_result_df)

    result_df = pd.concat(results, axis=0)
    result_df.index = mlt.array_utils.numpy_str_join(
        "_",
        result_df["event_id"].values.astype(str),
        result_df["site_int"].values.astype(str),
    )

    return result_df


def get_variable_att_model_predictions(
    run_config: gnn_gm.RunConfig,
    att_model: nn.Module,
    device: str,
    x_var: str,
    site_dist: float | None = None,
    soi_vs30: float | None = None,
    vs30_diff: float | None = None,
    ln_vs30_diff: float | None = None,
    angular_distance: float | None = None,
):
    """
    Get the predictions of the attention model (of the first convolutional layer)
    for a range of values of the specified variable. 
    The other variables are kept constant and can be specified as arguments.
    """
    input_tensor = torch.full(
        (1000, len(run_config.graph_feature_keys["edge"])),
        fill_value=torch.nan,
        device=device,
        dtype=torch.float32,
    )
    if x_var == "ln_vs30_diff":
        variable_input = torch.linspace(-2, 2, 1000)
    else:
        variable_input = torch.linspace(-1, 1, 1000)

    for i, var in enumerate(run_config.graph_feature_keys["edge"]):
        if var == x_var:
            input_tensor[:, i] = variable_input

        elif var == "dist":
            input_tensor[:, i] = features.scale_site_to_site_distances(
                site_dist, run_config.max_dist
            )
        elif var == "vs30_site_int":
            input_tensor[:, i] = features.scale_site_feature(soi_vs30, "vs30")
        elif var == "vs30_diff":
            input_tensor[:, i] = features.scale_vs30_diff(vs30_diff)
        elif var == "angular_dist":
            input_tensor[:, i] = features.scale_angular_distance(
                np.deg2rad(angular_distance)
            )
        elif var == "ln_vs30_diff":
            input_tensor[:, i] = ln_vs30_diff

        else:
            raise ValueError(f"Unexpected variable {var} in edge features")
        
    with torch.no_grad():
        att_model.eval()
        raw_att_coeff = att_model(input_tensor).numpy(force=True)

    return raw_att_coeff, variable_input.numpy(force=True)


