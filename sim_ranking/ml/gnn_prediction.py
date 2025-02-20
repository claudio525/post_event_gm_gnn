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
    allow_self: bool = True,
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
    obs_site_df : pd.DataFrame
        DataFrame containing observation site information.
    obs_event_site_df : pd.DataFrame
        DataFrame containing event-site information for observation sites.
    obs_im_data : pd.DataFrame
        DataFrame containing intensity measure (IM) data for observation sites.
    allow_self : bool
        Whether to allow the prediction site to be one of the observation sites.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the prediction results with columns for event_id, site_int, 
        obs_sites, predicted IMs, and their standard deviations.
    """
    run_config = gnn_gm.RunConfig.from_yaml(model_dir / "run_config.yaml")

    model = torch.load(model_dir / "model.pt")
    model.eval()

    # Scale the IM data
    obs_im_data = obs_im_data.loc[
        obs_im_data.event_id == event_id, run_config.ims + ["event_id", "site_id"]
    ].copy()
    obs_im_data[run_config.ims] = (
        np.log(obs_im_data[run_config.ims]) - run_config.im_scale_params["mean"][run_config.ims]
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
    assert np.all(np.isin(all_sites, comb_site_df.index))

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

    print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(all_sites, comb_site_df)

    event_sites = {event_id: all_sites}
    event_int_sites = {event_id: int_sites}

    print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites,
        event_info.to_frame().T,
        comb_site_df,
        comb_event_site_df,
        run_config,
        constants.SCALAR_FEATURE_KEYS,
        dist_matrix,
    )

    # Site combinations
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
        )
        graph_data.append(cur_graph_data)

    result_df = _run_prediction(model, graph_data, run_config)

    # Add site information
    result_df = pd.merge(result_df, comb_site_df, left_on="site_int", right_index=True, how="left")
    return result_df


def predict_scenarios(
    model_dir: Path,
    site_df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_site_df: pd.DataFrame,
    obs_im_data: pd.DataFrame,
    scenario_defs: list,
):
    """
    Perform predictions using a trained GNN model
    for the specified scenarios.

    Parameters
    ----------
    model_dir : Path
        Directory containing the trained model and configuration files.
    site_df : pd.DataFrame
        DataFrame containing site information.
    event_df : pd.DataFrame
        DataFrame containing event information.
    event_site_df : pd.DataFrame
        DataFrame containing event-site information (e.g. rrup).
    im_data : pd.DataFrame
        DataFrame containing intensity measure (IM) data
        for the obsversation sites.
    scenario_defs : list
        List of scenario definitions. Each scenario is a tuple containing:
        - event_id (str): Identifier for the event.
        - site_int (str): Identifier for the site of interest.
        - obs_sites (list): List of observation site identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the prediction results with columns for event_id, site_int, obs_sites,
        predicted IMs, and their standard deviations.
    """
    run_config = gnn_gm.RunConfig.from_yaml(model_dir / "run_config.yaml")

    model = torch.load(model_dir / "model.pt")
    model.eval()

    # Scale the IM data
    obs_im_data = obs_im_data[run_config.ims + ["event_id", "site_id"]].copy()
    obs_im_data[run_config.ims] = (
        np.log(obs_im_data[run_config.ims]) - run_config.im_scale_params["mean"][run_config.ims]
    ) / run_config.im_scale_params["std"][run_config.ims]

    # Get the events and relevant event site pairs
    events = list({cur_event for cur_event, _, __ in scenario_defs})
    event_sites = {cur_event: set() for cur_event in events}
    for cur_event, cur_site_int, cur_obs_sites in scenario_defs:
        event_sites[cur_event].add(cur_site_int)
        event_sites[cur_event].update(cur_obs_sites)
    event_sites = {
        cur_event: np.asarray(list(cur_sites))
        for cur_event, cur_sites in event_sites.items()
    }

    print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(site_df.index, site_df)

    print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites,
        event_df,
        site_df,
        event_site_df,
        run_config,
        constants.SCALAR_FEATURE_KEYS,
        dist_matrix,
    )

    # Event site combinations
    event_site_combs = {cur_event: [] for cur_event in events}
    for cur_event, cur_site_int, cur_obs_sites in scenario_defs:
        cur_event_sites = event_sites[cur_event]
        cur_site_int_ix = np.flatnonzero(cur_event_sites == cur_site_int)[0]
        cur_combs = [
            (cur_site_int_ix, np.flatnonzero(cur_event_sites == cur_obs_site)[0])
            for cur_obs_site in cur_obs_sites
        ]

        event_site_combs[cur_event].extend(cur_combs)

    event_site_combs = {
        cur_event: np.asarray(cur_combs)
        for cur_event, cur_combs in event_site_combs.items()
    }

    event_scalar_feature_dfs, scalar_feature_columns = (
        ml_data.create_event_scalar_feature_dfs(
            event_sites, scalar_features, event_site_combs
        )
    )

    # Generate the graph data
    graph_data = []
    for cur_event, cur_site_int, cur_obs_sites in tqdm(
        scenario_defs, desc="Creating graph data"
    ):
        cur_scalar_feature_df = event_scalar_feature_dfs[cur_event]

        cur_im_data = obs_im_data.loc[obs_im_data.event_id == cur_event].set_index(
            "site_id"
        )
        cur_event_sites = event_sites[cur_event]
        cur_event_site_combs = event_site_combs[cur_event]
        cur_site_int_ix = np.flatnonzero(cur_event_sites == cur_site_int)[0]

        cur_sc_data = _create_graph_data(
            cur_event,
            cur_site_int_ix,
            cur_event_site_combs,
            cur_event_sites,
            cur_scalar_feature_df,
            cur_im_data,
            run_config,
        )

        graph_data.append(cur_sc_data)

    result_df = _run_prediction(model, graph_data, run_config)
    return result_df


def _create_graph_data(
    event_id: str,
    site_int_ix: int,
    site_combs: np.ndarray[float],
    sites: np.ndarray[str],
    scalar_feature_df: pd.DataFrame,
    im_data: pd.DataFrame,
    run_config: gnn_gm.RunConfig,
):
    """
    Creates the graph data for the specified scenario

    Note: sites must match the sites used for site_combs!
    """
    site_combs_mask = site_combs[:, 0] == site_int_ix
    cur_site_int = sites[site_combs[site_combs_mask, 0][0]]
    cur_obs_sites = sites[site_combs[site_combs_mask, 1]]

    # Create the site_int node features
    site_int_features = scalar_feature_df.loc[
        site_combs_mask, run_config.graph_feature_keys["site_int"]
    ].values[0]

    # Get observation site IM values and deal with nan values
    obs_sites_im_values = (
        im_data.loc[cur_obs_sites, run_config.ims].replace(np.nan, 99).values
    )
    if (
        run_config.graph_feature_keys["site_obs"] is not None
        and len(run_config.graph_feature_keys["site_obs"]) > 0
    ):
        # Create the site_obs node features
        obs_sites_features = scalar_feature_df.loc[
            site_combs_mask,
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
        site_combs_mask, run_config.graph_feature_keys["edge"]
    ].values

    graph_data = gdata.HeteroData()
    graph_data["site_int"].x = torch.tensor(
        site_int_features, dtype=torch.float32
    )[None, :]
    graph_data["site_obs"].x = torch.tensor(
        obs_sites_features, dtype=torch.float32
    )

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

    graph_data["metadata"] = {
        "sc_id": f"{event_id}_{cur_site_int}",
        "event": event_id,
        "site_int": cur_site_int,
        "obs_sites": cur_obs_sites,
    }

    return graph_data


def _run_prediction(
    model: nn.Module, graph_data: list[gdata.HeteroData], run_config: gnn_gm.RunConfig
):
    pred_cols = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
    pred_std_cols = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_std")

    results = []
    loader = gloader.DataLoader(graph_data, batch_size=1024, shuffle=False)
    for cur_batch in tqdm(loader, desc="Running predictions"):
        cur_batch = cur_batch.to(run_config.device)

        # Get predictions
        cur_out = model(cur_batch)
        torch_pred_ln_im_mean, torch_pred_ln_im_ln_std = cur_out
        pred_ln_im_std = torch.exp(torch_pred_ln_im_ln_std).cpu().numpy(force=True)
        pred_ln_im_mean = torch_pred_ln_im_mean.cpu().numpy(force=True)

        # Revert the IM scaling
        if run_config.scale_IMs:
            pred_ln_im_mean, pred_ln_im_std = gnn_gm.revert_im_scaling(
                pred_ln_im_mean, run_config, pred_ln_im_std
            )

        cur_result_df = pd.DataFrame(
            {
                "event_id": cur_batch["metadata"]["event"],
                "site_int": cur_batch["metadata"]["site_int"],
                "obs_sites": cur_batch["metadata"]["obs_sites"],
            }
        )
        cur_result_df[pred_cols] = pred_ln_im_mean
        cur_result_df[pred_std_cols] = pred_ln_im_std

        results.append(cur_result_df)

    result_df = pd.concat(results, axis=0)
    result_df.index = mlt.array_utils.numpy_str_join(
        "_",
        result_df["event_id"].values.astype(str),
        result_df["site_int"].values.astype(str),
    )

    return result_df
