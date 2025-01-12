"""
Module for performing predictions using an already trained GNN
for post-event GM estimation
"""

import time
import os
import itertools
import pickle
import warnings
import multiprocessing as mp
from typing import NamedTuple, Sequence, Callable
from pathlib import Path
from dataclasses import dataclass

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import torch_geometric.data.batch as gbatch
import torch_geometric.utils as gutils
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

import ml_tools as mlt

from . import data as ml_data
from . import gnn_modules
from . import features
from . import gnn_gm
from .. import utils
from .. import constants
from ..data_classes import ObservedData, LBSiteCorrelationData


def predict_single(
    model_dir: Path,
    site_df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_site_df: pd.DataFrame,
    im_data: pd.DataFrame,
    scenario_defs: list,
):
    run_config = gnn_gm.RunConfig.from_yaml(model_dir / "run_config.yaml")

    model = torch.load(model_dir / "model.pt")
    model.eval()

    # Scale the IM data
    im_data = im_data[run_config.ims + ["event_id", "site_id"]].copy()
    im_data[run_config.ims] = (
        im_data[run_config.ims] - run_config.im_scale_params["mean"][run_config.ims]
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

    print(f"Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(site_df.index, site_df)

    print(f"Getting scalar features")
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

    event_scalar_feature_dfs, scalar_feature_columns = ml_data.create_event_scalar_feature_dfs(
        event_sites, scalar_features, event_site_combs
    )

    graph_data = []
    for cur_event, cur_site_int, cur_obs_sites in scenario_defs:
        cur_scalar_feature_df = event_scalar_feature_dfs[cur_event]

        cur_im_data = im_data.loc[im_data.event_id == cur_event].set_index("site_id")
        cur_event_sites = event_sites[cur_event]
        cur_event_site_combs = event_site_combs[cur_event]
        cur_site_int_ix = np.flatnonzero(cur_event_sites == cur_site_int)[0]

        cur_site_combs_mask = cur_event_site_combs[:, 0] == cur_site_int_ix
        cur_site_int = cur_event_sites[cur_event_site_combs[cur_site_combs_mask, 0][0]]
        cur_obs_sites = cur_event_sites[cur_event_site_combs[cur_site_combs_mask, 1]]

        # Create the site_int node features
        cur_site_int_features = cur_scalar_feature_df.loc[
            cur_site_combs_mask, run_config.graph_feature_keys["site_int"]
        ].values[0]

        # Get observation site IM values and deal with nan values
        cur_obs_sites_im_values = (
            cur_im_data.loc[cur_obs_sites, run_config.ims].replace(np.nan, 99).values
        )
        if (
            run_config.graph_feature_keys["site_obs"] is not None
            and len(run_config.graph_feature_keys["site_obs"]) > 0
        ):
            # Create the site_obs node features
            cur_obs_sites_features = cur_scalar_feature_df.loc[
                cur_site_combs_mask,
                run_config.graph_feature_keys["site_obs"],
            ].values
            # Add the IM values
            cur_obs_sites_features = np.concatenate(
                (
                    cur_obs_sites_features,
                    cur_obs_sites_im_values,
                ),
                axis=1,
            )
        else:
            cur_obs_sites_features = cur_obs_sites_im_values

        # Create the edge features
        cur_edge_features = cur_scalar_feature_df.loc[
            cur_site_combs_mask, run_config.graph_feature_keys["edge"]
        ].values

        cur_sc_data = gdata.HeteroData()
        cur_sc_data["site_int"].x = torch.tensor(
            cur_site_int_features, dtype=torch.float32
        )[None, :]
        cur_sc_data["site_obs"].x = torch.tensor(
            cur_obs_sites_features, dtype=torch.float32
        )

        cur_sc_data["site_obs", "informs", "site_int"].edge_index = torch.tensor(
            [[ix, 0] for ix, cur_obs_site in enumerate(cur_obs_sites)],
            dtype=torch.long,
        ).T
        cur_sc_data["site_obs", "informs", "site_int"].edge_attr = torch.tensor(
            cur_edge_features, dtype=torch.float32
        )

        cur_sc_data["site_obs", "self_loop", "site_obs"].edge_index = torch.tensor(
            [[ix, ix] for ix in range(len(cur_obs_sites))], dtype=torch.long
        ).T

        cur_sc_data["metadata"] = {
            "sc_id": f"{cur_event}_{cur_site_int}",
            "event": cur_event,
            "site_int": cur_site_int,
            "obs_sites": cur_obs_sites,
        }

        graph_data.append(cur_sc_data)

    # loader = gloader.DataLoader(graph_data, batch_size=512)
    results = model(graph_data[0].to("cuda"))

    return results


