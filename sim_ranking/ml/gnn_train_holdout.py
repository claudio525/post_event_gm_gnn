"""Module for training a GNN using the holdout method."""

from pathlib import Path

import numpy as np

import ml_tools as mlt

from . import gnn_gm
from . import data as ml_data
from . import features
from .. import constants
from .. import data
from .. import utils


def run_holdout(
    run_config_ffp: Path,
    holdout_config_ffp: Path,
    n_epochs: int = None,
    id_suffix: str = "",
    device: str = "cpu",
):
    ### Create the configs
    run_config = gnn_gm.RunConfig.from_config_kwargs(
        run_config_ffp, n_epochs=n_epochs, ims=constants.PSA_KEYS, device=device
    )
    holdout_config = gnn_gm.HoldoutConfig.from_yaml(holdout_config_ffp)

    ### Data loading
    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)
    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    ### Data setup
    # Get the set of valid site-interests per event
    print("Getting valid sites of interest")
    valid_int_sites, valid_event_int_sites, _ = ml_data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))
    print(f"Number of valid events: {len(events)}/{len(obs_data.events)}")

    if holdout_config.test_events is not None:
        events = np.setdiff1d(events, holdout_config.test_events)
        print(f"Number of events after removing test events: {len(events)}")

    # Set the random seed
    if run_config.seed is not None:
        print(f"Using numpy random seed: {run_config.seed}")
        np.random.seed(run_config.seed)

    # Split into training and validation
    val_events = np.random.choice(events, holdout_config.n_val_events, replace=False)
    if holdout_config.val_events is not None:
        val_events = np.union1d(val_events, holdout_config.val_events)
    train_events = np.setdiff1d(events, val_events)

    print("----------------- Events Summary -----------------")
    print(f"Number of available events: {len(events)}")
    print(f"Number of training events: {train_events.size}")
    print(f"Number of validation events: {val_events.size}")

    if holdout_config.val_sites_ffp is not None:
        val_int_sites = np.load(holdout_config.val_sites_ffp)
    else:
        val_int_sites = np.random.choice(
            valid_int_sites, holdout_config.n_val_sites, replace=False
        )
    train_int_sites = np.setdiff1d(valid_int_sites, val_int_sites)
    obs_sites = np.setdiff1d(all_sites, val_int_sites)

    print("----------------- Sites Summary -----------------")
    print(f"Number of available sites: {len(all_sites)}")
    print(f"Number of valid sites of interests: {valid_int_sites.size}")
    print(f"Number of training sites of interests: {train_int_sites.size}")
    print(f"Number of validation sites of interests: {val_int_sites.size}")
    print(f"Number of observation sites: {obs_sites.size}")
    print("------------------------------------------------")

    print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites, obs_data.event_df, obs_data.site_df, obs_data.record_df, run_config, constants.SCALAR_FEATURE_KEYS, dist_matrix
    )

    id_suffix = f"_{id_suffix}" if len(id_suffix) > 0 else ""
    cur_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    )

    gnn_gm.run_model_training(
        cur_out_dir,
        event_sites,
        valid_event_int_sites,
        train_events,
        val_events,
        train_int_sites,
        val_int_sites,
        obs_sites,
        dist_matrix,
        obs_data,
        scalar_features,
        run_config,
        graph_data_n_procs=1,
    )
