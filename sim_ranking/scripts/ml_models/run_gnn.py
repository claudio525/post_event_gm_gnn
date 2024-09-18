import time
import os
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import torch_geometric.data as gdata
import torch_geometric.transforms as T
import torch_geometric.loader as gloader
import tqdm
import typer

import ml_tools as mlt
import sim_ranking as sr
import spatial_hazard as sh

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer()

@app.command("run-holdout")
def run_holdout(run_config_ffp: Path, holdout_config_ffp: Path, n_epochs: int = None):
    ### Create the configs
    run_config = sr.ml.gnn_gm.RunConfig.from_config_kwargs(
        run_config_ffp, n_epochs=n_epochs, ims=sr.constants.PSA_KEYS, device=device
    )
    holdout_config = sr.ml.gnn_gm.HoldoutConfig.from_yaml(holdout_config_ffp)

    ### Data loading
    obs_data = sr.ObservedData.from_nzgmdb_flat(run_config.obs_data_ffp)
    obs_data.drop_nan()
    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")


    ### Data setup
    # Get the set of valid site-interests per event
    print(f"Getting valid sites of interest")
    valid_int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.IM_COLUMNS)
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

    print(f"----------------- Events Summary -----------------")
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

    print(f"----------------- Sites Summary -----------------")
    print(f"Number of available sites: {len(all_sites)}")
    print(f"Number of valid sites of interests: {valid_int_sites.size}")
    print(f"Number of training sites of interests: {train_int_sites.size}")
    print(f"Number of validation sites of interests: {val_int_sites.size}")
    print(f"Number of observation sites: {obs_sites.size}")
    print(f"------------------------------------------------")

    print(f"Computing distance matrix")
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, obs_data.site_df)

    print(f"Getting scalar features")
    scalar_features = sr.ml.features.get_scalar_features(
        event_sites, obs_data, run_config, sr.constants.SCALAR_FEATURE_KEYS, dist_matrix
    )

    id_suffix = ""
    cur_out_dir = run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"

    sr.ml.gnn_gm.run(
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
    )

@app.command("run-cv")
def run_cv(run_config_ffp: Path, n_event_folds: int, n_site_folds: int, n_epochs: int = None):
    run_config = sr.ml.gnn_gm.RunConfig.from_config_kwargs(
        run_config_ffp, n_epochs=n_epochs, ims=sr.constants.PSA_KEYS, device=device
    )


    print(f"wtf")
    pass


if __name__ == "__main__":
    app()
