from pathlib import Path

import numpy as np
import pandas as pd

import ml_tools as mlt
from labelled_data_array import LabelledDataArray

from . import gnn_gm
from . import data as ml_data
from . import features
from .. import constants
from .. import data
from .. import utils


def run_full(
    out_dir: Path,
    run_config: Path,
    n_epochs: int = None,
    device: str = None,
):
    """
    Run the full training process for the GNN model.

    Parameters
    ----------
    run_config_ffp : Path
        Path to the run configuration file.
    n_epochs : int, optional
        Number of epochs to run the model for.
    """
    # Create the config
    if isinstance(run_config, Path):
        run_config = gnn_gm.RunConfig.from_config_kwargs(
            run_config, n_epochs=n_epochs, device=device
        )

    ### Data loading
    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)
    if len(run_config.ignore_events) > 0:
        obs_data = obs_data.drop_events(run_config.ignore_events)

    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    # Get the set of valid site-interests per event
    print("Getting valid sites of interest")
    int_sites, valid_event_int_sites, _ = ml_data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))

    print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites, obs_data.event_df, obs_data.site_df, obs_data.record_df, run_config, constants.SCALAR_FEATURE_KEYS, dist_matrix
    )

    gnn_gm.run_model_training(
        out_dir,
        event_sites,
        event_sites,
        events,
        None,
        int_sites, 
        None, 
        all_sites,
        dist_matrix,
        obs_data,
        scalar_features,
        run_config,
    )
