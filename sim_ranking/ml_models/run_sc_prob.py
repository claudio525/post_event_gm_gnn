import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import typer

import sim_ranking as sr
import sim_ranking.ml.prob as prob
import sim_ranking.ml.sc_prob as sc_prob


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

def train_model(
    rel_db_ffp: Path,
    hyperparams_ffp: Path,
    rel_corr_dir: Path,
    n_epochs: int = 10,
    max_dist: float = 50,
    per_im_prob: bool = False,
    debug: bool = False,
    n_rels: int = None,
    id_suffix: str = "",
    data_source: str = None,
    im_set: str = "all",
    quiet: bool = False,
    seed: int = None,
    out_dir: Path = None,
):
    run_config = prob.RunParamsConfig(
        max_dist,
        n_rels,
        sr.constants.IM_SETS[im_set],
        sr.constants.IM_WEIGTHS_SETS[im_set],
        per_im_prob,
        debug,
        device,
        results_dir=out_dir,
    )
    hp_config = sc_prob.HyperParamsConfig.from_yaml(hyperparams_ffp, n_epochs)

    corr_dir = (
        Path(os.path.expandvars("$wdata")) / rel_corr_dir
        if rel_corr_dir is not None
        else None
    )

    ### Data loading
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp
    db = sr.db.DB(db_ffp)

    events = db.get_avail_events(data_source=data_source)
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    all_sites = db.get_avail_sites()

    ### Data setup
    # Get the sites per event
    event_sites = db.get_event_sites()

    # Split into training and validation
    if seed is not None:
        print(f"Using numpy random seed: {seed}")
        np.random.seed(seed)
    val_int_sites = np.random.choice(all_sites, 100, replace=False)
    train_sites = np.setdiff1d(all_sites, val_int_sites)

    val_events = np.random.choice(events, 75, replace=False)
    train_events = np.setdiff1d(events, val_events)

    train_dataset, val_dataset, scalar_features, data_metadata = sc_prob.data_prep(
        event_sites,
        train_events,
        val_events,
        train_sites,
        val_int_sites,
        events,
        run_config,
        hp_config,
        db,
        corr_dir=corr_dir,
    )

    prob_model = sc_prob.create_model(hp_config, scalar_features, run_config)
    prob_model.to(device)

    weight_model = sr.ml.models.WeightModel(
        run_config.n_ims, [16, 16], scalar_features.n_scalar_features
    )
    weight_model.to(device)

    print(f"Run training")
    sc_prob.train(
        prob_model,
        weight_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        data_metadata,
    )



if __name__ == "__main__":
    typer.run(train_model)

