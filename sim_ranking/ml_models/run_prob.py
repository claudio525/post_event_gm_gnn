import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import typer


import sim_ranking as sr
import sim_ranking.ml.prob as prob

app = typer.Typer()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")


@app.command("train-model")
def train_model(
    rel_db_ffp: Path,
    hyperparams_ffp: Path,
    n_epochs: int = 10,
    max_dist: float = 75,
    debug: bool = False,
    n_rels: int = None,
    rel_sim_corr_dir: Path = None,
    id_suffix: str = "",
    data_source: str = None,
    im_set: str = "all",
    quiet: bool = False,
    seed: int = None,
    out_dir: Path = None,
):
    """Trains a single model"""
    run_config = prob.RunParamsConfig(
        max_dist,
        n_rels,
        sr.constants.IM_SETS[im_set],
        sr.constants.IM_WEIGTHS_SETS[im_set],
        debug,
        device,
        results_dir=out_dir,
    )
    hp_config = prob.HyperParamsConfig.from_yaml(hyperparams_ffp, n_epochs)

    ### Data loading
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp

    sim_corr_dir = (
        Path(os.path.expandvars("$wdata")) / rel_sim_corr_dir
        if rel_sim_corr_dir is not None
        else None
    )
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

    train_dataset, val_dataset, scalar_features, data_metadata = prob.data_preb(
        event_sites,
        train_events,
        val_events,
        train_sites,
        val_int_sites,
        events,
        run_config,
        hp_config,
        db,
        sim_corr_dir=sim_corr_dir,
    )

    prob_model = prob.create_model(hp_config, scalar_features, run_config)
    prob_model.to(device)

    print(f"Run training")
    metrics, best_model_state, best_model_epoch = prob.train(
        prob_model, train_dataset, val_dataset, hp_config, run_config, quiet=quiet
    )
    prob_model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    print(f"Run post-processing")
    data_metadata["db"] = str(rel_db_ffp)
    data_metadata["sim_corr_dir"] = (
        sim_corr_dir if sim_corr_dir is None else str(sim_corr_dir)
    )
    prob.post_processing(
        prob_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        metrics,
        best_model_epoch,
        scalar_features,
        data_metadata,
        val_int_sites,
        train_sites,
        id_suffix=id_suffix,
    )




if __name__ == "__main__":
    app()


### Sanity checking code
# loader = prob.CustomTabularDataLoader(train_dataset, batch_size=1000, shuffle=False)
#
# for (
#     batch_ind,
#     site_int_sims,
#     site_obs_sims,
#     site_obs_obs,
#     scalar_features,
#     misfit_score,
# ) in loader:
#     event, site_int, site_obs = train_dataset.get_metadata(batch_ind)
#
#     for ix, (cur_event, cur_site_int, cur_site_obs) in enumerate(
#         zip(event, site_int, site_obs)
#     ):
#         cur_rels = train_dataset.event_rels[cur_event]
#
#         db_site_int_sims = db.get_sim_data(cur_event, [cur_site_int])
#         db_site_int_sims = db_site_int_sims.loc[
#             np.isin(db_site_int_sims.rel_id, cur_rels)
#         ].sort_values("rel_id")
#         assert np.all(
#             site_int_sims[ix].numpy()
#             == np.log(db_site_int_sims.loc[:, run_config.ims].values)
#         )
#
#         db_site_obs_sims = db.get_sim_data(cur_event, [cur_site_obs])
#         db_site_obs_sims = db_site_obs_sims.loc[
#             np.isin(db_site_obs_sims.rel_id, cur_rels)
#         ].sort_values("rel_id")
#         assert np.all(
#             site_obs_sims[ix].numpy()
#             == np.log(db_site_obs_sims.loc[:, run_config.ims].values)
#         )
#
#         db_site_obs_obs = db.get_obs_data(cur_event, [cur_site_obs]).loc[:, run_config.ims].values
#         assert np.all(site_obs_obs[ix].numpy() == np.log(db_site_obs_obs))
