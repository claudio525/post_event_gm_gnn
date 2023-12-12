import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import typer



import sim_ranking as sr

from sim_ranking.ml import pairwise as pr

app = typer.Typer()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


@app.command("train-model")
def train_model(
    rel_db_ffp: Path,
    hyperparams_ffp: Path,
    max_dist: float = 75,
    debug: bool = False,
    max_n_rels: int = 25,
    rel_sim_corr_dir: Path = None,
    id_suffix: str = "",
    data_source: str = None,
    im_set: str = "all",
):
    """Trains a single model"""
    run_config = pr.RunParamsConfig(
        max_dist,
        max_n_rels,
        sr.constants.IM_SETS[im_set],
        sr.constants.IM_WEIGTHS_SETS[im_set],
        debug,
        device,
    )
    hp_config = pr.HyperParamsConfig.from_yaml(hyperparams_ffp)

    ### Data loading
    # db_ffp_orig = "$wdata/sim_ranking/db/gm_db.sqlite"
    # db_ffp_orig = "$wdata/sim_ranking/db/gm_db_neil.sqlite"
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp

    sim_corr_dir = (
        Path(os.path.expandvars("$wdata")) / rel_sim_corr_dir
        if rel_sim_corr_dir is not None
        else None
    )
    db = sr.db.DB(db_ffp)

    # events = db.get_avail_events(data_source="specific")
    events = db.get_avail_events(data_source=data_source)
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    all_sites = db.get_avail_sites()

    ### Data setup
    # Get the sites per event
    event_sites = db.get_event_sites()

    # Split into training and validation
    np.random.seed(30)
    val_int_sites = np.random.choice(all_sites, 100, replace=False)
    train_sites = np.setdiff1d(all_sites, val_int_sites)

    val_events = np.random.choice(events, 100, replace=False)
    train_events = np.setdiff1d(events, val_events)

    # Data prep
    train_dataset, val_dataset, scalar_features, data_metadata = pr.data_prep(
        event_sites,
        train_events,
        val_events,
        train_sites,
        val_int_sites,
        events,
        run_config,
        db,
        sim_corr_dir=sim_corr_dir,
    )

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_loader = pr.CustomTabularDataLoader(train_dataset, hp_config.batch_size, True)
    iter_loop = tqdm(data_loader)
    iter_loop.set_description(f"Epoch 0/{hp_config.n_epochs}")

    for i, t in enumerate(iter_loop):
        pass

    # # n_workers = 0 if run_config.debug else 8
    # n_workers = 0
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=hp_config.batch_size,
    #     shuffle=True,
    #     num_workers=n_workers,
    #     pin_memory=True,
    #     persistent_workers=True if n_workers > 0 else False,
    #     prefetch_factor=25,
    # )
    #
    # iter_loop = tqdm(train_dataloader)
    # iter_loop.set_description(f"Epoch 0/{hp_config.n_epochs}")
    # for i, (
    #         _,
    #         scalar_features,
    #         int_sim_ims_rel_1,
    #         int_sim_ims_rel_2,
    #         obs_sim_ims_rel_1,
    #         obs_sim_ims_rel_2,
    #         obs_obs_ims,
    #         res_area_rel_1,
    #         res_area_rel_2,
    #         site_correlations,
    # ) in enumerate(iter_loop):
    #     pass

    exit()


    # Create the model
    ranking_model = pr.create_model(hp_config, scalar_features, len(run_config.ims))
    ranking_model.to(device)

    # Train
    metrics, best_model_state, best_model_epoch = pr.train(
        ranking_model, train_dataset, val_dataset, device, hp_config, run_config
    )
    ranking_model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
        f"\tAccuracy: {metrics['acc_hist_val'][best_model_epoch]:.4f}\n"
        f"\tBCELoss: {metrics['bce_loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    data_metadata["db"] = rel_db_ffp
    data_metadata["sim_corr_dir"] = rel_sim_corr_dir

    # Post-processing
    pr.post_processing(
        ranking_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        metrics,
        best_model_epoch,
        scalar_features,
        data_metadata,
        sim_corr_dir=sim_corr_dir,
        id_suffix=id_suffix,
    )


if __name__ == "__main__":
    app()
