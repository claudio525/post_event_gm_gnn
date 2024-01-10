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
    train_max_n_rels: int = 25,
    val_max_n_rels: int = 25,
    rel_sim_corr_dir: Path = None,
    id_suffix: str = "",
    data_source: str = None,
    im_set: str = "all",
    quiet: bool = False,
):
    """Trains a single model"""
    run_config = pr.RunParamsConfig(
        max_dist,
        train_max_n_rels,
        val_max_n_rels,
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

    val_events = np.random.choice(events, 75, replace=False)
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

    # from torch.utils.data import DataLoader
    # from tqdm import tqdm
    #
    # data_loader = pr.CustomTabularDataLoader(train_dataset, 64_000, False)
    # iter_loop = tqdm(data_loader)
    # iter_loop.set_description(f"Epoch 0/{hp_config.n_epochs}")
    #
    # ind = []
    # for i, t in enumerate(iter_loop):
    #     meta_df = pd.DataFrame(
    #         train_dataset.get_metadata(t[0]),
    #         index=["event_id", "site_int", "site_obs", "rel_1", "rel_2"],
    #     ).T
    #
    #     m = (meta_df.event_id == "2012p003376") & (meta_df.site_int == "AKSS") & (meta_df.site_obs == "KPOC")
    #     if np.count_nonzero(meta_df.loc[m]) > 0:
    #         print(f"wtf")
    #
    #     ind.append(t[0])
    #
    # print(f"wtf")

    # Create the model
    ranking_model = pr.create_model(hp_config, scalar_features, len(run_config.ims))
    ranking_model.to(device)

    # Train
    metrics, best_model_state, best_model_epoch = pr.train(
        ranking_model, train_dataset, val_dataset, device, hp_config, run_config, quiet=quiet
    )
    ranking_model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
        f"\tAccuracy: {metrics['acc_hist_val'][best_model_epoch]:.4f}\n"
        f"\tBCELoss: {metrics['bce_loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    data_metadata["db"] = str(rel_db_ffp)
    data_metadata["sim_corr_dir"] = str(rel_sim_corr_dir)

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
