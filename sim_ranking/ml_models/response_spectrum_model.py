"""
Trains a model that given
- Response spectrum from observed GM at the observation site
- Response spectrum from simulation realisation at the observation site
- Response spectrum from simulation realisation at the site of interest
- Site properties of observation site
- Site properties of site of interest
predicts the similarity score of the simulation realisation and the
(unknown) observed GM at the site of interest
"""
import os
from pathlib import Path

import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from torchview import draw_graph
import typer

import sim_ranking as sr
import spatial_hazard as sh
import ml_tools as mlt


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


def get_dataset_predictions(
    dataset: sr.ml.data.ResponseSpectrumDataset,
    model: nn.Module,
    device: str,
    loss_fn_key: str,
):
    # pred_train_meta_dataloader = DataLoader(
    #     train_meta_dataset, shuffle=False, batch_size=4096, num_workers=0
    # )
    pred_train_dataloader = DataLoader(
        dataset, shuffle=False, batch_size=4096, num_workers=0
    )

    model.eval()

    results = []
    pred_res_results, true_res_results = [], []

    for i, (
        data_ind,
        obs_obs_obs_sim_res,
        obs_obs_int_sim_res,
        obs_sim_int_sim_rel,
        site_features,
        true_res,
    ) in enumerate(pred_train_dataloader):
        # Forward pass
        pred = (
            _get_prediction(
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                site_features,
                model,
                device,
            )
            .cpu()
            .detach()
        )

        meta_df = pd.DataFrame(
            [dataset.get_metadata(data_ind[0]) for ix in data_ind],
            columns=["event_id", "rel_id", "site_int", "site_obs"],
        )
        meta_df.index = np.char.add(
                    np.char.add(
                        np.char.add(
                            np.char.add(
                                np.char.add(meta_df.event_id.values.astype(str), "_"),
                                np.char.add(meta_df.rel_id.values.astype(str), "_"),
                            ),
                            meta_df.site_int.values.astype(str),
                        ),
                        "_",
                    ),
                    meta_df.site_obs.values.astype(str),
                )
        meta_df["loss"] = _compute_loss(loss_fn_key, pred, true_res, reduction="none").cpu().detach().numpy().mean(axis=1)

        results.append(meta_df)
        true_res_results.append(true_res.numpy())
        pred_res_results.append(pred.numpy())

    results_df = pd.concat(results)
    results_df.loc[:, np.char.add(sr.constants.PSA_KEYS, "_true")] = np.concatenate(
        true_res_results, axis=0
    )
    results_df.loc[:, np.char.add(sr.constants.PSA_KEYS, "_pred")] = np.concatenate(
        pred_res_results, axis=0
    )

    return results_df


def _get_prediction(
    obs_obs_obs_sim_res,
    obs_obs_int_sim_res,
    obs_sim_int_sim_rel,
    site_features,
    model: nn.Module,
    device: str,
):
    # Put data onto the device
    obs_obs_obs_sim_res = obs_obs_obs_sim_res.to(device, dtype=torch.float32)[
        :, None, :
    ]
    obs_obs_int_sim_res = obs_obs_int_sim_res.to(device, dtype=torch.float32)[
        :, None, :
    ]
    obs_sim_int_sim_rel = obs_sim_int_sim_rel.to(device, dtype=torch.float32)[
        :, None, :
    ]
    site_features = site_features.to(device, dtype=torch.float32)

    # return model(rs_int_sim, rs_obs_sim, rs_obs_obs, site_features).ravel()
    return model(
        obs_obs_obs_sim_res, obs_obs_int_sim_res, obs_sim_int_sim_rel, site_features
    )


def _compute_loss(loss_fn_key: str, pred, sim_score, reduction: str = "mean"):
    return nn.functional.l1_loss(pred, sim_score, reduction=reduction)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_fn_key: str,
):
    loss_hist_train, loss_hist_val = torch.zeros(n_epochs), torch.zeros(n_epochs)

    best_val_loss = np.inf
    best_model_state = None
    best_model_epoch = None

    for epoch in range(n_epochs):
        print(f"Processing epoch {epoch+1}/{n_epochs}")

        ### Training
        model.train()
        for i, (
            _,
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_rel,
            site_features,
            int_obs_int_sim_res,
        ) in enumerate(train_dataloader):
            # Forward pass
            pred = _get_prediction(
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                site_features,
                model,
                device,
            )

            # Compute the loss
            int_obs_int_sim_res = int_obs_int_sim_res.to(device, dtype=torch.float32)
            loss = _compute_loss(loss_fn_key, pred, int_obs_int_sim_res)

            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_hist_train[epoch] += loss.item() * int_obs_int_sim_res.shape[0]

        loss_hist_train[epoch] /= len(train_dataloader.dataset)

        ### Validation
        model.eval()
        with torch.no_grad():
            for i, (
                _,
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                site_features,
                int_obs_int_sim_res,
            ) in enumerate(val_dataloader):
                # Forward pass
                pred = _get_prediction(
                    obs_obs_obs_sim_res,
                    obs_obs_int_sim_res,
                    obs_sim_int_sim_rel,
                    site_features,
                    model,
                    device,
                )

                # Loss
                int_obs_int_sim_res = int_obs_int_sim_res.to(
                    device, dtype=torch.float32
                )
                loss = _compute_loss(loss_fn_key, pred, int_obs_int_sim_res)
                loss_hist_val[epoch] += loss.item() * int_obs_int_sim_res.shape[0]

            loss_hist_val[epoch] /= len(val_dataloader.dataset)

        # Keep track of the best model
        if loss_hist_val[epoch] < best_val_loss:
            best_model_state = model.state_dict()
            best_val_loss = loss_hist_val[epoch]
            best_model_epoch = epoch

        print(
            f"Epoch {epoch+1} Loss:\n Train {loss_hist_train[epoch]:.4f} Val {loss_hist_val[epoch]:.4f}"
        )
    return loss_hist_train, loss_hist_val, best_model_state, best_model_epoch


def main(
    n_epochs: int = 25,
    batch_size: int = 2048,
    n_rels_used: int = 5,
    comment: str = "",
    loss_fn_key: str = "custom",
    max_dist: float = 100.0,
    save_best_val_model: bool = True,
    weight_decay: float = 0.0001,
):
    ### CONFIG ###
    # N_VAL_SITES = 10

    # RS_N_CHANNELS = [1, 16, 32]
    RS_N_CHANNELS = []
    RS_KERNEL_SIZES = []
    # RS_KERNEL_SIZES = [5, 3]
    # FC_UNITS = [128, 64, 32]
    FC_UNITS = [64, 32]

    SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angle"]
    ### END CONFIG ###

    # Fixing the random seed
    np.random.seed(42)

    db_ffp_orig = "$wdata/sim_ranking/db/gm_db.sqlite"
    db_ffp = Path(os.path.expandvars(db_ffp_orig))
    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))

    db = sr.db.DB(db_ffp)
    station_df = db.get_site_df()
    event_df = db.get_event_df()
    record_df = db.get_record_df()

    events = db.get_avail_events()
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    # all_sites = np.unique(np.concatenate(list(event_sites.values())))
    all_sites = db.get_avail_sites()

    # Compute the distance matrix
    print(f"Computing distance matrix")
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    # Select one of the events for validation
    # val_events = np.asarray(["3468575", "3528839"])
    val_events = np.asarray(
        [
            "3528839",
            "3497857",
            "2017p161601",
            "2013p543121",
            "2016p355041",
            "2017p512943",
        ]
    )
    train_events = np.setdiff1d(events, val_events)

    # Get the sites per event
    event_sites = db.get_event_sites()

    # Select a subset of the stations (of the validation events) as the validation sites
    # val_sites = np.random.choice(np.intersect1d() , N_VAL_SITES, replace=False)
    # val_sites = np.random.choice(
    #     list(
    #         set.intersection(
    #             *[set(list(event_sites[cur_event])) for cur_event in val_events]
    #         )
    #     ),
    #     N_VAL_SITES,
    #     replace=False,
    # )
    # train_sites = all_sites[np.isin(all_sites, val_sites, invert=True)]

    # Use all sites for training & validation for now
    # Once the model learns, this needs to be updated
    # TODO: Update this!!
    train_sites = all_sites
    val_sites = all_sites

    # Get the training and validation dataset site combinations
    print(f"Creating site combinations")
    train_site_combs, train_event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        train_events,
        dist_matrix,
        sites_to_use=train_sites,
        max_dist=max_dist,
    )
    val_site_combs, val_event_sites = sr.ml.data.compute_site_combinations(
        event_sites, val_events, dist_matrix, sites_to_use=val_sites, max_dist=max_dist
    )

    # Get the periods and corresponding pSA keys
    periods, pSA_keys = sr.constants.PERIODS, sr.constants.PSA_KEYS
    n_periods = len(periods)

    # Run pre-processing for the site features
    # TODO: This should be updated such that the normalisation
    # only happens on training sites, not all sites
    print(f"Pre-processing site features")
    site_features_df, site_feature_stats = sr.ml.features.preprocess_site_features(
        station_df, SITE_FEATURE_KEYS
    )

    # Computed the site-to-site features
    print(f"Computing scalar features")
    ### Site-to-site features
    site_to_site_features = {}

    # Scale the (used) site-to-site distances
    # such that they are between -1 and 1
    # as per the maximum allowed site-to-site
    # distance when computing the site combinations
    site_to_site_features["dist"] = ((dist_matrix.copy() / max_dist) * 2) - 1

    ### Event-site features
    event_site_features = {}
    for cur_event in events:
        event_site_features[cur_event] = sr.ml.features.pre_process_event_site_features(
            record_df.loc[record_df.event_id == cur_event]
            .set_index("site_id")
            .drop(columns=["event_id"])
        )

    ### Event-site-to-site features
    event_site_to_site_features = {}

    # Compute the site-to-site angle wrt. the epicentre
    event_site_to_site_features["angle"] = sr.ml.features.compute_angular_distance(
        station_df.copy(), event_df.copy(), events, event_sites
    )

    scalar_features = sr.ml.data.ScalarFeatures(
        site_features_df,
        SITE_FEATURE_KEYS,
        site_to_site_features,
        SITE_TO_SITE_FEATURE_KEYS,
        event_site_features,
        EVENT_SITE_FEATURE_KEYS,
        event_site_to_site_features,
        EVENT_SITE_TO_SITE_FEATURE_KEYS,
    )

    # Create the training and validation dataset
    print(f"Creating datasets and dataloaders")
    train_dataset = sr.ml.data.ResponseSpectrumResidualDataset(
        train_event_sites,
        train_site_combs,
        db,
        n_rels_used,
        site_features_df,
        periods,
        pSA_keys,
        scalar_features,
    )
    val_dataset = sr.ml.data.ResponseSpectrumResidualDataset(
        val_event_sites,
        val_site_combs,
        db,
        n_rels_used,
        site_features_df,
        periods,
        pSA_keys,
        scalar_features,
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # Create the model
    n_rs_layers = len(RS_KERNEL_SIZES)
    padding = (
        [mlt.dl_utils.compute_same_conv_padding(n_periods, RS_KERNEL_SIZES[0])]
        * n_rs_layers
        if n_rs_layers > 0
        else []
    )

    model = sr.ml.models.ResponseSpectrumSimModel(
        RS_KERNEL_SIZES,
        RS_N_CHANNELS,
        padding,
        FC_UNITS,
        len(periods),
        scalar_features.n_scalar_features,
        n_periods,
        apply_sigmoid=False,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=weight_decay
    )

    summary(
        model,
        input_size=[
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, scalar_features.n_scalar_features),
        ],
    )

    print(f"Running training")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    loss_hist_train, loss_hist_val, best_model_state, best_epoch = train(
        model,
        train_dataloader,
        val_dataloader,
        n_epochs,
        device,
        optimizer,
        loss_fn_key,
    )

    # Load the best model
    if save_best_val_model:
        model.load_state_dict(best_model_state)
        print(
            f"Best model epoch: {best_epoch + 1}, "
            f"with validation loss {loss_hist_val[best_epoch]:.4f}"
        )

    # Get predictions for the training and validation datasets
    print(f"Getting predictions")
    train_results_df = get_dataset_predictions(
        train_dataset, model, device, loss_fn_key
    )
    val_results_df = get_dataset_predictions(val_dataset, model, device, loss_fn_key)

    # Save the results
    run_id = mlt.utils.create_run_id(False)
    (results_dir := results_dir / run_id).mkdir(exist_ok=False)
    print(f"Savings results, run-id {run_id}")

    train_results_df.to_csv(results_dir / "train_results.csv", index=True)
    val_results_df.to_csv(results_dir / "val_results.csv", index=True)

    # Write the metadata
    meta_data = {
        "site_feature_stats": site_feature_stats.to_dict(),
        "n_rels_used": n_rels_used,
        # "n_val_sites": N_VAL_SITES,
        "val_events": val_events.astype(str).tolist(),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "train_events": train_events.astype(str).tolist(),
        "site_features": scalar_features.site_feature_keys,
        "site_to_site_features": scalar_features.site_to_site_feature_keys,
        "event_site_features": scalar_features.event_site_feature_keys,
        "event_site_to_site_features": scalar_features.event_site_to_site_feature_keys,
        "comment": comment,
        "model": {
            "n_channels": RS_N_CHANNELS,
            "kernel_sizes": RS_KERNEL_SIZES,
            "fc_units": FC_UNITS,
        },
        "training": {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "best_epoch": best_epoch,
        },
        "data": {
            "db": str(db_ffp_orig),
        },
    }
    mlt.utils.write_to_yaml(meta_data, results_dir / "meta.yaml")

    # Save the model
    torch.save(model, results_dir / "model.pt")

    # Save the loss history
    np.save(str(results_dir / "loss_hist_train.npy"), loss_hist_train.numpy())
    np.save(str(results_dir / "loss_hist_val.npy"), loss_hist_val.numpy())

    # Create a model visualisation
    draw_graph(
        model,
        input_size=[
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="model_vis",
        save_graph=True,
        directory=str(results_dir),
    )


if __name__ == "__main__":
    typer.run(main)
