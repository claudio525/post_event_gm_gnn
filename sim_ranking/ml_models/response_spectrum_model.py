"""
Trains a model that given
- Response spectrum from observed GM at the observation site
- Response spectrum from simulation realisation at the observation site
- Response spectrum from simulation realisation at the site of interest
- Site properties of observation site
- Site properties of site of interest
predicts the residual between the simulation and observation at
the site of interest.
"""
import pickle
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
    dataset: sr.ml.data.BaseDataset,
    res_model: nn.Module,
    weight_model: nn.Module,
    device: str,
    weight_norm_penalty: float,
):
    pred_train_dataloader = DataLoader(
        dataset, shuffle=False, batch_size=4096, num_workers=2
    )

    res_model.eval()

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
        weights, pred_res = (
            cur_r.cpu().detach()
            for cur_r in _get_prediction(
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                site_features,
                res_model,
                weight_model,
                device,
            )
        )

        meta_df = pd.DataFrame(
            [dataset.get_metadata(ix) for ix in data_ind],
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
        meta_df["misfit_loss_term"] = compute_misfit(
            pred_res, true_res, weights, dim=1
        ).numpy(force=True)
        meta_df["weight_penalty_loss_term"] = compute_weight_norm_penalty(
            weights, weight_norm_penalty, device
        ).numpy(force=True)
        meta_df["total_loss"] = (
            meta_df["misfit_loss_term"] + meta_df["weight_penalty_loss_term"]
        )
        meta_df["weight"] = weights.numpy()

        results.append(meta_df)
        true_res_results.append(true_res.numpy())
        pred_res_results.append(pred_res.numpy())

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
    scalar_features,
    res_model: nn.Module,
    weight_model: nn.Module,
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
    scalar_features = scalar_features.to(device, dtype=torch.float32)

    res = res_model(
        obs_obs_obs_sim_res, obs_obs_int_sim_res, obs_sim_int_sim_rel, scalar_features
    )
    weight = weight_model(scalar_features)

    # Normalise weights
    # if normalise_weights:
    #     weight = (weight / weight.sum()) * weight.shape[0]
    # assert weight.max() <= 1.0 and weight.min() >= 0.0, "Weights should be normalised"

    return weight, res


def compute_misfit(
    pred_res: torch.Tensor,
    true_res: torch.Tensor,
    sample_weights: torch.Tensor,
    dim: int = None,
):
    return torch.mean(torch.abs(pred_res - true_res) * sample_weights, dim=dim)


def compute_weight_norm_penalty(sample_weights: torch.Tensor, lambda_p: float, device):
    Nw = sample_weights.shape[0]
    lambda_p = torch.tensor(lambda_p, device=device)

    # See https://www.desmos.com/calculator/tyljt3ybpr
    lp = lambda_p * (
        (
            torch.maximum(
                torch.tensor(0.0, device=device), 0.75 * Nw - sample_weights.sum()
            )
            ** 2
        )
        / (Nw)
    )
    return lp
    # return ((sample_weights.shape[0] * 0.75) - sample_weights.sum()) / sample_weights.shape[0]


def compute_total_loss(
    pred_res: torch.Tensor,
    true_res: torch.Tensor,
    sample_weights: torch.Tensor,
    weight_norm_penalty: float,
    dim: int = None,
):
    misfit_loss = compute_misfit(pred_res, true_res, sample_weights, dim=dim)
    weight_penalty_loss = compute_weight_norm_penalty(
        sample_weights, weight_norm_penalty, device
    )

    return misfit_loss + weight_penalty_loss, misfit_loss, weight_penalty_loss


def train(
    res_model: nn.Module,
    weight_model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    weight_norm_penalty: float,
):
    metrics = {
        "loss_hist_train": torch.zeros(n_epochs),
        "loss_hist_val": torch.zeros(n_epochs),
        "misfit_loss_hist_train": torch.zeros(n_epochs),
        "misfit_loss_hist_val": torch.zeros(n_epochs),
        "weight_penalty_loss_hist_train": torch.zeros(n_epochs),
        "weight_penalty_loss_hist_val": torch.zeros(n_epochs),
        "weight_hist_train": torch.zeros(n_epochs),
        "weight_hist_val": torch.zeros(n_epochs),
    }

    best_epoch_loss_key = "loss_hist_val"
    best_val_loss = np.inf
    best_res_model_state = None
    best_weight_model_state = None
    best_model_epoch = None

    for epoch in range(n_epochs):
        ### Training
        res_model.train()
        weight_model.train()
        for i, (
            _,
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_rel,
            scalar_features,
            int_obs_int_sim_res,
        ) in enumerate(train_dataloader):
            # Forward pass
            weight, res_pred = _get_prediction(
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                scalar_features,
                res_model,
                weight_model,
                device,
            )

            # Compute the loss
            int_obs_int_sim_res = int_obs_int_sim_res.to(device, dtype=torch.float32)
            loss, misfit_loss, weight_penalty_loss = compute_total_loss(
                res_pred, int_obs_int_sim_res, weight, weight_norm_penalty
            )

            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics["loss_hist_train"][epoch] += (
                loss.item() * int_obs_int_sim_res.shape[0]
            )
            metrics["misfit_loss_hist_train"][epoch] += (
                misfit_loss.item() * int_obs_int_sim_res.shape[0]
            )
            metrics["weight_penalty_loss_hist_train"][epoch] += (
                weight_penalty_loss.item() * int_obs_int_sim_res.shape[0]
            )
            metrics["weight_hist_train"][epoch] += (
                weight.mean().item() * weight.shape[0]
            )

        metrics["loss_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["misfit_loss_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["weight_penalty_loss_hist_train"][epoch] /= len(
            train_dataloader.dataset
        )
        metrics["weight_hist_train"][epoch] /= len(train_dataloader.dataset)

        ### Validation
        res_model.eval()
        weight_model.eval()
        with torch.no_grad():
            for i, (
                _,
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_rel,
                scalar_features,
                int_obs_int_sim_res,
            ) in enumerate(val_dataloader):
                # Forward pass
                weight, res_pred = _get_prediction(
                    obs_obs_obs_sim_res,
                    obs_obs_int_sim_res,
                    obs_sim_int_sim_rel,
                    scalar_features,
                    res_model,
                    weight_model,
                    device,
                )

                # Loss
                int_obs_int_sim_res = int_obs_int_sim_res.to(
                    device, dtype=torch.float32
                )
                # loss = _compute_loss(loss_fn_key, res_pred, int_obs_int_sim_res)
                loss, misfit_loss, weight_penalty_loss = compute_total_loss(
                    res_pred, int_obs_int_sim_res, weight, weight_norm_penalty
                )

                metrics["loss_hist_val"][epoch] += (
                    loss.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["misfit_loss_hist_val"][epoch] += (
                    misfit_loss.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["weight_penalty_loss_hist_val"][epoch] += (
                    weight_penalty_loss.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["weight_hist_val"][epoch] += (
                    weight.mean().item() * weight.shape[0]
                )

            metrics["loss_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["misfit_loss_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["weight_penalty_loss_hist_val"][epoch] /= len(
                val_dataloader.dataset
            )
            metrics["weight_hist_val"][epoch] /= len(val_dataloader.dataset)

        # Keep track of the best model
        if metrics[best_epoch_loss_key][epoch] < best_val_loss:
            best_res_model_state = res_model.state_dict()
            best_weight_model_state = weight_model.state_dict()
            best_val_loss = metrics[best_epoch_loss_key][epoch]
            best_model_epoch = epoch

        print(
            f"Epoch {epoch+1}\n"
            f"\tLoss: Train {metrics['loss_hist_train'][epoch]:.4f} Val {metrics['loss_hist_val'][epoch]:.4f}\n"
            f"\tMisfit Loss: Train {metrics['misfit_loss_hist_train'][epoch]:.4f}, Val {metrics['misfit_loss_hist_val'][epoch]:.4f}\n"
            f"\tWeight Penalty Loss: Train {metrics['weight_penalty_loss_hist_train'][epoch]:.4f}, Val {metrics['weight_penalty_loss_hist_val'][epoch]:.4f}\n"
            f"\tWeight: Train {metrics['weight_hist_train'][epoch]:.4f}, Val {metrics['weight_hist_val'][epoch]:.4f}"
        )
    return metrics, best_res_model_state, best_weight_model_state, best_model_epoch


def main(
    n_epochs: int = 25,
    batch_size: int = 2048,
    n_rels_used: int = 5,
    comment: str = "",
    max_dist: float = 100.0,
    save_best_val_model: bool = True,
    weight_decay: float = 0.0001,
    weight_norm_penalty: float = 0.005,
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

    WEIGHT_FC_UNITS = [16, 8]
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
    train_events = np.asarray(list(train_event_sites.keys()))
    val_events = np.asarray(list(val_event_sites.keys()))

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
    (
        site_to_site_features,
        event_site_features,
        event_site_to_site_features,
    ) = sr.ml.features.compute_scalar_features(
        events, event_sites, event_df, station_df, record_df, dist_matrix, max_dist
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
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Create the model
    n_rs_layers = len(RS_KERNEL_SIZES)
    padding = (
        [mlt.dl_utils.compute_same_conv_padding(n_periods, RS_KERNEL_SIZES[0])]
        * n_rs_layers
        if n_rs_layers > 0
        else []
    )

    res_model = sr.ml.models.ResponseSpectrumSimModel(
        RS_KERNEL_SIZES,
        RS_N_CHANNELS,
        padding,
        FC_UNITS,
        len(periods),
        scalar_features.n_scalar_features,
        n_periods,
        apply_sigmoid=False,
    ).to(device)

    weight_model = sr.ml.models.WeightModel(
        scalar_features.n_scalar_features,
        WEIGHT_FC_UNITS,
    ).to(device)

    # See https://pytorch.org/docs/stable/optim.html#per-parameter-options
    optimizer = torch.optim.Adam(
        list(res_model.parameters()) + list(weight_model.parameters()),
        lr=0.001,
        weight_decay=weight_decay,
    )

    print(f"Residual model summary")
    summary(
        res_model,
        input_size=[
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, scalar_features.n_scalar_features),
        ],
    )

    print(f"Weight model summary")
    summary(
        weight_model,
        input_size=(batch_size, scalar_features.n_scalar_features),
    )

    print(f"Running training")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    metrics, best_res_model_state, best_weight_model_state, best_epoch = train(
        res_model,
        weight_model,
        train_dataloader,
        val_dataloader,
        n_epochs,
        device,
        optimizer,
        weight_norm_penalty,
    )

    # Load the best model
    if save_best_val_model:
        res_model.load_state_dict(best_res_model_state)
        weight_model.load_state_dict(best_weight_model_state)
        print(
            f"Best model epoch: {best_epoch + 1}, "
            f"Vvalidation loss\n"
            f"\tTotal {metrics['loss_hist_val'][best_epoch]:.4f}\n"
            f"\tMisfit {metrics['misfit_loss_hist_val'][best_epoch]:.4f}\n"
            f"\tWeight penalty {metrics['weight_penalty_loss_hist_val'][best_epoch]:.4f}\n"
        )

    # Get predictions for the training and validation datasets
    print(f"Getting predictions")
    train_results_df = get_dataset_predictions(
        train_dataset, res_model, weight_model, device, weight_norm_penalty
    )
    val_results_df = get_dataset_predictions(
        val_dataset, res_model, weight_model, device, weight_norm_penalty
    )

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
            "weight_decay": weight_decay,
        },
        "data": {
            "db": str(db_ffp_orig),
        },
    }
    mlt.utils.write_to_yaml(meta_data, results_dir / "meta.yaml")

    # Save the model
    torch.save(res_model, results_dir / "model.pt")
    torch.save(weight_model, results_dir / "weight_model.pt")

    # Save the loss history
    with (results_dir / "metrics.pickle").open("wb") as f:
        pickle.dump(metrics, f)

    # Create a model visualisation
    draw_graph(
        res_model,
        input_size=[
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, 1, 31),
            (batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="res_model_vis",
        save_graph=True,
        directory=str(results_dir),
    )

    draw_graph(
        weight_model,
        input_size=(batch_size, scalar_features.n_scalar_features),
        expand_nested=True,
        filename="weight_model_vis",
        save_graph=True,
        directory=str(results_dir),
    )


if __name__ == "__main__":
    typer.run(main)
