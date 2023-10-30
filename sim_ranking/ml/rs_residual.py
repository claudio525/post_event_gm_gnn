import time
import pickle
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from torchinfo import summary
from torchview import draw_graph
import spatial_hazard as sh
import ml_tools as mlt

from . import data
from . import models
from . import features
from ..db import DB
from .. import constants

@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    weight_penalty_factor: float
    l2_reg: float
    lr: float

    n_channels: List[int]
    kernel_sizes: List[int]
    fc_units: List[int]

    @classmethod
    def from_yaml(cls, ffp: Path):
        params = mlt.utils.load_yaml(ffp)

        return cls(params["n_epochs"],
                     params["batch_size"],
                        params["weight_penalty_factor"],
                        params["l2_reg"],
                        params["lr"],
                        params["n_channels"],
                        params["kernel_sizes"],
                        params["fc_units"])

@dataclass
class RunParamsConfig:
    n_rels_used: int
    max_dist: float

    debug: bool
    device: str

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))

def get_dataset_predictions(
    dataset: data.BaseDataset,
    res_model: nn.Module,
    weight_model: nn.Module,
    hp_config: HyperParamsConfig,
    device: str,
):
    pred_dataloader = DataLoader(dataset, shuffle=False, batch_size=4096, num_workers=0)

    res_model.eval()

    results = []
    pred_res_results, true_res_results = [], []

    for i, (
        data_ind,
        obs_obs_obs_sim_res,
        obs_obs_int_sim_res,
        obs_sim_int_sim_rel,
        scalar_features,
        true_res,
        weight_scalar_features,
    ) in enumerate(pred_dataloader):
        # Forward pass
        pred_res = _get_res_prediction(
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_rel,
            scalar_features,
            res_model,
            device,
        )
        weights = _get_weight_prediction(weight_scalar_features, weight_model, device)

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
        true_res = true_res.to(device, dtype=torch.float32)
        misfit = compute_misfit(pred_res, true_res, aggregate=False)
        weighted_misfit = compute_weighted_misfit(
            pred_res, true_res, weights, aggregate=False
        )

        misfit_keys = np.char.add(constants.PSA_KEYS, "_misfit")
        meta_df[misfit_keys] = misfit.numpy(force=True)

        weighted_misfit_keys = np.char.add(constants.PSA_KEYS, "_weighted_misfit")
        meta_df[weighted_misfit_keys] = weighted_misfit.numpy(force=True)

        weight_keys = np.char.add(constants.PSA_KEYS, "_weight")
        meta_df[weight_keys] = weights.numpy(force=True)

        meta_df["misfit"] = torch.mean(misfit, dim=1).numpy(force=True)
        meta_df["weighted_misfit"] = torch.mean(weighted_misfit, dim=1).numpy(
            force=True
        )
        loss, _, weight_penalty = compute_loss(
            pred_res, true_res, weights, hp_config.weight_penalty_factor, dim=1
        )
        meta_df["loss"], meta_df["weight_penalty"] = loss.numpy(force=True), weight_penalty.numpy(force=True)
        meta_df["weight"] = torch.mean(weights, dim=1).numpy(force=True)

        results.append(meta_df)
        true_res_results.append(true_res.numpy(force=True))
        pred_res_results.append(pred_res.numpy(force=True))

    results_df = pd.concat(results)
    results_df.loc[:, np.char.add(constants.PSA_KEYS, "_true")] = np.concatenate(
        true_res_results, axis=0
    )
    results_df.loc[:, np.char.add(constants.PSA_KEYS, "_pred")] = np.concatenate(
        pred_res_results, axis=0
    )

    return results_df

def _get_weight_prediction(
    scalar_features,
    weight_model: nn.Module,
    device: str,
):
    # Put data onto the device
    # scalar_features = scalar_features[:, [-2]].to(device, dtype=torch.float32)
    scalar_features = scalar_features.to(device, dtype=torch.float32)
    weights = weight_model(scalar_features)

    # Normalise
    weights = (weights / torch.sum(weights, dim=0)) * (
        torch.ones(weights.shape[1], dtype=torch.float32).to(weights.device)
        * weights.shape[0]
    )

    # Scale such that the max weight is 1 per period
    weights = weights / weights.max(dim=0)[0]

    return weights
    # return torch.ones(weights.shape, dtype=torch.float32).to(weights.device)


def compute_misfit(
    pred_res: torch.Tensor,
    true_res: torch.Tensor,
    aggregate: bool = True,
    dim: int = None,
):
    misfit = torch.abs(pred_res - true_res)
    if aggregate:
        return torch.mean(misfit, dim=dim)
    return misfit


def compute_weighted_misfit(
    pred_res: torch.Tensor,
    true_res: torch.Tensor,
    weights: torch.Tensor,
    aggregate: bool = True,
    dim: int = None,
):
    weighted_misfit = weights * torch.abs(pred_res - true_res)
    if aggregate:
        return torch.mean(weighted_misfit, dim=dim)
    return weighted_misfit


def compute_loss(
    pred_res: torch.Tensor,
    true_res: torch.Tensor,
    weights: torch.Tensor,
    weight_penalty_factor: float,
    aggregate: bool = True,
    dim: int = None,
):
    weighted_misfit = compute_weighted_misfit(
        pred_res, true_res, weights, aggregate, dim
    )

    # weight_penalty_term = weight_penalty_factor * ((1 / weights) - 1)
    # return (
    #     weighted_misfit + torch.mean(weight_penalty_term, dim=dim),
    #     weight_penalty_term,
    # )

    weight_penalty_term = torch.mean(
        weight_penalty_factor * ((1 / weights) - 1), dim=dim
    )

    # weight_diff = 0.01 * ((0.5 * weights.shape[0]) - torch.sum(weights, dim=0))
    # total_weight_penalty_term = torch.mean(torch.where(weight_diff > 0, weight_diff, 0))

    # loss = weighted_misfit + total_weight_penalty_term + weight_penalty_term
    # return loss, total_weight_penalty_term, weight_penalty_term

    loss = weighted_misfit + weight_penalty_term
    return loss, torch.Tensor([0.0]), weight_penalty_term

    # return weighted_misfit, torch.Tensor([0.0])

def _get_res_prediction(
    obs_obs_obs_sim_res,
    obs_obs_int_sim_res,
    obs_sim_int_sim_rel,
    scalar_features,
    res_model: nn.Module,
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

    return res


def train(
    res_model: nn.Module,
    weight_model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    hp_config: HyperParamsConfig,
):
    metrics = {
        "loss_hist_train": torch.zeros(hp_config.n_epochs),
        "misfit_hist_train": torch.zeros(hp_config.n_epochs),
        "weighted_misfit_hist_train": torch.zeros(hp_config.n_epochs),
        "total_weight_penalty_hist_train": torch.zeros(hp_config.n_epochs),
        "weight_penalty_hist_train": torch.zeros(hp_config.n_epochs),
        "loss_hist_val": torch.zeros(hp_config.n_epochs),
        "misfit_hist_val": torch.zeros(hp_config.n_epochs),
        "weighted_misfit_hist_val": torch.zeros(hp_config.n_epochs),
        "total_weight_penalty_hist_val": torch.zeros(hp_config.n_epochs),
        "weight_penalty_hist_val": torch.zeros(hp_config.n_epochs),
    }

    best_epoch_loss_key = "loss_hist_val"
    best_val_loss = np.inf
    best_res_model_state = None
    best_weight_model_state = None
    best_model_epoch = None

    optimizer = torch.optim.Adam(
        [
            dict(params=res_model.parameters(), lr=hp_config.lr, weight_decay=hp_config.l2_reg),
            dict(params=weight_model.parameters(), lr=0.01),
        ]
    )

    for epoch in range(hp_config.n_epochs):
        if epoch == 25:
            optimizer.param_groups[1]["lr"] = 0.001
        if epoch == 75:
            optimizer.param_groups[1]["lr"] = 0.0001
        if epoch == 75:
            optimizer.param_groups[1]["lr"] = 0.0

        ### Training
        res_model.train()
        weight_model.train()
        for i, (
            _,
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_res,
            scalar_features,
            int_obs_int_sim_res,
            weight_scalar_features,
        ) in enumerate(train_dataloader):
            int_obs_int_sim_res = int_obs_int_sim_res.to(device, dtype=torch.float32)

            pred = _get_res_prediction(
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_res,
                scalar_features,
                res_model,
                device,
            )
            weights = _get_weight_prediction(weight_scalar_features, weight_model, device)
            misfit = compute_misfit(pred, int_obs_int_sim_res)
            weighted_misfit = compute_weighted_misfit(
                pred, int_obs_int_sim_res, weights
            )
            loss, total_weight_penalty, weight_penalty = compute_loss(
                pred, int_obs_int_sim_res, weights, hp_config.weight_penalty_factor
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metrics["loss_hist_train"][epoch] += (
                loss.item() * int_obs_int_sim_res.shape[0]
            )
            metrics["misfit_hist_train"][epoch] += (
                torch.mean(misfit).item() * int_obs_int_sim_res.shape[0]
            )
            metrics["weighted_misfit_hist_train"][epoch] += (
                torch.mean(weighted_misfit).item() * int_obs_int_sim_res.shape[0]
            )
            metrics["total_weight_penalty_hist_train"][epoch] += (
                total_weight_penalty.item() * int_obs_int_sim_res.shape[0]
            )
            metrics["weight_penalty_hist_train"][epoch] += (
                # torch.mean(weight_penalty).item() * int_obs_int_sim_res.shape[0]
                weight_penalty.item()
                * int_obs_int_sim_res.shape[0]
            )

        metrics["loss_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["misfit_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["weighted_misfit_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["weight_penalty_hist_train"][epoch] /= len(train_dataloader.dataset)
        metrics["total_weight_penalty_hist_train"][epoch] /= len(
            train_dataloader.dataset
        )

        ### Validation
        res_model.eval()
        weight_model.eval()
        with torch.no_grad():
            for i, (
                _,
                obs_obs_obs_sim_res,
                obs_obs_int_sim_res,
                obs_sim_int_sim_res,
                scalar_features,
                int_obs_int_sim_res,
                weight_scalar_features,
            ) in enumerate(val_dataloader):
                # Forward pass
                res_pred = _get_res_prediction(
                    obs_obs_obs_sim_res,
                    obs_obs_int_sim_res,
                    obs_sim_int_sim_res,
                    scalar_features,
                    res_model,
                    device,
                )

                val_sample_weights = _get_weight_prediction(
                    weight_scalar_features, weight_model, device
                )

                # Loss
                int_obs_int_sim_res = int_obs_int_sim_res.to(
                    device, dtype=torch.float32
                )
                val_misfit = compute_misfit(res_pred, int_obs_int_sim_res)
                val_weighted_misfit = compute_weighted_misfit(
                    res_pred, int_obs_int_sim_res, val_sample_weights
                )
                val_loss, val_total_weigth_penalty, val_weight_penalty = compute_loss(
                    res_pred,
                    int_obs_int_sim_res,
                    val_sample_weights,
                    hp_config.weight_penalty_factor,
                )

                metrics["loss_hist_val"][epoch] += (
                    val_loss.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["misfit_hist_val"][epoch] += (
                    val_misfit.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["weighted_misfit_hist_val"][epoch] += (
                    val_weighted_misfit.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["total_weight_penalty_hist_val"][epoch] += (
                    val_total_weigth_penalty.item() * int_obs_int_sim_res.shape[0]
                )
                metrics["weight_penalty_hist_val"][epoch] += (
                    # torch.mean(val_weight_penalty).item() * int_obs_int_sim_res.shape[0]
                    val_weight_penalty.item()
                    * int_obs_int_sim_res.shape[0]
                )

            metrics["loss_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["misfit_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["weighted_misfit_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["weight_penalty_hist_val"][epoch] /= len(val_dataloader.dataset)
            metrics["total_weight_penalty_hist_val"][epoch] /= len(
                val_dataloader.dataset
            )

        # Keep track of the best model
        if metrics[best_epoch_loss_key][epoch] < best_val_loss:
            best_res_model_state = res_model.state_dict()
            best_weight_model_state = weight_model.state_dict()
            best_val_loss = metrics[best_epoch_loss_key][epoch]
            best_model_epoch = epoch

        print(
            f"Epoch {epoch+1}\n"
            f"\tLoss: Train {metrics['loss_hist_train'][epoch]:.4f} Val {metrics['loss_hist_val'][epoch]:.4f}\n"
            f"\tMisfit: Train {metrics['misfit_hist_train'][epoch]:.4f} Val {metrics['misfit_hist_val'][epoch]:.4f}\n"
            f"\tWeighted Misfit: Train {metrics['weighted_misfit_hist_train'][epoch]:.4f} Val {metrics['weighted_misfit_hist_val'][epoch]:.4f}\n"
            f"\tWeight Penalty: Train {metrics['weight_penalty_hist_train'][epoch]:.4f} Val {metrics['weight_penalty_hist_val'][epoch]:.4f}\n"
            f"\tTotal Weight Penalty: Train {metrics['total_weight_penalty_hist_train'][epoch]:.4f} Val {metrics['total_weight_penalty_hist_val'][epoch]:.4f}\n"
        )
    return metrics, best_res_model_state, best_weight_model_state, best_model_epoch


def post_processing(
    res_model: nn.Module,
    weight_model: nn.Module,
    train_dataset: data.BaseDataset,
    val_dataset: data.BaseDataset,
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
    scalar_features: data.ScalarFeatures,
    weight_scalar_features: data.WeightScalarFeatures,
    metrics: Dict,
    data_metadata: Dict,
    comment: str,
):
    """
    Runs the post-processing, specifically:
    - Gets and saves predictions for the training and validation datasets
    - Writes the metadat
    - Saves the models
    - Creates model visualisations
    """
    # Get predictions for the training and validation datasets
    print(f"Getting predictions")
    train_results_df = get_dataset_predictions(
        train_dataset, res_model, weight_model, hp_config, run_config.device
    )
    val_results_df = get_dataset_predictions(
        val_dataset, res_model, weight_model, hp_config, run_config.device
    )

    # Save the results
    run_id = mlt.utils.create_run_id(False)
    (results_dir := run_config.results_dir / run_id).mkdir(exist_ok=False)
    print(f"Savings results, run-id {run_id}")

    train_results_df.to_csv(results_dir / "train_results.csv", index=True)
    val_results_df.to_csv(results_dir / "val_results.csv", index=True)

    # Write the metadata
    metadata = {
        "data": data_metadata,
        "comment": comment,
    }
    mlt.utils.write_to_yaml(metadata, results_dir / "meta.yaml")

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
            (hp_config.batch_size, 1, 31),
            (hp_config.batch_size, 1, 31),
            (hp_config.batch_size, 1, 31),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="res_model_vis",
        save_graph=True,
        directory=str(results_dir),
    )

    draw_graph(
        weight_model,
        input_size=(hp_config.batch_size, weight_scalar_features.n_scalar_features),
        expand_nested=True,
        filename="weight_model_vis",
        save_graph=True,
        directory=str(results_dir),
    )

def create_models(
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
    n_periods: int,
    scalar_features: data.ScalarFeatures,
    weight_scalar_features: data.WeightScalarFeatures,
    print_summary: bool = True,
):
    """Creates the Residual prediction and weighting model"""

    ### Residual model
    n_rs_layers = len(hp_config.kernel_sizes)
    padding = (
        [mlt.dl_utils.compute_same_conv_padding(n_periods, hp_config.kernel_sizes[0])]
        * n_rs_layers
        if n_rs_layers > 0
        else []
    )

    res_model = models.ResponseSpectrumSimModel(
        hp_config.kernel_sizes,
        hp_config.n_channels,
        padding,
        hp_config.fc_units,
        n_periods,
        scalar_features.n_scalar_features,
        n_periods,
        apply_sigmoid=False,
    ).to(run_config.device)

    ### Weighting model
    # weight_model = sr.ml.models.ConstrainedWeightModel(len(periods)).to(device)
    # weight_model = sr.ml.models.MLPWeightModel(
    #     len(periods), WEIGHT_UNITS, weight_scalar_features.n_scalar_features
    # ).to(device)
    weight_model = models.ExpWeightModel(n_periods).to(run_config.device)

    if print_summary:
        print(f"Residual model summary")
        summary(
            res_model,
            input_size=[
                (hp_config.batch_size, 1, 31),
                (hp_config.batch_size, 1, 31),
                (hp_config.batch_size, 1, 31),
                (hp_config.batch_size, scalar_features.n_scalar_features),
            ],
        )

        print(f"Weight model summary")
        summary(
            weight_model,
            input_size=(hp_config.batch_size, weight_scalar_features.n_scalar_features),
        )

    return res_model, weight_model


def prep(
    val_events: np.ndarray,
    run_config: RunParamsConfig,
    hp_config: HyperParamsConfig,
):
    """
    Performs
    - Data loading
    - Computation of distance matrix
    - Splitting into validation and training events
    - Compute site-combinations (i.e. samples)
    - Compute and preprocesses scalar features for both main and weight model
    - Creates training and validation datasets & dataloaders
    - Collects relevant metadata
    """

    # Scalar features
    SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

    # WEIGHT_SITE_TO_SITE_FEATURE_KEYS = ["dist", "vs30_dist"]
    # WEIGHT_EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angle"]
    WEIGHT_SITE_TO_SITE_FEATURE_KEYS = ["dist", "vs30_dist"]
    WEIGHT_EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

    db_ffp_orig = "$wdata/sim_ranking/db/gm_db.sqlite"
    db_ffp = Path(os.path.expandvars(db_ffp_orig))

    db = DB(db_ffp)
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

    # Use all non-validation events for training
    train_events = np.setdiff1d(events, val_events)

    # Get the training and validation dataset site combinations
    print(f"Creating site combinations")
    train_site_combs, train_event_sites = data.compute_site_combinations(
        event_sites,
        train_events,
        dist_matrix,
        sites_to_use=train_sites,
        max_dist=run_config.max_dist,
    )
    val_site_combs, val_event_sites = data.compute_site_combinations(
        event_sites,
        val_events,
        dist_matrix,
        sites_to_use=val_sites,
        max_dist=run_config.max_dist,
    )
    train_events = np.asarray(list(train_event_sites.keys()))
    val_events = np.asarray(list(val_event_sites.keys()))

    # Run pre-processing for the site features
    # TODO: This should be updated such that the normalisation
    # only happens on training sites, not all sites
    print(f"Pre-processing site features")
    site_features_df, site_feature_stats = features.preprocess_site_features(
        station_df, SITE_FEATURE_KEYS
    )

    # Computed the site-to-site features
    print(f"Computing scalar features")
    (
        site_to_site_features,
        event_site_features,
        event_site_to_site_features,
    ) = features.compute_scalar_features(
        events,
        event_sites,
        event_df,
        station_df,
        record_df,
        dist_matrix,
        run_config.max_dist,
    )
    (
        weight_site_to_site_features,
        weight_event_site_to_site_features,
    ) = features.compute_weight_features(
        station_df, event_df, events, event_sites, dist_matrix, run_config.max_dist
    )

    scalar_features = data.ScalarFeatures(
        site_features_df,
        SITE_FEATURE_KEYS,
        site_to_site_features,
        SITE_TO_SITE_FEATURE_KEYS,
        event_site_features,
        EVENT_SITE_FEATURE_KEYS,
        event_site_to_site_features,
        EVENT_SITE_TO_SITE_FEATURE_KEYS,
    )

    weight_scalar_features = data.WeightScalarFeatures(
        weight_site_to_site_features,
        WEIGHT_SITE_TO_SITE_FEATURE_KEYS,
        weight_event_site_to_site_features,
        WEIGHT_EVENT_SITE_TO_SITE_FEATURE_KEYS,
    )

    # Create the training and validation dataset
    print(f"Creating datasets and dataloaders")
    start_time = time.time()
    train_dataset = data.WeightRSResidualDataset(
        train_event_sites,
        train_site_combs,
        db,
        run_config.n_rels_used,
        site_features_df,
        constants.PERIODS,
        constants.PSA_KEYS,
        scalar_features,
        weight_scalar_features,
    )
    print(f"Took {time.time() - start_time} to create train dataset")
    start_time = time.time()
    val_dataset = data.WeightRSResidualDataset(
        val_event_sites,
        val_site_combs,
        db,
        run_config.n_rels_used,
        site_features_df,
        constants.PERIODS,
        constants.PSA_KEYS,
        scalar_features,
        weight_scalar_features,
    )
    print(f"Took {time.time() - start_time} to create val dataset")

    # Create the dataloaders
    N_DATALOADER_WORKERS = 0 if run_config.debug else 2
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=N_DATALOADER_WORKERS,
        pin_memory=True,
        persistent_workers=True if N_DATALOADER_WORKERS > 0 else False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=N_DATALOADER_WORKERS,
        pin_memory=True,
        persistent_workers=True if N_DATALOADER_WORKERS > 0 else False,
    )

    data_summary = {
        "site_feature_stats": site_feature_stats.to_dict(),
        "n_rels_used": run_config.n_rels_used,
        # "n_val_sites": N_VAL_SITES,
        "val_events": val_events.astype(str).tolist(),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "train_events": train_events.astype(str).tolist(),
        "site_features": scalar_features.site_feature_keys,
        "site_to_site_features": scalar_features.site_to_site_feature_keys,
        "event_site_features": scalar_features.event_site_feature_keys,
        "event_site_to_site_features": scalar_features.event_site_to_site_feature_keys,
        "max_dist": run_config.max_dist,
    }

    return (
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        scalar_features,
        weight_scalar_features,
        data_summary,
    )