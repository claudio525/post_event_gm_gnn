import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, List
from collections import deque

import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchinfo import summary
from tqdm import tqdm
from torchview import draw_graph

import ml_tools as mlt
import spatial_hazard as sh

from . import data as ml_data
from . import features
from . import models
from ..db import DB
from .. import constants


def compute_res_area(rs_obs: np.ndarray, rs_sim: np.ndarray):
    """
    Computes the similarity score for response spectrum
    based on area under the (absolute) residual curve

    Parameters
    ----------
    rs_obs: array of floats
        Observed response spectrum
        Format [n_periods]
    rs_sim: array of floats
        Simulation realisation response spectra
        Format [n_realisations, n_periods]

    Returns
    -------
    array of floats
        Similarity score for each realisation
        Format [n_realisations]
    """
    # Compute the residual
    res = np.log(rs_obs[..., None]) - np.log(rs_sim)
    res_area = np.trapz(np.abs(res), axis=1)

    return res_area


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

        return cls(
            params["n_epochs"],
            params["batch_size"],
            params["weight_penalty_factor"],
            params["l2_reg"],
            params["lr"],
            params["n_channels"],
            params["kernel_sizes"],
            params["fc_units"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "weight_penalty_factor": self.weight_penalty_factor,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "n_channels": self.n_channels,
            "kernel_sizes": self.kernel_sizes,
            "fc_units": self.fc_units,
        }


class PairDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: ml_data.ScalarFeatures,
        pSA_mean: np.ndarray,
        pSA_std: np.ndarray,
        max_n_rels: int,
    ):
        self.db = db
        self.event_sites = event_sites
        self.site_combs = site_combs

        self.events = np.asarray(list(event_sites.keys()))

        self.periods = np.asarray(periods)
        self.pSA_keys = np.asarray(pSA_keys)

        self.pSA_mean = pSA_mean
        self.pSA_std = pSA_std

        self.scalar_features = scalar_features

        self.n_events = len(self.event_sites)

        # Compute the number of samples per event
        self.event_rels = {}
        self.n_rels_event = {}
        self.n_samples_event = []
        for cur_event, cur_sites in event_sites.items():
            cur_sim_data = db.get_sim_data(cur_event, cur_sites)

            # cur_n_rels = np.unique(cur_sim_data.rel_id.values).size
            # self.n_rels_event[cur_event] = cur_n_rels if cur_n_rels < max_n_rels else max_n_rels
            self.event_rels[cur_event] = np.random.choice(
                np.unique(cur_sim_data.rel_id.values.astype(str)),
                max_n_rels,
                replace=False,
            )
            cur_n_rels = self.n_rels_event[cur_event] = self.event_rels[cur_event].size

            cur_n_samples = self.site_combs[cur_event].shape[0] * int(
                (cur_n_rels ** 2 - cur_n_rels) / 2
            )
            self.n_samples_event.append(cur_n_samples)
        self.cum_n_samples_event = np.cumsum(self.n_samples_event)

        # Create the feature tensor
        self.scalar_features_tensor = ml_data.create_scalar_feature_tensor(
            self.events, self.event_sites, self.scalar_features
        )

        # Create the (normalised) pSA inputs in the format
        # obs: (n_sites, n_periods)
        # sim: (n_sites, n_periods, n_rels)
        # And create the residual area scores
        # format: (n_sites, n_rels)
        self.obs_pSA = {}
        self.sim_pSA = {}
        self.res_area = {}
        for cur_event, cur_sites in event_sites.items():
            # Observed
            cur_obs_df = db.get_obs_data(cur_event, cur_sites)
            # self.obs_pSA[cur_event] = cur_obs_df.loc[cur_sites, self.pSA_keys].values
            self.obs_pSA[cur_event] = (
                cur_obs_df.loc[cur_sites, self.pSA_keys].values
                - self.pSA_mean[self.pSA_keys].values
            ) / self.pSA_std[self.pSA_keys].values

            # Get the simulation data
            cur_sim_df = db.get_sim_data(cur_event, cur_sites)
            cur_sim_data = np.full(
                (cur_sites.size, self.periods.size, self.n_rels_event[cur_event]),
                fill_value=np.nan,
            )

            for ix, cur_rel in enumerate(self.event_rels[cur_event]):
                cur_rel_data = (
                    cur_sim_df.loc[cur_sim_df.rel_id == cur_rel]
                    .set_index("site_id")
                    .loc[cur_sites, self.pSA_keys]
                    .values
                )
                cur_sim_data[:, :, ix] = cur_rel_data

            # Need to compute residual area before normalizing
            cur_res_area = compute_res_area(
                cur_obs_df.loc[cur_sites, self.pSA_keys].values, cur_sim_data
            )
            self.res_area[cur_event] = cur_res_area

            # self.sim_pSA[cur_event] = cur_sim_data
            self.sim_pSA[cur_event] = (
                cur_sim_data - self.pSA_mean[self.pSA_keys].values[None, :, None]
            ) / self.pSA_std[self.pSA_keys].values[None, :, None]

    def __len__(self):
        return int(np.sum(self.n_samples_event))

    def get_metadata(self, idx: int):
        """Get the metadata for a specific sample"""
        event_ix, event, site_comb_ix, rel_1_ix, rel_2_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_comb_ix, 0]
        site_obs_ix = self.site_combs[event][site_comb_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]

        rel_1 = self.event_rels[event][rel_1_ix]
        rel_2 = self.event_rels[event][rel_2_ix]

        return (event, site_int, site_obs, rel_1, rel_2)

    def get_indices(self, idx: int):
        # Have to it this way, as some events may not have samples
        event_ix = np.flatnonzero(idx - self.cum_n_samples_event < 0)[0]

        within_event_ix = (
            idx - self.cum_n_samples_event[event_ix - 1] if event_ix > 0 else idx
        )

        event = self.events[event_ix]

        n_rels = self.n_rels_event[event]
        n_rel_combs = int((n_rels * (n_rels - 1)) / 2)

        within_site_ix = int(within_event_ix % n_rel_combs)

        # Get the realisation indices
        # Based on https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        rel_1_ix = int(
            n_rels
            - 2
            - np.floor(
                np.sqrt(-8 * within_site_ix + 4 * n_rels * (n_rels - 1) - 7) / 2 - 0.5
            )
        )
        rel_2_ix = int(
            within_site_ix
            + rel_1_ix
            + 1
            - n_rels * (n_rels - 1) / 2
            + (n_rels - rel_1_ix) * ((n_rels - rel_1_ix) - 1) / 2
        )

        site_comb_ix = within_event_ix // n_rel_combs

        return event_ix, event, site_comb_ix, rel_1_ix, rel_2_ix

    def __getitem__(self, idx: int):
        event_ix, event, site_comb_ix, rel_1_ix, rel_2_ix = self.get_indices(idx)

        site_int_ix = self.site_combs[event][site_comb_ix, 0]
        site_obs_ix = self.site_combs[event][site_comb_ix, 1]

        return (
            idx,
            self.scalar_features_tensor[event][site_int_ix, site_obs_ix, :],
            self.sim_pSA[event][site_int_ix, :, rel_1_ix],
            self.sim_pSA[event][site_int_ix, :, rel_2_ix],
            self.sim_pSA[event][site_obs_ix, :, rel_1_ix],
            self.sim_pSA[event][site_obs_ix, :, rel_2_ix],
            self.obs_pSA[event][site_obs_ix],
            self.res_area[event][site_int_ix, rel_1_ix],
            self.res_area[event][site_int_ix, rel_2_ix],
            # self.obs_pSA[event][site_int_ix],
        )


@dataclass
class RunParamsConfig:
    max_dist: float
    max_n_rels: int

    debug: bool
    device: str

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))


def compute_loss(
    loss: nn.Module,
    pred: torch.Tensor,
    res_area_1: torch.Tensor,
    res_area_2: torch.Tensor,
    device: str,
    reduce: bool = True,
):
    # Classification, is Rel 1 better than Rel 2?
    true = (res_area_1 < res_area_2)[:, None].to(device, dtype=torch.float32)
    loss_value = loss(pred, true)

    res_diff = torch.abs(res_area_1 - res_area_2).to(device, dtype=torch.float32)

    # Compute and normalize the sample weights
    cur_weights = 1 - torch.exp(-0.075 * res_diff) + 0.05
    cur_weights = cur_weights * (cur_weights.shape[0] / cur_weights.sum())

    weighted_loss_value = cur_weights * loss_value.ravel()

    if reduce:
        return loss_value.mean(), weighted_loss_value.mean(), true, cur_weights

    return loss_value, weighted_loss_value, true, cur_weights


def train(
    ranking_model: nn.Module,
    train_dataset: PairDataset,
    val_dataset: PairDataset,
    device: str,
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
):
    metrics = {
        "bce_loss_hist_train": torch.zeros(hp_config.n_epochs),
        "loss_hist_train": torch.zeros(hp_config.n_epochs),
        "acc_hist_train": torch.zeros(hp_config.n_epochs),
        "bce_loss_hist_val": torch.zeros(hp_config.n_epochs),
        "loss_hist_val": torch.zeros(hp_config.n_epochs),
        "acc_hist_val": torch.zeros(hp_config.n_epochs),
    }

    best_epoch_key = "acc_hist_val"
    best_val_acc = -np.inf
    best_model_state = None

    optimizer = torch.optim.Adam(
        ranking_model.parameters(), lr=hp_config.lr, weight_decay=hp_config.l2_reg
    )

    n_workers = 0 if run_config.debug else 4
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True if n_workers > 0 else False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True if n_workers > 0 else False,
    )

    # loss = nn.BCELoss()
    loss = nn.BCEWithLogitsLoss(reduction="none")
    for epoch_ix in range(hp_config.n_epochs):
        ranking_model.train()
        iter_loop = tqdm(train_dataloader)
        iter_loop.set_description(f"Epoch {epoch_ix}/{hp_config.n_epochs}")
        for i, (
            _,
            scalar_features,
            int_sim_pSA_rel_1,
            int_sim_pSA_rel_2,
            obs_sim_pSA_rel_1,
            obs_sim_pSA_rel_2,
            obs_obs_pSA,
            res_area_rel_1,
            res_area_rel_2,
        ) in enumerate(iter_loop):

            pred = get_prediction(
                ranking_model,
                scalar_features,
                int_sim_pSA_rel_1,
                int_sim_pSA_rel_2,
                obs_sim_pSA_rel_1,
                obs_sim_pSA_rel_2,
                obs_obs_pSA,
                device,
            )

            bce_loss_value, cur_weighted_bce_loss_value, true, _ = compute_loss(
                loss, pred, res_area_rel_1, res_area_rel_2, device
            )
            cur_loss_value = cur_weighted_bce_loss_value

            optimizer.zero_grad(set_to_none=True)
            cur_loss_value.backward()
            optimizer.step()

            metrics["loss_hist_train"][epoch_ix] += cur_loss_value.item()
            metrics["bce_loss_hist_train"][epoch_ix] += bce_loss_value.item()

            n_correct = (
                ((torch.nn.functional.sigmoid(pred) >= 0.5).float() == true)
                .sum()
                .item()
            )
            metrics["acc_hist_train"][epoch_ix] += n_correct

            iter_loop.set_postfix(
                {"loss": cur_loss_value.item(), "acc": n_correct / pred.size(0)}
            )

        metrics["loss_hist_train"][epoch_ix] /= len(train_dataloader)
        metrics["bce_loss_hist_train"][epoch_ix] /= len(train_dataloader)
        metrics["acc_hist_train"][epoch_ix] /= len(train_dataloader.dataset)

        # Validation
        with torch.no_grad():
            ranking_model.eval()
            for i, (
                _,
                scalar_features,
                int_sim_pSA_rel_1,
                int_sim_pSA_rel_2,
                obs_sim_pSA_rel_1,
                obs_sim_pSA_rel_2,
                obs_obs_pSA,
                res_area_rel_1,
                res_area_rel_2,
            ) in enumerate(val_dataloader):

                pred = get_prediction(
                    ranking_model,
                    scalar_features,
                    int_sim_pSA_rel_1,
                    int_sim_pSA_rel_2,
                    obs_sim_pSA_rel_1,
                    obs_sim_pSA_rel_2,
                    obs_obs_pSA,
                    device,
                )

                bce_loss_value, cur_weighted_bce_loss_value, true, _ = compute_loss(
                    loss, pred, res_area_rel_1, res_area_rel_2, device
                )
                cur_loss_value = cur_weighted_bce_loss_value

                metrics["loss_hist_val"][epoch_ix] += cur_loss_value.item()
                metrics["bce_loss_hist_val"][epoch_ix] += bce_loss_value.item()

                n_correct = (
                    ((torch.nn.functional.sigmoid(pred) >= 0.5).float() == true)
                    .sum()
                    .item()
                )
                metrics["acc_hist_val"][epoch_ix] += n_correct

            metrics["loss_hist_val"][epoch_ix] /= len(val_dataloader)
            metrics["bce_loss_hist_val"][epoch_ix] /= len(val_dataloader)
            metrics["acc_hist_val"][epoch_ix] /= len(val_dataloader.dataset)

            # Keep track of the best model
            if metrics[best_epoch_key][epoch_ix] > best_val_acc:
                best_model_state = ranking_model.state_dict()
                best_val_acc = metrics[best_epoch_key][epoch_ix]
                best_model_epoch = epoch_ix

        print(f"Epoch {epoch_ix + 1}/{hp_config.n_epochs}")
        print(
            f"\tTraining - Loss: {metrics['loss_hist_train'][epoch_ix]:.4f}, "
            f"Accuracy: {metrics['acc_hist_train'][epoch_ix]:.4f}, "
            f"BCELoss: {metrics['bce_loss_hist_train'][epoch_ix]:.4f}"
        )
        print(
            f"\tValidation - Loss: {metrics['loss_hist_val'][epoch_ix]:.4f}, "
            f"Accuracy: {metrics['acc_hist_val'][epoch_ix]:.4f}, "
            f"BCELoss: {metrics['bce_loss_hist_val'][epoch_ix]:.4f}"
        )

    return metrics, best_model_state, best_model_epoch



def get_prediction(
    ranking_model: nn.Module,
    scalar_features: torch.Tensor,
    int_sim_pSA_rel_1: torch.Tensor,
    int_sim_pSA_rel_2: torch.Tensor,
    obs_sim_pSA_rel_1: torch.Tensor,
    obs_sim_pSA_rel_2: torch.Tensor,
    obs_obs_pSA: torch.Tensor,
    device: str,
):
    pSA_data = torch.cat(
        (
            int_sim_pSA_rel_1[:, None, :],
            int_sim_pSA_rel_2[:, None, :],
            obs_sim_pSA_rel_1[:, None, :],
            obs_sim_pSA_rel_2[:, None, :],
            obs_obs_pSA[:, None, :],
        ),
        dim=1,
    ).to(device, dtype=torch.float32)

    scalar_features = scalar_features.to(device, dtype=torch.float32)

    pred = ranking_model(pSA_data, scalar_features)

    return pred


def get_dataset_predictions(
    dataset: PairDataset, ranking_model: nn.Module, device: str
):

    pred_dataloader = DataLoader(
        dataset, shuffle=False, batch_size=4096, num_workers=0, pin_memory=True
    )

    results = []
    with torch.no_grad():
        loss = nn.BCEWithLogitsLoss(reduction="none")
        ranking_model.eval()
        for i, (
            data_ind,
            scalar_features,
            int_sim_pSA_rel_1,
            int_sim_pSA_rel_2,
            obs_sim_pSA_rel_1,
            obs_sim_pSA_rel_2,
            obs_obs_pSA,
            res_area_rel_1,
            res_area_rel_2,
        ) in enumerate(pred_dataloader):
            pred = get_prediction(
                ranking_model,
                scalar_features,
                int_sim_pSA_rel_1,
                int_sim_pSA_rel_2,
                obs_sim_pSA_rel_1,
                obs_sim_pSA_rel_2,
                obs_obs_pSA,
                device,
            )
            bce_loss_value, weighted_loss_value, true, weights = compute_loss(
                loss, pred, res_area_rel_1, res_area_rel_2, device, reduce=False
            )

            meta_df = pd.DataFrame(
                [dataset.get_metadata(ix) for ix in data_ind],
                columns=["event_id", "site_int", "site_obs", "rel_1", "rel_2"],
            )
            meta_df["pred"] = (
                torch.nn.functional.sigmoid(pred).numpy(force=True).ravel()
            )
            meta_df["true"] = true.numpy(force=True).ravel()
            meta_df["bce_loss"] = bce_loss_value.numpy(force=True).ravel()
            meta_df["weighted_loss"] = weighted_loss_value.numpy(force=True).ravel()
            meta_df["weights"] = weights.numpy(force=True).ravel()

            results.append(meta_df)

        results = pd.concat(results, axis=0)
        return results


def post_processing(
    ranking_model: nn.Module,
    train_dataset: PairDataset,
    val_dataset: PairDataset,
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
    metrics: Dict,
    best_epoch: int,
    scalar_features: ml_data.ScalarFeatures,
    data_metadata: Dict,
):
    (cur_out_dir := run_config.results_dir / mlt.utils.create_run_id(False)).mkdir()

    # Get predictions
    print(f"Getting dataset predictions")
    train_results = get_dataset_predictions(
        train_dataset, ranking_model, run_config.device
    )
    val_results = get_dataset_predictions(val_dataset, ranking_model, run_config.device)

    # Save results
    train_results.to_csv(cur_out_dir / "train_results.csv")
    val_results.to_csv(cur_out_dir / "val_results.csv")

    # Save loss history
    pd.to_pickle(metrics, cur_out_dir / "metrics.pickle")

    # Save the model
    torch.save(ranking_model, cur_out_dir / "model.pt")

    # Metadata
    metadata = {
        "hp_config": hp_config.to_dict(),
        "best_epoch": best_epoch,
        "data": data_metadata,
    }
    mlt.utils.write_to_yaml(metadata, cur_out_dir / "meta.yaml")

    # Create model visualisation
    draw_graph(
        ranking_model,
        input_size=[
            (hp_config.batch_size, 5, 31),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="ranking_model_vis",
        save_graph=True,
        directory=str(cur_out_dir),
    )




def data_prep(
    event_sites: Dict[str, np.ndarray],
    train_events: np.ndarray,
    val_events: np.ndarray,
    train_sites: np.ndarray,
    val_int_sites: np.ndarray,
    events: np.ndarray,
    run_config: RunParamsConfig,
    db: DB,
):
    ### Constants
    # Scalar features
    SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    # SITE_FEATURE_KEYS = ["vs30"]
    SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    # EVENT_SITE_FEATURE_KEYS = []
    EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

    WEIGHT_SITE_TO_SITE_FEATURE_KEYS = ["dist", "vs30_dist"]
    WEIGHT_EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

    event_df = db.get_event_df()
    record_df = db.get_record_df()

    print(f"Computing distance matrix")
    station_df = db.get_site_df()
    all_sites = db.get_avail_sites()
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    # Get the training and validation dataset site combinations
    print(f"Creating site combinations")
    train_site_combs, train_event_sites = ml_data.compute_site_combinations(
        event_sites,
        train_events,
        dist_matrix,
        train_sites,
        train_sites,
        max_dist=run_config.max_dist,
    )
    val_site_combs, val_event_sites = ml_data.compute_site_combinations(
        event_sites,
        val_events,
        dist_matrix,
        train_sites,
        val_int_sites,
        max_dist=run_config.max_dist,
    )

    ### Scalar Features
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
    # (
    #     weight_site_to_site_features,
    #     weight_event_site_to_site_features,
    # ) = features.compute_weight_features(
    #     station_df, event_df, events, event_sites, dist_matrix, run_config.max_dist
    # )
    #
    scalar_features = ml_data.ScalarFeatures(
        site_features_df,
        SITE_FEATURE_KEYS,
        site_to_site_features,
        SITE_TO_SITE_FEATURE_KEYS,
        event_site_features,
        EVENT_SITE_FEATURE_KEYS,
        event_site_to_site_features,
        EVENT_SITE_TO_SITE_FEATURE_KEYS,
    )

    # weight_scalar_features = ml_data.WeightScalarFeatures(
    #     weight_site_to_site_features,
    #     WEIGHT_SITE_TO_SITE_FEATURE_KEYS,
    #     weight_event_site_to_site_features,
    #     WEIGHT_EVENT_SITE_TO_SITE_FEATURE_KEYS,
    # )

    # Compute mean and standard deviation for each period
    # for normalisation (only training events)
    obs_data = db.get_obs_df()
    pSA_mean = np.mean(obs_data.loc[:, constants.PSA_KEYS], axis=0)
    pSA_std = np.std(obs_data.loc[:, constants.PSA_KEYS], axis=0)

    # Create the datasets
    train_dataset = PairDataset(
        train_event_sites,
        train_site_combs,
        db,
        constants.PERIODS,
        constants.PSA_KEYS,
        scalar_features,
        pSA_mean,
        pSA_std,
        run_config.max_n_rels,
    )

    val_dataset = PairDataset(
        val_event_sites,
        val_site_combs,
        db,
        constants.PERIODS,
        constants.PSA_KEYS,
        scalar_features,
        pSA_mean,
        pSA_std,
        run_config.max_n_rels,
    )

    metadata = {
        "train_sites": train_sites.tolist(),
        "val_int_sites": val_int_sites.tolist(),
        "train_events": train_events.tolist(),
        "val_events": val_events.tolist(),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "max_dist": run_config.max_dist,
        "max_n_rels": run_config.max_n_rels,
        "features": {
            "site_features":  scalar_features.site_feature_keys,
            "site_to_site_features": scalar_features.site_to_site_feature_keys,
            "event_site_features": scalar_features.event_site_feature_keys,
            "event_site_to_site_features": scalar_features.event_site_to_site_feature_keys,
            "site_feature_stats": site_feature_stats.to_dict(),
            "pSA_mean": pSA_mean.to_dict(),
            "pSA_std": pSA_std.to_dict(),
            "n_scalar_features": int(scalar_features.n_scalar_features),
        },
    }

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, scalar_features, metadata


def create_model(hp_config: HyperParamsConfig, scalar_features: ml_data.ScalarFeatures):
    # Create the model
    n_conv_layers = len(hp_config.kernel_sizes)
    padding = (
        [
            mlt.dl_utils.compute_same_conv_padding(
                len(constants.PERIODS), hp_config.kernel_sizes[0]
            )
        ]
        * n_conv_layers
        if n_conv_layers > 0
        else []
    )

    ranking_model = models.PairWiseModel(
        hp_config.kernel_sizes,
        hp_config.n_channels,
        padding,
        hp_config.fc_units,
        scalar_features.n_scalar_features,
        len(constants.PERIODS),
    )

    print(f"Ranking model summary")
    summary(
        ranking_model,
        input_size=[
            (hp_config.batch_size, 5, 31),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
    )

    return ranking_model
