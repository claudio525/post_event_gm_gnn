import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, List

import einops
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import numba as nb
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


@dataclass
class RunParamsConfig:
    max_dist: float
    n_rels: int
    ims: Sequence[str]
    im_weights: np.ndarray

    debug: bool
    device: str

    results_dir: Path = None

    def __post_init__(self):
        if self.results_dir is None:
            self.results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))

    def to_dict(self):
        return {
            "max_dist": self.max_dist,
            "n_rels": self.n_rels,
            "ims": self.ims,
            "im_weights": self.im_weights.tolist(),
            "debug": self.debug,
            "device": self.device,
        }


@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    l2_reg: float
    lr: Sequence[float]
    lr_epochs: Sequence[int]

    misfit_fn: str

    use_im_sim_site_obs: bool
    use_im_sim_site_int: bool
    use_im_obs_site_obs: bool

    use_res_site_obs: bool
    use_res_sim_site_obs_sim_site_int: bool
    use_res_obs_site_obs_sim_site_int: bool

    ind_fc_units: Sequence[int]

    combined_model: bool
    comb_fc_units: Sequence[int]

    def __post_init__(self):
        self.n_im_features = sum(
            [
                self.use_im_sim_site_obs,
                self.use_im_sim_site_int,
                self.use_im_obs_site_obs,
                self.use_res_site_obs,
                self.use_res_sim_site_obs_sim_site_int,
                self.use_res_obs_site_obs_sim_site_int,
            ]
        )

    @classmethod
    def from_yaml(cls, ffp: Path, n_epochs: int):
        params = mlt.utils.load_yaml(ffp)

        return cls(
            n_epochs,
            params["batch_size"],
            params["l2_reg"],
            params["lr"],
            params["lr_epochs"],
            params["misfit_fn"],
            params["im_sim_site_obs"],
            params["im_sim_site_int"],
            params["im_obs_site_obs"],
            params["res_site_obs"],
            params["res_sim_site_obs_sim_site_int"],
            params["res_obs_site_obs_sim_site_int"],
            params["ind_fc_units"],
            params["use_combined_model"],
            params["comb_fc_units"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "lr_epochs": self.lr_epochs,
            "misfit_fn": self.misfit_fn,
            "im_sim_site_obs": self.use_im_sim_site_obs,
            "im_sim_site_int": self.use_im_sim_site_int,
            "im_obs_site_obs": self.use_im_obs_site_obs,
            "res_site_obs": self.use_res_site_obs,
            "res_sim_site_obs_sim_site_int": self.use_res_sim_site_obs_sim_site_int,
            "res_obs_site_obs_sim_site_int": self.use_res_obs_site_obs_sim_site_int,
            "n_im_features": self.n_im_features,
            "ind_fc_units": self.ind_fc_units,
            "use_combined_model": self.combined_model,
            "comb_fc_units": self.comb_fc_units,
        }


def data_preb(
    event_sites: Dict[str, np.ndarray],
    train_events: np.ndarray,
    val_events: np.ndarray,
    train_sites: np.ndarray,
    val_int_sites: np.ndarray,
    events: np.ndarray,
    run_config: RunParamsConfig,
    hp_config: HyperParamsConfig,
    db: DB,
    sim_corr_dir: Path = None,
):
    # Scalar features
    SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

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

    # Compute the site-to-site features
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

    # Compute mean and standard deviation for each period
    # for normalisation (only training events)
    obs_data = db.get_obs_df()
    ims_mean = np.mean(np.log(obs_data.loc[:, run_config.ims]), axis=0)
    ims_std = np.std(np.log(obs_data.loc[:, run_config.ims]), axis=0)

    # Create the datasets
    print(f"Creating datasets")
    train_dataset = ProbDataset(
        train_event_sites,
        train_site_combs,
        db,
        run_config,
        scalar_features,
        ims_mean,
        ims_std,
        hp_config,
        sim_corr_dir=sim_corr_dir,
    )

    val_dataset = ProbDataset(
        val_event_sites,
        val_site_combs,
        db,
        run_config,
        scalar_features,
        ims_mean,
        ims_std,
        hp_config,
        sim_corr_dir=sim_corr_dir,
    )

    metadata = {
        "train_sites": train_sites.tolist(),
        "val_int_sites": val_int_sites.tolist(),
        "train_events": train_events.tolist(),
        "val_events": val_events.tolist(),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "max_dist": run_config.max_dist,
        "n_rels": run_config.n_rels,
        "features": {
            "site_features": scalar_features.site_feature_keys,
            "site_to_site_features": scalar_features.site_to_site_feature_keys,
            "event_site_features": scalar_features.event_site_feature_keys,
            "event_site_to_site_features": scalar_features.event_site_to_site_feature_keys,
            "site_feature_stats": site_feature_stats.to_dict(),
            "ims_mean": ims_mean.to_dict(),
            "ims_std": ims_std.to_dict(),
            "n_scalar_features": int(scalar_features.n_scalar_features),
        },
    }

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, scalar_features, metadata


class ProbDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        event_site_combs: Dict[str, np.ndarray],
        db: DB,
        run_config: RunParamsConfig,
        scalar_features: ml_data.ScalarFeatures,
        ims_mean: np.ndarray,
        ims_std: np.ndarray,
        hp_config: HyperParamsConfig,
        sim_corr_dir: Path = None,
    ):
        self.db = db
        self.event_sites = event_sites
        self.event_site_combs = event_site_combs

        self.events = np.asarray(list(event_sites.keys()))

        self.ims = np.asarray(run_config.ims)
        self.im_weights = run_config.im_weights

        self.n_rels = run_config.n_rels

        self.ims_mean = ims_mean
        self.ims_std = ims_std

        self.scalar_features = scalar_features

        self.n_events = len(self.event_sites)

        # Create the feature tensor
        self.scalar_features_values = ml_data.create_scalar_feature_tensor(
            self.events, self.event_sites, self.scalar_features, self.event_site_combs
        )

        # Event details
        self.event_rels = {}
        self.n_samples_event = []
        self.n_sites_event = []
        # Create the (normalised) IM inputs
        self.norm_obs_ims, self.norm_sim_ims = [], []
        self.obs_ims, self.sim_ims = [], []
        self.misfit_score, self.corr = [], []
        self.residual = []

        event_loop = tqdm(event_sites.items(), desc="Processing events")
        for cur_event, cur_sites in event_loop:
            # for cur_event, cur_sites in event_sites.items():
            # Observed
            cur_obs_data = db.get_obs_data(cur_event, cur_sites)
            cur_obs_data = np.log(cur_obs_data.loc[cur_sites, self.ims]).values
            self.obs_ims.append(cur_obs_data)

            # Get the simulation data
            cur_sim_df = db.get_sim_data(cur_event, cur_sites)

            # Number of sites
            cur_n_sites = cur_sites.size
            self.n_sites_event.append(cur_n_sites)

            # Select realisations
            cur_avail_rels = np.unique(cur_sim_df.rel_id.values.astype(str))
            assert self.n_rels <= cur_avail_rels.size
            cur_rels = self.event_rels[cur_event] = np.sort(
                np.random.choice(
                    cur_avail_rels,
                    self.n_rels,
                    replace=False,
                )
            )

            # Number of samples
            self.n_samples_event.append(self.event_site_combs[cur_event].shape[0])

            # Sort, sanity check and convert IMs to log-space
            cur_sim_df = cur_sim_df.loc[
                mlt.array_utils.pandas_isin(cur_sim_df.rel_id, cur_rels)
            ].sort_values(["rel_id", "site_id"])
            assert np.all(cur_sim_df.iloc[:cur_n_sites]["site_id"].values == cur_sites)
            cur_sim_df.loc[:, self.ims] = np.log(cur_sim_df.loc[:, self.ims])

            # Rearrange to (n_records, n_rels, n_ims)
            cur_sim_data = einops.rearrange(
                cur_sim_df.loc[:, self.ims].values,
                "(rel rec) im -> rec rel im",
                rel=run_config.n_rels,
            )
            self.sim_ims.append(cur_sim_data)

            # Residual
            cur_residual = cur_obs_data[:, None, :] - cur_sim_data
            self.residual.append(cur_residual)

            # Compute misfit score
            if hp_config.misfit_fn == "mse":
                cur_misfit_score = np.sum(self.im_weights * cur_residual ** 2, axis=2)
            elif hp_config.misfit_fn == "mae":
                cur_misfit_score = np.sum(
                    self.im_weights * np.abs(cur_residual), axis=2
                )
            else:
                raise ValueError(f"Unknown misfit function: {hp_config.misfit_fn}")
            self.misfit_score.append(cur_misfit_score)

            # # Normalise & Append
            cur_obs_data = (
                cur_obs_data - self.ims_mean[self.ims].values
            ) / self.ims_std[self.ims].values
            self.norm_obs_ims.append(cur_obs_data)
            cur_sim_data = (
                cur_sim_data - self.ims_mean[self.ims].values
            ) / self.ims_std[self.ims].values
            self.norm_sim_ims.append(cur_sim_data)

            # Get the (absolute) spatial correlations
            # Stored per site-combination
            cur_site_combs = self.event_site_combs[cur_event]
            cur_corrs_values = np.ones((cur_site_combs.shape[0], len(self.ims)))
            if sim_corr_dir is not None:
                cur_corrs = pd.read_pickle(sim_corr_dir / f"{cur_event}.pickle")

                for im_ix, cur_im in enumerate(self.ims):
                    cur_corr_df = cur_corrs.get_im_corrs(cur_im).loc[
                        cur_sites, cur_sites
                    ]
                    cur_corrs_values[:, im_ix] = np.abs(
                        cur_corr_df.values[cur_site_combs[:, 0], cur_site_combs[:, 1]]
                    )

            self.corr.append(cur_corrs_values)

        self.n_sites_event = np.asarray(self.n_sites_event)
        self.cum_n_samples_event = np.cumsum(self.n_samples_event)
        self.cum_n_sites_event = np.cumsum(self.n_sites_event)

        self.norm_sim_ims = np.concatenate(self.norm_sim_ims, axis=0)
        self.norm_obs_ims = np.concatenate(self.norm_obs_ims, axis=0)
        self.sim_ims = np.concatenate(self.sim_ims, axis=0)
        self.obs_ims = np.concatenate(self.obs_ims, axis=0)
        self.residual = np.concatenate(self.residual, axis=0)
        self.misfit_score = np.concatenate(self.misfit_score, axis=0)
        self.corr = np.concatenate(self.corr, axis=0)
        self.corr_weights = einops.einsum(
            self.im_weights[None, :], self.corr, "r w, r w -> r"
        )

        # Convert all site-combination to a dataframe for fast indexing
        site_combs_df, sim_ims_df = [], []
        site_combs_ix, sim_ims_ix = 0, 0
        for cur_event in self.events:
            # Site Combinations
            cur_site_comb_df = pd.DataFrame(
                self.event_site_combs[cur_event], columns=["site_int", "site_obs"]
            )
            cur_site_comb_df["event"] = cur_event
            cur_site_comb_df["ix"] = np.arange(
                site_combs_ix, site_combs_ix + cur_site_comb_df.shape[0]
            )
            site_combs_ix += cur_site_comb_df.shape[0]
            site_combs_df.append(cur_site_comb_df)

        self.site_combs_df = pd.concat(site_combs_df)
        self.site_combs_df = self.site_combs_df.set_index("ix")
        self.site_combs = self.site_combs_df[["site_int", "site_obs"]].values

        self.sites = np.concatenate(
            [self.event_sites[cur_event] for cur_event in self.events]
        )
        self.rels = np.stack(
            [self.event_rels[cur_event] for cur_event in self.events], axis=0
        )

    def __len__(self):
        return int(np.sum(self.n_samples_event))

    def get_metadata(self, batch_ind: Sequence[int]):
        """Get the metadata"""
        (
            event_ind,
            site_int_ind,
            site_obs_ind,
            record_site_int_ind,
            record_site_obs_ind,
        ) = self.get_indices(batch_ind)

        return (
            self.events[event_ind],
            self.sites[record_site_int_ind],
            self.sites[record_site_obs_ind],
            self.rels[event_ind],
            self.residual[record_site_int_ind],
        )

    def get_indices(self, batch_ind: np.ndarray):
        # Have to it this way, as some events may not have samples
        event_ind = np.argmax(batch_ind - self.cum_n_samples_event[:, None] < 0, axis=0)

        site_int_ind = self.site_combs[batch_ind, 0]
        site_obs_ind = self.site_combs[batch_ind, 1]

        record_site_int_ind = np.where(
            event_ind > 0,
            self.cum_n_sites_event[event_ind - 1] + site_int_ind,
            site_int_ind,
        )
        record_site_obs_ind = np.where(
            event_ind > 0,
            self.cum_n_sites_event[event_ind - 1] + site_obs_ind,
            site_obs_ind,
        )

        return (
            event_ind,
            site_int_ind,
            site_obs_ind,
            record_site_int_ind,
            record_site_obs_ind,
        )

    def get_batch(self, batch_ind: np.ndarray):
        (
            event_ind,
            site_int_ind,
            site_obs_ind,
            record_site_int_ind,
            record_site_obs_ind,
        ) = self.get_indices(batch_ind)

        site_int_norm_sim_ims = self.norm_sim_ims[record_site_int_ind, :, :]
        site_obs_norm_sim_ims = self.norm_sim_ims[record_site_obs_ind, :, :]
        site_obs_norm_obs_ims = self.norm_obs_ims[record_site_obs_ind, :]

        site_int_sim_ims = self.sim_ims[record_site_int_ind, :, :]
        site_obs_sim_ims = self.sim_ims[record_site_obs_ind, :, :]
        site_obs_obs_ims = self.obs_ims[record_site_obs_ind, :]

        scalar_features = self.scalar_features_values[batch_ind]
        misfit_score = self.misfit_score[record_site_int_ind]

        site_corr_weights = self.corr_weights[batch_ind]

        return (
            batch_ind,
            site_int_norm_sim_ims,
            site_obs_norm_sim_ims,
            site_obs_norm_obs_ims,
            site_int_sim_ims,
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            misfit_score,
            site_corr_weights,
        )


class CustomTabularDataLoader:
    """
    Loosely based on
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, dataset: ProbDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        self.n_samples = len(self.dataset)
        self.n_batches = int(np.ceil(self.n_samples // self.batch_size))

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self.i = 0
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration

        batch_ind = self.indices[self.i : min(self.i + self.batch_size, self.n_samples)]
        self.i += self.batch_size

        # Convert to torch tensors
        return [
            torch.from_numpy(cur_array)
            for cur_array in self.dataset.get_batch(batch_ind)
        ]


def create_model(
    hp_config: HyperParamsConfig,
    scalar_features: ml_data.ScalarFeatures,
    run_config: RunParamsConfig,
):
    if hp_config.combined_model:
        prob_model = models.ProbCombModel(
            hp_config.ind_fc_units,
            hp_config.comb_fc_units,
            scalar_features.n_scalar_features,
            len(run_config.ims),
            hp_config.n_im_features,
        )

        print(f"Model summary")
        summary(
            prob_model,
            input_size=[
                (
                    hp_config.batch_size,
                    hp_config.n_im_features,
                    run_config.n_rels,
                    len(run_config.ims),
                ),
                (hp_config.batch_size, scalar_features.n_scalar_features),
            ],
        )

    else:
        prob_model = models.ProbIndModel(
            hp_config.ind_fc_units,
            scalar_features.n_scalar_features,
            len(run_config.ims),
            hp_config.n_im_features,
            is_sub_model=False,
        )

        print(f"Model summary")
        summary(
            prob_model,
            input_size=[
                (
                    hp_config.batch_size,
                    hp_config.n_im_features,
                    run_config.n_rels,
                    len(run_config.ims),
                ),
                (hp_config.batch_size, scalar_features.n_scalar_features),
            ],
        )

    return prob_model


def train(
    prob_model: models.ProbIndModel,
    train_dataset: ProbDataset,
    val_dataset: ProbDataset,
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
    quiet: bool = False,
):
    metrics = {
        "loss_hist_train": torch.zeros(hp_config.n_epochs),
        "loss_hist_val": torch.zeros(hp_config.n_epochs),
    }

    best_epoch_key = "loss_hist_val"
    best_val_loss = np.inf
    best_model_state, best_model_epoch = None, None

    optimizer = torch.optim.Adam(
        prob_model.parameters(), lr=hp_config.lr[0], weight_decay=hp_config.l2_reg
    )
    lr_ix = 0

    train_dataloader = CustomTabularDataLoader(
        train_dataset, hp_config.batch_size, True
    )
    val_dataloader = CustomTabularDataLoader(val_dataset, hp_config.batch_size, True)

    for epoch_ix in range(hp_config.n_epochs):
        if lr_ix < len(hp_config.lr_epochs) and epoch_ix == hp_config.lr_epochs[lr_ix]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = hp_config.lr[lr_ix + 1]

            print("----------------------------------------------------")
            print(f"Reduced learning rate to {hp_config.lr[lr_ix + 1]}")
            print("----------------------------------------------------")

            lr_ix += 1

        prob_model.train()
        iter_loop = tqdm(train_dataloader, disable=quiet)
        iter_loop.set_description(f"Epoch {epoch_ix}/{hp_config.n_epochs}")
        for i, (
            _,
            site_int_norm_sim_ims,
            site_obs_norm_sim_ims,
            site_obs_norm_obs_ims,
            site_int_sim_ims,
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            misfit_score,
            site_corr_weights,
        ) in enumerate(iter_loop):
            pred = get_prediction(
                prob_model,
                site_obs_norm_obs_ims,
                site_int_norm_sim_ims,
                site_obs_norm_sim_ims,
                site_obs_obs_ims,
                site_int_sim_ims,
                site_obs_sim_ims,
                scalar_features,
                run_config,
                hp_config,
            )

            if pred.isnan().any():
                print("NaNs in model predictions, Quitting!")
                exit()

            cur_loss = compute_loss(misfit_score, pred, site_corr_weights, run_config)

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            metrics["loss_hist_train"][epoch_ix] += cur_loss.item()

            iter_loop.set_postfix({"loss": cur_loss.item()})

        metrics["loss_hist_train"][epoch_ix] /= len(train_dataloader)

        with torch.no_grad():
            prob_model.eval()
            for i, (
                _,
                site_int_norm_sim_ims,
                site_obs_norm_sim_ims,
                site_obs_norm_obs_ims,
                site_int_sim_ims,
                site_obs_sim_ims,
                site_obs_obs_ims,
                scalar_features,
                misfit_score,
                site_corr_weights,
            ) in enumerate(val_dataloader):
                pred = get_prediction(
                    prob_model,
                    site_obs_norm_obs_ims,
                    site_int_norm_sim_ims,
                    site_obs_norm_sim_ims,
                    site_obs_obs_ims,
                    site_int_sim_ims,
                    site_obs_sim_ims,
                    scalar_features,
                    run_config,
                    hp_config,
                )

                cur_loss = compute_loss(
                    misfit_score, pred, site_corr_weights, run_config
                )

                metrics["loss_hist_val"][epoch_ix] += cur_loss.item()

        metrics["loss_hist_val"][epoch_ix] /= len(val_dataloader)

        # Keep track of the best model
        if metrics[best_epoch_key][epoch_ix] < best_val_loss:
            best_model_state = prob_model.state_dict()
            best_val_loss = metrics[best_epoch_key][epoch_ix]
            best_model_epoch = epoch_ix

        print(f"Epoch {epoch_ix + 1}/{hp_config.n_epochs}")
        print(f"\tTraining" f"\t\tLoss: {metrics['loss_hist_train'][epoch_ix]:.4f}")
        print(f"\tValidation" f"\t\tLoss: {metrics['loss_hist_val'][epoch_ix]:.4f}")

    return metrics, best_model_state, best_model_epoch


def compute_loss(
    misfit_score: torch.Tensor,
    pred: torch.Tensor,
    site_corr_weights: torch.Tensor,
    run_config: RunParamsConfig,
    reduce: bool = True,
):
    misfit_score = misfit_score.to(run_config.device, dtype=torch.float32)
    site_corr_weights = site_corr_weights.to(run_config.device, dtype=torch.float32)
    loss = einops.einsum(site_corr_weights, pred, misfit_score, "b, b r, b r -> b")
    if reduce:
        return loss.mean()
    return loss


def get_prediction(
    prob_model: models.ProbIndModel,
    site_obs_norm_obs_ims: torch.Tensor,
    site_int_norm_sim_ims: torch.Tensor,
    site_obs_norm_sim_ims: torch.Tensor,
    site_obs_obs_ims: torch.Tensor,
    site_int_sim_ims: torch.Tensor,
    site_obs_sim_ims: torch.Tensor,
    scalar_features: torch.Tensor,
    run_config: RunParamsConfig,
    hp_config: HyperParamsConfig,
):
    # Pre-allocate IM tensor on device
    im_tensor = torch.full(
        (
            site_obs_norm_obs_ims.shape[0],
            hp_config.n_im_features,
            run_config.n_rels,
            len(run_config.ims),
        ),
        torch.nan,
        dtype=torch.float32,
        device=run_config.device,
        requires_grad=False,
    )
    ix = 0

    if hp_config.use_im_obs_site_obs:
        # Need observed value (at observation site) per realisation
        im_tensor[:, ix, :, :] = site_obs_norm_obs_ims = einops.repeat(
            site_obs_norm_obs_ims[:, None, :],
            "batch rel im -> batch (n_rels rel) im",
            n_rels=run_config.n_rels,
        )
        ix += 1

    if hp_config.use_im_sim_site_obs:
        im_tensor[:, ix, :, :] = site_obs_norm_sim_ims
        ix += 1

    if hp_config.use_im_sim_site_int:
        im_tensor[:, ix, :, :] = site_int_norm_sim_ims
        ix += 1

    if hp_config.use_res_site_obs:
        im_tensor[:, ix, :, :] = site_obs_obs_ims[:, None, :] - site_obs_sim_ims
        ix += 1

    if hp_config.use_res_sim_site_obs_sim_site_int:
        im_tensor[:, ix, :, :] = site_obs_sim_ims - site_int_sim_ims
        ix += 1

    if hp_config.use_res_obs_site_obs_sim_site_int:
        im_tensor[:, ix, :, :] = site_obs_obs_ims[:, None, :] - site_int_sim_ims
        ix += 1

    scalar_features = scalar_features.to(
        run_config.device, dtype=torch.float32, non_blocking=True
    )

    return prob_model(im_tensor, scalar_features)


def get_dataset_predictions(
    dataset: ProbDataset,
    prob_model: models.ProbIndModel,
    run_config: RunParamsConfig,
    dist_matrix: pd.DataFrame,
    hp_config: HyperParamsConfig,
):
    pred_dataloader = CustomTabularDataLoader(
        dataset, hp_config.batch_size, shuffle=False
    )

    columns = [
        "event_id",
        "site_int",
        "site_obs",
        "rel_id",
        "prob",
        "misfit_score",
        "site_corr_weights",
        "s2s_distance",
    ] + run_config.ims
    results_df = pd.DataFrame(
        index=np.arange(len(dataset) * run_config.n_rels), columns=columns
    )

    num_cols = run_config.ims + [
        "prob",
        "misfit_score",
        "site_corr_weights",
        "s2s_distance",
    ]
    results_df = results_df.astype(dict(zip(num_cols, len(num_cols) * [np.float32])))

    results_df["event_id"] = pd.Categorical(
        categories=dataset.events, values=results_df.shape[0] * [pd.NA]
    )
    results_df["site_int"] = pd.Categorical(
        categories=np.unique(dataset.sites), values=results_df.shape[0] * [pd.NA]
    )
    results_df["site_obs"] = pd.Categorical(
        categories=np.unique(dataset.sites), values=results_df.shape[0] * [pd.NA]
    )
    results_df["rel_id"] = pd.Categorical(
        categories=np.unique(dataset.rels), values=results_df.shape[0] * [pd.NA]
    )

    t = []
    with torch.no_grad():
        prob_model.eval()
        for i, (
            batch_ind,
            site_int_norm_sim_ims,
            site_obs_norm_sim_ims,
            site_obs_norm_obs_ims,
            site_int_sim_ims,
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            misfit_score,
            site_corr_weights,
        ) in enumerate(pred_dataloader):
            pred = get_prediction(
                prob_model,
                site_obs_norm_obs_ims,
                site_int_norm_sim_ims,
                site_obs_norm_sim_ims,
                site_obs_obs_ims,
                site_int_sim_ims,
                site_obs_sim_ims,
                scalar_features,
                run_config,
                hp_config,
            )

            events, site_int, site_obs, rels, misfit = dataset.get_metadata(batch_ind)

            cur_start_ix = i * hp_config.batch_size * run_config.n_rels
            cur_end_ix = cur_start_ix + pred.shape[0] * run_config.n_rels - 1
            results_df.loc[cur_start_ix:cur_end_ix, "event_id"] = einops.repeat(
                events, "batch -> (batch rel)", rel=pred.shape[1]
            )
            df_site_int = einops.repeat(
                site_int, "batch -> (batch rel)", rel=pred.shape[1]
            )
            results_df.loc[cur_start_ix:cur_end_ix, "site_int"] = df_site_int
            df_site_obs = einops.repeat(
                site_obs, "batch -> (batch rel)", rel=pred.shape[1]
            )
            results_df.loc[cur_start_ix:cur_end_ix, "site_obs"] = df_site_obs
            results_df.loc[cur_start_ix:cur_end_ix, "rel_id"] = einops.rearrange(
                rels, "batch rel -> (batch rel)"
            )
            results_df.loc[cur_start_ix:cur_end_ix, "prob"] = (
                einops.rearrange(pred, "batch rel -> (batch rel)")
                .numpy(force=True)
                .astype(np.float32)
            )
            results_df.loc[cur_start_ix:cur_end_ix, "misfit_score"] = (
                einops.rearrange(misfit_score, "batch rel -> (batch rel)")
                .numpy(force=True)
                .astype(np.float32)
            )
            results_df.loc[cur_start_ix:cur_end_ix, "site_corr_weights"] = (
                einops.repeat(
                    site_corr_weights, "batch -> (batch rel)", rel=pred.shape[1]
                )
                .numpy(force=True)
                .astype(np.float32)
            )
            results_df.loc[cur_start_ix:cur_end_ix, run_config.ims] = einops.rearrange(
                misfit, "batch rel im -> (batch rel) im"
            ).astype(np.float32)
            results_df.loc[
                cur_start_ix:cur_end_ix, "s2s_distance"
            ] = dist_matrix.values[
                dist_matrix.index.get_indexer_for(df_site_int),
                dist_matrix.columns.get_indexer_for(df_site_obs),
            ].astype(
                np.float32
            )

    return results_df


def post_processing(
    prob_model: models.ProbIndModel,
    train_dataset: ProbDataset,
    val_dataset: ProbDataset,
    hp_config: HyperParamsConfig,
    run_config: RunParamsConfig,
    metrics: Dict,
    best_epoch: int,
    scalar_features: ml_data.ScalarFeatures,
    data_metadata: Dict,
    val_int_sites: np.ndarray,
    train_sites: np.ndarray,
    id_suffix: str = "",
):
    (
        cur_out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir()

    # Save the training sites and validation sites
    np.save(cur_out_dir / "val_int_sites.npy", val_int_sites)
    np.save(cur_out_dir / "train_sites.npy", train_sites)

    # Compute the distance matrix
    print(f"Computing distance matrix")
    station_df = DB(
        Path(os.path.expandvars("$wdata")) / data_metadata["db"]
    ).get_site_df()
    all_sites = np.unique(np.concatenate((train_dataset.sites, val_dataset.sites)))
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    ### Sample
    print(f"Computing sample distributions")
    train_sample_results = get_dataset_predictions(
        train_dataset, prob_model, run_config, dist_matrix, hp_config
    )
    print(
        f"\tTrain sample results - Memory usage: {train_sample_results.memory_usage(deep=True).sum() / 1e6} MB"
    )
    val_sample_results = get_dataset_predictions(
        val_dataset, prob_model, run_config, dist_matrix, hp_config
    )
    print(
        f"\tVal sample results - Memory usage: {val_sample_results.memory_usage(deep=True).sum() / 1e6}"
    )

    train_sample_results.to_parquet(cur_out_dir / "train_sample_results.parquet")
    val_sample_results.to_parquet(cur_out_dir / "val_sample_results.parquet")

    ### Scenario
    print(f"Computing scenario distributions")
    print("Training dataset")
    train_scenario_results = compute_scenario_distribution(
        train_sample_results, run_config
    )
    print("Validation dataset")
    val_scenario_results = compute_scenario_distribution(val_sample_results, run_config)

    train_scenario_results.to_parquet(cur_out_dir / "train_scenario_results.parquet")
    val_scenario_results.to_parquet(cur_out_dir / "val_scenario_results.parquet")

    # Save loss history
    pd.to_pickle(metrics, cur_out_dir / "metrics.pickle")

    # Save the model
    torch.save(prob_model, cur_out_dir / "model.pt")

    # Metadata
    metadata = {
        "method_type": constants.RankingMethod.ml_prob.value,
        "hp_config": hp_config.to_dict(),
        "best_epoch": best_epoch,
        "data": data_metadata,
        "run_config": run_config.to_dict(),
    }
    mlt.utils.write_to_yaml(metadata, cur_out_dir / "meta.yaml")

    # Create model visualisation
    draw_graph(
        prob_model,
        input_size=[
            (
                hp_config.batch_size,
                hp_config.n_im_features,
                run_config.n_rels,
                len(run_config.ims),
            ),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="prob_model_vis",
        save_graph=True,
        directory=str(cur_out_dir),
    )


def compute_scenario_distribution(
    sample_results: pd.DataFrame, run_config: RunParamsConfig
):
    """
    Computes the realisation distribution for each scenario
    """
    scenario_results = []
    groups = sample_results.groupby(["event_id", "site_int"], observed=True)
    iter_loop = tqdm(groups, desc="Processing scenarios")
    for (cur_event, cur_site_int), cur_group in iter_loop:
        cur_rel_group = cur_group.groupby("rel_id", observed=True)

        wm = lambda x: np.sum(
            x.prob.values
            * (x.site_corr_weights.values / x.site_corr_weights.values.sum())
        )
        cur_result = cur_rel_group.apply(wm).to_frame("prob")
        cur_result["event_id"] = cur_event
        cur_result["site_int"] = cur_site_int
        cur_result["rel_id"] = cur_result.index

        cur_result["n_obs_sites"] = cur_group.site_obs.nunique()
        cur_result["min_distance"] = cur_group.s2s_distance.min()
        cur_result["max_distance"] = cur_group.s2s_distance.max()

        cur_site_obs_first = cur_group.groupby("site_obs", observed=True).first()
        cur_result["weighted_mean_distance"] = (
            np.sum(
                cur_site_obs_first.s2s_distance.values
                * cur_site_obs_first.site_corr_weights.values
            )
            / cur_site_obs_first.site_corr_weights.values.sum()
        )

        cur_result["misfit_score"] = cur_rel_group.misfit_score.first().values

        cur_residuals = cur_rel_group.first().loc[:, run_config.ims]
        assert np.all(cur_residuals.index == cur_result.rel_id)
        cur_result.loc[:, run_config.ims] = cur_residuals.values

        cur_result.index = mlt.array_utils.numpy_str_join(
            "_", cur_event, cur_site_int, cur_result.index.values.astype(str)
        )

        scenario_results.append(cur_result)

    scenario_df = pd.concat(scenario_results, axis=0)
    return scenario_df


def compute_scenario_loss(scenario_rel_results: pd.DataFrame):
    """
    Computes the scenario loss for each scenario
    """
    scenario_loss = (
        scenario_rel_results.groupby(["event_id", "site_int"])
        .apply(lambda g: np.sum(g.prob * g.misfit_score))
        .to_frame("scenario_loss")
    )
    scenario_loss["event_id"] = scenario_loss.index.get_level_values(0).values.astype(
        str
    )
    scenario_loss["site_int"] = scenario_loss.index.get_level_values(1).values.astype(
        str
    )
    scenario_loss.index = mlt.array_utils.numpy_str_join(
        "_",
        scenario_loss.event_id.values.astype(str),
        scenario_loss.site_int.values.astype(str),
    )

    return scenario_loss
