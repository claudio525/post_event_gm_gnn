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


@dataclass
class RunParamsConfig:
    max_dist: float
    n_rels: int
    ims: Sequence[str]
    im_weights: np.ndarray

    debug: bool
    device: str

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))

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
    lr: float

    fc_units: Sequence[int]

    fc_im_units: Sequence[int]
    fc_ss_units: Sequence[int]
    fc_comb_units: Sequence[int]

    @classmethod
    def from_yaml(cls, ffp: Path, n_epochs: int):
        params = mlt.utils.load_yaml(ffp)

        return cls(
            n_epochs,
            params["batch_size"],
            params["l2_reg"],
            params["lr"],
            params["fc_units"],
            params["fc_im_units"],
            params["fc_ss_units"],
            params["fc_comb_units"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "fc_units": self.fc_units,
            "fc_im_units": self.fc_im_units,
            "fc_ss_units": self.fc_ss_units,
            "fc_comb_units": self.fc_comb_units,
        }


def data_preb(
    event_sites: Dict[str, np.ndarray],
    train_events: np.ndarray,
    val_events: np.ndarray,
    train_sites: np.ndarray,
    val_int_sites: np.ndarray,
    events: np.ndarray,
    run_config: RunParamsConfig,
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

        # Compute the number of samples per event
        self.event_rels = {}
        self.n_samples_event = []
        self.n_sites_event = []
        for cur_event, cur_sites in event_sites.items():
            cur_sim_data = db.get_sim_data(cur_event, cur_sites)
            self.n_sites_event.append(cur_sites.size)

            cur_avail_rels = np.unique(cur_sim_data.rel_id.values.astype(str))
            assert self.n_rels <= cur_avail_rels.size
            self.event_rels[cur_event] = np.sort(
                np.random.choice(
                    cur_avail_rels,
                    self.n_rels,
                    replace=False,
                )
            )

            self.n_samples_event.append(self.event_site_combs[cur_event].shape[0])

        self.n_sites_event = np.asarray(self.n_sites_event)

        self.cum_n_samples_event = np.cumsum(self.n_samples_event)
        self.cum_n_sites_event = np.cumsum(self.n_sites_event)

        # Create the feature tensor
        self.scalar_features_values = ml_data.create_scalar_feature_tensor(
            self.events, self.event_sites, self.scalar_features, self.event_site_combs
        )

        # Create the (normalised) IM inputs
        self.obs_ims, self.sim_ims = [], []
        self.misfit_score, self.corr = [], []
        self.misfit = []
        # self.sim_n_records_event = []
        for cur_event, cur_sites in event_sites.items():
            cur_n_sites = self.event_sites[cur_event].size

            # Observed
            cur_obs_data = db.get_obs_data(cur_event, cur_sites)
            cur_obs_data = np.log(cur_obs_data.loc[cur_sites, self.ims]).values

            # Get the simulation data
            cur_rels = self.event_rels[cur_event]
            cur_sim_df = db.get_sim_data(cur_event, cur_sites)
            cur_sim_df = cur_sim_df.loc[
                np.isin(cur_sim_df.rel_id, cur_rels)
            ].sort_values(["rel_id", "site_id"])
            assert np.all(cur_sim_df.iloc[:cur_n_sites]["site_id"].values == cur_sites)
            cur_sim_df.loc[:, self.ims] = np.log(cur_sim_df.loc[:, self.ims])

            cur_sim_data = einops.rearrange(
                cur_sim_df.loc[:, self.ims].values, "(rel rec) im -> rec rel im", rel=run_config.n_rels
            )
            cur_misfit = cur_obs_data[:, None, :] - cur_sim_data
            self.misfit.append(cur_misfit)

            # Compute misfit score
            cur_misfit_score = np.sum(self.im_weights * cur_misfit ** 2, axis=2)
            self.misfit_score.append(cur_misfit_score)

            # # Normalise & Append
            cur_obs_data = (
                cur_obs_data - self.ims_mean[self.ims].values
            ) / self.ims_std[self.ims].values
            self.obs_ims.append(cur_obs_data)
            #
            cur_sim_data = (
                cur_sim_data - self.ims_mean[self.ims].values
            ) / self.ims_std[self.ims].values
            self.sim_ims.append(cur_sim_data)

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

        self.sim_ims = np.concatenate(self.sim_ims, axis=0)
        self.obs_ims = np.concatenate(self.obs_ims, axis=0)
        self.misfit = np.concatenate(self.misfit, axis=0)
        self.misfit_score = np.concatenate(self.misfit_score, axis=0)
        self.corr = np.concatenate(self.corr, axis=0)
        self.corr_weights = einops.einsum(
            self.im_weights[None, :], self.corr, "r w, r w -> r"
        )

        # self.cum_sim_n_records_event = np.cumsum(self.sim_n_records_event)

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
        self.rels = np.stack([self.event_rels[cur_event] for cur_event in self.events], axis=0)

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
            self.misfit[record_site_int_ind],
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

        site_int_sims = self.sim_ims[record_site_int_ind, :, :]
        site_obs_sims = self.sim_ims[record_site_obs_ind, :, :]
        site_obs_obs = self.obs_ims[record_site_obs_ind, :]

        scalar_features = self.scalar_features_values[batch_ind]
        misfit_score = self.misfit_score[record_site_int_ind]

        site_corr_weights = self.corr_weights[batch_ind]

        return (
            batch_ind,
            site_int_sims,
            site_obs_sims,
            site_obs_obs,
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
    # prob_model = models.ProbCombModel(
    #     hp_config.fc_im_units,
    #     len(run_config.ims),
    #     hp_config.fc_ss_units,
    #     hp_config.fc_comb_units,
    #     scalar_features.n_scalar_features,
    #     run_config.n_rels,
    # )

    prob_model = models.ProbIndModel(
        hp_config.fc_units,
        scalar_features.n_scalar_features,
        len(run_config.ims),
    )

    print(f"Model summary")
    summary(
        prob_model,
        input_size=[
            (hp_config.batch_size, 3, run_config.n_rels, len(run_config.ims)),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
    )

    # summary(
    #     prob_model,
    #     input_size=[
    #         (hp_config.batch_size, 3, run_config.n_rels, len(run_config.ims)),
    #         (hp_config.batch_size, scalar_features.n_scalar_features),
    #     ],
    # )

    return prob_model


def train(
    prob_model: models.ProbCombModel,
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
        prob_model.parameters(), lr=hp_config.lr, weight_decay=hp_config.l2_reg
    )

    train_dataloader = CustomTabularDataLoader(
        train_dataset, hp_config.batch_size, True
    )
    val_dataloader = CustomTabularDataLoader(val_dataset, hp_config.batch_size, True)

    for epoch_ix in range(hp_config.n_epochs):
        prob_model.train()
        iter_loop = tqdm(train_dataloader, disable=quiet)
        iter_loop.set_description(f"Epoch {epoch_ix}/{hp_config.n_epochs}")
        for i, (
            _,
            site_int_sims,
            site_obs_sims,
            site_obs_obs,
            scalar_features,
            misfit_score,
            site_corr_weights,
        ) in enumerate(iter_loop):
            pred = get_prediction(
                prob_model,
                site_obs_obs,
                site_int_sims,
                site_obs_sims,
                scalar_features,
                run_config,
            )
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
                site_int_sims,
                site_obs_sims,
                site_obs_obs,
                scalar_features,
                misfit_score,
                site_corr_weights,
            ) in enumerate(val_dataloader):
                pred = get_prediction(
                    prob_model,
                    site_obs_obs,
                    site_int_sims,
                    site_obs_sims,
                    scalar_features,
                    run_config,
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
    site_obs_obs: torch.Tensor,
    site_int_sims: torch.Tensor,
    site_obs_sims: torch.Tensor,
    scalar_features: torch.Tensor,
    run_config: RunParamsConfig,
):
    # Need observed value (at observation site) per realisation
    site_obs_obs = einops.repeat(
        site_obs_obs[:, None, :],
        "batch rel im -> batch (n_rels rel) im",
        n_rels=run_config.n_rels,
    )

    im_tensor = einops.rearrange(
        [site_int_sims, site_obs_sims, site_obs_obs],
        "type batch rel im -> batch type rel im",
    ).to(run_config.device, dtype=torch.float32)

    scalar_features = scalar_features.to(run_config.device, dtype=torch.float32)

    return prob_model(im_tensor, scalar_features)

def get_dataset_predictions(dataset: ProbDataset, prob_model: models.ProbIndModel, run_config: RunParamsConfig):
    pred_dataloader = CustomTabularDataLoader(dataset, int(1e5), shuffle=False)

    results = []
    with torch.no_grad():
        prob_model.eval()
        for i, (
            batch_ind,
            site_int_sims,
            site_obs_sims,
            site_obs_obs,
            scalar_features,
            misfit_score,
            site_corr_weights,
        ) in enumerate(pred_dataloader):
            pred = get_prediction(
                prob_model,
                site_obs_obs,
                site_int_sims,
                site_obs_sims,
                scalar_features,
                run_config,
            )

            events, site_int, site_obs, rels, misfit = dataset.get_metadata(batch_ind)

            cur_df = pd.DataFrame(
                {
                    "event_id": pd.Categorical(einops.repeat(events, "batch -> (batch rel)", rel=pred.shape[1])),
                    "site_int": pd.Categorical(einops.repeat(site_int, "batch -> (batch rel)", rel=pred.shape[1])),
                    "site_obs": pd.Categorical(einops.repeat(site_obs, "batch -> (batch rel)", rel=pred.shape[1])),
                    "rel_id": pd.Categorical(einops.rearrange(rels, "batch rel -> (batch rel)")),
                    "prob": einops.rearrange(pred, "batch rel -> (batch rel)").numpy(force=True),
                    "misfit_score": einops.rearrange(misfit_score, "batch rel -> (batch rel)"),
                    "site_corr_weights": einops.repeat(site_corr_weights, "batch -> (batch rel)", rel=pred.shape[1]),
                }
            )
            cur_df.loc[:, run_config.ims] = einops.rearrange(misfit, "batch rel im -> (batch rel) im")

            results.append(cur_df)

        results_df = pd.concat(results, axis=0)

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
    id_suffix: str = "",
):
    (
        cur_out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir()

    train_sample_results = get_dataset_predictions(train_dataset, prob_model, run_config)
    val_sample_results = get_dataset_predictions(val_dataset, prob_model, run_config)

    train_sample_results.to_parquet(cur_out_dir / "train_sample_results.parquet")
    val_sample_results.to_parquet(cur_out_dir / "val_sample_results.parquet")

    train_scenario_results = compute_scenario_distribution(train_sample_results, run_config)
    val_scenario_results = compute_scenario_distribution(val_sample_results, run_config)

    train_scenario_results.to_parquet(cur_out_dir / "train_scenario_results.parquet")
    val_scenario_results.to_parquet(cur_out_dir / "val_scenario_results.parquet")

    # Save loss history
    pd.to_pickle(metrics, cur_out_dir / "metrics.pickle")

    # Save the model
    torch.save(prob_model, cur_out_dir / "model.pt")

    # Metadata
    metadata = {
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
            (hp_config.batch_size, 3, run_config.n_rels, len(run_config.ims)),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
        expand_nested=True,
        filename="prob_model_vis",
        save_graph=True,
        directory=str(cur_out_dir),
    )

def compute_scenario_distribution(sample_results: pd.DataFrame, run_config: RunParamsConfig):
    """
    Computes the realisation distribution for each scenario
    """
    scenario_results = []
    groups = sample_results.groupby(["event_id", "site_int"], observed=True)
    for (cur_event, cur_site_int), cur_group in groups:
        cur_rel_group = cur_group.groupby("rel_id", observed=True)

        wm = lambda x: np.sum(x.prob.values * (x.site_corr_weights.values / x.site_corr_weights.values.sum()))
        cur_result = cur_rel_group.apply(wm).to_frame("prob")
        cur_result["event_id"] = cur_event
        cur_result["site_int"] = cur_site_int
        cur_result["rel_id"] = cur_result.index

        cur_residuals = cur_rel_group.first().loc[:, run_config.ims]
        assert np.all(cur_residuals.index == cur_result.rel_id)
        cur_result.loc[:, run_config.ims] = cur_residuals.values

        cur_result.index = mlt.array_utils.numpy_str_join(
            "_", cur_event, cur_site_int, cur_result.index.values.astype(str)
        )

        scenario_results.append(cur_result)

    scenario_df = pd.concat(scenario_results, axis=0)
    return scenario_df

