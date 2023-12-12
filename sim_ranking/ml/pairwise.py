import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, List

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
from . import pairwise_pred
from ..db import DB
from .. import constants


def compute_res_score(obs_ims: np.ndarray, sim_ims: np.ndarray, weights: np.ndarray):
    """
    Computes the similarity score for response spectrum

    Parameters
    ----------
    obs_ims: array of floats
        Observed IMs
        Format [n_periods]
    sim_ims: array of floats
        Simulation realisation IMs
        Format [n_realisations, n_periods]
    weights: array of floats
        Weights for each IM

    Returns
    -------
    array of floats
        Similarity score for each realisation
        Format [n_realisations]
    """
    # Compute the residual
    res = obs_ims[..., None] - sim_ims
    # res_score = np.sum(np.abs(res), axis=1)
    # res_score = np.average(res**2, axis=1, weights=weights)
    res_score = np.sum(weights[None, :, None] * res ** 2, axis=1)

    return res_score


# def test(res_df: pd.DataFrame):
#     return np.sum(constants.IM_weights * (res_df.loc[:, constants.IMs].values ** 2), axis=1)


@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    l2_reg: float
    lr: float

    # n_channels: List[int]
    # kernel_sizes: List[int]
    fc_units: List[int]

    @classmethod
    def from_yaml(cls, ffp: Path):
        params = mlt.utils.load_yaml(ffp)

        return cls(
            params["n_epochs"],
            params["batch_size"],
            params["l2_reg"],
            params["lr"],
            # params["n_channels"],
            # params["kernel_sizes"],
            params["fc_units"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            # "n_channels": self.n_channels,
            # "kernel_sizes": self.kernel_sizes,
            "fc_units": self.fc_units,
        }


class PairDataset2(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        ims: Sequence[str],
        im_weights: np.ndarray,
        scalar_features: ml_data.ScalarFeatures,
        ims_mean: np.ndarray,
        ims_std: np.ndarray,
        max_n_rels: int,
        sim_corr_dir: Path = None,
    ):
        self.db = db
        self.event_sites = event_sites
        self.site_combs = site_combs

        self.events = np.asarray(list(event_sites.keys()))

        self.ims = np.asarray(ims)
        self.im_weights = im_weights

        self.ims_mean = ims_mean
        self.ims_std = ims_std

        self.scalar_features = scalar_features

        self.n_events = len(self.event_sites)

        self.n_rels_event = {}
        self.event_rels = {}
        self.sim_df = []
        self.obs_df = []
        for cur_event, cur_sites in event_sites.items():
            # Load IM data
            cur_sim_data = db.get_sim_data(cur_event, cur_sites)
            cur_obs_df = db.get_obs_data(cur_event, cur_sites)
            cur_obs_df.loc[:, self.ims] = np.log(cur_obs_df.loc[:, self.ims])

            # Select a random subset of realisations
            cur_rels = np.random.choice(
                np.unique(cur_sim_data.rel_id.values.astype(str)),
                max_n_rels,
                replace=False,
            )

            # Filter
            cur_sim_data = cur_sim_data.loc[
                np.isin(cur_sim_data.rel_id, cur_rels)
                & np.isin(cur_sim_data.site_id, cur_sites)
            ]
            cur_obs_df = cur_obs_df.loc[np.isin(cur_obs_df.site_id, cur_sites)]

            self.sim_df.append(cur_sim_data)
            self.obs_df.append(cur_obs_df)

        self.sim_df = pd.concat(self.sim_df)
        self.obs_df = pd.concat(self.obs_df)


class PairDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        event_site_combs: Dict[str, np.ndarray],
        db: DB,
        ims: Sequence[str],
        im_weights: np.ndarray,
        scalar_features: ml_data.ScalarFeatures,
        ims_mean: np.ndarray,
        ims_std: np.ndarray,
        max_n_rels: int,
        sim_corr_dir: Path = None,
    ):
        self.db = db
        self.event_sites = event_sites
        self.event_site_combs = event_site_combs

        self.events = np.asarray(list(event_sites.keys()))

        self.ims = np.asarray(ims)
        self.im_weights = im_weights

        self.ims_mean = ims_mean
        self.ims_std = ims_std

        self.scalar_features = scalar_features

        self.n_events = len(self.event_sites)

        # Compute the number of samples per event
        self.event_rels = {}
        self.n_rels_event = {}
        self.n_samples_event = []
        self.n_site_combs_event = []
        self.n_sites_event = []
        for cur_event, cur_sites in event_sites.items():
            cur_sim_data = db.get_sim_data(cur_event, cur_sites)
            self.n_sites_event.append(cur_sites.size)

            # cur_n_rels = np.unique(cur_sim_data.rel_id.values).size
            # self.n_rels_event[cur_event] = cur_n_rels if cur_n_rels < max_n_rels else max_n_rels
            cur_avail_rels = np.unique(cur_sim_data.rel_id.values.astype(str))
            assert max_n_rels <= cur_avail_rels.size
            self.event_rels[cur_event] = np.random.choice(
                cur_avail_rels,
                max_n_rels,
                replace=False,
            )
            cur_n_rels = self.n_rels_event[cur_event] = self.event_rels[cur_event].size

            cur_n_samples = self.event_site_combs[cur_event].shape[0] * int(
                cur_n_rels ** 2 - cur_n_rels
            )
            self.n_samples_event.append(cur_n_samples)
            self.n_site_combs_event.append(self.event_site_combs[cur_event].shape[0])

        self.n_sites_event = np.asarray(self.n_sites_event)

        self.cum_n_samples_event = np.cumsum(self.n_samples_event)
        self.cum_n_site_combs_event = np.cumsum(self.n_site_combs_event)
        self.cum_n_sites_event = np.cumsum(self.n_sites_event)

        # Create the feature tensor
        self.scalar_features_values = ml_data.create_scalar_feature_tensor(
            self.events, self.event_sites, self.scalar_features, self.event_site_combs
        )

        # Create the (normalised) IM inputs
        self.obs_ims, self.sim_ims = [], []
        self.misfit_score, self.corr = [], []
        self.sim_n_records_event = []
        for cur_event, cur_sites in event_sites.items():
            cur_n_sites = self.event_sites[cur_event].size

            # Observed
            cur_obs_data = db.get_obs_data(cur_event, cur_sites)
            cur_obs_data = np.log(cur_obs_data.loc[cur_sites, self.ims]).values

            # Get the simulation data
            cur_sim_df = db.get_sim_data(cur_event, cur_sites)
            cur_sim_df.loc[:, self.ims] = np.log(cur_sim_df.loc[:, self.ims])

            cur_sim_data = np.full(
                (cur_sites.size * self.n_rels_event[cur_event], self.ims.size),
                fill_value=np.nan,
            )
            cur_misfit_score = cur_sim_data.copy()

            for rel_ix, cur_rel in enumerate(self.event_rels[cur_event]):
                cur_rel_data = (
                    cur_sim_df.loc[cur_sim_df.rel_id == cur_rel]
                    .set_index("site_id")
                    .loc[cur_sites, self.ims]
                    .values
                )

                cur_misfit_score[
                    rel_ix * cur_n_sites : (rel_ix + 1) * cur_n_sites, :
                ] = (cur_obs_data - cur_rel_data)
                cur_sim_data[
                    rel_ix * cur_n_sites : (rel_ix + 1) * cur_n_sites, :
                ] = cur_rel_data
            self.sim_n_records_event.append(cur_sim_data.shape[0])

            # Compute misfit score
            cur_misfit_score = np.sum(im_weights * cur_misfit_score ** 2, axis=1)
            self.misfit_score.append(cur_misfit_score)

            # Normalise & Append
            cur_obs_data = (cur_obs_data - self.ims_mean[self.ims].values) / self.ims_std[self.ims].values
            self.obs_ims.append(cur_obs_data)

            cur_sim_data = (cur_sim_data - self.ims_mean[self.ims].values) / self.ims_std[self.ims].values
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
        self.misfit_score = np.concatenate(self.misfit_score, axis=0)
        self.corr = np.concatenate(self.corr, axis=0)

        self.n_rels_event = pd.Series(self.n_rels_event)
        self.cum_sim_n_records_event = np.cumsum(self.sim_n_records_event)

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

    def __len__(self):
        return int(np.sum(self.n_samples_event))

    def get_metadata(self, idx: int):
        """Get the metadata for a specific sample"""
        raise NotImplementedError()
        # event_ix, event, site_comb_ix, rel_1_ix, rel_2_ix = self.get_indices(idx)
        #
        # # Get the site of interest and observation site
        # site_int_ix = self.site_combs[event][site_comb_ix, 0]
        # site_obs_ix = self.site_combs[event][site_comb_ix, 1]
        #
        # site_int = self.event_sites[event][site_int_ix]
        # site_obs = self.event_sites[event][site_obs_ix]
        #
        # rel_1 = self.event_rels[event][rel_1_ix]
        # rel_2 = self.event_rels[event][rel_2_ix]
        #
        # return (event, site_int, site_obs, rel_1, rel_2)

    def get_indices(self, ind: np.ndarray):
        # Have to it this way, as some events may not have samples
        event_ind = np.argmax(ind - self.cum_n_samples_event[:, None] < 0, axis=0)

        within_event_ind = np.where(
            event_ind > 0, ind - self.cum_n_samples_event[event_ind - 1], ind
        )

        events = self.events[event_ind]

        n_rels = self.n_rels_event[events].values

        n_rel_combs = n_rels ** 2 - n_rels

        within_site_ind = within_event_ind % n_rel_combs

        row_length_m = n_rels - 1

        rel_1_ind = within_site_ind // row_length_m

        rel_2_ind = within_site_ind % row_length_m + (
            within_site_ind % row_length_m >= rel_1_ind
        )

        site_comb_ind = within_event_ind // n_rel_combs

        return event_ind, events, site_comb_ind, rel_1_ind, rel_2_ind

    def get_batch(self, batch_ind: np.ndarray):
        event_ind, events, site_comb_ind, rel_1_ind, rel_2_ind = self.get_indices(
            batch_ind
        )

        # Site Combination indices
        event_site_comb_ind = (
            np.where(event_ind > 0, self.cum_n_site_combs_event[event_ind - 1], 0)
            + site_comb_ind
        )
        site_int_ind = self.site_combs_df.loc[event_site_comb_ind, "site_int"].values
        site_obs_ind = self.site_combs_df.loc[event_site_comb_ind, "site_obs"].values

        # Event - Site
        event_site_int_ind = (
            np.where(event_ind > 0, self.cum_n_sites_event[event_ind - 1], 0)
            + site_int_ind
        )

        # Event - Realisation - Site
        sim_record_event_ind = np.where(
            event_ind > 0, self.cum_sim_n_records_event[event_ind - 1], 0
        )
        sim_int_rel_1_ind = (
            sim_record_event_ind
            + rel_1_ind * self.n_sites_event[event_ind]
            + site_int_ind
        )
        sim_int_rel_2_ind = (
            sim_record_event_ind
            + rel_2_ind * self.n_sites_event[event_ind]
            + site_int_ind
        )

        sim_obs_rel_1_ind = (
            sim_record_event_ind
            + rel_1_ind * self.n_sites_event[event_ind]
            + site_obs_ind
        )
        sim_obs_rel_2_ind = (
            sim_record_event_ind
            + rel_2_ind * self.n_sites_event[event_ind]
            + site_obs_ind
        )

        sim_int_rel_1 = self.sim_ims[sim_int_rel_1_ind]
        sim_int_rel_2 = self.sim_ims[sim_int_rel_2_ind]

        sim_obs_rel_1 = self.sim_ims[sim_obs_rel_1_ind]
        sim_obs_rel_2 = self.sim_ims[sim_obs_rel_2_ind]

        obs_int = self.obs_ims[event_site_int_ind]

        misfit_rel_1 = self.misfit_score[sim_int_rel_1_ind]
        misfit_rel_2 = self.misfit_score[sim_int_rel_2_ind]

        corr = self.corr[event_site_comb_ind]

        scalar_features = self.scalar_features_values[event_site_comb_ind]

        return (
            batch_ind,
            scalar_features,
            sim_int_rel_1,
            sim_int_rel_2,
            sim_obs_rel_1,
            sim_obs_rel_2,
            obs_int,
            misfit_rel_1,
            misfit_rel_2,
            corr,
        )

    def __getitem__(self, idx: int):
        raise NotImplementedError()

        # event_ix, event, site_comb_ix, rel_1_ix, rel_2_ix = self.get_single_indices(idx)
        #
        # site_int_ix = self.site_combs[event][site_comb_ix, 0]
        # site_obs_ix = self.site_combs[event][site_comb_ix, 1]
        #
        # return (
        #     idx,
        #     self.scalar_features_tensor[event][site_int_ix, site_obs_ix, :],
        #     self.sim_ims[event][site_int_ix, :, rel_1_ix],
        #     self.sim_ims[event][site_int_ix, :, rel_2_ix],
        #     self.sim_ims[event][site_obs_ix, :, rel_1_ix],
        #     self.sim_ims[event][site_obs_ix, :, rel_2_ix],
        #     self.obs_ims[event][site_obs_ix],
        #     self.res_area[event][site_int_ix, rel_1_ix],
        #     self.res_area[event][site_int_ix, rel_2_ix],
        #     self.corr[event][site_int_ix, site_obs_ix, :],
        # )


@dataclass
class RunParamsConfig:
    max_dist: float
    max_n_rels: int
    ims: Sequence[str]
    im_weights: np.ndarray

    debug: bool
    device: str

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))


def compute_loss(
    loss: nn.Module,
    pred: torch.Tensor,
    res_misfit_1: torch.Tensor,
    res_misfit_2: torch.Tensor,
    site_correlations: torch.Tensor,
    im_weights: torch.Tensor,
    device: str,
    reduce: bool = True,
):
    # Classification, is Rel 1 better than Rel 2?
    true = (res_misfit_1 < res_misfit_2)[:, None].to(device, dtype=torch.float32)
    loss_value = loss(pred, true)

    res_diff = torch.abs(res_misfit_1 - res_misfit_2).to(device, dtype=torch.float32)

    # Compute and normalize the sample weights
    res_weights = 1 - torch.exp(-0.075 * res_diff) + 0.05
    res_weights = res_weights * (res_weights.shape[0] / res_weights.sum())

    # Compute site-correlation weights
    # Normalize IM weights such that they sum to number of IMs
    im_weights = im_weights * len(im_weights)
    # Weighted (based on IM weights) average of the site-correlations
    site_weights = torch.mean(
        im_weights * site_correlations.to(device, dtype=torch.float32), dim=1
    )
    site_weights = site_weights * (site_weights.shape[0] / site_weights.sum())

    sample_weights = site_weights * res_weights
    sample_weights = sample_weights * (sample_weights.shape[0] / sample_weights.sum())

    weighted_loss_value = sample_weights * loss_value.ravel()

    if reduce:
        return loss_value.mean(), weighted_loss_value.mean(), true, sample_weights

    return loss_value, weighted_loss_value, true, sample_weights


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
        "weighted_acc_hist_train": torch.zeros(hp_config.n_epochs),
        "bce_loss_hist_val": torch.zeros(hp_config.n_epochs),
        "loss_hist_val": torch.zeros(hp_config.n_epochs),
        "acc_hist_val": torch.zeros(hp_config.n_epochs),
        "weighted_acc_hist_val": torch.zeros(hp_config.n_epochs),
    }

    best_epoch_key = "loss_hist_val"
    best_val_loss = np.inf
    best_model_state = None

    optimizer = torch.optim.Adam(
        ranking_model.parameters(), lr=hp_config.lr, weight_decay=hp_config.l2_reg
    )

    n_workers = 0 if run_config.debug else 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True if n_workers > 0 else False,
        prefetch_factor=25,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True if n_workers > 0 else False,
        prefetch_factor=25,
    )

    im_weights = torch.from_numpy(train_dataset.im_weights).to(
        device, dtype=torch.float32
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
            int_sim_ims_rel_1,
            int_sim_ims_rel_2,
            obs_sim_ims_rel_1,
            obs_sim_ims_rel_2,
            obs_obs_ims,
            res_area_rel_1,
            res_area_rel_2,
            site_correlations,
        ) in enumerate(iter_loop):

            pred = get_prediction(
                ranking_model,
                scalar_features,
                int_sim_ims_rel_1,
                int_sim_ims_rel_2,
                obs_sim_ims_rel_1,
                obs_sim_ims_rel_2,
                obs_obs_ims,
                device,
            )

            (
                bce_loss_value,
                cur_weighted_bce_loss_value,
                true,
                sample_weights,
            ) = compute_loss(
                loss,
                pred,
                res_area_rel_1,
                res_area_rel_2,
                site_correlations,
                im_weights,
                device,
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
            correct = (
                (torch.nn.functional.sigmoid(pred) >= 0.5).float() == true
            ).ravel()
            metrics["acc_hist_train"][epoch_ix] += correct.sum().item()
            metrics["weighted_acc_hist_train"][epoch_ix] += (
                sample_weights[correct].sum().item()
            )

            iter_loop.set_postfix(
                {"loss": cur_loss_value.item(), "acc": n_correct / pred.size(0)}
            )

        metrics["loss_hist_train"][epoch_ix] /= len(train_dataloader)
        metrics["bce_loss_hist_train"][epoch_ix] /= len(train_dataloader)
        metrics["acc_hist_train"][epoch_ix] /= len(train_dataloader.dataset)
        metrics["weighted_acc_hist_train"][epoch_ix] /= len(train_dataloader.dataset)

        # Validation
        with torch.no_grad():
            ranking_model.eval()
            for i, (
                _,
                scalar_features,
                int_sim_ims_rel_1,
                int_sim_ims_rel_2,
                obs_sim_ims_rel_1,
                obs_sim_ims_rel_2,
                obs_obs_ims,
                res_area_rel_1,
                res_area_rel_2,
                site_correlations,
            ) in enumerate(val_dataloader):

                pred = get_prediction(
                    ranking_model,
                    scalar_features,
                    int_sim_ims_rel_1,
                    int_sim_ims_rel_2,
                    obs_sim_ims_rel_1,
                    obs_sim_ims_rel_2,
                    obs_obs_ims,
                    device,
                )

                (
                    bce_loss_value,
                    cur_weighted_bce_loss_value,
                    true,
                    sample_weights,
                ) = compute_loss(
                    loss,
                    pred,
                    res_area_rel_1,
                    res_area_rel_2,
                    site_correlations,
                    im_weights,
                    device,
                )
                cur_loss_value = cur_weighted_bce_loss_value

                metrics["loss_hist_val"][epoch_ix] += cur_loss_value.item()
                metrics["bce_loss_hist_val"][epoch_ix] += bce_loss_value.item()

                correct = (
                    (torch.nn.functional.sigmoid(pred) >= 0.5).float() == true
                ).ravel()
                metrics["acc_hist_val"][epoch_ix] += correct.sum().item()
                metrics["weighted_acc_hist_val"][epoch_ix] += (
                    sample_weights[correct].sum().item()
                )

            metrics["loss_hist_val"][epoch_ix] /= len(val_dataloader)
            metrics["bce_loss_hist_val"][epoch_ix] /= len(val_dataloader)
            metrics["acc_hist_val"][epoch_ix] /= len(val_dataloader.dataset)
            metrics["weighted_acc_hist_val"][epoch_ix] /= len(val_dataloader.dataset)

            # Keep track of the best model
            if metrics[best_epoch_key][epoch_ix] < best_val_loss:
                best_model_state = ranking_model.state_dict()
                best_val_loss = metrics[best_epoch_key][epoch_ix]
                best_model_epoch = epoch_ix

        print(f"Epoch {epoch_ix + 1}/{hp_config.n_epochs}")
        print(
            f"\tTraining - Loss: {metrics['loss_hist_train'][epoch_ix]:.4f}, "
            f"Weighted Accuracy: {metrics['weighted_acc_hist_train'][epoch_ix]:.4f}, "
            f"Accuracy: {metrics['acc_hist_train'][epoch_ix]:.4f}, "
            f"BCELoss: {metrics['bce_loss_hist_train'][epoch_ix]:.4f}"
        )
        print(
            f"\tValidation - Loss: {metrics['loss_hist_val'][epoch_ix]:.4f}, "
            f"Weighted Accuracy: {metrics['weighted_acc_hist_val'][epoch_ix]:.4f}, "
            f"Accuracy: {metrics['acc_hist_val'][epoch_ix]:.4f}, "
            f"BCELoss: {metrics['bce_loss_hist_val'][epoch_ix]:.4f}"
        )

    return metrics, best_model_state, best_model_epoch


def get_prediction(
    ranking_model: nn.Module,
    scalar_features: torch.Tensor,
    int_sim_ims_rel_1: torch.Tensor,
    int_sim_ims_rel_2: torch.Tensor,
    obs_sim_ims_rel_1: torch.Tensor,
    obs_sim_ims_rel_2: torch.Tensor,
    obs_obs_ims: torch.Tensor,
    device: str,
):
    im_data = torch.cat(
        (
            int_sim_ims_rel_1[:, None, :],
            int_sim_ims_rel_2[:, None, :],
            obs_sim_ims_rel_1[:, None, :],
            obs_sim_ims_rel_2[:, None, :],
            obs_obs_ims[:, None, :],
        ),
        dim=1,
    ).to(device, dtype=torch.float32)

    scalar_features = scalar_features.to(device, dtype=torch.float32)

    pred = ranking_model(im_data, scalar_features)

    return pred


def get_dataset_predictions(
    dataset: PairDataset, ranking_model: nn.Module, device: str
):
    pred_dataloader = DataLoader(
        dataset, shuffle=False, batch_size=4096, num_workers=0, pin_memory=True
    )

    im_weights = torch.from_numpy(dataset.im_weights).to(device, dtype=torch.float32)

    results = []
    with torch.no_grad():
        loss = nn.BCEWithLogitsLoss(reduction="none")
        ranking_model.eval()
        for i, (
            data_ind,
            scalar_features,
            int_sim_ims_rel_1,
            int_sim_ims_rel_2,
            obs_sim_ims_rel_1,
            obs_sim_ims_rel_2,
            obs_obs_ims,
            res_misfit_rel_1,
            res_misfit_rel_2,
            site_correlations,
        ) in enumerate(pred_dataloader):
            pred = get_prediction(
                ranking_model,
                scalar_features,
                int_sim_ims_rel_1,
                int_sim_ims_rel_2,
                obs_sim_ims_rel_1,
                obs_sim_ims_rel_2,
                obs_obs_ims,
                device,
            )
            bce_loss_value, weighted_loss_value, true, weights = compute_loss(
                loss,
                pred,
                res_misfit_rel_1,
                res_misfit_rel_2,
                site_correlations,
                im_weights,
                device,
                reduce=False,
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

        results = pd.concat(results, axis=0, ignore_index=True)
        return results


def _get_sample_distribution(comp_results_df: pd.DataFrame):
    """
    Computes the ranking for each
    (event, site_int, site_obs) combination
    """
    rankings_df = []
    groups = comp_results_df.groupby(["event_id", "site_int", "site_obs"])
    for (cur_event, cur_site_int, cur_site_obs), cur_group in groups:
        cur_group = cur_group.sort_values(["rel_1", "rel_2"])
        cur_rel_combs = cur_group[["rel_1", "rel_2"]].values
        cur_pred = cur_group["pred"].values

        # pairwise_pred.get_site_ranking(cur_pred)
        cur_pred = pairwise_pred.normalize_preds(cur_rel_combs, cur_pred)
        cur_ranking, cur_comps_won = pairwise_pred.get_site_ranking(
            cur_pred, cur_rel_combs
        )

        comp_results_df.loc[cur_group.index, "pred"] = cur_pred
        cur_rank_df = pd.DataFrame(
            data=[cur_ranking, np.arange(1, cur_ranking.size + 1), cur_comps_won],
            index=["rel_id", "rank", "comps_won"],
        ).T
        cur_rank_df["event_id"] = cur_event
        cur_rank_df["site_int"] = cur_site_int
        cur_rank_df["site_obs"] = cur_site_obs
        # cur_rank_df["model_rel_prob"] = (
        #     cur_rank_df["comps_won"] / cur_rank_df["comps_won"].sum()
        # )

        rankings_df.append(cur_rank_df)

    rankings_df = pd.concat(rankings_df, axis=0, ignore_index=True)
    rankings_df.index = mlt.array_utils.numpy_str_join(
        "_",
        rankings_df.event_id.values.astype(str),
        rankings_df.site_int.values.astype(str),
        rankings_df.site_obs.values.astype(str),
        rankings_df.rel_id.values.astype(str),
    )

    return comp_results_df, rankings_df


@nb.njit
def _compute_model_residuals(
    group_keys: np.ndarray,
    rank_event_id: np.ndarray,
    rank_site_int_id: np.ndarray,
    rank_site_obs_id: np.ndarray,
    rank_rel_ids: np.ndarray,
    ranks: np.ndarray,
    sim_event_ids: np.ndarray,
    sim_site_ids: np.ndarray,
    sim_rel_ids: np.ndarray,
    sim_ims: np.ndarray,
    obs_event_ids: np.ndarray,
    obs_site_ids: np.ndarray,
    obs_ims: np.ndarray,
):
    """
    Computes the residual between the highest ranked realisation and
    the observed GM at the site of interest

    Note: All string arrays have to be converted to integers arrays
        using hash, as numba does not handle string arrays at this point
    """
    res_result = np.zeros((group_keys.shape[0], obs_ims.shape[1]))
    for ix in range(group_keys.shape[0]):
        cur_event, cur_site_int, cur_site_obs = group_keys[ix, :]
        m = (
            (rank_event_id == cur_event)
            & (rank_site_int_id == cur_site_int)
            & (rank_site_obs_id == cur_site_obs)
            & (ranks == 1)
        )
        best_rel = rank_rel_ids[m][0]

        cur_sim_ix = np.flatnonzero(
            (sim_event_ids == cur_event)
            & (sim_site_ids == cur_site_int)
            & (sim_rel_ids == best_rel)
        )[0]
        cur_obs_ix = np.flatnonzero(
            (obs_event_ids == cur_event) & (obs_site_ids == cur_site_int)
        )[0]

        res_result[ix, :] = np.log(obs_ims[cur_obs_ix, :]) - np.log(
            sim_ims[cur_sim_ix, :]
        )

    return res_result


# def compute_sample_best_residuals(
#     ranking_df: pd.DataFrame, sim_df: pd.DataFrame, obs_df: pd.DataFrame
# ):
#     """
#     Computes the residual between the highest ranked realisation and
#     the observed GM at the site of interest
#     """
#     groups = ranking_df.groupby(["event_id", "site_int", "site_obs"])
#     group_keys = np.asarray(list(groups.groups.keys()))
#
#     # Convert strings to integers as
#     # numba doesn't handle strings
#     string_to_int = np.vectorize(lambda x: hash(x))
#
#     res_results = _compute_model_residuals(
#         string_to_int(group_keys.astype(str)),
#         string_to_int(ranking_df.event_id.values.astype(str)),
#         string_to_int(ranking_df.site_int.values.astype(str)),
#         string_to_int(ranking_df.site_obs.values.astype(str)),
#         string_to_int(ranking_df.rel_id.values.astype(str)),
#         ranking_df["rank"].values.astype(int),
#         string_to_int(sim_df.event_id.values.astype(str)),
#         string_to_int(sim_df.site_id.values.astype(str)),
#         string_to_int(sim_df.rel_id.values.astype(str)),
#         sim_df.loc[:, constants.IMs].values,
#         string_to_int(obs_df.event_id.values.astype(str)),
#         string_to_int(obs_df.site_id.values.astype(str)),
#         obs_df.loc[:, constants.IMs].values,
#     )
#
#     res_df = pd.DataFrame(data=res_results, columns=constants.IMs)
#     res_df["event_id"] = group_keys[:, 0]
#     res_df["site_int"] = group_keys[:, 1]
#     res_df["site_obs"] = group_keys[:, 2]
#
#     return res_df


# def compute_scenario_residuals(
#     ranking_df: pd.DataFrame, sim_df: pd.DataFrame, obs_df: pd.DataFrame
# ):
#     """
#     Determines the best realisation for each scenario
#     and then computes the residuals with the observed GM
#     """
#     # Determine the best realisation for each scenario
#     rel_votes = {cur_event: {} for cur_event in ranking_df.event_id.unique()}
#     best_rel_df = []
#     groups = ranking_df.groupby(["event_id", "site_int"])
#     for (cur_event, cur_site_int), cur_group in groups:
#         cur_rel_comps_won
#
#
#         cur_best_rels = cur_group.loc[(cur_group["rank"] == 1)].copy()
#         assert cur_best_rels.shape[0] == cur_group.site_obs.unique().size
#
#         cur_best_rels["weight"] = np.ones(cur_best_rels.shape[0])
#         cur_rel_votes = (
#             cur_best_rels[["rel_id", "weight"]]
#             .groupby("rel_id")
#             .sum()
#             .sort_values("weight", ascending=False)
#         )
#         cur_best_rel = cur_rel_votes.index[0]
#
#         rel_votes[cur_event][cur_site_int] = cur_rel_votes
#         best_rel_df.append([cur_event, cur_site_int, cur_best_rel])
#
#     best_rel_df = pd.DataFrame(
#         data=best_rel_df, columns=["event_id", "site_int", "rel_id"]
#     )
#     best_rel_df.index = mlt.array_utils.numpy_str_join(
#         "_",
#         best_rel_df.event_id.values.astype(str),
#         best_rel_df.site_int.values.astype(str),
#         best_rel_df.rel_id.values.astype(str),
#     )
#
#     # Compute residuals
#     sim_idx = best_rel_df.index.values.astype(str)
#     obs_idx = np.stack(np.char.rsplit(sim_idx, "_", 1), axis=0)[:, 0]
#
#     res_df = pd.DataFrame(
#         data=np.log(obs_df.loc[obs_idx, constants.IMs].values)
#         - np.log(sim_df.loc[sim_idx, constants.IMs].values),
#         index=sim_idx,
#         columns=constants.IMs,
#     )
#     res_df = pd.concat([best_rel_df, res_df], axis=1)
#
#     return res_df, rel_votes


def compute_residuals(df: pd.DataFrame, sim_df: pd.DataFrame, obs_df: pd.DataFrame):
    """
    Computes the residual for each row
    Required columns: event_id, site_int, rel_id
    """
    df = df.copy()

    group = df.groupby(["event_id", "site_int"])
    for (cur_event, cur_site_int), cur_df in group:
        # Get the simulation data
        cur_sim_keys = mlt.array_utils.numpy_str_join(
            "_", cur_event, cur_site_int, cur_df.rel_id.values.astype(str)
        )
        cur_sim_df = sim_df.loc[cur_sim_keys, constants.IMs]

        # Compute the residual
        cur_res = (
            np.log(
                obs_df.loc[
                    mlt.array_utils.numpy_str_join("_", cur_event, cur_site_int),
                    constants.IMs,
                ].values.astype(float)
            )
            - np.log(cur_sim_df.values)
        )

        df.loc[cur_df.index, constants.IMs] = cur_res

    return df


def get_sample_weights(
    sim_corr_dir: Path, cur_event: str, site_int: int, site_obs: np.ndarray
):
    if sim_corr_dir is not None:
        site_corrs = pd.read_pickle(sim_corr_dir / f"{cur_event}.pickle")
        sample_weights = np.mean(
            np.abs(
                site_corrs.get_site_im_corrs(site_int, constants.IMs)
                .loc[site_obs]
                .values
            )
            * (constants.IM_weights * len(constants.IM_weights)),
            axis=1,
        )
    # Uniform weights otherwise
    else:
        print(f"Warning: Using uniform sample weights!")
        sample_weights = np.ones(site_obs.size) / site_obs.size

    # Normalize such that they add to one
    sample_weights = sample_weights / sample_weights.sum()

    return pd.Series(index=site_obs, data=sample_weights, name="sample_weight")


def compute_scenario_distribution(sample_results: pd.DataFrame, sim_corr_dir: Path):
    """
    Computes the realisation distribution for each scenario
    """
    scenario_results = []
    groups = sample_results.groupby(["event_id", "site_int"])
    for (cur_event, cur_site_int), cur_group in groups:
        site_obs = cur_group.site_obs.unique()

        sample_weights = get_sample_weights(
            sim_corr_dir, cur_event, cur_site_int, site_obs
        )

        # Compute the scenario realisation distribution
        wm = lambda x: np.average(
            x.comps_won.values.astype(float), weights=sample_weights.loc[x.site_obs]
        )
        cur_result = cur_group.groupby("rel_id").apply(wm).to_frame("comps_won")
        cur_result["event_id"] = cur_event
        cur_result["site_int"] = cur_site_int
        cur_result["rel_id"] = cur_result.index
        # cur_result["model_rel_prob"] = cur_result["comps_won"] / cur_result["comps_won"].sum()
        cur_result.index = mlt.array_utils.numpy_str_join(
            "_", cur_event, cur_site_int, cur_result.index.values.astype(str)
        )

        scenario_results.append(cur_result)

    scenario_df = pd.concat(scenario_results, axis=0)

    return scenario_df


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
    sim_corr_dir: Path = None,
    id_suffix: str = "",
):
    (
        cur_out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir()

    db = DB(os.path.expandvars(data_metadata["db"]))
    sim_df = db.get_sim_df()
    obs_df = db.get_obs_df(custom_record_id=True)

    # Get predictions
    print(f"Getting dataset predictions")
    start_time = time.time()
    train_comp_results = get_dataset_predictions(
        train_dataset, ranking_model, run_config.device
    )
    val_comp_results = get_dataset_predictions(
        val_dataset, ranking_model, run_config.device
    )
    print(f"Took {time.time() - start_time} to get predictions")

    print(f"Computing sample distributions")
    start_time = time.time()
    train_comp_results, train_sample_results = _get_sample_distribution(
        train_comp_results
    )
    val_comp_results, val_sample_results = _get_sample_distribution(val_comp_results)
    print(f"Took {time.time() - start_time}")

    print(f"Computing sample residuals")
    start_time = time.time()
    train_sample_results = compute_residuals(train_sample_results, sim_df, obs_df)
    val_sample_results = compute_residuals(val_sample_results, sim_df, obs_df)
    print(f"Took {time.time() - start_time} to compute residuals")

    print(f"Computing scenario distributions")
    start_time = time.time()
    train_scenario_results = compute_scenario_distribution(
        train_sample_results, sim_corr_dir
    )
    val_scenario_results = compute_scenario_distribution(
        val_sample_results, sim_corr_dir
    )
    print(f"Took {time.time() - start_time} to compute scenario distributions")

    print(f"Computing scenario residuals")
    start_time = time.time()
    train_scenario_results = compute_residuals(train_scenario_results, sim_df, obs_df)
    val_scenario_results = compute_residuals(val_scenario_results, sim_df, obs_df)
    print(f"Took {time.time() - start_time} to compute scenario residuals")

    # Save results
    train_comp_results.to_parquet(cur_out_dir / "train_comp_results.parquet")
    val_comp_results.to_parquet(cur_out_dir / "val_comp_results.parquet")

    train_sample_results.to_parquet(cur_out_dir / "train_sample_results.parquet")
    val_sample_results.to_parquet(cur_out_dir / "val_sample_results.parquet")

    train_scenario_results.to_parquet(cur_out_dir / "train_scenario_results.parquet")
    val_scenario_results.to_parquet(cur_out_dir / "val_scenario_results.parquet")

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
            (hp_config.batch_size, 5, len(constants.IMs)),
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
    sim_corr_dir: Path = None,
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
    ims_mean = np.mean(np.log(obs_data.loc[:, run_config.ims]), axis=0)
    ims_std = np.std(np.log(obs_data.loc[:, run_config.ims]), axis=0)

    # Create the datasets
    print(f"Creating datasets")
    train_dataset = PairDataset(
        train_event_sites,
        train_site_combs,
        db,
        run_config.ims,
        run_config.im_weights,
        scalar_features,
        ims_mean,
        ims_std,
        run_config.max_n_rels,
        sim_corr_dir=sim_corr_dir,
    )

    val_dataset = PairDataset(
        val_event_sites,
        val_site_combs,
        db,
        run_config.ims,
        run_config.im_weights,
        scalar_features,
        ims_mean,
        ims_std,
        # run_config.max_n_rels,
        max_n_rels=25,
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
        "max_n_rels": run_config.max_n_rels,
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


def create_model(
    hp_config: HyperParamsConfig, scalar_features: ml_data.ScalarFeatures, n_ims: int
):
    # # Create the model
    # n_conv_layers = len(hp_config.kernel_sizes)
    # padding = (
    #     [
    #         mlt.dl_utils.compute_same_conv_padding(
    #             len(constants.IMs), hp_config.kernel_sizes[0]
    #         )
    #     ]
    #     * n_conv_layers
    #     if n_conv_layers > 0
    #     else []
    # )

    ranking_model = models.PairWiseModel(
        hp_config.fc_units,
        scalar_features.n_scalar_features,
        n_ims,
    )

    print(f"Ranking model summary")
    summary(
        ranking_model,
        input_size=[
            (hp_config.batch_size, 5, n_ims),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
    )

    return ranking_model


class CustomTabularDataLoader:
    """
    Loosely based on
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, dataset: PairDataset, batch_size: int, shuffle: bool):
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

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration

        batch_ind = self.indices[
            self.i : min(self.i + self.batch_size, self.n_samples - 1)
        ]
        return self.dataset.get_batch(batch_ind)


# @nb.njit
# def _compute_single_best_sim_res(
#     event: int,
#     site_id: int,
#     sim_event_ids: np.ndarray,
#     sim_site_ids: np.ndarray,
#     rel_ids: np.ndarray,
#     sim_ims: np.ndarray,
#     obs_event_ids: np.ndarray,
#     obs_site_ids: np.ndarray,
#     obs_ims: np.ndarray,
# ):
#     """Helper function"""
#     cur_sim_mask = (sim_event_ids == event) & (sim_site_ids == site_id)
#     cur_sim_ims = sim_ims[cur_sim_mask, :]
#
#     cur_obs_mask = (obs_event_ids == event) & (obs_site_ids == site_id)
#     if np.count_nonzero(cur_obs_mask) == 0:
#         return np.full(sim_ims.shape[1], np.nan), 0
#
#     cur_obs_ims = obs_ims[cur_obs_mask, :]
#
#     cur_res = np.log(cur_obs_ims) - np.log(cur_sim_ims)
#     cur_best_res_ix = np.argmin(np.sum(np.abs(cur_res), axis=1))
#
#     return cur_res[cur_best_res_ix, :], rel_ids[cur_sim_mask][cur_best_res_ix]


# @nb.njit(parallel=True)
# def _compute_best_sim_res(
#     group_keys: np.ndarray,
#     sim_event_ids: np.ndarray,
#     sim_site_ids: np.ndarray,
#     rel_ids: np.ndarray,
#     sim_ims: np.ndarray,
#     obs_event_ids: np.ndarray,
#     obs_site_ids: np.ndarray,
#     obs_ims: np.ndarray,
# ):
#     """Computes the residual for each site using the best simulation residual"""
#     best_res = np.full((group_keys.shape[0], sim_ims.shape[1]), np.nan)
#     best_rel = np.zeros(group_keys.shape[0], dtype=np.int64)
#     for ix in nb.prange(group_keys.shape[0]):
#         best_res[ix, :], best_rel[ix] = _compute_single_best_sim_res(
#             group_keys[ix, 0],
#             group_keys[ix, 1],
#             sim_event_ids,
#             sim_site_ids,
#             rel_ids,
#             sim_ims,
#             obs_event_ids,
#             obs_site_ids,
#             obs_ims,
#         )
#
#     return best_res, best_rel


# def compute_best_sim_res(sim_df: pd.DataFrame, obs_df: pd.DataFrame):
#     """
#     Computes the residual for each event &
#     site using the best simulation realisation
#     """
#     string_to_int = np.vectorize(lambda x: hash(x))
#
#     group_keys = np.asarray(list(sim_df.groupby(["event_id", "site_id"]).groups.keys()))
#
#     best_res, best_rel = _compute_best_sim_res(
#         string_to_int(group_keys),
#         string_to_int(sim_df.event_id.values.astype(str)),
#         string_to_int(sim_df.site_id.values.astype(str)),
#         string_to_int(sim_df.rel_id.values.astype(str)),
#         sim_df[constants.IMs].values,
#         string_to_int(obs_df.event_id.values.astype(str)),
#         string_to_int(obs_df.site_id.values.astype(str)),
#         obs_df[constants.IMs].values,
#     )
#
#     # Drop rows with no observation values
#     drop_mask = np.all(np.isnan(best_res), axis=1)
#     best_res = best_res[~drop_mask, :]
#     best_rel = best_rel[~drop_mask]
#
#     unique_rels = np.unique(sim_df.rel_id.values.astype(str))
#     rel_lookup = pd.Series(index=string_to_int(unique_rels), data=unique_rels)
#
#     best_res_df = pd.DataFrame(
#         data=best_res,
#         index=mlt.array_utils.numpy_str_join(
#             "_", group_keys[~drop_mask, 0], group_keys[~drop_mask, 1]
#         ),
#         columns=constants.IMs,
#     )
#     best_res_df["event_id"] = group_keys[~drop_mask, 0]
#     best_res_df["site_id"] = group_keys[~drop_mask, 1]
#     best_res_df["rel_id"] = rel_lookup.loc[best_rel].values
#
#     return best_res_df
