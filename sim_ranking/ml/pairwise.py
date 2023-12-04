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


def compute_res_score(rs_obs: np.ndarray, rs_sim: np.ndarray):
    """
    Computes the similarity score for response spectrum

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
    res = rs_obs[..., None] - rs_sim
    res_score = np.sum(np.abs(res), axis=1)

    return res_score


@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    weight_penalty_factor: float
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
            params["weight_penalty_factor"],
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
            "weight_penalty_factor": self.weight_penalty_factor,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            # "n_channels": self.n_channels,
            # "kernel_sizes": self.kernel_sizes,
            "fc_units": self.fc_units,
        }


class PairDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        ims: Sequence[str],
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

        self.ims_mean = ims_mean
        self.ims_std = ims_std

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

            # cur_n_samples = self.site_combs[cur_event].shape[0] * int(
            #     (cur_n_rels ** 2 - cur_n_rels) / 2
            # )
            cur_n_samples = self.site_combs[cur_event].shape[0] * int(
                cur_n_rels ** 2 - cur_n_rels
            )
            self.n_samples_event.append(cur_n_samples)
        self.cum_n_samples_event = np.cumsum(self.n_samples_event)

        # Create the feature tensor
        self.scalar_features_tensor = ml_data.create_scalar_feature_tensor(
            self.events, self.event_sites, self.scalar_features
        )

        # Create the (normalised) IM inputs in the format
        # obs: (n_sites, n_ims)
        # sim: (n_sites, n_ims, n_rels)
        # And create the residual area scores
        #   format: (n_sites, n_rels)
        # And get the spatial correlations
        #   format: (n_sites, n_sites, n_ims)
        self.obs_ims = {}
        self.sim_ims = {}
        self.res_area = {}
        self.corr = {}
        for cur_event, cur_sites in event_sites.items():
            # Observed
            cur_obs_df = db.get_obs_data(cur_event, cur_sites)
            cur_obs_df.loc[:, self.ims] = np.log(
                cur_obs_df.loc[:, self.ims]
            )

            self.obs_ims[cur_event] = (
                cur_obs_df.loc[cur_sites, self.ims].values
                - self.ims_mean[self.ims].values
            ) / self.ims_std[self.ims].values

            # Get the simulation data
            cur_sim_df = db.get_sim_data(cur_event, cur_sites)
            cur_sim_df.loc[:, self.ims] = np.log(
                cur_sim_df.loc[:, self.ims]
            )
            cur_sim_data = np.full(
                (cur_sites.size, self.ims.size, self.n_rels_event[cur_event]),
                fill_value=np.nan,
            )

            for ix, cur_rel in enumerate(self.event_rels[cur_event]):
                cur_rel_data = (
                    cur_sim_df.loc[cur_sim_df.rel_id == cur_rel]
                    .set_index("site_id")
                    .loc[cur_sites, self.ims]
                    .values
                )
                cur_sim_data[:, :, ix] = cur_rel_data

            # Need to compute residual area before normalizing
            cur_res_area = compute_res_score(
                cur_obs_df.loc[cur_sites, self.ims].values, cur_sim_data
            )
            self.res_area[cur_event] = cur_res_area

            self.sim_ims[cur_event] = (
                cur_sim_data - self.ims_mean[self.ims].values[None, :, None]
            ) / self.ims_std[self.ims].values[None, :, None]

            # Get the (absolute) spatial correlations
            cur_corrs_values = np.ones(
                (cur_sites.size, cur_sites.size, len(self.ims))
            )
            if sim_corr_dir is not None:
                cur_corrs = pd.read_pickle(sim_corr_dir / f"{cur_event}.pickle")
                for ix, cur_im in enumerate(constants.IMs):
                    cur_corr_df = cur_corrs.get_im_corrs(cur_im)
                    cur_corrs_values[:, :, ix] = np.abs(
                        cur_corr_df.loc[cur_sites, cur_sites].values
                    )

            self.corr[cur_event] = cur_corrs_values

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
        # n_rel_combs = int((n_rels * (n_rels - 1)) / 2)
        n_rel_combs = n_rels ** 2 - n_rels

        within_site_ix = int(within_event_ix % n_rel_combs)

        # # Get the realisation indices
        # # Based on https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        # rel_1_ix = int(
        #     n_rels
        #     - 2
        #     - np.floor(
        #         np.sqrt(-8 * within_site_ix + 4 * n_rels * (n_rels - 1) - 7) / 2 - 0.5
        #     )
        # )
        # rel_2_ix = int(
        #     within_site_ix
        #     + rel_1_ix
        #     + 1
        #     - n_rels * (n_rels - 1) / 2
        #     + (n_rels - rel_1_ix) * ((n_rels - rel_1_ix) - 1) / 2
        # )

        row_length = n_rels - 1
        rel_1_ix = within_site_ix // row_length
        rel_2_ix = within_site_ix % row_length + (
            within_site_ix % row_length >= rel_1_ix
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
            self.sim_ims[event][site_int_ix, :, rel_1_ix],
            self.sim_ims[event][site_int_ix, :, rel_2_ix],
            self.sim_ims[event][site_obs_ix, :, rel_1_ix],
            self.sim_ims[event][site_obs_ix, :, rel_2_ix],
            self.obs_ims[event][site_obs_ix],
            self.res_area[event][site_int_ix, rel_1_ix],
            self.res_area[event][site_int_ix, rel_2_ix],
            self.corr[event][site_int_ix, site_obs_ix, :],
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
    site_correlations: torch.Tensor,
    device: str,
    reduce: bool = True,
):
    # Classification, is Rel 1 better than Rel 2?
    true = (res_area_1 < res_area_2)[:, None].to(device, dtype=torch.float32)
    loss_value = loss(pred, true)

    res_diff = torch.abs(res_area_1 - res_area_2).to(device, dtype=torch.float32)

    # Compute and normalize the sample weights
    res_weights = 1 - torch.exp(-0.075 * res_diff) + 0.05
    # res_weights = res_weights * (res_weights.shape[0] / res_weights.sum())

    # Compute site-correlation weights
    site_weights = torch.mean(site_correlations.to(device, dtype=torch.float32), dim=1)
    # site_weights = site_weights * (site_weights.shape[0] / site_weights.sum())

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

            bce_loss_value, cur_weighted_bce_loss_value, true, sample_weights = compute_loss(
                loss, pred, res_area_rel_1, res_area_rel_2, site_correlations, device
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
            correct = ((torch.nn.functional.sigmoid(pred) >= 0.5).float() == true).ravel()
            metrics["acc_hist_train"][epoch_ix] += correct.sum().item()
            metrics["weighted_acc_hist_train"][epoch_ix] += sample_weights[correct].sum().item()

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

                bce_loss_value, cur_weighted_bce_loss_value, true, sample_weights = compute_loss(
                    loss,
                    pred,
                    res_area_rel_1,
                    res_area_rel_2,
                    site_correlations,
                    device,
                )
                cur_loss_value = cur_weighted_bce_loss_value

                metrics["loss_hist_val"][epoch_ix] += cur_loss_value.item()
                metrics["bce_loss_hist_val"][epoch_ix] += bce_loss_value.item()

                correct = ((torch.nn.functional.sigmoid(pred) >= 0.5).float() == true).ravel()
                metrics["acc_hist_val"][epoch_ix] += correct.sum().item()
                metrics["weighted_acc_hist_val"][epoch_ix] += sample_weights[correct].sum().item()

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


def _get_ranking(results_df: pd.DataFrame):
    rankings_df = []
    groups = results_df.groupby(["event_id", "site_int", "site_obs"])
    for (cur_event, cur_site_int, cur_site_obs), cur_group in groups:
        cur_group = cur_group.sort_values(["rel_1", "rel_2"])
        cur_rel_combs = cur_group[["rel_1", "rel_2"]].values
        cur_pred = cur_group["pred"].values

        # pairwise_pred.get_site_ranking(cur_pred)
        cur_pred = pairwise_pred.normalize_preds(cur_rel_combs, cur_pred)
        cur_ranking, cur_comps_won = pairwise_pred.get_site_ranking(
            cur_pred, cur_rel_combs
        )

        results_df.loc[cur_group.index, "pred"] = cur_pred
        cur_rank_df = pd.DataFrame(
            data=[cur_ranking, np.arange(1, cur_ranking.size + 1), cur_comps_won],
            index=["rel_id", "rank", "comps_won"],
        ).T
        cur_rank_df["event_id"] = cur_event
        cur_rank_df["site_int"] = cur_site_int
        cur_rank_df["site_obs"] = cur_site_obs

        rankings_df.append(cur_rank_df)

    rankings_df = pd.concat(rankings_df, axis=0, ignore_index=True)

    return results_df, rankings_df


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


def compute_sample_residuals(
    ranking_df: pd.DataFrame, sim_df: pd.DataFrame, obs_df: pd.DataFrame
):
    """
    Computes the residual between the highest ranked realisation and
    the observed GM at the site of interest
    """
    groups = ranking_df.groupby(["event_id", "site_int", "site_obs"])
    group_keys = np.asarray(list(groups.groups.keys()))

    # Convert strings to integers as
    # numba doesn't handle strings
    string_to_int = np.vectorize(lambda x: hash(x))

    res_results = _compute_model_residuals(
        string_to_int(group_keys.astype(str)),
        string_to_int(ranking_df.event_id.values.astype(str)),
        string_to_int(ranking_df.site_int.values.astype(str)),
        string_to_int(ranking_df.site_obs.values.astype(str)),
        string_to_int(ranking_df.rel_id.values.astype(str)),
        ranking_df["rank"].values.astype(int),
        string_to_int(sim_df.event_id.values.astype(str)),
        string_to_int(sim_df.site_id.values.astype(str)),
        string_to_int(sim_df.rel_id.values.astype(str)),
        sim_df.loc[:, constants.IMs].values,
        string_to_int(obs_df.event_id.values.astype(str)),
        string_to_int(obs_df.site_id.values.astype(str)),
        obs_df.loc[:, constants.IMs].values,
    )

    res_df = pd.DataFrame(data=res_results, columns=constants.IMs)
    res_df["event_id"] = group_keys[:, 0]
    res_df["site_int"] = group_keys[:, 1]
    res_df["site_obs"] = group_keys[:, 2]

    return res_df


def compute_scenario_residuals(
    ranking_df: pd.DataFrame, sim_df: pd.DataFrame, obs_df: pd.DataFrame
):
    """
    Determines the best realisation for each scenario
    and then computes the residuals with the observed GM
    """
    # Determine the best realisation for each scenario
    rel_votes = {cur_event: {} for cur_event in ranking_df.event_id.unique()}
    best_rel_df = []
    groups = ranking_df.groupby(["event_id", "site_int"])
    for (cur_event, cur_site_int), cur_group in groups:
        cur_best_rels = cur_group.loc[(cur_group["rank"] == 1)].copy()
        assert cur_best_rels.shape[0] == cur_group.site_obs.unique().size

        cur_best_rels["weight"] = np.ones(cur_best_rels.shape[0])
        cur_rel_votes = (
            cur_best_rels[["rel_id", "weight"]]
            .groupby("rel_id")
            .sum()
            .sort_values("weight", ascending=False)
        )
        cur_best_rel = cur_rel_votes.index[0]

        rel_votes[cur_event][cur_site_int] = cur_rel_votes
        best_rel_df.append([cur_event, cur_site_int, cur_best_rel])

    best_rel_df = pd.DataFrame(
        data=best_rel_df, columns=["event_id", "site_int", "rel_id"]
    )
    best_rel_df.index = mlt.array_utils.numpy_str_join(
        "_",
        best_rel_df.event_id.values.astype(str),
        best_rel_df.site_int.values.astype(str),
        best_rel_df.rel_id.values.astype(str),
    )

    # Compute residuals
    sim_idx = best_rel_df.index.values.astype(str)
    obs_idx = np.stack(np.char.rsplit(sim_idx, "_", 1), axis=0)[:, 0]

    res_df = pd.DataFrame(
        data=np.log(obs_df.loc[obs_idx, constants.IMs].values)
        - np.log(sim_df.loc[sim_idx, constants.IMs].values),
        index=sim_idx,
        columns=constants.IMs,
    )
    res_df = pd.concat([best_rel_df, res_df], axis=1)

    return res_df, rel_votes


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

    db = DB(os.path.expandvars(data_metadata["db"]))
    sim_df = db.get_sim_df()
    obs_df = db.get_obs_df(custom_record_id=True)

    # Get predictions
    print(f"Getting dataset predictions")
    start_time = time.time()
    train_results = get_dataset_predictions(
        train_dataset, ranking_model, run_config.device
    )
    val_results = get_dataset_predictions(val_dataset, ranking_model, run_config.device)
    print(f"Took {time.time() - start_time} to get predictions")

    print(f"Getting rankings")
    start_time = time.time()
    train_results, train_ranking = _get_ranking(train_results)
    val_results, val_ranking = _get_ranking(val_results)
    print(f"Took {time.time() - start_time}")

    print(f"Computing sample residuals")
    start_time = time.time()
    train_sample_residuals = compute_sample_residuals(train_ranking, sim_df, obs_df)
    val_sample_residuals = compute_sample_residuals(val_ranking, sim_df, obs_df)
    print(f"Took {time.time() - start_time} to compute sample residuals")

    print(f"Computing scenario residuals")
    start_time = time.time()
    train_scenario_res_df, train_scenario_rel_votes = compute_scenario_residuals(
        train_ranking, sim_df, obs_df
    )
    val_scenario_res_df, val_scenario_rel_votes = compute_scenario_residuals(
        val_ranking, sim_df, obs_df
    )
    print(f"Took {time.time() - start_time} to compute scenario best rels")

    # Save results
    train_ranking.to_csv(cur_out_dir / "train_rankings.csv")
    val_ranking.to_csv(cur_out_dir / "val_rankings.csv")

    train_results.to_csv(cur_out_dir / "train_results.csv")
    val_results.to_csv(cur_out_dir / "val_results.csv")

    train_sample_residuals.to_csv(cur_out_dir / "train_sample_residuals.csv")
    val_sample_residuals.to_csv(cur_out_dir / "val_sample_residuals.csv")

    train_scenario_res_df.to_csv(cur_out_dir / "train_scenario_residuals.csv")
    val_scenario_res_df.to_csv(cur_out_dir / "val_scenario_residuals.csv")

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
    ims_mean = np.mean(np.log(obs_data.loc[:, constants.IMs]), axis=0)
    ims_std = np.std(np.log(obs_data.loc[:, constants.IMs]), axis=0)

    # Create the datasets
    train_dataset = PairDataset(
        train_event_sites,
        train_site_combs,
        db,
        constants.IMs,
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
        constants.IMs,
        scalar_features,
        ims_mean,
        ims_std,
        run_config.max_n_rels,
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


def create_model(hp_config: HyperParamsConfig, scalar_features: ml_data.ScalarFeatures):
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
        len(constants.IMs),
    )

    print(f"Ranking model summary")
    summary(
        ranking_model,
        input_size=[
            (hp_config.batch_size, 5, len(constants.IMs)),
            (hp_config.batch_size, scalar_features.n_scalar_features),
        ],
    )

    return ranking_model


@nb.njit
def _compute_single_best_sim_res(
    event: int,
    site_id: int,
    sim_event_ids: np.ndarray,
    sim_site_ids: np.ndarray,
    rel_ids: np.ndarray,
    sim_ims: np.ndarray,
    obs_event_ids: np.ndarray,
    obs_site_ids: np.ndarray,
    obs_ims: np.ndarray,
):
    """Helper function"""
    cur_sim_mask = (sim_event_ids == event) & (sim_site_ids == site_id)
    cur_sim_ims = sim_ims[cur_sim_mask, :]

    cur_obs_mask = (obs_event_ids == event) & (obs_site_ids == site_id)
    if np.count_nonzero(cur_obs_mask) == 0:
        return np.full(sim_ims.shape[1], np.nan), 0

    cur_obs_ims = obs_ims[cur_obs_mask, :]

    ## TODO: Update this for more IMs
    cur_res = np.log(cur_obs_ims) - np.log(cur_sim_ims)
    cur_best_res_ix = np.argmin(np.sum(np.abs(cur_res), axis=1))

    return cur_res[cur_best_res_ix, :], rel_ids[cur_sim_mask][cur_best_res_ix]


@nb.njit(parallel=True)
def _compute_best_sim_res(
    group_keys: np.ndarray,
    sim_event_ids: np.ndarray,
    sim_site_ids: np.ndarray,
    rel_ids: np.ndarray,
    sim_ims: np.ndarray,
    obs_event_ids: np.ndarray,
    obs_site_ids: np.ndarray,
    obs_ims: np.ndarray,
):
    """Computes the residual for each site using the best simulation residual"""
    best_res = np.full((group_keys.shape[0], sim_ims.shape[1]), np.nan)
    best_rel = np.zeros(group_keys.shape[0], dtype=np.int64)
    for ix in nb.prange(group_keys.shape[0]):
        best_res[ix, :], best_rel[ix] = _compute_single_best_sim_res(
            group_keys[ix, 0],
            group_keys[ix, 1],
            sim_event_ids,
            sim_site_ids,
            rel_ids,
            sim_ims,
            obs_event_ids,
            obs_site_ids,
            obs_ims,
        )

    return best_res, best_rel


def compute_best_sim_res(sim_df: pd.DataFrame, obs_df: pd.DataFrame):
    """
    Computes the residual for each event &
    site using the best simulation realisation
    """
    string_to_int = np.vectorize(lambda x: hash(x))

    group_keys = np.asarray(list(sim_df.groupby(["event_id", "site_id"]).groups.keys()))

    best_res, best_rel = _compute_best_sim_res(
        string_to_int(group_keys),
        string_to_int(sim_df.event_id.values.astype(str)),
        string_to_int(sim_df.site_id.values.astype(str)),
        string_to_int(sim_df.rel_id.values.astype(str)),
        sim_df[constants.IMs].values,
        string_to_int(obs_df.event_id.values.astype(str)),
        string_to_int(obs_df.site_id.values.astype(str)),
        obs_df[constants.IMs].values,
    )

    # Drop rows with no observation values
    drop_mask = np.all(np.isnan(best_res), axis=1)
    best_res = best_res[~drop_mask, :]
    best_rel = best_rel[~drop_mask]

    unique_rels = np.unique(sim_df.rel_id.values.astype(str))
    rel_lookup = pd.Series(index=string_to_int(unique_rels), data=unique_rels)

    best_res_df = pd.DataFrame(
        data=best_res,
        index=mlt.array_utils.numpy_str_join(
            "_", group_keys[~drop_mask, 0], group_keys[~drop_mask, 1]
        ),
        columns=constants.IMs,
    )
    best_res_df["event_id"] = group_keys[~drop_mask, 0]
    best_res_df["site_id"] = group_keys[~drop_mask, 1]
    best_res_df["rel_id"] = rel_lookup.loc[best_rel].values

    return best_res_df
