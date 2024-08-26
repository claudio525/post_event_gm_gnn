import os
import itertools
import pickle
import warnings
import multiprocessing as mp
from typing import NamedTuple, Sequence
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import tqdm

import ml_tools as mlt

from . import data as ml_data
from .. import constants
from ..data_classes import ObservedData


@dataclass
class RunConfig:

    ### General settings
    seed: int

    ### Input settings
    rel_obs_data_ffp: Path
    # Maximum distance between site-interest and observation sites
    max_dist: float
    # Number of validation events
    n_val_events: int
    # Number of validation sites
    n_val_sites: int
    # Validation events ffp
    rel_val_sites_ffp: str

    # Device to use
    device: str

    ### Model settings
    # IMs to predict
    ims: Sequence[str]
    # Whether to predict the standard deviation
    pred_std: bool

    ### Hyperparameters
    # Number of training epochs
    n_epochs: int
    # Batch size
    batch_size: int

    # Base output directory
    rel_results_dir: str

    ### Optional settings
    # Events to be used for validation
    val_events: Sequence[str] = None
    # Events to be ignored
    test_events: Sequence[str] = None

    def __post_init__(self):
        assert self.n_val_events > 0
        assert self.rel_val_sites_ffp is not None or self.n_val_sites > 0

        assert self.obs_data_ffp.exists()
        assert self.val_sites_ffp is None or self.val_sites_ffp.exists()

    @property
    def n_ims(self):
        return len(self.ims)

    @property
    def n_outputs(self):
        return self.n_ims * 2 if self.pred_std else self.n_ims

    @property
    def results_dir(self):
        return Path(os.path.expandvars("$wdata")) / self.rel_results_dir

    @property
    def val_sites_ffp(self):
        if self.rel_val_sites_ffp is not None:
            return Path(os.path.expandvars("$wdata")) / self.rel_val_sites_ffp
        return None

    @property
    def obs_data_ffp(self):
        return Path(os.path.expandvars("$wdata")) / self.rel_obs_data_ffp

    def to_dict(self):
        return {
            "seed": self.seed,
            "rel_obs_data_ffp": self.rel_obs_data_ffp,
            "max_dist": self.max_dist,
            "n_val_events": self.n_val_events,
            "n_val_sites": self.n_val_sites,
            "rel_val_sites_ffp": self.rel_val_sites_ffp,
            "val_events": (
                list(self.val_events) if self.val_events is not None else None
            ),
            "test_events": (
                list(self.test_events) if self.test_events is not None else None
            ),
            "device": self.device,
            "ims": list(self.ims),
            "pred_std": self.pred_std,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "rel_results_dir": self.rel_results_dir,
        }

    def to_yaml(self, ffp: Path):
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    @classmethod
    def from_config_kwargs(cls, config_ffp: Path, **kwargs):
        """
        Creates an instance from the given config.
        If kwargs are set then they overwrite the values
        specified in the config.
        """
        config_dict = mlt.utils.load_yaml(config_ffp)

        for cur_key, cur_val in kwargs.items():
            if cur_val is not None:
                config_dict[cur_key] = cur_val

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))


def _get_event_graph_data(
    event: str,
    sites: np.ndarray,
    site_combs: np.ndarray,
    obs_data: ObservedData,
    scalar_feature_values,
    site_int_feature_keys: list[str],
    site_obs_scalar_feature_keys: list[str],
    edge_feature_keys: list[str],
    ims: Sequence[str],
):
    graph_data = []

    # cur_sites = event_sites[cur_event]
    cur_site_int_inds = np.unique(site_combs[:, 0])

    # cur_scalar_feature_values = scalar_event_feature_values[cur_event]

    # Get and normalise the IM data
    cur_im_data = obs_data.get_event_data(event, sites)
    cur_im_data = np.log(cur_im_data.loc[:, ims])
    # cur_norm_im_data = (cur_im_data - ims_mean) / ims_std

    for cur_site_int_ix in cur_site_int_inds:
        cur_site_combs_mask = site_combs[:, 0] == cur_site_int_ix
        cur_site_int = sites[site_combs[cur_site_combs_mask, 0][0]]
        cur_obs_sites = sites[site_combs[cur_site_combs_mask, 1]]

        # Create the site_int node features
        cur_site_int_features = scalar_feature_values.loc[
            cur_site_combs_mask, site_int_feature_keys
        ].values[0]

        # Create the site_obs node features
        cur_obs_sites_features = scalar_feature_values.loc[
            cur_site_combs_mask, site_obs_scalar_feature_keys
        ].values
        # Add the IM values
        cur_obs_sites_features = np.concatenate(
            (
                cur_obs_sites_features,
                cur_im_data.loc[cur_obs_sites, ims].values,
            ),
            axis=1,
        )

        # Create the edge features
        cur_edge_features = scalar_feature_values.loc[
            cur_site_combs_mask, edge_feature_keys
        ].values

        cur_sc_data = gdata.HeteroData()
        cur_sc_data["site_int"].x = torch.tensor(
            cur_site_int_features, dtype=torch.float32
        )[None, :]
        cur_sc_data["site_obs"].x = torch.tensor(
            cur_obs_sites_features, dtype=torch.float32
        )

        cur_sc_data["site_obs", "informs", "site_int"].edge_index = torch.tensor(
            [[ix, 0] for ix, cur_obs_site in enumerate(cur_obs_sites)],
            dtype=torch.long,
        ).T
        cur_sc_data["site_obs", "informs", "site_int"].edge_attr = torch.tensor(
            cur_edge_features, dtype=torch.float32
        )

        cur_sc_data["metadata"] = {
            "event": event,
            "site_int": cur_site_int,
            "obs_sites": cur_obs_sites,
        }

        cur_sc_data["y"] = torch.tensor(
            cur_im_data.loc[cur_site_int, ims].values, dtype=torch.float32
        )[None, :]

        graph_data.append(cur_sc_data)

    return pickle.dumps(graph_data)


def get_graph_data(
    obs_data: ObservedData,
    event_sites: dict[str, np.ndarray],
    event_site_combs: dict[str, np.ndarray],
    scalar_features: ml_data.ScalarFeatures,
    site_int_feature_keys: list[str],
    site_obs_scalar_feature_keys: list[str],
    edge_feature_keys: list[str],
    # ims_mean: pd.DataFrame,
    # ims_std: pd.DataFrame,
    ims: Sequence[str],
    n_procs: int = 1,
):
    # Create the scalar features tensors
    scalar_event_feature_values, scalar_feature_columns = (
        ml_data.create_scalar_feature_tensor(
            event_sites, scalar_features, event_site_combs
        )
    )

    n_site_obs_scalar_features = len(site_obs_scalar_feature_keys)
    site_obs_scalar_feature_ind = np.arange(n_site_obs_scalar_features)

    # Create the graph data objects
    graph_data = []
    if n_procs == 1:
        for cur_event, cur_site_combs in tqdm.tqdm(event_site_combs.items()):
            graph_data.append(
                _get_event_graph_data(
                    cur_event,
                    event_sites[cur_event],
                    cur_site_combs,
                    obs_data,
                    scalar_event_feature_values[cur_event],
                    site_int_feature_keys,
                    site_obs_scalar_feature_keys,
                    edge_feature_keys,
                    ims,
                )
            )
    else:
        with mp.Pool(n_procs) as pool:
            graph_data = pool.starmap(
                _get_event_graph_data,
                [
                    (
                        cur_event,
                        event_sites[cur_event],
                        cur_site_combs,
                        obs_data,
                        scalar_event_feature_values[cur_event],
                        site_int_feature_keys,
                        site_obs_scalar_feature_keys,
                        edge_feature_keys,
                        ims,
                    )
                    for cur_event, cur_site_combs in event_site_combs.items()
                ],
            )
            graph_data = [pickle.loads(data) for data in graph_data]

    graph_data = list(itertools.chain(*graph_data))
    return graph_data, site_obs_scalar_feature_ind


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: bool,
    scale_constant: float = 10.0,
):
    """Computes the loss per sample and output"""
    if pred_std:
        pred_mean, pred_ln_std = pred.chunk(2, dim=1)
        loss = F.gaussian_nll_loss(
            pred_mean, target, torch.exp(pred_ln_std) ** 2, reduction="none"
        )
        return loss * scale_constant
    else:
        return F.mse_loss(pred, target, reduction="none") * scale_constant


def reduce_loss(loss: torch.Tensor, dim: int = None):
    """Computes the batch loss"""
    return loss.mean(dim=dim)


def train(
    run_config: RunConfig,
    gnn_model: torch.nn.Module,
    train_loader: gdata.DataLoader,
    val_loader: gdata.DataLoader,
):
    # Setup metrics to log
    metrics = {
        "loss_hist_train": np.zeros(run_config.n_epochs),
        "loss_hist_val": np.zeros(run_config.n_epochs),
    }

    best_val_loss = np.inf
    best_model_state, best_model_epoch = None, None

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    for cur_epoch_ix in range(run_config.n_epochs):
        print(f"Epoch: {cur_epoch_ix}")

        ### Training
        n_graphs = 0
        gnn_model.train()
        for cur_batch in tqdm.tqdm(train_loader):
            cur_batch = cur_batch.to(run_config.device)
            cur_y = cur_batch.y if cur_batch.y.dim() > 1 else cur_batch.y[:, None]

            optimizer.zero_grad()
            out = gnn_model(cur_batch)
            loss = reduce_loss(compute_loss(out, cur_y, run_config.pred_std))
            loss.backward()
            optimizer.step()

            metrics["loss_hist_train"][cur_epoch_ix] += loss.item()
            n_graphs += cur_batch.num_graphs

        metrics["loss_hist_train"][cur_epoch_ix] /= n_graphs

        ### Validation
        gnn_model.eval()
        n_graphs = 0
        with torch.no_grad():
            for cur_batch in val_loader:
                cur_batch = cur_batch.to(run_config.device)
                cur_y = cur_batch.y if cur_batch.y.dim() > 1 else cur_batch.y[:, None]

                out = gnn_model(cur_batch)
                loss = reduce_loss(compute_loss(out, cur_y, run_config.pred_std))

                metrics["loss_hist_val"][cur_epoch_ix] += loss.item()
                n_graphs += cur_batch.num_graphs

        metrics["loss_hist_val"][cur_epoch_ix] /= n_graphs

        # Keep track of the best model
        if metrics["loss_hist_val"][cur_epoch_ix] < best_val_loss:
            best_val_loss = metrics["loss_hist_val"][cur_epoch_ix]
            best_model_state = gnn_model.state_dict()
            best_model_epoch = cur_epoch_ix

        print(f"Epoch {cur_epoch_ix}/{run_config.n_epochs}")
        print(f"\tTraining" f"\t\tLoss: {metrics['loss_hist_train'][cur_epoch_ix]:.4f}")
        print(
            f"\tValidation" f"\t\tLoss: {metrics['loss_hist_val'][cur_epoch_ix] :.4f}"
        )

    return metrics, best_model_state, best_model_epoch


def get_predictions(
    run_config: RunConfig,
    gnn_model: torch.nn.Module,
    graph_data: Sequence[gdata.HeteroData],
):
    gnn_model.eval()
    loader = gloader.DataLoader(graph_data, batch_size=128, shuffle=False)

    pred_im_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
    pred_im_std_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_std")
    loss_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "loss")

    results = []
    for cur_batch in tqdm.tqdm(loader):
        cur_batch = cur_batch.to(run_config.device)
        cur_out = gnn_model(cur_batch)

        cur_result = pd.DataFrame(
            data=[
                cur_batch["metadata"]["event"],
                cur_batch["metadata"]["site_int"],
                cur_batch["metadata"]["obs_sites"],
            ],
            index=["event_id", "site_int", "obs_sites"],
        ).T
        cur_result.loc[:, run_config.ims] = cur_batch["y"].cpu().numpy(force=True)
        if run_config.pred_std:
            pred_im, pred_im_ln_std = cur_out.chunk(2, dim=1)
            cur_result.loc[:, pred_im_keys] = pred_im.cpu().numpy(force=True)
            cur_result.loc[:, pred_im_std_keys] = (
                torch.exp(pred_im_ln_std).cpu().numpy(force=True)
            )
        else:
            cur_result.loc[:, pred_im_keys] = cur_out.cpu().numpy(force=True)

        # Loss
        cur_loss = compute_loss(cur_out, cur_batch.y, run_config.pred_std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            cur_result.loc[:, loss_keys] = cur_loss.cpu().numpy(force=True)
            cur_result.loc[:, "loss"] = (
                reduce_loss(cur_loss, dim=1).cpu().numpy(force=True)
            )

        # Index
        # cur_obs_site_hash_values = [
        #     base64.urlsafe_b64encode(
        #         hashlib.sha256("_".join(list(cur_site_obs)).encode()).digest()
        #     )
        #     .rstrip(b"=")
        #     .decode("utf-8")[:10]
        #     for cur_site_obs in cur_result["obs_sites"]
        # ]
        cur_result = cur_result.set_index(
            mlt.array_utils.numpy_str_join(
                "_",
                cur_result["event_id"].values.astype(str),
                cur_result["site_int"].values.astype(str),
                # cur_obs_site_hash_values,
            )
        )

        # Number of observation sites
        cur_result.loc[:, "n_obs_sites"] = cur_result["obs_sites"].apply(len)

        results.append(cur_result)

    results = pd.concat(results, axis=0)
    return results


def get_residuals(gnn_results: pd.DataFrame, ims: Sequence[str] = constants.PSA_KEYS):
    """Computes the residual between the observed and predicted IMs for each scenario"""
    pred_im_keys = mlt.array_utils.numpy_str_join("_", ims, "pred")
    res_df = pd.DataFrame(
        data=gnn_results.loc[:, ims].values - gnn_results.loc[:, pred_im_keys].values,
        columns=ims,
    )

    res_df.index = gnn_results.index
    res_df["event_id"] = gnn_results["event_id"]
    res_df["site_int"] = gnn_results["site_int"]
    res_df["n_obs_sites"] = gnn_results["n_obs_sites"]
    return res_df
