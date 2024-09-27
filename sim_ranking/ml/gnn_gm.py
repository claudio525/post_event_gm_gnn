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
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import tqdm

import ml_tools as mlt

from . import data as ml_data
from . import gnn_modules
from .. import constants
from ..data_classes import ObservedData


@dataclass
class RunConfig:
    """Config for specyfing run settings"""

    ### General settings
    seed: int

    ### Input settings
    rel_obs_data_ffp: Path
    # Maximum distance between site-interest and observation sites
    max_dist: float

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
    # Number of site of interest node channels
    n_int_node_channels: Sequence[int]
    # Number of fully connected units for the output MLP
    fc_n_units: int

    # Base output directory
    rel_results_dir: str

    def __post_init__(self):
        assert self.obs_data_ffp.exists()

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
    def obs_data_ffp(self):
        return Path(os.path.expandvars("$wdata")) / self.rel_obs_data_ffp

    def to_dict(self):
        return {
            "seed": self.seed,
            "rel_obs_data_ffp": self.rel_obs_data_ffp,
            "max_dist": self.max_dist,
            "device": self.device,
            "ims": list(self.ims),
            "pred_std": self.pred_std,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "n_int_node_channels": list(self.n_int_node_channels),
            "fc_n_units": self.fc_n_units,
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


@dataclass
class HoldoutConfig:
    """Config for using holdout data"""

    n_val_events: int
    """Number of validation events"""

    n_val_sites: int
    """Number of validation sites"""

    rel_val_sites_ffp: str
    """Validation events ffp"""

    val_events: Sequence[str] = None
    """Events to be used for validation"""

    test_events: Sequence[str] = None
    """Events to be ignored"""

    def __post_init__(self):
        assert self.n_val_events > 0
        assert self.rel_val_sites_ffp is not None or self.n_val_sites > 0
        assert self.val_sites_ffp is None or self.val_sites_ffp.exists()

    @property
    def val_sites_ffp(self):
        if self.rel_val_sites_ffp is not None:
            return Path(os.path.expandvars("$wdata")) / self.rel_val_sites_ffp
        return None

    def to_dict(self):
        return {
            "n_val_events": self.n_val_events,
            "n_val_sites": self.n_val_sites,
            "rel_val_sites_ffp": self.rel_val_sites_ffp,
            "val_events": (
                list(self.val_events) if self.val_events is not None else None
            ),
            "test_events": (
                list(self.test_events) if self.test_events is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))


def run_model_training(
    out_dir: Path,
    event_sites: dict[str, np.ndarray[str]],
    valid_event_int_sites: dict[str, np.ndarray[str]],
    train_events: np.ndarray[str],
    val_events: np.ndarray[str],
    train_int_sites: np.ndarray[str],
    val_int_sites: np.ndarray[str],
    obs_sites: np.ndarray[str],
    dist_matrix: pd.DataFrame,
    obs_data: ObservedData,
    scalar_features: ml_data.ScalarFeatures,
    run_config: RunConfig,
    graph_data_n_procs: int = 8,
    verbose: bool = True,
):
    """
    Performs an individual run of the GNN model

    Parameters
    ----------
    out_dir: Path
    event_sites: dict[str, np.ndarray[str]]
        Available sites per event
    valid_event_int_sites: dict[str, np.ndarray[str]]
        Valid site of interests per event
    train_events: np.ndarray[str]
        Events for model training
    val_events: np.ndarray[str]
        Events for model evaluation
    train_int_sites: np.ndarray[str]
        Site of interests for model training
    val_int_sites
        Site of interests for model evaluation
    obs_sites: np.ndarray[str]
        Sites that can be used as observation sites
    dist_matrix: pd.DataFrame
    obs_data: ObservedData
    scalar_features: ml_data.ScalarFeatures
    run_config: RunConfig
    graph_data_n_procs: int, optional
        Number of processes to use for
        creating the graph data
    verbose: bool, optional
        Whether to print progress information, by default True
    """
    if verbose:
        print(f"Creating site combinations")
    train_site_combs, train_event_sites = ml_data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        train_events,
        dist_matrix,
        obs_sites,
        train_int_sites,
        max_dist=run_config.max_dist,
    )
    val_site_combs, val_event_sites = ml_data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        val_events,
        dist_matrix,
        obs_sites,
        val_int_sites,
        max_dist=run_config.max_dist,
    )

    graph_feature_keys = constants.GRAPH_FEATURE_KEYS

    if verbose:
        print(f"Getting graph data")
    train_graph_data, site_obs_scalar_feature_ind = get_graph_data(
        obs_data,
        train_event_sites,
        train_site_combs,
        scalar_features,
        constants.GRAPH_FEATURE_KEYS,
        run_config.ims,
        n_procs=graph_data_n_procs,
        verbose=verbose,
    )

    val_graph_data, _ = get_graph_data(
        obs_data,
        val_event_sites,
        val_site_combs,
        scalar_features,
        constants.GRAPH_FEATURE_KEYS,
        run_config.ims,
        n_procs=graph_data_n_procs,
        verbose=verbose,
    )

    train_loader = gloader.DataLoader(
        train_graph_data, batch_size=run_config.batch_size, shuffle=True
    )
    val_loader = gloader.DataLoader(
        val_graph_data, batch_size=run_config.batch_size, shuffle=True
    )

    gnn_model = gnn_modules.BasicAttentionGNN(
        len(graph_feature_keys["site_obs"]) + len(run_config.ims),
        len(graph_feature_keys["site_obs"]),
        len(graph_feature_keys["site_int"]),
        len(graph_feature_keys["edge"]),
        run_config,
        torch.from_numpy(site_obs_scalar_feature_ind),
    )
    gnn_model.to(run_config.device)

    if verbose:
        print(f"----------------- Training -----------------")
        print(f"Number of training graphs: {len(train_graph_data)}")
        print(f"Number of validation graphs: {len(val_graph_data)}")

    metrics, best_model_state, best_model_epoch = train(
        run_config, gnn_model, train_loader, val_loader, verbose=verbose
    )

    if verbose:
        print(
            f"Best model epoch: {best_model_epoch + 1}, "
            f"Validation: \tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
        )

    # Load the best model
    gnn_model.load_state_dict(best_model_state)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the training sites and validation sites
    np.save(out_dir / "val_int_sites.npy", val_int_sites)
    np.save(out_dir / "train_int_sites.npy", train_int_sites)
    np.save(out_dir / "obs_sites.npy", obs_sites)

    # Save the run config
    run_config.to_yaml(out_dir / "run_config.yaml")

    # Save loss history
    pd.to_pickle(metrics, out_dir / "metrics.pickle")

    # Save the model
    torch.save(gnn_model, out_dir / "model.pt")

    # Save the results
    train_results_df = get_predictions(
        run_config, gnn_model, train_graph_data, verbose=verbose
    )
    train_results_df.to_parquet(out_dir / "train_results.parquet")

    val_results_df = get_predictions(
        run_config, gnn_model, val_graph_data, verbose=verbose
    )
    val_results_df.to_parquet(out_dir / "val_results.parquet")

    # Write the metadata
    metadata = {
        "best_model_epoch": int(best_model_epoch),
        "best_model_loss": float(metrics["loss_hist_val"][best_model_epoch]),
        "n_train_scenarios": len(train_graph_data),
        "n_val_scenarios": len(val_graph_data),
    }
    mlt.utils.write_to_yaml(metadata, out_dir / "metadata.yaml")


def _get_event_graph_data(
    event: str,
    sites: np.ndarray,
    site_combs: np.ndarray,
    obs_data: ObservedData,
    scalar_feature_values: pd.DataFrame,
    graph_feature_keys: dict[str, Sequence[str]],
    ims: Sequence[str],
):
    """Helper function, see get_graph_data"""
    graph_data = []

    # cur_sites = event_sites[cur_event]
    cur_site_int_inds = np.unique(site_combs[:, 0])

    # cur_scalar_feature_values = scalar_event_feature_values[cur_event]

    # Get and normalise the IM data
    cur_im_data = obs_data.get_event_data(event, sites)
    cur_im_data = np.log(cur_im_data.loc[:, ims])

    for cur_site_int_ix in cur_site_int_inds:
        cur_site_combs_mask = site_combs[:, 0] == cur_site_int_ix
        cur_site_int = sites[site_combs[cur_site_combs_mask, 0][0]]
        cur_obs_sites = sites[site_combs[cur_site_combs_mask, 1]]

        # Create the site_int node features
        cur_site_int_features = scalar_feature_values.loc[
            cur_site_combs_mask, graph_feature_keys["site_int"]
        ].values[0]

        # Create the site_obs node features
        cur_obs_sites_features = scalar_feature_values.loc[
            cur_site_combs_mask,
            graph_feature_keys["site_obs"],
        ].values
        # Get observation site IM values and deal with nan values
        cur_obs_sites_im_values = (
            cur_im_data.loc[cur_obs_sites, ims].replace(np.nan, 99).values
        )
        # Add the IM values
        cur_obs_sites_features = np.concatenate(
            (
                cur_obs_sites_features,
                cur_obs_sites_im_values,
            ),
            axis=1,
        )

        # Create the edge features
        cur_edge_features = scalar_feature_values.loc[
            cur_site_combs_mask, graph_feature_keys["edge"]
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
    graph_feature_keys: dict[str, Sequence[str]],
    ims: Sequence[str],
    n_procs: int = 1,
    verbose: bool = True,
):
    """
    Get the graph data for the given scenarios

    Parameters
    ----------
    obs_data: ObservedData
    event_sites: dict[str, np.ndarray]
    event_site_combs: dict[str, np.ndarray]
        Scenario definitions
    scalar_features: ml_data.ScalarFeatures
    graph_feature_keys: dict[str, Sequence[str]]
        Graph feature keys for the
        different node & edge types
    ims: Sequence[str]
    n_procs: int, optional
        Number of processes to use, by default 1
    verbose: bool, optional
        Whether to print progress information, by default True

    Returns
    -------
    graph_data: list[gdata.HeteroData]
    site_obs_scalar_feature_ind: np.ndarray
        Indices of the scalar features for the site_obs node features
    """
    # Create the scalar features tensors
    scalar_event_feature_values, scalar_feature_columns = (
        ml_data.create_scalar_feature_tensor(
            event_sites, scalar_features, event_site_combs
        )
    )

    site_obs_scalar_feature_ind = np.arange(len(graph_feature_keys["site_obs"]))

    # Create the graph data objects
    graph_data = []
    if n_procs == 1:
        for cur_event, cur_site_combs in tqdm.tqdm(
            event_site_combs.items(), disable=not verbose
        ):
            graph_data.append(
                _get_event_graph_data(
                    cur_event,
                    event_sites[cur_event],
                    cur_site_combs,
                    obs_data,
                    scalar_event_feature_values[cur_event],
                    graph_feature_keys,
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
                        graph_feature_keys,
                        ims,
                    )
                    for cur_event, cur_site_combs in event_site_combs.items()
                ],
            )

    graph_data = [pickle.loads(data) for data in graph_data]
    graph_data = list(itertools.chain(*graph_data))
    return graph_data, site_obs_scalar_feature_ind


def compute_loss(
    pred_ln_im_mean: torch.Tensor,
    target: torch.Tensor,
    pred_ln_im_std: torch.Tensor = None,
    scale_constant: float = 10.0,
    reduction: str = "mean",
):
    """
    Computes either the MSE or the Gaussian NLL loss,
    depending on whether the standard deviation is predicted
    """
    if pred_ln_im_std is not None:
        loss = (
            F.gaussian_nll_loss(
                pred_ln_im_mean,
                target,
                torch.exp(pred_ln_im_std) ** 2,
                reduction=reduction,
            )
            * scale_constant
        )
    else:
        loss = F.mse_loss(pred_ln_im_mean, target, reduction=reduction) * scale_constant

    return loss


def train(
    run_config: RunConfig,
    gnn_model: torch.nn.Module,
    train_loader: gdata.DataLoader,
    val_loader: gdata.DataLoader,
    verbose: bool = True,
):
    """
    Runs the training of a model

    Parameters
    ----------
    run_config: RunConfig
    gnn_model: torch.nn.Module
    train_loader: gloader.DataLoader
        Training graph data loader
    val_loader: gloader.DataLoader
        Validation graph data loader
    verbose: bool, optional
        Whether to print progress information, by default True

    Returns
    -------
    metrics: dict
        Training and validation metric (e.g. loss) history
    best_model_state: dict
        Best model state
    best_model_epoch: int
        Best model epoch
    """
    # Setup metrics to log
    metrics = {
        "loss_hist_train": np.zeros(run_config.n_epochs),
        "loss_hist_val": np.zeros(run_config.n_epochs),
    }

    best_val_loss = np.inf
    best_model_state, best_model_epoch = None, None

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    for cur_epoch_ix in range(run_config.n_epochs):
        if verbose:
            print(f"Epoch: {cur_epoch_ix}")

        ### Training
        n_graphs = 0
        gnn_model.train()
        for cur_batch in tqdm.tqdm(train_loader, disable=not verbose):
            cur_batch = cur_batch.to(run_config.device)
            cur_y = cur_batch.y if cur_batch.y.dim() > 1 else cur_batch.y[:, None]

            optimizer.zero_grad()
            if run_config.pred_std:
                pred_ln_im_mean, pred_ln_im_std = gnn_model(cur_batch)
            else:
                pred_ln_im_mean, pred_ln_im_std = gnn_model(cur_batch), None

            nan_mask = torch.isnan(cur_y)
            loss = compute_loss(
                pred_ln_im_mean[~nan_mask],
                cur_y[~nan_mask],
                pred_ln_im_std=(
                    pred_ln_im_std[~nan_mask] if pred_ln_im_std is not None else None
                ),
            )
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

                if run_config.pred_std:
                    pred_ln_im_mean, pred_ln_im_std = gnn_model(cur_batch)
                else:
                    pred_ln_im_mean, pred_ln_im_std = gnn_model(cur_batch), None

                nan_mask = torch.isnan(cur_y)
                loss = compute_loss(
                    pred_ln_im_mean[~nan_mask],
                    cur_y[~nan_mask],
                    pred_ln_im_std=(
                        pred_ln_im_std[~nan_mask]
                        if pred_ln_im_std is not None
                        else None
                    ),
                )
                metrics["loss_hist_val"][cur_epoch_ix] += loss.item()
                n_graphs += cur_batch.num_graphs

        metrics["loss_hist_val"][cur_epoch_ix] /= n_graphs

        # Keep track of the best model
        if metrics["loss_hist_val"][cur_epoch_ix] < best_val_loss:
            best_val_loss = metrics["loss_hist_val"][cur_epoch_ix]
            best_model_state = gnn_model.state_dict()
            best_model_epoch = cur_epoch_ix

        if verbose:
            print(f"Epoch {cur_epoch_ix}/{run_config.n_epochs}")
            print(
                f"\tTraining"
                f"\t\tLoss: {metrics['loss_hist_train'][cur_epoch_ix]:.4f}"
            )
            print(
                f"\tValidation"
                f"\t\tLoss: {metrics['loss_hist_val'][cur_epoch_ix] :.4f}"
            )

    return metrics, best_model_state, best_model_epoch


def get_predictions(
    run_config: RunConfig,
    gnn_model: torch.nn.Module,
    graph_data: Sequence[gdata.HeteroData],
    verbose: bool = True,
):
    """
    Gets model prediction for the given graph data

    Parameters
    ----------
    run_config: RunConfig
    gnn_model: torch.nn.Module
    graph_data: Sequence[gdata.HeteroData]
    verbose: bool, optional
        Whether to print progress information, by default True

    Returns
    -------
    results: pd.DataFrame
        GM estimation for each scenario
    """
    gnn_model.eval()
    loader = gloader.DataLoader(graph_data, batch_size=128, shuffle=False)

    pred_im_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
    pred_im_std_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_std")
    loss_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "loss")

    results = []
    for cur_batch in tqdm.tqdm(loader, disable=not verbose):
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
            pred_ln_im_mean, pred_im_ln_std = cur_out
            cur_result.loc[:, pred_im_keys] = pred_ln_im_mean.cpu().numpy(force=True)
            cur_result.loc[:, pred_im_std_keys] = (
                torch.exp(pred_im_ln_std).cpu().numpy(force=True)
            )
        else:
            pred_ln_im_mean, pred_im_ln_std = cur_out, None
            cur_result.loc[:, pred_im_keys] = pred_ln_im_mean.cpu().numpy(force=True)

        # Loss
        cur_loss = compute_loss(
            pred_ln_im_mean,
            cur_batch.y,
            pred_ln_im_std=pred_im_ln_std,
            reduction="none",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            cur_result.loc[:, loss_keys] = cur_loss.cpu().numpy(force=True)
            cur_result.loc[:, "loss"] = cur_loss.nanmean(dim=1).cpu().numpy(force=True)

        # Index
        cur_result = cur_result.set_index(
            mlt.array_utils.numpy_str_join(
                "_",
                cur_result["event_id"].values.astype(str),
                cur_result["site_int"].values.astype(str),
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
