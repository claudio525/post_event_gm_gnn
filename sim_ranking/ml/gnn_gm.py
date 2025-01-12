import time
import os
import itertools
import pickle
import warnings
import multiprocessing as mp
from typing import NamedTuple, Sequence, Callable
from pathlib import Path
from dataclasses import dataclass

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import torch_geometric.data.batch as gbatch
import torch_geometric.utils as gutils
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

import ml_tools as mlt

from . import data as ml_data
from . import gnn_modules
from .. import utils
from .. import constants
from ..data_classes import ObservedData, LBSiteCorrelationData


@dataclass
class RunConfig:
    """Config for specifying run settings"""

    ### General settings
    seed: int

    # IMs to use
    im_set: str

    ### Input settings
    rel_obs_data_ffp: Path
    max_dist: float
    """Maximum distance between site-interest and observation sites"""
    closest_max_dist: float
    """Maximum distance between the site of interest and the closest observation site"""
    max_n_obs_sites: int
    """Maximum number of observation sites to consider"""
    min_n_obs_sites: int
    """Minimum number of observation sites required"""
    ignore_events: Sequence[str]
    """Events to ignore"""

    ### Scenario Weighting
    mag_scenario_weighting: bool
    """Whether to use magnitude scenario weighting"""
    mag_min_weight: float
    """Minimum weight for the magnitude"""
    mag_max_weight: float
    """Maximum weight for the magnitude"""
    mag_start: float
    """Start point for linear weighting of the magnitude"""
    mag_end: float
    """End point for linear weighting of the magnitude"""
    doc_scenario_weighting: bool
    """Whether to use DoC scenario weighting"""
    doc_min_weight: float
    """Minimum weight for the DoC"""
    doc_max_weight: float
    """Maximum weight for the DoC"""
    doc_start: float
    """Start point for linear weighting of the DoC"""
    doc_end: float
    """End point for linear weighting of the DoC"""

    device: str
    """Device to use"""

    ### Model settings
    scale_IMs: bool
    """Whether to scale the IMs"""

    ### Hyperparameters
    n_epochs: int
    """Number of training epochs"""
    batch_size: int
    """Batch size"""

    batch_norm: bool
    """Whether to use batch normalization"""

    fc_n_units: int
    """Number of FC units for the output model"""
    fc_act_fn: str | None
    """Activation function for the FC output model"""

    int_embedding_act_fn: str | None
    """Activation function for the SoI embedding update/transform model"""
    obs_embedding_act_fn: str | None
    """Activation function for the observation embedding update models"""
    edge_embedding_act_fn: str | None
    """Activation function for the edge embedding update model"""
    n_obs_node_channels: Sequence[int]
    """Number of observation site node channels"""
    n_int_node_channels: Sequence[int]
    """Number of site of interest node channels per attention coefficient. 
    Total number of channels = n_int_node_channels * n_attention_heads"""
    n_edge_channels: Sequence[int]
    """Number of edge channels"""
    n_att_heads: Sequence[int]
    """Number of attention heads to use"""
    gcn_act_fn: str | None
    """Activation function following a graph convolution"""

    att_act_fn: str | None
    """Activation function for the attention models"""
    att_n_units: Sequence[int]
    """Number of attention units"""

    l2_reg: float
    """L2 regularization coefficient"""

    rel_results_dir: str
    """Base output directory"""

    dropout_rate: float = 0.0
    """Dropout rate"""

    use_lr_scheduler: bool = False
    """Whether to use a learning rate scheduler"""
    lr_patience: int = 5
    """Number of epochs to wait before reducing the learning rate"""
    lr_factor: float = 0.5
    """Factor to reduce the learning rate by"""

    _im_scale_params: dict[str, pd.Series] | None = None

    ### Features
    graph_feature_keys: dict[str, Sequence[str]] = None

    def __post_init__(self):
        assert self.obs_data_ffp.exists()

        # Handle loading IM scale parameters from a dict
        if self._im_scale_params is not None:
            tmp = {
                cur_key: pd.Series(cur_dict) for cur_key, cur_dict in
                self._im_scale_params.items()
            }
            self._im_scale_params = tmp
        else:
            self.im_scale_params = None

    @property
    def ims(self):
        return constants.IM_SETS[self.im_set]

    @property
    def pSA_ims(self):
        return constants.PSA_KEYS

    @property
    def pSA_periods(self):
        return constants.PERIODS

    @property
    def non_pSA_ims(self):
        return constants.NON_PSA_IMs

    @property
    def pred_std_keys(self):
        return [f"{cur_key}_pred_std" for cur_key in self.ims]

    @property
    def pred_std_keys_pSA(self):
        return [f"{cur_key}_pred_std" for cur_key in self.pSA_ims]

    @property
    def pred_std_keys_non_pSA(self):
        return [f"{cur_key}_pred_std" for cur_key in self.non_pSA_ims]

    @property
    def n_ims(self):
        return len(self.ims)

    @property
    def im_scale_params(self):
        if self.scale_IMs:
            return self._im_scale_params

        raise ValueError("IM standardization is not enabled")

    @im_scale_params.setter
    def im_scale_params(self, value):
        if self.scale_IMs:
            if self._im_scale_params is None:
                self._im_scale_params = value
            else:
                raise ValueError("IM standardization is already set")
        else:
            raise ValueError("IM standardization is not enabled")

    @property
    def n_outputs(self):
        return self.n_ims * 2

    @property
    def site_obs_n_scalar_features(self):
        return (
            0
            if self.graph_feature_keys["site_obs"] is None
            else len(self.graph_feature_keys["site_obs"])
        )

    @property
    def site_obs_n_features(self):
        return self.site_obs_n_scalar_features + self.n_ims

    @property
    def site_int_n_features(self):
        return len(self.graph_feature_keys["site_int"])

    @property
    def n_edge_features(self):
        return len(self.graph_feature_keys["edge"])

    @property
    def wdata(self) -> str:
        return os.path.expandvars("$wdata")

    @property
    def results_dir(self):
        return Path(self.wdata) / self.rel_results_dir

    @property
    def obs_data_ffp(self):
        return Path(self.wdata) / self.rel_obs_data_ffp

    def to_dict(self):
        result = {
            "seed": self.seed,
            "rel_obs_data_ffp": self.rel_obs_data_ffp,
            "max_dist": self.max_dist,
            "closest_max_dist": self.closest_max_dist,
            "max_n_obs_sites": self.max_n_obs_sites,
            "min_n_obs_sites": self.min_n_obs_sites,
            "ignore_events": list(self.ignore_events),
            "mag_scenario_weighting": self.mag_scenario_weighting,
            "mag_min_weight": self.mag_min_weight,
            "mag_max_weight": self.mag_max_weight,
            "mag_start": self.mag_start,
            "mag_end": self.mag_end,
            "doc_scenario_weighting": self.doc_scenario_weighting,
            "doc_min_weight": self.doc_min_weight,
            "doc_max_weight": self.doc_max_weight,
            "doc_start": self.doc_start,
            "doc_end": self.doc_end,
            "device": self.device,
            "im_set": self.im_set,
            "scale_IMs": self.scale_IMs,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "n_obs_node_channels": list(self.n_obs_node_channels),
            "n_int_node_channels": list(self.n_int_node_channels),
            "n_edge_channels": list(self.n_edge_channels),
            "n_att_heads": self.n_att_heads,
            "fc_n_units": self.fc_n_units,
            "batch_norm": self.batch_norm,
            "int_embedding_act_fn": self.int_embedding_act_fn,
            "obs_embedding_act_fn": self.obs_embedding_act_fn,
            "edge_embedding_act_fn": self.edge_embedding_act_fn,
            "att_act_fn": self.att_act_fn,
            "gcn_act_fn": self.gcn_act_fn,
            "fc_act_fn": self.fc_act_fn,
            "att_n_units": list(self.att_n_units),
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
            "rel_results_dir": self.rel_results_dir,
            "use_lr_scheduler": self.use_lr_scheduler,
            "lr_patience": self.lr_patience,
            "lr_factor": self.lr_factor,
            "graph_feature_keys": self.graph_feature_keys,
        }
        if self.im_scale_params is not None:
            result["_im_scale_params"] = {
                cur_key: cur_df.to_dict()
                for cur_key, cur_df in self.im_scale_params.items()
            }
        return result

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
        return  cls(**d)

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
        run_config.max_dist,
        run_config.closest_max_dist,
        run_config.max_n_obs_sites,
        run_config.min_n_obs_sites,
    )
    val_site_combs, val_event_sites = ml_data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        val_events,
        dist_matrix,
        obs_sites,
        val_int_sites,
        run_config.max_dist,
        run_config.closest_max_dist,
        run_config.max_n_obs_sites,
        run_config.min_n_obs_sites,
    )

    # Sanity check
    assert np.isin(val_int_sites, train_int_sites).sum() == 0
    assert np.isin(val_events, train_events).sum() == 0

    if run_config.scale_IMs:
        scale_record_ids = []
        for cur_event in train_events:
            cur_sites = event_sites[cur_event]
            cur_sites = cur_sites[np.isin(cur_sites, train_int_sites)]

            scale_record_ids.append(
                mlt.array_utils.numpy_str_join("_", cur_event, cur_sites)
            )
        scale_record_ids = np.concatenate(scale_record_ids)

        run_config.im_scale_params = {
            "mean": np.log(
                obs_data.record_df.loc[scale_record_ids, run_config.ims]
            ).mean(skipna=True),
            "std": np.log(obs_data.record_df.loc[scale_record_ids, run_config.ims]).std(
                skipna=True
            ),
        }

    if verbose:
        print(f"Getting graph data")
    train_graph_data = get_graph_data(
        obs_data,
        train_event_sites,
        train_site_combs,
        scalar_features,
        run_config.graph_feature_keys,
        run_config,
        n_procs=graph_data_n_procs,
        verbose=verbose,
        dist_matrix=dist_matrix,
    )

    val_graph_data = get_graph_data(
        obs_data,
        val_event_sites,
        val_site_combs,
        scalar_features,
        run_config.graph_feature_keys,
        run_config,
        n_procs=graph_data_n_procs,
        verbose=verbose,
        dist_matrix=dist_matrix,
    )

    train_loader = gloader.DataLoader(
        train_graph_data, batch_size=run_config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = gloader.DataLoader(
        val_graph_data, batch_size=run_config.batch_size, shuffle=True
    )

    gnn_model = gnn_modules.CustomAttentionGNN(
        run_config,
    )
    gnn_model.to(run_config.device)

    if verbose:
        print(f"----------------- Training -----------------")
        print(f"Number of training graphs: {len(train_graph_data)}")
        print(f"Number of validation graphs: {len(val_graph_data)}")

    metrics, best_model_state, best_model_epoch = train(
        run_config, gnn_model, train_loader, val_loader, verbose=verbose
    )

    metrics_df = pd.DataFrame(metrics)
    agg_metrics = {
        "loss_train_best_epoch": metrics_df.loss_hist_train.argmin(),
        "loss_train_min": metrics_df.loss_hist_train.min(),
        "loss_val_best_epoch": metrics_df.loss_hist_val.argmin(),
        "loss_val_min": metrics_df.loss_hist_val.min(),
        "mse_train_best_epoch": metrics_df.mse_hist_train.argmin(),
        "mse_train_min": metrics_df.mse_hist_train.min(),
        "mse_val_best_epoch": metrics_df.mse_hist_val.argmin(),
        "mse_val_min": metrics_df.mse_hist_val.min(),
    }

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

    # Save loss history & aggregate metrics
    metrics_df.to_parquet(out_dir / "metrics.parquet")
    pd.to_pickle(agg_metrics, out_dir / "agg_metrics.pickle")

    # Save the model
    torch.save(gnn_model, out_dir / "model.pt")

    # Save the results
    dist_matrix = utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    train_results_df, train_attn_coeffs_df = get_predictions(
        run_config,
        gnn_model,
        train_graph_data,
        dist_matrix=dist_matrix,
        verbose=verbose,
    )
    train_results_df.to_parquet(out_dir / "train_results.parquet")
    train_attn_coeffs_df.to_parquet(out_dir / "train_attn_coeffs.parquet")

    val_results_df, val_attn_coeffs = get_predictions(
        run_config, gnn_model, val_graph_data, dist_matrix=dist_matrix, verbose=verbose
    )
    val_results_df.to_parquet(out_dir / "val_results.parquet")
    val_attn_coeffs.to_parquet(out_dir / "val_attn_coeffs.parquet")

    # Sanity checks
    assert ~np.any(val_results_df.event_id.isin(train_results_df.event_id))
    assert ~np.any(val_results_df.site_int.isin(train_results_df.site_int))

    # Write the metadata
    metadata = {
        "best_model_epoch": int(best_model_epoch),
        "best_model_loss": float(metrics["loss_hist_val"][best_model_epoch]),
        "n_train_scenarios": len(train_graph_data),
        "n_val_scenarios": len(val_graph_data),
        "n_model_parameters": int(gnn_model.n_train_params),
    }
    mlt.utils.write_to_yaml(metadata, out_dir / "metadata.yaml")


def _get_event_graph_data(
    event: str,
    sites: np.ndarray,
    site_combs: np.ndarray,
    obs_data: ObservedData,
    scalar_feature_df: pd.DataFrame,
    graph_feature_keys: dict[str, Sequence[str]],
    run_config: RunConfig,
    corr_data: LBSiteCorrelationData = None,
):
    """Helper function, see get_graph_data"""
    graph_data = []
    cur_site_int_inds = np.unique(site_combs[:, 0])

    ### Weighting functions
    # Have to get these here, as they are not picklable
    # Magnitude weighting function
    mag_weight_fn = (
        get_mag_weight_func(
            run_config.mag_min_weight,
            run_config.mag_max_weight,
            run_config.mag_start,
            run_config.mag_end,
        )
        if run_config.mag_scenario_weighting
        else None
    )
    # Degree of Constraint weighting function
    doc_weight_fn = None
    if run_config.doc_scenario_weighting:
        doc_weight_fn = get_doc_weight_func(
            run_config.doc_min_weight,
            run_config.doc_max_weight,
            run_config.doc_start,
            run_config.doc_end,
        )

    # Get magnitude weight
    assert (not run_config.mag_scenario_weighting) or (mag_weight_fn is not None)
    cur_mag_weight = (
        mag_weight_fn(obs_data.event_df.loc[event, "mag"])
        if run_config.mag_scenario_weighting
        else None
    )

    # Get and normalise the IM data
    cur_im_data = np.log(obs_data.get_event_data(event, sites).loc[:, run_config.ims])
    if run_config.scale_IMs:
        cur_im_data = (
            cur_im_data - run_config.im_scale_params["mean"][run_config.ims]
        ) / run_config.im_scale_params["std"][run_config.ims]

    for cur_site_int_ix in cur_site_int_inds:
        cur_site_combs_mask = site_combs[:, 0] == cur_site_int_ix
        cur_site_int = sites[site_combs[cur_site_combs_mask, 0][0]]
        cur_obs_sites = sites[site_combs[cur_site_combs_mask, 1]]

        # DoC scenario weighting
        doc_weight = doc_weight_fn(
            corr_data.corr_data.sel[cur_site_int, :, :]
            .loc[cur_obs_sites, constants.PSA_KEYS]
            .sum(axis=0)
            .mean()
            if run_config.doc_scenario_weighting
            else None
        )

        # Create the site_int node features
        cur_site_int_features = scalar_feature_df.loc[
            cur_site_combs_mask, graph_feature_keys["site_int"]
        ].values[0]

        # Get observation site IM values and deal with nan values
        cur_obs_sites_im_values = (
            cur_im_data.loc[cur_obs_sites, run_config.ims].replace(np.nan, 99).values
        )
        if (
            graph_feature_keys["site_obs"] is not None
            and len(graph_feature_keys["site_obs"]) > 0
        ):
            # Create the site_obs node features
            cur_obs_sites_features = scalar_feature_df.loc[
                cur_site_combs_mask,
                graph_feature_keys["site_obs"],
            ].values
            # Add the IM values
            cur_obs_sites_features = np.concatenate(
                (
                    cur_obs_sites_features,
                    cur_obs_sites_im_values,
                ),
                axis=1,
            )
        else:
            cur_obs_sites_features = cur_obs_sites_im_values

        # Create the edge features
        cur_edge_features = scalar_feature_df.loc[
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

        cur_sc_data["site_obs", "self_loop", "site_obs"].edge_index = torch.tensor(
            [[ix, ix] for ix in range(len(cur_obs_sites))], dtype=torch.long
        ).T

        cur_sc_data["metadata"] = {
            "sc_id": f"{event}_{cur_site_int}",
            "event": event,
            "site_int": cur_site_int,
            "obs_sites": cur_obs_sites,
        }

        cur_sc_data["y"] = torch.tensor(
            cur_im_data.loc[cur_site_int, run_config.ims].values, dtype=torch.float32
        )[None, :]

        cur_sc_weight = torch.tensor(1.0)
        if cur_mag_weight is not None:
            cur_sc_data["metadata"]["mag_weight"] = cur_mag_weight
            cur_sc_weight += cur_mag_weight
        if doc_weight is not None:
            cur_sc_data["metadata"]["doc_weight"] = doc_weight
            cur_sc_weight += doc_weight
        cur_sc_data["sc_weight"] = cur_sc_weight

        graph_data.append(cur_sc_data)

    return pickle.dumps(graph_data)


def get_graph_data(
    obs_data: ObservedData,
    event_sites: dict[str, np.ndarray],
    event_site_combs: dict[str, np.ndarray],
    scalar_features: ml_data.ScalarFeatures,
    graph_feature_keys: dict[str, Sequence[str]],
    run_config: RunConfig,
    n_procs: int = 1,
    verbose: bool = True,
    dist_matrix: pd.DataFrame = None,
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
    dist_matrix: pd.DataFrame, optional
        Distance matrix, by default None
        Required if DoC scenario weighting is used

    Returns
    -------
    graph_data: list[gdata.HeteroData]
    """
    # Create the scalar features tensors
    scalar_event_feature_values, scalar_feature_columns = (
        ml_data.create_event_scalar_feature_dfs(
            event_sites, scalar_features, event_site_combs
        )
    )

    # Site2Site Correlation for DoC weight calculation
    assert (not run_config.doc_scenario_weighting) or (dist_matrix is not None)
    if run_config.doc_scenario_weighting:
        corr_data = LBSiteCorrelationData.from_dist_matrix(
            dist_matrix, constants.PSA_KEYS
        )

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
                    run_config,
                    corr_data=corr_data,
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
                        run_config,
                        corr_data,
                    )
                    for cur_event, cur_site_combs in event_site_combs.items()
                ],
            )

    graph_data = [pickle.loads(data) for data in graph_data]
    graph_data = list(itertools.chain(*graph_data))
    return graph_data


def compute_loss(
    pred_ln_im_mean: torch.Tensor,
    target: torch.Tensor,
    pred_ln_im_ln_std: torch.Tensor = None,
    reduction: str = "mean",
):
    """
    Computes either the MSE or the Gaussian NLL loss,
    depending on whether the standard deviation is predicted
    """
    loss = F.gaussian_nll_loss(
        pred_ln_im_mean,
        target,
        torch.exp(pred_ln_im_ln_std) ** 2,
        reduction=reduction,
    )

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
        "w_loss_hist_train": np.zeros(run_config.n_epochs),
        "loss_hist_val": np.zeros(run_config.n_epochs),
        "w_loss_hist_val": np.zeros(run_config.n_epochs),
        "mse_hist_train": np.zeros(run_config.n_epochs),
        "mse_hist_val": np.zeros(run_config.n_epochs),
        "mean_sigma_hist_train": np.zeros(run_config.n_epochs),
        "mean_sigma_hist_val": np.zeros(run_config.n_epochs),
    }

    best_val_loss = np.inf
    best_model_state, best_model_epoch = None, None

    optimizer = torch.optim.Adam(
        gnn_model.parameters(), lr=0.001, weight_decay=run_config.l2_reg
    )
    scheduler = (
        lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config.lr_factor,
            patience=run_config.lr_patience,
        )
        if run_config.use_lr_scheduler
        else None
    )
    for cur_epoch_ix in range(run_config.n_epochs):
        if verbose:
            print(f"Epoch: {cur_epoch_ix + 1}/{run_config.n_epochs}")

        ### Training
        n_graphs = 0
        gnn_model.train()
        for cur_batch in tqdm.tqdm(train_loader, disable=not verbose):
            optimizer.zero_grad()

            cur_bresult = _get_batch_result(cur_batch, gnn_model, run_config)

            cur_bresult.w_loss.backward()
            optimizer.step()

            metrics = _save_metrics(
                cur_bresult, metrics, run_config, cur_epoch_ix, "train"
            )
            n_graphs += cur_batch.num_graphs

        metrics["loss_hist_train"][cur_epoch_ix] /= n_graphs
        metrics["w_loss_hist_train"][cur_epoch_ix] /= n_graphs
        metrics["mse_hist_train"][cur_epoch_ix] /= n_graphs
        metrics["mean_sigma_hist_train"][cur_epoch_ix] /= n_graphs

        ### Validation
        gnn_model.eval()
        n_graphs = 0
        with torch.no_grad():
            for cur_batch in val_loader:
                cur_bresult = _get_batch_result(cur_batch, gnn_model, run_config)

                metrics = _save_metrics(
                    cur_bresult, metrics, run_config, cur_epoch_ix, "val"
                )
                n_graphs += cur_batch.num_graphs

        metrics["loss_hist_val"][cur_epoch_ix] /= n_graphs
        metrics["w_loss_hist_val"][cur_epoch_ix] /= n_graphs
        metrics["mse_hist_val"][cur_epoch_ix] /= n_graphs
        metrics["mean_sigma_hist_val"][cur_epoch_ix] /= n_graphs

        if scheduler is not None:
            scheduler.step(metrics["w_loss_hist_val"][cur_epoch_ix])

        # Keep track of the best model
        if metrics["w_loss_hist_val"][cur_epoch_ix] < best_val_loss:
            best_val_loss = metrics["w_loss_hist_val"][cur_epoch_ix]
            best_model_state = gnn_model.state_dict()
            best_model_epoch = cur_epoch_ix

        if verbose:
            # print(f"Epoch {cur_epoch_ix}/{run_config.n_epochs}")
            print(
                f"\tTraining\t\t"
                f"Weighted Loss: {metrics['w_loss_hist_train'][cur_epoch_ix]:.4f}, "
                f"Loss: {metrics['loss_hist_train'][cur_epoch_ix]:.4f}, "
                f"MSE: {metrics['mse_hist_train'][cur_epoch_ix]:.5f}"
            )
            print(
                f"\tValidation\t\t"
                f"Weighted Loss: {metrics['w_loss_hist_val'][cur_epoch_ix]:.4f}, "
                f"Loss: {metrics['loss_hist_val'][cur_epoch_ix] :.4f}, "
                f"MSE: {metrics['mse_hist_val'][cur_epoch_ix]:.5f}"
            )

    return metrics, best_model_state, best_model_epoch


class BatchResult(NamedTuple):
    batch: gdata.Batch
    """The batch data"""

    y: torch.Tensor
    """The target IM values"""
    pred_ln_im_mean: torch.Tensor
    """The predicted lnIM mean values"""
    pred_ln_im_ln_std: torch.Tensor
    """The predicted lnIM standard deviation values in logspace"""
    pred_ln_im_std: torch.Tensor
    """The predicted lnIM standard deviation values"""

    nan_mask: torch.Tensor
    """Mask for nan values in y"""
    loss: torch.Tensor
    """The batch loss"""
    w_loss: torch.Tensor
    """The weighted batch loss"""
    ind_loss: torch.Tensor
    """The individual losses, this includes nan-values"""
    w_ind_loss: torch.Tensor
    """The weighted individual losses, this includes nan-values"""


def _save_metrics(
    batch_result: BatchResult,
    metrics: dict[str, np.ndarray[float]],
    run_config: RunConfig,
    epoch_ix: int,
    result_type: str,
):
    """Computes and saves the metrics for a single batch"""
    loss_hist_key = f"loss_hist_{result_type}"
    w_loss_hist_key = f"w_loss_hist_{result_type}"
    mse_hist_key = f"mse_hist_{result_type}"
    mean_sigma_hist_key = f"mean_sigma_hist_{result_type}"

    # Save metrics
    metrics[loss_hist_key][epoch_ix] += (
        batch_result.ind_loss.nanmean(dim=1).sum().item()
    )
    metrics[w_loss_hist_key][epoch_ix] += (
        batch_result.w_ind_loss.nanmean(dim=1).sum().item()
    )
    metrics[mse_hist_key][epoch_ix] += (
        F.mse_loss(
            batch_result.pred_ln_im_mean,
            batch_result.y,
            reduction="none",
        )
        .nanmean(dim=1)
        .sum()
        .item()
    )
    metrics[mean_sigma_hist_key][epoch_ix] += (
        batch_result.pred_ln_im_std.mean(dim=1).sum().item()
    )

    return metrics


def _get_batch_result(batch: gdata.Batch, gnn_model: nn.Module, run_config: RunConfig):
    """
    Gets the result for a single batch

    Parameters
    ----------
    batch: gdata.Batch
    gnn_model: nn.Module
    run_config: RunConfig

    Returns
    -------
    result: BatchResult
        The batch results
    """
    cur_batch = batch.to(run_config.device)
    cur_y = cur_batch.y if cur_batch.y.dim() > 1 else cur_batch.y[:, None]
    cur_sc_weights = einops.repeat(cur_batch.sc_weight, "i -> i j", j=run_config.n_ims)

    pred_ln_im_mean, pred_ln_im_ln_std = gnn_model(cur_batch)

    nan_mask = torch.isnan(cur_y)

    ind_loss = torch.full_like(cur_y, torch.nan)
    ind_loss_ravel = compute_loss(
        pred_ln_im_mean[~nan_mask],
        cur_y[~nan_mask],
        pred_ln_im_ln_std=pred_ln_im_ln_std[~nan_mask],
        reduction="none",
    )
    ind_loss[~nan_mask] = ind_loss_ravel
    loss = ind_loss_ravel.mean()

    w_ind_loss = ind_loss
    if run_config.mag_scenario_weighting or run_config.doc_scenario_weighting:
        w_ind_loss = ind_loss * cur_sc_weights

    w_loss = (cur_sc_weights[~nan_mask] * ind_loss_ravel).mean()

    return BatchResult(
        batch=batch,
        y=cur_y,
        pred_ln_im_mean=pred_ln_im_mean,
        pred_ln_im_ln_std=pred_ln_im_ln_std,
        pred_ln_im_std=(
            torch.exp(pred_ln_im_ln_std) if pred_ln_im_ln_std is not None else None
        ),
        nan_mask=nan_mask,
        loss=loss,
        w_loss=w_loss,
        ind_loss=ind_loss,
        w_ind_loss=w_ind_loss,
    )


def revert_im_scaling(
    scaled_ln_im_mean: np.ndarray[float],
    run_config: RunConfig,
    scaled_ln_im_std: np.ndarray[float] = None,
):
    """
    Reverts the IM scaling

    Parameters
    ----------
    scaled_ln_im_mean: np.ndarray[float]
        The scaled IM (mean) values
    run_config: RunConfig
    scaled_ln_im_std: np.ndarray[float], optional
        The scaled IM standard deviation values

    Returns
    -------
    ln_im_mean: np.ndarray[float]
        The unscaled IM (mean) values
    ln_im_std: np.ndarray[float]
        The unscaled IM standard deviation values.
        Only returned if scaled_ln_im_std is not None.
    """
    ln_im_mean = (
        scaled_ln_im_mean
        * run_config.im_scale_params["std"][run_config.ims].values[None, :]
        + +run_config.im_scale_params["mean"][run_config.ims].values[None, :]
    )

    if scaled_ln_im_std is not None:
        ln_im_std = (
            scaled_ln_im_std
            * run_config.im_scale_params["std"][run_config.ims].values[None, :]
        )
        return ln_im_mean, ln_im_std

    return ln_im_mean, None


def get_predictions(
    run_config: RunConfig,
    gnn_model: gnn_modules.CustomAttentionGNN,
    graph_data: Sequence[gdata.HeteroData],
    dist_matrix: pd.DataFrame = None,
    verbose: bool = True,
):
    """
    Gets model prediction for the given graph data

    Parameters
    ----------
    run_config: RunConfig
    gnn_model: torch.nn.Module
    graph_data: Sequence[gdata.HeteroData]
    dist_matrix: pd.DataFrame, optional
        Distance matrix, by default None
        Allows for computation of the closest observation site
    verbose: bool, optional
        Whether to print progress information, by default True

    Returns
    -------
    results: pd.DataFrame
        GM estimation for each scenario
    """
    gnn_model.eval()
    loader = gloader.DataLoader(graph_data, batch_size=1024, shuffle=False)

    pred_im_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
    pred_im_std_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred_std")
    loss_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "loss")
    w_loss_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "w_loss")

    results = []
    attn_coeffs = []
    for cur_batch in tqdm.tqdm(loader, disable=not verbose):
        cur_batch = cur_batch.to(run_config.device)
        cur_out = gnn_model(cur_batch)

        cur_attn_coeffs_df = gnn_model.get_attention_coeff(cur_batch)

        cur_result = pd.DataFrame(
            data=[
                cur_batch["metadata"]["event"],
                cur_batch["metadata"]["site_int"],
                cur_batch["metadata"]["obs_sites"],
            ],
            index=["event_id", "site_int", "obs_sites"],
        ).T
        if run_config.mag_scenario_weighting:
            cur_result.loc[:, "mag_weight"] = (
                cur_batch["metadata"]["mag_weight"].cpu().numpy(force=True)
            )
        if run_config.doc_scenario_weighting:
            if __name__ == "__main__":
                cur_result.loc[:, "doc_weight"] = (
                    cur_batch["metadata"]["doc_weight"].cpu().numpy(force=True)
                )

        ## Add observed
        obs_ims = cur_batch["y"].cpu().numpy(force=True)
        if run_config.scale_IMs:
            obs_ims, _ = revert_im_scaling(obs_ims, run_config)
        cur_result.loc[:, run_config.ims] = obs_ims

        ## Add predicted
        # Get the predicted mean and standard deviation
        torch_pred_ln_im_mean, torch_pred_ln_im_ln_std = cur_out
        pred_ln_im_std = torch.exp(torch_pred_ln_im_ln_std).cpu().numpy(force=True)
        pred_ln_im_mean = torch_pred_ln_im_mean.cpu().numpy(force=True)

        # Revert the scaling
        if run_config.scale_IMs:
            pred_ln_im_mean, pred_ln_im_std = revert_im_scaling(
                pred_ln_im_mean, run_config, pred_ln_im_std
            )

        # Save
        cur_result.loc[:, pred_im_keys] = pred_ln_im_mean
        cur_result.loc[:, pred_im_std_keys] = pred_ln_im_std

        cur_batch_result = _get_batch_result(cur_batch, gnn_model, run_config)

        # Loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            cur_result.loc[:, loss_keys] = cur_batch_result.ind_loss.cpu().numpy(
                force=True
            )
            cur_result.loc[:, "loss"] = cur_batch_result.loss.cpu().numpy(force=True)
            cur_result.loc[:, w_loss_keys] = cur_batch_result.w_ind_loss.cpu().numpy(
                force=True
            )
            cur_result.loc[:, "w_loss"] = cur_batch_result.w_loss.cpu().numpy(
                force=True
            )

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

        # Closest observation site distance
        cur_result.loc[:, "closest_dist"] = [
            dist_matrix.loc[cur_row.site_int, cur_row.obs_sites].min()
            for cur_key, cur_row in cur_result.iterrows()
        ]

        attn_coeffs.append(cur_attn_coeffs_df)
        results.append(cur_result)

    attn_coeffs_df = pd.concat(attn_coeffs, axis=0)
    results = pd.concat(results, axis=0)
    return results, attn_coeffs_df


def get_residuals(
    results: pd.DataFrame,
    ims: Sequence[str] = constants.PSA_KEYS,
    pred_suffix: str = "pred",
):
    """Computes the residual between the observed and predicted IMs for each scenario"""
    pred_im_keys = mlt.array_utils.numpy_str_join("_", ims, pred_suffix)
    res_df = pd.DataFrame(
        data=results.loc[:, ims].values - results.loc[:, pred_im_keys].values,
        columns=ims,
    )

    res_df.index = results.index
    res_df["event_id"] = results["event_id"]
    res_df["site_int"] = results["site_int"]
    res_df["n_obs_sites"] = results["n_obs_sites"]
    return res_df


def get_mag_weight_func(
    min_weight: float, max_weight: float, mag_start: float, mag_end: float
):
    """
    Creates a magnitude weighting function that is linear
    between the given magnitude values, and otherwise
    constant.

    Function is defined as:
    w(m) = min_weight, m <= mag_start
    w(m) = max_weight, m >= mag_end
    w(m) = grad*m + bias, mag_start < m < mag_end

    Parameters
    ----------
    min_weight: float
        Minimum weight
    max_weight: float
        Maximum weight
    mag_start: float
        Start of the linear weighting
    mag_end: float
        End of the linear weighting

    Returns
    -------
    mag_weight_func: Callable[float, float]
        The magnitude weighting function
    """
    grad = (max_weight - min_weight) / (mag_end - mag_start)
    bias = min_weight - grad * mag_start

    # def mag_weight_func(mag_values: np.ndarray[float]):
    #     weights = np.ones_like(mag_values) * min_weight
    #     weights[mag_values > mag_end] = max_weight
    #
    #     weights = np.where((mag_values > mag_start) & (mag_values <= mag_end), (grad*mag_values + bias), weights)
    #
    #     return weights

    def mag_weight_func(mag: float) -> float:
        if mag <= mag_start:
            return min_weight
        elif mag >= mag_end:
            return max_weight
        else:
            return grad * mag + bias

    return mag_weight_func


def get_doc_weight_func(
    min_weight: float, max_weight: float, doc_start: float, doc_end: float
):
    """
    Creates a degree of constraint (DoC) weighting function that is linear
    between the specified DoC values, and otherwise constant.

    Parameters
    ----------
    min_weight: float
        Minimum weight
    max_weight: float
        Maximum weight
    doc_start: float
        Start of the linear weighting
    doc_end: float
        End of the linear weighting

    Returns
    -------
    doc_weight_fn: Callable[float, float]
        The DoC weighting function
    """
    grad = (max_weight - min_weight) / (doc_end - doc_start)
    bias = max_weight - (grad * doc_end)

    def doc_weight_fn(doc: float) -> float:
        if doc < doc_start:
            return min_weight
        elif doc > doc_end:
            return max_weight
        else:
            return grad * doc + bias

    return doc_weight_fn
