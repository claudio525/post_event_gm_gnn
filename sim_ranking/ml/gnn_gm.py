import hashlib
import base64
from typing import NamedTuple, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import tqdm

import ml_tools as mlt

from ..db import DB
from . import data as ml_data


class RunConfig(NamedTuple):

    # Number of training epochs
    n_epochs: int
    # Batch size
    batch_size: int
    # Number of validation events
    n_val_events: int
    # Number of validation sites
    n_val_sites: int
    # Device to use
    device: str
    # IMs to predict
    ims: Sequence[str]

    results_dir: Path

    @property
    def n_ims(self):
        return len(self.ims)

    def to_dict(self):
        return {
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "n_val_events": int(self.n_val_events),
            "n_val_sites": int(self.n_val_sites),
            "device": self.device,
            "ims": list(self.ims),
            "results_dir": str(
                self.results_dir,
            ),
        }

    def to_yaml(self, ffp: Path):
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    @classmethod
    def from_dict(cls, d: dict):
        d["results_dir"] = Path(d["results_dir"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))


def get_graph_data(
    db: DB,
    event_sites: dict[str, np.ndarray],
    event_site_combs: dict[str, np.ndarray],
    scalar_features: ml_data.ScalarFeatures,
    site_int_feature_keys: list[str],
    site_obs_scalar_feature_keys: list[str],
    edge_feature_keys: list[str],
    ims_mean: pd.DataFrame,
    ims_std: pd.DataFrame,
    ims: Sequence[str],
    # site_int_site_feature_keys: list[str] = None,
    # site_obs_site_feature_keys: list[str] = None,
):
    # Either both or neither of site_int_site_features
    # and site_obs_site_features should be specified
    # assert (site_int_site_feature_keys is None and site_obs_site_feature_keys is None) or (
    #         site_int_site_feature_keys and site_obs_site_feature_keys
    # )

    scalar_event_feature_values, scalar_feature_columns = (
        ml_data.create_scalar_feature_tensor(
            event_sites, scalar_features, event_site_combs
        )
    )

    # site_int_site_feature_ind = None
    # site_obs_site_feature_ind = None
    # if site_int_site_feature_keys is not None:
    #     site_int_site_feature_ind = [site_int_feature_keys.index(cur_site_feature) for cur_site_feature in site_int_site_feature_keys]
    #     site_obs_site_feature_ind = [site_obs_scalar_feature_keys.index(cur_site_feature) for cur_site_feature in site_obs_site_feature_keys]
    n_site_obs_scalar_features = len(site_obs_scalar_feature_keys)
    site_obs_scalar_feature_ind = np.arange(n_site_obs_scalar_features)

    # Create the graph data objects
    graph_data = []
    for cur_event, cur_site_combs in tqdm.tqdm(event_site_combs.items()):
        cur_sites = event_sites[cur_event]
        cur_site_int_inds = np.unique(cur_site_combs[:, 0])

        cur_scalar_feature_values = scalar_event_feature_values[cur_event]

        # Get and normalise the IM data
        cur_im_data = db.get_obs_data(cur_event, cur_sites)
        cur_im_data = np.log(cur_im_data.loc[:, ims])
        # cur_norm_im_data = (cur_im_data - ims_mean) / ims_std

        for cur_site_int_ix in cur_site_int_inds:
            cur_site_combs_mask = cur_site_combs[:, 0] == cur_site_int_ix
            cur_site_int = cur_sites[cur_site_combs[cur_site_combs_mask, 0][0]]
            cur_obs_sites = cur_sites[cur_site_combs[cur_site_combs_mask, 1]]

            # Create the site_int node features
            cur_site_int_features = cur_scalar_feature_values.loc[
                cur_site_combs_mask, site_int_feature_keys
            ].values[0]

            # Create the site_obs node features
            cur_obs_sites_features = cur_scalar_feature_values.loc[
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
            cur_edge_features = cur_scalar_feature_values.loc[
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

            # cur_sc_data["site_obs"].scalar_feature_ind = torch.tensor(
            #     site_obs_scalar_feature_ind, dtype=torch.int
            # )[None, :]
            # cur_sc_data["site_int"].scalar_feature_ind = torch.tensor(
            #     np.arange(len(site_int_feature_keys)), dtype=torch.int
            # )[None, :]
            #
            # cur_sc_data["site_obs"].im_ind = torch.tensor(
            #     np.arange(
            #         n_site_obs_scalar_features + 1,
            #         n_site_obs_scalar_features + len(ims),
            #     ),
            #     dtype=torch.int,
            # )[None, :]

            # Indices for the site-specific features
            # if site_int_site_feature_ind is not None:
            #     cur_sc_data["site_int"].site_feature_ind = torch.tensor(
            #         site_int_site_feature_ind, dtype=torch.int16
            #     )
            #     cur_sc_data["site_obs"].site_feature_ind = torch.tensor(
            #         site_obs_site_feature_ind, dtype=torch.int16
            #     )

            cur_sc_data["metadata"] = {
                "event": cur_event,
                "site_int": cur_site_int,
                "obs_sites": cur_obs_sites,
            }

            cur_sc_data["y"] = torch.tensor(
                cur_im_data.loc[cur_site_int, ims].values, dtype=torch.float32
            )[None, :]

            graph_data.append(cur_sc_data)

    return graph_data, site_obs_scalar_feature_ind


def train(
    run_config: RunConfig,
    gnn_model: torch.nn.Module,
    train_loader: gdata.DataLoader,
    val_loader: gdata.DataLoader,
):
    # Setup metrics to log
    metrics = {
        "loss_hist_train": torch.zeros(run_config.n_epochs),
        "loss_hist_val": torch.zeros(run_config.n_epochs),
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
            loss = torch.nn.functional.mse_loss(out, cur_y)
            loss.backward()
            optimizer.step()

            metrics["loss_hist_train"][cur_epoch_ix] += loss.item()
            n_graphs += cur_batch.num_graphs

        metrics["loss_hist_train"][cur_epoch_ix] /= n_graphs

        ### Validation
        gnn_model.eval()
        n_graphs = 0
        for cur_batch in val_loader:
            cur_batch = cur_batch.to(run_config.device)
            cur_y = cur_batch.y if cur_batch.y.dim() > 1 else cur_batch.y[:, None]

            out = gnn_model(cur_batch)
            loss = torch.nn.functional.mse_loss(out, cur_y)

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
    loader = gloader.DataLoader(graph_data, batch_size=128, shuffle=False)

    pred_im_keys = mlt.array_utils.numpy_str_join("_", run_config.ims, "pred")
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
        cur_result.loc[:, pred_im_keys] = cur_out.cpu().numpy(force=True)

        cur_loss = (
            F.mse_loss(cur_out, cur_batch.y, reduction="none").cpu().numpy(force=True)
        )
        cur_result.loc[:, loss_keys] = cur_loss
        cur_result.loc[:, "loss"] = cur_loss.mean(axis=1)

        cur_obs_site_hash_values = [
            base64.urlsafe_b64encode(
                hashlib.sha256("_".join(list(cur_site_obs)).encode()).digest()
            )
            .rstrip(b"=")
            .decode("utf-8")[:10]
            for cur_site_obs in cur_result["obs_sites"]
        ]

        cur_result = cur_result.set_index(
            mlt.array_utils.numpy_str_join(
                "_",
                cur_result["event_id"].values.astype(str),
                cur_result["site_int"].values.astype(str),
                cur_obs_site_hash_values,
            )
        )

        # Add column for number of observation sites

        results.append(cur_result)

    results = pd.concat(results, axis=0)
    return results
