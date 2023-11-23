import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, List


import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

import ml_tools as mlt

from . import models


@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    l2_reg: float
    lr: float

    fc_units: List[int]

    @classmethod
    def from_yaml(cls, ffp: Path):
        params = mlt.utils.load_yaml(ffp)

        return cls(
            params["n_epochs"],
            params["batch_size"],
            params["l2_reg"],
            params["lr"],
            params["fc_units"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "fc_units": self.fc_units,
        }


@dataclass
class RunParamsConfig:
    max_dist: float

    debug: bool
    device: str

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml/corr_results"))


class CorrDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, ims: Sequence[str]):
        self.X = X
        self.y = y

        self.site_1 = X.site_1.values
        self.site_2 = X.site_2.values

        self.ims = ims

        self.X = self.X.drop(columns=["site_1", "site_2"])

        assert np.all(self.X.index == self.y.index)

    def __len__(self):
        return len(self.X)


    def get_sites(self, ind: np.ndarray):
        return self.site_1[ind], self.site_2[ind]

    def __getitem__(self, idx: int):
        return idx, self.X.iloc[idx].values, self.y.iloc[idx].values


def get_datasets(X: pd.DataFrame, y: pd.DataFrame, train_sites: np.ndarray, val_sites: np.ndarray, ims: Sequence[str]):
    X_val = X.loc[np.isin(X.site_1, val_sites) & np.isin(X.site_2, val_sites)]
    X_train = X.loc[np.isin(X.site_1, train_sites) & np.isin(X.site_2, train_sites)]

    return CorrDataset(X_train, y.loc[X_train.index], ims), CorrDataset(X_val, y.loc[X_val.index], ims)

def get_dataset_predictions(model: models.MLPModel, dataset: CorrDataset, run_params: RunParamsConfig):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        model.eval()
        for i, (batch_ind, cur_x, cur_y) in enumerate(loader):
            cur_x = cur_x.to(run_params.device, dtype=torch.float32)
            pred = model(cur_x).numpy(force=True)

            site_1, site_2 = dataset.get_sites(batch_ind.numpy(force=True))

            results.append(pd.DataFrame({
                "site_1": site_1,
                "site_2": site_2,
                **{f"pred_{cur_im}": pred[:, ix] for ix, cur_im in enumerate(dataset.ims)},
                **{f"sim_{cur_im}": cur_y[:, ix] for ix, cur_im in enumerate(dataset.ims)},
            }))

    results_df = pd.concat(results, axis=0)
    return results_df

def train(
    model: models.MLPModel,
    train_dataset: CorrDataset,
    val_dataset: CorrDataset,
    hp_config: HyperParamsConfig,
    run_params: RunParamsConfig,
):
    best_model_state = None
    best_val_loss = np.inf
    best_model_epoch = None

    metrics = {
        "train_loss_hist": torch.zeros(hp_config.n_epochs),
        "val_loss_hist": torch.zeros(hp_config.n_epochs),
    }

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp_config.lr, weight_decay=hp_config.l2_reg
    )

    n_workers = 0 if run_params.debug else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp_config.batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp_config.batch_size,
        shuffle=False,
        num_workers=n_workers,
    )

    loss = nn.MSELoss()
    for epoch_ix in range(hp_config.n_epochs):

        ### Training
        model.train()
        iter_loop = tqdm(train_loader, desc=f"Epoch {epoch_ix+1}/{hp_config.n_epochs}")
        for i, (_, cur_x, cur_y) in enumerate(iter_loop):
            cur_x = cur_x.to(run_params.device, dtype=torch.float32)
            cur_y = cur_y.to(run_params.device, dtype=torch.float32)
            pred = model(cur_x)

            cur_loss = loss(pred * 10, cur_y * 10)
            optimizer.zero_grad(set_to_none=True)
            cur_loss.backward()
            optimizer.step()

            metrics["train_loss_hist"][epoch_ix] += cur_loss.item()

            iter_loop.set_postfix({"loss": cur_loss.item()})

        metrics["train_loss_hist"][epoch_ix] /= len(train_loader)

        ### Validation
        with torch.no_grad():
            model.eval()
            for i, (_, cur_x, cur_y) in enumerate(val_loader):
                cur_x = cur_x.to(run_params.device, dtype=torch.float32)
                cur_y = cur_y.to(run_params.device, dtype=torch.float32)
                pred = model(cur_x)

                cur_loss = loss(pred * 10, cur_y * 10)

                metrics["val_loss_hist"][epoch_ix] += cur_loss.item()

            metrics["val_loss_hist"][epoch_ix] /= len(val_loader)

            # Keep track of the best model
            if metrics["val_loss_hist"][epoch_ix] < best_val_loss:
                best_model_state = model.state_dict()
                best_val_loss = metrics["val_loss_hist"][epoch_ix]
                best_model_epoch = epoch_ix

        print(f"Epoch {epoch_ix + 1}/{hp_config.n_epochs}")
        print(
            f"\tTraining - Loss: {metrics['train_loss_hist'][epoch_ix]:.4f}"
        )
        print(
            f"\tValidation - Loss: {metrics['val_loss_hist'][epoch_ix]:.4f}, "
        )

    return best_model_state, best_model_epoch, metrics







