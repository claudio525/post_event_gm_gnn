import os
import tempfile as tmp
from pathlib import Path
from typing import List, Dict, Sequence, Union
from dataclasses import dataclass

import einops
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchinfo import summary
from torchview import draw_graph

import ml_tools as mlt
import spatial_hazard as sh

from . import data as ml_data
from . import features
from . import models
from . import prob
from ..db import DB
from .. import constants


@dataclass
class HyperParamsConfig:
    n_epochs: int
    batch_size: int
    l2_reg: float
    lr: Sequence[float]
    lr_epochs: Sequence[int]

    # lr: float

    misfit_fn: str

    fc_units: List[int]

    use_im_sim_site_obs: bool
    use_im_sim_site_int: bool
    use_im_obs_site_obs: bool

    use_res_site_obs: bool
    use_res_sim_site_obs_sim_site_int: bool
    use_res_obs_site_obs_sim_site_int: bool

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
            params["fc_units"],
            params["im_sim_site_obs"],
            params["im_sim_site_int"],
            params["im_obs_site_obs"],
            params["res_site_obs"],
            params["res_sim_site_obs_sim_site_int"],
            params["res_obs_site_obs_sim_site_int"],
        )

    def to_dict(self):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "lr_epochs": self.lr_epochs,
            "misfit_fn": self.misfit_fn,
            "fc_units": self.fc_units,
            "n_im_features": self.n_im_features,
            "im_sim_site_obs": self.use_im_sim_site_obs,
            "im_sim_site_int": self.use_im_sim_site_int,
            "im_obs_site_obs": self.use_im_obs_site_obs,
            "res_site_obs": self.use_res_site_obs,
            "res_sim_site_obs_sim_site_int": self.use_res_sim_site_obs_sim_site_int,
            "res_obs_site_obs_sim_site_int": self.use_res_obs_site_obs_sim_site_int,
        }


def data_prep(
    event_sites: Dict[str, np.ndarray],
    train_events: np.ndarray,
    val_events: np.ndarray,
    train_sites: np.ndarray,
    val_int_sites: np.ndarray,
    events: np.ndarray,
    run_config: prob.RunParamsConfig,
    hp_config: HyperParamsConfig,
    db: DB,
    corr_dir: Path,
):
    # Scalar features
    # EVENT_FEATURE_KEYS = ["mag"]
    # SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    # SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    # EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    # EVENT_SITE_TO_SITE_FEATURE_KEYS = ["angular_dist"]

    EVENT_FEATURE_KEYS = ["mag"]
    SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5"]
    SITE_TO_SITE_FEATURE_KEYS = ["dist"]
    EVENT_SITE_FEATURE_KEYS = ["r_rup"]
    EVENT_SITE_TO_SITE_FEATURE_KEYS = []

    event_df = db.get_event_df()
    record_df = db.get_record_df()

    print(f"Computing distance matrix")
    station_df = db.get_site_df()
    all_sites = db.get_avail_sites()
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    ### Scalar Features
    # Run pre-processing for the site features
    # TODO: This should be updated such that the normalisation
    # only happens on training sites, not all sites
    print(f"Pre-processing site & event features")
    site_features_df, site_feature_stats = features.preprocess_site_features(
        station_df, SITE_FEATURE_KEYS
    )
    event_features_df = event_df.loc[events, EVENT_FEATURE_KEYS]
    event_features_df[EVENT_FEATURE_KEYS] = (
        event_df.loc[events, EVENT_FEATURE_KEYS]
        - event_df.loc[events, EVENT_FEATURE_KEYS].mean()
    ) / event_df.loc[events, EVENT_FEATURE_KEYS].std()

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
        event_features_df,
        EVENT_FEATURE_KEYS,
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

    # Create the datasets
    print(f"Creating datasets")
    train_dataset = SCProbDataset(
        train_event_sites,
        train_site_combs,
        db,
        run_config,
        scalar_features,
        ims_mean,
        ims_std,
        hp_config,
        corr_dir,
    )

    val_dataset = SCProbDataset(
        val_event_sites,
        val_site_combs,
        db,
        run_config,
        scalar_features,
        ims_mean,
        ims_std,
        hp_config,
        corr_dir,
    )

    metadata = {
        "train_sites": train_sites.tolist(),
        "val_int_sites": val_int_sites.tolist(),
        "train_events": train_events.tolist(),
        "val_events": val_events.tolist(),
        "n_train_scenarios": len(train_dataset),
        "n_val_scenarios": len(val_dataset),
        "max_dist": run_config.max_dist,
        "n_rels": run_config.n_rels,
        "features": {
            "event_features": scalar_features.event_feature_keys,
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

    print(f"Number of training samples (scenarios): {len(train_dataset)}")
    print(f"Number of validation samples (scenarios) : {len(val_dataset)}")

    return train_dataset, val_dataset, scalar_features, metadata

    ## Sanity checking
    # sim_df = db.get_sim_df(log=True)
    # obs_df = db.get_obs_df(log=True)
    # (
    #     batch_ind,
    #     scenario_ids,
    #     record_scenario_ids,
    #     _,
    #     __,
    #     __,
    #     site_int_sim,
    #     site_obs_sim,
    #     site_obs_obs,
    #     scalar_features,
    #     im_misfit,
    #     im_site_corrs,
    # ) = train_dataset.get_batch(np.arange(len(train_dataset), dtype=int))
    #
    # events, record_events, site_int, site_obs, rels, residuals = train_dataset.get_metadata(batch_ind)
    # for ix, (cur_event, cur_site_int, cur_site_obs) in enumerate(tqdm(zip(record_events, site_int, site_obs))):
    #     if ix % 1000 == 0:
    #         print(f"Checking {ix} of {len(record_events)}")
    #     cur_int_sims = sim_df.loc[
    #         (sim_df.event_id == cur_event) & (sim_df.site_id == cur_site_int)]
    #     cur_int_sims = cur_int_sims.set_index("rel_id")[constants.PSA_KEYS]
    #     assert np.allclose(site_int_sim[ix, :, :], cur_int_sims.loc[rels[0]].values)
    #
    #     cur_obs_sims = sim_df.loc[
    #         (sim_df.event_id == cur_event) & (sim_df.site_id == cur_site_obs)]
    #     cur_obs_sims = cur_obs_sims.set_index("rel_id")[constants.PSA_KEYS]
    #     assert np.allclose(site_obs_sim[ix, :, :], cur_obs_sims.loc[rels[0]].values)
    #
    #     cur_site_obs_obs = obs_df.loc[(obs_data.event_id == cur_event) & (obs_data.site_id == cur_site_obs), constants.PSA_KEYS].values
    #     assert np.allclose(site_obs_obs[ix, :], cur_site_obs_obs)


class SCProbDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        event_site_combs: Dict[str, np.ndarray],
        db: DB,
        run_config: prob.RunParamsConfig,
        scalar_features: ml_data.ScalarFeatures,
        ims_mean: np.ndarray,
        ims_std: np.ndarray,
        hp_config: HyperParamsConfig,
        corr_dir: Path,
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

        self.event_scenario_ids = {}
        self.event_rels, self.scenario_ids = {}, []
        self.n_sites_scenario, self.n_sites_event = [], []
        self.norm_obs_ims, self.norm_sim_ims = [], []
        self.obs_ims, self.sim_ims = [], []
        self.residual, self.im_site_corrs = [], []
        self.n_scenarios_event = []
        self.im_misfit_score = []
        for ix, (cur_event, cur_sites) in enumerate(
            tqdm(self.event_sites.items(), desc="Processsing events")
        ):
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

            # Number of scenarios
            cur_n_scenarios = np.unique(self.event_site_combs[cur_event][:, 0]).size
            self.n_scenarios_event.append(cur_n_scenarios)

            cur_ids = ix * 100 + np.arange(cur_n_scenarios)
            self.event_scenario_ids[cur_event] = cur_ids
            self.scenario_ids.append(cur_ids)

            # Number of sites per scenario
            assert np.all(np.diff(self.event_site_combs[cur_event][:, 0]) >= 0)
            cur_n_sites_scenario = np.unique(
                self.event_site_combs[cur_event][:, 0], return_counts=True
            )[1]
            self.n_sites_scenario.append(cur_n_sites_scenario)

            ## Observation IMs
            cur_obs_data = db.get_obs_data(cur_event, cur_sites)
            cur_obs_data = np.log(cur_obs_data.loc[cur_sites, self.ims]).values
            self.obs_ims.append(cur_obs_data)

            ## Simulation IMs
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

            # Compute misfit score for each IM
            if hp_config.misfit_fn == "mse":
                cur_im_misfit_score = np.abs(cur_residual) ** 2
            elif hp_config.misfit_fn == "mae":
                cur_im_misfit_score = np.abs(cur_residual)
            else:
                raise ValueError(f"Unknown misfit function: {hp_config.misfit_fn}")
            self.im_misfit_score.append(cur_im_misfit_score)

            # Normalise & Append
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
            cur_corrs = pd.read_pickle(corr_dir / f"{cur_event}.pickle")
            for im_ix, cur_im in enumerate(self.ims):
                cur_corr_df = cur_corrs.get_im_corrs(cur_im).loc[cur_sites, cur_sites]
                cur_corrs_values[:, im_ix] = np.abs(
                    cur_corr_df.values[cur_site_combs[:, 0], cur_site_combs[:, 1]]
                )

            self.im_site_corrs.append(cur_corrs_values)

        self.n_scenarios_event = np.asarray(self.n_scenarios_event)
        self.cum_n_scenarios_event = np.cumsum(self.n_scenarios_event)

        self.n_sites_event = np.asarray(self.n_sites_event)
        self.cum_n_sites_event = np.cumsum(self.n_sites_event)

        self.n_sites_scenario = np.concatenate(self.n_sites_scenario)
        self.cum_n_sites_scenario = np.cumsum(self.n_sites_scenario)

        self.norm_sim_ims = np.concatenate(self.norm_sim_ims, axis=0)
        self.norm_obs_ims = np.concatenate(self.norm_obs_ims, axis=0)
        self.sim_ims = np.concatenate(self.sim_ims, axis=0)
        self.obs_ims = np.concatenate(self.obs_ims, axis=0)
        self.residual = np.concatenate(self.residual, axis=0)
        self.im_misfit_score = np.concatenate(self.im_misfit_score, axis=0)
        self.im_site_corrs = np.concatenate(self.im_site_corrs, axis=0)

        self.scenario_ids = np.concatenate(self.scenario_ids)
        self.record_scenario_ids = np.repeat(self.scenario_ids, self.n_sites_scenario)

        self.record_events = np.repeat(
            np.repeat(self.events, self.n_scenarios_event), self.n_sites_scenario
        )

        # Convert all site-combination to a dataframe for fast indexing
        site_combs_df, sim_ims_df = [], []
        site_combs_ix, sim_ims_ix = 0, 0
        for cur_event in self.events:

            ### TODO: Give scenario id
            # Site Combinations
            cur_site_comb_df = pd.DataFrame(
                self.event_site_combs[cur_event], columns=["site_int", "site_obs"]
            )
            cur_site_comb_df["event"] = cur_event
            cur_site_comb_df["ix"] = np.arange(
                site_combs_ix, site_combs_ix + cur_site_comb_df.shape[0]
            )

            cur_site_comb_df["scenario_id"] = np.repeat(
                self.event_scenario_ids[cur_event],
                np.unique(self.event_site_combs[cur_event][:, 0], return_counts=True)[
                    1
                ],
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

    def get_indices(self, batch_ind: np.ndarray):
        # Have to it this way, as some events may not have samples
        event_ind = np.argmax(
            batch_ind - self.cum_n_scenarios_event[:, None] < 0, axis=0
        )
        scenario_ind = np.where(
            event_ind > 0,
            batch_ind - self.cum_n_scenarios_event[event_ind - 1],
            batch_ind,
        )

        n_records = self.n_sites_scenario[batch_ind].sum()
        n_scenarios = len(batch_ind)
        site_int_ind = np.full(n_scenarios, fill_value=-1, dtype=int)
        site_obs_ind = np.full(n_scenarios, fill_value=-1, dtype=int)
        record_site_int_ind = np.full(n_records, fill_value=-1, dtype=int)
        record_site_obs_ind = np.full(n_records, fill_value=-1, dtype=int)
        record_ind = np.full(n_records, fill_value=-1, dtype=int)
        i_counter = 0
        for i in range(n_scenarios):
            cur_b_ix, cur_ev_ix = batch_ind[i], event_ind[i]
            if cur_b_ix == 0:
                cur_record_ind = np.arange(0, self.cum_n_sites_scenario[cur_b_ix])
            else:
                cur_record_ind = np.arange(
                    self.cum_n_sites_scenario[cur_b_ix - 1],
                    self.cum_n_sites_scenario[cur_b_ix],
                )

            if cur_ev_ix == 0:
                cur_site_int_ind = self.site_combs[cur_record_ind, 0]
                cur_site_obs_ind = self.site_combs[cur_record_ind, 1]
            else:
                cur_site_int_ind = (
                    self.cum_n_sites_event[cur_ev_ix - 1]
                    + self.site_combs[cur_record_ind, 0]
                )
                cur_site_obs_ind = (
                    self.cum_n_sites_event[cur_ev_ix - 1]
                    + self.site_combs[cur_record_ind, 1]
                )

            record_ind[i_counter : i_counter + cur_record_ind.size] = cur_record_ind

            record_site_int_ind[
                i_counter : i_counter + cur_record_ind.size
            ] = cur_site_int_ind
            record_site_obs_ind[
                i_counter : i_counter + cur_record_ind.size
            ] = cur_site_obs_ind

            site_int_ind[i] = cur_site_int_ind[0]
            site_obs_ind[i] = cur_site_obs_ind[0]

            i_counter += cur_record_ind.size

        assert (
            np.all(record_site_int_ind >= 0)
            and np.all(record_site_obs_ind >= 0)
            and np.all(record_ind >= 0)
        )

        return (
            event_ind,
            scenario_ind,
            site_int_ind,
            site_obs_ind,
            record_ind,
            record_site_int_ind,
            record_site_obs_ind,
        )

    def get_metadata(self, batch_ind: np.ndarray, rel_shuffle_ind: np.ndarray):
        (
            event_ind,
            scenario_ind,
            site_int_ind,
            site_obs_ind,
            record_ind,
            record_site_int_ind,
            record_site_obs_ind,
        ) = self.get_indices(batch_ind)

        return (
            self.events[event_ind],
            self.record_events[record_ind],
            self.sites[site_int_ind],
            self.sites[record_site_int_ind],
            self.sites[site_obs_ind],
            self.sites[record_site_obs_ind],
            self.rels[event_ind][:, rel_shuffle_ind],
            self.residual[site_int_ind],
            self.residual[record_site_int_ind],
            self.n_sites_scenario[batch_ind],
        )

    def get_batch(self, batch_ind: np.ndarray, shuffle_rels: bool):
        (
            event_ind,
            sc_scenario_ind,
            sc_site_int_ind,
            sc_site_obs_ind,
            record_ind,
            record_site_int_ind,
            record_site_obs_ind,
        ) = self.get_indices(batch_ind)

        site_int_norm_sim_ims = self.norm_sim_ims[record_site_int_ind, :, :]
        site_obs_norm_sim_ims = self.norm_sim_ims[record_site_obs_ind, :, :]
        site_obs_norm_obs_ims = self.norm_obs_ims[record_site_obs_ind, :]

        site_int_sim_ims = self.sim_ims[record_site_int_ind, :, :]
        site_obs_sim_ims = self.sim_ims[record_site_obs_ind, :, :]
        site_obs_obs_ims = self.obs_ims[record_site_obs_ind, :]

        scalar_features = self.scalar_features_values[record_ind]
        record_im_misfit_score = self.im_misfit_score[record_site_int_ind]

        im_site_corrs = self.im_site_corrs[record_ind, :]

        rel_shuffle_ind = (
            np.random.permutation(self.n_rels)
            if shuffle_rels
            else np.arange(self.n_rels)
        )

        return (
            batch_ind,
            rel_shuffle_ind,
            self.scenario_ids[batch_ind],
            self.im_misfit_score[sc_site_int_ind][:, rel_shuffle_ind, :],
            self.record_scenario_ids[record_ind],
            site_int_norm_sim_ims[:, rel_shuffle_ind, :],
            site_obs_norm_sim_ims[:, rel_shuffle_ind, :],
            site_obs_norm_obs_ims,
            site_int_sim_ims[:, rel_shuffle_ind, :],
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            record_im_misfit_score[:, rel_shuffle_ind, :],
            im_site_corrs,
        )

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.cum_n_scenarios_event[-1]


def create_indRelModel(
    hp_config: HyperParamsConfig,
    scalar_features: ml_data.ScalarFeatures,
    run_config: prob.RunParamsConfig,
):

    prob_model = models.ProbIndModel(
        hp_config.fc_units,
        scalar_features.n_scalar_features,
        len(run_config.ims),
        hp_config.n_im_features,
        is_sub_model=False,
        per_im_prob=run_config.per_im_prob,
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
            (
                hp_config.batch_size,
                run_config.n_rels,
                scalar_features.n_scalar_features,
            ),
        ],
    )

    return prob_model


def create_IMmodel(
    hp_config: HyperParamsConfig,
    scalar_features: ml_data.ScalarFeatures,
    run_config: prob.RunParamsConfig,
):
    n_inputs = (
        run_config.n_rels * hp_config.n_im_features
    ) + scalar_features.n_scalar_features

    prob_model = models.ProbIMModel(
        n_inputs, hp_config.fc_units, run_config.n_rels, one_hot_n_ims=0
    )

    print(f"Model summary")
    summary(
        prob_model,
        input_size=[
            (
                hp_config.batch_size,
                hp_config.n_im_features,
                run_config.n_rels,
                run_config.n_ims,
            ),
            (
                hp_config.batch_size,
                run_config.n_rels,
                scalar_features.n_scalar_features,
            ),
        ],
    )

    return prob_model


def get_scenario_mask(record_scenario_ids: torch.Tensor, scenario_ids: torch.Tensor):
    n_scenarios = scenario_ids.shape[0]
    n_samples = record_scenario_ids.shape[0]

    scenario_mask = einops.repeat(
        record_scenario_ids,
        "n_samples -> n_samples n_scenarios",
        n_scenarios=n_scenarios,
    )
    scenario_mask = scenario_mask == einops.repeat(
        scenario_ids, "n_scenarios -> n_samples n_scenarios", n_samples=n_samples
    )

    return scenario_mask


def compute_single_loss(
    agg_probs: Union[torch.Tensor, np.ndarray],
    im_misfit_score: Union[torch.Tensor, np.ndarray],
    im_weights: Union[torch.Tensor, np.ndarray],
    sc_weights: Union[torch.Tensor, np.ndarray] = None,
):
    """
    Computes the scenario loss for the single-prob model
    i.e. gives P(R_i)

    Parameters
    ----------
    agg_probs: array of floats
        The aggregated probability for each scenario
        Shape: [n_scenarios, n_rels]
    im_misfit_score: array of floats
        The misfit score for each scenario, realisation and IM
        Shape: [n_scenarios, n_rels, n_ims]
    im_weights: array of floats
        IM weights
        Shape - [n_ims]

    Returns
    -------
    scenario_loss: torch.Tensor
        The loss for each scenario and
        Shape: [n_scenarios]
    loss: torch.Tensor
        The average loss across all scenarios
    """
    # Computes the scenario loss
    # L_{s} = \sum_{i} p_{i} \sum_{j} w_j M_{i, j}
    # where
    # - p_{i} is the aggregated probability for the i-th realisation
    # - M_{i, i} is the misfit score of the i-th realisation and j-th IM
    scenario_loss = einops.einsum(
        agg_probs,
        im_weights,
        im_misfit_score,
        "scenario rel, im, scenario rel im -> scenario",
    )

    if sc_weights is not None:
        weighted_scenario_loss = sc_weights * scenario_loss
        return scenario_loss, weighted_scenario_loss, weighted_scenario_loss.mean()
    else:
        return scenario_loss, None, scenario_loss.mean()


def compute_single_agg_prob(
    scenario_mask: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    w_pred: Union[torch.Tensor, np.ndarray],
    im_weights: Union[torch.Tensor, np.ndarray],
):
    """
    Computes the aggregated probability for
    the single-prob model

    Parameters
    ----------
    scenario_mask: array of floats
        Mask of shape [n_samples, n_scenarios]
        that shows which scenario each sample belongs to
    pred: array of floats
        Probability predictions of shape [n_samples, n_rels]
        Sum to 1.0 across realisations
        i.e. np.allclose(np.sum(pred, axis=1), 1.0)
    w_pred: array of floats
        Site-weights for each observation site and IM
        If the weight model only gives one weight
        (across all IMs) repeat this weight for each IM
        These have to be already normalised such that for a single
        scenario the site-weights sum to 1.0
        Shape: [n_samples, n_ims]
    im_weights: array of floats
        IM weights
        Shape - [n_ims]

    Returns
    -------
    agg_prob: torch.Tensor
        The aggregated probability for each scenario
        Shape: [n_scenarios, n_rels]
    """
    ### Aggregated probability for single scenario
    # P(R_i) = \sum_{r} p_{r,i} \sum_{j} w_{j} \rho_{r, j} \delta_{r, s}
    # where
    # - p_{r,i} is the predicted probability of the r-th observation site, i-th realisation
    # - rho_{r, j} is the site-weight of the r-th observation site and j-th im
    # - w_{j} is the weight of the j-th IM
    # - \delta_{r, s} is a boolean that shows if the r-th observation site is in the s-th scenario
    agg_prob = einops.einsum(
        pred,
        w_pred,
        im_weights,
        scenario_mask,
        "obs rel, obs im, im, obs scenario -> scenario rel",
    )
    return agg_prob


def compute_multi_agg_prob(
    scenario_mask: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    w_pred: Union[torch.Tensor, np.ndarray],
    im_weights: Union[torch.Tensor, np.ndarray],
):
    """
    Computes the aggregated probability for
    the single-prob model

    Parameters
    ----------
    scenario_mask: array of floats
        Mask of shape [n_samples, n_scenarios]
        that shows which scenario each sample belongs to
    pred: array of floats
        Probability predictions of shape [n_samples, n_rels, n_ims]
        Sum to 1.0 across realisations
        i.e. np.allclose(np.sum(pred, axis=1), 1.0)
    w_pred: array of floats
        Site-weights for each observation site and IM
        If the weight model only gives one weight
        (across all IMs) repeat this weight for each IM
        These have to be already normalised such that for a single
        scenario the site-weights sum to 1.0
        Shape: [n_samples, n_ims]
    im_weights: array of floats
        IM weights
        Shape - [n_ims]

    Returns
    -------
    agg_prob: torch.Tensor
        The aggregated probability for each scenario
        Shape: [n_scenarios, n_rels, n_ims]
    """
    # As this computation is per IM,
    # the weights need to add up to the number of IMs
    im_weights = im_weights * len(im_weights)

    ### Aggregated probability for single scenario
    # P(R_i) = \sum_{r} \sum_{j} p_{r,i, j} w_{j} \rho_{r, j} \delta_{r, s}
    # where
    # - p_{r,i} is the predicted probability of the r-th observation site, i-th realisation and j-th IM
    # - rho_{r} is the site-weight of the r-th observation site and j-th im
    # - w_{j} is the weight of the j-th IM
    # - \delta_{r, s} is a boolean that shows if the r-th observation site is in the s-th scenario
    agg_prob = einops.einsum(
        pred,
        w_pred,
        im_weights,
        scenario_mask,
        "obs rel im, obs im, im, obs scenario -> scenario rel im",
    )
    return agg_prob


def compute_multi_loss(
    agg_probs: Union[torch.Tensor, np.ndarray],
    im_misfit_score: Union[torch.Tensor, np.ndarray],
    im_weights: Union[torch.Tensor, np.ndarray],
    sc_weights: Union[torch.Tensor, np.ndarray] = None,
):
    """
    Computes the scenario loss for the single-prob model
    i.e. gives P(R_i|IM_j)

    Parameters
    ----------
    agg_probs: array of floats
        The aggregated probability for each scenario
        Shape: [n_scenarios, n_rels, n_ims]
    im_misfit_score: array of floats
        The misfit score for each scenario, realisation and IM
        Shape: [n_scenarios, n_rels, n_ims]
    im_weights: array of floats
        IM weights
        Shape - [n_ims]

    Returns
    -------
    scenario_loss: torch.Tensor
        The loss for each scenario and
        Shape: [n_scenarios]
    loss: torch.Tensor
        The average loss across all scenarios
    """
    # Computes the scenario loss
    # L_{s} = \sum_{i} \sum_{j} p_{i, j}  w_j M_{i, j}
    # where
    # - p_{i, j} is the aggregated probability for the i-th realisation and j-th IM
    # - M_{i, i} is the misfit score of the i-th realisation and j-th IM
    scenario_loss = einops.einsum(
        agg_probs,
        im_weights,
        im_misfit_score,
        "scenario rel im, im, scenario rel im -> scenario",
    )

    if sc_weights is not None:
        weighted_scenario_loss = sc_weights * scenario_loss
        return scenario_loss, weighted_scenario_loss, weighted_scenario_loss.mean()
    else:
        return scenario_loss, None, scenario_loss.mean()


def get_weight_prediction(
    weight_model: models.WeightModel,
    scalar_features: torch.Tensor,
    scenario_mask: torch.Tensor,
    run_config: prob.RunParamsConfig,
):
    """Gets the normalised weight predictions"""
    w_pred = weight_model(
        scalar_features.to(run_config.device, dtype=torch.float32, non_blocking=True)
    )

    ### Normalize weight predictions
    # Apply the scenario mask
    w_pred = w_pred[:, None, :] * scenario_mask[..., None]
    # Sum each scenario weights, and then remove scenario axis
    w_pred = (w_pred / w_pred.sum(axis=0)).sum(axis=1)

    return w_pred


def get_loth_weights(
    im_site_corrs: Union[torch.Tensor, np.ndarray],
    scenario_mask: Union[torch.Tensor, np.ndarray],
):
    """
    Normalises the loth & baker site-correlations
    such that they sum to one for each scenario

    Parameters
    ----------
    im_site_corrs: array of floats
        The loth & baker site-correlations
        [n_samples, n_ims]
    scenario_mask: array of bools
        Mask that defines which scenario
        each sample belongs to
        [n_samples, n_scenarios]

    Returns
    -------
    w_pred: array of floats
        Normalised loth & baker site-correlations
        [n_samples, n_ims]
    """

    w_pred = im_site_corrs
    w_pred = w_pred[:, None, :] * scenario_mask[..., None]

    # Sum each scenario weights, and then remove scenario axis
    w_pred = (w_pred / w_pred.sum(axis=0)).sum(axis=1)

    return w_pred


def compute_loth_baker_scenario_weights(
    im_site_corrs: torch.Tensor,
    scenario_mask: torch.Tensor,
    im_weights: torch.Tensor,
    min_weight: float,
    max_weight: float,
):
    return torch.clamp(
        einops.einsum(
            im_site_corrs,
            im_weights,
            scenario_mask,
            "obs im, im, obs scenario -> scenario",
        ),
        min_weight,
        max_weight,
    )


def train(
    prob_model: models.ProbIMModel,
    weight_model: models.WeightModel,
    train_dataset: SCProbDataset,
    val_dataset: SCProbDataset,
    hp_config: HyperParamsConfig,
    run_config: prob.RunParamsConfig,
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

    train_dataloader = prob.CustomTabularDataLoader(
        train_dataset, hp_config.batch_size, True, shuffle_rels=False
    )
    val_dataloader = prob.CustomTabularDataLoader(
        val_dataset, hp_config.batch_size, True, shuffle_rels=False
    )

    def save_grad(grad: torch.Tensor):
        grad_df.loc[len(grad_df)] = {
            "epoch": epoch_ix,
            "shape": str(tuple(grad.shape)),
            "min": torch.abs(grad).min().item(),
            "max": torch.abs(grad).max().item(),
            "norm": grad.norm().item(),
        }

    if run_config.debug:
        grad_df = pd.DataFrame(columns=["epoch", "shape", "min", "max", "norm"])
        for param in prob_model.parameters():
            param.register_hook(save_grad)

    im_weights = torch.from_numpy(run_config.im_weights).to(
        run_config.device, torch.float32
    )

    lr_ix = 0
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
            batch_ind,
            rel_shuffle_ind,
            sc_ids,
            sc_im_misfit_score,
            record_scenario_ids,
            site_int_norm_sim_ims,
            site_obs_norm_sim_ims,
            site_obs_norm_obs_ims,
            site_int_sim_ims,
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            im_misfit_score,
            im_site_corrs,
        ) in enumerate(iter_loop):
            pred = prob.get_prediction(
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
                if run_config.debug:
                    with tmp.TemporaryDirectory() as tmp_dir:
                        grad_df.to_csv(
                            Path(tmp_dir) / f"{mlt.utils.create_run_id()}_grads.csv",
                            index=False,
                        )
                        print(f"Saved grads to {tmp_dir}")
                exit()

            scenario_mask = get_scenario_mask(record_scenario_ids, sc_ids)
            scenario_mask = scenario_mask.to(run_config.device, torch.float32)

            im_site_corrs = im_site_corrs.to(run_config.device, dtype=torch.float32)

            # w_pred = get_weight_prediction(
            #     weight_model, scalar_features, scenario_mask, run_config)
            w_pred = get_loth_weights(im_site_corrs, scenario_mask)

            scenario_weights = compute_loth_baker_scenario_weights(
                im_site_corrs, scenario_mask, im_weights, 0.5, 2.0
            )

            if run_config.per_im_prob:
                agg_probs = compute_multi_agg_prob(
                    scenario_mask, pred, w_pred, im_weights
                )
                scenario_loss, weighted_scenario_loss, loss = compute_multi_loss(
                    agg_probs,
                    sc_im_misfit_score.to(run_config.device, torch.float32),
                    im_weights,
                    sc_weights=scenario_weights,
                )
            else:
                agg_probs = compute_single_agg_prob(
                    scenario_mask,
                    pred,
                    w_pred,
                    im_weights,
                )
                scenario_loss, weighted_scenario_loss, loss = compute_single_loss(
                    agg_probs,
                    sc_im_misfit_score.to(run_config.device, torch.float32),
                    im_weights,
                    sc_weights=scenario_weights,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["loss_hist_train"][epoch_ix] += loss.item()

        metrics["loss_hist_train"][epoch_ix] /= len(train_dataloader)

        prob_model.eval()
        with torch.no_grad():
            for i, (
                batch_ind,
                rel_shuffle_ind,
                sc_ids,
                sc_im_misfit_score,
                record_scenario_ids,
                site_int_norm_sim_ims,
                site_obs_norm_sim_ims,
                site_obs_norm_obs_ims,
                site_int_sim_ims,
                site_obs_sim_ims,
                site_obs_obs_ims,
                scalar_features,
                im_misfit_score,
                im_site_corrs,
            ) in enumerate(val_dataloader):
                pred = prob.get_prediction(
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

                scenario_mask = get_scenario_mask(record_scenario_ids, sc_ids)
                scenario_mask = scenario_mask.to(run_config.device, torch.float32)

                im_site_corrs = im_site_corrs.to(run_config.device, dtype=torch.float32)

                # w_pred = get_weight_prediction(
                #     weight_model, scalar_features, scenario_mask, run_config)
                w_pred = get_loth_weights(
                    im_site_corrs.to(run_config.device, dtype=torch.float32),
                    scenario_mask,
                )

                scenario_weights = compute_loth_baker_scenario_weights(
                    im_site_corrs, scenario_mask, im_weights, 0.5, 2.0
                )

                if run_config.per_im_prob:
                    agg_probs = compute_multi_agg_prob(
                        scenario_mask, pred, w_pred, im_weights
                    )
                    scenario_loss, weighted_scenario_loss, loss = compute_multi_loss(
                        agg_probs,
                        sc_im_misfit_score.to(run_config.device, torch.float32),
                        im_weights,
                        sc_weights=scenario_weights,
                    )
                else:
                    agg_probs = compute_single_agg_prob(
                        scenario_mask,
                        pred,
                        w_pred,
                        im_weights,
                    )
                    scenario_loss, weighted_scenario_loss, loss = compute_single_loss(
                        agg_probs,
                        sc_im_misfit_score.to(run_config.device, torch.float32),
                        im_weights,
                        sc_weights=scenario_weights,
                    )

                metrics["loss_hist_val"][epoch_ix] += loss.item()

            metrics["loss_hist_val"][epoch_ix] /= len(val_dataloader)

            if metrics["loss_hist_val"][epoch_ix] < best_val_loss:
                best_val_loss = metrics["loss_hist_val"][epoch_ix]
                best_model_state = prob_model.state_dict()
                best_model_epoch = epoch_ix

            print(f"Epoch {epoch_ix + 1}/{hp_config.n_epochs}")
            print(f"\tTraining" f"\t\tLoss: {metrics['loss_hist_train'][epoch_ix]:.4f}")
            print(f"\tValidation" f"\t\tLoss: {metrics['loss_hist_val'][epoch_ix]:.4f}")

    return metrics, best_model_state, best_model_epoch


def get_dataset_prediction(
    dataset: SCProbDataset,
    prob_model: models.ProbIMModel,
    weight_model: models.WeightModel,
    run_config: prob.RunParamsConfig,
    dist_matrix: pd.DataFrame,
    hp_config: HyperParamsConfig,
):
    pred_dataloader = prob.CustomTabularDataLoader(
        dataset, hp_config.batch_size, shuffle=False, shuffle_rels=False
    )

    assert np.isclose(np.sum(run_config.im_weights), 1.0)
    im_weights = torch.from_numpy(run_config.im_weights).to(
        run_config.device, torch.float32
    )

    ### Sample setup
    columns = [
        "event_id",
        "site_int",
        "site_obs",
        "rel_id",
        "s2s_distance",
    ]
    im_res_cols = np.char.add(run_config.ims, "_residual").tolist()
    im_site_weights_cols = np.char.add(run_config.ims, "_site_weights").tolist()
    im_misfit_cols = np.char.add(run_config.ims, "_misfit").tolist()
    columns += im_res_cols + im_site_weights_cols + im_misfit_cols

    if run_config.per_im_prob:
        prob_cols = np.char.add(run_config.ims, "_prob").tolist()
        columns += prob_cols
    else:
        columns += ["prob"]

    sample_results_df = pd.DataFrame(
        index=np.arange(dataset.n_sites_scenario.sum() * run_config.n_rels),
        columns=columns,
    )

    sample_results_df = sample_results_df.astype(
        dict(zip(columns[4:], len(columns[4:]) * [np.float32]))
    )

    sample_results_df["event_id"] = pd.Categorical(
        categories=dataset.events, values=sample_results_df.shape[0] * [pd.NA]
    )
    sample_results_df["site_int"] = pd.Categorical(
        categories=np.unique(dataset.sites), values=sample_results_df.shape[0] * [pd.NA]
    )
    sample_results_df["site_obs"] = pd.Categorical(
        categories=np.unique(dataset.sites), values=sample_results_df.shape[0] * [pd.NA]
    )
    sample_results_df["rel_id"] = pd.Categorical(
        categories=np.unique(dataset.rels), values=sample_results_df.shape[0] * [pd.NA]
    )

    ### Iterate over dataset
    sc_results, sc_sum_results = [], []
    with (torch.no_grad()):
        prob_model.eval()
        start_ix = 0
        for i, (
            batch_ind,
            rel_shuffle_ind,
            sc_ids,
            sc_im_misfit_score,
            record_scenario_ids,
            site_int_norm_sim_ims,
            site_obs_norm_sim_ims,
            site_obs_norm_obs_ims,
            site_int_sim_ims,
            site_obs_sim_ims,
            site_obs_obs_ims,
            scalar_features,
            im_misfit_score,
            im_site_corrs,
        ) in enumerate(tqdm(pred_dataloader)):
            # Metadata
            (
                events,
                record_events,
                site_int,
                record_site_int,
                site_obs,
                record_site_obs,
                rels,
                residual,
                record_residual,
                n_sites_scenario,
            ) = dataset.get_metadata(batch_ind, rel_shuffle_ind)

            # Predictions
            pred = prob.get_prediction(
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

            # Scenario mask
            scenario_mask = get_scenario_mask(record_scenario_ids, sc_ids)
            scenario_mask = scenario_mask.to(run_config.device, torch.float32)

            # Site weights
            # w_pred = get_weight_prediction(
            #     weight_model, scalar_features, scenario_mask, run_config)
            w_pred = get_loth_weights(
                im_site_corrs.to(run_config.device, dtype=torch.float32), scenario_mask
            )

            cur_n_samples = pred.shape[0]
            end_ix = start_ix + cur_n_samples * run_config.n_rels - 1

            ### Store sample results
            # Event
            sample_results_df.loc[start_ix:end_ix, "event_id"] = einops.repeat(
                record_events, "batch -> (batch rel)", rel=pred.shape[1]
            )
            # Site of Interest
            sample_results_df.loc[
                start_ix:end_ix, "site_int"
            ] = df_site_int = einops.repeat(
                record_site_int, "batch -> (batch rel)", rel=pred.shape[1]
            )
            # Observation site
            sample_results_df.loc[
                start_ix:end_ix, "site_obs"
            ] = df_site_obs = einops.repeat(
                record_site_obs, "batch -> (batch rel)", rel=pred.shape[1]
            )
            # Realisation
            sample_results_df.loc[start_ix:end_ix, "rel_id"] = einops.rearrange(
                np.repeat(rels, n_sites_scenario, axis=0), "batch rel -> (batch rel)"
            )
            # Probabilities
            if run_config.per_im_prob:
                sample_results_df.loc[start_ix:end_ix, prob_cols] = (
                    einops.rearrange(pred, "batch rel im -> (batch rel) im")
                    .numpy(force=True)
                    .astype(np.float32)
                )
            else:
                sample_results_df.loc[start_ix:end_ix, "prob"] = (
                    einops.rearrange(pred, "batch rel -> (batch rel)")
                    .numpy(force=True)
                    .astype(np.float32)
                )

            # Misfit score
            sample_results_df.loc[start_ix:end_ix, im_misfit_cols] = (
                einops.rearrange(im_misfit_score, "batch rel im -> (batch rel) im")
                .numpy(force=True)
                .astype(np.float32)
            )
            if not run_config.per_im_prob:
                sample_results_df.loc[start_ix:end_ix, "misfit_score"] = (
                    einops.rearrange(
                        einops.einsum(
                            im_misfit_score.to(run_config.device, torch.float32),
                            im_weights,
                            "obs rel im, im -> obs rel",
                        ),
                        "obs rel -> (obs rel)",
                    )
                    .numpy(force=True)
                    .astype(np.float32)
                )

            # Site weights
            sample_results_df.loc[start_ix:end_ix, im_site_weights_cols] = (
                einops.repeat(
                    w_pred,
                    "batch im -> (batch rel) im",
                    rel=pred.shape[1],
                )
                .numpy(force=True)
                .astype(np.float32)
            )

            # IM residuals
            sample_results_df.loc[start_ix:end_ix, im_res_cols] = einops.rearrange(
                record_residual, "batch rel im -> (batch rel) im"
            ).astype(np.float32)

            # Site-to-site distance
            sample_results_df.loc[start_ix:end_ix, "s2s_distance"] = dist_matrix.values[
                dist_matrix.index.get_indexer_for(df_site_int),
                dist_matrix.columns.get_indexer_for(df_site_obs),
            ].astype(np.float32)

            sc_weights = compute_loth_baker_scenario_weights(
                im_site_corrs.to(run_config.device, torch.float32),
                scenario_mask,
                im_weights,
                0.5,
                2.0,
            )

            ### Scenario results
            # Compute the scenario probabilities & loss
            cur_sc_result = pd.DataFrame(
                index=["event_id", "site_int", "rel_id"],
                data=[
                    einops.repeat(events, "sc -> (sc rel)", rel=run_config.n_rels),
                    einops.repeat(site_int, "sc -> (sc rel)", rel=run_config.n_rels),
                    einops.rearrange(rels, "sc rel -> (sc rel)"),
                ],
            ).T

            if run_config.per_im_prob:
                cur_agg_prob = compute_multi_agg_prob(
                    scenario_mask,
                    pred,
                    w_pred,
                    im_weights,
                )
                assert np.allclose(cur_agg_prob.sum(axis=1).numpy(force=True), 1.0)

                scenario_loss, weighted_scenario_loss, loss = compute_multi_loss(
                    cur_agg_prob,
                    sc_im_misfit_score.to(run_config.device, torch.float32),
                    im_weights,
                    sc_weights=sc_weights,
                )

                cur_sc_result[prob_cols] = einops.rearrange(
                    cur_agg_prob.numpy(force=True), "sc rel im -> (sc rel) im"
                )
            else:
                cur_agg_prob = compute_single_agg_prob(
                    scenario_mask,
                    pred,
                    w_pred,
                    im_weights,
                )
                assert np.allclose(cur_agg_prob.sum(axis=1).numpy(force=True), 1.0)

                scenario_loss, weighted_scenario_loss, loss = compute_single_loss(
                    cur_agg_prob,
                    sc_im_misfit_score.to(run_config.device, torch.float32),
                    im_weights,
                    sc_weights=sc_weights,
                )

                cur_sc_result["prob"] = einops.rearrange(
                    cur_agg_prob.numpy(force=True), "sc rel -> (sc rel)"
                )

                cur_sc_result["misfit_score"] = einops.rearrange(
                    einops.einsum(
                        sc_im_misfit_score.to(run_config.device, torch.float32),
                        im_weights,
                        "sc rel im, im -> sc rel",
                    )
                    .numpy(force=True)
                    .astype(np.float32),
                    "sc rel -> (sc rel)",
                )

            cur_sc_result[im_misfit_cols] = (
                einops.rearrange(sc_im_misfit_score, "sc rel im -> (sc rel) im")
                .numpy(force=True)
                .astype(np.float32)
            )

            cur_sc_result[im_res_cols] = einops.rearrange(
                residual, "sc rel im -> (sc rel) im"
            ).astype(np.float32)

            ### Scenario Summary
            sc_group = sample_results_df.loc[start_ix:end_ix].groupby(
                ["event_id", "site_int"], observed=True
            )
            n_obs_sites = sc_group.site_obs.nunique()

            cur_sc_sum = pd.DataFrame(
                index=["event_id", "site_int", "loss", "w_loss", "weight"],
                data=[
                    events,
                    site_int,
                    scenario_loss.numpy(force=True),
                    weighted_scenario_loss.numpy(force=True)
                    if weighted_scenario_loss is not None
                    else None,
                    sc_weights.numpy(force=True),
                ],
            ).T

            assert np.all(
                n_obs_sites.index.get_level_values(0).values == cur_sc_sum.event_id
            )
            assert np.all(
                n_obs_sites.index.get_level_values(1).values == cur_sc_sum.site_int
            )
            cur_sc_sum["n_obs_sites"] = n_obs_sites.values
            cur_sc_sum["min_s2s_dist"] = sc_group["s2s_distance"].min().values

            cur_sc_result.index = mlt.array_utils.numpy_str_join(
                "_",
                cur_sc_result.event_id.values.astype(str),
                cur_sc_result.site_int.values.astype(str),
                cur_sc_result.rel_id.values.astype(str),
            )
            cur_sc_sum.index = mlt.array_utils.numpy_str_join("_", events, site_int)

            sc_results.append(cur_sc_result)
            sc_sum_results.append(cur_sc_sum)

            start_ix = end_ix + 1

        sc_results = pd.concat(sc_results, axis=0)
        sc_sum_results = pd.concat(sc_sum_results, axis=0)

        return sample_results_df, sc_results, sc_sum_results


def post_processing(
    prob_model: models.ProbIMModel,
    weight_model: models.WeightModel,
    train_dataset: SCProbDataset,
    val_dataset: SCProbDataset,
    hp_config: HyperParamsConfig,
    run_config: prob.RunParamsConfig,
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
    print(f"Computing results")
    (
        train_sample_results,
        train_sc_results,
        train_sc_sum_results,
    ) = get_dataset_prediction(
        train_dataset, prob_model, weight_model, run_config, dist_matrix, hp_config
    )
    print(
        f"\tTrain sample results - Memory usage: {train_sample_results.memory_usage(deep=True).sum() / 1e6} MB"
    )
    val_sample_results, val_sc_results, val_sc_sum_results = get_dataset_prediction(
        val_dataset, prob_model, weight_model, run_config, dist_matrix, hp_config
    )
    print(
        f"\tVal sample results - Memory usage: {val_sample_results.memory_usage(deep=True).sum() / 1e6}"
    )

    train_sample_results.to_parquet(cur_out_dir / "train_sample_results.parquet")
    train_sc_results.to_parquet(cur_out_dir / "train_scenario_results.parquet")
    train_sc_sum_results.to_parquet(cur_out_dir / "train_scenario_summary.parquet")

    val_sample_results.to_parquet(cur_out_dir / "val_sample_results.parquet")
    val_sc_results.to_parquet(cur_out_dir / "val_scenario_results.parquet")
    val_sc_sum_results.to_parquet(cur_out_dir / "val_scenario_summary.parquet")

    # Save loss history
    pd.to_pickle(metrics, cur_out_dir / "metrics.pickle")

    # Save the model
    torch.save(prob_model, cur_out_dir / "model.pt")

    # Metadata
    metadata = {
        "method_type": constants.RankingMethod.ml_prob_per_im.value
        if run_config.per_im_prob
        else constants.RankingMethod.ml_prob.value,
        "hp_config": hp_config.to_dict(),
        "best_epoch": best_epoch,
        "data": data_metadata,
        "run_config": run_config.to_dict(),
    }
    mlt.utils.write_to_yaml(metadata, cur_out_dir / "meta.yaml")

    # n_inputs = (run_config.n_rels * hp_config.n_im_features) + scalar_features.n_scalar_features
    draw_graph(
        prob_model,
        input_size=[
            (
                hp_config.batch_size,
                hp_config.n_im_features,
                run_config.n_rels,
                run_config.n_ims,
            ),
            (
                hp_config.batch_size,
                run_config.n_rels,
                scalar_features.n_scalar_features,
            ),
        ],
        expand_nested=True,
        filename="prob_model_vis",
        save_graph=True,
        directory=str(cur_out_dir),
    )


def compute_scenario_distribution(
    sample_results: pd.DataFrame,
    run_config: prob.RunParamsConfig,
    im_site_weights_suffix: str = "_site_corr_weights",
):
    """
    Computes the realisation distribution for each scenario
    """
    im_prob_cols = np.char.add(run_config.ims, "_prob")
    im_site_weight_cols = np.char.add(run_config.ims, im_site_weights_suffix)
    im_res_cols = np.char.add(run_config.ims, "_residual")
    im_misfit_cols = np.char.add(run_config.ims, "_misfit")

    scenario_results = []
    scenario_sum_results = []
    groups = sample_results.groupby(["event_id", "site_int"], observed=True)
    iter_loop = tqdm(groups, desc="Processing scenarios")
    for (cur_event, cur_site_int), cur_group in iter_loop:
        cur_group = cur_group.sort_values(["site_obs", "rel_id"])

        cur_rels = cur_group.rel_id[: run_config.n_rels].values.astype(str)
        cur_scenario_mask = np.ones(
            (cur_group.shape[0] // run_config.n_rels, 1), dtype=float
        )
        cur_im_site_weights = cur_group[im_site_weight_cols][
            :: run_config.n_rels
        ].values
        assert np.allclose(cur_im_site_weights.sum(axis=0), 1.0)

        # Compute the scenario probabilities & loss
        if run_config.per_im_prob:
            cur_agg_prob = compute_multi_agg_prob(
                cur_scenario_mask,
                einops.rearrange(
                    cur_group[im_prob_cols].values,
                    "(obs rel) im -> obs rel im",
                    rel=run_config.n_rels,
                ),
                cur_im_site_weights,
                run_config.im_weights,
            ).squeeze()
            assert np.allclose(cur_agg_prob.sum(axis=0), 1.0)
            cur_result = pd.DataFrame(
                index=cur_rels, columns=im_prob_cols, data=cur_agg_prob
            )

            scenario_loss, weighted_scenario_loss, loss = compute_multi_loss(
                cur_agg_prob[None, ...],
                cur_group[im_misfit_cols].values[: run_config.n_rels][None, ...],
                run_config.im_weights,
            )
        else:
            cur_agg_prob = compute_single_agg_prob(
                cur_scenario_mask,
                einops.rearrange(
                    cur_group.prob.values, "(obs rel) -> obs rel", rel=run_config.n_rels
                ),
                cur_im_site_weights,
                run_config.im_weights,
            ).squeeze()
            assert np.isclose(cur_agg_prob.sum(), 1.0)
            cur_result = pd.DataFrame(
                index=cur_rels, columns=["prob"], data=cur_agg_prob
            )

            scenario_loss, weighted_scenario_loss, loss = compute_single_loss(
                cur_agg_prob[None, ...],
                cur_group[im_misfit_cols].values[: run_config.n_rels][None, ...],
                run_config.im_weights,
            )

        cur_result["event_id"] = cur_event
        cur_result["site_int"] = cur_site_int
        cur_result["rel_id"] = cur_result.index

        cur_rel_group = cur_group.groupby("rel_id", observed=True)
        assert np.all(cur_rel_group.first().index == cur_result.index)
        cur_result[im_misfit_cols] = cur_rel_group.first()[im_misfit_cols]

        cur_residuals = cur_rel_group.first().loc[:, im_res_cols]
        assert np.all(cur_residuals.index == cur_result.rel_id)
        cur_result.loc[:, im_res_cols] = cur_residuals.values

        cur_result.index = mlt.array_utils.numpy_str_join(
            "_", cur_event, cur_site_int, cur_result.index.values.astype(str)
        )

        scenario_results.append(cur_result)

        # Scenario summary details
        scenario_sum_results.append(
            (
                cur_event,
                cur_site_int,
                scenario_loss[0],
                cur_group.site_obs.nunique(),
                cur_group.s2s_distance.min(),
            )
        )

    scenario_df = pd.concat(scenario_results, axis=0)
    scenario_sum_df = pd.DataFrame(
        data=scenario_sum_results,
        columns=["event_id", "site_int", "loss", "n_sites", "min_s2s_distance"],
    )
    return scenario_sum_df, scenario_df
