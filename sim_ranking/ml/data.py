import time
from pathlib import Path
from typing import Dict, Sequence
import multiprocessing as mp

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from . import similarity_score as ss
from .. import db

def compute_site_combinations(
    sites: Dict[str, np.ndarray],
    events: Sequence[str],
    dist_matrix: pd.DataFrame,
    sites_to_use: np.ndarray = None,
    max_dist: float = 100,
):
    """
    Compute the site combinations for each event

    Returns allowed sites and site-combination
    indices (for allowed sites) for each event
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        cur_sites = sites[cur_event]
        cur_sites = (
            cur_sites
            if sites_to_use is None
            else cur_sites[np.isin(cur_sites, sites_to_use)]
        )

        if len(cur_sites) < 2:
            continue

        # Filter for the current event sites
        # and site-combinations less than max_dist km apart
        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]
        cur_dist_mask = (cur_dist_matrix.values < max_dist) & (cur_dist_matrix.values > 0)
        cur_row_ind, cur_col_ind = np.nonzero(cur_dist_mask)

        # Get the site combinations
        # First is the site of interest, second is the observation site
        # Indices into the sites to use for the current event
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)

        site_combs[cur_event] = cur_site_combs
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites


class ResponseSpectrumDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: db.DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        dist_matrix: pd.DataFrame,
        site_features: Sequence[str],
        max_dist: float = 100,
        sim_score_fn=ss.similiarity_score,
    ):
        self.sim_score_fn = sim_score_fn

        self.site_combs = site_combs
        self.dist_matrix = dist_matrix

        self.station_df = station_df
        self.site_features = site_features

        self.event_sites = event_sites
        self.events = np.asarray(list(self.event_sites.keys()))
        self.n_rels = n_rels

        self.all_sites = dist_matrix.index.values.astype(str)
        self.all_sites_ix_lookup = {
            cur_site: ix for ix, cur_site in enumerate(self.all_sites)
        }

        self.pSA_keys = pSA_keys
        self.periods = periods

        # Get the simulation and observation data
        self.sim_im_dfs, self.obs_im_dfs = {}, {}
        self.n_samples_event, self.rels = [], {}
        for cur_event, cur_sites in event_sites.items():
            cur_sim_data = db.get_sim_data(cur_event, cur_sites)
            # Only use a subset of the available realisations to
            # prevent over-fitting to these events
            if np.any(cur_sim_data.data_source == "specific"):
                self.rels[cur_event] = np.random.choice(
                    cur_sim_data.rel_id.unique(), self.n_rels, replace=False
                )
                cur_mask = (cur_sim_data.data_source.values == "specific") & np.isin(
                    cur_sim_data.rel_id.values, self.rels[cur_event]
                )
                cur_sim_data = cur_sim_data.loc[cur_mask]

                self.n_samples_event.append(
                    self.site_combs[cur_event].shape[0] * self.n_rels
                )
            else:
                self.rels[cur_event] = None
                self.n_samples_event.append(self.site_combs[cur_event].shape[0])

            # Get observation data
            cur_obs_data = db.get_obs_data(cur_event, cur_sites)

            # Sanity checks
            assert np.all(cur_obs_data.columns == self.pSA_keys)
            assert (
                cur_sim_data.shape[0] == cur_obs_data.shape[0]
                or cur_sim_data.shape[0] == cur_obs_data.shape[0] * self.n_rels
            )

            self.sim_im_dfs[cur_event] = cur_sim_data
            self.obs_im_dfs[cur_event] = cur_obs_data


        # Compute the number of samples per event
        self.n_samples_event = np.asarray(self.n_samples_event)
        self._cum_n_samples = np.cumsum(self.n_samples_event)
        self.n_rels_used = {cur_event: 1 if cur_rels is None else cur_rels.size for cur_event, cur_rels in self.rels.items()}

        # Only keep columns that are relevant
        self.station_df = self.station_df.loc[:, self.site_features]

        # Ensure pre-processing has been applied
        assert all(
            [
                np.isclose(self.station_df[cur_feature].mean(), 0)
                for cur_feature in self.site_features
            ]
        )
        assert all(
            [
                np.isclose(self.station_df[cur_feature].std(), 1)
                for cur_feature in self.site_features
            ]
        )

        # Scale the (used) site-to-site distances
        # such that they are between -1 and 1
        # as per the maximum allowed site-to-site
        # distance when computing the site combinations
        self.scaled_dist_matrix = dist_matrix.copy()
        self.scaled_dist_matrix = ((self.scaled_dist_matrix / max_dist) * 2) - 1

        # Create feature matrix for all site combinations
        self.feature_tensor = np.full(
            (self.all_sites.size, self.all_sites.size, len(self.site_features) * 2 + 1),
            fill_value=np.nan,
        )
        n_site_features = len(self.site_features)
        for i, site_i in enumerate(self.all_sites):
            cur_vals = self.station_df.loc[site_i, self.site_features]
            self.feature_tensor[i, :, : len(self.site_features)] = cur_vals
            self.feature_tensor[
                :, i, n_site_features : n_site_features + n_site_features
            ] = cur_vals
        self.feature_tensor[:, :, -1] = self.scaled_dist_matrix.values

        # Organize the sim response spectra such that it is
        # in the format [n_rels, n_periods, n_sites]
        # per event
        self.sim_im_data = {}
        # And observed response spectra in the format
        # [n_periods, n_sites]
        self.obs_im_data = {}
        for cur_event in self.events:
            cur_sites = self.event_sites[cur_event]
            cur_sim_df = self.sim_im_dfs[cur_event]

            if self.rels[cur_event] is None:
                self.sim_im_data[cur_event] = (
                    cur_sim_df.set_index("site_id")
                    .loc[cur_sites, pSA_keys]
                    .values.T[np.newaxis, ...]
                )
            else:
                assert cur_sim_df.shape[0] == cur_sites.size * self.n_rels
                self.sim_im_data[cur_event] = np.stack(
                    [
                        cur_sim_df.loc[cur_sim_df.rel_id == cur_rel]
                        .set_index("site_id")
                        .loc[cur_sites, pSA_keys]
                        .T.values
                        for cur_rel in self.rels[cur_event]
                    ],
                    axis=0,
                )

            self.obs_im_data[cur_event] = (
                self.obs_im_dfs[cur_event].loc[cur_sites, pSA_keys].values.T
            )

        # Create similarity score lookup
        # with format [n_rels, n_sites]
        # per event
        self.sim_score = {}
        for cur_event in self.events:
            cur_event_scores = []
            for i, site_i in enumerate(self.event_sites[cur_event]):
                site_int_obs = (
                    self.obs_im_dfs[cur_event].loc[site_i, self.pSA_keys].values
                )

                site_int_sim = self.sim_im_data[cur_event][:, :, i]
                cur_sim_scores = ss.similiarity_score(site_int_obs, site_int_sim)

                cur_event_scores.append(cur_sim_scores)

            self.sim_score[cur_event] = np.stack(cur_event_scores, axis=1)

        # Some more sanity checking
        for cur_event in self.events:
            assert self.sim_im_data[cur_event].shape[0] in [1, self.n_rels]
            assert (
                self.sim_im_data[cur_event].shape[2] == self.event_sites[cur_event].size
            )
            assert self.sim_score[cur_event].shape[0] in [1, self.n_rels]
            assert (
                self.sim_score[cur_event].shape[1] == self.event_sites[cur_event].size
            )

    @property
    def n_samples(self):
        return self._cum_n_samples[-1]

    def get_metadata(self, idx: int):
        """Get the metadata for a specific sample"""
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]
        rel = "NA" if self.rels[event] is None else self.rels[event][rel_ix]

        return (
            event,
            rel,
            site_int,
            site_obs,
        )

    def __len__(self):
        return self.n_samples

    def get_indices(self, idx: int):
        event_ix = np.argmin(idx // self._cum_n_samples)
        event = self.events[event_ix]
        n_rels = self.n_rels_used[event]

        site_ix = (idx % self._cum_n_samples[max(event_ix - 1, 0)]) // n_rels
        rel_ix = idx % n_rels

        return event, event_ix, site_ix, rel_ix

    def __getitem__(self, idx: int):
        # Break the index down
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]

        # Features
        site_int_sim = self.sim_im_data[event][rel_ix, :, site_int_ix]
        site_obs_sim = self.sim_im_data[event][rel_ix, :, site_obs_ix]
        site_obs_obs = self.obs_im_data[event][:, site_obs_ix]

        site_int_all_ix = self.all_sites_ix_lookup[site_int]
        site_obs_all_ix = self.all_sites_ix_lookup[site_obs]
        site_features = self.feature_tensor[site_int_all_ix, site_obs_all_ix, :]

        # Labels
        # sim_score = self.sim_score[event][rel_ix, site_int_ix]
        site_int_obs = self.obs_im_data[event][:, site_int_ix]

        return (
            np.log(site_int_sim),
            np.log(site_obs_sim),
            np.log(site_obs_obs),
            site_features,
            np.log(site_int_obs),
            self.dist_matrix.iat[site_int_all_ix, site_obs_all_ix],
        )


def preprocess_site_features(
    station_df: pd.DataFrame, site_features: Sequence[str], stats: pd.DataFrame = None
):
    """Performs normalisation pre-processing of the site features"""
    station_df = station_df.copy()
    stats_comp = {}
    for cur_feature in site_features:
        if stats is None:
            cur_mean, cur_std = (
                station_df[cur_feature].mean(),
                station_df[cur_feature].std(),
            )
            stats_comp[cur_feature] = {"mean": cur_mean, "std": cur_std}
            station_df[cur_feature] = (station_df[cur_feature] - cur_mean) / cur_std
        # Use given statistics
        else:
            station_df[cur_feature] = (
                station_df[cur_feature] - stats.loc[cur_feature, "mean"]
            ) / stats.loc[cur_feature, "std"]

    if stats is None:
        return station_df, pd.DataFrame(stats_comp)
    return station_df


class MetaDataDataset(Dataset):
    def __init__(self, main_dataset: ResponseSpectrumDataset):
        self.main_dataset = main_dataset

        self.sites = main_dataset.event_sites
        self.site_combs = main_dataset.site_combs
        self.events = main_dataset.events

    def __len__(self):
        return self.main_dataset.n_samples

    def __getitem__(self, idx):
        return self.main_dataset.get_metadata(idx)
