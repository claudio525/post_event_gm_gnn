from typing import Dict, Sequence

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from . import similarity_score as ss


def compute_site_combinations(
    sites: Dict[str, np.ndarray],
    events: Sequence[str],
    dist_matrix: pd.DataFrame,
    sites_to_use: np.ndarray = None,
):
    """
    Compute the site combinations for each event

    Note: The resulting indices are into the index/columns
    of the distance matrix,
    i.e. all stations, NOT the event specific sites
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        cur_sites = sites[cur_event]
        cur_sites = (
            cur_sites
            if sites_to_use is None
            else cur_sites[np.isin(cur_sites, sites_to_use)]
        )

        # Filter for the current event sites
        # and site-combinations less than 100km apart
        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]
        cur_dist_mask = (cur_dist_matrix.values < 100) & (cur_dist_matrix.values > 0)
        cur_row_ind, cur_col_ind = np.nonzero(cur_dist_mask)

        # Get the site combinations
        # First is the site of interest, second is the observation site
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)

        site_combs[cur_event] = cur_site_combs
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites




class ResponseSpectrumDataset(Dataset):
    def __init__(
        self,
        sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        events: Sequence[str],
        obs_im_data: Dict[str, pd.DataFrame],
        sim_im_data: Dict[str, pd.DataFrame],
        rels: Dict[str, np.ndarray],
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        dist_matrix: pd.DataFrame,
        site_features: Sequence[str],
        max_dist: float = 100,
        sim_score_fn=ss.similiarity_score,
    ):
        self.sim_score_fn = sim_score_fn
        self.sites = sites
        self.site_combs = site_combs
        self.events = events
        self.rels = rels
        self.dist_matrix = dist_matrix
        self.site_features = site_features

        self.all_sites = dist_matrix.index.values.astype(str)
        self.all_sites_ix_lookup = {
            cur_site: ix for ix, cur_site in enumerate(self.all_sites)
        }

        # Some sanity checking
        for cur_event in sites.keys():
            assert sites[cur_event].size == site_combs[cur_event].max() + 1

        # Number of realisations have to be the same for all events
        # TODO: Could in theory be different set of realisation
        # per model epoch
        assert np.all(
            [rels[self.events[0]].size == rels[cur_event].size for cur_event in events]
        )
        self.n_rels_used = rels[self.events[0]].size

        # Compute the number of samples per event
        self.n_samples_event = np.asarray(
            [
                self.site_combs[cur_event].shape[0] * self.n_rels_used
                for cur_event in self.events
            ]
        )
        self._cum_n_samples = np.cumsum(self.n_samples_event)

        self.pSA_keys = pSA_keys
        self.periods = periods

        # Observed IM data
        # Only keep columns that are relevant, to save
        # column lookup in __getitem__
        self.obs_im_data = obs_im_data.copy()
        self.obs_im_data = {
            cur_event: cur_im_data.loc[:, self.pSA_keys]
            for cur_event, cur_im_data in obs_im_data.items()
        }

        # Normalise the station data
        self.station_df = station_df.copy()
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
        for cur_event in self.events:
            cur_sites = self.sites[cur_event]

            cur_sim_im_data = []
            for cur_site in cur_sites:
                cur_sim_im_data.append(
                    sim_im_data[cur_event][cur_site].loc[self.rels[cur_event], pSA_keys]
                )

            self.sim_im_data[cur_event] = np.stack(cur_sim_im_data, axis=2)

        # Create similarity score lookup
        # with format [n_rels, n_sites]
        # per event
        self.sim_score = {}
        for cur_event in self.events:
            cur_event_scores = []
            for i, site_i in enumerate(self.sites[cur_event]):
                site_int_obs = (
                    self.obs_im_data[cur_event].loc[site_i, :].values.astype(float)
                )
                site_int_sim = self.sim_im_data[cur_event][:, :, i]
                cur_sim_scores = ss.similiarity_score(site_int_obs, site_int_sim)

                cur_event_scores.append(cur_sim_scores)

            self.sim_score[cur_event] = np.stack(cur_event_scores, axis=1)

        # Some more sanity checking
        for cur_event in self.events:
            assert self.sim_im_data[cur_event].shape[0] == self.n_rels_used
            assert self.sim_im_data[cur_event].shape[2] == self.sites[cur_event].size
            assert self.sim_score[cur_event].shape[0] == self.n_rels_used
            assert self.sim_score[cur_event].shape[1] == self.sites[cur_event].size

    @property
    def n_samples(self):
        return self._cum_n_samples[-1]

    def get_metadata(self, idx: int):
        """Get the metadata for a specific sample"""
        event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        event = self.events[event_ix]
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.sites[event][site_int_ix]
        site_obs = self.sites[event][site_obs_ix]
        rel = self.rels[event][rel_ix]

        return event, rel, site_int, site_obs,

    def __len__(self):
        return self.n_samples

    def get_indices(self, idx: int):
        event_ix = np.argmin(idx // self._cum_n_samples)
        site_ix = (idx % self._cum_n_samples[max(event_ix - 1, 0)]) // self.n_rels_used
        rel_ix = idx % self.n_rels_used

        return event_ix, site_ix, rel_ix

    def __getitem__(self, idx: int):
        # Break the index down
        event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        event = self.events[event_ix]
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.sites[event][site_int_ix]
        site_obs = self.sites[event][site_obs_ix]

        # Features
        site_int_sim = self.sim_im_data[event][rel_ix, :, site_int_ix]
        site_obs_sim = self.sim_im_data[event][rel_ix, :, site_obs_ix]
        site_obs_obs = self.obs_im_data[event].loc[site_obs].values.astype(float)

        site_int_all_ix = self.all_sites_ix_lookup[site_int]
        site_obs_all_ix = self.all_sites_ix_lookup[site_obs]
        site_features = self.feature_tensor[site_int_all_ix, site_obs_all_ix, :]

        # Labels
        sim_score = self.sim_score[event][rel_ix, site_int_ix]

        return (
            site_int_sim,
            site_obs_sim,
            site_obs_obs,
            site_features,
            sim_score,
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

        self.sites = main_dataset.sites
        self.site_combs = main_dataset.site_combs
        self.events = main_dataset.events

    def __len__(self):
        return self.main_dataset.n_samples

    def __getitem__(self, idx):
        return self.main_dataset.get_metadata(idx)
