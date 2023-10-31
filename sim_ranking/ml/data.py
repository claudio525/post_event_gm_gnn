from typing import Dict, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from ..db import DB


@dataclass
class ScalarFeatures:
    site_features_data: pd.DataFrame
    site_feature_keys: Sequence[str]
    site_to_site_features_data: Dict[str, pd.DataFrame]
    site_to_site_feature_keys: Sequence[str]
    event_site_features_data: Dict[str, pd.DataFrame]
    event_site_feature_keys: Sequence[str]
    event_site_to_site_features_data: Dict[str, Dict[str, pd.DataFrame]]
    event_site_to_site_feature_keys: Sequence[str]

    def __post_init__(self):
        self.n_scalar_features = (
            len(self.site_feature_keys) * 2
            + len(self.site_to_site_feature_keys)
            + len(self.event_site_feature_keys) * 2
            + len(self.event_site_to_site_feature_keys)
        )


@dataclass
class WeightScalarFeatures:
    site_to_site_features_data: Dict[str, pd.DataFrame]
    site_to_site_feature_keys: Sequence[str]
    event_site_to_site_features_data: Dict[str, Dict[str, pd.DataFrame]]
    event_site_to_site_feature_keys: Sequence[str]

    def __post_init__(self):
        self.n_scalar_features = len(self.site_to_site_feature_keys) + len(
            self.event_site_to_site_feature_keys
        )


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

        # Need at least two sites for the event
        if len(cur_sites) < 2:
            continue

        # Filter for the current event sites
        # and site-combinations less than max_dist km apart
        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]
        cur_dist_mask = (cur_dist_matrix.values < max_dist) & (
            cur_dist_matrix.values > 0
        )
        cur_row_ind, cur_col_ind = np.nonzero(cur_dist_mask)

        # Need at least one site combination within the
        # specified distance requirements
        if cur_row_ind.size == 0:
            continue

        # Get the site combinations
        # First is the site of interest, second is the observation site
        # Indices into the sites to use for the current event
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)

        site_combs[cur_event] = cur_site_combs
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites


def _get_event_sim_obs_pSA_data(
    event_sites: Dict[str, np.ndarray],
    site_combs: Dict[str, np.ndarray],
    db: DB,
    pSA_keys: np.ndarray,
    n_rels: int,
):
    """
    Gets the simulation and observation pSA data
    for each event, in addition to the number of
    samples per event and the realisations used
    """
    sim_im_dfs, obs_im_dfs = {}, {}
    n_samples_event, rels = [], {}
    for cur_event, cur_sites in event_sites.items():
        cur_sim_data = db.get_sim_data(cur_event, cur_sites)
        # Only use a subset of the available realisations to
        # prevent over-fitting to these events
        if np.any(cur_sim_data.data_source == "specific"):
            rels[cur_event] = np.random.choice(
                cur_sim_data.rel_id.unique(), n_rels, replace=False
            )
            cur_mask = (cur_sim_data.data_source.values == "specific") & np.isin(
                cur_sim_data.rel_id.values, rels[cur_event]
            )
            cur_sim_data = cur_sim_data.loc[cur_mask]

            n_samples_event.append(site_combs[cur_event].shape[0] * n_rels)
        else:
            rels[cur_event] = None
            n_samples_event.append(site_combs[cur_event].shape[0])

        # Get observation data
        cur_obs_data = db.get_obs_data(cur_event, cur_sites)

        # Sanity checks
        assert np.all(cur_obs_data.columns == pSA_keys)
        assert (
            cur_sim_data.shape[0] == cur_obs_data.shape[0]
            or cur_sim_data.shape[0] == cur_obs_data.shape[0] * n_rels
        )

        sim_im_dfs[cur_event] = cur_sim_data
        obs_im_dfs[cur_event] = cur_obs_data

    return sim_im_dfs, obs_im_dfs, n_samples_event, rels


def _get_event_n_sampels(n_samples_event: Sequence[int], rels: Dict[str, np.ndarray]):
    """Computes the number of samples per event"""
    n_samples_event = np.asarray(n_samples_event)
    cum_n_samples = np.cumsum(n_samples_event)
    n_rels_used = {
        cur_event: 1 if cur_rels is None else cur_rels.size
        for cur_event, cur_rels in rels.items()
    }

    return n_samples_event, cum_n_samples, n_rels_used


def _station_df_sanity_check(station_df: pd.DataFrame, site_features: Sequence[str]):
    """Checks that the station_df has been normalised"""
    assert all(
        [np.isclose(station_df[cur_feature].mean(), 0) for cur_feature in site_features]
    )
    assert all(
        [np.isclose(station_df[cur_feature].std(), 1) for cur_feature in site_features]
    )



