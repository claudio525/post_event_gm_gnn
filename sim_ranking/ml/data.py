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
    site_obs: np.ndarray,
    site_int: np.ndarray,
    max_dist: float = 100,
):
    """
    Compute the site combinations for each event

    Returns allowed sites and site-combination
    indices (for allowed sites) for each event

    Parameters
    ----------
    site_int: np.ndarray
        The site of interests that are allowed to be used
        Any site not in this array is not used as site of interest,
        can be used as observation site though
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        cur_sites = sites[cur_event]
        cur_sites = cur_sites[np.isin(cur_sites, site_int) | np.isin(cur_sites, site_obs)]

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

        # Filter based on allowed observation sites and sites of interest
        cur_mask = np.isin(cur_sites[cur_site_combs[:, 1]], site_obs) & np.isin(
            cur_sites[cur_site_combs[:, 0]], site_int
        )

        site_combs[cur_event] = cur_site_combs[cur_mask]
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites


def create_scalar_feature_tensor(
    events: np.ndarray,
    event_sites: Dict[str, np.ndarray],
    scalar_features: ScalarFeatures,
):
    """
    Create feature matrix for all site combinations
    of shape [n_sites, n_sites, n_features]
    per event, where i/axis 0 = site of interest
    and j/axis 1 = observation site
    Order of the features is:
    1) Site of interest - site features
    2) Observation site - site features
    3) Site of interest - event site features
    4) Observation site - event site features
    5) Site to site features
    6) Event site to site features
    """
    assert np.all(np.asarray(list(event_sites.keys())) == events)

    scalar_features_tensors = {}
    for cur_event in events:
        cur_sites = event_sites[cur_event]
        cur_tensor = np.full(
            (
                cur_sites.size,
                cur_sites.size,
                scalar_features.n_scalar_features,
            ),
            fill_value=np.nan,
        )

        # Set the site features
        n_site_features = len(scalar_features.site_feature_keys)
        for i, feature_i in enumerate(scalar_features.site_feature_keys):
            for j, site_j in enumerate(cur_sites):
                # Site of interest/observation site
                cur_tensor[j, :, i] = cur_tensor[
                    :, j, i + n_site_features
                ] = scalar_features.site_features_data.loc[site_j, feature_i]
        # Set the event site features
        cur_f_ix = n_site_features * 2
        n_event_site_features = len(scalar_features.event_site_feature_keys)
        for i, feature_i in enumerate(scalar_features.event_site_feature_keys):
            for j, site_j in enumerate(cur_sites):
                # Site of interest/observation site
                cur_tensor[j, :, cur_f_ix + i] = cur_tensor[
                    :, j, cur_f_ix + i + n_event_site_features
                ] = scalar_features.event_site_features_data[cur_event].loc[
                    site_j, feature_i
                ]
        # Set the site to site features
        cur_f_ix += n_event_site_features * 2
        for i, feature_i in enumerate(scalar_features.site_to_site_feature_keys):
            cur_tensor[:, :, cur_f_ix + i] = scalar_features.site_to_site_features_data[
                feature_i
            ].loc[cur_sites, cur_sites]
        # Set the event site to site features
        cur_f_ix += len(scalar_features.site_to_site_feature_keys)
        for i, feature_i in enumerate(scalar_features.event_site_to_site_feature_keys):
            cur_tensor[
                :, :, cur_f_ix + i
            ] = scalar_features.event_site_to_site_features_data[feature_i][
                cur_event
            ].loc[
                cur_sites, cur_sites
            ]

        scalar_features_tensors[cur_event] = cur_tensor

    return scalar_features_tensors


def _station_df_sanity_check(station_df: pd.DataFrame, site_features: Sequence[str]):
    """Checks that the station_df has been normalised"""
    assert all(
        [np.isclose(station_df[cur_feature].mean(), 0) for cur_feature in site_features]
    )
    assert all(
        [np.isclose(station_df[cur_feature].std(), 1) for cur_feature in site_features]
    )
