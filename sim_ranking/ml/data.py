import warnings
from pathlib import Path
from typing import Dict, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from labelled_data_array import LabelledDataArray


@dataclass
class ScalarFeatures:
    event_features_data: pd.DataFrame
    event_feature_keys: Sequence[str]
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
            len(self.event_feature_keys)
            + len(self.site_feature_keys) * 2
            + len(self.site_to_site_feature_keys)
            + len(self.event_site_feature_keys) * 2
            + len(self.event_site_to_site_feature_keys)
        )


def compute_site_combinations(
    event_sites: Dict[str, np.ndarray],
    valid_event_int_sites: Dict[str, np.ndarray],
    events: Sequence[str],
    dist_matrix: pd.DataFrame,
    site_obs: np.ndarray,
    site_int: np.ndarray,
    max_dist: float,
    max_n_obs_sites: int,
    min_n_obs_sites: int,
):
    """
    Compute the site combinations for each event

    Returns allowed sites and site-combination
    indices (for allowed sites) for each event

    Parameters
    ----------
    event_sites: dict
        Sites available for each event
    valid_event_int_sites: dict
        Valid sites of interest for each event
    site_obs: np.ndarray
        Sites that are allowed to be used as observation sites
    site_int: np.ndarray
        Sites that are allowed to be used as sites of interest
    max_dist: float
        Maximum allowed distance between site of interest
        and nearest observation site
    max_n_obs_sites: int
        Maximum number of observation sites to use
    min_n_obs_sites: int
        Minimum number of observation sites required
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        # Current sites of interest
        cur_int_sites = valid_event_int_sites[cur_event]
        cur_int_sites = cur_int_sites[np.isin(cur_int_sites, site_int)]

        # Current observation sites
        cur_obs_sites = event_sites[cur_event]
        cur_obs_sites = cur_obs_sites[np.isin(cur_obs_sites, site_obs)]

        # All sites for the current event
        cur_sites = np.union1d(cur_int_sites, cur_obs_sites)

        # cur_sites = event_sites[cur_event]
        # cur_sites = cur_sites[
        #     np.isin(cur_sites, site_int) | np.isin(cur_sites, site_obs)
        # ]

        # Need at least one site of interest and one observation site
        if len(cur_int_sites) < 1 or len(cur_obs_sites) < 1:
            continue

        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]

        # No observation sites
        if cur_dist_matrix.shape[1] < 2:
            continue

        # Get observation sites such that minimum number of observations sites is satisfied
        neigh = NearestNeighbors(
            metric="precomputed", n_jobs=1
        )
        neigh.fit(cur_dist_matrix)
        dist, n_neigh_ind = neigh.kneighbors(
            n_neighbors=min(max_n_obs_sites + 1, cur_dist_matrix.shape[1]), X=cur_dist_matrix, return_distance=True
        )

        # Apply distance filter
        dist_mask = dist < max_dist
        dist_mask[:, 0] = False
        cur_row_ind = np.repeat(np.arange(0, len(cur_sites)), np.count_nonzero(dist_mask, axis=1))
        cur_col_ind = n_neigh_ind[dist_mask]

        # Get the site combinations
        # First is the site of interest, second is the observation site
        # Indices into the sites to use for the current event
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)
        # t = np.stack((ri, ci), axis=1)
        # assert all([np.any(np.all(t[i, :] == cur_site_combs, axis=1)) for i in range(t.shape[0])])

        # cur_site_combs = t

        # TODO: Add filter for minimum number of observation sites

        # Filter based on allowed observation sites and sites of interest
        cur_mask = np.isin(cur_sites[cur_site_combs[:, 1]], cur_obs_sites) & np.isin(
            cur_sites[cur_site_combs[:, 0]], cur_int_sites
        )

        # Filter for minimum number of observation sites
        if min_n_obs_sites > 1:
            cur_int_ind, cur_int_count = np.unique(cur_site_combs[:, 0], return_counts=True)
            cur_valid_int_ind = cur_int_ind[cur_int_count >= min_n_obs_sites]
            cur_mask &= np.isin(cur_site_combs[:, 0], cur_valid_int_ind)

        if np.count_nonzero(cur_mask) == 0:
            continue

        site_combs[cur_event] = cur_site_combs[cur_mask]
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites


def create_scalar_feature_tensor(
    event_sites: Dict[str, np.ndarray],
    scalar_features: ScalarFeatures,
    event_site_combs: Dict[str, np.ndarray],
):
    """
    Create feature matrix for all site combinations
    of shape [n_event_site_combinations, n_features]
    Order of the features is:
        1) Event features
        2) Site of interest - site features
        3) Observation site - site features
        4) Site of interest - event site features
        5) Observation site - event site features
        6) Site to site features
        7) Event site to site features
    """
    events = np.asarray(list(event_sites.keys()))
    # assert np.all(np.asarray(list(event_sites.keys())) == events)

    scalar_feature_columns = np.asarray(
        [
            *scalar_features.event_feature_keys,
            *[f"{cur_key}_site_int" for cur_key in scalar_features.site_feature_keys],
            *[f"{cur_key}_site_obs" for cur_key in scalar_features.site_feature_keys],
            *[
                f"{cur_key}_site_int"
                for cur_key in scalar_features.event_site_feature_keys
            ],
            *[
                f"{cur_key}_site_obs"
                for cur_key in scalar_features.event_site_feature_keys
            ],
            *scalar_features.site_to_site_feature_keys,
            *scalar_features.event_site_to_site_feature_keys,
        ]
    )

    scalar_features_values = {}
    for cur_event in events:
        cur_sites = event_sites[cur_event]
        cur_site_combs = event_site_combs[cur_event]
        cur_site_ints = cur_sites[cur_site_combs[:, 0]]
        cur_site_obs = cur_sites[cur_site_combs[:, 1]]

        cur_tensor = np.full(
            (
                cur_site_combs.shape[0],
                scalar_features.n_scalar_features,
            ),
            fill_value=np.nan,
        )

        # Set the event features
        n_event_features = len(scalar_features.event_feature_keys)
        cur_tensor[:, :n_event_features] = scalar_features.event_features_data.loc[
            cur_event, scalar_features.event_feature_keys
        ].values

        # Set the site features
        cur_f_ix = n_event_features
        n_site_features = len(scalar_features.site_feature_keys)
        cur_tensor[:, cur_f_ix : cur_f_ix + n_site_features] = (
            scalar_features.site_features_data.loc[
                cur_site_ints, scalar_features.site_feature_keys
            ]
        )
        cur_tensor[:, cur_f_ix + n_site_features : cur_f_ix + n_site_features * 2] = (
            scalar_features.site_features_data.loc[
                cur_site_obs, scalar_features.site_feature_keys
            ]
        )

        # Set the event site features
        cur_f_ix += n_site_features * 2
        n_event_site_features = len(scalar_features.event_site_feature_keys)
        cur_tensor[:, cur_f_ix : cur_f_ix + n_event_site_features] = (
            scalar_features.event_site_features_data[cur_event].loc[
                cur_site_ints, scalar_features.event_site_feature_keys
            ]
        )
        cur_tensor[
            :, cur_f_ix + n_event_site_features : cur_f_ix + n_event_site_features * 2
        ] = scalar_features.event_site_features_data[cur_event].loc[
            cur_site_obs, scalar_features.event_site_feature_keys
        ]

        # Set the site to site features
        cur_f_ix += n_event_site_features * 2
        for i, feature_i in enumerate(scalar_features.site_to_site_feature_keys):
            cur_feature_df = scalar_features.site_to_site_features_data[feature_i]

            cur_tensor[:, cur_f_ix + i] = cur_feature_df.values[
                cur_feature_df.index.get_indexer_for(cur_site_ints),
                cur_feature_df.columns.get_indexer_for(cur_site_obs),
            ]

        # Set the event site to site features
        cur_f_ix += len(scalar_features.site_to_site_feature_keys)
        for i, feature_i in enumerate(scalar_features.event_site_to_site_feature_keys):
            cur_feature_df = scalar_features.event_site_to_site_features_data[
                feature_i
            ][cur_event]

            cur_tensor[:, cur_f_ix + i] = cur_feature_df.values[
                cur_feature_df.index.get_indexer_for(cur_site_ints),
                cur_feature_df.columns.get_indexer_for(cur_site_obs),
            ]

        scalar_features_values[cur_event] = pd.DataFrame(
            data=cur_tensor, columns=scalar_feature_columns
        )
        # scalar_features_values.append(cur_tensor)

    # scalar_features_values = np.concatenate(scalar_features_values, axis=0)
    return scalar_features_values, scalar_feature_columns


def get_valid_site_ints(
    event_sites: Dict[str, np.ndarray],
    record_df: pd.DataFrame,
):
    """
    Gets the list of site of interests per event that experience
    strong enough GM to be of interest.

    Based on Bradley 2013 model, with a threshold of
    PGA > 0.01g for Magnitude 6 event

    Parameters
    ----------
    event_sites: dict
        Available sites per event
    record_df: Dataframe
        Record data

    Returns
    -------
    valid_int_sites: np.ndarray
        All sites that are valid sites for
        at least one event
    valid_event_int_sites: dict
        Valid sites of interests per event
    """
    from empirical.util.classdef import TectType, GMM
    from empirical.util.openquake_wrapper_vectorized import oq_run

    # Check for nans
    assert np.all(~record_df.isna())

    # Create the rupture dataframe
    rupture_df = record_df.copy(True)

    # Constant inputs
    rupture_df["mag"] = 6.0
    rupture_df["rake"] = 45.0
    rupture_df["dip"] = 45.0
    rupture_df["z_tor"] = 0.0

    # Rename the columns to be in line what openquake expects
    rupture_df = rupture_df.rename(
        columns={
            "r_x": "rx",
            "z1p0": "z1pt0",
        }
    )
    rupture_df["vs30measured"] = True

    # Get PGA results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pga_result = oq_run(
            GMM.Br_10,
            TectType.ACTIVE_SHALLOW,
            rupture_df,
            "PGA",
        )
    pga_result.index = rupture_df.index

    assert np.all(pga_result.index == rupture_df.index)
    pga_result["event_id"] = rupture_df["event_id"]
    pga_result["site_id"] = rupture_df["site_id"]

    # Get the valid site of interests
    pga_result = pga_result.loc[pga_result["PGA_mean"] >= np.log(0.01)]
    valid_event_int_sites = {
        cur_event: np.intersect1d(
            cur_df.site_id.values.astype(str), event_sites[cur_event]
        )
        for cur_event, cur_df in pga_result.groupby("event_id")
    }
    valid_int_sites = np.unique(pga_result.site_id.values.astype(str))

    valid_record_ids = pga_result.index.values.astype(str)

    print(f"Valid SOI records: {pga_result.shape[0]}/{record_df.shape[0]}")
    return valid_int_sites, valid_event_int_sites, valid_record_ids


def load_cv_metrics(results_dir: Path):
    """
    Load the cross-validation metrics

    Parameters
    ----------
    results_dir: Path
        Path to the CV results directory

    Returns
    -------
    lda: LabelledDataArray
        Labelled data array of the metrics
    """
    metrics = pd.read_pickle(results_dir / "metrics.pickle")

    cv_keys = list(metrics.keys())
    metric_keys = list(metrics[cv_keys[0]].keys())

    results = []
    for cur_cv in cv_keys:
        cur_cv_results = []
        for cur_metric in metric_keys:
            cur_cv_results.append(metrics[cur_cv][cur_metric])

        results.append(np.stack(cur_cv_results, axis=1))

    data = np.stack(results, axis=2)
    lda = LabelledDataArray(
        data,
        (np.arange(data.shape[0]), metric_keys, cv_keys),
        ("epoch", "metric", "cv_iter"),
    )

    return lda
