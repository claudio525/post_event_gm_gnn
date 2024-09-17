from typing import Sequence, Dict

import einops
import numpy as np
import pandas as pd
from pyproj import Transformer

from . import data as ml_data
from . import gnn_gm
from .. import constants
from ..data_classes import ObservedData


def get_scalar_features(
    event_sites: dict[str, np.ndarray],
    obs_data: ObservedData,
    run_config: gnn_gm.RunConfig,
    scalar_feature_keys: dict[str, Sequence[str]],
    dist_matrix: pd.DataFrame,
):
    """Performs pre-processing of the data"""
    events = np.asarray(list(event_sites.keys()))

    event_df = obs_data.event_df.copy()
    record_df = obs_data.record_df.copy()
    site_df = obs_data.site_df.copy()

    ### Event features
    event_features_df = _pre_process_event_features(event_df, scalar_feature_keys["event"])

    ### Site features
    site_features_df = _pre_process_site_features(site_df, scalar_feature_keys["site"])

    ### Event-site features
    event_site_features_df = _pre_process_event_site_features(
        record_df.copy(), scalar_feature_keys["event_site"]
    )

    event_groups = event_site_features_df.groupby("event_id")
    event_site_features = {
        cur_event: cur_record_df.set_index("site_id")[scalar_feature_keys["event_site"]]
        for cur_event, cur_record_df in event_groups
    }

    ### Site-to-site features
    site_to_site_features = _compute_site_to_site_features(
        dist_matrix, run_config.max_dist
    )

    ### Event site-to-site features
    event_site_to_site_features = _compute_event_site_to_site_features(
        events, event_sites, event_df, site_df
    )

    scalar_features = ml_data.ScalarFeatures(
        event_features_df,
        scalar_feature_keys["event"],
        site_features_df,
        scalar_feature_keys["site"],
        site_to_site_features,
        scalar_feature_keys["site_to_site"],
        event_site_features,
        scalar_feature_keys["event_site"],
        event_site_to_site_features,
        scalar_feature_keys["event_site_to_site"],
    )
    return scalar_features


def _compute_site_to_site_features(dist_matrix: pd.DataFrame, max_dist: float):
    """Computes and pre-processes site-to-site features"""
    site_to_site_features = {}

    # Scale the (used) site-to-site distances
    # such that they are between -1 and 1
    # as per the maximum allowed site-to-site
    # distance when computing the site combinations
    site_to_site_features["dist"] = ((dist_matrix.copy() / max_dist) * 2) - 1

    return site_to_site_features


def _pre_process_site_features(site_df: pd.DataFrame, site_feature_keys: Sequence[str]):
    """Scales the site features to be between -1 and 1"""
    site_df = site_df.loc[:, site_feature_keys]
    for cur_key in site_feature_keys:
        cur_min, cur_max = constants.PRE_PROCESS_CONFIG[cur_key]
        site_df[cur_key] = 2 * (site_df[cur_key] - cur_min) / (cur_max - cur_min) - 1

    return site_df


def _pre_process_event_features(
    event_df: pd.DataFrame, event_feature_keys: Sequence[str]
):
    """Scales the event features to be between -1 and 1"""
    event_df = event_df.loc[:, event_feature_keys]
    for cur_key in event_feature_keys:
        cur_min, cur_max = constants.PRE_PROCESS_CONFIG[cur_key]
        event_df[cur_key] = 2 * (event_df[cur_key] - cur_min) / (cur_max - cur_min) - 1

    return event_df


def _compute_event_site_to_site_features(
    events: np.ndarray,
    event_sites: dict[str, np.ndarray],
    event_df: pd.DataFrame,
    site_df: pd.DataFrame,
):
    """
    Computes event site-to-site features
    """
    event_site_to_site_features = {}

    # Compute the site-to-site angle wrt. the epicentre
    event_site_to_site_features["angular_dist"] = compute_angular_distance(
        site_df, event_df, events, event_sites
    )

    return event_site_to_site_features


def _pre_process_event_site_features(
    record_df: pd.DataFrame, event_site_feature_keys: Sequence[str]
):
    """Pre-process event-site features"""
    for cur_key in event_site_feature_keys:
        cur_min, cur_max = constants.PRE_PROCESS_CONFIG[cur_key]
        record_df[cur_key] = (
            2 * (record_df[cur_key] - cur_min) / (cur_max - cur_min) - 1
        )

    return record_df


def compute_angular_distance(
    station_df: pd.DataFrame,
    event_df: pd.DataFrame,
    events: Sequence[str],
    event_sites: Dict[str, np.ndarray],
    pre_process: bool = True,
):
    """
    Computes the angle between the site-pairs
    (for each event) with respect to the epicentre
    """
    station_df = station_df.copy()
    event_df = event_df.copy()

    event_angular_distances = {}
    transformer = Transformer.from_crs(4326, 2193, always_xy=True)
    station_df["nztm_x"], station_df["nztm_y"] = transformer.transform(
        station_df.lon, station_df.lat
    )
    event_df["nztm_x"], event_df["nztm_y"] = transformer.transform(
        event_df.lon, event_df.lat
    )
    for cur_event in events:
        cur_epi = event_df.loc[cur_event, ["nztm_x", "nztm_y"]].values.astype(float)
        cur_sites = event_sites[cur_event]
        cur_site_coords = station_df.loc[cur_sites, ["nztm_x", "nztm_y"]].values

        source_to_site_vecs = cur_site_coords - cur_epi
        cur_site_to_site_angle = np.arccos(
            np.clip(
                einops.einsum(
                    source_to_site_vecs, source_to_site_vecs, "i k, j k -> i j"
                )
                / (
                    (
                        np.linalg.norm(source_to_site_vecs, axis=1)[:, None]
                        * np.linalg.norm(source_to_site_vecs, axis=1)[None, :]
                    )
                ),
                -1.0,
                1.0,
            )
        )

        # Scale such that -1 => 0 and 1 => pi
        if pre_process:
            cur_site_to_site_angle = (cur_site_to_site_angle / np.pi) * 2 - 1

        event_angular_distances[cur_event] = pd.DataFrame(
            data=cur_site_to_site_angle, index=cur_sites, columns=cur_sites
        )

    return event_angular_distances
