from typing import Sequence, Dict

import einops
import numpy as np
import pandas as pd
from pyproj import Transformer


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

        # t = np.full((cur_sites.size, cur_sites.size), np.nan)
        # for i in range(cur_sites.size):
        #     for j in range(cur_sites.size):
        #         if i == j:
        #             t[i, j] = 0.0
        #         else:
        #             v1 = cur_site_coords[i] - cur_epi
        #             v2 = cur_site_coords[j] - cur_epi
        #             t[i, j] = np.arccos(
        #                 np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #             )
        #
        # assert np.allclose(cur_site_to_site_angle, t)

        # Scale such that -1 => 0 and 1 => pi
        if pre_process:
            cur_site_to_site_angle = (cur_site_to_site_angle / np.pi) * 2 - 1

        event_angular_distances[cur_event] = pd.DataFrame(
            data=cur_site_to_site_angle, index=cur_sites, columns=cur_sites
        )

    return event_angular_distances


def pre_process_event_site_features(site_df: pd.DataFrame):
    """
    This is called per event

    R_Rup: Scaled such that -1 => 0, 0 => 100km and 1 => 200km
    R_X: Scaled such that -1 => -200km, 0 => 0km and 1 => 200km
    """
    event_site_feature_stats = {
        "rrup": (0, 200),
        "rx": (-200, 200),
    }
    supported_cols = list(event_site_feature_stats.keys())

    assert np.all(np.isin(site_df.columns, supported_cols)), "Unsupported features"

    for cur_f, (cur_min, cur_max) in event_site_feature_stats.items():
        site_df[cur_f] = ((site_df[cur_f] - cur_min) / (cur_max - cur_min)) * 2 - 1

    return site_df


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


def compute_scalar_features(
    events: np.ndarray,
    event_sites: Dict[str, np.ndarray],
    event_df: pd.DataFrame,
    station_df: pd.DataFrame,
    record_df: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    max_dist: float,
):
    ### Site-to-site features
    site_to_site_features = {}

    # Scale the (used) site-to-site distances
    # such that they are between -1 and 1
    # as per the maximum allowed site-to-site
    # distance when computing the site combinations
    site_to_site_features["dist"] = ((dist_matrix.copy() / max_dist) * 2) - 1

    ### Event-site features
    event_groups = record_df.groupby("event_id")
    event_site_features = {
        cur_event: pre_process_event_site_features(
            cur_record_df.set_index("site_id")[["rrup", "rx"]]
        )
        for cur_event, cur_record_df in event_groups
    }

    ### Event-site-to-site features
    event_site_to_site_features = {}

    # Compute the site-to-site angle wrt. the epicentre
    event_site_to_site_features["angular_dist"] = compute_angular_distance(
        station_df, event_df, events, event_sites
    )

    return site_to_site_features, event_site_features, event_site_to_site_features
