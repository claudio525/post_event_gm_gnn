import os
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import torch

import ml_tools as mlt
import spatial_hazard as sh

from .. import constants
from ..db import DB
from . import data
from . import features


def prep_data(results_dir: Path):
    metadata = mlt.utils.load_yaml(results_dir / "meta.yaml")

    db_ff = Path(os.path.expandvars(metadata["data"]["db"]))
    db = DB(db_ff)

    event_df = db.get_event_df()
    record_df = db.get_record_df()

    event_sites = db.get_event_sites()

    sim_df = db.get_sim_df()
    obs_df = db.get_obs_df()

    print(f"Computing distance matrix")
    station_df = db.get_site_df()
    all_sites = db.get_avail_sites()
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    print(f"Pre-processing site features")
    site_features_df = features.preprocess_site_features(
        station_df,
        metadata["data"]["features"]["site_features"],
        pd.DataFrame.from_dict(metadata["data"]["features"]["site_feature_stats"]).T,
    )

    print(f"Computing scalar features")
    (
        site_to_site_features,
        event_site_features,
        event_site_to_site_features,
    ) = features.compute_scalar_features(
        event_df.index.values.astype(str),
        event_sites,
        event_df,
        station_df,
        record_df,
        dist_matrix,
        metadata["data"]["max_dist"],
    )

    scalar_features = data.ScalarFeatures(
        site_features_df,
        metadata["data"]["features"]["site_features"],
        site_to_site_features,
        metadata["data"]["features"]["site_to_site_features"],
        event_site_features,
        metadata["data"]["features"]["event_site_features"],
        event_site_to_site_features,
        metadata["data"]["features"]["event_site_to_site_features"],
    )

    model = torch.load(results_dir / "model.pt", map_location=torch.device("cpu"))

    return scalar_features, sim_df, obs_df, model, metadata


def get_site_ranking(pred: np.ndarray, rel_combs: np.ndarray):
    """Gets the ranking of sites based on most comparisons won"""
    pred_class = (pred >= 0.5).astype(float)

    rels = np.unique(rel_combs)
    comps_won = pd.Series(
        data=[int(np.sum(pred_class[rel_combs[:, 0] == cur_rel])) for cur_rel in rels],
        index=rels,
    ).sort_values(ascending=False)

    return comps_won.index.values, comps_won


def get_site_prediction(
    event_id: str,
    site_int: str,
    site_obs: str,
    metadata: Dict,
    scalar_features: data.ScalarFeatures,
    model: torch.nn.Module,
    sim_df: pd.DataFrame,
    obs_df: pd.DataFrame,
):
    # Use lnIM
    sim_df.loc[:, constants.PSA_KEYS] = np.log(sim_df.loc[:, constants.PSA_KEYS])
    obs_df.loc[:, constants.PSA_KEYS] = np.log(obs_df.loc[:, constants.PSA_KEYS])

    ### Prep scalar features
    scalar_feature_tensor = []
    # Site features
    for cur_site_feature in scalar_features.site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.site_features_data.loc[site_int, cur_site_feature]
        )
    for cur_site_feature in scalar_features.site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.site_features_data.loc[site_obs, cur_site_feature]
        )
    # Event site features
    for cur_event_site_feature in scalar_features.event_site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.event_site_features_data[event_id].loc[
                site_int, cur_event_site_feature
            ]
        )
    for cur_event_site_feature in scalar_features.event_site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.event_site_features_data[event_id].loc[
                site_obs, cur_event_site_feature
            ]
        )
    # Site-to-site features
    for cur_site_to_site_feature in scalar_features.site_to_site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.site_to_site_features_data[cur_site_to_site_feature].at[
                site_int, site_obs
            ]
        )
    # Event site-to-site features
    for (
        cur_event_site_to_site_feature
    ) in scalar_features.event_site_to_site_feature_keys:
        scalar_feature_tensor.append(
            scalar_features.event_site_to_site_features_data[
                cur_event_site_to_site_feature
            ][event_id].at[site_int, site_obs]
        )
    scalar_feature_tensor = torch.tensor(
        scalar_feature_tensor, dtype=torch.float32, requires_grad=False
    )

    ### Prep IM data
    cur_sim_df = sim_df.loc[(sim_df.event_id == event_id)]
    rels = np.unique(cur_sim_df.rel_id.values.astype(str))

    pSA_tensor = []
    rel_combs = []
    for rel_1 in rels:
        for rel_2 in rels:
            if rel_1 == rel_2:
                continue

            int_sim_pSA_rel_1 = cur_sim_df.loc[
                (cur_sim_df.rel_id == rel_1) & (cur_sim_df.site_id == site_int),
                constants.PSA_KEYS,
            ].values
            int_sim_pSA_rel_2 = cur_sim_df.loc[
                (cur_sim_df.rel_id == rel_2) & (cur_sim_df.site_id == site_int),
                constants.PSA_KEYS,
            ].values

            obs_sim_pSA_rel_1 = cur_sim_df.loc[
                (cur_sim_df.rel_id == rel_1) & (cur_sim_df.site_id == site_obs),
                constants.PSA_KEYS,
            ].values
            obs_sim_pSA_rel_2 = cur_sim_df.loc[
                (cur_sim_df.rel_id == rel_2) & (cur_sim_df.site_id == site_obs),
                constants.PSA_KEYS,
            ].values

            obs_obs_pSA = obs_df.loc[
                (obs_df.event_id == event_id) & (obs_df.site_id == site_obs),
                constants.PSA_KEYS,
            ].values

            # Scenario 1
            pSA_tensor.append(
                torch.cat(
                    (
                        torch.tensor(
                            int_sim_pSA_rel_1, dtype=torch.float32, requires_grad=False
                        ),
                        torch.tensor(
                            int_sim_pSA_rel_2, dtype=torch.float32, requires_grad=False
                        ),
                        torch.tensor(
                            obs_sim_pSA_rel_1, dtype=torch.float32, requires_grad=False
                        ),
                        torch.tensor(
                            obs_sim_pSA_rel_2, dtype=torch.float32, requires_grad=False
                        ),
                        torch.tensor(
                            obs_obs_pSA, dtype=torch.float32, requires_grad=False
                        ),
                    ),
                )
            )
            rel_combs.append((rel_1, rel_2))

    pSA_tensor = torch.stack(pSA_tensor)

    # Normalize the pSA data
    pSA_mean = pd.Series(metadata["data"]["features"]["pSA_mean"])
    pSA_std = pd.Series(metadata["data"]["features"]["pSA_std"])
    pSA_tensor = (
        pSA_tensor
        - pSA_mean.loc[constants.PSA_KEYS].values[None, None, :].astype(np.float32)
    ) / pSA_std.loc[constants.PSA_KEYS].values[None, None, :].astype(np.float32)

    with torch.no_grad():
        model.eval()
        pred = model(
            pSA_tensor, scalar_feature_tensor[None, :].repeat((pSA_tensor.shape[0], 1))
        )
        pred = torch.nn.functional.sigmoid(pred).numpy(force=True)

    rel_combs = np.asarray(rel_combs)

    # Normalize predictions
    pred = normalize_preds(rel_combs, pred)

    return pred, rel_combs


def normalize_preds(rel_combs: np.ndarray, pred: np.ndarray):
    # Normalize such that P(i > j) + P(j > i) = 1
    rels = np.unique(rel_combs)
    for i in range(rels.size - 1):
        for j in range(i + 1, rels.size):
            # Conversion of matrix index to index into rel_combs
            ix_1 = i * (rels.size - 1) + (j - 1)
            ix_2 = (j * (rels.size - 1)) + i

            # Sanity check
            rel_1, rel_2 = rel_combs[ix_1], rel_combs[ix_2]
            assert rel_1[0] == rel_2[1] and rel_1[1] == rel_2[0]

            # Normalize
            pred[ix_1] = pred[ix_1] / (pred[ix_1] + pred[ix_2])
            pred[ix_2] = 1 - pred[ix_1]
            assert np.isclose(pred[ix_1] + pred[ix_2], 1)

    return pred
