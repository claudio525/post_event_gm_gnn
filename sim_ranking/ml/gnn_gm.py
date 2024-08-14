import numpy as np
import pandas as pd
import torch
import torch_geometric.data as gdata
import tqdm


from ..db import DB
from . import data as ml_data


def get_graph_data(
    db: DB,
    event_sites: dict[str, np.ndarray],
    event_site_combs: dict[str, np.ndarray],
    scalar_features: ml_data.ScalarFeatures,
    site_int_scalar_feature_keys: list[str],
    site_obs_scalar_feature_keys: list[str],
    edge_feature_keys: list[str],
    ims_mean: pd.DataFrame,
    ims_std: pd.DataFrame,
    ims: list[str],
):
    scalar_event_feature_values, scalar_feature_columns = (
        ml_data.create_scalar_feature_tensor(
            event_sites, scalar_features, event_site_combs
        )
    )

    graph_data = []
    for cur_event, cur_site_combs in tqdm.tqdm(event_site_combs.items()):
        cur_sites = event_sites[cur_event]
        cur_site_int_inds = np.unique(cur_site_combs[:, 0])

        cur_scalar_feature_values = scalar_event_feature_values[cur_event]

        # Get and normalise the IM data
        cur_im_data = db.get_obs_data(cur_event, cur_sites)
        cur_im_data = np.log(cur_im_data.loc[:, ims])
        # cur_norm_im_data = (cur_im_data - ims_mean) / ims_std

        for cur_site_int_ix in cur_site_int_inds:
            cur_site_combs_mask = cur_site_combs[:, 0] == cur_site_int_ix
            cur_site_int = cur_sites[cur_site_combs[cur_site_combs_mask, 0][0]]
            cur_obs_sites = cur_sites[cur_site_combs[cur_site_combs_mask, 1]]

            # Create the site_int node features
            cur_site_int_features = cur_scalar_feature_values.loc[
                cur_site_combs_mask, site_int_scalar_feature_keys
            ].values[0]

            # Create the site_obs node features
            cur_obs_sites_features = cur_scalar_feature_values.loc[
                cur_site_combs_mask, site_obs_scalar_feature_keys
            ].values
            # Add the IM values
            cur_obs_sites_features = np.concatenate(
                (
                    cur_obs_sites_features,
                    cur_im_data.loc[cur_obs_sites, ims].values,
                ),
                axis=1,
            )

            # Create the edge features
            cur_edge_features = cur_scalar_feature_values.loc[
                cur_site_combs_mask, edge_feature_keys
            ].values

            cur_sc_data = gdata.HeteroData()
            cur_sc_data["site_int"].x = torch.tensor(
                cur_site_int_features, dtype=torch.float32
            )[None, :]
            cur_sc_data["site_obs"].x = torch.tensor(
                cur_obs_sites_features, dtype=torch.float32
            )

            cur_sc_data["site_obs", "informs", "site_int"].edge_index = torch.tensor(
                [[ix, 0] for ix, cur_obs_site in enumerate(cur_obs_sites)],
                dtype=torch.long,
            ).T
            cur_sc_data["site_obs", "informs", "site_int"].edge_attr = torch.tensor(
                cur_edge_features, dtype=torch.float32
            )

            cur_sc_data["metadata"] = {
                "event": cur_event,
                "site_int": cur_site_int,
                "obs_sites": cur_obs_sites,
            }

            cur_sc_data["y"] = torch.tensor(
                cur_im_data.loc[cur_site_int, ims].values, dtype=torch.float32
            )

            graph_data.append(cur_sc_data)

    return graph_data
