import abc
import dataclasses
from typing import Sequence

import numpy as np
import pandas as pd

from .. import constants
from .. import data_classes as dc


def create_scenario_df(
    event_site_combs: dict[str, Sequence[str]],
    event_sites: dict[str, Sequence[str]],
    obs_data: dc.ObservedData,
    dist_matrix: pd.DataFrame = None,
    lb_corr_data: dc.LBSiteCorrelationData = None,
):
    """
    Creates a scenario dataframe, with each row corresponding to a scenario
    Should only be used for data analysis, not actually used in the
    model training process

    Parameters
    ----------
    event_site_combs: dict
        Site combinations to use per event
    event_sites: dict
        Sites per event
    obs_data: ObservedData
        Observed data
    dist_matrix: pd.DataFrame, optional
        Distance matrix, add distances to the dataframe if provided
    lb_corr_data: LBSiteCorrelationData, optional
        Site correlations based on Loth & Baker (2013)
    """
    results = {}
    for cur_event, cur_combs in event_site_combs.items():
        cur_site_int_inds = np.unique(cur_combs[:, 0])

        for cur_site_int_ix in cur_site_int_inds:
            cur_site_int = event_sites[cur_event][cur_site_int_ix]
            cur_obs_sites = event_sites[cur_event][
                cur_combs[cur_combs[:, 0] == cur_site_int_ix, 1]
            ]

            cur_result = [
                cur_event,
                cur_site_int,
                cur_obs_sites,
                len(cur_obs_sites),
                obs_data.record_df.loc[f"{cur_event}_{cur_site_int}", "rrup"],
                obs_data.event_df.loc[cur_event, "mag"],
            ]

            # Add site-to-site distances
            if dist_matrix is not None:
                cur_obs_distances = dist_matrix.loc[cur_site_int, cur_obs_sites].values
                cur_result.append(cur_obs_distances)
                cur_result.append(cur_obs_distances.min())
            else:
                cur_result.append(None)
                cur_result.append(None)

            # Add constraintness
            if lb_corr_data is not None:
                cur_result.append(
                    lb_corr_data.corr_data.sel[cur_site_int, :, :]
                    .loc[cur_obs_sites, constants.PSA_KEYS]
                    .sum(axis=0)
                    .mean()
                )
            else:
                cur_result.append(None)

            results[f"{cur_event}_{cur_site_int}"] = cur_result

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[
            "event",
            "site_int",
            "obs_sites",
            "n_obs_sites",
            "rrup",
            "mag",
            "obs_distances",
            "min_obs_distance",
            "constraintness",
        ],
    )


