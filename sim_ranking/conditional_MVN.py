from typing import Sequence, Dict
from pathlib import Path
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd

import gmhazard_calc as gc
import spatial_hazard as sh


@dataclass
class ConditionalMVNDistribution:
    IMs: Sequence[gc.im.IM]
    stations: Sequence[str]
    rupture: str
    cond_lnIM_mean_df: pd.DataFrame
    cond_lnIM_std_df: pd.DataFrame
    cond_lnIM_results: Dict[gc.im.IM, sh.im_dist.CondLnIMDistributionResult]

    def save(self, output_ffp: Path):
        with output_ffp.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, data_ffp: Path):
        with data_ffp.open("rb") as f:
            return pickle.load(f)


def compute_cond_MVN_distributions(
    IMs: Sequence[gc.im.IM],
    obs_df: pd.DataFrame,
    gmm_params_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    sim_data: Dict[str, pd.DataFrame],
    int_stations: Sequence[str],
):
    # Sanity checks
    assert np.unique(obs_df["evid"]).size == 1

    IMs_str = [str(cur_im) for cur_im in IMs]

    # Get the hypocentre location & event id
    hypo_loc = tuple(obs_df[["ev_lon", "ev_lat"]].iloc[0].values)
    rutpure = obs_df["evid"].values[0]

    # Only need IMs from here
    obs_df = obs_df.loc[:, IMs_str]

    # Compute the conditional distribution for all sites of interest
    # and all IMs
    cond_lnIM_results = {}
    for cur_im in IMs:
        print(f"\nComputing conditional MVN for {cur_im}")
        im_columns = [
            f"{str(cur_im)}_mean",
            f"{str(cur_im)}_std_Total",
            f"{str(cur_im)}_std_Inter",
            f"{str(cur_im)}_std_Intra",
        ]
        cur_gmm_params_df = gmm_params_df[im_columns]
        cur_gmm_params_df.columns = [
            "mu",
            "sigma_total",
            "sigma_between",
            "sigma_within",
        ]

        cond_lnIM_results[cur_im] = sh.im_dist.compute_cond_lnIM(
            cur_im,
            int_stations,
            stations_df,
            cur_gmm_params_df,
            np.log(obs_df[str(cur_im)]),
            hypo_loc,
            allow_obs_sites=True,
        )

    # Check that results have the same stations
    assert np.all(
        [
            np.all(
                cur_result.cond_lnIM_df.index
                == cond_lnIM_results[cur_im].cond_lnIM_df.index
            )
            for cur_result in cond_lnIM_results.values()
        ]
    )

    # Update sites of interest to the ones with results
    int_stations = cond_lnIM_results[cur_im].cond_lnIM_df.index.values.astype(str)

    # Drop any observation for which results were not computed
    # obs_df = obs_df.loc[np.isin(obs_df.index.values, cond_lmIM_results[cur_im].cond_lnIM_df.index)]
    # assert np.all(obs_df.index == cond_lmIM_results[cur_im].cond_lnIM_df.index)

    # Combine the cMVN data
    cond_lnIM_mean_df = pd.DataFrame.from_dict(
        {
            str(cur_im): cur_result.cond_lnIM_df["mu"]
            for cur_im, cur_result in cond_lnIM_results.items()
        }
    )
    cond_lnIM_std_df = pd.DataFrame.from_dict(
        {
            str(cur_im): cur_result.cond_lnIM_df["sigma"]
            for cur_im, cur_result in cond_lnIM_results.items()
        }
    )

    return ConditionalMVNDistribution(
        IMs, int_stations, rutpure, cond_lnIM_mean_df, cond_lnIM_std_df, cond_lnIM_results
    )
