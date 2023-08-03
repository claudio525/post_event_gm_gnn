import copy
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

    def get_obs_stations(self, site: str):
        obs_sites = self.cond_lnIM_results[self.IMs[0]].obs_stations_mask_df
        obs_sites = obs_sites.loc[site, obs_sites.loc[site, :]]
        return obs_sites.index.values.astype(str)


def run_conditional_mvn_ranking(
        output_dir: Path,
    stations_df: pd.DataFrame, IMs: Sequence[str], gm_params_df: pd.DataFrame,
        sim_data,
        obs_df: pd.DataFrame,
        int_stations: np.ndarray,
        R: Dict[str, pd.DataFrame] = None
):
    # Compute the conditional MVN distributions for each IM
    IMs_str = IMs
    IMs = [gc.im.IM.from_str(cur_im) for cur_im in IMs]

    cMVNs_result = compute_cond_MVN_distributions(
        IMs,
        obs_df,
        gm_params_df,
        stations_df,
        int_stations,
        R=R,
    )

    # Compute the misfit for each site of interest
    site_misfits = []
    for cur_site in int_stations:
        if (cur_sim_df := sim_data.get(cur_site)) is None:
            print(f"No simulation data available for site: {cur_site}, skipping")
            continue

        # Compute misfit for each IM
        cur_misfit = (
            cMVNs_result.cond_lnIM_mean_df.loc[cur_site, IMs_str].values
            - np.log(cur_sim_df[IMs_str].values)
        ) ** 2

        # Aggregate along IM axis
        site_misfits.append(
            pd.Series(
                index=cur_sim_df.index, data=cur_misfit.sum(axis=1), name=cur_site
            )
        )

    # Combine
    site_misfits_df = pd.concat(site_misfits, axis=1)

    # Select the best realisation for each site
    best_sim_id = pd.Series(
        data=site_misfits_df.index[np.argmin(site_misfits_df.values, axis=0)],
        index=site_misfits_df.columns,
    )

    # Save the results
    cMVNs_result.save(output_dir / "cMVN_distributions.pickle")
    site_misfits_df.to_csv(output_dir / "site_misfits.csv")
    best_sim_id.to_csv(output_dir / "best_sim_ids.csv")


def compute_cond_MVN_distributions(
    IMs: Sequence[gc.im.IM],
    obs_df: pd.DataFrame,
    gmm_params_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    int_stations: np.ndarray,
    R: Dict[str, pd.DataFrame] = None,
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
            R=R[str(cur_im)] if R is not None else None,
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
        IMs,
        int_stations,
        rutpure,
        cond_lnIM_mean_df,
        cond_lnIM_std_df,
        cond_lnIM_results,
    )
