from typing import Sequence, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
import pickle

import einops
import numpy as np
import pandas as pd

import gmhazard_calc as gc
import spatial_hazard as sh

from . import utils


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
    stations_df: pd.DataFrame,
    IMs: Sequence[str],
    im_weights: np.ndarray,
    gm_params_df: pd.DataFrame,
    sim_data: Dict,
    obs_data: pd.DataFrame,
    int_stations: np.ndarray,
    source_info: utils.SourceInfo,
    R: Dict[str, pd.DataFrame] = None,
    min_n_obs_stations: int = 5,
    n_obs_stations: int = 20,
    verbose: bool = True,
):
    """
    Computes the conditional IM distributions

    Parameters
    ----------
    output_dir: Path
    stations_df: DataFrame
    IMs: sequence of strings
        The IMs to compute the conditional MVN distributions
    im_weights: array of floats
        The weights for each IM
    gm_params_df: DataFrame
        The unconditional GM parameters, must have
        the columns for each IM:
            IM_mean, IM_std_Total, IM_std_Inter, IM_std_Intra
    sim_data: dict
        The IM values for the simulation realisations
        Dictionary with site as key and dataframe as value
    obs_data: DataFrame
        Observed IM values at the GM recording stations
    int_stations: array of strings
        Sites for which to perform cMVN ranking
        Can include observation sites in which case
        the observation data of that station is ignored
        during calculation
    source_info: SourceInfo
        The rupture source information
    R: dict of DataFrames, optional
        The within-event spatial correlation matrices
        for each IM
    n_obs_stations: int, optional
        The number of observation stations to use for
        the calculation of the conditional IM distributions

    Returns
    -------
    ConditionalMVNDistribution
    """
    # Compute the conditional MVN distributions for each IM
    IMs_str = IMs
    IMs = [gc.im.IM.from_str(cur_im) for cur_im in IMs]

    assert len(im_weights) == len(IMs_str)

    cIMs_result = compute_cond_MVN_distributions(
        IMs,
        obs_data,
        gm_params_df,
        stations_df,
        int_stations,
        source_info,
        R=R,
        min_n_obs_stations=min_n_obs_stations,
        n_obs_stations=n_obs_stations,
        verbose=verbose,
    )
    if cIMs_result is None:
        return None

    # Compute the realisation misfit for each site of interest
    rel_misfits = []
    for cur_site in int_stations:
        if (cur_sim_df := sim_data.get(cur_site)) is None:
            if verbose:
                print(f"No simulation data available for site: {cur_site}, skipping")
            continue

        # Fix the index
        cur_sim_df.index = np.char.replace(
            cur_sim_df.index.values.astype(str), f"_{cur_site}", ""
        )
        cur_sim_df = cur_sim_df.sort_index()

        # Compute misfit for each IM
        cur_rel_misfit = einops.einsum(
            im_weights,
            (
                cIMs_result.cond_lnIM_mean_df.loc[cur_site, IMs_str].values
                - np.log(cur_sim_df[IMs_str].values)
            )
            ** 2, "i, j i -> j",
        )

        # Aggregate along IM axis
        rel_misfits.append(
            pd.Series(
                index=cur_sim_df.index, data=cur_rel_misfit, name=cur_site
            )
        )

    # Combine
    rel_misfits_df = pd.concat(rel_misfits, axis=1)

    # # Select the best realisation for each site
    # best_sim_id = pd.Series(
    #     data=rel_misfits_df.index[np.argmin(rel_misfits_df.values, axis=0)],
    #     index=rel_misfits_df.columns,
    # )

    # Save the results
    cIMs_result.save(output_dir / "cMVN_distributions.pickle")
    rel_misfits_df.to_csv(output_dir / "rel_misfits.csv")
    # best_sim_id.to_csv(output_dir / "best_sim_ids.csv")


def compute_cond_MVN_distributions(
    IMs: Sequence[gc.im.IM],
    obs_data: pd.DataFrame,
    gmm_params_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    int_stations: np.ndarray,
    source_info: utils.SourceInfo,
    R: Dict[str, pd.DataFrame] = None,
    min_n_obs_stations: int = 5,
    n_obs_stations: int = 20,
    verbose: bool = True,
):
    IMs_str = [str(cur_im) for cur_im in IMs]

    # Get the hypocentre location & event id
    # hypo_loc = tuple(obs_data[["ev_lon", "ev_lat"]].iloc[0].values)
    # rutpure = obs_data["evid"].values[0]

    # Only need IMs from here
    obs_data = obs_data.loc[:, IMs_str]

    # Compute the conditional distribution for all sites of interest
    # and all IMs
    cond_lnIM_results = {}
    for cur_im in IMs:
        if verbose:
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
            np.log(obs_data[str(cur_im)]),
            source_info.hypo_loc,
            obs_site_filter_fn=sh.im_dist.get_nn_obs_site_filter_fn(
                min_n_obs_stations, n_obs_stations
            ),
            R=R[str(cur_im)] if R is not None else None,
            allow_obs_sites=True,
            verbose=verbose,
        )
        if cond_lnIM_results[cur_im] is None:
            if verbose:
                print(f"No results for {cur_im}, skipping")
            return None

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
        source_info.rupture_name,
        cond_lnIM_mean_df,
        cond_lnIM_std_df,
        cond_lnIM_results,
    )
