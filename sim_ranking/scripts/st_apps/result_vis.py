import os
import time
from typing import List, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import typer

import ml_tools as mlt
import gmhazard_calc as gc

import sim_ranking as sr
import spatial_hazard as sh


# @st.cache_data
# def _load_cmvn_results(results_dir: Path):
#     cMVN_result = sr.cmvn.ConditionalMVNDistribution.load(
#         results_dir / "cMVN_distributions.pickle"
#     )
#     sites = cMVN_result.stations
#
#     return cMVN_result, sites
#
#
# @st.cache_data
# def _load_best_sim_ids(results_dir: Path):
#     return pd.read_csv(results_dir / "best_sim_ids.csv", index_col=0).squeeze()
#
#
# @st.cache_data
# def _get_dist_matrix(sites: np.ndarray, stations_ffp: Path):
#     station_df = sr.data.load_ll_file(stations_ffp)
#
#     return sh.im_dist.calculate_distance_matrix(sites, station_df)
#
#
# @st.cache_data
# def _load_gm_params(gm_params_ffp: Path, event: str):
#     gm_params = pd.read_csv(gm_params_ffp, index_col=0)
#
#     if "event" in gm_params.columns:
#         gm_params.event = gm_params.event.values.astype(str)
#
#         gm_params = gm_params.loc[gm_params.event == event]
#         gm_params = gm_params.set_index("site")
#
#     return gm_params


### New


@st.cache_data
def _get_meta(results_dir: Path):
    meta = mlt.utils.load_yaml(results_dir / "meta.yaml")
    meta = {
        key: os.path.expandvars(val)
        if isinstance(val, str) and (key.endswith("_ffp") or key.endswith("_dir"))
        else val
        for key, val in meta.items()
    }
    return meta


def _get_method_type(results_dir: Path):
    return sr.constants.RankingMethod(_get_meta(results_dir)["method_type"])


@st.cache_data
def _load_cmvn_result(results_dir: Path):
    ffp = results_dir / "cMVN_distributions.pickle"
    if ffp.exists():
        return sr.cmvn.ConditionalMVNDistribution.load(ffp)
    return None


def _get_obs_data(results_dir: Path):
    meta = _get_meta(results_dir)
    return _load_obs_data(meta["obs_data_ffp"], meta["rupture"])


@st.cache_data
def _load_obs_data(obs_data_ffp: Path, rupture: str):
    return sr.data.load_obs_rupture_data(obs_data_ffp, rupture)


def _get_sim_data(results_dir: Path, sites: np.ndarray):
    meta = _get_meta(results_dir)
    return _load_sim_data(meta["sim_imdb_ffp"], sites)


@st.cache_data
def _load_sim_data(sim_imdb_ffp: Path, sites: np.ndarray):
    return sr.data.load_sim_data(sim_imdb_ffp, sites)


def _get_sites(results_dir: Path):
    sites = _load_cmvn_result(results_dir).stations
    method_type = _get_method_type(results_dir)

    if method_type is sr.constants.RankingMethod.emp_cMVN:
        sim_data = _get_sim_data(results_dir, sites)

        # Drop any sites for which there is no simulation data
        mask = np.isin(sites, list(sim_data.keys()))
        if np.any(~mask):
            print(
                f"Dropping the following sites as no simulation data exists:\n{sites[~mask]}"
            )
            sites = sites[mask]

    return sites


def _get_IMs(results_dir: Path):
    return _get_meta(results_dir)["IMs"]


def _get_periods(results_dir: Path):
    ims = _get_IMs(results_dir)

    pSA_keys = [cur_im for cur_im in ims if cur_im.startswith("pSA")]
    periods = [float(cur_im.split("_")[-1]) for cur_im in pSA_keys]
    sort_int = np.argsort(periods)
    periods = np.array(periods)[sort_int]
    pSA_keys = np.array(pSA_keys)[sort_int]

    return periods, pSA_keys


@st.cache_data
def _load_best_sim_ids(results_dir: Path):
    return pd.read_csv(results_dir / "best_sim_ids.csv", index_col=0).squeeze()


@st.cache_data
def _load_sim_gm_params(data_dir: Path):
    return sr.data.SimGMParams.load(data_dir)


@st.cache_data
def _load_emp_gm_params(gm_params_ffp: Path, event: str):
    gm_params = pd.read_csv(gm_params_ffp, index_col=0)

    gm_params.event = gm_params.event.values.astype(str)
    gm_params = gm_params.loc[gm_params.event == event]
    gm_params = gm_params.set_index("site")
    return gm_params


def _get_gm_params(results_dir: Path):
    method_type = _get_method_type(results_dir)
    meta = _get_meta(results_dir)
    if method_type is sr.constants.RankingMethod.emp_cMVN:
        return _load_emp_gm_params(meta["gm_params_ffp"], meta["rupture"])
    else:
        sim_gm_params = _load_sim_gm_params(Path(meta["sim_gm_params_dir"]))
        return sim_gm_params.gm_params


@st.cache_data
def _load_station_df(results_dir: Path):
    return sr.data.load_ll_file(_get_meta(results_dir)["stations_ll_ffp"])


@st.cache_data
def _load_dist_matrix(results_dir: Path, sites: Sequence[str]):
    station_df = _load_station_df(results_dir)

    return sh.im_dist.calculate_distance_matrix(sites, station_df)


def main(
    results_dir: Path,
):
    start_time = time.time()
    mlt.st_tools.utils.update_st_width(1600, 2, 0, 1, 1)

    col_1, col_2 = st.columns(2)

    # Event selection
    events = [cur_ffp.stem for cur_ffp in results_dir.iterdir() if cur_ffp.is_dir()]
    with col_1:
        event = st.selectbox("Event", events)

    # Method selection
    methods = [
        cur_ffp.stem
        for cur_ffp in (results_dir / event).iterdir()
        if cur_ffp.is_dir()
        and cur_ffp.stem in sr.constants.RESULTS_DIR_NAME_METHOD_MAPPING.keys()
    ]
    with col_2:
        method_dir = st.selectbox("Method", methods)

    cur_results_dir = results_dir / event / method_dir

    summary_tab, site_tab = st.tabs(["Summary", "Individual Site"])
    print(f"Took {time.time() - start_time} to run initial")

    start_time = time.time()
    with site_tab:
        _site_vis(
            cur_results_dir,
        )
    print(f"Took {time.time() - start_time} to run site vis")


def _get_observation_sites(results_dir: Path, site: str):
    cmvn_result = _load_cmvn_result(results_dir)

    obs_sites = cmvn_result.get_obs_stations(site)
    periods, pSA_keys = _get_periods(results_dir)

    obs_stations_rho = {
        cur_obs_site: np.mean(
            [
                cmvn_result.cond_lnIM_results[gc.im.IM.from_str(cur_pSA_key)].R.loc[
                    site, cur_obs_site
                ]
                for cur_pSA_key in pSA_keys
            ]
        )
        for cur_obs_site in obs_sites
    }

    return obs_sites, obs_stations_rho


def _site_vis(cur_results_dir: Path):
    method_type = _get_method_type(cur_results_dir)

    sites = _get_sites(cur_results_dir)
    cur_site = st.selectbox("Site", sites)

    col1, col2 = st.columns(2)
    with col1:
        show_marginal = st.checkbox("Show Marginal", value=False)
    with col2:
        show_conditional = st.checkbox("Show Conditional", value=True)

    obs_df = _get_obs_data(cur_results_dir)
    sim_data = _get_sim_data(cur_results_dir, sites)

    periods, pSA_keys = _get_periods(cur_results_dir)
    best_sim_ids = _load_best_sim_ids(cur_results_dir)
    gm_params = _get_gm_params(cur_results_dir)

    cmvn_result = _load_cmvn_result(cur_results_dir)

    ### Response Spectrum at site of interest
    st.markdown(
        """
        ### Response spectrum
        Figure shows, all simulation realisation in gray, 
        the best simulation realisation in red, 
        the observed data in black and the conditional MVN in blue,
        for the current site of interest.
        """
    )
    fig = plt.figure(figsize=(10, 6))
    fig = sr.plots.plot_response_spectrum(
        periods,
        pSA_keys,
        sim_data[cur_site],
        obs_df.loc[cur_site],
        cur_site,
        best_sim_ids.loc[cur_site],
        cMVN_result=cmvn_result if show_conditional else None,
        gm_params=gm_params.loc[cur_site] if show_marginal else None,
        show_all_sims=True,
        fig=fig,
    )
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    ### Response spectrum at the relevant observation sites
    obs_sites, obs_stations_rho = _get_observation_sites(cur_results_dir, cur_site)

    # Get the 5 sites with larges rho to site of interest
    largest_rho_obs_stations = np.flip(
        obs_sites[np.argsort(list(obs_stations_rho.values()))[-5:]]
    )

    # Get the distance matrix
    obs_stations_dist = _load_dist_matrix(cur_results_dir, list(obs_sites) + [cur_site])

    st.markdown("### Observed Response Spectra")
    st.markdown(f"Observation sites used:\n{', '.join(obs_sites)}")
    st.markdown(
        "Figure shows the observed response spectrum at the current "
        "site of interest and the 5 observation sites with largest rho used to to "
        "compute the conditional IM distributions."
    )

    fig = plt.figure(figsize=(10, 6))

    # Plot the most relevant observation sites
    colors = sns.color_palette("rocket", n_colors=5)
    for ix, cur_obs_site in enumerate(largest_rho_obs_stations):
        plt.semilogx(
            periods,
            obs_df.loc[cur_obs_site, pSA_keys],
            label=f"{cur_obs_site} ({obs_stations_dist.loc[cur_obs_site, cur_site]:.2f} km,"
            f" $\\rho$={obs_stations_rho[cur_obs_site]:.2f})",
            c=colors[ix],
            linewidth=0.9,
        )

    # Plot the observed at the site of interest
    plt.semilogx(
        periods,
        obs_df.loc[cur_site, pSA_keys],
        label=f"{cur_site} - Observed",
        color="black",
    )

    # Plot the conditional & marginal IM distributions
    if show_marginal:
        sr.plots.draw_marginal(plt.gca(), gm_params.loc[cur_site], periods, pSA_keys)
    if show_conditional:
        sr.plots.draw_cmnv(
            plt.gca(), periods, pSA_keys, _load_cmvn_result(cur_results_dir), cur_site
        )

    plt.xlabel(f"Period, T (s)")
    plt.ylabel(f"pSA (g)")
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.xlim([0.01, 10])
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    ### Difference between marginal and observed of most relevant sites
    st.markdown(
        """
        ## Observed - Marginal 
        Figure shows the difference between the marginal mean and the response 
        spectrum at the most relevant sites (largest $\\rho$).
        """
    )

    diff = pd.DataFrame(
        index=largest_rho_obs_stations,
        columns=pSA_keys,
        data=np.log(obs_df.loc[largest_rho_obs_stations, pSA_keys]).values
        - gm_params.loc[
            largest_rho_obs_stations, np.char.add(pSA_keys, "_mean")
        ].values,
    )

    fig = plt.figure(figsize=(10, 6))
    for ix, cur_obs_site in enumerate(largest_rho_obs_stations):
        plt.semilogx(
            periods,
            diff.loc[cur_obs_site, pSA_keys],
            label=f"{cur_obs_site} ({obs_stations_dist.loc[cur_obs_site, cur_site]:.2f} km,"
            f" $\\rho$={obs_stations_rho[cur_obs_site]:.2f})",
            c=colors[ix],
            linewidth=0.9,
        )

    plt.ylabel(r"$\mu_{lnIM} - lnIM$")
    plt.xlabel(f"Period, T (s)")
    plt.xlim([0.01, 10])
    plt.ylim([-2, 2])
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    print(f"wtf")


if __name__ == "__main__":
    typer.run(main)
