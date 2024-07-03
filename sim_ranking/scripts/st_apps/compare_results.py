import os
from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import typer
import seaborn as sns
from scipy import stats

import st_utils
import sim_ranking as sr
import sha_calc as sha
import ml_tools as mlt


@st.cache_data
def load_site_misfits(cim_results_dir: Path):
    site_misfits_df = pd.read_csv(cim_results_dir / "site_misfits.csv", index_col=0)
    return site_misfits_df


def create_pSA_dist_plot(
    ml_results_df: pd.DataFrame,
    ml_sum_df: pd.DataFrame,
    emp_cim: sr.conditional.ConditionalMVNDistribution,
    sim_cim: sr.conditional.ConditionalMVNDistribution,
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    site: str,
    tab_type: str,
    high_rels: List[str] = None,
    gen_gm_params: pd.DataFrame = None,
    syn_obs_gm_params: pd.DataFrame = None,
):
    col1, col2 = st.columns(2)
    with col1:
        show_gen = st.checkbox(
            "Show Generation Distribution", value=False, key=f"{tab_type}_gen"
        )
    with col2:
        show_syn_obs = st.checkbox(
            "Show Synthetic Observations Distribution",
            value=False,
            key=f"{tab_type}_syn_obs",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        show_emp_cim = st.checkbox(
            "Show Empirical CIM", value=True, key=f"{tab_type}_emp_cim"
        )
    with col2:
        show_sim_cim = st.checkbox(
            "Show Simulated CIM", value=True, key=f"{tab_type}_sim_cim"
        )
    with col3:
        show_ml = st.checkbox("Show ML", value=True, key=f"{tab_type}_ml")

    assert site_int_sims.index.equals(ml_results_df.index)

    fig, ax = plt.subplots(figsize=(12, 6))

    mean_cols = [f"{cur_key}_mean" for cur_key in sr.constants.PSA_KEYS]
    std_cols = [f"{cur_key}_std_Total" for cur_key in sr.constants.PSA_KEYS]

    # Synthethic observation distribution
    if syn_obs_gm_params is not None and show_syn_obs:
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(syn_obs_gm_params.loc[mean_cols].values.astype(float)),
            label="Synthetic observations distribution",
            c="purple",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                syn_obs_gm_params.loc[mean_cols].values.astype(float)
                + syn_obs_gm_params.loc[std_cols].values.astype(float)
            ),
            c="purple",
            linestyle="--",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                syn_obs_gm_params.loc[mean_cols].values.astype(float)
                - syn_obs_gm_params.loc[std_cols].values.astype(float)
            ),
            c="purple",
            linestyle="--",
            linewidth=1.0,
        )

    # Generation distribution
    if gen_gm_params is not None and show_gen:
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(gen_gm_params.loc[mean_cols].values.astype(float)),
            label="Generation distribution",
            c="k",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                gen_gm_params.loc[mean_cols].values.astype(float)
                + gen_gm_params.loc[std_cols].values.astype(float)
            ),
            c="k",
            linestyle="--",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                gen_gm_params.loc[mean_cols].values.astype(float)
                - gen_gm_params.loc[std_cols].values.astype(float)
            ),
            c="k",
            linestyle="--",
            linewidth=1.0,
        )

    ## Empirical CIM
    if show_emp_cim:
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(emp_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values),
            label="Empirical cIM - Mean",
            c="darkgreen",
        )
        ax.fill_between(
            sr.constants.PERIODS,
            np.exp(
                emp_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                + emp_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            np.exp(
                emp_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                - emp_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            alpha=0.4,
            label="Empirical cIM +/- 1 Std",
            color="lightgreen",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                emp_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                + emp_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            c="lightgreen",
            linestyle="--",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                emp_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                - emp_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            c="lightgreen",
            linestyle="--",
            linewidth=1.0,
        )

    ## Simulation-based CIM
    if show_sim_cim:
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(sim_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values),
            label="Simulated cIM",
            c="orange",
        )
        ax.fill_between(
            sr.constants.PERIODS,
            np.exp(
                sim_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                + sim_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            np.exp(
                sim_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                - sim_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            alpha=0.4,
            label="Simulated cIM +/- 1 Std",
            color="yellow",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                sim_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                + sim_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            c="yellow",
            linestyle="--",
            linewidth=1.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(
                sim_cim.cond_lnIM_mean_df.loc[site, sr.constants.PSA_KEYS].values
                - sim_cim.cond_lnIM_std_df.loc[site, sr.constants.PSA_KEYS].values
            ),
            c="yellow",
            linestyle="--",
            linewidth=1.0,
        )

    if show_ml:
        im_wavg_cols = mlt.array_utils.numpy_str_join(
            "_", sr.constants.PSA_KEYS, "wavg"
        )
        im_wstd_cols = mlt.array_utils.numpy_str_join(
            "_", sr.constants.PSA_KEYS, "wstd"
        )
        weighted_avg_ln = ml_sum_df.loc[im_wavg_cols].values.astype(float)
        weighted_std_ln = ml_sum_df.loc[im_wstd_cols].values.astype(float)

        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(weighted_avg_ln),
            label="ML - Mean",
            c="blue",
        )
        ax.fill_between(
            sr.constants.PERIODS,
            np.exp(weighted_avg_ln + weighted_std_ln),
            np.exp(weighted_avg_ln - weighted_std_ln),
            alpha=0.4,
            label="ML +/- 1 Std",
            color="lightblue",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(weighted_avg_ln + weighted_std_ln),
            c="lightblue",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(weighted_avg_ln - weighted_std_ln),
            c="lightblue",
        )

    if high_rels is not None:
        colors = sns.color_palette("dark", len(high_rels))
        for ix, cur_rel in enumerate(high_rels):
            ax.semilogx(
                sr.constants.PERIODS,
                site_int_sims.loc[cur_rel, sr.constants.PSA_KEYS].values,
                label=f"{cur_rel}",
                c=colors[ix],
                linestyle="dashdot",
            )
    ax.semilogx(
        sr.constants.PERIODS, site_int_obs, label="Observed - Site of Interest", c="r"
    )

    ax.set_xlabel(f"Period (s)")
    ax.set_ylabel(f"pSA")
    ax.set_xlim([0.01, 10])
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    # fig.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def _create_pot_sites_map(
    event: str,
    site_df: pd.DataFrame,
    int_sites: np.ndarray,
    obs_sites: np.ndarray,
    event_df: pd.DataFrame,
    site_int: str,
):
    all_sites = np.unique(np.concatenate([int_sites, obs_sites]))

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[obs_sites].lat,
                lon=site_df.loc[obs_sites].lon,
                mode="markers",
                marker=dict(size=10, color="blue"),
                hovertext=obs_sites,
                hoverinfo="text",
                name="Potential observation sites",
            ),
            go.Scattermapbox(
                lat=site_df.loc[int_sites].lat,
                lon=site_df.loc[int_sites].lon,
                mode="markers",
                marker=dict(size=10, color="orange"),
                hovertext=int_sites,
                hoverinfo="text",
                name="Potential sites of interest",
            ),
            go.Scattermapbox(
                lat=[site_df.loc[site_int, "lat"]],
                lon=[site_df.loc[site_int, "lon"]],
                mode="markers",
                marker=dict(size=20, color="red"),
                hovertext=site_int,
                hoverinfo="text",
                name="Site of interest",
            ),
            go.Scattermapbox(
                lat=[event_df.loc[event, "lat"]],
                lon=[event_df.loc[event, "lon"]],
                mode="markers",
                marker=dict(size=25, color="black"),
                hovertext=event,
                hoverinfo="text",
                name="Event",
            ),
        ]
    )

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=site_df.loc[all_sites].lat.mean(),
            lon=site_df.loc[all_sites].lon.mean(),
        ),
        zoom=8,
    )

    return fig


def _create_scenario_map(
    event: str,
    site_df: pd.DataFrame,
    emp_sites: np.ndarray,
    sim_sites: np.ndarray,
    ml_sites: np.ndarray,
    event_df: pd.DataFrame,
    site_int: str,
):
    all_sites = np.unique(np.concatenate([emp_sites, sim_sites, ml_sites]))

    # assert np.all(np.isin(emp_sites, sim_sites))
    cim_sites = np.unique(np.concatenate([emp_sites, sim_sites]))

    cim_only_sites = np.setdiff1d(cim_sites, ml_sites)
    ml_only_sites = np.setdiff1d(ml_sites, cim_sites)
    both_sites = np.intersect1d(cim_sites, ml_sites)
    assert np.all(
        np.isin(np.concatenate((cim_only_sites, ml_only_sites, both_sites)), all_sites)
    )

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[both_sites].lat,
                lon=site_df.loc[both_sites].lon,
                mode="markers",
                marker=dict(size=10, color="blue"),
                hovertext=both_sites,
                hoverinfo="text",
                name="Observation sites - cIM & ML",
            ),
            go.Scattermapbox(
                lat=site_df.loc[cim_only_sites].lat,
                lon=site_df.loc[cim_only_sites].lon,
                mode="markers",
                marker=dict(size=10, color="magenta"),
                hovertext=cim_only_sites,
                hoverinfo="text",
                name="Observation sites - cIM only",
            ),
            go.Scattermapbox(
                lat=site_df.loc[ml_only_sites].lat,
                lon=site_df.loc[ml_only_sites].lon,
                mode="markers",
                marker=dict(size=10, color="green"),
                hovertext=ml_only_sites,
                hoverinfo="text",
                name="Observation sites - ML only",
            ),
            go.Scattermapbox(
                lat=[site_df.loc[site_int, "lat"]],
                lon=[site_df.loc[site_int, "lon"]],
                mode="markers",
                marker=dict(size=20, color="red"),
                hovertext=site_int,
                hoverinfo="text",
                name="Site of interest",
            ),
            go.Scattermapbox(
                lat=[event_df.loc[event, "lat"]],
                lon=[event_df.loc[event, "lon"]],
                mode="markers",
                marker=dict(size=25, color="black"),
                hovertext=event,
                hoverinfo="text",
                name="Event",
            ),
        ]
    )

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=site_df.loc[all_sites].lat.mean(),
            lon=site_df.loc[all_sites].lon.mean(),
        ),
        zoom=8,
    )

    return fig


def run_ind_scenario(
    scenario_df: pd.DataFrame,
    scenario_sum_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    sample_sum_df: pd.DataFrame,
    ml_results_dir: Path,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    tab_type: str,
    syn_obs_gm_params: pd.DataFrame = None,
    gen_gm_params: pd.DataFrame = None,
):
    event_df = st_utils.ml_get_event_df(ml_results_dir)
    col1, col2 = st.columns([1, 6])

    metadata = st_utils.ml_get_metadata(ml_results_dir)

    (
        mean_ml_emp_cIM_residuals,
        std_ml_emp_cIM_residuals,
    ) = sr.ml.sc_prob.compute_mean_std_residuals_wrt_emp(
        scenario_sum_df, emp_cim_results_dir, metadata["run_config"]["ims"]
    )

    with col1:
        cur_events = scenario_df.event_id.unique().astype(str)
        cur_event = st.selectbox(
            "Event",
            event_df.loc[cur_events]
            .sort_values("mag", ascending=False)
            .index.values.astype(str),
            key=f"{tab_type}_event",
        )

        cur_sites = (
            scenario_df.loc[scenario_df.event_id == cur_event]
            .site_int.unique()
            .astype(str)
        )
        cur_site = st.selectbox("Site of Interest", cur_sites, key=f"{tab_type}_site")

        st.markdown(f"Magnitude: {event_df.loc[cur_event].mag}")

    # Load the cIM results
    cur_emp_cim_results_dir = emp_cim_results_dir / cur_event / "empirical_cMVN"
    cur_emp_cim = st_utils.cim_load_cmvn_result(cur_emp_cim_results_dir)

    cur_sim_cim_results_dir = sim_cim_results_dir / cur_event / "sim_cMVN"
    cur_sim_cim = st_utils.cim_load_cmvn_result(cur_sim_cim_results_dir)

    # Get observed and simulation IM values
    obs_df = st_utils.ml_get_obs_df(ml_results_dir)
    sim_df = st_utils.ml_get_sim_data(ml_results_dir, cur_event)

    # Get residuals wrt observed
    ml_obs_residuals, emp_cIM_obs_residuals, sim_cIM_obs_residuals = get_obs_residuals(
        scenario_sum_df,
        st_utils.ml_get_db_ffp(ml_results_dir),
        metadata["run_config"]["ims"],
        emp_cim_results_dir,
        sim_cim_results_dir,
    )

    ### Get the relevant data
    # ML - current scenario
    cur_scenario_df = (
        scenario_df.loc[
            (scenario_df.event_id == cur_event) & (scenario_df.site_int == cur_site)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    cur_sc_sum_df = scenario_sum_df.loc[
        (scenario_sum_df.event_id == cur_event) & (scenario_sum_df.site_int == cur_site)
    ].squeeze()

    # Event realisations
    cur_event_rels = cur_scenario_df.index.unique().astype(str)
    # Observation data at the site of interest
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == cur_event) & (obs_df.site_id == cur_site)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    # Simulation data at the site of interest
    site_int_sims = (
        sim_df.loc[
            (sim_df.site_id == cur_site) & np.isin(sim_df.rel_id, cur_event_rels)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    # ML - sample data for the current scenario
    cur_sample_results = sample_df.loc[
        (sample_df.event_id == cur_event) & (sample_df.site_int == cur_site)
    ]

    assert np.all(site_int_sims.index == cur_scenario_df.index)

    # Synthethic observation and generation distribution
    cur_syn_obs_gm_params = (
        syn_obs_gm_params.loc[
            (syn_obs_gm_params.event == cur_event)
            & (syn_obs_gm_params.site == cur_site)
        ].squeeze()
        if syn_obs_gm_params is not None
        else None
    )
    cur_gen_gm_params = (
        gen_gm_params.loc[
            (gen_gm_params.event == cur_event) & (gen_gm_params.site == cur_site)
        ].squeeze()
        if gen_gm_params is not None
        else None
    )

    with col2:
        fig = _create_pot_sites_map(
            cur_event,
            st_utils.ml_get_site_df(ml_results_dir),
            scenario_df.loc[scenario_df.event_id == cur_event]
            .site_int.unique()
            .astype(str),
            sample_df.loc[sample_df.event_id == cur_event]
            .site_obs.unique()
            .astype(str),
            event_df,
            cur_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("Scenario Map"):
        fig = _create_scenario_map(
            cur_event,
            st_utils.ml_get_site_df(ml_results_dir),
            cur_emp_cim.get_obs_stations(cur_site),
            cur_sim_cim.get_obs_stations(cur_site),
            cur_sample_results.site_obs.unique().astype(str),
            event_df,
            cur_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Plots
    create_pSA_dist_plot(
        cur_scenario_df,
        cur_sc_sum_df,
        cur_emp_cim,
        cur_sim_cim,
        site_int_sims,
        site_int_obs,
        cur_site,
        tab_type,
        syn_obs_gm_params=cur_syn_obs_gm_params,
        gen_gm_params=cur_gen_gm_params,
    )

    # Observation sites used
    st.write(
        f"Emp-cim Observation sites used: {cur_emp_cim.get_obs_stations(cur_site)}"
    )
    st.write(
        f"Sim-cIM Observation sites used: \t{cur_sim_cim.get_obs_stations(cur_site)}"
    )
    st.write(
        f"ML Observation sites used: {cur_sample_results.site_obs.unique().astype(str)}"
    )

    ### CDFs
    with st.expander("CDFs"):
        meta = st_utils.ml_get_metadata(ml_results_dir)
        ims = st.multiselect(
            "IMs",
            options=meta["run_config"]["ims"],
            default=[
                "pSA_0.01",
                "pSA_0.05",
                "pSA_0.1",
                "pSA_0.5",
                "pSA_1.0",
                "pSA_5.0",
            ],
            key=f"{tab_type}_ims_scenario",
        )

        n_ims = len(ims)
        n_rows = int(np.ceil(n_ims / 2))

        fig, axs = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))
        axs = axs.ravel()

        for ix, (cur_im, cur_ax) in enumerate(zip(ims, axs)):
            ### Empirical CIM
            # Get mu and sigma
            cur_emp_cim_mu = cur_emp_cim.cond_lnIM_mean_df.loc[cur_site, cur_im]
            cur_emp_cim_sigma = cur_emp_cim.cond_lnIM_std_df.loc[cur_site, cur_im]
            # Create values
            cur_emp_cim_im_values = np.linspace(
                cur_emp_cim_mu - 4 * cur_emp_cim_sigma,
                cur_emp_cim_mu + 4 * cur_emp_cim_sigma,
                100,
            )
            cur_emp_cim_prob_values = stats.norm.cdf(
                cur_emp_cim_im_values, cur_emp_cim_mu, cur_emp_cim_sigma
            )

            ### Simulation based CIM
            # Get mu and sigma
            cur_sim_cim_mu = cur_sim_cim.cond_lnIM_mean_df.loc[cur_site, cur_im]
            cur_sim_cim_sigma = cur_sim_cim.cond_lnIM_std_df.loc[cur_site, cur_im]
            # Create values
            cur_sim_cim_im_values = np.linspace(
                cur_sim_cim_mu - 4 * cur_sim_cim_sigma,
                cur_sim_cim_mu + 4 * cur_sim_cim_sigma,
                100,
            )
            cur_sim_cim_prob_values = stats.norm.cdf(
                cur_sim_cim_im_values, cur_sim_cim_mu, cur_sim_cim_sigma
            )

            ### ML
            # Get values
            ml_im_values = np.log(site_int_sims[cur_im].values.astype(float))
            ml_prob_values = cur_scenario_df[f"{cur_im}_prob"]
            # Sort
            sort_int = np.argsort(ml_im_values)
            ml_im_values = ml_im_values[sort_int]
            ml_cum_prob_values = np.cumsum(ml_prob_values.values[sort_int])

            # Plot
            cur_ax.plot(
                cur_emp_cim_im_values,
                cur_emp_cim_prob_values,
                label="cIM CDF",
                c="green",
            )
            cur_ax.step(ml_im_values, ml_cum_prob_values, label="ML CDF", c="blue")
            cur_ax.plot(
                cur_sim_cim_im_values,
                cur_sim_cim_prob_values,
                label="Sim-CIM CDF",
                c="orange",
            )

            if ix == 0:
                cur_ax.legend()

            if ix % 2 == 0:
                cur_ax.set_ylabel(f"Probability")

            if ix % 2 == 1:
                # cur_ax.set_yticklabels([])
                cur_ax.yaxis.tick_right()

            cur_ax.axvline(np.log(site_int_obs[cur_im]), c="red", linestyle="--", label="Observed")

            cur_ax.set_ylim(0.0, 1.0)
            cur_ax.set_xlim(cur_emp_cim_im_values.min(), cur_emp_cim_im_values.max())
            cur_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

            cur_ax.text(
                0.5,
                0.95,
                f"{cur_im}",
                horizontalalignment="center",
                verticalalignment="top",
                transform=cur_ax.transAxes,
                fontsize=12,
            )

        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        st.pyplot(fig, use_container_width=False)

    cur_mean_ml_emp_cIM_residuals = mean_ml_emp_cIM_residuals.loc[
        (mean_ml_emp_cIM_residuals.event_id == cur_event)
        & (mean_ml_emp_cIM_residuals.site_int == cur_site)
    ].squeeze()
    cur_std_ml_emp_cIM_residuals = std_ml_emp_cIM_residuals.loc[
        (std_ml_emp_cIM_residuals.event_id == cur_event)
        & (std_ml_emp_cIM_residuals.site_int == cur_site)
    ].squeeze()
    with st.expander("ML-Emp cIM Residual"):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        # Mean
        ax1.semilogx(
            sr.constants.PERIODS,
            cur_mean_ml_emp_cIM_residuals[sr.constants.PSA_KEYS].values,
            label="$\mu_{emp} - \mu_{ML}$",
        )

        ax1.set_title(f"Mean Residuals")
        ax1.set_xlabel(f"Period (s)")
        ax1.set_ylabel(f"Residual (lnIM_emp - lnIM_ml)")
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        ax1.set_xlim(0.01, 10)
        ax1.set_ylim(-1, 1)

        ax1.legend()

        # Std
        ax2.semilogx(
            sr.constants.PERIODS,
            cur_std_ml_emp_cIM_residuals[sr.constants.PSA_KEYS].values,
            label="$\sigma_{emp} - \sigma_{ML}$",
        )

        ax2.set_title(f"Std Residuals")
        ax2.set_xlabel(f"Period (s)")
        ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax2.yaxis.tick_right()

        ax2.set_xlim(0.01, 10)
        ax2.set_ylim(-1, 1)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        st.pyplot(fig, use_container_width=False)

    cur_ml_obs_residuals = ml_obs_residuals.loc[
        (ml_obs_residuals.event_id == cur_event)
        & (ml_obs_residuals.site_int == cur_site)
    ].squeeze()
    cur_emp_cIM_obs_residuals = emp_cIM_obs_residuals.loc[
        (emp_cIM_obs_residuals.event_id == cur_event)
        & (emp_cIM_obs_residuals.site_int == cur_site)
    ].squeeze()
    cur_sim_cIM_obs_residuals = sim_cIM_obs_residuals.loc[
        (sim_cIM_obs_residuals.event_id == cur_event)
        & (sim_cIM_obs_residuals.site_int == cur_site)
    ].squeeze()
    with st.expander("Residuals wrt. Observed"):
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        # Mean
        ax1.semilogx(
            sr.constants.PERIODS,
            cur_ml_obs_residuals[sr.constants.PSA_KEYS].values,
            label="$\ln_{IM}^{Obs} - \mu_{ML}$",
        )
        ax1.semilogx(
            sr.constants.PERIODS,
            cur_emp_cIM_obs_residuals[sr.constants.PSA_KEYS].values,
            label="$\ln_{IM}^{Obs} - \mu_{emp}$",
        )
        ax1.semilogx(
            sr.constants.PERIODS,
            cur_sim_cIM_obs_residuals[sr.constants.PSA_KEYS].values,
            label="$\ln_{IM}^{Obs} - \mu_{sim}$",
        )

        ax1.set_title(f"{cur_site}")
        ax1.set_xlabel(f"Period (s)")
        ax1.set_ylabel(f"Bias")
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        ax1.set_xlim(0.01, 10)
        ax1.set_ylim(-2, 2)

        ax1.legend()

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)


@st.cache_data
def get_obs_residuals(
    sc_sum_df: pd.DataFrame,
    db_ffp: Path,
    ims: np.ndarray,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
):
    ml_obs_residuals = sr.ml.sc_prob.compute_ml_residuals_wrt_obs(sc_sum_df, db_ffp, ims)

    emp_cIM_obs_residuals = sr.ml.sc_prob.compute_cIM_residuals_wrt_obs(
        emp_cim_results_dir, db_ffp, sr.constants.RankingMethod.emp_cMVN, ims
    )

    sim_cIM_obs_residuals = sr.ml.sc_prob.compute_cIM_residuals_wrt_obs(
        sim_cim_results_dir, db_ffp, sr.constants.RankingMethod.sim_cMVN, ims
    )

    index_intersection = np.intersect1d(
        np.intersect1d(
            ml_obs_residuals.index.values.astype(str),
            emp_cIM_obs_residuals.index.values.astype(str),
        ),
        sim_cIM_obs_residuals.index.values.astype(str),
    )

    ml_obs_residuals = ml_obs_residuals.loc[index_intersection]
    emp_cIM_obs_residuals = emp_cIM_obs_residuals.loc[index_intersection]
    sim_cIM_obs_residuals = sim_cIM_obs_residuals.loc[index_intersection]

    return ml_obs_residuals, emp_cIM_obs_residuals, sim_cIM_obs_residuals


@st.cache_data
def get_ml_emp_cIM_residuals(
    sc_sum_df: pd.DataFrame, emp_cim_dir: Path, ims: np.ndarray
):
    (
        mean_ml_residual_emp_cIM,
        std_ml_residuals_emp_cIM,
    ) = sr.ml.sc_prob.compute_mean_std_residuals_wrt_emp(sc_sum_df, emp_cim_dir, ims)

    return mean_ml_residual_emp_cIM, std_ml_residuals_emp_cIM


def run_stats_tab(
    sc_df: pd.DataFrame,
    sc_sum_df: pd.DataFrame,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    ml_results_dir: Path,
    tab_type: str,
):
    metadata = st_utils.ml_get_metadata(ml_results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    run_config = sr.ml.sc_prob.RunParamsConfig.from_dict(metadata["run_config"])

    ks_df, p_df = sr.ml.sc_prob.compute_ks_p_values(
        sc_df,
        emp_cim_results_dir,
        db_ffp,
        run_config,
    )

    # Get residuals wrt observed
    ml_obs_residuals, emp_cIM_obs_residuals, sim_cIM_obs_residuals = get_obs_residuals(
        sc_sum_df, db_ffp, run_config.ims, emp_cim_results_dir, sim_cim_results_dir
    )

    # Get ML residuals wrt empirical cIM
    mean_ml_residual_emp_cIM, std_ml_residuals_emp_cIM = get_ml_emp_cIM_residuals(
        sc_sum_df, emp_cim_results_dir, run_config.ims
    )

    ims = st.multiselect(
        "IMs",
        options=run_config.ims,
        default=["pSA_0.01", "pSA_0.05", "pSA_0.1", "pSA_0.5", "pSA_1.0", "pSA_5.0"],
        key=f"{tab_type}_ims",
    )
    n_ims = len(ims)
    n_rows = int(np.ceil(n_ims / 2))

    if n_ims > 0:
        with st.expander("KS-Statistic"):
            fig, axs = plt.subplots(n_rows, 2, figsize=(12, n_rows * 6), sharex=True)
            axs = axs.ravel()

            for ix, (cur_im, cur_ax) in enumerate(zip(ims, axs)):
                cur_ax.hist(ks_df[cur_im].values, bins=50)
                cur_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

                if ix >= 4:
                    cur_ax.set_xlabel(f"KS-Statisitc")
                if ix % 2 == 0:
                    cur_ax.set_ylabel(f"Count")

                if ix % 2 == 1:
                    # cur_ax.set_yticklabels([])
                    cur_ax.yaxis.tick_right()

                cur_ax.set_xlim(0, None)

                if ix > 0:
                    cur_ax.set_ylim(axs[ix - 1].get_ylim())

                cur_ax.text(
                    0.5,
                    0.95,
                    f"{cur_im}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=cur_ax.transAxes,
                    fontsize=12,
                )

            fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)
            st.pyplot(fig, use_container_width=False)

        with st.expander("P-Values"):
            fig, axs = plt.subplots(n_rows, 2, figsize=(12, n_rows * 6), sharex=True)
            axs = axs.ravel()

            for ix, (cur_im, cur_ax) in enumerate(zip(ims, axs)):
                cur_ax.hist(p_df[cur_im].loc[p_df[cur_im] < 0.1].values, bins=50)
                cur_ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

                if ix >= 4:
                    cur_ax.set_xlabel(f"P-Value")
                if ix % 2 == 0:
                    cur_ax.set_ylabel(f"Count")

                if ix % 2 == 1:
                    # cur_ax.set_yticklabels([])
                    cur_ax.yaxis.tick_right()

                cur_ax.set_xlim(0, None)

                if ix > 0:
                    cur_ax.set_ylim(axs[ix - 1].get_ylim())

                cur_ax.text(
                    0.5,
                    0.95,
                    f"{cur_im}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=cur_ax.transAxes,
                    fontsize=12,
                )

            fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)
            st.pyplot(fig, use_container_width=False)

    with st.expander("Residuals wrt. Observed"):
        if n_ims > 0:
            n_rows = n_ims
            fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 6), sharex=True)

            bins = np.linspace(-2.0, 2.0, 50)

            y_max = 0
            for ix, (cur_im, (cur_ax1, cur_ax2, cur_ax3)) in enumerate(zip(ims, axs)):
                cur_ax1.hist(ml_obs_residuals[cur_im].values, bins=bins)
                cur_ax1.axvline(
                    np.mean(ml_obs_residuals[cur_im].values), color="r", linestyle="--"
                )
                cur_ax1.axvline(0, color="k", linestyle="-")
                cur_ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")

                cur_ax2.hist(emp_cIM_obs_residuals[cur_im].values, bins=bins)
                cur_ax2.axvline(
                    np.mean(emp_cIM_obs_residuals[cur_im].values),
                    color="r",
                    linestyle="--",
                )
                cur_ax2.axvline(0, color="k", linestyle="-")
                cur_ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")

                cur_ax3.hist(sim_cIM_obs_residuals[cur_im].values, bins=bins)
                cur_ax3.axvline(
                    np.mean(sim_cIM_obs_residuals[cur_im].values),
                    color="r",
                    linestyle="--",
                )
                cur_ax3.axvline(0, color="k", linestyle="-")
                cur_ax3.grid(linewidth=0.5, alpha=0.5, linestyle="--")

                if ix == 0:
                    cur_ax1.set_xlim(-2.0, 2.0)

                y_max = max(y_max, cur_ax1.get_ylim()[1], cur_ax2.get_ylim()[1])

                cur_ax2.yaxis.tick_right()

                cur_ax1.text(
                    0.01,
                    0.98,
                    f"ML-Obs Residuals",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax1.transAxes,
                    fontsize=12,
                )
                cur_ax1.text(
                    0.98,
                    0.98,
                    f"{cur_im}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=cur_ax1.transAxes,
                    fontsize=12,
                )
                cur_ax1.text(
                    0.01,
                    0.94,
                    f"Mean: {np.mean(ml_obs_residuals[cur_im].values):.2f}, Std: {np.std(ml_obs_residuals[cur_im].values):.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax1.transAxes,
                    fontsize=12,
                )

                cur_ax2.text(
                    0.01,
                    0.98,
                    f"Emp cIM Residuals",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax2.transAxes,
                    fontsize=12,
                )
                cur_ax2.text(
                    0.01,
                    0.94,
                    f"Mean: {np.mean(emp_cIM_obs_residuals[cur_im].values):.2f}, Std: {np.std(emp_cIM_obs_residuals[cur_im].values):.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax2.transAxes,
                    fontsize=12,
                )

                cur_ax3.text(
                    0.01,
                    0.98,
                    f"Sim cIM Residuals",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax3.transAxes,
                    fontsize=12,
                )
                cur_ax3.text(
                    0.01,
                    0.94,
                    f"Mean: {np.mean(sim_cIM_obs_residuals[cur_im].values):.2f}, Std: {np.std(sim_cIM_obs_residuals[cur_im].values):.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=cur_ax3.transAxes,
                    fontsize=12,
                )

            for cur_ax in axs.ravel():
                cur_ax.set_ylim(0, y_max)

            fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)
            st.pyplot(fig, use_container_width=False)

        ml_obs_mean = ml_obs_residuals[sr.constants.PSA_KEYS].mean()
        ml_obs_std = ml_obs_residuals[sr.constants.PSA_KEYS].std()

        emp_cIM_obs_mean = emp_cIM_obs_residuals[sr.constants.PSA_KEYS].mean()
        emp_cIM_obs_std = emp_cIM_obs_residuals[sr.constants.PSA_KEYS].std()

        sim_cIM_obs_mean = sim_cIM_obs_residuals[sr.constants.PSA_KEYS].mean()
        sim_cIM_obs_std = sim_cIM_obs_residuals[sr.constants.PSA_KEYS].std()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.semilogx(
            sr.constants.PERIODS,
            ml_obs_mean,
            label=f"ML-Obs, $\mu$={ml_obs_mean.mean():.2f}",
            c="blue",
        )
        ax.semilogx(
            sr.constants.PERIODS, ml_obs_mean + ml_obs_std, c="blue", linestyle="--"
        )
        ax.semilogx(
            sr.constants.PERIODS, ml_obs_mean - ml_obs_std, c="blue", linestyle="--"
        )

        ax.semilogx(
            sr.constants.PERIODS,
            emp_cIM_obs_mean,
            label=f"Emp cIM-Obs, $\mu$={emp_cIM_obs_mean.mean():.2f} ",
            c="green",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            emp_cIM_obs_mean + emp_cIM_obs_std,
            c="green",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            emp_cIM_obs_mean - emp_cIM_obs_std,
            c="green",
            linestyle="--",
        )

        ax.semilogx(
            sr.constants.PERIODS,
            sim_cIM_obs_mean,
            label=f"Sim cIM-Obs, $\mu$={sim_cIM_obs_mean.mean():.2f}",
            c="orange",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            sim_cIM_obs_mean + sim_cIM_obs_std,
            c="orange",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            sim_cIM_obs_mean - sim_cIM_obs_std,
            c="orange",
            linestyle="--",
        )

        ax.set_ylim(-1.25, 1.25)
        ax.set_xlim(0.01, 10)
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend()

        ax.set_title("Mean Residuals")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with st.expander("ML Residuals wrt. Empirical CIM"):
        st.markdown("Empirical cIM - ML")

        n_rows = n_ims

        fig, axs = plt.subplots(n_rows, 2, figsize=(12, n_rows * 6), sharex=True)

        bins = np.linspace(-2.0, 2.0, 50)

        y_max = 0
        for ix, (cur_im, (cur_ax1, cur_ax2)) in enumerate(zip(ims, axs)):
            cur_ax1.hist(mean_ml_residual_emp_cIM[cur_im].values, bins=bins)
            cur_ax1.axvline(
                np.mean(mean_ml_residual_emp_cIM[cur_im].values),
                color="r",
                linestyle="--",
            )
            cur_ax1.axvline(0, color="k", linestyle="-")
            cur_ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")

            cur_ax2.hist(std_ml_residuals_emp_cIM[cur_im].values, bins=bins)
            cur_ax2.axvline(
                np.mean(std_ml_residuals_emp_cIM[cur_im].values),
                color="r",
                linestyle="--",
            )
            cur_ax2.axvline(0, color="k", linestyle="-")
            cur_ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")

            if ix == 0:
                cur_ax1.set_xlim(-1.25, 1.25)

            y_max = max(y_max, cur_ax1.get_ylim()[1], cur_ax2.get_ylim()[1])

            cur_ax2.yaxis.tick_right()

            cur_ax1.text(
                0.01,
                0.98,
                f"Mean Residuals - {cur_im}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=cur_ax1.transAxes,
                fontsize=12,
            )

            cur_ax2.text(
                0.01,
                0.98,
                f"Std Residuals - {cur_im}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=cur_ax2.transAxes,
                fontsize=12,
            )

        for cur_ax in axs.ravel():
            cur_ax.set_ylim(0, y_max)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        st.pyplot(fig, use_container_width=False)


def run_explore_tab(
    sc_sum_df: pd.DataFrame,
    ml_results_dir: Path,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    tab_type: str,
):
    sc_sum_df = sc_sum_df.sort_index()

    ims = np.asarray(st_utils.ml_get_metadata(ml_results_dir)["run_config"]["ims"])
    mean_ml_emp_cIM_res, std_ml_emp_cIM_res = get_ml_emp_cIM_residuals(
        sc_sum_df, emp_cim_results_dir, ims
    )
    mean_ml_emp_cIM_res = mean_ml_emp_cIM_res.sort_index()


    ml_obs_residuals, emp_cIM_obs_residuals, sim_cIM_obs_residuals = get_obs_residuals(sc_sum_df, st_utils.ml_get_db_ffp(ml_results_dir), ims, emp_cim_results_dir, sim_cim_results_dir)

    res_df = ml_obs_residuals[["event_id", "site_int"]].copy(deep=True)
    res_df["ml_obs_mse"] = (ml_obs_residuals[sr.constants.PSA_KEYS] ** 2).mean(axis=1)
    res_df["emp_cIM_obs_mse"] = (emp_cIM_obs_residuals[sr.constants.PSA_KEYS] ** 2).mean(
        axis=1
    )
    assert np.all(sim_cIM_obs_residuals.index == res_df.index)
    res_df["sim_cIM_obs_mse"] = (sim_cIM_obs_residuals[sr.constants.PSA_KEYS] ** 2).mean(
        axis=1
    )
    res_df["ml_emp_cIM_mse"] = (mean_ml_emp_cIM_res.loc[res_df.index, sr.constants.PSA_KEYS] ** 2).mean(
        axis=1
    )
    res_df["n_obs_sites"] = sc_sum_df.loc[res_df.index, "n_obs_sites"]
    res_df["min_s2s_dist"] = sc_sum_df.loc[res_df.index, "min_s2s_dist"]
    res_df["sc_weight"] = sc_sum_df.loc[res_df.index, "weight"]

    st.dataframe(res_df)


def main(
    ml_results_dir: Path,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    syn_obs_gm_params_ffp: Path = None,
    gen_gm_params_ffp: Path = None,
):
    """Compare the results of the ML, empirical CIM and simulated CIM results"""
    st.set_page_config(layout="wide")

    (ind_scenarios, stats_tab, explore_tab) = st.tabs(
        ["Individual Scenario", "Stats", "Explore"]
    )

    (
        train_sc_df,
        train_sc_sum_df,
        val_sc_df,
        val_sc_sum_results,
    ) = st_utils.ml_load_scenario_results(ml_results_dir)
    (
        train_sample_df,
        train_sample_sum_df,
        val_sample_df,
        val_sample_sum_df,
    ) = st_utils.ml_load_sample_results(ml_results_dir)

    syn_obs_gm_params = (
        st_utils.load_gm_params(syn_obs_gm_params_ffp)
        if syn_obs_gm_params_ffp is not None
        else None
    )
    gen_gm_params = (
        st_utils.load_gm_params(gen_gm_params_ffp)
        if gen_gm_params_ffp is not None
        else None
    )

    with ind_scenarios:
        train_tab, val_tab = st.tabs(["Train", "Validation"])
        with train_tab:
            run_ind_scenario(
                train_sc_df,
                train_sc_sum_df,
                train_sample_df,
                train_sample_sum_df,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "train",
                syn_obs_gm_params=syn_obs_gm_params,
                gen_gm_params=gen_gm_params,
            )
        with val_tab:
            run_ind_scenario(
                val_sc_df,
                val_sc_sum_results,
                val_sample_df,
                val_sample_sum_df,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "val",
                syn_obs_gm_params=syn_obs_gm_params,
                gen_gm_params=gen_gm_params,
            )

    with stats_tab:
        train_tab, val_tab = st.tabs(["Train", "Validation"])
        with train_tab:
            pass
            run_stats_tab(
                train_sc_df,
                train_sc_sum_df,
                emp_cim_results_dir,
                sim_cim_results_dir,
                ml_results_dir,
                "train",
            )
        with val_tab:
            run_stats_tab(
                val_sc_df,
                val_sc_sum_results,
                emp_cim_results_dir,
                sim_cim_results_dir,
                ml_results_dir,
                "val",
            )

    with explore_tab:
        train_tab, val_tab = st.tabs(["Train", "Validation"])

        with train_tab:
            run_explore_tab(
                train_sc_sum_df,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "train",
            )
        with val_tab:
            run_explore_tab(
                val_sc_sum_results,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "val",
            )


if __name__ == "__main__":
    typer.run(main)
