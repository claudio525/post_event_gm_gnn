from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import typer
import seaborn as sns

import st_utils
import sim_ranking as sr
import sha_calc as sha


@st.cache_data
def load_site_misfits(cim_results_dir: Path):
    site_misfits_df = pd.read_csv(cim_results_dir / "site_misfits.csv", index_col=0)
    return site_misfits_df


def create_pSA_dist_plot(
    ml_results_df: pd.DataFrame,
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

    ## ML - Compute the quantiles
    assert site_int_sims.index.equals(ml_results_df.index)
    cdf_x, cdf_y = [], []
    for cur_im in sr.constants.PSA_KEYS:
        cur_sort_ind = np.argsort(site_int_sims[cur_im].values)
        cdf_x.append(site_int_sims[cur_im].values[cur_sort_ind])
        cdf_y.append(np.cumsum(ml_results_df.prob.values[cur_sort_ind]))

    cdf_x = pd.DataFrame(np.asarray(cdf_x).T, columns=sr.constants.PSA_KEYS)
    cdf_y = pd.DataFrame(np.asarray(cdf_y).T, columns=sr.constants.PSA_KEYS)

    qt_2, qt_16, qt_50, qt_84, qt_98 = sha.query_non_parametric_multi_cdf_invs(
        np.asarray([0.02, 0.16, 0.5, 0.84, 0.98]), cdf_x.T.values, cdf_y.T.values
    )

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
        ## ML Mean
        weighted_avg = einops.einsum(
            ml_results_df.prob.values,
            np.log(site_int_sims.loc[:, sr.constants.PSA_KEYS].values),
            "i, i j -> j",
        )
        weighted_std = np.sqrt(
            einops.einsum(
                ml_results_df.prob.values,
                (
                    np.log(site_int_sims.loc[:, sr.constants.PSA_KEYS].values)
                    - weighted_avg
                )
                ** 2,
                "i, i j -> j",
            )
            / np.sum(ml_results_df.prob.values)
        )
        print(f"wtf")

        ax.semilogx(
            sr.constants.PERIODS,
            np.exp(weighted_avg),
            label="ML - Mean",
            c="blue",
        )
        ax.fill_between(
            sr.constants.PERIODS,
            np.exp(weighted_avg + weighted_std),
            np.exp(weighted_avg - weighted_std),
            alpha=0.4,
            label="ML +/- 1 Std",
            color="lightblue",
        )
        ax.semilogx(
            sr.constants.PERIODS, np.exp(weighted_avg + weighted_std), c="lightblue"
        )
        ax.semilogx(
            sr.constants.PERIODS, np.exp(weighted_avg - weighted_std), c="lightblue"
        )

        # ML Median and 16-84th
        # ax.semilogx(sr.constants.PERIODS, qt_50, label="Model - Median", c="blue")
        # ax.fill_between(
        #     sr.constants.PERIODS,
        #     qt_16,
        #     qt_84,
        #     alpha=0.4,
        #     label="Model - 16/84th",
        #     color="lightblue",
        # )
        # ax.semilogx(
        #     sr.constants.PERIODS, qt_16, c="lightblue", linestyle="--", linewidth=1.0
        # )
        # ax.semilogx(
        #     sr.constants.PERIODS, qt_84, c="lightblue", linestyle="--", linewidth=1.0
        # )

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


def run_ind_scenario(
    scenario_results: pd.DataFrame,
    sample_results: pd.DataFrame,
    ml_results_dir: Path,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    tab_type: str,
    syn_obs_gm_params: pd.DataFrame = None,
    gen_gm_params: pd.DataFrame = None,
):
    event_df = st_utils.ml_get_event_df(ml_results_dir)

    cur_events = scenario_results.event_id.unique().astype(str)

    cur_event = st.selectbox(
        "Event",
        event_df.loc[cur_events]
        .sort_values("mag", ascending=False)
        .index.values.astype(str),
        key=f"{tab_type}_event",
    )

    cur_sites = (
        scenario_results.loc[scenario_results.event_id == cur_event]
        .site_int.unique()
        .astype(str)
    )
    cur_site = st.selectbox("Site of Interest", cur_sites, key=f"{tab_type}_site")

    st.markdown(f"Magnitude: {event_df.loc[cur_event].mag}")

    # Load the cIM results
    cur_emp_cim_results_dir = emp_cim_results_dir / cur_event / "empirical_cMVN"
    cur_emp_cim = st_utils.cim_load_cmvn_result(cur_emp_cim_results_dir)
    # cur_emp_misfit_df = load_site_misfits(cur_emp_cim_results_dir, cur_event)

    cur_sim_cim_results_dir = sim_cim_results_dir / cur_event / "sim_cMVN"
    cur_sim_cim = st_utils.cim_load_cmvn_result(cur_sim_cim_results_dir)
    # cur_sim_misfit_df = load_site_misfits(cur_sim_cim_results_dir, cur_event)

    # Get observed and simulation IM values
    obs_df = st_utils.ml_get_obs_df(ml_results_dir)
    sim_df = st_utils.ml_get_sim_data(ml_results_dir, cur_event)

    # Get the relevant data
    cur_scenario_df = (
        scenario_results.loc[
            (scenario_results.event_id == cur_event)
            & (scenario_results.site_int == cur_site)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    cur_event_rels = cur_scenario_df.index.unique().astype(str)
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == cur_event) & (obs_df.site_id == cur_site)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = (
        sim_df.loc[
            (sim_df.site_id == cur_site) & np.isin(sim_df.rel_id, cur_event_rels)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    cur_sample_results = sample_results.loc[
        (sample_results.event_id == cur_event) & (sample_results.site_int == cur_site)
    ]

    assert np.all(site_int_sims.index == cur_scenario_df.index)

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

    # Plots
    create_pSA_dist_plot(
        cur_scenario_df,
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

    print(f"wtf")


def main(
    ml_results_dir: Path,
    emp_cim_results_dir: Path,
    sim_cim_results_dir: Path,
    syn_obs_gm_params_ffp: Path = None,
    gen_gm_params_ffp: Path = None,
):
    """Compare the results of the ML, empirical CIM and simulated CIM results"""
    st.set_page_config(layout="wide")

    (ind_scenarios,) = st.tabs(["Individual Scenario"])

    with ind_scenarios:
        (
            train_scenario_results,
            val_scenario_results,
        ) = st_utils.ml_load_scenario_results(ml_results_dir)
        train_sample_results, val_sample_results = st_utils.ml_load_sample_results(
            ml_results_dir
        )

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

        train_tab, val_tab = st.tabs(["Train", "Validation"])
        with train_tab:
            run_ind_scenario(
                train_scenario_results,
                train_sample_results,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "train",
                syn_obs_gm_params=syn_obs_gm_params,
                gen_gm_params=gen_gm_params,
            )

        with val_tab:
            run_ind_scenario(
                val_scenario_results,
                val_sample_results,
                ml_results_dir,
                emp_cim_results_dir,
                sim_cim_results_dir,
                "val",
                syn_obs_gm_params=syn_obs_gm_params,
                gen_gm_params=gen_gm_params,
            )

    print(f"wtf")


if __name__ == "__main__":
    typer.run(main)
