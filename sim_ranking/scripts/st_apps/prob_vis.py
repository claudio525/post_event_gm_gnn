import os
import time
from pathlib import Path
from typing import Sequence, List, Tuple, Any

import einops
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import typer
import scipy.stats as stats

import sha_calc as sha
import spatial_hazard as sh
import sim_ranking as sr
import ml_tools as mlt

import st_utils


@st.cache_data
def load_training_metrics(results_dir: Path):
    metrics = pd.read_pickle(results_dir / "metrics.pickle")

    return metrics


@st.cache_data
def get_results_group(
    results_df: pd.DataFrame,
    group_cols: List[str],
):
    return results_df.groupby(group_cols, observed=True)


def run_general_tab(results_dir: Path):
    # Load the metadata
    meta = st_utils.ml_get_metadata(results_dir)

    # Loss plot
    metrics = load_training_metrics(results_dir)
    metric_keys = sorted(list(metrics.keys()))

    avail_metrics = [key.rsplit("_", maxsplit=1)[0] for key in metric_keys[::2]]
    # avail_metrics = ["loss_hist", "misfit_loss_hist"]
    sel_metric_keys = st.multiselect(
        "Metrics", avail_metrics, default=[avail_metrics[0]]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    mlt.plotting.plot_metrics(
        metrics, sel_metric_keys, ax=ax, best_epoch=meta["best_epoch"]
    )
    # mlt.plotting.plot_metrics(load_training_metrics(results_dir), ax=ax)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    ### Scenario loss
    train_scenario_results, val_scenario_results = st_utils.ml_load_scenario_results(
        results_dir
    )
    if "prob" in train_scenario_results.columns:
        train_scenario_loss = sr.ml.prob.compute_scenario_loss(train_scenario_results)
        val_scenario_loss = sr.ml.prob.compute_scenario_loss(val_scenario_results)

        st.markdown(
            f"#### Mean train scenario loss: {train_scenario_loss.scenario_loss.mean():.4f}"
        )
        st.write(
            f"#### Mean val scenario loss: {val_scenario_loss.scenario_loss.mean():.4f}"
        )

    # Model visualization
    col_1, col_2 = st.columns(2)
    with col_1:
        if (model_vis_ffp := results_dir / "prob_model_vis.png").exists():
            st.image(str(model_vis_ffp))

    with col_2:
        padding = 30
        st.markdown("### Hyperparams")
        st.text(f"{'Batch size:':<{padding}} {meta['hp_config']['batch_size']}")
        st.text(f"{'Learning rate:':<{padding}} {meta['hp_config']['lr']}")
        if "fc_units" in meta["hp_config"]:
            st.text(f"{'FC Units:':<{padding}} {str(meta['hp_config']['fc_units'])}")
        else:
            st.text(
                f"{'Ind FC Units:':<{padding}} {str(meta['hp_config']['ind_fc_units'])}"
            )
            st.text(
                f"{'Comb FC Units:':<{padding}} {str(meta['hp_config']['comb_fc_units'])}"
            )
        st.text(f"{'L2:':<{padding}} {meta['hp_config']['l2_reg']}")
        st.divider()

        st.text(
            f"{'Use IM Sim Site Obs:':<{padding}} {meta['hp_config']['im_sim_site_obs']}"
        )
        st.text(
            f"{'Use IM Sim Site Int:':<{padding}} {meta['hp_config']['im_sim_site_int']}"
        )
        st.text(
            f"{'Use IM Obs Site Obs:':<{padding}} {meta['hp_config']['im_obs_site_obs']}"
        )

        st.text(
            f"{'Use Residual sim_site_obs obs_site_obs:':<{padding}} {meta['hp_config']['res_site_obs']}"
        )
        st.text(
            f"{'Use Residual sim_site_obs sim_site_int:':<{padding}} {meta['hp_config']['res_sim_site_obs_sim_site_int']}"
        )
        st.text(
            f"{'Use Residual obs_site_obs sim_site_int:':<{padding}} {meta['hp_config']['res_obs_site_obs_sim_site_int']}"
        )

        st.markdown("### Run Config")
        st.text(
            f"{'Number of realisations:':<{padding}} {meta['run_config']['n_rels']}"
        )
        st.text(f"{'Max distance:':<{padding}} {meta['run_config']['max_dist']}")


def _create_event_map(
    event: str,
    site_df: pd.DataFrame,
    all_site_int: np.ndarray,
    all_site_obs: np.ndarray,
    event_df: pd.DataFrame,
    site_int: str,
    site_obs: np.ndarray,
):
    cur_site_obs_obs = set(site_obs).intersection(set(all_site_obs)) - set(all_site_int)
    cur_site_obs_both = (
        set(site_obs).intersection(set(all_site_int)).intersection(set(all_site_obs))
    )

    pot_site_both = (
        set(all_site_int).intersection(set(all_site_obs))
        - set(site_obs)
        - set([site_int])
    )

    pot_site_obs = set(all_site_obs) - set(site_obs) - set([site_int]) - pot_site_both
    pot_site_int = set(all_site_int) - set(site_obs) - set([site_int]) - pot_site_both

    all_sites = np.unique(np.concatenate([all_site_obs, all_site_int]))
    assert cur_site_obs_obs.union(cur_site_obs_both).union(pot_site_obs).union(
        pot_site_int
    ).union(pot_site_both).union([site_int]).union(site_obs) == set(all_sites)

    # Convert sets to list
    cur_site_obs_obs = list(cur_site_obs_obs)
    cur_site_obs_both = list(cur_site_obs_both)
    pot_site_obs = list(pot_site_obs)
    pot_site_int = list(pot_site_int)
    pot_site_both = list(pot_site_both)

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[pot_site_both].lat,
                lon=site_df.loc[pot_site_both].lon,
                mode="markers",
                marker=dict(size=10, color="darkblue"),
                hovertext=pot_site_both,
                hoverinfo="text",
                name="Potential site of interest & observation site",
            ),
            go.Scattermapbox(
                lat=site_df.loc[pot_site_obs].lat,
                lon=site_df.loc[pot_site_obs].lon,
                mode="markers",
                marker=dict(size=10, color="gray"),
                hovertext=pot_site_obs,
                hoverinfo="text",
                name="Potential observation sites",
            ),
            go.Scattermapbox(
                lat=site_df.loc[pot_site_int].lat,
                lon=site_df.loc[pot_site_int].lon,
                mode="markers",
                marker=dict(size=10, color="blue"),
                hovertext=pot_site_int,
                hoverinfo="text",
                name="Potential site of interests",
            ),
            go.Scattermapbox(
                lat=site_df.loc[cur_site_obs_both].lat,
                lon=site_df.loc[cur_site_obs_both].lon,
                mode="markers",
                marker=dict(size=15, color="orange"),
                hovertext=cur_site_obs_both,
                hoverinfo="text",
                name=f"Observation site for {site_int} & site of interest",
            ),
            go.Scattermapbox(
                lat=site_df.loc[cur_site_obs_obs].lat,
                lon=site_df.loc[cur_site_obs_obs].lon,
                mode="markers",
                marker=dict(size=15, color="yellow"),
                hovertext=cur_site_obs_obs,
                hoverinfo="text",
                name=f"Observation site for {site_int} and other sites",
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


def _scenario_viewer(
    results_dir: Path,
    scenario_results: pd.DataFrame,
    sample_results: pd.DataFrame,
    tab_type: str,
    gen_gm_params: pd.DataFrame = None,
    syn_obs_gm_params: pd.DataFrame = None,
):
    site_df = st_utils.ml_get_site_df(results_dir)
    event_df = st_utils.ml_get_event_df(results_dir)
    obs_df = st_utils.ml_get_obs_df(results_dir)

    events = scenario_results.event_id.unique().astype("str")

    col1, col2 = st.columns([1, 6])

    with col1:
        event = st.selectbox(
            "Event",
            event_df.loc[events]
            .sort_values("mag", ascending=False)
            .index.values.astype(str),
            key=f"{tab_type}_event",
        )

        site_int = st.selectbox(
            "Site of Interest",
            scenario_results.loc[(scenario_results.event_id == event)]
            .site_int.unique()
            .astype("str"),
            key=f"{tab_type}_site_int",
        )

        sim_df = st_utils.ml_get_sim_data(results_dir, event)

        high_rels = st.multiselect(
            "Highlighted Realisations",
            sorted(sim_df.rel_id.unique().astype(str).tolist()),
            key=f"{tab_type}_high_rels",
        )

        st.markdown(f"Magnitude: {event_df.loc[event].mag}")

    with col2:
        fig = _create_event_map(
            event,
            site_df,
            np.unique(
                scenario_results.loc[
                    scenario_results.event_id == event
                ].site_int.values.astype(str)
            ),
            np.unique(
                sample_results.loc[
                    (sample_results.event_id == event)
                ].site_obs.values.astype(str)
            ),
            event_df,
            site_int,
            sample_results.loc[
                (sample_results.event_id == event)
                & (sample_results.site_int == site_int)
            ].site_obs.values.astype(str),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Get the relevant data
    cur_scenario_df = (
        scenario_results.loc[
            (scenario_results.event_id == event)
            & (scenario_results.site_int == site_int)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    cur_event_rels = cur_scenario_df.index.unique().astype(str)
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = (
        sim_df.loc[
            (sim_df.site_id == site_int) & np.isin(sim_df.rel_id, cur_event_rels)
        ]
        .set_index("rel_id")
        .sort_index()
    )

    cur_sample_results = sample_results.loc[
        (sample_results.event_id == event) & (sample_results.site_int == site_int)
    ]

    if "prob" in cur_scenario_df.columns:
        st.markdown(
            f"**Scenario loss: {sr.ml.prob.compute_scenario_loss(cur_scenario_df).scenario_loss.iloc[0]:.4f}**"
        )

    assert np.all(site_int_sims.index == cur_scenario_df.index)

    cur_gen_gm_params = (
        gen_gm_params.loc[
            (gen_gm_params.event == event) & (gen_gm_params.site == site_int)
        ].squeeze()
        if gen_gm_params is not None
        else None
    )

    cur_syn_obs_gm_params = (
        syn_obs_gm_params.loc[
            (syn_obs_gm_params.event == event) & (syn_obs_gm_params.site == site_int)
        ].squeeze()
        if syn_obs_gm_params is not None
        else None
    )

    create_pSA_dist_plot(
        cur_scenario_df,
        site_int_sims,
        site_int_obs,
        high_rels=high_rels if len(high_rels) > 0 else None,
        gen_gm_params=cur_gen_gm_params,
        syn_obs_gm_params=cur_syn_obs_gm_params,
    )

    st.divider()

    st.text(f"Number of Observation sites: {cur_scenario_df.n_obs_sites.iloc[0]}")
    st.text(
        f"Distance to closest observation site: {cur_scenario_df.min_distance.iloc[0]}"
    )

    # Get the observation sites (sorted by distance)
    cur_obs_sites_df = cur_sample_results.groupby("site_obs", observed=True).first()
    cur_obs_sites = (
        cur_obs_sites_df["s2s_distance"].sort_values().index.values.astype(str)
    )

    weight_cols = mlt.array_utils.numpy_str_join(
        "_", sr.constants.PSA_KEYS, "site_weights"
    )
    for cur_obs_site in cur_obs_sites:
        with st.expander(cur_obs_site):
            st.text(
                f"Observation site: {cur_obs_site}, "
                f"distance {cur_obs_sites_df.loc[cur_obs_site, 's2s_distance']:.2f}"
            )
            st.dataframe(cur_obs_sites_df.loc[cur_obs_site, weight_cols].to_frame().T)
            create_pSA_dist_plot(
                cur_sample_results.loc[cur_sample_results.site_obs == cur_obs_site],
                site_int_sims,
                site_int_obs,
                high_rels=high_rels if len(high_rels) > 0 else None,
                site_obs_obs=obs_df.loc[
                    (obs_df.event_id == event) & (obs_df.site_id == cur_obs_site)
                ],
                # gen_gm_params=cur_gen_gm_params if show_gen_dist else None,
                # syn_obs_gm_params=cur_syn_obs_gm_params if show_obs_dist else None,
            )

    if "prob" in cur_scenario_df.columns:
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col1:
            st.dataframe(
                cur_scenario_df[["prob", "misfit_score"]]
                .sort_values("prob", ascending=False)
                .head(10)
            )

        with col2:
            tmp_df = cur_sample_results[["site_obs", "rel_id", "prob"]].pivot(
                columns="site_obs", index="rel_id", values="prob"
            )
            assert np.all(tmp_df.index == cur_scenario_df.index)
            tmp_df["scenario_prop"] = cur_scenario_df["prob"]
            tmp_df["misfit_score"] = cur_scenario_df["misfit_score"]
            tmp_df = tmp_df.sort_values("scenario_prop", ascending=False)
            st.dataframe(tmp_df)

        with col3:
            st.dataframe(
                cur_scenario_df[["prob", "misfit_score"]]
                .sort_values("misfit_score", ascending=True)
                .head(10)
            )

        st.dataframe(
            cur_sample_results.groupby("site_obs", observed=True)
            .first()[["site_corr_weights", "s2s_distance", "angular_distance"]]
            .sort_values("site_corr_weights", ascending=False)
            .T
        )

        create_misfit_dist_plot(cur_scenario_df, tab_type)

        col1, col2, col3 = st.columns(3)
        with col1:
            ims = st.multiselect(
                "IMs",
                sr.constants.PSA_KEYS,
                key=f"{tab_type}_ims",
                default=[
                    "pSA_0.01",
                    "pSA_0.1",
                    "pSA_0.5",
                    "pSA_1.0",
                    "pSA_2.5",
                    "pSA_5.0",
                    "pSA_10.0",
                ],
            )
        with col2:
            n_bins = st.number_input(
                "Number of Bins", 5, 100, 10, key=f"{tab_type}_n_bins"
            )
        with col3:
            cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_cdf")

        create_dist_plot(
            site_int_sims,
            site_int_obs,
            cur_scenario_df,
            n_bins,
            ims,
            cdf,
            gen_gm_params=cur_gen_gm_params,
            syn_obs_gm_params=cur_syn_obs_gm_params,
        )


def create_misfit_dist_plot(results_df: pd.DataFrame, tab_type: str):
    col1, col2 = st.columns(2)
    with col1:
        misfit_n_bins = st.number_input(
            "Number of Bins", 5, 50, 10, key=f"{tab_type}_misfit_n_bins"
        )
    with col2:
        misfit_cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_misfit_cdf")

    fig, ax = plt.subplots(figsize=(12, 6))
    if misfit_cdf:
        sort_inds = np.argsort(results_df.misfit_score)
        ax.step(
            results_df.misfit_score.values[sort_inds],
            np.cumsum(results_df.prob.values[sort_inds]),
            label="Model distribution",
            c="b",
            where="post",
        )
        # sns.rugplot(x=cur_scenario_df.misfit_score, ax=ax, color="k")
        ax.axvline(
            x=results_df.misfit_score.min(),
            c="r",
            label=f"Lowest misfit - {results_df.misfit_score.idxmin()}",
        )
        ax.set_title(f"P(Misfit Score < x)")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend()
        # fig.tight_layout()
    else:
        ax.hist(
            results_df.misfit_score,
            weights=results_df.prob,
            bins=misfit_n_bins,
        )
        ax.axvline(
            x=results_df.misfit_score.min(),
            c="r",
            label=f"Lowest misfit - {results_df.misfit_score.idxmin()}",
        )
        ax.set_xlabel("Misfit Score")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        # fig.tight_layout()

    st.pyplot(fig, use_container_width=False)


def _sample_viewer(
    results_dir: Path,
    results_df: pd.DataFrame,
    tab_type: str,
    gen_gm_params: pd.DataFrame = None,
    syn_obs_gm_params: pd.DataFrame = None,
):
    site_df = st_utils.ml_get_site_df(results_dir)
    event_df = st_utils.ml_get_event_df(results_dir)
    obs_df = st_utils.ml_get_obs_df(results_dir)

    events = results_df.event_id.unique().astype("str")

    col1, col2 = st.columns([1, 6])

    with col1:
        event = st.selectbox(
            "Event",
            event_df.loc[events]
            .sort_values("mag", ascending=False)
            .index.values.astype(str),
            key=f"{tab_type}_event",
        )

        site_int = st.selectbox(
            "Site of Interest",
            results_df.loc[(results_df.event_id == event)]
            .site_int.unique()
            .astype("str"),
            key=f"{tab_type}_site_int",
        )

        site_obs = st.selectbox(
            "Observation Site",
            results_df.loc[
                (results_df.event_id == event) & (results_df.site_int == site_int)
            ]
            .site_obs.unique()
            .astype(str),
            key=f"{tab_type}_site_obs",
        )

        sim_df = st_utils.ml_get_sim_data(results_dir, event)
        high_rels = st.multiselect(
            "Highlighted Realisations",
            sorted(sim_df.rel_id.unique().astype(str).tolist()),
            key=f"{tab_type}_high_rels",
        )

        st.markdown(f"Magnitude: {event_df.loc[event].mag}")

    with col2:
        fig = _create_event_map(
            event,
            site_df,
            np.unique(
                results_df.loc[results_df.event_id == event].site_int.values.astype(str)
            ),
            np.unique(
                results_df.loc[results_df.event_id == event].site_obs.values.astype(str)
            ),
            event_df,
            site_int,
            np.asarray([site_obs]),
        )

        st.plotly_chart(fig, use_container_width=True)

    cur_event_rels = (
        results_df.loc[(results_df.event_id == event)].rel_id.unique().astype(str)
    )

    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = (
        sim_df.loc[
            (sim_df.site_id == site_int) & np.isin(sim_df.rel_id, cur_event_rels)
        ]
        .set_index("rel_id")
        .sort_index()
    )
    site_obs_obs = obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]

    # Site of interest distribution
    cur_results_df = (
        results_df.loc[
            (results_df.event_id == event)
            & (results_df.site_int == site_int)
            & (results_df.site_obs == site_obs)
        ]
        .set_index("rel_id")
        .sort_index()
    )

    assert np.all(site_int_sims.index == cur_results_df.index)

    cur_gen_gm_params = (
        gen_gm_params.loc[
            (gen_gm_params.event == event) & (gen_gm_params.site == site_int)
        ].squeeze()
        if gen_gm_params is not None
        else None
    )
    cur_syn_obs_gm_params = (
        syn_obs_gm_params.loc[
            (syn_obs_gm_params.event == event) & (syn_obs_gm_params.site == site_int)
        ].squeeze()
        if syn_obs_gm_params is not None
        else None
    )

    if "prob" in cur_results_df.columns:
        st.text(f"Probabilities Standard Deviation: {cur_results_df.prob.std():.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        show_gen_dist = st.checkbox(
            "Show generation distribution", value=False, key=f"{tab_type}_show_gen"
        )
    with col2:
        show_obs_dist = st.checkbox(
            "Show synthetic observation distribution",
            value=False,
            key=f"{tab_type}_show_syn_obs",
        )
    with col3:
        show_obs_site = st.checkbox(
            "Show observation site", value=True, key=f"{tab_type}_show_obs"
        )

    create_pSA_dist_plot(
        cur_results_df,
        site_int_sims,
        site_int_obs,
        high_rels=high_rels if len(high_rels) > 0 else None,
        site_obs_obs=site_obs_obs if show_obs_site else None,
        gen_gm_params=cur_gen_gm_params if show_gen_dist else None,
        syn_obs_gm_params=cur_syn_obs_gm_params if show_obs_dist else None,
    )

    if "prob" in cur_results_df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                cur_results_df[["prob", "misfit_score"]]
                .sort_values("prob", ascending=False)
                .head(10)
            )
        with col2:
            st.dataframe(
                cur_results_df[["prob", "misfit_score"]]
                .sort_values("misfit_score", ascending=True)
                .head(10)
            )

        create_misfit_dist_plot(cur_results_df, tab_type)

        col1, col2, col3 = st.columns(3)
        with col1:
            ims = st.multiselect(
                "IMs",
                sr.constants.PSA_KEYS,
                key=f"{tab_type}_im",
                default=[
                    "pSA_0.01",
                    "pSA_0.1",
                    "pSA_0.5",
                    "pSA_1.0",
                    "pSA_2.5",
                    "pSA_5.0",
                    "pSA_10.0",
                ],
            )
        with col2:
            n_bins = st.slider("Number of Bins", 5, 50, 10, key=f"{tab_type}_n_bins")
        with col3:
            cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_cdf")

        create_dist_plot(
            site_int_sims,
            site_int_obs,
            cur_results_df,
            n_bins,
            ims,
            cdf,
            site_obs_obs,
            cur_gen_gm_params,
            cur_syn_obs_gm_params,
        )

    with st.expander("Raw data"):
        st.dataframe(cur_results_df)


def create_pSA_dist_plot(
    results_df: pd.DataFrame,
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    high_rels: List[str] = None,
    site_obs_obs: pd.DataFrame = None,
    gen_gm_params: pd.DataFrame = None,
    syn_obs_gm_params: pd.DataFrame = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    mean_cols = [f"{cur_key}_mean" for cur_key in sr.constants.PSA_KEYS]
    std_cols = [f"{cur_key}_std_Total" for cur_key in sr.constants.PSA_KEYS]

    if syn_obs_gm_params is not None:
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

    if gen_gm_params is not None:
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

    if site_obs_obs is not None:
        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_obs.squeeze().loc[sr.constants.PSA_KEYS].values,
            label="Observed - Observation Site",
            c="magenta",
            linewidth=1.0,
        )

    if "prob" in results_df.columns:
        weighted_avg = einops.einsum(
            results_df.prob.values,
            np.log(site_int_sims.loc[:, sr.constants.PSA_KEYS].values),
            "i, i j -> j",
        )
        weighted_std = np.sqrt(
            einops.einsum(
                results_df.prob.values,
                (
                    np.log(site_int_sims.loc[:, sr.constants.PSA_KEYS].values)
                    - weighted_avg
                )
                ** 2,
                "i, i j -> j",
            )
            / np.sum(results_df.prob.values)
        )
    else:
        im_prob_cols = [f"{cur_im}_prob" for cur_im in sr.constants.PSA_KEYS]
        assert np.allclose(results_df[im_prob_cols].sum(), 1.0)

        weighted_avg = einops.einsum(
            results_df[im_prob_cols].values,
            site_int_sims.loc[:, sr.constants.PSA_KEYS].values,
            "i j, i j -> j",
        )
        weighted_std = np.sqrt(
            einops.einsum(
                results_df[im_prob_cols].values,
                (site_int_sims.loc[:, sr.constants.PSA_KEYS].values - weighted_avg)
                ** 2,
                "i j, i j -> j",
            )
        )

    ax.semilogx(
        sr.constants.PERIODS,
        weighted_avg,
        label="ML - Mean",
        c="blue",
    )
    ax.fill_between(
        sr.constants.PERIODS,
        weighted_avg + weighted_std,
        weighted_avg - weighted_std,
        alpha=0.4,
        label="ML +/- 1 Std",
        color="lightblue",
    )
    ax.semilogx(sr.constants.PERIODS, weighted_avg + weighted_std, c="lightblue")
    ax.semilogx(sr.constants.PERIODS, weighted_avg - weighted_std, c="lightblue")

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


def create_dist_plot(
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    result_df: pd.DataFrame,
    n_bins: int,
    ims: Sequence[str],
    cdf: bool = False,
    site_obs_obs: pd.DataFrame = None,
    gen_gm_params: pd.DataFrame = None,
    syn_obs_gm_params: pd.DataFrame = None,
):

    for im in ims:
        col1, col2 = st.columns(2)

        if syn_obs_gm_params is not None:
            syn_obs_mean = syn_obs_gm_params.loc[f"{im}_mean"]
            syn_obs_std = syn_obs_gm_params.loc[f"{im}_std_Total"]
            syn_obs_dist = stats.lognorm(s=syn_obs_std, scale=np.exp(syn_obs_mean))
            syn_obs_x = np.linspace(
                syn_obs_dist.ppf(0.0001), syn_obs_dist.ppf(0.98), 1000
            )

            if gen_gm_params is not None:
                gen_mean = gen_gm_params.loc[f"{im}_mean"]
                gen_std = gen_gm_params.loc[f"{im}_std_Total"]
                gen_dist = stats.lognorm(s=gen_std, scale=np.exp(gen_mean))
                gen_x = np.linspace(gen_dist.ppf(0.0001), gen_dist.ppf(0.98), 1000)

        # IMs
        with col1:
            if cdf:
                fig, ax = plt.subplots(figsize=(6, 4))

                sort_inds = np.argsort(site_int_sims[im])

                # Prior (Sample)
                ax.step(
                    site_int_sims[im][sort_inds],
                    np.linspace(0, 1, site_int_sims.shape[0] + 1)[1:],
                    label="Sampled Prior Dist",
                    c="k",
                    where="post",
                    linestyle="--",
                    linewidth=1.0,
                )

                # Posterior (Model)
                ax.step(
                    site_int_sims[im][sort_inds],
                    np.cumsum(
                        result_df.loc[site_int_sims.index, "prob"].values[sort_inds]
                    ),
                    label="Model (Posterior) Dist",
                    c="b",
                    where="post",
                )
                sns.rugplot(x=site_int_sims[im], ax=ax, color="k")
                ax.axvline(site_int_obs[im], c="r", label="Observed - Site of Interest")

                if site_obs_obs is not None:
                    if site_obs_obs.shape[0] < 5:
                        for ix, cur_obs_row in site_obs_obs.iterrows():
                            ax.axvline(
                                cur_obs_row[im],
                                linestyle="--",
                                label=f"Observed - {cur_obs_row.site_id}",
                                c="g",
                                linewidth=1.0,
                            )

                if syn_obs_gm_params is not None:
                    syn_obs_y = syn_obs_dist.cdf(syn_obs_x)
                    ax.plot(
                        syn_obs_x,
                        syn_obs_y,
                        label="Synthetic Observations Dist",
                        c="purple",
                        linewidth=1.0,
                    )

                    if gen_gm_params is not None:
                        gen_y = gen_dist.cdf(gen_x)
                        ax.plot(
                            gen_x,
                            gen_y,
                            label="Generation (Prior) Dist",
                            c="k",
                            linewidth=1.0,
                        )

                ax.legend(fontsize="small")
                ax.set_title(f"P({im} < x)")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                ax.set_ylim([-0.05, 1.05])
                # fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))

                ax.hist(
                    # np.log(site_int_sims[im]),
                    site_int_sims[im],
                    weights=result_df.loc[site_int_sims.index, "prob"].values,
                    bins=n_bins,
                    density=True,
                    label="Model distribution",
                )
                sns.rugplot(x=site_int_sims[im], ax=ax, color="k")

                ax.axvline(site_int_obs[im], c="r", label="Observed - Site of Interest")
                if site_obs_obs is not None:
                    if site_obs_obs.shape[0] < 5:
                        for ix, cur_obs_row in site_obs_obs.iterrows():
                            ax.axvline(
                                cur_obs_row[im],
                                linestyle="--",
                                label=f"Observed - {cur_obs_row.site_id}",
                                c="g",
                                linewidth=1.0,
                            )

                # Synthetic observations distribution
                if syn_obs_gm_params is not None:
                    syn_obs_y = syn_obs_dist.pdf(syn_obs_x)
                    ax.plot(
                        syn_obs_x,
                        syn_obs_y,
                        label="Synthetic Observations Distribution",
                        c="purple",
                        linewidth=1.0,
                    )

                    # Generation distribution
                    if gen_gm_params is not None:
                        gen_y = gen_dist.pdf(gen_x)
                        ax.plot(
                            gen_x,
                            gen_y,
                            label="Generation Distribution (Prior)",
                            c="k",
                            linewidth=1.0,
                        )

                ax.legend(fontsize="small")
                ax.set_title(f"{im} Distribution")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col2:
            if cdf:
                fig, ax = plt.subplots(figsize=(6, 4))

                sort_inds = np.argsort(result_df.loc[:, im].values)

                # Prior (Sample)
                ax.step(
                    result_df.loc[:, im].values[sort_inds],
                    np.linspace(0, 1, site_int_sims.shape[0] + 1)[1:],
                    label="Sampled Prior Dist",
                    c="k",
                    where="post",
                    linestyle="--",
                    linewidth=1.0,
                )

                ax.step(
                    result_df.loc[:, im].values[sort_inds],
                    np.cumsum(
                        result_df.loc[site_int_sims.index, "prob"].values[sort_inds]
                    ),
                    label="Model distribution",
                    c="b",
                    where="post",
                )
                sns.rugplot(x=result_df.loc[:, im], ax=ax, color="k")
                ax.axvline(0.0, c="r", label="Observed - Site of Interest")
                ax.set_title(f"P({im} < x)")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                ax.set_ylim([-0.05, 1.05])
                ax.set_xlim(-2, 2)
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                fig, ax = plt.subplots(figsize=(6, 4))

                ax.hist(
                    result_df.loc[:, im],
                    weights=result_df.loc[site_int_sims.index, "prob"].values,
                    bins=n_bins,
                    label="Model distribution",
                )
                sns.rugplot(x=result_df.loc[:, im], ax=ax, color="k")

                ax.set_title(f"{im} Residual Distribution")
                ax.set_xlim(-2.0, 2.0)
                ax.axvline(0.0, c="r")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


def run_ind_samples(
    results_dir: Path,
    sample_results: Tuple[pd.DataFrame, pd.DataFrame],
    gen_gm_params_ffp: Path = None,
    syn_obs_gm_params_ffp: Path = None,
):
    train_sample_results, val_sample_results = sample_results

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    gen_gm_params = (
        st_utils.load_gm_params(gen_gm_params_ffp)
        if gen_gm_params_ffp is not None
        else None
    )
    syn_obs_gm_params = (
        st_utils.load_gm_params(syn_obs_gm_params_ffp)
        if syn_obs_gm_params_ffp is not None
        else None
    )

    with train_tab:
        _sample_viewer(
            results_dir,
            train_sample_results,
            "train_sample",
            gen_gm_params,
            syn_obs_gm_params,
        )

    with val_tab:
        _sample_viewer(
            results_dir,
            val_sample_results,
            "val_sample",
            gen_gm_params,
            syn_obs_gm_params,
        )


def run_ind_scenario(
    results_dir: Path,
    sample_results: Tuple[pd.DataFrame, pd.DataFrame],
    scenario_results: Tuple[pd.DataFrame, pd.DataFrame],
    gen_gm_params_ffp: Path = None,
    syn_obs_gm_params_ffp: Path = None,
):
    train_sample_results, val_sample_results = sample_results
    train_scenario_results, val_scenario_results = scenario_results

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    gen_gm_params = (
        st_utils.load_gm_params(gen_gm_params_ffp)
        if gen_gm_params_ffp is not None
        else None
    )
    syn_obs_gm_params = (
        st_utils.load_gm_params(syn_obs_gm_params_ffp)
        if syn_obs_gm_params_ffp is not None
        else None
    )

    with train_tab:
        _scenario_viewer(
            results_dir,
            train_scenario_results,
            train_sample_results,
            "train_scenario",
            gen_gm_params,
            syn_obs_gm_params,
        )

    with val_tab:
        _scenario_viewer(
            results_dir,
            val_scenario_results,
            val_sample_results,
            "val_scenario",
            gen_gm_params,
            syn_obs_gm_params,
        )


def agg_residuals(
    result_dir: Path,
    results_df: pd.DataFrame,
    tab_type: str,
):
    im_residuals = results_df.loc[:, sr.constants.PSA_KEYS].values
    weights = results_df.loc[:, "prob"].values

    weighted_residual_mean = (
        einops.einsum(im_residuals, weights, "i j, i -> j") / results_df.prob.sum()
    )
    weighted_residual_std = np.sqrt(
        np.sum(
            weights[:, None] * (im_residuals - weighted_residual_mean[None, :]) ** 2,
            axis=0,
        )
        / results_df.prob.sum()
    )
    mean = np.mean(im_residuals, axis=0)
    std = np.std(im_residuals, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean,
        label=f"Weighted (Posterior) Residual Mean, {np.mean(weighted_residual_mean):.2f}",
        c="b",
    )
    ax.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean + weighted_residual_std,
        label=f"Weighted (Posterior) Residual Std, {np.mean(weighted_residual_std):.2f}",
        c="b",
        linestyle="--",
    )
    ax.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean - weighted_residual_std,
        c="b",
        linestyle="--",
    )
    ax.semilogx(
        sr.constants.PERIODS,
        mean,
        label=f"Uniform (Prior) Residual Mean, {np.mean(mean):.2f}",
        c="r",
    )
    ax.semilogx(
        sr.constants.PERIODS,
        mean + std,
        label=f"Uniform (Prior) Residual Std, {np.mean(std):.2f}",
        c="r",
        linestyle="--",
    )
    ax.semilogx(
        sr.constants.PERIODS,
        mean - std,
        c="r",
        linestyle="--",
    )

    ax.set_xlabel(f"Period (s)")
    ax.set_ylabel(f"pSA")
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def posterior_probs_inv(cur_results_dir: Path, results_df: pd.DataFrame, tab_type: str):
    if "prob" in results_df.columns:
        prob_key = "prob"
    else:
        prob_key = st.selectbox(
            "Probability Key",
            [cur_c for cur_c in results_df.columns if "_prob" in cur_c],
            key=f"{tab_type}_prob_key",
        )

    n_probs = st.number_input(
        "Number of Probabilities", 1, 100, 2, key=f"{tab_type}_n_probs"
    )

    group_cols = (
        ["event_id", "site_int", "site_obs"]
        if "site_obs" in results_df.columns
        else ["event_id", "site_int"]
    )
    groups = get_results_group(results_df, group_cols)
    top_k_probs = groups[prob_key].apply(lambda x: x.nlargest(n_probs).sum())

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(top_k_probs, bins=20)

    ax.set_xlabel(f"Top {n_probs} (summed) probabilities")
    ax.set_ylabel(f"Count")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def agg_scenario_vis(cur_results_dir: Path, sc_results_df: pd.DataFrame, sample_results_df: pd.DataFrame, tab_type: str):
    with st.expander("Posterior Probabilities"):
        posterior_probs_inv(cur_results_dir, sc_results_df, tab_type)
    st.divider()

    im = st.selectbox("IM", sr.constants.PSA_KEYS, key=f"{tab_type}_agg_scenario_im")
    with st.expander("Misfit"):
        misfit_hist(sc_results_df, im if f"{im}_misfit" in sc_results_df.columns else None)

    # Setup
    mean_sample_misfit = sum_result_df(sample_results_df, ["event_id", "site_int", "site_obs"])
    mean_scenario_misfit = sum_result_df(sc_results_df, ["event_id", "site_int"])
    group = mean_sample_misfit.sort_values(["event_id", "site_int"]).groupby(
        ["event_id", "site_int"], observed=True)
    group_keys = list(group.groups.keys())
    mean_scenario_misfit = mean_scenario_misfit.sort_values(["event_id", "site_int"])
    assert np.all(np.asarray(group_keys) == mean_scenario_misfit[["event_id", "site_int"]].values)

    sc_n_obs = group.size().values
    sc_min_dist = group["s2s_distance"].min().values
    sc_misfit = mean_scenario_misfit[f"{im}_misfit"].values

    with st.expander("Misfit vs Minimum Distance"):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(sc_min_dist, sc_misfit, s=5, alpha=0.5)
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlabel("Minimum distance")
        ax1.set_ylabel("Misfit")

        ax1.set_ylim(0, np.quantile(sc_misfit, 0.95))
        ax1.set_xlim(0, None)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with st.expander("Misfit vs Number of Observations"):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(sc_n_obs +  np.random.uniform(-0.4, 0.4, sc_n_obs.size), sc_misfit, s=5, alpha=0.5)
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlabel("Number of observations")
        ax1.set_ylabel("Misfit")

        ax1.set_ylim(0, np.quantile(sc_misfit, 0.95))
        ax1.set_xlim(0, None)


        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with st.expander("Number of Observations vs Minimum Distance"):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        cm = ax1.scatter(sc_n_obs + np.random.uniform(-0.4, 0.4, sc_n_obs.size), sc_min_dist, c=sc_misfit, s=5, alpha=0.5)
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlabel("Number of observations")
        ax1.set_ylabel("Minimum distance")

        fig.colorbar(cm, ax=ax1, label="Misfit", pad=0)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


    return


def sum_result_df(df: pd.DataFrame, group_keys=Sequence[str]):
    """
    Computes the mean misfit
    """
    ims = sr.constants.PSA_KEYS
    prob_cols = mlt.array_utils.numpy_str_join("_", ims, "prob")
    misfit_cols = mlt.array_utils.numpy_str_join("_", ims, "misfit")
    weight_cols = mlt.array_utils.numpy_str_join("_", ims, "site_weights")

    result_df = pd.DataFrame(
        data=df[prob_cols].values * df[misfit_cols].values,
        index=df.index,
        columns=misfit_cols,
    )
    result_df[group_keys] = df[group_keys]
    result_df = result_df.groupby(group_keys, observed=True).mean()

    if weight_cols[0] in df.columns:
        result_df[weight_cols] = df.groupby(group_keys, observed=True)[weight_cols].first()
    if "s2s_distance" in df.columns:
        result_df["s2s_distance"] = df.groupby(group_keys, observed=True)["s2s_distance"].first()

    return result_df.reset_index()

def run_agg_scenario(cur_results_dir: Path, sample_results: Tuple[pd.DataFrame, pd.DataFrame],
                     scenario_results: Tuple[pd.DataFrame, pd.DataFrame]):
    train_scenario_results, val_scenario_results = scenario_results
    train_sample_results, val_sample_results = sample_results

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        agg_scenario_vis(cur_results_dir, train_scenario_results, train_sample_results, "train_scenario")

    with val_tab:
        agg_scenario_vis(cur_results_dir, val_scenario_results, val_sample_results, "val_scenario")


def agg_single_viewer(results_df: pd.DataFrame, tab_type: str):
    im = st.selectbox("IM", sr.constants.PSA_KEYS, key=f"{tab_type}_agg_single_im")

    with st.expander("Site Weights Histogram"):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(results_df[f"{im}_site_weights"], bins=20, range=(0, 1))
        ax.set_title(f"{im}")
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel(f"{im} Site Weights")
        ax.set_xlim([0, 1])
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with st.expander("Sample Misfit"):
        misfit_hist(results_df, im if f"{im}_misfit" in results_df.columns else None)

    with st.expander("Misfit vs Site Weights"):
        misfit_vs_site_weights(results_df, im)


def misfit_vs_site_weights(results_df: pd.DataFrame, im: str):
    hist_range = ((0.0, 1.0), (0.0, 0.1))
    bins = (20, 20)
    fig, ax = plt.subplots(figsize=(12, 6))

    *_, cm = plt.hist2d(
        results_df[f"{im}_site_weights"],
        results_df[f"{im}_misfit"],
        bins=bins,
        range=hist_range,
        cmap="Blues",
        vmin=0,
        # vmax = 10_000
    )

    ax.set_xlabel(f"{im} Site Weights")
    ax.set_ylabel(f"{im} Misfit Score")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.colorbar(cm, ax=ax, pad=0, label="Number of Samples")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def misfit_hist(results_df: pd.DataFrame, im: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    if im is not None:
        ax.hist(results_df[f"{im}_misfit"], bins=20, range=(0, 1), label="Unweighted")
        ax.set_xlabel(f"{im} Misfit Score")
    else:
        ax.hist(results_df["misfit_score"], bins=20)
        ax.set_xlabel(f"Misfit Score")

    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylabel("Number of Samples")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def run_agg_single(cur_results_dir: Path, sample_results: Tuple[pd.DataFrame, pd.DataFrame]):
    train_tab, val_tab = st.tabs(["Training", "Validation"])

    train_results_df, val_results_df = sample_results

    with train_tab:
        agg_single_viewer(train_results_df, "train")

    with val_tab:
        agg_single_viewer(val_results_df, "val")


def main(
    results_dir: Path,
    gen_gm_params_ffp: Path = typer.Option(
        None, help="Only provide when using realisations from empirical GMMs"
    ),
):
    st.set_page_config(layout="wide")

    result_id = st.selectbox(
        "Results Directory",
        sorted(
            [
                cur_ffp.stem
                for cur_ffp in results_dir.iterdir()
                if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
            ]
        ),
    )
    cur_results_dir = results_dir / result_id

    if st_utils.ml_get_metadata(cur_results_dir)["method_type"] not in [4, 5]:
        st.error("This app is only for the results of the ML method type")
        return

    (
        general_tab,
        ind_sample_tab,
        ind_scenario_tab,
        agg_single_tab,
        agg_scenario_tab,
    ) = st.tabs(
        [
            "General",
            "Individual Sample",
            "Individual Scenario",
            "Aggregate Sample",
            "Aggregate Scenario",
        ]
    )

    meta = st_utils.ml_get_metadata(cur_results_dir)
    syn_obs_gm_params_ffp = (
        fp
        if (
            fp := (Path(os.path.expandvars("$wdata")) / meta["data"]["db"]).parent
            / "syn_gm_params.csv"
        ).exists()
        else None
    )

    sample_results = st_utils.ml_load_sample_results(cur_results_dir)
    scenario_results = st_utils.ml_load_scenario_results(cur_results_dir)

    with general_tab:
        # pass
        run_general_tab(cur_results_dir)

    with ind_sample_tab:
        # pass
        run_ind_samples(
            cur_results_dir,
            sample_results,
            gen_gm_params_ffp=gen_gm_params_ffp,
            syn_obs_gm_params_ffp=syn_obs_gm_params_ffp,
        )

    with ind_scenario_tab:
        # pass
        run_ind_scenario(
            cur_results_dir,
            sample_results,
            scenario_results,
            gen_gm_params_ffp=gen_gm_params_ffp,
            syn_obs_gm_params_ffp=syn_obs_gm_params_ffp,
        )

    with agg_single_tab:
        run_agg_single(cur_results_dir, sample_results)

    with agg_scenario_tab:
        # pass
        run_agg_scenario(cur_results_dir, sample_results, scenario_results)


if __name__ == "__main__":
    typer.run(main)
