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


@st.cache_data
def get_metadata(results_dir: Path):
    return sr.data.get_meta(results_dir)


@st.cache_data
def load_training_metrics(results_dir: Path):
    metrics = pd.read_pickle(results_dir / "metrics.pickle")

    return metrics


@st.cache_data
def get_site_df(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_site_df()


@st.cache_data
def get_event_df(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_df()


@st.cache_data
def get_event_sites(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_sites()


@st.cache_data
def get_event_angular_distances(results_dir: Path):
    station_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    event_sites = get_event_sites(results_dir)

    return sr.ml.features.compute_angular_distance(
        station_df,
        event_df,
        event_df.index.values.astype(str),
        event_sites,
        pre_process=False,
    )


@st.cache_data
def get_obs_df(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_obs_df()


@st.cache_data
def get_sim_df(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_sim_df()


@st.cache_data
def load_scenario_results(results_dir: Path):
    train_results = pd.read_parquet(results_dir / "train_scenario_results.parquet")
    val_results = pd.read_parquet(results_dir / "val_scenario_results.parquet")

    return train_results, val_results


@st.cache_data
def load_sample_results(results_dir: Path):
    train_results_df = pd.read_parquet(results_dir / "train_sample_results.parquet")
    val_results_df = pd.read_parquet(results_dir / "val_sample_results.parquet")

    # dist_matrix = get_dist_matrix(results_dir)
    #
    # train_results_df["s2s_distance"] = dist_matrix.values[
    #     dist_matrix.index.get_indexer_for(train_results_df.site_int.values),
    #     dist_matrix.columns.get_indexer_for(train_results_df.site_obs.values),
    # ]
    # val_results_df["s2s_distance"] = dist_matrix.values[
    #     dist_matrix.index.get_indexer_for(val_results_df.site_int.values),
    #     dist_matrix.columns.get_indexer_for(val_results_df.site_obs.values),
    # ]

    angular_distances = get_event_angular_distances(results_dir)
    for cur_event in train_results_df.event_id.unique():
        train_results_df.loc[
            train_results_df.event_id == cur_event, "angular_distance"
        ] = np.rad2deg(
            angular_distances[cur_event].values[
                angular_distances[cur_event].index.get_indexer_for(
                    train_results_df.loc[
                        train_results_df.event_id == cur_event
                    ].site_int.values
                ),
                angular_distances[cur_event].columns.get_indexer_for(
                    train_results_df.loc[
                        train_results_df.event_id == cur_event
                    ].site_obs.values
                ),
            ]
        )
    for cur_event in val_results_df.event_id.unique():
        val_results_df.loc[
            val_results_df.event_id == cur_event, "angular_distance"
        ] = np.rad2deg(
            angular_distances[cur_event].values[
                angular_distances[cur_event].index.get_indexer_for(
                    val_results_df.loc[
                        val_results_df.event_id == cur_event
                    ].site_int.values
                ),
                angular_distances[cur_event].columns.get_indexer_for(
                    val_results_df.loc[
                        val_results_df.event_id == cur_event
                    ].site_obs.values
                ),
            ]
        )

    return train_results_df, val_results_df


def run_general_tab(results_dir: Path):
    # Load the metadata
    meta = get_metadata(results_dir)

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
        st.text(f"{'FC Units:':<{padding}} {str(meta['hp_config']['fc_units'])}")
        st.text(f"{'L2:':<{padding}} {meta['hp_config']['l2_reg']}")

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
):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

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

        high_rels = st.multiselect(
            "Highlighted Realisations",
            sorted(
                sim_df.loc[(sim_df.event_id == event)]
                .rel_id.unique()
                .astype(str)
                .tolist()
            ),
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
    cur_event_rels = (
        scenario_results.loc[(scenario_results.event_id == event)]
        .rel_id.unique()
        .astype(str)
    )

    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = (
        sim_df.loc[
            (sim_df.event_id == event)
            & (sim_df.site_id == site_int)
            & np.isin(sim_df.rel_id, cur_event_rels)
        ]
        .set_index("rel_id")
        .sort_index()
    )

    cur_scenario_df = (
        scenario_results.loc[
            (scenario_results.event_id == event)
            & (scenario_results.site_int == site_int)
        ]
        .set_index("rel_id")
        .sort_index()
    )

    cur_sample_results = sample_results.loc[
        (sample_results.event_id == event) & (sample_results.site_int == site_int)
    ]

    assert np.all(site_int_sims.index == cur_scenario_df.index)

    create_pSA_dist_plot(
        cur_scenario_df,
        site_int_sims,
        site_int_obs,
        high_rels if len(high_rels) > 0 else None,
    )

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

    col1, col2 = st.columns(2)
    with col1:
        misfit_n_bins = st.number_input(
            "Number of Bins", 5, 50, 10, key=f"{tab_type}_misfit_n_bins"
        )
    with col2:
        misfit_cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_misfit_cdf")

    fig, ax = plt.subplots(figsize=(12, 6))
    if misfit_cdf:
        sort_inds = np.argsort(cur_scenario_df.misfit_score)
        ax.step(
            cur_scenario_df.misfit_score.values[sort_inds],
            np.cumsum(cur_scenario_df.prob.values[sort_inds]),
            label="Model distribution",
            c="b",
            where="post",
        )
        # sns.rugplot(x=cur_scenario_df.misfit_score, ax=ax, color="k")
        ax.axvline(
            x=cur_scenario_df.misfit_score.min(),
            c="r",
            label=f"Lowest misfit - {cur_scenario_df.misfit_score.idxmin()}",
        )
        ax.set_title(f"P(Misfit Score < x)")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend()
        fig.tight_layout()
    else:
        ax.hist(
            cur_scenario_df.misfit_score,
            weights=cur_scenario_df.prob,
            bins=misfit_n_bins,
        )

        ax.set_xlabel("Misfit Score")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        ims = st.multiselect("IMs", sr.constants.PSA_KEYS, key=f"{tab_type}_ims")
    with col2:
        n_bins = st.slider("Number of Bins", 5, 100, 10, key=f"{tab_type}_n_bins")
    with col3:
        cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_cdf")

    create_dist_plot(site_int_sims, site_int_obs, cur_scenario_df, n_bins, ims, cdf)


def _sample_viewer(results_dir: Path, results_df: pd.DataFrame, tab_type: str):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

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

        high_rels = st.multiselect(
            "Highlighted Realisations",
            sorted(
                sim_df.loc[(sim_df.event_id == event)]
                .rel_id.unique()
                .astype(str)
                .tolist()
            ),
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
            (sim_df.event_id == event)
            & (sim_df.site_id == site_int)
            & np.isin(sim_df.rel_id, cur_event_rels)
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

    create_pSA_dist_plot(
        cur_results_df,
        site_int_sims,
        site_int_obs,
        high_rels if len(high_rels) > 0 else None,
    )

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

    col1, col2, col3 = st.columns(3)
    with col1:
        ims = st.multiselect("IMs", sr.constants.PSA_KEYS, key=f"{tab_type}_im")
    with col2:
        n_bins = st.slider("Number of Bins", 5, 50, 10, key=f"{tab_type}_n_bins")
    with col3:
        cdf = st.checkbox("CDF", value=False, key=f"{tab_type}_cdf")

    create_dist_plot(
        site_int_sims, site_int_obs, cur_results_df, n_bins, ims, cdf, site_obs_obs
    )


def create_pSA_dist_plot(
    results_df: pd.DataFrame,
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    high_rels: List[str] = None,
):
    cdf_x, cdf_y = [], []
    for cur_im in sr.constants.PSA_KEYS:
        cur_sort_ind = np.argsort(site_int_sims[cur_im].values)
        cdf_x.append(site_int_sims[cur_im].values[cur_sort_ind])
        cdf_y.append(np.cumsum(results_df.prob.values[cur_sort_ind]))

    cdf_x = pd.DataFrame(np.asarray(cdf_x).T, columns=sr.constants.PSA_KEYS)
    cdf_y = pd.DataFrame(np.asarray(cdf_y).T, columns=sr.constants.PSA_KEYS)

    qt_2, qt_16, qt_50, qt_84, qt_98 = sha.query_non_parametric_multi_cdf_invs(
        np.asarray([0.02, 0.16, 0.5, 0.84, 0.98]), cdf_x.T.values, cdf_y.T.values
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.semilogx(
        sr.constants.PERIODS, site_int_obs, label="Observed - Site of Interest", c="r"
    )

    plt.semilogx(sr.constants.PERIODS, qt_50, label="Model - Median", c="blue")

    plt.fill_between(
        sr.constants.PERIODS,
        qt_2,
        qt_98,
        alpha=0.4,
        label="Model - 2/98th",
        color="lightgreen",
    )
    plt.semilogx(
        sr.constants.PERIODS, qt_2, c="lightgreen", linestyle="--", linewidth=1.0
    )
    plt.semilogx(
        sr.constants.PERIODS, qt_98, c="lightgreen", linestyle="--", linewidth=1.0
    )

    plt.fill_between(
        sr.constants.PERIODS,
        qt_16,
        qt_84,
        alpha=0.4,
        label="Model - 16/84th",
        color="lightblue",
    )
    plt.semilogx(
        sr.constants.PERIODS, qt_16, c="lightblue", linestyle="--", linewidth=1.0
    )
    plt.semilogx(
        sr.constants.PERIODS, qt_84, c="lightblue", linestyle="--", linewidth=1.0
    )

    if high_rels is not None:
        colors = sns.color_palette("dark", len(high_rels))
        for ix, cur_rel in enumerate(high_rels):
            plt.semilogx(
                sr.constants.PERIODS,
                site_int_sims.loc[cur_rel, sr.constants.PSA_KEYS].values,
                label=f"{cur_rel}",
                c=colors[ix],
                linestyle="dashdot",
            )

    plt.xlabel(f"Period (s)")
    plt.ylabel(f"pSA")
    plt.xlim([0.01, 10])
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()

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
    emp_gm_params: pd.DataFrame = None,
):

    for im in ims:
        col1, col2 = st.columns(2)

        with col1:
            if cdf:
                fig, ax = plt.subplots(figsize=(6, 4))

                sort_inds = np.argsort(site_int_sims[im])

                plt.step(
                    site_int_sims[im][sort_inds],
                    np.cumsum(
                        result_df.loc[site_int_sims.rel_id, "prob"].values[sort_inds]
                    ),
                    label="Model distribution",
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
                            )

                ax.legend()
                ax.set_title(f"P({im} < x)")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))

                ax.hist(
                    # np.log(site_int_sims[im]),
                    site_int_sims[im],
                    weights=result_df.loc[site_int_sims.index, "prob"].values,
                    bins=n_bins,
                    # density=True,
                    label="Model distribution",
                )
                sns.rugplot(x=site_int_sims[im], ax=ax, color="k")

                # ax.axvline(np.log(site_int_obs[im]), c="r", label="Observed")
                ax.axvline(site_int_obs[im], c="r", label="Observed - Site of Interest")
                if site_obs_obs is not None:
                    if site_obs_obs.shape[0] < 5:
                        for ix, cur_obs_row in site_obs_obs.iterrows():
                            ax.axvline(
                                cur_obs_row[im],
                                linestyle="--",
                                label=f"Observed - {cur_obs_row.site_id}",
                            )

                if site_obs_obs is not None:
                    print(f"wtf")

                ax.legend()
                ax.set_title(f"{im} Distribution")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col2:
            if cdf:
                fig, ax = plt.subplots(figsize=(6, 4))

                sort_inds = np.argsort(result_df.loc[:, im].values)

                plt.step(
                    result_df.loc[:, im].values[sort_inds],
                    np.cumsum(
                        result_df.loc[site_int_sims.rel_id, "prob"].values[sort_inds]
                    ),
                    label="Model distribution",
                    c="b",
                    where="post",
                )
                sns.rugplot(x=result_df.loc[:, im], ax=ax, color="k")
                ax.axvline(0.0, c="r", label="Observed - Site of Interest")
                ax.set_title(f"P({im} < x)")
                ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
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


def run_ind_samples(results_dir: Path, emp_gm_params_ffp: Path = None):
    train_sample_results, val_sample_results = load_sample_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _sample_viewer(
            results_dir,
            train_sample_results,
            "train_sample",
        )

    with val_tab:
        _sample_viewer(
            results_dir,
            val_sample_results,
            "val_sample",
        )


def run_ind_scenario(results_dir: Path, emp_gm_params_ffp: Path = None):
    train_sample_results, val_sample_results = load_sample_results(results_dir)
    train_scenario_results, val_scenario_results = load_scenario_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _scenario_viewer(
            results_dir,
            train_scenario_results,
            train_sample_results,
            "train_scenario",
        )

    with val_tab:
        _scenario_viewer(
            results_dir,
            val_scenario_results,
            val_sample_results,
            "val_scenario",
        )


def agg_residuals(
    result_dir: Path,
    results_df: pd.DataFrame,
    tab_type: str,
    filters: List[Tuple[str, str, Any, Any, Any, Any]],
):
    # slider_vals = []
    # for ix, filter in enumerate(filters):
    #     slider_vals.append(
    #         st.slider(
    #             filter[0],
    #             filter[2],
    #             filter[3],
    #             filter[4],
    #             step=filter[5],
    #             key=f"{tab_type}_{ix}",
    #         )
    #     )
    #
    # for filter, val in zip(filters, slider_vals):
    #     filtered_results_df = results_df.loc[filtered_results_df[filter[1]] <= val]
    #
    # st.text(f"Number of (filtered) samples/scenarios: {int(np.ceil(filtered_results_df.prob.sum()))}")

    filtered_results_df = results_df

    im_residuals = filtered_results_df.loc[:, sr.constants.PSA_KEYS].values
    weights = filtered_results_df.loc[:, "prob"].values

    weighted_residual_mean = (
        einops.einsum(im_residuals, weights, "i j, i -> j")
        / filtered_results_df.prob.sum()
    )
    weighted_residual_std = np.sqrt(
        np.sum(
            weights[:, None] * (im_residuals - weighted_residual_mean[None, :]) ** 2,
            axis=0,
        )
        / filtered_results_df.prob.sum()
    )
    mean = np.mean(im_residuals, axis=0)
    std = np.std(im_residuals, axis=0)

    fig = plt.figure(figsize=(12, 6))

    plt.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean,
        label="Weighted Residual Mean",
        c="b",
    )
    plt.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean + weighted_residual_std,
        label="Weighted Residual Std",
        c="b",
        linestyle="--",
    )
    plt.semilogx(
        sr.constants.PERIODS,
        weighted_residual_mean - weighted_residual_std,
        c="b",
        linestyle="--",
    )
    plt.semilogx(
        sr.constants.PERIODS,
        mean,
        label="Residual Mean",
        c="r",
    )
    plt.semilogx(
        sr.constants.PERIODS,
        mean + std,
        label="Residual Std",
        c="r",
        linestyle="--",
    )
    plt.semilogx(
        sr.constants.PERIODS,
        mean - std,
        c="r",
        linestyle="--",
    )

    plt.xlabel(f"Period (s)")
    plt.ylabel(f"pSA")
    plt.xlim([0.01, 10])
    plt.ylim([-1.5, 1.5])
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def run_agg_single(cur_results_dir: Path):
    train_sample_results, val_sample_results = load_sample_results(cur_results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        agg_residuals(
            cur_results_dir,
            train_sample_results,
            "train_sample",
            [("Max Distance", "s2s_distance", 0.0, 50.0, 50.0, 5.0)],
        )

    with val_tab:
        agg_residuals(
            cur_results_dir,
            val_sample_results,
            "val_sample",
            [("Max Distance", "s2s_distance", 0.0, 50.0, 50.0, 5.0)],
        )


def run_agg_scenario(cur_results_dir: Path):
    train_scenario_results, val_scenario_results = load_scenario_results(
        cur_results_dir
    )

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        agg_residuals(
            cur_results_dir,
            train_scenario_results,
            "train_scenario",
            [
                (
                    "Max Distance",
                    "max_distance",
                    float(train_scenario_results.min_distance.min()),
                    float(train_scenario_results.min_distance.max()),
                    float(train_scenario_results.min_distance.max()),
                    5.0,
                ),
            ],
        )

    with val_tab:
        agg_residuals(
            cur_results_dir,
            val_scenario_results,
            "val_scenario",
            [
                (
                    "Min Distance",
                    "min_distance",
                    float(train_scenario_results.min_distance.min()),
                    float(train_scenario_results.min_distance.max()),
                    float(train_scenario_results.min_distance.max()),
                    5.0,
                ),
            ],
        )


def main(
    results_dir: Path,
    # emp_gm_params_ffp: Path = typer.Option(
    #     None, help="Only provide when using realisations from empirical GMMs"
    # ),
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

    with general_tab:
        # pass
        run_general_tab(cur_results_dir)

    with ind_sample_tab:
        # pass
        run_ind_samples(cur_results_dir)

    with ind_scenario_tab:
        # pass
        run_ind_scenario(cur_results_dir)

    with agg_single_tab:
        # pass
        run_agg_single(cur_results_dir)

    with agg_scenario_tab:
        # pass
        run_agg_scenario(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
