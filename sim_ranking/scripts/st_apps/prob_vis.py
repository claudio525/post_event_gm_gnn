import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import typer
import scipy.stats as stats

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
def get_dist_matrix(results_dir: Path):
    site_df = get_site_df(results_dir)

    return sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )

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

    dist_matrix = get_dist_matrix(results_dir)

    train_results_df["s2s_distance"] = dist_matrix.values[
        dist_matrix.index.get_indexer_for(train_results_df.site_int.values),
        dist_matrix.columns.get_indexer_for(train_results_df.site_obs.values),
    ]
    val_results_df["s2s_distance"] = dist_matrix.values[
        dist_matrix.index.get_indexer_for(val_results_df.site_int.values),
        dist_matrix.columns.get_indexer_for(val_results_df.site_obs.values),
    ]

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
        st.text(f"{'Number of realisations:':<{padding}} {meta['run_config']['n_rels']}")
        st.text(f"{'Max distance:':<{padding}} {meta['run_config']['max_dist']}")


def _create_event_map(
    event: str,
    site_df: pd.DataFrame,
    results_df: pd.DataFrame,
    event_df: pd.DataFrame,
    site_int: str,
    site_obs: str = None,
):
    # Map
    event_sites = np.unique(results_df[results_df.event_id == event].site_int.values)
    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[event_sites].lat,
                lon=site_df.loc[event_sites].lon,
                mode="markers",
                marker=dict(size=10),
                hovertext=event_sites,
                hoverinfo="text",
                name="Sites",
            ),
            go.Scattermapbox(
                lat=[event_df.loc[event, "lat"]],
                lon=[event_df.loc[event, "lon"]],
                mode="markers",
                marker=dict(size=20, color="orange"),
                hovertext=event,
                hoverinfo="text",
                name="Event",
            ),
            go.Scattermapbox(
                lat=[site_df.loc[site_int, "lat"]],
                lon=[site_df.loc[site_int, "lon"]],
                mode="markers",
                marker=dict(size=10, color="red"),
                hovertext=site_int,
                hoverinfo="text",
                name="Site of Interest",
            ),
        ]
    )
    if site_obs is not None:
        fig.add_trace(
            go.Scattermapbox(
                lat=[site_df.loc[site_obs, "lat"]],
                lon=[site_df.loc[site_obs, "lon"]],
                mode="markers",
                marker=dict(size=10, color="maroon"),
                hovertext=site_obs,
                hoverinfo="text",
                name="Observation Site",
            ),
        )

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=site_df.loc[event_sites].lat.mean(),
            lon=site_df.loc[event_sites].lon.mean(),
        ),
        zoom=8,
    )

    return fig

def _scenario_viewer(results_dir: Path, results_df: pd.DataFrame, tab_type: str):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

    events = results_df.event_id.unique().astype("str")

    col1, col2 = st.columns([1, 6])

    with col1:
        event = st.selectbox("Event", events, key=f"{tab_type}_event")

        site_int = st.selectbox(
            "Site of Interest",
            results_df.loc[(results_df.event_id == event)]
            .site_int.unique()
            .astype("str"),
            key=f"{tab_type}_site_int",
        )

        high_rel = st.selectbox(
            "Highlighted Realisation",
            ["---"]
            + sorted(
                sim_df.loc[(sim_df.event_id == event)]
                .rel_id.unique()
                .astype(str)
                .tolist()
            ),
            key=f"{tab_type}_high_rel",
        )

        st.markdown(f"Magnitude: {event_df.loc[event].mag}")

    with col2:
        fig = _create_event_map(event, site_df, results_df, event_df, site_int)
        st.plotly_chart(fig, use_container_width=True)

    # Get the relevant data
    cur_event_rels = results_df.loc[
        (results_df.event_id == event)].rel_id.unique().astype(str)

    cur_scenario_df = (
        results_df.loc[
            (results_df.event_id == event)
            & (results_df.site_int == site_int)
        ]
    )
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_int)
        & np.isin(sim_df.rel_id, cur_event_rels)
    ]

    cur_results_df = results_df.loc[
        (results_df.event_id == event)
        & (results_df.site_int == site_int)
        ].set_index("rel_id")

    fig = create_dist_plot(site_int_sims, site_int_obs, cur_results_df, tab_type)
    st.pyplot(fig, use_container_width=False)


def _sample_viewer(results_dir: Path, results_df: pd.DataFrame , type: str):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

    events = results_df.event_id.unique().astype("str")

    col1, col2 = st.columns([1, 6])

    with col1:
        event = st.selectbox("Event", events, key=f"{type}_event")

        site_int = st.selectbox(
            "Site of Interest",
            results_df.loc[(results_df.event_id == event)]
            .site_int.unique()
            .astype("str"),
            key=f"{type}_site_int",
        )

        site_obs = st.selectbox(
            "Observation Site",
            results_df.loc[
                (results_df.event_id == event) & (results_df.site_int == site_int)
            ]
            .site_obs.unique()
            .astype(str),
            key=f"{type}_site_obs",
        )

        high_rel = st.selectbox(
            "Highlighted Realisation",
            ["---"]
            + sorted(
                sim_df.loc[(sim_df.event_id == event)]
                .rel_id.unique()
                .astype(str)
                .tolist()
            ),
            key=f"{type}_high_rel",
        )

    with col2:
        fig = _create_event_map(
            event, site_df, results_df, event_df, site_int, site_obs
        )

        st.plotly_chart(fig, use_container_width=True)

    cur_event_rels = results_df.loc[(results_df.event_id == event)].rel_id.unique().astype(str)

    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_int)
        & np.isin(sim_df.rel_id, cur_event_rels)
    ]
    site_obs_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_obs_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_obs)
        & np.isin(sim_df.rel_id, cur_event_rels)
    ]

    cur_results_df = results_df.loc[
        (results_df.event_id == event)
        & (results_df.site_int == site_int)
        & (results_df.site_obs == site_obs)
        ].set_index("rel_id")

    fig = create_dist_plot(site_int_sims, site_int_obs, cur_results_df, type)
    st.pyplot(fig, use_container_width=False)

def create_dist_plot(
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    result_df: pd.DataFrame,
    type: str,
    emp_gm_params: pd.DataFrame = None,
):
    col1, col2, col3 = st.columns(3)
    with col1:
        im = st.selectbox("IM", sr.constants.PSA_KEYS, key=f"{type}_im")
    with col2:
        n_bins = st.slider("Number of Bins", 5, 50, 10, key=f"{type}_n_bins")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        np.log(site_int_sims[im]),
        weights=result_df.loc[site_int_sims.rel_id, "prob"].values,
        bins=n_bins,
        density=True,
        label="Model distribution",
    )

    if emp_gm_params is not None:
        mean = emp_gm_params[f"{im}_mean"]
        std = emp_gm_params[f"{im}_std_Total"]

        x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
        rv = stats.norm(mean, std)
        t = rv.pdf(x)

        ax.plot(x, t, c="g", label="Empirical GMM distribution")

    ax.axvline(np.log(site_int_obs[im]), c="r", label="Observed")
    ax.legend()
    fig.tight_layout()
    return fig

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
    train_scenario_results, val_scenario_results = load_scenario_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _scenario_viewer(
            results_dir,
            train_scenario_results,
            "train_scenario",
        )

    with val_tab:
        _scenario_viewer(
            results_dir,
            val_scenario_results,
            "val_scenario",
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
        pass
        # run_agg_single(cur_results_dir)

    with agg_scenario_tab:
        pass
        # run_agg_scenario(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
