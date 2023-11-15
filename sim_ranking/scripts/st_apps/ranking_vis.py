import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import typer

import spatial_hazard as sh
import sim_ranking as sr
import ml_tools as mlt


@st.cache_data
def _get_metadata(results_dir: Path):
    return sr.data.get_meta(results_dir)


@st.cache_data
def load_training_metrics(results_dir: Path):
    # meta = _get_metadata(results_dir)
    metrics = pd.read_pickle(results_dir / "metrics.pickle")

    return metrics


@st.cache_data
def get_dist_matrix(results_dir: Path):
    site_df = get_site_df(results_dir)

    return sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )


@st.cache_data
def get_site_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_site_df()


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
def get_event_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_event_df()


@st.cache_data
def get_event_sites(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_event_sites()


@st.cache_data
def load_ranking_data(results_dir: Path):
    train_rankings = pd.read_csv(results_dir / "train_rankings.csv", index_col=0)
    val_rankings = pd.read_csv(results_dir / "val_rankings.csv", index_col=0)

    return train_rankings, val_rankings


@st.cache_data
def get_obs_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_obs_df()


@st.cache_data
def get_sim_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_sim_df()


@st.cache_data
def _get_prediction_data(results_dir: Path):
    scalar_features, sim_df, obs_df, model, metadata = sr.ml.pairwise_pred.prep_data(
        results_dir
    )

    return scalar_features, sim_df, obs_df, model, metadata


@st.cache_data
def get_record_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_record_df()


@st.cache_data
def _load_sample_residuals(results_dir: Path):
    train_residuals = pd.read_csv(
        results_dir / "train_sample_residuals.csv", index_col=0
    )
    val_residuals = pd.read_csv(results_dir / "val_sample_residuals.csv", index_col=0)

    return train_residuals, val_residuals


@st.cache_data
def _load_scenario_residuals(results_dir: Path):
    train_residuals = pd.read_csv(
        results_dir / "train_scenario_residuals.csv", index_col=0
    )
    val_residuals = pd.read_csv(results_dir / "val_scenario_residuals.csv", index_col=0)

    return train_residuals, val_residuals


@st.cache_data
def _load_results(results_dir: Path):
    train_results_df = pd.read_csv(
        results_dir / "train_results.csv",
        dtype=dict(event_id=str),
        index_col=0,
        na_filter=False,
    )
    val_results_df = pd.read_csv(
        results_dir / "val_results.csv",
        dtype=dict(event_id=str),
        index_col=0,
        na_filter=False,
    )

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
    meta = _get_metadata(results_dir)

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
        if (model_vis_ffp := results_dir / "res_model_vis.png").exists():
            st.image(str(model_vis_ffp))


def run_ind_samples(results_dir):
    train_results, val_results = _load_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _sample_viewer(results_dir, train_results, "train")

    with val_tab:
        _sample_viewer(results_dir, val_results, "val")


def _get_site_ranking(
    results_dir: Path,
    event_id: str,
    site_int: str,
    site_obs: str,
):
    scalar_features, sim_df, obs_df, model, metadata = _get_prediction_data(results_dir)

    pred, rel_combs = sr.ml.pairwise_pred.get_site_prediction(
        event_id,
        site_int,
        site_obs,
        _get_metadata(results_dir),
        scalar_features,
        model,
        sim_df,
        obs_df,
    )

    ranked_rels, combs_won = sr.ml.pairwise_pred.get_site_ranking(pred, rel_combs)
    return ranked_rels, combs_won


def _create_event_map(
    event: str,
    site_df: pd.DataFrame,
    results_df: pd.DataFrame,
    event_df: pd.DataFrame,
    site_int: str,
    site_obs: str = None,
):
    # Map
    event_sites = np.union1d(
        results_df[results_df.event_id == event].site_int.values,
        results_df[results_df.event_id == event].site_obs.values,
    )
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


def _sample_viewer(
    results_dir: Path,
    results_df: pd.DataFrame,
    type: str,
):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)
    record_df = get_record_df(results_dir)
    metadata = _get_metadata(results_dir)
    event_angular_distances = get_event_angular_distances(results_dir)
    dist_df = get_dist_matrix(results_dir)

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

    # Get realisation ranking
    ranked_rels, combs_won = _get_site_ranking(results_dir, event, site_int, site_obs)

    # Get the relevant data
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event) & (sim_df.site_id == site_int)
    ]
    site_obs_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_obs_sims = sim_df.loc[
        (sim_df.event_id == event) & (sim_df.site_id == site_obs)
    ]

    # Site of interest figure
    fig, ax = plt.subplots(figsize=(12, 6))
    for ix, (cur_id, cur_row) in enumerate(site_int_sims.iterrows()):
        c = "g" if cur_row.rel_id == ranked_rels[0] else "gray"
        if high_rel == cur_row.rel_id:
            c = "orange"

        plt.semilogx(
            sr.constants.PERIODS,
            cur_row[sr.constants.PSA_KEYS].values,
            label=r"$IM^{sim}_s$" if ix == 0 else None,
            c=c,
            linestyle="--",
            linewidth=None if c != "gray" else 1.0,
        )

    ax.semilogx(
        sr.constants.PERIODS,
        site_obs_obs[sr.constants.PSA_KEYS].values,
        label=r"$IM^{obs}_i$",
        linestyle="-",
        c="b",
    )

    ax.semilogx(
        sr.constants.PERIODS,
        site_int_obs[sr.constants.PSA_KEYS].values,
        label=r"$IM^{obs}_s$",
        linestyle="-",
        c="r",
    )

    ax.set_title("Site of Interest")
    ax.set_xlabel("Period")
    ax.set_ylabel("pSA")
    ax.set_xlim(0.01, 10.0)
    # ax.set_ylim(-2.0, 2.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

    # Observation site figure
    fig, ax = plt.subplots(figsize=(12, 6))
    for ix, (cur_id, cur_row) in enumerate(site_obs_sims.iterrows()):
        c = "g" if cur_row.rel_id == ranked_rels[0] else "gray"
        if high_rel == cur_row.rel_id:
            c = "orange"

        plt.semilogx(
            sr.constants.PERIODS,
            cur_row[sr.constants.PSA_KEYS].values,
            label=r"$IM^{sim}_i$" if ix == 0 else None,
            c=c,
            linestyle="--",
            linewidth=None if c != "gray" else 1.0,
        )

    ax.semilogx(
        sr.constants.PERIODS,
        site_obs_obs[sr.constants.PSA_KEYS].values,
        label=r"$IM^{obs}_i$",
        linestyle="-",
        c="b",
    )

    ax.semilogx(
        sr.constants.PERIODS,
        site_int_obs[sr.constants.PSA_KEYS].values,
        label=r"$IM^{obs}_s$",
        linestyle="-",
        c="r",
    )

    ax.set_title("Observation Site")
    ax.set_xlabel("Period")
    ax.set_ylabel("pSA")
    ax.set_xlim(0.01, 10.0)
    # ax.set_ylim(-2.0, 2.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

    cur_scalar_features_df = site_df.loc[
        [site_int, site_obs], metadata["data"]["features"]["site_features"]
    ]
    cur_scalar_features_df["site_to_site_distance"] = dist_df.loc[site_int, site_obs]
    cur_scalar_features_df["r_rup"] = (
        record_df.loc[(record_df.event_id == event)]
        .set_index("site_id")
        .loc[[site_int, site_obs], "r_rup"]
        .values
    )
    cur_scalar_features_df["angular_distance"] = event_angular_distances[event].loc[
        site_int, site_obs
    ]

    st.dataframe(cur_scalar_features_df)
    st.dataframe(combs_won.to_frame("Combinations Won").T)


def _run_res_tab(residuals_df: pd.DataFrame):

    mean = residuals_df.loc[:, sr.constants.PSA_KEYS].mean(axis=0)
    std = residuals_df.loc[:, sr.constants.PSA_KEYS].std(axis=0)

    st.markdown(
        """
        ### Residuals
        Residuals between the best realisation (based on model predictions) and the observations at the site of interest.
        """
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, cur_row in residuals_df.iloc[::5, :].iterrows():
        ax.semilogx(
            sr.constants.PERIODS,
            cur_row.loc[sr.constants.PSA_KEYS].values,
            c="gray",
            alpha=0.5,
            linewidth=1.0,
        )

    ax.semilogx(
        sr.constants.PERIODS,
        mean,
        c="b",
        label="Mean",
        marker="o",
        linestyle="-",
        markersize=2.5,
    )
    ax.semilogx(
        sr.constants.PERIODS, mean + std, c="b", linestyle="--", label="Std"
    )
    ax.semilogx(sr.constants.PERIODS, mean - std, c="b", linestyle="--")

    ax.set_xlabel("Period")
    ax.set_ylabel("pSA")
    ax.set_xlim(0.01, 10.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)


def run_agg_single(cur_results_dir):

    train_residuals, val_residuals = _load_sample_residuals(cur_results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _run_res_tab(train_residuals)
    with val_tab:
        _run_res_tab(val_residuals)

    pass


def _scenario_viewer(
    results_dir: Path, results_df: pd.DataFrame, ranking_df: pd.DataFrame, tab_type: str
):
    events = results_df.event_id.unique().astype("str")

    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

    col1, col2 = st.columns([1, 6])

    with col1:
        event = st.selectbox("Event", events, key=f"scenario_{tab_type}_event")

        site_int = st.selectbox(
            "Site of Interest",
            results_df.loc[(results_df.event_id == event)]
            .site_int.unique()
            .astype("str"),
            key=f"scenario_{tab_type}_site_int",
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
            key=f"scenario_{tab_type}_high_rel",
        )
    with col2:
        fig = _create_event_map(event, site_df, results_df, event_df, site_int)
        st.plotly_chart(fig, use_container_width=True)

    cur_rank_df = ranking_df.loc[
        (ranking_df.event_id == event)
        & (ranking_df.site_int == site_int)
        & (ranking_df["rank"] == 1),
        :,
    ].copy()

    # Perform weighted voting
    cur_rank_df.loc[:, "weight"] = 1.0
    weighted_rels = (
        cur_rank_df[["rel_id", "weight"]]
        .groupby("rel_id")
        .sum()
        .sort_values("weight", ascending=False)
    )
    best_rel = weighted_rels.index[0]

    # Get the relevant data
    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event) & (sim_df.site_id == site_int)
    ]

    # Site of interest figure
    fig, ax = _create_pSA_plot(
        site_int_obs.loc[sr.constants.PSA_KEYS].values,
        site_int_sims,
        best_rel,
        high_rel,
        "Site of Interest",
    )
    st.pyplot(fig, use_container_width=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(weighted_rels)
    with col2:
        st.dataframe(cur_rank_df.drop(columns=["rank", "comps_won"]))

    # Observation site plots
    site_obs = st.selectbox(
        "Observation Site",
        results_df.loc[
            (results_df.event_id == event) & (results_df.site_int == site_int)
        ]
        .site_obs.unique()
        .astype(str),
        key=f"scenario_{tab_type}_site_obs",
    )

    site_obs_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_obs_sims = sim_df.loc[
        (sim_df.event_id == event) & (sim_df.site_id == site_obs)
    ]

    fig, ax = _create_pSA_plot(
        site_int_obs.loc[sr.constants.PSA_KEYS].values,
        site_obs_sims,
        best_rel,
        high_rel,
        "Observation Site",
        site_obs_obs.loc[sr.constants.PSA_KEYS].values,
    )
    st.pyplot(fig, use_container_width=False)


def _create_pSA_plot(
    site_int_obs_values: np.ndarray,
    sim_rels: pd.DataFrame,
    best_rel: str = None,
    high_rel: str = None,
    title: str = None,
    site_obs_obs_values: np.ndarray = None,
):
    label = None
    fig, ax = plt.subplots(figsize=(12, 6))
    for ix, (cur_id, cur_row) in enumerate(sim_rels.iterrows()):
        cur_label, c = None, "gray"
        if cur_row.rel_id == best_rel:
            cur_label, c = best_rel, "g"
        if high_rel == cur_row.rel_id:
            cur_label, c = best_rel, "orange"
        if cur_label is None and label is None:
            cur_label = label = r"$IM^{sim}_s$"

        plt.semilogx(
            sr.constants.PERIODS,
            cur_row[sr.constants.PSA_KEYS].values,
            label=cur_label,
            c=c,
            linestyle="--",
            linewidth=None if c != "gray" else 1.0,
        )

    if site_obs_obs_values is not None:
        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_obs_values,
            label=r"$IM^{obs}_i$",
            linestyle="-",
            c="b",
        )

    ax.semilogx(
        sr.constants.PERIODS,
        site_int_obs_values,
        label=r"$IM^{obs}_s$",
        linestyle="-",
        c="r",
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("pSA")
    ax.set_xlim(0.01, 10.0)
    # ax.set_ylim(-2.0, 2.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def run_ind_scenario(results_dir):
    train_results, val_results = _load_results(results_dir)
    train_rankings_df, val_rankings_df = load_ranking_data(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _scenario_viewer(results_dir, train_results, train_rankings_df, "train")

    with val_tab:
        _scenario_viewer(results_dir, val_results, val_rankings_df, "val")


def run_agg_scenario(results_dir):
    train_residuals, val_residuals = _load_scenario_residuals(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _run_res_tab(train_residuals)
    with val_tab:
        _run_res_tab(val_residuals)

def main(results_dir: Path):
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
        run_ind_scenario(cur_results_dir)

    with agg_single_tab:
        run_agg_single(cur_results_dir)

    with agg_scenario_tab:
        run_agg_scenario(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
