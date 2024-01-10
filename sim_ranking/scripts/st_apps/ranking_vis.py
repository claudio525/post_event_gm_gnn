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


# plt.rcParams['text.usetex'] = True


@st.cache_data
def get_metadata(results_dir: Path):
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
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

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
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_df()


@st.cache_data
def get_event_sites(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_sites()


@st.cache_data
def load_sample_data(results_dir: Path):
    train_comps = pd.read_parquet(results_dir / "train_comp_results.parquet")
    val_comps = pd.read_parquet(results_dir / "val_comp_results.parquet")

    train_rankings = pd.read_parquet(results_dir / "train_sample_results.parquet")
    val_rankings = pd.read_parquet(results_dir / "val_sample_results.parquet")

    return train_rankings, val_rankings, train_comps, val_comps


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


# @st.cache_data
# def _get_prediction_data(results_dir: Path):
#     scalar_features, sim_df, obs_df, model, metadata = sr.ml.pairwise_pred.prep_data(
#         results_dir
#     )
#
#     return scalar_features, sim_df, obs_df, model, metadata


@st.cache_data
def get_record_df(results_dir: Path):
    metadata = get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_record_df()


# @st.cache_data
# def _load_sample_residuals(results_dir: Path):
#     train_residuals = pd.read_csv(
#         results_dir / "train_sample_residuals.csv", index_col=0
#     )
#     val_residuals = pd.read_csv(results_dir / "val_sample_residuals.csv", index_col=0)
#
#     return train_residuals, val_residuals


@st.cache_data
def load_scenario_results(results_dir: Path):
    train_results = pd.read_parquet(results_dir / "train_scenario_results.parquet")
    val_results = pd.read_parquet(results_dir / "val_scenario_results.parquet")

    return train_results, val_results

@st.cache_data
def get_emp_gm_params(emp_gm_params_ffp: Path):
    return (
        None
        if emp_gm_params_ffp is None
        else pd.read_csv(emp_gm_params_ffp, index_col=0)
    )


# @st.cache_data
# def _get_sim_residuals(results_dir: Path):
#     metadata = _get_metadata(results_dir)
#     db_ffp = os.path.expandvars(metadata["data"]["db"])
#
#     db = sr.db.DB(db_ffp)
#
#     return sr.ml.pairwise.compute_best_sim_res(db.get_sim_df(), db.get_obs_df())


@st.cache_data
def load_results(results_dir: Path):
    # train_results_df = pd.read_csv(
    #     results_dir / "train_comp_results.csv",
    #     dtype=dict(event_id=str),
    #     index_col=0,
    #     na_filter=False,
    # )
    # val_results_df = pd.read_csv(
    #     results_dir / "val_comp_results.csv",
    #     dtype=dict(event_id=str),
    #     index_col=0,
    #     na_filter=False,
    # )

    train_results_df = pd.read_parquet(results_dir / "train_comp_results.parquet")
    val_results_df = pd.read_parquet(results_dir / "val_comp_results.parquet")

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
        if (model_vis_ffp := results_dir / "res_model_vis.png").exists():
            st.image(str(model_vis_ffp))


def run_ind_samples(results_dir: Path, emp_gm_params_ffp: Path = None):
    emp_gm_params = get_emp_gm_params(emp_gm_params_ffp)

    train_results, val_results = load_results(results_dir)
    (
        train_rankings,
        val_rankings,
        train_comps,
        val_comps,
    ) = load_sample_data(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _sample_viewer(
            results_dir,
            train_results,
            train_rankings,
            train_comps,
            "train",
            emp_gm_params=emp_gm_params,
        )

    with val_tab:
        _sample_viewer(
            results_dir,
            val_results,
            val_rankings,
            val_comps,
            "val",
            emp_gm_params=emp_gm_params,
        )


# def _get_site_ranking(
#     results_dir: Path,
#     event_id: str,
#     site_int: str,
#     site_obs: str,
# ):
#     scalar_features, sim_df, obs_df, model, metadata = _get_prediction_data(results_dir)
#
#     pred, rel_combs = sr.ml.pairwise_pred.get_site_prediction(
#         event_id,
#         site_int,
#         site_obs,
#         get_metadata(results_dir),
#         scalar_features,
#         model,
#         sim_df,
#         obs_df,
#     )
#
#     ranked_rels, combs_won = sr.ml.pairwise_pred.get_site_ranking(pred, rel_combs)
#     return ranked_rels, combs_won


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
    ranking_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    type: str,
    emp_gm_params: pd.DataFrame = None,
):
    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)
    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)
    metadata = get_metadata(results_dir)

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

    # Get the relevant data
    cur_ranking_df = ranking_df.loc[
        (ranking_df.event_id == event)
        & (ranking_df.site_int == site_int)
        & (ranking_df.site_obs == site_obs)
    ].set_index("rel_id")
    assert cur_ranking_df.iloc[0]["rank"] == 1
    ranked_rels = cur_ranking_df.index.values.astype(str)

    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_int)
        & np.isin(sim_df.rel_id, ranked_rels)
    ]
    site_obs_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_obs_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_obs)
        & np.isin(sim_df.rel_id, ranked_rels)
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
        label=r"$IM^{obs}_{i}$",
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
    # fig.tight_layout()

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

    cur_scalar_features_df = get_scalar_features(
        results_dir, site_int, [site_obs], event
    )

    cur_comps_df = comps_df.loc[
        (comps_df.event_id == event)
        & (comps_df.site_int == site_int)
        & (comps_df.site_obs == site_obs)
    ].copy()
    cur_comps_df["rel_1_win"] = cur_comps_df.pred >= 0.5
    comps_won = (
        cur_comps_df[["rel_1", "rel_1_win"]]
        .groupby("rel_1")
        .sum()
        .sort_values("rel_1_win", ascending=False)
        .rename(columns={"rel_1_win": "Comparisons Won"})
    )

    st.dataframe(cur_scalar_features_df)
    st.dataframe(comps_won.T)

    # Plot the distribution for a specific IM
    st.markdown("### Realisation Distribution")
    st.markdown(f"Number of realisations: {ranked_rels.size}")
    cur_emp_gm_params = (
        None
        if emp_gm_params is None
        else emp_gm_params.loc[
            (emp_gm_params.event == event) & (emp_gm_params.site == site_int)
        ].squeeze()
    )
    fig = create_dist_plot(
        site_int_sims,
        site_int_obs,
        cur_ranking_df,
        f"{type}_sample",
        emp_gm_params=cur_emp_gm_params,
    )

    st.pyplot(fig, use_container_width=False)


def create_dist_plot(
    site_int_sims: pd.DataFrame,
    site_int_obs: pd.DataFrame,
    ranking_df: pd.DataFrame,
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
        weights=ranking_df.loc[site_int_sims.rel_id, "model_rel_prob"].values,
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


def get_scalar_features(
    results_dir: Path, site_int: str, site_obs: Sequence[str], event: str
):
    metadata = get_metadata(results_dir)
    site_df = get_site_df(results_dir)
    dist_df = get_dist_matrix(results_dir)
    record_df = get_record_df(results_dir)
    event_angular_distances = get_event_angular_distances(results_dir)

    site_int_features = (
        site_df.loc[site_int, metadata["data"]["features"]["site_features"]]
        .copy()
        .to_frame()
        .T
    )
    site_int_features["r_rup"] = (
        record_df.loc[(record_df.event_id == event)]
        .set_index("site_id")
        .loc[site_int, "r_rup"]
    )
    dfs = [site_int_features]

    for cur_site_obs in site_obs:
        cur_scalar_features_df = site_df.loc[
            [cur_site_obs], metadata["data"]["features"]["site_features"]
        ].copy()
        cur_scalar_features_df["r_rup"] = (
            record_df.loc[(record_df.event_id == event)]
            .set_index("site_id")
            .loc[cur_site_obs, "r_rup"]
        )
        cur_scalar_features_df["site_to_site_distance"] = dist_df.loc[
            site_int, cur_site_obs
        ]
        cur_scalar_features_df["angular_distance"] = event_angular_distances[event].loc[
            site_int, cur_site_obs
        ]

        dfs.append(cur_scalar_features_df)

    return pd.concat(dfs, axis=0)


def _run_res_tab(ranking_df: pd.DataFrame):
    # ### Model based residual
    # st.markdown(
    #     """
    #     ### Model Residuals
    #     Residuals between the best realisation (based on model predictions) and
    #     the observations at the site of interest.
    #     """
    # )
    #
    model_mean = ranking_df.loc[ranking_df["rank"].values == 1][sr.constants.PSA_KEYS].mean(axis=0)
    model_std = ranking_df.loc[ranking_df["rank"].values == 1][sr.constants.PSA_KEYS].std(axis=0)
    #
    # fig, ax = plt.subplots(figsize=(12, 6))
    #
    # # for _, cur_row in ranking_df.iloc[::100, :].iterrows():
    # #     ax.semilogx(
    # #         sr.constants.PERIODS,
    # #         cur_row.loc[sr.constants.PSA_KEYS].values,
    # #         c="gray",
    # #         alpha=0.5,
    # #         linewidth=1.0,
    # #     )
    #
    # ax.semilogx(
    #     sr.constants.PERIODS,
    #     model_mean,
    #     c="b",
    #     label="Mean",
    #     marker="o",
    #     linestyle="-",
    #     markersize=2.5,
    # )
    # ax.semilogx(
    #     sr.constants.PERIODS, model_mean + model_std, c="b", linestyle="--", label="Std"
    # )
    # ax.semilogx(sr.constants.PERIODS, model_mean - model_std, c="b", linestyle="--")
    #
    # ax.set_xlabel("Period")
    # ax.set_ylabel("pSA")
    # ax.set_xlim(0.01, 10.0)
    # ax.set_ylim(-2.0, 2.0)
    # ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    # ax.legend()
    # fig.tight_layout()
    #
    # st.pyplot(fig, use_container_width=False)
    #
    # ### Best simulation residual
    # st.markdown(
    #     """
    #     ### Simulation Residuals
    #     Residuals between the best realisation and
    #     the observations at the site of interest.
    #     """
    # )
    #
    best_rels = ranking_df.sort_values("misfit_score", ascending=True).groupby(
        ["event_id", "site_int"]).first()
    best_rels["event_id"] = best_rels.index.get_level_values(0)
    best_rels["site_int"] = best_rels.index.get_level_values(1)
    best_rels.index = mlt.array_utils.numpy_str_join("_", best_rels.index.get_level_values(0).values.astype(str), best_rels.index.get_level_values(1).values.astype(str), best_rels.rel_id.values.astype(str))

    sim_mean = best_rels.loc[:, sr.constants.PSA_KEYS].mean(axis=0)
    sim_std = best_rels.loc[:, sr.constants.PSA_KEYS].std(axis=0)
    #
    # fig, ax = plt.subplots(figsize=(12, 6))
    #
    # # for _, cur_row in best_rels.iloc[::5, :].iterrows():
    # #     ax.semilogx(
    # #         sr.constants.PERIODS,
    # #         cur_row.loc[sr.constants.PSA_KEYS].values,
    # #         c="gray",
    # #         alpha=0.5,
    # #         linewidth=1.0,
    # #     )
    #
    # ax.semilogx(
    #     sr.constants.PERIODS,
    #     sim_mean,
    #     c="b",
    #     label="Mean",
    #     marker="o",
    #     linestyle="-",
    #     markersize=2.5,
    # )
    # ax.semilogx(
    #     sr.constants.PERIODS, sim_mean + sim_std, c="b", linestyle="--", label="Std"
    # )
    # ax.semilogx(sr.constants.PERIODS, sim_mean - sim_std, c="b", linestyle="--")
    #
    # ax.set_xlabel("Period")
    # ax.set_ylabel("pSA")
    # ax.set_xlim(0.01, 10.0)
    # ax.set_ylim(-2.0, 2.0)
    # ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    # ax.legend()
    # fig.tight_layout()
    #
    # st.pyplot(fig, use_container_width=False)

    ### Comparison
    st.markdown("""
        ### Comparison
        Comparison between the best realisation (based on model predictions) and
        the best realisation (based on misfit).
    """)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogx(
        sr.constants.PERIODS,
        sim_mean,
        c="b",
        label="Best Realisation - Mean",
        marker="o",
        linestyle="-",
        markersize=2.5,
    )
    ax.semilogx(
        sr.constants.PERIODS,
        sim_mean + sim_std,
        c="b",
        linestyle="--",
        label="Best Realisation - Std",
    )
    ax.semilogx(sr.constants.PERIODS, sim_mean - sim_std, c="b", linestyle="--")

    ax.semilogx(
        sr.constants.PERIODS,
        model_mean,
        c="r",
        label="Best Model Realisation - Mean",
        marker="o",
        linestyle="-",
        markersize=2.5,
    )
    ax.semilogx(
        sr.constants.PERIODS,
        model_mean + model_std,
        c="r",
        linestyle="--",
        label="Best Model Realisation - Std",
    )
    ax.semilogx(sr.constants.PERIODS, model_mean - model_std, c="r", linestyle="--")

    ax.set_xlabel("Period")
    ax.set_ylabel("pSA")
    ax.set_xlim(0.01, 10.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)


def run_agg_single(cur_results_dir):

    train_rankings, val_rankings, train_comps, val_comps = load_sample_data(cur_results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _run_res_tab(train_rankings)
    with val_tab:
        _run_res_tab(val_rankings)


def _scenario_viewer(
    results_dir: Path,
    results_df: pd.DataFrame,
    scenario_results: pd.DataFrame,
    tab_type: str,
    emp_gm_params: pd.DataFrame = None,
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

        st.markdown(f"Magnitude: {event_df.loc[event].mag}")

    with col2:
        fig = _create_event_map(event, site_df, results_df, event_df, site_int)
        st.plotly_chart(fig, use_container_width=True)

    # Get the relevant data
    cur_scenario_df = (
        scenario_results.loc[
            (scenario_results.event_id == event)
            & (scenario_results.site_int == site_int)
        ]
        .sort_values("comps_won", ascending=False)
        .set_index("rel_id")
    )
    ranked_rels = cur_scenario_df.index.values.astype(str)

    site_int_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_int_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_int)
        & np.isin(sim_df.rel_id, ranked_rels)
    ]

    # Site of interest figure
    fig, ax = _create_pSA_plot(
        site_int_obs.loc[sr.constants.PSA_KEYS].values,
        site_int_sims,
        cur_scenario_df.index.values[0],
        # best_rel,
        high_rel=high_rel,
        title="Site of Interest",
    )
    st.pyplot(fig, use_container_width=False)

    obs_sites = (
        results_df.loc[
            (results_df.event_id == event) & (results_df.site_int == site_int)
        ]
        .site_obs.unique()
        .astype(str)
    )
    cur_scalar_features_df = get_scalar_features(
        results_dir, site_int, obs_sites, event
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        pass
        # st.dataframe(weighted_rels)
    with col2:
        st.dataframe(cur_scalar_features_df)
    with col3:
        st.dataframe(cur_scenario_df["comps_won"].to_frame())

    # Observation site plots
    site_obs = st.selectbox(
        "Observation Site",
        obs_sites,
        key=f"scenario_{tab_type}_site_obs",
    )

    site_obs_obs = (
        obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
        .iloc[0][sr.constants.PSA_KEYS]
        .astype(float)
    )
    site_obs_sims = sim_df.loc[
        (sim_df.event_id == event)
        & (sim_df.site_id == site_obs)
        & np.isin(sim_df.rel_id, ranked_rels)
    ]
    fig, ax = _create_pSA_plot(
        site_int_obs.loc[sr.constants.PSA_KEYS].values,
        site_obs_sims,
        best_model_rel=cur_scenario_df.index.values[0],
        high_rel=high_rel,
        title="Observation Site",
        site_obs_obs_values=site_obs_obs.loc[sr.constants.PSA_KEYS].values,
    )
    st.pyplot(fig, use_container_width=False)

    # Create IM distribution plot
    st.markdown("### Realisation Distribution")
    st.markdown(f"Number of realisations: {ranked_rels.size}")
    cur_emp_gm_params = (
        None
        if emp_gm_params is None
        else emp_gm_params.loc[
            (emp_gm_params.event == event) & (emp_gm_params.site == site_int)
        ].squeeze()
    )
    fig = create_dist_plot(
        site_int_sims,
        site_int_obs,
        cur_scenario_df,
        f"{tab_type}_scenario",
        emp_gm_params=cur_emp_gm_params,
    )

    st.pyplot(fig, use_container_width=False)


def _create_pSA_plot(
    site_int_obs_values: np.ndarray,
    sim_rels: pd.DataFrame,
    best_model_rel: str = None,
    best_rel: str = None,
    high_rel: str = None,
    title: str = None,
    site_obs_obs_values: np.ndarray = None,
):
    label = None
    fig, ax = plt.subplots(figsize=(12, 6))
    for ix, (cur_id, cur_row) in enumerate(sim_rels.iterrows()):
        cur_label, c = None, "gray"
        if cur_row.rel_id == best_model_rel:
            cur_label, c = f"Best Model Rel: {best_model_rel}", "g"
        if high_rel == cur_row.rel_id:
            cur_label, c = f"Highlighted Rel: {high_rel}", "orange"
        if best_rel == cur_row.rel_id:
            cur_label, c = f"Best Rel: {best_rel}", "m"
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
    # fig.tight_layout()

    return fig, ax




def run_ind_scenario(results_dir, emp_gm_params_ffp: Path = None):
    train_results, val_results = load_results(results_dir)
    train_scenario_results, val_scenario_results = load_scenario_results(results_dir)

    emp_gm_params = get_emp_gm_params(emp_gm_params_ffp)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _scenario_viewer(
            results_dir,
            train_results,
            train_scenario_results,
            "train",
            emp_gm_params=emp_gm_params,
        )

    with val_tab:
        _scenario_viewer(
            results_dir,
            val_results,
            val_scenario_results,
            "val",
            emp_gm_params=emp_gm_params,
        )


def run_agg_scenario(results_dir):
    train_results, val_results = load_scenario_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _run_res_tab(train_results)
    with val_tab:
        _run_res_tab(val_results)


def main(
    results_dir: Path,
    emp_gm_params_ffp: Path = typer.Option(
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
        run_ind_samples(cur_results_dir, emp_gm_params_ffp=emp_gm_params_ffp)

    with ind_scenario_tab:
        # pass
        run_ind_scenario(cur_results_dir, emp_gm_params_ffp=emp_gm_params_ffp)

    with agg_single_tab:
        # pass
        run_agg_single(cur_results_dir)

    with agg_scenario_tab:
        # pass
        run_agg_scenario(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
