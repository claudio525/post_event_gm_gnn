import os
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
def get_obs_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_obs_df()


@st.cache_data
def get_event_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_event_df()


@st.cache_data
def get_sim_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_sim_df()


@st.cache_data
def get_site_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_site_df()


@st.cache_data
def get_record_df(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_record_df()


@st.cache_data
def get_event_sites(results_dir: Path):
    metadata = _get_metadata(results_dir)
    db_ffp = os.path.expandvars(metadata["data"]["db"])

    return sr.db.DB(db_ffp).get_event_sites()


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

    # Add other stuff
    dist_matrix = get_dist_matrix(results_dir)
    train_results_df["s2s_distance"] = [
        dist_matrix.loc[cur_row.site_int, cur_row.site_obs]
        for cur_ix, cur_row in train_results_df.iterrows()
    ]
    val_results_df["s2s_distance"] = [
        dist_matrix.loc[cur_row.site_int, cur_row.site_obs]
        for cur_ix, cur_row in val_results_df.iterrows()
    ]

    vs30_dist = get_vs30_dist(results_dir)
    train_results_df["vs30_distance"] = [
        vs30_dist.loc[cur_row.site_int, cur_row.site_obs]
        for cur_ix, cur_row in train_results_df.iterrows()
    ]
    val_results_df["vs30_distance"] = [
        vs30_dist.loc[cur_row.site_int, cur_row.site_obs]
        for cur_ix, cur_row in val_results_df.iterrows()
    ]

    angular_distances = get_event_angular_distances(results_dir)
    train_results_df["angular_distance"] = np.rad2deg(
        [
            angular_distances[cur_row.event_id].loc[cur_row.site_int, cur_row.site_obs]
            for cur_ix, cur_row in train_results_df.iterrows()
        ]
    )
    val_results_df["angular_distance"] = np.rad2deg(
        [
            angular_distances[cur_row.event_id].loc[cur_row.site_int, cur_row.site_obs]
            for cur_ix, cur_row in val_results_df.iterrows()
        ]
    )

    event_df = get_event_df(results_dir)
    train_results_df["mag"] = event_df.loc[train_results_df.event_id, "mag"].values
    val_results_df["mag"] = event_df.loc[val_results_df.event_id, "mag"].values

    return train_results_df, val_results_df


@st.cache_data
def get_dist_matrix(results_dir: Path):
    site_df = get_site_df(results_dir)

    return sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )


@st.cache_data
def get_vs30_dist(results_dir: Path):
    site_df = get_site_df(results_dir)

    return sr.ml.features.compute_vs30_dist(site_df)


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


def plot_pred_vs_true(result_df: pd.DataFrame, ax: plt.Axes):
    t = ax.scatter(
        result_df["sim_score"],
        result_df["predicted_sim_score"],
        s=1 / result_df["distance"] * 100,
        c=result_df["distance"],
        cmap="hot",
        vmin=0,
        vmax=100,
        alpha=0.8,
    )
    plt.colorbar(t, pad=0, label="Site-to-Site Distance")
    ax.plot([0, 1], [0, 1], c="k", linestyle="--")

    ax.set_xlabel("True Similarity Score")
    ax.set_ylabel("Predicted Similarity Score")
    ax.set_xlim(-0.025, 1)
    ax.set_ylim(-0.025, 1)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")


def run_general_tab(results_dir: Path):
    # Load the metadata
    meta = _get_metadata(results_dir)

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown(
            f"""
            ### Data
            Max Distance: {meta["data"].get('max_distance')}\n
            
            Number of Realisations: {meta["data"]['n_rels_used']}\n
            Number of Training Events: {len(meta["data"]['train_events'])}\n
            Number of Validation Events: {len(meta["data"]['val_events'])}\n
            Number of Training Samples: {meta["data"]['n_train_samples']}\n
            Number of Validation Samples: {meta["data"]['n_val_samples']}\n
            
            #### Features
            Site Features: {', '.join(meta["data"]['site_features'])}\n
            Site-to-site Features: {', '.join(meta["data"]['site_to_site_features'])}\n
            Event-site Features: {', '.join(meta["data"]['event_site_features'])}\n
            Event-site-to-site Features: {', '.join(meta["data"]['event_site_to_site_features'])}\n
            
            Comment: {meta['comment']}\n
            """
        )
    with col_2:
        st.markdown(
            f"""
            ### Model
            Number of Channels: {meta["hyperparams"]['n_channels']}\n
            Kernel Sizes: {meta["hyperparams"]['kernel_sizes']}\n
            Fully Connected Units: {meta["hyperparams"]['fc_units']}\n
            
            ### Training
            Number of Epochs: {meta["hyperparams"]['n_epochs']}\n
            Batch Size: {meta["hyperparams"]['batch_size']}\n
            L2 Regularisation: {meta["hyperparams"].get("l2_reg")}\n
            Weight Penalty Factor: {meta["hyperparams"].get("weight_penalty_factor")}
            Learning rate: {meta["hyperparams"]['lr']}\n
            """
        )

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
    with col_2:
        if (model_vis_ffp := results_dir / "weight_model_vis.png").exists():
            st.image(str(model_vis_ffp))


def run_individual_samples_tab(results_dir: Path):
    metadata = _get_metadata(results_dir)
    train_results, val_results = _load_results(results_dir)

    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

    site_df = get_site_df(results_dir)
    event_df = get_event_df(results_dir)

    dist_df = get_dist_matrix(results_dir)
    record_df = get_record_df(results_dir)

    event_angular_distances = get_event_angular_distances(results_dir)

    def _sample_viewer(results_df: pd.DataFrame, events: np.ndarray, type: str):
        col1, col2 = st.columns([1, 6])

        with col1:
            event = st.selectbox("Event", events, key=f"{type}_event")

            rel = st.selectbox(
                "Realisation",
                results_df[(results_df.event_id == event)]
                .rel_id.unique()
                .astype("str"),
                key=f"{type}_rel",
            )

            site_int = st.selectbox(
                "Site of Interest",
                results_df.loc[
                    (results_df.event_id == event) & (results_df.rel_id == rel)
                ]
                .site_int.unique()
                .astype("str"),
                key=f"{type}_site_int",
            )

            site_obs = st.selectbox(
                "Observation Site",
                results_df.loc[
                    (results_df.event_id == event)
                    & (results_df.site_int == site_int)
                    & (results_df.rel_id == rel)
                ].site_obs,
                key=f"{type}_site_obs",
            )

        with col2:
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
                    go.Scattermapbox(
                        lat=[site_df.loc[site_obs, "lat"]],
                        lon=[site_df.loc[site_obs, "lon"]],
                        mode="markers",
                        marker=dict(size=10, color="maroon"),
                        hovertext=site_obs,
                        hoverinfo="text",
                        name="Observation Site",
                    ),
                ]
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
            st.plotly_chart(fig, use_container_width=True)

        site_int_obs = (
            obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_int)]
            .iloc[0][sr.constants.PSA_KEYS]
            .astype(float)
        )
        site_int_sim = (
            sim_df.loc[
                (sim_df.event_id == event)
                & (sim_df.site_id == site_int)
                & (sim_df.rel_id == rel)
            ]
            .iloc[0][sr.constants.PSA_KEYS]
            .astype(float)
        )
        site_obs_obs = (
            obs_df.loc[(obs_df.event_id == event) & (obs_df.site_id == site_obs)]
            .iloc[0][sr.constants.PSA_KEYS]
            .astype(float)
        )
        site_obs_sim = (
            sim_df.loc[
                (sim_df.event_id == event)
                & (sim_df.site_id == site_obs)
                & (sim_df.rel_id == rel)
            ]
            .iloc[0][sr.constants.PSA_KEYS]
            .astype(float)
        )

        m = (
            (results_df.event_id == event)
            & (results_df.site_int == site_int)
            & (results_df.site_obs == site_obs)
            & (results_df.rel_id == rel)
        )
        pred = (
            results_df.loc[m]
            .iloc[0][np.char.add(sr.constants.PSA_KEYS, "_pred")]
            .values.astype(float)
        )

        st.markdown(
            f"##### Misfit Loss: {results_df.loc[m, ['misfit']].iloc[0].values[0]}"
        )
        st.markdown(f"##### Loss: {results_df.loc[m, ['loss']].iloc[0].values[0]}")

        # Residuals
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_obs.values) - np.log(site_int_sim.values),
            label=r"$lnIM^{obs}_i - lnIM^{sim}_s$ (Input)",
            c="magenta",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_obs.values) - np.log(site_obs_sim.values),
            label=r"$lnIM^{obs}_i - lnIM^{sim}_i$ (Input)",
            c="blue",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_sim.values) - np.log(site_int_sim.values),
            label=r"$lnIM^{sim}_i - lnIM^{sim}_s$ (Input)",
            c="cyan",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            pred,
            c="k",
            linestyle="-",
            label=r"$lnIM^{obs}_s - lnIM^{sim}_s$ (Predicted)",
            linewidth=2.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_int_obs.values) - np.log(site_int_sim.values),
            c="r",
            linestyle="-",
            label=r"$lnIM^{obs}_s - lnIM^{sim}_s$ (True)",
            linewidth=2.0,
        )

        ax.set_xlabel("Period")
        ax.set_ylabel("Residual")
        ax.set_xlim(0.01, 10.0)
        ax.set_ylim(-2.0, 2.0)
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)

        ## Response Spectrum
        pred_pSA = site_int_sim * np.exp(pred)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogx(
            sr.constants.PERIODS,
            site_int_obs,
            label=r"$lnIM^{obs}_s$",
            linestyle="-",
            c="r",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            site_int_sim,
            label=r"$lnIM^{sim}_s$",
            c="r",
            linestyle="--",
        )

        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_obs,
            label=r"$lnIM^{obs}_i$",
            linestyle="-",
            c="b",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_sim,
            label=r"$lnIM^{sim}_i$",
            c="b",
            linestyle="--",
        )

        ax.semilogx(
            sr.constants.PERIODS,
            pred_pSA,
            label=r"$lnIM^{pred}_s$",
            c="k",
            linestyle="-",
        )

        ax.set_xlabel("Period")
        ax.set_ylabel("pSA (g)")
        ax.set_xlim(0.01, 10.0)
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)

        ## Info table

        cur_scalar_features_df = site_df.loc[
            [site_int, site_obs], metadata["data"]["site_features"]
        ]
        cur_scalar_features_df["site_to_site_distance"] = dist_df.loc[
            site_int, site_obs
        ]
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

    table_cols = [
        "event_id",
        "site_int",
        "site_obs",
        "loss",
        "misfit",
        "weighted_misfit",
        "weight",
        "s2s_distance"
    ]
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("Sort By", table_cols, index=4)
    with col2:
        ascending = st.checkbox("Ascending", value=False)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        st.table(
            train_results[table_cols].sort_values(sort_by, ascending=ascending).head(10)
        )
        _sample_viewer(train_results, metadata["data"]["train_events"], "train")
    with val_tab:
        st.table(
            val_results[table_cols].sort_values(sort_by, ascending=ascending).head(10)
        )
        _sample_viewer(val_results, metadata["data"]["val_events"], "val")


def run_scatter_tab(results_dir: Path):
    def create_loss_dist_plot(
        results_df: pd.DataFrame,
        im_key: str,
        x_key: str,
        y_key: str,
        show_avg: bool,
        color_key: str = None,
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        # fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(5, 1, (2, 5))
        # ax_hist = fig.add_subplot(5, 1, 1, sharex=ax)

        ax_hist = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_hist.tick_params(axis="x", labelbottom=False)

        y_key = y_key if im_key == "Mean" else f"{im_key}_{y_key}"

        if color_key == "mag":
            t = ax.scatter(
                results_df[x_key],
                results_df[y_key],
                s=2.0,
                c=results_df.mag.values,
                cmap="viridis_r",
                vmin=3.5,
                vmax=7,
            )
            plt.colorbar(t, pad=0, label="Magnitude")
        elif color_key == "weight":
            t = ax.scatter(
                results_df[x_key],
                results_df[y_key],
                s=2.0,
                c=results_df.weight.values,
                cmap="viridis_r",
                vmin=0.0,
                vmax=1.0,
            )
            plt.colorbar(t, pad=0, label="Weight")
        elif color_key in ["vs30_distance", "s2s_distance"]:
            t = ax.scatter(
                results_df[x_key],
                results_df[y_key],
                s=2.0,
                c=results_df[color_key].values,
                cmap="viridis_r",
                vmin=0.0,
                vmax=results_df[color_key].values.max()
                if color_key == "s2s_distance"
                else 800,
            )
            plt.colorbar(t, pad=0, label=color_key)
        else:
            ax.scatter(results_df[x_key], results_df[y_key], s=2.0, alpha=0.5)

        if show_avg:
            bins = np.linspace(-1e-9, results_df[x_key].max(), 20)
            t = np.digitize(results_df[x_key], bins, right=True)
            results_df = results_df.copy(True)
            results_df["bin"] = t

            mid_points = np.diff(bins) / 2 + bins[:-1]
            mean_values = results_df[[y_key, "bin"]].groupby("bin").mean()
            std_values = results_df[[y_key, "bin"]].groupby("bin").std()
            ax.plot(mid_points, mean_values[y_key], c="k", linestyle="-", linewidth=1.0)
            ax.plot(
                mid_points,
                mean_values[y_key] + std_values[y_key],
                c="k",
                linestyle="--",
                linewidth=1.0,
            )
            ax.plot(
                mid_points,
                mean_values[y_key] - std_values[y_key],
                c="k",
                linestyle="--",
                linewidth=1.0,
            )

        ax_hist.hist(results_df[x_key], bins=50)

        ax.set_xlabel(f"{x_key}")
        ax.set_ylabel(f"{y_key}")
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

        if y_key != "weight":
            ax.set_ylim(0, min(2.0, results_df[y_key].max()))
        else:
            ax.set_ylim(0, 1.0)

        ax.set_xlim(results_df[x_key].min(), results_df[x_key].max())

        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)

    train_results_df, val_results_df = _load_results(results_dir)

    col_1, col_2, col_3, col_4, col_5 = st.columns(5)

    with col_1:
        y_key = st.selectbox(
            "y-axis", ["loss", "misfit", "weighted_misfit", "weight"], index=0
        )

    with col_2:
        x_key = ["s2s_distance", "vs30_distance", "angular_distance"]
        x_key = st.selectbox("x-axis", x_key, index=0)

    with col_3:
        if y_key == "loss":
            im_options = ["Mean"]
        else:
            im_options = ["Mean"] + sr.constants.PSA_KEYS

        im_key = st.selectbox("IM", im_options, index=0)

    with col_4:
        color_key_options = [
            "weight",
            "mag",
            "vs30_distance",
            "s2s_distance",
            "angular_distance",
        ]
        avail_options = [
            cur_option
            for cur_option in color_key_options
            if cur_option in train_results_df.columns
        ] + ["no_color"]
        color_key = st.selectbox(
            "Color Key", avail_options, index=len(avail_options) - 1
        )
    with col_5:
        show_avg = st.checkbox("Show Average", value=True)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        create_loss_dist_plot(
            train_results_df, im_key, x_key, y_key, show_avg, color_key=color_key
        )
    with val_tab:
        create_loss_dist_plot(
            val_results_df, im_key, x_key, y_key, show_avg, color_key=color_key
        )


def run_rs_agg_tab(results_dir: Path):
    def gen_plot(results_df: pd.DataFrame, y_key: str, show_individual: bool = False, event: str = None):
        if event is not None:
            results_df = results_df[results_df.event_id == event]

        cur_result = {}
        for cur_period, cur_key in zip(sr.constants.PERIODS, sr.constants.PSA_KEYS):
            cur_result[cur_period] = [
                results_df.loc[:, f"{cur_key}_{y_key}"].mean(),
                results_df.loc[:, f"{cur_key}_{y_key}"].std(),
            ]

        cur_df = pd.DataFrame(cur_result, index=["mean", "std"]).T

        fig, ax = plt.subplots(figsize=(12, 6))

        if show_individual:
            ax.plot(
                cur_df.index.values[::2],
                results_df[
                    [f"{cur_key}_{y_key}" for cur_key in sr.constants.PSA_KEYS]
                ].T.values[::2],
                c="b",
                alpha=0.25,
            )

        ax.semilogx(cur_df.index.values, cur_df["mean"].values, c="r", linestyle="-")
        ax.fill_between(
            cur_df.index.values,
            cur_df["mean"].values - cur_df["std"].values,
            cur_df["mean"].values + cur_df["std"].values,
            color="r",
            alpha=0.1,
        )
        ax.plot(
            cur_df.index.values,
            cur_df["mean"].values - cur_df["std"].values,
            c="r",
            linestyle="--",
        )
        ax.plot(
            cur_df.index.values,
            cur_df["mean"].values + cur_df["std"].values,
            c="r",
            linestyle="--",
        )

        ax.set_xlabel("Period")
        ax.set_ylabel(f"{y_key}")

        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        if y_key in ["pred", "true"]:
            ax.set_ylim(-2, 2)
        else:
            ax.set_ylim(0, min(2.0, results_df[y_key].max()))

        ax.set_xlim(0.01, 10.0)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)

    train_results_df, val_results_df = _load_results(results_dir)

    col1, col2, col3 = st.columns(3)
    with col1:
        y_key = st.selectbox("y_key", ["misfit", "weighted_misfit", "pred", "true"])
    with col2:
        show_individual = st.checkbox("Show Individual", value=False)
    with col3:
        event = st.selectbox("Event", [None] + train_results_df.event_id.unique().tolist())

    train_tab, val_tab = st.tabs(["Training", "Validation"])
    with train_tab:
        gen_plot(train_results_df, y_key, show_individual=show_individual, event=event)
    with val_tab:
        gen_plot(val_results_df, y_key, show_individual=show_individual, event=event)


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

    general_tab, individual_samples_tab, loss_scatter, rs_agg = st.tabs(
        ["General", "Sample Explorer", "Loss/Weights", "RS-Agg"]
    )

    with general_tab:
        run_general_tab(cur_results_dir)

    with individual_samples_tab:
        run_individual_samples_tab(cur_results_dir)

    with loss_scatter:
        run_scatter_tab(cur_results_dir)

    with rs_agg:
        run_rs_agg_tab(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)


# def get_filtering_mask(results_df: pd.DataFrame, label: str):
#     # Filtering
#     sel_events = st.multiselect(
#         "Events", results_df.event.unique().tolist(), key=f"{label}_events"
#     )
#     sel_sites = st.multiselect(
#         "Sites of Interest", results_df.site_int.unique().tolist(), key=f"{label}_sites"
#     )
#
#     col1, col2 = st.columns(2)
#     with col1:
#         min_distance = st.slider(
#             "Min Distance",
#             0,
#             100,
#             value=0,
#             step=1,
#             format="%d km",
#             key=f"{label}_min_distance",
#         )
#     with col2:
#         max_distance = st.slider(
#             "Max Distance",
#             0,
#             100,
#             value=100,
#             step=1,
#             format="%d km",
#             key=f"{label}_max_distance",
#         )
#
#     m = np.ones(results_df.shape[0], dtype=bool)
#     if len(sel_events) > 0:
#         m = m & np.isin(results_df.event.values, sel_events)
#     if len(sel_sites) > 0:
#         m = m & np.isin(results_df.site_int.values, sel_sites)
#     if min_distance > 0:
#         m = m & (results_df.distance.values >= min_distance)
#     if max_distance < 100:
#         m = m & (results_df.distance.values <= max_distance)
#
#     return m
