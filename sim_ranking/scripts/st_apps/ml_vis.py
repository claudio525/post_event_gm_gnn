import os
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import typer

import sim_ranking as sr
import ml_tools as mlt


@st.cache_data
def _get_metadata(results_dir: Path):
    return sr.data.get_meta(results_dir)


@st.cache_data
def _load_loss_history(results_dir: Path):
    return {
        "loss": np.load(str(results_dir / "loss_hist_train.npy")),
        "val_loss": np.load(str(results_dir / "loss_hist_val.npy")),
    }


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
def _load_results(results_dir: Path):
    train_results_df = pd.read_csv(
        results_dir / "train_results.csv",
        dtype=dict(event=str),
        index_col=0,
        na_filter=False,
    )
    val_results_df = pd.read_csv(
        results_dir / "val_results.csv",
        dtype=dict(event=str),
        index_col=0,
        na_filter=False,
    )

    return train_results_df, val_results_df


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


def get_filtering_mask(results_df: pd.DataFrame, label: str):
    # Filtering
    sel_events = st.multiselect(
        "Events", results_df.event.unique().tolist(), key=f"{label}_events"
    )
    sel_sites = st.multiselect(
        "Sites of Interest", results_df.site_int.unique().tolist(), key=f"{label}_sites"
    )

    col1, col2 = st.columns(2)
    with col1:
        min_distance = st.slider(
            "Min Distance",
            0,
            100,
            value=0,
            step=1,
            format="%d km",
            key=f"{label}_min_distance",
        )
    with col2:
        max_distance = st.slider(
            "Max Distance",
            0,
            100,
            value=100,
            step=1,
            format="%d km",
            key=f"{label}_max_distance",
        )

    m = np.ones(results_df.shape[0], dtype=bool)
    if len(sel_events) > 0:
        m = m & np.isin(results_df.event.values, sel_events)
    if len(sel_sites) > 0:
        m = m & np.isin(results_df.site_int.values, sel_sites)
    if min_distance > 0:
        m = m & (results_df.distance.values >= min_distance)
    if max_distance < 100:
        m = m & (results_df.distance.values <= max_distance)

    return m


def run_general_tab(results_dir: Path):
    # Load the metadata
    meta = _get_metadata(results_dir)

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown(
            f"""
            ### Data
            DB File: {meta["data"]['db']}\n

            Site Features: {meta['site_features']}\n
            
            Number of Realisations: {meta['n_rels_used']}\n
            
            Number of Training Events: {len(meta['train_events'])}\n
            Validation Events: {meta['val_events']}\n
            
            Number of Training Samples: {meta['n_train_samples']}\n
            Validation Samples: {meta['n_val_samples']}\n
            
            Comment: {meta['comment']}\n
            """
        )
    with col_2:
        st.markdown(
            f"""
            ### Model
            Number of Channels: {meta["model"]['n_channels']}\n
            Kernel Sizes: {meta["model"]['kernel_sizes']}\n
            Fully Connected Units: {meta["model"]['fc_units']}\n

            ### Training
            Number of Epochs: {meta["training"]['n_epochs']}\n
            Batch Size: {meta["training"]['batch_size']}\n
            """
        )

    # Loss plot
    fig, ax = plt.subplots(figsize=(12, 6))
    mlt.plotting.plot_loss(_load_loss_history(results_dir), ax=ax)
    st.pyplot(fig, use_container_width=False)

    # Model visualization
    if (model_vis_ffp := results_dir / "model_vis.png").exists():
        st.image(str(model_vis_ffp))


def run_one_to_one_tab(results_dir: Path):
    # True vs Predicted plot
    train_results_df, val_results_df = _load_results(results_dir)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        m = get_filtering_mask(train_results_df, "train")

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_pred_vs_true(train_results_df.loc[m], ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with val_tab:
        m = get_filtering_mask(val_results_df, "val")

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_pred_vs_true(val_results_df.loc[m], ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)


def run_individual_samples_tab(results_dir: Path):
    metadata = _get_metadata(results_dir)
    train_results, val_results = _load_results(results_dir)

    obs_df = get_obs_df(results_dir)
    sim_df = get_sim_df(results_dir)

    site_df = get_site_df(results_dir)

    def _sample_viewer(results_df: pd.DataFrame, events: np.ndarray, type: str):
        event = st.selectbox("Event", events, key=f"{type}_event")

        rel = st.selectbox(
            "Realisation",
            results_df[(results_df.event == event)].rel.unique().astype("str"),
            key=f"{type}_rel",
        )

        site_int = st.selectbox(
            "Site of Interest",
            results_df.loc[(results_df.event == event) & (results_df.rel == rel)]
            .site_int.unique()
            .astype("str"),
            key=f"{type}_site_int",
        )
        site_obs = st.selectbox(
            "Observation Site",
            results_df.loc[
                (results_df.event == event)
                & (results_df.site_int == site_int)
                & (results_df.rel == rel)
            ].site_obs,
            key=f"{type}_site_obs",
        )

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
            (results_df.event == event)
            & (results_df.site_int == site_int)
            & (results_df.site_obs == site_obs)
            & (results_df.rel == rel)
        )
        pred = (
            results_df.loc[m]
            .iloc[0][np.char.add(sr.constants.PSA_KEYS, "_pred")]
            .values.astype(float)
        )

        st.markdown(f"##### Loss: {results_df.loc[m, ['loss']].iloc[0].values[0]}")

        # Residuals
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_obs.values) - np.log(site_int_sim.values),
            label="Site Obs Obs - Site Int Sim (Input)",
            c="magenta",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_obs.values) - np.log(site_obs_sim.values),
            label="Site Obs Obs - Site Obs Sim (Input)",
            c="blue",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_obs_sim.values) - np.log(site_int_sim.values),
            label="Site Obs Sim - Site Int Sim (Input)",
            c="cyan",
            linestyle="--",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            pred,
            c="k",
            linestyle="-",
            label="Site Int Obs - Site Int Sim (Predicted)",
            linewidth=2.0,
        )
        ax.semilogx(
            sr.constants.PERIODS,
            np.log(site_int_obs.values) - np.log(site_int_sim.values),
            c="r",
            linestyle="-",
            label="Site Int Obs - Site Int Sim (True)",
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
            label="Site of Interest (Observed)",
            linestyle="-",
            c="r",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            site_int_sim,
            label="Site of Interest (Simulated)",
            c="r",
            linestyle="--",
        )

        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_obs,
            label="Observation Site (Observed)",
            linestyle="-",
            c="b",
        )
        ax.semilogx(
            sr.constants.PERIODS,
            site_obs_sim,
            label="Observation Site (Simulated)",
            c="b",
            linestyle="--",
        )

        ax.semilogx(
            sr.constants.PERIODS,
            pred_pSA,
            label="Site of Interest (Predicted)",
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
        cur_site_df = site_df.loc[[site_int, site_obs], metadata["site_features"]]
        cur_site_df["distance"] = results_df["distance"].loc[m].iloc[0]

        st.dataframe(cur_site_df)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        _sample_viewer(train_results, metadata["train_events"], "train")
    with val_tab:
        _sample_viewer(val_results, metadata["val_events"], "val")


def run_rs_agg_tab(results_dir: Path):
    def create_loss_dist_plot(results_df: pd.DataFrame, mag_coloring: bool = False):
        fig, ax = plt.subplots(figsize=(12, 6))

        if mag_coloring:
            event_df = get_event_df(results_dir)
            t = ax.scatter(
                results_df.distance,
                results_df.loss,
                s=2.0,
                c=event_df.loc[results_df.event, "mag"].values,
                cmap="viridis_r",
                vmin=3.5,
                vmax=7
                # alpha=0.5,
            )
            plt.colorbar(t, pad=0, label="Magnitude")
        else:
            ax.scatter(results_df.distance, results_df.loss, s=2.0, c="k", alpha=0.5)

        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Loss")
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_ylim(0, 2.0)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)

    train_results_df, val_results_df = _load_results(results_dir)

    mag_coloring = st.checkbox("Color by Magnitude", value=False)

    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        create_loss_dist_plot(train_results_df, mag_coloring=mag_coloring)
    with val_tab:
        create_loss_dist_plot(val_results_df, mag_coloring=mag_coloring)


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

    general_tab, one_to_one_tab, individual_samples_tab, rs_agg_tab = st.tabs(
        ["General", "One-To-One", "Sample Explorer", "RS-Agg"]
    )

    with general_tab:
        run_general_tab(cur_results_dir)

    with one_to_one_tab:
        pass
        # run_one_to_one_tab(cur_results_dir)

    with individual_samples_tab:
        run_individual_samples_tab(cur_results_dir)

    with rs_agg_tab:
        run_rs_agg_tab(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
