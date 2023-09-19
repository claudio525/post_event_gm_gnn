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
def _load_results(results_dir: Path):
    train_results_df = pd.read_csv(
        results_dir / "train_results.csv", dtype=dict(event=str)
    )
    val_results_df = pd.read_csv(results_dir / "val_results.csv", dtype=dict(event=str))

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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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
            Observed Data: \{meta["data"]['obs_ffp']}\n
            Simulation IMDB: \{meta["data"]['sim_imdb_ffp']}\n
            Sites Directory: \{meta["data"]['sites_dir']}\n

            Site Features: {meta['site_features']}\n
            Train Events: {meta['train_events']}\n
            Validation Events: {meta['val_events']}\n
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


def run_results_tab(results_dir: Path):
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


def main(results_dir: Path):
    st.set_page_config(layout="wide")

    result_id = st.selectbox(
        "Results Directory",
        [
            cur_ffp.stem
            for cur_ffp in results_dir.iterdir()
            if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
        ],
    )
    cur_results_dir = results_dir / result_id

    general_tab, results_tab = st.tabs(["General", "Results"])

    with general_tab:
        run_general_tab(cur_results_dir)

    with results_tab:
        run_results_tab(cur_results_dir)


if __name__ == "__main__":
    typer.run(main)
