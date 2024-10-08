import time
from typing import NamedTuple, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import streamlit as st
import plotly.graph_objects as go

import ml_tools as mlt
import sim_ranking as sr

import st_utils


class SharedData(NamedTuple):

    emp_cim_data: dict[str, sr.conditional.ConditionalMVNDistribution] = None








def run_ind_scenario(shared_data: SharedData):
    train_tab, val_tab = st.tabs(["Training", "Validation"])

    with train_tab:
        scenario_viewer(shared_data.gnn_train_results, shared_data, "train")

    with val_tab:
        scenario_viewer(shared_data.gnn_val_results, shared_data, "val")


def run_general(shared_data: SharedData):
    metrics = shared_data.gnn_metrics

    metric_keys = sorted(list(metrics.keys()))
    avail_metrics = [key.rsplit("_", maxsplit=1)[0] for key in metric_keys[::2]]
    # avail_metrics = ["loss_hist", "misfit_loss_hist"]
    sel_metric_keys = st.multiselect(
        "Metrics", avail_metrics, default=[avail_metrics[0]]
    )

    # Loss plot
    fig, ax = plt.subplots(figsize=(12, 6))
    mlt.plotting.plot_metrics(
        metrics,
        sel_metric_keys,
        ax=ax,
        best_epoch=shared_data.gnn_metadata["best_model_epoch"],
        y_lim=(0.0, 0.02),
    )
    # mlt.plotting.plot_metrics(load_training_metrics(results_dir), ax=ax)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    im_loss_keys = [
        f"{cur_im}_loss"
        for cur_im in shared_data.gnn_run_config.ims
        if cur_im.startswith("pSA")
    ]
    with st.expander("IM Loss"):
        train_loss_mean = shared_data.gnn_train_results[im_loss_keys].mean()
        val_loss_mean = shared_data.gnn_val_results[im_loss_keys].mean()

        train_loss_std = shared_data.gnn_train_results[im_loss_keys].std()
        val_loss_std = shared_data.gnn_val_results[im_loss_keys].std()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.semilogx(sr.constants.PERIODS, train_loss_mean, c="b", label="Training Loss")
        ax.semilogx(
            sr.constants.PERIODS,
            np.stack(
                (
                    train_loss_mean + train_loss_std,
                    train_loss_mean - train_loss_std,
                ),
                axis=1,
            ),
            linestyle="--",
            linewidth=1.0,
            c="b",
        )

        ax.semilogx(sr.constants.PERIODS, val_loss_mean, c="r", label="Validation Loss")
        ax.semilogx(
            sr.constants.PERIODS,
            np.stack(
                (
                    val_loss_mean + val_loss_std,
                    val_loss_mean - val_loss_std,
                ),
                axis=1,
            ),
            linestyle="--",
            linewidth=1.0,
            c="r",
        )

        ax.set_xlabel(f"Period (s)")
        ax.set_ylabel(f"Loss")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_xlim([0.01, 10])

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        print(f"wtf")

    col1, col2 = st.columns(2)
    with col1:
        st.title("Run Config")
        st.json(shared_data.gnn_run_config.to_dict())

    with col2:
        st.title("Metadata")
        st.json(shared_data.gnn_metadata)

        st.markdown(
            f"Observed Data Source: {shared_data.obs_data.data_source} - {shared_data.obs_data.nzgmdb_version}"
        )


def main(
    nzgmdb_ffp: Path = typer.Argument(..., help="Path to the NZGMDB flat file"),
    gnn_results_dir: Path = typer.Argument(
        ..., help="Path to the directory containing the GNN results"
    ),
    emp_cim_results_dir: Path = typer.Option(
        None, help="Path to the directory containing the CIM results"
    ),
):
    st.set_page_config(layout="wide")

    # Get observed data
    obs_data = get_observed_data(nzgmdb_ffp)
    # Get distance matrix
    dist_matrix = get_dist_matrix(obs_data)

    # Select GNN results
    gnn_result_id = st.selectbox(
        "Results Directory",
        sorted(
            [
                cur_ffp.stem
                for cur_ffp in gnn_results_dir.iterdir()
                if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
            ]
        ),
    )
    gnn_result_ffp = gnn_results_dir / gnn_result_id
    gnn_metrics, gnn_train_results, gnn_val_results = get_gnn_result(gnn_result_ffp)

    gnn_run_config = sr.ml.gnn_gm.RunConfig.from_yaml(
        gnn_result_ffp / "run_config.yaml"
    )

    gnn_metadata = mlt.utils.load_yaml(gnn_result_ffp / "metadata.yaml")

    start_time = time.time()
    emp_cim_data = (
        load_emp_cim_data(emp_cim_results_dir)
        if emp_cim_results_dir is not None
        else None
    )
    print(f"Took {time.time() - start_time} to load cIM data")

    ## Add check here to ensure that validation sites are matching!!

    shared_data = SharedData(
        obs_data=obs_data,
        obs_sites=obs_data.sites,
        dist_matrix=dist_matrix,
        gnn_result_id=gnn_result_id,
        gnn_metrics=gnn_metrics,
        gnn_train_results=gnn_train_results,
        gnn_val_results=gnn_val_results,
        gnn_run_config=gnn_run_config,
        gnn_metadata=gnn_metadata,
        emp_cim_data=emp_cim_data,
    )

    general_tab, ind_sc_tab = st.tabs(["General", "Individual Scenarios"])

    with general_tab:
        run_general(shared_data)

    with ind_sc_tab:
        run_ind_scenario(shared_data)


if __name__ == "__main__":
    typer.run(main)
