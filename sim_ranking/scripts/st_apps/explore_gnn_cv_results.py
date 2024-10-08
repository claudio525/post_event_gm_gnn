import time
from typing import NamedTuple, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns

import ml_tools as mlt
import sim_ranking as sr

import st_utils


class CVResults(NamedTuple):
    run_config: sr.ml.gnn_gm.RunConfig
    obs_data: sr.ObservedData
    dist_matrix: pd.DataFrame
    run_results: dict[str, st_utils.GNNRunResults]
    comb_val_results: pd.DataFrame
    metrics: dict


def get_n_scenarios_df(cv_results: CVResults) -> pd.DataFrame:
    """Returns a DataFrame with the number of scenarios per CV iteration."""
    cv_val_n_scenarios = pd.DataFrame(
        {
            cur_key: [cur_run_result.val_results.shape[0], cur_key, "val"]
            for cur_key, cur_run_result in cv_results.run_results.items()
        },
        index=["n_scenarios", "cv_iter", "type"],
    ).T
    cv_train_n_scenarios = pd.DataFrame(
        {
            cur_key: [cur_run_result.train_results.shape[0], cur_key, "train"]
            for cur_key, cur_run_result in cv_results.run_results.items()
        },
        index=["n_scenarios", "cv_iter", "type"],
    ).T

    n_scenarios_df = (
        pd.concat((cv_val_n_scenarios, cv_train_n_scenarios), axis=0)
        .sort_values("cv_iter")
        .reset_index(drop=True)
    )

    return n_scenarios_df


@st.cache_data()
def load_cv_results(results_dir: Path):
    """Loads the results of a cross-validation run."""
    run_config = sr.ml.gnn_gm.RunConfig.from_yaml(results_dir / "run_config.yaml")

    # obs_data = st_utils.get_observed_data(run_config.obs_data_ffp)
    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)
    dist_matrix = st_utils.get_dist_matrix(obs_data)

    run_results = {}
    for cur_cv_dir in results_dir.iterdir():
        if not cur_cv_dir.is_dir() or not cur_cv_dir.stem.startswith("cv_"):
            continue

        run_results[cur_cv_dir.stem] = st_utils.get_gnn_result(cur_cv_dir)

    comb_val_results = pd.read_parquet(results_dir / "val_results.parquet")

    return CVResults(
        run_config,
        obs_data,
        dist_matrix,
        run_results,
        comb_val_results,
        pd.read_pickle(results_dir / "metrics.pickle"),
    )


def is_valid_results_dir(results_dir: Path) -> bool:
    """Checks if the directory is a valid CV run results directory."""
    return (
        results_dir.is_dir()
        and not results_dir.stem.startswith("_")
        and "_cv_" in results_dir.stem
    )


def run_general(cv_results: CVResults):
    """General Tab."""

    # Number of scenarios per CV iteration plot
    n_scenarios_df = get_n_scenarios_df(cv_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=n_scenarios_df[n_scenarios_df["type"] == "train"],
        x="cv_iter",
        y="n_scenarios",
        color="blue",
        ax=ax1,
    )
    ax1.set_title("Number of Training Scenarios")

    sns.barplot(
        data=n_scenarios_df[n_scenarios_df["type"] == "val"],
        x="cv_iter",
        y="n_scenarios",
        color="red",
        ax=ax2,
    )
    ax2.set_title("Number of Validation Scenarios")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # Run config
    st.markdown("### Run Config")
    st.json(cv_results.run_config.to_dict())


def run_ind_cv_result_tab(cv_results: CVResults):
    """Allows exploration of results for a specific CV iteration."""
    # Select CV iteration
    cv_iter = st.selectbox(
        "CV Iteration",
        sorted(cv_results.run_results.keys()),
    )

    cur_results = cv_results.run_results[cv_iter]

    train_tab, val_tab = st.tabs(["Train", "Validation"])

    with train_tab:
        st_utils.scenario_viewer(
            cur_results.train_results,
            cv_results.obs_data,
            cv_results.dist_matrix,
            "train",
        )

    with val_tab:
        st_utils.scenario_viewer(
            cur_results.val_results, cv_results.obs_data, cv_results.dist_matrix, "val"
        )


def run_combined_val_results_tab(cv_results: CVResults):
    st_utils.scenario_viewer(
        cv_results.comb_val_results, cv_results.obs_data, cv_results.dist_matrix, "comb"
    )


def main(gnn_results_dir: Path):
    st.set_page_config(layout="wide")

    # Select GNN results
    gnn_result_id = st.selectbox(
        "Results Directory",
        sorted(
            [
                cur_ffp.stem
                for cur_ffp in gnn_results_dir.iterdir()
                if is_valid_results_dir(cur_ffp)
            ]
        ),
    )

    cur_result_dir = gnn_results_dir / gnn_result_id

    cv_results = load_cv_results(cur_result_dir)

    general_tab, ind_cv_result_tab, comb_val_results_tab = st.tabs(
        ["General", "Individual CV Results", "Combined Validation Results"]
    )

    with general_tab:
        run_general(cv_results)

    with ind_cv_result_tab:
        run_ind_cv_result_tab(cv_results)

    with comb_val_results_tab:
        run_combined_val_results_tab(cv_results)


if __name__ == "__main__":
    typer.run(main)
