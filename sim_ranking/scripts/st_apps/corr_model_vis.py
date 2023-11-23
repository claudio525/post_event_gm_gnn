from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import typer

import spatial_hazard as sh
import sim_ranking as sr
import sha_calc as sha
import ml_tools as mlt


@st.cache_data
def load_results(results_dir: Path):
    train_results = pd.read_csv(results_dir / "train_results.csv", index_col=0)
    val_results = pd.read_csv(results_dir / "val_results.csv", index_col=0)

    return train_results, val_results

@st.cache_data
def get_metadata(results_dir: Path):
    return mlt.utils.load_yaml(results_dir / "metadata.yaml")

def one_to_one_plot(results_df: pd.DataFrame, im: str):
    fig = plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df[f"sim_{im}"], results_df[f"pred_{im}"], s=1, c="k", alpha=0.5
    )

    plt.xlabel(f"Correlation ({im})")
    plt.ylabel(f"Predicted Correlation ({im})")
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

def run_results_tab(results_dir: Path):
    train_results, val_results = load_results(results_dir)
    meta = get_metadata(results_dir)

    im = st.selectbox("IM", meta["ims"])

    train_tab, val_tab = st.tabs(["Training", "Validation"])
    with train_tab:
        one_to_one_plot(train_results, im)
    with val_tab:
        one_to_one_plot(val_results, im)

def run_model_tab(results_dir: Path):
    meta = get_metadata(results_dir)

    vs30 = 250

    dist_range = np.linspace(0, meta["max_dist"], 100)

    print(f"wtf")

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

    results_tab, model_tab = st.tabs(["Results", "Model"])
    with results_tab:
        run_results_tab(cur_results_dir)
    with model_tab:
        run_model_tab(cur_results_dir)


if __name__ == '__main__':
    typer.run(main)

