"""
Allows for the comparison of event-specific site-correlations
for two different result folders
"""
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import typer
import matplotlib.pyplot as plt


import sha_calc as sha
import spatial_hazard as sh
import sim_ranking as sr


def main(results_dir_1: Path, results_dir_2: Path, sites_ffp: Path, label_1: str, label_2: str):
    st.set_page_config(layout="wide")

    site_df = pd.read_csv(sites_ffp, index_col="sta")[["lon", "lat"]]
    dist_matrix = sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )

    event_ffps_1 = results_dir_1.glob("*.pickle")
    event_ffps_2 = results_dir_2.glob("*.pickle")

    events_1 = [cur_ffp.stem for cur_ffp in event_ffps_1]
    events_2 = [cur_ffp.stem for cur_ffp in event_ffps_2]

    common_events = set(events_1).intersection(set(events_2))

    col1, col2 = st.columns(2)
    with col1:
        cur_event = st.selectbox("Event", list(common_events))
    with col2:
        cur_im = st.selectbox("IM", sr.constants.PSA_KEYS)

    cur_corr_1 = pd.read_pickle(results_dir_1 / f"{cur_event}.pickle")
    cur_corr_2 = pd.read_pickle(results_dir_2 / f"{cur_event}.pickle")
    assert np.all(cur_corr_1.sites == cur_corr_2.sites)

    cur_sites = cur_corr_1.sites
    cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]

    cur_corr_df_1 = cur_corr_1.get_im_corrs(cur_im)
    cur_corr_df_2 = cur_corr_2.get_im_corrs(cur_im)

    assert np.all(cur_dist_matrix.index == cur_corr_df_1.index) and np.all(
        cur_dist_matrix.index == cur_corr_df_1.columns
    )
    assert np.all(cur_dist_matrix.index == cur_corr_df_2.index) and np.all(
        cur_dist_matrix.index == cur_corr_df_2.columns
    )

    st.write(f"Event: {cur_event}")

    mask = np.tril(cur_dist_matrix.values).astype(bool)

    dist = np.linspace(0, 300, 100)
    loth_baker_vals = sha.loth_baker_corr_model.get_correlations(cur_im, cur_im, dist)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        cur_dist_matrix.values[mask],
        cur_corr_df_1.values[mask],
        s=1.5,
        c="b",
        label=label_1
    )
    ax.scatter(
        cur_dist_matrix.values[mask],
        cur_corr_df_2.values[mask],
        s=1.5,
        c="r",
        label=label_2
    )
    ax.plot(dist, loth_baker_vals, c="k", linewidth=1.0)

    ax.set_xlabel(f"Distance (km)")
    ax.set_ylabel(f"Site-Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0.0, 100)
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

    print(f"wtf")


if __name__ == "__main__":
    typer.run(main)
