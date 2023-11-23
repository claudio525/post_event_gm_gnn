from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import typer

import spatial_hazard as sh
import sim_ranking as sr
import sha_calc as sha
from gmhazard_calc.im import IM


@st.cache_data
def get_corr(corr_dir: Path, im: str):
    corr_path = corr_dir / f"{im}.csv"
    corr_df = pd.read_csv(corr_path, index_col=0)
    return corr_df


@st.cache_data
def get_ims(corr_dir: Path):
    ims = [p.stem for p in corr_dir.glob("*.csv")]
    return ims


@st.cache_data
def get_dist_matrix(db_ffp: Path, sites: np.ndarray = None):
    site_df = get_site_df(db_ffp)

    if sites is not None:
        site_mask = np.isin(sites, site_df.index.values)
        if np.count_nonzero(~site_mask) > 0:
            print(f"Warning: {np.count_nonzero(~site_mask)} sites not found in DB")
        site_df = site_df.loc[sites[site_mask]]

    return sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )


@st.cache_data
def get_site_df(db_ffp: Path):

    return sr.db.DB(db_ffp).get_site_df()

@st.cache_data
def get_obs_corr_df(obs_corr_ffp: Path, im: str):
    obs_corrs = pd.read_pickle(obs_corr_ffp)

    return obs_corrs[im]


def create_corr_plot(dist_matrix: pd.DataFrame, corr_df: pd.DataFrame, mask: np.ndarray, im: str):
    dist = np.linspace(0, 300, 100)
    loth_baker_vals = sha.loth_baker_corr_model.get_correlations(im, im, dist)

    # Get modified Loth & Baker values
    # based on eq 11.8 and tau and phi (estimates) from Ask14
    lb_tau = 0.5
    cp = IM.from_str(im).period
    if cp < 0.1:
        lb_phi = 0.65
    elif cp < 1.0:
        lb_phi = 0.6
    else:
        lb_phi = 0.55
    lb_updated = (loth_baker_vals * lb_phi ** 2 + lb_tau ** 2) / np.sqrt(
        lb_phi ** 2 + lb_tau ** 2
    )

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        dist_matrix.values[mask],
        corr_df.values[mask],
        s=1.0,
        alpha=0.75,
    )
    plt.plot(dist, loth_baker_vals, c="k", linewidth=1.0)
    plt.plot(dist, lb_updated, c="k", linestyle="--", linewidth=1.0)

    plt.xlabel(f"Distance (km)")
    plt.ylabel(f"Site-Correlation")
    plt.ylim(-1.0, 1.0)
    plt.xlim(0.0, 100)
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()

    return fig


def main(sim_corr_dir: Path, obs_corr_ffp: Path, db_ffp: Path):
    st.set_page_config(layout="wide")

    st.title("Correlations")

    ims = sorted(get_ims(sim_corr_dir))
    im = st.selectbox("IM", ims)

    ### Sim-based correlations
    st.markdown("### Simulation Within-Event Correlations")

    # Get and transform relevant data
    sim_corr_df = get_corr(sim_corr_dir, im)
    sim_corr_df = sim_corr_df.pivot(index="site_1", columns="site_2")
    sim_sites = sim_corr_df.index.values.astype(str)
    sim_site_df = get_site_df(db_ffp).loc[sim_sites]
    sim_dist_matrix = get_dist_matrix(db_ffp, sites=sim_sites)

    sim_mask = np.tril(sim_dist_matrix.values).astype(bool) & ~sim_corr_df.isna().values

    fig = create_corr_plot(sim_dist_matrix, sim_corr_df, sim_mask, im)
    st.pyplot(fig)

    # ### Observation-based correlations
    # st.markdown("### Observation Correlations")
    #
    # obs_corr_df = get_obs_corr_df(obs_corr_ffp, im)
    # obs_dist_matrix = get_dist_matrix(db_ffp, sites=obs_corr_df.index.values.astype(str))
    #
    # # Drop sites which aren't in the DB
    # obs_corr_df = obs_corr_df.loc[obs_dist_matrix.index.values, obs_dist_matrix.index.values]
    #
    # obs_mask = np.tril(obs_dist_matrix.values).astype(bool) & ~obs_corr_df.isna().values
    #
    # fig = create_corr_plot(obs_dist_matrix, obs_corr_df, obs_mask, im)
    # st.pyplot(fig)



if __name__ == "__main__":
    typer.run(main)
