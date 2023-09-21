from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import typer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns

import sim_ranking as sr


# @st.cache_resource
def get_db(db_ffp: Path):
    return sr.db.DB(db_ffp)


@st.cache_data
def get_obs_df(db_ffp: Path):
    return get_db(db_ffp).get_obs_df()


def get_sim_df(db_ffp: Path):
    return get_db(db_ffp).get_sim_df()


@st.cache_data
def get_full_obs_df(db_ffp: Path):
    return get_db(db_ffp).get_full_obs_df()


@st.cache_data
def get_site_df(db_ffp: Path):
    return get_db(db_ffp).get_site_df()


@st.cache_data
def get_event_df(db_ffp: Path):
    return get_db(db_ffp).get_event_df()


@st.cache_data
def compute_residuals(db_ffp: Path):
    sim_obs_df = get_db(db_ffp).get_sim_obs_df()

    sim_pSA_cols = np.char.add(sr.constants.PSA_KEYS, "_sim")
    obs_pSA_cols = np.char.add(sr.constants.PSA_KEYS, "_obs")

    res_df = pd.DataFrame(
        data=np.log(sim_obs_df.loc[:, sim_pSA_cols].values)
             - np.log(sim_obs_df.loc[:, obs_pSA_cols].values),
        columns=sr.constants.PSA_KEYS,
        index=sim_obs_df.index,
    )
    res_df["event_id"] = sim_obs_df.event_id.values
    res_df["site_id"] = sim_obs_df.site_id.values
    res_df["rel_id"] = sim_obs_df.rel_id.values
    res_df["data_source"] = sim_obs_df.data_source.values

    return res_df


@st.cache_data
def compute_residual_area(db_ffp: Path):
    res_df = compute_residuals(db_ffp)

    res_area = np.trapz(np.abs(res_df.loc[:, sr.constants.PSA_KEYS].values), axis=1)
    res_area_df = res_df.loc[:, ["event_id", "site_id", "rel_id", "data_source"]].copy()
    res_area_df["res_area"] = res_area

    return res_area_df

def run_general_tab(db_ffp: Path):
    obs_df = get_full_obs_df(db_ffp)

    res_area_df = compute_residual_area(db_ffp)

    with st.expander("Residual area histogram"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(res_area_df.res_area, bins=25, ax=ax)
        plt.xlabel(f"Residual Area")
        plt.ylabel(f"Count")
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        plt.tight_layout()

        st.pyplot(fig)


        print(f"wtf")


    with st.expander("Distribution of max(pSA) values across all periods"):
        max_pSA_df = pd.concat(
            (obs_df.loc[:, sr.constants.PSA_KEYS].max(axis=1), obs_df.mag), axis=1
        )
        max_pSA_df["mag_class"] = pd.cut(
            max_pSA_df.mag, bins=[0, 4, 6, 7], labels=["small", "moderate", "large"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            data=max_pSA_df,
            x=0,
            hue="mag_class",
            bins=25,
            log_scale=(False, True),
            ax=ax,
            multiple="dodge",
        )

        plt.xlabel(f"pSA (g)")
        plt.ylabel(f"Count")
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        plt.tight_layout()

        st.pyplot(fig)




def main(db_ffp: Path):
    st.set_page_config(layout="wide")

    (general_tab,) = st.tabs(["General"])



    with general_tab:
        run_general_tab(db_ffp)


if __name__ == "__main__":
    typer.run(main)
