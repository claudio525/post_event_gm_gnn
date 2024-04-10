import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

import sim_ranking as sr


@st.cache_data
def load_gm_params(gm_params_ffp: Path):
    return pd.read_csv(gm_params_ffp, index_col=0)


@st.cache_data
def ml_get_sim_data(results_dir: Path, event: str):
    metadata = ml_get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    sites = ml_get_event_sites(results_dir)[event]

    return sr.db.DB(db_ffp).get_sim_data(event, sites)


@st.cache_data
def ml_get_obs_df(results_dir: Path):
    metadata = ml_get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_obs_df()


@st.cache_data
def cim_load_cmvn_result(results_dir: Path):
    ffp = results_dir / "cMVN_distributions.pickle"
    if ffp.exists():
        return sr.conditional.ConditionalMVNDistribution.load(ffp)
    return None


@st.cache_data
def ml_get_event_sites(results_dir: Path):
    metadata = ml_get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_sites()


@st.cache_data
def ml_get_metadata(results_dir: Path):
    return sr.data.get_meta(results_dir)


@st.cache_data
def ml_get_site_df(results_dir: Path):
    metadata = ml_get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_site_df()


@st.cache_data
def ml_get_event_df(results_dir: Path):
    metadata = ml_get_metadata(results_dir)
    db_ffp = Path(os.path.expandvars("$wdata")) / metadata["data"]["db"]

    return sr.db.DB(db_ffp).get_event_df()


@st.cache_data
def ml_get_event_angular_distances(results_dir: Path):
    station_df = ml_get_site_df(results_dir)
    event_df = ml_get_event_df(results_dir)
    event_sites = ml_get_event_sites(results_dir)

    return sr.ml.features.compute_angular_distance(
        station_df,
        event_df,
        event_df.index.values.astype(str),
        event_sites,
        pre_process=False,
    )

@st.cache_data
def ml_load_scenario_results(results_dir: Path):
    train_results = pd.read_parquet(results_dir / "train_scenario_results.parquet")
    train_sum_results = pd.read_parquet(results_dir / "train_scenario_summary.parquet")

    val_results = pd.read_parquet(results_dir / "val_scenario_results.parquet")
    val_sum_results = pd.read_parquet(results_dir / "val_scenario_summary.parquet")

    return train_results, train_sum_results, val_results, val_sum_results


@st.cache_data
def ml_load_sample_results(results_dir: Path):
    train_results_df = pd.read_parquet(results_dir / "train_sample_results.parquet")
    val_results_df = pd.read_parquet(results_dir / "val_sample_results.parquet")

    angular_distances = ml_get_event_angular_distances(results_dir)
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
