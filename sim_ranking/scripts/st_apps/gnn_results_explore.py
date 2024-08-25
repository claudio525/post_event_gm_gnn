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
import spatial_hazard as sh

import st_utils


class SharedData(NamedTuple):
    obs_data: sr.ObservedData
    obs_sites: np.ndarray

    dist_matrix: pd.DataFrame

    gnn_result_id: str
    gnn_metrics: dict[str, pd.DataFrame]
    gnn_train_results: pd.DataFrame
    gnn_val_results: pd.DataFrame
    gnn_run_config: sr.ml.gnn_gm.RunConfig
    gnn_metadata: dict

    emp_cim_data: dict[str, sr.conditional.ConditionalMVNDistribution] = None


@st.cache_data
def get_observed_data(nzgmdb_ffp: Path) -> sr.ObservedData:
    return sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)


@st.cache_data(hash_funcs={sr.ObservedData: lambda x: hash(x.data_source)})
def get_dist_matrix(obs_data: sr.ObservedData) -> pd.DataFrame:
    return sh.im_dist.calculate_distance_matrix(obs_data.sites, obs_data.site_df)


@st.cache_data
def get_gnn_result(result_ffp: Path):
    metrics = pd.read_pickle(result_ffp / "metrics.pickle")
    train_results = pd.read_parquet(result_ffp / "train_results.parquet")
    val_results = pd.read_parquet(result_ffp / "val_results.parquet")

    return metrics, train_results, val_results


def _create_pot_sites_map(
    event: str,
    site_df: pd.DataFrame,
    int_sites: np.ndarray,
    obs_sites: np.ndarray,
    event_df: pd.DataFrame,
    site_int: str,
):
    all_sites = np.unique(np.concatenate([int_sites, obs_sites]))

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[obs_sites].lat,
                lon=site_df.loc[obs_sites].lon,
                mode="markers",
                marker=dict(size=10, color="blue"),
                hovertext=obs_sites,
                hoverinfo="text",
                name="Potential observation sites",
            ),
            go.Scattermapbox(
                lat=site_df.loc[int_sites].lat,
                lon=site_df.loc[int_sites].lon,
                mode="markers",
                marker=dict(size=10, color="orange"),
                hovertext=int_sites,
                hoverinfo="text",
                name="Potential sites of interest",
            ),
            go.Scattermapbox(
                lat=[site_df.loc[site_int, "lat"]],
                lon=[site_df.loc[site_int, "lon"]],
                mode="markers",
                marker=dict(size=20, color="red"),
                hovertext=site_int,
                hoverinfo="text",
                name="Site of interest",
            ),
            go.Scattermapbox(
                lat=[event_df.loc[event, "lat"]],
                lon=[event_df.loc[event, "lon"]],
                mode="markers",
                marker=dict(size=25, color="black"),
                hovertext=event,
                hoverinfo="text",
                name="Event",
            ),
        ]
    )

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=site_df.loc[all_sites].lat.mean(),
            lon=site_df.loc[all_sites].lon.mean(),
        ),
        zoom=8,
    )

    return fig


def _create_scenario_map(
    event: str,
    site_df: pd.DataFrame,
    emp_sites: np.ndarray,
    sim_sites: np.ndarray,
    ml_sites: np.ndarray,
    event_df: pd.DataFrame,
    site_int: str,
):
    all_sites = np.unique(np.concatenate([emp_sites, sim_sites, ml_sites]))

    # assert np.all(np.isin(emp_sites, sim_sites))
    cim_sites = np.unique(np.concatenate([emp_sites, sim_sites]))

    cim_only_sites = np.setdiff1d(cim_sites, ml_sites)
    ml_only_sites = np.setdiff1d(ml_sites, cim_sites)
    both_sites = np.intersect1d(cim_sites, ml_sites)
    assert np.all(
        np.isin(np.concatenate((cim_only_sites, ml_only_sites, both_sites)), all_sites)
    )

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=site_df.loc[both_sites].lat,
                lon=site_df.loc[both_sites].lon,
                mode="markers",
                marker=dict(size=10, color="blue"),
                hovertemplate="<b>Name: %{customdata.site}<br>Vs30: %{customdata.vs30}<br>Z1.0: %{customdata.z1p0}<extra></extra>",
                customdata=[
                    {
                        "site": cur_site,
                        "vs30": site_df.loc[cur_site, "vs30"],
                        "z1p0": site_df.loc[cur_site, "z1p0"],
                    }
                    for cur_site in both_sites
                ],
                name="Observation sites - cIM & ML",
            ),
            go.Scattermapbox(
                lat=site_df.loc[cim_only_sites].lat,
                lon=site_df.loc[cim_only_sites].lon,
                mode="markers",
                marker=dict(size=10, color="magenta"),
                customdata=[
                    {
                        "site": cur_site,
                        "vs30": site_df.loc[cur_site, "vs30"],
                        "z1p0": site_df.loc[cur_site, "z1p0"],
                    }
                    for cur_site in cim_only_sites
                ],
                hovertemplate="<b>Name: %{customdata.site}<br>Vs30: %{customdata.vs30}<br>Z1.0: %{customdata.z1p0}<extra></extra>",
                name="Observation sites - cIM only",
            ),
            go.Scattermapbox(
                lat=site_df.loc[ml_only_sites].lat,
                lon=site_df.loc[ml_only_sites].lon,
                mode="markers",
                marker=dict(size=10, color="green"),
                customdata=[
                    {
                        "site": cur_site,
                        "vs30": site_df.loc[cur_site, "vs30"],
                        "z1p0": site_df.loc[cur_site, "z1p0"],
                    }
                    for cur_site in ml_only_sites
                ],
                hovertemplate="<b>Name: %{customdata.site}<br>Vs30: %{customdata.vs30}<br>Z1.0: %{customdata.z1p0}<extra></extra>",
                name="Observation sites - ML only",
            ),
            go.Scattermapbox(
                lat=[site_df.loc[site_int, "lat"]],
                lon=[site_df.loc[site_int, "lon"]],
                mode="markers",
                marker=dict(size=20, color="red"),
                customdata=[
                    {
                        "site": cur_site,
                        "vs30": site_df.loc[cur_site, "vs30"],
                        "z1p0": site_df.loc[cur_site, "z1p0"],
                    }
                    for cur_site in [site_int]
                ],
                hovertemplate="<b>Name: %{customdata.site}<br>Vs30: %{customdata.vs30}<br>Z1.0: %{customdata.z1p0}<extra></extra>",
                name="Site of interest",
            ),
            go.Scattermapbox(
                lat=[event_df.loc[event, "lat"]],
                lon=[event_df.loc[event, "lon"]],
                mode="markers",
                marker=dict(size=25, color="black"),
                hovertext=event,
                hoverinfo="text",
                name="Event",
            ),
        ]
    )

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=site_df.loc[all_sites].lat.mean(),
            lon=site_df.loc[all_sites].lon.mean(),
        ),
        zoom=8,
    )

    return fig


def scenario_viewer(gnn_results: pd.DataFrame, shared_data: SharedData, tab_type: str):
    events = gnn_results.event_id.unique().astype(str)

    col1, col2 = st.columns([1, 6])

    with col1:
        # Select site and event
        cur_event = st.selectbox(
            "Event",
            shared_data.obs_data.event_df.loc[events]
            .sort_values("mag", ascending=False)
            .index.values.astype(str),
            key=f"{tab_type}_event",
        )

        cur_gnn_results = gnn_results.loc[gnn_results.event_id == cur_event]

        cur_int_sites = cur_gnn_results.site_int.unique().astype(str)
        cur_ob_sites = np.unique(
            np.concatenate(
                [cur_result.obs_sites for _, cur_result in cur_gnn_results.iterrows()]
            ).astype(str)
        )
        cur_int_site = st.selectbox(
            "Site of Interest", cur_int_sites, key=f"{tab_type}_site"
        )

        # Get GNN results
        cur_gnn_results = cur_gnn_results.loc[
            cur_gnn_results.site_int == cur_int_site
        ].squeeze()

        cur_obs_df = shared_data.obs_data.record_df
        cur_obs_df = cur_obs_df[cur_obs_df.event_id == cur_event].set_index("site_id")

        # Get empirical cIM results
        cur_emp_cim = (
            shared_data.emp_cim_data.get(cur_event, None)
            if shared_data.emp_cim_data is not None
            else None
        )
        if cur_emp_cim is not None:
            cur_emp_cim_mean = cur_emp_cim.cond_lnIM_mean_df
            cur_emp_cim_std = cur_emp_cim.cond_lnIM_std_df

        st.markdown(f"Magnitude: {shared_data.obs_data.event_df.loc[cur_event].mag}")

    with col2:
        fig = _create_pot_sites_map(
            cur_event,
            shared_data.obs_data.site_df,
            cur_int_sites,
            cur_ob_sites,
            shared_data.obs_data.event_df,
            cur_int_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("Scenario Map"):
        fig = _create_scenario_map(
            cur_event,
            shared_data.obs_data.site_df,
            [],
            [],
            # cur_emp_cim.get_obs_stations(cur_site),
            # cur_sim_cim.get_obs_stations(cur_site),
            cur_gnn_results.obs_sites,
            shared_data.obs_data.event_df,
            cur_int_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    pred_im_keys = mlt.array_utils.numpy_str_join("_", sr.constants.PSA_KEYS, "pred")

    col1, col2 = st.columns(2)

    line_options = ["ML", "Obs"]
    if cur_emp_cim is not None:
        line_options.append("emp_cIM")
    line_options.extend(cur_gnn_results.obs_sites.tolist())

    with col1:
        log_log = st.checkbox("Log-Log Plot", value=False, key=f"{tab_type}_log_log")
    with col2:
        sel_line_options = st.multiselect(
            "Lines",
            line_options,
            default=["ML", "Obs"] if cur_emp_cim is None else ["ML", "Obs", "emp_cIM"],
            key=f"{tab_type}_lines",
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    # ML
    if "ML" in sel_line_options:
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_results.loc[pred_im_keys].values.astype(float)),
            c="blue",
            label="ML - Mean",
            marker=".",
        )

    # Observed
    if "Obs" in sel_line_options:
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_results.loc[sr.constants.PSA_KEYS].values.astype(float)),
            c="red",
            label="Observed - SoI",
        )

    # Empirical cIM
    if cur_emp_cim is not None and "emp_cIM" in sel_line_options:
        cur_emp_cim_mean_values = cur_emp_cim_mean.loc[
            cur_int_site, sr.constants.PSA_KEYS
        ].values
        cur_emp_cim_std_values = cur_emp_cim_std.loc[
            cur_int_site, sr.constants.PSA_KEYS
        ].values

        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_emp_cim_mean_values),
            c="green",
            label="cIM",
        )
        ax.plot(
            sr.constants.PERIODS,
            np.stack(
                (
                    np.exp(cur_emp_cim_mean_values + cur_emp_cim_std_values),
                    np.exp(cur_emp_cim_mean_values - cur_emp_cim_std_values),
                ),
                axis=1,
            ),
            c="green",
            linestyle="--",
            linewidth=1.0,
        )

    # Observation sites
    obs_sites_to_plot = np.intersect1d(sel_line_options, cur_gnn_results.obs_sites)
    for cur_site in obs_sites_to_plot:
        cur_site_values = cur_obs_df.loc[cur_site, sr.constants.PSA_KEYS].values.astype(
            float
        )
        ax.plot(
            sr.constants.PERIODS,
            cur_site_values,
            label=f"Observed - {cur_site}",
            linestyle="dotted",
            linewidth=1.25,
        )

    ax.set_xscale("log")
    if log_log:
        ax.set_yscale("log")

    ax.set_xlabel(f"Period (s)")
    ax.set_ylabel(f"pSA")
    ax.set_xlim([0.01, 10])
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.legend()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    print(f"wtf")


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
        y_lim=(0.0, 0.2),
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

    print(f"wtf")


def run(
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

    emp_cim_data = None
    if emp_cim_results_dir is not None:
        emp_cim_events = [
            cur_ffp.stem
            for cur_ffp in emp_cim_results_dir.iterdir()
            if cur_ffp.is_dir()
        ]
        emp_cim_data = {
            cur_event: st_utils.cim_load_cmvn_result(
                emp_cim_results_dir / cur_event / "empirical_cMVN"
            )
            for cur_event in emp_cim_events
        }

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
    typer.run(run)
