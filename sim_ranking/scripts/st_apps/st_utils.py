import os
from pathlib import Path
from typing import Sequence, NamedTuple

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import sim_ranking as sr
import ml_tools as mlt
import spatial_hazard as sh


class GNNRunResults(NamedTuple):
    run_config: sr.ml.gnn_gm.RunConfig
    train_results: pd.DataFrame
    val_results: pd.DataFrame
    val_int_sites: np.ndarray[str]
    train_int_sites: np.ndarray[str]
    metrics: dict
    metadata: dict


@st.cache_data
def get_gnn_result(result_ffp: Path):
    run_config = sr.ml.gnn_gm.RunConfig.from_yaml(result_ffp / "run_config.yaml")
    metrics = pd.read_pickle(result_ffp / "metrics.pickle")
    train_results = pd.read_parquet(result_ffp / "train_results.parquet")
    train_int_sites = np.load(result_ffp / "train_int_sites.npy")
    val_results = pd.read_parquet(result_ffp / "val_results.parquet")
    val_int_sites = np.load(result_ffp / "val_int_sites.npy")
    metadata = mlt.utils.load_yaml(result_ffp / "metadata.yaml")

    return GNNRunResults(
        run_config,
        train_results,
        val_results,
        val_int_sites,
        train_int_sites,
        metrics,
        metadata,
    )


@st.cache_data(hash_funcs={sr.ObservedData: lambda x: hash(x.data_ffp)})
def get_dist_matrix(obs_data: sr.ObservedData) -> pd.DataFrame:
    return sh.im_dist.calculate_distance_matrix(obs_data.sites, obs_data.site_df)


@st.cache_data
def get_observed_data(nzgmdb_ffp: Path) -> sr.ObservedData:
    return sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)


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


def scenario_viewer(
    gnn_results: pd.DataFrame,
    obs_data: sr.ObservedData,
    dist_matrix: pd.DataFrame,
    tab_type: str,
):
    events = gnn_results.event_id.unique().astype(str)

    col1, col2 = st.columns([1, 6])

    with col1:
        # Select site and event
        cur_event = st.selectbox(
            "Event",
            obs_data.event_df.loc[events]
            .sort_values("mag", ascending=False)
            .index.values.astype(str),
            key=f"{tab_type}_event",
        )

        cur_gnn_result = gnn_results.loc[gnn_results.event_id == cur_event]

        int_sites = cur_gnn_result.site_int.unique().astype(str)
        obs_sites = np.unique(
            np.concatenate(
                [cur_result.obs_sites for _, cur_result in cur_gnn_result.iterrows()]
            ).astype(str)
        )
        cur_int_site = st.selectbox(
            "Site of Interest", int_sites, key=f"{tab_type}_site"
        )

        # Get GNN results
        cur_gnn_result = cur_gnn_result.loc[
            cur_gnn_result.site_int == cur_int_site
        ].squeeze()

        cur_obs_df = obs_data.record_df
        cur_obs_df = cur_obs_df[cur_obs_df.event_id == cur_event].set_index("site_id")

        # Get empirical cIM results
        # cur_emp_cim = (
        #     shared_data.emp_cim_data.get(cur_event, None)
        #     if shared_data.emp_cim_data is not None
        #     else None
        # )
        # if cur_emp_cim is not None:
        #     cur_emp_cim_mean = cur_emp_cim.cond_lnIM_mean_df
        #     cur_emp_cim_std = cur_emp_cim.cond_lnIM_std_df

        st.markdown(f"Magnitude: {obs_data.event_df.loc[cur_event].mag}")
        st.markdown(f"Loss: {cur_gnn_result.loss:.4f}")

    with col2:
        fig = _create_pot_sites_map(
            cur_event,
            obs_data.site_df,
            int_sites,
            obs_sites,
            obs_data.event_df,
            cur_int_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("Scenario Map"):
        fig = _create_scenario_map(
            cur_event,
            obs_data.site_df,
            [],
            [],
            # cur_emp_cim.get_obs_stations(cur_site),
            # cur_sim_cim.get_obs_stations(cur_site),
            cur_gnn_result.obs_sites,
            obs_data.event_df,
            cur_int_site,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    pred_im_keys = mlt.array_utils.numpy_str_join("_", sr.constants.PSA_KEYS, "pred")
    pred_std_im_keys = mlt.array_utils.numpy_str_join(
        "_", sr.constants.PSA_KEYS, "pred_std"
    )

    col1, col2 = st.columns(2)

    line_options = ["ML", "Obs"]
    # if cur_emp_cim is not None:
    #     line_options.append("emp_cIM")
    cur_obs_sites = (
        dist_matrix.loc[cur_int_site]
        .loc[cur_gnn_result.obs_sites.astype(str)]
        .sort_values()
        .index.values.astype(str)
    )
    line_options.extend(cur_obs_sites)

    with col1:
        log_log = st.checkbox("Log-Log Plot", value=False, key=f"{tab_type}_log_log")
    with col2:
        sel_line_options = st.multiselect(
            "Lines",
            line_options,
            default=[
                "ML",
                "Obs",
            ],  # if cur_emp_cim is None else ["ML", "Obs", "emp_cIM"],
            key=f"{tab_type}_lines",
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    # ML
    if "ML" in sel_line_options:
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_result.loc[pred_im_keys].values.astype(float)),
            c="blue",
            label="ML",
            marker=".",
        )
        ax.plot(
            sr.constants.PERIODS,
            np.stack(
                (
                    np.exp(
                        cur_gnn_result.loc[pred_im_keys].values.astype(float)
                        + cur_gnn_result.loc[pred_std_im_keys].values.astype(float)
                    ),
                    np.exp(
                        cur_gnn_result.loc[pred_im_keys].values.astype(float)
                        - cur_gnn_result.loc[pred_std_im_keys].values.astype(float)
                    ),
                ),
                axis=1,
            ),
            c="blue",
            linestyle="--",
            linewidth=1.0,
        )

    # Observed
    if "Obs" in sel_line_options:
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_result.loc[sr.constants.PSA_KEYS].values.astype(float)),
            c="red",
            label="Observed - SoI",
        )

    # Empirical cIM
    # if cur_emp_cim is not None and "emp_cIM" in sel_line_options:
    #     cur_emp_cim_mean_values = cur_emp_cim_mean.loc[
    #         cur_int_site, sr.constants.PSA_KEYS
    #     ].values
    #     cur_emp_cim_std_values = cur_emp_cim_std.loc[
    #         cur_int_site, sr.constants.PSA_KEYS
    #     ].values
    #
    #     ax.plot(
    #         sr.constants.PERIODS,
    #         np.exp(cur_emp_cim_mean_values),
    #         c="green",
    #         label="cIM",
    #     )
    #     ax.plot(
    #         sr.constants.PERIODS,
    #         np.stack(
    #             (
    #                 np.exp(cur_emp_cim_mean_values + cur_emp_cim_std_values),
    #                 np.exp(cur_emp_cim_mean_values - cur_emp_cim_std_values),
    #             ),
    #             axis=1,
    #         ),
    #         c="green",
    #         linestyle="--",
    #         linewidth=1.0,
    #     )

    # Observation sites
    obs_sites_to_plot = np.intersect1d(sel_line_options, cur_gnn_result.obs_sites)
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

    # Site information
    st.markdown("### Site Information")
    site_cols = ["rrup", "vs30", "z1p0", "z2p5", "tsite"]
    site_info_sites = [cur_int_site] + cur_obs_sites.tolist()
    site_info_records = mlt.array_utils.numpy_str_join("_", cur_event, site_info_sites)

    site_info_df = obs_data.record_df.loc[site_info_records, site_cols].copy()
    site_info_df["site_int_distance"] = (
        dist_matrix.loc[cur_int_site].loc[site_info_sites].values
    )

    st.table(site_info_df.sort_values("site_int_distance", ascending=True))

    # IM loss
    st.markdown("### IM Loss")
    im_loss_keys = mlt.array_utils.numpy_str_join("_", sr.constants.PSA_KEYS, "loss")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sr.constants.PERIODS,
        cur_gnn_result.loc[im_loss_keys].values.astype(float),
        c="blue",
        label="IM Loss",
        marker=".",
    )
    ax.set_xscale("log")
    ax.set_xlim([0.01, 10])
    ax.set_xlabel(f"Period (s)")
    ax.set_ylabel(f"Loss")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# @st.cache_data
# def cim_load_cmvn_result(results_dir: Path):
#     ffp = results_dir / "cMVN_distributions.pickle"
#     if ffp.exists():
#         return sr.conditional.ConditionalMVNDistribution.load(ffp)
#     return None

# @st.cache_resource
# def load_emp_cim_data(emp_cim_results_dir: Path):
#     print(emp_cim_results_dir)
#     emp_cim_events = [
#         cur_ffp.stem for cur_ffp in emp_cim_results_dir.iterdir() if cur_ffp.is_dir()
#     ]
#     emp_cim_data = {
#         cur_event: cim_load_cmvn_result(emp_cim_results_dir / cur_event)
#         for cur_event in emp_cim_events
#     }
#     return emp_cim_data


def scatter_options_form(df: pd.DataFrame, cols: Sequence[str], key_prefix: str):
    COLORS = ["green", "blue", "red", "black"]
    with st.form(key=f"{key_prefix}_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ## X-axis
            st.markdown("### X-axis")
            x_axis = st.selectbox("Select X-axis", cols)

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                x_min_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_x_min_use_qt"
                )
                x_min = st.number_input(
                    "Min", key=f"{key_prefix}_x_min", value=float(df[x_axis].min())
                )
                st.text(f"Min data value: {float(df[x_axis].min()):.4f}")
            with c2:
                x_max_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_x_max_use_qt"
                )
                x_max = st.number_input(
                    "Max", key=f"{key_prefix}_x_max", value=float(df[x_axis].max())
                )
                st.text(f"Max data value: {float(df[x_axis].max()):.4f}")

            st.divider()

            ## Other Options
            st.markdown("### Other Options")
            alpha = st.number_input("Alpha", 0.0, 1.0, 1.0, key=f"{key_prefix}_alpha")
            marker_size = st.number_input(
                "Marker Size", 1, 100, 10, key=f"{key_prefix}_marker_size"
            )

        with col2:
            ## Y-axis
            st.markdown("### Y-axis")
            y_axis = st.selectbox("Select Y-axis", cols, key=f"{key_prefix}_y_axis")

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                y_min_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_y_min_use_qt"
                )
                y_min = st.number_input(
                    "Min", key=f"{key_prefix}_y_min", value=float(df[y_axis].min())
                )
                st.text(f"Min data value: {float(df[y_axis].min()):.4f}")
            with c2:
                y_max_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_y_max_use_qt"
                )
                y_max = st.number_input(
                    "Max", key=f"{key_prefix}_y_max", value=float(df[y_axis].max())
                )
                st.text(f"Max data value: {float(df[y_axis].max()):.4f}")

            st.divider()

            ## Trend line
            st.markdown("### Binned trend line")
            c1, c2 = st.columns(2)
            with c1:
                show_trend_mean_line = st.checkbox(
                    "Show Binned Trend Line",
                    key=f"{key_prefix}_show_trend_mean_line",
                )
            with c2:
                show_trend_std_line = st.checkbox(
                    "Show Binned Trend Std Line",
                    key=f"{key_prefix}_show_trend_std_line",
                )
            trend_n_bins = st.number_input(
                "Number of Bins", 1, 100, 10, key=f"{key_prefix}_trend_n_bins"
            )

            trend_line_style = st.selectbox(
                "Trend Line Style",
                ["-", "--", "-.", ":"],
                key=f"{key_prefix}_trend_line_style",
            )
            trend_line_width = st.number_input(
                "Trend Line Width",
                0.5,
                10.0,
                1.0,
                key=f"{key_prefix}_trend_line_width",
            )
            trend_color = st.selectbox(
                "Trend Line Color", COLORS, key=f"{key_prefix}_trend_color"
            )

        with col3:
            ## Color
            st.markdown("### Color")
            st.markdown("#### Fixed Color")
            fixed_color = st.checkbox(
                "Use Fixed Color", key=f"{key_prefix}_use_fixed_color"
            )
            color = st.selectbox("Select Color:", COLORS, key=f"{key_prefix}_color")

            st.divider()

            st.markdown("#### Colormap")
            color_axis = st.selectbox(
                "Select Color Axis", cols, key=f"{key_prefix}_color_axis"
            )
            cmap = st.selectbox(
                "Select Color Map",
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "Blues",
                    "Blues_r",
                ],
                key=f"{key_prefix}_cmap",
            )

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                vmin_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_color_min_use_qt"
                )
                vmin = st.number_input(
                    "Min",
                    key=f"{key_prefix}_color_min",
                    value=float(df[color_axis].min()),
                )
                st.text(f"Min data value: {float(df[color_axis].min()):.4f}")
            with c2:
                vmax_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_color_max_use_qt"
                )
                vmax = st.number_input(
                    "Max",
                    key=f"{key_prefix}_color_max",
                    value=float(df[color_axis].max()),
                )
                st.text(f"Max data value: {float(df[color_axis].max()):.4f}")

        submitted = st.form_submit_button("Submit")
        if submitted:
            scatter_options = mlt.plotting.ScatterOptions(
                x_axis=x_axis,
                x_min_use_qt=x_min_use_qt,
                x_max_use_qt=x_max_use_qt,
                x_min=x_min,
                x_max=x_max,
                y_axis=y_axis,
                y_min_use_qt=y_min_use_qt,
                y_max_use_qt=y_max_use_qt,
                y_min=y_min,
                y_max=y_max,
                use_fixed_color=fixed_color,
                color=color,
                color_axis=color_axis,
                cmap=cmap,
                vmin_use_qt=vmin_use_qt,
                vmax_use_qt=vmax_use_qt,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                marker_size=marker_size,
                show_trend_mean_line=show_trend_mean_line,
                show_trend_std_line=show_trend_std_line,
                trend_n_bins=trend_n_bins,
                trend_line_style=trend_line_style,
                trend_line_width=trend_line_width,
                trend_color=trend_color,
            )
            return scatter_options
    return None
