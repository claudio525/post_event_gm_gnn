import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import ml_tools as mlt

from . import constants
from .data_classes import ObservedData


def get_obs_sites(
    event: str,
    int_site: str,
    gnn_results: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    n_obs_sites: int = 5,
):
    """
    Gets the observation sites for a 
    given event and location of interest.
    """
    cur_id = f"{event}_{int_site}"
    obs_sites = (
        dist_matrix.loc[int_site]
        .loc[gnn_results.loc[cur_id].obs_sites.astype(str)]
        .sort_values()
        .index.values.astype(str)
    )[:n_obs_sites]

    return obs_sites


def exp_formatter(y, pos):
    return f"{np.exp(y):.3f}"


def plot_gnn_cim(
    ax: plt.Axes,
    cur_id: str,
    gnn_results: pd.DataFrame,
    cim_results: pd.DataFrame = None,
    emp_gmm_params: pd.DataFrame = None,
    log_values: bool = False,
):
    emp_gmm_pSA_mean_keys = [
        f"pSA_{cur_period}_mean" for cur_period in constants.PERIODS
    ]
    emp_gmm_pSA_std_keys = [
        f"pSA_{cur_period}_std_Total" for cur_period in constants.PERIODS
    ]

    # Empirical GMM
    if emp_gmm_params is not None:
        cur_emp_gmm_mean = emp_gmm_params.loc[
            cur_id, emp_gmm_pSA_mean_keys
        ].values.astype(float)
        cur_emp_gmm_std = emp_gmm_params.loc[
            cur_id, emp_gmm_pSA_std_keys
        ].values.astype(float)
        ax.plot(
            constants.PERIODS,
            cur_emp_gmm_mean if log_values else np.exp(cur_emp_gmm_mean),
            label="Empirical GMM",
            c="gray",
        )
        ax.plot(
            constants.PERIODS,
            (
                cur_emp_gmm_mean + cur_emp_gmm_std
                if log_values
                else np.exp(cur_emp_gmm_mean + cur_emp_gmm_std)
            ),
            c="gray",
            linestyle="--",
        )
        ax.plot(
            constants.PERIODS,
            (
                cur_emp_gmm_mean - cur_emp_gmm_std
                if log_values
                else np.exp(cur_emp_gmm_mean - cur_emp_gmm_std)
            ),
            c="gray",
            linestyle="--",
        )

    # CIM
    if cim_results is not None:
        cur_cim_mean = cim_results.loc[
            cur_id, constants.CIM_PRED_PSA_KEYS
        ].values.astype(float)
        cur_cim_std = cim_results.loc[
            cur_id, constants.CIM_PRED_STD_PSA_KEYS
        ].values.astype(float)
        ax.plot(
            constants.PERIODS,
            cur_cim_mean if log_values else np.exp(cur_cim_mean),
            label="cIM",
            c="g",
        )
        ax.plot(
            constants.PERIODS,
            (
                cur_cim_mean + cur_cim_std
                if log_values
                else np.exp(cur_cim_mean + cur_cim_std)
            ),
            c="g",
            linestyle="--",
        )
        ax.plot(
            constants.PERIODS,
            (
                cur_cim_mean - cur_cim_std
                if log_values
                else np.exp(cur_cim_mean - cur_cim_std)
            ),
            c="g",
            linestyle="--",
        )

    # GNN
    cur_gnn_mean = gnn_results.loc[cur_id, constants.GNN_PRED_PSA_KEYS].values.astype(
        float
    )
    cur_gnn_std = gnn_results.loc[
        cur_id, constants.GNN_PRED_STD_PSA_KEYS
    ].values.astype(float)
    ax.plot(
        constants.PERIODS,
        cur_gnn_mean if log_values else np.exp(cur_gnn_mean),
        label="GNN",
        c="b",
    )
    ax.plot(
        constants.PERIODS,
        (
            cur_gnn_mean + cur_gnn_std
            if log_values
            else np.exp(cur_gnn_mean + cur_gnn_std)
        ),
        c="b",
        linestyle="--",
    )
    ax.plot(
        constants.PERIODS,
        (
            cur_gnn_mean - cur_gnn_std
            if log_values
            else np.exp(cur_gnn_mean - cur_gnn_std)
        ),
        c="b",
        linestyle="--",
    )

    # Observed
    obs_values = gnn_results.loc[cur_id, constants.PSA_KEYS].values.astype(float)
    ax.plot(
        constants.PERIODS,
        obs_values if log_values else np.exp(obs_values),
        label="Observed",
        c="r",
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    if log_values:
        ax.yaxis.set_major_formatter(FuncFormatter(exp_formatter))

    ax.set_xscale("log")
    ax.set_xlim(0.01, 10)
    ax.set_xlabel("Period (s)")

    # Get smallest/largest y-values
    min_y = min([min(line.get_ydata()) for line in ax.get_lines()])
    max_y = max([max(line.get_ydata()) for line in ax.get_lines()])
    if log_values:
        return np.exp(min_y), np.exp(max_y)
    return min_y, max_y


def plot_obs_sites(
    ax: plt.Axes,
    event: str,
    int_site: str,
    obs_sites: np.ndarray[str],
    obs_data: ObservedData,
    dist_matrix: pd.DataFrame,
    log_values: bool = False,
):
    cur_obs_sites_dist = dist_matrix.loc[int_site].loc[obs_sites]
    cur_obs_data = obs_data.get_event_data(event, list(obs_sites) + [int_site])

    for cur_site in obs_sites:
        cur_values = cur_obs_data.loc[cur_site, constants.PSA_KEYS].values.astype(float)
        ax.plot(
            constants.PERIODS,
            np.log(cur_values) if log_values else cur_values,
            label=f"{cur_site}, {cur_obs_sites_dist[cur_site]:.2f} km",
            linestyle="--",
        )

    site_int_values = cur_obs_data.loc[int_site, constants.PSA_KEYS].values.astype(
        float
    )
    ax.plot(
        constants.PERIODS,
        np.log(site_int_values) if log_values else site_int_values,
        label=f"{int_site}",
        c="r",
        linewidth=2.5,
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    if log_values:
        ax.yaxis.set_major_formatter(FuncFormatter(exp_formatter))

    ax.set_xscale("log")
    ax.set_xlim(0.01, 10)
    ax.set_xlabel("Period (s)")

    # Get smallest/largest y-values
    min_y = min([min(line.get_ydata()) for line in ax.get_lines()])
    max_y = max([max(line.get_ydata()) for line in ax.get_lines()])
    if log_values:
        return np.exp(min_y), np.exp(max_y)
    return min_y, max_y


def create_2plot_log(
    event: str,
    int_site: str,
    obs_sites: np.ndarray[str],
    gnn_results: pd.DataFrame,
    cim_results: pd.DataFrame,
    obs_data: ObservedData,
    dist_matrix: pd.DataFrame,
    emp_gmm_params: pd.DataFrame = None,
    plot_gnn_with_obs: bool = False,
    plot_cim_with_obs: bool = False,
):
    cur_id = f"{event}_{int_site}"

    fig, (ax1, ax2) = mlt.plotting.get_fig_axes(2, 2, 1, ind_figsize=(8, 6))

    min_y2, max_y2 = plot_gnn_cim(
        ax1,
        cur_id,
        gnn_results,
        cim_results,
        log_values=True,
        emp_gmm_params=emp_gmm_params,
    )
    ax1.legend()

    min_y4, max_y4 = plot_obs_sites(
        ax2, event, int_site, obs_sites, obs_data, dist_matrix, log_values=True
    )
    # GNN
    if plot_gnn_with_obs:
        ax2.plot(
            constants.PERIODS,
            gnn_results.loc[cur_id, constants.GNN_PRED_PSA_KEYS].values.astype(float),
            c="b",
            label="GNN",
            linewidth=2.5,
        )
    # cIM
    if cim_results is not None and plot_cim_with_obs:
        ax2.plot(
            constants.PERIODS,
            cim_results.loc[cur_id, constants.CIM_PRED_PSA_KEYS].values.astype(float),
            c="g",
            label="cIM",
            linewidth=2.5,
        )

    ax2.legend()

    # y-axis limits
    min_y = min(min_y2, min_y4)
    max_y = max(max_y2, max_y4)
    ax1.set_ylim(np.log(min_y), np.log(max_y))
    ax2.set_ylim(np.log(min_y), np.log(max_y))

    fig.tight_layout()
    return fig


def get_site_info_df(
    event: str,
    int_site: str,
    obs_sites: np.ndarray[str],
    obs_data: ObservedData,
    dist_matrix: pd.DataFrame,
):
    site_cols = ["rrup", "vs30", "z1p0", "z2p5", "tsite"]
    site_info_sites = [int_site] + obs_sites.tolist()
    site_info_records = mlt.array_utils.numpy_str_join("_", event, site_info_sites)

    site_info_df = obs_data.record_df.loc[site_info_records, site_cols].copy()
    site_info_df["site_int_distance"] = (
        dist_matrix.loc[int_site].loc[site_info_sites].values
    )

    return site_info_df
