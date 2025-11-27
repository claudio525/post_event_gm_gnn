from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from tqdm import tqdm

import spatial_hazard as sh
import sim_ranking as sr
import ml_tools as mlt


app = typer.Typer()


@app.command("event-scenarios")
def event_scenarios(
    events: List[str],
    gnn_results_ffp: Path,
    emp_cim_results_dir: Path,
    nzgmdb_ffp: Path,
    output_dir: Path,
):
    """
    Visualises individual scenarios, shows
    - GNN mean and std
    - Empirical CIM mean and std
    - Observed GM at the SOI
    """
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    gnn_results = pd.read_parquet(gnn_results_ffp)
    scenario_ids = gnn_results.loc[
        gnn_results.event_id.isin(events)
    ].index.values.astype(str)

    emp_cim_results = sr.CIMResults.from_dir(emp_cim_results_dir, events)
    emp_mean_df = emp_cim_results.mean_df
    emp_std_df = emp_cim_results.std_df

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)

    pred_im_keys = mlt.array_utils.numpy_str_join("_", sr.constants.PSA_KEYS, "pred")
    pred_std_im_keys = mlt.array_utils.numpy_str_join(
        "_", sr.constants.PSA_KEYS, "pred_std"
    )

    for cur_sc_id in tqdm(scenario_ids):
        cur_gnn_results = gnn_results.loc[cur_sc_id]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

        ### ML
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_results.loc[pred_im_keys].values.astype(float)),
            c="blue",
            label="GNN",
        )
        ax.plot(
            sr.constants.PERIODS,
            np.stack(
                (
                    np.exp(
                        cur_gnn_results.loc[pred_im_keys].values.astype(float)
                        + cur_gnn_results.loc[pred_std_im_keys].values.astype(float)
                    ),
                    np.exp(
                        cur_gnn_results.loc[pred_im_keys].values.astype(float)
                        - cur_gnn_results.loc[pred_std_im_keys].values.astype(float)
                    ),
                ),
                axis=1,
            ),
            c="blue",
            linestyle="--",
            linewidth=1.0,
        )

        ### Observed
        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_gnn_results.loc[sr.constants.PSA_KEYS].values.astype(float)),
            c="red",
            label="SOI Observed",
        )

        cur_emp_cim_mean_values = (
            emp_mean_df[sr.constants.PSA_KEYS].loc[cur_sc_id].values
        )
        cur_emp_cim_std_values = emp_std_df[sr.constants.PSA_KEYS].loc[cur_sc_id].values

        ax.plot(
            sr.constants.PERIODS,
            np.exp(cur_emp_cim_mean_values),
            c="green",
            label="Cond. Dist.",
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

        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Pseudo Spectral Acceleration (g)")
        ax.set_xlim((0.01, 10))
        ax.set_title(
            f"{cur_gnn_results.site_int}, "
            f"$R_{{rup}}$ = {obs_data.record_df.loc[cur_sc_id, 'rrup']:.2f} (km), "
            f"$V_{{s30}}$ = {int(obs_data.site_df.loc[cur_gnn_results.site_int, 'vs30'])} (m/s)"
        )
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_xscale("log")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"{cur_sc_id}.png")


def __load_residuals(
    gnn_results_dir: Path, emp_cim_results_dir: Path, obs_data: sr.ObservedData
):
    # Load GNN results & compute residuals
    gnn_train_results = pd.read_parquet(gnn_results_dir / "train_results.parquet")
    gnn_val_results = pd.read_parquet(gnn_results_dir / "val_results.parquet")

    gnn_train_events = gnn_train_results.event_id.unique().astype(str)
    gnn_val_events = gnn_val_results.event_id.unique().astype(str)

    gnn_train_res_df = sr.analysis.get_residuals(gnn_train_results)
    gnn_val_res_df = sr.analysis.get_residuals(gnn_val_results)

    # Load empirical cIM results & compute residuals
    train_emp_cim_result = sr.CIMResults.from_dir(emp_cim_results_dir, gnn_train_events)
    val_emp_cim_result = sr.CIMResults.from_dir(emp_cim_results_dir, gnn_val_events)

    train_emp_cim_res_df = train_emp_cim_result.get_residual_df(obs_data)
    val_emp_cim_res_df = val_emp_cim_result.get_residual_df(obs_data)

    # Get shared scenario IDs
    shared_train_sc_ids = np.intersect1d(
        gnn_train_res_df.index.values, train_emp_cim_res_df.index.values
    )
    shared_val_sc_ids = np.intersect1d(
        gnn_val_res_df.index.values, val_emp_cim_res_df.index.values
    )
    print(
        f"Shared train scenarios: {shared_train_sc_ids.shape[0]}/{gnn_train_res_df.shape[0]}"
    )
    print(
        f"Shared val scenarios: {shared_val_sc_ids.shape[0]}/{gnn_val_res_df.shape[0]}"
    )

    return (
        gnn_train_res_df,
        gnn_val_res_df,
        train_emp_cim_res_df,
        val_emp_cim_res_df,
        shared_train_sc_ids,
        shared_val_sc_ids,
    )


@app.command("pSA-bias-std")
def pSA_bias_std(
    gnn_results_dir: Path,
    emp_cim_results_dir: Path,
    nzgmdb_ffp: Path,
    output_dir: Path,
):
    """
    Visualises the bias and the residual bias
    of the GNN and empirical cIM predictions
    """
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    # Load observed data
    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)

    (
        gnn_train_res_df,
        gnn_val_res_df,
        train_emp_cim_res_df,
        val_emp_cim_res_df,
        shared_train_sc_ids,
        shared_val_sc_ids,
    ) = __load_residuals(gnn_results_dir, emp_cim_results_dir, obs_data)

    # Compute mean and std of residuals
    gnn_train_res_mean_std_df = pd.concat(
        (
            gnn_train_res_df.loc[shared_train_sc_ids, sr.constants.PSA_KEYS].mean(
                axis=0
            ),
            gnn_train_res_df.loc[shared_train_sc_ids, sr.constants.PSA_KEYS].std(
                axis=0
            ),
        ),
        axis=1,
    )
    gnn_train_res_mean_std_df.columns = ["mean", "std"]

    gnn_val_res_mean_std_df = pd.concat(
        (
            gnn_val_res_df.loc[shared_val_sc_ids, sr.constants.PSA_KEYS].mean(axis=0),
            gnn_val_res_df.loc[shared_val_sc_ids, sr.constants.PSA_KEYS].std(axis=0),
        ),
        axis=1,
    )
    gnn_val_res_mean_std_df.columns = ["mean", "std"]

    emp_cim_train_res_mean_std_df = pd.concat(
        (
            train_emp_cim_res_df.loc[shared_train_sc_ids, sr.constants.PSA_KEYS].mean(
                axis=0
            ),
            train_emp_cim_res_df.loc[shared_train_sc_ids, sr.constants.PSA_KEYS].std(
                axis=0
            ),
        ),
        axis=1,
    )
    emp_cim_train_res_mean_std_df.columns = ["mean", "std"]

    emp_cim_val_res_mean_std_df = pd.concat(
        (
            val_emp_cim_res_df.loc[shared_val_sc_ids, sr.constants.PSA_KEYS].mean(
                axis=0
            ),
            val_emp_cim_res_df.loc[shared_val_sc_ids, sr.constants.PSA_KEYS].std(
                axis=0
            ),
        ),
        axis=1,
    )
    emp_cim_val_res_mean_std_df.columns = ["mean", "std"]

    ### Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.semilogx(
        sr.constants.PERIODS,
        gnn_train_res_mean_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        label="GNN - Training",
        c="b",
    )
    ax1.semilogx(
        sr.constants.PERIODS,
        gnn_val_res_mean_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        label="GNN - Validation",
        c="b",
        linestyle="--",
    )
    ax1.semilogx(
        sr.constants.PERIODS,
        emp_cim_train_res_mean_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        label="Emp. Cond. - Training",
        c="g",
    )
    ax1.semilogx(
        sr.constants.PERIODS,
        emp_cim_val_res_mean_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        label="Emp. Cond. - Validation",
        linestyle="--",
        c="g",
    )
    ax1.axhline(0, color="k", linestyle="--")
    ax1.set_xlabel(f"Period (s)")
    ax1.set_ylabel(f"Bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlim(0.01, 10)
    ax1.set_ylim(-0.5, 0.5)
    ax1.legend()

    ax2.semilogx(
        sr.constants.PERIODS,
        gnn_train_res_mean_std_df.loc[sr.constants.PSA_KEYS, "std"],
        c="b",
    )
    ax2.semilogx(
        sr.constants.PERIODS,
        gnn_val_res_mean_std_df.loc[sr.constants.PSA_KEYS, "std"],
        c="b",
        linestyle="--",
    )
    ax2.semilogx(
        sr.constants.PERIODS,
        emp_cim_train_res_mean_std_df.loc[sr.constants.PSA_KEYS, "std"],
        c="g",
    )
    ax2.semilogx(
        sr.constants.PERIODS,
        emp_cim_val_res_mean_std_df.loc[sr.constants.PSA_KEYS, "std"],
        linestyle="--",
        c="g",
    )
    ax2.set_xlabel(f"Period (s)")
    ax2.set_ylabel(f"Standard deviation of residuals")
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlim(0.01, 10)

    fig.tight_layout()
    fig.savefig(output_dir / "pSA_bias_std.png")


def _plot_res_hist(
    emp_cim_res_df: pd.DataFrame,
    gnn_res_df: pd.DataFrame,
    ims: Sequence[str],
    n_bins: int,
    output_dir: Path,
    res_type: str,
):
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    bins = np.linspace(-2, 2, n_bins)
    for cur_im in ims:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cur_mean = emp_cim_res_df[cur_im].mean(axis=0)
        cur_std = emp_cim_res_df[cur_im].std(axis=0)
        ax1.axvline(cur_mean, color="k", linestyle="-", label="Mean", linewidth=1.5)
        ax1.axvline(
            cur_mean + cur_std, color="k", linestyle="--", label="Std", linewidth=1.0
        )
        ax1.axvline(cur_mean - cur_std, color="k", linestyle="--", linewidth=1.0)
        ax1.hist(emp_cim_res_df[cur_im], bins=bins)
        ax1.set_xlabel(f"Residual")
        ax1.set_ylabel(f"Count")
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlim(-2, 2)
        ax1.set_title("Empirical Conditional")
        ax1.text(
            0.02,
            0.98,
            f"$\mu = {cur_mean:.2f}$, $\sigma = {cur_std:.2}$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax1.transAxes,
        )
        ax1.text(
            0.98,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax1.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        cur_mean = gnn_res_df[cur_im].mean(axis=0)
        cur_std = gnn_res_df[cur_im].std(axis=0)
        ax2.axvline(cur_mean, color="k", linestyle="-", label="Mean", linewidth=1.5)
        ax2.axvline(
            cur_mean + cur_std, color="k", linestyle="--", label="Std", linewidth=1.0
        )
        ax2.axvline(cur_mean - cur_std, color="k", linestyle="--", linewidth=1.0)
        ax2.hist(gnn_res_df[cur_im], bins=bins)
        ax2.set_xlabel(f"Residual")
        ax2.set_ylabel(f"Count")
        ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax2.set_xlim(-2, 2)
        ax2.set_title("GNN")

        ax2.text(
            0.02,
            0.98,
            f"$\mu = {cur_mean:.2f}$, $\sigma = {cur_std:.2}$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.98,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax2.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        # fig.suptitle(f"{sr.utils.get_nice_im_name(cur_im)}")
        fig.tight_layout()

        fig.savefig(output_dir / f"{res_type}_residual_distribution_{cur_im}.png")


@app.command("im-residual-hist")
def im_residual_hist(
    gnn_results_dir: Path,
    emp_cim_results_dir: Path,
    ims: List[str],
    nzgmdb_ffp: Path,
    output_dir: Path,
):
    """
    Visualises the residual distribution of the GNN
    and empirical cIM predictions per specified IM
    """
    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)

    (
        gnn_train_res_df,
        gnn_val_res_df,
        train_emp_cim_res_df,
        val_emp_cim_res_df,
        shared_train_sc_ids,
        shared_val_sc_ids,
    ) = __load_residuals(gnn_results_dir, emp_cim_results_dir, obs_data)

    _plot_res_hist(
        train_emp_cim_res_df.loc[shared_train_sc_ids],
        gnn_train_res_df.loc[shared_train_sc_ids],
        ims,
        50,
        output_dir,
        "train",
    )
    _plot_res_hist(
        val_emp_cim_res_df.loc[shared_val_sc_ids],
        gnn_val_res_df.loc[shared_val_sc_ids],
        ims,
        50,
        output_dir,
        "val",
    )


@app.command("event-site-map")
def event_site_map(
    event: str,
    site_int_ids: List[str],
    nzgmdb_ffp: Path,
    output_dir: Path,
    val_int_site_ids_ffp: Path = None,
    map_data_ffp: Path = None,
):
    """
    Create a map plot of the event and the site locations
    """
    from pygmt_helper import plotting

    min_lon, max_lon, min_lat, max_lat = (172.0, 173.12, -43.75, -43.35)

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)
    obs_sites = obs_data.record_df.loc[
        obs_data.record_df.event_id == event, "site_id"
    ].values.astype(str)
    event_data = obs_data.event_df.loc[event]

    # Don't use the validation sites
    val_int_sites = np.load(val_int_site_ids_ffp)
    obs_sites = obs_sites[~np.isin(obs_sites, val_int_sites)]

    # Load map data
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )
    fig = plotting.gen_region_fig(
        region=(min_lon, max_lon, min_lat, max_lat),
        map_data=map_data,
        plot_kwargs=dict(frame_args=["+n"]),
        # config_options=dict(
        #     MAP_FRAME_TYPE="plain",
        #     FORMAT_GEO_MAP="ddd.xx",
        #     MAP_FRAME_PEN="thinner,black",
        #     FONT_ANNOT_PRIMARY="6p,Helvetica,black",
        # ),
    )

    fig.meca(
        spec=dict(
            strike=event_data.strike,
            dip=event_data.dip,
            rake=event_data.rake,
            magnitude=event_data.mag,
        ),
        scale=f"{0.05 * event_data.mag}c",
        longitude=event_data.lon,
        latitude=event_data.lat,
        depth=event_data.depth,
        G="red",
        W="0.05p,black,solid",
    )

    # Plot the observation sites
    fig.plot(
        x=obs_data.site_df.loc[obs_sites, "lon"].values,
        y=obs_data.site_df.loc[obs_sites, "lat"].values,
        style="t0.25c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    # Plot the sites of interest
    fig.plot(
        x=obs_data.site_df.loc[site_int_ids, "lon"].values,
        y=obs_data.site_df.loc[site_int_ids, "lat"].values,
        style="a0.3c",
        fill="orange",
        pen="0.1p,black",
    )

    fig.savefig(
        output_dir / f"{event}_site_map.png",
        dpi=900,
        anti_alias=True,
    )


@app.command("pred-std-constraintness")
def pred_std_constraintness(
    gnn_resuls_dir: Path, nzmdb_ffp: Path, ims: List[str], output_dir: Path
):
    """
    Plots predicted standard deviation against
    the degree of constraint of the SOI
    """
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzmdb_ffp)
    obs_data.drop_nan()

    dist_matrix = sh.im_dist.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    lb_corr_data = sr.LBSiteCorrelationData.from_dist_matrix(dist_matrix, ims)

    gnn_train_results = pd.read_parquet(gnn_resuls_dir / "train_results.parquet")
    gnn_val_results = pd.read_parquet(gnn_resuls_dir / "val_results.parquet")

    # Compute the constraintness for each scenario
    gnn_train_constraint = {}
    for cur_ix, cur_row in gnn_train_results.iterrows():
        cur_corr_data = lb_corr_data.corr_data.sel[cur_row.site_int, :, :].loc[
            cur_row.obs_sites
        ]
        cur_constraint = cur_corr_data.sum(axis=0).mean()
        gnn_train_constraint[cur_ix] = cur_constraint
    gnn_train_constraint = pd.Series(gnn_train_constraint, name="constraint")

    gnn_val_constraint = {}
    for cur_ix, cur_row in gnn_val_results.iterrows():
        cur_corr_data = lb_corr_data.corr_data.sel[cur_row.site_int, :, :].loc[
            cur_row.obs_sites
        ]
        cur_constraint = cur_corr_data.sum(axis=0).mean()
        gnn_val_constraint[cur_ix] = cur_constraint
    gnn_val_constraint = pd.Series(gnn_val_constraint, name="constraint")

    # Generate the plots
    cur_x_min = min(gnn_train_constraint.min(), gnn_val_constraint.min())
    cur_x_max = max(gnn_train_constraint.max(), gnn_val_constraint.max())

    for cur_im in ims:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cur_x = gnn_train_constraint
        cur_y = gnn_train_results[f"{cur_im}_pred_std"]
        cur_bin_centres, cur_bin_means, cur_bin_stds = (
            mlt.utils.compute_count_binned_trend(
                cur_x.values, cur_y.values, n_points_per_bin=1000
            )
        )
        ax1.scatter(cur_x, cur_y, label="Training", alpha=0.25, s=10)
        ax1.plot(
            cur_bin_centres,
            cur_bin_means,
            c="k",
            label="Mean",
            linewidth=2.0,
        )
        ax1.plot(
            cur_bin_centres,
            cur_bin_means + cur_bin_stds,
            c="k",
            linestyle="--",
            label="Std",
            linewidth=2.0,
        )
        ax1.plot(cur_bin_centres, cur_bin_means - cur_bin_stds, c="k", linestyle="--",
                 linewidth=2.0)
        ax1.set_xlabel(f"Constraint Level of SOI")
        ax1.set_ylabel(f"Predicted Standard Deviation")
        ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlim(cur_x_min, cur_x_max)
        ax1.set_ylim(0.2, 1.5)
        ax1.set_xscale("log")
        ax1.set_title("Training Data")
        ax1.text(
            0.175,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax1.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        cur_x = gnn_val_constraint
        cur_y = gnn_val_results[f"{cur_im}_pred_std"]
        cur_bin_centres, cur_bin_means, cur_bin_stds = (
            mlt.utils.compute_count_binned_trend(
                cur_x.values, cur_y.values, n_points_per_bin=100
            )
        )
        ax2.scatter(cur_x, cur_y, label="Validation", alpha=0.25, s=10)
        ax2.plot(cur_bin_centres, cur_bin_means, c="k", label="Mean", linewidth=2.0)
        ax2.plot(
            cur_bin_centres,
            cur_bin_means + cur_bin_stds,
            c="k",
            linestyle="--",
            label="Std",
            linewidth=2.0,
        )
        ax2.plot(cur_bin_centres, cur_bin_means - cur_bin_stds, c="k", linestyle="--", linewidth=2.0)
        ax2.set_xlabel(f"Constraint Level of SOI")
        ax2.set_ylabel(f"Predicted Standard Deviation")
        ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        ax2.set_xlim(cur_x_min, cur_x_max)
        ax2.set_ylim(0.2, 1.5)
        ax2.set_xscale("log")
        ax2.set_title("Validation Data")
        ax2.text(
            0.175,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax2.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        fig.tight_layout()

        fig.savefig(output_dir / f"{cur_im}_pred_std_constraintness.png")

@app.command("residual-constraintness")
def residual_constraintness(gnn_resuls_dir: Path, nzmdb_ffp: Path, ims: List[str], output_dir: Path):
    """
    Visualises how the GNN residual changes
    the constraint level of the SOI
    One plot per IM
    """
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzmdb_ffp)
    obs_data.drop_nan()

    dist_matrix = sh.im_dist.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    lb_corr_data = sr.LBSiteCorrelationData.from_dist_matrix(dist_matrix, ims)

    gnn_train_results = pd.read_parquet(gnn_resuls_dir / "train_results.parquet")
    gnn_val_results = pd.read_parquet(gnn_resuls_dir / "val_results.parquet")

    # Compute the constraintness for each scenario
    gnn_train_constraint = {}
    for cur_ix, cur_row in gnn_train_results.iterrows():
        cur_corr_data = lb_corr_data.corr_data.sel[cur_row.site_int, :, :].loc[
            cur_row.obs_sites
        ]
        cur_constraint = cur_corr_data.sum(axis=0).mean()
        gnn_train_constraint[cur_ix] = cur_constraint
    gnn_train_constraint = pd.Series(gnn_train_constraint, name="constraint")

    gnn_val_constraint = {}
    for cur_ix, cur_row in gnn_val_results.iterrows():
        cur_corr_data = lb_corr_data.corr_data.sel[cur_row.site_int, :, :].loc[
            cur_row.obs_sites
        ]
        cur_constraint = cur_corr_data.sum(axis=0).mean()
        gnn_val_constraint[cur_ix] = cur_constraint
    gnn_val_constraint = pd.Series(gnn_val_constraint, name="constraint")

    # Compute the residuals
    gnn_train_res_df = sr.analysis.get_residuals(gnn_train_results)
    gnn_val_res_df = sr.analysis.get_residuals(gnn_val_results)

    # Generate the plot
    cur_x_min = min(gnn_train_constraint.min(), gnn_val_constraint.min())
    cur_x_max = max(gnn_train_constraint.max(), gnn_val_constraint.max())

    for cur_im in ims:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cur_x = gnn_train_constraint
        cur_y = gnn_train_res_df[cur_im]
        cur_bin_centres, cur_bin_means, cur_bin_stds = mlt.utils.compute_count_binned_trend(
            cur_x.values, cur_y.values, n_points_per_bin=1000)
        ax1.scatter(cur_x, cur_y, label="Training", alpha=0.25, zorder=0, s=10)
        ax1.plot(cur_bin_centres, cur_bin_means, c="k", label="Mean", linewidth=2.0)
        ax1.plot(cur_bin_centres, cur_bin_means + cur_bin_stds, c="k", linestyle="--",
                 label="Std", linewidth=2.0)
        ax1.plot(cur_bin_centres, cur_bin_means - cur_bin_stds, c="k", linestyle="--", linewidth=2.0)
        ax1.set_xlabel(f"Constraint Level of SOI")
        ax1.set_ylabel(f"Residual")
        ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax1.set_xlim(cur_x_min, cur_x_max)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_xscale("log")
        ax1.set_title("Training Data")
        ax1.text(
            0.175,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax1.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        cur_x = gnn_val_constraint
        cur_y = gnn_val_res_df[cur_im]
        cur_bin_centres, cur_bin_means, cur_bin_stds = mlt.utils.compute_count_binned_trend(
            cur_x.values, cur_y.values, n_points_per_bin=100)
        ax2.scatter(cur_x, cur_y, label="Validation", alpha=0.25, s=10)
        ax2.plot(cur_bin_centres, cur_bin_means, c="k", label="Mean", linewidth=2.0)
        ax2.plot(cur_bin_centres, cur_bin_means + cur_bin_stds, c="k", linestyle="--",
                 label="Std", linewidth=2.0)
        ax2.plot(cur_bin_centres, cur_bin_means - cur_bin_stds, c="k", linestyle="--", linewidth=2.0)
        ax2.set_xlabel(f"Constraint Level of SOI")
        ax2.set_ylabel(f"Residual")
        ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax2.set_xlim(cur_x_min, cur_x_max)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xscale("log")
        ax2.set_title("Validation Data")
        ax2.text(
            0.175,
            0.98,
            f"{sr.utils.get_nice_im_name(cur_im)}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax2.transAxes,
            fontsize="medium",
            fontweight="bold",
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"{cur_im}_residual_constraintness.png")


if __name__ == "__main__":
    app()
