from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import typer
from tqdm import tqdm
import seaborn as sns

import sim_ranking as sr
import ml_tools as mlt


app = typer.Typer()


@app.command("mag-rrup-scatter")
def mag_rrup_scatter(
    nzgmdb_ffp: Path, output_dir: Path, n_rrup_bins: int = 20, n_mag_bins: int = 20
):
    """
    Creates a scatter of rrup vs magnitude
    with the marginal distributions on the sides
    """
    print("Using figure size: ", sr.constants.FIG_SIZE)
    print("Using figure format: ", sr.constants.FIG_FORMAT)

    nzgmdb_df = pd.read_csv(
        nzgmdb_ffp,
        dtype={"evid": str, "loc": str},
        engine="c",
        index_col="record_id",
    ).sort_index()

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)

    _, __, valid_record_ids = sr.ml.data.get_valid_site_ints_Lee2024(
        obs_data.event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )

    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        layout="constrained",
        figsize=sr.constants.FIG_SIZE,
    )
    ax_scatter = axs["scatter"]
    ax_histx = axs["histx"]
    ax_histy = axs["histy"]

    # Scatter plot
    ax_scatter.scatter(
        nzgmdb_df["r_rup"],
        nzgmdb_df["mag"],
        s=2.5,
        c="grey",
        alpha=0.5,
        label=f"All records - N: {nzgmdb_df.shape[0]}",
    )
    ax_scatter.scatter(
        obs_data.record_df.loc[valid_record_ids, "rrup"],
        obs_data.record_df.loc[valid_record_ids, "mag"],
        s=2.5,
        c="red",
        alpha=0.5,
        label=f"Filtered records - N: {valid_record_ids.shape[0]}",
    )
    ax_scatter.plot(
        sr.constants.MW_RRUP_LIMITS[:, 1],
        sr.constants.MW_RRUP_LIMITS[:, 0],
        c="blue",
        label="Magnitude-distance filter",
    )

    ax_scatter.set_xlabel("Source-to-site distance, $R_{Rup}$ (km)")
    ax_scatter.set_ylabel("Magnitude, $M_{W}$")
    ax_scatter.legend()
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlim(0.1, 1000)
    ax_scatter.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax_histx.hist(
        obs_data.record_df.loc[valid_record_ids, "rrup"],
        bins=np.logspace(np.log10(0.1), np.log10(1000), n_rrup_bins),
        color="red",
    )
    ax_histx.set_xscale("log")
    ax_histx.set_xlim(0.1, 1000)
    ax_histx.spines[["top", "right"]].set_visible(False)
    ax_histx.set_xticklabels([])

    ax_histy.hist(
        obs_data.record_df.loc[valid_record_ids, "mag"],
        bins=n_mag_bins,
        color="red",
        orientation="horizontal",
    )
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.spines[["top", "right"]].set_visible(False)
    ax_histy.set_yticklabels([])

    plt.savefig(output_dir / f"rrup_vs_mag.{sr.constants.FIG_FORMAT}")


@app.command("sample-weighting")
def sample_weighting(
    nzgmdb_ffp: Path,
    output_dir: Path,
    max_dist: float,
    closest_max_dist: float,
    max_n_obs_sites: int,
    min_n_obs_sites: int,
    mag_n_bins: int = 20,
    rrup_n_bins: int = 20,
    mag_max_weight: float = 4.0,
    mag_start: float = 4.5,
    mag_end: float = 6.0,
    doc_max_weight: float = 1.0,
    doc_start: float = 1.0,
    doc_end: float = 6.0,
):
    # Load observed data
    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    # Get the set of valid site-interests per event
    print("Getting valid sites of interest")
    int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints_Lee2024(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))

    # Distance matrix
    dist_matrix = sr.utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    # Loth & Baker spatial correlations
    corr_data = sr.LBSiteCorrelationData.from_dist_matrix(
        dist_matrix, sr.constants.PSA_KEYS
    )

    # Compute available scenarios
    obs_sites = all_sites
    site_combs, event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        events,
        dist_matrix,
        obs_sites,
        int_sites,
        max_dist,
        closest_max_dist,
        max_n_obs_sites,
        min_n_obs_sites,
    )
    # Create scenario dataframe
    scenario_df = sr.ml.utils.create_scenario_df(
        site_combs,
        event_sites,
        obs_data,
        dist_matrix=dist_matrix,
        lb_corr_data=corr_data,
    )
    print(f"Number of scenarios: {len(scenario_df)}")

    ### Magnitude
    fig, ax = plt.subplots(1, 1, figsize=sr.constants.FIG_SIZE)
    sns.histplot(scenario_df["mag"], bins=mag_n_bins, ax=ax)
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # Weighting
    weight_func = sr.ml.gnn_gm.get_mag_weight_func(
        0.0, mag_max_weight, mag_start, mag_end
    )
    mag_values = np.linspace(scenario_df.mag.min(), scenario_df.mag.max(), 100)
    weights = np.asarray([weight_func(cur_mag) for cur_mag in mag_values])
    ax_weight = ax.twinx()
    ax_weight.plot(mag_values, weights, color="red", linestyle="--")
    ax_weight.set_ylim(0.0, None)

    ax.set_title("Magnitude Distribution (Original) + Weighting Function")
    fig.tight_layout()

    plt.savefig(output_dir / "mag_weighting.png")
    plt.close(fig)

    ### RRUP
    fig, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(scenario_df["rrup"], bins=rrup_n_bins, ax=ax_hist)
    ax_hist.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_hist.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax_hist.set_xlim(0.0)
    fig.tight_layout()

    plt.savefig(output_dir / "rrup_hist.png")
    plt.close(fig)

    ### Degree of constraint
    # Distribution
    fig, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(scenario_df["constraintness"], bins=20, ax=ax_hist)
    ax_hist.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_hist.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax_hist.set_xlim(0.0)

    # Weighting
    doc_weight_fn = sr.ml.gnn_gm.get_doc_weight_func(
        0.0, doc_max_weight, doc_start, doc_end
    )
    doc_values = np.linspace(
        scenario_df.constraintness.min(), scenario_df.constraintness.max(), 100
    )
    weights = [doc_weight_fn(cur_doc) for cur_doc in doc_values]

    ax_weight = ax_hist.twinx()
    ax_weight.plot(doc_values, weights, color="red", linestyle="--")
    ax_weight.set_ylim(0.0, None)
    ax_hist.set_title(
        "Degree of Constraint Distribution (Original) + Weighting Function"
    )

    fig.tight_layout()
    plt.savefig(output_dir / f"doc_weighting.{sr.constants.FIG_FORMAT}")


@app.command("bias-res-std")
def bias_res_std(
    gnn_only_result_dir: Path,
    gnn_residual_result_dir: Path,
    cim_result_dir: Path,
    emp_gm_params_ffp: Path,
    output_dir: Path,
):
    """
    Generates a figure comparing the bias and residual standard deviation
    of the GNN-Only, GNN-Residual, marginal and cIM models.
    """
    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    gnn_only_run_config = sr.ml.RunConfig.from_yaml(
        gnn_only_result_dir / "run_config.yaml"
    )
    gnn_residual_run_config = sr.ml.RunConfig.from_yaml(
        gnn_residual_result_dir / "run_config.yaml"
    )

    assert gnn_only_run_config.obs_data_ffp == gnn_residual_run_config.obs_data_ffp
    obs_data = sr.data.load_obs_nzgmdb(gnn_only_run_config.obs_data_ffp)

    # Load GNN validation results
    gnn_only_results = pd.read_parquet(
        gnn_only_result_dir / "val_results.parquet"
    ).sort_index()
    gnn_res_results = pd.read_parquet(
        gnn_residual_result_dir / "val_results.parquet"
    ).sort_index()

    # Load cIM results
    cim_results = pd.read_parquet(cim_result_dir / "val_results.parquet").sort_index()

    # Load marginal GMM residuals
    _, emp_res_df = sr.ml.gnn_gm.load_emp_gm_params_res(emp_gm_params_ffp, obs_data)
    emp_res_df = emp_res_df.loc[gnn_only_results.index]

    # Sanity check
    assert (
        emp_res_df.index.equals(gnn_only_results.index)
        and gnn_res_results.index.equals(gnn_only_results.index)
        and cim_results.index.equals(gnn_only_results.index)
    ), "Index mixmatch"

    # Compute residuals
    gnn_only_res_df = sr.ml.gnn_gm.get_residuals(
        gnn_only_results, ims=gnn_only_run_config.ims
    )
    gnn_only_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(
        gnn_only_res_df, ims=gnn_only_run_config.ims
    )
    cim_res_df = sr.ml.gnn_gm.get_residuals(cim_results, pred_suffix="cond_mean")

    gnn_res_res_df = sr.ml.gnn_gm.get_residuals(gnn_res_results)
    gnn_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(gnn_res_res_df)
    cim_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(cim_res_df)
    marg_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(emp_res_df)

    fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        left=0.08,
        main_wspace=0.175,
        sub_wspace=0.05,
        right=0.99,
        std_y_axis_limits=(0.0, 1.25),
        bottom=0.125,
    )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        marg_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="grey",
        linewidth=2.5,
        label="MVN-Marginal",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="green",
        linewidth=2.5,
        label="MVN-CIM",
    )
    ax1.plot(
        sr.constants.PERIODS,
        gnn_only_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="blue",
        linewidth=2.5,
        label="GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="purple",
        linewidth=2.5,
        label="GNN-Residual",
    )

    ax1.legend(fontsize=sr.constants.FIG_FONT_SIZE)

    ax1.text(
        0.03,
        0.03,
        "Overprediction",
        transform=ax1.transAxes,
        fontsize=sr.constants.FIG_FONT_SIZE,
        va="bottom",
        ha="left",
    )
    ax1.text(
        0.03,
        0.97,
        "Underprediction",
        transform=ax1.transAxes,
        fontsize=sr.constants.FIG_FONT_SIZE,
        va="top",
        ha="left",
    )

    if gnn_only_run_config.non_pSA_ims is not None:
        ax2.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_only_run_config.non_pSA_ims
            ],
            gnn_only_bias_std_df.loc[gnn_only_run_config.non_pSA_ims, "mean"],
            color="blue",
            zorder=10,
        )
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax2.set_xlim(-0.75, len(gnn_only_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        marg_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="grey",
        linewidth=2.5,
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="green",
        linewidth=2.5,
    )
    ax3.plot(
        sr.constants.PERIODS,
        gnn_only_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="blue",
        linewidth=2.5,
    )
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df["std"],
        color="purple",
        linewidth=2.5,
    )
    if gnn_only_run_config.non_pSA_ims is not None:
        ax4.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_only_run_config.non_pSA_ims
            ],
            gnn_only_bias_std_df.loc[gnn_only_run_config.non_pSA_ims, "std"],
            color="blue",
        )
        ax4.set_xticklabels(
            ax4.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax4.set_xlim(-0.75, len(gnn_only_run_config.non_pSA_ims) - 0.25)

    plt.savefig(output_dir / f"bias_residual_std.{sr.constants.FIG_FORMAT}")


@app.command("mag-bias-res-std")
def mag_bias_res_std(
    gnn_results_dir: Path,
    cim_result_dir: Path,
    output_dir: Path,
    plot_labels: str = None,
    show_legend: bool = True,
    legend_ax: int = 1,
    output_name: str = None,
):
    """
    Generates a figure comparing the bias and residual standard deviation
    for different magnitude bins

    Parameters
    ----------
    gnn_results_dir : Path
        Directory containing the GNN results
    cim_result_dir : Path
        Directory containing the cIM results
    output_dir : Path
        Directory to save the figure
    plot_labels : str
        Labels for the plot
        (comma separated)
    show_legend : bool
        Whether to show the legend
    legend_ax : int
        Axis to show the legend on
    output_name : str
        Name of the output file
    """
    plot_labels = plot_labels.split(",") if plot_labels is not None else None

    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    gnn_run_config = sr.ml.RunConfig.from_yaml(gnn_results_dir / "run_config.yaml")

    # Load observed data
    obs_data = sr.data.load_obs_nzgmdb(gnn_run_config.obs_data_ffp)

    # Load results
    gnn_results = pd.read_parquet(gnn_results_dir / "val_results.parquet").sort_index()
    cim_results = pd.read_parquet(cim_result_dir / "val_results.parquet").sort_index()

    # Sanity check
    assert gnn_results.index.equals(cim_results.index), "Index mixmatch"

    # Compute residuals
    gnn_res_df = sr.ml.gnn_gm.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )
    cim_res_df = sr.ml.gnn_gm.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(cim_res_df)

    # Apply magnitude binning
    gnn_res_df["mag"] = obs_data.record_df.loc[gnn_res_df.index, "mag"].values
    gnn_res_df["mag_bin"] = pd.cut(
        gnn_res_df["mag"],
        bins=sr.constants.MAG_BINS,
        labels=sr.constants.MAG_BIN_LABELS,
    )
    gnn_res_mag_groups = gnn_res_df.groupby("mag_bin")
    gnn_res_mag_bias = gnn_res_mag_groups[gnn_run_config.ims].mean()
    gnn_res_mag_std = gnn_res_mag_groups[gnn_run_config.ims].std()

    cim_res_df["mag"] = obs_data.record_df.loc[cim_res_df.index, "mag"].values
    cim_res_df["mag_bin"] = pd.cut(
        cim_res_df["mag"],
        bins=sr.constants.MAG_BINS,
        labels=sr.constants.MAG_BIN_LABELS,
    )
    cim_res_mag_groups = cim_res_df.groupby("mag_bin")
    cim_res_mag_bias = cim_res_mag_groups[sr.constants.PSA_KEYS].mean()
    cim_res_mag_std = cim_res_mag_groups[sr.constants.PSA_KEYS].std()

    # colors = list(reversed(cm.viridis(np.linspace(0, 1, len(sr.constants.MAG_BINS)))))
    group_colors = sr.constants.MAG_COLORS
    group_linewidth = 2.0
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_mag_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            cim_res_mag_bias.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_res_mag_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"{cur_key}, N: {gnn_res_mag_groups.size()[cur_key]}",
            c=group_colors[i],
            linewidth=group_linewidth,
        )
    if show_legend and legend_ax == 1:
        ax1.legend()

    ax1.text(
        0.03,
        0.03,
        "Overprediction",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
    )
    ax1.text(
        0.03,
        0.97,
        "Underprediction",
        transform=ax1.transAxes,
        va="top",
        ha="left",
    )

    if plot_labels is not None:
        ax1.text(
            -0.2 if gnn_run_config.non_pSA_ims is not None else -0.175,
            0.98,
            plot_labels[0],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax2.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "mean"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_mag_groups):
            ax2.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_mag_bias.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )

        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_mag_groups):
        ax3.semilogx(
            sr.constants.PERIODS,
            cim_res_mag_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax3.semilogx(
            sr.constants.PERIODS,
            gnn_res_mag_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linewidth=group_linewidth,
            label=f"{cur_key}, N: {gnn_res_mag_groups.size()[cur_key]}",
        )

    if show_legend and legend_ax == 3:
        ax3.legend()

    if plot_labels is not None:
        ax3.text(
            -0.175 if gnn_run_config.non_pSA_ims is not None else -0.15,
            0.98,
            plot_labels[1],
            transform=ax3.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax4.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "std"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_mag_groups):
            ax4.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_mag_std.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )

        ax4.set_xticklabels(
            ax4.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax4.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    output_name = (
        f"{output_name}.{sr.constants.FIG_FORMAT}"
        if output_name is not None
        else f"mag_bias_residual_std.{sr.constants.FIG_FORMAT}"
    )
    fig.savefig(
        output_dir / output_name,
    )


@app.command("rrup-bias-res-std")
def rrup_bias_res_std(
    gnn_results_dir: Path,
    cim_result_dir: Path,
    output_dir: Path,
    plot_labels: str = None,
    show_legend: bool = True,
    legend_ax: int = 1,
    output_name: str = None,
):
    """
    Generates a figure comparing the bias and residual standard deviation
    for different rrup bins

    Parameters
    ----------
    gnn_results_dir : Path
        Directory containing the GNN results
    cim_result_dir : Path
        Directory containing the cIM results
    output_dir : Path
        Directory to save the figure
    plot_labels : str
        Labels for the plot
        (comma separated)
    show_legend : bool
        Whether to show the legend
    legend_ax : int
        Axis to show the legend on
    output_name : str
        Name of the output file
    """
    plot_labels = plot_labels.split(",") if plot_labels is not None else None

    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    gnn_run_config = sr.ml.RunConfig.from_yaml(gnn_results_dir / "run_config.yaml")

    # Load observed data
    obs_data = sr.data.load_obs_nzgmdb(gnn_run_config.obs_data_ffp)

    # Load results
    gnn_results = pd.read_parquet(gnn_results_dir / "val_results.parquet").sort_index()
    cim_results = pd.read_parquet(cim_result_dir / "val_results.parquet").sort_index()

    # Sanity check
    assert gnn_results.index.equals(cim_results.index), "Index mixmatch"

    # Compute residuals
    gnn_res_df = sr.ml.gnn_gm.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )

    cim_res_df = sr.ml.gnn_gm.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(cim_res_df)

    # Apply rrup binning
    gnn_res_df["rrup"] = obs_data.record_df.loc[gnn_res_df.index, "rrup"].values
    gnn_res_df["rrup_bin"] = pd.cut(
        gnn_res_df["rrup"],
        bins=sr.constants.RRUP_BINS,
        labels=sr.constants.RRUP_BIN_LABELS,
    )

    gnn_res_rrup_groups = gnn_res_df.groupby("rrup_bin")
    gnn_res_rrup_bias = gnn_res_rrup_groups[gnn_run_config.ims].mean()
    gnn_res_rrup_std = gnn_res_rrup_groups[gnn_run_config.ims].std()

    cim_res_df["rrup"] = obs_data.record_df.loc[cim_res_df.index, "rrup"].values
    cim_res_df["rrup_bin"] = pd.cut(
        cim_res_df["rrup"],
        bins=sr.constants.RRUP_BINS,
        labels=sr.constants.RRUP_BIN_LABELS,
    )

    cim_res_rrup_groups = cim_res_df.groupby("rrup_bin")
    cim_res_rrup_bias = cim_res_rrup_groups[sr.constants.PSA_KEYS].mean()
    cim_res_rrup_std = cim_res_rrup_groups[sr.constants.PSA_KEYS].std()

    group_colors = sr.constants.RRUP_COLORS
    group_linewidth = 2.0
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_rrup_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            cim_res_rrup_bias.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_res_rrup_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"{cur_key}, N: {gnn_res_rrup_groups.size()[cur_key]}",
            c=group_colors[i],
            linewidth=group_linewidth,
        )
    if show_legend and legend_ax == 1:
        ax1.legend()

    ax1.text(
        0.03,
        0.03,
        "Overprediction",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
    )
    ax1.text(
        0.03,
        0.03,
        "Overprediction",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
    )

    if plot_labels is not None:
        ax1.text(
            -0.2 if gnn_run_config.non_pSA_ims is not None else -0.175,
            0.98,
            plot_labels[0],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax2.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "mean"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_rrup_groups):
            ax2.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_rrup_bias.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )

        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_rrup_groups):
        ax3.semilogx(
            sr.constants.PERIODS,
            cim_res_rrup_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax3.semilogx(
            sr.constants.PERIODS,
            gnn_res_rrup_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linewidth=group_linewidth,
            label=f"{cur_key}, N: {gnn_res_rrup_groups.size()[cur_key]}",
        )
    if show_legend and legend_ax == 3:
        ax3.legend()

    if plot_labels is not None:
        ax3.text(
            -0.175 if gnn_run_config.non_pSA_ims is not None else -0.15,
            0.98,
            plot_labels[1],
            transform=ax3.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax4.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "std"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_rrup_groups):
            ax4.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_rrup_std.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )
        ax4.set_xticklabels(
            ax4.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax4.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    output_name = (
        f"{output_name}.{sr.constants.FIG_FORMAT}"
        if output_name is not None
        else f"rrup_bias_residual_std.{sr.constants.FIG_FORMAT}"
    )
    fig.savefig(
        output_dir / output_name,
    )


@app.command("doc-bias-res-std")
def doc_bias_res_std(
    gnn_results_dir: Path,
    cim_result_dir: Path,
    output_dir: Path,
    plot_labels: str = None,
    show_legend: bool = True,
    legend_ax: int = 1,
    output_name: str = None,
):
    """
    Generates a figure comparing the bias and residual standard deviation
    for different degree of constraint bins

    Parameters
    ----------
    gnn_results_dir : Path
        Directory containing the GNN results
    cim_result_dir : Path
        Directory containing the cIM results
    output_dir : Path
        Directory to save the figure
    plot_labels : str
        Labels for the plot
        (comma separated)
    show_legend : bool
        Whether to show the legend
    legend_ax : int
        Axis to show the legend on
    output_name : str
        Name of the output file
    """
    plot_labels = plot_labels.split(",") if plot_labels is not None else None

    # Update font size
    plt.rcParams.update(
        {
            "font.size": sr.constants.FIG_FONT_SIZE,
        }
    )

    gnn_run_config = sr.ml.RunConfig.from_yaml(gnn_results_dir / "run_config.yaml")

    # Load observed data
    obs_data = sr.data.load_obs_nzgmdb(gnn_run_config.obs_data_ffp)

    # Load results
    gnn_results = pd.read_parquet(gnn_results_dir / "val_results.parquet").sort_index()
    cim_results = pd.read_parquet(cim_result_dir / "val_results.parquet").sort_index()

    # Add DoC
    dist_matrix = sr.utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    corr_data = sr.LBSiteCorrelationData.from_dist_matrix(
        dist_matrix, gnn_run_config.pSA_ims
    )
    gnn_results = sr.utils.compute_degree_of_constraint(gnn_results, corr_data)

    # Sanity check
    assert gnn_results.index.equals(cim_results.index), "Index mixmatch"

    # Compute residuals
    gnn_res_df = sr.ml.gnn_gm.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )

    cim_res_df = sr.ml.gnn_gm.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.ml.gnn_gm.get_res_mean_std(cim_res_df)

    # Apply degree of constraint binning
    gnn_res_df["doc"] = gnn_results.loc[gnn_res_df.index, "doc"].values
    gnn_res_df["doc_bin"] = pd.cut(
        gnn_res_df["doc"],
        bins=sr.constants.DOC_BINS,
        labels=sr.constants.DOC_BIN_LABELS,
    )

    gnn_res_doc_groups = gnn_res_df.groupby("doc_bin")
    gnn_res_doc_bias = gnn_res_doc_groups[gnn_run_config.ims].mean()
    gnn_res_doc_std = gnn_res_doc_groups[gnn_run_config.ims].std()

    cim_res_df["doc"] = gnn_results.loc[cim_res_df.index, "doc"].values
    cim_res_df["doc_bin"] = pd.cut(
        cim_res_df["doc"],
        bins=sr.constants.DOC_BINS,
        labels=sr.constants.DOC_BIN_LABELS,
    )

    cim_res_doc_groups = cim_res_df.groupby("doc_bin")
    cim_res_doc_bias = cim_res_doc_groups[sr.constants.PSA_KEYS].mean()
    cim_res_doc_std = cim_res_doc_groups[sr.constants.PSA_KEYS].std()

    group_colors = sr.constants.DOC_COLORS
    group_linewidth = 2.0
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            std_y_axis_limits=(0.0, 1.25),
            bottom=0.125,
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_doc_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            cim_res_doc_bias.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_res_doc_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"{cur_key}, N: {gnn_res_doc_groups.size()[cur_key]}",
            c=group_colors[i],
            linewidth=group_linewidth,
        )
    if show_legend and legend_ax == 1:
        ax1.legend()

    ax1.text(
        0.03, 0.03, "Overprediction", transform=ax1.transAxes, va="bottom", ha="left"
    )
    ax1.text(
        0.03, 0.97, "Underprediction", transform=ax1.transAxes, va="top", ha="left"
    )

    if plot_labels is not None:
        ax1.text(
            -0.2 if gnn_run_config.non_pSA_ims is not None else -0.175,
            0.98,
            plot_labels[0],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax2.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "mean"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_doc_groups):
            ax2.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_doc_bias.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )

        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=2.5,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_res_doc_groups):
        ax3.semilogx(
            sr.constants.PERIODS,
            cim_res_doc_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linestyle="--",
            linewidth=group_linewidth,
        )
        ax3.semilogx(
            sr.constants.PERIODS,
            gnn_res_doc_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=group_colors[i],
            linewidth=group_linewidth,
            label=f"{cur_key}, N: {gnn_res_doc_groups.size()[cur_key]}",
        )
    if show_legend and legend_ax == 3:
        ax3.legend()

    if plot_labels is not None:
        ax3.text(
            -0.175 if gnn_run_config.non_pSA_ims is not None else -0.15,
            0.98,
            plot_labels[1],
            transform=ax3.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    if gnn_run_config.non_pSA_ims is not None:
        ax4.scatter(
            [
                sr.utils.get_nice_im_name(cur_im, use_latex=True)
                for cur_im in gnn_run_config.non_pSA_ims
            ],
            gnn_res_bias_std_df.loc[gnn_run_config.non_pSA_ims, "std"],
            color="black",
            s=17.5,
        )
        for i, (cur_key, cur_group) in enumerate(gnn_res_doc_groups):
            ax4.scatter(
                [
                    sr.utils.get_nice_im_name(cur_im, use_latex=True)
                    for cur_im in gnn_run_config.non_pSA_ims
                ],
                gnn_res_doc_std.loc[cur_key, gnn_run_config.non_pSA_ims],
                c=group_colors[i],
                s=group_scatter_size,
            )
        ax4.set_xticklabels(
            ax4.get_xticklabels(), rotation=90, fontsize=sr.constants.FIG_FONT_SIZE
        )
        ax4.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    output_name = (
        f"{output_name}.{sr.constants.FIG_FORMAT}"
        if output_name is not None
        else f"doc_bias_residual_std.{sr.constants.FIG_FORMAT}"
    )
    fig.savefig(
        output_dir / output_name,
    )


@app.command("fmin-filter")
def fmin_filter(nzgmdb_ffp: Path, gnn_result_dir: Path, output_dir: Path):
    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)
    record_count_df = (~obs_data.record_df[sr.constants.PSA_KEYS].isna()).sum(axis=0)

    gnn_results = pd.read_parquet(gnn_result_dir / "val_results.parquet")
    scenario_count = (~gnn_results[sr.constants.PSA_KEYS].isna()).sum(axis=0)

    fig, ax = plt.subplots(figsize=sr.constants.FIG_SIZE)

    ax.plot(
        sr.constants.PERIODS,
        record_count_df.loc[sr.constants.PSA_KEYS],
        color="black",
        linewidth=2.5,
        label="Number of records",
    )
    ax.plot(
        sr.constants.PERIODS,
        scenario_count.loc[sr.constants.PSA_KEYS],
        color="blue",
        linewidth=2.5,
        label="Number of scenarios",
    )
    ax.set_xlabel("Period (s)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    ax.set_xlim(0.01, 10.0)
    ax.legend()
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()

    fig.savefig(output_dir / f"fmin_filter.{sr.constants.FIG_FORMAT}")
    plt.close(fig)


if __name__ == "__main__":
    app()
