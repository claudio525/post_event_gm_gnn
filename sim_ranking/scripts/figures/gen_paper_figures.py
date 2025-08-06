from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import typer
import seaborn as sns
from tqdm import tqdm

import ml_tools as mlt
import sim_ranking as sr


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
    print("Using figure dpi: ", sr.constants.FIG_DPI)

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)
    obs_data = obs_data.to_event_site_index()

    # # Ignore tectonic types other than 
    # # crustal, subduction interface, and subduction slab
    # records_to_keep = obs_data.record_df[
    #     obs_data.record_df["tect_type"].isin(
    #         [
    #             sr.constants.TectonicType.CRUSTAL,
    #             sr.constants.TectonicType.SUBDUCTION_INTERFACE,
    #             sr.constants.TectonicType.SUBDUCTION_SLAB,
    #         ]
    #     )
    # ].index.values.astype(str)
    # obs_data = obs_data.filter_record_ids(records_to_keep)

    # Load basic filtered data
    filtered_obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    # Apply Lee magnitude-distance filter
    _, __, valid_record_ids = sr.ml.data.get_valid_site_ints_Lee2024(
    filtered_obs_data.event_sites, filtered_obs_data.record_df.drop(columns=filtered_obs_data.ims)
    )
    filtered_obs_data = filtered_obs_data.filter_record_ids(valid_record_ids)

    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        layout="constrained",
        figsize=sr.constants.FIG_SIZE,
        dpi=sr.constants.FIG_DPI,
    )
    ax_scatter = axs["scatter"]
    ax_histx = axs["histx"]
    ax_histy = axs["histy"]

    # Scatter plot
    ax_scatter.scatter(
        obs_data.record_df["rrup"],
        obs_data.record_df["mag"],
        s=2.5,
        c="grey",
        alpha=0.5,
        label=f"All (N={obs_data.n_records:,})",
    )
    ax_scatter.scatter(
        filtered_obs_data.record_df["rrup"],
        filtered_obs_data.record_df["mag"],
        s=2.5,
        c="red",
        alpha=0.5,
        label=f"Filtered (N={filtered_obs_data.n_records:,})",
    )
    ax_scatter.plot(
        sr.constants.MW_RRUP_LIMITS[:, 1],
        sr.constants.MW_RRUP_LIMITS[:, 0],
        c="blue",
        label="Magnitude-distance filter",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )

    ax_scatter.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax_scatter.set_ylabel("Magnitude, $M_{W}$")
    ax_scatter.legend()
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlim(0.1, 1000)
    ax_scatter.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax_histx.hist(
        filtered_obs_data.record_df["rrup"],
        bins=np.logspace(np.log10(0.1), np.log10(1000), n_rrup_bins),
        color="red",
    )
    ax_histx.set_xscale("log")
    ax_histx.set_xlim(0.1, 1000)
    ax_histx.spines[["top", "right"]].set_visible(False)
    ax_histx.set_xticklabels([])

    ax_histy.hist(
        filtered_obs_data.record_df["mag"],
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
    bias_limit: float = 1.0,
    std_limit: float = 1.0,
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
    _, emp_res_df = sr.analysis.load_emp_gm_params_res(emp_gm_params_ffp, obs_data)
    emp_res_df = emp_res_df.loc[gnn_only_results.index]

    # Sanity check
    assert (
        emp_res_df.index.equals(gnn_only_results.index)
        and gnn_res_results.index.equals(gnn_only_results.index)
        and cim_results.index.equals(gnn_only_results.index)
    ), "Index mixmatch"

    # Compute residuals
    gnn_only_res_df = sr.analysis.get_residuals(
        gnn_only_results, ims=gnn_only_run_config.ims
    )
    gnn_only_bias_std_df = sr.analysis.get_res_mean_std(
        gnn_only_res_df, ims=gnn_only_run_config.ims
    )
    cim_res_df = sr.analysis.get_residuals(cim_results, pred_suffix="cond_mean")

    gnn_res_res_df = sr.analysis.get_residuals(gnn_res_results)
    gnn_res_bias_std_df = sr.analysis.get_res_mean_std(gnn_res_res_df)
    cim_res_bias_std_df = sr.analysis.get_res_mean_std(cim_res_df)
    marg_res_bias_std_df = sr.analysis.get_res_mean_std(emp_res_df)

    fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        left=0.08,
        main_wspace=0.175,
        sub_wspace=0.05,
        right=0.99,
        std_y_axis_limits=(0.0, std_limit),
        bias_y_axis_limits=(-bias_limit, bias_limit),
        bottom=0.125,
    )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        marg_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="grey",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="MVN-Marginal",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="green",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="MVN-CIM",
    )
    ax1.plot(
        sr.constants.PERIODS,
        gnn_only_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="blue",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="purple",
        linewidth=sr.constants.FIG_LINEWIDTH,
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

        ax2.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
        ax2.set_xlim(-0.75, len(gnn_only_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        marg_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="grey",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="green",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax3.plot(
        sr.constants.PERIODS,
        gnn_only_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="blue",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df["std"],
        color="purple",
        linewidth=sr.constants.FIG_LINEWIDTH,
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

        ax4.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
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
    bias_limit: float = 1.0,
    std_limit: float = 1.0,
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
    gnn_res_df = sr.analysis.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.analysis.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )
    cim_res_df = sr.analysis.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.analysis.get_res_mean_std(cim_res_df)

    # Apply magnitude binning
    gnn_res_df["mag"] = obs_data.record_df.loc[gnn_res_df.index, "mag"].values
    gnn_res_df["mag_bin"] = pd.cut(
        gnn_res_df["mag"],
        bins=sr.constants.MAG_BINS,
        labels=sr.constants.MAG_BIN_LABELS,
    )
    gnn_res_mag_groups = gnn_res_df.groupby("mag_bin", observed=True)
    gnn_res_mag_bias = gnn_res_mag_groups[gnn_run_config.ims].mean()
    gnn_res_mag_std = gnn_res_mag_groups[gnn_run_config.ims].std()

    cim_res_df["mag"] = obs_data.record_df.loc[cim_res_df.index, "mag"].values
    cim_res_df["mag_bin"] = pd.cut(
        cim_res_df["mag"],
        bins=sr.constants.MAG_BINS,
        labels=sr.constants.MAG_BIN_LABELS,
    )
    cim_res_mag_groups = cim_res_df.groupby("mag_bin", observed=True)
    cim_res_mag_bias = cim_res_mag_groups[sr.constants.PSA_KEYS].mean()
    cim_res_mag_std = cim_res_mag_groups[sr.constants.PSA_KEYS].std()

    # colors = list(reversed(cm.viridis(np.linspace(0, 1, len(sr.constants.MAG_BINS)))))
    group_colors = sr.constants.MAG_COLORS
    group_linewidth = sr.constants.FIG_GROUP_LINEWIDTH
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} (N={gnn_res_mag_groups.size()[cur_key]:,})",
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

        ax2.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} (N={gnn_res_mag_groups.size()[cur_key]:,})",
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

        ax4.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
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
    bias_limit: float = 1.0,
    std_limit: float = 1.0,
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
    gnn_res_df = sr.analysis.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.analysis.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )

    cim_res_df = sr.analysis.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.analysis.get_res_mean_std(cim_res_df)

    # Apply rrup binning
    gnn_res_df["rrup"] = obs_data.record_df.loc[gnn_res_df.index, "rrup"].values
    gnn_res_df["rrup_bin"] = pd.cut(
        gnn_res_df["rrup"],
        bins=sr.constants.RRUP_BINS,
        labels=sr.constants.RRUP_BIN_LABELS,
    )

    gnn_res_rrup_groups = gnn_res_df.groupby("rrup_bin", observed=True)
    gnn_res_rrup_bias = gnn_res_rrup_groups[gnn_run_config.ims].mean()
    gnn_res_rrup_std = gnn_res_rrup_groups[gnn_run_config.ims].std()

    cim_res_df["rrup"] = obs_data.record_df.loc[cim_res_df.index, "rrup"].values
    cim_res_df["rrup_bin"] = pd.cut(
        cim_res_df["rrup"],
        bins=sr.constants.RRUP_BINS,
        labels=sr.constants.RRUP_BIN_LABELS,
    )

    cim_res_rrup_groups = cim_res_df.groupby("rrup_bin", observed=True)
    cim_res_rrup_bias = cim_res_rrup_groups[sr.constants.PSA_KEYS].mean()
    cim_res_rrup_std = cim_res_rrup_groups[sr.constants.PSA_KEYS].std()

    group_colors = sr.constants.RRUP_COLORS
    group_linewidth = sr.constants.FIG_GROUP_LINEWIDTH
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} km (N={gnn_res_rrup_groups.size()[cur_key]:,})",
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

        ax2.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} (N={gnn_res_rrup_groups.size()[cur_key]:,})",
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

        ax4.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
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
    bias_limit: float = 1.0,
    std_limit: float = 1.0,
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
    gnn_res_df = sr.analysis.get_residuals(gnn_results, ims=gnn_run_config.ims)
    gnn_res_bias_std_df = sr.analysis.get_res_mean_std(
        gnn_res_df, ims=gnn_run_config.ims
    )

    cim_res_df = sr.analysis.get_residuals(cim_results, pred_suffix="cond_mean")
    cim_res_bias_std_df = sr.analysis.get_res_mean_std(cim_res_df)

    # Apply degree of constraint binning
    gnn_res_df["doc"] = gnn_results.loc[gnn_res_df.index, "doc"].values
    gnn_res_df["doc_bin"] = pd.cut(
        gnn_res_df["doc"],
        bins=sr.constants.DOC_BINS,
        labels=sr.constants.DOC_BIN_LABELS,
    )

    gnn_res_doc_groups = gnn_res_df.groupby("doc_bin", observed=True)
    gnn_res_doc_bias = gnn_res_doc_groups[gnn_run_config.ims].mean()
    gnn_res_doc_std = gnn_res_doc_groups[gnn_run_config.ims].std()

    cim_res_df["doc"] = gnn_results.loc[cim_res_df.index, "doc"].values
    cim_res_df["doc_bin"] = pd.cut(
        cim_res_df["doc"],
        bins=sr.constants.DOC_BINS,
        labels=sr.constants.DOC_BIN_LABELS,
    )

    cim_res_doc_groups = cim_res_df.groupby("doc_bin", observed=True)
    cim_res_doc_bias = cim_res_doc_groups[sr.constants.PSA_KEYS].mean()
    cim_res_doc_std = cim_res_doc_groups[sr.constants.PSA_KEYS].std()

    group_colors = sr.constants.DOC_COLORS
    group_linewidth = sr.constants.FIG_GROUP_LINEWIDTH
    group_scatter_size = 12.5

    if gnn_run_config.non_pSA_ims is not None:
        fig, ax1, ax2, ax3, ax4 = sr.plot_utils.get_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            left=0.08,
            main_wspace=0.175,
            sub_wspace=0.05,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )
    else:
        fig, ax1, ax3 = sr.plot_utils.get_pSA_bias_residual_fig(
            sr.constants.FIG_SIZE,
            fig_dpi=sr.constants.FIG_DPI,
            main_wspace=0.175,
            left=0.08,
            right=0.99,
            bottom=0.125,
            std_y_axis_limits=(0.0, std_limit),
            bias_y_axis_limits=(-bias_limit, bias_limit),
        )

    ### Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax1.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "mean"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} (N={gnn_res_doc_groups.size()[cur_key]:,})",
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

        ax2.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
        ax2.set_xlim(-0.75, len(gnn_run_config.non_pSA_ims) - 0.25)

    ### Residual Standard Deviation
    ax3.plot(
        sr.constants.PERIODS,
        gnn_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Residual" if gnn_run_config.use_emp_gm_model else "GNN-Only",
    )
    ax3.plot(
        sr.constants.PERIODS,
        cim_res_bias_std_df.loc[sr.constants.PSA_KEYS, "std"],
        color="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
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
            label=f"{cur_key} (N={gnn_res_doc_groups.size()[cur_key]:,})",
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

        ax4.xaxis.set_tick_params(rotation=90, labelsize=sr.constants.FIG_FONT_SIZE)
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
    """
    Generates figure showing the number of records and scenarios
    as a function of pSA period
    """
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

    fig, ax = plt.subplots(figsize=sr.constants.FIG_SIZE, dpi=sr.constants.FIG_DPI)

    ax.plot(
        sr.constants.PERIODS,
        record_count_df.loc[sr.constants.PSA_KEYS],
        color="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="Number of records",
    )
    ax.plot(
        sr.constants.PERIODS,
        scenario_count.loc[sr.constants.PSA_KEYS],
        color="blue",
        linewidth=sr.constants.FIG_LINEWIDTH,
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


@app.command("spatial-corr-trends")
def spatial_corr_trends(
    gnn_only_results_dir: Path,
    gnn_residual_results_dir: Path,
    emp_gm_params_ffp: Path,
    output_dir: Path,
    plot_labels: str = None,
    bias_limit: float = 1.0,
    std_limit: float = 1.0,
):
    plot_labels = plot_labels.split(",") if plot_labels is not None else None

    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    gnn_only_run_config = sr.ml.RunConfig.from_yaml(
        gnn_only_results_dir / "run_config.yaml"
    )
    gnn_residual_run_config = sr.ml.RunConfig.from_yaml(
        gnn_residual_results_dir / "run_config.yaml"
    )
    assert gnn_only_run_config.obs_data_ffp == gnn_residual_run_config.obs_data_ffp

    # Load data
    obs_data = sr.data.load_obs_nzgmdb(gnn_only_run_config.obs_data_ffp)
    gnn_only_results = pd.read_parquet(
        gnn_only_results_dir / "val_results.parquet"
    ).sort_index()
    gnn_residual_results = pd.read_parquet(
        gnn_residual_results_dir / "val_results.parquet"
    ).sort_index()
    emp_gm_params = pd.read_parquet(emp_gm_params_ffp)

    if (gnn_only_results_dir / "cim_results/val_results.parquet").exists():
        cim_results = pd.read_parquet(
            gnn_only_results_dir / "cim_results/val_results.parquet"
        ).sort_index()
    elif (gnn_residual_results_dir / "cim_results/val_results.parquet").exists():
        cim_results = pd.read_parquet(
            gnn_residual_results_dir / "cim_results/val_results.parquet"
        ).sort_index()
    else:
        print("No CIM results found")

    assert gnn_only_results.index.equals(cim_results.index), "Index mismatch"
    assert gnn_only_results.index.equals(gnn_residual_results.index), "Index mismatch"

    # Get the correlations
    print("Computing correlations")
    (
        gnn_only_site_pair_corrs,
        obs_site_pair_corrs,
        cim_site_pair_corrs,
        site_pairs_df,
    ) = sr.analysis.compute_site_int_obs_correlation_residuals(
        gnn_only_results, obs_data, emp_gm_params, cim_results=cim_results
    )

    (
        gnn_residual_site_pair_corrs,
        obs_site_pair_corrs_2,
        *_,
    ) = sr.analysis.compute_site_int_obs_correlation_residuals(
        gnn_residual_results, obs_data, emp_gm_params
    )
    assert obs_site_pair_corrs_2.equals(obs_site_pair_corrs)

    # Compute fisher transform residuals
    gnn_only_corr_residuals = sr.analysis.get_fisher_transform_residuals(
        gnn_only_site_pair_corrs, obs_site_pair_corrs
    )
    gnn_residual_corr_residuals = sr.analysis.get_fisher_transform_residuals(
        gnn_residual_site_pair_corrs, obs_site_pair_corrs
    )
    cim_corr_residuals = None
    if cim_results is not None:
        cim_corr_residuals = sr.analysis.get_fisher_transform_residuals(
            cim_site_pair_corrs, obs_site_pair_corrs
        )

    # Add site-to-site distance
    dist_matrix = sr.utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)
    site_int_ind = dist_matrix.index.get_indexer_for(site_pairs_df["site_int"].values)
    obs_site_ind = dist_matrix.columns.get_indexer_for(site_pairs_df["obs_site"].values)
    site_dist = dist_matrix.values[site_int_ind, obs_site_ind]
    site_pairs_df["dist"] = site_dist

    # Add Vs30 Difference
    site_pairs_df["site_int_vs30"] = obs_data.site_df.loc[
        site_pairs_df["site_int"].values, "vs30"
    ].values
    site_pairs_df["obs_site_vs30"] = obs_data.site_df.loc[
        site_pairs_df["obs_site"].values, "vs30"
    ].values
    site_pairs_df["site_int_ln_vs30"] = np.log(
        obs_data.site_df.loc[site_pairs_df["site_int"].values, "vs30"].values
    )
    site_pairs_df["obs_site_ln_vs30"] = np.log(
        obs_data.site_df.loc[site_pairs_df["obs_site"].values, "vs30"].values
    )

    site_pairs_df["ln_vs30_diff"] = (
        site_pairs_df["site_int_ln_vs30"] - site_pairs_df["obs_site_ln_vs30"]
    )
    site_pairs_df["abs_ln_vs30_diff"] = np.abs(site_pairs_df["ln_vs30_diff"])

    # Compute bias and residual standard deviation
    gnn_only_corr_res_bias = gnn_only_corr_residuals.mean(axis=0)
    gnn_only_corr_res_std = gnn_only_corr_residuals.std(axis=0)

    gnn_residual_corr_res_bias = gnn_residual_corr_residuals.mean(axis=0)
    gnn_residual_corr_res_std = gnn_residual_corr_residuals.std(axis=0)

    if cim_corr_residuals is not None:
        cim_corr_res_bias = cim_corr_residuals.mean(axis=0)
        cim_corr_res_std = cim_corr_residuals.std(axis=0)

    # Plot total bias and residual standard deviation
    print("Plotting")
    fig, ax1, ax2 = sr.plot_utils.get_pSA_bias_residual_fig(
        sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.08,
        right=0.99,
        bottom=0.125,
        bias_y_axis_limits=(-bias_limit, bias_limit),
        std_y_axis_limits=(0.0, std_limit),
    )

    ax1.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-only",
        c="blue",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax1.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-residual",
        c="purple",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )

    ax2.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_std[sr.constants.PSA_KEYS],
        label="GNN-only",
        c="blue",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax2.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_std[sr.constants.PSA_KEYS],
        label="GNN-residual",
        c="purple",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )

    if cim_corr_res_bias is not None:
        ax1.plot(
            sr.constants.PERIODS,
            cim_corr_res_bias[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="green",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )
        ax2.plot(
            sr.constants.PERIODS,
            cim_corr_res_std[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="green",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    ax1.legend()
    # ax1.yaxis.set_major_locator(ticker.MaxNLocator(9))
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

    fig.savefig(output_dir / f"spatial_corr_bias_std.{sr.constants.FIG_FORMAT}")
    plt.close(fig)

    ### Site-to-site distance
    gnn_only_corr_residuals["dist"] = site_pairs_df["dist"]
    gnn_only_corr_residuals["dist_bin"] = pd.cut(
        gnn_only_corr_residuals["dist"],
        bins=sr.constants.SITE_TO_SITE_DIST_BINS,
        labels=sr.constants.SITE_TO_SITE_DIST_BIN_LABELS,
    )
    gnn_only_corr_residual_groups = gnn_only_corr_residuals.groupby(
        "dist_bin", observed=False
    )
    gnn_only_res_dist_corr_bias = gnn_only_corr_residual_groups.mean()
    gnn_only_res_dist_corr_std = gnn_only_corr_residual_groups.std()

    gnn_residual_corr_residuals["dist"] = site_pairs_df["dist"]
    gnn_residual_corr_residuals["dist_bin"] = pd.cut(
        gnn_residual_corr_residuals["dist"],
        bins=sr.constants.SITE_TO_SITE_DIST_BINS,
        labels=sr.constants.SITE_TO_SITE_DIST_BIN_LABELS,
    )
    gnn_residual_corr_residual_groups = gnn_residual_corr_residuals.groupby(
        "dist_bin", observed=False
    )
    gnn_residual_res_dist_corr_bias = gnn_residual_corr_residual_groups.mean()
    gnn_residual_res_dist_corr_std = gnn_residual_corr_residual_groups.std()

    if cim_corr_residuals is not None:
        cim_corr_residuals["dist"] = site_pairs_df["dist"]
        cim_corr_residuals["dist_bin"] = pd.cut(
            cim_corr_residuals["dist"],
            bins=sr.constants.SITE_TO_SITE_DIST_BINS,
            labels=sr.constants.SITE_TO_SITE_DIST_BIN_LABELS,
        )
        cim_corr_residual_groups = cim_corr_residuals.groupby(
            "dist_bin", observed=False
        )
        cim_res_dist_corr_bias = cim_corr_residual_groups.mean()
        cim_res_dist_corr_std = cim_corr_residual_groups.std()

    ## GNN-Only
    fig, ax1, ax2 = sr.plot_utils.get_pSA_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.08,
        right=0.99,
        bottom=0.125,
        bias_y_axis_limits=(-bias_limit, bias_limit),
        std_y_axis_limits=(0, std_limit),
    )

    # Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-Only",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax1.plot(
            sr.constants.PERIODS,
            cim_corr_res_bias[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_only_corr_residual_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_only_res_dist_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"S2S {cur_key} km (N = {gnn_only_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )

        if cim_corr_res_bias is not None:
            ax1.semilogx(
                sr.constants.PERIODS,
                cim_res_dist_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    # ax1.legend()
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
    if plot_labels is not None:
        ax1.text(
            -0.175,
            0.98,
            plot_labels[0],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    # Residual Standard Deviation
    ax2.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_std[sr.constants.PSA_KEYS],
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="GNN-Only",
    )
    ax2.plot(
        sr.constants.PERIODS,
        cim_corr_res_std[sr.constants.PSA_KEYS],
        c="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
        label="MVN-CIM",
    )
    for i, (cur_key, cur_group) in enumerate(gnn_only_corr_residual_groups):
        ax2.semilogx(
            sr.constants.PERIODS,
            gnn_only_res_dist_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
            c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            label=f"S2S {cur_key} km (N={gnn_only_corr_residual_groups.size()[cur_key]})",
        )

        if cim_corr_res_std is not None:
            ax2.semilogx(
                sr.constants.PERIODS,
                cim_res_dist_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    ax2.legend()
    fig.savefig(output_dir / f"spatial_corr_s2s_gnn_only.{sr.constants.FIG_FORMAT}")
    plt.close(fig)

    ## GNN-Residual
    fig, ax1, ax2 = sr.plot_utils.get_pSA_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.08,
        right=0.99,
        bottom=0.125,
        bias_y_axis_limits=(-bias_limit, bias_limit),
        std_y_axis_limits=(0, std_limit),
    )

    # Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-Residual",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax1.plot(
            sr.constants.PERIODS,
            cim_corr_res_bias[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_residual_corr_residual_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_residual_res_dist_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"S2S {cur_key} km (N={gnn_residual_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )

        if cim_corr_res_bias is not None:
            ax1.semilogx(
                sr.constants.PERIODS,
                cim_res_dist_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    # ax1.legend()
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
    if plot_labels is not None:
        ax1.text(
            -0.175,
            0.98,
            plot_labels[1],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    # Residual Standard Deviation
    ax2.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_std[sr.constants.PSA_KEYS],
        label="GNN-Residual",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    ax2.plot(
        sr.constants.PERIODS,
        cim_corr_res_std[sr.constants.PSA_KEYS],
        label="MVN-CIM",
        c="black",
        linestyle="--",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )

    for i, (cur_key, cur_group) in enumerate(gnn_residual_corr_residual_groups):
        ax2.semilogx(
            sr.constants.PERIODS,
            gnn_residual_res_dist_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
            label=f"S2S {cur_key} km (N={gnn_residual_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )

        if cim_corr_res_std is not None:
            ax2.semilogx(
                sr.constants.PERIODS,
                cim_res_dist_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.SITE_TO_SITE_DIST_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    ax2.legend()

    fig.savefig(output_dir / f"spatial_corr_s2s_gnn_residual.{sr.constants.FIG_FORMAT}")
    plt.close(fig)

    ### Vs30 Difference
    gnn_only_corr_residuals["abs_ln_vs30_diff"] = site_pairs_df["abs_ln_vs30_diff"]
    gnn_only_corr_residuals["ln_vs30_diff_bin"] = pd.cut(
        gnn_only_corr_residuals["abs_ln_vs30_diff"],
        bins=sr.constants.LN_VS30_DIFF_BINS,
        labels=sr.constants.LN_VS30_DIFF_BIN_LABELS,
    )
    gnn_only_corr_residual_groups = gnn_only_corr_residuals.groupby(
        "ln_vs30_diff_bin", observed=False
    )
    gnn_only_res_vs30_corr_bias = gnn_only_corr_residual_groups[
        sr.constants.PSA_KEYS
    ].mean()
    gnn_only_res_vs30_corr_std = gnn_only_corr_residual_groups[
        sr.constants.PSA_KEYS
    ].std()

    gnn_residual_corr_residuals["abs_ln_vs30_diff"] = site_pairs_df["abs_ln_vs30_diff"]
    gnn_residual_corr_residuals["ln_vs30_diff_bin"] = pd.cut(
        gnn_residual_corr_residuals["abs_ln_vs30_diff"],
        bins=sr.constants.LN_VS30_DIFF_BINS,
        labels=sr.constants.LN_VS30_DIFF_BIN_LABELS,
    )
    gnn_residual_corr_residual_groups = gnn_residual_corr_residuals.groupby(
        "ln_vs30_diff_bin", observed=False
    )
    gnn_residual_res_vs30_corr_bias = gnn_residual_corr_residual_groups[
        sr.constants.PSA_KEYS
    ].mean()
    gnn_residual_res_vs30_corr_std = gnn_residual_corr_residual_groups[
        sr.constants.PSA_KEYS
    ].std()

    if cim_corr_residuals is not None:
        cim_corr_residuals["abs_ln_vs30_diff"] = site_pairs_df["abs_ln_vs30_diff"]
        cim_corr_residuals["ln_vs30_diff_bin"] = pd.cut(
            cim_corr_residuals["abs_ln_vs30_diff"],
            bins=sr.constants.LN_VS30_DIFF_BINS,
            labels=sr.constants.LN_VS30_DIFF_BIN_LABELS,
        )
        cim_corr_residual_groups = cim_corr_residuals.groupby(
            "ln_vs30_diff_bin", observed=False
        )
        cim_res_vs30_corr_bias = cim_corr_residual_groups[sr.constants.PSA_KEYS].mean()
        cim_res_vs30_corr_std = cim_corr_residual_groups[sr.constants.PSA_KEYS].std()

    ## GNN-Only
    fig, ax1, ax2 = sr.plot_utils.get_pSA_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.08,
        right=0.99,
        bottom=0.125,
        bias_y_axis_limits=(-bias_limit, bias_limit),
        std_y_axis_limits=(0, std_limit),
    )

    # Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-Only",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax1.plot(
            sr.constants.PERIODS,
            cim_corr_res_bias[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_only_corr_residual_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_only_res_vs30_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=rf"$\Delta_{{V_{{S30}}}}$ {cur_key} km (N={gnn_only_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.LN_VS30_DIFF_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )
        if cim_corr_res_bias is not None:
            ax1.semilogx(
                sr.constants.PERIODS,
                cim_res_vs30_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.LN_VS30_DIFF_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

        # ax1.legend()
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
    if plot_labels is not None:
        ax1.text(
            -0.175,
            0.98,
            plot_labels[0],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    # Residual Standard Deviation
    ax2.plot(
        sr.constants.PERIODS,
        gnn_only_corr_res_std[sr.constants.PSA_KEYS],
        label="GNN-Only",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax2.plot(
            sr.constants.PERIODS,
            cim_corr_res_std[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_only_corr_residual_groups):
        ax2.semilogx(
            sr.constants.PERIODS,
            gnn_only_res_vs30_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
            label=rf"$\Delta_{{V_{{S30}}}}$ {cur_key} km (N={gnn_only_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.LN_VS30_DIFF_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )
        if cim_corr_res_std is not None:
            ax2.semilogx(
                sr.constants.PERIODS,
                cim_res_vs30_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.LN_VS30_DIFF_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    ax2.legend()
    fig.savefig(output_dir / f"spatial_corr_vs30_gnn_only.{sr.constants.FIG_FORMAT}")
    plt.close(fig)

    ## GNN-Residual
    fig, ax1, ax2 = sr.plot_utils.get_pSA_bias_residual_fig(
        figsize=sr.constants.FIG_SIZE,
        fig_dpi=sr.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.08,
        right=0.99,
        bottom=0.125,
        bias_y_axis_limits=(-bias_limit, bias_limit),
        std_y_axis_limits=(0, std_limit),
    )

    # Bias
    ax1.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_bias[sr.constants.PSA_KEYS],
        label="GNN-Residual",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax1.plot(
            sr.constants.PERIODS,
            cim_corr_res_bias[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_residual_corr_residual_groups):
        ax1.semilogx(
            sr.constants.PERIODS,
            gnn_residual_res_vs30_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
            label=rf"$\Delta_{{V_{{S30}}}}$ {cur_key} km (N={gnn_residual_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.LN_VS30_DIFF_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )
        if cim_corr_res_bias is not None:
            ax1.semilogx(
                sr.constants.PERIODS,
                cim_res_vs30_corr_bias.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.LN_VS30_DIFF_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

        # ax1.legend()
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
    if plot_labels is not None:
        ax1.text(
            -0.175,
            0.98,
            plot_labels[1],
            transform=ax1.transAxes,
            va="center",
            ha="left",
            fontsize=14,
        )

    # Residual Standard Deviation
    ax2.plot(
        sr.constants.PERIODS,
        gnn_residual_corr_res_std[sr.constants.PSA_KEYS],
        label="GNN-Residual",
        c="black",
        linewidth=sr.constants.FIG_LINEWIDTH,
    )
    if cim_corr_residuals is not None:
        ax2.plot(
            sr.constants.PERIODS,
            cim_corr_res_std[sr.constants.PSA_KEYS],
            label="MVN-CIM",
            c="black",
            linestyle="--",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

    for i, (cur_key, cur_group) in enumerate(gnn_residual_corr_residual_groups):
        ax2.semilogx(
            sr.constants.PERIODS,
            gnn_residual_res_vs30_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
            label=rf"$\Delta_{{V_{{S30}}}}$ {cur_key} km (N={gnn_residual_corr_residual_groups.size()[cur_key]})",
            c=sr.constants.LN_VS30_DIFF_COLORS[i],
            linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        )
        if cim_corr_res_std is not None:
            ax2.semilogx(
                sr.constants.PERIODS,
                cim_res_vs30_corr_std.loc[cur_key, sr.constants.PSA_KEYS],
                c=sr.constants.LN_VS30_DIFF_COLORS[i],
                linestyle="--",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )

    ax2.legend()
    fig.savefig(
        output_dir / f"spatial_corr_vs30_gnn_residual.{sr.constants.FIG_FORMAT}"
    )
    plt.close(fig)


@app.command("ind-scenario-pSA")
def ind_scenario_pSA(
    event_id: str,
    # gnn_only_ffp: Path,
    gnn_residual_ffp: Path,
    cim_results_ffp: Path,
    emp_gm_params_ffp: Path,
    nzgmdb_ffp: Path,
    output_dir: Path,
):
    # Update font size
    if sr.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": sr.constants.FIG_FONT_SIZE,
            }
        )

    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    # gnn_only_pred_df = pd.read_parquet(gnn_only_ffp)
    gnn_residual_pred_df = pd.read_parquet(gnn_residual_ffp)

    cim_results = pd.read_parquet(cim_results_ffp)
    emp_gm_params = pd.read_parquet(emp_gm_params_ffp)

    dist_matrix = sr.utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)

    obs_color_boundaries = np.array([0, 1.0, 2.5, 5.0, 10.0, 30.0])
    obs_colors = sns.color_palette("hls", 5)

    site_ints = obs_data.event_sites[event_id]
    for cur_site_int in tqdm(site_ints):
        cur_id = f"{event_id}_{cur_site_int}"

        if cur_id not in gnn_residual_pred_df.index:
            continue

        cur_obs_sites = sr.plot_ind_scenarios.get_obs_sites(
            event_id, cur_site_int, gnn_residual_pred_df, dist_matrix, n_obs_sites=5
        )

        fig, ax = plt.subplots(figsize=sr.constants.FIG_SIZE, dpi=sr.constants.FIG_DPI)

        # Observation sites
        obs_lines = []
        for cur_obs_site in cur_obs_sites:
            cur_obs_site_id = f"{event_id}_{cur_obs_site}"

            (cur_line,) = ax.plot(
                sr.constants.PERIODS,
                np.log(
                    obs_data.record_df.loc[
                        cur_obs_site_id, sr.constants.PSA_KEYS
                    ].values.astype(float)
                ),
                c=obs_colors[
                    mlt.array_utils.find_nearest_larger(
                        obs_color_boundaries,
                        dist_matrix.loc[cur_site_int, cur_obs_site],
                    )
                    - 1
                ],
                linestyle="--",
                label=f"{cur_obs_site} - S2S: {dist_matrix.loc[cur_site_int, cur_obs_site]:.1f} km",
                linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
            )
            obs_lines.append(cur_line)

        # Empirical GM
        (emp_line,) = ax.plot(
            sr.constants.PERIODS,
            emp_gm_params.loc[cur_id, sr.constants.GMM_PRED_PSA_KEYS],
            label="Empirical GM",
            c="gray",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

        # cIM
        (cim_line,) = ax.plot(
            sr.constants.PERIODS,
            cim_results.loc[cur_id, sr.constants.CIM_PRED_PSA_KEYS],
            label="MVN-CIM",
            c="green",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )
        ax.fill_between(
            sr.constants.PERIODS,
            cim_results.loc[cur_id, sr.constants.CIM_PRED_PSA_KEYS].values.astype(float)
            + cim_results.loc[cur_id, sr.constants.CIM_PRED_STD_PSA_KEYS].values.astype(
                float
            ),
            cim_results.loc[cur_id, sr.constants.CIM_PRED_PSA_KEYS].values.astype(float)
            - cim_results.loc[cur_id, sr.constants.CIM_PRED_STD_PSA_KEYS].values.astype(
                float
            ),
            color="green",
            alpha=0.2,
        )

        # ax.plot(
        #     sr.constants.PERIODS,
        #     cim_results.loc[cur_id, sr.constants.CIM_PRED_PSA_KEYS].values
        #     + cim_results.loc[cur_id, sr.constants.CIM_PRED_STD_PSA_KEYS].values,
        #     c="green",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )
        # ax.plot(
        #     sr.constants.PERIODS,
        #     cim_results.loc[cur_id, sr.constants.CIM_PRED_PSA_KEYS].values
        #     - cim_results.loc[cur_id, sr.constants.CIM_PRED_STD_PSA_KEYS].values,
        #     c="green",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )

        # # GNN Only
        # gnn_line, = ax.plot(
        #     sr.constants.PERIODS,
        #     gnn_only_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS],
        #     label="GNN",
        #     c="blue",
        #     linewidth=sr.constants.FIG_LINEWIDTH,
        # )
        # ax.plot(
        #     sr.constants.PERIODS,
        #     gnn_only_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS].values
        #     + gnn_only_pred_df.loc[cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS].values,
        #     c="blue",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )
        # ax.plot(
        #     sr.constants.PERIODS,
        #     gnn_only_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS].values
        #     - gnn_only_pred_df.loc[cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS].values,
        #     c="blue",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )

        # GNN Residual
        (gnn_line,) = ax.plot(
            sr.constants.PERIODS,
            gnn_residual_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS],
            label="GNN-Residual",
            c="purple",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )
        ax.fill_between(
            sr.constants.PERIODS,
            gnn_residual_pred_df.loc[
                cur_id, sr.constants.GNN_PRED_PSA_KEYS
            ].values.astype(float)
            + gnn_residual_pred_df.loc[
                cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS
            ].values.astype(float),
            gnn_residual_pred_df.loc[
                cur_id, sr.constants.GNN_PRED_PSA_KEYS
            ].values.astype(float)
            - gnn_residual_pred_df.loc[
                cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS
            ].values.astype(float),
            color="purple",
            alpha=0.2,
        )

        # ax.plot(
        #     sr.constants.PERIODS,
        #     gnn_residual_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS].values
        #     + gnn_residual_pred_df.loc[cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS].values,
        #     c="purple",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )
        # ax.plot(
        #     sr.constants.PERIODS,
        #     gnn_residual_pred_df.loc[cur_id, sr.constants.GNN_PRED_PSA_KEYS].values
        #     - gnn_residual_pred_df.loc[cur_id, sr.constants.GNN_PRED_STD_PSA_KEYS].values,
        #     c="purple",
        #     linestyle="--",
        #     linewidth=sr.constants.FIG_GROUP_LINEWIDTH,
        # )

        # Observed
        (obs_line,) = ax.plot(
            sr.constants.PERIODS,
            np.log(
                obs_data.record_df.loc[cur_id, sr.constants.PSA_KEYS].values.astype(
                    float
                )
            ),
            label="Observed",
            c="red",
            linewidth=sr.constants.FIG_LINEWIDTH,
        )

        ax.set_xscale("log")
        ax.grid(
            which="both",
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("pSA (g)")
        ax.set_xlim(0.01, 10.0)
        ax.yaxis.set_major_formatter(FuncFormatter(sr.plot_ind_scenarios.exp_formatter))

        ax.set_ylim(np.log(0.005), np.log(5.0))

        legend_1 = ax.legend(
            handles=[emp_line, obs_line, gnn_line, cim_line], loc="upper right"
        )
        ax.add_artist(legend_1)

        ax.legend(handles=obs_lines, loc="lower left")

        ax.text(
            0.02,
            0.95,
            f"{cur_site_int}",
            transform=ax.transAxes,
            fontweight="bold",
            va="center",
            ha="left",
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"{event_id}_{cur_site_int}.{sr.constants.FIG_FORMAT}")
        plt.close(fig)


if __name__ == "__main__":
    app()
