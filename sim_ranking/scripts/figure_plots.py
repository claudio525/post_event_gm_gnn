from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import typer
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

import sim_ranking as sr
import spatial_hazard as sh
import sha_calc as sha

app = typer.Typer()


@app.command("perturbed-waveforms")
def plot_perturbed_waveforms(
    sim_rupture_dir: Path,
    site: str,
    output_ffp: Path,
    rel_ids: List[str] = None,
    n_records: int = None,
    max_t: float = None,
):
    # Select some random realisations
    if not len(rel_ids) > 0:
        # Get the available realisation ids
        avail_rel_ids = [
            cur_dir.stem for cur_dir in list(sim_rupture_dir.glob("*REL*"))
        ]
        rel_ids = np.random.choice(avail_rel_ids, n_records, replace=False)

    # Load the waveform data
    time_data, acc_data = [], []
    for cur_rel_id in rel_ids:
        cur_t, cur_acc = sr.data.load_sim_waveform(sim_rupture_dir, cur_rel_id, site)

        if max_t is not None:
            cur_mask = cur_t < max_t
            cur_t = cur_t[cur_mask]
            cur_acc = cur_acc[cur_mask]

        time_data.append(cur_t)
        acc_data.append(cur_acc)

    fig = plt.figure(figsize=(6, 4.5))

    axes = sr.plots.draw_waveforms(
        fig, acc_data, time_data, colors=["r"] * n_records, add_comp_text=False
    )
    for cur_ax in axes:
        sns.despine(ax=cur_ax, left=True, bottom=True)
        cur_ax.set(xticklabels=[])
        cur_ax.set(yticklabels=[])
        cur_ax.xaxis.set_minor_locator(ticker.NullLocator())
        cur_ax.xaxis.set_major_locator(ticker.NullLocator())
        cur_ax.yaxis.set_major_locator(ticker.NullLocator())
        cur_ax.yaxis.set_minor_locator(ticker.NullLocator())
        cur_ax.tick_params(bottom="off", left="off")

    fig.tight_layout()
    plt.savefig(output_ffp)
    plt.close()


@app.command("perturbed-response-spectrum")
def plot_perturbed_response_spectrum(
    sim_imdb_ffp: Path, site: str, n_rels: int, output_ffp: Path
):
    # Load Simulation data
    sim_df = sr.data.load_sim_data(sim_imdb_ffp, [site])[site]

    # Select realisations
    rel_ids = np.random.choice(sim_df.index.values.astype(str), n_rels, replace=False)

    periods, pSA_keys = sr.utils.get_periods(sim_df.columns.values.astype(str))

    # fig = plt.figure(figsize=(6, 4.5))
    fig = plt.figure(figsize=(8, 6))

    for cur_rel_id in rel_ids:
        plt.plot(
            periods,
            sim_df.loc[cur_rel_id, pSA_keys],
            c="gray",
            linewidth=0.75,
            alpha=0.5,
        )

    plt.semilogx()
    plt.xlim(periods.min(), periods.max())

    plt.xlabel(f"Period (s)")
    plt.ylabel(f"Pseudo-spectral acceleration, Sa (g)")
    plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()

    plt.savefig(output_ffp)
    plt.close()


@app.command("observation-sites")
def plot_observation_sites(sites_ffp: Path, map_data_ffp: Path, output_ffp: Path):
    from pygmt_helper import plotting

    sites_df = pd.read_csv(sites_ffp, index_col="sta")

    min_lon, max_lon, min_lat, max_lat = sr.constants.CANTERBURY_REGION

    sites_df = sites_df.loc[
        (sites_df.lon > min_lon)
        & (sites_df.lon < max_lon)
        & (sites_df.lat > min_lat)
        & (sites_df.lat < max_lat)
    ]

    map_data = plotting.NZMapData.load(map_data_ffp, high_res_topo=True)

    fig = plotting.gen_region_fig(
        region=(min_lon, max_lon, min_lat, max_lat),
        map_data=map_data,
        # plot_kwargs={"topo_cmap": "oleron"},
    )

    for cur_site, cur_row in sites_df.iterrows():
        fig.plot(
            x=cur_row.lon,
            y=cur_row.lat,
            style="d0.1c",
            fill="black",
            pen="black",
        )

    fig.savefig(
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


@app.command("historic-events")
def plot_historic_events(events_ffp: Path, map_data_ffp: Path, output_ffp: Path):
    from pygmt_helper import plotting

    event_df = pd.read_csv(events_ffp)

    min_lon, max_lon, min_lat, max_lat = sr.constants.CANTERBURY_REGION

    # Filter by region
    event_df = event_df.loc[
        (event_df.lon > min_lon)
        & (event_df.lon < max_lon)
        & (event_df.lat > min_lat)
        & (event_df.lat < max_lat)
    ]

    # Magnitude filter
    event_df = event_df.loc[event_df.mag > 4]

    # Load map data
    map_data = plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
    fig = plotting.gen_region_fig(
        region=(min_lon, max_lon, min_lat, max_lat), map_data=map_data
    )

    for cur_event, cur_row in event_df.iterrows():
        fig.meca(
            spec=dict(
                strike=cur_row.strike,
                dip=cur_row.dip,
                rake=cur_row.rake,
                magnitude=cur_row.mag,
            ),
            scale=f"{0.05 * cur_row.mag}c",
            longitude=cur_row.lon,
            latitude=cur_row.lat,
            depth=cur_row.depth,
        )

    fig.savefig(
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


@app.command("event-sites-example")
def plot_event_sites_example(
    events_ffp: Path, sites_ffp: Path, map_data_ffp: Path, output_ffp: Path, events_to_highlight: List[str] = None,
        sites_to_highlight: List[str] = None
):
    """Creates a figure of Canterbury showing
    all historic events"""
    from pygmt_helper import plotting

    event_df = pd.read_csv(events_ffp, index_col="evid")
    event_df.index = event_df.index.values.astype(str)
    sites_df = pd.read_csv(sites_ffp, index_col="sta")

    min_lon, max_lon, min_lat, max_lat = sr.constants.CANTERBURY_REGION

    # Filter by region
    event_df = event_df.loc[
        (event_df.lon > min_lon)
        & (event_df.lon < max_lon)
        & (event_df.lat > min_lat)
        & (event_df.lat < max_lat)
    ]

    sites_df = sites_df.loc[
        (sites_df.lon > min_lon)
        & (sites_df.lon < max_lon)
        & (sites_df.lat > min_lat)
        & (sites_df.lat < max_lat)
    ]

    # Magnitude filter
    event_df = event_df.loc[event_df.mag > 4]

    # Load map data
    map_data = (
        None
        if map_data_ffp is None
        else plotting.NZMapData.load(map_data_ffp, high_res_topo=False)
    )
    # map_data = None

    # Generate the figure
    fig = plotting.gen_region_fig(
        region=(min_lon, max_lon, min_lat, max_lat),
        map_data=map_data,
        plot_kwargs=dict(frame_args=["+n"]),
        config_options=dict(
            # MAP_FRAME_TYPE="plain",
            # FORMAT_GEO_MAP="ddd.xx",
            # MAP_FRAME_PEN="thinner,black",
            # FONT_ANNOT_PRIMARY="6p,Helvetica,black",
        ),
    )

    # Plot the events
    for ix, (cur_event, cur_row) in enumerate(event_df.iterrows()):
        cur_c = "green" if str(cur_event) in events_to_highlight else "red"
        fig.meca(
            spec=dict(
                strike=cur_row.strike,
                dip=cur_row.dip,
                rake=cur_row.rake,
                magnitude=cur_row.mag,
            ),
            scale=f"{0.06 * cur_row.mag}c",
            G=cur_c,
            W="0.05p,black,solid",
            longitude=cur_row.lon,
            latitude=cur_row.lat,
            depth=cur_row.depth,
        )

    # Create the inset rectangle
    inset_region = [
        172.60,
        172.69,
        -43.545,
        -43.495,
    ]
    fig.plot(
        data=[[inset_region[0], inset_region[2], inset_region[1], inset_region[3]]],
        style="r+s",
        pen="0.5p,black",
    )

    # Plot the sites
    for ix, (cur_site, cur_row) in enumerate(sites_df.iterrows()):
        fig.plot(
            x=cur_row.lon,
            y=cur_row.lat,
            style="t0.25c",
            fill="darkblue",
            pen="0.1p,darkblue",
        )

        if cur_site in sites_to_highlight:
            # Draw circle around the site and add text
            fig.plot(
                x=cur_row.lon,
                y=cur_row.lat,
                style="c0.4c",
                fill=None,
                pen="0.5p,magenta",
            )
            # fig.text(
            #     text=cur_site,
            #     x=cur_row.lon,
            #     y=cur_row.lat,
            #     justify="LM",
            #     font="6p,Helvetica,black",
            #     offset="0.1c",
            # )

    # Create the inset
    with fig.inset(
        position="jTR",#+o0.2c",
        region=inset_region,
        projection="M4c",
        margin=0,
        box="+p0.5p,black",
    ):
        fig.basemap(frame=False)

        # Plots the default coast (sea & inland lakes/rivers)
        if map_data is None:
            fig.coast(
                shorelines=["1/0.1p,black", "2/0.1p,black"],
                resolution="f",
                land="#666666",
                water="skyblue",
            )
        # Use the custom NZ data
        else:
            plotting._draw_map_data(
                fig, map_data, plot_kwargs=plotting.DEFAULT_PLT_KWARGS
            )  # , plot_kwargs=dict(topo_cmap="oleron"))

        # Plot the sites
        inset_sites_df = sites_df.loc[
            (sites_df.lon > inset_region[0])
            & (sites_df.lon < inset_region[1])
            & (sites_df.lat > inset_region[2])
            & (sites_df.lat < inset_region[3])
        ]
        for ix, (cur_site, cur_row) in enumerate(inset_sites_df.iterrows()):
            fig.plot(
                x=cur_row.lon,
                y=cur_row.lat,
                style="t0.15c",
                fill="darkblue",
                pen="0.1p,black",
            )

        # Plot the site of interest
        fig.plot(
            x=172.636849,
            y=-43.530954,
            style="a0.2c",
            fill="orange",
            pen="0.1p,black",
        )
        fig.text(
            text="Site of Interest",
            x=172.64,
            y=-43.530954,
            justify="LM",
            font="6p,Helvetica,black",
        )

    fig.savefig(
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


@app.command("sim-site-correlations")
def sim_site_correlations(sim_corr_dir: Path, site_ffp: Path, ims: List[str], event: str, output_dir: Path):
    """Generates a simulation site-pair
    correlation plot with respect to site-to-site distance."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})

    MAX_DIST = 200

    sim_corrs = sr.data.load_correlations(sim_corr_dir)[event]
    site_df = sr.data.load_ll_file(site_ffp)

    dist_matrix = sh.im_dist.calculate_distance_matrix(sim_corrs.sites, site_df)

    for cur_im in ims:
        # Get the model values
        dist = np.logspace(np.log10(0.1), np.log(MAX_DIST), 1000)
        loth_baker_vals = sha.loth_baker_corr_model.get_correlations(cur_im, cur_im, dist)

        # Get the simulation values
        lower_tri_mask = np.tril(dist_matrix.values).astype(bool)
        cur_sim_site_corrs = sim_corrs.get_im_corrs(cur_im)

        # Compute the moving average
        sim_avg_values = []
        sim_std_values = []
        n_bins = 10
        bins = np.logspace(np.log10(1), np.log10(MAX_DIST), n_bins)
        bin_inds = np.digitize(dist_matrix.values[lower_tri_mask], bins)
        for ix in np.unique(bin_inds):
            if ix == 0 or ix == n_bins:
                continue

            cur_mask = bin_inds == ix
            sim_avg_values.append(np.mean(cur_sim_site_corrs.values[lower_tri_mask][cur_mask]))
            sim_std_values.append(np.std(cur_sim_site_corrs.values[lower_tri_mask][cur_mask]))

        sim_avg_values = np.asarray(sim_avg_values)
        sim_std_values = np.asarray(sim_std_values)
        bin_centres = np.asarray([np.mean(bins[i : i + 2]) for i in range(n_bins - 1)])

        # Create the plot
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(
            dist_matrix.values[lower_tri_mask],
            cur_sim_site_corrs.values[lower_tri_mask],
            s=1.0,
            alpha=0.75,
            label="Simulation Site-Pair Correlations"
        )

        plt.semilogx(bin_centres, sim_avg_values, c="r", linewidth=1.0, label="Simulation Average & Standard Deviation")
        plt.semilogx(bin_centres, sim_avg_values + sim_std_values, c="r", linewidth=1.0, linestyle="--")
        plt.semilogx(bin_centres, sim_avg_values - sim_std_values, c="r", linewidth=1.0, linestyle="--")

        plt.semilogx(dist, loth_baker_vals, c="k", linewidth=1.0, label="Loth & Baker (2013)")

        plt.title(f"{sr.utils.get_nice_im_name(cur_im)}")
        plt.xlabel(f"Distance (km)")
        plt.ylabel(f"Within-Event Site Correlation")
        plt.ylim(-1.0, 1.0)
        plt.xlim(1, 300)
        plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        plt.legend()
        plt.tight_layout()

        fig.savefig(output_dir / f"{event}_{cur_im}_site_correlations.{sr.constants.FIG_FORMAT}")
        plt.close()
        # plt.show()
        # print(f"wtf")



if __name__ == "__main__":
    app()


# @app.command("correlation-period")
# def correlation_period(
#     sim_corr_dir: Path, event: str, station_ffp: Path, output_dir: Path
# ):
#     """Plots the average correlation as a function period."""
#     sim_corrs = sr.data.load_correlations(sim_corr_dir)[event]
#
#     periods, pSA_keys = sr.utils.get_periods(sim_corrs.ims)
#
#     station_df = sr.data.load_ll_file(station_ffp)
#     dist_matrix = sh.im_dist.calculate_distance_matrix(sim_corrs.sites, station_df)
#
#     upper_mask = np.triu(sim_corrs.get_im_corrs(pSA_keys[0]), 1).astype(bool)
#
#     # Compute the average absolute correlation for each period
#     # across all site pairs
#     rho_avg = {}
#     rho_std = {}
#     for cur_im in pSA_keys:
#         cur_corrs = sim_corrs.get_im_corrs(cur_im)
#         rho_avg[cur_im] = np.mean(cur_corrs.values[upper_mask])
#         rho_std[cur_im] = np.std(cur_corrs.values[upper_mask])
#
#     rho_avg = pd.Series(rho_avg)
#     rho_std = pd.Series(rho_std)
#
#     # Compute the average absolute correlation for different distance bins
#     distance_bins = np.array([0, 10, 30, 80, 150])
#     dist_rho_avg = {i: {} for i in range(len(distance_bins) - 1)}
#     dist_rho_std = {i: {} for i in range(len(distance_bins) - 1)}
#     dist_emp_corr = {i: {} for i in range(len(distance_bins) - 1)}
#     for i in range(len(distance_bins) - 1):
#
#         cur_dist_mask = (dist_matrix >= distance_bins[i]) & (
#             dist_matrix < distance_bins[i + 1]
#         )
#         cur_mask = cur_dist_mask & upper_mask
#
#         for cur_im in pSA_keys:
#             cur_corrs = sim_corrs.get_im_corrs(cur_im)
#             dist_rho_avg[i][cur_im] = np.mean(cur_corrs.values[cur_mask])
#             dist_rho_std[i][cur_im] = np.std(cur_corrs.values[cur_mask])
#             dist_emp_corr[i][cur_im] = sha.loth_baker_corr_model.get_correlations(
#                 cur_im, cur_im, np.asarray([np.mean(distance_bins[i : i + 2])])
#             )[0]
#
#     dist_rho_avg = pd.DataFrame(dist_rho_avg)
#     dist_rho_std = pd.DataFrame(dist_rho_std)
#     dist_emp_corr = pd.DataFrame(dist_emp_corr)
#
#     fig = plt.figure(figsize=(8, 6))
#
#     c = ["maroon", "red", "blue", "magenta"]
#     for i in range(len(distance_bins) - 1):
#         plt.semilogx(
#             periods,
#             dist_emp_corr.loc[:, i],
#             label=f"Loth & Baker (2013)" if i == 0 else None,
#             c=c[i],
#             linewidth=1.0,
#             linestyle="--",
#         )
#         plt.semilogx(
#             periods,
#             dist_rho_avg.loc[:, i],
#             label=f"{distance_bins[i]}-{distance_bins[i + 1]} km",
#             c=c[i],
#             linewidth=1.0,
#         )
#
#     plt.semilogx(periods, rho_avg, label="All site pairs", c="k", linewidth=1.0)
#
#     # plt.title(f"Average Absolute Correlation vs Period")
#     plt.xlabel(f"Period (s)")
#     plt.ylabel(f"Average Absolute Correlation")
#     plt.xlim(periods.min(), periods.max())
#     plt.ylim(-0.6, 0.6)
#     plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig(
#         output_dir / f"{event}_average_correlation_vs_period.{sr.constants.FIG_FORMAT}"
#     )
#
#     # Standard deviation plot
#     fig = plt.figure(figsize=(8, 6))
#
#     for i in range(len(distance_bins) - 1):
#         plt.semilogx(
#             periods,
#             dist_rho_std.loc[:, i],
#             label=f"{distance_bins[i]}-{distance_bins[i + 1]} km",
#             c=c[i],
#             linewidth=1.0,
#         )
#
#     plt.semilogx(periods, rho_std, label="All site pairs", c="k", linewidth=1.0)
#
#     plt.xlabel(f"Period (s)")
#     plt.ylabel(f"Standard Deviation Of Absolute Correlation")
#     plt.xlim(periods.min(), periods.max())
#     plt.ylim(0, 0.3)
#     plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig(
#         output_dir / f"{event}_std_correlation_vs_period.{sr.constants.FIG_FORMAT}"
#     )
