from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import typer
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pygmt

import gmhazard_calc as gc
import sim_ranking as sr
from qcore.timeseries import BBSeis, read_ascii

from pygmt_helper import plotting

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
        cur_t, cur_acc = sr.load_sim_waveform(sim_rupture_dir, cur_rel_id, site)

        if max_t is not None:
            cur_mask = cur_t < max_t
            cur_t = cur_t[cur_mask]
            cur_acc = cur_acc[cur_mask]

        time_data.append(cur_t)
        acc_data.append(cur_acc)

    fig = plt.figure(figsize=(6, 4.5))

    axes = sr.draw_waveforms(
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
    sim_df = sr.load_sim_data(sim_imdb_ffp, [site])[site]

    # Select realisations
    rel_ids = np.random.choice(sim_df.index.values.astype(str), n_rels, replace=False)

    pSA_keys = np.asarray(
        [cur_c for cur_c in sim_df.columns if cur_c.startswith("pSA")]
    )
    periods = np.asarray(
        [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_keys]
    )
    sort_ind = np.argsort(periods)

    # fig = plt.figure(figsize=(6, 4.5))
    fig = plt.figure(figsize=(8, 6))

    for cur_rel_id in rel_ids:
        plt.plot(
            periods[sort_ind],
            sim_df.loc[cur_rel_id, pSA_keys[sort_ind]],
            c="gray",
            linewidth=0.75,
            alpha=0.5
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


if __name__ == "__main__":
    app()
