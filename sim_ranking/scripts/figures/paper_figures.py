from pathlib import Path
from typing import List, Sequence, Tuple
from typing_extensions import Annotated

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

@app.command("plot-mag-distribution")
def db_mag_distribution(db_ffp: Path, output_ffp: Path, figsize: Tuple[float, float] = (6, 4.5)):
    """Creates a magnitude histogram"""
    db = sr.db.DB(db_ffp)
    events = db.get_avail_events(data_source="neil")
    events_df = db.get_event_df().loc[events]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(events_df["mag"], bins=10, color="darkblue", edgecolor="black")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()

    plt.savefig(output_ffp)

@app.command("plot-vs30-distribution")
def db_vs30_distribution(db_ffp: Path, output_ffp: Path, figsize: Tuple[float, float] = (6, 4.5)):
    """Creates a Vs30 histogram"""
    db = sr.db.DB(db_ffp)
    sites_df = db.get_site_df()

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(sites_df["vs30"], bins=10, color="darkblue", edgecolor="black")
    ax.set_xlabel("Vs30")
    ax.set_ylabel("Count")
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()

    plt.savefig(output_ffp)


@app.command("events-stations-map")
def plot_events_map(
    event_ffp: Annotated[
        Path,
        typer.Argument(
            help="NZGMDB event file path", exists=True, dir_okay=False, file_okay=True
        ),
    ],
    db_ffp: Annotated[
        Path,
        typer.Argument(
            help="DB file path", exists=True, dir_okay=False, file_okay=True
        ),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(
            help="Output file path", exists=False, dir_okay=False, file_okay=True
        ),
    ],
    map_data_ffp: Annotated[Path, typer.Option(help="Map data file path")] = None,
    data_source: Annotated[str, typer.Option(help="DB Data source")] = "neil",
    region_code: Annotated[str, typer.Option(help="Region to plot")] = "CAN",
):
    """Creates a pygmt map of the events & stations in the specified region"""
    from pygmt_helper import plotting

    db = sr.db.DB(db_ffp)
    events = db.get_avail_events(data_source=data_source)

    sites_df = db.get_site_df()
    events_df = pd.read_csv(event_ffp, index_col="evid").loc[events]

    min_lon, max_lon, min_lat, max_lat = region = (
        sr.constants.CANTERBURY_REGION
        if region_code == "CAN"
        else sr.constants.WELLINGTON_REGION
    )

    # Filter by region
    events_df = events_df.loc[
        (events_df.lon > min_lon)
        & (events_df.lon < max_lon)
        & (events_df.lat > min_lat)
        & (events_df.lat < max_lat)
    ]
    sites_df = sites_df.loc[
        (sites_df.lon > min_lon)
        & (sites_df.lon < max_lon)
        & (sites_df.lat > min_lat)
        & (sites_df.lat < max_lat)
    ]

    # Load map data
    map_data = (
        None
        if map_data_ffp is None
        else plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
    )
    fig = plotting.gen_region_fig(
        region=region,
        map_data=map_data,
        plot_kwargs=dict(
            topo_cmap="oleron",
            topo_cmap_min=0,
            topo_cmap_max=3000,
            topo_cmap_inc=10,
            topo_cmap_reverse=False,
        ),
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_FRAME_PEN="thinner,black",
            FONT_ANNOT_PRIMARY="6p,Helvetica,black",
        ),
    )

    # Plot the sites
    # for ix, (cur_site, cur_row) in enumerate(sites_df.iterrows()):
    fig.plot(
        x=sites_df.lon,
        y=sites_df.lat,
        style="t0.25c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    for cur_event, cur_row in events_df.iterrows():
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
