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


@app.command("event-site-map")
def event_site_map(
    event: str,
    site_int_ids: List[str],
    max_lon: float,
    min_lon: float,
    max_lat: float,
    min_lat: float,
    nzgmdb_ffp: Path,
    output_ffp: Path,
    val_int_site_ids_ffp: Path = None,
    map_data_ffp: Path = None,
):
    """
    Create a map plot of the event and the site locations
    """
    from pygmt_helper import plotting
    assert max_lon is None or min_lon is not None

    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)
    obs_sites = obs_data.record_df.loc[
        obs_data.record_df.event_id == event, "site_id"
    ].values.astype(str)
    event_data = obs_data.event_df.loc[event]

    # Don't use the validation sites
    val_int_sites = (
        np.concatenate((np.load(val_int_site_ids_ffp), site_int_ids), axis=0)
        if val_int_site_ids_ffp is not None
        else site_int_ids
    )
    obs_sites = obs_sites[~np.isin(obs_sites, val_int_sites)]

    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    # Create figure
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
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


@app.command("test")
def test():
    pass

@app.command("plot-event-predictions")
def plot_event_predictions(model_dir: Path, event_predictions_ffp: Path, output_dir: Path, map_data_ffp: Path = None, region: tuple[float, float, float, float] = sr.constants.CANTERBURY_REGION): 
    from pygmt_helper import plotting

    run_config = sr.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")

    pred_df_ln = pd.read_parquet(event_predictions_ffp)
    pred_df = pred_df_ln.copy(deep=True)
    pred_df[run_config.ims] = np.exp(pred_df[run_config.ims])
    event_id = pred_df.event_id.iloc[0]

    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)
    obs_sites = obs_data.event_sites[event_id]
    event_data = obs_data.event_df.loc[event_id]

    im = "pSA_1.0"

    grid = plotting.create_grid(
        pred_df, 
        im,
        region=region,
    )

    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    # Create figure
    fig = plotting.gen_region_fig(
        region=region,
        map_data=map_data,
        plot_kwargs=dict(frame_args=["+n"]),
        # config_options=dict(
        #     MAP_FRAME_TYPE="plain",
        #     FORMAT_GEO_MAP="ddd.xx",
        #     MAP_FRAME_PEN="thinner,black",
        #     FONT_ANNOT_PRIMARY="6p,Helvetica,black",
        # ),
    )

    # Plot the IM values
    plotting.plot_grid(fig, grid, "hot", (0, 0.4, 0.02), ("white", "black"), im, continuous_cmap=True, reverse_cmap=True, plot_contours=True, transparency=25)

    # Plot the prediction sites
    fig.plot(
        x=pred_df.lon.values,
        y=pred_df.lat.values,
        style="c0.01c",
        fill="black",
        pen="0.1p,black",
    )

    # Plot the source
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
        style="t0.2c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    fig.savefig(
        output_dir / "event_predictions_map.png",
        dpi=900,
        anti_alias=True,
    )



if __name__ == "__main__":
    app()
