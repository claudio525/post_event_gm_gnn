from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer
from tqdm import tqdm
from pygmt_helper import plotting

import sim_ranking as sr


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


@app.command("plot-event-cim-predictions")
def plot_event_cim_predictions(
    cim_results_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: List[str],
    map_data_ffp: Path = None,
    region: tuple[float, float, float, float] = sr.constants.CANTERBURY_REGION,
):
    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    sr.plot_spatial.plot_event_cim_predictions(
        cim_results_ffp,
        nzgmdb_ffp,
        event_id,
        output_dir,
        ims,
        map_data=map_data,
        region=region,
    )


@app.command("plot-event-gmm-predictions")
def plot_event_gmm_predictions(
    emp_gm_params_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: List[str],
    map_data_ffp: Path = None,
    region: tuple[float, float, float, float] = sr.constants.CANTERBURY_REGION,
):
    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    sr.plot_spatial.plot_event_gmm_predictions(
        emp_gm_params_ffp,
        nzgmdb_ffp,
        event_id,
        output_dir,
        ims,
        map_data=map_data,
        region=region,
    )


@app.command("plot-event-gnn-predictions")
def plot_event_gnn_predictions(
    model_dir: Path,
    event_predictions_ffp: Path,
    output_dir: Path,
    ims: List[str],
    map_data_ffp: Path = None,
    region: tuple[float, float, float, float] = sr.constants.CANTERBURY_REGION,
):
    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    sr.plot_spatial.plot_event_gnn_predictions(
        model_dir,
        event_predictions_ffp,
        output_dir,
        ims,
        map_data=map_data,
        region=region,
    )

@app.command("gen-event-prediction-plots")
def get_event_prediction_plots(
    gnn_model_dir: Path,
    gnn_results_ffp: Path,
    gnn_out_dir: Path,
    emp_gmm_params_ffp: Path,
    emp_gmm_out_dir: Path,
    cim_results_ffp: Path,
    cim_out_dir: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    ims: List[str],
    map_data_ffp: Path = None,
    region: tuple[float, float, float, float] = sr.constants.CANTERBURY_REGION,
):
    # Load map data
    print(f"Loading map data from {map_data_ffp}")
    map_data = (
        plotting.NZMapData.load(map_data_ffp, high_res_topo=True)
        if map_data_ffp is not None
        else None
    )

    print("Plotting GNN predictions")
    sr.plot_spatial.plot_event_gnn_predictions(
        gnn_model_dir,
        gnn_results_ffp,
        gnn_out_dir,
        ims,
        map_data=map_data,
        region=region,
    )

    print("Plotting marginal GMM predictions")
    sr.plot_spatial.plot_event_gmm_predictions(
        emp_gmm_params_ffp,
        nzgmdb_ffp,
        event_id,
        emp_gmm_out_dir,
        ims,
        map_data=map_data,
        region=region,
    )

    print("Plotting cIM predictions")
    sr.plot_spatial.plot_event_cim_predictions(
        cim_results_ffp,
        nzgmdb_ffp,
        event_id,
        cim_out_dir,
        ims,
        map_data=map_data,
        region=region,
    )


if __name__ == "__main__":
    app()
