from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer


import gmhazard_calc as gc

import sim_ranking as sr

app = typer.Typer()


@app.command("cmvn-emp")
def emp_cmvn_ranking(
    rupture: str,
    gm_params_ffp: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    output_dir: Path,
    IMs: List[str] = None,
):
    """
    Performs simulation ranking based on
    empirical conditional MVN

    Note: Currently computes it for all given
    observation sites

    To add:
     - Support for specifying sites of interest
     - Support for setting IM weights
    """
    assert len(list(output_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station data
    stations_df = pd.read_csv(
        stations_ll_ffp, sep=" ", index_col=2, header=None, names=["lon", "lat"]
    )

    # IMs to use for ranking
    IMs = (
        [f"pSA_{cur_period}" for cur_period in sr.constants.PERIODS]
        if len(IMs) == 0
        else IMs
    )

    # Get GMM parameters
    gm_params_df = pd.read_csv(gm_params_ffp, index_col=0, dtype={"event": str})
    gm_params_df = gm_params_df.loc[gm_params_df.event == rupture]
    gm_params_df = gm_params_df.set_index("site").sort_index()

    # Loading Observations
    obs_df = sr.data.load_obs_rupture_data(obs_data_ffp, rupture)

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # Load the simulation IM data
    sim_data = sr.data.load_sim_data(sim_imdb_ffp, int_stations)

    # Run the conditional MVN based ranking
    sr.conditional_MVN.run_conditional_mvn_ranking(
        output_dir, stations_df, IMs, gm_params_df, sim_data, obs_df, int_stations
    )


@app.command("cmvn-sim")
def sim_cmvn_ranking(
    rupture: str,
    sim_gm_params_dir: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    output_dir: Path,
    corr_dir: Path,
    IMs: List[str] = None,
):
    """
    Performs simulation ranking based on
    simulation conditional MVN
    """
    assert len(list(output_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station data
    stations_df = pd.read_csv(
        stations_ll_ffp, sep=" ", index_col=2, header=None, names=["lon", "lat"]
    )

    # Get simulation GM parameters
    sim_gm_params = sr.data.SimGMParams.load(sim_gm_params_dir)

    # Loading Observations
    obs_df = sr.data.load_obs_rupture_data(obs_data_ffp, rupture)

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # Load the simulation IM data
    sim_data = sr.data.load_sim_data(sim_imdb_ffp, int_stations)

    # IMs to use for ranking
    IMs = (
        [f"pSA_{cur_period}" for cur_period in sr.constants.PERIODS]
        if len(IMs) == 0
        else IMs
    )

    # Load the within-event site correlations
    R = None if corr_dir is None else sr.data.load_correlations(corr_dir)

    # Need the absolute value
    R = {cur_im: cur_R.abs() for cur_im, cur_R in R.items()}

    # Run the conditional MVN based ranking
    sr.conditional_MVN.run_conditional_mvn_ranking(
        output_dir,
        stations_df,
        IMs,
        sim_gm_params.gm_params,
        sim_data,
        obs_df,
        int_stations,
        R=R,
    )


if __name__ == "__main__":
    app()
