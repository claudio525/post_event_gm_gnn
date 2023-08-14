from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer

import ml_tools as mlt
import sim_ranking as sr

app = typer.Typer()


@app.command("cmvn-emp")
def emp_cmvn_ranking(
    rupture: str,
    gm_params_ffp: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
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
    method_type = sr.constants.RankingMethod.emp_cMVN
    (
        output_dir := results_dir
        / sr.constants.METHOD_RESULT_DIR_NAME_MAPPING[method_type]
    ).mkdir(exist_ok=True)
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

    meta = dict(
        method_type=method_type.value,
        rupture=rupture,
        IMs=IMs,
        sim_imdb_ffp=str(sim_imdb_ffp),
        obs_data_ffp=str(obs_data_ffp),
        stations_ll_ffp=str(stations_ll_ffp),
        gm_params_ffp=str(gm_params_ffp),
    )
    mlt.utils.write_to_yaml(meta, output_dir / "meta.yaml")


@app.command("cmvn-sim")
def sim_cmvn_ranking(
    rupture: str,
    sim_gm_params_dir: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
    corr_dir: Path,
    IMs: List[str] = None,
):
    """
    Performs simulation ranking based on
    simulation conditional MVN
    """
    run_sim_cmvn_ranking(
        sr.constants.RankingMethod.sim_cMVN,
        rupture,
        sim_gm_params_dir,
        obs_data_ffp,
        stations_ll_ffp,
        sim_imdb_ffp,
        results_dir,
        corr_dir=corr_dir,
        IMs=IMs,
    )


@app.command("cmvn-sim-emp-corr")
def sim_cmvn_ranking_emp_corr(
    rupture: str,
    sim_gm_params_dir: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
    IMs: List[str] = None,
):
    """
    Same as cmvn-sim but uses correlation
    coefficients from the empirical model
    """
    run_sim_cmvn_ranking(
        sr.constants.RankingMethod.sim_cMVN_emp_corr,
        rupture,
        sim_gm_params_dir,
        obs_data_ffp,
        stations_ll_ffp,
        sim_imdb_ffp,
        results_dir,
        IMs=IMs,
    )


def run_sim_cmvn_ranking(
    method_type: sr.constants.RankingMethod,
    rupture: str,
    sim_gm_params_dir: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
    corr_dir: Path = None,
    IMs: List[str] = None,
):
    (
        output_dir := results_dir
        / sr.constants.METHOD_RESULT_DIR_NAME_MAPPING[method_type]
    ).mkdir(exist_ok=True)
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
    R = None
    if corr_dir is not None:
        # Load and convert to absolute values
        R = {
            # cur_im: cur_R.abs()
            cur_im: cur_R
            for cur_im, cur_R in sr.data.load_correlations(corr_dir).items()
        }

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

    # Save the meta data
    meta = dict(
        method_type=method_type.value,
        rupture=rupture,
        IMs=IMs,
        sim_gm_params_dir=str(sim_gm_params_dir),
        obs_data_ffp=str(obs_data_ffp),
        stations_ll_ffp=str(stations_ll_ffp),
        sim_imdb_ffp=str(sim_imdb_ffp),
        corr_dir=str(corr_dir) if corr_dir is not None else None,
    )

    mlt.utils.write_to_yaml(meta, output_dir / "meta.yaml")


if __name__ == "__main__":
    app()
