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
    run_emp_cmvn_ranking(
        rupture,
        gm_params_ffp,
        obs_data_ffp,
        stations_ll_ffp,
        sim_imdb_ffp,
        results_dir,
        IMs=IMs,
    )

@app.command("cmvn-emp-all")
def emp_cmvn_ranking_all(
    gm_params_ffp: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
    IMs: List[str] = None,
):
    # Find all events for which empirical and simualtion data is available
    sim_events = sr.data.load_avail_sim_events(sim_imdb_ffp)
    emp_events = np.unique(pd.read_csv(gm_params_ffp, index_col=0, dtype={"event": str}).event.values.astype(str))
    events = np.intersect1d(sim_events, emp_events)

    for cur_event in events:
        run_emp_cmvn_ranking(
            str(cur_event),
            gm_params_ffp,
            obs_data_ffp,
            stations_ll_ffp,
            sim_imdb_ffp,
            results_dir / cur_event,
            IMs=IMs,
        )


def run_emp_cmvn_ranking(
        rupture: str,
        gm_params_ffp: Path,
        obs_data_ffp: Path,
        stations_ll_ffp: Path,
        sim_imdb_ffp: Path,
        results_dir: Path,
        IMs: List[str] = None,
):
    method_type = sr.constants.RankingMethod.emp_cMVN
    (
        output_dir := results_dir
        / sr.constants.METHOD_RESULT_DIR_NAME_MAPPING[method_type]
    ).mkdir(exist_ok=True, parents=True)
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
    sim_data = sr.data.load_sim_data(sim_imdb_ffp, sites=int_stations, event=rupture)

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


@app.command("cmvn-sim-all")
def sim_cmvn_ranking_all(
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
    simulation conditional MVN for
    all ruptures with data
    """
    events = [
        cur_dir.stem
        for cur_dir in sim_gm_params_dir.iterdir()
        if cur_dir.is_dir() and not cur_dir.stem.startswith("_")
    ]

    for cur_event in events:
        run_sim_cmvn_ranking(
            sr.constants.RankingMethod.sim_cMVN,
            cur_event,
            sim_gm_params_dir / cur_event,
            obs_data_ffp,
            stations_ll_ffp,
            sim_imdb_ffp,
            results_dir / cur_event,
            corr_dir=corr_dir / cur_event,
            IMs=IMs,
        )


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
    simulation conditional MVN for the
    specified rupture
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


@app.command("cmvn-sim-emp-corr-all")
def sim_cmvn_ranking_emp_corr_all(
    sim_gm_params_dir: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    results_dir: Path,
    IMs: List[str] = None,
):
    """
    Same as cmvn-sim-all but uses correlation
    coefficients from the empirical model
    Runs for all ruptures with data
    """
    events = [
        cur_dir.stem
        for cur_dir in sim_gm_params_dir.iterdir()
        if cur_dir.is_dir() and not cur_dir.stem.startswith("_")
    ]

    for cur_event in events:
        run_sim_cmvn_ranking(
            sr.constants.RankingMethod.sim_cMVN_emp_corr,
            cur_event,
            sim_gm_params_dir / cur_event,
            obs_data_ffp,
            stations_ll_ffp,
            sim_imdb_ffp,
            results_dir / cur_event,
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
    ).mkdir(exist_ok=True, parents=True)
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
    sim_data = sr.data.load_sim_data(sim_imdb_ffp, sites=int_stations, event=rupture)

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
