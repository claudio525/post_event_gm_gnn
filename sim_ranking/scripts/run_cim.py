import os.path
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import typer
from tqdm import tqdm

import ml_tools as mlt
import sim_ranking as sr

app = typer.Typer()


@app.command("cmvn-emp")
def emp_cmvn(
    rupture: str,
    rel_gmm_params_ffp: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    min_n_obs_stations: int = 5,
    n_stations: int = 20,
    IMs: List[str] = None,
    quiet: bool = False,
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
    obs_site_sel_params = {
        "min_n_obs_stations": min_n_obs_stations,
        "n_stations": n_stations,
        # "max_obs_dist": max_obs_dist,
        # "min_n_obs": min_n_obs,
    }

    run_emp_cmvn(
        rupture,
        rel_gmm_params_ffp,
        rel_db_ffp,
        results_dir,
        obs_site_sel_params,
        IMs=IMs,
        quiet=quiet,
    )


@app.command("cmvn-emp-all")
def emp_cmvn_all(
    rel_gm_params_ffp: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    val_int_sites_ffp: Path = None,
    data_source: str = None,
    IMs: List[str] = None,
    min_n_obs_stations: int = 5,
    n_stations: int = 20,
    quiet: bool = True,
):
    # Find all events for which empirical and simulation data is available
    db = sr.db.DB(Path(os.path.expandvars("$wdata")) / rel_db_ffp)
    sim_events = db.get_avail_events(data_source=data_source)
    obs_events = db.get_obs_df().event_id.unique().astype(str)

    events = np.intersect1d(sim_events, obs_events)

    obs_site_sel_params = {
        "min_n_obs_stations": min_n_obs_stations,
        "n_stations": n_stations,
        # "max_obs_dist": max_obs_dist,
        # "min_n_obs": min_n_obs,
    }

    for cur_event in tqdm(events):
        run_emp_cmvn(
            str(cur_event),
            rel_gm_params_ffp,
            rel_db_ffp,
            results_dir / cur_event,
            obs_site_sel_params,
            val_int_sites_ffp=val_int_sites_ffp,
            IMs=IMs,
            quiet=quiet,
        )


def run_emp_cmvn(
    rupture: str,
    rel_gm_params_ffp: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    obs_site_sel_params: Dict[str, Any],
    val_int_sites_ffp: Path = None,
    IMs: List[str] = None,
    quiet: bool = False,
):
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp
    gm_params_ffp = Path(os.path.expandvars("$wdata")) / rel_gm_params_ffp

    method_type = sr.constants.RankingMethod.emp_cMVN
    (
        output_dir := results_dir
        / sr.constants.METHOD_RESULT_DIR_NAME_MAPPING[method_type]
    ).mkdir(exist_ok=True, parents=True)
    assert len(list(output_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station & event data
    db = sr.db.DB(db_ffp)
    stations_df = db.get_site_df()
    event_df = db.get_event_df()

    # IMs to use for ranking
    IMs = (
        [f"pSA_{cur_period}" for cur_period in sr.constants.PERIODS]
        if IMs is None or len(IMs) == 0
        else IMs
    )
    im_weights = sr.constants.IM_WEIGTHS_SETS["pSA"]

    # Get GMM parameters
    gm_params_df = pd.read_csv(gm_params_ffp, index_col=0, dtype={"event": str})
    gm_params_df = gm_params_df.loc[gm_params_df.event == rupture]
    gm_params_df = gm_params_df.set_index("site").sort_index()

    # Loading Observations
    obs_df = db.get_obs_data(rupture, stations_df.index.values)
    obs_df = obs_df.sort_index()

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # If val_int_sites_ffp is not None,
    # then the sites in the file are not used
    # as observation sites
    if val_int_sites_ffp is not None:
        val_int_sites = np.load(val_int_sites_ffp)
        mask = obs_df.index.isin(val_int_sites)
        if not quiet:
            print(
                f"Excluding {mask.sum()} sites from the analysis"
                f" as specified by val_int_sites"
            )
        obs_df = obs_df.loc[~mask]

    # Load the simulation IM data
    sim_data = db.get_sim_data(rupture, int_stations)
    sim_data = {cur_site: cur_g for cur_site, cur_g in sim_data.groupby("site_id")}

    # Run the conditional MVN based ranking
    sr.conditional.run_conditional_mvn_ranking(
        output_dir,
        stations_df,
        IMs,
        im_weights,
        gm_params_df,
        sim_data,
        obs_df,
        int_stations,
        sr.utils.SourceInfo(
            rupture, tuple(event_df.loc[rupture, ["lon", "lat"]].values)
        ),
        obs_site_sel_params,
        verbose=not quiet,
    )

    meta = dict(
        method_type=method_type.value,
        rupture=rupture,
        IMs=IMs,
        db_ffp=str(rel_db_ffp),
        gm_params_ffp=str(rel_gm_params_ffp),
        obs_site_sel_params=obs_site_sel_params,
    )
    mlt.utils.write_to_yaml(meta, output_dir / "meta.yaml")


@app.command("cmvn-sim-all")
def sim_cmvn_all(
    rel_sim_gm_params_dir: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    rel_corr_dir: Path,
    data_source: str,
    suffix: str = None,
    val_int_sites_ffp: Path = None,
    IMs: List[str] = None,
    min_n_obs_stations: int = 5,
    n_obs_stations: int = 20,
    quiet: bool = True,
):
    """
    Performs simulation ranking based on
    simulation conditional MVN for
    all ruptures with data
    """
    # Create run_id
    run_id = mlt.utils.create_run_id()
    if suffix is not None:
        run_id = f"{run_id}{suffix}"
    results_dir = results_dir / run_id
    results_dir.mkdir(parents=False, exist_ok=False)

    obs_site_sel_params = {
        "min_n_obs_sites": min_n_obs_stations,
        "n_obs_sites": n_obs_stations,
        # "max_obs_dist": max_obs_dist,
        # "min_n_obs": min_n_obs,
    }

    # Find all events for which empirical and simulation data is available
    db = sr.db.DB(Path(os.path.expandvars("$wdata")) / rel_db_ffp)
    sim_events = db.get_avail_events(data_source=data_source)
    obs_events = db.get_obs_df().event_id.unique().astype(str)

    events = np.intersect1d(sim_events, obs_events)

    for cur_event in tqdm(events):
        run_sim_cmvn(
            sr.constants.RankingMethod.sim_cMVN,
            str(cur_event),
            rel_sim_gm_params_dir / cur_event,
            rel_db_ffp,
            results_dir / cur_event,
            obs_site_sel_params,
            val_int_sites_ffp=val_int_sites_ffp,
            rel_corr_dir=rel_corr_dir,
            IMs=IMs,
            quiet=quiet,
        )


@app.command("cmvn-sim")
def sim_ranking(
    rupture: str,
    rel_sim_gm_params_dir: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    rel_corr_dir: Path,
    IMs: List[str] = None,
    min_n_obs_stations: int = 5,
    n_obs_stations: int = 20,
    quiet: bool = False,
):
    """
    Performs simulation ranking based on
    simulation conditional MVN for the
    specified rupture
    """
    obs_site_sel_params = {
        "min_n_obs_stations": min_n_obs_stations,
        "n_stations": n_obs_stations,
        # "max_obs_dist": max_obs_dist,
        # "min_n_obs": min_n_obs,
    }

    run_sim_cmvn(
        sr.constants.RankingMethod.sim_cMVN,
        rupture,
        rel_sim_gm_params_dir,
        rel_db_ffp,
        results_dir,
        obs_site_sel_params,
        rel_corr_dir=rel_corr_dir,
        IMs=IMs,
        quiet=quiet,
    )


def run_sim_cmvn(
    method_type: sr.constants.RankingMethod,
    rupture: str,
    rel_sim_gm_params_dir: Path,
    rel_db_ffp: Path,
    results_dir: Path,
    obs_site_sel_params: Dict[str, Any],
    val_int_sites_ffp: Path = None,
    rel_corr_dir: Path = None,
    IMs: List[str] = None,
    quiet: bool = False,
):
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp
    sim_gm_params_dir = Path(os.path.expandvars("$wdata")) / rel_sim_gm_params_dir
    corr_dir = (
        None
        if rel_corr_dir is None
        else Path(os.path.expandvars("$wdata")) / rel_corr_dir
    )

    results_dir.mkdir(parents=False, exist_ok=True)
    assert len(list(results_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station & event data
    db = sr.db.DB(db_ffp)
    stations_df = db.get_site_df()
    event_df = db.get_event_df()

    # Get simulation GM parameters
    sim_gm_params = sr.data.SimGMParams.load(sim_gm_params_dir)

    # Loading Observations
    obs_df = db.get_obs_data(rupture, stations_df.index.values)
    obs_df = obs_df.sort_index()

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # If val_int_sites_ffp is not None,
    # then the sites in the file are not used
    # as observation sites
    if val_int_sites_ffp is not None:
        val_int_sites = np.load(val_int_sites_ffp)
        mask = obs_df.index.isin(val_int_sites)
        if not quiet:
            print(
                f"Excluding {mask.sum()} sites from the analysis"
                f" as specified by val_int_sites"
            )
        obs_df = obs_df.loc[~mask]

    # Load the simulation IM data
    sim_data = db.get_sim_data(rupture, int_stations)
    sim_data = {cur_site: cur_g for cur_site, cur_g in sim_data.groupby("site_id")}

    # IMs to use
    IMs = (
        [f"pSA_{cur_period}" for cur_period in sr.constants.PERIODS]
        if len(IMs) == 0
        else IMs
    )
    im_weights = sr.constants.IM_WEIGTHS_SETS["pSA"]

    # Load the within-event site correlations
    R = None
    if corr_dir is not None:
        R = sr.data.load_correlations(corr_dir)[rupture].to_im_dict()

    # Run the conditional MVN based ranking
    sr.conditional.run_conditional_mvn_ranking(
        results_dir,
        stations_df,
        IMs,
        im_weights,
        sim_gm_params.gm_params,
        sim_data,
        obs_df,
        int_stations,
        sr.utils.SourceInfo(
            rupture, tuple(event_df.loc[rupture, ["lon", "lat"]].values)
        ),
        obs_site_sel_params,
        R=R,
        verbose=not quiet,
    )

    # Save the meta data
    meta = dict(
        method_type=method_type.value,
        rupture=rupture,
        IMs=IMs,
        sim_gm_params_dir=str(sim_gm_params_dir),
        db_ffp=str(rel_db_ffp),
        corr_dir=str(rel_corr_dir) if corr_dir is not None else None,
        obs_site_sel_params=obs_site_sel_params,
    )

    mlt.utils.write_to_yaml(meta, results_dir / "meta.yaml")





if __name__ == "__main__":
    app()


# @app.command("cmvn-sim-emp-corr-all")
# def sim_cmvn_emp_corr_all(
#     sim_gm_params_dir: Path,
#     obs_data_ffp: Path,
#     stations_ll_ffp: Path,
#     sim_imdb_ffp: Path,
#     results_dir: Path,
#     IMs: List[str] = None,
#     n_stations: int = 20,
# ):
#     """
#     Same as cmvn-sim-all but uses correlation
#     coefficients from the empirical model
#     Runs for all ruptures with data
#     """
#     events = [
#         cur_dir.stem
#         for cur_dir in sim_gm_params_dir.iterdir()
#         if cur_dir.is_dir() and not cur_dir.stem.startswith("_")
#     ]
#
#     for cur_event in events:
#         run_sim_cmvn(
#             sr.constants.RankingMethod.sim_cMVN_emp_corr,
#             cur_event,
#             sim_gm_params_dir / cur_event,
#             obs_data_ffp,
#             stations_ll_ffp,
#             sim_imdb_ffp,
#             results_dir / cur_event,
#             IMs=IMs,
#             n_obs_stations=n_stations,
#         )


# @app.command("cmvn-sim-emp-corr")
# def sim_cmvn_ranking_emp_corr(
#     rupture: str,
#     sim_gm_params_dir: Path,
#     obs_data_ffp: Path,
#     stations_ll_ffp: Path,
#     sim_imdb_ffp: Path,
#     results_dir: Path,
#     IMs: List[str] = None,
#     n_stations: int = 20,
# ):
#     """
#     Same as cmvn-sim but uses correlation
#     coefficients from the empirical model
#     """
#     run_sim_cmvn(
#         sr.constants.RankingMethod.sim_cMVN_emp_corr,
#         rupture,
#         sim_gm_params_dir,
#         obs_data_ffp,
#         stations_ll_ffp,
#         sim_imdb_ffp,
#         results_dir,
#         IMs=IMs,
#         n_obs_stations=n_stations,
#     )
