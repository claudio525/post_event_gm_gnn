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


@app.command("run-cIM-for-GNN")
def run_cim_for_GNN(
    gnn_result_dir: Path,
    emp_gm_params_ffp: Path,
    on_train: bool = False,
    n_procs: int = 1,
):
    """Runs conditional IM ranking for the GNN results"""
    sr.new_conditional.run_cim_for_GNN(
        gnn_result_dir, emp_gm_params_ffp, on_train=on_train, n_procs=n_procs
    )


@app.command("run-cIM-for-CV-GNN")
def run_cim_for_CV_GNN(
    gnn_cv_results_dir: Path,
    emp_gm_params_ffp: Path,
    n_procs: int = 1,
    include_train: bool = False,
):
    """Runs conditional IM ranking for the GNN CV results"""
    sr.new_conditional.run_cim_for_CV_GNN(
        gnn_cv_results_dir,
        emp_gm_params_ffp,
        n_procs=n_procs,
        include_train=include_train,
    )



@app.command("cmvn-emp")
def emp_cmvn_all(
    nzgmdb_flat_rel_ffp: Path,
    gm_params_rel_ffp: Path,
    results_dir: Path,
    val_int_sites_rel_ffp: Path = None,
    IMs: List[str] = None,
    min_n_obs_stations: int = 5,
    n_stations: int = 20,
    quiet: bool = True,
):
    nzgmdb_flat_ffp = Path(os.path.expandvars("$wdata")) / nzgmdb_flat_rel_ffp
    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_flat_ffp)

    gm_params_ffp = Path(os.path.expandvars("$wdata")) / gm_params_rel_ffp
    gm_params_df = pd.read_csv(gm_params_ffp, index_col=0, dtype={"event": str})

    events = np.intersect1d(
        obs_data.events.astype(str), gm_params_df.event.unique().astype(str)
    )

    val_int_sites = None
    if val_int_sites_rel_ffp is not None:
        val_int_sites = np.load(
            Path(os.path.expandvars("$wdata")) / val_int_sites_rel_ffp
        )

    obs_site_sel_params = {
        "min_n_obs_sites": min_n_obs_stations,
        "n_obs_sites": n_stations,
    }

    (output_dir := results_dir / mlt.utils.create_run_id()).mkdir(
        parents=False, exist_ok=False
    )
    for cur_event in tqdm(events):
        (cur_out_dir := output_dir / cur_event).mkdir(parents=False, exist_ok=False)
        run_emp_cmvn(
            str(cur_event),
            gm_params_df.copy(),
            obs_data,
            cur_out_dir,
            obs_site_sel_params,
            val_int_sites=val_int_sites,
            IMs=IMs,
            quiet=quiet,
        )

    meta = dict(
        IMs=IMs,
        nzgmdb_flat_rel_ffp=str(nzgmdb_flat_rel_ffp),
        gm_params_rel_ffp=str(gm_params_rel_ffp),
        val_int_sites_rel_ffp=str(val_int_sites_rel_ffp),
        obs_site_sel_params=obs_site_sel_params,
    )
    mlt.utils.write_to_yaml(meta, output_dir / "meta.yaml")


def run_emp_cmvn(
    rupture: str,
    gm_params_df: pd.DataFrame,
    obs_data: sr.ObservedData,
    output_dir: Path,
    obs_site_sel_params: Dict[str, Any],
    val_int_sites: np.ndarray = None,
    IMs: List[str] = None,
    quiet: bool = False,
):
    assert len(list(output_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station & event data
    # stations_df = obs_data.site_df
    # event_df = obs_data.event_df

    # IMs to use for ranking
    IMs = (
        [f"pSA_{cur_period}" for cur_period in sr.constants.PERIODS]
        if IMs is None or len(IMs) == 0
        else IMs
    )
    im_weights = sr.constants.IM_WEIGTHS_SETS["pSA"]

    # Get GMM parameters
    gm_params_df = gm_params_df.loc[gm_params_df.event == rupture]
    gm_params_df = gm_params_df.set_index("site").sort_index()

    # Loading Observations
    obs_df = obs_data.get_event_data(rupture)
    obs_df = obs_df.sort_index()

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # If val_int_sites_ffp is not None,
    # then the sites in the file are not used
    # as observation sites
    if val_int_sites is not None:
        mask = obs_df.index.isin(val_int_sites)
        if not quiet:
            print(
                f"Excluding {mask.sum()} sites from the analysis"
                f" as specified by val_int_sites"
            )
        obs_df = obs_df.loc[~mask]

    # Load the simulation IM data
    # sim_data = db.get_sim_data(rupture, int_stations)
    # sim_data = {cur_site: cur_g for cur_site, cur_g in sim_data.groupby("site_id")}

    # Run the conditional MVN based ranking
    sr.conditional.run_conditional_cIM(
        output_dir,
        obs_data.site_df,
        IMs,
        im_weights,
        gm_params_df,
        obs_df,
        int_stations,
        sr.utils.SourceInfo(
            rupture, tuple(obs_data.event_df.loc[rupture, ["lon", "lat"]].values)
        ),
        obs_site_sel_params,
        verbose=not quiet,
    )


if __name__ == "__main__":
    app()
