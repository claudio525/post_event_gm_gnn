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
    sr.conditional.run_cim_for_GNN(
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
    sr.conditional.run_cim_for_CV_GNN(
        gnn_cv_results_dir,
        emp_gm_params_ffp,
        n_procs=n_procs,
        include_train=include_train,
    )

@app.command("predict-event-cIM")
def predict_event_cIM(
    event_id: str,
    nzgmdb_ffp: Path,
    nzgmdb_emp_gm_params_ffp: Path,
    non_uniform_grid_dir: Path,
    non_uniform_emp_gm_params_ffp: Path,
    max_rjb: float,
    output_ffp: Path,
):
    """Compute single event conditional IM distribution for the non-uniform grid sites"""
    non_uniform_site_df = sr.data.load_non_uniform_grid(
        non_uniform_grid_dir
    )
    obs_data = sr.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)

    # Load GM params
    grid_gm_params_df = pd.read_parquet(non_uniform_emp_gm_params_ffp)
    nzgmdb_gm_params = pd.read_parquet(nzgmdb_emp_gm_params_ffp)
    nzgmdb_gm_params = nzgmdb_gm_params.loc[nzgmdb_gm_params.event_id == event_id]

    # Combine the empirical GM params
    gm_params_df = pd.concat(
        [nzgmdb_gm_params, grid_gm_params_df.loc[~np.isin(grid_gm_params_df.index, nzgmdb_gm_params.index)]], axis=0
    )

    # Ignore observation sites that exceed the max RJB
    obs_event_data = obs_data.get_event_data(event_id)
    obs_sites = obs_event_data.loc[obs_event_data.rjb < max_rjb].index.values.astype(str)

    sr.conditional.predict_event_cIM(
        event_id,
        non_uniform_site_df,
        obs_data,
        obs_sites,
        gm_params_df,
        grid_gm_params_df["site_id"].values.astype(str),
        output_ffp
    )


if __name__ == "__main__":
    app()

