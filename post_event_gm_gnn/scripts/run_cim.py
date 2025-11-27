from pathlib import Path

import pandas as pd
import numpy as np
import typer

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
    site_emp_gm_params_ffp: Path,
    output_ffp: Path,
    non_uniform_grid_dir: Path = None,
    uniform_site_ffp: Path = None,
    allow_self: bool = False,
):
    """Compute single event conditional IM distribution for the non-uniform grid sites"""
    assert (
        non_uniform_grid_dir is not None or uniform_site_ffp is not None
    ), "Either non-uniform grid dir or uniform site file must be provided"

    # Load the site data
    if non_uniform_grid_dir is not None:
        site_df = sr.data.load_non_uniform_grid(non_uniform_grid_dir)
    else:
        site_df = pd.read_parquet(uniform_site_ffp)

    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    # Load GM params
    grid_gm_params_df = pd.read_parquet(site_emp_gm_params_ffp)
    nzgmdb_gm_params = pd.read_parquet(nzgmdb_emp_gm_params_ffp)
    nzgmdb_gm_params = nzgmdb_gm_params.loc[nzgmdb_gm_params.event_id == event_id]

    nan_mask = grid_gm_params_df["PGA_mean"].isna()
    if np.any(nan_mask):
        print(f"Dropping {np.sum(nan_mask)} rows in grid_gm_params_df due to NaN values")
        grid_gm_params_df = grid_gm_params_df[~nan_mask]

    # Combine the empirical GM params
    gm_params_df = pd.concat(
        [
            nzgmdb_gm_params,
            grid_gm_params_df.loc[
                ~np.isin(grid_gm_params_df.index, nzgmdb_gm_params.index)
            ],
        ],
        axis=0,
    )

    sr.conditional.predict_event_cIM(
        event_id,
        site_df,
        obs_data,
        obs_data.get_event_data(event_id).index.values.astype(str),
        gm_params_df,
        site_df.index.values.astype(str),
        output_ffp,
        allow_self=allow_self,
    )


if __name__ == "__main__":
    app()
