from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import typer

import sim_ranking as sr
from source_modelling.srf import read_srf
from qcore import src_site_dist

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("train-holdout")
def run_holdout(
    run_config_ffp: Path,
    holdout_config_ffp: Path,
    n_epochs: int = None,
    id_suffix: str = "",
):
    sr.ml.run_holdout(
        run_config_ffp,
        holdout_config_ffp,
        n_epochs=n_epochs,
        id_suffix=id_suffix,
        device=device,
    )


@app.command("train-cv")
def run_cv(
    run_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int = None,
    id_suffix: str = "",
    n_procs: int = mp.cpu_count(),
):
    mp.set_start_method("spawn")

    sr.ml.run_cv(
        run_config_ffp,
        n_event_folds,
        n_site_folds,
        n_epochs=n_epochs,
        id_suffix=id_suffix,
        n_procs=n_procs,
        device=device,
    )


@app.command("train-full")
def run_full(
    output_dir: Path,
    run_config_ffp: Path,
    n_epochs: int,
):
    mp.set_start_method("spawn")

    sr.ml.run_full(
        output_dir,
        run_config_ffp,
        n_epochs,
        device=device,
    )


@app.command("predict-event-3468575")
def predict_event_3468575(
    model_dir: Path,
    non_uniform_site_dir: Path,
    srf_ffp: Path,
    out_ffp: Path,
    allow_self: bool = True,
):
    region = sr.constants.CANTERBURY_REGION
    event_id = "3468575"

    # Prediction site data
    pred_site_df = sr.data.load_non_uniform_grid(
        non_uniform_site_dir
    )
    region_mask = (
        (pred_site_df["lon"] >= region[0])
        & (pred_site_df["lon"] <= region[1])
        & (pred_site_df["lat"] >= region[2])
        & (pred_site_df["lat"] <= region[3])
    )
    pred_site_df = pred_site_df.loc[region_mask]

    # Compute rrup
    srf = read_srf(srf_ffp)
    loc_values = pred_site_df[["lon", "lat"]].values
    loc_values = np.hstack((loc_values, np.zeros((loc_values.shape[0], 1))))
    srf_points = srf.points[["lon", "lat", "dep"]].values
    rrup, _ = src_site_dist.calc_rrup_rjb(srf_points, loc_values)
    pred_site_df["rrup"] = rrup

    # Observation data
    run_config = sr.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")
    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)

    # Run prediction
    result_df = sr.ml.predict_event(
        model_dir,
        event_id,
        obs_data.event_df.loc[event_id],
        pred_site_df,
        obs_data.site_df.loc[obs_data.event_sites[event_id]],
        obs_data.record_df[["event_id", "site_id", "rrup"]],
        obs_data.record_df[sr.constants.IMs + ["event_id", "site_id"]],
        allow_self=allow_self,
    )

    result_df.to_parquet(out_ffp)


@app.command("continue-hp-opt")
def continue_hp_opt(
    rel_results_dir: str,
    n_trials: int,
):
    mp.set_start_method("spawn")

    sr.ml.gnn_hp.continue_hp_opt(
        rel_results_dir,
        n_trials,
    )


@app.command("run-hp-opt")
def run_hp_opt(
    base_run_config_ffp: Path,
    hp_opt_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    rel_results_dir: str,
    n_trials: int,
    n_epochs: int,
    n_procs: int = mp.cpu_count(),
):
    mp.set_start_method("spawn")

    sr.ml.gnn_hp.run_hp_opt(
        base_run_config_ffp,
        hp_opt_config_ffp,
        n_event_folds,
        n_site_folds,
        rel_results_dir,
        n_trials,
        n_epochs,
        n_procs,
        device,
    )

@app.command("copy-cim-cv-results")
def copy_cim_cv_results(
    src_dir: Path,
    dest_dir: Path,
):
    sr.ml.data.copy_cim_cv_results(src_dir, dest_dir)


if __name__ == "__main__":
    app()
