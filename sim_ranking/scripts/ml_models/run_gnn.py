from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import typer

import sim_ranking as sr
from source_modelling.srf import read_srf
from qcore import src_site_dist
from tqdm import tqdm

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


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
    srf_ffp: Path,
    out_ffp: Path,
    non_uniform_site_dir: Path = None,
    grid_site_ffp: Path = None,
    emp_gm_params_ffp: Path = None,
    allow_self: bool = True,
    batch_size: int = None,
):
    if non_uniform_site_dir is None and grid_site_ffp is None:
        raise ValueError(
            "Either non-uniform site dir or grid site file must be provided"
        )

    event_id = "3468575"

    # Prediction site data
    if non_uniform_site_dir is not None:
        region = sr.constants.CANTERBURY_REGION
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
    else:
        pred_site_df = pd.read_parquet(grid_site_ffp)
        # Drop any values with nan
        nan_mask = pred_site_df.isna().any(axis=1)
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        pred_site_df = pred_site_df[~nan_mask]

    # Compute rrup
    srf = read_srf(srf_ffp)
    loc_values = pred_site_df[["lon", "lat"]].values
    loc_values = np.hstack((loc_values, np.zeros((loc_values.shape[0], 1))))
    srf_points = srf.points[["lon", "lat", "dep"]].values
    rrup, _ = src_site_dist.calc_rrup_rjb(srf_points, loc_values)
    pred_site_df["rrup"] = rrup

    assert pred_site_df.z1p0.max() > 5, "Ensure Z1.0 is in metres"
    assert pred_site_df.z2p5.max() < 15, "Ensure Z2.5 is in kilometres"

    # Observation data
    run_config = sr.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")
    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)

    emp_gm_params, obs_emp_res_df = None, None
    if run_config.use_emp_gm_model:
        assert emp_gm_params_ffp is not None, "emp_gm_params_ffp must be provided"

        # Load empirical GMM residuals for observation sites
        _, obs_emp_res_df = sr.analysis.load_emp_gm_params_res(
            run_config.emp_gm_params_ffp, obs_data
        )
        # Load empirical GMM parameters for prediction sites
        emp_gm_params = pd.read_parquet(emp_gm_params_ffp)

    # Run prediction
    if batch_size is None:
        result_df = sr.ml.predict_event(
            model_dir,
            event_id,
            obs_data.event_df.loc[event_id],
            pred_site_df,
            obs_data.site_df.loc[obs_data.event_sites[event_id]],
            obs_data.record_df[["event_id", "site_id", "rrup"]],
            obs_data.record_df[sr.constants.IMs + ["event_id", "site_id"]],
            emp_gm_params=emp_gm_params,
            obs_emp_res_df=obs_emp_res_df,
            allow_self=allow_self,
        )
    else:
        n_batches = int(np.ceil(pred_site_df.shape[0] / batch_size))
        result_df = []

        for i in tqdm(range(n_batches)):
            cur_pred_site_df = pred_site_df.iloc[i * batch_size : (i + 1) * batch_size]
            cur_result_df = sr.ml.predict_event(
                model_dir,
                event_id,
                obs_data.event_df.loc[event_id],
                cur_pred_site_df,
                obs_data.site_df.loc[obs_data.event_sites[event_id]],
                obs_data.record_df[["event_id", "site_id", "rrup"]],
                obs_data.record_df[sr.constants.IMs + ["event_id", "site_id"]],
                emp_gm_params=emp_gm_params,
                obs_emp_res_df=obs_emp_res_df,
                allow_self=allow_self,
                verbose=False,
            )

            result_df.append(cur_result_df)

        result_df = pd.concat(result_df, axis=0)


    result_df.to_parquet(out_ffp)


@app.command("copy-cim-cv-results")
def copy_cim_cv_results(
    src_dir: Path,
    dest_dir: Path,
):
    sr.ml.data.copy_cim_cv_results(src_dir, dest_dir)


if __name__ == "__main__":
    app()
