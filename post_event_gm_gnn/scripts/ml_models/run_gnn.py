from pathlib import Path

import shap
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import typer

from source_modelling.srf import read_srf
from qcore import src_site_dist
from tqdm import tqdm

import post_event_gm_gnn as pg
import ml_tools as mlt

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
    """Runs cross-validation training for the GNN model"""
    mp.set_start_method("spawn")

    pg.ml.run_cv(
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
    """Trains a full model (i.e. using all data for training)"""
    mp.set_start_method("spawn")

    pg.ml.run_full(
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
    """Runs prediction for event 3468575 for given model result directory"""
    if non_uniform_site_dir is None and grid_site_ffp is None:
        raise ValueError(
            "Either non-uniform site dir or grid site file must be provided"
        )

    event_id = "3468575"

    # Prediction site data
    if non_uniform_site_dir is not None:
        region = pg.constants.CANTERBURY_REGION
        pred_site_df = pg.data.load_non_uniform_grid(non_uniform_site_dir)
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
    run_config = pg.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")
    obs_data = pg.data.load_obs_nzgmdb(run_config.obs_data_ffp)

    emp_gm_params, obs_emp_res_df = None, None
    if run_config.use_emp_gm_model:
        assert emp_gm_params_ffp is not None, "emp_gm_params_ffp must be provided"

        # Load empirical GMM residuals for observation sites
        _, obs_emp_res_df = pg.analysis.load_emp_gm_params_res(
            run_config.emp_gm_params_ffp, obs_data
        )
        # Load empirical GMM parameters for prediction sites
        emp_gm_params = pd.read_parquet(emp_gm_params_ffp)

    # Run prediction
    if batch_size is None:
        result_df = pg.ml.predict_event(
            model_dir,
            event_id,
            obs_data.event_df.loc[event_id],
            pred_site_df,
            obs_data.site_df.loc[obs_data.event_sites[event_id]],
            obs_data.record_df[["event_id", "site_id", "rrup"]],
            obs_data.record_df[pg.constants.IMs + ["event_id", "site_id"]],
            emp_gm_params=emp_gm_params,
            obs_emp_res_df=obs_emp_res_df,
            allow_self=allow_self,
        )
    else:
        n_batches = int(np.ceil(pred_site_df.shape[0] / batch_size))
        result_df = []

        for i in tqdm(range(n_batches)):
            cur_pred_site_df = pred_site_df.iloc[i * batch_size : (i + 1) * batch_size]
            cur_result_df = pg.ml.predict_event(
                model_dir,
                event_id,
                obs_data.event_df.loc[event_id],
                cur_pred_site_df,
                obs_data.site_df.loc[obs_data.event_sites[event_id]],
                obs_data.record_df[["event_id", "site_id", "rrup"]],
                obs_data.record_df[pg.constants.IMs + ["event_id", "site_id"]],
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
    """
    Copies cIM CV results from source to destination directory
    """
    pg.ml.data.copy_cim_cv_results(src_dir, dest_dir)


@app.command("get-att-SHAP-values")
def get_att_SHAP_values(model_dir: Path):
    """
    Gets attention SHAP values for a given model
    """
    run_config = pg.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")
    input_vars = run_config.graph_feature_keys["edge"]
    assert input_vars == ["dist", "angular_dist", "ln_vs30_diff"]

    att_models = [
        cur_conv.convs[("site_obs", "informs", "site_int")].att_model
        for cur_conv in torch.load(model_dir / "model.pt", map_location=device).convs
    ]
    # Only care about the first attention model,
    # as that is the only one that uses user-features.
    att_model = att_models[0]

    site_df = pg.data.load_obs_nzgmdb(run_config.obs_data_ffp).site_df

    dist_matrix = pg.utils.calculate_distance_matrix(site_df.index.values, site_df)

    # Identify valid site-pairs
    mask = ~np.diag(np.ones(dist_matrix.shape[0], dtype=bool)) & (
        dist_matrix.values <= run_config.max_dist
    )
    site_pairs_df = dist_matrix.where(mask).stack().reset_index()
    site_pairs_df.columns = ["site_int", "obs_site", "dist"]

    # Add ln_vs30_diff feature
    site_pairs_df["ln_vs30_diff"] = np.log(
        site_df.loc[site_pairs_df.site_int, "vs30"].values
    ) - np.log(site_df.loc[site_pairs_df.obs_site, "vs30"].values)

    # Scale distance feature
    site_pairs_df["dist"] = pg.ml.features.scale_site_to_site_distances(
        site_pairs_df["dist"].values, run_config.max_dist
    )

    # Select random angular distance values
    site_pairs_df["angular_dist"] = np.random.uniform(
        0, np.pi, size=site_pairs_df.shape[0]
    )
    site_pairs_df["angular_dist"] = pg.ml.features.scale_angular_distance(
        site_pairs_df["angular_dist"].values
    )
    input_df = site_pairs_df[input_vars]

    def _pred_fn(x: np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            return att_model(x).cpu().numpy()

    # Compute SHAP values for each attention model
    explainer = shap.KernelExplainer(_pred_fn, input_df)
    explainer_values = explainer(input_df)
    mlt.utils.write_pickle(
        explainer_values, model_dir / "att_shap_explainer_values.pkl", clobber=True
    )


if __name__ == "__main__":
    app()
