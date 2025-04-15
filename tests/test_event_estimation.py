import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
import numpy.testing as npt

import sim_ranking as sr

wdata = Path(os.environ.get("wdata"))
config_ffp = Path(__file__).parent / "test_config.yaml"
event_est_config = yaml.safe_load(config_ffp.read_text())["gnn_event_estimation_config"]


@pytest.fixture(scope="module", params=event_est_config["model_dirs"])
def model_dir(request):
    return wdata / request.param

@pytest.fixture(scope="module")
def train_results(model_dir):
    return pd.read_parquet(model_dir / "train_results.parquet")

@pytest.fixture(scope="module")
def obs_data(run_config: sr.ml.RunConfig):
    return sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)

@pytest.fixture(scope="module")
def run_config(model_dir):
    return sr.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")

@pytest.fixture(scope="module")
def emp_gm_data(run_config: sr.ml.RunConfig, obs_data: sr.ObservedData):
    if run_config.use_emp_gm_model:
        return sr.analysis.load_emp_gm_params_res(run_config.emp_gm_params_ffp, obs_data)
    return None

@pytest.mark.parametrize("event_id", event_est_config["events"])
def test_event_estimation(event_id: str, model_dir: Path, run_config: sr.ml.RunConfig, train_results: pd.DataFrame, obs_data: sr.ObservedData, emp_gm_data: tuple[pd.DataFrame, pd.DataFrame] | None):
    train_results = train_results.loc[train_results.event_id == event_id]
    event_data = obs_data.get_event_data(event_id)

    pred_site_df = event_data.loc[train_results.site_int, ["vs30", "z1p0", "z2p5", "rrup"]].copy()
    pred_site_df[["lon", "lat"]] = obs_data.site_df.loc[train_results.site_int, ["lon", "lat"]]

    emp_gm_params, obs_emp_res_df = emp_gm_data if emp_gm_data else (None, None)

    result_df = sr.ml.predict_event(
        model_dir,
        event_id,
        obs_data.event_df.loc[event_id],
        pred_site_df,    
        obs_data.site_df.loc[obs_data.event_sites[event_id]],    
        obs_data.record_df[["event_id", "site_id", "rrup"]],
        obs_data.record_df[run_config.ims + ["event_id", "site_id"]],
        emp_gm_params=emp_gm_params,
        obs_emp_res_df=obs_emp_res_df,
        allow_self=False
    )

    pred_cols = [f"{im}_pred" for im in run_config.ims]
    pred_std_cols = [f"{im}_pred_std" for im in run_config.ims]

    npt.assert_allclose(result_df[pred_cols], train_results[pred_cols], rtol=1e-3)
    npt.assert_allclose(result_df[pred_std_cols], train_results[pred_std_cols], rtol=1e-3)


# if __name__ == "__main__": 
    # test_scenario_estimation("sim_ranking/results/gnn/0218_2000_full_v4p2NZGMDB_v2p1GNN", "2016p860234_WEL")
    # test_event_estimation("3528839")
    
