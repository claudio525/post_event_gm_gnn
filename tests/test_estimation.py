import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
import numpy.testing as npt

import sim_ranking as sr

wdata = Path(os.environ.get("wdata"))

config_ffp = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_ffp.read_text())


@pytest.mark.parametrize("event_id", config["events"])
def test_event_estimation(event_id: str):
    model_dir = wdata / config["benchmark_model_dir"]
    run_config = sr.ml.RunConfig.from_yaml(model_dir / "run_config.yaml")

    train_results = pd.read_parquet(model_dir / "train_results.parquet")
    train_results = train_results.loc[train_results.event_id == event_id]

    obs_data = sr.data.load_obs_nzgmdb(wdata / config["nzgmdb_ffp"])
    event_data = obs_data.get_event_data(event_id)

    pred_site_df = event_data.loc[train_results.site_int, ["vs30", "z1p0", "z2p5", "rrup"]].copy()
    pred_site_df[["lon", "lat"]] = obs_data.site_df.loc[train_results.site_int, ["lon", "lat"]]

    result_df = sr.ml.predict_event(
        model_dir,
        event_id,
        obs_data.event_df.loc[event_id],
        pred_site_df,    
        obs_data.site_df.loc[obs_data.event_sites[event_id]],    
        obs_data.record_df[["event_id", "site_id", "rrup"]],
        obs_data.record_df[sr.constants.IMs + ["event_id", "site_id"]],
    )

    pred_cols = [f"{im}_pred" for im in sr.constants.IMs]
    pred_std_cols = [f"{im}_pred_std" for im in sr.constants.IMs]
    std_cols = [f"{im}_std" for im in sr.constants.IMs]

    npt.assert_allclose(result_df[run_config.ims], train_results[pred_cols], rtol=1e-3)
    npt.assert_allclose(result_df[std_cols], train_results[pred_std_cols], rtol=1e-3)



# if __name__ == "__main__":
#     test_event_estimation("3528839")
    
