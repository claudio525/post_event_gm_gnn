import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sim_ranking as sr

wdata = Path(os.environ.get("wdata"))

config_ffp = Path(__file__).parent / "test_config.yaml"
config = yaml.safe_load(config_ffp.read_text())["compute_emp_gm_params_config"]


def test_nzgmdb_emp_gm_params(tmpdir: Path):
    # Load benchmark data
    benchmark_emp_gm_params_ffp = wdata / config["benchmark_emp_gm_params_ffp"]
    bench_emp_gm_params = pd.read_parquet(benchmark_emp_gm_params_ffp)

    nzgmdb_ffp = wdata / config["nzgmdb_ffp"]
    output_ffp = tmpdir / "test.parquet"

    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    sr.data.compute_nzgmdb_emp_gm_params(
        output_ffp,
        obs_data,
        config["max_rjb"],
    )

    pd.testing.assert_frame_equal(bench_emp_gm_params, pd.read_parquet(output_ffp))

    
