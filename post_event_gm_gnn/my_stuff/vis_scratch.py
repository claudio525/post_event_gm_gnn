from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sim_ranking as sr

new_df = pd.read_parquet("/home/claudy/dev/work/data/sim_ranking/results/gnn/0214_1350_test_full_noTsite/train_results.parquet")
old_df = pd.read_parquet("/home/claudy/dev/work/data/sim_ranking/results/gnn/0408_1030_test_full_noTsite/train_results.parquet")


nzgmdb_ffp = Path("/home/claudy/dev/work/data/gm_datasets/nz_gmdb/v4.1/Tables/ground_motion_im_table_rotd50_flat.csv")

obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)
dist_matrix = sr.utils.calculate_distance_matrix(obs_data.sites, obs_data.site_df)

new_scenarios = new_df.index.values.astype(str)[~np.isin(new_df.index, old_df.index)]

added_df = new_df.loc[new_scenarios]


print("wtf")