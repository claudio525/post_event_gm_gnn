# from pathlib import Path

# import pandas as pd
# import numpy as np

# import ml_tools as mlt
# import sim_ranking as sr

# def chiou_young_08_calc_z1p0(vs30):
#     """
#     Calculates the z2p5 value for the Chiou and Youngs (2008) model

#     Parameters
#     ----------
#     vs30 : Union[float, np.ndarray, pd.DataFrame]
#         The Vs30 value or values, in meters per second

#     Returns
#     -------
#     Union[float, np.ndarray]
#         The z1p0 value or values, in km
#     """
#     z1p0 = np.exp(28.5 - 3.82 / 8 * np.log(vs30**8 + 378.7**8)) / 1000  # In km
#     return z1p0

# # Combine the different dataframes
# data_dir = Path("/Users/claudy/dev/work/tmp/marginal_model_test/Output_v20p10p8_emp")

# gm_df = pd.read_csv(data_dir / "gm.csv", index_col=0)
# site_df = pd.read_csv(data_dir / "stations.csv", index_col=0)
# event_df = pd.read_csv(data_dir / "events.csv", index_col=0)
# obs_df = pd.read_csv(data_dir / "im_obs.csv", index_col=0)

# pSA_cols = [col for col in obs_df.columns if col.startswith("pSA")]
# periods = np.asarray([float(col.split("_")[1]) for col in pSA_cols])

# record_df = obs_df[["event_id", "stat_id"]].copy()
# record_df[["site_name", "vs30", "site_lon", "site_lat"]] = site_df.loc[record_df["stat_id"], ["stat_name", "vs30", "lon", "lat"]].values
# record_df[["event_name", "mag", "depth", "strike", "dip", "rake"]] = event_df.loc[record_df["event_id"], ["event_name", "mag", "hdepth", "strike", "dip", "rake"]].values
# record_df["rrup"] = gm_df.loc[record_df.index, "rrup"].values
# record_df["rjb"] = gm_df.loc[record_df.index, "rjbs"].values

# record_df["z1p0"] = chiou_young_08_calc_z1p0(record_df["vs30"].values.astype(float))

# record_df["rx"]  = record_df["rjb"]
# record_df["tect_type"] = "crustal"

# record_df["ztor"] = record_df["depth"]

# # record_df = record_df.drop(columns=["stat_id", "event_id"])
# # record_df = record_df.rename(columns={"site_name": "site_id", "event_name": "event_id"})
# record_df = record_df.rename(columns={"stat_id": "site_id"})

# # record_df["gm_id"] = record_df.index
# # record_df.index = mlt.array_utils.numpy_str_join("_", record_df["event_id"].values.astype(str), record_df["site_id"].values.astype(str))

# record_df.to_csv("/Users/claudy/dev/work/tmp/marginal_model_test/record.csv", index=True)

# sr.data.run_emp_gmms(
#     "/Users/claudy/dev/work/tmp/marginal_model_test/test.parquet",
#     "/Users/claudy/dev/work/tmp/marginal_model_test/record.csv",
#     500,
#     periods=periods
# )

# print("wtf")

# exit()

# ---------------------------------------------------------------------------------------------

from pathlib import Path

import pandas as pd
import numpy as np

import ml_tools as mlt
import sim_ranking as sr

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

obs_df = pd.read_csv("/Users/claudy/dev/work/tmp/marginal_model_test/Output_v20p10p8_emp/im_obs.csv", index_col=0).sort_index()

# Reference results
r_emp_df = pd.read_csv("/Users/claudy/dev/work/tmp/marginal_model_test/Output_v20p10p8_emp/im_sim.csv", index_col=0).sort_index()

assert obs_df.index.equals(r_emp_df.index)

r_pSA_cols = np.asarray([col for col in obs_df.columns if col.startswith("pSA")])
r_pred_pSA_cols = mlt.array_utils.numpy_str_join("_", r_pSA_cols, "pred")
periods = np.asarray([float(col.split("_")[1]) for col in r_pSA_cols])

r_results = np.log(r_emp_df[r_pSA_cols])
r_results.columns = r_pred_pSA_cols
r_results[r_pSA_cols] = np.log(obs_df[r_pSA_cols])
r_results["site_int"] = obs_df["stat_id"]
r_results["event_id"] = obs_df["event_id"]
r_results["n_obs_sites"] = -1
r_results = r_results.copy()

r_res = sr.ml.gnn_gm.get_residuals(r_results, ims=r_pSA_cols)

r_bias = r_res[r_pSA_cols].mean(axis=0)
r_std = r_res[r_pSA_cols].std(axis=0)

# My results
m_emp_df = pd.read_parquet("/Users/claudy/dev/work/tmp/marginal_model_test/test.parquet").sort_index()
m_emp_df = m_emp_df.rename(columns={cur_col: cur_col.replace("_mean", "") for cur_col in m_emp_df.columns if cur_col.endswith("_mean")})

assert m_emp_df.index.equals(obs_df.index)

m_pSA_cols = np.asarray([col for col in m_emp_df.columns if col.startswith("pSA") and "std" not in col])
m_pred_pSA_cols = mlt.array_utils.numpy_str_join("_", m_pSA_cols, "pred")

m_results = m_emp_df[["event_id", "site_id"]].copy()
m_results[m_pred_pSA_cols] = m_emp_df[m_pSA_cols]
m_results["site_int"] = m_results["site_id"]
m_results["n_obs_sites"] = -1
m_results = m_results.copy()

m_results[m_pSA_cols] = np.log(obs_df[m_pSA_cols])

m_res = sr.ml.gnn_gm.get_residuals(m_results, ims=m_pSA_cols)

m_bias = m_res[m_pSA_cols].mean(axis=0)
m_std = m_res[m_pSA_cols].std(axis=0)

robin_bias_df = pd.read_csv("/Users/claudy/dev/work/tmp/marginal_model_test/Output_v20p10p8_emp/Residuals/PJSvarCompsBiased_sim.csv", index_col=0)

fig, ax1, ax2, ax3, ax4 =  sr.plot_utils.get_bias_residual_fig()

ax1.semilogx(periods, r_bias, label="Reference")
ax1.semilogx(periods, m_bias, label="Mine")
ax1.semilogx(periods, robin_bias_df.loc[m_pSA_cols, "bias"], label="Robin")

ax3.semilogx(periods, r_std, label="Reference CS")
ax3.semilogx(periods, m_std, label="Mine")
ax3.semilogx(periods, robin_bias_df.loc[m_pSA_cols, "sigma"], label="Robin")

ax1.legend()

# plt.show()
fig.savefig("/Users/claudy/dev/work/tmp/marginal_model_test/plot.png")

print("wtf")


