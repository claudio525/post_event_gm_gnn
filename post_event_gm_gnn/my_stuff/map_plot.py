from pathlib import Path

import pandas as pd
import numpy as np

from pygmt_helper import plotting
from qcore import coordinates
from source_modelling.srf import read_srf

srf_ffp = Path("/Users/claudy/Downloads/rupture_1190.srf")
sites_ffp = Path(
    "/Users/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"
)

output_ffp = "./map_plot.png"


# srf_ffp = Path("/home/claudy/dev/work/tmp/srf/rupture_1190.srf")
# sites_ffp = Path(
#     "/home/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll")

srf = read_srf(srf_ffp)

srf_points = srf.points

# fault_nztm_coords = coordinates.wgs_depth_to_nztm(
#     srf.points[["lat", "lon", "dep"]].values
# )
map_data_ffp = Path("/Users/claudy/dev/work/code/qcore/qcore/data")
map_data = plotting.NZMapData.load(map_data_ffp, high_res_topo=False)
# map_data = None

region = (165.5, 178.5, -47.0, -37.0)

# Generate the figure
fig = plotting.gen_region_fig(
    region="NZ",
    # region=region,
    map_data=map_data,
    # plot_kwargs=dict(frame_args=["+n"]),
)

# Plot the srf points
fig.plot(
    x=srf_points["lon"].values[::5],
    y=srf_points["lat"].values[::5],
    style="c0.1c",
    fill="black",
    pen="0.1p,black",
)


# Plot the site
site_lon= 172.619876
site_lat = -43.529347

# fig.plot(
#     x=site_lon,
#     y=site_lat,
#     style="a0.5c",
#     fill="orange",
#     pen="0.1p,black",
# )

sites_df = pd.read_csv(sites_ffp, sep=" ", header=None, names=["lon", "lat"], index_col=2)

fig.plot(
    x=sites_df["lon"],
    y=sites_df["lat"],
    style="+0.1c",
    fill="red",
    pen="0.1p,red",
)

fig.savefig(output_ffp, dpi=900, anti_alias=True)
