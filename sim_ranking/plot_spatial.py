from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pygmt_helper import plotting

from . import utils
from . import constants
from . import ml
from . import data

IM_LIMITS_MAPPING = {
    "pSA_0.1": (0.0, 1.0, 0.05),
    "pSA_0.5": (0.0, 1.0, 0.05),
    "pSA_1.0": (0.0, 0.8, 0.04),
    "pSA_5.0": (0.0, 0.1, 0.005),
    "pSA_10.0": (0.0, 0.02, 0.001),
}


def plot_im_values(
    im_values: pd.DataFrame,
    im: str,
    event_info: pd.Series,
    obs_site_df: pd.DataFrame,
    region: tuple[float, float, float, float],
    map_data: plotting.NZMapData,
    output_ffp: str = None,
):
    grid = plotting.create_grid(
        im_values,
        "im_value",
        region=region,
    )

    # Create figure
    fig = plotting.gen_region_fig(
        region=region,
        map_data=map_data,
        plot_kwargs=dict(frame_args=["+n"]),
        # config_options=dict(
        #     MAP_FRAME_TYPE="plain",
        #     FORMAT_GEO_MAP="ddd.xx",
        #     MAP_FRAME_PEN="thinner,black",
        #     FONT_ANNOT_PRIMARY="6p,Helvetica,black",
        # ),
    )

    # Plot the IM values
    im_cmap_limits = IM_LIMITS_MAPPING[im]
    assert im_cmap_limits is not None, f"No cmaps limits set for IM {im}"
    plotting.plot_grid(
        fig,
        grid,
        "hot",
        im_cmap_limits,
        ("white", "black"),
        utils.get_nice_im_name(im),
        continuous_cmap=True,
        reverse_cmap=True,
        plot_contours=True,
        transparency=25,
    )

    # Plot the prediction sites
    fig.plot(
        x=im_values.lon.values,
        y=im_values.lat.values,
        style="c0.01c",
        fill="black",
        pen="0.1p,black",
    )

    # Plot the source
    fig.meca(
        spec=dict(
            strike=event_info.strike,
            dip=event_info.dip,
            rake=event_info.rake,
            magnitude=event_info.mag,
        ),
        scale=f"{0.04 * event_info.mag}c",
        longitude=event_info.lon,
        latitude=event_info.lat,
        depth=event_info.depth,
        G="red",
        W="0.05p,black,solid",
    )

    # Plot the observation sites
    fig.plot(
        x=obs_site_df["lon"].values,
        y=obs_site_df["lat"].values,
        style="t0.2c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    fig.savefig(
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


def plot_event_gnn_predictions(
    model_dir: Path,
    event_predictions_ffp: Path,
    output_dir: Path,
    ims: list[str],
    map_data: plotting.NZMapData,
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
):
    run_config = ml.RunConfig.from_yaml(model_dir / "run_config.yaml")

    pred_df_ln = pd.read_parquet(event_predictions_ffp)
    pred_df = pred_df_ln.copy(deep=True)
    pred_df[constants.GNN_PRED_PSA_KEYS] = np.exp(
        pred_df[constants.GNN_PRED_PSA_KEYS]
    )
    event_id = pred_df.event_id.iloc[0]

    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)
    obs_site_df = obs_data.site_df.loc[obs_data.event_sites[event_id]]
    event_data = obs_data.event_df.loc[event_id]

    for cur_im in tqdm(ims):
        plot_im_values(
            pred_df[["lon", "lat", f"{cur_im}_pred"]].rename(
                columns={f"{cur_im}_pred": "im_value"}
            ),
            cur_im,
            event_data,
            obs_site_df,
            region,
            map_data=map_data,
            output_ffp=output_dir
            / f"{event_id}_{utils.get_im_filename(cur_im)}.png",
        )

def plot_event_gmm_predictions(
    emp_gm_params_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: list[str],
    map_data: plotting.NZMapData,
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
):
    emp_gm_params = pd.read_parquet(emp_gm_params_ffp)
    emp_gm_params[constants.GMM_PRED_PSA_KEYS] = np.exp(
        emp_gm_params[constants.GMM_PRED_PSA_KEYS]
    )

    obs_data = data.load_obs_nzgmdb(nzgmdb_ffp)
    obs_site_df = obs_data.site_df.loc[obs_data.event_sites[event_id]]
    event_data = obs_data.event_df.loc[event_id]

    for cur_im in tqdm(ims):
        plot_im_values(
            emp_gm_params[["lon", "lat", f"{cur_im}_mean"]].rename(
                columns={f"{cur_im}_mean": "im_value"}
            ),
            cur_im,
            event_data,
            obs_site_df,
            region,
            map_data=map_data,
            output_ffp=output_dir
            / f"{event_id}_{utils.get_im_filename(cur_im)}.png",
        )

def plot_event_cim_predictions(
    cim_results_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: list[str],
    map_data: plotting.NZMapData,
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
):
    cim_results = pd.read_parquet(cim_results_ffp)
    cim_results[constants.CIM_PRED_PSA_KEYS] = np.exp(
        cim_results[constants.CIM_PRED_PSA_KEYS]
    )

    obs_data = data.load_obs_nzgmdb(nzgmdb_ffp)
    obs_site_df = obs_data.site_df.loc[obs_data.event_sites[event_id]]
    event_data = obs_data.event_df.loc[event_id]

    for cur_im in tqdm(ims):
        plot_im_values(
            cim_results[["lon", "lat", f"{cur_im}_cond_mean"]].rename(
                columns={f"{cur_im}_cond_mean": "im_value"}
            ),
            cur_im,
            event_data,
            obs_site_df,
            region,
            map_data=map_data,
            output_ffp=output_dir
            / f"{event_id}_{utils.get_im_filename(cur_im)}.png",
        )