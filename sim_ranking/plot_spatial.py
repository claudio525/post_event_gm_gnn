from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from pygmt_helper import plotting

from . import utils
from . import constants
from . import ml
from . import data

IM_LIMITS_MAPPING = {
    "pSA_0.01": (0.0, 1.0, 0.05),
    "pSA_0.1": (0.0, 2.5, 0.125),
    "pSA_0.5": (0.0, 1.5, 0.075),
    "pSA_1.0": (0.0, 0.8, 0.04),
    "pSA_5.0": (0.0, 0.25, 0.0125),
    "pSA_10.0": (0.0, 0.025, 0.00125),
}


def custom_shading_fn(
    topo_grid: xr.DataArray, topo_shading_grid: xr.DataArray
) -> xr.DataArray:
    return topo_shading_grid.where(topo_shading_grid > 0.1, np.nan)


def plot_im_values(
    im_values: pd.DataFrame,
    im: str,
    event_info: pd.Series,
    obs_site_df: pd.DataFrame,
    region: tuple[float, float, float, float],
    output_ffp,
    grid_spacing: str = "25e/25e",
):
    grid = plotting.create_grid(
        im_values,
        "im_value",
        region=region,
        grid_spacing=grid_spacing,
        high_quality=True,
    )

    # Create figure
    fig = plotting.gen_region_fig(
        region=region,
        plot_kwargs={
            "topo_cmap": "gray",
            "topo_cmap_min": 0,
            "topo_cmap_max": 1500,
            "topo_cmap_inc": 25,
            "topo_cmap_reverse": True,
            "land_color": "white",
            "road_pen_color": "black",
            "highway_pen_color": "orange",
        },
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            # MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="wsne",
            FONT_ANNOT_PRIMARY="11p,Helvetica,black",
            FONT_LABEL="12p,Helvetica,black",
        ),
        high_res_topo=True,
        high_quality=True,
        plot_roads=True,
        custom_shading_fn=custom_shading_fn,
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
        transparency=30,
        encode_cb_label=False,
        cb_position="JBC+o0c/0.25c",
        cb_box="+gwhite+c0c/0c/0.3c/0c" 
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
        scale=f"{0.075 * event_info.mag}c",
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
        style="t0.3c",
        fill="darkblue",
        pen="0.1p,darkblue",
    )

    fig.savefig(
        output_ffp,
        dpi=900,
        anti_alias=True,
    )


def plot_res_values(
    res_values: pd.DataFrame,
    im: str,
    event_info: pd.Series,
    obs_site_df: pd.DataFrame,
    region: tuple[float, float, float, float],
    output_ffp: Path,
    grid_spacing: str = "25e/25e",
):
    grid = plotting.create_grid(
        res_values,
        "im_value",
        region=region,
        grid_spacing=grid_spacing,
        high_quality=True,
    )

    # Create figure
    fig = plotting.gen_region_fig(
        region=region,
        plot_kwargs={
            "topo_cmap": "gray",
            "topo_cmap_min": 0,
            "topo_cmap_max": 1500,
            "topo_cmap_inc": 25,
            "topo_cmap_reverse": True,
            "land_color": "white",
            "road_pen_color": "black",
            "highway_pen_color": "orange",
        },
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            # MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="wsne",
            FONT_ANNOT_PRIMARY="11p,Helvetica,black",
            FONT_LABEL="12p,Helvetica,black",
        ),
        high_res_topo=True,
        high_quality=True,
        plot_roads=True,
        custom_shading_fn=custom_shading_fn,
    )

    # Plot residuals
    label_math = r"\mu_{\text{lnIM}}^{(\text{cIM})} - \mu_{\text{lnIM}}^{(\text{GNN})}"
    plotting.plot_grid(
        fig,
        grid,
        "polar",
        (-0.5, 0.5, 0.1),
        ("darkred", "darkblue"),
        f"{utils.get_nice_im_name(im)} Differences (@[ {label_math}  @[)",
        continuous_cmap=True,
        transparency=50,
        reverse_cmap=True,
        encode_cb_label=False,
        cb_position="JBC+o0c/0.25c",
        cb_box="+gwhite+c0c/0c/0.3c/0c" 
    )

    # Plot the prediction sites
    fig.plot(
        x=res_values.lon.values,
        y=res_values.lat.values,
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
        scale=f"{0.075 * event_info.mag}c",
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
        style="t0.3c",
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
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
    use_obs_at_int_sites: bool = True,
):
    """
    Plot the GNN predictions for a given event.

    Parameters
    ----------
    model_dir : Path
        The directory where the GNN model is stored.
    event_predictions_ffp : Path
        The file path to the event predictions.
    output_dir : Path
        The directory where the output plots will be saved.
    ims : list[str]
        The list of IMs to plot.
    region : tuple[float, float, float, float], optional
        The region to plot.
    use_obs_at_int_sites : bool, optional
        If True, use the observed values at the
        sites of interest instead of the predicted values.
        Default is True.
    """
    run_config = ml.RunConfig.from_yaml(model_dir / "run_config.yaml")

    pred_df_ln = pd.read_parquet(event_predictions_ffp)
    pred_df = pred_df_ln.copy(deep=True)
    pred_df[constants.GNN_PRED_PSA_KEYS] = np.exp(pred_df[constants.GNN_PRED_PSA_KEYS])
    event_id = pred_df.event_id.iloc[0]

    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)
    obs_site_df = obs_data.site_df.loc[obs_data.event_sites[event_id]]
    event_data = obs_data.event_df.loc[event_id]

    # At sites of interest with observed values
    # use these instead of the predicted values
    soi_with_obs_ids = pred_df[
        pred_df.index.isin(obs_data.record_df.index)
    ].index.values.astype(str)
    obs_nan_mask = obs_data.record_df.loc[soi_with_obs_ids, constants.PSA_KEYS].isna()
    if use_obs_at_int_sites:
        pred_df.loc[soi_with_obs_ids, constants.GNN_PRED_PSA_KEYS] = np.where(
            obs_nan_mask,
            pred_df.loc[soi_with_obs_ids, constants.GNN_PRED_PSA_KEYS],
            obs_data.record_df.loc[soi_with_obs_ids, constants.PSA_KEYS],
        )

    for cur_im in tqdm(ims):
        cur_obs_sites = pred_df.loc[soi_with_obs_ids].site_int.values[
            ~obs_nan_mask[cur_im]
        ]
        plot_im_values(
            pred_df[["lon", "lat", f"{cur_im}_pred"]].rename(
                columns={f"{cur_im}_pred": "im_value"}
            ),
            cur_im,
            event_data,
            obs_site_df.loc[cur_obs_sites],
            region,
            output_ffp=output_dir
            / f"gnn_{event_id}_{utils.get_im_filename(cur_im)}.png",
        )


def plot_event_gmm_predictions(
    emp_gm_params_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: list[str],
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
):
    emp_gm_params = pd.read_parquet(emp_gm_params_ffp)
    emp_gm_params[constants.GMM_PRED_PSA_KEYS] = np.exp(
        emp_gm_params[constants.GMM_PRED_PSA_KEYS]
    )

    # Drop any rows with NaN values
    nan_mask = emp_gm_params.isna().any(axis=1)
    print(f"Removing {nan_mask.sum()} rows with NaN values")
    emp_gm_params = emp_gm_params[~nan_mask]

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
            output_ffp=output_dir
            / f"gmm_{event_id}_{utils.get_im_filename(cur_im)}.png",
        )


def plot_event_cim_predictions(
    cim_results_ffp: Path,
    nzgmdb_ffp: Path,
    event_id: str,
    output_dir: Path,
    ims: list[str],
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
    use_obs_at_int_sites: bool = True,
):
    """
    Plot the cIM predictions for a given event.

    Parameters
    ----------
    cim_results_ffp : Path
        The file path to the cIM results.
    nzgmdb_ffp : Path
        The file path to the NZGMDB data.
    event_id : str
        The event ID to plot.
    output_dir : Path
        The directory where the output plots will be saved.
    ims : list[str]
        The list of IMs to plot.
    region : tuple[float, float, float, float], optional
        The region to plot.
    use_obs_at_int_sites : bool, optional
        If True, use the observed values at the
        sites of interest instead of the predicted values.
        Default is True.
    """
    cim_results = pd.read_parquet(cim_results_ffp)
    cim_results[constants.CIM_PRED_PSA_KEYS] = np.exp(
        cim_results[constants.CIM_PRED_PSA_KEYS]
    )

    obs_data = data.load_obs_nzgmdb(nzgmdb_ffp)
    obs_site_df = obs_data.site_df.loc[obs_data.event_sites[event_id]]
    event_data = obs_data.event_df.loc[event_id]

    # At sites of interest with observed values
    # use these instead of the predicted values
    if use_obs_at_int_sites:
        soi_with_obs_ids = cim_results[
            cim_results.index.isin(obs_data.record_df.index)
        ].index.values.astype(str)
        obs_nan_mask = obs_data.record_df.loc[
            soi_with_obs_ids, constants.PSA_KEYS
        ].isna()
        cim_results.loc[soi_with_obs_ids, constants.CIM_PRED_PSA_KEYS] = np.where(
            obs_nan_mask,
            cim_results.loc[soi_with_obs_ids, constants.CIM_PRED_PSA_KEYS],
            obs_data.record_df.loc[soi_with_obs_ids, constants.PSA_KEYS],
        )

    for cur_im in tqdm(ims):
        cur_obs_sites = cim_results.loc[soi_with_obs_ids].site_int.values[
            ~obs_nan_mask[cur_im]
        ]
        plot_im_values(
            cim_results[["lon", "lat", f"{cur_im}_cond_mean"]].rename(
                columns={f"{cur_im}_cond_mean": "im_value"}
            ),
            cur_im,
            event_data,
            obs_site_df.loc[cur_obs_sites],
            region,
            output_ffp=output_dir
            / f"cim_{event_id}_{utils.get_im_filename(cur_im)}.png",
        )


def plot_event_cim_gnn_residuals(
    gnn_model_dir: Path,
    gnn_event_results_ffp: Path,
    cim_event_results_ffp: Path,
    output_dir: Path,
    ims: list[str],
    region: tuple[float, float, float, float] = constants.CANTERBURY_REGION,
    use_obs_at_int_sites: bool = True,
):
    run_config = ml.RunConfig.from_yaml(gnn_model_dir / "run_config.yaml")

    # GNN results
    gnn_pred_df = pd.read_parquet(gnn_event_results_ffp)
    event_id = gnn_pred_df.event_id.iloc[0]

    # cIM results
    cim_results = pd.read_parquet(cim_event_results_ffp)
    cim_results[constants.CIM_PRED_PSA_KEYS] = np.exp(
        cim_results[constants.CIM_PRED_PSA_KEYS]
    )

    # Load the observation data
    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)

    # Compute the residuals
    res_df = pd.DataFrame(
        data=np.log(
            cim_results.loc[gnn_pred_df.index, constants.CIM_PRED_PSA_KEYS].values
        )
        - gnn_pred_df[constants.GNN_PRED_PSA_KEYS].values,
        index=gnn_pred_df.index,
        columns=constants.PSA_KEYS,
    )
    res_df["lon"] = gnn_pred_df.lon.values
    res_df["lat"] = gnn_pred_df.lat.values
    res_df["site_int"] = gnn_pred_df.site_int.values

    if use_obs_at_int_sites:
        soi_with_obs_ids = res_df[
            res_df.index.isin(obs_data.record_df.index)
        ].index.values.astype(str)
        obs_nan_mask = obs_data.record_df.loc[
            soi_with_obs_ids, constants.PSA_KEYS
        ].isna()
        res_df.loc[soi_with_obs_ids, constants.PSA_KEYS] = np.where(
            obs_nan_mask, res_df.loc[soi_with_obs_ids, constants.PSA_KEYS], 0.0
        )

    for cur_im in tqdm(ims):
        cur_obs_sites = res_df.loc[soi_with_obs_ids].site_int.values[
            ~obs_nan_mask[cur_im]
        ]

        plot_res_values(
            res_df[["lon", "lat", cur_im]].rename(columns={cur_im: "im_value"}),
            cur_im,
            obs_data.event_df.loc[event_id],
            obs_data.site_df.loc[cur_obs_sites],
            region,
            output_ffp=output_dir
            / f"cim_gnn_res_{event_id}_{utils.get_im_filename(cur_im)}.png",
        )
