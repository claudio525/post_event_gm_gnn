from pathlib import Path

import pandas as pd
import numpy as np

from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.classdef import TectType, GMM
from IM_calculation.source_site_dist import src_site_dist
from qcore import srf

from . import constants
from . import utils


def run_emp_gmms(
    output_ffp: Path,
    site_dir: Path,
    srf_dir: Path,
    nz_gmdb_source_ffp: Path,
    rjb_max: float,
):
    """
    Computes the empirical GMM parameters for all
        specified sites and sources

    Parameters
    ----------
    output_ffp: Path
    site_dir: Path
        Directory that contains all the site
        information files (i.e. vs30, ll, and z)
    srf_dir: Path
        Directory that contains the srf files
    nz_gmdb_source_ffp: Path
        Path to the NZ-GMDB source file
    rjb_max: float
        RJB distance threshold

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """

    ### Constants
    GMM_MAPPING = {
        TectType.ACTIVE_SHALLOW: GMM.Br_10,
        TectType.SUBDUCTION_SLAB: GMM.ZA_06,
        TectType.SUBDUCTION_INTERFACE: GMM.ZA_06,
    }

    TECT_CLASS_MAPPING = {
        "Crustal": TectType.ACTIVE_SHALLOW,
        "Slab": TectType.SUBDUCTION_SLAB,
        "Interface": TectType.SUBDUCTION_INTERFACE,
        "Undetermined": TectType.ACTIVE_SHALLOW,
    }

    OQ_INPUT_COLUMNS = [
        "vs30",
        "rrup",
        "rjb",
        "z1pt0",
        "mag",
        "rake",
        "dip",
        "vs30measured",
        "ztor",
        "rx",
        "hypo_depth",
    ]

    ### Data loading
    # Get all srf files
    srf_ffps = list(srf_dir.rglob("*.srf"))
    events = [cur_ffp.stem for cur_ffp in srf_ffps]

    # Load source info
    source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)

    # Load srf data
    srf_points, plane_infos = {}, {}
    for cur_srf_ffp in srf_ffps:
        srf_points[cur_srf_ffp.stem] = srf.read_srf_points(str(cur_srf_ffp))
        plane_infos[cur_srf_ffp.stem] = srf.read_header(str(cur_srf_ffp), idx=True)

    # Load the site_data
    stations_df = pd.read_csv(
        site_dir / f"{constants.STATION_FN_NAME}.ll",
        sep=" ",
        index_col=2,
        header=None,
        names=["lon", "lat"],
    )
    vs30_df = pd.read_csv(
        site_dir / f"{constants.STATION_FN_NAME}.vs30",
        sep=" ",
        index_col=0,
        header=None,
        names=["vs30"],
    )
    z_df = pd.read_csv(site_dir / f"{constants.STATION_FN_NAME}.z", index_col=0)

    ### Data merging/re-naming and tidy up
    assert np.all(stations_df.index == vs30_df.index) and np.all(
        stations_df.index == z_df.index
    )
    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    site_df = site_df.rename(columns={"Z_1.0(km)": "z1pt0"})
    del stations_df, vs30_df, z_df

    ### Distance calculation
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    data_dfs = []
    for cur_event in events:
        cur_data_df = site_df.copy(True)
        cur_data_df["rrup"], cur_data_df["rjb"] = src_site_dist.calc_rrup_rjb(
            srf_points[cur_event], site_locs
        )

        cur_data_df["rx"], cur_data_df["ry"] = src_site_dist.calc_rx_ry(
            srf_points[cur_event], plane_infos[cur_event], site_locs
        )
        # Enforce distance threshold
        cur_data_df = cur_data_df.loc[cur_data_df.rjb <= rjb_max]
        cur_data_df["site"] = cur_data_df.index.values
        cur_data_df["event"] = str(cur_event)
        cur_data_df.index = np.add(f"{cur_event}_", cur_data_df.index.values)

        # Add event data
        cur_data_df[
            ["mag", "tect_class", "ztor", "rake", "dip", "hypo_depth"]
        ] = source_df.loc[
            cur_event, ["mag", "tect_class", "z_tor", "rake", "dip", "depth"]
        ]

        data_dfs.append(cur_data_df)

    data_df = pd.concat(data_dfs, axis=0)
    data_df["vs30measured"] = False

    ### GM prediction
    dfs = []
    sites = np.unique(data_df.site)
    for site_ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {site_ix + 1}/{len(sites)}")

        cur_site_mask = data_df.site.values == cur_site

        for cur_tect_class in np.unique(data_df.loc[cur_site_mask].tect_class):
            cur_tect_mask = cur_site_mask & (data_df.tect_class == cur_tect_class)

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]
            pga_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "pSA",
                constants.PERIODS,
            )

            cur_df = pd.concat((pga_result, psa_result), axis=1)
            cur_df.index = data_df.loc[cur_tect_mask].index
            cur_df[["event", "site"]] = data_df[["event", "site"]]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)

    # Rename the columns
    print(f"wtf")

    result_df.to_csv(output_ffp, index_label="id")


def compute_sim_gm_parameters(simulation_imdb_ffp: Path):
    # Get the simulation data
    sim_data = utils.load_sim_data(simulation_imdb_ffp)

    # Compute the GM parameters for each site
    results = {}
    for cur_site, cur_im_data in sim_data.items():
        # Compute mean and sigma



        print(f"wtf")



    print(f"wtf")